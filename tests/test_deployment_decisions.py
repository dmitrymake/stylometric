"""Deployment panels and commitments must not manufacture authorship decisions."""
from types import SimpleNamespace

import numpy as np
import pytest

from stylo.config import artifact_config_id, deployment_candidates, load_config, with_overrides
from stylo import workdoc
from stylo.jsonio import dump_strict, load_strict
from stylo.pipeline import predict, train
from stylo.pipeline.bundle import BundleError
from stylo.report import evidence


class ToyClassifier:
    def fit(self, texts, labels):
        self.classes_ = np.unique(labels)
        return self

    def predict_proba(self, texts):
        scores = np.full(len(self.classes_), 0.01)
        scores[-1] = 1.0 - 0.01 * (len(scores) - 1)
        return np.tile(scores, (len(texts), 1))


class ToyDelta(ToyClassifier):
    def __init__(self, **_kwargs):
        pass

    def distances(self, texts):
        return np.tile(np.arange(len(self.classes_), 0, -1), (len(texts), 1))


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    data = tmp_path / "data"
    frags = data / "train"
    unknown = data / "unknown"
    unknown.mkdir(parents=True)
    (unknown / "target.txt").write_text("Неизвестный синтетический образец.", encoding="utf-8")
    panel = ["bulgakov", "ilf-petrov", "sholohov"]
    chunker_hash = workdoc.chunker_config_hash(load_config())
    for author in panel + ["background"]:
        directory = frags / author / "reference"
        directory.mkdir(parents=True)
        texts = []
        names = []
        for i in range(5):
            text = " ".join(f"{author}{i}слово{j}" for j in range(100))
            name = f"{i}.txt"
            (directory / name).write_text(text, encoding="utf-8")
            texts.append(text)
            names.append(name)
        manifest = workdoc.build_work_manifest(
            f"{author}/reference", author, texts, names,
            provenance_sha256=workdoc.sha256_text(" ".join(texts)),
            chunker_config_hash=chunker_hash, overlap=0.0,
        )
        dump_strict(manifest.to_dict(), directory / workdoc.MANIFEST_NAME)
    docs = tmp_path / "docs"
    docs.mkdir()
    cfg = with_overrides(load_config(), {
        "paths.data": str(data), "paths.docs": str(docs),
        "deployment.candidate_authors": panel,
    })
    snapshot = SimpleNamespace(train_root=frags, unknown_root=unknown, root=data,
                               generation_id="synthetic-generation")
    monkeypatch.setattr(train, "resolve_fragment_roots", lambda cfg: snapshot)
    monkeypatch.setattr(predict, "resolve_fragment_roots", lambda cfg: snapshot)
    monkeypatch.setattr(evidence, "_current_fragment_identity", lambda cfg: {
        "fragment_generation_id": snapshot.generation_id,
        "fragment_root": str(data.resolve()), "unknown_root": str(unknown.resolve()),
    })
    monkeypatch.setattr(train, "_code_tree_sha256", lambda: "a" * 64)
    monkeypatch.setattr(train, "make_factory", lambda *a, **kw: ToyClassifier)
    monkeypatch.setattr(train, "BurrowsDelta", ToyDelta)
    receipt = train.run(cfg, warm=False, weighting="chunk_weighted_legacy")
    return cfg, receipt


def test_explicit_panel_includes_benchmark_excluded_authors_and_abstains(deployment, capsys):
    cfg, receipt = deployment
    result = predict.run(cfg, expected_bundle_token=receipt["bundle_token"])
    assert result["candidate_authors"] == ["bulgakov", "ilf-petrov", "sholohov"]
    assert result["diagnostic_closed_set_top"] == "sholohov"
    assert result["winner"] is None
    assert result["abstained"] is True
    assert result["abstention_reason"] == "open_set_applicability_unavailable"
    assert result["score_scope"] == "closed_set_ensemble"
    assert result["margin"] > 0.05  # Even a decisive relative score cannot grant applicability.
    assert set(result["ensemble"]) == set(result["candidate_authors"])
    assert evidence.verify_prediction(cfg)
    assert "Победитель:" not in capsys.readouterr().out


def test_pin_token_after_training_keeps_model_identity_and_verifies_report(deployment):
    cfg, receipt = deployment
    pinned = with_overrides(cfg, {"deployment.expected_bundle_token": receipt["bundle_token"]})
    assert artifact_config_id(pinned) == receipt["config_id"]
    assert predict.run(pinned)["abstained"]
    assert evidence.verify_prediction(pinned)
    wrong = with_overrides(pinned, {"deployment.expected_bundle_token": "f" * 32})
    with pytest.raises(BundleError, match="trusted expected token"):
        predict.run(wrong)
    with pytest.raises(evidence.SectionEvidenceError, match="trusted bundle token"):
        evidence.verify_prediction(wrong)
    with pytest.raises(BundleError, match="tokens conflict"):
        predict.run(wrong, expected_bundle_token=receipt["bundle_token"])


def test_other_config_drift_still_rejected(deployment):
    cfg, receipt = deployment
    changed = with_overrides(cfg, {
        "deployment.expected_bundle_token": receipt["bundle_token"],
        "model.classifier.C": 17.0,
    })
    assert artifact_config_id(changed) != receipt["config_id"]
    with pytest.raises(evidence.SectionEvidenceError, match="config does not match"):
        predict.run(changed)


def test_prediction_panel_must_match_bundle(deployment):
    cfg, receipt = deployment
    cfg = with_overrides(cfg, {"deployment.candidate_authors": ["bulgakov", "sholohov"]})
    with pytest.raises(BundleError, match="bundle authors do not match"):
        predict.run(cfg, expected_bundle_token=receipt["bundle_token"])


def test_missing_training_candidate_fails_before_fit(deployment, monkeypatch):
    cfg, _ = deployment
    monkeypatch.setattr(train, "make_factory", lambda *a, **kw: pytest.fail("unexpected fit"))
    changed = with_overrides(cfg, {"deployment.candidate_authors": ["bulgakov", "absent"]})
    with pytest.raises(ValueError, match="candidates absent"):
        train.run(changed, warm=False, weighting="chunk_weighted_legacy")


def test_bundle_chunker_identity_comes_from_verified_records(deployment):
    cfg, receipt = deployment
    path = train.resolve_fragment_roots(cfg).train_root / "bulgakov" / "reference" / workdoc.MANIFEST_NAME
    assert receipt["chunker_config_hash"] == load_strict(path)["chunker_config_hash"]
    assert receipt["chunker_config_hash"] == workdoc.chunker_config_hash(cfg)


@pytest.mark.parametrize("fault", ["missing", "old", "mixed"])
def test_legacy_training_rejects_unverified_chunker_identity_before_warm_or_factory(
    deployment, monkeypatch, fault,
):
    cfg, _receipt = deployment
    frags = train.resolve_fragment_roots(cfg).train_root
    paths = [frags / author / "reference" / workdoc.MANIFEST_NAME
             for author in deployment_candidates(cfg)]
    if fault == "missing":
        paths[0].unlink()
    else:
        with monkeypatch.context() as old:
            old.setattr(workdoc, "CHUNKER_ALGORITHM", "stylo.sent_chunks/v1")
            old.setattr(workdoc, "NORMALIZATION_CONTRACT", "stylo.clean/v1")
            old_hash = workdoc.chunker_config_hash(cfg)
        for path in paths if fault == "old" else paths[:1]:
            manifest = load_strict(path)
            manifest["chunker_config_hash"] = old_hash
            dump_strict(manifest, path)
    monkeypatch.setattr(train, "make_rep_cache", lambda *_args: pytest.fail("unexpected warm-up"))
    monkeypatch.setattr(train, "make_factory", lambda *a, **kw: pytest.fail("unexpected factory"))
    with pytest.raises(ValueError, match="rerun clean/split"):
        train.run(cfg, warm=True, weighting="chunk_weighted_legacy")


@pytest.mark.parametrize("panel", [None, [], ["a"], "a,b", ["a", "a"],
                                  ["a", "unknown"], ["a", " ../b"], ["a", True]])
def test_invalid_or_implicit_panel_fails_before_reading_corpus(panel, tmp_path):
    cfg = with_overrides(load_config(), {
        "paths.data": str(tmp_path / "uncreated"), "deployment.candidate_authors": panel,
    })
    with pytest.raises(ValueError, match="deployment.candidate_authors"):
        train.run(cfg, warm=False, weighting="chunk_weighted_legacy")
    assert not (tmp_path / "uncreated").exists()


def test_config_hash_preserves_legacy_without_token():
    import hashlib
    from stylo.jsonio import dumps_strict

    cfg = load_config()
    previous = hashlib.sha256(dumps_strict(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
    assert artifact_config_id(cfg) == previous
    pinned = with_overrides(cfg, {"deployment.expected_bundle_token": "a" * 32})
    assert artifact_config_id(pinned) == previous
    for key, value in [("model.classifier.C", 3.0),
                       ("deployment.candidate_authors", ["a", "b"]),
                       ("deployment.other_setting", "changed")]:
        assert artifact_config_id(with_overrides(pinned, {key: value})) != previous


def test_cli_accepts_explicit_panel_for_both_commands(monkeypatch):
    from stylo.cli import main

    seen = []
    def trained(cfg, **kwargs):
        seen.append(deployment_candidates(cfg))
        return {"bundle_version": "synthetic", "bundle_token": "a" * 32,
                "training_weighting": kwargs["weighting"]}
    monkeypatch.setattr(train, "run", trained)
    monkeypatch.setattr(predict, "run", lambda cfg, **kw: seen.append(deployment_candidates(cfg)))
    flags = ["--candidate-author", "sholohov", "--candidate-author", "bulgakov"]
    main(["train", *flags])
    main(["predict", *flags, "--model-bundle-token", "a" * 32])
    assert seen == [("bulgakov", "sholohov"), ("bulgakov", "sholohov")]
