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
from stylo.pipeline.train import _attestation as installed_attestation


@pytest.fixture(autouse=True)
def synthetic_training_attestation(monkeypatch, tmp_path):
    """Exercise deployment semantics in checkouts and Git-free source archives.

    Workspace discovery has separate package tests. Synthetic model tests bind
    their real configuration and fixture code identity without an enclosing Git
    repository.
    """
    monkeypatch.setattr(train, "_require_source_workspace", lambda: tmp_path)
    monkeypatch.setattr(train, "_attestation", lambda cfg: {
        "git_commit": "synthetic-deployment-test",
        "git_dirty": False,
        "code_tree_sha256": train._code_tree_sha256(),
        "config_id": artifact_config_id(cfg),
    })


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
    assert result["score_scope"] == "closed_set_method_diagnostics"
    assert result["margin"] > 0.05  # Even a decisive relative score cannot grant applicability.
    assert set(result["methods"]["stylo_lr"]["scores"]) == set(result["candidate_authors"])
    assert set(result["methods"]["delta"]["distances"]) == set(result["candidate_authors"])
    assert evidence.verify_prediction(cfg)
    output = capsys.readouterr().out
    assert "Победитель:" not in output
    assert "Топ-3 по Stylo LR:" in output and "Топ-3 по Delta" in output
    assert "legacy Delta" not in output
    assert "checked_matching_training_references" not in output
    assert "точные совпадения с эталонами проверены" in output


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


def test_config_hash_excludes_only_presentation_and_external_token():
    cfg = load_config()
    previous = artifact_config_id(cfg)
    pinned = with_overrides(cfg, {"deployment.expected_bundle_token": "a" * 32})
    assert artifact_config_id(pinned) == previous
    assert artifact_config_id(with_overrides(cfg, {"evaluation.top_k_candidates": 1,
                                                 "paths.docs": "different-results"})) == previous
    for key, value in [("model.classifier.C", 3.0),
                       ("deployment.candidate_authors", ["a", "b"]),
                       ("deployment.other_setting", "changed")]:
        assert artifact_config_id(with_overrides(pinned, {key: value})) != previous
    for key, value in [("features.char_ngrams.max_features", 123),
                       ("language.spacy_model", "different"),
                       ("evaluation.training_weighting", "work_balanced"),
                       ("chunking.chunk_size", 123)]:
        assert artifact_config_id(with_overrides(cfg, {key: value})) != previous


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


def test_training_without_git_preserves_mandatory_hashes(deployment, monkeypatch):
    cfg, _receipt = deployment
    import subprocess
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("git unavailable")))
    monkeypatch.setattr(train, "_require_source_workspace", lambda: pytest.fail("ordinary training must not require Git"))
    monkeypatch.setattr(train, "_attestation", installed_attestation)
    receipt = train.run(cfg, warm=False, weighting="chunk_weighted_legacy")
    assert receipt["git_commit"] is None and receipt["git_dirty"] is None
    assert receipt["code_tree_sha256"] == "a" * 64
    assert receipt["config_id"] == artifact_config_id(cfg)
    assert receipt["rows_digest"] and receipt["chunker_config_hash"]
    assert predict.run(cfg, expected_bundle_token=receipt["bundle_token"])["abstained"]


def test_deployment_code_identity_ignores_modes_and_binds_bytes(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = train._code_tree_sha256(tmp_path)
    source.chmod(0o600)
    assert train._code_tree_sha256(tmp_path) == before
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert train._code_tree_sha256(tmp_path) != before


def test_deployment_code_identity_excludes_presentation_and_keeps_unknown_compute_modules(tmp_path):
    model = tmp_path / "models" / "custom.py"
    model.parent.mkdir()
    model.write_text("VALUE = 1\n", encoding="utf-8")
    before = train._code_tree_sha256(tmp_path)
    for relative in ("cli.py", "report/build.py", "release/source_inventory.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PRESENTATION = 'first caption'\n", encoding="utf-8")
        assert train._code_tree_sha256(tmp_path) == before
        path.write_text("PRESENTATION = 'second caption'\n", encoding="utf-8")
        assert train._code_tree_sha256(tmp_path) == before
    model.write_text("VALUE = 2\n", encoding="utf-8")
    assert train._code_tree_sha256(tmp_path) != before


def _nested_targets(cfg):
    root = predict.resolve_fragment_roots(cfg).unknown_root
    (root / "target.txt").unlink()
    for work in ("one", "two"):
        directory = root / "unknown" / work
        directory.mkdir(parents=True)
        (directory / "0.txt").write_text(f"Независимый синтетический target {work}.", encoding="utf-8")
    return root


def test_multiple_unknown_works_require_selection_before_deserialization(deployment, monkeypatch):
    cfg, receipt = deployment
    _nested_targets(cfg)
    monkeypatch.setattr(predict.joblib, "load", lambda *_a, **_kw: pytest.fail("ambiguous input must fail before loading"))
    with pytest.raises(BundleError, match="select --target-work"):
        predict.run(cfg, expected_bundle_token=receipt["bundle_token"])


def test_each_target_gets_independent_json_and_report_without_sweep(deployment):
    from stylo.report import build
    cfg, receipt = deployment
    root = _nested_targets(cfg)
    for work in ("unknown/one", "unknown/two"):
        result = predict.run(cfg, target_work=work, expected_bundle_token=receipt["bundle_token"])
        assert result["target_work_id"] == work and result["n_fragments"] == 1
        assert len(result["fragments"]) == 1 and "text" not in result["fragments"][0]
        stored = evidence.verify_structured_prediction(cfg, work)
        assert stored == result
        output = build.run_prediction(cfg, work)
        assert output.is_file() and work in output.read_text()
    assert evidence.prediction_directory(cfg, "unknown/one") != evidence.prediction_directory(cfg, "unknown/two")
    (root / "unknown/two/0.txt").write_text("Изменённая другая книга.", encoding="utf-8")
    assert evidence.verify_prediction(cfg, "unknown/one")
    with pytest.raises(evidence.SectionEvidenceError, match="drifted"):
        evidence.verify_prediction(cfg, "unknown/two")


@pytest.mark.parametrize("fault", ["same_work_id", "exact_fragment", "exact_whole_work"])
def test_selected_target_cannot_leak_training_references(deployment, fault):
    cfg, receipt = deployment
    root = predict.resolve_fragment_roots(cfg).unknown_root
    (root / "target.txt").unlink()
    work = "bulgakov/reference" if fault == "same_work_id" else "unknown/copied"
    target = root.joinpath(*work.split("/")); target.mkdir(parents=True)
    reference = train.resolve_fragment_roots(cfg).train_root / "bulgakov/reference"
    text = (reference / "0.txt").read_text() if fault == "exact_fragment" else " ".join((reference / f"{i}.txt").read_text() for i in range(5))
    (target / "0.txt").write_text(text, encoding="utf-8")
    with pytest.raises(BundleError, match="overlaps|exactly copies"):
        predict.run(cfg, target_work=work, expected_bundle_token=receipt["bundle_token"])


def test_source_free_inference_reports_unavailable_content_check(deployment, monkeypatch, tmp_path):
    cfg, receipt = deployment
    target = tmp_path / "standalone"
    (target / "unknown/book").mkdir(parents=True)
    (target / "unknown/book/0.txt").write_text("Новый отдельный учебный текст.", encoding="utf-8")
    monkeypatch.setattr(predict, "resolve_fragment_roots", lambda *_: (_ for _ in ()).throw(FileNotFoundError("references absent")))
    result = predict.run(cfg, unknown_dir=str(target), target_work="unknown/book", expected_bundle_token=receipt["bundle_token"])
    assert result["overlap_check"]["work_id"] == "checked_bundle_metadata"
    assert result["overlap_check"]["exact_content"] == "unavailable_no_training_texts"
    assert evidence.verify_prediction(cfg, "unknown/book")


def test_work_balanced_mode_never_falls_back_to_legacy_bundle(deployment, monkeypatch):
    cfg, receipt = deployment
    cfg = with_overrides(cfg, {"evaluation.training_weighting": "work_balanced"})
    monkeypatch.setattr(predict.joblib, "load", lambda *_a, **_kw: pytest.fail("legacy fallback forbidden"))
    with pytest.raises(BundleError, match="trusted expected token version missing"):
        predict.run(cfg, expected_bundle_token=receipt["bundle_token"])


def test_work_balanced_namespace_rejects_legacy_weighting_metadata(deployment, monkeypatch):
    from stylo.pipeline.bundle import publish_bundle, load_bundle
    cfg, receipt = deployment
    data = train.pathlib.Path(cfg.paths.data)
    meta, paths = load_bundle(data / "deployment/chunk_weighted_legacy", expected_token=receipt["bundle_token"])
    metadata = {k: v for k, v in meta.items() if k not in {"bundle_version", "files"}}
    copied = publish_bundle(data / "deployment/work_balanced", {name: lambda p, source=source: p.write_bytes(source.read_bytes()) for name, source in paths.items()}, metadata)
    cfg = with_overrides(cfg, {"evaluation.training_weighting": "work_balanced"})
    monkeypatch.setattr(predict.joblib, "load", lambda *_a, **_kw: pytest.fail("mismatched metadata must fail before loading"))
    with pytest.raises(BundleError, match="training weighting"):
        predict.run(cfg, expected_bundle_token=copied["bundle_token"])


def test_v2_bundle_loading_remains_compatible_and_nonnull_git_required(deployment):
    from stylo.pipeline import bundle
    cfg, receipt = deployment
    root = train.pathlib.Path(cfg.paths.data) / "deployment/chunk_weighted_legacy"
    sidecar = root / "versions" / receipt["bundle_token"] / bundle.SIDECAR_NAME
    original = load_strict(sidecar)
    meta = {k: v for k, v in original.items() if k not in {"bundle_version", "files"}}
    token = bundle._content_token(original["files"], meta, version=bundle.LEGACY_BUNDLE_VERSION)
    directory = sidecar.parent
    directory.rename(root / "versions" / token)
    sidecar = root / "versions" / token / bundle.SIDECAR_NAME
    dump_strict({**original, "bundle_version": bundle.LEGACY_BUNDLE_VERSION}, sidecar)
    dump_strict({"bundle_version": bundle.LEGACY_BUNDLE_VERSION, "version": token}, root / bundle.CURRENT_NAME)
    loaded, _paths = bundle.load_bundle(root, expected_token=token)
    assert loaded["bundle_version"] == bundle.LEGACY_BUNDLE_VERSION
    with pytest.raises(BundleError, match="non-null"):
        bundle._validate_meta_schema({**meta, "git_commit": None, "git_dirty": None}, version=bundle.LEGACY_BUNDLE_VERSION)


def test_analyze_composes_stages_and_carries_fresh_trusted_token(monkeypatch, tmp_path):
    from stylo.cli import main
    from stylo.pipeline import clean, split
    from stylo.corpus_tools import validate_corpus
    from stylo.report import build
    events = []
    monkeypatch.setattr(clean, "run", lambda cfg: events.append("clean"))
    def validated(cfg, **kwargs):
        assert kwargs == {"target_work_id": "alpha/heldout", "reference_authors": ("alpha", "beta")}
        events.append("validate")
    monkeypatch.setattr(validate_corpus, "run", validated)
    monkeypatch.setattr(split, "run", lambda cfg, **kw: events.append(("split", kw["leave_out"])))
    monkeypatch.setattr(train, "run", lambda cfg, **kw: events.append(("train", kw["weighting"])) or {"bundle_token": "f" * 32})
    def predicted(cfg, **kw):
        assert cfg.deployment.expected_bundle_token == kw["expected_bundle_token"] == "f" * 32
        events.append(("predict", kw["target_work"]))
    monkeypatch.setattr(predict, "run", predicted)
    def reported(cfg, **kw):
        assert cfg.deployment.expected_bundle_token == "f" * 32
        events.append(("report", kw["target_work"], kw["prediction_only"]))
    monkeypatch.setattr(build, "run", reported)
    path = tmp_path / "case.yaml"
    path.write_text("deployment:\n  candidate_authors: [alpha, beta]\n", encoding="utf-8")
    assert main(["analyze", "--config", str(path), "--target-work", "alpha/heldout"]) == 0
    assert events == ["clean", "validate", ("split", ("alpha/heldout",)),
                      ("train", "chunk_weighted_legacy"), ("predict", "alpha/heldout"),
                      ("report", "alpha/heldout", True)]


@pytest.mark.parametrize("weighting", ["chunk_weighted_legacy", "work_balanced"])
def test_real_grouped_fit_save_load_predict_preserves_method_outputs(tmp_path, monkeypatch, weighting):
    pytest.importorskip("ru_core_news_lg")
    import joblib
    from stylo.pipeline.bundle import load_bundle
    from stylo.report import build
    data, clean = tmp_path / "data", tmp_path / "clean"
    frags, unknown = data / "train", data / "unknown"
    cfg = with_overrides(load_config(), {
        "paths.data": str(data), "paths.input_clean": str(clean),
        "paths.docs": str(tmp_path / "reports"), "paths.doc_cache": str(tmp_path / "doc-cache"),
        "deployment.candidate_authors": ["alpha", "beta"],
        "evaluation.training_weighting": weighting, "evaluation.n_jobs": 1,
        "language.parse_n_process": 1, "chunking.chunk_size": 80, "chunking.min_words": 20,
        "features.char_ngrams.min_df": 1, "features.char_ngrams.max_features": 100,
        "features.pos_ngrams.min_df": 1, "features.pos_ngrams.max_features": 50,
        "features.function_words.mfw_count": 30, "delta.mfw_sizes": [30],
    })
    for author in ("alpha", "beta"):
        for book in range(3):
            directory = frags / author / f"book{book}"
            directory.mkdir(parents=True)
            pieces = [
                (f"Это учебный образец {author} номер {book} часть {chunk}. "
                 + ("И он спокойно пишет, а затем смотрит на берег. " if author == "alpha"
                    else "Но почему ветер сильнее? Она снова спрашивает и отвечает! ")) * 4
                for chunk in range(3)
            ]
            names = [f"{chunk}.txt" for chunk in range(3)]
            for name, text in zip(names, pieces):
                (directory / name).write_text(text, encoding="utf-8")
            source = clean / author / f"book{book}.txt"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(" ".join(pieces), encoding="utf-8")
            manifest = workdoc.build_work_manifest(
                f"{author}/book{book}", author, pieces, names,
                provenance_sha256=workdoc.source_provenance_sha256(source),
                chunker_config_hash=workdoc.chunker_config_hash(cfg), overlap=0.0,
            )
            dump_strict(manifest.to_dict(), directory / workdoc.MANIFEST_NAME)
    target = unknown / "unknown" / "query"
    target.mkdir(parents=True)
    query = ["Отдельный учебный рассказ описывает путь через сад. Почему герой остановился?",
             "Потом наступил вечер, и спокойный разговор продолжился на веранде."]
    for i, text in enumerate(query):
        (target / f"{i}.txt").write_text(text, encoding="utf-8")
    snapshot = SimpleNamespace(train_root=frags, unknown_root=unknown, root=data,
                               generation_id="real-synthetic-inputs")
    monkeypatch.setattr(train, "resolve_fragment_roots", lambda _cfg: snapshot)
    monkeypatch.setattr(predict, "resolve_fragment_roots", lambda _cfg: snapshot)
    fitted = {}
    original_factory, original_delta = train.make_factory, train.BurrowsDelta
    def track_factory(*args, **kwargs):
        factory = original_factory(*args, **kwargs)
        def created():
            fitted["pipe"] = factory()
            return fitted["pipe"]
        return created
    def track_delta(**kwargs):
        fitted["delta"] = original_delta(**kwargs)
        return fitted["delta"]
    monkeypatch.setattr(train, "make_factory", track_factory)
    monkeypatch.setattr(train, "BurrowsDelta", track_delta)
    receipt = train.run(cfg, warm=False, weighting=weighting)
    meta, paths = load_bundle(data / "deployment" / weighting, expected_token=receipt["bundle_token"])
    assert meta["training_weighting"] == weighting
    assert (data / "deployment" / weighting / "current.json").is_file()
    loaded_pipe, loaded_delta = joblib.load(paths["model.pkl"]), joblib.load(paths["delta.pkl"])
    np.testing.assert_allclose(loaded_pipe.predict_proba(query), fitted["pipe"].predict_proba(query), rtol=0, atol=0)
    np.testing.assert_allclose(loaded_delta.distances(query), fitted["delta"].distances(query), rtol=0, atol=0)
    result = predict.run(cfg, target_work="unknown/query", expected_bundle_token=receipt["bundle_token"])
    assert result["training_weighting"] == weighting and result["abstained"] is True
    assert result["n_fragments"] == 2 and result["candidate_authors"] == ["alpha", "beta"]
    np.testing.assert_allclose(list(result["methods"]["stylo_lr"]["scores"].values()), loaded_pipe.predict_proba(query).mean(axis=0))
    np.testing.assert_allclose(list(result["methods"]["delta"]["distances"].values()), loaded_delta.distances(query).mean(axis=0))
    assert result["methods"]["delta"]["frequency_denominator"] == (
        "all_analyzer_events" if weighting == "work_balanced" else "sum_selected_mfw_counts"
    )
    assert result["methods"]["delta"]["training_weighting"] == weighting
    assert evidence.verify_structured_prediction(cfg, "unknown/query") == result
    assert build.run_prediction(cfg, "unknown/query").is_file()
    with pytest.raises(BundleError, match="trusted expected token"):
        predict.run(cfg, target_work="unknown/query", expected_bundle_token="f" * 32)
    # Keep a second immutable model/config version and a second book report.
    # The first report must remain reproducible with its own trusted token.
    changed_cfg = with_overrides(cfg, {"model.classifier.C": 0.5})
    second_target = unknown / "unknown" / "second"
    second_target.mkdir()
    (second_target / "0.txt").write_text("Другая учебная книга говорит о дороге и вечернем разговоре.", encoding="utf-8")
    second_receipt = train.run(changed_cfg, warm=False, weighting=weighting)
    assert second_receipt["bundle_token"] != receipt["bundle_token"]
    second_result = predict.run(changed_cfg, target_work="unknown/second", expected_bundle_token=second_receipt["bundle_token"])
    assert evidence.verify_structured_prediction(cfg, "unknown/query") == result
    assert evidence.verify_structured_prediction(changed_cfg, "unknown/second") == second_result
    assert build.run_prediction(cfg, "unknown/query").is_file()
    assert build.run_prediction(changed_cfg, "unknown/second").is_file()
    original_meta, original_paths = load_bundle(data / "deployment" / weighting, expected_token=receipt["bundle_token"])
    assert original_meta["config_id"] == receipt["config_id"]
    np.testing.assert_allclose(joblib.load(original_paths["model.pkl"]).predict_proba(query), loaded_pipe.predict_proba(query), rtol=0, atol=0)
    # Explicit selection also remains independent of an absent CURRENT pointer.
    (data / "deployment" / weighting / "current.json").unlink()
    assert evidence.verify_structured_prediction(cfg, "unknown/query") == result
    assert evidence.verify_structured_prediction(changed_cfg, "unknown/second") == second_result


@pytest.mark.parametrize("target_author", ["bulgakov", "unknown"])
def test_predict_rejects_normalized_collection_before_model_scoring(deployment, monkeypatch, target_author):
    cfg, receipt = deployment
    root = predict.resolve_fragment_roots(cfg).unknown_root
    (root / "target.txt").unlink()
    directory = root / target_author / "edition"
    directory.mkdir(parents=True)
    reference = train.resolve_fragment_roots(cfg).train_root / "bulgakov/reference"
    original = " ".join((reference / f"{i}.txt").read_text() for i in range(5))
    changed = "Учебное предисловие. " + original.replace(" ", "; ") + " Учебное послесловие."
    (directory / "0.txt").write_text(changed, encoding="utf-8")
    monkeypatch.setattr(ToyClassifier, "predict_proba", lambda *_: pytest.fail("overlap must fail before scoring"))
    monkeypatch.setattr(ToyDelta, "distances", lambda *_: pytest.fail("overlap must fail before scoring"))
    with pytest.raises(BundleError, match="normalized/shingle content overlaps"):
        predict.run(cfg, target_work=f"{target_author}/edition", expected_bundle_token=receipt["bundle_token"])


def test_prediction_scope_excludes_inactive_authors_and_unselected_unknown(deployment):
    cfg, receipt = deployment
    root = _nested_targets(cfg)
    references = train.resolve_fragment_roots(cfg).train_root
    (root / "unknown/one/0.txt").write_text((references / "background/reference/0.txt").read_text(), encoding="utf-8")
    (root / "unknown/two/0.txt").write_text((references / "bulgakov/reference/0.txt").read_text(), encoding="utf-8")
    result = predict.run(cfg, target_work="unknown/one", expected_bundle_token=receipt["bundle_token"])
    assert result["overlap_check"]["near_duplicates"] == "checked_matching_training_references"


def test_existing_model_accepts_view_only_changes(deployment, tmp_path):
    cfg, receipt = deployment
    changed = with_overrides(cfg, {"evaluation.top_k_candidates": 1, "paths.docs": str(tmp_path / "other-output")})
    result = predict.run(changed, expected_bundle_token=receipt["bundle_token"])
    assert result["view_settings"] == {"top_k_candidates": 1, "output_directory": str(tmp_path / "other-output")}
