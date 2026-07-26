"""Focused fail-closed regressions for the HIGH remediation block."""
from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import pathlib
import runpy
import subprocess
import sys
import types

import numpy as np
import pytest
import spacy

from stylo.benchmarks import (
    BenchmarkSubmission,
    BenchmarkTruth,
    PredictionRecord,
    ScoringFormatError,
    TruthRecord,
    load_submission,
    score_files,
    score_submission,
    validate_manifest,
)
from stylo.config import load_config, with_overrides
from stylo.corpus import Dataset, CorpusLoadError, load_dataset
from stylo.corpus_tools.validate_corpus import (
    CorpusValidationError,
    run as validate_corpus,
)
from stylo.domain.prediction_contract import (
    PredictionContractError,
    validate_class_indices,
    validate_probabilities,
)
from stylo.domain.corpus_identity import ContentIsolationError
from stylo.eval.ensemble import reliability_weighted
from stylo.eval.run_attestation import (
    LiveRunAttestationError,
    LiveRunAttestor,
)
from stylo.jsonio import dumps_strict, load_strict
from stylo.pipeline.bundle import load_bundle, publish_bundle


def _bundle_meta() -> dict:
    return {
        "training_weighting": "chunk_weighted_legacy",
        "dataset_contract": "legacy_recursive",
        "rows_digest": "a" * 64,
        "chunker_config_hash": "b" * 64,
        "code_tree_sha256": "c" * 64,
        "config_id": "d" * 64,
        "git_commit": "abc123",
        "git_dirty": False,
    }


def _publish(root: pathlib.Path, marker: bytes) -> dict:
    return publish_bundle(
        root,
        {
            "model.pkl": lambda path: path.write_bytes(marker),
            "delta.pkl": lambda path: path.write_bytes(b"delta-" + marker),
            "authors.json": lambda path: path.write_text(
                '["alpha","beta"]', encoding="utf-8"
            ),
        },
        _bundle_meta(),
    )


def test_concurrent_bundle_publishers_never_mix_generations(tmp_path):
    root = tmp_path / "deployment"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda value: _publish(root, value), (b"A", b"B")))
    metadata, files = load_bundle(root)
    current_model = files["model.pkl"].read_bytes()
    assert current_model in {b"A", b"B"}
    expected_delta = b"delta-" + current_model
    assert files["delta.pkl"].read_bytes() == expected_delta
    assert metadata["files"]["model.pkl"] == hashlib.sha256(current_model).hexdigest()
    assert metadata["files"]["delta.pkl"] == hashlib.sha256(expected_delta).hexdigest()
    assert metadata["bundle_version"] == receipts[0]["bundle_version"]


def _legacy_corpus(root: pathlib.Path) -> None:
    for author in ("alpha", "beta"):
        work = root / author / "work"
        work.mkdir(parents=True)
        for index in range(5):
            (work / f"{index}.txt").write_text(
                f"{author} work fragment {index}", encoding="utf-8"
            )


@pytest.mark.parametrize("fault", ("empty", "non_utf8", "zero_row_author"))
def test_legacy_loader_never_silently_drops_expected_rows(tmp_path, fault):
    root = tmp_path / "frags"
    _legacy_corpus(root)
    if fault == "empty":
        (root / "alpha" / "work" / "0.txt").write_text("", encoding="utf-8")
    elif fault == "non_utf8":
        (root / "alpha" / "work" / "0.txt").write_bytes(b"\xff")
    else:
        (root / "gamma" / "empty-work").mkdir(parents=True)
    with pytest.raises(CorpusLoadError):
        load_dataset(root)


def test_probability_contract_rejects_partial_classes_sentinel_and_nan():
    with pytest.raises(PredictionContractError):
        validate_class_indices(np.asarray([0, 2]), 3)
    with pytest.raises(PredictionContractError):
        validate_probabilities(
            np.asarray([[0.5, np.nan, 0.5]]), rows=1, n_classes=3
        )
    with pytest.raises(PredictionContractError, match="sentinel"):
        reliability_weighted(
            {"channel": np.asarray([[1.0, -1.0e9]])},
            {"channel": 0.8},
            chance=0.5,
        )


def test_resolved_nlp_identity_changes_between_fallback_and_primary(monkeypatch):
    from stylo import nlp as nlp_module

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()

    def fallback_load(name, **_kwargs):
        if name == "primary":
            raise OSError("missing")
        return spacy.blank("ru")

    monkeypatch.setattr(nlp_module.spacy, "load", fallback_load)
    fallback_pipeline = nlp_module.load_nlp(
        "primary", "fallback", max_length=1234
    )
    fallback_identity = nlp_module.resolved_nlp_identity(fallback_pipeline)
    assert fallback_identity.resolved_model == "fallback"
    assert fallback_identity.fallback_used is True

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    monkeypatch.setattr(
        nlp_module.spacy,
        "load",
        lambda name, **_kwargs: spacy.blank("ru"),
    )
    primary_pipeline = nlp_module.load_nlp(
        "primary", "fallback", max_length=1234
    )
    primary_identity = nlp_module.resolved_nlp_identity(primary_pipeline)
    assert primary_identity.resolved_model == "primary"
    assert primary_identity.fallback_used is False
    assert primary_identity.identity_sha256 != fallback_identity.identity_sha256


def test_doc_cache_rejects_swapped_text_payload(tmp_path, monkeypatch):
    from stylo import nlp as nlp_module

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    monkeypatch.setattr(
        nlp_module.spacy,
        "load",
        lambda _name, **_kwargs: spacy.blank("ru"),
    )
    cache = nlp_module.DocCache(tmp_path, "model", "configured-v1")
    key = nlp_module._text_key(
        "expected", cache.identity.identity_sha256, cache.version
    )
    cache._store_one(key, cache.nlp("different"))
    assert cache._load_one(key, "expected") is None


def test_corpus_validation_errors_are_fatal_unless_report_only(tmp_path):
    corpus = tmp_path / "clean"
    for author in ("alpha", "beta"):
        path = corpus / author / "book.txt"
        path.parent.mkdir(parents=True)
        path.write_text("одинаковый текст " * 80, encoding="utf-8")
    cfg = with_overrides(
        load_config(),
        {"paths.docs": str(tmp_path / "docs")},
    )
    with pytest.raises(CorpusValidationError, match="exact_dup"):
        validate_corpus(cfg, corpus_dir=str(corpus))
    report = validate_corpus(cfg, corpus_dir=str(corpus), report_only=True)
    assert any(item.code == "exact_dup" for item in report.errors)


def test_split_failure_preserves_both_current_roots(tmp_path, monkeypatch):
    from stylo.pipeline import split

    clean = tmp_path / "clean"
    source = clean / "alpha" / "book.txt"
    source.parent.mkdir(parents=True)
    source.write_text(
        "Первое предложение содержит достаточно слов. "
        "Второе предложение тоже содержит достаточно слов.",
        encoding="utf-8",
    )
    data = tmp_path / "data"
    for name in ("frags_train", "frags_unknown"):
        current = data / name
        current.mkdir(parents=True)
        (current / "old.marker").write_text("old", encoding="utf-8")
    cfg = with_overrides(
        load_config(),
        {
            "paths.input_clean": str(clean),
            "paths.data": str(data),
            "chunking.chunk_size": 10,
            "chunking.min_words": 2,
        },
    )
    monkeypatch.setattr(
        split,
        "make_sent_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        split.run(cfg)
    assert (data / "frags_train" / "old.marker").read_text() == "old"
    assert (data / "frags_unknown" / "old.marker").read_text() == "old"


def _versioned_split_fixture(tmp_path, monkeypatch):
    from stylo.pipeline import split

    clean = tmp_path / "clean"
    for author in ("alpha", "beta"):
        source = clean / author / "same.txt"
        source.parent.mkdir(parents=True)
        source.write_text(f"{author} first generation", encoding="utf-8")
    cfg = with_overrides(
        load_config(),
        {
            "paths.input_clean": str(clean),
            "paths.data": str(tmp_path / "data"),
            "chunking.chunk_size": 10,
            "chunking.min_words": 1,
        },
    )
    monkeypatch.setattr(
        split,
        "sentences_for_text",
        lambda raw, _nlp: [raw],
    )
    monkeypatch.setattr(
        split,
        "make_sent_chunks",
        lambda document, *_args, **_kwargs: [str(document.sents[0])],
    )
    monkeypatch.setattr(split, "load_sentencizer", lambda _language: object())
    return split, cfg, clean, tmp_path / "data"


def test_split_switches_train_unknown_and_map_with_one_pointer(
    tmp_path, monkeypatch
):
    split, cfg, _clean, data = _versioned_split_fixture(tmp_path, monkeypatch)
    split.run(cfg, leave_out=["alpha/same"])
    snapshot = split.resolve_fragment_snapshot(data, require_versioned=True)
    assert snapshot.versioned is True
    assert (snapshot.unknown_root / "alpha" / "same").is_dir()
    assert (snapshot.train_root / "beta" / "same").is_dir()
    assert not (snapshot.unknown_root / "beta" / "same").exists()
    mapping = load_strict(snapshot.chunk_map)
    assert {row["path"].split("/", 1)[0] for row in mapping} == {
        "frags_train",
        "frags_unknown",
    }


def test_split_pointer_failure_preserves_the_entire_prior_generation(
    tmp_path, monkeypatch
):
    split, cfg, clean, data = _versioned_split_fixture(tmp_path, monkeypatch)
    split.run(cfg, leave_out=["alpha/same"])
    before = split.resolve_fragment_snapshot(data, require_versioned=True)
    pointer_before = (
        data / split.SNAPSHOT_DIRECTORY / split.CURRENT_POINTER
    ).read_bytes()
    (clean / "alpha" / "same.txt").write_text(
        "alpha changed second generation", encoding="utf-8"
    )
    monkeypatch.setattr(
        split,
        "_publish_current_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected pointer failure")
        ),
    )
    with pytest.raises(RuntimeError, match="injected pointer failure"):
        split.run(cfg, leave_out=["alpha/same"])
    assert (
        data / split.SNAPSHOT_DIRECTORY / split.CURRENT_POINTER
    ).read_bytes() == pointer_before
    after = split.resolve_fragment_snapshot(data, require_versioned=True)
    assert after.generation_id == before.generation_id
    assert (
        after.unknown_root / "alpha" / "same" / "same_00000.txt"
    ).read_text(encoding="utf-8") == "alpha first generation"


def test_split_resolver_rejects_current_generation_tamper(tmp_path, monkeypatch):
    split, cfg, _clean, data = _versioned_split_fixture(tmp_path, monkeypatch)
    split.run(cfg)
    snapshot = split.resolve_fragment_snapshot(data, require_versioned=True)
    target = next(snapshot.train_root.rglob("*.txt"))
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(split.FragmentSnapshotError, match="inventory/hash mismatch"):
        split.resolve_fragment_snapshot(data, require_versioned=True)


def test_reproducible_benchmark_resolves_the_current_fragment_snapshot(
    tmp_path, monkeypatch
):
    namespace = runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "scripts/run_benchmark.py"),
        run_name="stylo_benchmark_route_test",
    )
    current_train = tmp_path / "versions" / ("a" * 64) / "frags_train"
    current_train.mkdir(parents=True)
    monkeypatch.setitem(
        namespace["resolve_benchmark_data"].__globals__,
        "resolve_fragment_roots",
        lambda _cfg: types.SimpleNamespace(train_root=current_train),
    )
    assert namespace["resolve_benchmark_data"](object()) == current_train
    assert "data/frags_train" not in (
        pathlib.Path(__file__).resolve().parents[1] / "scripts/run_benchmark.py"
    ).read_text(encoding="utf-8")


def _cross_work_duplicate_dataset():
    duplicate = "один и тот же зарегистрированный фрагмент " * 20
    return Dataset(
        texts=np.array([duplicate, duplicate], dtype=object),
        y=np.array([0, 1], dtype=int),
        groups=np.array(["alpha/work", "beta/work"], dtype=object),
        authors=["alpha", "beta"],
    )


def test_all_public_cv_entrypoints_reject_content_overlap_before_workers(
    monkeypatch,
):
    from stylo.eval import dispatch, final, groupkfold, lobo, provenance, sweep
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    dataset = _cross_work_duplicate_dataset()
    monkeypatch.setattr(
        dispatch,
        "frozen_run_contract",
        lambda _cfg: object(),
    )
    monkeypatch.setattr(
        provenance,
        "verify_dataset_against_disk",
        lambda _cfg, _dataset, weighting, _contract: weighting,
    )
    reached = []

    def worker(*_args, **_kwargs):
        reached.append("worker")
        raise AssertionError("raw evaluator reached after failed content gate")

    monkeypatch.setattr(lobo, "_lobo_run", worker)
    monkeypatch.setattr(groupkfold, "_gkf_run", worker)
    monkeypatch.setattr(final, "_lobo_run", worker)
    monkeypatch.setattr(sweep, "_gkf_run", worker)
    monkeypatch.setattr(sweep, "_lobo_run", worker)

    calls = (
        lambda: lobo.lobo_evaluate(
            object(),
            dataset,
            weighting=CHUNK_WEIGHTED_LEGACY,
        ),
        lambda: groupkfold.gkf_evaluate(
            object(),
            dataset,
            weighting=CHUNK_WEIGHTED_LEGACY,
        ),
        lambda: final.run_final(
            object(),
            dataset,
            weighting=CHUNK_WEIGHTED_LEGACY,
        ),
        lambda: sweep.run_sweep(
            object(),
            dataset,
            strategy="gkf",
            weighting=CHUNK_WEIGHTED_LEGACY,
        ),
        lambda: sweep.run_sweep(
            object(),
            dataset,
            strategy="lobo",
            weighting=CHUNK_WEIGHTED_LEGACY,
        ),
    )
    for call in calls:
        with pytest.raises(
            ContentIsolationError,
            match="exact_cross_work_chunk",
        ):
            call()
    assert reached == []


def test_raw_cv_kernels_reject_bare_dataset_before_factory_or_cache():
    from stylo.eval import groupkfold, lobo, sweep
    from stylo.eval.provenance import ProvenanceError

    dataset = Dataset(
        texts=np.array(["alpha unique", "beta unique"], dtype=object),
        y=np.array([0, 1], dtype=int),
        groups=np.array(["alpha/work", "beta/work"], dtype=object),
        authors=["alpha", "beta"],
    )
    calls = (
        lambda: lobo._lobo_run(
            object(),
            dataset,
            "stylo",
            None,
            0,
            1,
        ),
        lambda: groupkfold._gkf_run(
            object(),
            dataset,
            "stylo",
            None,
            2,
        ),
        lambda: groupkfold._gkf_run_panel(
            object(),
            dataset,
            "stylo",
            None,
            {},
        ),
        lambda: groupkfold.evaluate_frozen_panel_factory(
            object(),
            dataset,
            lambda: object(),
            {},
        ),
        lambda: sweep._evaluate_case(
            object(),
            dataset,
            sweep.EvalCase("raw"),
            weighting="chunk_weighted_legacy",
        ),
    )
    for call in calls:
        with pytest.raises(ProvenanceError, match="sealed"):
            call()


def test_scientific_context_remains_sealed_across_process_serialization():
    import pickle

    from stylo.eval.provenance import (
        prepare_synthetic_scientific_evaluation,
        require_scientific_evaluation_context,
    )
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    dataset = Dataset(
        texts=np.array(["alpha unique", "beta unique"], dtype=object),
        y=np.array([0, 1], dtype=int),
        groups=np.array(["alpha/work", "beta/work"], dtype=object),
        authors=["alpha", "beta"],
    )
    context = prepare_synthetic_scientific_evaluation(
        dataset,
        CHUNK_WEIGHTED_LEGACY,
    )
    restored = pickle.loads(pickle.dumps(context))
    assert require_scientific_evaluation_context(restored) is restored
    assert all(
        not values.flags.writeable
        for values in (restored.texts, restored.y, restored.groups)
    )
    assert restored.isolation_receipt_sha256 == (
        context.isolation_receipt_sha256
    )


def test_raw_cv_kernel_callers_are_exactly_the_guarded_orchestrators():
    root = pathlib.Path(__file__).resolve().parents[1]
    expected = {
        "_lobo_run": {
            "src/stylo/eval/final.py",
            "src/stylo/eval/lobo.py",
            "src/stylo/eval/sweep.py",
        },
        "_gkf_run": {
            "src/stylo/eval/groupkfold.py",
            "src/stylo/eval/sweep.py",
        },
        "_gkf_run_panel": {"src/stylo/eval/groupkfold.py"},
        "evaluate_frozen_panel_factory": {
            "src/stylo/eval/groupkfold.py",
        },
    }
    observed = {name: set() for name in expected}
    for source_root in ("src", "scripts"):
        for path in (root / source_root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else None
                    )
                )
                if name in observed:
                    observed[name].add(path.relative_to(root).as_posix())
    assert observed == expected


def test_run_all_checks_content_isolation_before_cache_and_training():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "run.sh"
    ).read_text(encoding="utf-8")
    split = source.index('run split "$@"')
    isolation = source.index('run verify-evaluation-corpus "$@"')
    warm = source.index('run warm "$@"')
    train = source.index('train_receipt="$(run train "$@")"')
    assert split < isolation < warm < train


def test_benchmark_rejects_uncapped_overlap_before_rng_or_cache(
    tmp_path,
    monkeypatch,
):
    namespace = runpy.run_path(
        str(
            pathlib.Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_benchmark.py"
        ),
        run_name="stylo_benchmark_uncapped_gate_test",
    )
    root = tmp_path / "frags"
    hidden = "скрытый за лимитом одинаковый фрагмент " * 20
    for author in ("alpha", "beta"):
        for work in ("one", "two"):
            work_root = root / author / work
            work_root.mkdir(parents=True)
            for index in range(36):
                text = f"{author} {work} unique fragment {index}"
                if work == "one" and index == 35:
                    text = hidden
                (work_root / f"{index:03d}.txt").write_text(
                    text,
                    encoding="utf-8",
                )
    parent = load_dataset(root)
    prepare = namespace["prepare_benchmark_contexts"]
    globals_ = prepare.__globals__
    from stylo.eval.provenance import (
        prepare_synthetic_scientific_evaluation,
    )

    monkeypatch.setitem(
        globals_,
        "prepare_scientific_evaluation",
        lambda _cfg, dataset, weighting: (
            prepare_synthetic_scientific_evaluation(dataset, weighting)
        ),
    )

    class NeverCap:
        calls = 0

        def choice(self, *_args, **_kwargs):
            self.calls += 1
            return np.arange(35)

    rng = NeverCap()
    with pytest.raises(
        ContentIsolationError,
        match="exact_cross_work_chunk",
    ):
        prepare(object(), parent, pd_only=False, rng=rng)
    assert rng.calls == 0


def test_benchmark_candidate_writer_preserves_historical_site_inputs(
    tmp_path,
):
    namespace = runpy.run_path(
        str(
            pathlib.Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_benchmark.py"
        ),
        run_name="stylo_benchmark_writer_test",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    historical = {
        "validation.json": b"frozen-full",
        "validation_pd.json": b"frozen-pd",
    }
    for name, payload in historical.items():
        (docs / name).write_bytes(payload)
    out = {
        "claim_status": "exploratory_internal",
        "public_headline_authorized": False,
        "dataset_identity": {
            "rows_digest": "a" * 64,
            "isolation_contract_version": "test-v1",
        },
        "attestation": {
            "git_commit": "b" * 40,
            "git_dirty": True,
            "code_tree_sha256": "c" * 64,
            "config_id": "d" * 64,
        },
    }
    published = namespace["publish_benchmark_result"](
        out,
        docs=docs,
        pd_only=False,
        fast=False,
    )
    assert set(published) == {
        "run_provenance.json",
        "validation.full.all_channels.candidate.json",
    }
    assert "exploratory/channel_benchmark/full/all_channels" in (
        published["run_provenance.json"].as_posix()
    )
    for name, payload in historical.items():
        assert (docs / name).read_bytes() == payload
    candidate = load_strict(
        published["validation.full.all_channels.candidate.json"]
    )
    provenance = load_strict(published["run_provenance.json"])
    assert candidate["claim_status"] == "exploratory_internal"
    assert provenance["public_headline_authorized"] is False
    assert provenance["supersedes"] is None


@pytest.mark.parametrize(
    "relative",
    ["scripts/clean_text.py", "scripts/split.py"],
)
def test_legacy_corpus_builders_are_hard_disabled(relative, tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    before = set(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, str(root / relative)],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "retired" in (result.stdout + result.stderr)
    assert set(tmp_path.iterdir()) == before


def test_legacy_report_name_delegates_fail_closed_without_evidence(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    before = set(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "report.py")],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "docs root" in (result.stdout + result.stderr)
    assert set(tmp_path.iterdir()) == before


def test_live_attestor_detects_mid_run_source_change(tmp_path):
    source = tmp_path / "code.py"
    config = tmp_path / "config.yaml"
    cache = tmp_path / "cache.sqlite3"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("seed: 42\n", encoding="utf-8")
    cache.write_bytes(b"immutable-cache")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    attestor = LiveRunAttestor.build(
        repository_root=tmp_path,
        code_hashes={"code.py": digest(source)},
        config_path=config,
        config_sha256=digest(config),
        cache_path=cache,
        cache_sha256=digest(cache),
        cache_size_bytes=cache.stat().st_size,
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(LiveRunAttestationError, match="drifted"):
        attestor.verify("before-checkpoint")


def test_report_uses_verified_v2_strategy_and_rejects_tamper(
    tmp_path, monkeypatch
):
    from stylo.report import build, evidence
    from stylo.pipeline import train
    from stylo import dataset

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "corpus_validation.txt").write_text("corpus ok", encoding="utf-8")
    (docs / "prediction.txt").write_text("prediction ok", encoding="utf-8")
    text = "screen results"
    csv = "model,score\nx,1\n"
    attestation = {
        "code_tree_sha256": "a" * 64,
        "config_id": "b" * 64,
        "git_commit": "commit",
    }
    rows_digest = "c" * 64
    provenance = {
        "schema_version": "stylo.sweep.v2.provenance",
        "training_weighting": "chunk_weighted_legacy",
        "strategy": "gkf",
        "dataset_contract": "legacy_recursive",
        "rows_digest": rows_digest,
        "attestation": attestation,
        "files": {
            "sweep_table.v2.csv": hashlib.sha256(csv.encode()).hexdigest(),
            "sweep_table.v2.txt": hashlib.sha256(text.encode()).hexdigest(),
        },
        "note": "verified test",
    }
    from stylo.eval.provenance import safe_write_batch

    published_sweep = safe_write_batch(
        docs,
        {
            "sweep_table.v2.txt": text,
            "sweep_table.v2.csv": csv,
            "sweep_table.v2.provenance.json": dumps_strict(provenance),
        },
        publication_id="sweep-table-v2",
    )
    cfg = with_overrides(load_config(), {"paths.docs": str(docs)})
    monkeypatch.setattr(
        evidence, "verify_corpus_validation", lambda _cfg: "corpus ok"
    )
    monkeypatch.setattr(
        evidence, "verify_prediction", lambda _cfg: "prediction ok"
    )
    monkeypatch.setattr(train, "_attestation", lambda _cfg: attestation)
    monkeypatch.setattr(
        dataset,
        "resolve_dataset",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            provenance=types.SimpleNamespace(rows_digest=rows_digest)
        ),
    )
    build.run(cfg)
    rendered = (docs / "index.html").read_text(encoding="utf-8")
    assert "GKF screening proxy" in rendered
    assert "not LOBO" in rendered

    published_sweep["sweep_table.v2.txt"].write_text("tampered", encoding="utf-8")
    with pytest.raises(build.ReportEvidenceError, match="digest mismatch"):
        build.run(cfg)


def test_report_corpus_section_binds_bytes_and_current_corpus(
    tmp_path, monkeypatch
):
    from stylo.report import evidence

    docs = tmp_path / "docs"
    corpus = tmp_path / "clean"
    docs.mkdir()
    (corpus / "alpha").mkdir(parents=True)
    source = corpus / "alpha" / "book.txt"
    source.write_text("clean corpus", encoding="utf-8")
    cfg = with_overrides(
        load_config(),
        {
            "paths.docs": str(docs),
            "paths.input_clean": str(corpus),
        },
    )
    monkeypatch.setattr(evidence, "_code_tree_sha256", lambda: "a" * 64)
    evidence.publish_corpus_validation(
        cfg,
        corpus_root=corpus,
        text="corpus ok",
        structured={"status": "ok"},
    )
    assert evidence.verify_corpus_validation(cfg) == "corpus ok"

    (docs / "corpus_validation.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(evidence.SectionEvidenceError, match="hash mismatch"):
        evidence.verify_corpus_validation(cfg)

    evidence.publish_corpus_validation(
        cfg,
        corpus_root=corpus,
        text="corpus ok",
        structured={"status": "ok"},
    )
    source.write_text("different corpus", encoding="utf-8")
    with pytest.raises(evidence.SectionEvidenceError, match="stale"):
        evidence.verify_corpus_validation(cfg)


def test_report_prediction_section_binds_bundle_and_unknown_inputs(
    tmp_path, monkeypatch
):
    from stylo.pipeline.bundle import load_bundle
    from stylo.report import evidence

    docs = tmp_path / "docs"
    data = tmp_path / "data"
    unknown = data / "frags_unknown"
    docs.mkdir()
    unknown.mkdir(parents=True)
    (data / "frags_train").mkdir()
    fragment = unknown / "unknown.txt"
    fragment.write_text("unknown evidence", encoding="utf-8")
    cfg = with_overrides(
        load_config(),
        {
            "paths.data": str(data),
            "paths.docs": str(docs),
        },
    )
    code_hash = "a" * 64
    monkeypatch.setattr(evidence, "_code_tree_sha256", lambda: code_hash)
    meta = _bundle_meta()
    meta["config_id"] = evidence._config_id(cfg)
    meta["code_tree_sha256"] = code_hash
    published = publish_bundle(
        data / "deployment" / "chunk_weighted_legacy",
        {
            "model.pkl": lambda path: path.write_bytes(b"model"),
            "delta.pkl": lambda path: path.write_bytes(b"delta"),
            "authors.json": lambda path: path.write_text(
                '["alpha","beta"]', encoding="utf-8"
            ),
        },
        meta,
    )
    loaded_meta, _paths = load_bundle(
        data / "deployment" / "chunk_weighted_legacy",
        expected_token=published["bundle_token"],
    )
    evidence.publish_prediction(
        cfg,
        unknown_root=unknown,
        report="prediction ok",
        bundle_token=published["bundle_token"],
        bundle_meta=loaded_meta,
    )
    assert evidence.verify_prediction(cfg) == "prediction ok"

    current_fragment = evidence._current_fragment_identity(cfg)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            evidence,
            "_current_fragment_identity",
            lambda _cfg: {
                **current_fragment,
                "fragment_generation_id": "f" * 64,
                "fragment_root": str(data / "fragment_snapshots" / "versions" / ("f" * 64)),
            },
        )
        with pytest.raises(
            evidence.SectionEvidenceError,
            match="current fragment snapshot",
        ):
            evidence.verify_prediction(cfg)

    (docs / "prediction.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(evidence.SectionEvidenceError, match="hash mismatch"):
        evidence.verify_prediction(cfg)

    evidence.publish_prediction(
        cfg,
        unknown_root=unknown,
        report="prediction ok",
        bundle_token=published["bundle_token"],
        bundle_meta=loaded_meta,
    )
    fragment.write_text("different unknown", encoding="utf-8")
    with pytest.raises(evidence.SectionEvidenceError, match="drifted"):
        evidence.verify_prediction(cfg)


def _blind_manifest_raw() -> dict:
    return {
        "schema_version": "1.0",
        "dataset": {
            "name": "synthetic",
            "version": "1",
            "license": "CC0",
            "offset_unit": "token",
            "tokenizer": "stylo_unicode_word_punct_v1",
        },
        "task_types": ["spoof"],
        "documents": [
            {
                "doc_id": "doc_0000000000000001",
                "source": {
                    "source_id": "sealed",
                    "provenance": "sealed:item",
                    "revision": "v1",
                    "sha256": "a" * 64,
                },
                "split": "blind",
                "task_types": ["spoof"],
                "text_path": "text.txt",
                "spans": [],
            }
        ],
    }


def test_score_envelope_binds_parameters_and_typed_abstention():
    manifest = validate_manifest(_blind_manifest_raw())
    manifest_sha = "b" * 64
    truth = BenchmarkTruth(
        "1.0",
        "synthetic",
        "1",
        manifest_sha,
        (TruthRecord("doc_0000000000000001", "author_a", "e", None, None, ()),),
        "c" * 64,
        True,
    )
    abstaining = BenchmarkSubmission(
        "1.0",
        "synthetic",
        "1",
        (PredictionRecord("doc_0000000000000001", None, None, ()),),
        "d" * 64,
    )
    score_a = score_submission(
        manifest,
        truth,
        abstaining,
        manifest_sha256=manifest_sha,
        bootstrap_iters=5,
        seed=1,
        synthetic_integration_only=True,
    )
    score_b = score_submission(
        manifest,
        truth,
        abstaining,
        manifest_sha256=manifest_sha,
        bootstrap_iters=6,
        seed=1,
        synthetic_integration_only=True,
    )
    assert score_a.authorship.coverage == 0.0
    assert score_a.input_bindings == {
        "manifest_sha256": manifest_sha,
        "truth_sha256": "c" * 64,
        "submission_sha256": "d" * 64,
    }
    assert score_a.self_hash != score_b.self_hash
    assert score_a.to_dict()["self_hash"] == score_a.self_hash
    assert set(score_a.code_binding) == {
        "benchmarks/artifacts.py",
        "benchmarks/loader.py",
        "benchmarks/schema.py",
        "benchmarks/scoring.py",
        "benchmarks/validator.py",
        "domain/segmentation.py",
        "eval/segmentation.py",
        "jsonio.py",
    }
    assert score_a.runtime_binding["scipy"] != "not-installed"

    for forbidden in ("__abstain__", "unknown_author"):
        submission = BenchmarkSubmission(
            "1.0",
            "synthetic",
            "1",
            (
                PredictionRecord(
                    "doc_0000000000000001", forbidden, None, ()
                ),
            ),
            "e" * 64,
        )
        with pytest.raises(ScoringFormatError, match="truth universe"):
            score_submission(
                manifest,
                truth,
                submission,
                manifest_sha256=manifest_sha,
                bootstrap_iters=5,
                synthetic_integration_only=True,
            )


def test_submission_parser_reserves_abstention_string_and_file_scorer_requires_root(
    tmp_path,
):
    manifest_raw = _blind_manifest_raw()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_raw), encoding="utf-8")
    manifest = validate_manifest(manifest_raw)
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_name": "synthetic",
                "dataset_version": "1",
                "predictions": [
                    {
                        "doc_id": "doc_0000000000000001",
                        "author_label": "__abstain__",
                        "spans": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ScoringFormatError, match="reserved abstention"):
        load_submission(submission_path)
    with pytest.raises(ScoringFormatError, match="requires artifact_root"):
        score_files(
            manifest,
            manifest_path,
            tmp_path / "truth.json",
            submission_path,
            expected_truth_sha256="f" * 64,
        )


def test_submission_null_is_explicit_abstention_and_missing_field_rejects(
    tmp_path,
):
    manifest = validate_manifest(_blind_manifest_raw())
    manifest_sha = "b" * 64
    truth = BenchmarkTruth(
        "1.0",
        "synthetic",
        "1",
        manifest_sha,
        (TruthRecord("doc_0000000000000001", "author_a", "e", None, None, ()),),
        "c" * 64,
        True,
    )
    body = {
        "schema_version": "1.0",
        "dataset_name": "synthetic",
        "dataset_version": "1",
        "predictions": [
            {
                "doc_id": "doc_0000000000000001",
                "author_label": None,
                "spans": [],
            }
        ],
    }
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(body), encoding="utf-8")
    explicit = load_submission(submission_path)
    assert explicit.predictions[0].author_label is None
    assert explicit.predictions[0].author_label_present is True
    score = score_submission(
        manifest,
        truth,
        explicit,
        manifest_sha256=manifest_sha,
        bootstrap_iters=5,
        synthetic_integration_only=True,
    )
    assert score.authorship.coverage == 0.0

    del body["predictions"][0]["author_label"]
    submission_path.write_text(json.dumps(body), encoding="utf-8")
    omitted = load_submission(submission_path)
    assert omitted.predictions[0].author_label_present is False
    with pytest.raises(ScoringFormatError, match="explicit JSON null"):
        score_submission(
            manifest,
            truth,
            omitted,
            manifest_sha256=manifest_sha,
            bootstrap_iters=5,
            synthetic_integration_only=True,
        )
