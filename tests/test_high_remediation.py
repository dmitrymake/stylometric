"""Focused fail-closed regressions for the HIGH remediation block."""
from __future__ import annotations

import ast
import concurrent.futures
import dataclasses
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

    def fallback_load(binding, **_kwargs):
        name = binding[0]
        if name == "primary":
            raise OSError("missing")
        return spacy.blank("ru")

    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        lambda name: (
            "test-version",
            hashlib.sha256(name.encode("utf-8")).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        nlp_module,
        "_verified_model_package_binding",
        lambda name, identity: (name, identity),
    )
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        fallback_load,
    )
    fallback_pipeline = nlp_module.load_nlp(
        "primary", "fallback", max_length=1234
    )
    fallback_identity = nlp_module.resolved_nlp_identity(fallback_pipeline)
    assert fallback_identity.resolved_model == "fallback"
    assert fallback_identity.fallback_used is True

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        lambda _binding, **_kwargs: spacy.blank("ru"),
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
        nlp_module,
        "_load_verified_model_from_binding",
        lambda _binding, **_kwargs: spacy.blank("ru"),
    )
    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        lambda name: (
            "test-version",
            hashlib.sha256(name.encode("utf-8")).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        nlp_module,
        "_verified_model_package_binding",
        lambda name, identity: (name, identity),
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
    tmp_path,
    monkeypatch,
):
    from stylo.eval import dispatch, final, groupkfold, lobo, provenance, sweep
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    duplicate = "один и тот же зарегистрированный фрагмент " * 20
    root = tmp_path / "frags"
    for author in ("alpha", "beta"):
        work = root / author / "work"
        work.mkdir(parents=True)
        for index in range(5):
            (work / f"{index}.txt").write_text(
                duplicate,
                encoding="utf-8",
            )
    dataset = load_dataset(root)
    monkeypatch.setattr(
        dispatch,
        "frozen_run_contract",
        lambda _cfg: provenance.RunContract.build(root, (), "unknown"),
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


def test_production_scientific_kernels_reject_synthetic_authority(tmp_path):
    from stylo.eval import (
        groupkfold,
        lobo,
        sweep,
        work_balanced_ablation_screen,
    )
    from stylo.eval.provenance import (
        ProvenanceError,
        prepare_synthetic_scientific_evaluation,
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
    calls = (
        lambda: lobo._lobo_run(
            object(), context, "stylo", None, 0, 1
        ),
        lambda: groupkfold._gkf_run(
            object(), context, "stylo", None, 2
        ),
        lambda: groupkfold._gkf_run_panel(
            object(), context, "stylo", None, {}
        ),
        lambda: groupkfold.evaluate_frozen_panel_factory(
            object(), context, lambda: object(), {}
        ),
        lambda: sweep._evaluate_case(
            object(),
            context,
            sweep.EvalCase("synthetic"),
            weighting=CHUNK_WEIGHTED_LEGACY,
        ),
        lambda: work_balanced_ablation_screen.run_ablation_screen(
            object(), context, {}, tmp_path / "ablation.json"
        ),
    )
    for call in calls:
        with pytest.raises(ProvenanceError, match="disk-verified"):
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


def test_scientific_context_cannot_be_cloned_or_changed_after_authorization():
    from stylo.eval.provenance import (
        ProvenanceError,
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
    duplicate = np.array(
        ["cross work duplicate", "cross work duplicate"],
        dtype=object,
    )
    duplicate.setflags(write=False)
    with pytest.raises(TypeError):
        dataclasses.replace(context, texts=duplicate)

    object.__setattr__(context, "texts", duplicate)
    with pytest.raises(ProvenanceError, match="receipt|changed"):
        require_scientific_evaluation_context(context)


def test_disk_authority_cannot_be_minted_by_freeze_or_pickle(
    tmp_path,
):
    import pickle

    from stylo.eval import provenance
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    root = tmp_path / "frags"
    _legacy_corpus(root)
    dataset = load_dataset(root)
    with pytest.raises(
        provenance.ProvenanceError,
        match="registered.*verify_dataset_against_disk",
    ):
        provenance._freeze_scientific_context(
            dataset,
            CHUNK_WEIGHTED_LEGACY,
            disk_verified=True,
        )

    transport = provenance.prepare_synthetic_scientific_evaluation(
        dataset,
        CHUNK_WEIGHTED_LEGACY,
    )
    restored = pickle.loads(pickle.dumps(transport))
    assert restored.disk_verified is False
    with pytest.raises(provenance.ProvenanceError, match="disk-verified"):
        provenance.require_disk_verified_scientific_context(restored)


def test_context_registry_value_binds_every_provenance_field(tmp_path):
    from stylo.eval import provenance
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    root = tmp_path / "frags"
    _legacy_corpus(root)
    dataset = load_dataset(root)
    context = provenance.prepare_synthetic_scientific_evaluation(
        dataset,
        CHUNK_WEIGHTED_LEGACY,
    )
    object.__setattr__(
        context.provenance,
        "frags_root",
        "/mutated-after-authorization",
    )
    with pytest.raises(provenance.ProvenanceError, match="changed"):
        provenance.require_scientific_evaluation_context(context)


def test_context_restore_rechecks_content_instead_of_trusting_serialized_seal():
    from stylo.eval import provenance
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    duplicate = "одинаковый межкнижный фрагмент " * 20
    texts = np.array([duplicate, duplicate], dtype=object)
    y = np.array([0, 1], dtype=int)
    groups = np.array(["alpha/work", "beta/work"], dtype=object)
    authors = ("alpha", "beta")
    receipt = provenance._scientific_rows_receipt(
        texts,
        y,
        groups,
        authors,
    )
    with pytest.raises(
        ContentIsolationError,
        match="exact_cross_work_chunk",
    ):
        provenance._restore_scientific_evaluation_context(
            texts,
            y,
            groups,
            authors,
            None,
            CHUNK_WEIGHTED_LEGACY,
            receipt,
            receipt,
            provenance.SCIENTIFIC_ISOLATION_CONTRACT_VERSION,
        )


def test_derived_scientific_context_requires_true_parent_subsequence(tmp_path):
    from stylo.domain.corpus_identity import RowIdentity
    from stylo.eval import provenance
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    root = tmp_path / "frags"
    _legacy_corpus(root)
    parent = load_dataset(root)
    parent_context = provenance.prepare_synthetic_scientific_evaluation(
        parent,
        CHUNK_WEIGHTED_LEGACY,
    )
    legitimate = provenance.derive_dataset(parent, [0, 1, 5, 6])
    accepted = provenance.prepare_synthetic_derived_scientific_evaluation(
        parent_context,
        legitimate,
    )
    assert accepted.rows_digest == legitimate.provenance.rows_digest

    text = "fabricated but internally consistent child row"
    group = "gamma/fabricated"
    row_ids = (
        RowIdentity(
            group=group,
            ordinal=0,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ),
    )
    parent_prov = parent.provenance
    fake_prov = provenance.build_provenance(
        loader_kind=parent_prov.loader_kind,
        texts=[text],
        y=[0],
        groups=[group],
        authors=["gamma"],
        row_ids=row_ids,
        frags_root=parent_prov.frags_root,
        corpus_policy=parent_prov.corpus_policy,
        chunker_config_hash=parent_prov.chunker_config_hash,
        manifest_hash=parent_prov.manifest_hash,
        config_id=parent_prov.config_id,
        parent_rows_digest=parent_prov.rows_digest,
        selection_manifest_digest=provenance._selection_digest(row_ids),
    )
    fabricated = Dataset(
        texts=np.array([text], dtype=object),
        y=np.array([0], dtype=int),
        groups=np.array([group], dtype=object),
        authors=["gamma"],
    )
    fabricated.provenance = fake_prov
    with pytest.raises(
        provenance.ProvenanceError,
        match="ordered subsequence",
    ):
        provenance.prepare_synthetic_derived_scientific_evaluation(
            parent_context,
            fabricated,
        )


def test_disk_verified_parent_authorizes_only_its_validated_subsequence(
    tmp_path,
    monkeypatch,
):
    from stylo.eval import dispatch, provenance
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    root = tmp_path / "frags"
    _legacy_corpus(root)
    parent = load_dataset(root)
    monkeypatch.setattr(
        dispatch,
        "frozen_run_contract",
        lambda _cfg: provenance.RunContract.build(root, (), "unknown"),
    )
    parent_context = provenance.prepare_scientific_evaluation(
        object(),
        parent,
        CHUNK_WEIGHTED_LEGACY,
    )
    child = provenance.derive_dataset(parent, [0, 1, 5, 6])
    child_context = provenance.prepare_derived_scientific_evaluation(
        parent_context,
        child,
    )
    assert child_context.disk_verified is True
    assert (
        provenance.require_disk_verified_scientific_context(child_context)
        is child_context
    )


def test_serialized_production_context_must_reverify_against_disk(
    tmp_path,
    monkeypatch,
):
    import pickle

    from stylo.eval import dispatch, provenance
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    class _Cfg:
        def to_dict(self):
            return {"paths": {"data": "test"}}

    root = tmp_path / "frags"
    _legacy_corpus(root)
    dataset = load_dataset(root)
    monkeypatch.setattr(
        dispatch,
        "frozen_run_contract",
        lambda _cfg: provenance.RunContract.build(root, (), "unknown"),
    )
    cfg = _Cfg()
    context = provenance.prepare_scientific_evaluation(
        cfg,
        dataset,
        CHUNK_WEIGHTED_LEGACY,
    )
    transport = pickle.loads(pickle.dumps(context))
    assert transport.disk_verified is False
    restored = provenance.reverify_scientific_context_from_disk(
        cfg,
        transport,
    )
    assert restored.disk_verified is True
    assert restored.rows_digest == context.rows_digest


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


def test_raw_cv_kernel_symbols_cannot_hide_behind_aliases_or_dynamic_lookup():
    root = pathlib.Path(__file__).resolve().parents[1]
    allowed = {
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
        "_evaluate_frozen_panel_factory_validated": {
            "src/stylo/eval/groupkfold.py",
        },
    }
    observed = {name: set() for name in allowed}
    for source_root in ("src", "scripts"):
        for path in (root / source_root).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                references = []
                if isinstance(node, ast.Name):
                    references.append(node.id)
                elif isinstance(node, ast.Attribute):
                    references.append(node.attr)
                elif isinstance(node, ast.alias):
                    references.append(node.name.rsplit(".", 1)[-1])
                elif isinstance(node, ast.Constant) and type(node.value) is str:
                    references.append(node.value)
                for name in references:
                    if name in observed:
                        observed[name].add(relative)
    assert observed == allowed


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


def test_spacy_wheel_record_identity_verifies_installed_member_bytes(
    tmp_path,
    monkeypatch,
):
    import base64

    from stylo import nlp as nlp_module

    payload = b"registered model bytes"
    member = "example_model/data.bin"
    target = tmp_path / member
    target.parent.mkdir()
    target.write_bytes(payload)
    encoded = (
        base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    dist_info = tmp_path / "example_model-3.8.0.dist-info"
    dist_info.mkdir()
    record_member = "example_model-3.8.0.dist-info/RECORD"
    record_path = dist_info / "RECORD"
    record = (
        f"{member},sha256={encoded},{len(payload)}\n"
        f"{record_member},,\n"
    )
    base_record = record
    record_path.write_text(record, encoding="utf-8")

    class Distribution:
        version = "3.8.0"
        _path = dist_info

        @staticmethod
        def read_text(name):
            assert name == "RECORD"
            return record

        @staticmethod
        def locate_file(path):
            return tmp_path / path

    monkeypatch.setattr(
        nlp_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    assert nlp_module.verified_installed_package_record("example_model") == (
        "3.8.0",
        hashlib.sha256(record.encode("utf-8")).hexdigest(),
    )

    target.write_bytes(b"tampered model bytes")
    with pytest.raises(RuntimeError, match="RECORD mismatch"):
        nlp_module.verified_installed_package_record("example_model")

    target.write_bytes(payload)
    unhashed = tmp_path / "example_model" / "vectors.bin"
    unhashed.write_bytes(b"unhashed vectors")
    record += "example_model/vectors.bin,,\n"
    record_path.write_text(record, encoding="utf-8")
    with pytest.raises(RuntimeError, match="unhashed model payload"):
        nlp_module.verified_installed_package_record("example_model")

    record = base_record + "other-1.0.dist-info/RECORD,,\n"
    record_path.write_text(record, encoding="utf-8")
    with pytest.raises(RuntimeError, match="unhashed model payload"):
        nlp_module.verified_installed_package_record("example_model")

    record = f"{member},sha256={encoded},{len(payload)}\n"
    record_path.write_text(record, encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind its own wheel RECORD"):
        nlp_module.verified_installed_package_record("example_model")


def test_spacy_wheel_record_accepts_only_source_derived_unhashed_pyc(
    tmp_path,
    monkeypatch,
):
    import base64
    import importlib.util
    import marshal
    import py_compile

    from stylo import nlp as nlp_module

    source = tmp_path / "example_model" / "__init__.py"
    source.parent.mkdir()
    source_bytes = b'VALUE = "trusted"\n'
    source.write_bytes(source_bytes)
    bytecode = pathlib.Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(
        str(source),
        cfile=str(bytecode),
        dfile=str(source.resolve()),
        doraise=True,
        optimize=0,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )

    source_member = "example_model/__init__.py"
    bytecode_member = bytecode.relative_to(tmp_path).as_posix()
    source_digest = (
        base64.urlsafe_b64encode(hashlib.sha256(source_bytes).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    dist_info = tmp_path / "example_model-3.8.0.dist-info"
    dist_info.mkdir()
    record_member = "example_model-3.8.0.dist-info/RECORD"
    record = (
        f"{source_member},sha256={source_digest},{len(source_bytes)}\n"
        f"{record_member},,\n"
        f"{bytecode_member},,\n"
    )
    (dist_info / "RECORD").write_text(record, encoding="utf-8")

    class Distribution:
        version = "3.8.0"
        _path = dist_info

        @staticmethod
        def read_text(name):
            assert name == "RECORD"
            return record

        @staticmethod
        def locate_file(path):
            return tmp_path / path

    monkeypatch.setattr(
        nlp_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    assert nlp_module.verified_installed_package_record("example_model") == (
        "3.8.0",
        hashlib.sha256(record.encode("utf-8")).hexdigest(),
    )

    raw = bytecode.read_bytes()
    forged_code = compile(
        b'VALUE = "changed"\n',
        str(source.resolve()),
        "exec",
        dont_inherit=True,
        optimize=0,
    )
    bytecode.write_bytes(raw[:16] + marshal.dumps(forged_code))
    with pytest.raises(RuntimeError, match="does not derive"):
        nlp_module.verified_installed_package_record("example_model")

    py_compile.compile(
        str(source),
        cfile=str(bytecode),
        dfile=str(source.resolve()),
        doraise=True,
        optimize=0,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    record = (
        f"{source_member},sha256={source_digest},{len(source_bytes)}\n"
        f"{record_member},,\n"
    )
    (dist_info / "RECORD").write_text(record, encoding="utf-8")
    assert nlp_module.verified_installed_package_record("example_model") == (
        "3.8.0",
        hashlib.sha256(record.encode("utf-8")).hexdigest(),
    )

    raw = bytecode.read_bytes()
    expected_code = marshal.loads(raw[16:])
    forged_code = expected_code.replace(co_qualname="forged_module")
    assert forged_code == expected_code
    bytecode.write_bytes(raw[:16] + marshal.dumps(forged_code))
    with pytest.raises(RuntimeError, match="does not derive"):
        nlp_module.verified_installed_package_record("example_model")

    py_compile.compile(
        str(source),
        cfile=str(bytecode),
        dfile=str(source.resolve()),
        doraise=True,
        optimize=0,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    extension = source.parent / (
        "__init__" + nlp_module.importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    extension.write_bytes(b"unrecorded native extension fixture")
    with pytest.raises(RuntimeError, match="unrecorded model payload"):
        nlp_module.verified_installed_package_record("example_model")


@pytest.mark.parametrize("fallback_route", [False, True])
def test_spacy_load_rejects_sys_path_shadow_of_verified_wheel(
    fallback_route,
    tmp_path,
    monkeypatch,
):
    import base64

    from stylo import nlp as nlp_module

    model = "stylo_verified_model_shadow_fixture"
    installed_root = tmp_path / "installed"
    installed_source = installed_root / model / "__init__.py"
    installed_source.parent.mkdir(parents=True)
    source_bytes = b"VERIFIED = True\n"
    installed_source.write_bytes(source_bytes)
    source_digest = (
        base64.urlsafe_b64encode(hashlib.sha256(source_bytes).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    dist_info = installed_root / f"{model}-1.0.dist-info"
    dist_info.mkdir()
    source_member = f"{model}/__init__.py"
    record_member = f"{model}-1.0.dist-info/RECORD"
    record = (
        f"{source_member},sha256={source_digest},{len(source_bytes)}\n"
        f"{record_member},,\n"
    )
    (dist_info / "RECORD").write_text(record, encoding="utf-8")

    shadow_root = tmp_path / "shadow"
    shadow_source = shadow_root / model / "__init__.py"
    shadow_source.parent.mkdir(parents=True)
    shadow_source.write_text("SHADOW = True\n", encoding="utf-8")

    class Distribution:
        version = "1.0"
        _path = dist_info

        @staticmethod
        def read_text(name):
            assert name == "RECORD"
            return record

        @staticmethod
        def locate_file(path):
            return installed_root / path

    def distribution(name):
        if name == model:
            return Distribution()
        raise nlp_module.importlib.metadata.PackageNotFoundError(name)

    imported = []
    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    assert model not in sys.modules
    monkeypatch.setattr(
        nlp_module.importlib.metadata,
        "distribution",
        distribution,
    )
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        lambda binding, **_kwargs: imported.append(binding[0]),
    )
    monkeypatch.syspath_prepend(str(shadow_root))

    with pytest.raises(RuntimeError, match="outside its verified wheel"):
        if fallback_route:
            nlp_module.load_nlp("missing_primary_fixture", model)
        else:
            nlp_module.load_nlp(model)
    assert imported == []


@pytest.mark.parametrize("fallback_route", [False, True])
@pytest.mark.parametrize(
    "loader_order",
    [("load_nlp", "load_ner"), ("load_ner", "load_nlp")],
)
def test_verified_model_load_bypasses_mutable_preloaded_package_namespace(
    fallback_route,
    loader_order,
    tmp_path,
    monkeypatch,
):
    from stylo import nlp as nlp_module

    model = "stylo_preloaded_model_fixture"
    root = (tmp_path / "installed").resolve()
    origin = root / model / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("TRUSTED = True\n", encoding="utf-8")
    identity = ("1.0", hashlib.sha256(model.encode()).hexdigest())
    binding = (
        str(root),
        f"{model}/__init__.py",
        "sha256=fixture",
        str(origin.stat().st_size),
        (model,),
    )

    poison_calls = []
    poisoned_module = types.ModuleType(model)

    def poisoned_load(**_kwargs):
        poison_calls.append("package.load")
        return spacy.blank("ru")

    poisoned_module.load = poisoned_load
    poisoned_module.load_model_from_init_py = poisoned_load
    monkeypatch.setitem(sys.modules, model, poisoned_module)
    monkeypatch.setattr(
        nlp_module.spacy,
        "load",
        lambda *_args, **_kwargs: poison_calls.append("spacy.load"),
    )

    def verify(name):
        if name == "missing_primary_fixture":
            raise nlp_module.importlib.metadata.PackageNotFoundError(name)
        assert name == model
        return identity

    direct_calls = []

    def direct_load(init_path, *, disable):
        direct_calls.append((pathlib.Path(init_path), tuple(disable)))
        return spacy.blank("ru")

    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        verify,
    )
    monkeypatch.setattr(
        nlp_module,
        "_verified_model_package_binding",
        lambda name, observed: binding
        if (name, observed) == (model, identity)
        else pytest.fail("unexpected model binding"),
    )
    monkeypatch.setattr(
        nlp_module,
        "_spacy_load_model_from_init_py",
        direct_load,
    )
    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()

    primary = "missing_primary_fixture" if fallback_route else model
    fallback = model if fallback_route else None
    for loader_name in loader_order:
        getattr(nlp_module, loader_name)(primary, fallback)
    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    getattr(nlp_module, loader_order[0])(primary, fallback)

    assert poison_calls == []
    assert [call[0] for call in direct_calls] == [origin, origin, origin]
    assert len(direct_calls) == 3
    assert sys.modules[model] is poisoned_module


@pytest.mark.parametrize(
    ("primary", "fallback", "resolved"),
    [
        ("ru_core_news_lg", None, "ru_core_news_lg"),
        ("missing_primary_fixture", "ru_core_news_md", "ru_core_news_md"),
    ],
)
@pytest.mark.parametrize("loader_name", ["load_nlp", "load_ner"])
def test_real_model_load_ignores_substituted_preloaded_load(
    primary,
    fallback,
    resolved,
    loader_name,
    monkeypatch,
):
    from stylo import nlp as nlp_module

    package = pytest.importorskip(resolved)
    poison_calls = []

    def poisoned_load(**_kwargs):
        poison_calls.append(resolved)
        return spacy.blank("ru")

    monkeypatch.setattr(package, "load", poisoned_load)
    monkeypatch.setattr(package, "load_model_from_init_py", poisoned_load)
    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    loaded = getattr(nlp_module, loader_name)(primary, fallback)

    assert poison_calls == []
    if loader_name == "load_nlp":
        identity = nlp_module.resolved_nlp_identity(loaded)
        assert identity.resolved_model == resolved
        assert identity.fallback_used is (fallback is not None)
        assert "morphologizer" in loaded.pipe_names
    else:
        assert "ner" in loaded.pipe_names


def test_spacy_model_packages_are_verified_before_primary_and_fallback_load(
    monkeypatch,
):
    from stylo import nlp as nlp_module

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    events = []

    def verify(name):
        events.append(("verify", name))
        return ("test-version", hashlib.sha256(name.encode()).hexdigest())

    def load(binding, **_kwargs):
        name = binding[0]
        events.append(("load", name))
        if name == "primary":
            raise OSError("primary load failed")
        return spacy.blank("ru")

    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        verify,
    )
    monkeypatch.setattr(
        nlp_module,
        "_verified_model_package_binding",
        lambda name, identity: (name, identity),
    )
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        load,
    )
    loaded = nlp_module.load_nlp("primary", "fallback")
    assert nlp_module.resolved_nlp_identity(loaded).resolved_model == "fallback"
    assert events == [
        ("verify", "primary"),
        ("load", "primary"),
        ("verify", "primary"),
        ("verify", "fallback"),
        ("load", "fallback"),
        ("verify", "fallback"),
    ]

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    events.clear()

    def verify_missing_primary(name):
        events.append(("verify", name))
        if name == "primary":
            raise nlp_module.importlib.metadata.PackageNotFoundError(name)
        return ("test-version", hashlib.sha256(name.encode()).hexdigest())

    def load_fallback_only(binding, **_kwargs):
        name = binding[0]
        events.append(("load", name))
        assert name == "fallback"
        return spacy.blank("ru")

    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        verify_missing_primary,
    )
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        load_fallback_only,
    )
    loaded = nlp_module.load_nlp("primary", "fallback")
    assert nlp_module.resolved_nlp_identity(loaded).resolved_model == "fallback"
    assert events == [
        ("verify", "primary"),
        ("verify", "fallback"),
        ("load", "fallback"),
        ("verify", "fallback"),
    ]


def test_primary_load_oserror_cannot_launder_integrity_drift_into_fallback(
    monkeypatch,
):
    from stylo import nlp as nlp_module

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    observations = iter(
        [
            ("1.0", "a" * 64),
            ("1.0", "b" * 64),
        ]
    )
    fallback_touched = []

    def verify(name):
        if name == "primary":
            return next(observations)
        fallback_touched.append(name)
        return ("1.0", "c" * 64)

    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        verify,
    )
    monkeypatch.setattr(
        nlp_module,
        "_verified_model_package_binding",
        lambda name, identity: (name, *identity),
    )
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        lambda _binding, **_kwargs: (_ for _ in ()).throw(
            OSError("load failed")
        ),
    )

    with pytest.raises(RuntimeError, match="changed during a failed direct load"):
        nlp_module.load_nlp("primary", "fallback")
    assert fallback_touched == []


@pytest.mark.parametrize("loader_name", ["load_nlp", "load_ner"])
@pytest.mark.parametrize("direct_load_succeeds", [False, True])
def test_post_load_integrity_oserror_never_enters_fallback(
    loader_name,
    direct_load_succeeds,
    monkeypatch,
):
    from stylo import nlp as nlp_module

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    primary_checks = 0
    fallback_touched = []
    identity = ("1.0", "a" * 64)

    def verify(name):
        nonlocal primary_checks
        if name == "primary":
            primary_checks += 1
            if primary_checks == 2:
                raise FileNotFoundError("post-load integrity read failed")
            return identity
        fallback_touched.append(name)
        return ("1.0", "b" * 64)

    def direct_load(binding, **_kwargs):
        assert binding[0] == "primary"
        if not direct_load_succeeds:
            raise OSError("direct load failed")
        return spacy.blank("ru")

    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        verify,
    )
    monkeypatch.setattr(
        nlp_module,
        "_verified_model_package_binding",
        lambda name, observed: (name, *observed),
    )
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        direct_load,
    )

    with pytest.raises(RuntimeError, match="cannot reverify.*direct load"):
        getattr(nlp_module, loader_name)("primary", "fallback")
    assert primary_checks == 2
    assert fallback_touched == []


@pytest.mark.parametrize("loader_name", ["load_nlp", "load_ner"])
def test_spacy_record_failure_blocks_model_import_and_fallback(
    loader_name,
    monkeypatch,
):
    from stylo import nlp as nlp_module

    nlp_module._NLP_CACHE.clear()
    nlp_module._NLP_IDENTITIES.clear()
    imported = []

    def reject(_name):
        raise RuntimeError("RECORD integrity failed")

    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        reject,
    )
    monkeypatch.setattr(
        nlp_module,
        "_load_verified_model_from_binding",
        lambda binding, **_kwargs: imported.append(binding[0]),
    )
    with pytest.raises(RuntimeError, match="RECORD integrity failed"):
        getattr(nlp_module, loader_name)("primary", "fallback")
    assert imported == []


def test_benchmark_snapshot_rejects_actual_fallback_and_binds_live_state(
    monkeypatch,
):
    from stylo import nlp as nlp_module

    namespace = runpy.run_path(
        str(
            pathlib.Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_benchmark.py"
        ),
        run_name="stylo_benchmark_live_nlp_identity_test",
    )
    snapshot = namespace["snapshot_benchmark_nlp_identity"]
    model = "benchmark-model"
    cfg = with_overrides(
        load_config(),
        {
            "language.spacy_model": model,
            "language.spacy_model_version": "0.0.0",
        },
    )

    first = spacy.blank("ru")
    first.add_pipe("sentencizer", config={"punct_chars": ["."]})
    second = spacy.blank("ru")
    second.add_pipe("sentencizer", config={"punct_chars": ["!"]})
    fallback = spacy.blank("ru")
    for pipeline in (first, second, fallback):
        pipeline.max_length = 1_234

    first_identity = nlp_module._build_nlp_identity(
        requested=model,
        resolved=model,
        nlp=first,
        max_length=first.max_length,
    )
    second_identity = nlp_module._build_nlp_identity(
        requested=model,
        resolved=model,
        nlp=second,
        max_length=second.max_length,
    )
    fallback_identity = nlp_module._build_nlp_identity(
        requested=model,
        resolved="fallback-model",
        nlp=fallback,
        max_length=fallback.max_length,
    )
    # The package/pipe-name identity alone cannot distinguish component state.
    assert first_identity.identity_sha256 == second_identity.identity_sha256

    for pipeline, identity in (
        (first, first_identity),
        (second, second_identity),
        (fallback, fallback_identity),
    ):
        monkeypatch.setitem(nlp_module._NLP_IDENTITIES, id(pipeline), identity)
    monkeypatch.setattr(
        nlp_module,
        "verified_installed_package_record",
        lambda _name: (
            first_identity.package_version,
            first_identity.package_record_sha256,
        ),
    )

    first_snapshot = snapshot(cfg, first)
    second_snapshot = snapshot(cfg, second)
    assert (
        first_snapshot["identity_sha256"]
        == second_snapshot["identity_sha256"]
    )
    assert (
        first_snapshot["live_pipeline_sha256"]
        != second_snapshot["live_pipeline_sha256"]
    )
    assert (
        first_snapshot["benchmark_identity_sha256"]
        != second_snapshot["benchmark_identity_sha256"]
    )
    first("Совершенно новые слова. Ещё предложение!")
    assert snapshot(cfg, first) == first_snapshot
    with pytest.raises(RuntimeError, match="refuses a fallback"):
        snapshot(cfg, fallback)


def test_benchmark_attestation_binds_runner_lock_runtime_and_rejects_dirty(
    tmp_path,
    monkeypatch,
):
    from importlib.metadata import version

    namespace = runpy.run_path(
        str(
            pathlib.Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_benchmark.py"
        ),
        run_name="stylo_benchmark_attestation_test",
    )
    root = tmp_path / "checkout"
    runner = root / "scripts" / "run_benchmark.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("RUNNER = 1\n", encoding="utf-8")
    (root / "requirements.lock").write_text(
        "locked-dependencies\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='test'\n",
        encoding="utf-8",
    )
    base = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "code_tree_sha256": "b" * 64,
        "config_id": "c" * 64,
    }
    cfg = load_config()
    identity_body = {
        "requested_model": cfg.get_path("language.spacy_model"),
        "resolved_model": cfg.get_path("language.spacy_model"),
        "fallback_used": False,
        "package_version": str(
            cfg.get_path("language.spacy_model_version")
        ),
        "package_record_sha256": "d" * 64,
        "spacy_version": version("spacy"),
        "disabled_pipes": ["ner"],
        "active_pipes": ["tok2vec", "morphologizer", "parser", "lemmatizer"],
        "max_length": 5_000_000,
    }
    core_identity = {
        **identity_body,
        "identity_sha256": hashlib.sha256(
            dumps_strict(
                identity_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    benchmark_identity_body = {
        **core_identity,
        "live_pipeline_sha256": "1" * 64,
    }
    nlp_identity = {
        **benchmark_identity_body,
        "benchmark_identity_sha256": hashlib.sha256(
            dumps_strict(
                benchmark_identity_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    environment_body = {
        "schema_version": "stylo.canonical-environment.v1",
        "python_implementation": "CPython",
        "python_major_minor": "3.11",
        "distributions": {"spacy": version("spacy")},
        "environment_lock_identity_sha256": "e" * 64,
    }
    environment = {
        **environment_body,
        "contract_sha256": hashlib.sha256(
            dumps_strict(
                environment_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    helper = namespace["benchmark_attestation"]
    verified_roots = []
    monkeypatch.setitem(
        helper.__globals__,
        "_attestation",
        lambda _cfg: dict(base),
    )
    monkeypatch.setitem(
        helper.__globals__,
        "verify_installed_environment",
        lambda observed_root: (
            verified_roots.append(pathlib.Path(observed_root)),
            dict(environment),
        )[1],
    )
    first = helper(cfg, nlp_identity=nlp_identity, root=root)
    assert set(first["runtime"]) == {
        "python",
        "python_implementation",
        "numpy",
        "scipy",
        "scikit_learn",
        "spacy",
    }
    assert first["installed_environment"] == environment
    assert first["nlp_model_identity"] == nlp_identity
    assert first["cache_authority"] == {
        "schema_version": "stylo.channel-benchmark-cache-authority.v1",
        "mode": "fresh_ephemeral_recompute",
        "persistent_cache_reads_allowed": False,
        "representation_cache": "unique_empty_temporary_root",
        "doc_cache": "unique_empty_temporary_root",
        "dsp_cache": "run_local_empty_mapping",
        "process_memory_precondition": (
            "representation_doc_and_nlp_caches_empty"
        ),
    }
    assert verified_roots == [root.resolve()]
    runner.write_text("RUNNER = 2\n", encoding="utf-8")
    second = helper(cfg, nlp_identity=nlp_identity, root=root)
    assert second["runner_sha256"] != first["runner_sha256"]
    assert (
        second["requirements_lock_sha256"]
        == first["requirements_lock_sha256"]
    )

    monkeypatch.setitem(
        helper.__globals__,
        "_attestation",
        lambda _cfg: {**base, "git_dirty": True},
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        helper(cfg, nlp_identity=nlp_identity, root=root)

    monkeypatch.setitem(
        helper.__globals__,
        "_attestation",
        lambda _cfg: dict(base),
    )
    changed_identity = {
        **nlp_identity,
        "package_record_sha256": "f" * 64,
    }
    with pytest.raises(RuntimeError, match="digest does not match"):
        helper(cfg, nlp_identity=changed_identity, root=root)

    bad_environment = {
        **environment,
        "python_major_minor": "0.0",
    }
    monkeypatch.setitem(
        helper.__globals__,
        "verify_installed_environment",
        lambda _root: bad_environment,
    )
    with pytest.raises(RuntimeError, match="contract digest is invalid"):
        helper(cfg, nlp_identity=nlp_identity, root=root)


def test_benchmark_uses_only_empty_ephemeral_disk_caches(
    tmp_path,
    monkeypatch,
):
    from stylo import nlp as nlp_module
    from stylo.features import reps as reps_module

    namespace = runpy.run_path(
        str(
            pathlib.Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_benchmark.py"
        ),
        run_name="stylo_benchmark_cache_authority_test",
    )
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    poison = persistent / "dsp_bench_cache.json"
    poison.write_text('{"forged": [1]}', encoding="utf-8")
    cfg = with_overrides(
        load_config(),
        {
            "paths.data": str(persistent),
            "paths.doc_cache": str(persistent / "doc"),
        },
    )
    workspace = tmp_path / "ephemeral"
    workspace.mkdir()
    monkeypatch.setattr(reps_module, "_MEM_REPS", {})
    monkeypatch.setattr(nlp_module, "_MEM_DOCS", {})
    monkeypatch.setattr(nlp_module, "_NLP_CACHE", {})
    monkeypatch.setattr(nlp_module, "_NLP_IDENTITIES", {})

    runtime_cfg = namespace["isolated_benchmark_config"](cfg, workspace)
    assert pathlib.Path(runtime_cfg.get_path("paths.data")).parent == workspace
    assert (
        pathlib.Path(runtime_cfg.get_path("paths.doc_cache")).parent
        == workspace
    )
    assert poison.read_text(encoding="utf-8") == '{"forged": [1]}'
    assert not list(workspace.iterdir())

    (workspace / "preexisting").write_text("untrusted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must start empty"):
        namespace["isolated_benchmark_config"](cfg, workspace)

    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_benchmark.py"
    ).read_text(encoding="utf-8")
    assert "dsp_bench_cache.json" not in source
    assert "make_channels(runtime_cfg)" in source
    assert "make_rep_cache(runtime_cfg)" in source


def test_benchmark_dsp_cache_is_run_local_and_nlp_identity_scoped():
    namespace = runpy.run_path(
        str(
            pathlib.Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_benchmark.py"
        ),
        run_name="stylo_benchmark_dsp_cache_test",
    )

    class Token:
        lemma_ = "мудрость"
        pos_ = "NOUN"

    class Doc(list):
        def __init__(self, text):
            super().__init__([Token()])
            self.text = text

    class NLP:
        calls = 0

        def pipe(self, texts, *, batch_size):
            assert batch_size == 64
            self.calls += 1
            return [Doc(text) for text in texts]

    cache = {}
    nlp = NLP()
    helper = namespace["dsp_matrix"]
    first = helper(
        ["первый текст"],
        nlp=nlp,
        nlp_identity_sha256="a" * 64,
        cache=cache,
    )
    second = helper(
        ["первый текст"],
        nlp=nlp,
        nlp_identity_sha256="a" * 64,
        cache=cache,
    )
    assert np.array_equal(first, second)
    assert first.shape == (1, len(namespace["SUF"]) + 2)
    assert nlp.calls == 1

    helper(
        ["первый текст"],
        nlp=nlp,
        nlp_identity_sha256="b" * 64,
        cache=cache,
    )
    assert nlp.calls == 2
    assert len(cache) == 2


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
