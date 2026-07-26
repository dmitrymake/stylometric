"""Focused anti-false-green tests for the 47-class resumable stylo LOBO validation."""
from __future__ import annotations

import copy
import hashlib
import threading

import numpy as np
import pytest
from sklearn.metrics import f1_score

from stylo.corpus import Dataset
from stylo.eval import stylo_lobo_validation as tl
from stylo.eval.provenance import prepare_synthetic_scientific_evaluation
from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY
from stylo.jsonio import dump_strict, load_strict


class _Cfg:
    def get_path(self, path, default=None):
        return {"evaluation.top_k_candidates": 3}.get(path, default)


@pytest.fixture
def tiny_target():
    # The singleton deliberately sits between tested labels: metric order must be [0, 2], not [0, 1].
    authors = ["aa", "singleton", "bb"]
    rows = [
        ("bb/b2", "bb/b2:1"),
        ("aa/a1", "aa/a1:0"),
        ("singleton/s1", "singleton/s1:1"),
        ("bb/b1", "bb/b1:0"),
        ("aa/a2", "aa/a2:1"),
        ("bb/b2", "bb/b2:0"),
        ("aa/a1", "aa/a1:1"),
        ("singleton/s1", "singleton/s1:0"),
        ("bb/b1", "bb/b1:1"),
        ("aa/a2", "aa/a2:0"),
    ]
    groups = np.asarray([group for group, _ in rows], dtype=object)
    texts = np.asarray([text for _, text in rows], dtype=object)
    a2i = {author: index for index, author in enumerate(authors)}
    y = np.asarray([a2i[group.split("/", 1)[0]] for group in groups], dtype=int)
    dataset = Dataset(
        texts=texts,
        y=y,
        groups=groups,
        authors=authors,
    )
    context = prepare_synthetic_scientific_evaluation(
        dataset,
        CHUNK_WEIGHTED_LEGACY,
    )
    inventory = tl.derive_inventory(context)
    return _Cfg(), context, inventory


_CHUNK_PROBABILITIES = {
    "aa/a1:0": (0.8, 0.1, 0.1),
    "aa/a1:1": (0.6, 0.2, 0.2),
    "aa/a2:0": (0.2, 0.7, 0.1),
    "aa/a2:1": (0.4, 0.5, 0.1),
    "bb/b1:0": (0.1, 0.2, 0.7),
    "bb/b1:1": (0.1, 0.4, 0.5),
    "bb/b2:0": (0.7, 0.2, 0.1),
    "bb/b2:1": (0.5, 0.3, 0.2),
}


class _SpyEstimator:
    needs_groups = True

    def __init__(self, registry):
        self.registry = registry
        self.classes_ = np.arange(3, dtype=int)
        self.fit_groups = None
        self.predict_texts = None
        registry.append(self)

    def fit(self, texts, y, *, groups):
        self.fit_groups = tuple(str(group) for group in groups)
        self.fit_labels = tuple(int(label) for label in y)
        return self

    def predict_proba(self, texts):
        self.predict_texts = tuple(str(text) for text in texts)
        return np.asarray([_CHUNK_PROBABILITIES[text] for text in self.predict_texts])


def _runtime_binding(**overrides):
    value = {
        "python": "test",
        "python_implementation": "CPython",
        "python_compiler": "test-compiler",
        "system": "Linux",
        "machine": "x86_64",
        "processor": "test-cpu",
        "libc": {"name": "glibc", "version": "test"},
        "numpy": "test",
        "scipy": "test",
        "sklearn": "test",
        "spacy": "test",
        "joblib": "test",
        "threadpoolctl": "test",
        "spacy_model": {
            "name": "test-model",
            "lang": "ru",
            "version": "test",
            "spacy_version": "test",
            "spacy_git_version": None,
        },
    }
    value.update(overrides)
    return value


def _identity(
    dataset,
    inventory,
    *,
    runtime_fingerprint=None,
    config_path="synthetic.yaml",
    cache_path="synthetic-reps.pkl",
):
    return tl.build_synthetic_run_identity(
        dataset=dataset,
        inventory=inventory,
        config={
            "path": config_path,
            "sha256": "c" * 64,
            "resolved_sha256": "e" * 64,
        },
        code_hashes={"src/stylo/eval/stylo_lobo_validation.py": "b" * 64},
        git_commit="a" * 40,
        git_dirty=True,
        runtime_fingerprint=runtime_fingerprint or _runtime_binding(),
        thread_fingerprint={"worker_policy": "one"},
        representation_cache={
            "path": cache_path,
            "size_bytes": 1,
            "sha256": "7" * 64,
            "rep_version": "synthetic-v1",
        },
        reference_sha256="f" * 64,
        seed=42,
        bootstrap_iters=100,
        ci_level=0.95,
        noninferiority_margin=0.02,
    )


def test_kernel_release_is_absent_while_libc_and_runtime_inventory_are_binding(tiny_target):
    _, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    assert "platform" not in identity["runtime_fingerprint"]
    assert "release" not in identity["runtime_fingerprint"]

    libc_drift = _runtime_binding(libc={"name": "glibc", "version": "different"})
    changed = _identity(dataset, inventory, runtime_fingerprint=libc_drift)
    assert changed["run_id"] != identity["run_id"]

    for forbidden in ("platform", "release"):
        invalid = _runtime_binding(**{forbidden: "must-not-bind"})
        with pytest.raises(tl.TrueLoboError, match="field mismatch"):
            _identity(dataset, inventory, runtime_fingerprint=invalid)


def test_true_lobo_identity_and_runners_bind_evaluation_authority(
    tiny_target,
    tmp_path,
    monkeypatch,
):
    cfg, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    assert identity["evaluation_authority"] == {
        "mode": tl.SYNTHETIC_AUTHORITY_MODE,
        "context_rows_digest": dataset.rows_digest,
        "evaluator": tl.SYNTHETIC_EVALUATOR_ID,
    }
    with pytest.raises(Exception, match="disk-verified"):
        tl.build_run_identity(
            dataset=dataset,
            inventory=inventory,
            config={
                "path": "synthetic.yaml",
                "sha256": "c" * 64,
                "resolved_sha256": "e" * 64,
            },
            code_hashes={"runner.py": "b" * 64},
            git_commit="a" * 40,
            git_dirty=True,
            runtime_fingerprint=_runtime_binding(),
            thread_fingerprint={"worker_policy": "one"},
            representation_cache={
                "path": "synthetic-reps.pkl",
                "size_bytes": 1,
                "sha256": "7" * 64,
                "rep_version": "synthetic-v1",
            },
        )

    monkeypatch.setattr(
        tl,
        "require_disk_verified_scientific_context",
        lambda value: value,
    )
    with pytest.raises(tl.TrueLoboError, match="injected evaluator"):
        tl.run_true_lobo(
            cfg,
            dataset,
            identity,
            _reference(identity),
            output_path=tmp_path / "forbidden.json",
            checkpoint_root=tmp_path / "forbidden.checkpoints",
            n_jobs=1,
            evaluator=_fake_evaluator(identity, []),
        )
    with pytest.raises(tl.TrueLoboError, match="injected clock"):
        tl.run_true_lobo(
            cfg,
            dataset,
            identity,
            _reference(identity),
            output_path=tmp_path / "forbidden-clock.json",
            checkpoint_root=tmp_path / "forbidden-clock.checkpoints",
            n_jobs=1,
            clock=lambda: 0.0,
        )

    forged = copy.deepcopy(identity)
    forged["evaluation_authority"] = {
        "mode": tl.DISK_AUTHORITY_MODE,
        "context_rows_digest": dataset.rows_digest,
        "evaluator": tl.PRODUCTION_EVALUATOR_ID,
    }
    forged["run_id"] = tl.canonical_hash(tl._run_id_material(forged))
    forged["self_hash"] = tl.artifact_self_hash(forged)
    tl.validate_run_identity(forged)
    with pytest.raises(tl.CheckpointError, match="authority"):
        tl._validate_context_against_run_identity(dataset, forged)


def test_run_id_is_relocation_stable_while_absolute_paths_remain_display_only(
        tiny_target, tmp_path):
    _, dataset, inventory = tiny_target
    first = _identity(
        dataset,
        inventory,
        config_path="/checkout-a/configs/default.yaml",
        cache_path="/checkout-a/data/reps.sqlite3",
    )
    relocated = _identity(
        dataset,
        inventory,
        config_path="/different/root/configs/default.yaml",
        cache_path="/different/root/data/reps.sqlite3",
    )
    assert first["run_id"] == relocated["run_id"]
    assert first["self_hash"] != relocated["self_hash"]
    assert first["config"]["display_path"] != relocated["config"]["display_path"]
    assert (
        first["representation_cache"]["display_path"]
        != relocated["representation_cache"]["display_path"]
    )
    tl.validate_run_identity(first)
    tl.validate_run_identity(relocated)

    # A moved checkpoint tree remains resumable because RUN.json compares the
    # scientific content identity, not checkout-local display paths.
    root = tmp_path / "relocated.checkpoints"
    tl.CheckpointStore(root, first)
    tl.CheckpointStore(root, relocated)


def test_runtime_ledger_tamper_fails_closed(tiny_target, tmp_path):
    _, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    store = tl.CheckpointStore(tmp_path / "checkpoints", identity)
    store.add_runtime("A0", 1.0, 2)
    runtime_path = tmp_path / "checkpoints" / "RUNTIME.json"
    tampered = load_strict(runtime_path)
    tampered["n_jobs_history"] = [999]
    dump_strict(tampered, runtime_path)
    with pytest.raises(tl.CheckpointError, match="runtime ledger mismatch"):
        store.load_runtime()


def _probability_for(cell, fold_index):
    predictions = {
        "A0": (0, 1, 2, 0),
        "A4": (0, 0, 2, 2),
        "A1": (0, 0, 1, 2),
    }
    return predictions[cell][fold_index]


def _evaluated(identity, cell, fold):
    work_ids = [item["work_id"] for item in identity["work_universe"]]
    train_ids = [work_id for work_id in work_ids if work_id != fold["work_id"]]
    singleton_ids = [item["work_id"] for item in identity["singleton_train_only"]]
    pred = _probability_for(cell, int(fold["fold_index"]))
    probabilities = np.full(len(identity["probability_class_order"]), 0.1, dtype=float)
    probabilities[pred] = 0.8
    true_label = int(fold["true_label"])
    rank = int((probabilities >= probabilities[true_label]).sum())
    return {
        "fold_index": int(fold["fold_index"]),
        "work_id": fold["work_id"],
        "true_label": true_label,
        "true_author": fold["true_author"],
        "split": {
            "n_train_chunks": int(identity["dataset"]["n_chunks"] - fold["n_chunks"]),
            "n_test_chunks": int(fold["n_chunks"]),
            "n_train_works": len(work_ids) - 1,
            "n_train_authors": len(identity["probability_class_order"]),
            "train_work_inventory_sha256": tl.canonical_hash(train_ids),
            "singleton_work_ids_present": singleton_ids,
        },
        "result": {
            "pred_label": pred,
            "pred_author": identity["probability_class_order"][pred]["author"],
            "rank": rank,
            "correct": pred == true_label,
            "probabilities": probabilities.tolist(),
        },
        "timing": {
            "fit_wall_seconds": 2.0,
            "predict_wall_seconds": 1.0,
            "total_wall_seconds": 3.0,
            "fit_cpu_seconds": 1.5,
            "predict_cpu_seconds": 0.5,
            "total_cpu_seconds": 2.0,
            "peak_rss_kib": 1000 + int(fold["fold_index"]),
        },
    }


def _reference(identity):
    records = []
    for fold in identity["tested_inventory"]:
        evaluated = _evaluated(identity, "A0", fold)
        result = evaluated["result"]
        records.append({
            "fold_index": int(fold["fold_index"]),
            "work_id": fold["work_id"],
            "true_label": int(fold["true_label"]),
            "true_author": fold["true_author"],
            "pred_label": result["pred_label"],
            "pred_author": result["pred_author"],
            "rank": result["rank"],
            "correct": result["correct"],
        })
    correct = sum(record["correct"] for record in records)
    return {
        "path": "synthetic-reference.txt",
        "sha256": "f" * 64,
        "expected_correct": correct,
        "expected_tested": len(records),
        "expected_accuracy": correct / len(records),
        "records": records,
        "records_sha256": tl.canonical_hash(records),
    }


def _fake_evaluator(identity, calls):
    def evaluate(cfg, dataset, cell, fold):
        calls.append((cell, int(fold["fold_index"])))
        return _evaluated(identity, cell, fold)
    return evaluate


def _parallel_fake_evaluator(cfg, dataset, cell, fold):
    inventory = tl.derive_inventory(dataset)
    identity = _identity(dataset, inventory)
    return _evaluated(identity, cell, fold)


class _CounterClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        value = self.value
        self.value += 1.0
        return value


def test_inventory_and_real_fold_keep_singleton_train_only_and_average_once(tiny_target):
    cfg, dataset, inventory = tiny_target
    assert [item["label"] for item in inventory["metric_label_order"]] == [0, 2]
    assert [item["work_id"] for item in inventory["tested_inventory"]] == [
        "aa/a1", "aa/a2", "bb/b1", "bb/b2"]
    assert inventory["singleton_train_only"] == [
        {"label": 1, "author": "singleton", "work_id": "singleton/s1"}]

    registry = []
    observed = []
    all_works = {item["work_id"] for item in inventory["work_universe"]}
    for fold in inventory["tested_inventory"]:
        result = tl.evaluate_synthetic_true_lobo_fold(
            cfg, dataset, fold, lambda: _SpyEstimator(registry))
        observed.append(result)
        estimator = registry[-1]
        assert set(estimator.fit_groups) == all_works - {fold["work_id"]}
        assert "singleton/s1" in estimator.fit_groups
        assert set(estimator.fit_labels) == {0, 1, 2}
        assert {text.split(":", 1)[0] for text in estimator.predict_texts} == {fold["work_id"]}

    assert len(registry) == len({id(estimator) for estimator in registry}) == 4
    np.testing.assert_allclose(observed[0]["result"]["probabilities"], [0.7, 0.15, 0.15])
    np.testing.assert_allclose(observed[1]["result"]["probabilities"], [0.3, 0.6, 0.1])
    assert all(len(row["result"]["probabilities"]) == 3 for row in observed)


def test_full_width_predictions_average_over_frozen_tested_author_order(tiny_target):
    _, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    checkpoints = {}
    for fold in identity["tested_inventory"]:
        checkpoint = tl.build_checkpoint(identity, "A0", fold, _evaluated(identity, "A0", fold))
        checkpoints[int(fold["fold_index"])] = checkpoint
    cell = tl.assemble_cell(identity, "A0", checkpoints)
    truth = np.asarray([work["true_label"] for work in cell["works"]])
    pred = np.asarray([work["pred_label"] for work in cell["works"]])
    expected = f1_score(
        truth,
        pred,
        labels=[0, 2],
        average="macro",
        zero_division=0,
    )
    wrong_probability_universe = f1_score(
        truth,
        pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    assert cell["metrics"]["macro_f1"] == expected
    assert expected != wrong_probability_universe
    assert all(len(work["probabilities"]) == 3 for work in cell["works"])


def _write_reference(path, inventory, predictions):
    names = {item["label"]: item["author"] for item in inventory["probability_class_order"]}
    lines = ["=== LOBO: топ-кандидаты по каждой книге (leakage-free) ==="]
    for fold, pred in zip(inventory["tested_inventory"], predictions, strict=True):
        correct = pred == fold["true_label"]
        rank = 1 if correct else 3
        mark = "OK  " if correct else "MISS"
        book = fold["work_id"].split("/", 1)[1]
        lines.append(
            f"[{mark}] {fold['true_author']} / {book}  (rank истинного автора: {rank})")
        lines.append(
            f"        топ: {names[pred]} (0.800), {names[(pred + 1) % 3]} (0.100), "
            f"{names[(pred + 2) % 3]} (0.100)")
    path.write_text("\n".join(lines), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_is_hashed_before_strict_parse_and_binds_exact_order(tiny_target, tmp_path):
    _, _, inventory = tiny_target
    bad = tmp_path / "bad.txt"
    bad.write_text("not even parseable", encoding="utf-8")
    with pytest.raises(tl.A0ParityError, match="SHA256 mismatch"):
        tl.load_pinned_a0_reference(bad, inventory, expected_sha256="0" * 64)

    path = tmp_path / "reference.txt"
    predictions = [0, 1, 2, 0]
    digest = _write_reference(path, inventory, predictions)
    parsed = tl.load_pinned_a0_reference(
        path, inventory, expected_sha256=digest, expected_correct=2)
    assert [record["work_id"] for record in parsed["records"]] == [
        fold["work_id"] for fold in inventory["tested_inventory"]]
    assert [record["pred_label"] for record in parsed["records"]] == predictions

    reordered = copy.deepcopy(inventory)
    reordered["tested_inventory"] = list(reversed(reordered["tested_inventory"]))
    with pytest.raises(tl.A0ParityError, match="inventory mismatch"):
        tl.load_pinned_a0_reference(
            path, reordered, expected_sha256=digest, expected_correct=2)


def test_a0_reference_mismatch_blocks_all_a4_a1_scheduling(tiny_target, tmp_path):
    cfg, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    reference = _reference(identity)
    # Keep correctness/rank unchanged but alter the exact wrong top-1 identity.
    bad = copy.deepcopy(reference)
    row = bad["records"][1]
    row["pred_label"] = 2
    row["pred_author"] = "bb"
    bad["records_sha256"] = tl.canonical_hash(bad["records"])
    calls = []
    with pytest.raises(tl.A0ParityError, match="per-work frozen parity"):
        tl.run_synthetic_true_lobo(
            cfg,
            dataset,
            identity,
            bad,
            output_path=tmp_path / "blocked.json",
            checkpoint_root=tmp_path / "blocked.checkpoints",
            n_jobs=1,
            evaluator=_fake_evaluator(identity, calls),
            clock=lambda: 0.0,
        )
    assert calls == [("A0", index) for index in range(4)]
    assert not list((tmp_path / "blocked.checkpoints" / "A4").glob("*.json"))
    assert not list((tmp_path / "blocked.checkpoints" / "A1").glob("*.json"))


def test_context_mutation_is_rejected_before_checkpoint_creation(
    tiny_target,
    tmp_path,
):
    cfg, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    reference = _reference(identity)
    dataset.texts.setflags(write=True)
    dataset.texts[0] = "changed after run identity was constructed"
    dataset.texts.setflags(write=False)
    checkpoint_root = tmp_path / "mutated.checkpoints"
    with pytest.raises(Exception, match="receipt|changed"):
        tl.run_synthetic_true_lobo(
            cfg,
            dataset,
            identity,
            reference,
            output_path=tmp_path / "mutated.json",
            checkpoint_root=checkpoint_root,
            n_jobs=1,
            cells=["A0"],
            smoke_only=True,
            evaluator=_fake_evaluator(identity, []),
            clock=lambda: 0.0,
        )
    assert not checkpoint_root.exists()


def test_interrupted_resume_only_computes_missing_and_matches_uninterrupted_bytes(
        tiny_target, tmp_path):
    cfg, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    reference = _reference(identity)
    resumed_calls = []
    resumed_clock = _CounterClock()
    resume_root = tmp_path / "resume.checkpoints"
    resume_output = tmp_path / "resume.json"
    smoke = tl.run_synthetic_true_lobo(
        cfg,
        dataset,
        identity,
        reference,
        output_path=resume_output,
        checkpoint_root=resume_root,
        n_jobs=1,
        cells=["A0"],
        smoke_only=True,
        evaluator=_fake_evaluator(identity, resumed_calls),
        clock=resumed_clock,
    )
    assert smoke["completed_a0_checkpoints"] == 1
    assert resumed_calls == [("A0", 0)]
    resumed = tl.run_synthetic_true_lobo(
        cfg,
        dataset,
        identity,
        reference,
        output_path=resume_output,
        checkpoint_root=resume_root,
        n_jobs=1,
        evaluator=_fake_evaluator(identity, resumed_calls),
        clock=resumed_clock,
    )
    assert resumed_calls.count(("A0", 0)) == 1
    assert len(resumed_calls) == 12  # 1 smoke + 3 remaining A0 + 4 A4 + 4 A1

    uninterrupted_calls = []
    uninterrupted_clock = _CounterClock()
    uninterrupted_output = tmp_path / "uninterrupted.json"
    uninterrupted = tl.run_synthetic_true_lobo(
        cfg,
        dataset,
        identity,
        reference,
        output_path=uninterrupted_output,
        checkpoint_root=tmp_path / "uninterrupted.checkpoints",
        n_jobs=1,
        evaluator=_fake_evaluator(identity, uninterrupted_calls),
        clock=uninterrupted_clock,
    )
    assert len(uninterrupted_calls) == 12
    assert resumed == uninterrupted
    assert resume_output.read_bytes() == uninterrupted_output.read_bytes()
    assert resumed["self_hash"] == tl.artifact_self_hash(resumed)

    # Pure resume validates all 12 checkpoints but evaluates nothing and preserves exact bytes.
    before = resume_output.read_bytes()
    calls = []
    again = tl.run_synthetic_true_lobo(
        cfg,
        dataset,
        identity,
        reference,
        output_path=resume_output,
        checkpoint_root=resume_root,
        n_jobs=1,
        evaluator=_fake_evaluator(identity, calls),
        clock=resumed_clock,
    )
    assert calls == [] and again == resumed and resume_output.read_bytes() == before


def test_outer_parallel_generator_publishes_a_smoke_checkpoint(tiny_target, tmp_path):
    cfg, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    result = tl.run_synthetic_true_lobo(
        cfg,
        dataset,
        identity,
        _reference(identity),
        output_path=tmp_path / "parallel.json",
        checkpoint_root=tmp_path / "parallel.checkpoints",
        n_jobs=2,
        cells=["A0"],
        smoke_only=True,
        evaluator=_parallel_fake_evaluator,
        clock=lambda: 0.0,
    )
    assert result["completed_a0_checkpoints"] == 1
    store = tl.CheckpointStore(tmp_path / "parallel.checkpoints", identity)
    assert list(store.scan_cell("A0")) == [0]


def test_checkpoint_rejects_rehashed_semantic_tamper_extra_and_metadata_mismatch(
        tiny_target, tmp_path):
    _, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    store = tl.CheckpointStore(tmp_path / "checkpoints", identity)
    fold = identity["tested_inventory"][0]
    checkpoint = tl.build_checkpoint(identity, "A0", fold, _evaluated(identity, "A0", fold))
    path = store.save(checkpoint)

    tampered = copy.deepcopy(checkpoint)
    tampered["result"]["probabilities"] = [0.9, 0.9, 0.0]
    tampered["self_hash"] = tl.artifact_self_hash(tampered)
    dump_strict(tampered, path)
    with pytest.raises(tl.CheckpointError, match="sum to one"):
        store.scan_cell("A0")

    dump_strict(checkpoint, path)
    extra = tmp_path / "checkpoints" / "A0" / "9999-extra.json"
    dump_strict(checkpoint, extra)
    with pytest.raises(tl.CheckpointError, match="extra/conflicting"):
        store.scan_cell("A0")

    other = copy.deepcopy(identity)
    other["config"]["sha256"] = "9" * 64
    other["bindings"]["config_sha256"] = "9" * 64
    other["run_id"] = tl.canonical_hash(tl._run_id_material(other))
    other["self_hash"] = tl.artifact_self_hash(other)
    with pytest.raises(tl.CheckpointError, match="different run metadata"):
        tl.CheckpointStore(tmp_path / "checkpoints", other)


def test_concurrent_conflicting_checkpoint_writers_never_overwrite(
        tiny_target, tmp_path, monkeypatch):
    _, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    store = tl.CheckpointStore(tmp_path / "concurrent.checkpoints", identity)
    fold = identity["tested_inventory"][0]
    first = tl.build_checkpoint(
        identity,
        "A0",
        fold,
        _evaluated(identity, "A0", fold),
    )
    changed_evaluation = _evaluated(identity, "A0", fold)
    changed_evaluation["timing"].update(
        {
            "fit_wall_seconds": 4.0,
            "total_wall_seconds": 5.0,
            "fit_cpu_seconds": 3.0,
            "total_cpu_seconds": 3.5,
            "peak_rss_kib": changed_evaluation["timing"]["peak_rss_kib"] + 1,
        }
    )
    conflicting = tl.build_checkpoint(
        identity,
        "A0",
        fold,
        changed_evaluation,
    )
    assert first != conflicting

    barrier = threading.Barrier(2)
    real_link = tl.os.link

    def synchronized_link(source, target):
        barrier.wait(timeout=5)
        return real_link(source, target)

    monkeypatch.setattr(tl.os, "link", synchronized_link)
    outcomes = []

    def writer(checkpoint):
        try:
            outcomes.append(("ok", store.save(checkpoint)))
        except BaseException as exc:  # surfaced below
            outcomes.append(("error", exc))

    threads = [
        threading.Thread(target=writer, args=(checkpoint,))
        for checkpoint in (first, conflicting)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert [kind for kind, _value in outcomes].count("ok") == 1
    errors = [value for kind, value in outcomes if kind == "error"]
    assert len(errors) == 1
    assert isinstance(errors[0], tl.CheckpointError)
    path = store.path_for("A0", fold)
    published = load_strict(path)
    assert published in (first, conflicting)

    # A crash before a later create-if-absent commit cannot damage the winner.
    prior = path.read_bytes()
    monkeypatch.setattr(
        tl.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic pre-link crash")
        ),
    )
    with pytest.raises(OSError, match="pre-link crash"):
        store.save(first if published == conflicting else conflicting)
    assert path.read_bytes() == prior
    assert not list(path.parent.glob("*.tmp"))


def test_incomplete_assembly_fails_but_missing_checkpoint_is_pending(tiny_target, tmp_path):
    _, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    store = tl.CheckpointStore(tmp_path / "pending", identity)
    assert store.scan_cell("A0") == {}
    fold = identity["tested_inventory"][0]
    store.save(tl.build_checkpoint(identity, "A0", fold, _evaluated(identity, "A0", fold)))
    partial = store.scan_cell("A0")
    with pytest.raises(tl.CheckpointError, match="incomplete"):
        tl.assemble_cell(identity, "A0", partial)


def test_pairing_loao_and_primary_gate_boundaries_are_exact(tiny_target):
    _, dataset, inventory = tiny_target
    identity = _identity(dataset, inventory)
    cells = {}
    for cell in tl.CELL_ORDER:
        checkpoints = {}
        for fold in identity["tested_inventory"]:
            checkpoints[int(fold["fold_index"])] = tl.build_checkpoint(
                identity, cell, fold, _evaluated(identity, cell, fold))
        cells[cell] = tl.assemble_cell(identity, cell, checkpoints)
    one = tl.paired_analysis(
        cells["A1"], cells["A0"], iterations=100, seed=42,
        include_leave_one_author_out=True)
    two = tl.paired_analysis(
        cells["A1"], cells["A0"], iterations=100, seed=42,
        include_leave_one_author_out=True)
    assert one == two and one["self_hash"] == tl.artifact_self_hash(one)
    assert one["gains_count"] == 2 and one["losses_count"] == 1
    assert len(one["leave_one_author_out"]) == 2

    reordered = copy.deepcopy(cells["A0"])
    reordered["works"] = list(reversed(reordered["works"]))
    with pytest.raises(tl.TrueLoboError, match="paired inventory"):
        tl.paired_analysis(cells["A1"], reordered)

    boundary = -0.02
    assert tl.primary_a4_gate(boundary, 0.01) == "inconclusive"
    assert tl.primary_a4_gate(np.nextafter(boundary, np.inf), 0.01) == "noninferior"
    assert tl.primary_a4_gate(-0.05, np.nextafter(boundary, -np.inf)) == "inferior"
    with pytest.raises(tl.TrueLoboError, match="invalid interval"):
        tl.primary_a4_gate(0.01, -0.01)
