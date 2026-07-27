from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from stylo.domain.lobo_vnext import (
    ContentCandidate,
    ContentComponent,
    ContentComponentManifest,
    CorpusVNextManifest,
    InferenceSpec,
    LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION,
    ModelSpec,
    WorkIdentity,
    build_fold_manifest,
    build_inner_cv_plan,
    inventory_raw_files,
)
from stylo.eval import lobo_vnext as lv
from stylo.jsonio import dump_strict


@dataclass(frozen=True)
class Harness:
    root: Path
    corpus: CorpusVNextManifest
    content: ContentComponentManifest
    folds: object
    inner: object
    model: ModelSpec
    inference: InferenceSpec
    execution: dict


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _receipts() -> dict[str, dict]:
    return {
        "config": lv.build_identity_receipt(
            "config",
            {
                "verified": True,
                "drift_free": True,
                "config_sha256": "a" * 64,
            },
        ),
        "executable_sources": lv.build_identity_receipt(
            "executable_sources",
            {
                "verified": True,
                "drift_free": True,
                "worktree_clean": True,
                "closure_sha256": "b" * 64,
            },
        ),
        "dependencies": lv.build_identity_receipt(
            "dependencies",
            {"verified": True, "lock_sha256": "c" * 64},
        ),
        "runtime": lv.build_identity_receipt(
            "runtime",
            {
                "verified": True,
                "drift_free": True,
                "runtime_sha256": "d" * 64,
            },
        ),
        "thread_contract": lv.build_identity_receipt(
            "thread_contract",
            {
                "verified": True,
                "deterministic": True,
                "omp_num_threads": 1,
                "mkl_num_threads": 1,
                "openblas_num_threads": 1,
            },
        ),
    }


def _harness(
    tmp_path: Path,
    *,
    mode: str = "isolated",
    works_per_test_author: int = 2,
    singleton: bool = True,
    component_pairs: tuple[tuple[str, str], ...] = (),
    requires_inner_cv: bool = False,
    duplicate_pair: tuple[str, str] | None = None,
) -> Harness:
    root = tmp_path / "corpus"
    root.mkdir(parents=True)
    work_rows: list[tuple[str, str]] = []
    for author in ("aa", "bb"):
        for index in range(1, works_per_test_author + 1):
            work_rows.append((f"{author}/{author}{index}", author))
    if singleton:
        work_rows.append(("singleton/s1", "singleton"))

    works: list[WorkIdentity] = []
    for work_id, author_id in work_rows:
        filename = work_id.replace("/", "-") + ".txt"
        text = f"literal synthetic row {work_id} token-{len(works):03d}"
        if duplicate_pair and work_id == duplicate_pair[1]:
            text = f"literal synthetic row {duplicate_pair[0]} token-000"
        (root / filename).write_text(text, encoding="utf-8")
        works.append(
            WorkIdentity(
                work_id=work_id,
                author_id=author_id,
                edition_id=f"edition:{work_id}",
                source_id=f"source:{work_id}",
                work_kind="work",
                raw_paths=(filename,),
            )
        )

    component_by_work = {work.work_id: work.work_id for work in works}
    manual_candidates: list[ContentCandidate] = []
    for ordinal, (left, right) in enumerate(component_pairs):
        component_id = f"paired-{ordinal:02d}"
        component_by_work[left] = component_id
        component_by_work[right] = component_id
        manual_candidates.append(
            ContentCandidate(
                candidate_id=f"manual-{ordinal:02d}",
                left_work_id=left,
                right_work_id=right,
                edge_type="manual",
                origin="manual",
                disposition="same_component",
                evidence_sha256=_sha(f"manual evidence {left} {right}"),
            )
        )
    components = [
        ContentComponent(component_id, tuple(sorted(member_ids)))
        for component_id, member_ids in sorted(
            {
                component_id: [
                    work_id
                    for work_id, assigned in component_by_work.items()
                    if assigned == component_id
                ]
                for component_id in set(component_by_work.values())
            }.items()
        )
    ]
    content = ContentComponentManifest.build(
        automatic_candidate_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        works=works,
        components=components,
        candidates=manual_candidates,
    )
    corpus = CorpusVNextManifest.build(
        corpus_kind="synthetic_fixture",
        generation_id="synthetic-generation-001",
        approved_for_exploratory=True,
        owner_selected=False,
        raw_inventory=inventory_raw_files(root),
        author_ids=sorted({author for _, author in work_rows}),
        works=works,
        canonical_model_row_digest=_sha("unchanged-canonical-model-rows"),
        chunker_policy_version="synthetic-whole-file.v1",
        canonicalizer_policy_version="synthetic-identity-utf8.v1",
        content_policy_version=LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION,
        content_component_manifest_digest=content.self_hash,
    )
    model = ModelSpec.build(
        model_id=lv.SYNTHETIC_UNIFORM_ESTIMATOR,
        family="synthetic_contract_probe",
        features=["constant"],
        weighting="one_book_one_vote",
        hyperparameters={"distribution": "uniform"},
        seeds={"model": 0},
        requires_inner_cv=requires_inner_cv,
        inner_cv_splits=2 if requires_inner_cv else None,
        supports_component_aware_inner_cv=requires_inner_cv,
        approved_for_exploratory=True,
        owner_selected=False,
    )
    inference = InferenceSpec.build(
        primary_metric="book_accuracy",
        primary_uncertainty="author_clustered_percentile_bootstrap",
        secondary_metrics=["macro_f1", "top2", "per_author"],
        macro_f1_uncertainty="point_only",
        bootstrap_seed=17,
        bootstrap_iterations=40,
        confidence_level=0.95,
        approved_for_exploratory=True,
        owner_selected=False,
    )
    folds = build_fold_manifest(corpus, content, mode=mode)
    inner = build_inner_cv_plan(folds, corpus, content, model)
    representation = lv.build_representation_receipt(
        corpus,
        representation_policy_version="synthetic-raw-utf8.v1",
    )
    execution = lv.build_execution_spec(
        estimator_key=model.model_id,
        identity_receipts=_receipts(),
        representation_receipt=representation,
    )
    return Harness(
        root, corpus, content, folds, inner, model, inference, execution
    )


def _run(harness: Harness, output: Path, *, n_jobs: int = 1, **kwargs):
    return lv.run_lobo_vnext(
        corpus_root=harness.root,
        corpus_manifest=harness.corpus,
        content_manifest=harness.content,
        fold_manifest=harness.folds,
        inner_cv_plan=harness.inner,
        model_spec=harness.model,
        inference_spec=harness.inference,
        execution_spec=harness.execution,
        output_namespace=output,
        n_jobs=n_jobs,
        **kwargs,
    )


def test_serial_parallel_resume_are_byte_identical_and_keep_fixed_p_m(tmp_path):
    harness = _harness(tmp_path / "fixture")

    serial = _run(harness, tmp_path / "exploratory-serial", n_jobs=1)
    parallel = _run(harness, tmp_path / "exploratory-parallel", n_jobs=2)
    resumed = _run(harness, tmp_path / "exploratory-parallel", n_jobs=1)

    assert serial.run_id == parallel.run_id == resumed.run_id
    assert lv._canonical_bytes(serial.artifact) == lv._canonical_bytes(
        parallel.artifact
    )
    assert parallel.artifact == resumed.artifact
    assert resumed.computed_folds == 0
    assert resumed.resumed_folds == len(harness.folds.folds)
    assert "telemetry" not in serial.artifact
    assert serial.telemetry["scientific_result_hashed"] is False

    checkpoints = serial.artifact["checkpoints"]
    assert len(checkpoints) == len(harness.folds.folds)
    assert [item["result"]["work_id"] for item in checkpoints] == [
        fold.test_work_id for fold in harness.folds.folds
    ]
    assert all(item["result"]["chunk_count"] == 1 for item in checkpoints)
    assert serial.artifact["metrics"]["probability_class_order"] == [
        "aa",
        "bb",
        "singleton",
    ]
    assert serial.artifact["metrics"]["metric_label_order"] == ["aa", "bb"]
    # Uniform P: lowest index wins; the true label receives the worst tie rank.
    assert all(item["result"]["predicted_label"] == 0 for item in checkpoints)
    assert all(item["result"]["true_rank"] == 3 for item in checkpoints)
    assert serial.artifact["metrics"]["macro_f1"]["uncertainty"] == "point_only"
    assert (
        serial.artifact["metrics"]["primary_accuracy"]["method"]
        == "author_clustered_percentile_bootstrap"
    )


class _RecordingEstimator:
    def __init__(self, probability_order, records, fold):
        self.probability_order = probability_order
        self.records = records
        self.fold = fold
        self.classes_ = np.arange(len(probability_order), dtype=np.int64)

    def fit(self, texts, labels, *, groups, inner_splits):
        self.records.append(
            {
                "fold_id": self.fold.fold_id,
                "fit_groups": tuple(sorted(set(groups.tolist()))),
                "inner_splits": inner_splits,
            }
        )
        return self

    def predict_proba(self, texts):
        return np.full(
            (len(texts), len(self.probability_order)),
            1.0 / len(self.probability_order),
        )


def test_purged_outer_and_component_aware_inner_receipts_reach_fit_exactly(
    tmp_path,
):
    pairs = (("aa/aa1", "aa/aa2"), ("bb/bb1", "bb/bb2"))
    harness = _harness(
        tmp_path / "fixture",
        mode="purged",
        works_per_test_author=4,
        singleton=False,
        component_pairs=pairs,
        requires_inner_cv=True,
    )
    records = []

    def factory(model, fold):
        assert model is harness.model
        return _RecordingEstimator(
            fold.probability_class_order, records, fold
        )

    outcome = _run(
        harness,
        tmp_path / "exploratory-purged",
        factory=factory,
    )

    assert len(records) == len(harness.folds.folds)
    by_fold = {record["fold_id"]: record for record in records}
    component_by_work = harness.content.component_by_work
    for fold in harness.folds.folds:
        record = by_fold[fold.fold_id]
        assert record["fit_groups"] == fold.train_work_ids
        assert not (
            set(record["fit_groups"])
            & ({fold.test_work_id} | set(fold.purged_work_ids))
        )
        planned_inner = harness.inner.by_fold[fold.fold_id].splits
        assert record["inner_splits"] == planned_inner
        for inner in planned_inner:
            validation_components = set(inner.validation_component_ids)
            assert not any(
                component_by_work[work_id] in validation_components
                for work_id in inner.train_work_ids
            )
    assert all(
        checkpoint["split"]["fold_spec_sha256"]
        if "fold_spec_sha256" in checkpoint["split"]
        else checkpoint["fold_spec_sha256"]
        for checkpoint in outcome.artifact["checkpoints"]
    )


def test_prefit_raw_drift_blocks_representation_factory_and_fit(tmp_path):
    harness = _harness(tmp_path / "fixture")
    calls = {"representation": 0, "factory": 0, "fit": 0}

    def representation_builder(preflight):
        calls["representation"] += 1
        return preflight.execution_spec["representation_receipt"]

    def factory(model, fold):
        calls["factory"] += 1
        estimator = _RecordingEstimator(
            fold.probability_class_order, [], fold
        )
        original = estimator.fit

        def fit(*args, **kwargs):
            calls["fit"] += 1
            return original(*args, **kwargs)

        estimator.fit = fit
        return estimator

    first_path = harness.root / harness.corpus.raw_inventory[0].relative_path
    first_path.write_bytes(first_path.read_bytes() + b" ")
    with pytest.raises(lv.VNextPreflightError, match="raw corpus inventory"):
        _run(
            harness,
            tmp_path / "exploratory-rejected",
            factory=factory,
            representation_builder=representation_builder,
        )
    assert calls == {"representation": 0, "factory": 0, "fit": 0}
    assert not (tmp_path / "exploratory-rejected").exists()


def test_historical_model_gate_blocks_before_representation_factory_and_output(
    tmp_path,
):
    harness = _harness(tmp_path / "fixture")
    blocked_model = ModelSpec.build(
        model_id="A0",
        family="historical_a0",
        features=["constant"],
        weighting="one_book_one_vote",
        hyperparameters={},
        seeds={"model": 0},
        requires_inner_cv=False,
        inner_cv_splits=None,
        supports_component_aware_inner_cv=False,
        approved_for_exploratory=True,
        owner_selected=False,
    )
    blocked_inner = build_inner_cv_plan(
        harness.folds,
        harness.corpus,
        harness.content,
        blocked_model,
    )
    blocked_execution = lv.build_execution_spec(
        estimator_key=blocked_model.model_id,
        identity_receipts=_receipts(),
        representation_receipt=harness.execution["representation_receipt"],
    )
    blocked = replace(
        harness,
        inner=blocked_inner,
        model=blocked_model,
        execution=blocked_execution,
    )
    calls = {"representation": 0, "factory": 0}

    def representation_builder(preflight):
        calls["representation"] += 1
        return preflight.execution_spec["representation_receipt"]

    def factory(model, fold):
        calls["factory"] += 1
        return _RecordingEstimator(fold.probability_class_order, [], fold)

    output = tmp_path / "exploratory-historical-rejected"
    with pytest.raises(lv.VNextPreflightError, match="synthetic probe"):
        _run(
            blocked,
            output,
            factory=factory,
            representation_builder=representation_builder,
        )
    assert calls == {"representation": 0, "factory": 0}
    assert not output.exists()


def test_omitted_cross_component_literal_overlap_blocks_before_factory(tmp_path):
    harness = _harness(
        tmp_path / "fixture",
        duplicate_pair=("aa/aa1", "bb/bb1"),
    )
    calls = {"representation": 0, "factory": 0}

    def representation_builder(preflight):
        calls["representation"] += 1
        return preflight.execution_spec["representation_receipt"]

    def factory(model, fold):
        calls["factory"] += 1
        return _RecordingEstimator(fold.probability_class_order, [], fold)

    with pytest.raises(
        lv.VNextPreflightError,
        match="automatic content candidate preflight",
    ):
        _run(
            harness,
            tmp_path / "exploratory-overlap-rejected",
            factory=factory,
            representation_builder=representation_builder,
        )
    assert calls == {"representation": 0, "factory": 0}
    assert not (tmp_path / "exploratory-overlap-rejected").exists()


def test_raw_byte_mutation_changes_run_and_checkpoint_namespace(tmp_path):
    harness = _harness(tmp_path / "fixture")
    first = _run(harness, tmp_path / "exploratory-output")

    path = harness.root / harness.corpus.raw_inventory[0].relative_path
    path.write_bytes(path.read_bytes() + b" ")
    mutated_corpus = CorpusVNextManifest.build(
        corpus_kind=harness.corpus.corpus_kind,
        generation_id=harness.corpus.generation_id,
        approved_for_exploratory=True,
        owner_selected=False,
        raw_inventory=inventory_raw_files(harness.root),
        author_ids=harness.corpus.author_ids,
        works=harness.corpus.works,
        canonical_model_row_digest=(
            harness.corpus.canonical_model_row_digest
        ),
        chunker_policy_version=harness.corpus.chunker_policy_version,
        canonicalizer_policy_version=(
            harness.corpus.canonicalizer_policy_version
        ),
        content_policy_version=harness.corpus.content_policy_version,
        content_component_manifest_digest=harness.content.self_hash,
    )
    mutated_folds = build_fold_manifest(
        mutated_corpus, harness.content, mode="isolated"
    )
    mutated_inner = build_inner_cv_plan(
        mutated_folds,
        mutated_corpus,
        harness.content,
        harness.model,
    )
    mutated_execution = lv.build_execution_spec(
        estimator_key=harness.model.model_id,
        identity_receipts=_receipts(),
        representation_receipt=lv.build_representation_receipt(
            mutated_corpus,
            representation_policy_version="synthetic-raw-utf8.v1",
        ),
    )
    mutated = replace(
        harness,
        corpus=mutated_corpus,
        folds=mutated_folds,
        inner=mutated_inner,
        execution=mutated_execution,
    )
    second = _run(mutated, tmp_path / "exploratory-output")

    assert second.run_id != first.run_id
    assert (
        mutated_corpus.canonical_model_row_digest
        == harness.corpus.canonical_model_row_digest
    )
    assert (tmp_path / "exploratory-output" / first.run_id).is_dir()
    assert (tmp_path / "exploratory-output" / second.run_id).is_dir()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra="rehashed-extra"),
        lambda value: value["result"]["probabilities"].__setitem__(0, "1.0"),
        lambda value: value.__setitem__("fold_index", True),
        lambda value: value["split"].__setitem__("train_work_ids", {}),
        lambda value: value["result"].__setitem__("chunk_count", False),
    ],
)
def test_checkpoint_schema_rejects_rehashed_noncanonical_payloads(
    tmp_path, mutation
):
    harness = _harness(tmp_path / "fixture")
    outcome = _run(harness, tmp_path / "exploratory-output")
    checkpoint = copy.deepcopy(outcome.artifact["checkpoints"][0])
    mutation(checkpoint)
    checkpoint["self_hash"] = lv._self_hash(checkpoint)

    with pytest.raises(lv.VNextCheckpointError):
        lv.validate_vnext_checkpoint(
            checkpoint,
            identity=outcome.artifact["run_identity"],
            model_spec=harness.model,
            inner_cv_plan=harness.inner,
            fold_index=0,
            fold=harness.folds.folds[0],
        )


def test_extra_and_conflicting_checkpoint_files_block_resume(tmp_path):
    harness = _harness(tmp_path / "fixture")
    outcome = _run(harness, tmp_path / "exploratory-output")
    checkpoint_dir = (
        tmp_path
        / "exploratory-output"
        / outcome.run_id
        / "checkpoints"
        / harness.model.self_hash
    )
    (checkpoint_dir / "unexpected.json").write_text("{}", encoding="utf-8")
    representation_calls = 0

    def representation_builder(preflight):
        nonlocal representation_calls
        representation_calls += 1
        return preflight.execution_spec["representation_receipt"]

    with pytest.raises(lv.VNextCheckpointError, match="extra/conflicting"):
        _run(
            harness,
            tmp_path / "exploratory-output",
            representation_builder=representation_builder,
        )
    assert representation_calls == 0


def test_invalid_existing_final_blocks_before_representation_and_factory(tmp_path):
    harness = _harness(tmp_path / "fixture")
    outcome = _run(harness, tmp_path / "exploratory-output")
    final_path = outcome.artifact_path
    tampered = copy.deepcopy(outcome.artifact)
    tampered["metrics"]["primary_accuracy"]["point"] = 0.25
    tampered["self_hash"] = lv._self_hash(tampered)
    final_path.write_bytes(lv._canonical_file_bytes(tampered))
    calls = {"representation": 0, "factory": 0}

    def representation_builder(preflight):
        calls["representation"] += 1
        return preflight.execution_spec["representation_receipt"]

    def factory(model, fold):
        calls["factory"] += 1
        return _RecordingEstimator(fold.probability_class_order, [], fold)

    with pytest.raises(lv.VNextArtifactError):
        _run(
            harness,
            tmp_path / "exploratory-output",
            factory=factory,
            representation_builder=representation_builder,
        )
    assert calls == {"representation": 0, "factory": 0}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact.update(extra="forbidden"),
        lambda artifact: artifact["metrics"]["primary_accuracy"].__setitem__(
            "point", 0.25
        ),
        lambda artifact: artifact["metrics"]["macro_f1"].__setitem__(
            "point", "0.5"
        ),
        lambda artifact: artifact["checkpoints"][0]["result"].__setitem__(
            "correct", 1
        ),
    ],
)
def test_final_validator_recomputes_all_derived_values(tmp_path, mutation):
    harness = _harness(tmp_path / "fixture")
    artifact = copy.deepcopy(
        _run(harness, tmp_path / "exploratory-output").artifact
    )
    mutation(artifact)
    if type(artifact.get("checkpoints")) is list:
        for checkpoint in artifact["checkpoints"]:
            if type(checkpoint) is dict:
                checkpoint["self_hash"] = lv._self_hash(checkpoint)
    artifact["self_hash"] = lv._self_hash(artifact)

    with pytest.raises(lv.VNextArtifactError):
        lv.validate_vnext_final_artifact(artifact)


def test_legacy_projection_is_descriptive_and_never_resumable():
    legacy = {
        "schema": "b4_true_lobo_checkpoint_v2",
        "run_id": "a" * 64,
    }
    projection = lv.project_legacy_artifact_read_only(legacy)
    assert projection["resumable"] is False
    assert projection["scientific_evidence"] is False
    with pytest.raises(lv.VNextCheckpointError, match="read-only"):
        lv.reject_legacy_checkpoint_resume(legacy)


def test_file_oriented_entrypoint_strictly_loads_every_bound_spec(tmp_path):
    harness = _harness(tmp_path / "fixture")
    spec_dir = tmp_path / "specs"
    paths = {}
    for name, payload in {
        "corpus": harness.corpus.to_dict(),
        "content": harness.content.to_dict(),
        "fold": harness.folds.to_dict(),
        "inner": harness.inner.to_dict(),
        "model": harness.model.to_dict(),
        "inference": harness.inference.to_dict(),
        "execution": harness.execution,
    }.items():
        path = spec_dir / f"{name}.json"
        dump_strict(payload, path, sort_keys=True)
        paths[name] = path

    artifact = lv.run_lobo_vnext_from_specs(
        corpus_root=harness.root,
        corpus_manifest_path=paths["corpus"],
        content_manifest_path=paths["content"],
        fold_manifest_path=paths["fold"],
        inner_cv_plan_path=paths["inner"],
        model_spec_path=paths["model"],
        inference_spec_path=paths["inference"],
        execution_spec_path=paths["execution"],
        output_namespace=tmp_path / "exploratory-file-entrypoint",
        n_jobs=2,
    )

    assert artifact["schema_version"] == lv.FINAL_ARTIFACT_SCHEMA_VERSION
    assert artifact["confirmatory_authorized"] is False
    assert artifact["inner_cv_plan"]["self_hash"] == harness.inner.self_hash
