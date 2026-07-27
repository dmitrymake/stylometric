from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from stylo.config import load_config
from stylo.domain.lobo_vnext import InferenceSpec, canonical_sha256
from stylo.domain.lobo_vnext_real import (
    REQUIRED_RECEIPT_KINDS,
    OutputNamespaceContract,
    RealCorpusExecutionSpec,
    RealExecutionBindings,
    inner_cv_receipt_subject_digest,
)
from stylo.eval import lobo_vnext_receipts as receipts
from stylo.eval.lobo_vnext_models import build_r1_model_spec
from stylo.release.source_inventory import (
    SourceInventoryReport,
    SourceSnapshot,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _inner_subject() -> tuple[str, str, str]:
    primary = _digest("primary-inner")
    baseline = _digest("baseline-inner")
    return (
        primary,
        baseline,
        inner_cv_receipt_subject_digest(
            primary_inner_cv_plan_digest=primary,
            baseline_inner_cv_plan_digest=baseline,
        ),
    )


def _observations() -> tuple[receipts.DerivedObservation, ...]:
    _primary_inner, _baseline_inner, inner_subject = _inner_subject()
    return tuple(
        receipts.DerivedObservation(
            kind=kind,
            derivation_version=f"test.{kind}.v1",
            digest=(
                inner_subject
                if kind == "inner_cv"
                else _digest(f"{kind}-digest")
            ),
            evidence_digest=_digest(f"{kind}-evidence"),
            observation_count=1,
        ).validate()
        for kind in REQUIRED_RECEIPT_KINDS
    )


def _execution(
    observations: tuple[receipts.DerivedObservation, ...],
) -> RealCorpusExecutionSpec:
    by_kind = {row.kind: row for row in observations}
    binding_values = {
        field.name: _digest(field.name)
        for field in dataclasses.fields(RealExecutionBindings)
    }
    binding_values.update(
        {
            "packet_manifest_digest": by_kind[
                "packet_selection"
            ].digest,
            "candidate_inventory_digest": by_kind[
                "content_candidates"
            ].digest,
            "content_component_manifest_digest": by_kind[
                "content_components"
            ].digest,
            "fold_manifest_digest": by_kind["folds"].digest,
            "config_digest": by_kind["config"].digest,
        }
    )
    # The two exact plan digests are selected so their canonical subject equals
    # the independently observed composite receipt.
    primary_inner, baseline_inner, _inner_subject_digest = _inner_subject()
    binding_values["primary_inner_cv_plan_digest"] = primary_inner
    binding_values["baseline_inner_cv_plan_digest"] = baseline_inner
    bindings = RealExecutionBindings(
        **binding_values
    )
    return RealCorpusExecutionSpec.build(
        bindings=bindings,
        independent_receipts=receipts.build_independent_receipts(
            observations
        ),
        output_namespace=OutputNamespaceContract.build(
            namespace_id="receipt-test"
        ),
    )


def test_live_observations_must_cover_exact_canonical_kind_order():
    rows = _observations()

    built = receipts.build_independent_receipts(rows)

    assert tuple(row.kind for row in built) == REQUIRED_RECEIPT_KINDS
    with pytest.raises(receipts.RealReceiptError, match="canonical order"):
        receipts.build_independent_receipts(rows[::-1])
    with pytest.raises(receipts.RealReceiptError, match="canonical order"):
        receipts.build_independent_receipts(rows[:-1])


def test_execution_receipts_are_rebuilt_and_compared_exactly():
    rows = _observations()
    execution = _execution(rows)

    assert receipts.assert_independent_receipts(execution, rows) is execution

    changed = list(rows)
    original = changed[0]
    changed[0] = dataclasses.replace(
        original, digest=_digest("changed-live-observation")
    )
    with pytest.raises(receipts.RealReceiptError, match="differ"):
        receipts.assert_independent_receipts(execution, changed)


def test_self_hashed_observation_uses_payload_hash_and_full_evidence():
    spec = InferenceSpec.build(
        primary_metric="book_accuracy",
        primary_uncertainty="author_clustered_percentile_bootstrap",
        secondary_metrics=("macro_f1", "top2", "per_author"),
        macro_f1_uncertainty="point_only",
        bootstrap_seed=42,
        bootstrap_iterations=10_000,
        confidence_level=0.95,
        approved_for_exploratory=True,
        owner_selected=True,
    )

    observed = receipts.observation_from_self_hashed(
        kind="folds", value=spec
    )

    assert observed.digest == spec.self_hash
    assert observed.evidence_digest == canonical_sha256(spec.to_dict())


def test_config_and_adapter_observations_rebuild_live_r1_contracts():
    cfg = load_config()
    primary = build_r1_model_spec(role="primary", cfg=cfg)
    baseline = build_r1_model_spec(role="baseline", cfg=cfg)

    config, primary_adapter, baseline_adapter = (
        receipts.derive_config_and_adapter_observations(
            cfg=cfg,
            primary_model_spec=primary,
            baseline_model_spec=baseline,
        )
    )

    assert (
        config.kind,
        primary_adapter.kind,
        baseline_adapter.kind,
    ) == (
        "config",
        "primary_model_adapter",
        "baseline_model_adapter",
    )
    assert primary_adapter.digest != baseline_adapter.digest


def test_thread_contract_requires_exact_env_and_live_single_thread_pools(
    monkeypatch,
):
    for key, value in receipts.REQUIRED_THREAD_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        receipts,
        "blas_thread_fingerprint",
        lambda: {
            "threadpools": [
                {
                    "internal_api": "openblas",
                    "version": "1",
                    "num_threads": 1,
                    "threading_layer": "pthreads",
                    "architecture": "x86",
                }
            ],
            "thread_env": dict(receipts.REQUIRED_THREAD_ENV),
        },
    )

    observed = receipts.derive_thread_observation()

    assert observed.kind == "thread_contract"
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(receipts.RealReceiptError, match="exact thread"):
        receipts.derive_thread_observation()


def test_thread_contract_rejects_unpinned_loaded_pool(monkeypatch):
    for key, value in receipts.REQUIRED_THREAD_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        receipts,
        "blas_thread_fingerprint",
        lambda: {
            "threadpools": [
                {
                    "internal_api": "openblas",
                    "version": "1",
                    "num_threads": 2,
                    "threading_layer": "pthreads",
                    "architecture": "x86",
                }
            ],
            "thread_env": dict(receipts.REQUIRED_THREAD_ENV),
        },
    )

    with pytest.raises(receipts.RealReceiptError, match="one thread"):
        receipts.derive_thread_observation()


def test_executable_closure_is_live_file_and_commit_derived(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "src/stylo").mkdir(parents=True)
    (tmp_path / "release").mkdir()
    (tmp_path / "README.md").write_text("reviewed\n", encoding="utf-8")
    source = tmp_path / "src/stylo/live.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest_path = tmp_path / "release/executable_sources.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    snapshot = SourceSnapshot(
        file_count=1,
        paths_sha256=_digest("path-set"),
        paths=("src/stylo/live.py",),
    )
    monkeypatch.setattr(
        receipts,
        "_git",
        lambda _root, *args: (
            "main\n"
            if args[:3] == ("symbolic-ref", "--quiet", "--short")
            else (
                ""
                if args[:2] == ("status", "--porcelain=v1")
                else (
                    "sha1\n"
                    if args[:2] == ("rev-parse", "--show-object-format")
                    else f"{'a' * 40}\n"
                )
            )
        ),
    )
    monkeypatch.setattr(
        receipts,
        "check_source_inventory",
        lambda *_args, **_kwargs: SourceInventoryReport(snapshot, ()),
    )
    monkeypatch.setattr(
        receipts,
        "load_strict",
        lambda _path: {"required_release_files": ["README.md"]},
    )

    first = receipts.derive_executable_source_observation(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = receipts.derive_executable_source_observation(tmp_path)

    assert first.kind == "executable_sources"
    assert first.observation_count == 3
    assert first.digest != second.digest


@pytest.mark.parametrize(
    ("object_format", "commit"),
    (
        ("sha1", "a" * 64),
        ("sha256", "a" * 40),
        ("sha512", "a" * 128),
        ("sha1", "A" * 40),
    ),
)
def test_executable_closure_rejects_git_object_format_length_mismatch(
    tmp_path: Path,
    monkeypatch,
    object_format: str,
    commit: str,
):
    monkeypatch.setattr(
        receipts,
        "_git",
        lambda _root, *args: (
            "main\n"
            if args[:3] == ("symbolic-ref", "--quiet", "--short")
            else (
                ""
                if args[:2] == ("status", "--porcelain=v1")
                else (
                    f"{object_format}\n"
                    if args[:2] == ("rev-parse", "--show-object-format")
                    else f"{commit}\n"
                )
            )
        ),
    )

    with pytest.raises(
        receipts.RealReceiptError,
        match="git object format|git commit",
    ):
        receipts.derive_executable_source_observation(tmp_path)


def test_git_object_id_accepts_exact_sha1_and_sha256_lengths():
    assert receipts._require_git_object_id(
        "a" * 40,
        object_format="sha1",
        label="git commit",
    ) == "a" * 40
    assert receipts._require_git_object_id(
        "b" * 64,
        object_format="sha256",
        label="git commit",
    ) == "b" * 64


def test_executable_closure_rejects_dirty_and_forbidden_branch(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        receipts,
        "_git",
        lambda _root, *args: (
            receipts.FORBIDDEN_PRIVATE_BRANCH + "\n"
            if args[:3] == ("symbolic-ref", "--quiet", "--short")
            else ""
        ),
    )
    with pytest.raises(receipts.RealReceiptError, match="forbidden"):
        receipts.derive_executable_source_observation(tmp_path)

    monkeypatch.setattr(
        receipts,
        "_git",
        lambda _root, *args: (
            "main\n"
            if args[:3] == ("symbolic-ref", "--quiet", "--short")
            else " M src/stylo/live.py\n"
        ),
    )
    with pytest.raises(receipts.RealReceiptError, match="clean"):
        receipts.derive_executable_source_observation(tmp_path)


def test_dependency_and_runtime_observations_are_path_free(monkeypatch):
    monkeypatch.setattr(
        receipts,
        "verify_installed_environment",
        lambda _root: {
            "schema_version": "test.environment.v1",
            "distributions": {"numpy": "1.0"},
        },
    )
    monkeypatch.setattr(
        receipts,
        "runtime_fingerprint",
        lambda: {"python": "3.11", "numpy": "1.0"},
    )

    dependency = receipts.derive_dependency_observation("/tmp/repo")
    runtime = receipts.derive_runtime_observation()

    assert dependency.kind == "dependencies"
    assert runtime.kind == "runtime"
    assert dependency.observation_count == 1
