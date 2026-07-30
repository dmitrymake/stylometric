from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from stylo.config import load_config
from stylo.domain.lobo_vnext import (
    LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION,
    ContentComponent,
    ContentComponentManifest,
    InferenceSpec,
    WorkIdentity,
    build_corpus_vnext_manifest,
    build_fold_manifest,
    build_inner_cv_plan,
    canonical_sha256,
    inventory_raw_files,
)
from stylo.domain.lobo_vnext_approval import (
    DecisionBindings,
    ReviewedEvidence,
    build_owner_decision_record,
)
from stylo.domain.lobo_vnext_packet import (
    CanonicalRepresentationReceipt,
    CanonicalRowEntry,
    PacketFileEntry,
    R1AcquisitionBinding,
    R1PacketManifest,
)
from stylo.domain.lobo_vnext_policy import (
    AutomaticCandidateMechanisms,
    CandidateInventory,
    ChunkerPolicy,
    ContentPolicySpec,
    LiteralCandidateMechanism,
    RawByteIdentityPolicy,
    StrictUTF8Policy,
    VersionedTextPolicy,
    Word5ContainmentPolicy,
)
from stylo.domain.lobo_vnext_real import (
    REQUIRED_RECEIPT_KINDS,
    CampaignManifest,
    ModelRoleManifest,
    OutputNamespaceContract,
    RealCorpusExecutionSpec,
    RealExecutionBindings,
    inner_cv_receipt_subject_digest,
)
from stylo.eval import lobo_vnext_real as real
from stylo.eval.lobo_vnext_models import build_r1_model_spec
from stylo.eval.lobo_vnext_receipts import (
    DerivedObservation,
    build_independent_receipts,
    derive_config_and_adapter_observations,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _text_policy(label: str, disposition: str) -> VersionedTextPolicy:
    return VersionedTextPolicy.from_dict(
        {
            "policy_version": f"test-{label}.v1",
            "disposition": disposition,
            "policy_document_sha256": _digest(f"{label}-document"),
        },
        label=label,
    )


def _content_policy() -> ContentPolicySpec:
    return ContentPolicySpec.build(
        policy_id="test-r1-policy.v1",
        raw_byte_identity=RawByteIdentityPolicy.from_dict(
            {
                "policy_version": "test-literal-bytes.v1",
                "identity_fields": [
                    "relative_path",
                    "byte_size",
                    "sha256",
                ],
                "digest_algorithm": "sha256",
            }
        ),
        strict_utf8=StrictUTF8Policy.from_dict(
            {
                "policy_version": "test-strict-utf8.v1",
                "encoding": "utf-8",
                "errors": "strict",
                "bom_disposition": "reject",
            }
        ),
        canonical_row_policy=_text_policy(
            "canonical-row", "transform_versioned"
        ),
        chunker_policy=ChunkerPolicy.from_dict(
            {
                "policy_version": "stylo.sent-chunks.v1",
                "mode": "external_versioned",
                "policy_document_sha256": _digest("chunker"),
            }
        ),
        yo_e_policy=_text_policy("yo-e", "transform_versioned"),
        historical_orthography_policy=_text_policy(
            "historical", "transform_versioned"
        ),
        ocr_policy=_text_policy("ocr", "preserve"),
        markup_policy=_text_policy("markup", "transform_versioned"),
        automatic_candidates=AutomaticCandidateMechanisms(
            exact_duplicate=LiteralCandidateMechanism.from_dict(
                {
                    "policy_version": "test-exact.v1",
                    "comparison": "literal_bytes_equal",
                },
                label="exact",
                expected_comparison="literal_bytes_equal",
            ),
            literal_contains=LiteralCandidateMechanism.from_dict(
                {
                    "policy_version": "test-contains.v1",
                    "comparison": "literal_byte_subsequence",
                },
                label="contains",
                expected_comparison="literal_byte_subsequence",
            ),
            word5_containment=Word5ContainmentPolicy.from_dict(
                {
                    "policy_version": "stylo.overlap-contract.v2",
                    "shingle_size": 5,
                    "comparison": "asymmetric_containment",
                    "threshold": {"numerator": 9, "denominator": 10},
                    "threshold_boundary": "inclusive",
                    "min_shingles": 20,
                    "sample_size": 64,
                    "final_verification": (
                        "exact_intersection_authoritative"
                    ),
                }
            ),
        ),
        manual_disposition_required=True,
    )


@dataclass(frozen=True)
class Harness:
    packet_root: Path
    packet_manifest: R1PacketManifest
    corpus: object
    policy: ContentPolicySpec
    candidates: CandidateInventory
    content: ContentComponentManifest
    folds: object
    primary_inner: object
    baseline_inner: object
    primary: object
    baseline: object
    inference: InferenceSpec
    roles: ModelRoleManifest
    campaign: CampaignManifest
    execution: RealCorpusExecutionSpec
    owner: object
    representation: CanonicalRepresentationReceipt
    cfg: object
    observations: tuple[DerivedObservation, ...]


def _work(work_id: str) -> WorkIdentity:
    author, leaf = work_id.split("/", 1)
    return WorkIdentity.from_dict(
        {
            "work_id": work_id,
            "author_id": author,
            "edition_id": f"edition-{leaf}",
            "source_id": f"source-{leaf}",
            "work_kind": "work",
            "raw_paths": [f"{work_id}.txt"],
        }
    )


def _observation(
    kind: str,
    digest: str,
    *,
    evidence: str | None = None,
) -> DerivedObservation:
    return DerivedObservation(
        kind=kind,
        derivation_version="stylo.test-live-derivation.v1",
        digest=digest,
        evidence_digest=evidence or digest,
        observation_count=1,
    ).validate()


def _harness(tmp_path: Path) -> Harness:
    raw_root = tmp_path / "raw"
    packet_root = tmp_path / "packet"
    works = tuple(
        _work(work_id)
        for work_id in (
            "a/a1",
            "a/a2",
            "b/b1",
            "b/b2",
        )
    )
    for work in works:
        path = raw_root / work.raw_paths[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"literal source for {work.work_id}\n", encoding="utf-8"
        )
    policy = _content_policy()
    raw_inventory = inventory_raw_files(raw_root)
    generation_id = _digest("acquisition-generation")
    content = ContentComponentManifest.build(
        automatic_candidate_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        works=works,
        components=tuple(
            ContentComponent(
                f"component-{work.work_id}", (work.work_id,)
            )
            for work in works
        ),
        candidates=(),
    )
    entries: list[CanonicalRowEntry] = []
    for work in works:
        text = f"canonical text for {work.work_id}"
        relative = f"canonical_rows/{work.work_id}/000000.txt"
        path = packet_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        source_bytes = (raw_root / work.raw_paths[0]).read_bytes()
        encoded = text.encode("utf-8")
        entries.append(
            CanonicalRowEntry.from_dict(
                {
                    "row_id": f"{work.work_id}#000000",
                    "relative_path": relative,
                    "work_id": work.work_id,
                    "author_id": work.author_id,
                    "ordinal": 0,
                    "source_relative_path": work.raw_paths[0],
                    "source_raw_sha256": hashlib.sha256(
                        source_bytes
                    ).hexdigest(),
                    "canonical_byte_size": len(encoded),
                    "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
                    "word_count": len(text.split()),
                }
            )
        )
    entries.sort(key=lambda row: (row.work_id, row.ordinal, row.relative_path))
    model_row_digest = canonical_sha256(
        [entry.to_dict() for entry in entries]
    )
    corpus = build_corpus_vnext_manifest(
        raw_root,
        corpus_kind="real_corpus",
        generation_id=generation_id,
        approved_for_exploratory=True,
        owner_selected=True,
        author_ids=("a", "b"),
        works=works,
        canonical_model_row_digest=model_row_digest,
        chunker_policy_version="stylo.sent_chunks/v1",
        canonicalizer_policy_version="stylo.clean/v1",
        content_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        content_component_manifest_digest=content.self_hash,
    )
    representation = CanonicalRepresentationReceipt.build(
        generation_id=corpus.generation_id,
        corpus_manifest_sha256=corpus.self_hash,
        canonicalizer_policy_document_sha256=_digest("canonicalizer"),
        chunker_policy_document_sha256=_digest("chunker"),
        rows=entries,
    )
    candidates = CandidateInventory.build(
        generation_id=corpus.generation_id,
        work_identity_catalog_digest=canonical_sha256(
            [work.to_dict() for work in works]
        ),
        raw_inventory_digest=canonical_sha256(
            [entry.to_dict() for entry in corpus.raw_inventory]
        ),
        content_policy_spec_digest=policy.self_hash,
        included_work_ids=tuple(work.work_id for work in works),
        candidates=(),
    )
    folds = build_fold_manifest(corpus, content, mode="isolated")
    cfg = load_config()
    primary = build_r1_model_spec(role="primary", cfg=cfg)
    baseline = build_r1_model_spec(role="baseline", cfg=cfg)
    primary_inner = build_inner_cv_plan(
        folds, corpus, content, primary
    )
    baseline_inner = build_inner_cv_plan(
        folds, corpus, content, baseline
    )
    inference = InferenceSpec.build(
        primary_metric="book_accuracy",
        primary_uncertainty="author_clustered_percentile_bootstrap",
        secondary_metrics=("macro_f1", "top2", "per_author"),
        macro_f1_uncertainty="point_only",
        bootstrap_seed=42,
        bootstrap_iterations=40,
        confidence_level=0.95,
        approved_for_exploratory=True,
        owner_selected=True,
    )
    roles = ModelRoleManifest.build(
        primary_model_spec=primary,
        baseline_model_spec=baseline,
        primary_inner_cv_plan=primary_inner,
        baseline_inner_cv_plan=baseline_inner,
    )
    campaign = CampaignManifest.build(
        campaign_id="test-r1-bounded-exploratory",
        fold_manifest_digest=folds.self_hash,
        inference_spec_digest=inference.self_hash,
        model_role_manifest=roles,
    )
    acquisition_binding = R1AcquisitionBinding.build(
        generation_id=generation_id,
        acquisition_manifest_self_hash=_digest("acquisition-manifest"),
        acquisition_receipt_self_hash=_digest("acquisition-receipt"),
        selected_audit_file_sha256=_digest("selected-audit-file"),
        selected_audit_self_hash=_digest("selected-audit-self"),
        raw_inventory_digest=canonical_sha256(
            [entry.to_dict() for entry in raw_inventory]
        ),
        work_identity_catalog_digest=canonical_sha256(
            [work.to_dict() for work in works]
        ),
        upstream_excluded_work_ids=(
            "serafimovich/у_нас_и_у_них",
            "sevsky/дон_на_костылях",
            "turgenev/записки_охотника",
        ),
        content_policy_spec_digest=policy.self_hash,
        post_selection_candidate_inventory_sha256=candidates.self_hash,
        work_count=len(works),
        author_count=len({work.author_id for work in works}),
    )
    packet_manifest = R1PacketManifest.build(
        acquisition_binding=acquisition_binding,
        candidate_inventory_sha256=candidates.self_hash,
        corpus_manifest_sha256=corpus.self_hash,
        content_component_manifest_sha256=content.self_hash,
        fold_manifest_sha256=folds.self_hash,
        primary_model_spec_sha256=primary.self_hash,
        baseline_model_spec_sha256=baseline.self_hash,
        inference_spec_sha256=inference.self_hash,
        primary_inner_cv_plan_sha256=primary_inner.self_hash,
        baseline_inner_cv_plan_sha256=baseline_inner.self_hash,
        model_role_manifest_sha256=roles.self_hash,
        campaign_manifest_sha256=campaign.self_hash,
        representation_receipt_sha256=representation.self_hash,
        files=(
            PacketFileEntry(
                entries[0].relative_path,
                entries[0].canonical_byte_size,
                entries[0].canonical_sha256,
            ),
        ),
    )
    config_obs, primary_obs, baseline_obs = (
        derive_config_and_adapter_observations(
            cfg=cfg,
            primary_model_spec=primary,
            baseline_model_spec=baseline,
        )
    )
    subject = {
        "packet_selection": packet_manifest.self_hash,
        "raw_inventory": canonical_sha256(
            [entry.to_dict() for entry in corpus.raw_inventory]
        ),
        "canonical_model_rows": corpus.canonical_model_row_digest,
        "content_candidates": candidates.self_hash,
        "content_components": content.self_hash,
        "folds": folds.self_hash,
        "inner_cv": inner_cv_receipt_subject_digest(
            primary_inner_cv_plan_digest=primary_inner.self_hash,
            baseline_inner_cv_plan_digest=baseline_inner.self_hash,
        ),
        "config": config_obs.digest,
        "primary_model_adapter": primary_obs.digest,
        "baseline_model_adapter": baseline_obs.digest,
        "executable_sources": _digest("sources"),
        "dependencies": _digest("dependencies"),
        "runtime": _digest("runtime"),
        "thread_contract": _digest("threads"),
        "representation": representation.self_hash,
    }
    exact_observations = {
        config_obs.kind: config_obs,
        primary_obs.kind: primary_obs,
        baseline_obs.kind: baseline_obs,
    }
    observations = tuple(
        exact_observations.get(kind)
        or _observation(kind, subject[kind])
        for kind in REQUIRED_RECEIPT_KINDS
    )
    bindings = RealExecutionBindings(
        packet_manifest_digest=packet_manifest.self_hash,
        content_policy_spec_digest=policy.self_hash,
        candidate_inventory_digest=candidates.self_hash,
        corpus_manifest_digest=corpus.self_hash,
        content_component_manifest_digest=content.self_hash,
        fold_manifest_digest=folds.self_hash,
        primary_inner_cv_plan_digest=primary_inner.self_hash,
        baseline_inner_cv_plan_digest=baseline_inner.self_hash,
        primary_model_spec_digest=primary.self_hash,
        baseline_model_spec_digest=baseline.self_hash,
        model_role_manifest_digest=roles.self_hash,
        inference_spec_digest=inference.self_hash,
        campaign_manifest_digest=campaign.self_hash,
        config_digest=config_obs.digest,
    )
    execution = RealCorpusExecutionSpec.build(
        bindings=bindings,
        independent_receipts=build_independent_receipts(observations),
        output_namespace=OutputNamespaceContract.build(
            namespace_id="test-r1"
        ),
    )
    owner = build_owner_decision_record(
        decision_id="test-r1-owner-decision",
        decision_revision=1,
        decision_date="2026-07-27",
        bindings=DecisionBindings(
            corpus_manifest_digest=corpus.self_hash,
            content_component_manifest_digest=content.self_hash,
            policy_manifest_digest=policy.self_hash,
            fold_manifest_digest=folds.self_hash,
            campaign_manifest_digest=campaign.self_hash,
            model_role_manifest_digest=roles.self_hash,
            inference_spec_digest=inference.self_hash,
            execution_spec_digest=execution.self_hash,
        ),
        reviewed_evidence=(
            ReviewedEvidence(
                "test/evidence.json", _digest("reviewed-evidence")
            ),
        ),
        affected_contract_versions=("stylo.test-r1.v1",),
    )
    return Harness(
        packet_root,
        packet_manifest,
        corpus,
        policy,
        candidates,
        content,
        folds,
        primary_inner,
        baseline_inner,
        primary,
        baseline,
        inference,
        roles,
        campaign,
        execution,
        owner,
        representation,
        cfg,
        observations,
    )


class _Estimator:
    def __init__(self, role, fold, calls):
        self.role = role
        self.fold = fold
        self.calls = calls
        self.classes_ = np.arange(
            len(fold.probability_class_order), dtype=np.int64
        )

    def fit(self, texts, labels, *, groups, inner_splits):
        self.calls.append(
            (
                "fit",
                self.role,
                self.fold.fold_id,
                tuple(sorted(set(groups.tolist()))),
                inner_splits,
            )
        )
        assert set(groups.tolist()) == set(self.fold.train_work_ids)
        assert inner_splits == ()
        assert set(labels.tolist()) == set(range(len(self.classes_)))
        return self

    def predict_proba(self, texts):
        probabilities = np.full(
            (len(texts), len(self.classes_)),
            1.0 / len(self.classes_),
            dtype=np.float64,
        )
        if self.role == "primary":
            for index, text in enumerate(texts.tolist()):
                author = str(text).split()[-1].split("/", 1)[0]
                label = self.fold.probability_class_order.index(author)
                probabilities[index] = 0.1
                probabilities[index, label] = 0.9
        return probabilities


def _factories(harness, calls):
    return {
        role: (
            lambda model, fold, role=role: _Estimator(role, fold, calls)
        )
        for role in ("primary", "baseline")
    }


def _output(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "docs"
        / "exploratory"
        / "lobo_vnext"
        / "real_corpus"
    )


def _run(
    harness: Harness,
    output: Path,
    *,
    n_jobs: int,
    calls: list,
    **kwargs,
):
    return real.run_lobo_vnext_real(
        packet_root=harness.packet_root,
        packet_manifest=harness.packet_manifest,
        corpus_manifest=harness.corpus,
        content_policy_spec=harness.policy,
        candidate_inventory=harness.candidates,
        content_manifest=harness.content,
        fold_manifest=harness.folds,
        primary_inner_cv_plan=harness.primary_inner,
        baseline_inner_cv_plan=harness.baseline_inner,
        primary_model_spec=harness.primary,
        baseline_model_spec=harness.baseline,
        inference_spec=harness.inference,
        model_role_manifest=harness.roles,
        campaign_manifest=harness.campaign,
        execution_spec=harness.execution,
        owner_decision=harness.owner,
        representation_receipt=harness.representation,
        cfg=harness.cfg,
        observations=harness.observations,
        output_namespace=output,
        n_jobs=n_jobs,
        _test_factory_map=_factories(harness, calls),
        **kwargs,
    )


def _run_root_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fresh_imports_preserve_evaluator_and_domain_boundaries():
    checks = (
        (
            "import sys; import stylo.eval.lobo_vnext_real; "
            "assert 'stylo.eval.lobo_vnext' not in sys.modules"
        ),
        (
            "import sys; import stylo.domain.lobo_vnext_packet; "
            "assert not any(name.startswith('stylo.eval') "
            "for name in sys.modules)"
        ),
    )
    for code in checks:
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )


def test_serial_parallel_resume_are_byte_identical_and_paired(tmp_path):
    harness = _harness(tmp_path / "fixture")
    serial_calls: list = []
    parallel_calls: list = []
    serial_output = _output(tmp_path / "serial")
    parallel_output = _output(tmp_path / "parallel")

    serial = _run(
        harness,
        serial_output,
        n_jobs=1,
        calls=serial_calls,
    )
    parallel = _run(
        harness,
        parallel_output,
        n_jobs=2,
        calls=parallel_calls,
    )
    serial_map = _run_root_bytes(serial_output / serial.run_id)
    parallel_before_resume = _run_root_bytes(
        parallel_output / parallel.run_id
    )
    resumed = _run(
        harness,
        parallel_output,
        n_jobs=1,
        calls=[],
    )
    parallel_after_resume = _run_root_bytes(
        parallel_output / resumed.run_id
    )

    assert serial.run_id == parallel.run_id == resumed.run_id
    assert serial.run_id == (
        "8c74fe68dc5e3277fafa5b292700148f3"
        "6211d416ea5e3e1efcad4a14a3b8459"
    )
    assert serial.artifact["self_hash"] == (
        "8b76c336513af3af0d5c9ea0518bc831"
        "f4392a27e28ef8a0d4f49b67d77ae6e5"
    )
    assert hashlib.sha256(serial.artifact_path.read_bytes()).hexdigest() == (
        "09d3d4b0c7d0030a74700d1e2783ae58"
        "b665ae18ced7e00ade60306ecce47091"
    )
    assert serial.artifact == parallel.artifact == resumed.artifact
    assert serial_map == parallel_before_resume == parallel_after_resume
    absolute_identity = copy.deepcopy(serial.artifact["run_identity"])
    absolute_identity["corpus"]["generation_id"] = "/host/path"
    run_material = {
        key: value
        for key, value in absolute_identity.items()
        if key not in {"run_id", "self_hash"}
    }
    absolute_identity["run_id"] = canonical_sha256(run_material)
    absolute_identity["self_hash"] = canonical_sha256(
        {
            key: value
            for key, value in absolute_identity.items()
            if key != "self_hash"
        }
    )
    with pytest.raises(
        real.RealVNextArtifactError,
        match="absolute host path",
    ):
        real.validate_real_run_identity(absolute_identity)
    with pytest.raises(real.RealVNextPreflightError):
        real._output_path_contract(tmp_path / "real-corpus", harness.execution)
    assert resumed.computed_checkpoints == 0
    assert resumed.resumed_checkpoints == 8
    assert "telemetry" not in serial.artifact
    assert serial.telemetry["scientific_result_hashed"] is False
    assert set(serial.artifact["checkpoints"]) == {"primary", "baseline"}
    for role in ("primary", "baseline"):
        assert [
            row["result"]["work_id"]
            for row in serial.artifact["checkpoints"][role]
        ] == [fold.test_work_id for fold in harness.folds.folds]
        assert all(
            row["inner_cv_plan_sha256"]
            == (
                harness.primary_inner.self_hash
                if role == "primary"
                else harness.baseline_inner.self_hash
            )
            for row in serial.artifact["checkpoints"][role]
        )
        assert (
            serial.artifact["metrics_by_role"][role]["macro_f1"][
                "uncertainty"
            ]
            == "point_only"
        )
    assert serial.artifact["paired_inference"]["shared_paired_draws"] is True
    assert len(serial_calls) == len(parallel_calls) == 8

def test_authorization_binding_blocks_before_rows_factory_fit_and_output(
    tmp_path,
):
    harness = _harness(tmp_path / "fixture")
    calls = {"rows": 0, "factory": 0}
    raw = harness.owner.to_dict()
    raw["bindings"]["campaign_manifest_digest"] = "0" * 64
    raw["self_hash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    from stylo.domain.lobo_vnext_approval import (
        ExploratoryOwnerDecisionRecord,
    )

    drifted = ExploratoryOwnerDecisionRecord.from_dict(raw)

    def loader(*args):
        calls["rows"] += 1
        raise AssertionError("row loader must not run")

    factories = {
        role: lambda *args: calls.__setitem__(
            "factory", calls["factory"] + 1
        )
        for role in ("primary", "baseline")
    }
    output = _output(tmp_path / "blocked")
    with pytest.raises(real.RealVNextPreflightError, match="owner"):
        real.run_lobo_vnext_real(
            packet_root=harness.packet_root,
            packet_manifest=harness.packet_manifest,
            corpus_manifest=harness.corpus,
            content_policy_spec=harness.policy,
            candidate_inventory=harness.candidates,
            content_manifest=harness.content,
            fold_manifest=harness.folds,
            primary_inner_cv_plan=harness.primary_inner,
            baseline_inner_cv_plan=harness.baseline_inner,
            primary_model_spec=harness.primary,
            baseline_model_spec=harness.baseline,
            inference_spec=harness.inference,
            model_role_manifest=harness.roles,
            campaign_manifest=harness.campaign,
            execution_spec=harness.execution,
            owner_decision=drifted,
            representation_receipt=harness.representation,
            cfg=harness.cfg,
            observations=harness.observations,
            output_namespace=output,
            n_jobs=1,
            representation_loader=loader,
            _test_factory_map=factories,
        )
    assert calls == {"rows": 0, "factory": 0}
    assert not output.exists()


def test_production_preflight_rejects_self_consistent_noncanonical_acquisition(
    tmp_path,
):
    harness = _harness(tmp_path / "fixture")
    with pytest.raises(
        real.RealVNextPreflightError,
        match="exact canonical selected-134 acquisition packet",
    ):
        real.preflight_lobo_vnext_real(
            packet_manifest=harness.packet_manifest,
            corpus_manifest=harness.corpus,
            content_policy_spec=harness.policy,
            candidate_inventory=harness.candidates,
            content_manifest=harness.content,
            fold_manifest=harness.folds,
            primary_inner_cv_plan=harness.primary_inner,
            baseline_inner_cv_plan=harness.baseline_inner,
            primary_model_spec=harness.primary,
            baseline_model_spec=harness.baseline,
            inference_spec=harness.inference,
            model_role_manifest=harness.roles,
            campaign_manifest=harness.campaign,
            execution_spec=harness.execution,
            owner_decision=harness.owner,
            representation_receipt=harness.representation,
            cfg=harness.cfg,
            observations=harness.observations,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(extra="forbidden"),
        lambda row: row["result"]["probabilities"].__setitem__(0, "0.9"),
        lambda row: row.__setitem__("fold_index", True),
        lambda row: row["result"].__setitem__("correct", 1),
        lambda row: row["split"].__setitem__("train_work_ids", {}),
    ],
)
def test_checkpoint_schema_rejects_rehashed_adversarial_payload(
    tmp_path, mutation
):
    harness = _harness(tmp_path / "fixture")
    outcome = _run(
        harness, _output(tmp_path / "run"), n_jobs=1, calls=[]
    )
    checkpoint = copy.deepcopy(
        outcome.artifact["checkpoints"]["primary"][0]
    )
    mutation(checkpoint)
    checkpoint["self_hash"] = canonical_sha256(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "self_hash"
        }
    )
    preflight = real.preflight_lobo_vnext_real(
        packet_manifest=harness.packet_manifest,
        corpus_manifest=harness.corpus,
        content_policy_spec=harness.policy,
        candidate_inventory=harness.candidates,
        content_manifest=harness.content,
        fold_manifest=harness.folds,
        primary_inner_cv_plan=harness.primary_inner,
        baseline_inner_cv_plan=harness.baseline_inner,
        primary_model_spec=harness.primary,
        baseline_model_spec=harness.baseline,
        inference_spec=harness.inference,
        model_role_manifest=harness.roles,
        campaign_manifest=harness.campaign,
        execution_spec=harness.execution,
        owner_decision=harness.owner,
        representation_receipt=harness.representation,
        cfg=harness.cfg,
        observations=harness.observations,
        _test_factory_injected=True,
    )
    with pytest.raises(real.RealVNextCheckpointError):
        real.validate_real_checkpoint(
            checkpoint,
            preflight=preflight,
            role="primary",
            fold_index=0,
            fold=harness.folds.folds[0],
        )


def test_extra_checkpoint_blocks_resume_before_row_loading(tmp_path):
    harness = _harness(tmp_path / "fixture")
    output = _output(tmp_path / "run")
    outcome = _run(harness, output, n_jobs=1, calls=[])
    checkpoint_dir = (
        output
        / outcome.run_id
        / "checkpoints"
        / "primary"
    )
    (checkpoint_dir / "unexpected.json").write_text("{}", encoding="utf-8")
    row_calls = 0

    def loader(*args):
        nonlocal row_calls
        row_calls += 1
        raise AssertionError("loader must not run")

    with pytest.raises(
        real.RealVNextCheckpointError, match="extra/conflicting"
    ):
        _run(
            harness,
            output,
            n_jobs=1,
            calls=[],
            representation_loader=loader,
        )
    assert row_calls == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact.update(extra="forbidden"),
        lambda artifact: artifact["metrics_by_role"]["primary"][
            "primary_accuracy"
        ].__setitem__("point", 0.25),
        lambda artifact: artifact["paired_inference"].__setitem__(
            "point", "0.5"
        ),
        lambda artifact: artifact["checkpoints"]["baseline"][0][
            "result"
        ].__setitem__("correct", 1),
    ],
)
def test_final_validator_recomputes_nested_results(tmp_path, mutation):
    harness = _harness(tmp_path / "fixture")
    artifact = copy.deepcopy(
        _run(
            harness, _output(tmp_path / "run"), n_jobs=1, calls=[]
        ).artifact
    )
    mutation(artifact)
    for role in ("primary", "baseline"):
        for checkpoint in artifact["checkpoints"][role]:
            checkpoint["self_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in checkpoint.items()
                    if key != "self_hash"
                }
            )
    artifact["self_hash"] = canonical_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "self_hash"
        }
    )
    with pytest.raises(real.RealVNextArtifactError):
        real.validate_real_final_artifact(artifact)
