"""Fail-closed evaluator for the owner-bound real-corpus LOBO-vNext campaign.

This module is intentionally separate from the synthetic v1 harness and from
the historical A0/A1/A4 runner.  It evaluates exactly two R1 roles on the same
frozen folds, probability universe, metric universe, and canonical text rows:

* ``primary`` -- ``stylo`` with work-balanced training;
* ``baseline`` -- ``char_cos`` with work-balanced training.

All authority, campaign, model, inference, receipt, and output contracts are
validated before canonical rows are loaded and before a model factory is
constructed.  Checkpoints are immutable role/fold records.  Scientific output
is deterministic; scheduling telemetry is returned separately and is never
part of the scientific artifact hash.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import pathlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..config import ConfigNode
from ..domain.lobo_vnext import (
    ContentComponentManifest,
    CorpusVNextManifest,
    FoldManifest,
    FoldSpec,
    InferenceSpec,
    InnerCVPlan,
    InnerFoldPlan,
    ModelSpec,
    canonical_sha256,
)
from ..domain.lobo_vnext_approval import ExploratoryOwnerDecisionRecord
from ..domain.lobo_vnext_packet import R1PacketManifest, VNextTextRow
from ..domain.lobo_vnext_policy import CandidateInventory, ContentPolicySpec
from ..domain.lobo_vnext_real import (
    BASELINE_ROLE,
    PRIMARY_ROLE,
    REQUIRED_RECEIPT_KINDS,
    CampaignManifest,
    ModelRoleManifest,
    RealCorpusExecutionSpec,
)
from ..domain.prediction_contract import (
    PREDICTION_CONTRACT_VERSION,
    PredictionContractError,
    stable_top1_and_worst_tie_rank,
    validate_probability_matrix,
)
from . import _lobo_vnext_shared as _shared
from .lobo_vnext_models import (
    make_r1_model_factory,
    r1_scientific_config_sha256,
    validate_r1_model_spec,
)
from .lobo_vnext_prepare import (
    R1_ACQUISITION_GENERATION_ID,
    R1_ACQUISITION_MANIFEST_SELF_HASH,
    R1_ACQUISITION_RECEIPT_SELF_HASH,
    R1_AUTHOR_COUNT,
    R1_RAW_INVENTORY_DIGEST,
    R1_SELECTED_AUDIT_FILE_SHA256,
    R1_SELECTED_AUDIT_SELF_HASH,
    R1_UPSTREAM_EXCLUDED_WORK_IDS,
    R1_WORK_COUNT,
    R1_WORK_IDENTITY_CATALOG_DIGEST,
)
from .lobo_vnext_receipts import (
    DerivedObservation,
    assert_independent_receipts,
    derive_config_and_adapter_observations,
    observations_by_kind,
)
from .metrics import accuracy
from .significance import paired_bootstrap_diff_clustered


REAL_RUN_IDENTITY_SCHEMA_VERSION = (
    "stylo.lobo-vnext.real-run-identity.v1"
)
REAL_CHECKPOINT_SCHEMA_VERSION = "stylo.lobo-vnext.real-checkpoint.v1"
REAL_FINAL_ARTIFACT_SCHEMA_VERSION = (
    "stylo.lobo-vnext.real-final-artifact.v1"
)
REAL_PAIRED_INFERENCE_SCHEMA_VERSION = (
    "stylo.lobo-vnext.real-paired-author-bootstrap.v1"
)
REAL_TELEMETRY_SCHEMA_VERSION = "stylo.lobo-vnext.real-telemetry.v1"

REAL_EXPLORATORY_STATUS = "exploratory_dry_run_only"
REAL_EXECUTION_MODE = "real_corpus"
REAL_EVALUATION_STRATEGY = "lobo"
PRODUCTION_FACTORY_CONTRACT = "r1_exact_live"
TEST_FACTORY_CONTRACT = "test_injected_nonproduction"

_ROLES = (PRIMARY_ROLE, BASELINE_ROLE)
_MODEL_BY_ROLE = {PRIMARY_ROLE: "stylo", BASELINE_ROLE: "char_cos"}


class RealLoboVNextError(ValueError):
    """Base error for the real-corpus exploratory evaluator."""


class RealVNextPreflightError(RealLoboVNextError):
    """The real campaign was rejected before factory construction or fit."""


class RealVNextCheckpointError(RealLoboVNextError):
    """A role/fold checkpoint namespace is malformed or conflicting."""


class RealVNextArtifactError(RealLoboVNextError):
    """A real-campaign scientific artifact is malformed or inconsistent."""


@dataclasses.dataclass(frozen=True)
class RealVNextPreflight:
    packet_manifest: R1PacketManifest
    corpus_manifest: CorpusVNextManifest
    content_policy_spec: ContentPolicySpec
    candidate_inventory: CandidateInventory
    content_manifest: ContentComponentManifest
    fold_manifest: FoldManifest
    primary_inner_cv_plan: InnerCVPlan
    baseline_inner_cv_plan: InnerCVPlan
    primary_model_spec: ModelSpec
    baseline_model_spec: ModelSpec
    inference_spec: InferenceSpec
    model_role_manifest: ModelRoleManifest
    campaign_manifest: CampaignManifest
    execution_spec: RealCorpusExecutionSpec
    owner_decision: ExploratoryOwnerDecisionRecord
    representation_receipt: Any
    cfg: ConfigNode
    observations: tuple[DerivedObservation, ...]
    factory_contract: str
    run_identity: dict[str, Any]
    rows: tuple[Any, ...] = ()

    @property
    def run_id(self) -> str:
        return self.run_identity["run_id"]

    @property
    def model_by_role(self) -> dict[str, ModelSpec]:
        return {
            PRIMARY_ROLE: self.primary_model_spec,
            BASELINE_ROLE: self.baseline_model_spec,
        }

    @property
    def inner_by_role(self) -> dict[str, InnerCVPlan]:
        return {
            PRIMARY_ROLE: self.primary_inner_cv_plan,
            BASELINE_ROLE: self.baseline_inner_cv_plan,
        }


@dataclasses.dataclass(frozen=True)
class RealVNextRunOutcome:
    artifact: dict[str, Any]
    artifact_path: pathlib.Path
    telemetry: dict[str, Any]
    computed_checkpoints: int
    resumed_checkpoints: int

    @property
    def run_id(self) -> str:
        return self.artifact["run_identity"]["run_id"]


def _manifest_dict(value: object, *, label: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise RealVNextPreflightError(
            f"{label} must be a strict self-hashed domain object"
        )
    try:
        raw = to_dict()
    except Exception as exc:
        raise RealVNextPreflightError(
            f"{label} cannot be projected canonically: {exc}"
        ) from exc
    if type(raw) is not dict:
        raise RealVNextPreflightError(
            f"{label}.to_dict() must return an exact object"
        )
    try:
        _shared._require_sha256(
            raw.get("self_hash"),
            path=f"{label}.self_hash",
            error_type=RealVNextPreflightError,
        )
    except Exception as exc:
        raise RealVNextPreflightError(str(exc)) from exc
    return raw


def _raw_inventory_digest(corpus_manifest: CorpusVNextManifest) -> str:
    return canonical_sha256(
        [entry.to_dict() for entry in corpus_manifest.raw_inventory]
    )


def _representation_type() -> type:
    try:
        from ..domain.lobo_vnext_packet import (
            CanonicalRepresentationReceipt,
        )
    except (ImportError, AttributeError) as exc:
        raise RealVNextPreflightError(
            "canonical representation packet contract is unavailable"
        ) from exc
    return CanonicalRepresentationReceipt


def _validate_representation_receipt(
    receipt: object,
    *,
    corpus_manifest: CorpusVNextManifest,
) -> object:
    receipt_type = _representation_type()
    if type(receipt) is not receipt_type:
        raise RealVNextPreflightError(
            "representation_receipt must be exactly "
            "CanonicalRepresentationReceipt"
        )
    try:
        raw = receipt.to_dict()
        restored = receipt_type.from_dict(raw)
        restored.validate(corpus_manifest=corpus_manifest)
    except Exception as exc:
        raise RealVNextPreflightError(
            f"canonical representation receipt is invalid: {exc}"
        ) from exc
    if restored != receipt:
        raise RealVNextPreflightError(
            "canonical representation receipt is noncanonical"
        )
    if raw.get("corpus_manifest_sha256") != corpus_manifest.self_hash:
        raise RealVNextPreflightError(
            "representation receipt/corpus manifest digest mismatch"
        )
    row_digest = raw.get("canonical_model_row_digest")
    if row_digest is None:
        row_digest = raw.get("canonical_row_inventory_digest")
    if row_digest != corpus_manifest.canonical_model_row_digest:
        raise RealVNextPreflightError(
            "representation receipt/canonical model-row digest mismatch"
        )
    return receipt


def _require_domain_types(
    *,
    packet_manifest: object,
    corpus_manifest: object,
    content_policy_spec: object,
    candidate_inventory: object,
    content_manifest: object,
    fold_manifest: object,
    primary_inner_cv_plan: object,
    baseline_inner_cv_plan: object,
    primary_model_spec: object,
    baseline_model_spec: object,
    inference_spec: object,
    model_role_manifest: object,
    campaign_manifest: object,
    execution_spec: object,
    owner_decision: object,
    cfg: object,
) -> None:
    expected = (
        ("packet_manifest", packet_manifest, R1PacketManifest),
        ("corpus_manifest", corpus_manifest, CorpusVNextManifest),
        ("content_policy_spec", content_policy_spec, ContentPolicySpec),
        ("candidate_inventory", candidate_inventory, CandidateInventory),
        ("content_manifest", content_manifest, ContentComponentManifest),
        ("fold_manifest", fold_manifest, FoldManifest),
        (
            "primary_inner_cv_plan",
            primary_inner_cv_plan,
            InnerCVPlan,
        ),
        (
            "baseline_inner_cv_plan",
            baseline_inner_cv_plan,
            InnerCVPlan,
        ),
        ("primary_model_spec", primary_model_spec, ModelSpec),
        ("baseline_model_spec", baseline_model_spec, ModelSpec),
        ("inference_spec", inference_spec, InferenceSpec),
        ("model_role_manifest", model_role_manifest, ModelRoleManifest),
        ("campaign_manifest", campaign_manifest, CampaignManifest),
        ("execution_spec", execution_spec, RealCorpusExecutionSpec),
        (
            "owner_decision",
            owner_decision,
            ExploratoryOwnerDecisionRecord,
        ),
        ("cfg", cfg, ConfigNode),
    )
    for label, value, expected_type in expected:
        if type(value) is not expected_type:
            raise RealVNextPreflightError(
                f"{label} must be exactly {expected_type.__name__}; "
                "legacy and duck-typed objects are rejected"
            )


def _assert_execution_bindings(
    *,
    packet_manifest: R1PacketManifest,
    corpus_manifest: CorpusVNextManifest,
    content_policy_spec: ContentPolicySpec,
    candidate_inventory: CandidateInventory,
    content_manifest: ContentComponentManifest,
    fold_manifest: FoldManifest,
    primary_inner_cv_plan: InnerCVPlan,
    baseline_inner_cv_plan: InnerCVPlan,
    primary_model_spec: ModelSpec,
    baseline_model_spec: ModelSpec,
    inference_spec: InferenceSpec,
    model_role_manifest: ModelRoleManifest,
    campaign_manifest: CampaignManifest,
    execution_spec: RealCorpusExecutionSpec,
    owner_decision: ExploratoryOwnerDecisionRecord,
    cfg: ConfigNode,
) -> None:
    """Validate every authority/campaign binding before canonical row access."""

    try:
        execution_spec.assert_owner_decision(owner_decision)
        execution_spec.assert_campaign(
            campaign_manifest=campaign_manifest,
            model_role_manifest=model_role_manifest,
            primary_model_spec=primary_model_spec,
            baseline_model_spec=baseline_model_spec,
            primary_inner_cv_plan=primary_inner_cv_plan,
            baseline_inner_cv_plan=baseline_inner_cv_plan,
        )
    except Exception as exc:
        raise RealVNextPreflightError(
            f"owner/execution/campaign binding failed: {exc}"
        ) from exc

    binding_checks = {
        "packet_manifest_digest": packet_manifest.self_hash,
        "content_policy_spec_digest": content_policy_spec.self_hash,
        "candidate_inventory_digest": candidate_inventory.self_hash,
        "corpus_manifest_digest": corpus_manifest.self_hash,
        "content_component_manifest_digest": content_manifest.self_hash,
        "fold_manifest_digest": fold_manifest.self_hash,
        "primary_inner_cv_plan_digest": primary_inner_cv_plan.self_hash,
        "baseline_inner_cv_plan_digest": baseline_inner_cv_plan.self_hash,
        "primary_model_spec_digest": primary_model_spec.self_hash,
        "baseline_model_spec_digest": baseline_model_spec.self_hash,
        "model_role_manifest_digest": model_role_manifest.self_hash,
        "inference_spec_digest": inference_spec.self_hash,
        "campaign_manifest_digest": campaign_manifest.self_hash,
        "config_digest": r1_scientific_config_sha256(cfg),
    }
    for field, expected in binding_checks.items():
        if getattr(execution_spec.bindings, field) != expected:
            raise RealVNextPreflightError(
                f"execution binding mismatch for {field}"
            )


def _assert_live_receipts(
    *,
    execution_spec: RealCorpusExecutionSpec,
    observations: Sequence[DerivedObservation],
    packet_manifest: R1PacketManifest,
    cfg: ConfigNode,
    primary_model_spec: ModelSpec,
    baseline_model_spec: ModelSpec,
    representation_receipt: object,
) -> tuple[DerivedObservation, ...]:
    if type(observations) not in (list, tuple):
        raise RealVNextPreflightError(
            "observations must be an exact list or tuple"
        )
    rows = tuple(observations)
    try:
        assert_independent_receipts(execution_spec, rows)
        by_kind = observations_by_kind(rows)
        live_config, live_primary, live_baseline = (
            derive_config_and_adapter_observations(
                cfg=cfg,
                primary_model_spec=primary_model_spec,
                baseline_model_spec=baseline_model_spec,
            )
        )
    except Exception as exc:
        raise RealVNextPreflightError(
            f"independent receipt validation failed: {exc}"
        ) from exc
    for live in (live_config, live_primary, live_baseline):
        observed = by_kind[live.kind]
        if observed != live:
            raise RealVNextPreflightError(
                f"live {live.kind} receipt differs from the execution packet"
            )
    if by_kind["packet_selection"].digest != packet_manifest.self_hash:
        raise RealVNextPreflightError(
            "live packet-selection receipt differs from the execution packet"
        )
    representation_hash = _manifest_dict(
        representation_receipt, label="representation_receipt"
    )["self_hash"]
    if by_kind["representation"].digest != representation_hash:
        raise RealVNextPreflightError(
            "live representation receipt differs from the execution packet"
        )
    return rows


def _validate_scientific_contracts(
    *,
    packet_manifest: R1PacketManifest,
    corpus_manifest: CorpusVNextManifest,
    content_policy_spec: ContentPolicySpec,
    candidate_inventory: CandidateInventory,
    content_manifest: ContentComponentManifest,
    fold_manifest: FoldManifest,
    primary_inner_cv_plan: InnerCVPlan,
    baseline_inner_cv_plan: InnerCVPlan,
    primary_model_spec: ModelSpec,
    baseline_model_spec: ModelSpec,
    inference_spec: InferenceSpec,
    campaign_manifest: CampaignManifest,
    cfg: ConfigNode,
    test_factory_injected: bool,
) -> None:
    try:
        packet_manifest.validate()
        content_policy_spec.validate()
        candidate_inventory.validate(
            content_policy_spec=content_policy_spec
        ).assert_resolved_for_component_manifest()
        corpus_manifest.validate(content_manifest=content_manifest)
        corpus_manifest.assert_exploratory_authorized(
            synthetic_fixture=False
        )
        primary_model_spec.assert_exploratory_authorized(
            synthetic_fixture=False
        )
        baseline_model_spec.assert_exploratory_authorized(
            synthetic_fixture=False
        )
        inference_spec.validate().assert_exploratory_authorized(
            synthetic_fixture=False
        )
        validate_r1_model_spec(primary_model_spec, cfg=cfg)
        validate_r1_model_spec(baseline_model_spec, cfg=cfg)
        fold_manifest.validate_against(corpus_manifest, content_manifest)
    except Exception as exc:
        raise RealVNextPreflightError(
            f"scientific manifest validation failed: {exc}"
        ) from exc

    packet_checks = {
        "candidate_inventory_sha256": candidate_inventory.self_hash,
        "corpus_manifest_sha256": corpus_manifest.self_hash,
        "content_component_manifest_sha256": content_manifest.self_hash,
        "fold_manifest_sha256": fold_manifest.self_hash,
        "primary_model_spec_sha256": primary_model_spec.self_hash,
        "baseline_model_spec_sha256": baseline_model_spec.self_hash,
        "inference_spec_sha256": inference_spec.self_hash,
        "primary_inner_cv_plan_sha256": primary_inner_cv_plan.self_hash,
        "baseline_inner_cv_plan_sha256": baseline_inner_cv_plan.self_hash,
        "model_role_manifest_sha256": (
            campaign_manifest.model_role_manifest_digest
        ),
        "campaign_manifest_sha256": campaign_manifest.self_hash,
    }
    for field, expected in packet_checks.items():
        if getattr(packet_manifest, field) != expected:
            raise RealVNextPreflightError(
                f"packet manifest binding mismatch for {field}"
            )
    binding = packet_manifest.acquisition_binding
    if (
        packet_manifest.generation_id != corpus_manifest.generation_id
        or binding.raw_inventory_digest
        != _raw_inventory_digest(corpus_manifest)
        or binding.work_identity_catalog_digest
        != canonical_sha256(
            [work.to_dict() for work in corpus_manifest.works]
        )
        or binding.content_policy_spec_digest
        != content_policy_spec.self_hash
        or binding.post_selection_candidate_inventory_sha256
        != candidate_inventory.self_hash
        or binding.work_count != len(corpus_manifest.works)
        or binding.author_count != len(corpus_manifest.author_ids)
        or set(binding.upstream_excluded_work_ids)
        & {work.work_id for work in corpus_manifest.works}
    ):
        raise RealVNextPreflightError(
            "packet acquisition binding differs from the "
            "scientific corpus"
        )

    if not test_factory_injected:
        author_support: dict[str, int] = {}
        for work in corpus_manifest.works:
            author_support[work.author_id] = (
                author_support.get(work.author_id, 0) + 1
            )
        if (
            packet_manifest.generation_id
            != R1_ACQUISITION_GENERATION_ID
            or binding.acquisition_manifest_self_hash
            != R1_ACQUISITION_MANIFEST_SELF_HASH
            or binding.acquisition_receipt_self_hash
            != R1_ACQUISITION_RECEIPT_SELF_HASH
            or binding.selected_audit_file_sha256
            != R1_SELECTED_AUDIT_FILE_SHA256
            or binding.selected_audit_self_hash
            != R1_SELECTED_AUDIT_SELF_HASH
            or binding.raw_inventory_digest != R1_RAW_INVENTORY_DIGEST
            or binding.work_identity_catalog_digest
            != R1_WORK_IDENTITY_CATALOG_DIGEST
            or binding.upstream_excluded_work_ids
            != R1_UPSTREAM_EXCLUDED_WORK_IDS
            or binding.work_count != R1_WORK_COUNT
            or binding.author_count != R1_AUTHOR_COUNT
            or packet_manifest.selected_work_count != R1_WORK_COUNT
            or len(corpus_manifest.works) != R1_WORK_COUNT
            or len(corpus_manifest.raw_inventory) != R1_WORK_COUNT
            or len(corpus_manifest.author_ids) != R1_AUTHOR_COUNT
            or not author_support
            or min(author_support.values()) < 2
            or candidate_inventory.candidates
            or content_manifest.candidates
            or any(
                len(component.work_ids) != 1
                for component in content_manifest.components
            )
        ):
            raise RealVNextPreflightError(
                "production real preflight requires the exact canonical "
                "selected-134 acquisition packet"
            )

    if (
        candidate_inventory.generation_id != corpus_manifest.generation_id
        or candidate_inventory.included_work_ids
        != tuple(work.work_id for work in corpus_manifest.works)
        or candidate_inventory.work_identity_catalog_digest
        != canonical_sha256(
            [work.to_dict() for work in corpus_manifest.works]
        )
        or candidate_inventory.raw_inventory_digest
        != _raw_inventory_digest(corpus_manifest)
    ):
        raise RealVNextPreflightError(
            "candidate inventory differs from the exact corpus "
            "generation/work/raw inventory"
        )
    if (
        corpus_manifest.content_component_manifest_digest
        != content_manifest.self_hash
    ):
        raise RealVNextPreflightError(
            "corpus/content component digest mismatch"
        )
    if (
        corpus_manifest.content_policy_version
        != content_manifest.automatic_candidate_policy_version
    ):
        raise RealVNextPreflightError(
            "corpus/content automatic candidate policy mismatch"
        )
    if (
        campaign_manifest.fold_manifest_digest != fold_manifest.self_hash
        or campaign_manifest.inference_spec_digest != inference_spec.self_hash
    ):
        raise RealVNextPreflightError(
            "campaign differs from the exact fold/inference manifests"
        )
    if (
        primary_inner_cv_plan.fold_manifest_digest
        != fold_manifest.self_hash
        or baseline_inner_cv_plan.fold_manifest_digest
        != fold_manifest.self_hash
        or primary_inner_cv_plan.content_component_manifest_digest
        != content_manifest.self_hash
        or baseline_inner_cv_plan.content_component_manifest_digest
        != content_manifest.self_hash
    ):
        raise RealVNextPreflightError(
            "model-specific inner plans do not bind the common folds/content"
        )
    for label, plan, model in (
        ("primary", primary_inner_cv_plan, primary_model_spec),
        ("baseline", baseline_inner_cv_plan, baseline_model_spec),
    ):
        if (
            plan.model_spec_digest != model.self_hash
            or tuple(row.fold_id for row in plan.plans)
            != tuple(fold.fold_id for fold in fold_manifest.folds)
            or any(row.splits for row in plan.plans)
        ):
            raise RealVNextPreflightError(
                f"{label} inner plan is not the exact empty R1 fold plan"
            )
        for row, fold in zip(
            plan.plans, fold_manifest.folds, strict=True
        ):
            if row.fold_spec_digest != fold.self_hash:
                raise RealVNextPreflightError(
                    f"{label} inner fold plan/fold digest mismatch"
                )
    if inference_spec.primary_metric != "book_accuracy":
        raise RealVNextPreflightError(
            "real vNext primary metric must be book_accuracy"
        )
    if (
        inference_spec.primary_uncertainty
        != "author_clustered_percentile_bootstrap"
        or inference_spec.macro_f1_uncertainty != "point_only"
        or inference_spec.secondary_metrics
        != ("macro_f1", "top2", "per_author")
    ):
        raise RealVNextPreflightError(
            "real vNext inference contract is not exact R1"
        )


def _build_run_identity(
    *,
    packet_manifest: R1PacketManifest,
    corpus_manifest: CorpusVNextManifest,
    content_policy_spec: ContentPolicySpec,
    candidate_inventory: CandidateInventory,
    content_manifest: ContentComponentManifest,
    fold_manifest: FoldManifest,
    primary_inner_cv_plan: InnerCVPlan,
    baseline_inner_cv_plan: InnerCVPlan,
    primary_model_spec: ModelSpec,
    baseline_model_spec: ModelSpec,
    inference_spec: InferenceSpec,
    model_role_manifest: ModelRoleManifest,
    campaign_manifest: CampaignManifest,
    execution_spec: RealCorpusExecutionSpec,
    owner_decision: ExploratoryOwnerDecisionRecord,
    representation_receipt: object,
    observations: Sequence[DerivedObservation],
    factory_contract: str,
) -> dict[str, Any]:
    representation_hash = _manifest_dict(
        representation_receipt, label="representation_receipt"
    )["self_hash"]
    receipt_hashes = {
        receipt.kind: receipt.self_hash
        for receipt in execution_spec.independent_receipts
    }
    material = {
        "schema_version": REAL_RUN_IDENTITY_SCHEMA_VERSION,
        "status": REAL_EXPLORATORY_STATUS,
        "confirmatory_authorized": False,
        "execution_mode": REAL_EXECUTION_MODE,
        "evaluation_strategy": REAL_EVALUATION_STRATEGY,
        "factory_contract": factory_contract,
        "packet_manifest_sha256": packet_manifest.self_hash,
        "corpus": {
            "schema_version": corpus_manifest.schema_version,
            "generation_id": corpus_manifest.generation_id,
            "manifest_sha256": corpus_manifest.self_hash,
            "raw_inventory_sha256": _raw_inventory_digest(corpus_manifest),
            "canonical_model_row_digest": (
                corpus_manifest.canonical_model_row_digest
            ),
            "chunker_policy_version": (
                corpus_manifest.chunker_policy_version
            ),
            "canonicalizer_policy_version": (
                corpus_manifest.canonicalizer_policy_version
            ),
            "content_policy_version": (
                corpus_manifest.content_policy_version
            ),
        },
        "content_policy_spec_sha256": content_policy_spec.self_hash,
        "candidate_inventory_sha256": candidate_inventory.self_hash,
        "content_component_manifest_sha256": content_manifest.self_hash,
        "fold_manifest_sha256": fold_manifest.self_hash,
        "fold_spec_sha256": [
            fold.self_hash for fold in fold_manifest.folds
        ],
        "probability_class_order": list(
            fold_manifest.probability_class_order
        ),
        "metric_label_order": list(fold_manifest.metric_label_order),
        "model_roles": [
            {
                "role": PRIMARY_ROLE,
                "model_spec_sha256": primary_model_spec.self_hash,
                "inner_cv_plan_sha256": primary_inner_cv_plan.self_hash,
            },
            {
                "role": BASELINE_ROLE,
                "model_spec_sha256": baseline_model_spec.self_hash,
                "inner_cv_plan_sha256": baseline_inner_cv_plan.self_hash,
            },
        ],
        "model_role_manifest_sha256": model_role_manifest.self_hash,
        "campaign_manifest_sha256": campaign_manifest.self_hash,
        "inference_spec_sha256": inference_spec.self_hash,
        "execution_spec_sha256": execution_spec.self_hash,
        "owner_decision_sha256": owner_decision.self_hash,
        "representation_receipt_sha256": representation_hash,
        "independent_receipt_sha256": receipt_hashes,
        "observation_evidence_sha256": {
            row.kind: row.evidence_digest for row in observations
        },
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "checkpoint_schema_version": REAL_CHECKPOINT_SCHEMA_VERSION,
        "final_artifact_schema_version": (
            REAL_FINAL_ARTIFACT_SCHEMA_VERSION
        ),
    }
    run_id = _shared._canonical_hash(
        material,
        error_type=RealVNextArtifactError,
    )
    identity = {**material, "run_id": run_id}
    identity["self_hash"] = _shared._self_hash(
        identity,
        error_type=RealVNextArtifactError,
    )
    validate_real_run_identity(identity)
    return identity


_IDENTITY_KEYS = {
    "schema_version",
    "status",
    "confirmatory_authorized",
    "execution_mode",
    "evaluation_strategy",
    "factory_contract",
    "packet_manifest_sha256",
    "corpus",
    "content_policy_spec_sha256",
    "candidate_inventory_sha256",
    "content_component_manifest_sha256",
    "fold_manifest_sha256",
    "fold_spec_sha256",
    "probability_class_order",
    "metric_label_order",
    "model_roles",
    "model_role_manifest_sha256",
    "campaign_manifest_sha256",
    "inference_spec_sha256",
    "execution_spec_sha256",
    "owner_decision_sha256",
    "representation_receipt_sha256",
    "independent_receipt_sha256",
    "observation_evidence_sha256",
    "prediction_contract_version",
    "checkpoint_schema_version",
    "final_artifact_schema_version",
    "run_id",
    "self_hash",
}


def validate_real_run_identity(identity: object) -> dict[str, Any]:
    error = RealVNextArtifactError
    raw = _shared._require_exact_dict(
        identity, _IDENTITY_KEYS, path="real_run_identity", error_type=error
    )
    if (
        raw["schema_version"] != REAL_RUN_IDENTITY_SCHEMA_VERSION
        or raw["status"] != REAL_EXPLORATORY_STATUS
        or raw["confirmatory_authorized"] is not False
        or raw["execution_mode"] != REAL_EXECUTION_MODE
        or raw["evaluation_strategy"] != REAL_EVALUATION_STRATEGY
        or raw["factory_contract"]
        not in {PRODUCTION_FACTORY_CONTRACT, TEST_FACTORY_CONTRACT}
        or raw["prediction_contract_version"]
        != PREDICTION_CONTRACT_VERSION
        or raw["checkpoint_schema_version"]
        != REAL_CHECKPOINT_SCHEMA_VERSION
        or raw["final_artifact_schema_version"]
        != REAL_FINAL_ARTIFACT_SCHEMA_VERSION
    ):
        raise error("real run identity contract/version mismatch")
    corpus = _shared._require_exact_dict(
        raw["corpus"],
        {
            "schema_version",
            "generation_id",
            "manifest_sha256",
            "raw_inventory_sha256",
            "canonical_model_row_digest",
            "chunker_policy_version",
            "canonicalizer_policy_version",
            "content_policy_version",
        },
        path="real_run_identity.corpus",
        error_type=error,
    )
    for key in (
        "schema_version",
        "generation_id",
        "chunker_policy_version",
        "canonicalizer_policy_version",
        "content_policy_version",
    ):
        _shared._require_str(
            corpus[key],
            path=f"real_run_identity.corpus.{key}",
            error_type=error,
        )
    for key in (
        "manifest_sha256",
        "raw_inventory_sha256",
        "canonical_model_row_digest",
    ):
        _shared._require_sha256(
            corpus[key],
            path=f"real_run_identity.corpus.{key}",
            error_type=error,
        )
    for key in (
        "packet_manifest_sha256",
        "content_policy_spec_sha256",
        "candidate_inventory_sha256",
        "content_component_manifest_sha256",
        "fold_manifest_sha256",
        "model_role_manifest_sha256",
        "campaign_manifest_sha256",
        "inference_spec_sha256",
        "execution_spec_sha256",
        "owner_decision_sha256",
        "representation_receipt_sha256",
        "run_id",
        "self_hash",
    ):
        _shared._require_sha256(
            raw[key], path=f"real_run_identity.{key}", error_type=error
        )
    fold_hashes = _shared._require_list(
        raw["fold_spec_sha256"],
        path="real_run_identity.fold_spec_sha256",
        nonempty=True,
        error_type=error,
    )
    for index, digest in enumerate(fold_hashes):
        _shared._require_sha256(
            digest,
            path=f"real_run_identity.fold_spec_sha256[{index}]",
            error_type=error,
        )
    probability_order = _shared._require_string_array(
        raw["probability_class_order"],
        path="real_run_identity.probability_class_order",
        nonempty=True,
        unique=True,
        error_type=error,
    )
    metric_order = _shared._require_string_array(
        raw["metric_label_order"],
        path="real_run_identity.metric_label_order",
        nonempty=True,
        unique=True,
        error_type=error,
    )
    if tuple(
        item for item in probability_order if item in frozenset(metric_order)
    ) != metric_order:
        raise error("real run identity M is not a P-ordered subset")
    model_roles = _shared._require_list(
        raw["model_roles"],
        path="real_run_identity.model_roles",
        nonempty=True,
        error_type=error,
    )
    if len(model_roles) != 2:
        raise error("real run identity must contain exactly two model roles")
    for index, (row, role) in enumerate(zip(model_roles, _ROLES, strict=True)):
        record = _shared._require_exact_dict(
            row,
            {"role", "model_spec_sha256", "inner_cv_plan_sha256"},
            path=f"real_run_identity.model_roles[{index}]",
            error_type=error,
        )
        if record["role"] != role:
            raise error("real run identity model role order mismatch")
        for field in ("model_spec_sha256", "inner_cv_plan_sha256"):
            _shared._require_sha256(
                record[field],
                path=f"real_run_identity.model_roles[{index}].{field}",
                error_type=error,
            )
    for field in (
        "independent_receipt_sha256",
        "observation_evidence_sha256",
    ):
        record = _shared._require_exact_dict(
            raw[field],
            set(REQUIRED_RECEIPT_KINDS),
            path=f"real_run_identity.{field}",
            error_type=error,
        )
        for kind in REQUIRED_RECEIPT_KINDS:
            _shared._require_sha256(
                record[kind],
                path=f"real_run_identity.{field}.{kind}",
                error_type=error,
            )
    _shared._require_self_hash(
        raw, path="real_run_identity", error_type=error
    )
    material = {
        key: value
        for key, value in raw.items()
        if key not in {"run_id", "self_hash"}
    }
    if raw["run_id"] != _shared._canonical_hash(
        material,
        error_type=error,
    ):
        raise error("real run identity run_id mismatch")
    _shared._reject_absolute_paths(
        raw,
        path="real_run_identity",
        error_type=error,
    )
    return raw


def preflight_lobo_vnext_real(
    *,
    packet_manifest: R1PacketManifest,
    corpus_manifest: CorpusVNextManifest,
    content_policy_spec: ContentPolicySpec,
    candidate_inventory: CandidateInventory,
    content_manifest: ContentComponentManifest,
    fold_manifest: FoldManifest,
    primary_inner_cv_plan: InnerCVPlan,
    baseline_inner_cv_plan: InnerCVPlan,
    primary_model_spec: ModelSpec,
    baseline_model_spec: ModelSpec,
    inference_spec: InferenceSpec,
    model_role_manifest: ModelRoleManifest,
    campaign_manifest: CampaignManifest,
    execution_spec: RealCorpusExecutionSpec,
    owner_decision: ExploratoryOwnerDecisionRecord,
    representation_receipt: object,
    cfg: ConfigNode,
    observations: Sequence[DerivedObservation],
    _test_factory_injected: bool = False,
) -> RealVNextPreflight:
    """Validate the complete real packet without reading canonical row bytes."""

    _require_domain_types(
        packet_manifest=packet_manifest,
        corpus_manifest=corpus_manifest,
        content_policy_spec=content_policy_spec,
        candidate_inventory=candidate_inventory,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        primary_inner_cv_plan=primary_inner_cv_plan,
        baseline_inner_cv_plan=baseline_inner_cv_plan,
        primary_model_spec=primary_model_spec,
        baseline_model_spec=baseline_model_spec,
        inference_spec=inference_spec,
        model_role_manifest=model_role_manifest,
        campaign_manifest=campaign_manifest,
        execution_spec=execution_spec,
        owner_decision=owner_decision,
        cfg=cfg,
    )
    _assert_execution_bindings(
        packet_manifest=packet_manifest,
        corpus_manifest=corpus_manifest,
        content_policy_spec=content_policy_spec,
        candidate_inventory=candidate_inventory,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        primary_inner_cv_plan=primary_inner_cv_plan,
        baseline_inner_cv_plan=baseline_inner_cv_plan,
        primary_model_spec=primary_model_spec,
        baseline_model_spec=baseline_model_spec,
        inference_spec=inference_spec,
        model_role_manifest=model_role_manifest,
        campaign_manifest=campaign_manifest,
        execution_spec=execution_spec,
        owner_decision=owner_decision,
        cfg=cfg,
    )
    _validate_scientific_contracts(
        packet_manifest=packet_manifest,
        corpus_manifest=corpus_manifest,
        content_policy_spec=content_policy_spec,
        candidate_inventory=candidate_inventory,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        primary_inner_cv_plan=primary_inner_cv_plan,
        baseline_inner_cv_plan=baseline_inner_cv_plan,
        primary_model_spec=primary_model_spec,
        baseline_model_spec=baseline_model_spec,
        inference_spec=inference_spec,
        campaign_manifest=campaign_manifest,
        cfg=cfg,
        test_factory_injected=_test_factory_injected,
    )
    validated_representation = _validate_representation_receipt(
        representation_receipt, corpus_manifest=corpus_manifest
    )
    if (
        packet_manifest.representation_receipt_sha256
        != validated_representation.self_hash
    ):
        raise RealVNextPreflightError(
            "packet/representation receipt digest mismatch"
        )
    validated_observations = _assert_live_receipts(
        execution_spec=execution_spec,
        observations=observations,
        packet_manifest=packet_manifest,
        cfg=cfg,
        primary_model_spec=primary_model_spec,
        baseline_model_spec=baseline_model_spec,
        representation_receipt=validated_representation,
    )
    factory_contract = (
        TEST_FACTORY_CONTRACT
        if _test_factory_injected
        else PRODUCTION_FACTORY_CONTRACT
    )
    identity = _build_run_identity(
        packet_manifest=packet_manifest,
        corpus_manifest=corpus_manifest,
        content_policy_spec=content_policy_spec,
        candidate_inventory=candidate_inventory,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        primary_inner_cv_plan=primary_inner_cv_plan,
        baseline_inner_cv_plan=baseline_inner_cv_plan,
        primary_model_spec=primary_model_spec,
        baseline_model_spec=baseline_model_spec,
        inference_spec=inference_spec,
        model_role_manifest=model_role_manifest,
        campaign_manifest=campaign_manifest,
        execution_spec=execution_spec,
        owner_decision=owner_decision,
        representation_receipt=validated_representation,
        observations=validated_observations,
        factory_contract=factory_contract,
    )
    return RealVNextPreflight(
        packet_manifest=packet_manifest,
        corpus_manifest=corpus_manifest,
        content_policy_spec=content_policy_spec,
        candidate_inventory=candidate_inventory,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        primary_inner_cv_plan=primary_inner_cv_plan,
        baseline_inner_cv_plan=baseline_inner_cv_plan,
        primary_model_spec=primary_model_spec,
        baseline_model_spec=baseline_model_spec,
        inference_spec=inference_spec,
        model_role_manifest=model_role_manifest,
        campaign_manifest=campaign_manifest,
        execution_spec=execution_spec,
        owner_decision=owner_decision,
        representation_receipt=validated_representation,
        cfg=cfg,
        observations=validated_observations,
        factory_contract=factory_contract,
        run_identity=identity,
    )


def _default_row_loader(
    packet_root: pathlib.Path,
    receipt: object,
    corpus_manifest: CorpusVNextManifest,
) -> tuple[Any, ...]:
    try:
        from ..domain.lobo_vnext_packet import (
            load_canonical_representation_rows,
        )
    except (ImportError, AttributeError) as exc:
        raise RealVNextPreflightError(
            "canonical representation row loader is unavailable"
        ) from exc
    return load_canonical_representation_rows(
        packet_root, receipt, corpus_manifest
    )


def _validate_loaded_rows(
    rows: object,
    *,
    corpus_manifest: CorpusVNextManifest,
    representation_receipt: object,
) -> tuple[Any, ...]:
    if type(rows) is not tuple or not rows:
        raise RealVNextPreflightError(
            "canonical row loader must return an exact non-empty tuple"
        )
    if any(type(row) is not VNextTextRow for row in rows):
        raise RealVNextPreflightError(
            "canonical row loader returned a non-VNextTextRow record"
        )
    row_ids = tuple(row.row_id for row in rows)
    if len(set(row_ids)) != len(row_ids):
        raise RealVNextPreflightError(
            "canonical model rows must have unique row_id values"
        )
    receipt_rows = representation_receipt.rows
    expected_identity = tuple(
        (
            row.row_id,
            row.relative_path,
            row.work_id,
            row.author_id,
            row.source_raw_sha256,
        )
        for row in receipt_rows
    )
    observed_identity = tuple(
        (
            row.row_id,
            row.relative_path,
            row.work_id,
            row.author_id,
            row.raw_sha256,
        )
        for row in rows
    )
    if observed_identity != expected_identity:
        raise RealVNextPreflightError(
            "loaded canonical rows differ from the exact receipt order/identity"
        )
    works = {work.work_id: work.author_id for work in corpus_manifest.works}
    observed_works: set[str] = set()
    for index, row in enumerate(rows):
        for field in (
            "row_id",
            "relative_path",
            "work_id",
            "author_id",
            "text",
            "raw_sha256",
        ):
            value = getattr(row, field, None)
            if type(value) is not str or not value:
                raise RealVNextPreflightError(
                    f"canonical row[{index}].{field} is not an exact "
                    "non-empty string"
                )
        if row.work_id not in works or works[row.work_id] != row.author_id:
            raise RealVNextPreflightError(
                "canonical row author/work identity mismatch"
            )
        observed_works.add(row.work_id)
    if observed_works != set(works):
        raise RealVNextPreflightError(
            "canonical rows do not cover every included work exactly"
        )
    return rows


def _restore_fold(
    preflight: RealVNextPreflight, fold_index: int
) -> FoldSpec:
    fold = preflight.fold_manifest.folds[fold_index]
    try:
        restored = FoldSpec.from_dict(fold.to_dict())
    except Exception as exc:
        raise RealVNextPreflightError(
            f"worker FoldSpec restore failed: {exc}"
        ) from exc
    if (
        restored != fold
        or restored.self_hash
        != preflight.run_identity["fold_spec_sha256"][fold_index]
    ):
        raise RealVNextPreflightError(
            "worker FoldSpec identity mismatch"
        )
    return restored


def _restore_inner_fold(
    preflight: RealVNextPreflight,
    *,
    role: str,
    fold: FoldSpec,
) -> InnerFoldPlan:
    plan = preflight.inner_by_role[role]
    planned = plan.by_fold.get(fold.fold_id)
    if planned is None:
        raise RealVNextPreflightError(
            f"{role} has no inner-fold plan for {fold.fold_id}"
        )
    try:
        restored = InnerFoldPlan.from_dict(planned.to_dict())
    except Exception as exc:
        raise RealVNextPreflightError(
            f"{role} inner-fold restore failed: {exc}"
        ) from exc
    identity_role = {
        row["role"]: row for row in preflight.run_identity["model_roles"]
    }[role]
    if (
        restored != planned
        or restored.fold_spec_digest != fold.self_hash
        or plan.self_hash != identity_role["inner_cv_plan_sha256"]
        or restored.splits
    ):
        raise RealVNextPreflightError(
            f"{role} inner-fold identity/locality mismatch"
        )
    return restored


def _split_record(fold: FoldSpec) -> dict[str, Any]:
    return {
        "fold_id": fold.fold_id,
        "test_work_id": fold.test_work_id,
        "content_component_id": fold.content_component_id,
        "train_work_ids": list(fold.train_work_ids),
        "purged_work_ids": list(fold.purged_work_ids),
        "probability_class_order": list(fold.probability_class_order),
        "metric_label_order": list(fold.metric_label_order),
    }


_CHECKPOINT_KEYS = {
    "schema_version",
    "status",
    "confirmatory_authorized",
    "run_id",
    "campaign_manifest_sha256",
    "role",
    "model_spec_sha256",
    "inner_cv_plan_sha256",
    "inner_fold_plan_sha256",
    "fold_index",
    "fold_spec_sha256",
    "split",
    "result",
    "self_hash",
}
_SPLIT_KEYS = {
    "fold_id",
    "test_work_id",
    "content_component_id",
    "train_work_ids",
    "purged_work_ids",
    "probability_class_order",
    "metric_label_order",
}
_RESULT_KEYS = {
    "work_id",
    "true_author_id",
    "true_label",
    "predicted_author_id",
    "predicted_label",
    "correct",
    "true_rank",
    "probabilities",
    "chunk_count",
}


def build_real_checkpoint(
    *,
    preflight: RealVNextPreflight,
    role: str,
    fold_index: int,
    fold: FoldSpec,
    result: dict[str, Any],
) -> dict[str, Any]:
    model = preflight.model_by_role[role]
    inner = preflight.inner_by_role[role]
    checkpoint = {
        "schema_version": REAL_CHECKPOINT_SCHEMA_VERSION,
        "status": REAL_EXPLORATORY_STATUS,
        "confirmatory_authorized": False,
        "run_id": preflight.run_id,
        "campaign_manifest_sha256": (
            preflight.campaign_manifest.self_hash
        ),
        "role": role,
        "model_spec_sha256": model.self_hash,
        "inner_cv_plan_sha256": inner.self_hash,
        "inner_fold_plan_sha256": inner.by_fold[fold.fold_id].self_hash,
        "fold_index": fold_index,
        "fold_spec_sha256": fold.self_hash,
        "split": _split_record(fold),
        "result": result,
    }
    checkpoint["self_hash"] = _shared._self_hash(
        checkpoint,
        error_type=RealVNextCheckpointError,
    )
    validate_real_checkpoint(
        checkpoint,
        preflight=preflight,
        role=role,
        fold_index=fold_index,
        fold=fold,
    )
    return checkpoint


def validate_real_checkpoint(
    checkpoint: object,
    *,
    preflight: RealVNextPreflight,
    role: str,
    fold_index: int,
    fold: FoldSpec,
) -> dict[str, Any]:
    error = RealVNextCheckpointError
    raw = _shared._require_exact_dict(
        checkpoint, _CHECKPOINT_KEYS, path="real_checkpoint", error_type=error
    )
    if (
        raw["schema_version"] != REAL_CHECKPOINT_SCHEMA_VERSION
        or raw["status"] != REAL_EXPLORATORY_STATUS
        or raw["confirmatory_authorized"] is not False
    ):
        raise error(
            "legacy, non-exploratory, or unsupported real checkpoint"
        )
    if role not in _ROLES or raw["role"] != role:
        raise error("checkpoint role mismatch")
    model = preflight.model_by_role[role]
    inner = preflight.inner_by_role[role]
    inner_fold = inner.by_fold.get(fold.fold_id)
    if inner_fold is None:
        raise error("checkpoint fold has no role-specific inner plan")
    expected_hashes = {
        "run_id": preflight.run_id,
        "campaign_manifest_sha256": (
            preflight.campaign_manifest.self_hash
        ),
        "model_spec_sha256": model.self_hash,
        "inner_cv_plan_sha256": inner.self_hash,
        "inner_fold_plan_sha256": inner_fold.self_hash,
        "fold_spec_sha256": fold.self_hash,
    }
    for field, expected in expected_hashes.items():
        _shared._require_sha256(
            raw[field], path=f"real_checkpoint.{field}", error_type=error
        )
        if raw[field] != expected:
            raise error(f"checkpoint identity mismatch for {field}")
    _shared._require_int(
        raw["fold_index"],
        path="real_checkpoint.fold_index",
        minimum=0,
        error_type=error,
    )
    if raw["fold_index"] != fold_index:
        raise error("checkpoint fold index mismatch")
    split = _shared._require_exact_dict(
        raw["split"],
        _SPLIT_KEYS,
        path="real_checkpoint.split",
        error_type=error,
    )
    _shared._require_exact_structure(
        split,
        _split_record(fold),
        path="real_checkpoint.split",
        error_type=error,
    )
    result = _shared._require_exact_dict(
        raw["result"],
        _RESULT_KEYS,
        path="real_checkpoint.result",
        error_type=error,
    )
    for field in ("work_id", "true_author_id", "predicted_author_id"):
        _shared._require_str(
            result[field],
            path=f"real_checkpoint.result.{field}",
            error_type=error,
        )
    for field in ("true_label", "predicted_label"):
        _shared._require_int(
            result[field],
            path=f"real_checkpoint.result.{field}",
            minimum=0,
            error_type=error,
        )
    _shared._require_bool(
        result["correct"],
        path="real_checkpoint.result.correct",
        error_type=error,
    )
    _shared._require_int(
        result["true_rank"],
        path="real_checkpoint.result.true_rank",
        minimum=1,
        error_type=error,
    )
    _shared._require_int(
        result["chunk_count"],
        path="real_checkpoint.result.chunk_count",
        minimum=1,
        error_type=error,
    )
    probabilities = _shared._require_list(
        result["probabilities"],
        path="real_checkpoint.result.probabilities",
        nonempty=True,
        error_type=error,
    )
    order = tuple(fold.probability_class_order)
    if len(probabilities) != len(order):
        raise error("checkpoint probability width differs from P")
    for index, value in enumerate(probabilities):
        _shared._require_float(
            value,
            path=f"real_checkpoint.result.probabilities[{index}]",
            minimum=0.0,
            maximum=1.0,
            error_type=error,
        )
    if result["work_id"] != fold.test_work_id:
        raise error("checkpoint result/test work mismatch")
    if (
        not 0 <= result["true_label"] < len(order)
        or not 0 <= result["predicted_label"] < len(order)
    ):
        raise error("checkpoint labels are outside P")
    if (
        order[result["true_label"]] != result["true_author_id"]
        or order[result["predicted_label"]]
        != result["predicted_author_id"]
    ):
        raise error("checkpoint author/label mapping mismatch")
    try:
        from ..domain.prediction_contract import validate_prediction_record

        validate_prediction_record(
            probabilities=probabilities,
            pred_label=result["predicted_label"],
            true_label=result["true_label"],
            correct=result["correct"],
            rank=result["true_rank"],
            expected_width=len(order),
        )
    except PredictionContractError as exc:
        raise error(f"checkpoint prediction contract failed: {exc}") from exc
    _shared._require_self_hash(
        raw, path="real_checkpoint", error_type=error
    )
    return raw


def _evaluate_role_fold(
    preflight: RealVNextPreflight,
    *,
    role: str,
    fold_index: int,
    factory: Callable[[ModelSpec, FoldSpec], Any],
) -> dict[str, Any]:
    fold = _restore_fold(preflight, fold_index)
    inner_fold = _restore_inner_fold(preflight, role=role, fold=fold)
    train_ids = frozenset(fold.train_work_ids)
    train_rows = tuple(
        row for row in preflight.rows if row.work_id in train_ids
    )
    test_rows = tuple(
        row
        for row in preflight.rows
        if row.work_id == fold.test_work_id
    )
    forbidden = frozenset(fold.purged_work_ids) | {fold.test_work_id}
    if (
        not train_rows
        or not test_rows
        or {row.work_id for row in train_rows} != train_ids
        or any(row.work_id in forbidden for row in train_rows)
        or {row.work_id for row in test_rows} != {fold.test_work_id}
    ):
        raise RealVNextPreflightError(
            f"{role}/{fold.fold_id} failed the restored outer split receipt"
        )
    probability_order = tuple(fold.probability_class_order)
    label_by_author = {
        author_id: index
        for index, author_id in enumerate(probability_order)
    }
    try:
        train_labels = np.asarray(
            [label_by_author[row.author_id] for row in train_rows],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RealVNextPreflightError(
            f"{role}/{fold.fold_id} train author is outside P"
        ) from exc
    test_authors = {row.author_id for row in test_rows}
    if len(test_authors) != 1:
        raise RealVNextPreflightError(
            f"{role}/{fold.fold_id} test work spans multiple authors"
        )
    true_author = next(iter(test_authors))
    if true_author not in label_by_author:
        raise RealVNextPreflightError(
            f"{role}/{fold.fold_id} true author is outside P"
        )
    true_label = label_by_author[true_author]

    # First factory interaction: all global, row, fold, and locality receipts
    # have already succeeded.
    model_spec = preflight.model_by_role[role]
    estimator = factory(model_spec, fold)
    if estimator is None:
        raise RealLoboVNextError(f"{role} factory returned None")
    train_texts = np.asarray(
        [row.text for row in train_rows], dtype=object
    )
    train_groups = np.asarray(
        [row.work_id for row in train_rows], dtype=object
    )
    estimator.fit(
        train_texts,
        train_labels,
        groups=train_groups,
        inner_splits=inner_fold.splits,
    )
    test_texts = np.asarray(
        [row.text for row in test_rows], dtype=object
    )
    raw_probabilities = estimator.predict_proba(test_texts)
    object_probabilities = np.asarray(raw_probabilities, dtype=object)
    if any(
        isinstance(value, (bool, np.bool_))
        or isinstance(value, (str, bytes))
        for value in object_probabilities.flat
    ):
        raise RealLoboVNextError(
            f"{role}/{fold.fold_id} probabilities contain coercible scalars"
        )
    try:
        chunk_probabilities = validate_probability_matrix(
            raw_probabilities,
            getattr(estimator, "classes_", None),
            n_classes=len(probability_order),
            n_rows=len(test_rows),
        )
    except PredictionContractError as exc:
        raise RealLoboVNextError(
            f"{role}/{fold.fold_id} probability/P contract failed: {exc}"
        ) from exc
    book_probabilities = chunk_probabilities.mean(axis=0)
    decision = stable_top1_and_worst_tie_rank(
        book_probabilities,
        true_label=true_label,
        expected_width=len(probability_order),
    )
    result = {
        "work_id": fold.test_work_id,
        "true_author_id": true_author,
        "true_label": true_label,
        "predicted_author_id": probability_order[decision.top1],
        "predicted_label": decision.top1,
        "correct": bool(decision.top1 == true_label),
        "true_rank": decision.true_rank,
        "probabilities": [
            float(value) for value in book_probabilities.tolist()
        ],
        "chunk_count": len(test_rows),
    }
    checkpoint = build_real_checkpoint(
        preflight=preflight,
        role=role,
        fold_index=fold_index,
        fold=fold,
        result=result,
    )
    restored_fold = _restore_fold(preflight, fold_index)
    _restore_inner_fold(preflight, role=role, fold=restored_fold)
    validate_real_checkpoint(
        checkpoint,
        preflight=preflight,
        role=role,
        fold_index=fold_index,
        fold=restored_fold,
    )
    return checkpoint


def _derive_paired_inference(
    *,
    primary: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    inference_spec: InferenceSpec,
) -> dict[str, Any]:
    if len(primary) != len(baseline) or not primary:
        raise RealVNextArtifactError(
            "paired inference requires complete primary/baseline rows"
        )
    primary_work = [
        row["result"]["work_id"] for row in primary
    ]
    baseline_work = [
        row["result"]["work_id"] for row in baseline
    ]
    primary_truth = np.asarray(
        [row["result"]["true_label"] for row in primary],
        dtype=np.int64,
    )
    baseline_truth = np.asarray(
        [row["result"]["true_label"] for row in baseline],
        dtype=np.int64,
    )
    primary_pred = np.asarray(
        [row["result"]["predicted_label"] for row in primary],
        dtype=np.int64,
    )
    baseline_pred = np.asarray(
        [row["result"]["predicted_label"] for row in baseline],
        dtype=np.int64,
    )
    primary_authors = np.asarray(
        [row["result"]["true_author_id"] for row in primary],
        dtype=object,
    )
    baseline_authors = np.asarray(
        [row["result"]["true_author_id"] for row in baseline],
        dtype=object,
    )
    if (
        primary_work != baseline_work
        or not np.array_equal(primary_truth, baseline_truth)
        or not np.array_equal(primary_authors, baseline_authors)
    ):
        raise RealVNextArtifactError(
            "primary and baseline do not have byte-identical paired folds"
        )
    difference = paired_bootstrap_diff_clustered(
        lambda indexes: accuracy(
            primary_truth[indexes], primary_pred[indexes]
        ),
        lambda indexes: accuracy(
            baseline_truth[indexes], baseline_pred[indexes]
        ),
        primary_authors,
        iters=inference_spec.bootstrap_iterations,
        level=inference_spec.confidence_level,
        seed=inference_spec.bootstrap_seed,
    )
    record = {
        "schema_version": REAL_PAIRED_INFERENCE_SCHEMA_VERSION,
        "metric": "book_accuracy_primary_minus_baseline",
        "sampling_unit": "author",
        "shared_paired_draws": True,
        "point": float(difference.diff),
        "lo": float(difference.lo),
        "hi": float(difference.hi),
        "method": "paired_author_clustered_percentile_bootstrap",
        "inference_spec_sha256": inference_spec.self_hash,
    }
    _validate_paired_inference(record)
    return record


def _validate_paired_inference(value: object) -> dict[str, Any]:
    error = RealVNextArtifactError
    raw = _shared._require_exact_dict(
        value,
        {
            "schema_version",
            "metric",
            "sampling_unit",
            "shared_paired_draws",
            "point",
            "lo",
            "hi",
            "method",
            "inference_spec_sha256",
        },
        path="real_final.paired_inference",
        error_type=error,
    )
    expected_literals = {
        "schema_version": REAL_PAIRED_INFERENCE_SCHEMA_VERSION,
        "metric": "book_accuracy_primary_minus_baseline",
        "sampling_unit": "author",
        "shared_paired_draws": True,
        "method": "paired_author_clustered_percentile_bootstrap",
    }
    for field, expected in expected_literals.items():
        if type(raw[field]) is not type(expected) or raw[field] != expected:
            raise error(f"paired inference {field} contract mismatch")
    for field in ("point", "lo", "hi"):
        _shared._require_float(
            raw[field],
            path=f"real_final.paired_inference.{field}",
            minimum=-1.0,
            maximum=1.0,
            error_type=error,
        )
    _shared._require_sha256(
        raw["inference_spec_sha256"],
        path="real_final.paired_inference.inference_spec_sha256",
        error_type=error,
    )
    return raw


def _derive_role_metrics(
    checkpoints_by_role: Mapping[str, Sequence[dict[str, Any]]],
    *,
    fold_manifest: FoldManifest,
    inference_spec: InferenceSpec,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for role in _ROLES:
        try:
            metrics[role] = _shared._derive_metrics(
                checkpoints_by_role[role],
                fold_manifest,
                inference_spec,
                error_type=RealVNextArtifactError,
            )
        except Exception as exc:
            raise RealVNextArtifactError(
                f"{role} metric derivation failed: {exc}"
            ) from exc
    return metrics


_FINAL_KEYS = {
    "schema_version",
    "status",
    "confirmatory_authorized",
    "run_identity",
    "packet_manifest",
    "owner_decision",
    "execution_spec",
    "content_policy_spec",
    "candidate_inventory",
    "corpus_manifest",
    "content_component_manifest",
    "fold_manifest",
    "primary_inner_cv_plan",
    "baseline_inner_cv_plan",
    "primary_model_spec",
    "baseline_model_spec",
    "model_role_manifest",
    "campaign_manifest",
    "inference_spec",
    "representation_receipt",
    "checkpoints",
    "metrics_by_role",
    "paired_inference",
    "self_hash",
}


def build_real_final_artifact(
    *,
    preflight: RealVNextPreflight,
    checkpoints_by_role: Mapping[
        str, Sequence[dict[str, Any]]
    ],
) -> dict[str, Any]:
    ordered = {
        role: list(checkpoints_by_role[role]) for role in _ROLES
    }
    metrics = _derive_role_metrics(
        ordered,
        fold_manifest=preflight.fold_manifest,
        inference_spec=preflight.inference_spec,
    )
    paired = _derive_paired_inference(
        primary=ordered[PRIMARY_ROLE],
        baseline=ordered[BASELINE_ROLE],
        inference_spec=preflight.inference_spec,
    )
    artifact = {
        "schema_version": REAL_FINAL_ARTIFACT_SCHEMA_VERSION,
        "status": REAL_EXPLORATORY_STATUS,
        "confirmatory_authorized": False,
        "run_identity": preflight.run_identity,
        "packet_manifest": preflight.packet_manifest.to_dict(),
        "owner_decision": preflight.owner_decision.to_dict(),
        "execution_spec": preflight.execution_spec.to_dict(),
        "content_policy_spec": preflight.content_policy_spec.to_dict(),
        "candidate_inventory": preflight.candidate_inventory.to_dict(),
        "corpus_manifest": preflight.corpus_manifest.to_dict(),
        "content_component_manifest": preflight.content_manifest.to_dict(),
        "fold_manifest": preflight.fold_manifest.to_dict(),
        "primary_inner_cv_plan": (
            preflight.primary_inner_cv_plan.to_dict()
        ),
        "baseline_inner_cv_plan": (
            preflight.baseline_inner_cv_plan.to_dict()
        ),
        "primary_model_spec": preflight.primary_model_spec.to_dict(),
        "baseline_model_spec": preflight.baseline_model_spec.to_dict(),
        "model_role_manifest": preflight.model_role_manifest.to_dict(),
        "campaign_manifest": preflight.campaign_manifest.to_dict(),
        "inference_spec": preflight.inference_spec.to_dict(),
        "representation_receipt": (
            preflight.representation_receipt.to_dict()
        ),
        "checkpoints": ordered,
        "metrics_by_role": metrics,
        "paired_inference": paired,
    }
    artifact["self_hash"] = _shared._self_hash(
        artifact,
        error_type=RealVNextArtifactError,
    )
    validate_real_final_artifact(artifact, preflight=preflight)
    return artifact


def _rehydrate_final(
    artifact: dict[str, Any],
) -> dict[str, object]:
    receipt_type = _representation_type()
    try:
        return {
            "packet_manifest": R1PacketManifest.from_dict(
                artifact["packet_manifest"]
            ),
            "owner_decision": ExploratoryOwnerDecisionRecord.from_dict(
                artifact["owner_decision"]
            ),
            "execution_spec": RealCorpusExecutionSpec.from_dict(
                artifact["execution_spec"]
            ),
            "content_policy_spec": ContentPolicySpec.from_dict(
                artifact["content_policy_spec"]
            ),
            "candidate_inventory": CandidateInventory.from_dict(
                artifact["candidate_inventory"]
            ),
            "corpus_manifest": CorpusVNextManifest.from_dict(
                artifact["corpus_manifest"]
            ),
            "content_manifest": ContentComponentManifest.from_dict(
                artifact["content_component_manifest"]
            ),
            "fold_manifest": FoldManifest.from_dict(
                artifact["fold_manifest"]
            ),
            "primary_inner": InnerCVPlan.from_dict(
                artifact["primary_inner_cv_plan"]
            ),
            "baseline_inner": InnerCVPlan.from_dict(
                artifact["baseline_inner_cv_plan"]
            ),
            "primary_model": ModelSpec.from_dict(
                artifact["primary_model_spec"]
            ),
            "baseline_model": ModelSpec.from_dict(
                artifact["baseline_model_spec"]
            ),
            "model_roles": ModelRoleManifest.from_dict(
                artifact["model_role_manifest"]
            ),
            "campaign": CampaignManifest.from_dict(
                artifact["campaign_manifest"]
            ),
            "inference": InferenceSpec.from_dict(
                artifact["inference_spec"]
            ),
            "representation": receipt_type.from_dict(
                artifact["representation_receipt"]
            ),
        }
    except Exception as exc:
        raise RealVNextArtifactError(
            f"embedded real-vNext contract validation failed: {exc}"
        ) from exc


def _standalone_preflight_projection(
    artifact: dict[str, Any],
    objects: Mapping[str, object],
) -> RealVNextPreflight:
    """Build the validation-only projection required by checkpoint validators."""

    identity = validate_real_run_identity(artifact["run_identity"])
    return RealVNextPreflight(
        packet_manifest=objects["packet_manifest"],
        corpus_manifest=objects["corpus_manifest"],
        content_policy_spec=objects["content_policy_spec"],
        candidate_inventory=objects["candidate_inventory"],
        content_manifest=objects["content_manifest"],
        fold_manifest=objects["fold_manifest"],
        primary_inner_cv_plan=objects["primary_inner"],
        baseline_inner_cv_plan=objects["baseline_inner"],
        primary_model_spec=objects["primary_model"],
        baseline_model_spec=objects["baseline_model"],
        inference_spec=objects["inference"],
        model_role_manifest=objects["model_roles"],
        campaign_manifest=objects["campaign"],
        execution_spec=objects["execution_spec"],
        owner_decision=objects["owner_decision"],
        representation_receipt=objects["representation"],
        cfg=object(),  # never consulted by artifact/checkpoint validation
        observations=(),
        factory_contract=identity["factory_contract"],
        run_identity=identity,
    )


def validate_real_final_artifact(
    artifact: object,
    *,
    preflight: RealVNextPreflight | None = None,
) -> dict[str, Any]:
    error = RealVNextArtifactError
    raw = _shared._require_exact_dict(
        artifact, _FINAL_KEYS, path="real_final", error_type=error
    )
    if (
        raw["schema_version"] != REAL_FINAL_ARTIFACT_SCHEMA_VERSION
        or raw["status"] != REAL_EXPLORATORY_STATUS
        or raw["confirmatory_authorized"] is not False
    ):
        raise error(
            "legacy, non-exploratory, or unsupported real final artifact"
        )
    identity = validate_real_run_identity(raw["run_identity"])
    objects = _rehydrate_final(raw)
    projected = _standalone_preflight_projection(raw, objects)
    try:
        objects["packet_manifest"].validate()
        objects["execution_spec"].assert_owner_decision(
            objects["owner_decision"]
        )
        objects["execution_spec"].assert_campaign(
            campaign_manifest=objects["campaign"],
            model_role_manifest=objects["model_roles"],
            primary_model_spec=objects["primary_model"],
            baseline_model_spec=objects["baseline_model"],
            primary_inner_cv_plan=objects["primary_inner"],
            baseline_inner_cv_plan=objects["baseline_inner"],
        )
        objects["corpus_manifest"].validate(
            content_manifest=objects["content_manifest"]
        )
        objects["candidate_inventory"].validate(
            content_policy_spec=objects["content_policy_spec"]
        ).assert_resolved_for_component_manifest()
        objects["fold_manifest"].validate_against(
            objects["corpus_manifest"], objects["content_manifest"]
        )
        objects["representation"].validate(
            corpus_manifest=objects["corpus_manifest"]
        )
    except Exception as exc:
        raise error(f"embedded real-vNext binding failed: {exc}") from exc
    bindings = objects["execution_spec"].bindings
    binding_checks = {
        "packet_manifest_digest": objects["packet_manifest"].self_hash,
        "content_policy_spec_digest": objects[
            "content_policy_spec"
        ].self_hash,
        "candidate_inventory_digest": objects[
            "candidate_inventory"
        ].self_hash,
        "corpus_manifest_digest": objects["corpus_manifest"].self_hash,
        "content_component_manifest_digest": objects[
            "content_manifest"
        ].self_hash,
        "fold_manifest_digest": objects["fold_manifest"].self_hash,
        "primary_inner_cv_plan_digest": objects["primary_inner"].self_hash,
        "baseline_inner_cv_plan_digest": objects["baseline_inner"].self_hash,
        "primary_model_spec_digest": objects["primary_model"].self_hash,
        "baseline_model_spec_digest": objects["baseline_model"].self_hash,
        "model_role_manifest_digest": objects["model_roles"].self_hash,
        "inference_spec_digest": objects["inference"].self_hash,
        "campaign_manifest_digest": objects["campaign"].self_hash,
    }
    for field, expected in binding_checks.items():
        if getattr(bindings, field) != expected:
            raise error(f"final execution binding mismatch for {field}")
    packet = objects["packet_manifest"]
    packet_checks = {
        "candidate_inventory_sha256": objects[
            "candidate_inventory"
        ].self_hash,
        "corpus_manifest_sha256": objects["corpus_manifest"].self_hash,
        "content_component_manifest_sha256": objects[
            "content_manifest"
        ].self_hash,
        "fold_manifest_sha256": objects["fold_manifest"].self_hash,
        "primary_model_spec_sha256": objects["primary_model"].self_hash,
        "baseline_model_spec_sha256": objects["baseline_model"].self_hash,
        "inference_spec_sha256": objects["inference"].self_hash,
        "primary_inner_cv_plan_sha256": objects["primary_inner"].self_hash,
        "baseline_inner_cv_plan_sha256": objects["baseline_inner"].self_hash,
        "model_role_manifest_sha256": objects["model_roles"].self_hash,
        "campaign_manifest_sha256": objects["campaign"].self_hash,
        "representation_receipt_sha256": objects[
            "representation"
        ].self_hash,
    }
    for field, expected in packet_checks.items():
        if getattr(packet, field) != expected:
            raise error(f"final packet binding mismatch for {field}")
    hash_checks = {
        "packet_manifest_sha256": objects["packet_manifest"].self_hash,
        "content_policy_spec_sha256": objects[
            "content_policy_spec"
        ].self_hash,
        "candidate_inventory_sha256": objects[
            "candidate_inventory"
        ].self_hash,
        "content_component_manifest_sha256": objects[
            "content_manifest"
        ].self_hash,
        "fold_manifest_sha256": objects["fold_manifest"].self_hash,
        "model_role_manifest_sha256": objects["model_roles"].self_hash,
        "campaign_manifest_sha256": objects["campaign"].self_hash,
        "inference_spec_sha256": objects["inference"].self_hash,
        "execution_spec_sha256": objects["execution_spec"].self_hash,
        "owner_decision_sha256": objects["owner_decision"].self_hash,
        "representation_receipt_sha256": objects[
            "representation"
        ].self_hash,
    }
    for field, expected in hash_checks.items():
        if identity[field] != expected:
            raise error(f"final/run identity mismatch for {field}")
    if (
        identity["corpus"]["manifest_sha256"]
        != objects["corpus_manifest"].self_hash
        or identity["corpus"]["raw_inventory_sha256"]
        != _raw_inventory_digest(objects["corpus_manifest"])
        or identity["corpus"]["canonical_model_row_digest"]
        != objects["corpus_manifest"].canonical_model_row_digest
    ):
        raise error("final corpus/run identity mismatch")
    corpus_identity = identity["corpus"]
    corpus = objects["corpus_manifest"]
    expected_corpus_literals = {
        "schema_version": corpus.schema_version,
        "generation_id": corpus.generation_id,
        "chunker_policy_version": corpus.chunker_policy_version,
        "canonicalizer_policy_version": corpus.canonicalizer_policy_version,
        "content_policy_version": corpus.content_policy_version,
    }
    for field, expected in expected_corpus_literals.items():
        if corpus_identity[field] != expected:
            raise error(f"final corpus identity mismatch for {field}")
    folds = objects["fold_manifest"]
    if (
        identity["fold_spec_sha256"]
        != [fold.self_hash for fold in folds.folds]
        or identity["probability_class_order"]
        != list(folds.probability_class_order)
        or identity["metric_label_order"]
        != list(folds.metric_label_order)
    ):
        raise error("final fold/P/M run identity mismatch")
    candidates = objects["candidate_inventory"]
    if (
        candidates.generation_id != corpus.generation_id
        or candidates.included_work_ids
        != tuple(work.work_id for work in corpus.works)
        or candidates.work_identity_catalog_digest
        != canonical_sha256([work.to_dict() for work in corpus.works])
        or candidates.raw_inventory_digest != _raw_inventory_digest(corpus)
    ):
        raise error("final candidate/corpus generation/work/raw mismatch")
    expected_receipt_hashes = {
        receipt.kind: receipt.self_hash
        for receipt in objects["execution_spec"].independent_receipts
    }
    expected_evidence_hashes = {
        receipt.kind: receipt.evidence_digest
        for receipt in objects["execution_spec"].independent_receipts
    }
    if (
        identity["independent_receipt_sha256"]
        != expected_receipt_hashes
        or identity["observation_evidence_sha256"]
        != expected_evidence_hashes
    ):
        raise error("final independent receipt/run identity mismatch")
    representation_receipt = next(
        receipt
        for receipt in objects["execution_spec"].independent_receipts
        if receipt.kind == "representation"
    )
    packet_selection_receipt = next(
        receipt
        for receipt in objects["execution_spec"].independent_receipts
        if receipt.kind == "packet_selection"
    )
    if (
        packet_selection_receipt.expected_digest
        != objects["packet_manifest"].self_hash
        or representation_receipt.expected_digest
        != objects["representation"].self_hash
    ):
        raise error(
            "final packet/representation receipt observation mismatch"
        )
    expected_model_rows = (
        (PRIMARY_ROLE, objects["primary_model"], objects["primary_inner"]),
        (BASELINE_ROLE, objects["baseline_model"], objects["baseline_inner"]),
    )
    for identity_row, (role, model, inner) in zip(
        identity["model_roles"], expected_model_rows, strict=True
    ):
        if identity_row != {
            "role": role,
            "model_spec_sha256": model.self_hash,
            "inner_cv_plan_sha256": inner.self_hash,
        }:
            raise error("final model role/run identity mismatch")
    if preflight is not None:
        expected = {
            "run_identity": preflight.run_identity,
            "packet_manifest": preflight.packet_manifest.to_dict(),
            "owner_decision": preflight.owner_decision.to_dict(),
            "execution_spec": preflight.execution_spec.to_dict(),
            "content_policy_spec": preflight.content_policy_spec.to_dict(),
            "candidate_inventory": preflight.candidate_inventory.to_dict(),
            "corpus_manifest": preflight.corpus_manifest.to_dict(),
            "content_component_manifest": (
                preflight.content_manifest.to_dict()
            ),
            "fold_manifest": preflight.fold_manifest.to_dict(),
            "primary_inner_cv_plan": (
                preflight.primary_inner_cv_plan.to_dict()
            ),
            "baseline_inner_cv_plan": (
                preflight.baseline_inner_cv_plan.to_dict()
            ),
            "primary_model_spec": preflight.primary_model_spec.to_dict(),
            "baseline_model_spec": preflight.baseline_model_spec.to_dict(),
            "model_role_manifest": (
                preflight.model_role_manifest.to_dict()
            ),
            "campaign_manifest": preflight.campaign_manifest.to_dict(),
            "inference_spec": preflight.inference_spec.to_dict(),
            "representation_receipt": (
                preflight.representation_receipt.to_dict()
            ),
        }
        observed = {key: raw[key] for key in expected}
        _shared._require_exact_structure(
            observed,
            expected,
            path="real_final.preflight_bindings",
            error_type=error,
        )
        projected = preflight
    checkpoint_map = _shared._require_exact_dict(
        raw["checkpoints"],
        set(_ROLES),
        path="real_final.checkpoints",
        error_type=error,
    )
    expected_count = len(objects["fold_manifest"].folds)
    validated_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in _ROLES:
        rows = _shared._require_list(
            checkpoint_map[role],
            path=f"real_final.checkpoints.{role}",
            nonempty=True,
            error_type=error,
        )
        if len(rows) != expected_count:
            raise error(
                f"real final {role} checkpoint inventory is incomplete"
            )
        validated_by_role[role] = []
        for index, (checkpoint, fold) in enumerate(
            zip(rows, objects["fold_manifest"].folds, strict=True)
        ):
            try:
                validated_by_role[role].append(
                    validate_real_checkpoint(
                        checkpoint,
                        preflight=projected,
                        role=role,
                        fold_index=index,
                        fold=fold,
                    )
                )
            except RealVNextCheckpointError as exc:
                raise error(
                    f"invalid final {role} checkpoint[{index}]: {exc}"
                ) from exc
    metrics = _shared._require_exact_dict(
        raw["metrics_by_role"],
        set(_ROLES),
        path="real_final.metrics_by_role",
        error_type=error,
    )
    expected_metrics = _derive_role_metrics(
        validated_by_role,
        fold_manifest=objects["fold_manifest"],
        inference_spec=objects["inference"],
    )
    for role in _ROLES:
        _shared._validate_metrics_schema(
            metrics[role],
            error_type=error,
        )
        _shared._require_exact_structure(
            metrics[role],
            expected_metrics[role],
            path=f"real_final.metrics_by_role.{role}",
            error_type=error,
        )
        if metrics[role]["macro_f1"]["uncertainty"] != "point_only":
            raise error("macro-F1 must remain point-only")
    expected_paired = _derive_paired_inference(
        primary=validated_by_role[PRIMARY_ROLE],
        baseline=validated_by_role[BASELINE_ROLE],
        inference_spec=objects["inference"],
    )
    _validate_paired_inference(raw["paired_inference"])
    _shared._require_exact_structure(
        raw["paired_inference"],
        expected_paired,
        path="real_final.paired_inference",
        error_type=error,
    )
    _shared._require_self_hash(
        raw, path="real_final", error_type=error
    )
    return raw


def _output_path_contract(
    output_namespace: str | pathlib.Path,
    execution_spec: RealCorpusExecutionSpec,
) -> pathlib.Path:
    path = pathlib.Path(output_namespace)
    _shared._guard_output_namespace(
        path,
        error_type=RealVNextPreflightError,
    )
    required = pathlib.PurePosixPath(
        execution_spec.output_namespace.root_relative_path
    ).parts
    if tuple(path.parts[-len(required) :]) != required:
        raise RealVNextPreflightError(
            "output namespace does not end in the exact owner-bound "
            "exploratory root"
        )
    return path


class _RealCheckpointStore:
    def __init__(
        self,
        output_namespace: pathlib.Path,
        preflight: RealVNextPreflight,
    ) -> None:
        self.output_namespace = output_namespace
        self.preflight = preflight
        self.run_root = output_namespace / preflight.run_id
        self.identity_path = self.run_root / "run-identity.json"
        self.checkpoint_root = self.run_root / "checkpoints"
        self.final_path = self.run_root / "scientific-result.json"

    def _path(self, role: str, fold_index: int) -> pathlib.Path:
        return (
            self.checkpoint_root
            / role
            / f"{fold_index:06d}.json"
        )

    def _expected_paths(self) -> set[pathlib.Path]:
        count = len(self.preflight.fold_manifest.folds)
        return {
            self._path(role, index)
            for role in _ROLES
            for index in range(count)
        }

    def _guard_existing_tree(self) -> None:
        if self.run_root.is_symlink():
            raise RealVNextCheckpointError(
                "symlinked real run directory is forbidden"
            )
        allowed_files = self._expected_paths() | {
            self.identity_path,
            self.final_path,
        }
        allowed_dirs = {
            self.run_root,
            self.checkpoint_root,
            *(self.checkpoint_root / role for role in _ROLES),
        }
        for child in self.run_root.rglob("*"):
            if child.is_symlink():
                raise RealVNextCheckpointError(
                    f"symlinked checkpoint namespace entry: {child}"
                )
            if child.is_dir():
                if child not in allowed_dirs:
                    raise RealVNextCheckpointError(
                        f"extra/conflicting checkpoint directory: {child}"
                    )
            elif child not in allowed_files:
                raise RealVNextCheckpointError(
                    f"extra/conflicting checkpoint file: {child}"
                )

    def inspect_existing(
        self,
    ) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any] | None]:
        _shared._guard_output_namespace(
            self.output_namespace,
            error_type=RealVNextPreflightError,
        )
        if not self.run_root.exists():
            return {}, None
        if self.run_root.is_symlink() or not self.run_root.is_dir():
            raise RealVNextCheckpointError(
                "unsafe real checkpoint run directory"
            )
        identity = _shared._load_json_exact(
            self.identity_path, error_type=RealVNextCheckpointError
        )
        try:
            validate_real_run_identity(identity)
        except RealLoboVNextError as exc:
            raise RealVNextCheckpointError(
                f"existing run identity is invalid: {exc}"
            ) from exc
        if identity != self.preflight.run_identity:
            raise RealVNextCheckpointError(
                "checkpoint namespace belongs to a conflicting run identity"
            )
        self._guard_existing_tree()
        found = self.scan()
        final: dict[str, Any] | None = None
        if self.final_path.exists():
            final = _shared._load_json_exact(
                self.final_path, error_type=RealVNextArtifactError
            )
            validate_real_final_artifact(final, preflight=self.preflight)
            expected_keys = {
                (role, index)
                for role in _ROLES
                for index in range(len(self.preflight.fold_manifest.folds))
            }
            if set(found) != expected_keys:
                raise RealVNextCheckpointError(
                    "final artifact exists beside an incomplete checkpoint set"
                )
        return found, final

    def initialize(self) -> None:
        _shared._guard_output_namespace(
            self.output_namespace,
            error_type=RealVNextPreflightError,
        )
        self.output_namespace.mkdir(parents=True, exist_ok=True)
        if self.output_namespace.is_symlink():
            raise RealVNextCheckpointError(
                "symlinked output namespace is forbidden"
            )
        self.run_root.mkdir(exist_ok=True)
        if self.run_root.is_symlink():
            raise RealVNextCheckpointError(
                "symlinked real run directory is forbidden"
            )
        created = _shared._durable_create(
            self.identity_path,
            self.preflight.run_identity,
            error_type=RealVNextCheckpointError,
        )
        if not created:
            existing = _shared._load_json_exact(
                self.identity_path, error_type=RealVNextCheckpointError
            )
            if existing != self.preflight.run_identity:
                raise RealVNextCheckpointError(
                    "conflicting immutable real run identity"
                )
        for role in _ROLES:
            directory = self.checkpoint_root / role
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink():
                raise RealVNextCheckpointError(
                    f"symlinked checkpoint role directory: {role}"
                )

    def scan(self) -> dict[tuple[str, int], dict[str, Any]]:
        found: dict[tuple[str, int], dict[str, Any]] = {}
        count = len(self.preflight.fold_manifest.folds)
        if not self.checkpoint_root.exists():
            return found
        self._guard_existing_tree()
        for role in _ROLES:
            for index, fold in enumerate(self.preflight.fold_manifest.folds):
                path = self._path(role, index)
                if not path.exists():
                    continue
                checkpoint = _shared._load_json_exact(
                    path, error_type=RealVNextCheckpointError
                )
                validate_real_checkpoint(
                    checkpoint,
                    preflight=self.preflight,
                    role=role,
                    fold_index=index,
                    fold=fold,
                )
                found[(role, index)] = checkpoint
            role_dir = self.checkpoint_root / role
            if role_dir.exists():
                expected_names = {
                    f"{index:06d}.json" for index in range(count)
                }
                observed_names = {item.name for item in role_dir.iterdir()}
                extras = observed_names - expected_names
                if extras:
                    raise RealVNextCheckpointError(
                        f"extra/conflicting {role} checkpoints: "
                        f"{sorted(extras)}"
                    )
        return found

    def save(
        self,
        checkpoint: dict[str, Any],
        *,
        role: str,
        fold_index: int,
    ) -> None:
        fold = self.preflight.fold_manifest.folds[fold_index]
        validate_real_checkpoint(
            checkpoint,
            preflight=self.preflight,
            role=role,
            fold_index=fold_index,
            fold=fold,
        )
        path = self._path(role, fold_index)
        if not _shared._durable_create(
            path,
            checkpoint,
            error_type=RealVNextCheckpointError,
        ):
            existing = _shared._load_json_exact(
                path, error_type=RealVNextCheckpointError
            )
            if existing != checkpoint:
                raise RealVNextCheckpointError(
                    f"conflicting immutable checkpoint {role}/{fold_index}"
                )

    def publish_final(self, artifact: dict[str, Any]) -> pathlib.Path:
        validate_real_final_artifact(artifact, preflight=self.preflight)
        if not _shared._durable_create(
            self.final_path,
            artifact,
            error_type=RealVNextArtifactError,
        ):
            existing = _shared._load_json_exact(
                self.final_path, error_type=RealVNextArtifactError
            )
            validate_real_final_artifact(existing, preflight=self.preflight)
            if existing != artifact:
                raise RealVNextArtifactError(
                    "conflicting immutable real scientific result"
                )
        return self.final_path


def _factory_map(
    *,
    preflight: RealVNextPreflight,
    test_factories: Mapping[
        str, Callable[[ModelSpec, FoldSpec], Any]
    ]
    | None,
) -> dict[str, Callable[[ModelSpec, FoldSpec], Any]]:
    if test_factories is None:
        return {
            role: make_r1_model_factory(
                cfg=preflight.cfg,
                model_spec=preflight.model_by_role[role],
            )
            for role in _ROLES
        }
    if type(test_factories) is not dict or set(test_factories) != set(_ROLES):
        raise RealVNextPreflightError(
            "_test_factory_map must be an exact primary/baseline dict"
        )
    if any(not callable(test_factories[role]) for role in _ROLES):
        raise RealVNextPreflightError(
            "_test_factory_map values must be callable"
        )
    return dict(test_factories)


def run_lobo_vnext_real(
    *,
    packet_root: str | pathlib.Path,
    packet_manifest: R1PacketManifest,
    corpus_manifest: CorpusVNextManifest,
    content_policy_spec: ContentPolicySpec,
    candidate_inventory: CandidateInventory,
    content_manifest: ContentComponentManifest,
    fold_manifest: FoldManifest,
    primary_inner_cv_plan: InnerCVPlan,
    baseline_inner_cv_plan: InnerCVPlan,
    primary_model_spec: ModelSpec,
    baseline_model_spec: ModelSpec,
    inference_spec: InferenceSpec,
    model_role_manifest: ModelRoleManifest,
    campaign_manifest: CampaignManifest,
    execution_spec: RealCorpusExecutionSpec,
    owner_decision: ExploratoryOwnerDecisionRecord,
    representation_receipt: object,
    cfg: ConfigNode,
    observations: Sequence[DerivedObservation],
    output_namespace: str | pathlib.Path,
    n_jobs: int,
    representation_loader: Callable[
        [pathlib.Path, object, CorpusVNextManifest], tuple[Any, ...]
    ]
    | None = None,
    _test_factory_map: Mapping[
        str, Callable[[ModelSpec, FoldSpec], Any]
    ]
    | None = None,
) -> RealVNextRunOutcome:
    """Run/resume the exact two-role bounded real-corpus exploratory campaign."""

    if type(n_jobs) is not int or not 1 <= n_jobs <= 8:
        raise RealVNextPreflightError(
            "n_jobs must be an exact integer in [1, 8]"
        )
    if representation_loader is not None and not callable(
        representation_loader
    ):
        raise RealVNextPreflightError(
            "representation_loader must be callable"
        )
    preflight = preflight_lobo_vnext_real(
        packet_manifest=packet_manifest,
        corpus_manifest=corpus_manifest,
        content_policy_spec=content_policy_spec,
        candidate_inventory=candidate_inventory,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        primary_inner_cv_plan=primary_inner_cv_plan,
        baseline_inner_cv_plan=baseline_inner_cv_plan,
        primary_model_spec=primary_model_spec,
        baseline_model_spec=baseline_model_spec,
        inference_spec=inference_spec,
        model_role_manifest=model_role_manifest,
        campaign_manifest=campaign_manifest,
        execution_spec=execution_spec,
        owner_decision=owner_decision,
        representation_receipt=representation_receipt,
        cfg=cfg,
        observations=observations,
        _test_factory_injected=_test_factory_map is not None,
    )
    output_path = _output_path_contract(output_namespace, execution_spec)
    store = _RealCheckpointStore(output_path, preflight)
    present, existing_final = store.inspect_existing()

    # Existing namespace corruption is rejected above.  Row loading is the
    # first packet-byte/cache interaction and remains downstream of every
    # authority, campaign, receipt, fold, and model validation.
    loader = representation_loader or _default_row_loader
    try:
        loaded = loader(
            pathlib.Path(packet_root),
            preflight.representation_receipt,
            preflight.corpus_manifest,
        )
    except RealLoboVNextError:
        raise
    except Exception as exc:
        raise RealVNextPreflightError(
            f"canonical representation loading failed: {exc}"
        ) from exc
    rows = _validate_loaded_rows(
        loaded,
        corpus_manifest=preflight.corpus_manifest,
        representation_receipt=preflight.representation_receipt,
    )
    preflight = dataclasses.replace(preflight, rows=rows)
    store = _RealCheckpointStore(output_path, preflight)

    if existing_final is not None:
        return RealVNextRunOutcome(
            artifact=existing_final,
            artifact_path=store.final_path,
            telemetry={
                "schema_version": REAL_TELEMETRY_SCHEMA_VERSION,
                "scientific_result_hashed": False,
                "run_id": preflight.run_id,
                "n_jobs": n_jobs,
                "computed_checkpoints": 0,
                "resumed_checkpoints": len(present),
                "checkpoint_seconds": [],
            },
            computed_checkpoints=0,
            resumed_checkpoints=len(present),
        )

    factories = _factory_map(
        preflight=preflight, test_factories=_test_factory_map
    )
    store.initialize()
    present = store.scan()
    keys = [
        (role, index)
        for role in _ROLES
        for index in range(len(preflight.fold_manifest.folds))
    ]
    pending = [key for key in keys if key not in present]
    timings: dict[tuple[str, int], float] = {}

    def evaluate(
        key: tuple[str, int],
    ) -> tuple[tuple[str, int], dict[str, Any], float]:
        role, fold_index = key
        started = time.perf_counter()
        checkpoint = _evaluate_role_fold(
            preflight,
            role=role,
            fold_index=fold_index,
            factory=factories[role],
        )
        return key, checkpoint, float(time.perf_counter() - started)

    if n_jobs == 1:
        for key in pending:
            completed_key, checkpoint, duration = evaluate(key)
            store.save(
                checkpoint,
                role=completed_key[0],
                fold_index=completed_key[1],
            )
            timings[completed_key] = duration
    elif pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_jobs,
            thread_name_prefix="stylo-lobo-vnext-real",
        ) as executor:
            futures = {
                executor.submit(evaluate, key): key for key in pending
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    key, checkpoint, duration = future.result()
                    store.save(
                        checkpoint, role=key[0], fold_index=key[1]
                    )
                    timings[key] = duration
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    complete = store.scan()
    if set(complete) != set(keys):
        missing = sorted(set(keys) - set(complete))
        raise RealVNextCheckpointError(
            f"real checkpoint set is incomplete: {missing}"
        )
    checkpoints_by_role = {
        role: [
            complete[(role, index)]
            for index in range(len(preflight.fold_manifest.folds))
        ]
        for role in _ROLES
    }
    artifact = build_real_final_artifact(
        preflight=preflight, checkpoints_by_role=checkpoints_by_role
    )
    artifact_path = store.publish_final(artifact)
    telemetry = {
        "schema_version": REAL_TELEMETRY_SCHEMA_VERSION,
        "scientific_result_hashed": False,
        "run_id": preflight.run_id,
        "n_jobs": n_jobs,
        "computed_checkpoints": len(pending),
        "resumed_checkpoints": len(present),
        "checkpoint_seconds": [
            {
                "role": role,
                "fold_index": index,
                "seconds": timings[(role, index)],
            }
            for role, index in keys
            if (role, index) in timings
        ],
    }
    return RealVNextRunOutcome(
        artifact=artifact,
        artifact_path=artifact_path,
        telemetry=telemetry,
        computed_checkpoints=len(pending),
        resumed_checkpoints=len(present),
    )


__all__ = [
    "PRODUCTION_FACTORY_CONTRACT",
    "REAL_CHECKPOINT_SCHEMA_VERSION",
    "REAL_FINAL_ARTIFACT_SCHEMA_VERSION",
    "REAL_PAIRED_INFERENCE_SCHEMA_VERSION",
    "REAL_RUN_IDENTITY_SCHEMA_VERSION",
    "REAL_TELEMETRY_SCHEMA_VERSION",
    "RealLoboVNextError",
    "RealVNextArtifactError",
    "RealVNextCheckpointError",
    "RealVNextPreflight",
    "RealVNextPreflightError",
    "RealVNextRunOutcome",
    "build_real_checkpoint",
    "build_real_final_artifact",
    "preflight_lobo_vnext_real",
    "run_lobo_vnext_real",
    "validate_real_checkpoint",
    "validate_real_final_artifact",
    "validate_real_run_identity",
]
