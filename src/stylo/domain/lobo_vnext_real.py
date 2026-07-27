"""Strict control-plane contracts for bounded real-corpus LOBO-vNext runs.

This module contains no corpus transformation, representation construction,
estimator factory, fitting, scoring, checkpoint, or publication code.  It
freezes the owner-selected two-model campaign and the exact pre-execution
receipts which a future real-corpus runner must validate before it performs any
learned operation.

The owner-decision v1 schema contains an ``execution_spec_digest``.  To avoid a
hash cycle, ExecutionSpec v2 deliberately does *not* contain an owner-decision
digest.  The complete execution spec is built and hashed first; the external
owner-decision record then binds that ``self_hash`` plus the independently
addressed corpus/content/fold/campaign/model/inference manifests.  A runner must
load both objects and call
:meth:`RealCorpusExecutionSpec.assert_owner_decision` before any learned
operation.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

from ..jsonio import StrictJSONError, load_strict, loads_strict
from .lobo_vnext import (
    InnerCVPlan,
    ModelSpec,
    VNextContractError,
    canonical_sha256,
)
from .lobo_vnext_approval import ExploratoryOwnerDecisionRecord


MODEL_ROLE_MANIFEST_SCHEMA_VERSION = (
    "stylo.lobo-vnext.real-model-role-manifest.v1"
)
CAMPAIGN_MANIFEST_SCHEMA_VERSION = "stylo.lobo-vnext.real-campaign-manifest.v1"
INDEPENDENT_RECEIPT_SCHEMA_VERSION = (
    "stylo.lobo-vnext.independent-derivation-receipt.v1"
)
OUTPUT_NAMESPACE_SCHEMA_VERSION = (
    "stylo.lobo-vnext.real-output-namespace.v1"
)
REAL_EXECUTION_SPEC_SCHEMA_VERSION = "stylo.lobo-vnext.execution-spec.v2"

REAL_CORPUS_EXECUTION_MODE = "real_corpus"
BOUNDED_EXPLORATORY_AUTHORIZATION = (
    "owner_bound_real_corpus_exploratory_dry_run_only"
)
LOBO_EVALUATION_STRATEGY = "lobo"
REAL_EXPLORATORY_OUTPUT_ROOT = (
    "docs/exploratory/lobo_vnext/real_corpus"
)

PRIMARY_ROLE = "primary"
BASELINE_ROLE = "baseline"
PRIMARY_MODEL_ID = "stylo"
BASELINE_MODEL_ID = "char_cos"
REQUIRED_WEIGHTING = "work_balanced"

REQUIRED_RECEIPT_KINDS = (
    "packet_selection",
    "raw_inventory",
    "canonical_model_rows",
    "content_candidates",
    "content_components",
    "folds",
    "inner_cv",
    "config",
    "primary_model_adapter",
    "baseline_model_adapter",
    "executable_sources",
    "dependencies",
    "runtime",
    "thread_contract",
    "representation",
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]*$")


def _exact_object(
    value: object,
    keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise VNextContractError(f"{label} must be an exact JSON object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise VNextContractError(
            f"{label} keys must be exact; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _exact_list(
    value: object,
    label: str,
    *,
    nonempty: bool = False,
) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = " non-empty" if nonempty else ""
        raise VNextContractError(f"{label} must be an exact{qualifier} array")
    return value


def _exact_str(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise VNextContractError(
            f"{label} must be an exact non-empty trimmed string"
        )
    return value


def _token(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if _TOKEN_RE.fullmatch(text) is None:
        raise VNextContractError(f"{label} must be a path-free canonical token")
    return text


def _sha256(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if _HEX64_RE.fullmatch(text) is None:
        raise VNextContractError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return text


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise VNextContractError(f"{label} must be an exact boolean")
    return value


def _exact_int(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise VNextContractError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _literal(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise VNextContractError(
            f"{label} must be the exact literal {expected!r}"
        )


def _relative_path(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\\" in text or text.startswith(("~", "/")):
        raise VNextContractError(
            f"{label} must be a canonical relative POSIX path"
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or text in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise VNextContractError(
            f"{label} must be a canonical relative POSIX path"
        )
    return text


def _checked_payload(
    value: object,
    keys: set[str] | frozenset[str],
    label: str,
) -> tuple[dict[str, Any], str]:
    raw = _exact_object(value, keys, label)
    recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
    payload = {key: child for key, child in raw.items() if key != "self_hash"}
    if recorded != canonical_sha256(payload):
        raise VNextContractError(f"{label} self_hash mismatch")
    return payload, recorded


def _strict_raw(text: str, label: str) -> object:
    try:
        return loads_strict(text)
    except (StrictJSONError, TypeError) as exc:
        raise VNextContractError(f"{label}: {exc}") from exc


def _strict_file(path: str | os.PathLike[str], label: str) -> object:
    try:
        return load_strict(path)
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise VNextContractError(f"{label}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class ModelRoleBinding:
    ordinal: int
    role: str
    model_id: str
    family: str
    weighting: str
    model_spec_digest: str
    inner_cv_plan_digest: str

    @classmethod
    def from_dict(cls, value: object) -> "ModelRoleBinding":
        raw = _exact_object(
            value,
            {
                "ordinal",
                "role",
                "model_id",
                "family",
                "weighting",
                "model_spec_digest",
                "inner_cv_plan_digest",
            },
            "model role binding",
        )
        return cls(
            _exact_int(raw["ordinal"], "model role binding.ordinal", minimum=0),
            _token(raw["role"], "model role binding.role"),
            _token(raw["model_id"], "model role binding.model_id"),
            _token(raw["family"], "model role binding.family"),
            _token(raw["weighting"], "model role binding.weighting"),
            _sha256(
                raw["model_spec_digest"],
                "model role binding.model_spec_digest",
            ),
            _sha256(
                raw["inner_cv_plan_digest"],
                "model role binding.inner_cv_plan_digest",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "role": self.role,
            "model_id": self.model_id,
            "family": self.family,
            "weighting": self.weighting,
            "model_spec_digest": self.model_spec_digest,
            "inner_cv_plan_digest": self.inner_cv_plan_digest,
        }


def _validate_empty_inner_plan(
    plan: InnerCVPlan,
    *,
    model_spec: ModelSpec,
    label: str,
) -> InnerCVPlan:
    if type(plan) is not InnerCVPlan:
        raise VNextContractError(f"{label} must be exactly InnerCVPlan")
    rebuilt = InnerCVPlan.from_dict(plan.to_dict())
    if rebuilt != plan:
        raise VNextContractError(f"{label} is noncanonical")
    if plan.model_spec_digest != model_spec.self_hash:
        raise VNextContractError(
            f"{label} does not bind its exact ModelSpec"
        )
    if model_spec.requires_inner_cv:
        raise VNextContractError(
            f"{label} R1 ModelSpec must not require inner CV"
        )
    if any(fold_plan.splits for fold_plan in plan.plans):
        raise VNextContractError(
            f"{label} must contain exact empty inner splits for R1"
        )
    return plan


@dataclasses.dataclass(frozen=True)
class ModelRoleManifest:
    schema_version: str
    roles: tuple[ModelRoleBinding, ModelRoleBinding]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        primary_model_spec: ModelSpec,
        baseline_model_spec: ModelSpec,
        primary_inner_cv_plan: InnerCVPlan,
        baseline_inner_cv_plan: InnerCVPlan,
    ) -> "ModelRoleManifest":
        for label, spec in (
            ("primary_model_spec", primary_model_spec),
            ("baseline_model_spec", baseline_model_spec),
        ):
            if type(spec) is not ModelSpec:
                raise VNextContractError(f"{label} must be exactly ModelSpec")
            spec.validate()
        _validate_empty_inner_plan(
            primary_inner_cv_plan,
            model_spec=primary_model_spec,
            label="primary_inner_cv_plan",
        )
        _validate_empty_inner_plan(
            baseline_inner_cv_plan,
            model_spec=baseline_model_spec,
            label="baseline_inner_cv_plan",
        )
        if (
            primary_inner_cv_plan.fold_manifest_digest
            != baseline_inner_cv_plan.fold_manifest_digest
            or primary_inner_cv_plan.content_component_manifest_digest
            != baseline_inner_cv_plan.content_component_manifest_digest
        ):
            raise VNextContractError(
                "primary and baseline inner plans must bind the same "
                "fold/content manifests"
            )
        payload = {
            "schema_version": MODEL_ROLE_MANIFEST_SCHEMA_VERSION,
            "roles": [
                {
                    "ordinal": 0,
                    "role": PRIMARY_ROLE,
                    "model_id": primary_model_spec.model_id,
                    "family": primary_model_spec.family,
                    "weighting": primary_model_spec.weighting,
                    "model_spec_digest": primary_model_spec.self_hash,
                    "inner_cv_plan_digest": primary_inner_cv_plan.self_hash,
                },
                {
                    "ordinal": 1,
                    "role": BASELINE_ROLE,
                    "model_id": baseline_model_spec.model_id,
                    "family": baseline_model_spec.family,
                    "weighting": baseline_model_spec.weighting,
                    "model_spec_digest": baseline_model_spec.self_hash,
                    "inner_cv_plan_digest": baseline_inner_cv_plan.self_hash,
                },
            ],
        }
        manifest = cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )
        return manifest.assert_model_specs(
            primary_model_spec=primary_model_spec,
            baseline_model_spec=baseline_model_spec,
            primary_inner_cv_plan=primary_inner_cv_plan,
            baseline_inner_cv_plan=baseline_inner_cv_plan,
        )

    @classmethod
    def from_dict(cls, value: object) -> "ModelRoleManifest":
        payload, recorded = _checked_payload(
            value,
            {"schema_version", "roles", "self_hash"},
            "model role manifest",
        )
        _literal(
            payload["schema_version"],
            MODEL_ROLE_MANIFEST_SCHEMA_VERSION,
            "model role manifest.schema_version",
        )
        rows = _exact_list(
            payload["roles"], "model role manifest.roles", nonempty=True
        )
        if len(rows) != 2:
            raise VNextContractError(
                "model role manifest.roles must contain exactly two records"
            )
        roles = tuple(ModelRoleBinding.from_dict(row) for row in rows)
        expected = (
            (
                0,
                PRIMARY_ROLE,
                PRIMARY_MODEL_ID,
                PRIMARY_MODEL_ID,
                REQUIRED_WEIGHTING,
            ),
            (
                1,
                BASELINE_ROLE,
                BASELINE_MODEL_ID,
                BASELINE_MODEL_ID,
                REQUIRED_WEIGHTING,
            ),
        )
        observed = tuple(
            (
                row.ordinal,
                row.role,
                row.model_id,
                row.family,
                row.weighting,
            )
            for row in roles
        )
        if observed != expected:
            raise VNextContractError(
                "model roles must be ordered exactly as "
                "primary stylo/work_balanced then "
                "baseline char_cos/work_balanced"
            )
        if roles[0].model_spec_digest == roles[1].model_spec_digest:
            raise VNextContractError(
                "primary and baseline ModelSpec digests must differ"
            )
        if roles[0].inner_cv_plan_digest == roles[1].inner_cv_plan_digest:
            raise VNextContractError(
                "model-specific primary and baseline inner-plan digests "
                "must differ"
            )
        return cls(
            MODEL_ROLE_MANIFEST_SCHEMA_VERSION,
            (roles[0], roles[1]),
            recorded,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "roles": [row.to_dict() for row in self.roles],
        }

    def validate(self) -> "ModelRoleManifest":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError("model role manifest is noncanonical")
        return self

    def assert_model_specs(
        self,
        *,
        primary_model_spec: ModelSpec,
        baseline_model_spec: ModelSpec,
        primary_inner_cv_plan: InnerCVPlan,
        baseline_inner_cv_plan: InnerCVPlan,
    ) -> "ModelRoleManifest":
        self.validate()
        expected_specs = (primary_model_spec, baseline_model_spec)
        expected_plans = (primary_inner_cv_plan, baseline_inner_cv_plan)
        for row, spec, label in zip(
            self.roles,
            expected_specs,
            ("primary", "baseline"),
            strict=True,
        ):
            if type(spec) is not ModelSpec:
                raise VNextContractError(
                    f"{label} model spec must be exactly ModelSpec"
                )
            spec.validate()
            if (
                row.model_id != spec.model_id
                or row.family != spec.family
                or row.weighting != spec.weighting
                or row.model_spec_digest != spec.self_hash
            ):
                raise VNextContractError(
                    f"{label} ModelSpec differs from its role binding"
                )
            spec.assert_exploratory_authorized(synthetic_fixture=False)
        for row, plan, spec, label in zip(
            self.roles,
            expected_plans,
            expected_specs,
            ("primary_inner_cv_plan", "baseline_inner_cv_plan"),
            strict=True,
        ):
            _validate_empty_inner_plan(plan, model_spec=spec, label=label)
            if row.inner_cv_plan_digest != plan.self_hash:
                raise VNextContractError(
                    f"{label} differs from its role binding"
                )
        if (
            primary_inner_cv_plan.fold_manifest_digest
            != baseline_inner_cv_plan.fold_manifest_digest
            or primary_inner_cv_plan.content_component_manifest_digest
            != baseline_inner_cv_plan.content_component_manifest_digest
        ):
            raise VNextContractError(
                "primary and baseline inner plans must bind the same "
                "fold/content manifests"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


@dataclasses.dataclass(frozen=True)
class CampaignRun:
    ordinal: int
    role: str
    model_spec_digest: str
    inner_cv_plan_digest: str

    @classmethod
    def from_dict(cls, value: object) -> "CampaignRun":
        raw = _exact_object(
            value,
            {
                "ordinal",
                "role",
                "model_spec_digest",
                "inner_cv_plan_digest",
            },
            "campaign run",
        )
        return cls(
            _exact_int(raw["ordinal"], "campaign run.ordinal", minimum=0),
            _token(raw["role"], "campaign run.role"),
            _sha256(
                raw["model_spec_digest"], "campaign run.model_spec_digest"
            ),
            _sha256(
                raw["inner_cv_plan_digest"],
                "campaign run.inner_cv_plan_digest",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "role": self.role,
            "model_spec_digest": self.model_spec_digest,
            "inner_cv_plan_digest": self.inner_cv_plan_digest,
        }


@dataclasses.dataclass(frozen=True)
class CampaignManifest:
    schema_version: str
    campaign_id: str
    fold_manifest_digest: str
    inference_spec_digest: str
    model_role_manifest_digest: str
    ordered_runs: tuple[CampaignRun, CampaignRun]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        campaign_id: str,
        fold_manifest_digest: str,
        inference_spec_digest: str,
        model_role_manifest: ModelRoleManifest,
    ) -> "CampaignManifest":
        if type(model_role_manifest) is not ModelRoleManifest:
            raise VNextContractError(
                "model_role_manifest must be exactly ModelRoleManifest"
            )
        model_role_manifest.validate()
        payload = {
            "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "fold_manifest_digest": fold_manifest_digest,
            "inference_spec_digest": inference_spec_digest,
            "model_role_manifest_digest": model_role_manifest.self_hash,
            "ordered_runs": [
                {
                    "ordinal": role.ordinal,
                    "role": role.role,
                    "model_spec_digest": role.model_spec_digest,
                    "inner_cv_plan_digest": role.inner_cv_plan_digest,
                }
                for role in model_role_manifest.roles
            ],
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "CampaignManifest":
        payload, recorded = _checked_payload(
            value,
            {
                "schema_version",
                "campaign_id",
                "fold_manifest_digest",
                "inference_spec_digest",
                "model_role_manifest_digest",
                "ordered_runs",
                "self_hash",
            },
            "campaign manifest",
        )
        _literal(
            payload["schema_version"],
            CAMPAIGN_MANIFEST_SCHEMA_VERSION,
            "campaign manifest.schema_version",
        )
        run_rows = _exact_list(
            payload["ordered_runs"],
            "campaign manifest.ordered_runs",
            nonempty=True,
        )
        if len(run_rows) != 2:
            raise VNextContractError(
                "campaign must contain exactly primary and baseline runs"
            )
        runs = tuple(CampaignRun.from_dict(row) for row in run_rows)
        if tuple((row.ordinal, row.role) for row in runs) != (
            (0, PRIMARY_ROLE),
            (1, BASELINE_ROLE),
        ):
            raise VNextContractError(
                "campaign runs must be ordered primary then baseline"
            )
        return cls(
            CAMPAIGN_MANIFEST_SCHEMA_VERSION,
            _token(payload["campaign_id"], "campaign manifest.campaign_id"),
            _sha256(
                payload["fold_manifest_digest"],
                "campaign manifest.fold_manifest_digest",
            ),
            _sha256(
                payload["inference_spec_digest"],
                "campaign manifest.inference_spec_digest",
            ),
            _sha256(
                payload["model_role_manifest_digest"],
                "campaign manifest.model_role_manifest_digest",
            ),
            (runs[0], runs[1]),
            recorded,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "fold_manifest_digest": self.fold_manifest_digest,
            "inference_spec_digest": self.inference_spec_digest,
            "model_role_manifest_digest": self.model_role_manifest_digest,
            "ordered_runs": [run.to_dict() for run in self.ordered_runs],
        }

    def validate(self) -> "CampaignManifest":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError("campaign manifest is noncanonical")
        return self

    def assert_model_roles(
        self, model_role_manifest: ModelRoleManifest
    ) -> "CampaignManifest":
        self.validate()
        if type(model_role_manifest) is not ModelRoleManifest:
            raise VNextContractError(
                "model_role_manifest must be exactly ModelRoleManifest"
            )
        model_role_manifest.validate()
        if self.model_role_manifest_digest != model_role_manifest.self_hash:
            raise VNextContractError(
                "campaign/model-role manifest digest mismatch"
            )
        expected = tuple(
            (
                row.ordinal,
                row.role,
                row.model_spec_digest,
                row.inner_cv_plan_digest,
            )
            for row in model_role_manifest.roles
        )
        observed = tuple(
            (
                row.ordinal,
                row.role,
                row.model_spec_digest,
                row.inner_cv_plan_digest,
            )
            for row in self.ordered_runs
        )
        if observed != expected:
            raise VNextContractError(
                "campaign runs differ from ordered model-role bindings"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


@dataclasses.dataclass(frozen=True)
class IndependentDerivationReceipt:
    schema_version: str
    kind: str
    derivation_version: str
    expected_digest: str
    observed_digest: str
    evidence_digest: str
    observation_count: int
    host_path_free: bool
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        kind: str,
        derivation_version: str,
        expected_digest: str,
        observed_digest: str,
        evidence_digest: str,
        observation_count: int,
    ) -> "IndependentDerivationReceipt":
        payload = {
            "schema_version": INDEPENDENT_RECEIPT_SCHEMA_VERSION,
            "kind": kind,
            "derivation_version": derivation_version,
            "expected_digest": expected_digest,
            "observed_digest": observed_digest,
            "evidence_digest": evidence_digest,
            "observation_count": observation_count,
            "host_path_free": True,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "IndependentDerivationReceipt":
        payload, recorded = _checked_payload(
            value,
            {
                "schema_version",
                "kind",
                "derivation_version",
                "expected_digest",
                "observed_digest",
                "evidence_digest",
                "observation_count",
                "host_path_free",
                "self_hash",
            },
            "independent derivation receipt",
        )
        _literal(
            payload["schema_version"],
            INDEPENDENT_RECEIPT_SCHEMA_VERSION,
            "independent derivation receipt.schema_version",
        )
        kind = _token(
            payload["kind"], "independent derivation receipt.kind"
        )
        if kind not in REQUIRED_RECEIPT_KINDS:
            raise VNextContractError(
                f"unsupported independent receipt kind {kind!r}"
            )
        expected = _sha256(
            payload["expected_digest"],
            "independent derivation receipt.expected_digest",
        )
        observed = _sha256(
            payload["observed_digest"],
            "independent derivation receipt.observed_digest",
        )
        if observed != expected:
            raise VNextContractError(
                "independently observed digest differs from expected digest"
            )
        _literal(
            payload["host_path_free"],
            True,
            "independent derivation receipt.host_path_free",
        )
        return cls(
            INDEPENDENT_RECEIPT_SCHEMA_VERSION,
            kind,
            _token(
                payload["derivation_version"],
                "independent derivation receipt.derivation_version",
            ),
            expected,
            observed,
            _sha256(
                payload["evidence_digest"],
                "independent derivation receipt.evidence_digest",
            ),
            _exact_int(
                payload["observation_count"],
                "independent derivation receipt.observation_count",
                minimum=1,
            ),
            True,
            recorded,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "derivation_version": self.derivation_version,
            "expected_digest": self.expected_digest,
            "observed_digest": self.observed_digest,
            "evidence_digest": self.evidence_digest,
            "observation_count": self.observation_count,
            "host_path_free": self.host_path_free,
        }

    def validate(self) -> "IndependentDerivationReceipt":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError(
                "independent derivation receipt is noncanonical"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


@dataclasses.dataclass(frozen=True)
class OutputNamespaceContract:
    schema_version: str
    root_relative_path: str
    namespace_id: str
    run_directory_key: str
    create_if_absent_only: bool
    reject_symlinks: bool
    immutable_checkpoints: bool
    separate_scientific_and_telemetry: bool
    public_evidence_update_authorized: bool
    frozen_evidence_mutation_authorized: bool
    confirmatory_output_authorized: bool
    self_hash: str

    @classmethod
    def build(cls, *, namespace_id: str) -> "OutputNamespaceContract":
        payload = {
            "schema_version": OUTPUT_NAMESPACE_SCHEMA_VERSION,
            "root_relative_path": REAL_EXPLORATORY_OUTPUT_ROOT,
            "namespace_id": namespace_id,
            "run_directory_key": "run_id",
            "create_if_absent_only": True,
            "reject_symlinks": True,
            "immutable_checkpoints": True,
            "separate_scientific_and_telemetry": True,
            "public_evidence_update_authorized": False,
            "frozen_evidence_mutation_authorized": False,
            "confirmatory_output_authorized": False,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "OutputNamespaceContract":
        keys = {
            "schema_version",
            "root_relative_path",
            "namespace_id",
            "run_directory_key",
            "create_if_absent_only",
            "reject_symlinks",
            "immutable_checkpoints",
            "separate_scientific_and_telemetry",
            "public_evidence_update_authorized",
            "frozen_evidence_mutation_authorized",
            "confirmatory_output_authorized",
            "self_hash",
        }
        payload, recorded = _checked_payload(
            value, keys, "output namespace contract"
        )
        _literal(
            payload["schema_version"],
            OUTPUT_NAMESPACE_SCHEMA_VERSION,
            "output namespace contract.schema_version",
        )
        root = _relative_path(
            payload["root_relative_path"],
            "output namespace contract.root_relative_path",
        )
        _literal(
            root,
            REAL_EXPLORATORY_OUTPUT_ROOT,
            "output namespace contract.root_relative_path",
        )
        _literal(
            payload["run_directory_key"],
            "run_id",
            "output namespace contract.run_directory_key",
        )
        for field in (
            "create_if_absent_only",
            "reject_symlinks",
            "immutable_checkpoints",
            "separate_scientific_and_telemetry",
        ):
            _literal(
                payload[field], True, f"output namespace contract.{field}"
            )
        for field in (
            "public_evidence_update_authorized",
            "frozen_evidence_mutation_authorized",
            "confirmatory_output_authorized",
        ):
            _literal(
                payload[field], False, f"output namespace contract.{field}"
            )
        return cls(
            OUTPUT_NAMESPACE_SCHEMA_VERSION,
            root,
            _token(
                payload["namespace_id"],
                "output namespace contract.namespace_id",
            ),
            "run_id",
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            recorded,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root_relative_path": self.root_relative_path,
            "namespace_id": self.namespace_id,
            "run_directory_key": self.run_directory_key,
            "create_if_absent_only": self.create_if_absent_only,
            "reject_symlinks": self.reject_symlinks,
            "immutable_checkpoints": self.immutable_checkpoints,
            "separate_scientific_and_telemetry": (
                self.separate_scientific_and_telemetry
            ),
            "public_evidence_update_authorized": (
                self.public_evidence_update_authorized
            ),
            "frozen_evidence_mutation_authorized": (
                self.frozen_evidence_mutation_authorized
            ),
            "confirmatory_output_authorized": (
                self.confirmatory_output_authorized
            ),
        }

    def validate(self) -> "OutputNamespaceContract":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError("output namespace contract is noncanonical")
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


_EXECUTION_BINDING_KEYS = {
    "packet_manifest_digest",
    "content_policy_spec_digest",
    "candidate_inventory_digest",
    "corpus_manifest_digest",
    "content_component_manifest_digest",
    "fold_manifest_digest",
    "primary_inner_cv_plan_digest",
    "baseline_inner_cv_plan_digest",
    "primary_model_spec_digest",
    "baseline_model_spec_digest",
    "model_role_manifest_digest",
    "inference_spec_digest",
    "campaign_manifest_digest",
    "config_digest",
}


@dataclasses.dataclass(frozen=True)
class RealExecutionBindings:
    packet_manifest_digest: str
    content_policy_spec_digest: str
    candidate_inventory_digest: str
    corpus_manifest_digest: str
    content_component_manifest_digest: str
    fold_manifest_digest: str
    primary_inner_cv_plan_digest: str
    baseline_inner_cv_plan_digest: str
    primary_model_spec_digest: str
    baseline_model_spec_digest: str
    model_role_manifest_digest: str
    inference_spec_digest: str
    campaign_manifest_digest: str
    config_digest: str

    @classmethod
    def from_dict(cls, value: object) -> "RealExecutionBindings":
        raw = _exact_object(
            value, _EXECUTION_BINDING_KEYS, "real execution bindings"
        )
        ordered = (
            "packet_manifest_digest",
            "content_policy_spec_digest",
            "candidate_inventory_digest",
            "corpus_manifest_digest",
            "content_component_manifest_digest",
            "fold_manifest_digest",
            "primary_inner_cv_plan_digest",
            "baseline_inner_cv_plan_digest",
            "primary_model_spec_digest",
            "baseline_model_spec_digest",
            "model_role_manifest_digest",
            "inference_spec_digest",
            "campaign_manifest_digest",
            "config_digest",
        )
        return cls(
            *(
                _sha256(raw[key], f"real execution bindings.{key}")
                for key in ordered
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }


def inner_cv_receipt_subject_digest(
    *,
    primary_inner_cv_plan_digest: str,
    baseline_inner_cv_plan_digest: str,
) -> str:
    """Hash the exact ordered role-to-inner-plan binding for one receipt."""

    return canonical_sha256(
        {
            "primary": _sha256(
                primary_inner_cv_plan_digest,
                "primary_inner_cv_plan_digest",
            ),
            "baseline": _sha256(
                baseline_inner_cv_plan_digest,
                "baseline_inner_cv_plan_digest",
            ),
        }
    )


_EXECUTION_KEYS = {
    "schema_version",
    "execution_mode",
    "authorization_scope",
    "evaluation_strategy",
    "bindings",
    "independent_receipts",
    "output_namespace",
    "safety",
    "self_hash",
}
_EXECUTION_SAFETY_KEYS = {
    "confirmatory_execution_authorized",
    "public_evidence_update_authorized",
    "headline_update_authorized",
    "frozen_evidence_mutation_authorized",
}


@dataclasses.dataclass(frozen=True)
class RealCorpusExecutionSpec:
    schema_version: str
    execution_mode: str
    authorization_scope: str
    evaluation_strategy: str
    bindings: RealExecutionBindings
    independent_receipts: tuple[IndependentDerivationReceipt, ...]
    output_namespace: OutputNamespaceContract
    confirmatory_execution_authorized: bool
    public_evidence_update_authorized: bool
    headline_update_authorized: bool
    frozen_evidence_mutation_authorized: bool
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        bindings: RealExecutionBindings,
        independent_receipts: Sequence[IndependentDerivationReceipt],
        output_namespace: OutputNamespaceContract,
    ) -> "RealCorpusExecutionSpec":
        if type(bindings) is not RealExecutionBindings:
            raise VNextContractError(
                "bindings must be exactly RealExecutionBindings"
            )
        if type(independent_receipts) not in (list, tuple):
            raise VNextContractError(
                "independent_receipts must be an exact list or tuple"
            )
        if type(output_namespace) is not OutputNamespaceContract:
            raise VNextContractError(
                "output_namespace must be exactly OutputNamespaceContract"
            )
        receipt_rows = tuple(independent_receipts)
        if any(
            type(receipt) is not IndependentDerivationReceipt
            for receipt in receipt_rows
        ):
            raise VNextContractError(
                "independent_receipts contain an invalid record"
            )
        payload = {
            "schema_version": REAL_EXECUTION_SPEC_SCHEMA_VERSION,
            "execution_mode": REAL_CORPUS_EXECUTION_MODE,
            "authorization_scope": BOUNDED_EXPLORATORY_AUTHORIZATION,
            "evaluation_strategy": LOBO_EVALUATION_STRATEGY,
            "bindings": bindings.to_dict(),
            "independent_receipts": [
                receipt.to_dict() for receipt in receipt_rows
            ],
            "output_namespace": output_namespace.to_dict(),
            "safety": {
                "confirmatory_execution_authorized": False,
                "public_evidence_update_authorized": False,
                "headline_update_authorized": False,
                "frozen_evidence_mutation_authorized": False,
            },
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "RealCorpusExecutionSpec":
        payload, recorded = _checked_payload(
            value, _EXECUTION_KEYS, "real corpus execution spec"
        )
        _literal(
            payload["schema_version"],
            REAL_EXECUTION_SPEC_SCHEMA_VERSION,
            "real corpus execution spec.schema_version",
        )
        _literal(
            payload["execution_mode"],
            REAL_CORPUS_EXECUTION_MODE,
            "real corpus execution spec.execution_mode",
        )
        _literal(
            payload["authorization_scope"],
            BOUNDED_EXPLORATORY_AUTHORIZATION,
            "real corpus execution spec.authorization_scope",
        )
        strategy = _exact_str(
            payload["evaluation_strategy"],
            "real corpus execution spec.evaluation_strategy",
        )
        if strategy.lower() in {"gkf", "groupkfold", "group_k_fold"}:
            raise VNextContractError("GKF is never accepted as LOBO")
        _literal(
            strategy,
            LOBO_EVALUATION_STRATEGY,
            "real corpus execution spec.evaluation_strategy",
        )
        bindings = RealExecutionBindings.from_dict(payload["bindings"])
        receipt_rows = _exact_list(
            payload["independent_receipts"],
            "real corpus execution spec.independent_receipts",
            nonempty=True,
        )
        receipts = tuple(
            IndependentDerivationReceipt.from_dict(row)
            for row in receipt_rows
        )
        if tuple(receipt.kind for receipt in receipts) != REQUIRED_RECEIPT_KINDS:
            raise VNextContractError(
                "independent receipts must contain every required kind "
                "exactly once in canonical order"
            )
        receipts_by_kind = {receipt.kind: receipt for receipt in receipts}
        expected_receipt_subjects = {
            "packet_selection": bindings.packet_manifest_digest,
            "content_candidates": bindings.candidate_inventory_digest,
            "content_components": (
                bindings.content_component_manifest_digest
            ),
            "folds": bindings.fold_manifest_digest,
            "inner_cv": inner_cv_receipt_subject_digest(
                primary_inner_cv_plan_digest=(
                    bindings.primary_inner_cv_plan_digest
                ),
                baseline_inner_cv_plan_digest=(
                    bindings.baseline_inner_cv_plan_digest
                ),
            ),
            "config": bindings.config_digest,
        }
        for kind, expected_digest in expected_receipt_subjects.items():
            if receipts_by_kind[kind].expected_digest != expected_digest:
                raise VNextContractError(
                    f"{kind} receipt subject mismatch with execution bindings"
                )
        output = OutputNamespaceContract.from_dict(
            payload["output_namespace"]
        )
        safety = _exact_object(
            payload["safety"],
            _EXECUTION_SAFETY_KEYS,
            "real corpus execution spec.safety",
        )
        for field in sorted(_EXECUTION_SAFETY_KEYS):
            _literal(
                safety[field],
                False,
                f"real corpus execution spec.safety.{field}",
            )
        return cls(
            REAL_EXECUTION_SPEC_SCHEMA_VERSION,
            REAL_CORPUS_EXECUTION_MODE,
            BOUNDED_EXPLORATORY_AUTHORIZATION,
            LOBO_EVALUATION_STRATEGY,
            bindings,
            receipts,
            output,
            False,
            False,
            False,
            False,
            recorded,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_mode": self.execution_mode,
            "authorization_scope": self.authorization_scope,
            "evaluation_strategy": self.evaluation_strategy,
            "bindings": self.bindings.to_dict(),
            "independent_receipts": [
                receipt.to_dict() for receipt in self.independent_receipts
            ],
            "output_namespace": self.output_namespace.to_dict(),
            "safety": {
                "confirmatory_execution_authorized": (
                    self.confirmatory_execution_authorized
                ),
                "public_evidence_update_authorized": (
                    self.public_evidence_update_authorized
                ),
                "headline_update_authorized": self.headline_update_authorized,
                "frozen_evidence_mutation_authorized": (
                    self.frozen_evidence_mutation_authorized
                ),
            },
        }

    def validate(self) -> "RealCorpusExecutionSpec":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError(
                "real corpus execution spec is noncanonical"
            )
        return self

    def assert_owner_decision(
        self,
        owner_decision: ExploratoryOwnerDecisionRecord,
    ) -> "RealCorpusExecutionSpec":
        self.validate()
        if type(owner_decision) is not ExploratoryOwnerDecisionRecord:
            raise VNextContractError(
                "owner_decision must be exactly ExploratoryOwnerDecisionRecord"
            )
        owner_decision.validate()
        decision = owner_decision.bindings
        expected = {
            "corpus_manifest_digest": self.bindings.corpus_manifest_digest,
            "content_component_manifest_digest": (
                self.bindings.content_component_manifest_digest
            ),
            "policy_manifest_digest": (
                self.bindings.content_policy_spec_digest
            ),
            "fold_manifest_digest": self.bindings.fold_manifest_digest,
            "campaign_manifest_digest": (
                self.bindings.campaign_manifest_digest
            ),
            "model_role_manifest_digest": (
                self.bindings.model_role_manifest_digest
            ),
            "inference_spec_digest": self.bindings.inference_spec_digest,
            "execution_spec_digest": self.self_hash,
        }
        if decision.to_dict() != expected:
            raise VNextContractError(
                "owner decision does not bind the exact real execution packet"
            )
        return self

    def assert_campaign(
        self,
        *,
        campaign_manifest: CampaignManifest,
        model_role_manifest: ModelRoleManifest,
        primary_model_spec: ModelSpec,
        baseline_model_spec: ModelSpec,
        primary_inner_cv_plan: InnerCVPlan,
        baseline_inner_cv_plan: InnerCVPlan,
    ) -> "RealCorpusExecutionSpec":
        self.validate()
        if type(campaign_manifest) is not CampaignManifest:
            raise VNextContractError(
                "campaign_manifest must be exactly CampaignManifest"
            )
        if type(model_role_manifest) is not ModelRoleManifest:
            raise VNextContractError(
                "model_role_manifest must be exactly ModelRoleManifest"
            )
        model_role_manifest.assert_model_specs(
            primary_model_spec=primary_model_spec,
            baseline_model_spec=baseline_model_spec,
            primary_inner_cv_plan=primary_inner_cv_plan,
            baseline_inner_cv_plan=baseline_inner_cv_plan,
        )
        campaign_manifest.assert_model_roles(model_role_manifest)
        if (
            primary_inner_cv_plan.fold_manifest_digest
            != campaign_manifest.fold_manifest_digest
            or baseline_inner_cv_plan.fold_manifest_digest
            != campaign_manifest.fold_manifest_digest
        ):
            raise VNextContractError(
                "campaign fold manifest differs from model-specific "
                "inner plans"
            )
        if (
            primary_inner_cv_plan.content_component_manifest_digest
            != self.bindings.content_component_manifest_digest
            or baseline_inner_cv_plan.content_component_manifest_digest
            != self.bindings.content_component_manifest_digest
        ):
            raise VNextContractError(
                "execution content manifest differs from model-specific "
                "inner plans"
            )
        checks = {
            "campaign_manifest_digest": campaign_manifest.self_hash,
            "model_role_manifest_digest": model_role_manifest.self_hash,
            "primary_model_spec_digest": primary_model_spec.self_hash,
            "baseline_model_spec_digest": baseline_model_spec.self_hash,
            "primary_inner_cv_plan_digest": primary_inner_cv_plan.self_hash,
            "baseline_inner_cv_plan_digest": baseline_inner_cv_plan.self_hash,
            "fold_manifest_digest": campaign_manifest.fold_manifest_digest,
            "inference_spec_digest": campaign_manifest.inference_spec_digest,
        }
        for field, expected in checks.items():
            if getattr(self.bindings, field) != expected:
                raise VNextContractError(
                    f"execution/campaign binding mismatch for {field}"
                )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


def loads_model_role_manifest(text: str) -> ModelRoleManifest:
    return ModelRoleManifest.from_dict(_strict_raw(text, "model role manifest"))


def load_model_role_manifest(
    path: str | os.PathLike[str],
) -> ModelRoleManifest:
    return ModelRoleManifest.from_dict(
        _strict_file(path, "model role manifest")
    )


def loads_campaign_manifest(text: str) -> CampaignManifest:
    return CampaignManifest.from_dict(_strict_raw(text, "campaign manifest"))


def load_campaign_manifest(
    path: str | os.PathLike[str],
) -> CampaignManifest:
    return CampaignManifest.from_dict(_strict_file(path, "campaign manifest"))


def loads_real_execution_spec(text: str) -> RealCorpusExecutionSpec:
    return RealCorpusExecutionSpec.from_dict(
        _strict_raw(text, "real corpus execution spec")
    )


def load_real_execution_spec(
    path: str | os.PathLike[str],
) -> RealCorpusExecutionSpec:
    return RealCorpusExecutionSpec.from_dict(
        _strict_file(path, "real corpus execution spec")
    )


__all__ = [
    "BASELINE_MODEL_ID",
    "BASELINE_ROLE",
    "BOUNDED_EXPLORATORY_AUTHORIZATION",
    "CAMPAIGN_MANIFEST_SCHEMA_VERSION",
    "INDEPENDENT_RECEIPT_SCHEMA_VERSION",
    "LOBO_EVALUATION_STRATEGY",
    "MODEL_ROLE_MANIFEST_SCHEMA_VERSION",
    "OUTPUT_NAMESPACE_SCHEMA_VERSION",
    "PRIMARY_MODEL_ID",
    "PRIMARY_ROLE",
    "REAL_CORPUS_EXECUTION_MODE",
    "REAL_EXECUTION_SPEC_SCHEMA_VERSION",
    "REAL_EXPLORATORY_OUTPUT_ROOT",
    "REQUIRED_RECEIPT_KINDS",
    "REQUIRED_WEIGHTING",
    "CampaignManifest",
    "CampaignRun",
    "IndependentDerivationReceipt",
    "ModelRoleBinding",
    "ModelRoleManifest",
    "OutputNamespaceContract",
    "RealCorpusExecutionSpec",
    "RealExecutionBindings",
    "load_campaign_manifest",
    "load_model_role_manifest",
    "load_real_execution_spec",
    "inner_cv_receipt_subject_digest",
    "loads_campaign_manifest",
    "loads_model_role_manifest",
    "loads_real_execution_spec",
]
