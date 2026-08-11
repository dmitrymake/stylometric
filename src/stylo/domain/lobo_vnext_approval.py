"""Strict non-personal authorization for a bounded LOBO-vNext dry run.

The record binds an explicit interactive user authorization to exact
content-addressed inputs.  It intentionally contains no personal identity or
role fields.  Its canonical SHA-256 self-hash detects accidental or unreviewed
byte changes; it is not a signature and does not authenticate authorization.
Authentication, if ever required, needs a separate versioned authority.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .._strict_fields import ExactFieldReader
from ..jsonio import StrictJSONError, canonical_hash, loads_strict


OWNER_DECISION_SCHEMA_VERSION = (
    "stylo.lobo-vnext.exploratory-authorization.v2"
)
AUTHORIZATION_BASIS = "explicit_interactive_user_authorization"
REAL_CORPUS_EXPLORATORY_SCOPE = "real_corpus_exploratory_dry_run_only"
EXPLORATORY_CHOSEN_OPTION = "authorize_exact_bound_exploratory_dry_run"
SELF_HASH_SEMANTICS = (
    "canonical_sha256_integrity_only_not_authorization_authentication"
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]*$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "decision_id",
        "decision_revision",
        "decision_date",
        "authorization_basis",
        "scope",
        "chosen_option",
        "approved_for_exploratory",
        "bindings",
        "reviewed_evidence",
        "affected_contract_versions",
        "safety",
        "integrity_contract",
        "self_hash",
    }
)
_BINDING_KEYS = frozenset(
    {
        "corpus_manifest_digest",
        "content_component_manifest_digest",
        "policy_manifest_digest",
        "fold_manifest_digest",
        "campaign_manifest_digest",
        "model_role_manifest_digest",
        "inference_spec_digest",
        "execution_spec_digest",
    }
)
_EVIDENCE_KEYS = frozenset({"relative_path", "sha256"})
_SAFETY_KEYS = frozenset(
    {
        "confirmatory_execution_authorized",
        "public_evidence_update_authorized",
        "headline_update_authorized",
        "frozen_evidence_mutation_authorized",
    }
)
_INTEGRITY_KEYS = frozenset(
    {"self_hash_semantics", "cryptographic_authentication"}
)


class OwnerDecisionContractError(ValueError):
    """The exploratory owner-decision record is malformed or unsafe."""


def _validate_json_value(value: object, path: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise OwnerDecisionContractError(f"{path} must be finite")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise OwnerDecisionContractError(
                    f"{path} has a non-string object key"
                )
            _validate_json_value(child, f"{path}.{key}")
        return
    raise OwnerDecisionContractError(
        f"{path} has unsupported exact JSON type {type(value).__name__}"
    )


def _canonical_sha256(value: object) -> str:
    _validate_json_value(value, "$")
    try:
        return canonical_hash(value)
    except (StrictJSONError, TypeError, ValueError) as exc:
        raise OwnerDecisionContractError(
            f"value is not canonical strict JSON: {exc}"
        ) from exc


_STRICT = ExactFieldReader(
    OwnerDecisionContractError,
    string_policy="trimmed_control_separate",
)
_exact_object = _STRICT.object
_exact_string = _STRICT.string


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise OwnerDecisionContractError(f"{label} must be an exact boolean")
    return value


_exact_int = _STRICT.integer
_sha256 = _STRICT.sha256


def _looks_like_absolute_host_path(value: str) -> bool:
    return (
        value.startswith(("/", "\\", "~"))
        or _WINDOWS_ABSOLUTE_RE.match(value) is not None
    )


def _token(value: object, label: str) -> str:
    text = _exact_string(value, label)
    if _TOKEN_RE.fullmatch(text) is None:
        raise OwnerDecisionContractError(
            f"{label} must be a path-free canonical token"
        )
    return text


def _relative_path(value: object, label: str) -> str:
    text = _exact_string(value, label)
    if (
        _looks_like_absolute_host_path(text)
        or "\\" in text
        or _WINDOWS_ABSOLUTE_RE.match(text) is not None
    ):
        raise OwnerDecisionContractError(
            f"{label} must be a canonical relative POSIX path"
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or text in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise OwnerDecisionContractError(
            f"{label} must be a canonical relative POSIX path"
        )
    return text


def _decision_date(value: object) -> str:
    text = _exact_string(value, "decision_date")
    try:
        parsed = dt.date.fromisoformat(text)
    except ValueError as exc:
        raise OwnerDecisionContractError(
            "decision_date must be a valid ISO-8601 calendar date"
        ) from exc
    if parsed.isoformat() != text:
        raise OwnerDecisionContractError(
            "decision_date must use canonical YYYY-MM-DD form"
        )
    return text


def _literal(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise OwnerDecisionContractError(
            f"{label} must be the exact literal {expected!r}"
        )


@dataclasses.dataclass(frozen=True)
class ReviewedEvidence:
    relative_path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedEvidence":
        raw = _exact_object(value, _EVIDENCE_KEYS, "reviewed evidence")
        return cls(
            _relative_path(
                raw["relative_path"], "reviewed_evidence[].relative_path"
            ),
            _sha256(raw["sha256"], "reviewed_evidence[].sha256"),
        )

    def validate(self) -> "ReviewedEvidence":
        if type(self) is not ReviewedEvidence:
            raise OwnerDecisionContractError(
                "reviewed evidence must be exactly ReviewedEvidence"
            )
        _relative_path(
            self.relative_path, "reviewed_evidence[].relative_path"
        )
        _sha256(self.sha256, "reviewed_evidence[].sha256")
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {"relative_path": self.relative_path, "sha256": self.sha256}


@dataclasses.dataclass(frozen=True)
class DecisionBindings:
    corpus_manifest_digest: str
    content_component_manifest_digest: str
    policy_manifest_digest: str
    fold_manifest_digest: str
    campaign_manifest_digest: str
    model_role_manifest_digest: str
    inference_spec_digest: str
    execution_spec_digest: str

    @classmethod
    def from_dict(cls, value: object) -> "DecisionBindings":
        raw = _exact_object(value, _BINDING_KEYS, "decision bindings")
        return cls(
            *(
                _sha256(raw[key], f"bindings.{key}")
                for key in (
                    "corpus_manifest_digest",
                    "content_component_manifest_digest",
                    "policy_manifest_digest",
                    "fold_manifest_digest",
                    "campaign_manifest_digest",
                    "model_role_manifest_digest",
                    "inference_spec_digest",
                    "execution_spec_digest",
                )
            )
        )

    def validate(self) -> "DecisionBindings":
        if type(self) is not DecisionBindings:
            raise OwnerDecisionContractError(
                "bindings must be exactly DecisionBindings"
            )
        for key, value in self.to_dict(validate=False).items():
            _sha256(value, f"bindings.{key}")
        return self

    def to_dict(self, *, validate: bool = True) -> dict[str, object]:
        if validate:
            self.validate()
        return {
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "content_component_manifest_digest": (
                self.content_component_manifest_digest
            ),
            "policy_manifest_digest": self.policy_manifest_digest,
            "fold_manifest_digest": self.fold_manifest_digest,
            "campaign_manifest_digest": self.campaign_manifest_digest,
            "model_role_manifest_digest": self.model_role_manifest_digest,
            "inference_spec_digest": self.inference_spec_digest,
            "execution_spec_digest": self.execution_spec_digest,
        }


@dataclasses.dataclass(frozen=True)
class ExploratoryOwnerDecisionRecord:
    schema_version: str
    decision_id: str
    decision_revision: int
    decision_date: str
    authorization_basis: str
    scope: str
    chosen_option: str
    approved_for_exploratory: bool
    bindings: DecisionBindings
    reviewed_evidence: tuple[ReviewedEvidence, ...]
    affected_contract_versions: tuple[str, ...]
    confirmatory_execution_authorized: bool
    public_evidence_update_authorized: bool
    headline_update_authorized: bool
    frozen_evidence_mutation_authorized: bool
    self_hash_semantics: str
    cryptographic_authentication: bool
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        decision_id: str,
        decision_revision: int,
        decision_date: str,
        bindings: DecisionBindings,
        reviewed_evidence: Sequence[ReviewedEvidence],
        affected_contract_versions: Sequence[str],
    ) -> "ExploratoryOwnerDecisionRecord":
        if type(bindings) is not DecisionBindings:
            raise OwnerDecisionContractError(
                "bindings must be exactly DecisionBindings"
            )
        bindings.validate()
        if type(reviewed_evidence) not in (list, tuple):
            raise OwnerDecisionContractError(
                "reviewed_evidence must be an exact list or tuple"
            )
        evidence_items = tuple(reviewed_evidence)
        if not evidence_items or any(
            type(item) is not ReviewedEvidence for item in evidence_items
        ):
            raise OwnerDecisionContractError(
                "reviewed_evidence must contain exact ReviewedEvidence records"
            )
        for evidence in evidence_items:
            evidence.validate()
        if type(affected_contract_versions) not in (list, tuple):
            raise OwnerDecisionContractError(
                "affected_contract_versions must be an exact list or tuple"
            )
        affected_items = tuple(
            _token(item, "affected_contract_versions[]")
            for item in affected_contract_versions
        )
        if not affected_items:
            raise OwnerDecisionContractError(
                "affected_contract_versions must be non-empty"
            )
        payload = {
            "schema_version": OWNER_DECISION_SCHEMA_VERSION,
            "decision_id": decision_id,
            "decision_revision": decision_revision,
            "decision_date": decision_date,
            "authorization_basis": AUTHORIZATION_BASIS,
            "scope": REAL_CORPUS_EXPLORATORY_SCOPE,
            "chosen_option": EXPLORATORY_CHOSEN_OPTION,
            "approved_for_exploratory": True,
            "bindings": bindings.to_dict(),
            "reviewed_evidence": [
                evidence.to_dict()
                for evidence in sorted(
                    evidence_items,
                    key=lambda item: item.relative_path,
                )
            ],
            "affected_contract_versions": sorted(affected_items),
            "safety": {
                "confirmatory_execution_authorized": False,
                "public_evidence_update_authorized": False,
                "headline_update_authorized": False,
                "frozen_evidence_mutation_authorized": False,
            },
            "integrity_contract": {
                "self_hash_semantics": SELF_HASH_SEMANTICS,
                "cryptographic_authentication": False,
            },
        }
        return cls._from_payload(payload, _canonical_sha256(payload))

    @classmethod
    def from_dict(cls, value: object) -> "ExploratoryOwnerDecisionRecord":
        if (
            type(value) is dict
            and value.get("schema_version") != OWNER_DECISION_SCHEMA_VERSION
        ):
            raise OwnerDecisionContractError(
                "exploratory authorization is legacy or unsupported"
            )
        raw = _exact_object(
            value, _TOP_LEVEL_KEYS, "exploratory authorization record"
        )
        recorded_hash = _sha256(raw["self_hash"], "self_hash")
        payload = {
            key: child for key, child in raw.items() if key != "self_hash"
        }
        record = cls._from_payload(payload, recorded_hash)
        if record.self_hash != _canonical_sha256(payload):
            raise OwnerDecisionContractError(
                "exploratory authorization self_hash mismatch"
            )
        return record

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "ExploratoryOwnerDecisionRecord":
        expected_payload_keys = _TOP_LEVEL_KEYS - {"self_hash"}
        _exact_object(
            payload,
            frozenset(expected_payload_keys),
            "exploratory authorization payload",
        )
        bindings = DecisionBindings.from_dict(payload["bindings"])
        evidence_raw = payload["reviewed_evidence"]
        if type(evidence_raw) is not list or not evidence_raw:
            raise OwnerDecisionContractError(
                "reviewed_evidence must be an exact non-empty array"
            )
        evidence = tuple(
            ReviewedEvidence.from_dict(item) for item in evidence_raw
        )
        affected_raw = payload["affected_contract_versions"]
        if type(affected_raw) is not list or not affected_raw:
            raise OwnerDecisionContractError(
                "affected_contract_versions must be an exact non-empty array"
            )
        affected = tuple(
            _token(item, "affected_contract_versions[]")
            for item in affected_raw
        )
        safety = _exact_object(payload["safety"], _SAFETY_KEYS, "safety")
        integrity = _exact_object(
            payload["integrity_contract"],
            _INTEGRITY_KEYS,
            "integrity_contract",
        )
        record = cls(
            _exact_string(payload["schema_version"], "schema_version"),
            _token(payload["decision_id"], "decision_id"),
            _exact_int(
                payload["decision_revision"],
                "decision_revision",
                minimum=1,
            ),
            _decision_date(payload["decision_date"]),
            _exact_string(
                payload["authorization_basis"], "authorization_basis"
            ),
            _exact_string(payload["scope"], "scope"),
            _exact_string(payload["chosen_option"], "chosen_option"),
            _exact_bool(
                payload["approved_for_exploratory"],
                "approved_for_exploratory",
            ),
            bindings,
            evidence,
            affected,
            _exact_bool(
                safety["confirmatory_execution_authorized"],
                "safety.confirmatory_execution_authorized",
            ),
            _exact_bool(
                safety["public_evidence_update_authorized"],
                "safety.public_evidence_update_authorized",
            ),
            _exact_bool(
                safety["headline_update_authorized"],
                "safety.headline_update_authorized",
            ),
            _exact_bool(
                safety["frozen_evidence_mutation_authorized"],
                "safety.frozen_evidence_mutation_authorized",
            ),
            _exact_string(
                integrity["self_hash_semantics"],
                "integrity_contract.self_hash_semantics",
            ),
            _exact_bool(
                integrity["cryptographic_authentication"],
                "integrity_contract.cryptographic_authentication",
            ),
            _sha256(self_hash, "self_hash"),
        )
        record.validate()
        return record

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "decision_revision": self.decision_revision,
            "decision_date": self.decision_date,
            "authorization_basis": self.authorization_basis,
            "scope": self.scope,
            "chosen_option": self.chosen_option,
            "approved_for_exploratory": self.approved_for_exploratory,
            "bindings": self.bindings.to_dict(),
            "reviewed_evidence": [
                evidence.to_dict() for evidence in self.reviewed_evidence
            ],
            "affected_contract_versions": list(
                self.affected_contract_versions
            ),
            "safety": {
                "confirmatory_execution_authorized": (
                    self.confirmatory_execution_authorized
                ),
                "public_evidence_update_authorized": (
                    self.public_evidence_update_authorized
                ),
                "headline_update_authorized": (
                    self.headline_update_authorized
                ),
                "frozen_evidence_mutation_authorized": (
                    self.frozen_evidence_mutation_authorized
                ),
            },
            "integrity_contract": {
                "self_hash_semantics": self.self_hash_semantics,
                "cryptographic_authentication": (
                    self.cryptographic_authentication
                ),
            },
        }

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    def validate(self) -> "ExploratoryOwnerDecisionRecord":
        if type(self) is not ExploratoryOwnerDecisionRecord:
            raise OwnerDecisionContractError(
                "record must be exactly ExploratoryOwnerDecisionRecord"
            )
        _literal(
            self.schema_version,
            OWNER_DECISION_SCHEMA_VERSION,
            "schema_version",
        )
        _token(self.decision_id, "decision_id")
        _exact_int(self.decision_revision, "decision_revision", minimum=1)
        _decision_date(self.decision_date)
        _literal(
            self.authorization_basis,
            AUTHORIZATION_BASIS,
            "authorization_basis",
        )
        _literal(self.scope, REAL_CORPUS_EXPLORATORY_SCOPE, "scope")
        _literal(
            self.chosen_option,
            EXPLORATORY_CHOSEN_OPTION,
            "chosen_option",
        )
        _literal(
            self.approved_for_exploratory,
            True,
            "approved_for_exploratory",
        )
        self.bindings.validate()
        if not self.reviewed_evidence or any(
            type(item) is not ReviewedEvidence
            for item in self.reviewed_evidence
        ):
            raise OwnerDecisionContractError(
                "reviewed_evidence must contain exact ReviewedEvidence records"
            )
        for evidence in self.reviewed_evidence:
            evidence.validate()
        evidence_paths = tuple(
            evidence.relative_path for evidence in self.reviewed_evidence
        )
        if evidence_paths != tuple(sorted(set(evidence_paths))):
            raise OwnerDecisionContractError(
                "reviewed_evidence must be path-sorted and duplicate-free"
            )
        if not self.affected_contract_versions:
            raise OwnerDecisionContractError(
                "affected_contract_versions must be non-empty"
            )
        for version in self.affected_contract_versions:
            _token(version, "affected_contract_versions[]")
        if self.affected_contract_versions != tuple(
            sorted(set(self.affected_contract_versions))
        ):
            raise OwnerDecisionContractError(
                "affected_contract_versions must be sorted and duplicate-free"
            )
        for label, value in (
            (
                "confirmatory_execution_authorized",
                self.confirmatory_execution_authorized,
            ),
            (
                "public_evidence_update_authorized",
                self.public_evidence_update_authorized,
            ),
            ("headline_update_authorized", self.headline_update_authorized),
            (
                "frozen_evidence_mutation_authorized",
                self.frozen_evidence_mutation_authorized,
            ),
        ):
            _literal(value, False, f"safety.{label}")
        _literal(
            self.self_hash_semantics,
            SELF_HASH_SEMANTICS,
            "integrity_contract.self_hash_semantics",
        )
        _literal(
            self.cryptographic_authentication,
            False,
            "integrity_contract.cryptographic_authentication",
        )
        _sha256(self.self_hash, "self_hash")
        if self.self_hash != _canonical_sha256(self._payload()):
            raise OwnerDecisionContractError(
                "exploratory authorization self_hash mismatch"
            )
        return self


def loads_owner_decision_record(text: str) -> ExploratoryOwnerDecisionRecord:
    """Load strict JSON, rejecting duplicate keys and non-finite numbers."""

    try:
        raw = loads_strict(text)
    except (StrictJSONError, TypeError) as exc:
        raise OwnerDecisionContractError(
            f"exploratory authorization is not strict JSON: {exc}"
        ) from exc
    return ExploratoryOwnerDecisionRecord.from_dict(raw)


def build_owner_decision_record(
    *,
    decision_id: str,
    decision_revision: int,
    decision_date: str,
    bindings: DecisionBindings,
    reviewed_evidence: Sequence[ReviewedEvidence],
    affected_contract_versions: Sequence[str],
) -> ExploratoryOwnerDecisionRecord:
    """Build one canonical approved exploratory record with safe literals."""

    return ExploratoryOwnerDecisionRecord.build(
        decision_id=decision_id,
        decision_revision=decision_revision,
        decision_date=decision_date,
        bindings=bindings,
        reviewed_evidence=reviewed_evidence,
        affected_contract_versions=affected_contract_versions,
    )


def load_owner_decision_record(
    path: str | os.PathLike[str],
) -> ExploratoryOwnerDecisionRecord:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise OwnerDecisionContractError(
            f"cannot read exploratory authorization record: {exc}"
        ) from exc
    return loads_owner_decision_record(text)


__all__ = [
    "AUTHORIZATION_BASIS",
    "EXPLORATORY_CHOSEN_OPTION",
    "OWNER_DECISION_SCHEMA_VERSION",
    "REAL_CORPUS_EXPLORATORY_SCOPE",
    "SELF_HASH_SEMANTICS",
    "DecisionBindings",
    "ExploratoryOwnerDecisionRecord",
    "OwnerDecisionContractError",
    "ReviewedEvidence",
    "build_owner_decision_record",
    "load_owner_decision_record",
    "loads_owner_decision_record",
]
