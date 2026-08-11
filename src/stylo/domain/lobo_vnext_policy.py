"""Strict owner-decision contracts for a future real-corpus LOBO-vNext packet.

The executable LOBO-vNext harness intentionally remains synthetic-only.  This
module freezes the policy and candidate-review inputs which must exist before a
real corpus can be considered for that harness.  It performs no corpus
normalisation, candidate discovery, component assignment, or model execution.

``CandidateDraft`` and ``CandidateInventory`` are deliberately pre-execution
surfaces: they may retain unresolved content candidates.  The executable
``ContentComponentManifest`` in :mod:`stylo.domain.lobo_vnext` remains the
resolved-only boundary.
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .._strict_fields import ExactFieldReader
from ..jsonio import StrictJSONError, load_strict, loads_strict
from .lobo_vnext import VNextContractError, canonical_sha256


CONTENT_POLICY_SPEC_SCHEMA_VERSION = "stylo.lobo-vnext.content-policy-spec.v1"
CANDIDATE_DRAFT_SCHEMA_VERSION = "stylo.lobo-vnext.candidate-draft.v1"
CANDIDATE_INVENTORY_SCHEMA_VERSION = "stylo.lobo-vnext.candidate-inventory.v1"

RAW_IDENTITY_FIELDS = ("relative_path", "byte_size", "sha256")
TEXT_POLICY_DISPOSITIONS = frozenset(
    {"preserve", "transform_versioned", "manual_required"}
)
CHUNKER_MODES = frozenset(
    {
        "whole_work",
        "fixed_words",
        "fixed_tokens",
        "fixed_characters",
        "external_versioned",
    }
)
CANDIDATE_EDGE_TYPES = frozenset(
    {
        "exact_duplicate",
        "edition_of",
        "contains",
        "excerpt_of",
        "collection_member",
        "manual",
        "word5_asymmetric_containment",
    }
)
CANDIDATE_ORIGINS = frozenset({"automatic", "manual"})
CANDIDATE_DISPOSITIONS = frozenset(
    {"same_component", "separate_components", "unresolved"}
)
RESOLVED_DISPOSITIONS = frozenset(
    {"same_component", "separate_components"}
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$")


_STRICT = ExactFieldReader(
    VNextContractError,
    string_policy="nul_separate",
    hash_message="must be 64 lowercase hex characters",
    default_minimum=None,
)
_exact_object = _STRICT.object
_exact_list = _STRICT.array
_exact_str = _STRICT.string


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise VNextContractError(f"{label} must be an exact boolean")
    return value


_exact_int = _STRICT.integer
_sha256 = _STRICT.sha256


def _opaque_id(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise VNextContractError(
            f"{label} must be an opaque identifier, not a filesystem path"
        )
    return text


def _relative_work_id(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise VNextContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise VNextContractError(
            f"{label} must be a canonical relative work identifier"
        )
    return text


def _sorted_unique_work_ids(value: object, label: str) -> tuple[str, ...]:
    rows = _exact_list(value, label, nonempty=True)
    work_ids = tuple(
        _relative_work_id(item, f"{label}[]")
        for item in rows
    )
    if work_ids != tuple(sorted(set(work_ids))):
        raise VNextContractError(f"{label} must be sorted and duplicate-free")
    return work_ids


def _payload_self_hash(payload: Mapping[str, object]) -> str:
    return canonical_sha256(dict(payload))


def _checked_payload(raw: dict[str, Any], label: str) -> dict[str, Any]:
    recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    if recorded != _payload_self_hash(payload):
        raise VNextContractError(f"{label} self_hash mismatch")
    return payload


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
class RawByteIdentityPolicy:
    """The mandatory path-independent literal-byte identity mechanism."""

    policy_version: str
    identity_fields: tuple[str, ...]
    digest_algorithm: str

    @classmethod
    def from_dict(cls, value: object) -> "RawByteIdentityPolicy":
        raw = _exact_object(
            value,
            {"policy_version", "identity_fields", "digest_algorithm"},
            "raw_byte_identity",
        )
        policy_version = _opaque_id(
            raw["policy_version"], "raw_byte_identity.policy_version"
        )
        fields = tuple(
            _exact_str(item, "raw_byte_identity.identity_fields[]")
            for item in _exact_list(
                raw["identity_fields"],
                "raw_byte_identity.identity_fields",
                nonempty=True,
            )
        )
        if fields != RAW_IDENTITY_FIELDS:
            raise VNextContractError(
                "raw_byte_identity.identity_fields must be exactly "
                "['relative_path','byte_size','sha256'] in that order"
            )
        algorithm = _exact_str(
            raw["digest_algorithm"], "raw_byte_identity.digest_algorithm"
        )
        if algorithm != "sha256":
            raise VNextContractError(
                "raw_byte_identity.digest_algorithm must be exact 'sha256'"
            )
        return cls(policy_version, fields, algorithm)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "identity_fields": list(self.identity_fields),
            "digest_algorithm": self.digest_algorithm,
        }


@dataclasses.dataclass(frozen=True)
class StrictUTF8Policy:
    """Strict decoding is fixed; BOM handling remains an explicit decision."""

    policy_version: str
    encoding: str
    errors: str
    bom_disposition: str

    @classmethod
    def from_dict(cls, value: object) -> "StrictUTF8Policy":
        raw = _exact_object(
            value,
            {"policy_version", "encoding", "errors", "bom_disposition"},
            "strict_utf8",
        )
        policy_version = _opaque_id(
            raw["policy_version"], "strict_utf8.policy_version"
        )
        encoding = _exact_str(raw["encoding"], "strict_utf8.encoding")
        errors = _exact_str(raw["errors"], "strict_utf8.errors")
        bom = _exact_str(
            raw["bom_disposition"], "strict_utf8.bom_disposition"
        )
        if encoding != "utf-8" or errors != "strict":
            raise VNextContractError(
                "strict_utf8 must use encoding='utf-8' and errors='strict'"
            )
        if bom not in {"preserve", "reject", "strip_versioned"}:
            raise VNextContractError(
                "strict_utf8.bom_disposition must be explicit"
            )
        return cls(policy_version, encoding, errors, bom)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "encoding": self.encoding,
            "errors": self.errors,
            "bom_disposition": self.bom_disposition,
        }


@dataclasses.dataclass(frozen=True)
class VersionedTextPolicy:
    """A version/hash-bound choice to preserve, transform, or review text."""

    policy_version: str
    disposition: str
    policy_document_sha256: str

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        label: str,
    ) -> "VersionedTextPolicy":
        raw = _exact_object(
            value,
            {"policy_version", "disposition", "policy_document_sha256"},
            label,
        )
        disposition = _exact_str(
            raw["disposition"], f"{label}.disposition"
        )
        if disposition not in TEXT_POLICY_DISPOSITIONS:
            raise VNextContractError(
                f"{label}.disposition must be one of "
                f"{sorted(TEXT_POLICY_DISPOSITIONS)}"
            )
        return cls(
            _opaque_id(raw["policy_version"], f"{label}.policy_version"),
            disposition,
            _sha256(
                raw["policy_document_sha256"],
                f"{label}.policy_document_sha256",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "disposition": self.disposition,
            "policy_document_sha256": self.policy_document_sha256,
        }


@dataclasses.dataclass(frozen=True)
class ChunkerPolicy:
    """An explicit chunker family plus a hash-bound complete parameter document."""

    policy_version: str
    mode: str
    policy_document_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "ChunkerPolicy":
        raw = _exact_object(
            value,
            {"policy_version", "mode", "policy_document_sha256"},
            "chunker_policy",
        )
        mode = _exact_str(raw["mode"], "chunker_policy.mode")
        if mode not in CHUNKER_MODES:
            raise VNextContractError(
                f"chunker_policy.mode must be one of {sorted(CHUNKER_MODES)}"
            )
        return cls(
            _opaque_id(
                raw["policy_version"], "chunker_policy.policy_version"
            ),
            mode,
            _sha256(
                raw["policy_document_sha256"],
                "chunker_policy.policy_document_sha256",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "mode": self.mode,
            "policy_document_sha256": self.policy_document_sha256,
        }


@dataclasses.dataclass(frozen=True)
class ExactRational:
    numerator: int
    denominator: int

    @classmethod
    def from_dict(cls, value: object, *, label: str) -> "ExactRational":
        raw = _exact_object(value, {"numerator", "denominator"}, label)
        numerator = _exact_int(
            raw["numerator"], f"{label}.numerator", minimum=1
        )
        denominator = _exact_int(
            raw["denominator"], f"{label}.denominator", minimum=1
        )
        if numerator > denominator:
            raise VNextContractError(
                f"{label} must lie in the exact interval (0, 1]"
            )
        if math.gcd(numerator, denominator) != 1:
            raise VNextContractError(
                f"{label} must be in lowest terms for canonical identity"
            )
        return cls(numerator, denominator)

    def to_dict(self) -> dict[str, int]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


@dataclasses.dataclass(frozen=True)
class LiteralCandidateMechanism:
    policy_version: str
    comparison: str

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        label: str,
        expected_comparison: str,
    ) -> "LiteralCandidateMechanism":
        raw = _exact_object(
            value, {"policy_version", "comparison"}, label
        )
        comparison = _exact_str(
            raw["comparison"], f"{label}.comparison"
        )
        if comparison != expected_comparison:
            raise VNextContractError(
                f"{label}.comparison must be exact {expected_comparison!r}"
            )
        return cls(
            _opaque_id(raw["policy_version"], f"{label}.policy_version"),
            comparison,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "policy_version": self.policy_version,
            "comparison": self.comparison,
        }


@dataclasses.dataclass(frozen=True)
class Word5ContainmentPolicy:
    policy_version: str
    shingle_size: int
    comparison: str
    threshold: ExactRational
    threshold_boundary: str
    min_shingles: int
    sample_size: int
    final_verification: str

    @classmethod
    def from_dict(cls, value: object) -> "Word5ContainmentPolicy":
        raw = _exact_object(
            value,
            {
                "policy_version",
                "shingle_size",
                "comparison",
                "threshold",
                "threshold_boundary",
                "min_shingles",
                "sample_size",
                "final_verification",
            },
            "automatic_candidates.word5_containment",
        )
        shingle_size = _exact_int(
            raw["shingle_size"],
            "automatic_candidates.word5_containment.shingle_size",
            minimum=1,
        )
        if shingle_size != 5:
            raise VNextContractError(
                "word5_containment.shingle_size must be exact integer 5"
            )
        comparison = _exact_str(
            raw["comparison"],
            "automatic_candidates.word5_containment.comparison",
        )
        if comparison != "asymmetric_containment":
            raise VNextContractError(
                "word5_containment.comparison must be "
                "'asymmetric_containment'"
            )
        boundary = _exact_str(
            raw["threshold_boundary"],
            "automatic_candidates.word5_containment.threshold_boundary",
        )
        if boundary != "inclusive":
            raise VNextContractError(
                "word5_containment.threshold_boundary must be 'inclusive'"
            )
        final_verification = _exact_str(
            raw["final_verification"],
            "automatic_candidates.word5_containment.final_verification",
        )
        if final_verification != "exact_intersection_authoritative":
            raise VNextContractError(
                "word5_containment.final_verification must be "
                "'exact_intersection_authoritative'"
            )
        return cls(
            _opaque_id(
                raw["policy_version"],
                "automatic_candidates.word5_containment.policy_version",
            ),
            shingle_size,
            comparison,
            ExactRational.from_dict(
                raw["threshold"],
                label="automatic_candidates.word5_containment.threshold",
            ),
            boundary,
            _exact_int(
                raw["min_shingles"],
                "automatic_candidates.word5_containment.min_shingles",
                minimum=1,
            ),
            _exact_int(
                raw["sample_size"],
                "automatic_candidates.word5_containment.sample_size",
                minimum=1,
            ),
            final_verification,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "shingle_size": self.shingle_size,
            "comparison": self.comparison,
            "threshold": self.threshold.to_dict(),
            "threshold_boundary": self.threshold_boundary,
            "min_shingles": self.min_shingles,
            "sample_size": self.sample_size,
            "final_verification": self.final_verification,
        }


@dataclasses.dataclass(frozen=True)
class AutomaticCandidateMechanisms:
    exact_duplicate: LiteralCandidateMechanism
    literal_contains: LiteralCandidateMechanism
    word5_containment: Word5ContainmentPolicy | None

    @classmethod
    def from_dict(cls, value: object) -> "AutomaticCandidateMechanisms":
        raw = _exact_object(
            value,
            {"exact_duplicate", "literal_contains", "word5_containment"},
            "automatic_candidates",
        )
        word5_raw = raw["word5_containment"]
        if word5_raw is None:
            word5 = None
        else:
            word5 = Word5ContainmentPolicy.from_dict(word5_raw)
        return cls(
            LiteralCandidateMechanism.from_dict(
                raw["exact_duplicate"],
                label="automatic_candidates.exact_duplicate",
                expected_comparison="literal_bytes_equal",
            ),
            LiteralCandidateMechanism.from_dict(
                raw["literal_contains"],
                label="automatic_candidates.literal_contains",
                expected_comparison="literal_byte_subsequence",
            ),
            word5,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "exact_duplicate": self.exact_duplicate.to_dict(),
            "literal_contains": self.literal_contains.to_dict(),
            "word5_containment": (
                None
                if self.word5_containment is None
                else self.word5_containment.to_dict()
            ),
        }


@dataclasses.dataclass(frozen=True)
class ContentPolicySpec:
    schema_version: str
    policy_id: str
    raw_byte_identity: RawByteIdentityPolicy
    strict_utf8: StrictUTF8Policy
    canonical_row_policy: VersionedTextPolicy
    chunker_policy: ChunkerPolicy
    yo_e_policy: VersionedTextPolicy
    historical_orthography_policy: VersionedTextPolicy
    ocr_policy: VersionedTextPolicy
    markup_policy: VersionedTextPolicy
    automatic_candidates: AutomaticCandidateMechanisms
    manual_disposition_required: bool
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        policy_id: str,
        raw_byte_identity: RawByteIdentityPolicy,
        strict_utf8: StrictUTF8Policy,
        canonical_row_policy: VersionedTextPolicy,
        chunker_policy: ChunkerPolicy,
        yo_e_policy: VersionedTextPolicy,
        historical_orthography_policy: VersionedTextPolicy,
        ocr_policy: VersionedTextPolicy,
        markup_policy: VersionedTextPolicy,
        automatic_candidates: AutomaticCandidateMechanisms,
        manual_disposition_required: bool,
    ) -> "ContentPolicySpec":
        nested = {
            "raw_byte_identity": raw_byte_identity,
            "strict_utf8": strict_utf8,
            "canonical_row_policy": canonical_row_policy,
            "chunker_policy": chunker_policy,
            "yo_e_policy": yo_e_policy,
            "historical_orthography_policy": historical_orthography_policy,
            "ocr_policy": ocr_policy,
            "markup_policy": markup_policy,
            "automatic_candidates": automatic_candidates,
        }
        expected_types = {
            "raw_byte_identity": RawByteIdentityPolicy,
            "strict_utf8": StrictUTF8Policy,
            "canonical_row_policy": VersionedTextPolicy,
            "chunker_policy": ChunkerPolicy,
            "yo_e_policy": VersionedTextPolicy,
            "historical_orthography_policy": VersionedTextPolicy,
            "ocr_policy": VersionedTextPolicy,
            "markup_policy": VersionedTextPolicy,
            "automatic_candidates": AutomaticCandidateMechanisms,
        }
        for label, value in nested.items():
            if type(value) is not expected_types[label]:
                raise VNextContractError(
                    f"{label} must be exactly {expected_types[label].__name__}"
                )
        payload = {
            "schema_version": CONTENT_POLICY_SPEC_SCHEMA_VERSION,
            "policy_id": policy_id,
            "raw_byte_identity": raw_byte_identity.to_dict(),
            "strict_utf8": strict_utf8.to_dict(),
            "canonical_row_policy": canonical_row_policy.to_dict(),
            "chunker_policy": chunker_policy.to_dict(),
            "yo_e_policy": yo_e_policy.to_dict(),
            "historical_orthography_policy": (
                historical_orthography_policy.to_dict()
            ),
            "ocr_policy": ocr_policy.to_dict(),
            "markup_policy": markup_policy.to_dict(),
            "automatic_candidates": automatic_candidates.to_dict(),
            "manual_disposition_required": manual_disposition_required,
        }
        return cls.from_dict(
            {**payload, "self_hash": _payload_self_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "ContentPolicySpec":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "policy_id",
                "raw_byte_identity",
                "strict_utf8",
                "canonical_row_policy",
                "chunker_policy",
                "yo_e_policy",
                "historical_orthography_policy",
                "ocr_policy",
                "markup_policy",
                "automatic_candidates",
                "manual_disposition_required",
                "self_hash",
            },
            "content policy spec",
        )
        payload = _checked_payload(raw, "content policy spec")
        if payload["schema_version"] != CONTENT_POLICY_SPEC_SCHEMA_VERSION:
            raise VNextContractError(
                "content policy spec is legacy, unversioned, or unsupported"
            )
        manual_required = _exact_bool(
            payload["manual_disposition_required"],
            "manual_disposition_required",
        )
        if manual_required is not True:
            raise VNextContractError(
                "manual_disposition_required must be exact true"
            )
        return cls(
            CONTENT_POLICY_SPEC_SCHEMA_VERSION,
            _opaque_id(payload["policy_id"], "policy_id"),
            RawByteIdentityPolicy.from_dict(payload["raw_byte_identity"]),
            StrictUTF8Policy.from_dict(payload["strict_utf8"]),
            VersionedTextPolicy.from_dict(
                payload["canonical_row_policy"],
                label="canonical_row_policy",
            ),
            ChunkerPolicy.from_dict(payload["chunker_policy"]),
            VersionedTextPolicy.from_dict(
                payload["yo_e_policy"], label="yo_e_policy"
            ),
            VersionedTextPolicy.from_dict(
                payload["historical_orthography_policy"],
                label="historical_orthography_policy",
            ),
            VersionedTextPolicy.from_dict(
                payload["ocr_policy"], label="ocr_policy"
            ),
            VersionedTextPolicy.from_dict(
                payload["markup_policy"], label="markup_policy"
            ),
            AutomaticCandidateMechanisms.from_dict(
                payload["automatic_candidates"]
            ),
            manual_required,
            _sha256(raw["self_hash"], "content policy spec.self_hash"),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "raw_byte_identity": self.raw_byte_identity.to_dict(),
            "strict_utf8": self.strict_utf8.to_dict(),
            "canonical_row_policy": self.canonical_row_policy.to_dict(),
            "chunker_policy": self.chunker_policy.to_dict(),
            "yo_e_policy": self.yo_e_policy.to_dict(),
            "historical_orthography_policy": (
                self.historical_orthography_policy.to_dict()
            ),
            "ocr_policy": self.ocr_policy.to_dict(),
            "markup_policy": self.markup_policy.to_dict(),
            "automatic_candidates": self.automatic_candidates.to_dict(),
            "manual_disposition_required": self.manual_disposition_required,
        }

    def validate(self) -> "ContentPolicySpec":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError("content policy spec is noncanonical")
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


@dataclasses.dataclass(frozen=True)
class ManualDisposition:
    decision_id: str
    disposition: str
    evidence_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "ManualDisposition":
        raw = _exact_object(
            value,
            {"decision_id", "disposition", "evidence_sha256"},
            "manual_disposition",
        )
        disposition = _exact_str(
            raw["disposition"], "manual_disposition.disposition"
        )
        if disposition not in RESOLVED_DISPOSITIONS:
            raise VNextContractError(
                "manual_disposition.disposition must resolve the candidate"
            )
        return cls(
            _opaque_id(raw["decision_id"], "manual_disposition.decision_id"),
            disposition,
            _sha256(
                raw["evidence_sha256"],
                "manual_disposition.evidence_sha256",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "decision_id": self.decision_id,
            "disposition": self.disposition,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclasses.dataclass(frozen=True)
class CandidateDraft:
    schema_version: str
    candidate_id: str
    left_work_id: str
    right_work_id: str
    edge_type: str
    origin: str
    evidence_sha256: str
    disposition: str
    manual_disposition: ManualDisposition | None
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        candidate_id: str,
        left_work_id: str,
        right_work_id: str,
        edge_type: str,
        origin: str,
        evidence_sha256: str,
        disposition: str,
        manual_disposition: ManualDisposition | None,
    ) -> "CandidateDraft":
        if (
            manual_disposition is not None
            and type(manual_disposition) is not ManualDisposition
        ):
            raise VNextContractError(
                "manual_disposition must be exactly ManualDisposition or null"
            )
        payload = {
            "schema_version": CANDIDATE_DRAFT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "left_work_id": left_work_id,
            "right_work_id": right_work_id,
            "edge_type": edge_type,
            "origin": origin,
            "evidence_sha256": evidence_sha256,
            "disposition": disposition,
            "manual_disposition": (
                None
                if manual_disposition is None
                else manual_disposition.to_dict()
            ),
        }
        return cls.from_dict(
            {**payload, "self_hash": _payload_self_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "CandidateDraft":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "candidate_id",
                "left_work_id",
                "right_work_id",
                "edge_type",
                "origin",
                "evidence_sha256",
                "disposition",
                "manual_disposition",
                "self_hash",
            },
            "candidate draft",
        )
        payload = _checked_payload(raw, "candidate draft")
        if payload["schema_version"] != CANDIDATE_DRAFT_SCHEMA_VERSION:
            raise VNextContractError(
                "candidate draft is legacy, unversioned, or unsupported"
            )
        left = _relative_work_id(
            payload["left_work_id"], "candidate draft.left_work_id"
        )
        right = _relative_work_id(
            payload["right_work_id"], "candidate draft.right_work_id"
        )
        if left == right:
            raise VNextContractError("candidate draft cannot be a self-edge")
        edge_type = _exact_str(
            payload["edge_type"], "candidate draft.edge_type"
        )
        if edge_type not in CANDIDATE_EDGE_TYPES:
            raise VNextContractError(
                f"unsupported candidate draft edge_type {edge_type!r}"
            )
        origin = _exact_str(payload["origin"], "candidate draft.origin")
        if origin not in CANDIDATE_ORIGINS:
            raise VNextContractError(
                f"unsupported candidate draft origin {origin!r}"
            )
        if edge_type == "manual" and origin != "manual":
            raise VNextContractError(
                "manual candidate edge must have manual origin"
            )
        disposition = _exact_str(
            payload["disposition"], "candidate draft.disposition"
        )
        if disposition not in CANDIDATE_DISPOSITIONS:
            raise VNextContractError(
                f"unsupported candidate draft disposition {disposition!r}"
            )
        manual_raw = payload["manual_disposition"]
        manual = (
            None
            if manual_raw is None
            else ManualDisposition.from_dict(manual_raw)
        )
        if disposition == "unresolved":
            if manual is not None:
                raise VNextContractError(
                    "unresolved candidate must not carry a manual disposition"
                )
        elif manual is None:
            raise VNextContractError(
                "resolved candidate requires an exact manual disposition"
            )
        elif manual.disposition != disposition:
            raise VNextContractError(
                "candidate/manual disposition mismatch"
            )
        return cls(
            CANDIDATE_DRAFT_SCHEMA_VERSION,
            _opaque_id(payload["candidate_id"], "candidate draft.candidate_id"),
            left,
            right,
            edge_type,
            origin,
            _sha256(
                payload["evidence_sha256"],
                "candidate draft.evidence_sha256",
            ),
            disposition,
            manual,
            _sha256(raw["self_hash"], "candidate draft.self_hash"),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "left_work_id": self.left_work_id,
            "right_work_id": self.right_work_id,
            "edge_type": self.edge_type,
            "origin": self.origin,
            "evidence_sha256": self.evidence_sha256,
            "disposition": self.disposition,
            "manual_disposition": (
                None
                if self.manual_disposition is None
                else self.manual_disposition.to_dict()
            ),
        }

    def validate(self) -> "CandidateDraft":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError("candidate draft is noncanonical")
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    @property
    def is_resolved(self) -> bool:
        return self.disposition in RESOLVED_DISPOSITIONS


@dataclasses.dataclass(frozen=True)
class CandidateInventory:
    schema_version: str
    generation_id: str
    work_identity_catalog_digest: str
    raw_inventory_digest: str
    content_policy_spec_digest: str
    included_work_ids: tuple[str, ...]
    candidates: tuple[CandidateDraft, ...]
    unresolved_candidate_ids: tuple[str, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        generation_id: str,
        work_identity_catalog_digest: str,
        raw_inventory_digest: str,
        content_policy_spec_digest: str,
        included_work_ids: Sequence[str],
        candidates: Sequence[CandidateDraft],
    ) -> "CandidateInventory":
        if isinstance(included_work_ids, (str, bytes)):
            raise VNextContractError(
                "included_work_ids must be a sequence, not a string"
            )
        if isinstance(candidates, (str, bytes)):
            raise VNextContractError(
                "candidates must be a sequence, not a string"
            )
        candidate_rows = list(candidates)
        if any(type(candidate) is not CandidateDraft for candidate in candidate_rows):
            raise VNextContractError(
                "candidates must contain exact CandidateDraft objects"
            )
        sorted_candidates = sorted(
            candidate_rows, key=lambda item: item.candidate_id
        )
        unresolved = sorted(
            candidate.candidate_id
            for candidate in sorted_candidates
            if not candidate.is_resolved
        )
        payload = {
            "schema_version": CANDIDATE_INVENTORY_SCHEMA_VERSION,
            "generation_id": generation_id,
            "work_identity_catalog_digest": work_identity_catalog_digest,
            "raw_inventory_digest": raw_inventory_digest,
            "content_policy_spec_digest": content_policy_spec_digest,
            "included_work_ids": sorted(included_work_ids),
            "candidates": [
                candidate.to_dict() for candidate in sorted_candidates
            ],
            "unresolved_candidate_ids": unresolved,
        }
        return cls.from_dict(
            {**payload, "self_hash": _payload_self_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "CandidateInventory":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "generation_id",
                "work_identity_catalog_digest",
                "raw_inventory_digest",
                "content_policy_spec_digest",
                "included_work_ids",
                "candidates",
                "unresolved_candidate_ids",
                "self_hash",
            },
            "candidate inventory",
        )
        payload = _checked_payload(raw, "candidate inventory")
        if payload["schema_version"] != CANDIDATE_INVENTORY_SCHEMA_VERSION:
            raise VNextContractError(
                "candidate inventory is legacy, unversioned, or unsupported"
            )
        work_ids = _sorted_unique_work_ids(
            payload["included_work_ids"], "included_work_ids"
        )
        candidate_rows = _exact_list(
            payload["candidates"], "candidates"
        )
        candidates = tuple(
            CandidateDraft.from_dict(item) for item in candidate_rows
        )
        candidate_ids = tuple(
            candidate.candidate_id for candidate in candidates
        )
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise VNextContractError(
                "candidates must be sorted by unique candidate_id"
            )
        known_works = set(work_ids)
        for candidate in candidates:
            if (
                candidate.left_work_id not in known_works
                or candidate.right_work_id not in known_works
            ):
                raise VNextContractError(
                    "candidate references a work outside included_work_ids"
                )
        expected_unresolved = tuple(
            candidate.candidate_id
            for candidate in candidates
            if not candidate.is_resolved
        )
        unresolved = tuple(
            _opaque_id(item, "unresolved_candidate_ids[]")
            for item in _exact_list(
                payload["unresolved_candidate_ids"],
                "unresolved_candidate_ids",
            )
        )
        if unresolved != expected_unresolved:
            raise VNextContractError(
                "unresolved_candidate_ids must exactly match candidate records"
            )
        resolved_by_pair: dict[tuple[str, str], set[str]] = {}
        for candidate in candidates:
            if not candidate.is_resolved:
                continue
            pair = tuple(
                sorted((candidate.left_work_id, candidate.right_work_id))
            )
            resolved_by_pair.setdefault(pair, set()).add(candidate.disposition)
        if any(len(dispositions) > 1 for dispositions in resolved_by_pair.values()):
            raise VNextContractError(
                "candidate inventory has conflicting manual dispositions"
            )
        return cls(
            CANDIDATE_INVENTORY_SCHEMA_VERSION,
            _opaque_id(payload["generation_id"], "generation_id"),
            _sha256(
                payload["work_identity_catalog_digest"],
                "work_identity_catalog_digest",
            ),
            _sha256(
                payload["raw_inventory_digest"], "raw_inventory_digest"
            ),
            _sha256(
                payload["content_policy_spec_digest"],
                "content_policy_spec_digest",
            ),
            work_ids,
            candidates,
            unresolved,
            _sha256(raw["self_hash"], "candidate inventory.self_hash"),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "work_identity_catalog_digest": self.work_identity_catalog_digest,
            "raw_inventory_digest": self.raw_inventory_digest,
            "content_policy_spec_digest": self.content_policy_spec_digest,
            "included_work_ids": list(self.included_work_ids),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "unresolved_candidate_ids": list(self.unresolved_candidate_ids),
        }

    def validate(
        self,
        *,
        content_policy_spec: ContentPolicySpec | None = None,
    ) -> "CandidateInventory":
        rebuilt = type(self).from_dict(
            {**self._payload(), "self_hash": self.self_hash}
        )
        if rebuilt != self:
            raise VNextContractError("candidate inventory is noncanonical")
        if content_policy_spec is not None:
            if type(content_policy_spec) is not ContentPolicySpec:
                raise VNextContractError(
                    "content_policy_spec must be exactly ContentPolicySpec"
                )
            content_policy_spec.validate()
            if self.content_policy_spec_digest != content_policy_spec.self_hash:
                raise VNextContractError(
                    "candidate inventory/content policy digest mismatch"
                )
        return self

    def assert_resolved_for_component_manifest(self) -> "CandidateInventory":
        self.validate()
        if self.unresolved_candidate_ids:
            raise VNextContractError(
                "unresolved candidate drafts block ContentComponentManifest: "
                f"{list(self.unresolved_candidate_ids)}"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


def loads_content_policy_spec(text: str) -> ContentPolicySpec:
    return ContentPolicySpec.from_dict(
        _strict_raw(text, "content policy spec")
    )


def load_content_policy_spec(
    path: str | os.PathLike[str],
) -> ContentPolicySpec:
    return ContentPolicySpec.from_dict(
        _strict_file(path, "content policy spec")
    )


def loads_candidate_draft(text: str) -> CandidateDraft:
    return CandidateDraft.from_dict(_strict_raw(text, "candidate draft"))


def load_candidate_draft(path: str | os.PathLike[str]) -> CandidateDraft:
    return CandidateDraft.from_dict(
        _strict_file(path, "candidate draft")
    )


def loads_candidate_inventory(text: str) -> CandidateInventory:
    return CandidateInventory.from_dict(
        _strict_raw(text, "candidate inventory")
    )


def load_candidate_inventory(
    path: str | os.PathLike[str],
) -> CandidateInventory:
    return CandidateInventory.from_dict(
        _strict_file(path, "candidate inventory")
    )


__all__ = [
    "CONTENT_POLICY_SPEC_SCHEMA_VERSION",
    "CANDIDATE_DRAFT_SCHEMA_VERSION",
    "CANDIDATE_INVENTORY_SCHEMA_VERSION",
    "RAW_IDENTITY_FIELDS",
    "TEXT_POLICY_DISPOSITIONS",
    "CHUNKER_MODES",
    "CANDIDATE_EDGE_TYPES",
    "CANDIDATE_ORIGINS",
    "CANDIDATE_DISPOSITIONS",
    "RESOLVED_DISPOSITIONS",
    "RawByteIdentityPolicy",
    "StrictUTF8Policy",
    "VersionedTextPolicy",
    "ChunkerPolicy",
    "ExactRational",
    "LiteralCandidateMechanism",
    "Word5ContainmentPolicy",
    "AutomaticCandidateMechanisms",
    "ContentPolicySpec",
    "ManualDisposition",
    "CandidateDraft",
    "CandidateInventory",
    "loads_content_policy_spec",
    "load_content_policy_spec",
    "loads_candidate_draft",
    "load_candidate_draft",
    "loads_candidate_inventory",
    "load_candidate_inventory",
]
