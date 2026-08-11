"""Strict canonical-row packet for real-corpus LOBO-vNext execution.

The corpus manifest identifies literal source bytes.  Model rows are a distinct
derived representation: normalization or chunking can make two different raw
files produce the same model text, so the two identities must never be
collapsed.  This module binds both sides and verifies an immutable canonical
row directory before a cache, model factory, or fit may be created.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import stat
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .._strict_fields import ExactFieldReader
from ..jsonio import StrictJSONError, load_strict, loads_strict
from .lobo_vnext import (
    CorpusVNextManifest,
    VNextContractError,
    canonical_sha256,
)


CANONICAL_REPRESENTATION_SCHEMA_VERSION = (
    "stylo.lobo-vnext.canonical-representation.v1"
)
R1_ACQUISITION_BINDING_SCHEMA_VERSION = (
    "stylo.lobo-vnext.ruaa-r1-acquisition-binding.v2"
)
R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION = (
    "stylo.lobo-vnext.ruaa-r1-corpus-generation-material.v1"
)
R1_PACKET_GENERATION_MATERIAL_SCHEMA_VERSION = (
    "stylo.lobo-vnext.ruaa-r1-packet-generation-material.v1"
)
R1_PACKET_MANIFEST_SCHEMA_VERSION = (
    "stylo.lobo-vnext.ruaa-r1-packet.v5"
)
R1_PACKET_STATUS = "owner_selected_exploratory_packet_no_fit"
CANONICAL_ROWS_DIRECTORY = "canonical_rows"

_HEX64 = frozenset("0123456789abcdef")


class VNextPacketError(VNextContractError):
    """The canonical representation packet is malformed or has drifted."""


@dataclasses.dataclass(frozen=True)
class VNextTextRow:
    """One deterministic model row selected by a work's ``raw_paths``."""

    row_id: str
    relative_path: str
    work_id: str
    author_id: str
    text: str
    raw_sha256: str


_STRICT = ExactFieldReader(
    VNextPacketError,
    string_policy="trimmed_nul",
    hash_message="must be 64 lowercase hex characters",
)
_exact_object = _STRICT.object
_exact_list = _STRICT.array
_exact_str = _STRICT.string
_exact_int = _STRICT.integer


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise VNextPacketError(f"{label} must be an exact boolean")
    return value


_sha256 = _STRICT.sha256


def _relative_path(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise VNextPacketError(f"{label} must use POSIX separators")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise VNextPacketError(f"{label} must be a canonical relative path")
    return text


def _strict_raw(text: str, label: str) -> object:
    try:
        return loads_strict(text)
    except (StrictJSONError, TypeError) as exc:
        raise VNextPacketError(f"{label}: {exc}") from exc


def _strict_file(path: str | os.PathLike[str], label: str) -> object:
    try:
        return load_strict(path)
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise VNextPacketError(f"{label}: {exc}") from exc


def _sorted_unique_strings(
    value: object,
    label: str,
    *,
    nonempty: bool,
) -> tuple[str, ...]:
    rows = tuple(
        _exact_str(item, f"{label}[]")
        for item in _exact_list(value, label, nonempty=nonempty)
    )
    if rows != tuple(sorted(set(rows))):
        raise VNextPacketError(f"{label} must be sorted and unique")
    return rows


@dataclasses.dataclass(frozen=True)
class PacketFileEntry:
    """One literal non-self file in an immutable R1 packet."""

    relative_path: str
    byte_size: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "PacketFileEntry":
        raw = _exact_object(
            value,
            {"relative_path", "byte_size", "sha256"},
            "packet file entry",
        )
        relative_path = _relative_path(
            raw["relative_path"], "packet file.relative_path"
        )
        if relative_path == "packet.json":
            raise VNextPacketError(
                "packet.json is self-hashed and must not inventory itself"
            )
        return cls(
            relative_path,
            _exact_int(
                raw["byte_size"], "packet file.byte_size", minimum=1
            ),
            _sha256(raw["sha256"], "packet file.sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclasses.dataclass(frozen=True)
class R1AcquisitionBinding:
    """Independent pre-corpus identity bound to the selected acquisition."""

    schema_version: str
    acquisition_generation_id: str
    acquisition_manifest_self_hash: str
    acquisition_receipt_self_hash: str
    selected_audit_file_sha256: str
    selected_audit_self_hash: str
    raw_inventory_digest: str
    work_identity_catalog_digest: str
    upstream_excluded_work_ids: tuple[str, ...]
    content_policy_spec_digest: str
    work_count: int
    author_count: int
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        acquisition_generation_id: str,
        acquisition_manifest_self_hash: str,
        acquisition_receipt_self_hash: str,
        selected_audit_file_sha256: str,
        selected_audit_self_hash: str,
        raw_inventory_digest: str,
        work_identity_catalog_digest: str,
        upstream_excluded_work_ids: Sequence[str],
        content_policy_spec_digest: str,
        work_count: int,
        author_count: int,
    ) -> "R1AcquisitionBinding":
        payload = {
            "schema_version": R1_ACQUISITION_BINDING_SCHEMA_VERSION,
            "acquisition_generation_id": acquisition_generation_id,
            "acquisition_manifest_self_hash": (
                acquisition_manifest_self_hash
            ),
            "acquisition_receipt_self_hash": (
                acquisition_receipt_self_hash
            ),
            "selected_audit_file_sha256": selected_audit_file_sha256,
            "selected_audit_self_hash": selected_audit_self_hash,
            "raw_inventory_digest": raw_inventory_digest,
            "work_identity_catalog_digest": (
                work_identity_catalog_digest
            ),
            "upstream_excluded_work_ids": sorted(
                upstream_excluded_work_ids
            ),
            "content_policy_spec_digest": content_policy_spec_digest,
            "work_count": work_count,
            "author_count": author_count,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1AcquisitionBinding":
        if (
            type(value) is not dict
            or value.get("schema_version")
            != R1_ACQUISITION_BINDING_SCHEMA_VERSION
        ):
            raise VNextPacketError(
                "R1 acquisition binding is legacy or unsupported"
            )
        raw = _exact_object(
            value,
            {
                "schema_version",
                "acquisition_generation_id",
                "acquisition_manifest_self_hash",
                "acquisition_receipt_self_hash",
                "selected_audit_file_sha256",
                "selected_audit_self_hash",
                "raw_inventory_digest",
                "work_identity_catalog_digest",
                "upstream_excluded_work_ids",
                "content_policy_spec_digest",
                "work_count",
                "author_count",
                "self_hash",
            },
            "R1 acquisition binding",
        )
        recorded = _sha256(
            raw["self_hash"], "R1 acquisition binding.self_hash"
        )
        payload = {
            key: child for key, child in raw.items() if key != "self_hash"
        }
        if canonical_sha256(payload) != recorded:
            raise VNextPacketError(
                "R1 acquisition binding self_hash mismatch"
            )
        excluded = _sorted_unique_strings(
            raw["upstream_excluded_work_ids"],
            "R1 acquisition binding.upstream_excluded_work_ids",
            nonempty=True,
        )
        if len(excluded) != 3:
            raise VNextPacketError(
                "R1 acquisition binding must record exactly three "
                "upstream exclusions"
            )
        return cls(
            R1_ACQUISITION_BINDING_SCHEMA_VERSION,
            _sha256(
                raw["acquisition_generation_id"],
                "R1 acquisition binding.acquisition_generation_id",
            ),
            _sha256(
                raw["acquisition_manifest_self_hash"],
                "R1 acquisition binding.acquisition_manifest_self_hash",
            ),
            _sha256(
                raw["acquisition_receipt_self_hash"],
                "R1 acquisition binding.acquisition_receipt_self_hash",
            ),
            _sha256(
                raw["selected_audit_file_sha256"],
                "R1 acquisition binding.selected_audit_file_sha256",
            ),
            _sha256(
                raw["selected_audit_self_hash"],
                "R1 acquisition binding.selected_audit_self_hash",
            ),
            _sha256(
                raw["raw_inventory_digest"],
                "R1 acquisition binding.raw_inventory_digest",
            ),
            _sha256(
                raw["work_identity_catalog_digest"],
                "R1 acquisition binding.work_identity_catalog_digest",
            ),
            excluded,
            _sha256(
                raw["content_policy_spec_digest"],
                "R1 acquisition binding.content_policy_spec_digest",
            ),
            _exact_int(
                raw["work_count"],
                "R1 acquisition binding.work_count",
                minimum=1,
            ),
            _exact_int(
                raw["author_count"],
                "R1 acquisition binding.author_count",
                minimum=1,
            ),
            recorded,
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "acquisition_generation_id": self.acquisition_generation_id,
            "acquisition_manifest_self_hash": (
                self.acquisition_manifest_self_hash
            ),
            "acquisition_receipt_self_hash": (
                self.acquisition_receipt_self_hash
            ),
            "selected_audit_file_sha256": (
                self.selected_audit_file_sha256
            ),
            "selected_audit_self_hash": self.selected_audit_self_hash,
            "raw_inventory_digest": self.raw_inventory_digest,
            "work_identity_catalog_digest": (
                self.work_identity_catalog_digest
            ),
            "upstream_excluded_work_ids": list(
                self.upstream_excluded_work_ids
            ),
            "content_policy_spec_digest": (
                self.content_policy_spec_digest
            ),
            "work_count": self.work_count,
            "author_count": self.author_count,
        }
        if (
            type(self) is not R1AcquisitionBinding
            or canonical_sha256(payload) != self.self_hash
        ):
            raise VNextPacketError(
                "R1 acquisition binding is noncanonical"
            )
        return {**payload, "self_hash": self.self_hash}

    def validate(self) -> "R1AcquisitionBinding":
        if type(self).from_dict(self.to_dict()) != self:
            raise VNextPacketError(
                "R1 acquisition binding is noncanonical"
            )
        return self


@dataclasses.dataclass(frozen=True)
class R1CorpusGenerationMaterial:
    """Versioned pre-corpus material whose self-hash is the corpus identity."""

    schema_version: str
    acquisition_binding_self_hash: str
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        acquisition_binding_self_hash: str,
    ) -> "R1CorpusGenerationMaterial":
        payload = {
            "schema_version": R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION,
            "acquisition_binding_self_hash": acquisition_binding_self_hash,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1CorpusGenerationMaterial":
        if (
            type(value) is not dict
            or value.get("schema_version")
            != R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION
        ):
            raise VNextPacketError(
                "R1 corpus generation material is legacy or unsupported"
            )
        raw = _exact_object(
            value,
            {
                "schema_version",
                "acquisition_binding_self_hash",
                "self_hash",
            },
            "R1 corpus generation material",
        )
        recorded = _sha256(
            raw["self_hash"], "R1 corpus generation material.self_hash"
        )
        payload = {
            key: child for key, child in raw.items() if key != "self_hash"
        }
        if canonical_sha256(payload) != recorded:
            raise VNextPacketError(
                "R1 corpus generation material self_hash mismatch"
            )
        return cls(
            R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION,
            _sha256(
                raw["acquisition_binding_self_hash"],
                "R1 corpus generation material."
                "acquisition_binding_self_hash",
            ),
            recorded,
        )

    @property
    def corpus_generation_id(self) -> str:
        return self.self_hash

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "acquisition_binding_self_hash": (
                self.acquisition_binding_self_hash
            ),
        }
        if (
            type(self) is not R1CorpusGenerationMaterial
            or canonical_sha256(payload) != self.self_hash
        ):
            raise VNextPacketError(
                "R1 corpus generation material is noncanonical"
            )
        return {**payload, "self_hash": self.self_hash}

    def validate(self) -> "R1CorpusGenerationMaterial":
        if type(self).from_dict(self.to_dict()) != self:
            raise VNextPacketError(
                "R1 corpus generation material is noncanonical"
            )
        return self


@dataclasses.dataclass(frozen=True)
class R1PacketGenerationMaterial:
    """All non-packet bytes and contract digests defining a packet root."""

    schema_version: str
    packet_schema_version: str
    corpus_generation_id: str
    acquisition_binding_self_hash: str
    file_inventory_sha256: str
    content_policy_spec_sha256: str
    candidate_inventory_sha256: str
    corpus_manifest_sha256: str
    content_component_manifest_sha256: str
    fold_manifest_sha256: str
    primary_model_spec_sha256: str
    baseline_model_spec_sha256: str
    inference_spec_sha256: str
    primary_inner_cv_plan_sha256: str
    baseline_inner_cv_plan_sha256: str
    model_role_manifest_sha256: str
    campaign_manifest_sha256: str
    representation_receipt_sha256: str

    @classmethod
    def build(
        cls,
        *,
        packet_schema_version: str = R1_PACKET_MANIFEST_SCHEMA_VERSION,
        corpus_generation_id: str,
        acquisition_binding_self_hash: str,
        file_inventory_sha256: str,
        content_policy_spec_sha256: str,
        candidate_inventory_sha256: str,
        corpus_manifest_sha256: str,
        content_component_manifest_sha256: str,
        fold_manifest_sha256: str,
        primary_model_spec_sha256: str,
        baseline_model_spec_sha256: str,
        inference_spec_sha256: str,
        primary_inner_cv_plan_sha256: str,
        baseline_inner_cv_plan_sha256: str,
        model_role_manifest_sha256: str,
        campaign_manifest_sha256: str,
        representation_receipt_sha256: str,
    ) -> "R1PacketGenerationMaterial":
        payload = {
            "schema_version": R1_PACKET_GENERATION_MATERIAL_SCHEMA_VERSION,
            "packet_schema_version": packet_schema_version,
            "corpus_generation_id": corpus_generation_id,
            "acquisition_binding_self_hash": acquisition_binding_self_hash,
            "file_inventory_sha256": file_inventory_sha256,
            "content_policy_spec_sha256": content_policy_spec_sha256,
            "candidate_inventory_sha256": candidate_inventory_sha256,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "content_component_manifest_sha256": (
                content_component_manifest_sha256
            ),
            "fold_manifest_sha256": fold_manifest_sha256,
            "primary_model_spec_sha256": primary_model_spec_sha256,
            "baseline_model_spec_sha256": baseline_model_spec_sha256,
            "inference_spec_sha256": inference_spec_sha256,
            "primary_inner_cv_plan_sha256": (
                primary_inner_cv_plan_sha256
            ),
            "baseline_inner_cv_plan_sha256": (
                baseline_inner_cv_plan_sha256
            ),
            "model_role_manifest_sha256": model_role_manifest_sha256,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "representation_receipt_sha256": (
                representation_receipt_sha256
            ),
        }
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, value: object) -> "R1PacketGenerationMaterial":
        if (
            type(value) is not dict
            or value.get("schema_version")
            != R1_PACKET_GENERATION_MATERIAL_SCHEMA_VERSION
        ):
            raise VNextPacketError(
                "R1 packet generation material is legacy or unsupported"
            )
        raw = _exact_object(
            value,
            {
                "schema_version",
                "packet_schema_version",
                "corpus_generation_id",
                "acquisition_binding_self_hash",
                "file_inventory_sha256",
                "content_policy_spec_sha256",
                "candidate_inventory_sha256",
                "corpus_manifest_sha256",
                "content_component_manifest_sha256",
                "fold_manifest_sha256",
                "primary_model_spec_sha256",
                "baseline_model_spec_sha256",
                "inference_spec_sha256",
                "primary_inner_cv_plan_sha256",
                "baseline_inner_cv_plan_sha256",
                "model_role_manifest_sha256",
                "campaign_manifest_sha256",
                "representation_receipt_sha256",
            },
            "R1 packet generation material",
        )
        if raw["packet_schema_version"] != R1_PACKET_MANIFEST_SCHEMA_VERSION:
            raise VNextPacketError(
                "R1 packet generation material packet schema is unsupported"
            )
        sha_fields = (
            "corpus_generation_id",
            "acquisition_binding_self_hash",
            "file_inventory_sha256",
            "content_policy_spec_sha256",
            "candidate_inventory_sha256",
            "corpus_manifest_sha256",
            "content_component_manifest_sha256",
            "fold_manifest_sha256",
            "primary_model_spec_sha256",
            "baseline_model_spec_sha256",
            "inference_spec_sha256",
            "primary_inner_cv_plan_sha256",
            "baseline_inner_cv_plan_sha256",
            "model_role_manifest_sha256",
            "campaign_manifest_sha256",
            "representation_receipt_sha256",
        )
        digests = {
            field: _sha256(
                raw[field], f"R1 packet generation material.{field}"
            )
            for field in sha_fields
        }
        return cls(
            R1_PACKET_GENERATION_MATERIAL_SCHEMA_VERSION,
            R1_PACKET_MANIFEST_SCHEMA_VERSION,
            *(digests[field] for field in sha_fields),
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "packet_schema_version": self.packet_schema_version,
            "corpus_generation_id": self.corpus_generation_id,
            "acquisition_binding_self_hash": (
                self.acquisition_binding_self_hash
            ),
            "file_inventory_sha256": self.file_inventory_sha256,
            "content_policy_spec_sha256": (
                self.content_policy_spec_sha256
            ),
            "candidate_inventory_sha256": (
                self.candidate_inventory_sha256
            ),
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "content_component_manifest_sha256": (
                self.content_component_manifest_sha256
            ),
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "primary_model_spec_sha256": self.primary_model_spec_sha256,
            "baseline_model_spec_sha256": self.baseline_model_spec_sha256,
            "inference_spec_sha256": self.inference_spec_sha256,
            "primary_inner_cv_plan_sha256": (
                self.primary_inner_cv_plan_sha256
            ),
            "baseline_inner_cv_plan_sha256": (
                self.baseline_inner_cv_plan_sha256
            ),
            "model_role_manifest_sha256": (
                self.model_role_manifest_sha256
            ),
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "representation_receipt_sha256": (
                self.representation_receipt_sha256
            ),
        }
        if (
            type(self) is not R1PacketGenerationMaterial
            or type(self).from_dict(payload) != self
        ):
            raise VNextPacketError(
                "R1 packet generation material is noncanonical"
            )
        return payload

    @property
    def packet_generation_id(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate(self) -> "R1PacketGenerationMaterial":
        if type(self).from_dict(self.to_dict()) != self:
            raise VNextPacketError(
                "R1 packet generation material is noncanonical"
            )
        return self


@dataclasses.dataclass(frozen=True)
class R1PacketManifest:
    """Self-hashed v5 index for one content-addressed R1 packet."""

    schema_version: str
    status: str
    confirmatory_authorized: bool
    acquisition_binding: R1AcquisitionBinding
    corpus_generation_material: R1CorpusGenerationMaterial
    corpus_generation_id: str
    packet_generation_material: R1PacketGenerationMaterial
    packet_generation_id: str
    selected_work_count: int
    files: tuple[PacketFileEntry, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        acquisition_binding: R1AcquisitionBinding,
        corpus_generation_material: R1CorpusGenerationMaterial,
        content_policy_spec_sha256: str,
        candidate_inventory_sha256: str,
        corpus_manifest_sha256: str,
        content_component_manifest_sha256: str,
        fold_manifest_sha256: str,
        primary_model_spec_sha256: str,
        baseline_model_spec_sha256: str,
        inference_spec_sha256: str,
        primary_inner_cv_plan_sha256: str,
        baseline_inner_cv_plan_sha256: str,
        model_role_manifest_sha256: str,
        campaign_manifest_sha256: str,
        representation_receipt_sha256: str,
        files: Sequence[PacketFileEntry],
    ) -> "R1PacketManifest":
        if type(acquisition_binding) is not R1AcquisitionBinding:
            raise VNextPacketError(
                "acquisition_binding must be exactly R1AcquisitionBinding"
            )
        acquisition_binding.validate()
        if (
            type(corpus_generation_material)
            is not R1CorpusGenerationMaterial
        ):
            raise VNextPacketError(
                "corpus_generation_material must be exactly "
                "R1CorpusGenerationMaterial"
            )
        corpus_generation_material.validate()
        if (
            corpus_generation_material.acquisition_binding_self_hash
            != acquisition_binding.self_hash
        ):
            raise VNextPacketError(
                "corpus generation material differs from acquisition binding"
            )
        if (
            content_policy_spec_sha256
            != acquisition_binding.content_policy_spec_digest
        ):
            raise VNextPacketError(
                "packet content policy differs from acquisition binding"
            )
        if type(files) not in (list, tuple):
            raise VNextPacketError("packet files must be an exact list or tuple")
        file_rows = tuple(files)
        if not file_rows or any(
            type(row) is not PacketFileEntry for row in file_rows
        ):
            raise VNextPacketError(
                "packet files must contain exact PacketFileEntry records"
            )
        file_dicts = [row.to_dict() for row in file_rows]
        generation_material = R1PacketGenerationMaterial.build(
            corpus_generation_id=(
                corpus_generation_material.corpus_generation_id
            ),
            acquisition_binding_self_hash=acquisition_binding.self_hash,
            file_inventory_sha256=canonical_sha256(file_dicts),
            content_policy_spec_sha256=content_policy_spec_sha256,
            candidate_inventory_sha256=candidate_inventory_sha256,
            corpus_manifest_sha256=corpus_manifest_sha256,
            content_component_manifest_sha256=(
                content_component_manifest_sha256
            ),
            fold_manifest_sha256=fold_manifest_sha256,
            primary_model_spec_sha256=primary_model_spec_sha256,
            baseline_model_spec_sha256=baseline_model_spec_sha256,
            inference_spec_sha256=inference_spec_sha256,
            primary_inner_cv_plan_sha256=primary_inner_cv_plan_sha256,
            baseline_inner_cv_plan_sha256=baseline_inner_cv_plan_sha256,
            model_role_manifest_sha256=model_role_manifest_sha256,
            campaign_manifest_sha256=campaign_manifest_sha256,
            representation_receipt_sha256=representation_receipt_sha256,
        )
        payload = {
            "schema_version": R1_PACKET_MANIFEST_SCHEMA_VERSION,
            "status": R1_PACKET_STATUS,
            "confirmatory_authorized": False,
            "acquisition_binding": acquisition_binding.to_dict(),
            "corpus_generation_material": (
                corpus_generation_material.to_dict()
            ),
            "corpus_generation_id": (
                corpus_generation_material.corpus_generation_id
            ),
            "packet_generation_material": generation_material.to_dict(),
            "packet_generation_id": generation_material.packet_generation_id,
            "selected_work_count": acquisition_binding.work_count,
            "files": file_dicts,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1PacketManifest":
        if (
            type(value) is not dict
            or value.get("schema_version")
            != R1_PACKET_MANIFEST_SCHEMA_VERSION
        ):
            raise VNextPacketError(
                "R1 packet manifest is legacy or unsupported"
            )
        raw = _exact_object(
            value,
            {
                "schema_version",
                "status",
                "confirmatory_authorized",
                "acquisition_binding",
                "corpus_generation_material",
                "corpus_generation_id",
                "packet_generation_material",
                "packet_generation_id",
                "selected_work_count",
                "files",
                "self_hash",
            },
            "R1 packet manifest",
        )
        recorded = _sha256(raw["self_hash"], "R1 packet manifest.self_hash")
        payload = {
            key: child for key, child in raw.items() if key != "self_hash"
        }
        if canonical_sha256(payload) != recorded:
            raise VNextPacketError("R1 packet manifest self_hash mismatch")
        if (
            raw["status"] != R1_PACKET_STATUS
            or _exact_bool(
                raw["confirmatory_authorized"],
                "R1 packet manifest.confirmatory_authorized",
            )
            is not False
        ):
            raise VNextPacketError(
                "R1 packet manifest is not exploratory/no-fit"
            )
        binding = R1AcquisitionBinding.from_dict(
            raw["acquisition_binding"]
        )
        corpus_material = R1CorpusGenerationMaterial.from_dict(
            raw["corpus_generation_material"]
        )
        if (
            corpus_material.acquisition_binding_self_hash
            != binding.self_hash
        ):
            raise VNextPacketError(
                "corpus generation material differs from acquisition binding"
            )
        corpus_generation_id = _sha256(
            raw["corpus_generation_id"],
            "R1 packet manifest.corpus_generation_id",
        )
        if corpus_generation_id != corpus_material.corpus_generation_id:
            raise VNextPacketError(
                "R1 packet corpus_generation_id differs from material"
            )
        packet_material = R1PacketGenerationMaterial.from_dict(
            raw["packet_generation_material"]
        )
        if (
            packet_material.corpus_generation_id != corpus_generation_id
            or packet_material.acquisition_binding_self_hash
            != binding.self_hash
        ):
            raise VNextPacketError(
                "packet generation material has inconsistent identities"
            )
        if (
            packet_material.content_policy_spec_sha256
            != binding.content_policy_spec_digest
        ):
            raise VNextPacketError(
                "packet content policy differs from acquisition binding"
            )
        packet_generation_id = _sha256(
            raw["packet_generation_id"],
            "R1 packet manifest.packet_generation_id",
        )
        if packet_generation_id != packet_material.packet_generation_id:
            raise VNextPacketError(
                "R1 packet_generation_id differs from material"
            )
        files = tuple(
            PacketFileEntry.from_dict(item)
            for item in _exact_list(
                raw["files"], "R1 packet manifest.files", nonempty=True
            )
        )
        ordering = tuple(row.relative_path for row in files)
        if ordering != tuple(sorted(set(ordering))):
            raise VNextPacketError(
                "R1 packet files must be sorted by unique relative path"
            )
        if packet_material.file_inventory_sha256 != canonical_sha256(
            [row.to_dict() for row in files]
        ):
            raise VNextPacketError(
                "R1 packet file inventory digest mismatch"
            )
        selected_work_count = _exact_int(
            raw["selected_work_count"],
            "R1 packet manifest.selected_work_count",
            minimum=1,
        )
        if selected_work_count != binding.work_count:
            raise VNextPacketError(
                "R1 packet selected work count mismatch"
            )
        return cls(
            R1_PACKET_MANIFEST_SCHEMA_VERSION,
            R1_PACKET_STATUS,
            False,
            binding,
            corpus_material,
            corpus_generation_id,
            packet_material,
            packet_generation_id,
            selected_work_count,
            files,
            recorded,
        )

    @property
    def acquisition_generation_id(self) -> str:
        return self.acquisition_binding.acquisition_generation_id

    @property
    def file_inventory_sha256(self) -> str:
        return self.packet_generation_material.file_inventory_sha256

    @property
    def content_policy_spec_sha256(self) -> str:
        return self.packet_generation_material.content_policy_spec_sha256

    @property
    def candidate_inventory_sha256(self) -> str:
        return self.packet_generation_material.candidate_inventory_sha256

    @property
    def corpus_manifest_sha256(self) -> str:
        return self.packet_generation_material.corpus_manifest_sha256

    @property
    def content_component_manifest_sha256(self) -> str:
        return (
            self.packet_generation_material.content_component_manifest_sha256
        )

    @property
    def fold_manifest_sha256(self) -> str:
        return self.packet_generation_material.fold_manifest_sha256

    @property
    def primary_model_spec_sha256(self) -> str:
        return self.packet_generation_material.primary_model_spec_sha256

    @property
    def baseline_model_spec_sha256(self) -> str:
        return self.packet_generation_material.baseline_model_spec_sha256

    @property
    def inference_spec_sha256(self) -> str:
        return self.packet_generation_material.inference_spec_sha256

    @property
    def primary_inner_cv_plan_sha256(self) -> str:
        return self.packet_generation_material.primary_inner_cv_plan_sha256

    @property
    def baseline_inner_cv_plan_sha256(self) -> str:
        return self.packet_generation_material.baseline_inner_cv_plan_sha256

    @property
    def model_role_manifest_sha256(self) -> str:
        return self.packet_generation_material.model_role_manifest_sha256

    @property
    def campaign_manifest_sha256(self) -> str:
        return self.packet_generation_material.campaign_manifest_sha256

    @property
    def representation_receipt_sha256(self) -> str:
        return self.packet_generation_material.representation_receipt_sha256

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "status": self.status,
            "confirmatory_authorized": self.confirmatory_authorized,
            "acquisition_binding": self.acquisition_binding.to_dict(),
            "corpus_generation_material": (
                self.corpus_generation_material.to_dict()
            ),
            "corpus_generation_id": self.corpus_generation_id,
            "packet_generation_material": (
                self.packet_generation_material.to_dict()
            ),
            "packet_generation_id": self.packet_generation_id,
            "selected_work_count": self.selected_work_count,
            "files": [row.to_dict() for row in self.files],
        }
        if (
            type(self) is not R1PacketManifest
            or canonical_sha256(payload) != self.self_hash
        ):
            raise VNextPacketError("R1 packet manifest is noncanonical")
        rebuilt = type(self).from_dict({**payload, "self_hash": self.self_hash})
        if rebuilt != self:
            raise VNextPacketError("R1 packet manifest is noncanonical")
        return {**payload, "self_hash": self.self_hash}

    def validate(self) -> "R1PacketManifest":
        if type(self).from_dict(self.to_dict()) != self:
            raise VNextPacketError("R1 packet manifest is noncanonical")
        return self


@dataclasses.dataclass(frozen=True)
class CanonicalRowEntry:
    """One immutable canonical chunk and its literal-source binding."""

    row_id: str
    relative_path: str
    work_id: str
    author_id: str
    ordinal: int
    source_relative_path: str
    source_raw_sha256: str
    canonical_byte_size: int
    canonical_sha256: str
    word_count: int

    @classmethod
    def from_dict(cls, value: object) -> "CanonicalRowEntry":
        raw = _exact_object(
            value,
            {
                "row_id",
                "relative_path",
                "work_id",
                "author_id",
                "ordinal",
                "source_relative_path",
                "source_raw_sha256",
                "canonical_byte_size",
                "canonical_sha256",
                "word_count",
            },
            "canonical row entry",
        )
        relative_path = _relative_path(
            raw["relative_path"], "canonical row.relative_path"
        )
        if (
            not relative_path.startswith(CANONICAL_ROWS_DIRECTORY + "/")
            or not relative_path.endswith(".txt")
        ):
            raise VNextPacketError(
                "canonical row paths must be .txt files below canonical_rows/"
            )
        return cls(
            _exact_str(raw["row_id"], "canonical row.row_id"),
            relative_path,
            _relative_path(raw["work_id"], "canonical row.work_id"),
            _exact_str(raw["author_id"], "canonical row.author_id"),
            _exact_int(raw["ordinal"], "canonical row.ordinal", minimum=0),
            _relative_path(
                raw["source_relative_path"],
                "canonical row.source_relative_path",
            ),
            _sha256(
                raw["source_raw_sha256"],
                "canonical row.source_raw_sha256",
            ),
            _exact_int(
                raw["canonical_byte_size"],
                "canonical row.canonical_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["canonical_sha256"],
                "canonical row.canonical_sha256",
            ),
            _exact_int(
                raw["word_count"], "canonical row.word_count", minimum=1
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "relative_path": self.relative_path,
            "work_id": self.work_id,
            "author_id": self.author_id,
            "ordinal": self.ordinal,
            "source_relative_path": self.source_relative_path,
            "source_raw_sha256": self.source_raw_sha256,
            "canonical_byte_size": self.canonical_byte_size,
            "canonical_sha256": self.canonical_sha256,
            "word_count": self.word_count,
        }


@dataclasses.dataclass(frozen=True)
class CanonicalRepresentationReceipt:
    """Complete path-independent inventory of canonical model rows."""

    schema_version: str
    generation_id: str
    corpus_manifest_sha256: str
    canonicalizer_policy_document_sha256: str
    chunker_policy_document_sha256: str
    canonical_model_row_digest: str
    row_inventory_sha256: str
    n_rows: int
    rows: tuple[CanonicalRowEntry, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        generation_id: str,
        corpus_manifest_sha256: str,
        canonicalizer_policy_document_sha256: str,
        chunker_policy_document_sha256: str,
        rows: Sequence[CanonicalRowEntry],
    ) -> "CanonicalRepresentationReceipt":
        if type(rows) not in (list, tuple):
            raise VNextPacketError("rows must be an exact list or tuple")
        ordered = tuple(rows)
        if any(type(row) is not CanonicalRowEntry for row in ordered):
            raise VNextPacketError(
                "rows must contain exact CanonicalRowEntry records"
            )
        row_dicts = [row.to_dict() for row in ordered]
        row_digest = canonical_sha256(row_dicts)
        payload = {
            "schema_version": CANONICAL_REPRESENTATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "canonicalizer_policy_document_sha256": (
                canonicalizer_policy_document_sha256
            ),
            "chunker_policy_document_sha256": (
                chunker_policy_document_sha256
            ),
            "canonical_model_row_digest": row_digest,
            "row_inventory_sha256": row_digest,
            "n_rows": len(ordered),
            "rows": row_dicts,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "CanonicalRepresentationReceipt":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "generation_id",
                "corpus_manifest_sha256",
                "canonicalizer_policy_document_sha256",
                "chunker_policy_document_sha256",
                "canonical_model_row_digest",
                "row_inventory_sha256",
                "n_rows",
                "rows",
                "self_hash",
            },
            "canonical representation receipt",
        )
        recorded = _sha256(
            raw["self_hash"], "canonical representation receipt.self_hash"
        )
        payload = {key: child for key, child in raw.items() if key != "self_hash"}
        if canonical_sha256(payload) != recorded:
            raise VNextPacketError(
                "canonical representation receipt self_hash mismatch"
            )
        if (
            payload["schema_version"]
            != CANONICAL_REPRESENTATION_SCHEMA_VERSION
        ):
            raise VNextPacketError(
                "canonical representation receipt is legacy or unsupported"
            )
        rows = tuple(
            CanonicalRowEntry.from_dict(item)
            for item in _exact_list(
                payload["rows"],
                "canonical representation receipt.rows",
                nonempty=True,
            )
        )
        row_dicts = [row.to_dict() for row in rows]
        row_digest = canonical_sha256(row_dicts)
        n_rows = _exact_int(
            payload["n_rows"],
            "canonical representation receipt.n_rows",
            minimum=1,
        )
        if n_rows != len(rows):
            raise VNextPacketError(
                "canonical representation receipt row count mismatch"
            )
        canonical_digest = _sha256(
            payload["canonical_model_row_digest"],
            "canonical representation receipt.canonical_model_row_digest",
        )
        inventory_digest = _sha256(
            payload["row_inventory_sha256"],
            "canonical representation receipt.row_inventory_sha256",
        )
        if canonical_digest != row_digest or inventory_digest != row_digest:
            raise VNextPacketError(
                "canonical representation row inventory digest mismatch"
            )
        receipt = cls(
            CANONICAL_REPRESENTATION_SCHEMA_VERSION,
            _exact_str(
                payload["generation_id"],
                "canonical representation receipt.generation_id",
            ),
            _sha256(
                payload["corpus_manifest_sha256"],
                "canonical representation receipt.corpus_manifest_sha256",
            ),
            _sha256(
                payload["canonicalizer_policy_document_sha256"],
                "canonical representation receipt."
                "canonicalizer_policy_document_sha256",
            ),
            _sha256(
                payload["chunker_policy_document_sha256"],
                "canonical representation receipt."
                "chunker_policy_document_sha256",
            ),
            canonical_digest,
            inventory_digest,
            n_rows,
            rows,
            recorded,
        )
        return receipt.validate()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "canonicalizer_policy_document_sha256": (
                self.canonicalizer_policy_document_sha256
            ),
            "chunker_policy_document_sha256": (
                self.chunker_policy_document_sha256
            ),
            "canonical_model_row_digest": self.canonical_model_row_digest,
            "row_inventory_sha256": self.row_inventory_sha256,
            "n_rows": self.n_rows,
            "rows": [row.to_dict() for row in self.rows],
        }

    def validate(
        self,
        *,
        corpus_manifest: CorpusVNextManifest | None = None,
    ) -> "CanonicalRepresentationReceipt":
        if type(self) is not CanonicalRepresentationReceipt:
            raise VNextPacketError(
                "receipt must be exactly CanonicalRepresentationReceipt"
            )
        rows = self.rows
        ordering = tuple(
            (row.work_id, row.ordinal, row.relative_path) for row in rows
        )
        if ordering != tuple(sorted(ordering)):
            raise VNextPacketError(
                "canonical representation rows must be canonically sorted"
            )
        if (
            len({row.row_id for row in rows}) != len(rows)
            or len({row.relative_path for row in rows}) != len(rows)
        ):
            raise VNextPacketError(
                "canonical representation has duplicate row ids or paths"
            )
        rows_by_work: dict[str, list[CanonicalRowEntry]] = defaultdict(list)
        for row in rows:
            rows_by_work[row.work_id].append(row)
        for work_id, work_rows in rows_by_work.items():
            if tuple(row.ordinal for row in work_rows) != tuple(
                range(len(work_rows))
            ):
                raise VNextPacketError(
                    f"canonical row ordinals are not contiguous for {work_id!r}"
                )
        if canonical_sha256(self._payload()) != self.self_hash:
            raise VNextPacketError(
                "canonical representation receipt self_hash mismatch"
            )
        if corpus_manifest is not None:
            if type(corpus_manifest) is not CorpusVNextManifest:
                raise VNextPacketError(
                    "corpus_manifest must be exactly CorpusVNextManifest"
                )
            corpus_manifest.validate()
            works = {work.work_id: work for work in corpus_manifest.works}
            inventory = {
                row.relative_path: row
                for row in corpus_manifest.raw_inventory
            }
            if self.generation_id != corpus_manifest.generation_id:
                raise VNextPacketError(
                    "canonical representation generation mismatch"
                )
            if self.corpus_manifest_sha256 != corpus_manifest.self_hash:
                raise VNextPacketError(
                    "canonical representation corpus digest mismatch"
                )
            if (
                self.canonical_model_row_digest
                != corpus_manifest.canonical_model_row_digest
            ):
                raise VNextPacketError(
                    "canonical model-row digest differs from corpus manifest"
                )
            if set(rows_by_work) != set(works):
                raise VNextPacketError(
                    "canonical rows do not cover every included work exactly"
                )
            for row in rows:
                work = works[row.work_id]
                if row.author_id != work.author_id:
                    raise VNextPacketError(
                        f"canonical row author mismatch for {row.row_id!r}"
                    )
                if row.source_relative_path not in work.raw_paths:
                    raise VNextPacketError(
                        f"canonical row source is outside work {row.work_id!r}"
                    )
                source = inventory.get(row.source_relative_path)
                if (
                    source is None
                    or source.sha256 != row.source_raw_sha256
                ):
                    raise VNextPacketError(
                        f"canonical row source SHA mismatch for {row.row_id!r}"
                    )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}


def _inventory_canonical_row_files(root: Path) -> tuple[str, ...]:
    rows_root = root / CANONICAL_ROWS_DIRECTORY
    try:
        root_stat = root.lstat()
        rows_stat = rows_root.lstat()
    except OSError as exc:
        raise VNextPacketError(
            "packet/canonical_rows directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(rows_stat.st_mode)
        or not stat.S_ISDIR(rows_stat.st_mode)
    ):
        raise VNextPacketError(
            "packet root and canonical_rows must be real directories"
        )
    found: list[str] = []
    stack = [rows_root]
    while stack:
        directory = stack.pop()
        for child in sorted(os.scandir(directory), key=lambda item: item.name):
            child_stat = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(child_stat.st_mode):
                raise VNextPacketError(
                    f"symlink rejected in canonical rows: {child.path}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                stack.append(Path(child.path))
            elif stat.S_ISREG(child_stat.st_mode):
                found.append(Path(child.path).relative_to(root).as_posix())
            else:
                raise VNextPacketError(
                    f"special file rejected in canonical rows: {child.path}"
                )
    return tuple(sorted(found))


def load_canonical_representation_rows(
    packet_root: str | os.PathLike[str],
    receipt: CanonicalRepresentationReceipt,
    corpus_manifest: CorpusVNextManifest,
) -> tuple[VNextTextRow, ...]:
    """Verify every canonical row byte and return exact model inputs."""

    if type(receipt) is not CanonicalRepresentationReceipt:
        raise VNextPacketError(
            "receipt must be exactly CanonicalRepresentationReceipt"
        )
    receipt.validate(corpus_manifest=corpus_manifest)
    root = Path(packet_root)
    expected_paths = tuple(sorted(row.relative_path for row in receipt.rows))
    observed_paths = _inventory_canonical_row_files(root)
    if observed_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(observed_paths))
        extra = sorted(set(observed_paths) - set(expected_paths))
        raise VNextPacketError(
            f"canonical row path mismatch; missing={missing}, extra={extra}"
        )
    loaded: list[VNextTextRow] = []
    for row in receipt.rows:
        path = root.joinpath(*PurePosixPath(row.relative_path).parts)
        if path.is_symlink() or not path.is_file():
            raise VNextPacketError(
                f"canonical row is missing or symlinked: {row.relative_path}"
            )
        payload = path.read_bytes()
        if (
            len(payload) != row.canonical_byte_size
            or hashlib.sha256(payload).hexdigest() != row.canonical_sha256
        ):
            raise VNextPacketError(
                f"canonical row bytes drifted: {row.relative_path}"
            )
        if payload.startswith(b"\xef\xbb\xbf"):
            raise VNextPacketError(
                f"canonical row contains a UTF-8 BOM: {row.relative_path}"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VNextPacketError(
                f"canonical row is not strict UTF-8: {row.relative_path}"
            ) from exc
        if not text or text != text.strip():
            raise VNextPacketError(
                f"canonical row is empty or noncanonical: {row.relative_path}"
            )
        if len(text.split()) != row.word_count:
            raise VNextPacketError(
                f"canonical row word count drifted: {row.relative_path}"
            )
        loaded.append(
            VNextTextRow(
                row_id=row.row_id,
                relative_path=row.relative_path,
                work_id=row.work_id,
                author_id=row.author_id,
                text=text,
                raw_sha256=row.source_raw_sha256,
            )
        )
    return tuple(loaded)


def loads_canonical_representation_receipt(
    text: str,
) -> CanonicalRepresentationReceipt:
    return CanonicalRepresentationReceipt.from_dict(
        _strict_raw(text, "canonical representation receipt")
    )


def load_canonical_representation_receipt(
    path: str | os.PathLike[str],
) -> CanonicalRepresentationReceipt:
    return CanonicalRepresentationReceipt.from_dict(
        _strict_file(path, "canonical representation receipt")
    )


__all__ = [
    "CANONICAL_REPRESENTATION_SCHEMA_VERSION",
    "CANONICAL_ROWS_DIRECTORY",
    "R1_ACQUISITION_BINDING_SCHEMA_VERSION",
    "R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION",
    "R1_PACKET_GENERATION_MATERIAL_SCHEMA_VERSION",
    "R1_PACKET_MANIFEST_SCHEMA_VERSION",
    "R1_PACKET_STATUS",
    "CanonicalRepresentationReceipt",
    "CanonicalRowEntry",
    "PacketFileEntry",
    "R1AcquisitionBinding",
    "R1CorpusGenerationMaterial",
    "R1PacketGenerationMaterial",
    "R1PacketManifest",
    "VNextTextRow",
    "VNextPacketError",
    "load_canonical_representation_receipt",
    "load_canonical_representation_rows",
    "loads_canonical_representation_receipt",
]
