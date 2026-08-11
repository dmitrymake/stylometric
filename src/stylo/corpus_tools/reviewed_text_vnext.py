"""Hash-bound publication of already reviewed final corpus texts.

This provider deliberately has no discovery, fetching, extraction, or repair
surface.  Editorial work happens upstream.  Its only input is a strict,
self-hashed campaign spec and a local content-addressed cache containing the
exact reviewed UTF-8 text artifacts named by that spec.

Publication is create-if-absent.  A complete existing generation is validated
byte-for-byte before the cache is inspected.  A generation contains only the
canonical campaign spec, one self-hashed receipt, and the reviewed raw texts;
builder and provenance artifacts remain compact hash-and-size references.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import os
import pathlib
import re
import shutil
import stat
import tempfile
from pathlib import PurePosixPath
from typing import Any, Sequence

from .._strict_fields import ExactFieldReader
from ..jsonio import (
    StrictJSONError,
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
    loads_strict,
)


REVIEWED_TEXT_CAMPAIGN_SPEC_SCHEMA_VERSION = (
    "stylo.reviewed-text.campaign-spec.v1"
)
REVIEWED_TEXT_CAMPAIGN_RECEIPT_SCHEMA_VERSION = (
    "stylo.reviewed-text.campaign-receipt.v1"
)
REVIEWED_TEXT_CAMPAIGN_KIND = "hash_bound_reviewed_final_texts"
REVIEWED_TEXT_ARTIFACT_KEY_POLICY_VERSION = (
    "stylo.reviewed-text.sha256-artifact-key.v1"
)
REVIEWED_TEXT_IDENTITY_POLICY_VERSION = (
    "stylo.reviewed-text.canonical-utf8-final-lf.v1"
)

UPSTREAM_INVENTORY_FILE_SHA256 = (
    "abcda1abbe6c2ba495fd1d27093d81fc7ad3880c2d73f770adb443c524086cff"
)
UPSTREAM_QUALITY_AUDIT_FILE_SHA256 = (
    "63e4da1df7186133aa05b18806ba29390ff0e9b5e3ef996021426942fb705ece"
)
UPSTREAM_QUALITY_AUDIT_SELF_HASH = (
    "3be9ed08062ba68a2411e862a282a87e84b33ed900fb57585af61dc5daa1b5aa"
)

REVIEWED_TEXT_CAMPAIGN_SPEC_NAME = "campaign-spec.json"
REVIEWED_TEXT_CAMPAIGN_RECEIPT_NAME = "campaign-receipt.json"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)


class ReviewedTextMaterializationError(ValueError):
    """A reviewed-text spec, cache entry, or generation is unsafe."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_STRICT = ExactFieldReader(ReviewedTextMaterializationError)
_exact_object = _STRICT.object
_exact_list = _STRICT.array
_exact_str = _STRICT.string
_exact_int = _STRICT.integer
_sha256 = _STRICT.sha256


def _work_id(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise ReviewedTextMaterializationError(
            f"{label} must use POSIX separators"
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReviewedTextMaterializationError(
            f"{label} must be a canonical author/work identifier"
        )
    return text


def _logical_name(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if (
        text != text.strip()
        or "/" in text
        or "\\" in text
        or "\r" in text
        or "\n" in text
    ):
        raise ReviewedTextMaterializationError(
            f"{label} must be a path-free, trimmed single-line logical name"
        )
    return text


def _self_hashed_payload(
    raw: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    if canonical_hash(payload) != recorded:
        raise ReviewedTextMaterializationError(f"{label} self_hash mismatch")
    return payload


def _canonical_json_text(value: object) -> str:
    return dumps_strict(value, indent=2, sort_keys=True) + "\n"


@dataclasses.dataclass(frozen=True)
class _TextIdentity:
    byte_size: int
    sha256: str
    word_count: int
    line_count: int
    first_nonblank_line_sha256: str
    last_nonblank_line_sha256: str


def _text_identity(payload: bytes, *, label: str) -> _TextIdentity:
    if type(payload) is not bytes or not payload:
        raise ReviewedTextMaterializationError(
            f"{label} must be exact non-empty bytes"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReviewedTextMaterializationError(
            f"{label} is not strict UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise ReviewedTextMaterializationError(f"{label} has a UTF-8 BOM")
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise ReviewedTextMaterializationError(
            f"{label} must have exactly one final LF"
        )
    if "\r" in text:
        raise ReviewedTextMaterializationError(
            f"{label} contains a rejected CR character"
        )
    if "\t" in text:
        raise ReviewedTextMaterializationError(
            f"{label} contains a rejected tab character"
        )
    body = text[:-1]
    if not body or body != body.strip():
        raise ReviewedTextMaterializationError(
            f"{label} body must have no edge whitespace"
        )
    lines = body.split("\n")
    if any(line != line.rstrip() for line in lines):
        raise ReviewedTextMaterializationError(
            f"{label} contains trailing line whitespace"
        )
    if any(not left and not right for left, right in zip(lines, lines[1:])):
        raise ReviewedTextMaterializationError(
            f"{label} contains repeated blank lines"
        )
    nonblank = [line for line in lines if line]
    if not nonblank:
        raise ReviewedTextMaterializationError(
            f"{label} has no nonblank lines"
        )
    return _TextIdentity(
        len(payload),
        _sha256_bytes(payload),
        len(_WORD_RE.findall(body)),
        len(lines),
        _sha256_bytes(nonblank[0].encode("utf-8")),
        _sha256_bytes(nonblank[-1].encode("utf-8")),
    )


@dataclasses.dataclass(frozen=True)
class ReviewedTextArtifactRef:
    """Compact identity for a builder or provenance artifact."""

    logical_name: str
    sha256: str
    byte_size: int

    @classmethod
    def build(
        cls,
        *,
        logical_name: str,
        payload: bytes,
    ) -> "ReviewedTextArtifactRef":
        if type(payload) is not bytes or not payload:
            raise ReviewedTextMaterializationError(
                "referenced artifact payload must be exact non-empty bytes"
            )
        return cls.from_dict(
            {
                "logical_name": logical_name,
                "sha256": _sha256_bytes(payload),
                "byte_size": len(payload),
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedTextArtifactRef":
        raw = _exact_object(
            value,
            {"logical_name", "sha256", "byte_size"},
            "reviewed artifact ref",
        )
        return cls(
            _logical_name(
                raw["logical_name"],
                "reviewed artifact ref.logical_name",
            ),
            _sha256(raw["sha256"], "reviewed artifact ref.sha256"),
            _exact_int(
                raw["byte_size"],
                "reviewed artifact ref.byte_size",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def validate(self) -> "ReviewedTextArtifactRef":
        if ReviewedTextArtifactRef.from_dict(self.to_dict()) != self:
            raise ReviewedTextMaterializationError(
                "reviewed artifact ref is noncanonical"
            )
        return self


def _artifact_refs(
    value: object,
    label: str,
) -> tuple[ReviewedTextArtifactRef, ...]:
    refs = tuple(
        ReviewedTextArtifactRef.from_dict(item)
        for item in _exact_list(value, label, nonempty=True)
    )
    names = tuple(row.logical_name for row in refs)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ReviewedTextMaterializationError(
            f"{label} must be sorted by unique logical_name"
        )
    return refs


def _checked_artifact_refs(
    value: Sequence[ReviewedTextArtifactRef],
    label: str,
) -> tuple[ReviewedTextArtifactRef, ...]:
    if type(value) not in {list, tuple} or not value:
        raise ReviewedTextMaterializationError(
            f"{label} must be an exact non-empty list or tuple"
        )
    refs: list[ReviewedTextArtifactRef] = []
    for index, item in enumerate(value):
        if type(item) is not ReviewedTextArtifactRef:
            raise ReviewedTextMaterializationError(
                f"{label}[{index}] must be exactly ReviewedTextArtifactRef"
            )
        refs.append(item.validate())
    refs.sort(key=lambda row: row.logical_name)
    if len({row.logical_name for row in refs}) != len(refs):
        raise ReviewedTextMaterializationError(
            f"{label} contains duplicate logical names"
        )
    return tuple(refs)


@dataclasses.dataclass(frozen=True)
class ReviewedTextWorkSpec:
    """One exact reviewed text and its compact reproducibility references."""

    work_id: str
    artifact_key: str
    byte_size: int
    sha256: str
    word_count: int
    line_count: int
    first_nonblank_line_sha256: str
    last_nonblank_line_sha256: str
    builder_artifacts: tuple[ReviewedTextArtifactRef, ...]
    provenance_artifacts: tuple[ReviewedTextArtifactRef, ...]
    source_part_count: int
    reviewed_part_count: int

    @classmethod
    def build(
        cls,
        *,
        work_id: str,
        text_payload: bytes,
        builder_artifacts: Sequence[ReviewedTextArtifactRef],
        provenance_artifacts: Sequence[ReviewedTextArtifactRef],
        source_part_count: int,
        reviewed_part_count: int,
    ) -> "ReviewedTextWorkSpec":
        work = _work_id(work_id, "reviewed work spec.work_id")
        identity = _text_identity(
            text_payload,
            label=f"reviewed text {work!r}",
        )
        builders = _checked_artifact_refs(
            builder_artifacts,
            "reviewed work spec.builder_artifacts",
        )
        provenance = _checked_artifact_refs(
            provenance_artifacts,
            "reviewed work spec.provenance_artifacts",
        )
        return cls.from_dict(
            {
                "work_id": work,
                "artifact_key": f"sha256/{identity.sha256}.txt",
                "byte_size": identity.byte_size,
                "sha256": identity.sha256,
                "word_count": identity.word_count,
                "line_count": identity.line_count,
                "first_nonblank_line_sha256": (
                    identity.first_nonblank_line_sha256
                ),
                "last_nonblank_line_sha256": (
                    identity.last_nonblank_line_sha256
                ),
                "builder_artifacts": [row.to_dict() for row in builders],
                "provenance_artifacts": [
                    row.to_dict() for row in provenance
                ],
                "source_part_count": source_part_count,
                "reviewed_part_count": reviewed_part_count,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedTextWorkSpec":
        raw = _exact_object(
            value,
            {
                "work_id",
                "artifact_key",
                "byte_size",
                "sha256",
                "word_count",
                "line_count",
                "first_nonblank_line_sha256",
                "last_nonblank_line_sha256",
                "builder_artifacts",
                "provenance_artifacts",
                "source_part_count",
                "reviewed_part_count",
            },
            "reviewed work spec",
        )
        work = _work_id(raw["work_id"], "reviewed work spec.work_id")
        digest = _sha256(raw["sha256"], "reviewed work spec.sha256")
        artifact_key = _exact_str(
            raw["artifact_key"],
            "reviewed work spec.artifact_key",
        )
        if artifact_key != f"sha256/{digest}.txt":
            raise ReviewedTextMaterializationError(
                "reviewed work artifact_key must be "
                "exactly sha256/<sha256>.txt"
            )
        return cls(
            work,
            artifact_key,
            _exact_int(
                raw["byte_size"],
                "reviewed work spec.byte_size",
                minimum=1,
            ),
            digest,
            _exact_int(
                raw["word_count"],
                "reviewed work spec.word_count",
                minimum=1,
            ),
            _exact_int(
                raw["line_count"],
                "reviewed work spec.line_count",
                minimum=1,
            ),
            _sha256(
                raw["first_nonblank_line_sha256"],
                "reviewed work spec.first_nonblank_line_sha256",
            ),
            _sha256(
                raw["last_nonblank_line_sha256"],
                "reviewed work spec.last_nonblank_line_sha256",
            ),
            _artifact_refs(
                raw["builder_artifacts"],
                "reviewed work spec.builder_artifacts",
            ),
            _artifact_refs(
                raw["provenance_artifacts"],
                "reviewed work spec.provenance_artifacts",
            ),
            _exact_int(
                raw["source_part_count"],
                "reviewed work spec.source_part_count",
                minimum=1,
            ),
            _exact_int(
                raw["reviewed_part_count"],
                "reviewed work spec.reviewed_part_count",
                minimum=1,
            ),
        )

    @property
    def output_relative_path(self) -> str:
        return f"raw/{self.work_id}.txt"

    def to_dict(self) -> dict[str, object]:
        return {
            "work_id": self.work_id,
            "artifact_key": self.artifact_key,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "first_nonblank_line_sha256": self.first_nonblank_line_sha256,
            "last_nonblank_line_sha256": self.last_nonblank_line_sha256,
            "builder_artifacts": [
                row.to_dict() for row in self.builder_artifacts
            ],
            "provenance_artifacts": [
                row.to_dict() for row in self.provenance_artifacts
            ],
            "source_part_count": self.source_part_count,
            "reviewed_part_count": self.reviewed_part_count,
        }

    def validate(self) -> "ReviewedTextWorkSpec":
        if ReviewedTextWorkSpec.from_dict(self.to_dict()) != self:
            raise ReviewedTextMaterializationError(
                "reviewed work spec is noncanonical"
            )
        return self


def _campaign_payload(
    *,
    work_ids: Sequence[str],
    works: Sequence[ReviewedTextWorkSpec],
) -> dict[str, object]:
    return {
        "schema_version": REVIEWED_TEXT_CAMPAIGN_SPEC_SCHEMA_VERSION,
        "campaign_kind": REVIEWED_TEXT_CAMPAIGN_KIND,
        "artifact_key_policy_version": (
            REVIEWED_TEXT_ARTIFACT_KEY_POLICY_VERSION
        ),
        "text_identity_policy_version": (
            REVIEWED_TEXT_IDENTITY_POLICY_VERSION
        ),
        "upstream_inventory_file_sha256": (
            UPSTREAM_INVENTORY_FILE_SHA256
        ),
        "upstream_quality_audit_file_sha256": (
            UPSTREAM_QUALITY_AUDIT_FILE_SHA256
        ),
        "upstream_quality_audit_self_hash": (
            UPSTREAM_QUALITY_AUDIT_SELF_HASH
        ),
        "work_ids": list(work_ids),
        "works": [row.to_dict() for row in works],
    }


@dataclasses.dataclass(frozen=True)
class ReviewedTextCampaignSpec:
    """Strict self-hashed inventory of reviewed final texts."""

    work_ids: tuple[str, ...]
    works: tuple[ReviewedTextWorkSpec, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        works: Sequence[ReviewedTextWorkSpec],
    ) -> "ReviewedTextCampaignSpec":
        if type(works) not in {list, tuple} or not works:
            raise ReviewedTextMaterializationError(
                "reviewed campaign works must be an exact non-empty list or "
                "tuple"
            )
        checked: list[ReviewedTextWorkSpec] = []
        for index, work in enumerate(works):
            if type(work) is not ReviewedTextWorkSpec:
                raise ReviewedTextMaterializationError(
                    f"reviewed campaign works[{index}] must be exactly "
                    "ReviewedTextWorkSpec"
                )
            checked.append(work.validate())
        checked.sort(key=lambda row: row.work_id)
        work_ids = tuple(row.work_id for row in checked)
        if len(work_ids) != len(set(work_ids)):
            raise ReviewedTextMaterializationError(
                "reviewed campaign contains duplicate work ids"
            )
        payload = _campaign_payload(work_ids=work_ids, works=checked)
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedTextCampaignSpec":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "campaign_kind",
                "artifact_key_policy_version",
                "text_identity_policy_version",
                "upstream_inventory_file_sha256",
                "upstream_quality_audit_file_sha256",
                "upstream_quality_audit_self_hash",
                "work_ids",
                "works",
                "self_hash",
            },
            "reviewed campaign spec",
        )
        _self_hashed_payload(raw, "reviewed campaign spec")
        expected_scalars = {
            "schema_version": REVIEWED_TEXT_CAMPAIGN_SPEC_SCHEMA_VERSION,
            "campaign_kind": REVIEWED_TEXT_CAMPAIGN_KIND,
            "artifact_key_policy_version": (
                REVIEWED_TEXT_ARTIFACT_KEY_POLICY_VERSION
            ),
            "text_identity_policy_version": (
                REVIEWED_TEXT_IDENTITY_POLICY_VERSION
            ),
            "upstream_inventory_file_sha256": (
                UPSTREAM_INVENTORY_FILE_SHA256
            ),
            "upstream_quality_audit_file_sha256": (
                UPSTREAM_QUALITY_AUDIT_FILE_SHA256
            ),
            "upstream_quality_audit_self_hash": (
                UPSTREAM_QUALITY_AUDIT_SELF_HASH
            ),
        }
        for key, expected in expected_scalars.items():
            if raw[key] != expected:
                raise ReviewedTextMaterializationError(
                    f"reviewed campaign spec {key} must be {expected!r}"
                )
        work_ids = tuple(
            _work_id(
                item,
                f"reviewed campaign spec.work_ids[{index}]",
            )
            for index, item in enumerate(
                _exact_list(
                    raw["work_ids"],
                    "reviewed campaign spec.work_ids",
                    nonempty=True,
                )
            )
        )
        if work_ids != tuple(sorted(work_ids)):
            raise ReviewedTextMaterializationError(
                "reviewed campaign work_ids must be sorted exactly"
            )
        if len(work_ids) != len(set(work_ids)):
            raise ReviewedTextMaterializationError(
                "reviewed campaign work_ids must be unique"
            )
        works = tuple(
            ReviewedTextWorkSpec.from_dict(item)
            for item in _exact_list(
                raw["works"],
                "reviewed campaign spec.works",
                nonempty=True,
            )
        )
        if tuple(row.work_id for row in works) != work_ids:
            raise ReviewedTextMaterializationError(
                "reviewed campaign works must exactly match sorted work_ids"
            )
        payload = _campaign_payload(work_ids=work_ids, works=works)
        recorded = _sha256(
            raw["self_hash"],
            "reviewed campaign spec.self_hash",
        )
        if canonical_hash(payload) != recorded:
            raise ReviewedTextMaterializationError(
                "reviewed campaign spec payload is noncanonical"
            )
        return cls(work_ids, works, recorded)

    def to_dict(self) -> dict[str, object]:
        return {
            **_campaign_payload(work_ids=self.work_ids, works=self.works),
            "self_hash": self.self_hash,
        }

    def validate(self) -> "ReviewedTextCampaignSpec":
        if ReviewedTextCampaignSpec.from_dict(self.to_dict()) != self:
            raise ReviewedTextMaterializationError(
                "reviewed campaign spec is noncanonical"
            )
        return self


def loads_reviewed_text_campaign_spec(
    text: str,
) -> ReviewedTextCampaignSpec:
    try:
        return ReviewedTextCampaignSpec.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise ReviewedTextMaterializationError(
            f"reviewed campaign spec: {exc}"
        ) from exc


def load_reviewed_text_campaign_spec(
    path: str | os.PathLike[str],
) -> ReviewedTextCampaignSpec:
    try:
        return ReviewedTextCampaignSpec.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise ReviewedTextMaterializationError(
            f"reviewed campaign spec: {exc}"
        ) from exc


@dataclasses.dataclass(frozen=True)
class ReviewedTextWorkReceipt:
    work_id: str
    artifact_key: str
    output_relative_path: str
    byte_size: int
    sha256: str
    word_count: int
    line_count: int
    first_nonblank_line_sha256: str
    last_nonblank_line_sha256: str
    source_part_count: int
    reviewed_part_count: int

    @classmethod
    def build(
        cls,
        spec: ReviewedTextWorkSpec,
    ) -> "ReviewedTextWorkReceipt":
        if type(spec) is not ReviewedTextWorkSpec:
            raise ReviewedTextMaterializationError(
                "work receipt requires exactly ReviewedTextWorkSpec"
            )
        spec.validate()
        return cls.from_dict(
            {
                "work_id": spec.work_id,
                "artifact_key": spec.artifact_key,
                "output_relative_path": spec.output_relative_path,
                "byte_size": spec.byte_size,
                "sha256": spec.sha256,
                "word_count": spec.word_count,
                "line_count": spec.line_count,
                "first_nonblank_line_sha256": (
                    spec.first_nonblank_line_sha256
                ),
                "last_nonblank_line_sha256": (
                    spec.last_nonblank_line_sha256
                ),
                "source_part_count": spec.source_part_count,
                "reviewed_part_count": spec.reviewed_part_count,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedTextWorkReceipt":
        raw = _exact_object(
            value,
            {
                "work_id",
                "artifact_key",
                "output_relative_path",
                "byte_size",
                "sha256",
                "word_count",
                "line_count",
                "first_nonblank_line_sha256",
                "last_nonblank_line_sha256",
                "source_part_count",
                "reviewed_part_count",
            },
            "reviewed work receipt",
        )
        work = _work_id(raw["work_id"], "reviewed work receipt.work_id")
        digest = _sha256(raw["sha256"], "reviewed work receipt.sha256")
        artifact_key = _exact_str(
            raw["artifact_key"],
            "reviewed work receipt.artifact_key",
        )
        if artifact_key != f"sha256/{digest}.txt":
            raise ReviewedTextMaterializationError(
                "reviewed work receipt artifact key is noncanonical"
            )
        output = _exact_str(
            raw["output_relative_path"],
            "reviewed work receipt.output_relative_path",
        )
        if output != f"raw/{work}.txt":
            raise ReviewedTextMaterializationError(
                "reviewed work receipt output path is noncanonical"
            )
        return cls(
            work,
            artifact_key,
            output,
            _exact_int(
                raw["byte_size"],
                "reviewed work receipt.byte_size",
                minimum=1,
            ),
            digest,
            _exact_int(
                raw["word_count"],
                "reviewed work receipt.word_count",
                minimum=1,
            ),
            _exact_int(
                raw["line_count"],
                "reviewed work receipt.line_count",
                minimum=1,
            ),
            _sha256(
                raw["first_nonblank_line_sha256"],
                "reviewed work receipt.first_nonblank_line_sha256",
            ),
            _sha256(
                raw["last_nonblank_line_sha256"],
                "reviewed work receipt.last_nonblank_line_sha256",
            ),
            _exact_int(
                raw["source_part_count"],
                "reviewed work receipt.source_part_count",
                minimum=1,
            ),
            _exact_int(
                raw["reviewed_part_count"],
                "reviewed work receipt.reviewed_part_count",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReviewedTextCampaignReceipt:
    campaign_spec_sha256: str
    work_ids: tuple[str, ...]
    works: tuple[ReviewedTextWorkReceipt, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        spec: ReviewedTextCampaignSpec,
    ) -> "ReviewedTextCampaignReceipt":
        if type(spec) is not ReviewedTextCampaignSpec:
            raise ReviewedTextMaterializationError(
                "campaign receipt requires exactly ReviewedTextCampaignSpec"
            )
        spec.validate()
        payload: dict[str, object] = {
            "schema_version": (
                REVIEWED_TEXT_CAMPAIGN_RECEIPT_SCHEMA_VERSION
            ),
            "campaign_kind": REVIEWED_TEXT_CAMPAIGN_KIND,
            "campaign_spec_sha256": spec.self_hash,
            "upstream_inventory_file_sha256": (
                UPSTREAM_INVENTORY_FILE_SHA256
            ),
            "upstream_quality_audit_file_sha256": (
                UPSTREAM_QUALITY_AUDIT_FILE_SHA256
            ),
            "upstream_quality_audit_self_hash": (
                UPSTREAM_QUALITY_AUDIT_SELF_HASH
            ),
            "work_ids": list(spec.work_ids),
            "works": [
                ReviewedTextWorkReceipt.build(row).to_dict()
                for row in spec.works
            ],
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedTextCampaignReceipt":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "campaign_kind",
                "campaign_spec_sha256",
                "upstream_inventory_file_sha256",
                "upstream_quality_audit_file_sha256",
                "upstream_quality_audit_self_hash",
                "work_ids",
                "works",
                "self_hash",
            },
            "reviewed campaign receipt",
        )
        _self_hashed_payload(raw, "reviewed campaign receipt")
        expected_scalars = {
            "schema_version": (
                REVIEWED_TEXT_CAMPAIGN_RECEIPT_SCHEMA_VERSION
            ),
            "campaign_kind": REVIEWED_TEXT_CAMPAIGN_KIND,
            "upstream_inventory_file_sha256": (
                UPSTREAM_INVENTORY_FILE_SHA256
            ),
            "upstream_quality_audit_file_sha256": (
                UPSTREAM_QUALITY_AUDIT_FILE_SHA256
            ),
            "upstream_quality_audit_self_hash": (
                UPSTREAM_QUALITY_AUDIT_SELF_HASH
            ),
        }
        for key, expected in expected_scalars.items():
            if raw[key] != expected:
                raise ReviewedTextMaterializationError(
                    f"reviewed campaign receipt {key} must be {expected!r}"
                )
        work_ids = tuple(
            _work_id(
                item,
                f"reviewed campaign receipt.work_ids[{index}]",
            )
            for index, item in enumerate(
                _exact_list(
                    raw["work_ids"],
                    "reviewed campaign receipt.work_ids",
                    nonempty=True,
                )
            )
        )
        if (
            work_ids != tuple(sorted(work_ids))
            or len(work_ids) != len(set(work_ids))
        ):
            raise ReviewedTextMaterializationError(
                "reviewed campaign receipt work_ids must be sorted and unique"
            )
        works = tuple(
            ReviewedTextWorkReceipt.from_dict(item)
            for item in _exact_list(
                raw["works"],
                "reviewed campaign receipt.works",
                nonempty=True,
            )
        )
        if tuple(row.work_id for row in works) != work_ids:
            raise ReviewedTextMaterializationError(
                "reviewed campaign receipt works do not match work_ids"
            )
        return cls(
            _sha256(
                raw["campaign_spec_sha256"],
                "reviewed campaign receipt.campaign_spec_sha256",
            ),
            work_ids,
            works,
            _sha256(
                raw["self_hash"],
                "reviewed campaign receipt.self_hash",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": (
                REVIEWED_TEXT_CAMPAIGN_RECEIPT_SCHEMA_VERSION
            ),
            "campaign_kind": REVIEWED_TEXT_CAMPAIGN_KIND,
            "campaign_spec_sha256": self.campaign_spec_sha256,
            "upstream_inventory_file_sha256": (
                UPSTREAM_INVENTORY_FILE_SHA256
            ),
            "upstream_quality_audit_file_sha256": (
                UPSTREAM_QUALITY_AUDIT_FILE_SHA256
            ),
            "upstream_quality_audit_self_hash": (
                UPSTREAM_QUALITY_AUDIT_SELF_HASH
            ),
            "work_ids": list(self.work_ids),
            "works": [row.to_dict() for row in self.works],
            "self_hash": self.self_hash,
        }

    def validate_for(
        self,
        spec: ReviewedTextCampaignSpec,
        payloads: Sequence[bytes],
    ) -> "ReviewedTextCampaignReceipt":
        if type(spec) is not ReviewedTextCampaignSpec:
            raise ReviewedTextMaterializationError(
                "receipt validation requires exactly "
                "ReviewedTextCampaignSpec"
            )
        spec.validate()
        if type(payloads) not in {list, tuple} or len(payloads) != len(
            spec.works
        ):
            raise ReviewedTextMaterializationError(
                "receipt validation payload count differs from campaign"
            )
        for work, payload in zip(spec.works, payloads, strict=True):
            _validate_payload_for_work(work, payload)
        if self != ReviewedTextCampaignReceipt.build(spec):
            raise ReviewedTextMaterializationError(
                "reviewed campaign receipt/spec/output mismatch"
            )
        return self


def loads_reviewed_text_campaign_receipt(
    text: str,
) -> ReviewedTextCampaignReceipt:
    try:
        return ReviewedTextCampaignReceipt.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise ReviewedTextMaterializationError(
            f"reviewed campaign receipt: {exc}"
        ) from exc


def load_reviewed_text_campaign_receipt(
    path: str | os.PathLike[str],
) -> ReviewedTextCampaignReceipt:
    try:
        return ReviewedTextCampaignReceipt.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise ReviewedTextMaterializationError(
            f"reviewed campaign receipt: {exc}"
        ) from exc


def _validate_payload_for_work(
    spec: ReviewedTextWorkSpec,
    payload: bytes,
) -> None:
    identity = _text_identity(
        payload,
        label=f"reviewed text {spec.work_id!r}",
    )
    expected = _TextIdentity(
        spec.byte_size,
        spec.sha256,
        spec.word_count,
        spec.line_count,
        spec.first_nonblank_line_sha256,
        spec.last_nonblank_line_sha256,
    )
    if identity != expected:
        raise ReviewedTextMaterializationError(
            f"reviewed text identity mismatch: {spec.work_id}"
        )


@dataclasses.dataclass(frozen=True)
class MaterializedReviewedTextCampaign:
    root: pathlib.Path
    receipt: ReviewedTextCampaignReceipt
    resumed: bool

    @property
    def spec_path(self) -> pathlib.Path:
        return self.root / REVIEWED_TEXT_CAMPAIGN_SPEC_NAME

    @property
    def receipt_path(self) -> pathlib.Path:
        return self.root / REVIEWED_TEXT_CAMPAIGN_RECEIPT_NAME

    @property
    def output_paths(self) -> tuple[pathlib.Path, ...]:
        return tuple(
            self.root.joinpath(
                *PurePosixPath(row.output_relative_path).parts
            )
            for row in self.receipt.works
        )

    def output_path(self, work_id: str) -> pathlib.Path:
        work = _work_id(work_id, "materialized output work_id")
        for row in self.receipt.works:
            if row.work_id == work:
                return self.root.joinpath(
                    *PurePosixPath(row.output_relative_path).parts
                )
        raise ReviewedTextMaterializationError(
            f"work is not in materialized reviewed campaign: {work}"
        )


def _reject_symlink_components(
    path: pathlib.Path,
    *,
    label: str,
) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ReviewedTextMaterializationError(
                f"{label} must not contain symlink components: {component}"
            )


@contextlib.contextmanager
def _publication_lock(parent: pathlib.Path):
    lock = parent / ".reviewed-text-vnext.lock"
    if lock.is_symlink():
        raise ReviewedTextMaterializationError(
            "reviewed-text publication lock must not be a symlink"
        )
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as exc:
        raise ReviewedTextMaterializationError(
            "cannot open reviewed-text publication lock safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReviewedTextMaterializationError(
                "reviewed-text publication lock is not a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _tree_inventory(
    root: pathlib.Path,
) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        for entry in os.scandir(directory):
            metadata = entry.stat(follow_symlinks=False)
            path = pathlib.Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise ReviewedTextMaterializationError(
                    f"symlink rejected in reviewed-text generation: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise ReviewedTextMaterializationError(
                    f"special file rejected in reviewed-text generation: "
                    f"{relative}"
                )
    return files, directories


def _expected_inventory(
    spec: ReviewedTextCampaignSpec,
) -> tuple[set[str], set[str]]:
    files = {
        REVIEWED_TEXT_CAMPAIGN_SPEC_NAME,
        REVIEWED_TEXT_CAMPAIGN_RECEIPT_NAME,
    }
    files.update(row.output_relative_path for row in spec.works)
    directories: set[str] = set()
    for relative in files:
        directories.update(
            parent.as_posix()
            for parent in PurePosixPath(relative).parents
            if parent.as_posix() != "."
        )
    return files, directories


def _load_canonical_spec_file(
    path: pathlib.Path,
) -> ReviewedTextCampaignSpec:
    spec = load_reviewed_text_campaign_spec(path)
    try:
        observed = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReviewedTextMaterializationError(
            "materialized reviewed campaign spec is unreadable"
        ) from exc
    if observed != _canonical_json_text(spec.to_dict()):
        raise ReviewedTextMaterializationError(
            "materialized reviewed campaign spec has noncanonical JSON bytes"
        )
    return spec


def _load_canonical_receipt_file(
    path: pathlib.Path,
) -> ReviewedTextCampaignReceipt:
    receipt = load_reviewed_text_campaign_receipt(path)
    try:
        observed = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReviewedTextMaterializationError(
            "materialized reviewed campaign receipt is unreadable"
        ) from exc
    if observed != _canonical_json_text(receipt.to_dict()):
        raise ReviewedTextMaterializationError(
            "materialized reviewed campaign receipt has noncanonical JSON "
            "bytes"
        )
    return receipt


def _load_existing(
    root: pathlib.Path,
    spec: ReviewedTextCampaignSpec,
) -> MaterializedReviewedTextCampaign:
    if root.is_symlink() or not root.is_dir():
        raise ReviewedTextMaterializationError(
            f"reviewed-text generation is not a real directory: {root}"
        )
    observed_files, observed_directories = _tree_inventory(root)
    expected_files, expected_directories = _expected_inventory(spec)
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise ReviewedTextMaterializationError(
            "reviewed-text generation has missing or extra files/directories"
        )
    persisted_spec = _load_canonical_spec_file(
        root / REVIEWED_TEXT_CAMPAIGN_SPEC_NAME
    )
    if persisted_spec != spec:
        raise ReviewedTextMaterializationError(
            "materialized reviewed campaign spec differs from requested spec"
        )
    payloads: list[bytes] = []
    for work in spec.works:
        path = root.joinpath(
            *PurePosixPath(work.output_relative_path).parts
        )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ReviewedTextMaterializationError(
                f"materialized reviewed text is unreadable: {work.work_id}"
            ) from exc
        _validate_payload_for_work(work, payload)
        payloads.append(payload)
    receipt = _load_canonical_receipt_file(
        root / REVIEWED_TEXT_CAMPAIGN_RECEIPT_NAME
    )
    receipt.validate_for(spec, payloads)
    return MaterializedReviewedTextCampaign(root, receipt, True)


def _prepare_cache_root(
    artifact_cache: str | os.PathLike[str],
) -> pathlib.Path:
    cache = pathlib.Path(artifact_cache)
    _reject_symlink_components(cache, label="reviewed artifact cache")
    if cache.is_symlink() or not cache.is_dir():
        raise ReviewedTextMaterializationError(
            "reviewed artifact cache must be an existing real directory"
        )
    return cache.resolve(strict=True)


def _read_cache_artifact(
    cache: pathlib.Path,
    spec: ReviewedTextWorkSpec,
) -> bytes:
    path = cache.joinpath(*PurePosixPath(spec.artifact_key).parts)
    _reject_symlink_components(
        path,
        label=f"reviewed artifact cache entry {spec.work_id!r}",
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReviewedTextMaterializationError(
            f"reviewed artifact cache entry is missing or unsafe: "
            f"{spec.artifact_key}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReviewedTextMaterializationError(
                f"reviewed artifact cache entry is not a regular file: "
                f"{spec.artifact_key}"
            )
        if before.st_size != spec.byte_size:
            raise ReviewedTextMaterializationError(
                f"reviewed artifact cache entry byte size mismatch: "
                f"{spec.artifact_key}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ReviewedTextMaterializationError(
                f"reviewed artifact cache entry changed while reading: "
                f"{spec.artifact_key}"
            )
    finally:
        os.close(descriptor)
    _validate_payload_for_work(spec, payload)
    return payload


def materialize_reviewed_text_campaign(
    spec: ReviewedTextCampaignSpec,
    *,
    artifact_cache: str | os.PathLike[str],
    output_parent: str | os.PathLike[str],
) -> MaterializedReviewedTextCampaign:
    """Immutably publish one exact campaign from a local artifact cache."""

    if type(spec) is not ReviewedTextCampaignSpec:
        raise ReviewedTextMaterializationError(
            "reviewed-text materialization requires exactly "
            "ReviewedTextCampaignSpec"
        )
    spec.validate()
    parent = pathlib.Path(output_parent)
    _reject_symlink_components(parent, label="reviewed-text output parent")
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ReviewedTextMaterializationError(
            "reviewed-text output parent must be a real directory"
        )
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    target = parent / spec.self_hash

    with _publication_lock(parent):
        if target.exists() or target.is_symlink():
            return _load_existing(target, spec)

    cache = _prepare_cache_root(artifact_cache)
    payloads = [_read_cache_artifact(cache, work) for work in spec.works]
    receipt = ReviewedTextCampaignReceipt.build(spec)
    stage = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".reviewed-text.{spec.self_hash[:12]}.",
            dir=parent,
        )
    )
    try:
        for work, payload in zip(spec.works, payloads, strict=True):
            output = stage.joinpath(
                *PurePosixPath(work.output_relative_path).parts
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
        dump_strict(
            spec.to_dict(),
            stage / REVIEWED_TEXT_CAMPAIGN_SPEC_NAME,
            sort_keys=True,
            trailing_newline=True,
        )
        dump_strict(
            receipt.to_dict(),
            stage / REVIEWED_TEXT_CAMPAIGN_RECEIPT_NAME,
            sort_keys=True,
            trailing_newline=True,
        )
        staged = _load_existing(stage, spec)
        with _publication_lock(parent):
            if target.exists() or target.is_symlink():
                return _load_existing(target, spec)
            os.rename(stage, target)
        return MaterializedReviewedTextCampaign(
            target,
            staged.receipt,
            False,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


__all__ = [
    "MaterializedReviewedTextCampaign",
    "REVIEWED_TEXT_ARTIFACT_KEY_POLICY_VERSION",
    "REVIEWED_TEXT_CAMPAIGN_KIND",
    "REVIEWED_TEXT_CAMPAIGN_RECEIPT_NAME",
    "REVIEWED_TEXT_CAMPAIGN_RECEIPT_SCHEMA_VERSION",
    "REVIEWED_TEXT_CAMPAIGN_SPEC_NAME",
    "REVIEWED_TEXT_CAMPAIGN_SPEC_SCHEMA_VERSION",
    "REVIEWED_TEXT_IDENTITY_POLICY_VERSION",
    "ReviewedTextArtifactRef",
    "ReviewedTextCampaignReceipt",
    "ReviewedTextCampaignSpec",
    "ReviewedTextMaterializationError",
    "ReviewedTextWorkReceipt",
    "ReviewedTextWorkSpec",
    "UPSTREAM_INVENTORY_FILE_SHA256",
    "UPSTREAM_QUALITY_AUDIT_FILE_SHA256",
    "UPSTREAM_QUALITY_AUDIT_SELF_HASH",
    "load_reviewed_text_campaign_receipt",
    "load_reviewed_text_campaign_spec",
    "loads_reviewed_text_campaign_receipt",
    "loads_reviewed_text_campaign_spec",
    "materialize_reviewed_text_campaign",
]
