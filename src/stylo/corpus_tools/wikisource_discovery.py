"""Fail-closed promotion of reviewed Wikisource discovery into pinned specs.

Discovery is intentionally not acquisition.  A discovery candidate may record
missing pages or unresolved edition/content decisions, but such a candidate
must never become a resumable corpus input.  This module performs the narrow
promotion step:

* parse the candidate as strict, duplicate-key-free JSON and verify its hash;
* reject every unresolved work, page, or choice before the first HTTP request;
* fetch each exact revision by ``revid`` and render it by ``oldid``;
* verify the returned revision identity and immutable wikitext metadata;
* derive the hashes and counts required by :class:`PinnedWorkSpec`; and
* build the deterministic multi-work acquisition campaign.

HTTP responses are cached in create-if-absent, self-hashed files whose names
bind the revision, request kind, and canonical response SHA-256.  A malformed
or conflicting cache is an error; it is never silently refreshed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import re
import stat
from collections.abc import Mapping
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from ..jsonio import (
    StrictJSONError,
    canonical_hash,
    dumps_strict,
    load_strict,
    loads_strict,
)
from .wikisource_campaign import WikisourceCampaignSpec
from .wikisource_vnext import (
    API,
    ASSEMBLY_POLICY_VERSION,
    BODY_BOUNDARY_POLICY_VERSION_V2,
    BODY_DISPOSITIONS,
    EXTRACTION_POLICY_VERSION,
    PINNED_WORK_SPEC_SCHEMA_VERSION_V2,
    PINNED_WORK_SPEC_SCHEMA_VERSION_V3,
    PINNED_WORK_SPEC_SCHEMA_VERSION_V4,
    RESIDUE_POLICY_VERSION,
    SOURCE_REPAIR_POLICY_VERSION_V1,
    WORD_COUNT_POLICY_VERSION,
    JSONTransport,
    PinnedPartSpec,
    PinnedWorkSpec,
    RedirectHop,
    ReviewedLiteralSourceRepairV1,
    WikisourceAcquisitionError,
    apply_reviewed_source_repair_v1,
    apply_reviewed_source_repairs_v1,
    assemble_plain_parts,
    build_body_selection_v2,
    count_words,
    extract_rendered_html,
)


DISCOVERY_CANDIDATE_SCHEMA_VERSION = (
    "stylo.ruaa-r1-wikisource-source-spec-candidate.v1"
)
DISCOVERY_CANDIDATE_SCHEMA_VERSION_V2 = (
    "stylo.ruaa-r1-wikisource-source-spec-candidate.v2"
)
DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3 = (
    "stylo.ruaa-r1-wikisource-source-spec-candidate.v3"
)
DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4 = (
    "stylo.ruaa-r1-wikisource-source-spec-candidate.v4"
)
PINNING_CACHE_SCHEMA_VERSION = "stylo.wikisource.pinning-response-cache.v1"
READY_STATUS = "ready_for_pinning"
BLOCKED_STATUS = "blocked"
SELECTED_WORK_STATUS = "selected"
EXTERNAL_PROVIDER_WORK_STATUS = "external_provider_selected"
SOURCE_QUALITY_REJECTED_WORK_STATUS = "source_quality_rejected"
RESOLVED_PART_STATUS = "resolved"
PARSE_OLDID_STRATEGY = "parse_oldid"
RENDERED_OLDID_MATERIALIZATION = (
    "rendered HTML pinned by page revision oldid"
)
TRUSTED_SITE = "ru.wikisource.org"
TRUSTED_API = API

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "source",
        "generated_at",
        "rejected_work_receipts",
        "selection_contract",
        "summary",
        "unresolved_pages",
        "unresolved_choices",
        "works",
        "candidate_hash",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "site",
        "api",
        "root_probe_schema_version",
        "root_probe_sha256",
    }
)
_SELECTION_CONTRACT_KEYS = frozenset(
    {
        "source_work_count",
        "selected_work_count",
        "excluded_work_ids",
        "part_order_source",
        "one_edition_per_work",
        "materialization",
        "candidate_only",
    }
)
_WORK_KEYS = frozenset(
    {
        "work_id",
        "include_in_corpus",
        "legacy_requested_root_title",
        "legacy_root_revision_id",
        "selection_basis",
        "selection_status",
        "issues",
        "parts",
    }
)
_PART_KEYS = frozenset(
    {
        "ordinal",
        "requested_title",
        "resolved_title",
        "redirect_chain",
        "page_id",
        "revision_id",
        "revision_parent_id",
        "revision_sha1",
        "revision_timestamp",
        "status",
        "acquisition_strategy",
    }
)
_PART_BOUNDARY_V2_KEYS = frozenset(
    {
        "body_start_line",
        "body_end_line_exclusive",
        "body_disposition",
    }
)
_PART_REPAIR_V3_KEYS = frozenset({"source_repair"})
_PART_REPAIRS_V4_KEYS = frozenset({"source_repairs"})
_SUMMARY_KEYS = frozenset(
    {
        "work_count",
        "part_count",
        "resolved_part_count",
        "missing_part_count",
        "blocked_work_count",
        "authorship_rejected_work_count",
    }
)
_ISSUE_KEYS = frozenset({"chosen_disposition", "kind", "reason"})
_EDITION_ISSUE_KEYS = frozenset(
    {*_ISSUE_KEYS, "candidate", "alternatives"}
)
_UNRESOLVED_CHOICE_KEYS = frozenset({"work_id", "issues"})
_REJECTED_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "work_id",
        "reason_code",
        "disposition",
        "evidence",
        "self_hash",
    }
)
_REJECTED_EVIDENCE_KEYS = frozenset(
    {
        "requested_title",
        "resolved_title",
        "page_id",
        "revision_id",
        "revision_sha1",
        "publication",
        "closing_signature",
        "body_characterization",
    }
)
_REJECTED_RECEIPT_SCHEMA_VERSION = "stylo.ruaa-r1-rejected-work-receipt.v1"
_REDIRECT_HOP_KEYS = frozenset({"from", "to"})
_CACHE_KEYS = frozenset(
    {
        "schema_version",
        "request_kind",
        "request",
        "revision_id",
        "response_sha256",
        "response",
        "self_hash",
    }
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_WIKI_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)
_CACHE_NAME_RE = re.compile(
    r"^(query|parse)-([1-9][0-9]*)-([0-9a-f]{64})\.json$"
)
_REDIRECT_WIKITEXT_RE = re.compile(
    r"(?im)^\s*#\s*(?:redirect|перенаправление)\b"
)
_ALLOWED_WORK_STATUSES = frozenset(
    {
        SELECTED_WORK_STATUS,
        "content_boundary_unresolved",
        "source_incomplete",
        "authorship_rejected",
        EXTERNAL_PROVIDER_WORK_STATUS,
        SOURCE_QUALITY_REJECTED_WORK_STATUS,
    }
)


class WikisourceDiscoveryError(WikisourceAcquisitionError):
    """A discovery candidate, exact response, or pinning cache is unsafe."""


def _exact_object(
    value: object,
    keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise WikisourceDiscoveryError(f"{label} must be an exact JSON object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise WikisourceDiscoveryError(
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
        raise WikisourceDiscoveryError(
            f"{label} must be an exact{qualifier} array"
        )
    return value


def _exact_str(value: object, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise WikisourceDiscoveryError(
            f"{label} must be an exact non-empty NUL-free string"
        )
    return value


def _single_line(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\r" in text or "\n" in text:
        raise WikisourceDiscoveryError(f"{label} must be a single line")
    return text


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WikisourceDiscoveryError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise WikisourceDiscoveryError(f"{label} must be an exact boolean")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _exact_str(value, label)
    if _HEX64_RE.fullmatch(digest) is None:
        raise WikisourceDiscoveryError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return digest


def _wiki_sha1(value: object, label: str) -> str:
    digest = _exact_str(value, label)
    if _WIKI_SHA1_RE.fullmatch(digest) is None:
        raise WikisourceDiscoveryError(
            f"{label} must be 40 lowercase hexadecimal characters"
        )
    return digest


def _timestamp(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise WikisourceDiscoveryError(
            f"{label} must be a canonical UTC timestamp"
        )
    return text


def _work_id(value: object, label: str = "work_id") -> str:
    text = _single_line(value, label)
    if "\\" in text:
        raise WikisourceDiscoveryError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WikisourceDiscoveryError(
            f"{label} must be a canonical author/work identifier"
        )
    return text


def _string_array(value: object, label: str) -> tuple[str, ...]:
    rows = tuple(
        _single_line(item, f"{label}[{index}]")
        for index, item in enumerate(_exact_list(value, label))
    )
    if len(rows) != len(set(rows)):
        raise WikisourceDiscoveryError(f"{label} must not contain duplicates")
    return rows


def _redirect_chain(
    value: object,
    *,
    requested_title: str,
    resolved_title: str,
    label: str,
) -> tuple[RedirectHop, ...]:
    rows: list[RedirectHop] = []
    for index, item in enumerate(_exact_list(value, label)):
        raw = _exact_object(
            item,
            _REDIRECT_HOP_KEYS,
            f"{label}[{index}]",
        )
        rows.append(
            RedirectHop(
                _single_line(raw["from"], f"{label}[{index}].from"),
                _single_line(raw["to"], f"{label}[{index}].to"),
            )
        )
    # Reuse the strict continuity/cycle validation in PinnedPartSpec.
    probe = {
        "ordinal": 0,
        "requested_title": requested_title,
        "resolved_title": resolved_title,
        "redirect_chain": [row.to_dict() for row in rows],
        "page_id": 1,
        "revision_id": 1,
        "mediawiki_sha1": "0" * 40,
        "wikitext_sha256": "0" * 64,
        "rendered_html_sha256": "0" * 64,
        "plain_byte_size": 1,
        "plain_sha256": "0" * 64,
        "word_count": 1,
    }
    try:
        PinnedPartSpec.from_dict(probe)
    except WikisourceAcquisitionError as exc:
        raise WikisourceDiscoveryError(f"{label}: {exc}") from exc
    return tuple(rows)


def _summary(value: object) -> dict[str, int]:
    raw = _exact_object(
        value,
        _SUMMARY_KEYS,
        "discovery candidate.summary",
    )
    return {
        key: _exact_int(
            raw[key],
            f"discovery candidate.summary.{key}",
            minimum=0,
        )
        for key in sorted(_SUMMARY_KEYS)
    }


@dataclasses.dataclass(frozen=True)
class DiscoveryIssue:
    chosen_disposition: str
    kind: str
    reason: str
    candidate: str | None
    alternatives: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object, *, label: str) -> "DiscoveryIssue":
        if type(value) is not dict:
            raise WikisourceDiscoveryError(
                f"{label} must be an exact JSON object"
            )
        actual = set(value)
        if actual == set(_ISSUE_KEYS):
            raw = value
            candidate = None
            alternatives: tuple[str, ...] = ()
        elif actual == set(_EDITION_ISSUE_KEYS):
            raw = value
            candidate = _single_line(raw["candidate"], f"{label}.candidate")
            alternatives = _string_array(
                raw["alternatives"],
                f"{label}.alternatives",
            )
            if not alternatives or candidate in alternatives:
                raise WikisourceDiscoveryError(
                    f"{label} edition alternatives must be non-empty and "
                    "exclude the selected candidate"
                )
        else:
            raise WikisourceDiscoveryError(
                f"{label} keys must match an exact supported issue shape; "
                f"observed={sorted(actual)}"
            )
        return cls(
            _single_line(
                raw["chosen_disposition"],
                f"{label}.chosen_disposition",
            ),
            _single_line(raw["kind"], f"{label}.kind"),
            _exact_str(raw["reason"], f"{label}.reason"),
            candidate,
            alternatives,
        )


def _issues(value: object, *, label: str) -> tuple[DiscoveryIssue, ...]:
    rows = tuple(
        DiscoveryIssue.from_dict(item, label=f"{label}[{index}]")
        for index, item in enumerate(_exact_list(value, label))
    )
    identities = tuple(
        (
            row.chosen_disposition,
            row.kind,
            row.reason,
            row.candidate,
            row.alternatives,
        )
        for row in rows
    )
    if len(identities) != len(set(identities)):
        raise WikisourceDiscoveryError(f"{label} contains duplicate issues")
    return rows


@dataclasses.dataclass(frozen=True)
class UnresolvedChoice:
    work_id: str
    issues: tuple[DiscoveryIssue, ...]

    @classmethod
    def from_dict(cls, value: object, *, index: int) -> "UnresolvedChoice":
        label = f"discovery candidate.unresolved_choices[{index}]"
        raw = _exact_object(value, _UNRESOLVED_CHOICE_KEYS, label)
        issues = _issues(raw["issues"], label=f"{label}.issues")
        if not issues:
            raise WikisourceDiscoveryError(
                f"{label}.issues must be non-empty"
            )
        return cls(
            _work_id(raw["work_id"], f"{label}.work_id"),
            issues,
        )


@dataclasses.dataclass(frozen=True)
class RejectedWorkReceipt:
    work_id: str
    reason_code: str
    disposition: str
    self_hash: str

    @classmethod
    def from_dict(cls, value: object, *, index: int) -> "RejectedWorkReceipt":
        label = f"discovery candidate.rejected_work_receipts[{index}]"
        raw = _exact_object(value, _REJECTED_RECEIPT_KEYS, label)
        if raw["schema_version"] != _REJECTED_RECEIPT_SCHEMA_VERSION:
            raise WikisourceDiscoveryError(
                f"{label} is legacy or unsupported"
            )
        recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
        payload = {key: item for key, item in raw.items() if key != "self_hash"}
        if canonical_hash(payload) != recorded:
            raise WikisourceDiscoveryError(f"{label} self_hash mismatch")
        evidence = _exact_object(
            raw["evidence"],
            _REJECTED_EVIDENCE_KEYS,
            f"{label}.evidence",
        )
        _single_line(
            evidence["requested_title"],
            f"{label}.evidence.requested_title",
        )
        _single_line(
            evidence["resolved_title"],
            f"{label}.evidence.resolved_title",
        )
        _exact_int(
            evidence["page_id"],
            f"{label}.evidence.page_id",
            minimum=1,
        )
        _exact_int(
            evidence["revision_id"],
            f"{label}.evidence.revision_id",
            minimum=1,
        )
        _wiki_sha1(
            evidence["revision_sha1"],
            f"{label}.evidence.revision_sha1",
        )
        for key in (
            "publication",
            "closing_signature",
            "body_characterization",
        ):
            _exact_str(evidence[key], f"{label}.evidence.{key}")
        disposition = _single_line(
            raw["disposition"],
            f"{label}.disposition",
        )
        if disposition != "exclude_from_corpus":
            raise WikisourceDiscoveryError(
                f"{label}.disposition must be 'exclude_from_corpus'"
            )
        return cls(
            _work_id(raw["work_id"], f"{label}.work_id"),
            _single_line(raw["reason_code"], f"{label}.reason_code"),
            disposition,
            recorded,
        )


@dataclasses.dataclass(frozen=True)
class DiscoverySource:
    site: str
    api: str
    root_probe_schema_version: str
    root_probe_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "DiscoverySource":
        raw = _exact_object(value, _SOURCE_KEYS, "discovery candidate.source")
        site = _single_line(raw["site"], "discovery candidate.source.site")
        api = _single_line(raw["api"], "discovery candidate.source.api")
        if site != TRUSTED_SITE or api != TRUSTED_API:
            raise WikisourceDiscoveryError(
                "discovery candidate source is not the trusted "
                f"{TRUSTED_SITE} Action API"
            )
        return cls(
            site,
            api,
            _single_line(
                raw["root_probe_schema_version"],
                "discovery candidate.source.root_probe_schema_version",
            ),
            _sha256(
                raw["root_probe_sha256"],
                "discovery candidate.source.root_probe_sha256",
            ),
        )


@dataclasses.dataclass(frozen=True)
class SelectionContract:
    source_work_count: int
    selected_work_count: int
    excluded_work_ids: tuple[str, ...]
    part_order_source: str
    one_edition_per_work: bool
    materialization: str
    candidate_only: bool

    @classmethod
    def from_dict(cls, value: object) -> "SelectionContract":
        raw = _exact_object(
            value,
            _SELECTION_CONTRACT_KEYS,
            "discovery candidate.selection_contract",
        )
        excluded = tuple(
            _work_id(
                item,
                f"discovery candidate.selection_contract."
                f"excluded_work_ids[{index}]",
            )
            for index, item in enumerate(
                _exact_list(
                    raw["excluded_work_ids"],
                    "discovery candidate.selection_contract."
                    "excluded_work_ids",
                )
            )
        )
        if excluded != tuple(sorted(excluded)) or len(excluded) != len(
            set(excluded)
        ):
            raise WikisourceDiscoveryError(
                "selection_contract.excluded_work_ids must be sorted and unique"
            )
        one_edition = _exact_bool(
            raw["one_edition_per_work"],
            "selection_contract.one_edition_per_work",
        )
        candidate_only = _exact_bool(
            raw["candidate_only"],
            "selection_contract.candidate_only",
        )
        if not one_edition or not candidate_only:
            raise WikisourceDiscoveryError(
                "selection contract must freeze one edition per work and "
                "remain candidate-only"
            )
        materialization = _single_line(
            raw["materialization"],
            "selection_contract.materialization",
        )
        if materialization != RENDERED_OLDID_MATERIALIZATION:
            raise WikisourceDiscoveryError(
                "selection contract materialization must be parse-oldid only"
            )
        return cls(
            _exact_int(
                raw["source_work_count"],
                "selection_contract.source_work_count",
                minimum=1,
            ),
            _exact_int(
                raw["selected_work_count"],
                "selection_contract.selected_work_count",
                minimum=1,
            ),
            excluded,
            _single_line(
                raw["part_order_source"],
                "selection_contract.part_order_source",
            ),
            one_edition,
            materialization,
            candidate_only,
        )


@dataclasses.dataclass(frozen=True)
class DiscoveryPart:
    ordinal: int
    requested_title: str
    resolved_title: str
    redirect_chain: tuple[RedirectHop, ...]
    page_id: int
    revision_id: int
    revision_parent_id: int
    revision_sha1: str
    revision_timestamp: str
    status: str
    acquisition_strategy: str
    body_start_line: int
    body_end_line_exclusive: int | None
    body_disposition: str
    source_repair_v1: ReviewedLiteralSourceRepairV1 | None
    source_repairs_v1: tuple[ReviewedLiteralSourceRepairV1, ...] | None

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        label: str,
        candidate_schema_version: str,
    ) -> "DiscoveryPart":
        if candidate_schema_version == DISCOVERY_CANDIDATE_SCHEMA_VERSION:
            boundary_keys: frozenset[str] = frozenset()
        elif (
            candidate_schema_version
            == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V2
        ):
            boundary_keys = _PART_BOUNDARY_V2_KEYS
        elif (
            candidate_schema_version
            == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3
        ):
            boundary_keys = _PART_BOUNDARY_V2_KEYS | _PART_REPAIR_V3_KEYS
        elif (
            candidate_schema_version
            == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4
        ):
            boundary_keys = _PART_BOUNDARY_V2_KEYS | _PART_REPAIRS_V4_KEYS
        else:  # Guarded at the candidate boundary.
            raise WikisourceDiscoveryError(
                f"{label} belongs to an unsupported candidate schema"
            )
        raw = _exact_object(value, _PART_KEYS | boundary_keys, label)
        requested = _single_line(
            raw["requested_title"],
            f"{label}.requested_title",
        )
        resolved = _single_line(
            raw["resolved_title"],
            f"{label}.resolved_title",
        )
        strategy = _single_line(
            raw["acquisition_strategy"],
            f"{label}.acquisition_strategy",
        )
        if strategy != PARSE_OLDID_STRATEGY:
            raise WikisourceDiscoveryError(
                f"{label}.acquisition_strategy must be "
                f"{PARSE_OLDID_STRATEGY!r}"
            )
        if candidate_schema_version == DISCOVERY_CANDIDATE_SCHEMA_VERSION:
            body_start = 0
            body_end: int | None = None
            body_disposition = "whole_rendered_body"
            source_repair = None
            source_repairs = None
        else:
            body_start = _exact_int(
                raw["body_start_line"],
                f"{label}.body_start_line",
                minimum=0,
            )
            raw_end = raw["body_end_line_exclusive"]
            body_end = (
                None
                if raw_end is None
                else _exact_int(
                    raw_end,
                    f"{label}.body_end_line_exclusive",
                    minimum=1,
                )
            )
            body_disposition = _single_line(
                raw["body_disposition"],
                f"{label}.body_disposition",
            )
            if body_disposition not in BODY_DISPOSITIONS:
                raise WikisourceDiscoveryError(
                    f"{label}.body_disposition must be one of "
                    f"{sorted(BODY_DISPOSITIONS)!r}"
                )
            valid_shape = {
                "whole_rendered_body": body_start == 0 and body_end is None,
                "strip_leading_apparatus": (
                    body_start > 0 and body_end is None
                ),
                "strip_trailing_apparatus": (
                    body_start == 0 and body_end is not None
                ),
                "strip_both_apparatus": (
                    body_start > 0 and body_end is not None
                ),
            }[body_disposition]
            if not valid_shape:
                raise WikisourceDiscoveryError(
                    f"{label} body boundary conflicts with disposition"
                )
            if body_end is not None and body_start >= body_end:
                raise WikisourceDiscoveryError(
                    f"{label} body boundary must be non-empty"
                )
            if (
                candidate_schema_version
                == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3
            ):
                raw_repair = raw["source_repair"]
                try:
                    source_repair = (
                        None
                        if raw_repair is None
                        else ReviewedLiteralSourceRepairV1.from_dict(
                            raw_repair,
                            label=f"{label}.source_repair",
                        )
                    )
                except WikisourceAcquisitionError as exc:
                    raise WikisourceDiscoveryError(str(exc)) from exc
                source_repairs = None
            elif (
                candidate_schema_version
                == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4
            ):
                source_repair = None
                try:
                    source_repairs = tuple(
                        ReviewedLiteralSourceRepairV1.from_dict(
                            item,
                            label=f"{label}.source_repairs[{index}]",
                        )
                        for index, item in enumerate(
                            _exact_list(
                                raw["source_repairs"],
                                f"{label}.source_repairs",
                            )
                        )
                    )
                except WikisourceAcquisitionError as exc:
                    raise WikisourceDiscoveryError(str(exc)) from exc
            else:
                source_repair = None
                source_repairs = None
        return cls(
            _exact_int(raw["ordinal"], f"{label}.ordinal", minimum=0),
            requested,
            resolved,
            _redirect_chain(
                raw["redirect_chain"],
                requested_title=requested,
                resolved_title=resolved,
                label=f"{label}.redirect_chain",
            ),
            _exact_int(raw["page_id"], f"{label}.page_id", minimum=1),
            _exact_int(
                raw["revision_id"],
                f"{label}.revision_id",
                minimum=1,
            ),
            _exact_int(
                raw["revision_parent_id"],
                f"{label}.revision_parent_id",
                minimum=0,
            ),
            _wiki_sha1(raw["revision_sha1"], f"{label}.revision_sha1"),
            _timestamp(
                raw["revision_timestamp"],
                f"{label}.revision_timestamp",
            ),
            _single_line(raw["status"], f"{label}.status"),
            strategy,
            body_start,
            body_end,
            body_disposition,
            source_repair,
            source_repairs,
        )


@dataclasses.dataclass(frozen=True)
class DiscoveryWork:
    work_id: str
    include_in_corpus: bool
    legacy_requested_root_title: str
    legacy_root_revision_id: int | None
    selection_basis: str
    selection_status: str
    issues: tuple[DiscoveryIssue, ...]
    parts: tuple[DiscoveryPart, ...]

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        index: int,
        candidate_schema_version: str,
    ) -> "DiscoveryWork":
        label = f"discovery candidate.works[{index}]"
        raw = _exact_object(value, _WORK_KEYS, label)
        status = _single_line(
            raw["selection_status"],
            f"{label}.selection_status",
        )
        if status not in _ALLOWED_WORK_STATUSES:
            raise WikisourceDiscoveryError(
                f"{label}.selection_status is unsupported: {status!r}"
            )
        parts = tuple(
            DiscoveryPart.from_dict(
                item,
                label=f"{label}.parts[{part_index}]",
                candidate_schema_version=candidate_schema_version,
            )
            for part_index, item in enumerate(
                _exact_list(raw["parts"], f"{label}.parts")
            )
        )
        if parts and tuple(part.ordinal for part in parts) != tuple(
            range(len(parts))
        ):
            raise WikisourceDiscoveryError(
                f"{label}.parts must have contiguous zero-based order"
            )
        if len({part.page_id for part in parts}) != len(parts):
            raise WikisourceDiscoveryError(
                f"{label}.parts contains duplicate page ids"
            )
        if len({part.revision_id for part in parts}) != len(parts):
            raise WikisourceDiscoveryError(
                f"{label}.parts contains duplicate revision ids"
            )
        include = _exact_bool(
            raw["include_in_corpus"],
            f"{label}.include_in_corpus",
        )
        if include != (status == SELECTED_WORK_STATUS):
            raise WikisourceDiscoveryError(
                f"{label}.include_in_corpus conflicts with selection_status"
            )
        legacy_revision = raw["legacy_root_revision_id"]
        if legacy_revision is not None:
            legacy_revision = _exact_int(
                legacy_revision,
                f"{label}.legacy_root_revision_id",
                minimum=1,
            )
        return cls(
            _work_id(raw["work_id"], f"{label}.work_id"),
            include,
            _single_line(
                raw["legacy_requested_root_title"],
                f"{label}.legacy_requested_root_title",
            ),
            legacy_revision,
            _single_line(raw["selection_basis"], f"{label}.selection_basis"),
            status,
            _issues(raw["issues"], label=f"{label}.issues"),
            parts,
        )


@dataclasses.dataclass(frozen=True)
class DiscoveryCandidate:
    schema_version: str
    status: str
    source: DiscoverySource
    generated_at: str
    selection_contract: SelectionContract
    summary: Mapping[str, int]
    unresolved_pages: tuple[object, ...]
    unresolved_choices: tuple[UnresolvedChoice, ...]
    rejected_work_receipts: tuple[RejectedWorkReceipt, ...]
    works: tuple[DiscoveryWork, ...]
    candidate_hash: str

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryCandidate":
        raw = _exact_object(value, _TOP_LEVEL_KEYS, "discovery candidate")
        schema_version = raw["schema_version"]
        if schema_version not in {
            DISCOVERY_CANDIDATE_SCHEMA_VERSION,
            DISCOVERY_CANDIDATE_SCHEMA_VERSION_V2,
            DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3,
            DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4,
        }:
            raise WikisourceDiscoveryError(
                "discovery candidate is legacy or unsupported"
            )
        recorded_hash = _sha256(
            raw["candidate_hash"],
            "discovery candidate.candidate_hash",
        )
        payload = {
            key: item for key, item in raw.items() if key != "candidate_hash"
        }
        if canonical_hash(payload) != recorded_hash:
            raise WikisourceDiscoveryError(
                "discovery candidate candidate_hash mismatch"
            )
        status = _single_line(raw["status"], "discovery candidate.status")
        if status not in {READY_STATUS, BLOCKED_STATUS}:
            raise WikisourceDiscoveryError(
                f"discovery candidate.status is unsupported: {status!r}"
            )
        contract = SelectionContract.from_dict(raw["selection_contract"])
        works = tuple(
            DiscoveryWork.from_dict(
                item,
                index=index,
                candidate_schema_version=schema_version,
            )
            for index, item in enumerate(
                _exact_list(
                    raw["works"],
                    "discovery candidate.works",
                    nonempty=True,
                )
            )
        )
        work_ids = tuple(work.work_id for work in works)
        if work_ids != tuple(sorted(work_ids)):
            raise WikisourceDiscoveryError(
                "discovery candidate works must be sorted by work_id"
            )
        if len(work_ids) != len(set(work_ids)):
            raise WikisourceDiscoveryError(
                "discovery candidate contains duplicate work ids"
            )
        if contract.source_work_count != len(works):
            raise WikisourceDiscoveryError(
                "selection_contract.source_work_count differs from works"
            )
        included = tuple(work for work in works if work.include_in_corpus)
        if contract.selected_work_count != len(included):
            raise WikisourceDiscoveryError(
                "selection_contract.selected_work_count differs from included "
                "works"
            )
        included_ids = {work.work_id for work in included}
        if included_ids.intersection(contract.excluded_work_ids):
            raise WikisourceDiscoveryError(
                "selected and excluded work inventories overlap"
            )
        seen_pages: set[int] = set()
        seen_revisions: set[int] = set()
        for work in works:
            for part in work.parts:
                if part.page_id in seen_pages:
                    raise WikisourceDiscoveryError(
                        "discovery candidate reuses a page across works"
                    )
                if part.revision_id in seen_revisions:
                    raise WikisourceDiscoveryError(
                        "discovery candidate reuses a revision across works"
                    )
                seen_pages.add(part.page_id)
                seen_revisions.add(part.revision_id)
        unresolved_pages = tuple(
            _exact_list(
                raw["unresolved_pages"],
                "discovery candidate.unresolved_pages",
            )
        )
        unresolved_choices = tuple(
            UnresolvedChoice.from_dict(item, index=index)
            for index, item in enumerate(
                _exact_list(
                    raw["unresolved_choices"],
                    "discovery candidate.unresolved_choices",
                )
            )
        )
        if len({row.work_id for row in unresolved_choices}) != len(
            unresolved_choices
        ):
            raise WikisourceDiscoveryError(
                "unresolved_choices contains duplicate work ids"
            )
        rejected_receipts = tuple(
            RejectedWorkReceipt.from_dict(item, index=index)
            for index, item in enumerate(
                _exact_list(
                    raw["rejected_work_receipts"],
                    "discovery candidate.rejected_work_receipts",
                )
            )
        )
        if len({row.work_id for row in rejected_receipts}) != len(
            rejected_receipts
        ):
            raise WikisourceDiscoveryError(
                "rejected_work_receipts contains duplicate work ids"
            )
        rejected_work_ids = {
            work.work_id
            for work in works
            if work.selection_status == "authorship_rejected"
        }
        if {row.work_id for row in rejected_receipts} != rejected_work_ids:
            raise WikisourceDiscoveryError(
                "rejected_work_receipts do not exactly cover rejected works"
            )
        summary = _summary(raw["summary"])
        expected_counts = {
            "work_count": len(works),
            "part_count": sum(len(work.parts) for work in works),
            "resolved_part_count": sum(
                part.status == RESOLVED_PART_STATUS
                for work in works
                for part in work.parts
            ),
            "missing_part_count": sum(
                part.status != RESOLVED_PART_STATUS
                for work in works
                for part in work.parts
            ),
            "blocked_work_count": sum(
                work.selection_status != SELECTED_WORK_STATUS
                for work in works
            ),
            "authorship_rejected_work_count": len(rejected_work_ids),
        }
        if summary != {
            key: expected_counts[key] for key in sorted(expected_counts)
        }:
            raise WikisourceDiscoveryError(
                "discovery candidate.summary differs from work/part records"
            )
        return cls(
            schema_version,
            status,
            DiscoverySource.from_dict(raw["source"]),
            _timestamp(raw["generated_at"], "discovery candidate.generated_at"),
            contract,
            summary,
            unresolved_pages,
            unresolved_choices,
            rejected_receipts,
            works,
            recorded_hash,
        )

    def assert_ready(self) -> "DiscoveryCandidate":
        blockers: list[str] = []
        if self.schema_version not in {
            DISCOVERY_CANDIDATE_SCHEMA_VERSION_V2,
            DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3,
            DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4,
        }:
            blockers.append(
                "candidate schema has no explicit v2 body-boundary policy"
            )
        if self.status != READY_STATUS:
            blockers.append(f"top-level status={self.status!r}")
        if self.unresolved_pages:
            blockers.append(
                f"unresolved_pages={len(self.unresolved_pages)}"
            )
        if self.unresolved_choices:
            blockers.append(
                f"unresolved_choices={len(self.unresolved_choices)}"
            )
        for work in self.works:
            if work.selection_status in {
                "content_boundary_unresolved",
                "source_incomplete",
            }:
                blockers.append(
                    f"{work.work_id}: selection_status="
                    f"{work.selection_status!r}"
                )
            if work.include_in_corpus and not work.parts:
                blockers.append(f"{work.work_id}: no ordered parts")
            if work.selection_status == EXTERNAL_PROVIDER_WORK_STATUS:
                if not work.issues or any(
                    issue.chosen_disposition
                    != "pinned_external_provider"
                    for issue in work.issues
                ):
                    blockers.append(
                        f"{work.work_id}: external provider disposition "
                        "is not exact"
                    )
            if work.selection_status == SOURCE_QUALITY_REJECTED_WORK_STATUS:
                if not work.issues or any(
                    issue.chosen_disposition
                    != "exclude_source_quality"
                    for issue in work.issues
                ):
                    blockers.append(
                        f"{work.work_id}: source-quality exclusion "
                        "disposition is not exact"
                    )
            if work.include_in_corpus:
                for issue in work.issues:
                    if issue.chosen_disposition != "selected_candidate":
                        blockers.append(
                            f"{work.work_id}: included issue "
                            "chosen_disposition="
                            f"{issue.chosen_disposition!r}"
                        )
            for part in work.parts:
                if part.status != RESOLVED_PART_STATUS:
                    blockers.append(
                        f"{work.work_id} part {part.ordinal}: "
                        f"status={part.status!r}"
                    )
        if blockers:
            raise WikisourceDiscoveryError(
                "discovery candidate is not eligible for pinning: "
                + "; ".join(blockers)
            )
        return self


def loads_discovery_candidate(text: str) -> DiscoveryCandidate:
    try:
        return DiscoveryCandidate.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise WikisourceDiscoveryError(
            f"discovery candidate: {exc}"
        ) from exc


def load_discovery_candidate(
    path: str | os.PathLike[str],
) -> DiscoveryCandidate:
    try:
        return DiscoveryCandidate.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise WikisourceDiscoveryError(
            f"discovery candidate: {exc}"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha1_bytes(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _request_parameters(
    request_kind: str,
    revision_id: int,
) -> dict[str, str]:
    revision = _exact_int(
        revision_id,
        "cache request revision_id",
        minimum=1,
    )
    if request_kind == "query":
        return {
            "action": "query",
            "prop": "revisions",
            "rvprop": "ids|timestamp|sha1|content",
            "rvslots": "main",
            "revids": str(revision),
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        }
    if request_kind == "parse":
        return {
            "action": "parse",
            "oldid": str(revision),
            "prop": "text|revid",
            "disableeditsection": "1",
            "disablelimitreport": "1",
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        }
    raise WikisourceDiscoveryError(
        f"unsupported pinning request kind: {request_kind!r}"
    )


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise WikisourceDiscoveryError(
                f"{label} must not contain symlink components: {component}"
            )


def _validate_cache_directory(path: pathlib.Path) -> pathlib.Path:
    _reject_symlink_components(path, label="pinning cache")
    if path.exists() and not path.is_dir():
        raise WikisourceDiscoveryError(
            "pinning cache must be a real directory"
        )
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    entries: list[tuple[pathlib.Path, str, int]] = []
    identities: list[tuple[str, int]] = []
    for entry in os.scandir(resolved):
        metadata = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise WikisourceDiscoveryError(
                "pinning cache may contain only regular cache files"
            )
        match = _CACHE_NAME_RE.fullmatch(entry.name)
        if match is None:
            raise WikisourceDiscoveryError(
                f"pinning cache contains unexpected file: {entry.name}"
            )
        request_kind = match.group(1)
        revision_id = int(match.group(2))
        identities.append((request_kind, revision_id))
        entries.append((pathlib.Path(entry.path), request_kind, revision_id))
    if len(identities) != len(set(identities)):
        for request_kind, revision_id in identities:
            if identities.count((request_kind, revision_id)) > 1:
                break
        else:  # pragma: no cover - guarded by the cardinality check.
            raise AssertionError("duplicate cache identity not found")
        raise WikisourceDiscoveryError(
            f"conflicting {request_kind} cache entries for revision "
            f"{revision_id}"
        )
    for cache_path, request_kind, revision_id in entries:
        try:
            raw = load_strict(cache_path)
        except (OSError, UnicodeError, StrictJSONError) as exc:
            raise WikisourceDiscoveryError(
                f"pinning cache {cache_path.name} is not strict JSON: {exc}"
            ) from exc
        _validate_cache_envelope(
            raw,
            path=cache_path,
            request_kind=request_kind,
            revision_id=revision_id,
            request=_request_parameters(request_kind, revision_id),
        )
    return resolved


def _cache_envelope(
    *,
    request_kind: str,
    request: Mapping[str, str],
    revision_id: int,
    response: object,
) -> dict[str, object]:
    response_sha = canonical_hash(response)
    payload: dict[str, object] = {
        "schema_version": PINNING_CACHE_SCHEMA_VERSION,
        "request_kind": request_kind,
        "request": dict(request),
        "revision_id": revision_id,
        "response_sha256": response_sha,
        "response": response,
    }
    return {**payload, "self_hash": canonical_hash(payload)}


def _same_strict_json_tree(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(  # type: ignore[arg-type]
            type(key) is str
            and _same_strict_json_tree(left[key], right[key])  # type: ignore[index]
            for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _same_strict_json_tree(a, b)
            for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return left == right


def _strict_json_clone(value: object, *, label: str) -> object:
    try:
        encoded = dumps_strict(value, sort_keys=True)
        decoded = loads_strict(encoded)
    except (StrictJSONError, TypeError, ValueError) as exc:
        raise WikisourceDiscoveryError(
            f"{label} is not strict JSON: {exc}"
        ) from exc
    if not _same_strict_json_tree(value, decoded):
        raise WikisourceDiscoveryError(
            f"{label} contains noncanonical JSON scalar/container types"
        )
    return decoded


def _validate_cache_envelope(
    value: object,
    *,
    path: pathlib.Path,
    request_kind: str,
    revision_id: int,
    request: Mapping[str, str],
) -> object:
    raw = _exact_object(value, _CACHE_KEYS, f"pinning cache {path.name}")
    if raw["schema_version"] != PINNING_CACHE_SCHEMA_VERSION:
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} is legacy or unsupported"
        )
    if raw["request_kind"] != request_kind:
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} request kind mismatch"
        )
    observed_request = raw["request"]
    if (
        type(observed_request) is not dict
        or any(
            type(key) is not str or type(item) is not str
            for key, item in observed_request.items()
        )
        or observed_request != dict(request)
    ):
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} exact request mismatch"
        )
    if (
        _exact_int(
            raw["revision_id"],
            f"pinning cache {path.name}.revision_id",
            minimum=1,
        )
        != revision_id
    ):
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} revision mismatch"
        )
    response_sha = _sha256(
        raw["response_sha256"],
        f"pinning cache {path.name}.response_sha256",
    )
    if canonical_hash(raw["response"]) != response_sha:
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} response hash mismatch"
        )
    expected_name = f"{request_kind}-{revision_id}-{response_sha}.json"
    if path.name != expected_name:
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} filename/hash mismatch"
        )
    recorded_self_hash = _sha256(
        raw["self_hash"],
        f"pinning cache {path.name}.self_hash",
    )
    payload = {key: item for key, item in raw.items() if key != "self_hash"}
    if canonical_hash(payload) != recorded_self_hash:
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} self_hash mismatch"
        )
    return raw["response"]


def _read_cached_response(
    cache: pathlib.Path,
    *,
    request_kind: str,
    revision_id: int,
    request: Mapping[str, str],
) -> object | None:
    prefix = f"{request_kind}-{revision_id}-"
    matches = sorted(
        path for path in cache.iterdir() if path.name.startswith(prefix)
    )
    if len(matches) > 1:
        raise WikisourceDiscoveryError(
            f"conflicting {request_kind} cache entries for revision "
            f"{revision_id}"
        )
    if not matches:
        return None
    path = matches[0]
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise WikisourceDiscoveryError(
            f"pinning cache entry is not a regular file: {path.name}"
        )
    try:
        raw = load_strict(path)
    except (OSError, UnicodeError, StrictJSONError) as exc:
        raise WikisourceDiscoveryError(
            f"pinning cache {path.name} is not strict JSON: {exc}"
        ) from exc
    return _validate_cache_envelope(
        raw,
        path=path,
        request_kind=request_kind,
        revision_id=revision_id,
        request=request,
    )


def _write_cached_response(
    cache: pathlib.Path,
    *,
    request_kind: str,
    revision_id: int,
    request: Mapping[str, str],
    response: object,
) -> object:
    response = _strict_json_clone(
        response,
        label=f"{request_kind} revision {revision_id} response",
    )
    envelope = _cache_envelope(
        request_kind=request_kind,
        request=request,
        revision_id=revision_id,
        response=response,
    )
    response_sha = str(envelope["response_sha256"])
    path = cache / f"{request_kind}-{revision_id}-{response_sha}.json"
    payload = (
        dumps_strict(envelope, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing = _read_cached_response(
            cache,
            request_kind=request_kind,
            revision_id=revision_id,
            request=request,
        )
        if canonical_hash(existing) != response_sha:
            raise WikisourceDiscoveryError(
                f"conflicting cache race for revision {revision_id}"
            )
        return existing
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return _validate_cache_envelope(
        envelope,
        path=path,
        request_kind=request_kind,
        revision_id=revision_id,
        request=request,
    )


def _cached_transport_call(
    cache: pathlib.Path,
    *,
    request_kind: str,
    revision_id: int,
    params: dict[str, str],
    transport: JSONTransport,
) -> object:
    cached = _read_cached_response(
        cache,
        request_kind=request_kind,
        revision_id=revision_id,
        request=params,
    )
    if cached is not None:
        return cached
    response = transport(params)
    return _write_cached_response(
        cache,
        request_kind=request_kind,
        revision_id=revision_id,
        request=params,
        response=response,
    )


def _single_revision(
    response: object,
    *,
    part: DiscoveryPart,
) -> tuple[bytes, int, str]:
    if type(response) is not dict:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} query response must be an object"
        )
    if "error" in response or "warnings" in response or "continue" in response:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} query response is incomplete or warned"
        )
    query = response.get("query")
    if type(query) is not dict:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} query response has no query object"
        )
    bad = query.get("badrevids")
    if bad not in (None, {}):
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} was reported missing"
        )
    pages = query.get("pages")
    if type(pages) is not list or len(pages) != 1:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} query must return exactly one page"
        )
    page = pages[0]
    if type(page) is not dict or page.get("missing") is not None:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} page is missing"
        )
    revisions = page.get("revisions")
    if type(revisions) is not list or len(revisions) != 1:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} query must return one revision"
        )
    revision = revisions[0]
    if type(revision) is not dict:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} record must be an object"
        )
    observed = {
        "title": page.get("title"),
        "page_id": page.get("pageid"),
        "revision_id": revision.get("revid"),
        "parent_id": revision.get("parentid"),
        "sha1": revision.get("sha1"),
        "timestamp": revision.get("timestamp"),
    }
    expected = {
        "title": part.resolved_title,
        "page_id": part.page_id,
        "revision_id": part.revision_id,
        "parent_id": part.revision_parent_id,
        "sha1": part.revision_sha1,
        "timestamp": part.revision_timestamp,
    }
    if any(
        type(observed[key]) is not type(expected[key])
        for key in expected
    ) or observed != expected:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} identity differs from discovery"
        )
    slots = revision.get("slots")
    main = slots.get("main") if type(slots) is dict else None
    wikitext = main.get("content") if type(main) is dict else None
    if type(wikitext) is not str or not wikitext:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} has no main-slot wikitext"
        )
    if _REDIRECT_WIKITEXT_RE.search(wikitext):
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} is a redirect, not prose"
        )
    wikitext_payload = wikitext.encode("utf-8")
    if _sha1_bytes(wikitext_payload) != part.revision_sha1:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} wikitext does not match "
            "MediaWiki SHA-1"
        )
    return (
        wikitext_payload,
        part.revision_parent_id,
        part.revision_timestamp,
    )


def _single_parse(response: object, *, part: DiscoveryPart) -> bytes:
    if type(response) is not dict:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} parse response must be an object"
        )
    if "error" in response or "warnings" in response or "continue" in response:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} parse response is incomplete or warned"
        )
    parsed = response.get("parse")
    if type(parsed) is not dict:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} parse response has no parse object"
        )
    observed = {
        "title": parsed.get("title"),
        "page_id": parsed.get("pageid"),
        "revision_id": parsed.get("revid"),
    }
    expected = {
        "title": part.resolved_title,
        "page_id": part.page_id,
        "revision_id": part.revision_id,
    }
    if any(
        type(observed[key]) is not type(expected[key])
        for key in expected
    ) or observed != expected:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} parsed identity differs from discovery"
        )
    rendered_html = parsed.get("text")
    if type(rendered_html) is not str or not rendered_html:
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} parse has no rendered HTML"
        )
    return rendered_html.encode("utf-8")


def _pin_part(
    part: DiscoveryPart,
    *,
    candidate_schema_version: str,
    cache: pathlib.Path,
    transport: JSONTransport,
) -> tuple[PinnedPartSpec, str]:
    query_response = _cached_transport_call(
        cache,
        request_kind="query",
        revision_id=part.revision_id,
        params=_request_parameters("query", part.revision_id),
        transport=transport,
    )
    wikitext_payload, _, _ = _single_revision(query_response, part=part)
    parse_response = _cached_transport_call(
        cache,
        request_kind="parse",
        revision_id=part.revision_id,
        params=_request_parameters("parse", part.revision_id),
        transport=transport,
    )
    rendered_payload = _single_parse(parse_response, part=part)
    try:
        rendered_html = rendered_payload.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - str encoded above.
        raise WikisourceDiscoveryError(
            f"revision {part.revision_id} rendered HTML is not UTF-8"
        ) from exc
    full_plain = extract_rendered_html(rendered_html)
    end_line = (
        len(full_plain.split("\n"))
        if part.body_end_line_exclusive is None
        else part.body_end_line_exclusive
    )
    selection = build_body_selection_v2(
        full_plain,
        start_line=part.body_start_line,
        end_line_exclusive=end_line,
        body_disposition=part.body_disposition,
    )
    pre_repair_plain = selection.selected_plain
    if part.source_repairs_v1 is not None:
        final_plain = apply_reviewed_source_repairs_v1(
            pre_repair_plain,
            part.source_repairs_v1,
            label=f"revision {part.revision_id}",
        )
    elif part.source_repair_v1 is not None:
        final_plain = apply_reviewed_source_repair_v1(
            pre_repair_plain,
            part.source_repair_v1,
            label=f"revision {part.revision_id}",
        )
    else:
        final_plain = pre_repair_plain
    if candidate_schema_version in {
        DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3,
        DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4,
    }:
        schema_version = (
            PINNED_WORK_SPEC_SCHEMA_VERSION_V3
            if candidate_schema_version
            == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3
            else PINNED_WORK_SPEC_SCHEMA_VERSION_V4
        )
        pre_payload = pre_repair_plain.encode("utf-8")
        final_payload = final_plain.encode("utf-8")
        if candidate_schema_version == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3:
            source_repair_fields: dict[str, object] = {
                "source_repair": (
                    None
                    if part.source_repair_v1 is None
                    else part.source_repair_v1.to_dict()
                )
            }
        else:
            if part.source_repairs_v1 is None:
                raise WikisourceDiscoveryError(
                    "v4 discovery part has no plural source repairs"
                )
            source_repair_fields = {
                "source_repairs": [
                    repair.to_dict()
                    for repair in part.source_repairs_v1
                ]
            }
        repair_fields: dict[str, object] = {
            "pre_repair_plain_byte_size": len(pre_payload),
            "pre_repair_plain_sha256": _sha256_bytes(pre_payload),
            "pre_repair_word_count": count_words(pre_repair_plain),
            **source_repair_fields,
            "plain_byte_size": len(final_payload),
            "plain_sha256": _sha256_bytes(final_payload),
            "word_count": count_words(final_plain),
        }
    else:
        schema_version = PINNED_WORK_SPEC_SCHEMA_VERSION_V2
        repair_fields = {}
    spec = PinnedPartSpec.from_dict(
        {
            "ordinal": part.ordinal,
            "requested_title": part.requested_title,
            "resolved_title": part.resolved_title,
            "redirect_chain": [
                hop.to_dict() for hop in part.redirect_chain
            ],
            "page_id": part.page_id,
            "revision_id": part.revision_id,
            "mediawiki_sha1": part.revision_sha1,
            "wikitext_sha256": _sha256_bytes(wikitext_payload),
            "rendered_html_sha256": _sha256_bytes(rendered_payload),
            **selection.to_part_fields(),
            **repair_fields,
        },
        schema_version=schema_version,
    )
    return spec, final_plain


def _pin_work(
    work: DiscoveryWork,
    *,
    candidate_schema_version: str,
    cache: pathlib.Path,
    transport: JSONTransport,
) -> tuple[PinnedWorkSpec, bytes]:
    pinned_parts: list[PinnedPartSpec] = []
    plain_parts: list[str] = []
    for part in work.parts:
        pinned, plain = _pin_part(
            part,
            candidate_schema_version=candidate_schema_version,
            cache=cache,
            transport=transport,
        )
        pinned_parts.append(pinned)
        plain_parts.append(plain)
    output = assemble_plain_parts(plain_parts)
    if candidate_schema_version == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3:
        pinned_schema_version = PINNED_WORK_SPEC_SCHEMA_VERSION_V3
    elif candidate_schema_version == DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4:
        pinned_schema_version = PINNED_WORK_SPEC_SCHEMA_VERSION_V4
    else:
        pinned_schema_version = PINNED_WORK_SPEC_SCHEMA_VERSION_V2
    payload: dict[str, object] = {
        "schema_version": pinned_schema_version,
        "work_id": work.work_id,
        "assembly_policy_version": ASSEMBLY_POLICY_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "residue_policy_version": RESIDUE_POLICY_VERSION,
        "word_count_policy_version": WORD_COUNT_POLICY_VERSION,
        "body_boundary_policy_version": BODY_BOUNDARY_POLICY_VERSION_V2,
        "parts": [part.to_dict() for part in pinned_parts],
        "output_relative_path": f"raw/{work.work_id}.txt",
        "output_byte_size": len(output),
        "output_sha256": _sha256_bytes(output),
        "word_count": count_words(output.decode("utf-8")),
    }
    if pinned_schema_version in {
        PINNED_WORK_SPEC_SCHEMA_VERSION_V3,
        PINNED_WORK_SPEC_SCHEMA_VERSION_V4,
    }:
        payload["source_repair_policy_version"] = (
            SOURCE_REPAIR_POLICY_VERSION_V1
        )
    spec = PinnedWorkSpec.from_dict(
        {**payload, "self_hash": canonical_hash(payload)}
    )
    return spec, output


@dataclasses.dataclass(frozen=True)
class PinnedDiscoveryCampaign:
    """Pinning result; assembled bytes are verification-only and unpublished."""

    candidate_hash: str
    campaign_spec: WikisourceCampaignSpec
    assembled_outputs: Mapping[str, bytes]


def pin_discovery_candidate(
    candidate: DiscoveryCandidate,
    *,
    cache_dir: str | os.PathLike[str],
    transport: JSONTransport,
) -> PinnedDiscoveryCampaign:
    """Promote one fully resolved candidate without publishing corpus files."""

    if type(candidate) is not DiscoveryCandidate:
        raise WikisourceDiscoveryError(
            "pinning requires exactly DiscoveryCandidate"
        )
    if not callable(transport):
        raise WikisourceDiscoveryError("transport must be callable")
    candidate.assert_ready()
    cache = _validate_cache_directory(pathlib.Path(cache_dir))
    works: list[PinnedWorkSpec] = []
    outputs: dict[str, bytes] = {}
    for work in candidate.works:
        if not work.include_in_corpus:
            continue
        pinned, output = _pin_work(
            work,
            candidate_schema_version=candidate.schema_version,
            cache=cache,
            transport=transport,
        )
        works.append(pinned)
        outputs[work.work_id] = output
    campaign = WikisourceCampaignSpec.build(works)
    return PinnedDiscoveryCampaign(
        candidate.candidate_hash,
        campaign,
        MappingProxyType(dict(outputs)),
    )


def write_campaign_spec_create_if_absent(
    spec: WikisourceCampaignSpec,
    path: str | os.PathLike[str],
) -> pathlib.Path:
    """Write an exact campaign spec once; accept only byte-identical replay."""

    if type(spec) is not WikisourceCampaignSpec:
        raise WikisourceDiscoveryError(
            "campaign output requires exactly WikisourceCampaignSpec"
        )
    spec.validate()
    target = pathlib.Path(path)
    _reject_symlink_components(target, label="campaign spec output")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        dumps_strict(spec.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if target.is_symlink() or not target.is_file():
            raise WikisourceDiscoveryError(
                "campaign spec output exists but is not a regular file"
            )
        if target.read_bytes() != payload:
            raise WikisourceDiscoveryError(
                "campaign spec output already exists with conflicting bytes"
            )
        return target
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return target


__all__ = [
    "BLOCKED_STATUS",
    "DISCOVERY_CANDIDATE_SCHEMA_VERSION",
    "DISCOVERY_CANDIDATE_SCHEMA_VERSION_V2",
    "DISCOVERY_CANDIDATE_SCHEMA_VERSION_V3",
    "DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4",
    "DiscoveryCandidate",
    "DiscoveryIssue",
    "DiscoveryPart",
    "DiscoverySource",
    "DiscoveryWork",
    "EXTERNAL_PROVIDER_WORK_STATUS",
    "PARSE_OLDID_STRATEGY",
    "PINNING_CACHE_SCHEMA_VERSION",
    "PinnedDiscoveryCampaign",
    "READY_STATUS",
    "RENDERED_OLDID_MATERIALIZATION",
    "RESOLVED_PART_STATUS",
    "SELECTED_WORK_STATUS",
    "SOURCE_QUALITY_REJECTED_WORK_STATUS",
    "SelectionContract",
    "RejectedWorkReceipt",
    "TRUSTED_API",
    "TRUSTED_SITE",
    "WikisourceDiscoveryError",
    "load_discovery_candidate",
    "loads_discovery_candidate",
    "pin_discovery_candidate",
    "write_campaign_spec_create_if_absent",
]
