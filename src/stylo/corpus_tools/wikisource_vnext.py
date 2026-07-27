"""Pinned, fail-closed Wikisource acquisition for whole-work corpus inputs.

This module is deliberately separate from :mod:`fetch_classics`, whose
best-effort title/subpage fetcher predates the LOBO-vNext corpus contract.
Scientific acquisition here has two boundaries:

* title resolution is discovery-only and returns an explicit redirect receipt;
* materialisation accepts only a self-hashed spec whose exact revisions and
  rendered/cleaned bytes are already pinned.

MediaWiki ``action=parse`` is used for materialisation because Wikisource prose
is commonly provided through ``<pages>`` and nested transclusions.  Fetching
raw wikitext and stripping markup would discard that prose.
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
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any, Protocol

from ..jsonio import (
    StrictJSONError,
    canonical_hash,
    dump_strict,
    load_strict,
    loads_strict,
)


API = "https://ru.wikisource.org/w/api.php"
PINNED_WORK_SPEC_SCHEMA_VERSION_V1 = "stylo.wikisource.pinned-work-spec.v1"
PINNED_WORK_SPEC_SCHEMA_VERSION_V2 = "stylo.wikisource.pinned-work-spec.v2"
PINNED_PART_RECEIPT_SCHEMA_VERSION_V1 = (
    "stylo.wikisource.pinned-part-receipt.v1"
)
PINNED_PART_RECEIPT_SCHEMA_VERSION_V2 = (
    "stylo.wikisource.pinned-part-receipt.v2"
)
WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V1 = (
    "stylo.wikisource.whole-work-receipt.v1"
)
WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2 = (
    "stylo.wikisource.whole-work-receipt.v2"
)
# These aliases deliberately remain v1.  Existing pinned specs and receipts
# are read-only compatibility inputs; new discovery code must opt into the
# explicit *_V2 constants and body-boundary builder below.
PINNED_WORK_SPEC_SCHEMA_VERSION = PINNED_WORK_SPEC_SCHEMA_VERSION_V1
PINNED_PART_RECEIPT_SCHEMA_VERSION = PINNED_PART_RECEIPT_SCHEMA_VERSION_V1
WHOLE_WORK_RECEIPT_SCHEMA_VERSION = WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V1
ASSEMBLY_POLICY_VERSION = "stylo.wikisource.ordered-parts-lf.v1"
EXTRACTION_POLICY_VERSION = "stylo.wikisource.rendered-html-text.v1"
RESIDUE_POLICY_VERSION = "stylo.wikisource.residue-reject.v1"
WORD_COUNT_POLICY_VERSION = "stylo.unicode-word-count.v1"
BODY_BOUNDARY_POLICY_VERSION_V2 = (
    "stylo.wikisource.exact-rendered-line-boundary.v2"
)
BODY_DISPOSITIONS = frozenset(
    {
        "whole_rendered_body",
        "strip_leading_apparatus",
        "strip_trailing_apparatus",
        "strip_both_apparatus",
    }
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_WIKI_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)
_REDIRECT_WIKITEXT_RE = re.compile(
    r"(?im)^\s*#\s*(?:redirect|перенаправление)\b"
)
_WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)
_MIN_CONTAINMENT_TOKENS = 20


class WikisourceAcquisitionError(ValueError):
    """A source response, pinned spec, or materialised work is unsafe."""


class JSONTransport(Protocol):
    """Minimal injectable Action API transport used by acquisition primitives."""

    def __call__(self, params: Mapping[str, str]) -> object: ...


def _exact_object(
    value: object,
    keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise WikisourceAcquisitionError(f"{label} must be an exact JSON object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise WikisourceAcquisitionError(
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
        raise WikisourceAcquisitionError(
            f"{label} must be an exact{qualifier} array"
        )
    return value


def _exact_str(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise WikisourceAcquisitionError(
            f"{label} must be an exact non-empty string"
        )
    if "\x00" in value:
        raise WikisourceAcquisitionError(f"{label} must not contain NUL")
    return value


def _title(value: object, label: str) -> str:
    title = _exact_str(value, label)
    if "\r" in title or "\n" in title:
        raise WikisourceAcquisitionError(
            f"{label} must be one MediaWiki title"
        )
    return title


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise WikisourceAcquisitionError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _sha256(value: object, label: str) -> str:
    digest = _exact_str(value, label)
    if _HEX64_RE.fullmatch(digest) is None:
        raise WikisourceAcquisitionError(
            f"{label} must be 64 lowercase hex characters"
        )
    return digest


def _wiki_sha1(value: object, label: str) -> str:
    digest = _exact_str(value, label)
    if _WIKI_SHA1_RE.fullmatch(digest) is None:
        raise WikisourceAcquisitionError(
            f"{label} must be 40 lowercase hex characters"
        )
    return digest


def _timestamp(value: object, label: str) -> str:
    timestamp = _exact_str(value, label)
    if _TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise WikisourceAcquisitionError(
            f"{label} must be a canonical UTC MediaWiki timestamp"
        )
    return timestamp


def _work_id(value: object, label: str = "work_id") -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise WikisourceAcquisitionError(
            f"{label} must use POSIX separators"
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WikisourceAcquisitionError(
            f"{label} must be a canonical author/work identifier"
        )
    return text


def _relative_path(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise WikisourceAcquisitionError(
            f"{label} must use POSIX separators"
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WikisourceAcquisitionError(
            f"{label} must be a canonical relative path"
        )
    return text


def _self_hashed_payload(raw: dict[str, Any], label: str) -> dict[str, Any]:
    recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    if canonical_hash(payload) != recorded:
        raise WikisourceAcquisitionError(f"{label} self_hash mismatch")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _title_key(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


@dataclasses.dataclass(frozen=True)
class RedirectHop:
    source_title: str
    target_title: str

    @classmethod
    def from_dict(cls, value: object) -> "RedirectHop":
        raw = _exact_object(
            value,
            {"from", "to"},
            "redirect hop",
        )
        return cls(
            _title(raw["from"], "redirect hop.from"),
            _title(raw["to"], "redirect hop.to"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"from": self.source_title, "to": self.target_title}


def _redirect_chain(
    value: object,
    *,
    requested_title: str,
    resolved_title: str,
    label: str,
) -> tuple[RedirectHop, ...]:
    rows = tuple(
        RedirectHop.from_dict(item)
        for item in _exact_list(value, label)
    )
    if not rows:
        if _title_key(requested_title) != _title_key(resolved_title):
            raise WikisourceAcquisitionError(
                f"{label} is empty but requested/resolved titles differ"
            )
        return rows
    if _title_key(rows[0].source_title) != _title_key(requested_title):
        raise WikisourceAcquisitionError(
            f"{label} does not start at requested_title"
        )
    for left, right in zip(rows, rows[1:], strict=False):
        if _title_key(left.target_title) != _title_key(right.source_title):
            raise WikisourceAcquisitionError(
                f"{label} is not contiguous"
            )
    if _title_key(rows[-1].target_title) != _title_key(resolved_title):
        raise WikisourceAcquisitionError(
            f"{label} does not end at resolved_title"
        )
    visited = [_title_key(rows[0].source_title)]
    visited.extend(_title_key(row.target_title) for row in rows)
    if len(visited) != len(set(visited)):
        raise WikisourceAcquisitionError(f"{label} contains a cycle")
    return rows


@dataclasses.dataclass(frozen=True)
class PageResolution:
    requested_title: str
    resolved_title: str
    redirect_chain: tuple[RedirectHop, ...]
    page_id: int
    revision_id: int
    mediawiki_sha1: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_title": self.requested_title,
            "resolved_title": self.resolved_title,
            "redirect_chain": [row.to_dict() for row in self.redirect_chain],
            "page_id": self.page_id,
            "revision_id": self.revision_id,
            "mediawiki_sha1": self.mediawiki_sha1,
        }


def _single_query_page(
    response: object,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(response) is not dict:
        raise WikisourceAcquisitionError(
            f"{label} response must be a JSON object"
        )
    query = response.get("query")
    if type(query) is not dict:
        raise WikisourceAcquisitionError(
            f"{label} response has no query object"
        )
    pages = query.get("pages")
    if type(pages) is not list or len(pages) != 1:
        raise WikisourceAcquisitionError(
            f"{label} response must contain exactly one page"
        )
    page = pages[0]
    if type(page) is not dict or page.get("missing") is not None:
        raise WikisourceAcquisitionError(f"{label} page is missing")
    revisions = page.get("revisions")
    if type(revisions) is not list or len(revisions) != 1:
        raise WikisourceAcquisitionError(
            f"{label} response must contain exactly one revision"
        )
    revision = revisions[0]
    if type(revision) is not dict:
        raise WikisourceAcquisitionError(
            f"{label} revision must be an object"
        )
    return page, revision


def resolve_page(title: str, *, transport: JSONTransport) -> PageResolution:
    """Resolve one live title into a draftable pinned revision.

    This discovery helper never materialises corpus bytes.  Its result must be
    copied into a self-hashed :class:`PinnedWorkSpec` before acquisition.
    """

    requested = _title(title, "title")
    response = transport(
        {
            "action": "query",
            "prop": "revisions",
            "rvprop": "ids|timestamp|sha1|content",
            "rvslots": "main",
            "titles": requested,
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        }
    )
    page, revision = _single_query_page(response, label="title resolution")
    resolved = _title(page.get("title"), "resolved page.title")
    raw_chain: object = response.get("query", {}).get("redirects", [])
    if type(raw_chain) is not list:
        raise WikisourceAcquisitionError(
            "title resolution redirects must be an array"
        )
    chain_value = [
        {"from": row.get("from"), "to": row.get("to")}
        if type(row) is dict
        else row
        for row in raw_chain
    ]
    chain = _redirect_chain(
        chain_value,
        requested_title=requested,
        resolved_title=resolved,
        label="title resolution.redirect_chain",
    )
    return PageResolution(
        requested,
        resolved,
        chain,
        _exact_int(page.get("pageid"), "resolved page.pageid", minimum=1),
        _exact_int(
            revision.get("revid"),
            "resolved revision.revid",
            minimum=1,
        ),
        _wiki_sha1(
            revision.get("sha1"),
            "resolved revision.sha1",
        ),
    )


_BODY_BOUNDARY_V2_KEYS = frozenset(
    {
        "full_plain_byte_size",
        "full_plain_sha256",
        "full_word_count",
        "full_line_count",
        "start_line",
        "end_line_exclusive",
        "first_selected_nonblank_line_sha256",
        "last_selected_nonblank_line_sha256",
        "body_disposition",
    }
)


def _body_disposition(value: object, label: str) -> str:
    disposition = _exact_str(value, label)
    if disposition not in BODY_DISPOSITIONS:
        raise WikisourceAcquisitionError(
            f"{label} must be one of {sorted(BODY_DISPOSITIONS)!r}"
        )
    return disposition


@dataclasses.dataclass(frozen=True)
class _PinnedBodyBoundaryV2:
    full_plain_byte_size: int
    full_plain_sha256: str
    full_word_count: int
    full_line_count: int
    start_line: int
    end_line_exclusive: int
    first_selected_nonblank_line_sha256: str
    last_selected_nonblank_line_sha256: str
    body_disposition: str

    @classmethod
    def from_part_dict(
        cls,
        raw: Mapping[str, object],
        *,
        label: str,
    ) -> "_PinnedBodyBoundaryV2":
        full_line_count = _exact_int(
            raw["full_line_count"],
            f"{label}.full_line_count",
            minimum=1,
        )
        start = _exact_int(
            raw["start_line"],
            f"{label}.start_line",
            minimum=0,
        )
        end = _exact_int(
            raw["end_line_exclusive"],
            f"{label}.end_line_exclusive",
            minimum=1,
        )
        if start >= end or end > full_line_count:
            raise WikisourceAcquisitionError(
                f"{label} body boundary must be a non-empty in-range span"
            )
        disposition = _body_disposition(
            raw["body_disposition"],
            f"{label}.body_disposition",
        )
        expected_span = {
            "whole_rendered_body": (
                start == 0 and end == full_line_count
            ),
            "strip_leading_apparatus": (
                start > 0 and end == full_line_count
            ),
            "strip_trailing_apparatus": (
                start == 0 and end < full_line_count
            ),
            "strip_both_apparatus": (
                start > 0 and end < full_line_count
            ),
        }[disposition]
        if not expected_span:
            raise WikisourceAcquisitionError(
                f"{label} body_disposition conflicts with body boundary"
            )
        return cls(
            _exact_int(
                raw["full_plain_byte_size"],
                f"{label}.full_plain_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["full_plain_sha256"],
                f"{label}.full_plain_sha256",
            ),
            _exact_int(
                raw["full_word_count"],
                f"{label}.full_word_count",
                minimum=1,
            ),
            full_line_count,
            start,
            end,
            _sha256(
                raw["first_selected_nonblank_line_sha256"],
                f"{label}.first_selected_nonblank_line_sha256",
            ),
            _sha256(
                raw["last_selected_nonblank_line_sha256"],
                f"{label}.last_selected_nonblank_line_sha256",
            ),
            disposition,
        )

    def to_part_fields(self) -> dict[str, object]:
        return {
            "full_plain_byte_size": self.full_plain_byte_size,
            "full_plain_sha256": self.full_plain_sha256,
            "full_word_count": self.full_word_count,
            "full_line_count": self.full_line_count,
            "start_line": self.start_line,
            "end_line_exclusive": self.end_line_exclusive,
            "first_selected_nonblank_line_sha256": (
                self.first_selected_nonblank_line_sha256
            ),
            "last_selected_nonblank_line_sha256": (
                self.last_selected_nonblank_line_sha256
            ),
            "body_disposition": self.body_disposition,
        }


@dataclasses.dataclass(frozen=True)
class PinnedPartSpec:
    ordinal: int
    requested_title: str
    resolved_title: str
    redirect_chain: tuple[RedirectHop, ...]
    page_id: int
    revision_id: int
    mediawiki_sha1: str
    wikitext_sha256: str
    rendered_html_sha256: str
    plain_byte_size: int
    plain_sha256: str
    word_count: int
    body_boundary_v2: _PinnedBodyBoundaryV2 | None = dataclasses.field(
        default=None,
        repr=False,
    )

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        schema_version: str | None = None,
    ) -> "PinnedPartSpec":
        base_keys = {
            "ordinal",
            "requested_title",
            "resolved_title",
            "redirect_chain",
            "page_id",
            "revision_id",
            "mediawiki_sha1",
            "wikitext_sha256",
            "rendered_html_sha256",
            "plain_byte_size",
            "plain_sha256",
            "word_count",
        }
        if schema_version is None:
            if type(value) is not dict:
                raise WikisourceAcquisitionError(
                    "pinned part spec must be an exact JSON object"
                )
            actual_keys = set(value)
            if actual_keys == base_keys:
                schema_version = PINNED_WORK_SPEC_SCHEMA_VERSION_V1
            elif actual_keys == base_keys | set(_BODY_BOUNDARY_V2_KEYS):
                schema_version = PINNED_WORK_SPEC_SCHEMA_VERSION_V2
            else:
                schema_version = PINNED_WORK_SPEC_SCHEMA_VERSION_V1
        if schema_version not in {
            PINNED_WORK_SPEC_SCHEMA_VERSION_V1,
            PINNED_WORK_SPEC_SCHEMA_VERSION_V2,
        }:
            raise WikisourceAcquisitionError(
                "pinned part spec schema is unsupported"
            )
        raw = _exact_object(
            value,
            (
                base_keys
                if schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V1
                else base_keys | set(_BODY_BOUNDARY_V2_KEYS)
            ),
            "pinned part spec",
        )
        requested = _title(
            raw["requested_title"],
            "pinned part spec.requested_title",
        )
        resolved = _title(
            raw["resolved_title"],
            "pinned part spec.resolved_title",
        )
        plain_size = _exact_int(
            raw["plain_byte_size"],
            "pinned part spec.plain_byte_size",
            minimum=1,
        )
        plain_sha256 = _sha256(
            raw["plain_sha256"],
            "pinned part spec.plain_sha256",
        )
        word_count = _exact_int(
            raw["word_count"],
            "pinned part spec.word_count",
            minimum=1,
        )
        boundary = (
            None
            if schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V1
            else _PinnedBodyBoundaryV2.from_part_dict(
                raw,
                label="pinned part spec",
            )
        )
        if boundary is not None:
            if (
                plain_size > boundary.full_plain_byte_size
                or word_count > boundary.full_word_count
            ):
                raise WikisourceAcquisitionError(
                    "pinned part spec selected identity exceeds full identity"
                )
            if boundary.body_disposition == "whole_rendered_body" and (
                plain_size != boundary.full_plain_byte_size
                or plain_sha256 != boundary.full_plain_sha256
                or word_count != boundary.full_word_count
            ):
                raise WikisourceAcquisitionError(
                    "pinned part spec whole-body selected/full identity mismatch"
                )
        return cls(
            _exact_int(
                raw["ordinal"],
                "pinned part spec.ordinal",
                minimum=0,
            ),
            requested,
            resolved,
            _redirect_chain(
                raw["redirect_chain"],
                requested_title=requested,
                resolved_title=resolved,
                label="pinned part spec.redirect_chain",
            ),
            _exact_int(
                raw["page_id"],
                "pinned part spec.page_id",
                minimum=1,
            ),
            _exact_int(
                raw["revision_id"],
                "pinned part spec.revision_id",
                minimum=1,
            ),
            _wiki_sha1(
                raw["mediawiki_sha1"],
                "pinned part spec.mediawiki_sha1",
            ),
            _sha256(
                raw["wikitext_sha256"],
                "pinned part spec.wikitext_sha256",
            ),
            _sha256(
                raw["rendered_html_sha256"],
                "pinned part spec.rendered_html_sha256",
            ),
            plain_size,
            plain_sha256,
            word_count,
            boundary,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ordinal": self.ordinal,
            "requested_title": self.requested_title,
            "resolved_title": self.resolved_title,
            "redirect_chain": [row.to_dict() for row in self.redirect_chain],
            "page_id": self.page_id,
            "revision_id": self.revision_id,
            "mediawiki_sha1": self.mediawiki_sha1,
            "wikitext_sha256": self.wikitext_sha256,
            "rendered_html_sha256": self.rendered_html_sha256,
            "plain_byte_size": self.plain_byte_size,
            "plain_sha256": self.plain_sha256,
            "word_count": self.word_count,
        }
        if self.body_boundary_v2 is not None:
            payload.update(self.body_boundary_v2.to_part_fields())
        return payload


@dataclasses.dataclass(frozen=True)
class PinnedWorkSpec:
    schema_version: str
    work_id: str
    assembly_policy_version: str
    extraction_policy_version: str
    residue_policy_version: str
    word_count_policy_version: str
    body_boundary_policy_version: str | None
    parts: tuple[PinnedPartSpec, ...]
    output_relative_path: str
    output_byte_size: int
    output_sha256: str
    word_count: int
    self_hash: str

    @classmethod
    def from_dict(cls, value: object) -> "PinnedWorkSpec":
        if type(value) is not dict:
            raise WikisourceAcquisitionError(
                "pinned work spec must be an exact JSON object"
            )
        schema_version = value.get("schema_version")
        if schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V1:
            versioned_keys: set[str] = set()
        elif schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V2:
            versioned_keys = {"body_boundary_policy_version"}
        else:
            raise WikisourceAcquisitionError(
                "pinned work spec is legacy or unsupported"
            )
        raw = _exact_object(
            value,
            {
                "schema_version",
                "work_id",
                "assembly_policy_version",
                "extraction_policy_version",
                "residue_policy_version",
                "word_count_policy_version",
                "parts",
                "output_relative_path",
                "output_byte_size",
                "output_sha256",
                "word_count",
                "self_hash",
            }
            | versioned_keys,
            "pinned work spec",
        )
        _self_hashed_payload(raw, "pinned work spec")
        policy_values = {
            "assembly_policy_version": ASSEMBLY_POLICY_VERSION,
            "extraction_policy_version": EXTRACTION_POLICY_VERSION,
            "residue_policy_version": RESIDUE_POLICY_VERSION,
            "word_count_policy_version": WORD_COUNT_POLICY_VERSION,
        }
        if schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V2:
            policy_values["body_boundary_policy_version"] = (
                BODY_BOUNDARY_POLICY_VERSION_V2
            )
        for key, expected in policy_values.items():
            if raw[key] != expected:
                raise WikisourceAcquisitionError(
                    f"pinned work spec {key} must be {expected!r}"
                )
        work = _work_id(raw["work_id"])
        parts = tuple(
            PinnedPartSpec.from_dict(item, schema_version=schema_version)
            for item in _exact_list(
                raw["parts"],
                "pinned work spec.parts",
                nonempty=True,
            )
        )
        if tuple(part.ordinal for part in parts) != tuple(range(len(parts))):
            raise WikisourceAcquisitionError(
                "pinned work spec parts must have contiguous manifest order"
            )
        if len({part.page_id for part in parts}) != len(parts):
            raise WikisourceAcquisitionError(
                "pinned work spec parts contain duplicate page ids"
            )
        if len({part.revision_id for part in parts}) != len(parts):
            raise WikisourceAcquisitionError(
                "pinned work spec parts contain duplicate revisions"
            )
        output_relative = _relative_path(
            raw["output_relative_path"],
            "pinned work spec.output_relative_path",
        )
        if output_relative != f"raw/{work}.txt":
            raise WikisourceAcquisitionError(
                "pinned work output path must be exactly raw/<work_id>.txt"
            )
        return cls(
            schema_version,
            work,
            ASSEMBLY_POLICY_VERSION,
            EXTRACTION_POLICY_VERSION,
            RESIDUE_POLICY_VERSION,
            WORD_COUNT_POLICY_VERSION,
            (
                BODY_BOUNDARY_POLICY_VERSION_V2
                if schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V2
                else None
            ),
            parts,
            output_relative,
            _exact_int(
                raw["output_byte_size"],
                "pinned work spec.output_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["output_sha256"],
                "pinned work spec.output_sha256",
            ),
            _exact_int(
                raw["word_count"],
                "pinned work spec.word_count",
                minimum=1,
            ),
            _sha256(raw["self_hash"], "pinned work spec.self_hash"),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "work_id": self.work_id,
            "assembly_policy_version": self.assembly_policy_version,
            "extraction_policy_version": self.extraction_policy_version,
            "residue_policy_version": self.residue_policy_version,
            "word_count_policy_version": self.word_count_policy_version,
            "parts": [part.to_dict() for part in self.parts],
            "output_relative_path": self.output_relative_path,
            "output_byte_size": self.output_byte_size,
            "output_sha256": self.output_sha256,
            "word_count": self.word_count,
            "self_hash": self.self_hash,
        }
        if self.schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V2:
            payload["body_boundary_policy_version"] = (
                self.body_boundary_policy_version
            )
        return payload

    def validate(self) -> "PinnedWorkSpec":
        if PinnedWorkSpec.from_dict(self.to_dict()) != self:
            raise WikisourceAcquisitionError("pinned work spec is noncanonical")
        return self


def loads_pinned_work_spec(text: str) -> PinnedWorkSpec:
    try:
        return PinnedWorkSpec.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise WikisourceAcquisitionError(f"pinned work spec: {exc}") from exc


def load_pinned_work_spec(
    path: str | os.PathLike[str],
) -> PinnedWorkSpec:
    try:
        return PinnedWorkSpec.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise WikisourceAcquisitionError(f"pinned work spec: {exc}") from exc


_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_SUPPRESSED_TAGS = frozenset({"script", "style", "noscript"})
_SUPPRESSED_CLASSES = frozenset(
    {
        "catlinks",
        "headertemplate",
        "mw-editsection",
        "noprint",
        "printfooter",
        "reference",
        "references",
        "searchaux",
        "ws-noexport",
    }
)
_SUPPRESSED_IDS = frozenset(
    {"catlinks", "headertemplate", "mw-normal-catlinks", "ws-footer"}
)


class _RenderedTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._suppressed_stack: list[str] = []

    @staticmethod
    def _is_suppressed(
        tag: str,
        attrs: Sequence[tuple[str, str | None]],
    ) -> bool:
        if tag in _SUPPRESSED_TAGS:
            return True
        values = {key: value or "" for key, value in attrs}
        classes = frozenset(values.get("class", "").split())
        return (
            values.get("id") in _SUPPRESSED_IDS
            or not classes.isdisjoint(_SUPPRESSED_CLASSES)
        )

    def _boundary(self) -> None:
        if self._pieces and self._pieces[-1] != "\n":
            self._pieces.append("\n")

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if self._suppressed_stack:
            if tag not in _VOID_TAGS:
                self._suppressed_stack.append(tag)
            return
        if self._is_suppressed(tag, attrs):
            if tag not in _VOID_TAGS:
                self._suppressed_stack.append(tag)
            return
        if tag in _BLOCK_TAGS:
            self._boundary()

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if not self._suppressed_stack and tag.lower() in _BLOCK_TAGS:
            self._boundary()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._suppressed_stack:
            for index in range(len(self._suppressed_stack) - 1, -1, -1):
                if self._suppressed_stack[index] == tag:
                    del self._suppressed_stack[index:]
                    return
            return
        if tag in _BLOCK_TAGS:
            self._boundary()

    def handle_data(self, data: str) -> None:
        if not self._suppressed_stack and data:
            self._pieces.append(data)

    def text(self) -> str:
        return "".join(self._pieces)


def count_words(text: str) -> int:
    """Count words with the versioned acquisition word-count policy."""

    if type(text) is not str:
        raise TypeError("text must be exactly str")
    return len(_WORD_RE.findall(text))


def extract_rendered_html(rendered_html: str) -> str:
    """Extract body text from formatversion=2 ``action=parse`` HTML."""

    if type(rendered_html) is not str or not rendered_html:
        raise WikisourceAcquisitionError(
            "rendered HTML must be an exact non-empty string"
        )
    parser = _RenderedTextExtractor()
    try:
        parser.feed(rendered_html)
        parser.close()
    except Exception as exc:  # HTMLParser errors are uncommon but fail closed.
        raise WikisourceAcquisitionError(
            f"cannot parse rendered Wikisource HTML: {exc}"
        ) from exc
    text = parser.text().replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\ufeff", "")
    )
    lines = []
    previous_blank = True
    for raw_line in text.split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw_line).strip()
        if line:
            lines.append(line)
            previous_blank = False
        elif not previous_blank:
            lines.append("")
            previous_blank = True
    while lines and not lines[-1]:
        lines.pop()
    plain = "\n".join(lines).strip()
    if not plain or count_words(plain) == 0:
        raise WikisourceAcquisitionError(
            "rendered Wikisource page produced no prose"
        )
    return plain


@dataclasses.dataclass(frozen=True)
class BodySelectionV2:
    """Deterministic v2 selection plus JSON fields for a pinned part."""

    selected_plain: str
    full_plain_byte_size: int
    full_plain_sha256: str
    full_word_count: int
    full_line_count: int
    start_line: int
    end_line_exclusive: int
    first_selected_nonblank_line_sha256: str
    last_selected_nonblank_line_sha256: str
    body_disposition: str
    plain_byte_size: int
    plain_sha256: str
    word_count: int

    def boundary(self) -> _PinnedBodyBoundaryV2:
        return _PinnedBodyBoundaryV2(
            self.full_plain_byte_size,
            self.full_plain_sha256,
            self.full_word_count,
            self.full_line_count,
            self.start_line,
            self.end_line_exclusive,
            self.first_selected_nonblank_line_sha256,
            self.last_selected_nonblank_line_sha256,
            self.body_disposition,
        )

    def to_part_fields(self) -> dict[str, object]:
        return {
            **self.boundary().to_part_fields(),
            "plain_byte_size": self.plain_byte_size,
            "plain_sha256": self.plain_sha256,
            "word_count": self.word_count,
        }


def build_body_selection_v2(
    full_plain: str,
    *,
    start_line: int,
    end_line_exclusive: int,
    body_disposition: str,
) -> BodySelectionV2:
    """Build the exact, reviewable v2 body boundary for one rendered part.

    This helper belongs to discovery/pinning.  Runtime materialisation applies
    only these exact LF line coordinates and never runs regex trimming or
    searches for an apparatus marker.
    """

    if (
        type(full_plain) is not str
        or not full_plain
        or full_plain != full_plain.strip()
        or "\r" in full_plain
    ):
        raise WikisourceAcquisitionError(
            "full rendered plain text must be exact non-empty stripped LF text"
        )
    lines = full_plain.split("\n")
    # Parse all boundary values through the same exact schema validation used
    # by persisted specs, including the disposition/span relationship.
    boundary_raw: dict[str, object] = {
        "full_plain_byte_size": len(full_plain.encode("utf-8")),
        "full_plain_sha256": _sha256_bytes(full_plain.encode("utf-8")),
        "full_word_count": count_words(full_plain),
        "full_line_count": len(lines),
        "start_line": start_line,
        "end_line_exclusive": end_line_exclusive,
        "first_selected_nonblank_line_sha256": "0" * 64,
        "last_selected_nonblank_line_sha256": "0" * 64,
        "body_disposition": body_disposition,
    }
    preliminary = _PinnedBodyBoundaryV2.from_part_dict(
        boundary_raw,
        label="body selection v2",
    )
    selected_lines = lines[
        preliminary.start_line : preliminary.end_line_exclusive
    ]
    nonblank = [line for line in selected_lines if line]
    if not nonblank:
        raise WikisourceAcquisitionError(
            "body selection v2 must contain a nonblank line"
        )
    selected = "\n".join(selected_lines)
    if not selected or selected != selected.strip():
        raise WikisourceAcquisitionError(
            "body selection v2 must start and end on nonblank lines"
        )
    first_anchor = _sha256_bytes(nonblank[0].encode("utf-8"))
    last_anchor = _sha256_bytes(nonblank[-1].encode("utf-8"))
    boundary_raw["first_selected_nonblank_line_sha256"] = first_anchor
    boundary_raw["last_selected_nonblank_line_sha256"] = last_anchor
    boundary = _PinnedBodyBoundaryV2.from_part_dict(
        boundary_raw,
        label="body selection v2",
    )
    selected_payload = selected.encode("utf-8")
    selected_word_count = count_words(selected)
    if selected_word_count == 0:
        raise WikisourceAcquisitionError(
            "body selection v2 must contain at least one word"
        )
    return BodySelectionV2(
        selected,
        boundary.full_plain_byte_size,
        boundary.full_plain_sha256,
        boundary.full_word_count,
        boundary.full_line_count,
        boundary.start_line,
        boundary.end_line_exclusive,
        boundary.first_selected_nonblank_line_sha256,
        boundary.last_selected_nonblank_line_sha256,
        boundary.body_disposition,
        len(selected_payload),
        _sha256_bytes(selected_payload),
        selected_word_count,
    )


def _apply_body_boundary_v2(
    full_plain: str,
    expected: _PinnedBodyBoundaryV2,
    *,
    label: str,
) -> BodySelectionV2:
    observed = build_body_selection_v2(
        full_plain,
        start_line=expected.start_line,
        end_line_exclusive=expected.end_line_exclusive,
        body_disposition=expected.body_disposition,
    )
    if observed.boundary() != expected:
        raise WikisourceAcquisitionError(
            f"{label} full plain text or body-boundary anchor drifted"
        )
    return observed


_RESIDUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "redirect",
        re.compile(r"(?iu)(?:#\s*)?\b(?:redirect|перенаправление)\b"),
    ),
    ("template_open", re.compile(r"\{\{")),
    ("template_close", re.compile(r"\}\}")),
    ("wikilink_open", re.compile(r"\[\[")),
    ("wikilink_close", re.compile(r"\]\]")),
    ("pages_tag", re.compile(r"(?iu)<\s*/?\s*pages\b")),
    ("ref_tag", re.compile(r"(?iu)<\s*/?\s*ref\b")),
    ("html_tag", re.compile(r"(?u)<\s*/?\s*[a-z][^>]*>")),
    ("nbsp_literal", re.compile(r"(?iu)(?:&(?:amp;)?nbsp;|\bnbsp\b)")),
    ("wikisource_chrome", re.compile(r"(?iu)Материал\s+из\s+Викитеки")),
    ("edit_chrome", re.compile(r"(?iu)\[\s*править\s*\]")),
    ("category_chrome", re.compile(r"(?iu)(?:Категория|Category)\s*:")),
    ("replacement_character", re.compile("\ufffd")),
    ("zero_width", re.compile("[\u200b\u200c\u200d\u2060]")),
    ("nonbreaking_space", re.compile("[\xa0\u202f]")),
    ("control_character", re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")),
)


def residue_kinds(text: str) -> tuple[str, ...]:
    """Return sorted acquisition-residue kinds present in final plain text."""

    if type(text) is not str:
        raise TypeError("text must be exactly str")
    return tuple(
        name for name, pattern in _RESIDUE_PATTERNS if pattern.search(text)
    )


def _assert_no_residue(text: str, *, label: str) -> None:
    kinds = residue_kinds(text)
    if kinds:
        raise WikisourceAcquisitionError(
            f"{label} contains rejected acquisition residue: {list(kinds)!r}"
        )


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(text))


def _contains_tokens(
    needle: tuple[str, ...],
    haystack: tuple[str, ...],
) -> bool:
    if len(needle) > len(haystack):
        return False
    # A NUL separator cannot occur in accepted plain text.
    return ("\x00".join(needle) + "\x00") in (
        "\x00".join(haystack) + "\x00"
    )


def _assert_distinct_parts(parts: Sequence[str]) -> None:
    digests = [_sha256_bytes(part.encode("utf-8")) for part in parts]
    if len(digests) != len(set(digests)):
        raise WikisourceAcquisitionError(
            "ordered work contains exact duplicate parts"
        )
    tokens = [_normalized_tokens(part) for part in parts]
    for left_index, left in enumerate(tokens):
        if len(left) < _MIN_CONTAINMENT_TOKENS:
            continue
        for right_index, right in enumerate(tokens):
            if left_index == right_index or len(left) > len(right):
                continue
            if _contains_tokens(left, right):
                raise WikisourceAcquisitionError(
                    "ordered work contains one part wholly inside another: "
                    f"{left_index} in {right_index}"
                )


def assemble_plain_parts(parts: Sequence[str]) -> bytes:
    """Assemble exact manifest-order parts under the frozen LF policy."""

    if type(parts) not in {list, tuple} or not parts:
        raise WikisourceAcquisitionError(
            "parts must be an exact non-empty list or tuple"
        )
    checked: list[str] = []
    for index, value in enumerate(parts):
        if type(value) is not str or not value or value != value.strip():
            raise WikisourceAcquisitionError(
                f"parts[{index}] must be exact non-empty stripped text"
            )
        _assert_no_residue(value, label=f"parts[{index}]")
        checked.append(value)
    _assert_distinct_parts(checked)
    return ("\n\n".join(checked) + "\n").encode("utf-8")


@dataclasses.dataclass(frozen=True)
class PinnedPartReceipt:
    schema_version: str
    ordinal: int
    requested_title: str
    resolved_title: str
    redirect_chain: tuple[RedirectHop, ...]
    page_id: int
    revision_id: int
    parent_revision_id: int
    timestamp: str
    mediawiki_sha1: str
    wikitext_byte_size: int
    wikitext_sha256: str
    rendered_html_byte_size: int
    rendered_html_sha256: str
    plain_byte_size: int
    plain_sha256: str
    word_count: int
    output_byte_start: int
    output_byte_end: int
    self_hash: str
    body_boundary_v2: _PinnedBodyBoundaryV2 | None = dataclasses.field(
        default=None,
        repr=False,
    )

    @classmethod
    def build(
        cls,
        *,
        spec: PinnedPartSpec,
        parent_revision_id: int,
        timestamp: str,
        wikitext_payload: bytes,
        rendered_html_payload: bytes,
        full_plain_payload: bytes,
        plain_payload: bytes,
        output_byte_start: int,
        output_byte_end: int,
    ) -> "PinnedPartReceipt":
        if spec.body_boundary_v2 is None:
            schema_version = PINNED_PART_RECEIPT_SCHEMA_VERSION_V1
            if full_plain_payload != plain_payload:
                raise WikisourceAcquisitionError(
                    "v1 part receipt cannot select a plain-text subset"
                )
            boundary = None
        else:
            schema_version = PINNED_PART_RECEIPT_SCHEMA_VERSION_V2
            try:
                full_plain = full_plain_payload.decode("utf-8")
                selected_plain = plain_payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WikisourceAcquisitionError(
                    "v2 part receipt plain text is not strict UTF-8"
                ) from exc
            observed = _apply_body_boundary_v2(
                full_plain,
                spec.body_boundary_v2,
                label=f"pinned part {spec.ordinal}",
            )
            if observed.selected_plain != selected_plain:
                raise WikisourceAcquisitionError(
                    "v2 part receipt selected plain text differs from boundary"
                )
            boundary = observed.boundary()
        payload: dict[str, object] = {
            "schema_version": schema_version,
            "ordinal": spec.ordinal,
            "requested_title": spec.requested_title,
            "resolved_title": spec.resolved_title,
            "redirect_chain": [
                row.to_dict() for row in spec.redirect_chain
            ],
            "page_id": spec.page_id,
            "revision_id": spec.revision_id,
            "parent_revision_id": parent_revision_id,
            "timestamp": timestamp,
            "mediawiki_sha1": spec.mediawiki_sha1,
            "wikitext_byte_size": len(wikitext_payload),
            "wikitext_sha256": _sha256_bytes(wikitext_payload),
            "rendered_html_byte_size": len(rendered_html_payload),
            "rendered_html_sha256": _sha256_bytes(rendered_html_payload),
            "plain_byte_size": len(plain_payload),
            "plain_sha256": _sha256_bytes(plain_payload),
            "word_count": count_words(plain_payload.decode("utf-8")),
            "output_byte_start": output_byte_start,
            "output_byte_end": output_byte_end,
        }
        if boundary is not None:
            payload.update(boundary.to_part_fields())
        return cls.from_dict({**payload, "self_hash": canonical_hash(payload)})

    @classmethod
    def from_dict(cls, value: object) -> "PinnedPartReceipt":
        if type(value) is not dict:
            raise WikisourceAcquisitionError(
                "pinned part receipt must be an exact JSON object"
            )
        schema_version = value.get("schema_version")
        if schema_version == PINNED_PART_RECEIPT_SCHEMA_VERSION_V1:
            versioned_keys: set[str] = set()
        elif schema_version == PINNED_PART_RECEIPT_SCHEMA_VERSION_V2:
            versioned_keys = set(_BODY_BOUNDARY_V2_KEYS)
        else:
            raise WikisourceAcquisitionError(
                "pinned part receipt is legacy or unsupported"
            )
        keys = {
            "schema_version",
            "ordinal",
            "requested_title",
            "resolved_title",
            "redirect_chain",
            "page_id",
            "revision_id",
            "parent_revision_id",
            "timestamp",
            "mediawiki_sha1",
            "wikitext_byte_size",
            "wikitext_sha256",
            "rendered_html_byte_size",
            "rendered_html_sha256",
            "plain_byte_size",
            "plain_sha256",
            "word_count",
            "output_byte_start",
            "output_byte_end",
            "self_hash",
        } | versioned_keys
        raw = _exact_object(value, keys, "pinned part receipt")
        _self_hashed_payload(raw, "pinned part receipt")
        requested = _title(
            raw["requested_title"],
            "pinned part receipt.requested_title",
        )
        resolved = _title(
            raw["resolved_title"],
            "pinned part receipt.resolved_title",
        )
        start = _exact_int(
            raw["output_byte_start"],
            "pinned part receipt.output_byte_start",
            minimum=0,
        )
        end = _exact_int(
            raw["output_byte_end"],
            "pinned part receipt.output_byte_end",
            minimum=1,
        )
        plain_size = _exact_int(
            raw["plain_byte_size"],
            "pinned part receipt.plain_byte_size",
            minimum=1,
        )
        if end - start != plain_size:
            raise WikisourceAcquisitionError(
                "pinned part receipt output span/size mismatch"
            )
        plain_sha256 = _sha256(
            raw["plain_sha256"],
            "pinned part receipt.plain_sha256",
        )
        word_count = _exact_int(
            raw["word_count"],
            "pinned part receipt.word_count",
            minimum=1,
        )
        boundary = (
            None
            if schema_version == PINNED_PART_RECEIPT_SCHEMA_VERSION_V1
            else _PinnedBodyBoundaryV2.from_part_dict(
                raw,
                label="pinned part receipt",
            )
        )
        if boundary is not None:
            if (
                plain_size > boundary.full_plain_byte_size
                or word_count > boundary.full_word_count
            ):
                raise WikisourceAcquisitionError(
                    "pinned part receipt selected identity exceeds full identity"
                )
            if boundary.body_disposition == "whole_rendered_body" and (
                plain_size != boundary.full_plain_byte_size
                or plain_sha256 != boundary.full_plain_sha256
                or word_count != boundary.full_word_count
            ):
                raise WikisourceAcquisitionError(
                    "pinned part receipt whole-body identity mismatch"
                )
        return cls(
            schema_version,
            _exact_int(raw["ordinal"], "pinned part receipt.ordinal"),
            requested,
            resolved,
            _redirect_chain(
                raw["redirect_chain"],
                requested_title=requested,
                resolved_title=resolved,
                label="pinned part receipt.redirect_chain",
            ),
            _exact_int(
                raw["page_id"],
                "pinned part receipt.page_id",
                minimum=1,
            ),
            _exact_int(
                raw["revision_id"],
                "pinned part receipt.revision_id",
                minimum=1,
            ),
            _exact_int(
                raw["parent_revision_id"],
                "pinned part receipt.parent_revision_id",
                minimum=0,
            ),
            _timestamp(
                raw["timestamp"],
                "pinned part receipt.timestamp",
            ),
            _wiki_sha1(
                raw["mediawiki_sha1"],
                "pinned part receipt.mediawiki_sha1",
            ),
            _exact_int(
                raw["wikitext_byte_size"],
                "pinned part receipt.wikitext_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["wikitext_sha256"],
                "pinned part receipt.wikitext_sha256",
            ),
            _exact_int(
                raw["rendered_html_byte_size"],
                "pinned part receipt.rendered_html_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["rendered_html_sha256"],
                "pinned part receipt.rendered_html_sha256",
            ),
            plain_size,
            plain_sha256,
            word_count,
            start,
            end,
            _sha256(
                raw["self_hash"],
                "pinned part receipt.self_hash",
            ),
            boundary,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "requested_title": self.requested_title,
            "resolved_title": self.resolved_title,
            "redirect_chain": [row.to_dict() for row in self.redirect_chain],
            "page_id": self.page_id,
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "timestamp": self.timestamp,
            "mediawiki_sha1": self.mediawiki_sha1,
            "wikitext_byte_size": self.wikitext_byte_size,
            "wikitext_sha256": self.wikitext_sha256,
            "rendered_html_byte_size": self.rendered_html_byte_size,
            "rendered_html_sha256": self.rendered_html_sha256,
            "plain_byte_size": self.plain_byte_size,
            "plain_sha256": self.plain_sha256,
            "word_count": self.word_count,
            "output_byte_start": self.output_byte_start,
            "output_byte_end": self.output_byte_end,
            "self_hash": self.self_hash,
        }
        if self.schema_version == PINNED_PART_RECEIPT_SCHEMA_VERSION_V2:
            if self.body_boundary_v2 is None:
                raise WikisourceAcquisitionError(
                    "v2 pinned part receipt has no body boundary"
                )
            payload.update(self.body_boundary_v2.to_part_fields())
        return payload


@dataclasses.dataclass(frozen=True)
class WholeWorkReceipt:
    schema_version: str
    work_id: str
    pinned_work_spec_sha256: str
    assembly_policy_version: str
    extraction_policy_version: str
    residue_policy_version: str
    word_count_policy_version: str
    body_boundary_policy_version: str | None
    parts: tuple[PinnedPartReceipt, ...]
    output_relative_path: str
    output_byte_size: int
    output_sha256: str
    word_count: int
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        spec: PinnedWorkSpec,
        parts: Sequence[PinnedPartReceipt],
        output_payload: bytes,
    ) -> "WholeWorkReceipt":
        schema_version = (
            WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2
            if spec.schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V2
            else WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V1
        )
        payload: dict[str, object] = {
            "schema_version": schema_version,
            "work_id": spec.work_id,
            "pinned_work_spec_sha256": spec.self_hash,
            "assembly_policy_version": ASSEMBLY_POLICY_VERSION,
            "extraction_policy_version": EXTRACTION_POLICY_VERSION,
            "residue_policy_version": RESIDUE_POLICY_VERSION,
            "word_count_policy_version": WORD_COUNT_POLICY_VERSION,
            "parts": [part.to_dict() for part in parts],
            "output_relative_path": spec.output_relative_path,
            "output_byte_size": len(output_payload),
            "output_sha256": _sha256_bytes(output_payload),
            "word_count": count_words(output_payload.decode("utf-8")),
        }
        if schema_version == WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2:
            payload["body_boundary_policy_version"] = (
                BODY_BOUNDARY_POLICY_VERSION_V2
            )
        return cls.from_dict({**payload, "self_hash": canonical_hash(payload)})

    @classmethod
    def from_dict(cls, value: object) -> "WholeWorkReceipt":
        if type(value) is not dict:
            raise WikisourceAcquisitionError(
                "whole-work receipt must be an exact JSON object"
            )
        schema_version = value.get("schema_version")
        if schema_version == WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V1:
            versioned_keys: set[str] = set()
            expected_part_schema = PINNED_PART_RECEIPT_SCHEMA_VERSION_V1
        elif schema_version == WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2:
            versioned_keys = {"body_boundary_policy_version"}
            expected_part_schema = PINNED_PART_RECEIPT_SCHEMA_VERSION_V2
        else:
            raise WikisourceAcquisitionError(
                "whole-work receipt is legacy or unsupported"
            )
        raw = _exact_object(
            value,
            {
                "schema_version",
                "work_id",
                "pinned_work_spec_sha256",
                "assembly_policy_version",
                "extraction_policy_version",
                "residue_policy_version",
                "word_count_policy_version",
                "parts",
                "output_relative_path",
                "output_byte_size",
                "output_sha256",
                "word_count",
                "self_hash",
            }
            | versioned_keys,
            "whole-work receipt",
        )
        _self_hashed_payload(raw, "whole-work receipt")
        policies = {
            "assembly_policy_version": ASSEMBLY_POLICY_VERSION,
            "extraction_policy_version": EXTRACTION_POLICY_VERSION,
            "residue_policy_version": RESIDUE_POLICY_VERSION,
            "word_count_policy_version": WORD_COUNT_POLICY_VERSION,
        }
        if schema_version == WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2:
            policies["body_boundary_policy_version"] = (
                BODY_BOUNDARY_POLICY_VERSION_V2
            )
        for key, expected in policies.items():
            if raw[key] != expected:
                raise WikisourceAcquisitionError(
                    f"whole-work receipt {key} must be {expected!r}"
                )
        work = _work_id(raw["work_id"], "whole-work receipt.work_id")
        parts = tuple(
            PinnedPartReceipt.from_dict(item)
            for item in _exact_list(
                raw["parts"],
                "whole-work receipt.parts",
                nonempty=True,
            )
        )
        if any(part.schema_version != expected_part_schema for part in parts):
            raise WikisourceAcquisitionError(
                "whole-work receipt contains a mismatched part schema"
            )
        if tuple(part.ordinal for part in parts) != tuple(range(len(parts))):
            raise WikisourceAcquisitionError(
                "whole-work receipt parts are not in contiguous manifest order"
            )
        cursor = 0
        for index, part in enumerate(parts):
            expected_start = cursor + (2 if index else 0)
            if part.output_byte_start != expected_start:
                raise WikisourceAcquisitionError(
                    "whole-work receipt part spans are noncanonical"
                )
            cursor = part.output_byte_end
        output_size = _exact_int(
            raw["output_byte_size"],
            "whole-work receipt.output_byte_size",
            minimum=1,
        )
        if cursor + 1 != output_size:
            raise WikisourceAcquisitionError(
                "whole-work receipt final newline/size mismatch"
            )
        relative = _relative_path(
            raw["output_relative_path"],
            "whole-work receipt.output_relative_path",
        )
        if relative != f"raw/{work}.txt":
            raise WikisourceAcquisitionError(
                "whole-work receipt output path is noncanonical"
            )
        return cls(
            schema_version,
            work,
            _sha256(
                raw["pinned_work_spec_sha256"],
                "whole-work receipt.pinned_work_spec_sha256",
            ),
            ASSEMBLY_POLICY_VERSION,
            EXTRACTION_POLICY_VERSION,
            RESIDUE_POLICY_VERSION,
            WORD_COUNT_POLICY_VERSION,
            (
                BODY_BOUNDARY_POLICY_VERSION_V2
                if schema_version == WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2
                else None
            ),
            parts,
            relative,
            output_size,
            _sha256(
                raw["output_sha256"],
                "whole-work receipt.output_sha256",
            ),
            _exact_int(
                raw["word_count"],
                "whole-work receipt.word_count",
                minimum=1,
            ),
            _sha256(raw["self_hash"], "whole-work receipt.self_hash"),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "work_id": self.work_id,
            "pinned_work_spec_sha256": self.pinned_work_spec_sha256,
            "assembly_policy_version": self.assembly_policy_version,
            "extraction_policy_version": self.extraction_policy_version,
            "residue_policy_version": self.residue_policy_version,
            "word_count_policy_version": self.word_count_policy_version,
            "parts": [part.to_dict() for part in self.parts],
            "output_relative_path": self.output_relative_path,
            "output_byte_size": self.output_byte_size,
            "output_sha256": self.output_sha256,
            "word_count": self.word_count,
            "self_hash": self.self_hash,
        }
        if self.schema_version == WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2:
            payload["body_boundary_policy_version"] = (
                self.body_boundary_policy_version
            )
        return payload

    def validate_for(
        self,
        spec: PinnedWorkSpec,
        output_payload: bytes,
    ) -> "WholeWorkReceipt":
        if type(spec) is not PinnedWorkSpec:
            raise WikisourceAcquisitionError(
                "receipt validation requires exactly PinnedWorkSpec"
            )
        spec.validate()
        expected_receipt_schema = (
            WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2
            if spec.schema_version == PINNED_WORK_SPEC_SCHEMA_VERSION_V2
            else WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V1
        )
        if (
            self.schema_version != expected_receipt_schema
            or self.work_id != spec.work_id
            or self.pinned_work_spec_sha256 != spec.self_hash
            or self.output_relative_path != spec.output_relative_path
            or self.output_byte_size != len(output_payload)
            or self.output_sha256 != _sha256_bytes(output_payload)
            or self.word_count != count_words(output_payload.decode("utf-8"))
        ):
            raise WikisourceAcquisitionError(
                "whole-work receipt/spec/output mismatch"
            )
        if (
            self.output_byte_size != spec.output_byte_size
            or self.output_sha256 != spec.output_sha256
            or self.word_count != spec.word_count
            or len(self.parts) != len(spec.parts)
        ):
            raise WikisourceAcquisitionError(
                "whole-work receipt differs from pinned expectations"
            )
        for observed, expected in zip(self.parts, spec.parts, strict=True):
            if (
                observed.ordinal != expected.ordinal
                or observed.requested_title != expected.requested_title
                or observed.resolved_title != expected.resolved_title
                or observed.redirect_chain != expected.redirect_chain
                or observed.page_id != expected.page_id
                or observed.revision_id != expected.revision_id
                or observed.mediawiki_sha1 != expected.mediawiki_sha1
                or observed.wikitext_sha256 != expected.wikitext_sha256
                or observed.rendered_html_sha256
                != expected.rendered_html_sha256
                or observed.plain_byte_size != expected.plain_byte_size
                or observed.plain_sha256 != expected.plain_sha256
                or observed.word_count != expected.word_count
                or observed.body_boundary_v2
                != expected.body_boundary_v2
            ):
                raise WikisourceAcquisitionError(
                    "whole-work part receipt differs from pinned expectations"
                )
            plain = output_payload[
                observed.output_byte_start : observed.output_byte_end
            ]
            if (
                len(plain) != observed.plain_byte_size
                or _sha256_bytes(plain) != observed.plain_sha256
            ):
                raise WikisourceAcquisitionError(
                    "whole-work output span differs from part receipt"
                )
        _assert_no_residue(
            output_payload.decode("utf-8"),
            label="whole-work output",
        )
        if WholeWorkReceipt.from_dict(self.to_dict()) != self:
            raise WikisourceAcquisitionError(
                "whole-work receipt is noncanonical"
            )
        return self


def loads_whole_work_receipt(text: str) -> WholeWorkReceipt:
    try:
        return WholeWorkReceipt.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise WikisourceAcquisitionError(
            f"whole-work receipt: {exc}"
        ) from exc


def load_whole_work_receipt(
    path: str | os.PathLike[str],
) -> WholeWorkReceipt:
    try:
        return WholeWorkReceipt.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise WikisourceAcquisitionError(
            f"whole-work receipt: {exc}"
        ) from exc


@dataclasses.dataclass(frozen=True)
class _FetchedPart:
    spec: PinnedPartSpec
    parent_revision_id: int
    timestamp: str
    wikitext_payload: bytes
    rendered_html_payload: bytes
    full_plain: str
    plain: str


def _fetch_pinned_part(
    part: PinnedPartSpec,
    *,
    transport: JSONTransport,
) -> _FetchedPart:
    query_response = transport(
        {
            "action": "query",
            "prop": "revisions",
            "rvprop": "ids|timestamp|sha1|content",
            "rvslots": "main",
            "revids": str(part.revision_id),
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        }
    )
    page, revision = _single_query_page(
        query_response,
        label=f"pinned part {part.ordinal} query",
    )
    page_id = _exact_int(
        page.get("pageid"),
        f"pinned part {part.ordinal} pageid",
        minimum=1,
    )
    returned_title = _title(
        page.get("title"),
        f"pinned part {part.ordinal} title",
    )
    revision_id = _exact_int(
        revision.get("revid"),
        f"pinned part {part.ordinal} revid",
        minimum=1,
    )
    parent_id = _exact_int(
        revision.get("parentid"),
        f"pinned part {part.ordinal} parentid",
        minimum=0,
    )
    timestamp = _timestamp(
        revision.get("timestamp"),
        f"pinned part {part.ordinal} timestamp",
    )
    wiki_sha1 = _wiki_sha1(
        revision.get("sha1"),
        f"pinned part {part.ordinal} sha1",
    )
    slots = revision.get("slots")
    main = slots.get("main") if type(slots) is dict else None
    wikitext = main.get("content") if type(main) is dict else None
    if type(wikitext) is not str or not wikitext:
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} has no exact main-slot wikitext"
        )
    if _REDIRECT_WIKITEXT_RE.search(wikitext):
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} points at a redirect revision"
        )
    wikitext_payload = wikitext.encode("utf-8")
    observed = {
        "page_id": page_id,
        "resolved_title": returned_title,
        "revision_id": revision_id,
        "mediawiki_sha1": wiki_sha1,
        "wikitext_sha256": _sha256_bytes(wikitext_payload),
    }
    expected = {
        "page_id": part.page_id,
        "resolved_title": part.resolved_title,
        "revision_id": part.revision_id,
        "mediawiki_sha1": part.mediawiki_sha1,
        "wikitext_sha256": part.wikitext_sha256,
    }
    if observed != expected:
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} revision identity mismatch"
        )

    parse_response = transport(
        {
            "action": "parse",
            "oldid": str(part.revision_id),
            "prop": "text|revid|title",
            "disableeditsection": "1",
            "disablelimitreport": "1",
            "format": "json",
            "formatversion": "2",
            "maxlag": "5",
        }
    )
    if type(parse_response) is not dict or type(parse_response.get("parse")) is not dict:
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} parse response is malformed"
        )
    parsed = parse_response["parse"]
    parsed_title = _title(
        parsed.get("title"),
        f"pinned part {part.ordinal} parsed title",
    )
    parsed_page_id = _exact_int(
        parsed.get("pageid"),
        f"pinned part {part.ordinal} parsed pageid",
        minimum=1,
    )
    parsed_revision = _exact_int(
        parsed.get("revid"),
        f"pinned part {part.ordinal} parsed revid",
        minimum=1,
    )
    rendered_html = parsed.get("text")
    if type(rendered_html) is not str or not rendered_html:
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} parse response has no exact HTML"
        )
    if (
        parsed_title != part.resolved_title
        or parsed_page_id != part.page_id
        or parsed_revision != part.revision_id
    ):
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} parsed identity mismatch"
        )
    rendered_payload = rendered_html.encode("utf-8")
    if _sha256_bytes(rendered_payload) != part.rendered_html_sha256:
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} rendered HTML drifted"
        )
    full_plain = extract_rendered_html(rendered_html)
    if part.body_boundary_v2 is None:
        plain = full_plain
    else:
        plain = _apply_body_boundary_v2(
            full_plain,
            part.body_boundary_v2,
            label=f"pinned part {part.ordinal}",
        ).selected_plain
    _assert_no_residue(plain, label=f"pinned part {part.ordinal}")
    plain_payload = plain.encode("utf-8")
    if (
        len(plain_payload) != part.plain_byte_size
        or _sha256_bytes(plain_payload) != part.plain_sha256
        or count_words(plain) != part.word_count
    ):
        raise WikisourceAcquisitionError(
            f"pinned part {part.ordinal} extracted plain text drifted"
        )
    return _FetchedPart(
        part,
        parent_id,
        timestamp,
        wikitext_payload,
        rendered_payload,
        full_plain,
        plain,
    )


@dataclasses.dataclass(frozen=True)
class MaterializedWork:
    root: pathlib.Path
    output_path: pathlib.Path
    receipt: WholeWorkReceipt
    resumed: bool


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise WikisourceAcquisitionError(
                f"{label} must not contain symlink components: {component}"
            )


@contextlib.contextmanager
def _publication_lock(parent: pathlib.Path):
    lock = parent / ".wikisource-vnext-acquisition.lock"
    if lock.is_symlink():
        raise WikisourceAcquisitionError(
            "acquisition publication lock must not be a symlink"
        )
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _inventory_files(root: pathlib.Path) -> set[str]:
    files: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        for entry in os.scandir(directory):
            metadata = entry.stat(follow_symlinks=False)
            path = pathlib.Path(entry.path)
            if stat.S_ISLNK(metadata.st_mode):
                raise WikisourceAcquisitionError(
                    f"symlink rejected in materialized work: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(path.relative_to(root).as_posix())
            else:
                raise WikisourceAcquisitionError(
                    f"special file rejected in materialized work: {path}"
                )
    return files


def _load_existing(
    root: pathlib.Path,
    spec: PinnedWorkSpec,
) -> MaterializedWork:
    if root.is_symlink() or not root.is_dir():
        raise WikisourceAcquisitionError(
            f"materialized work namespace is not a real directory: {root}"
        )
    expected_files = {spec.output_relative_path, "receipt.json"}
    if _inventory_files(root) != expected_files:
        raise WikisourceAcquisitionError(
            "materialized work has missing or extra files"
        )
    output = root.joinpath(*PurePosixPath(spec.output_relative_path).parts)
    receipt_path = root / "receipt.json"
    if output.is_symlink() or not output.is_file():
        raise WikisourceAcquisitionError(
            "materialized whole-work output is missing or unsafe"
        )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise WikisourceAcquisitionError(
            "materialized whole-work receipt is missing or unsafe"
        )
    payload = output.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WikisourceAcquisitionError(
            "materialized whole-work output is not strict UTF-8"
        ) from exc
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise WikisourceAcquisitionError(
            "materialized whole-work output has noncanonical final newline"
        )
    receipt = load_whole_work_receipt(receipt_path)
    receipt.validate_for(spec, payload)
    return MaterializedWork(root, output, receipt, True)


def materialize_pinned_work(
    spec: PinnedWorkSpec,
    *,
    output_parent: str | os.PathLike[str],
    transport: JSONTransport,
) -> MaterializedWork:
    """Fetch, validate, and immutably publish one exact whole-work generation."""

    if type(spec) is not PinnedWorkSpec:
        raise WikisourceAcquisitionError(
            "materialization requires exactly PinnedWorkSpec"
        )
    spec.validate()
    parent = pathlib.Path(output_parent)
    _reject_symlink_components(parent, label="acquisition output parent")
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise WikisourceAcquisitionError(
            "acquisition output parent must be a real directory"
        )
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    target = parent / spec.self_hash

    with _publication_lock(parent):
        if target.exists() or target.is_symlink():
            return _load_existing(target, spec)

    fetched = [
        _fetch_pinned_part(part, transport=transport)
        for part in spec.parts
    ]
    plain_parts = [row.plain for row in fetched]
    output_payload = assemble_plain_parts(plain_parts)
    if (
        len(output_payload) != spec.output_byte_size
        or _sha256_bytes(output_payload) != spec.output_sha256
        or count_words(output_payload.decode("utf-8")) != spec.word_count
    ):
        raise WikisourceAcquisitionError(
            "assembled whole-work output differs from pinned expectations"
        )

    receipts: list[PinnedPartReceipt] = []
    cursor = 0
    for index, row in enumerate(fetched):
        if index:
            cursor += 2
        plain_payload = row.plain.encode("utf-8")
        start = cursor
        end = start + len(plain_payload)
        receipts.append(
            PinnedPartReceipt.build(
                spec=row.spec,
                parent_revision_id=row.parent_revision_id,
                timestamp=row.timestamp,
                wikitext_payload=row.wikitext_payload,
                rendered_html_payload=row.rendered_html_payload,
                full_plain_payload=row.full_plain.encode("utf-8"),
                plain_payload=plain_payload,
                output_byte_start=start,
                output_byte_end=end,
            )
        )
        cursor = end
    receipt = WholeWorkReceipt.build(
        spec=spec,
        parts=receipts,
        output_payload=output_payload,
    )
    receipt.validate_for(spec, output_payload)

    stage = pathlib.Path(
        tempfile.mkdtemp(prefix=".wikisource-work.", dir=parent)
    )
    try:
        output = stage.joinpath(
            *PurePosixPath(spec.output_relative_path).parts
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(output_payload)
        dump_strict(
            receipt.to_dict(),
            stage / "receipt.json",
            sort_keys=True,
            trailing_newline=True,
        )
        if _inventory_files(stage) != {
            spec.output_relative_path,
            "receipt.json",
        }:
            raise WikisourceAcquisitionError(
                "staged whole-work inventory is noncanonical"
            )
        with _publication_lock(parent):
            if target.exists() or target.is_symlink():
                existing = _load_existing(target, spec)
                return existing
            os.rename(stage, target)
        return MaterializedWork(
            target,
            target.joinpath(
                *PurePosixPath(spec.output_relative_path).parts
            ),
            receipt,
            False,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


__all__ = [
    "API",
    "ASSEMBLY_POLICY_VERSION",
    "BODY_BOUNDARY_POLICY_VERSION_V2",
    "BODY_DISPOSITIONS",
    "BodySelectionV2",
    "EXTRACTION_POLICY_VERSION",
    "MaterializedWork",
    "PINNED_PART_RECEIPT_SCHEMA_VERSION",
    "PINNED_PART_RECEIPT_SCHEMA_VERSION_V1",
    "PINNED_PART_RECEIPT_SCHEMA_VERSION_V2",
    "PINNED_WORK_SPEC_SCHEMA_VERSION",
    "PINNED_WORK_SPEC_SCHEMA_VERSION_V1",
    "PINNED_WORK_SPEC_SCHEMA_VERSION_V2",
    "PageResolution",
    "PinnedPartReceipt",
    "PinnedPartSpec",
    "PinnedWorkSpec",
    "RESIDUE_POLICY_VERSION",
    "RedirectHop",
    "WHOLE_WORK_RECEIPT_SCHEMA_VERSION",
    "WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V1",
    "WHOLE_WORK_RECEIPT_SCHEMA_VERSION_V2",
    "WORD_COUNT_POLICY_VERSION",
    "WholeWorkReceipt",
    "WikisourceAcquisitionError",
    "assemble_plain_parts",
    "build_body_selection_v2",
    "count_words",
    "extract_rendered_html",
    "load_pinned_work_spec",
    "load_whole_work_receipt",
    "loads_pinned_work_spec",
    "loads_whole_work_receipt",
    "materialize_pinned_work",
    "residue_kinds",
    "resolve_page",
]
