"""Deterministic content-quality audit for newly acquired corpus texts.

Acquisition hashes answer only whether the bytes changed.  This module checks
whether those bytes look like usable whole literary works: canonical UTF-8,
non-trivial prose, no transport markup or publishing colophon, no obvious
whole-text duplication, and no exact/near-contained work copied into another
work.

The audit is intentionally independent of model fitting and representation
caches.  It accepts an exact in-memory work/bytes inventory and returns a
path-independent, self-hashed report.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any

from ..domain.corpus_identity import (
    CONTENT_OVERLAP_POLICY_VERSION,
    find_cross_work_content_overlaps,
)
from ..jsonio import canonical_hash
from .wikisource_vnext import count_words


TEXT_QUALITY_AUDIT_SCHEMA_VERSION = "stylo.corpus-text-quality-audit.v1"
TEXT_QUALITY_POLICY_VERSION = "stylo.corpus-text-quality-policy.v1"
DEFAULT_MINIMUM_WORDS = 200
DEFAULT_CONTAINMENT_THRESHOLD = Fraction(9, 10)
DEFAULT_MINIMUM_SHINGLES = 20
DEFAULT_SAMPLE_SIZE = 64

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
_TRANSPORT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nul", re.compile("\x00")),
    ("unicode_replacement_character", re.compile("\ufffd")),
    ("html_tag", re.compile(r"</?[A-Za-z][^>\n]{0,200}>")),
    (
        "html_entity",
        re.compile(
            r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9a-f]+);",
            re.IGNORECASE,
        ),
    ),
    ("wiki_link", re.compile(r"\[\[|\]\]")),
    ("wiki_template", re.compile(r"\{\{|\}\}")),
    ("wiki_bold_italic", re.compile(r"'{3,5}")),
    (
        "wiki_category",
        re.compile(r"(?:Категория|Category)\s*:", re.IGNORECASE),
    ),
    (
        "wiki_magic_word",
        re.compile(r"__[A-ZА-Я][A-ZА-Я0-9_]+__"),
    ),
    (
        "redirect_directive",
        re.compile(
            r"^\s*#\s*(?:redirect|перенаправление)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rendered_redirect_notice",
        re.compile(r"\bперенаправлени[ея]\b", re.IGNORECASE),
    ),
    ("web_url", re.compile(r"https?://|www\.", re.IGNORECASE)),
)
_TAIL_APPARATUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "print_run_colophon",
        re.compile(r"\b(?:тираж|уч\.-изд\.\s*л\.|заказ\s+\d+)\b", re.I),
    ),
    (
        "publisher_colophon",
        re.compile(
            r"\b(?:издательств[оа]|книжная\s+фабрика|"
            r"госкомитет[а-я]*\s+.*издательств)\b",
            re.I,
        ),
    ),
    ("isbn_colophon", re.compile(r"\bISBN(?:-1[03])?\b", re.I)),
    (
        "digital_source_colophon",
        re.compile(
            r"\b(?:OCR|proofread|project\s+gutenberg|"
            r"электронн(?:ая|ое)\s+(?:библиотека|издание))\b",
            re.I,
        ),
    ),
    (
        "source_navigation",
        re.compile(r"^\s*(?:См\.\s+также|Викитека|Wikisource)\s*$", re.I),
    ),
)
_SOURCE_SEPARATOR_RE = re.compile(r"^\s*[-_=]{20,}\s*$")
_CONTROL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")


class CorpusTextQualityError(ValueError):
    """The literal corpus bytes fail the versioned content-quality policy."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _HEX64_RE.fullmatch(value) is None:
        raise CorpusTextQualityError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _line_sha256(line: str) -> str:
    return _sha256_bytes(line.encode("utf-8"))


def _work_id(value: object, label: str = "work_id") -> str:
    if type(value) is not str or not value or "\\" in value:
        raise CorpusTextQualityError(
            f"{label} must be a non-empty canonical POSIX work id"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CorpusTextQualityError(
            f"{label} must be a canonical author/work identifier"
        )
    return value


def _exact_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise CorpusTextQualityError(
            f"{label} must be an exact positive integer"
        )
    return value


def _finding(
    *,
    kind: str,
    line_number: int,
    line: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "line_number": line_number,
        "line_sha256": _line_sha256(line),
        "excerpt": line[:240],
    }


def _line_findings(
    lines: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    transport: list[dict[str, object]] = []
    tail_apparatus: list[dict[str, object]] = []
    tail_start = max(0, len(lines) - 100)
    for index, line in enumerate(lines):
        line_number = index + 1
        for kind, pattern in _TRANSPORT_PATTERNS:
            if pattern.search(line):
                transport.append(
                    _finding(
                        kind=kind,
                        line_number=line_number,
                        line=line,
                    )
                )
        if _SOURCE_SEPARATOR_RE.fullmatch(line):
            transport.append(
                _finding(
                    kind="source_separator",
                    line_number=line_number,
                    line=line,
                )
            )
        if index >= tail_start:
            for kind, pattern in _TAIL_APPARATUS_PATTERNS:
                if pattern.search(line):
                    tail_apparatus.append(
                        _finding(
                            kind=kind,
                            line_number=line_number,
                            line=line,
                        )
                    )
    return transport, tail_apparatus


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _WORD_RE.findall(text))


def _whole_prefix_repeated_at_tail(
    text: str,
) -> dict[str, object] | None:
    """Detect a long exact prefix repeated through the end of the work."""

    tokens = _tokenize(text)
    if len(tokens) < 400:
        return None
    anchor_width = 32
    anchor = tokens[:anchor_width]
    search_start = max(anchor_width, len(tokens) // 3)
    for offset in range(search_start, len(tokens) - anchor_width + 1):
        if tokens[offset : offset + anchor_width] != anchor:
            continue
        repeated = len(tokens) - offset
        if repeated < 200:
            continue
        if tokens[:repeated] == tokens[offset:]:
            return {
                "kind": "whole_prefix_repeated_at_tail",
                "second_copy_token_offset": offset,
                "repeated_token_count": repeated,
            }
    return None


def _canonical_text(payload: bytes, *, work_id: str) -> str:
    if type(payload) is not bytes or not payload:
        raise CorpusTextQualityError(
            f"{work_id}: text payload must be exact non-empty bytes"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusTextQualityError(
            f"{work_id}: text payload is not strict UTF-8: {exc}"
        ) from exc
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise CorpusTextQualityError(
            f"{work_id}: text must have exactly one final LF"
        )
    if "\r" in text:
        raise CorpusTextQualityError(f"{work_id}: CR line endings rejected")
    if "\t" in text:
        raise CorpusTextQualityError(f"{work_id}: tab characters rejected")
    body = text[:-1]
    if not body or body != body.strip():
        raise CorpusTextQualityError(
            f"{work_id}: text body must have no edge whitespace"
        )
    lines = body.split("\n")
    if any(line != line.rstrip() for line in lines):
        raise CorpusTextQualityError(
            f"{work_id}: trailing line whitespace rejected"
        )
    if any(not left and not right for left, right in zip(lines, lines[1:])):
        raise CorpusTextQualityError(
            f"{work_id}: repeated blank lines rejected"
        )
    return body


@dataclasses.dataclass(frozen=True)
class CorpusTextAuditReport:
    """Path-independent result of auditing an exact literal corpus inventory."""

    payload: Mapping[str, object]

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def self_hash(self) -> str:
        return str(self.payload["self_hash"])

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)

    def validate(self) -> "CorpusTextAuditReport":
        raw = self.payload
        if type(raw) is not dict:
            raise CorpusTextQualityError("audit report must be an exact object")
        expected_keys = {
            "schema_version",
            "policy_version",
            "content_overlap_policy_version",
            "minimum_words",
            "containment_threshold",
            "minimum_shingles",
            "sample_size",
            "status",
            "expected_work_ids",
            "work_count",
            "total_word_count",
            "works",
            "cross_work_overlaps",
            "blocking_findings",
            "self_hash",
        }
        if set(raw) != expected_keys:
            raise CorpusTextQualityError(
                "audit report has missing or extra top-level keys"
            )
        if raw["schema_version"] != TEXT_QUALITY_AUDIT_SCHEMA_VERSION:
            raise CorpusTextQualityError("audit report schema is unsupported")
        if raw["policy_version"] != TEXT_QUALITY_POLICY_VERSION:
            raise CorpusTextQualityError("audit report policy is unsupported")
        if raw["content_overlap_policy_version"] != (
            CONTENT_OVERLAP_POLICY_VERSION
        ):
            raise CorpusTextQualityError(
                "audit report overlap policy is unsupported"
            )
        for key in ("minimum_words", "minimum_shingles", "sample_size"):
            _exact_positive_int(raw[key], f"audit report.{key}")
        threshold = raw["containment_threshold"]
        if type(threshold) is not str or re.fullmatch(
            r"[1-9]\d*/[1-9]\d*",
            threshold,
        ) is None:
            raise CorpusTextQualityError(
                "audit report containment_threshold must be a rational string"
            )
        numerator, denominator = (int(item) for item in threshold.split("/"))
        if not Fraction(0, 1) < Fraction(numerator, denominator) <= Fraction(
            1,
            1,
        ):
            raise CorpusTextQualityError(
                "audit report containment_threshold is outside (0, 1]"
            )
        recorded = raw["self_hash"]
        if type(recorded) is not str or _HEX64_RE.fullmatch(recorded) is None:
            raise CorpusTextQualityError("audit report self_hash is malformed")
        unhashed = {key: value for key, value in raw.items() if key != "self_hash"}
        if canonical_hash(unhashed) != recorded:
            raise CorpusTextQualityError("audit report self_hash mismatch")
        expected_raw = raw["expected_work_ids"]
        if type(expected_raw) is not list or not expected_raw:
            raise CorpusTextQualityError(
                "audit report expected_work_ids must be a non-empty array"
            )
        expected = tuple(
            _work_id(value, f"audit report.expected_work_ids[{index}]")
            for index, value in enumerate(expected_raw)
        )
        if expected != tuple(sorted(expected)) or len(expected) != len(
            set(expected)
        ):
            raise CorpusTextQualityError(
                "audit report expected_work_ids must be sorted and unique"
            )
        work_count = _exact_positive_int(
            raw["work_count"],
            "audit report.work_count",
        )
        if work_count != len(expected):
            raise CorpusTextQualityError(
                "audit report work_count differs from expected_work_ids"
            )
        if type(raw["total_word_count"]) is not int or (
            raw["total_word_count"] < work_count
        ):
            raise CorpusTextQualityError(
                "audit report total_word_count is malformed"
            )
        work_rows = raw["works"]
        if type(work_rows) is not list or len(work_rows) != work_count:
            raise CorpusTextQualityError(
                "audit report works must exactly cover expected_work_ids"
            )
        work_keys = {
            "work_id",
            "byte_size",
            "sha256",
            "word_count",
            "line_count",
            "nonblank_line_count",
            "first_nonblank_line",
            "first_nonblank_line_sha256",
            "last_nonblank_line",
            "last_nonblank_line_sha256",
            "transport_residue_findings",
            "tail_apparatus_findings",
            "internal_duplication_findings",
        }
        observed_words = 0
        for index, (work_id, value) in enumerate(
            zip(expected, work_rows, strict=True)
        ):
            if type(value) is not dict or set(value) != work_keys:
                raise CorpusTextQualityError(
                    f"audit report.works[{index}] keys must be exact"
                )
            if value["work_id"] != work_id:
                raise CorpusTextQualityError(
                    "audit report works are not in expected_work_ids order"
                )
            for key in (
                "byte_size",
                "word_count",
                "line_count",
                "nonblank_line_count",
            ):
                _exact_positive_int(
                    value[key],
                    f"audit report.works[{index}].{key}",
                )
            observed_words += value["word_count"]
            _sha256(
                value["sha256"],
                f"audit report.works[{index}].sha256",
            )
            for edge in ("first", "last"):
                line = value[f"{edge}_nonblank_line"]
                if type(line) is not str or not line or "\n" in line:
                    raise CorpusTextQualityError(
                        f"audit report.works[{index}].{edge}_nonblank_line "
                        "must be one non-empty line"
                    )
                line_hash = _sha256(
                    value[f"{edge}_nonblank_line_sha256"],
                    f"audit report.works[{index}]."
                    f"{edge}_nonblank_line_sha256",
                )
                if _line_sha256(line) != line_hash:
                    raise CorpusTextQualityError(
                        f"audit report.works[{index}] {edge} line hash mismatch"
                    )
            for key in (
                "transport_residue_findings",
                "tail_apparatus_findings",
            ):
                findings = value[key]
                if type(findings) is not list:
                    raise CorpusTextQualityError(
                        f"audit report.works[{index}].{key} must be an array"
                    )
                for finding_index, finding in enumerate(findings):
                    label = (
                        f"audit report.works[{index}].{key}"
                        f"[{finding_index}]"
                    )
                    if type(finding) is not dict or set(finding) != {
                        "kind",
                        "line_number",
                        "line_sha256",
                        "excerpt",
                    }:
                        raise CorpusTextQualityError(
                            f"{label} keys must be exact"
                        )
                    if type(finding["kind"]) is not str or not finding["kind"]:
                        raise CorpusTextQualityError(
                            f"{label}.kind must be a non-empty string"
                        )
                    _exact_positive_int(
                        finding["line_number"],
                        f"{label}.line_number",
                    )
                    _sha256(finding["line_sha256"], f"{label}.line_sha256")
                    if (
                        type(finding["excerpt"]) is not str
                        or "\n" in finding["excerpt"]
                        or len(finding["excerpt"]) > 240
                    ):
                        raise CorpusTextQualityError(
                            f"{label}.excerpt is malformed"
                        )
            internal = value["internal_duplication_findings"]
            if type(internal) is not list or len(internal) > 1:
                raise CorpusTextQualityError(
                    "audit internal duplication findings must have length 0/1"
                )
            if internal:
                finding = internal[0]
                if type(finding) is not dict or set(finding) != {
                    "kind",
                    "second_copy_token_offset",
                    "repeated_token_count",
                }:
                    raise CorpusTextQualityError(
                        "audit internal duplication finding keys must be exact"
                    )
                if finding["kind"] != "whole_prefix_repeated_at_tail":
                    raise CorpusTextQualityError(
                        "audit internal duplication kind is unsupported"
                    )
                _exact_positive_int(
                    finding["second_copy_token_offset"],
                    "audit internal duplication offset",
                )
                _exact_positive_int(
                    finding["repeated_token_count"],
                    "audit internal duplication token count",
                )
        if observed_words != raw["total_word_count"]:
            raise CorpusTextQualityError(
                "audit report total_word_count differs from work rows"
            )
        overlaps = raw["cross_work_overlaps"]
        if type(overlaps) is not list:
            raise CorpusTextQualityError(
                "audit report cross_work_overlaps must be an array"
            )
        expected_set = set(expected)
        overlap_keys = {
            "left_work",
            "right_work",
            "kind",
            "containment",
            "evidence",
        }
        for index, overlap in enumerate(overlaps):
            label = f"audit report.cross_work_overlaps[{index}]"
            if type(overlap) is not dict or set(overlap) != overlap_keys:
                raise CorpusTextQualityError(f"{label} keys must be exact")
            if (
                overlap["left_work"] not in expected_set
                or overlap["right_work"] not in expected_set
                or overlap["left_work"] == overlap["right_work"]
            ):
                raise CorpusTextQualityError(
                    f"{label} work ids are not distinct corpus works"
                )
            for key in ("kind", "evidence"):
                if type(overlap[key]) is not str or not overlap[key]:
                    raise CorpusTextQualityError(
                        f"{label}.{key} must be a non-empty string"
                    )
            containment = overlap["containment"]
            if type(containment) is not str:
                raise CorpusTextQualityError(
                    f"{label}.containment must be a decimal string"
                )
            try:
                decimal = Decimal(containment)
            except InvalidOperation as exc:
                raise CorpusTextQualityError(
                    f"{label}.containment is malformed"
                ) from exc
            if not decimal.is_finite() or not Decimal(0) < decimal <= Decimal(
                1
            ):
                raise CorpusTextQualityError(
                    f"{label}.containment is outside (0, 1]"
                )
        status = raw["status"]
        blockers = raw["blocking_findings"]
        if type(blockers) is not list:
            raise CorpusTextQualityError(
                "audit report blocking_findings must be an array"
            )
        if status not in {"passed", "blocked"}:
            raise CorpusTextQualityError("audit report status is unsupported")
        if (status == "passed") != (not blockers):
            raise CorpusTextQualityError(
                "audit report status conflicts with blocking findings"
            )
        for index, blocker in enumerate(blockers):
            label = f"audit report.blocking_findings[{index}]"
            if type(blocker) is not dict or set(blocker) != {
                "kind",
                "work_ids",
                "evidence",
            }:
                raise CorpusTextQualityError(f"{label} keys must be exact")
            if type(blocker["kind"]) is not str or not blocker["kind"]:
                raise CorpusTextQualityError(
                    f"{label}.kind must be a non-empty string"
                )
            ids = blocker["work_ids"]
            if (
                type(ids) is not list
                or not 1 <= len(ids) <= 2
                or any(work not in expected_set for work in ids)
                or len(ids) != len(set(ids))
            ):
                raise CorpusTextQualityError(
                    f"{label}.work_ids must contain one/two corpus works"
                )
            if type(blocker["evidence"]) is not str or not blocker["evidence"]:
                raise CorpusTextQualityError(
                    f"{label}.evidence must be a non-empty string"
                )
        return self


def audit_corpus_texts(
    work_payloads: Mapping[str, bytes],
    *,
    expected_work_ids: Sequence[str],
    minimum_words: int = DEFAULT_MINIMUM_WORDS,
    containment_threshold: Fraction = DEFAULT_CONTAINMENT_THRESHOLD,
    minimum_shingles: int = DEFAULT_MINIMUM_SHINGLES,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> CorpusTextAuditReport:
    """Audit every exact work and all cross-work relationships.

    The returned report is always complete, including when it is blocked.
    Call :func:`require_text_quality` when a fail-closed exception is desired.
    """

    if type(work_payloads) is not dict:
        raise CorpusTextQualityError(
            "work_payloads must be an exact dictionary"
        )
    expected = tuple(
        _work_id(value, f"expected_work_ids[{index}]")
        for index, value in enumerate(expected_work_ids)
    )
    if not expected or expected != tuple(sorted(expected)):
        raise CorpusTextQualityError(
            "expected_work_ids must be a non-empty sorted sequence"
        )
    if len(expected) != len(set(expected)):
        raise CorpusTextQualityError("expected_work_ids must be unique")
    if set(work_payloads) != set(expected):
        raise CorpusTextQualityError(
            "literal work inventory differs from expected_work_ids"
        )
    minimum_words = _exact_positive_int(minimum_words, "minimum_words")
    minimum_shingles = _exact_positive_int(
        minimum_shingles,
        "minimum_shingles",
    )
    sample_size = _exact_positive_int(sample_size, "sample_size")
    if type(containment_threshold) is not Fraction:
        raise CorpusTextQualityError(
            "containment_threshold must be exactly Fraction"
        )
    if not Fraction(0, 1) < containment_threshold <= Fraction(1, 1):
        raise CorpusTextQualityError(
            "containment_threshold must be in (0, 1]"
        )

    texts: list[str] = []
    work_rows: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    total_words = 0
    for work_id in expected:
        payload = work_payloads[work_id]
        text = _canonical_text(payload, work_id=work_id)
        lines = text.split("\n")
        nonblank = [line for line in lines if line]
        words = count_words(text)
        total_words += words
        transport, apparatus = _line_findings(lines)
        internal = _whole_prefix_repeated_at_tail(text)
        row: dict[str, object] = {
            "work_id": work_id,
            "byte_size": len(payload),
            "sha256": _sha256_bytes(payload),
            "word_count": words,
            "line_count": len(lines),
            "nonblank_line_count": len(nonblank),
            "first_nonblank_line": nonblank[0],
            "first_nonblank_line_sha256": _line_sha256(nonblank[0]),
            "last_nonblank_line": nonblank[-1],
            "last_nonblank_line_sha256": _line_sha256(nonblank[-1]),
            "transport_residue_findings": transport,
            "tail_apparatus_findings": apparatus,
            "internal_duplication_findings": (
                [] if internal is None else [internal]
            ),
        }
        work_rows.append(row)
        if words < minimum_words:
            blockers.append(
                {
                    "kind": "work_below_minimum_words",
                    "work_ids": [work_id],
                    "evidence": f"{words} < {minimum_words}",
                }
            )
        if _CONTROL_RE.search(text):
            blockers.append(
                {
                    "kind": "control_character",
                    "work_ids": [work_id],
                    "evidence": "non-LF C0/DEL control character present",
                }
            )
        for label, findings in (
            ("transport_residue", transport),
            ("tail_apparatus", apparatus),
        ):
            if findings:
                blockers.append(
                    {
                        "kind": label,
                        "work_ids": [work_id],
                        "evidence": (
                            f"{len(findings)} finding(s); "
                            f"first={findings[0]['kind']}@"
                            f"{findings[0]['line_number']}"
                        ),
                    }
                )
        if internal is not None:
            blockers.append(
                {
                    "kind": "internal_whole_text_duplication",
                    "work_ids": [work_id],
                    "evidence": (
                        f"offset={internal['second_copy_token_offset']}; "
                        f"tokens={internal['repeated_token_count']}"
                    ),
                }
            )
        texts.append(text)

    overlaps = find_cross_work_content_overlaps(
        texts,
        expected,
        containment_threshold=containment_threshold,
        min_shingles=minimum_shingles,
        sample_size=sample_size,
    )
    overlap_rows = [
        {
            "left_work": row.left_work,
            "right_work": row.right_work,
            "kind": row.kind,
            "containment": format(row.containment, ".12g"),
            "evidence": row.evidence,
        }
        for row in overlaps
    ]
    blockers.extend(
        {
            "kind": f"cross_work_{row.kind}",
            "work_ids": [row.left_work, row.right_work],
            "evidence": row.evidence,
        }
        for row in overlaps
    )
    blockers.sort(
        key=lambda row: (
            str(row["kind"]),
            tuple(str(value) for value in row["work_ids"]),
            str(row["evidence"]),
        )
    )
    payload: dict[str, object] = {
        "schema_version": TEXT_QUALITY_AUDIT_SCHEMA_VERSION,
        "policy_version": TEXT_QUALITY_POLICY_VERSION,
        "content_overlap_policy_version": CONTENT_OVERLAP_POLICY_VERSION,
        "minimum_words": minimum_words,
        "containment_threshold": (
            f"{containment_threshold.numerator}/"
            f"{containment_threshold.denominator}"
        ),
        "minimum_shingles": minimum_shingles,
        "sample_size": sample_size,
        "status": "passed" if not blockers else "blocked",
        "expected_work_ids": list(expected),
        "work_count": len(expected),
        "total_word_count": total_words,
        "works": work_rows,
        "cross_work_overlaps": overlap_rows,
        "blocking_findings": blockers,
    }
    report = CorpusTextAuditReport(
        {**payload, "self_hash": canonical_hash(payload)}
    )
    return report.validate()


def require_text_quality(report: CorpusTextAuditReport) -> None:
    """Fail closed unless a validated report has no blocking findings."""

    if type(report) is not CorpusTextAuditReport:
        raise CorpusTextQualityError(
            "text quality gate requires exactly CorpusTextAuditReport"
        )
    report.validate()
    if report.status != "passed":
        blockers = report.payload["blocking_findings"]
        assert type(blockers) is list
        sample = "; ".join(
            f"{row['kind']}:{','.join(row['work_ids'])}"
            for row in blockers[:8]
        )
        more = "" if len(blockers) <= 8 else f"; +{len(blockers) - 8} more"
        raise CorpusTextQualityError(
            f"corpus text quality audit blocked: {sample}{more}"
        )


__all__ = [
    "CorpusTextAuditReport",
    "CorpusTextQualityError",
    "DEFAULT_CONTAINMENT_THRESHOLD",
    "DEFAULT_MINIMUM_SHINGLES",
    "DEFAULT_MINIMUM_WORDS",
    "DEFAULT_SAMPLE_SIZE",
    "TEXT_QUALITY_AUDIT_SCHEMA_VERSION",
    "TEXT_QUALITY_POLICY_VERSION",
    "audit_corpus_texts",
    "require_text_quality",
]
