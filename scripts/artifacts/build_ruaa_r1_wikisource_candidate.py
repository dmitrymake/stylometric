#!/usr/bin/env python3
"""Apply the compact RuAA R1 dispositions to the exact 136-work candidate.

The output is a large generated v4 discovery candidate and is intentionally
kept outside Git.  This builder is deterministic and has no network surface.
"""
from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import stat
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.corpus_tools.reviewed_text_vnext import (  # noqa: E402
    loads_reviewed_text_campaign_spec,
)
from stylo.corpus_tools.wikisource_discovery import (  # noqa: E402
    DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4,
    EXTERNAL_PROVIDER_WORK_STATUS,
    READY_STATUS,
    SOURCE_QUALITY_REJECTED_WORK_STATUS,
    DiscoveryCandidate,
)
from stylo.jsonio import canonical_hash, loads_strict  # noqa: E402


SOURCE_CANDIDATE_FILE_SHA256 = (
    "91debfaca03b3e4e692b922bc68591a50a12db186e6aeff0eac13bf069db0f0e"
)
SOURCE_CANDIDATE_HASH = (
    "50967141c0b98913a583481288b6ef9a13650186bf2f0a68ab035273a7104dd9"
)
DISPOSITIONS_PATH = (
    ROOT
    / "research"
    / "corpus_sources"
    / "ruaa_r1_source_dispositions_v1.json"
)
DISPOSITIONS_SELF_HASH = (
    "9903e3ced1156f42e669d4a199f75ca704e94bbf843ed658c705b428a5c776ed"
)
DISPOSITIONS_FILE_SHA256 = (
    "b05049760fbe18192bd998e8de5b08698c93be75ea7fcd1d32bea6b7b71eb9b0"
)
REVIEWED_CAMPAIGN_PATH = (
    ROOT
    / "research"
    / "corpus_sources"
    / "ruaa_r1_reviewed_text_campaign_v1.json"
)
EXPECTED_WORK_COUNT = 136
EXPECTED_PART_COUNT = 1360
EXPECTED_WIKISOURCE_WORK_COUNT = 127
EXPECTED_REVIEWED_WORK_COUNT = 6
EXPECTED_FEB_WORK_COUNT = 1
EXPECTED_INCLUDED_WORK_COUNT = 134
EXPECTED_DISPOSITION_COUNT = 10
SELECTED_QUALITY_REVIEW_RECEIPT_SHA256 = (
    "20a53dcca41b31021708fff797425691ba4cccb668e439395a53fcb82d955a61"
)
SELECTED_QUALITY_PART_PATCHES = {
    ("bunin/деревня", 0, 5585755): {
        "body_start_line": 0,
        "body_end_line_exclusive": None,
        "body_disposition": "whole_rendered_body",
        "source_repairs": [
            {
                "policy_version": (
                    "stylo.wikisource.reviewed-literal-source-repair.v1"
                ),
                "literal": (
                    "Оригинал здесь: Электронная библиотека Яблучанского."
                ),
                "replacement": "",
                "expected_count": 1,
                "occurrence_line_sha256": [
                    "109eaa4d1662f9c091019b6788b508c26c5ffc047e2a50531b842a82c47f3d56"
                ],
                "review_receipt_sha256": (
                    SELECTED_QUALITY_REVIEW_RECEIPT_SHA256
                ),
            }
        ],
    },
    ("bunin/суходол", 0, 5588281): {
        "body_start_line": 0,
        "body_end_line_exclusive": None,
        "body_disposition": "whole_rendered_body",
        "source_repairs": [
            {
                "policy_version": (
                    "stylo.wikisource.reviewed-literal-source-repair.v1"
                ),
                "literal": (
                    "Оригинал здесь: Электронная библиотека Яблучанского."
                ),
                "replacement": "",
                "expected_count": 1,
                "occurrence_line_sha256": [
                    "109eaa4d1662f9c091019b6788b508c26c5ffc047e2a50531b842a82c47f3d56"
                ],
                "review_receipt_sha256": (
                    SELECTED_QUALITY_REVIEW_RECEIPT_SHA256
                ),
            }
        ],
    },
    ("dostoevsky/бесы", 23, 5625680): {
        "body_start_line": 0,
        "body_end_line_exclusive": None,
        "body_disposition": "whole_rendered_body",
        "source_repairs": [
            {
                "policy_version": (
                    "stylo.wikisource.reviewed-literal-source-repair.v1"
                ),
                "literal": "<III>",
                "replacement": "III",
                "expected_count": 1,
                "occurrence_line_sha256": [
                    "5699a51036ab8d8d2bd4eeb47b50218b18f3fad1aab284d2161b68a0a3b7fe36"
                ],
                "review_receipt_sha256": (
                    SELECTED_QUALITY_REVIEW_RECEIPT_SHA256
                ),
            }
        ],
    },
    ("furmanov/красный_десант", 0, 5604500): {
        "body_start_line": 0,
        "body_end_line_exclusive": 531,
        "body_disposition": "strip_trailing_apparatus",
        "source_repairs": [],
    },
    ("furmanov/мятеж", 0, 5307372): {
        "body_start_line": 0,
        "body_end_line_exclusive": 6526,
        "body_disposition": "strip_trailing_apparatus",
        "source_repairs": [],
    },
    ("gogol/мёртвые_души", 16, 3575878): {
        "body_start_line": 0,
        "body_end_line_exclusive": None,
        "body_disposition": "whole_rendered_body",
        "source_repairs": [
            {
                "policy_version": (
                    "stylo.wikisource.reviewed-literal-source-repair.v1"
                ),
                "literal": "<Myразов>",
                "replacement": "<Муразов>",
                "expected_count": 1,
                "occurrence_line_sha256": [
                    "d96b545baf5edd063376bd43d73d8923e3f1df1229060fc3292f39a7677539aa"
                ],
                "review_receipt_sha256": (
                    SELECTED_QUALITY_REVIEW_RECEIPT_SHA256
                ),
            }
        ],
    },
    ("grin/алые_паруса", 0, 5636844): {
        "body_start_line": 20,
        "body_end_line_exclusive": None,
        "body_disposition": "strip_leading_apparatus",
        "source_repairs": [],
    },
    ("korolenko/соколинец", 0, 5588120): {
        "body_start_line": 8,
        "body_end_line_exclusive": None,
        "body_disposition": "strip_leading_apparatus",
        "source_repairs": [],
    },
    ("serafimovich/пески", 0, 5595521): {
        "body_start_line": 0,
        "body_end_line_exclusive": 885,
        "body_disposition": "strip_trailing_apparatus",
        "source_repairs": [],
    },
    ("serafimovich/степные_люди", 0, 5596256): {
        "body_start_line": 0,
        "body_end_line_exclusive": 339,
        "body_disposition": "strip_trailing_apparatus",
        "source_repairs": [],
    },
    ("uspensky/будка", 0, 5588776): {
        "body_start_line": 0,
        "body_end_line_exclusive": 1691,
        "body_disposition": "strip_trailing_apparatus",
        "source_repairs": [],
    },
}
class R1CandidateBuilderError(ValueError):
    """The exact source candidate or disposition patch drifted."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: pathlib.Path, *, label: str) -> bytes:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise R1CandidateBuilderError(
                f"{label} contains a symlink component: {component}"
            )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise R1CandidateBuilderError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise R1CandidateBuilderError(
                f"{label} must be a regular file"
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
            raise R1CandidateBuilderError(
                f"{label} changed while being read"
            )
        return payload
    finally:
        os.close(descriptor)


def _load_exact_object(payload: bytes, *, label: str) -> dict:
    try:
        value = loads_strict(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise R1CandidateBuilderError(f"cannot load {label}: {exc}") from exc
    if type(value) is not dict:
        raise R1CandidateBuilderError(f"{label} must be an exact object")
    return value


def _validate_dispositions(
    *,
    reviewed_campaign_path: pathlib.Path | None = None,
) -> tuple[dict[str, dict], dict]:
    disposition_payload = _read_regular(
        DISPOSITIONS_PATH,
        label="R1 dispositions",
    )
    if _sha256(disposition_payload) != DISPOSITIONS_FILE_SHA256:
        raise R1CandidateBuilderError("R1 dispositions file SHA-256 mismatch")
    raw = _load_exact_object(
        disposition_payload,
        label="R1 dispositions",
    )
    recorded = raw.get("self_hash")
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    if recorded != DISPOSITIONS_SELF_HASH or canonical_hash(payload) != recorded:
        raise R1CandidateBuilderError("R1 dispositions self-hash mismatch")
    if (
        raw.get("schema_version")
        != "stylo.ruaa-r1.source-dispositions.v1"
        or raw.get("models_executed") is not False
        or raw.get("source_candidate")
        != {
            "candidate_hash": SOURCE_CANDIDATE_HASH,
            "file_sha256": SOURCE_CANDIDATE_FILE_SHA256,
        }
        or raw.get("expected_inventory")
        != {
            "feb_work_count": EXPECTED_FEB_WORK_COUNT,
            "included_work_count": EXPECTED_INCLUDED_WORK_COUNT,
            "reviewed_text_work_count": EXPECTED_REVIEWED_WORK_COUNT,
            "source_part_count": EXPECTED_PART_COUNT,
            "source_work_count": EXPECTED_WORK_COUNT,
            "wikisource_work_count": EXPECTED_WIKISOURCE_WORK_COUNT,
        }
    ):
        raise R1CandidateBuilderError(
            "R1 dispositions top-level contract drifted"
        )
    rows = raw.get("work_dispositions")
    if type(rows) is not list or len(rows) != EXPECTED_DISPOSITION_COUNT:
        raise R1CandidateBuilderError(
            "R1 dispositions work inventory must be an exact array"
        )
    result: dict[str, dict] = {}
    for row in rows:
        if type(row) is not dict or type(row.get("work_id")) is not str:
            raise R1CandidateBuilderError(
                "R1 disposition row is malformed"
            )
        work_id = row["work_id"]
        if work_id in result:
            raise R1CandidateBuilderError(
                f"duplicate R1 disposition: {work_id}"
            )
        result[work_id] = row
    reviewed_payload = _read_regular(
        REVIEWED_CAMPAIGN_PATH
        if reviewed_campaign_path is None
        else reviewed_campaign_path,
        label="reviewed campaign",
    )
    reviewed = loads_reviewed_text_campaign_spec(
        reviewed_payload.decode("utf-8")
    )
    reviewed_identity = raw.get("reviewed_text_campaign")
    if type(reviewed_identity) is not dict or reviewed_identity != {
        "file_sha256": _sha256(reviewed_payload),
        "self_hash": reviewed.self_hash,
    }:
        raise R1CandidateBuilderError(
            "reviewed campaign top-level identity mismatch"
        )
    reviewed_ids = {
        work_id
        for work_id, row in result.items()
        if row.get("provider") == "reviewed_text"
    }
    reviewed_outputs = {
        work_id: row.get("output_sha256")
        for work_id, row in result.items()
        if row.get("provider") == "reviewed_text"
    }
    if (
        reviewed_ids != set(reviewed.work_ids)
        or reviewed_outputs
        != {work.work_id: work.sha256 for work in reviewed.works}
    ):
        raise R1CandidateBuilderError(
            "reviewed campaign and disposition identities differ"
        )
    generated = raw.get("generated_ready_candidate")
    if type(generated) is not dict:
        raise R1CandidateBuilderError(
            "generated ready candidate identity is malformed"
        )
    return result, generated


def build_candidate(
    source_path: pathlib.Path,
    *,
    reviewed_campaign_path: pathlib.Path | None = None,
) -> tuple[DiscoveryCandidate, dict]:
    payload = _read_regular(source_path, label="source candidate")
    if _sha256(payload) != SOURCE_CANDIDATE_FILE_SHA256:
        raise R1CandidateBuilderError("source candidate file SHA-256 mismatch")
    raw = _load_exact_object(payload, label="source candidate")
    if raw.get("candidate_hash") != SOURCE_CANDIDATE_HASH:
        raise R1CandidateBuilderError("source candidate canonical hash mismatch")
    # Validate the original strict v1 candidate before applying any patch.
    DiscoveryCandidate.from_dict(raw)
    dispositions, generated_identity = _validate_dispositions(
        reviewed_campaign_path=reviewed_campaign_path,
    )
    candidate = copy.deepcopy(raw)
    candidate["schema_version"] = DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4
    candidate["status"] = READY_STATUS
    candidate["unresolved_choices"] = []

    candidate_ids = {
        row["work_id"] for row in candidate["works"]
    }
    outside_source = {
        work_id for work_id in dispositions if work_id not in candidate_ids
    }
    if outside_source != {"turgenev/записки_охотника"}:
        raise R1CandidateBuilderError(
            "R1 dispositions outside-source inventory drifted"
        )
    applied_quality_patches: set[tuple[str, int, int]] = set()
    for work in candidate["works"]:
        work_id = work["work_id"]
        disposition = dispositions.get(work_id)
        if disposition is not None:
            chosen = disposition["disposition"]
            if chosen == "pinned_external_provider":
                work["include_in_corpus"] = False
                work["selection_status"] = EXTERNAL_PROVIDER_WORK_STATUS
                work["issues"] = [
                    {
                        "chosen_disposition": "pinned_external_provider",
                        "kind": "source_disposition",
                        "reason": disposition["reason"],
                    }
                ]
            elif chosen == "exclude_source_quality":
                work["include_in_corpus"] = False
                work["selection_status"] = (
                    SOURCE_QUALITY_REJECTED_WORK_STATUS
                )
                work["issues"] = [
                    {
                        "chosen_disposition": "exclude_source_quality",
                        "kind": "source_disposition",
                        "reason": disposition["reason"],
                    }
                ]
            elif chosen == "exclude_authorship_mismatch":
                # Preserve the exact rejected-work receipt and its original
                # evidence-bound work record.
                pass
            elif chosen != "exclude_collection_umbrella":
                raise R1CandidateBuilderError(
                    f"unsupported disposition for {work_id}: {chosen!r}"
                )
        for part in work["parts"]:
            part.update(
                {
                    "body_start_line": 0,
                    "body_end_line_exclusive": None,
                    "body_disposition": "whole_rendered_body",
                    "source_repairs": [],
                }
            )
            patch_key = (
                work_id,
                part["ordinal"],
                part["revision_id"],
            )
            quality_patch = SELECTED_QUALITY_PART_PATCHES.get(patch_key)
            if quality_patch is not None:
                part.update(copy.deepcopy(quality_patch))
                applied_quality_patches.add(patch_key)
    if applied_quality_patches != set(SELECTED_QUALITY_PART_PATCHES):
        missing = sorted(
            set(SELECTED_QUALITY_PART_PATCHES) - applied_quality_patches
        )
        raise R1CandidateBuilderError(
            "selected-quality source identities drifted: "
            f"missing exact part(s) {missing!r}"
        )

    selected = sum(
        work["include_in_corpus"] for work in candidate["works"]
    )
    part_count = sum(len(work["parts"]) for work in candidate["works"])
    blocked = sum(
        not work["include_in_corpus"] for work in candidate["works"]
    )
    candidate["selection_contract"]["selected_work_count"] = selected
    candidate["summary"] = {
        "authorship_rejected_work_count": 1,
        "blocked_work_count": blocked,
        "missing_part_count": 0,
        "part_count": part_count,
        "resolved_part_count": part_count,
        "work_count": len(candidate["works"]),
    }
    unhashed = {
        key: value
        for key, value in candidate.items()
        if key != "candidate_hash"
    }
    candidate["candidate_hash"] = canonical_hash(unhashed)
    result = DiscoveryCandidate.from_dict(candidate).assert_ready()
    if (
        len(result.works) != EXPECTED_WORK_COUNT
        or sum(len(work.parts) for work in result.works)
        != EXPECTED_PART_COUNT
        or len([work for work in result.works if work.include_in_corpus])
        != EXPECTED_WIKISOURCE_WORK_COUNT
    ):
        raise R1CandidateBuilderError(
            "ready R1 candidate inventory differs from the frozen contract"
        )
    if (
        result.candidate_hash
        != generated_identity.get("candidate_hash")
        or result.schema_version
        != generated_identity.get("schema_version")
    ):
        raise R1CandidateBuilderError(
            "ready R1 candidate identity differs from the frozen contract"
        )
    return result, candidate
