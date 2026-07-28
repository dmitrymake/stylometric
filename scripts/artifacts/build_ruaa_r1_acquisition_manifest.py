#!/usr/bin/env python3
"""Build the generated 134-work RuAA R1 acquisition manifest offline.

The large ready candidate, pinned Wikisource campaign, and resulting R1
manifest are generated artifacts and normally live outside Git.  This compact
builder binds them to the tracked dispositions, FEB spec, and reviewed-text
campaign.  It has no transport or model surface.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import sys
from collections.abc import Mapping


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.corpus_tools.feb_vnext import (  # noqa: E402
    PinnedFEBWorkSpec,
)
from stylo.corpus_tools.reviewed_text_vnext import (  # noqa: E402
    loads_reviewed_text_campaign_spec,
)
from stylo.corpus_tools.ruaa_r1_acquisition import (  # noqa: E402
    R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V3,
    R1AcquisitionManifest,
)
from stylo.corpus_tools.wikisource_campaign import (  # noqa: E402
    loads_campaign_spec,
)
from stylo.corpus_tools.wikisource_discovery import (  # noqa: E402
    DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4,
    loads_discovery_candidate,
)
from stylo.corpus_tools.wikisource_vnext import (  # noqa: E402
    PINNED_WORK_SPEC_SCHEMA_VERSION_V4,
)
from stylo.jsonio import (  # noqa: E402
    canonical_hash,
    loads_strict,
)


DISPOSITIONS_PATH = (
    ROOT
    / "research"
    / "corpus_sources"
    / "ruaa_r1_source_dispositions_v1.json"
)
REVIEWED_CAMPAIGN_PATH = (
    ROOT
    / "research"
    / "corpus_sources"
    / "ruaa_r1_reviewed_text_campaign_v1.json"
)
FEB_SPEC_PATH = (
    ROOT
    / "research"
    / "corpus_sources"
    / "ruaa_r1_pushkin_pugachev_feb_v1.json"
)
DISPOSITIONS_SCHEMA = "stylo.ruaa-r1.source-dispositions.v1"
DISPOSITIONS_FILE_SHA256 = (
    "b05049760fbe18192bd998e8de5b08698c93be75ea7fcd1d32bea6b7b71eb9b0"
)
DISPOSITIONS_SELF_HASH = (
    "9903e3ced1156f42e669d4a199f75ca704e94bbf843ed658c705b428a5c776ed"
)
EXPECTED_WIKISOURCE_WORK_COUNT = 127
EXPECTED_REVIEWED_WORK_COUNT = 6
EXPECTED_INCLUDED_WORK_COUNT = 134
EXPECTED_SOURCE_WORK_COUNT = 136
EXPECTED_SOURCE_PART_COUNT = 1360


class R1ManifestBuilderError(ValueError):
    """The bounded R1 provider union differs from the tracked contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: pathlib.Path, *, label: str) -> bytes:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise R1ManifestBuilderError(
                f"{label} contains a symlink component: {component}"
            )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise R1ManifestBuilderError(f"{label} cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise R1ManifestBuilderError(
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
            raise R1ManifestBuilderError(
                f"{label} changed while being read"
            )
        return payload
    finally:
        os.close(descriptor)


def _exact_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise R1ManifestBuilderError(f"{label} must be an exact object")
    return value


def _load_dispositions() -> Mapping[str, object]:
    payload = _read_regular(DISPOSITIONS_PATH, label="source dispositions")
    if _sha256(payload) != DISPOSITIONS_FILE_SHA256:
        raise R1ManifestBuilderError(
            "source dispositions file SHA-256 mismatch"
        )
    raw = _exact_mapping(
        loads_strict(payload.decode("utf-8")),
        label="source dispositions",
    )
    if (
        raw.get("schema_version") != DISPOSITIONS_SCHEMA
        or raw.get("self_hash") != DISPOSITIONS_SELF_HASH
    ):
        raise R1ManifestBuilderError(
            "source dispositions identity mismatch"
        )
    core = {key: value for key, value in raw.items() if key != "self_hash"}
    if canonical_hash(core) != DISPOSITIONS_SELF_HASH:
        raise R1ManifestBuilderError(
            "source dispositions canonical self-hash mismatch"
        )
    expected = _exact_mapping(
        raw.get("expected_inventory"),
        label="source dispositions expected_inventory",
    )
    counts = {
        "wikisource_work_count": EXPECTED_WIKISOURCE_WORK_COUNT,
        "reviewed_text_work_count": EXPECTED_REVIEWED_WORK_COUNT,
        "feb_work_count": 1,
        "included_work_count": EXPECTED_INCLUDED_WORK_COUNT,
        "source_work_count": EXPECTED_SOURCE_WORK_COUNT,
        "source_part_count": EXPECTED_SOURCE_PART_COUNT,
    }
    if expected != counts:
        raise R1ManifestBuilderError(
            "source dispositions expected inventory drifted"
        )
    return raw


def _disposition_rows(
    dispositions: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    values = dispositions.get("work_dispositions")
    if type(values) is not list:
        raise R1ManifestBuilderError(
            "source dispositions work_dispositions must be an array"
        )
    rows: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(values):
        row = _exact_mapping(
            value,
            label=f"source disposition[{index}]",
        )
        work_id = row.get("work_id")
        if type(work_id) is not str or not work_id or work_id in rows:
            raise R1ManifestBuilderError(
                "source disposition work ids must be unique strings"
            )
        rows[work_id] = row
    return rows


def _validate_wikisource_mapping(candidate, wikisource) -> None:
    """Bind every pinned selected part to the ready-candidate instruction."""

    candidate_by_id = {work.work_id: work for work in candidate.works}
    for pinned_work in wikisource.works:
        discovered_work = candidate_by_id.get(pinned_work.work_id)
        if (
            discovered_work is None
            or len(pinned_work.parts) != len(discovered_work.parts)
        ):
            raise R1ManifestBuilderError(
                "pinned Wikisource part count differs from ready candidate: "
                f"{pinned_work.work_id}"
            )
        for discovered, pinned in zip(
            discovered_work.parts,
            pinned_work.parts,
            strict=True,
        ):
            boundary = pinned.body_boundary_v2
            expected_end = (
                None
                if boundary is None
                else (
                    boundary.full_line_count
                    if discovered.body_end_line_exclusive is None
                    else discovered.body_end_line_exclusive
                )
            )
            if (
                pinned.ordinal != discovered.ordinal
                or pinned.requested_title != discovered.requested_title
                or pinned.resolved_title != discovered.resolved_title
                or pinned.redirect_chain != discovered.redirect_chain
                or pinned.page_id != discovered.page_id
                or pinned.revision_id != discovered.revision_id
                or pinned.mediawiki_sha1 != discovered.revision_sha1
                or boundary is None
                or boundary.start_line != discovered.body_start_line
                or boundary.end_line_exclusive != expected_end
                or boundary.body_disposition
                != discovered.body_disposition
                or pinned.source_repair_v1
                != discovered.source_repair_v1
                or pinned.source_repairs_v1
                != discovered.source_repairs_v1
            ):
                raise R1ManifestBuilderError(
                    "pinned Wikisource part identity differs from ready "
                    f"candidate: {pinned_work.work_id}/{pinned.ordinal}"
                )


def _validate_wikisource_campaign_identity(
    *,
    binding: object,
    campaign_payload: bytes,
    campaign: object,
) -> None:
    """Require the exact tracked campaign before inspecting part mappings."""

    expected = _exact_mapping(
        binding,
        label="pinned Wikisource campaign identity",
    )
    if set(expected) != {"file_sha256", "self_hash", "generation_id"}:
        raise R1ManifestBuilderError(
            "pinned Wikisource campaign identity keys must be exact"
        )
    actual = {
        "file_sha256": _sha256(campaign_payload),
        "self_hash": getattr(campaign, "self_hash", None),
        "generation_id": getattr(campaign, "generation_id", None),
    }
    if expected != actual:
        raise R1ManifestBuilderError(
            "pinned Wikisource campaign exact identity mismatch"
        )


def build_manifest(
    *,
    ready_candidate_path: pathlib.Path,
    wikisource_campaign_path: pathlib.Path,
    reviewed_campaign_path: pathlib.Path | None = None,
) -> R1AcquisitionManifest:
    dispositions = _load_dispositions()
    rows = _disposition_rows(dispositions)

    candidate_identity = _exact_mapping(
        dispositions.get("generated_ready_candidate"),
        label="generated ready candidate identity",
    )
    candidate_payload = _read_regular(
        ready_candidate_path,
        label="generated ready candidate",
    )
    if _sha256(candidate_payload) != candidate_identity.get("file_sha256"):
        raise R1ManifestBuilderError(
            "generated ready candidate file SHA-256 mismatch"
        )
    candidate = loads_discovery_candidate(
        candidate_payload.decode("utf-8")
    ).assert_ready()
    if (
        candidate.schema_version != DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4
        or candidate.candidate_hash
        != candidate_identity.get("candidate_hash")
    ):
        raise R1ManifestBuilderError(
            "generated ready candidate identity mismatch"
        )
    if (
        len(candidate.works) != EXPECTED_SOURCE_WORK_COUNT
        or sum(len(work.parts) for work in candidate.works)
        != EXPECTED_SOURCE_PART_COUNT
    ):
        raise R1ManifestBuilderError(
            "generated ready candidate source inventory mismatch"
        )
    selected_candidate_ids = tuple(
        sorted(
            work.work_id
            for work in candidate.works
            if work.include_in_corpus
        )
    )

    wikisource_payload = _read_regular(
        wikisource_campaign_path,
        label="pinned Wikisource campaign",
    )
    wikisource = loads_campaign_spec(
        wikisource_payload.decode("utf-8")
    )
    _validate_wikisource_campaign_identity(
        binding=dispositions.get("pinned_wikisource_campaign"),
        campaign_payload=wikisource_payload,
        campaign=wikisource,
    )
    if (
        len(wikisource.works) != EXPECTED_WIKISOURCE_WORK_COUNT
        or any(
            work.schema_version != PINNED_WORK_SPEC_SCHEMA_VERSION_V4
            for work in wikisource.works
        )
        or wikisource.work_ids != selected_candidate_ids
    ):
        raise R1ManifestBuilderError(
            "pinned Wikisource campaign differs from ready candidate"
        )
    candidate_by_id = {work.work_id: work for work in candidate.works}
    _validate_wikisource_mapping(candidate, wikisource)

    reviewed_identity = _exact_mapping(
        dispositions.get("reviewed_text_campaign"),
        label="reviewed campaign identity",
    )
    reviewed_payload = _read_regular(
        REVIEWED_CAMPAIGN_PATH
        if reviewed_campaign_path is None
        else reviewed_campaign_path,
        label="reviewed campaign",
    )
    reviewed = loads_reviewed_text_campaign_spec(
        reviewed_payload.decode("utf-8")
    )
    reviewed_rows = {
        work_id: row
        for work_id, row in rows.items()
        if row.get("provider") == "reviewed_text"
    }
    if (
        _sha256(reviewed_payload)
        != reviewed_identity.get("file_sha256")
        or reviewed.self_hash != reviewed_identity.get("self_hash")
        or len(reviewed.works) != EXPECTED_REVIEWED_WORK_COUNT
        or set(reviewed.work_ids) != set(reviewed_rows)
        or any(
            reviewed_rows[work.work_id].get("output_sha256")
            != work.sha256
            for work in reviewed.works
        )
        or any(
            work.work_id not in candidate_by_id
            or work.source_part_count
            != len(candidate_by_id[work.work_id].parts)
            for work in reviewed.works
        )
    ):
        raise R1ManifestBuilderError(
            "reviewed campaign differs from tracked dispositions"
        )

    feb_payload = _read_regular(FEB_SPEC_PATH, label="FEB work spec")
    feb = PinnedFEBWorkSpec.from_dict(
        loads_strict(feb_payload.decode("utf-8"))
    )
    feb_row = rows.get(feb.work_id)
    if (
        feb_row is None
        or feb_row.get("provider") != "feb"
        or _sha256(feb_payload)
        != feb_row.get("provider_spec_file_sha256")
        or feb.self_hash != feb_row.get("provider_spec_self_hash")
        or feb.output_sha256 != feb_row.get("output_sha256")
        or feb.work_id not in candidate_by_id
        or len(candidate_by_id[feb.work_id].parts) != 1
    ):
        raise R1ManifestBuilderError(
            "FEB work spec differs from tracked dispositions"
        )

    authorship = rows.get("serafimovich/у_нас_и_у_них")
    source_quality = rows.get("sevsky/дон_на_костылях")
    umbrella = rows.get("turgenev/записки_охотника")
    if (
        authorship is None
        or authorship.get("disposition") != "exclude_authorship_mismatch"
        or source_quality is None
        or source_quality.get("disposition") != "exclude_source_quality"
        or umbrella is None
        or umbrella.get("disposition") != "exclude_collection_umbrella"
    ):
        raise R1ManifestBuilderError(
            "required R1 exclusion dispositions are incomplete"
        )

    included = tuple(
        sorted((*wikisource.work_ids, feb.work_id, *reviewed.work_ids))
    )
    if len(included) != EXPECTED_INCLUDED_WORK_COUNT:
        raise R1ManifestBuilderError(
            "R1 provider union is not the exact 134-work inventory"
        )
    manifest = R1AcquisitionManifest.build(
        wikisource_campaign=wikisource,
        wikisource_discovery_candidate_sha256=candidate.candidate_hash,
        source_curation_receipt_sha256=DISPOSITIONS_SELF_HASH,
        feb_work_spec=feb,
        reviewed_text_campaign=reviewed,
        included_work_ids=included,
        collection_umbrella_evidence_sha256=DISPOSITIONS_SELF_HASH,
        authorship_mismatch_evidence_sha256=DISPOSITIONS_SELF_HASH,
        authorship_mismatch_receipt_sha256=str(
            authorship.get("rejected_work_receipt_self_hash")
        ),
        source_quality_rejected_evidence_sha256=str(
            source_quality.get("evidence_file_sha256")
        ),
        source_quality_rejected_receipt_sha256=str(
            source_quality.get("evidence_self_hash")
        ),
    )
    if (
        manifest.schema_version
        != R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V3
        or len(manifest.included_work_ids) != EXPECTED_INCLUDED_WORK_COUNT
    ):
        raise R1ManifestBuilderError(
            "generated R1 acquisition manifest is not canonical v3"
        )
    return manifest
