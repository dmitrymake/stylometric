#!/usr/bin/env python3
"""Build and materialize the exact pinned hybrid RuAA R1 corpus."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.artifacts import (  # noqa: E402
    build_ruaa_r1_acquisition_manifest as manifest_builder,
)
from scripts.artifacts import (  # noqa: E402
    build_ruaa_r1_wikisource_candidate as candidate_builder,
)
from stylo import jsonio as jsonio_module  # noqa: E402
from stylo.corpus_tools import (  # noqa: E402
    feb_vnext as feb_module,
    reviewed_text_vnext as reviewed_module,
    ruaa_r1_acquisition as acquisition_module,
    ruaa_r1_disk_backed_audit as disk_audit,
    text_quality_vnext as text_quality_module,
    wikisource_campaign as campaign_module,
    wikisource_discovery as discovery_module,
    wikisource_vnext as wikisource_module,
)
from stylo.domain import corpus_identity as corpus_identity_module  # noqa: E402
from stylo.corpus_tools.feb_vnext import (  # noqa: E402
    FEB_CONTENT_TYPE,
    FEB_SOURCE_URL,
    FEBHTTPResponse,
)
from stylo.corpus_tools.ruaa_r1_acquisition import (  # noqa: E402
    ACQUISITION_RECEIPT_NAME,
    AUDIT_REPORT_NAME,
    R1_AUTHORSHIP_MISMATCH_WORK_ID,
    R1_SOURCE_QUALITY_REJECTED_WORK_ID,
    R1AcquisitionAuditError,
    R1AcquisitionError,
    materialize_r1_acquisition,
)
from stylo.corpus_tools.wikisource_discovery import (  # noqa: E402
    _read_cached_response,
    _request_parameters,
    _validate_cache_directory,
)
from stylo.jsonio import (  # noqa: E402
    artifact_self_hash,
    dump_strict,
    dumps_strict,
    loads_strict,
)


BASE_COMMIT = "eebc1de4c07da6bccfd048b016b98c729cbfec15"
CONTRACT_NAME = "ruaa_r1_corpus_contract_execution_v1.json"
REVIEWED_CAMPAIGN_NAME = "ruaa_r1_reviewed_text_campaign_v1.json"
CORPUS_SOURCES_ROOT = ROOT / "research" / "corpus_sources"
REVIEWED_CAMPAIGN_PATH = CORPUS_SOURCES_ROOT / REVIEWED_CAMPAIGN_NAME
REVIEWED_PROVENANCE_PATH = (
    CORPUS_SOURCES_ROOT / "ruaa_r1_reviewed_text_provenance_v1.json"
)
FULL_INVENTORY_NAME = "full-136-inventory.json"
SELECTED_INVENTORY_NAME = "selected-134-inventory.json"
EXCLUDED_EVIDENCE_DIRECTORY_NAME = "excluded-evidence"
EXPECTED_SELECTED_AUDIT_FILE_SHA256 = (
    "a6887053a928a687c4fc12607515fdb10a5aa99d3912e054d14edc5410e5408b"
)
EXPECTED_SELECTED_AUDIT_SELF_HASH = (
    "233326b39ef7dcf3593a3bd5607ef64dde72620d6cbe20f2a5f786cf32440780"
)
EXCLUDED_EVIDENCE_TEXTS = {
    R1_AUTHORSHIP_MISMATCH_WORK_ID: {
        "byte_size": 17979,
        "sha256": (
            "6d4a8b820bc75d0c12d72528086cde792dbb9d7e68cbafcf53f213c1076aef5f"
        ),
    },
    R1_SOURCE_QUALITY_REJECTED_WORK_ID: {
        "byte_size": 186486,
        "sha256": (
            "25e81281db93ad39fcc9d585ac765722dd72506ff5258739a64242c7c40b9ae6"
        ),
    },
}
EXPECTED_SOURCE_WORK_COUNT = 136
EXPECTED_SOURCE_PART_COUNT = 1360
EXPECTED_WIKISOURCE_WORK_COUNT = 127
EXPECTED_WIKISOURCE_PART_COUNT = 1345
EXPECTED_REVIEWED_WORK_COUNT = 6
EXPECTED_FEB_WORK_COUNT = 1
EXPECTED_SELECTED_WORK_COUNT = 134
EXPECTED_FULL_WORK_COUNT = 136
EXPECTED_WIKISOURCE_CACHE_ENTRY_COUNT = 2690
EXPECTED_WIKISOURCE_CACHE_QUERY_COUNT = 1345
EXPECTED_WIKISOURCE_CACHE_PARSE_COUNT = 1345
EXPECTED_PARSE_AUDIT_FILE_SHA256 = (
    "ee7e9dcdd466dfdda334df7ef33f7b911b5ab9edb7cb3227deb3ff290ead4177"
)
EXPECTED_PARSE_AUDIT_SELF_HASH = (
    "a7630ae7c923769fc13d002363e9744b201efbfaa55b8ffe66c117e6b0d8d6b0"
)
EXPECTED_REVIEWED_CAMPAIGN_FILE_SHA256 = (
    "6ec111c42943412bcefaab0a3b4557973c5df7cede09a8fba1fe9916065eb220"
)
EXPECTED_REVIEWED_CAMPAIGN_SELF_HASH = (
    "c87deecb01ea9db922e02305efee9cfda9c76fe4e4e03d38e28dc6438eb63f7f"
)
EXPECTED_READY_CANDIDATE_FILE_SHA256 = (
    "236dfe586c6307914ec3f47ee85f9be002ab9d04bb463e0d49079076a477be00"
)
EXPECTED_READY_CANDIDATE_HASH = (
    "1a08416984adb84dcdf4523f32cff1c4a51d1c9cca9efc64c721210f1beba531"
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "65657c7b206475709de4c1675ba2c24e9084caed82bffb5bb7b3aca2306995e5"
)
EXPECTED_MANIFEST_SELF_HASH = (
    "7f6f8efba31c1c99d7b124708bb5331d5e5803a3ad9ebbafae21a6959646f209"
)
EXPECTED_MANIFEST_GENERATION_ID = (
    "7a930a56390ff8e310bfba75e35d028c3f260a2311a1f469dc687d235923ce4c"
)
EXPECTED_FEB_RESPONSE_FILE_SHA256 = (
    "8016f5313a561f90c112836d46831ba16ac0e75990e2ea691ebf263a8cb94684"
)


class R1AcquisitionCLIError(ValueError):
    """The hybrid acquisition CLI rejected an unsafe request."""


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise R1AcquisitionCLIError(
                f"{label} must not contain symlink components: {component}"
            )


def _require_file(path: pathlib.Path, *, label: str) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    _reject_symlink_components(candidate, label=label)
    if candidate.is_symlink() or not candidate.is_file():
        raise R1AcquisitionCLIError(
            f"{label} must be a regular non-symlink file: {candidate}"
        )
    return candidate.resolve(strict=True)


def _require_directory(path: pathlib.Path, *, label: str) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    _reject_symlink_components(candidate, label=label)
    if candidate.is_symlink() or not candidate.is_dir():
        raise R1AcquisitionCLIError(
            f"{label} must be an existing real directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def _prepare_output_parent(path: pathlib.Path) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    _reject_symlink_components(candidate, label="R1 replay output parent")
    if candidate.exists() and (
        candidate.is_symlink() or not candidate.is_dir()
    ):
        raise R1AcquisitionCLIError(
            "R1 replay output parent must be a real directory"
        )
    candidate.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(candidate, label="R1 replay output parent")
    return candidate.resolve(strict=True)


class _PinnedCacheTransport:
    """Closed transport over one validated official pinning cache."""

    def __init__(self, cache: pathlib.Path) -> None:
        self._cache = _validate_cache_directory(
            _require_directory(cache, label="Wikisource pinning cache")
        )
        names = tuple(sorted(path.name for path in self._cache.iterdir()))
        self.entry_count = len(names)
        self.query_count = sum(name.startswith("query-") for name in names)
        self.parse_count = sum(name.startswith("parse-") for name in names)
        self.inventory_sha256 = _sha256_bytes(
            ("\n".join(names) + "\n").encode("utf-8")
        )
        self.call_count = 0

    def __call__(self, params: Mapping[str, str]) -> object:
        if type(params) is not dict or any(
            type(key) is not str or type(value) is not str
            for key, value in params.items()
        ):
            raise R1AcquisitionCLIError(
                "closed Wikisource cache requires an exact str:str request"
            )
        action = params.get("action")
        if action == "query":
            request_kind, revision_key = "query", "revids"
        elif action == "parse":
            request_kind, revision_key = "parse", "oldid"
        else:
            raise R1AcquisitionCLIError(
                f"closed Wikisource cache rejects action {action!r}"
            )
        try:
            revision_id = int(params.get(revision_key, ""), 10)
        except ValueError as exc:
            raise R1AcquisitionCLIError(
                "closed Wikisource cache revision is invalid"
            ) from exc
        expected = _request_parameters(request_kind, revision_id)
        if params != expected:
            raise R1AcquisitionCLIError(
                "closed Wikisource cache request differs from the pinned "
                f"contract: {request_kind}/{revision_id}"
            )
        response = _read_cached_response(
            self._cache,
            request_kind=request_kind,
            revision_id=revision_id,
            request=expected,
        )
        if response is None:
            raise R1AcquisitionCLIError(
                f"OFFLINE_{request_kind.upper()}_CACHE_MISS "
                f"revision {revision_id}"
            )
        self.call_count += 1
        return response


class _FEBFileTransport:
    """Closed FEB transport over one exact captured response body."""

    def __init__(self, path: pathlib.Path) -> None:
        source = _require_file(path, label="captured FEB response")
        self._payload = source.read_bytes()
        self.logical_name = source.name
        self.byte_size = len(self._payload)
        self.file_sha256 = _sha256_bytes(self._payload)
        if self.file_sha256 != EXPECTED_FEB_RESPONSE_FILE_SHA256:
            raise R1AcquisitionCLIError(
                "captured FEB response file SHA-256 mismatch"
            )
        self.call_count = 0

    def __call__(self, url: str) -> FEBHTTPResponse:
        if url != FEB_SOURCE_URL:
            raise R1AcquisitionCLIError(
                f"closed FEB transport rejects URL {url!r}"
            )
        self.call_count += 1
        return FEBHTTPResponse(
            FEB_SOURCE_URL,
            200,
            FEB_CONTENT_TYPE,
            self._payload,
        )


def _replay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{pathlib.Path(__file__).name} replay",
        description=(
            "Copy the tracked reviewed campaign, then build the ready "
            "candidate, acquisition manifest, canonical offline "
            "materialization, bounded full audit, and receipt-derived "
            "execution contract in that order."
        ),
    )
    parser.add_argument(
        "--artifact-cache",
        required=True,
        type=pathlib.Path,
        help="content-addressed reviewed-text artifact cache",
    )
    parser.add_argument(
        "--source-candidate",
        required=True,
        type=pathlib.Path,
        help="frozen 136-work source discovery candidate",
    )
    parser.add_argument(
        "--wikisource-campaign",
        required=True,
        type=pathlib.Path,
        help="pinned 127-work Wikisource campaign",
    )
    parser.add_argument(
        "--wikisource-cache",
        required=True,
        type=pathlib.Path,
        help="official query/parse cache for the pinned campaign",
    )
    parser.add_argument(
        "--feb-response",
        required=True,
        type=pathlib.Path,
        help="captured exact FEB HTTP response body",
    )
    parser.add_argument(
        "--excluded-evidence-root",
        required=True,
        type=pathlib.Path,
        help=(
            "directory holding the two hash-bound excluded evidence texts "
            "as <work_id>.txt"
        ),
    )
    parser.add_argument(
        "--parse-audit",
        required=True,
        type=pathlib.Path,
        help="1360-part pinned parse audit report",
    )
    parser.add_argument(
        "--output-parent",
        required=True,
        type=pathlib.Path,
        help="ignored workspace for generated contracts and acquisition",
    )
    parser.add_argument(
        "--scratch-parent",
        required=True,
        type=pathlib.Path,
        help="existing real directory for bounded shingle arrays",
    )
    return parser


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(
    path: pathlib.Path,
    *,
    label: str,
) -> tuple[pathlib.Path, bytes, dict[str, Any]]:
    source = _require_file(path, label=label)
    payload = source.read_bytes()
    try:
        value = loads_strict(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise R1AcquisitionCLIError(
            f"{label} is not strict UTF-8 JSON: {exc}"
        ) from exc
    if type(value) is not dict:
        raise R1AcquisitionCLIError(f"{label} must be an exact JSON object")
    return source, payload, value


def _require_json_identity(
    path: pathlib.Path,
    *,
    label: str,
    file_sha256: str,
    self_hash: str,
) -> tuple[pathlib.Path, bytes, dict[str, Any]]:
    source, payload, value = _read_json_object(path, label=label)
    observed_file_hash = _sha256_bytes(payload)
    if observed_file_hash != file_sha256:
        raise R1AcquisitionCLIError(
            f"{label} file SHA-256 mismatch: {observed_file_hash}"
        )
    if (
        value.get("self_hash") != self_hash
        or artifact_self_hash(value) != self_hash
    ):
        raise R1AcquisitionCLIError(f"{label} self-hash mismatch")
    return source, payload, value


def _artifact_ref(
    path: pathlib.Path,
    *,
    logical_name: str,
    expected_document: Mapping[str, object],
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = _require_file(path, label=logical_name)
    payload = source.read_bytes()
    try:
        observed = loads_strict(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise R1AcquisitionCLIError(
            f"{logical_name} is not strict UTF-8 JSON: {exc}"
        ) from exc
    expected_tree = dumps_strict(
        dict(expected_document),
        sort_keys=True,
        separators=(",", ":"),
    )
    observed_tree = dumps_strict(
        observed,
        sort_keys=True,
        separators=(",", ":"),
    )
    if observed_tree != expected_tree:
        raise R1AcquisitionCLIError(
            f"{logical_name} differs from its in-memory receipt/report"
        )
    result: dict[str, object] = {
        "logical_name": logical_name,
        "byte_size": len(payload),
        "file_sha256": _sha256_bytes(payload),
    }
    if details is not None:
        result.update(details)
    return result


def _dump_exact(
    value: Mapping[str, object],
    path: pathlib.Path,
    *,
    label: str,
    expected_file_sha256: str,
) -> pathlib.Path:
    dump_strict(value, path, sort_keys=True, trailing_newline=True)
    payload = path.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_file_sha256:
        raise R1AcquisitionCLIError(
            f"{label} file SHA-256 mismatch: {observed}"
        )
    return path


def _dump_generated(
    value: Mapping[str, object],
    path: pathlib.Path,
) -> pathlib.Path:
    """Write one generated artifact into the ignored replay workspace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dump_strict(value, path, sort_keys=True, trailing_newline=True)
    return path


def _relative_to_root(path: pathlib.Path) -> str:
    if not path.is_relative_to(ROOT):
        raise R1AcquisitionCLIError(
            f"contract path is outside the repository: {path}"
        )
    return path.relative_to(ROOT).as_posix()


def _work_relative_parts(work_id: str) -> tuple[str, ...]:
    relative = pathlib.PurePosixPath(f"{work_id}.txt")
    if (
        relative.is_absolute()
        or relative.as_posix() != f"{work_id}.txt"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise R1AcquisitionCLIError(f"noncanonical work id: {work_id!r}")
    return relative.parts


def _load_tracked_reviewed_campaign(
    path: pathlib.Path,
) -> tuple[object, bytes]:
    """Load the tracked reviewed-text campaign under its pinned identity.

    The reviewed provider consumes finished hash-bound texts, so the campaign
    is an input of the replay rather than something it recomputes: the exact
    tracked bytes, the strict schema, the recorded self-hash, and the pinned
    file SHA-256 all have to agree before anything downstream reads it.
    """

    source = _require_file(path, label="tracked reviewed campaign")
    payload = source.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != EXPECTED_REVIEWED_CAMPAIGN_FILE_SHA256:
        raise R1AcquisitionCLIError(
            f"tracked reviewed campaign file SHA-256 mismatch: {observed}"
        )
    try:
        spec = reviewed_module.loads_reviewed_text_campaign_spec(
            payload.decode("utf-8")
        )
    except (
        UnicodeError,
        reviewed_module.ReviewedTextMaterializationError,
    ) as exc:
        raise R1AcquisitionCLIError(
            f"tracked reviewed campaign is invalid: {exc}"
        ) from exc
    if spec.self_hash != EXPECTED_REVIEWED_CAMPAIGN_SELF_HASH:
        raise R1AcquisitionCLIError(
            "tracked reviewed campaign self-hash mismatch"
        )
    return spec, payload


def _executable_source_refs() -> list[dict[str, object]]:
    modules = (
        candidate_builder,
        manifest_builder,
        disk_audit,
        jsonio_module,
        feb_module,
        reviewed_module,
        acquisition_module,
        text_quality_module,
        campaign_module,
        discovery_module,
        wikisource_module,
        corpus_identity_module,
    )
    paths = {pathlib.Path(__file__).resolve(strict=True)}
    for module in modules:
        source_name = inspect.getsourcefile(module)
        if source_name is None:
            raise R1AcquisitionCLIError(
                f"cannot locate executable source for {module.__name__}"
            )
        paths.add(pathlib.Path(source_name).resolve(strict=True))
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        if not path.is_relative_to(ROOT):
            raise R1AcquisitionCLIError(
                f"executable source is outside the repository: {path}"
            )
        payload = path.read_bytes()
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "byte_size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def _stage_pinned_payload(
    payload: bytes,
    target: pathlib.Path,
    *,
    sha256: str,
    label: str,
) -> None:
    """Stage one pinned payload without ever clobbering a target.

    A missing target is created with ``O_EXCL | O_NOFOLLOW``, so a planted
    symlink or an existing file at the final component fails instead of
    redirecting the write.  An existing target is never overwritten: it is read
    back through the bounded no-follow reader and accepted only on an exact
    size, SHA-256, and byte match, which is what an exact resume observes.

    Scope: the ancestor directories are checked by path, not traversed by
    descriptor, so this rejects planted symlinks and mismatched targets but
    does not exclude a concurrent same-user replacement of an ancestor between
    the check and the open.  The replay writes only into its own
    ``--output-parent``, and that residual race is not defended against here.
    """

    _reject_symlink_components(target, label=label)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(target, label=label)
        try:
            descriptor = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
            )
        except OSError as exc:
            raise R1AcquisitionCLIError(
                f"{label} could not be created exclusively: {exc}"
            ) from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
    try:
        staged = disk_audit._read_stable_regular(
            target,
            label=label,
            expected_size=len(payload),
            expected_sha256=sha256,
        )
    except disk_audit.DiskBackedAuditError as exc:
        raise R1AcquisitionCLIError(
            f"{label} is unsafe or drifted: {exc}"
        ) from exc
    if staged.payload != payload:
        raise R1AcquisitionCLIError(f"{label} differs from the pinned bytes")


def _excluded_evidence_rows(
    source_root: pathlib.Path,
    *,
    destination_root: pathlib.Path,
    output_parent: pathlib.Path,
) -> list[dict[str, object]]:
    """Verify and stage the two hash-bound excluded evidence texts.

    Each text is checked against its frozen byte size and SHA-256 before it is
    staged inside the replay workspace, so the audited full-136 inventory never
    depends on a previously materialized tree.
    """

    rows: list[dict[str, object]] = []
    for work_id in sorted(EXCLUDED_EVIDENCE_TEXTS):
        expected = EXCLUDED_EVIDENCE_TEXTS[work_id]
        parts = _work_relative_parts(work_id)
        source = _require_file(
            source_root.joinpath(*parts),
            label=f"excluded evidence text {work_id}",
        )
        payload = source.read_bytes()
        observed = _sha256_bytes(payload)
        if (
            len(payload) != expected["byte_size"]
            or observed != expected["sha256"]
        ):
            raise R1AcquisitionCLIError(
                f"excluded evidence text {work_id} identity mismatch: "
                f"byte_size={len(payload)}, sha256={observed}"
            )
        target = destination_root.joinpath(*parts)
        _stage_pinned_payload(
            payload,
            target,
            sha256=observed,
            label=f"staged excluded evidence text {work_id}",
        )
        rows.append(
            {
                "work_id": work_id,
                "path": target.relative_to(output_parent).as_posix(),
                "byte_size": len(payload),
                "sha256": observed,
            }
        )
    return rows


def _write_inventory(
    rows: Sequence[Mapping[str, object]],
    path: pathlib.Path,
    *,
    label: str,
    expected_work_count: int,
) -> disk_audit.LoadedInventory:
    """Write and re-read one generated production inventory."""

    ordered = sorted(rows, key=lambda row: str(row["work_id"]))
    if len(ordered) != expected_work_count:
        raise R1AcquisitionCLIError(
            f"{label} work count is {len(ordered)}, "
            f"expected {expected_work_count}"
        )
    core = {
        "schema_version": disk_audit.INPUT_INVENTORY_SCHEMA,
        "works": [dict(row) for row in ordered],
    }
    _dump_generated({**core, "self_hash": artifact_self_hash(core)}, path)
    try:
        inventory = disk_audit.load_input_inventory(path)
    except disk_audit.DiskBackedAuditError as exc:
        raise R1AcquisitionCLIError(f"{label} is invalid: {exc}") from exc
    if len(inventory.entries) != expected_work_count:
        raise R1AcquisitionCLIError(f"{label} did not round-trip exactly")
    return inventory


def _build_inventories(
    *,
    materialized: object,
    manifest: object,
    output_parent: pathlib.Path,
    excluded_evidence_root: pathlib.Path,
) -> tuple[
    disk_audit.LoadedInventory,
    disk_audit.LoadedInventory,
    tuple[str, ...],
]:
    """Derive both audit inventories from this run's materialized output."""

    root = getattr(materialized, "root")
    if not root.is_relative_to(output_parent):
        raise R1AcquisitionCLIError(
            "materialized acquisition is outside the replay output parent"
        )
    receipt_rows = {
        row.work_id: row
        for row in getattr(getattr(materialized, "receipt"), "raw_inventory")
    }
    manifest_ids = tuple(getattr(manifest, "included_work_ids"))
    if (
        tuple(sorted(receipt_rows)) != manifest_ids
        or len(manifest_ids) != EXPECTED_SELECTED_WORK_COUNT
    ):
        raise R1AcquisitionCLIError(
            "acquisition receipt inventory differs from the manifest"
        )
    selected_rows: list[dict[str, object]] = []
    for work_id in manifest_ids:
        row = receipt_rows[work_id]
        text = root.joinpath("raw", *_work_relative_parts(work_id))
        selected_rows.append(
            {
                "work_id": work_id,
                "path": text.relative_to(output_parent).as_posix(),
                "byte_size": row.byte_size,
                "sha256": row.sha256,
            }
        )
    excluded_rows = _excluded_evidence_rows(
        excluded_evidence_root,
        destination_root=output_parent / EXCLUDED_EVIDENCE_DIRECTORY_NAME,
        output_parent=output_parent,
    )
    exclusions = tuple(sorted(str(row["work_id"]) for row in excluded_rows))
    if exclusions != tuple(
        sorted(
            (
                R1_AUTHORSHIP_MISMATCH_WORK_ID,
                R1_SOURCE_QUALITY_REJECTED_WORK_ID,
            )
        )
    ) or set(exclusions) & set(manifest_ids):
        raise R1AcquisitionCLIError(
            "excluded evidence texts are not the two frozen exclusions"
        )
    selected = _write_inventory(
        selected_rows,
        output_parent / SELECTED_INVENTORY_NAME,
        label="selected inventory",
        expected_work_count=EXPECTED_SELECTED_WORK_COUNT,
    )
    full = _write_inventory(
        [*selected_rows, *excluded_rows],
        output_parent / FULL_INVENTORY_NAME,
        label="full inventory",
        expected_work_count=EXPECTED_FULL_WORK_COUNT,
    )
    selected_ids = tuple(selected.entries)
    full_ids = tuple(full.entries)
    if (
        selected_ids != manifest_ids
        or tuple(work for work in full_ids if work not in set(exclusions))
        != selected_ids
        or set(full_ids) - set(selected_ids) != set(exclusions)
    ):
        raise R1AcquisitionCLIError(
            "full inventory is not selected plus the two exclusions"
        )
    for work_id in selected_ids:
        entry = full.entries[work_id]
        receipt_row = receipt_rows[work_id]
        if (
            entry != selected.entries[work_id]
            or entry.byte_size != receipt_row.byte_size
            or entry.sha256 != receipt_row.sha256
            or not entry.path.is_relative_to(root)
        ):
            raise R1AcquisitionCLIError(
                f"{work_id}: inventory differs from this run's acquisition"
            )
    return full, selected, exclusions


def _parse_audit_ref(path: pathlib.Path) -> dict[str, object]:
    source, payload, report = _require_json_identity(
        path,
        label="1360-part parse audit",
        file_sha256=EXPECTED_PARSE_AUDIT_FILE_SHA256,
        self_hash=EXPECTED_PARSE_AUDIT_SELF_HASH,
    )
    summary = report.get("summary")
    if type(summary) is not dict:
        raise R1AcquisitionCLIError("parse audit summary is malformed")
    expected = {
        "work_count": EXPECTED_SOURCE_WORK_COUNT,
        "part_count": EXPECTED_SOURCE_PART_COUNT,
        "distinct_parsed_revision_count": EXPECTED_SOURCE_PART_COUNT,
        "oldid_query_error_count": 0,
        "parse_error_count": 0,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise R1AcquisitionCLIError("parse audit inventory drifted")
    return {
        "logical_name": source.name,
        "byte_size": len(payload),
        "file_sha256": EXPECTED_PARSE_AUDIT_FILE_SHA256,
        "self_hash": EXPECTED_PARSE_AUDIT_SELF_HASH,
        "status": report.get("status"),
        **expected,
    }


def _inventory_ref(
    inventory: disk_audit.LoadedInventory,
    *,
    logical_name: str,
) -> dict[str, object]:
    return {
        "logical_name": logical_name,
        "byte_size": inventory.byte_size,
        "file_sha256": inventory.sha256,
        "self_hash": inventory.self_hash,
        "work_count": len(inventory.entries),
    }


def _audit_ref(
    path: pathlib.Path,
    report: object,
    *,
    logical_name: str,
) -> dict[str, object]:
    payload = getattr(report, "to_dict")()
    result = _artifact_ref(
        path,
        logical_name=logical_name,
        expected_document=payload,
        details={
            "self_hash": getattr(report, "self_hash"),
            "status": payload["status"],
            "work_count": payload["work_count"],
            "total_word_count": payload["total_word_count"],
            "blocking_finding_count": len(payload["blocking_findings"]),
            "cross_work_overlap_count": len(
                payload["cross_work_overlaps"]
            ),
        },
    )
    if result["self_hash"] != artifact_self_hash(payload):
        raise R1AcquisitionCLIError(
            f"{logical_name} has an invalid self-hash"
        )
    return result


def _tracked_contract_refs() -> list[dict[str, object]]:
    paths = (
        REVIEWED_CAMPAIGN_PATH,
        REVIEWED_PROVENANCE_PATH,
        candidate_builder.DISPOSITIONS_PATH,
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        source, payload, value = _read_json_object(
            path,
            label=f"tracked contract {path.name}",
        )
        self_hash = value.get("self_hash")
        if (
            type(self_hash) is not str
            or artifact_self_hash(value) != self_hash
        ):
            raise R1AcquisitionCLIError(
                f"tracked contract {path.name} has an invalid self-hash"
            )
        rows.append(
            {
                "relative_path": _relative_to_root(source),
                "byte_size": len(payload),
                "file_sha256": _sha256_bytes(payload),
                "self_hash": self_hash,
            }
        )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def _build_execution_contract(
    *,
    reviewed_campaign_path: pathlib.Path,
    reviewed_campaign: object,
    ready_candidate_path: pathlib.Path,
    ready_candidate: object,
    ready_candidate_document: Mapping[str, object],
    wikisource_campaign_path: pathlib.Path,
    manifest_path: pathlib.Path,
    manifest: object,
    materialized: object,
    full_inventory: disk_audit.LoadedInventory,
    selected_inventory: disk_audit.LoadedInventory,
    parse_audit: Mapping[str, object],
    full_audit_path: pathlib.Path,
    full_audit: object,
    wikisource_transport: _PinnedCacheTransport,
    feb_transport: _FEBFileTransport,
    executable_sources: Sequence[Mapping[str, object]],
    output_parent: pathlib.Path,
) -> tuple[dict[str, object], str]:
    receipt_path = materialized.root / ACQUISITION_RECEIPT_NAME
    selected_audit_path = materialized.root / AUDIT_REPORT_NAME
    receipt = materialized.receipt
    receipt_details = _artifact_ref(
        receipt_path,
        logical_name=ACQUISITION_RECEIPT_NAME,
        expected_document=receipt.to_dict(),
        details={
            "schema_version": receipt.schema_version,
            "self_hash": receipt.self_hash,
            "manifest_sha256": receipt.manifest_sha256,
            "generation_id": receipt.generation_id,
            "wikisource_campaign_spec_sha256": (
                receipt.wikisource_campaign_spec_sha256
            ),
            "wikisource_campaign_receipt_sha256": (
                receipt.wikisource_campaign_receipt_sha256
            ),
            "feb_work_spec_sha256": receipt.feb_work_spec_sha256,
            "feb_work_receipt_sha256": receipt.feb_work_receipt_sha256,
            "reviewed_text_campaign_spec_sha256": (
                receipt.reviewed_text_campaign_spec_sha256
            ),
            "reviewed_text_campaign_receipt_sha256": (
                receipt.reviewed_text_campaign_receipt_sha256
            ),
            "text_quality_audit_sha256": (
                receipt.text_quality_audit_sha256
            ),
            "work_count": len(receipt.raw_inventory),
        },
    )
    reviewed_ref = _artifact_ref(
        reviewed_campaign_path,
        logical_name=reviewed_campaign_path.name,
        expected_document=reviewed_campaign.to_dict(),
        details={
            "self_hash": reviewed_campaign.self_hash,
            "work_count": len(reviewed_campaign.works),
        },
    )
    candidate_ref = _artifact_ref(
        ready_candidate_path,
        logical_name=ready_candidate_path.name,
        expected_document=ready_candidate_document,
        details={
            "candidate_hash": ready_candidate.candidate_hash,
            "work_count": len(ready_candidate.works),
            "part_count": sum(
                len(work.parts) for work in ready_candidate.works
            ),
            "selected_work_count": sum(
                work.include_in_corpus for work in ready_candidate.works
            ),
        },
    )
    pinned_source, pinned_payload, pinned = _read_json_object(
        wikisource_campaign_path,
        label="pinned Wikisource campaign",
    )
    pinned_works = pinned.get("works")
    if type(pinned_works) is not list:
        raise R1AcquisitionCLIError(
            "pinned Wikisource campaign works are malformed"
        )
    pinned_ref = {
        "logical_name": pinned_source.name,
        "byte_size": len(pinned_payload),
        "file_sha256": _sha256_bytes(pinned_payload),
        "self_hash": pinned.get("self_hash"),
        "generation_id": pinned.get("generation_id"),
        "work_count": len(pinned_works),
        "part_count": sum(
            len(work.get("parts", []))
            for work in pinned_works
            if type(work) is dict
        ),
    }
    manifest_ref = _artifact_ref(
        manifest_path,
        logical_name=manifest_path.name,
        expected_document=manifest.to_dict(),
        details={
            "self_hash": manifest.self_hash,
            "generation_id": manifest.generation_id,
            "included_work_count": len(manifest.included_work_ids),
        },
    )
    feb_spec_ref = _artifact_ref(
        manifest_builder.FEB_SPEC_PATH,
        logical_name=manifest_builder.FEB_SPEC_PATH.name,
        expected_document=manifest.feb_work_spec.to_dict(),
        details={
            "self_hash": manifest.feb_work_spec.self_hash,
            "work_id": manifest.feb_work_spec.work_id,
            "output_sha256": manifest.feb_work_spec.output_sha256,
        },
    )
    full_audit_ref = _audit_ref(
        full_audit_path,
        full_audit,
        logical_name=full_audit_path.name,
    )
    selected_audit_ref = _audit_ref(
        selected_audit_path,
        materialized.audit_report,
        logical_name=AUDIT_REPORT_NAME,
    )
    quality_gate = {
        "gating_inventory": "selected_134",
        "selected_passed": (
            selected_audit_ref["status"] == "passed"
            and selected_audit_ref["blocking_finding_count"] == 0
            and selected_audit_ref["cross_work_overlap_count"] == 0
        ),
        "full_inventory_report_is_evidence_only": True,
        "full_cross_work_isolated": (
            full_audit_ref["cross_work_overlap_count"] == 0
        ),
    }
    # The full-136 report is a diagnostic over the two excluded evidence
    # texts as well, so its own status and blocking-finding count are
    # recorded but never gate the execution status.
    failed = tuple(
        name
        for name in ("selected_passed", "full_cross_work_isolated")
        if not quality_gate[name]
    )
    if failed:
        raise R1AcquisitionCLIError(
            f"declared R1 quality gates failed: {', '.join(failed)}"
        )
    contract: dict[str, object] = {
        "schema_version": "stylo.ruaa-r1.corpus-contract-execution.v1",
        "base_commit": BASE_COMMIT,
        "status": "materialized_audit_passed",
        "publication_authorized": False,
        "controls": {
            "network_used": False,
            "corpus_workflow_models_or_lobo_executed": False,
            "scope": (
                "reviewed campaign -> ready candidate -> acquisition "
                "manifest -> canonical materialization -> audit -> contract"
            ),
        },
        "expected_inventory": {
            "source_work_count": EXPECTED_SOURCE_WORK_COUNT,
            "source_part_count": EXPECTED_SOURCE_PART_COUNT,
            "wikisource_work_count": EXPECTED_WIKISOURCE_WORK_COUNT,
            "wikisource_part_count": EXPECTED_WIKISOURCE_PART_COUNT,
            "reviewed_text_work_count": EXPECTED_REVIEWED_WORK_COUNT,
            "feb_work_count": EXPECTED_FEB_WORK_COUNT,
            "selected_work_count": EXPECTED_SELECTED_WORK_COUNT,
            "full_work_count": EXPECTED_FULL_WORK_COUNT,
        },
        "quality_gate": quality_gate,
        "generated_contracts": {
            "reviewed_campaign": reviewed_ref,
            "ready_wikisource_candidate": candidate_ref,
            "pinned_wikisource_campaign": pinned_ref,
            "pinned_feb_work": feb_spec_ref,
            "acquisition_manifest": manifest_ref,
        },
        "acquisition_receipt": receipt_details,
        "transport_inputs": {
            "wikisource_cache": {
                "entry_count": wikisource_transport.entry_count,
                "query_count": wikisource_transport.query_count,
                "parse_count": wikisource_transport.parse_count,
                "filename_inventory_sha256": (
                    wikisource_transport.inventory_sha256
                ),
            },
            "feb_response": {
                "logical_name": feb_transport.logical_name,
                "byte_size": feb_transport.byte_size,
                "file_sha256": feb_transport.file_sha256,
            },
        },
        "audit_inputs": {
            "full_136": _inventory_ref(
                full_inventory,
                logical_name=FULL_INVENTORY_NAME,
            ),
            "selected_134": _inventory_ref(
                selected_inventory,
                logical_name=SELECTED_INVENTORY_NAME,
            ),
        },
        "text_audits": {
            "full_136": full_audit_ref,
            "selected_134": selected_audit_ref,
        },
        "parse_audit": dict(parse_audit),
        "tracked_contracts": _tracked_contract_refs(),
        "executable_sources": [dict(row) for row in executable_sources],
    }
    if _executable_source_refs() != contract["executable_sources"]:
        raise R1AcquisitionCLIError(
            "executable sources changed during canonical replay"
        )
    contract["self_hash"] = artifact_self_hash(contract)
    contract_path = _dump_generated(contract, output_parent / CONTRACT_NAME)
    stored = loads_strict(contract_path.read_text(encoding="utf-8"))
    if (
        stored != contract
        or type(stored) is not dict
        or artifact_self_hash(stored) != stored.get("self_hash")
    ):
        raise R1AcquisitionCLIError(
            "stored execution contract failed self-validation"
        )
    return contract, _sha256_bytes(contract_path.read_bytes())


def _run_replay(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    args = _replay_parser().parse_args(argv)
    stage = "input validation"
    try:
        artifact_cache = _require_directory(
            args.artifact_cache,
            label="R1 reviewed artifact cache",
        )
        source_candidate = _require_file(
            args.source_candidate,
            label="source discovery candidate",
        )
        wikisource_campaign = _require_file(
            args.wikisource_campaign,
            label="pinned Wikisource campaign",
        )
        wikisource_cache = _require_directory(
            args.wikisource_cache,
            label="Wikisource pinning cache",
        )
        feb_response = _require_file(
            args.feb_response,
            label="captured FEB response",
        )
        excluded_evidence_root = _require_directory(
            args.excluded_evidence_root,
            label="excluded evidence root",
        )
        parse_audit_path = _require_file(
            args.parse_audit,
            label="1360-part parse audit",
        )
        output_parent = _prepare_output_parent(args.output_parent)
        scratch_parent = _require_directory(
            args.scratch_parent,
            label="disk-audit scratch parent",
        )
        executable_sources = _executable_source_refs()

        stage = "reviewed campaign"
        reviewed_campaign, reviewed_campaign_payload = (
            _load_tracked_reviewed_campaign(REVIEWED_CAMPAIGN_PATH)
        )
        reviewed_campaign_path = output_parent / REVIEWED_CAMPAIGN_NAME
        _stage_pinned_payload(
            reviewed_campaign_payload,
            reviewed_campaign_path,
            sha256=EXPECTED_REVIEWED_CAMPAIGN_FILE_SHA256,
            label="staged reviewed campaign",
        )

        stage = "ready candidate"
        ready_candidate, ready_candidate_raw = (
            candidate_builder.build_candidate(
                source_candidate,
                reviewed_campaign_path=reviewed_campaign_path,
            )
        )
        ready_candidate_path = _dump_exact(
            ready_candidate_raw,
            output_parent / "ready-wikisource-candidate.json",
            label="ready Wikisource candidate",
            expected_file_sha256=EXPECTED_READY_CANDIDATE_FILE_SHA256,
        )
        if ready_candidate.candidate_hash != EXPECTED_READY_CANDIDATE_HASH:
            raise R1AcquisitionCLIError("ready candidate hash mismatch")

        stage = "acquisition manifest"
        manifest = manifest_builder.build_manifest(
            ready_candidate_path=ready_candidate_path,
            wikisource_campaign_path=wikisource_campaign,
            reviewed_campaign_path=reviewed_campaign_path,
        )
        manifest_path = _dump_exact(
            manifest.to_dict(),
            output_parent / "acquisition-manifest.json",
            label="R1 acquisition manifest",
            expected_file_sha256=EXPECTED_MANIFEST_FILE_SHA256,
        )
        if (
            manifest.self_hash != EXPECTED_MANIFEST_SELF_HASH
            or manifest.generation_id != EXPECTED_MANIFEST_GENERATION_ID
        ):
            raise R1AcquisitionCLIError(
                "R1 acquisition manifest identity mismatch"
            )

        stage = "canonical materialization"
        wikisource_transport = _PinnedCacheTransport(wikisource_cache)
        feb_transport = _FEBFileTransport(feb_response)
        if (
            wikisource_transport.entry_count
            != EXPECTED_WIKISOURCE_CACHE_ENTRY_COUNT
            or wikisource_transport.query_count
            != EXPECTED_WIKISOURCE_CACHE_QUERY_COUNT
            or wikisource_transport.parse_count
            != EXPECTED_WIKISOURCE_CACHE_PARSE_COUNT
        ):
            raise R1AcquisitionCLIError(
                "Wikisource cache inventory is not the exact "
                f"{EXPECTED_WIKISOURCE_CACHE_QUERY_COUNT}-part "
                "query/parse pair set"
            )
        materialized = materialize_r1_acquisition(
            manifest,
            output_parent=output_parent / "acquisition",
            wikisource_transport=wikisource_transport,
            feb_transport=feb_transport,
            reviewed_artifact_cache=artifact_cache,
        )
        selected_audit_path = materialized.root / AUDIT_REPORT_NAME
        selected_audit_payload = selected_audit_path.read_bytes()
        if (
            _sha256_bytes(selected_audit_payload)
            != EXPECTED_SELECTED_AUDIT_FILE_SHA256
            or materialized.audit_report.self_hash
            != EXPECTED_SELECTED_AUDIT_SELF_HASH
        ):
            raise R1AcquisitionCLIError(
                "selected-134 text-derived audit identity changed"
            )

        stage = "fresh inventory generation"
        full_inventory, selected_inventory, exclusions = _build_inventories(
            materialized=materialized,
            manifest=manifest,
            output_parent=output_parent,
            excluded_evidence_root=excluded_evidence_root,
        )
        parse_audit = _parse_audit_ref(parse_audit_path)

        stage = "disk-backed text audit"
        audit_result = disk_audit.audit_disk_backed(
            full_inventory.entries,
            selected_exclusions=exclusions,
            scratch_parent=scratch_parent,
            expected_full_count=EXPECTED_FULL_WORK_COUNT,
            expected_selected_count=EXPECTED_SELECTED_WORK_COUNT,
        )
        if (
            audit_result.selected_report.to_dict()
            != materialized.audit_report.to_dict()
        ):
            raise R1AcquisitionCLIError(
                "disk-backed selected audit differs from acquisition audit"
            )
        full_audit_path = _dump_generated(
            audit_result.full_report.to_dict(),
            output_parent / "full-corpus-text-quality-audit.json",
        )

        stage = "execution contract"
        contract, contract_file_sha256 = _build_execution_contract(
            reviewed_campaign_path=reviewed_campaign_path,
            reviewed_campaign=reviewed_campaign,
            ready_candidate_path=ready_candidate_path,
            ready_candidate=ready_candidate,
            ready_candidate_document=ready_candidate_raw,
            wikisource_campaign_path=wikisource_campaign,
            manifest_path=manifest_path,
            manifest=manifest,
            materialized=materialized,
            full_inventory=full_inventory,
            selected_inventory=selected_inventory,
            parse_audit=parse_audit,
            full_audit_path=full_audit_path,
            full_audit=audit_result.full_report,
            wikisource_transport=wikisource_transport,
            feb_transport=feb_transport,
            executable_sources=executable_sources,
            output_parent=output_parent,
        )
    except R1AcquisitionAuditError as exc:
        raise R1AcquisitionCLIError(
            f"{stage} blocked: {exc}; report={exc.report_path}"
        ) from exc
    except R1AcquisitionCLIError:
        raise
    except (OSError, UnicodeError, ValueError, R1AcquisitionError) as exc:
        raise R1AcquisitionCLIError(
            f"{stage} rejected: {exc}"
        ) from exc
    return {
        "status": contract["status"],
        "generation_id": manifest.generation_id,
        "acquisition_receipt_sha256": materialized.receipt.self_hash,
        "selected_audit_file_sha256": EXPECTED_SELECTED_AUDIT_FILE_SHA256,
        "selected_audit_self_hash": EXPECTED_SELECTED_AUDIT_SELF_HASH,
        "full_audit_file_sha256": _sha256_bytes(
            full_audit_path.read_bytes()
        ),
        "full_audit_self_hash": audit_result.full_report.self_hash,
        "full_audit_status": audit_result.full_report.status,
        "full_inventory_file_sha256": full_inventory.sha256,
        "full_inventory_self_hash": full_inventory.self_hash,
        "selected_inventory_file_sha256": selected_inventory.sha256,
        "selected_inventory_self_hash": selected_inventory.self_hash,
        "execution_contract_file_sha256": contract_file_sha256,
        "execution_contract_self_hash": contract["self_hash"],
        "work_count": len(manifest.included_work_ids),
        "wikisource_cache_call_count": wikisource_transport.call_count,
        "feb_response_call_count": feb_transport.call_count,
        "resumed": materialized.resumed,
        "publication_authorized": False,
    }


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        raise R1AcquisitionCLIError(
            "the offline replay subcommand is required"
        )
    if values[0] != "replay":
        raise R1AcquisitionCLIError(
            "only the offline replay subcommand is supported"
        )
    return _run_replay(values[1:])


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(argv)
    except R1AcquisitionCLIError as exc:
        print(f"R1 acquisition rejected: {exc}", file=sys.stderr)
        return 2
    print(dumps_strict(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
