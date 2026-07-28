"""Audit a materialized RuAA R1 corpus with bounded resident memory.

The registered text-quality policy accepts an in-memory mapping.  This helper
preserves its exact report payload while keeping only one work payload and
disk-backed word-5 shingle vectors resident at a time.  It performs no network
request, model construction, fitting, prediction, representation loading, or
artifact publication.

Production input is a compact, self-hashed inventory.  The inventory and every
text are each opened exactly once through an ``O_NOFOLLOW`` descriptor and
checked with ``fstat`` before and after the read.  Publication and execution
contract construction belong to the canonical R1 acquisition path.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import os
import pathlib
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from . import text_quality_vnext as tq
from ..domain import corpus_identity as ci
from ..jsonio import (
    StrictJSONError,
    canonical_hash,
    loads_strict,
)


INPUT_INVENTORY_SCHEMA = "stylo.disk-backed-corpus-audit.inventory.v1"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
INTERSECTION_CHUNK_SIZE = 65_536
READ_CHUNK_SIZE = 1024 * 1024
MAX_INVENTORY_BYTES = 16 * 1024 * 1024


class DiskBackedAuditError(ValueError):
    """The disk-backed audit or its input contract was violated."""


@dataclasses.dataclass(frozen=True)
class StableRead:
    payload: bytes
    byte_size: int
    sha256: str
    identity: tuple[int, int]


@dataclasses.dataclass(frozen=True)
class InventoryEntry:
    work_id: str
    path: pathlib.Path
    byte_size: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class LoadedInventory:
    entries: dict[str, InventoryEntry]
    byte_size: int
    sha256: str
    self_hash: str


@dataclasses.dataclass(frozen=True)
class ShingleArtifact:
    canonical_text_sha256: str
    path: pathlib.Path
    count: int


@dataclasses.dataclass(frozen=True)
class AuditResult:
    full_report: tq.CorpusTextAuditReport
    selected_report: tq.CorpusTextAuditReport
    full_work_count: int
    selected_work_count: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or HEX64_RE.fullmatch(value) is None:
        raise DiskBackedAuditError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DiskBackedAuditError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _work_id(value: object, label: str = "work_id") -> str:
    if type(value) is not str or not value or "\\" in value:
        raise DiskBackedAuditError(
            f"{label} must be a canonical POSIX work id"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DiskBackedAuditError(
            f"{label} must be a canonical author/work identifier"
        )
    return value


def _exact_object(
    value: object,
    keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        raise DiskBackedAuditError(f"{label} keys must be exact")
    return value


def _lexical_absolute(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _safe_open(path: pathlib.Path, *, directory: bool, label: str) -> int:
    """Open an absolute path without following any symlink component."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise DiskBackedAuditError(
            "this audit requires O_NOFOLLOW and O_DIRECTORY support"
        )
    absolute = _lexical_absolute(path)
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        raise DiskBackedAuditError(f"{label} has an invalid absolute path")
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    current = os.open("/", directory_flags)
    try:
        for part in parts[1:-1]:
            try:
                following = os.open(
                    part,
                    directory_flags,
                    dir_fd=current,
                )
            except OSError as exc:
                raise DiskBackedAuditError(
                    f"{label} has a missing or unsafe directory component"
                ) from exc
            os.close(current)
            current = following
        final_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        if directory:
            final_flags |= os.O_DIRECTORY
        try:
            return os.open(parts[-1], final_flags, dir_fd=current)
        except OSError as exc:
            raise DiskBackedAuditError(
                f"{label} is missing, unsafe, or has the wrong file type: "
                f"{absolute}"
            ) from exc
    finally:
        os.close(current)


def _stable_tuple(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_regular(
    path: pathlib.Path,
    *,
    label: str,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    maximum_size: int | None = None,
) -> StableRead:
    """Read one regular file once through one stable, no-follow descriptor."""

    descriptor = _safe_open(path, directory=False, label=label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DiskBackedAuditError(f"{label} must be a regular file")
        if before.st_nlink <= 0:
            raise DiskBackedAuditError(f"{label} is unlinked")
        if expected_size is not None and before.st_size != expected_size:
            raise DiskBackedAuditError(
                f"{label} byte size differs from inventory"
            )
        if maximum_size is not None and before.st_size > maximum_size:
            raise DiskBackedAuditError(
                f"{label} exceeds the maximum accepted byte size"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            observed_size += len(chunk)
        after = os.fstat(descriptor)
        if _stable_tuple(before) != _stable_tuple(after):
            raise DiskBackedAuditError(f"{label} changed while being read")
        if observed_size != before.st_size:
            raise DiskBackedAuditError(
                f"{label} read length differs from stable fstat size"
            )
        observed_sha = digest.hexdigest()
        if (
            expected_sha256 is not None
            and observed_sha != expected_sha256
        ):
            raise DiskBackedAuditError(
                f"{label} SHA-256 differs from inventory; "
                f"expected={expected_sha256}, observed={observed_sha}"
            )
        return StableRead(
            b"".join(chunks),
            observed_size,
            observed_sha,
            (before.st_dev, before.st_ino),
        )
    finally:
        os.close(descriptor)


def _require_existing_directory(
    path: pathlib.Path,
    *,
    label: str,
) -> pathlib.Path:
    absolute = _lexical_absolute(path)
    descriptor = _safe_open(absolute, directory=True, label=label)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise DiskBackedAuditError(f"{label} must be a directory")
    finally:
        os.close(descriptor)
    return absolute


def _canonical_inventory_path(
    value: object,
    *,
    base: pathlib.Path,
    label: str,
) -> pathlib.Path:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
    ):
        raise DiskBackedAuditError(
            f"{label} must be a non-empty canonical POSIX path"
        )
    pure = PurePosixPath(value)
    if (
        pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DiskBackedAuditError(f"{label} must be a canonical POSIX path")
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return _lexical_absolute(candidate)


def load_input_inventory(
    path: str | os.PathLike[str],
) -> LoadedInventory:
    """Read and validate the production inventory without reopening it."""

    inventory_path = _lexical_absolute(pathlib.Path(path))
    observed = _read_stable_regular(
        inventory_path,
        label="materialization inventory",
        maximum_size=MAX_INVENTORY_BYTES,
    )
    try:
        raw = loads_strict(observed.payload.decode("utf-8"))
    except (UnicodeError, StrictJSONError) as exc:
        raise DiskBackedAuditError(
            f"materialization inventory is not strict UTF-8 JSON: {exc}"
        ) from exc
    document = _exact_object(
        raw,
        {"schema_version", "works", "self_hash"},
        "materialization inventory",
    )
    if document["schema_version"] != INPUT_INVENTORY_SCHEMA:
        raise DiskBackedAuditError(
            "materialization inventory schema is unsupported"
        )
    recorded_hash = _sha256(
        document["self_hash"],
        "materialization inventory.self_hash",
    )
    unhashed = {
        key: value for key, value in document.items() if key != "self_hash"
    }
    if canonical_hash(unhashed) != recorded_hash:
        raise DiskBackedAuditError(
            "materialization inventory self_hash mismatch"
        )
    works = document["works"]
    if type(works) is not list or not works:
        raise DiskBackedAuditError(
            "materialization inventory.works must be a non-empty array"
        )
    entries: dict[str, InventoryEntry] = {}
    lexical_paths: set[pathlib.Path] = set()
    for index, value in enumerate(works):
        label = f"materialization inventory.works[{index}]"
        row = _exact_object(
            value,
            {"work_id", "path", "byte_size", "sha256"},
            label,
        )
        work = _work_id(row["work_id"], f"{label}.work_id")
        candidate = _canonical_inventory_path(
            row["path"],
            base=inventory_path.parent,
            label=f"{label}.path",
        )
        if work in entries:
            raise DiskBackedAuditError(
                f"materialization inventory repeats work_id {work}"
            )
        if candidate in lexical_paths:
            raise DiskBackedAuditError(
                "materialization inventory aliases one path across work ids"
            )
        lexical_paths.add(candidate)
        entries[work] = InventoryEntry(
            work,
            candidate,
            _exact_int(
                row["byte_size"],
                f"{label}.byte_size",
                minimum=1,
            ),
            _sha256(row["sha256"], f"{label}.sha256"),
        )
    if tuple(entries) != tuple(sorted(entries)):
        raise DiskBackedAuditError(
            "materialization inventory works must be sorted by work_id"
        )
    return LoadedInventory(
        entries,
        observed.byte_size,
        observed.sha256,
        recorded_hash,
    )


def _save_shingles(path: pathlib.Path, shingles: np.ndarray) -> None:
    if (
        type(shingles) is not np.ndarray
        or shingles.dtype != np.dtype(np.uint64)
        or shingles.ndim != 1
    ):
        raise DiskBackedAuditError(
            "registered shingle builder returned a non-uint64 vector"
        )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.save(handle, shingles, allow_pickle=False)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_shingle_map(
    work_ids: Sequence[str],
    artifacts: Mapping[str, ShingleArtifact],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    try:
        for work in work_ids:
            artifact = artifacts[work]
            array = np.load(
                artifact.path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if (
                array.dtype != np.dtype(np.uint64)
                or array.ndim != 1
                or len(array) != artifact.count
            ):
                raise DiskBackedAuditError(
                    f"{work}: persisted shingle array identity drifted"
                )
            arrays[work] = array
    except BaseException:
        _close_shingle_map(arrays)
        raise
    return arrays


def _close_shingle_map(arrays: Mapping[str, np.ndarray]) -> None:
    for array in arrays.values():
        mmap_object = getattr(array, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()


def _intersection_count(short: np.ndarray, long: np.ndarray) -> int:
    """Count a sorted-unique intersection with bounded temporary vectors."""

    common = 0
    for start in range(0, len(short), INTERSECTION_CHUNK_SIZE):
        chunk = np.asarray(
            short[start : start + INTERSECTION_CHUNK_SIZE],
            dtype=np.uint64,
        )
        locations = np.searchsorted(long, chunk)
        inside = locations < len(long)
        if bool(np.any(inside)):
            common += int(
                np.count_nonzero(
                    long[locations[inside]] == chunk[inside]
                )
            )
    return common


def _disk_backed_overlaps(
    work_ids: Sequence[str],
    artifacts: Mapping[str, ShingleArtifact],
    *,
    containment_threshold: Fraction,
    minimum_shingles: int,
    sample_size: int,
) -> tuple[ci.ContentOverlap, ...]:
    """Exact disk-backed equivalent of the registered overlap function."""

    works = tuple(work_ids)
    if (
        not works
        or works != tuple(sorted(works))
        or len(works) != len(set(works))
        or set(works) != set(artifacts)
    ):
        raise DiskBackedAuditError(
            "overlap work ids and shingle artifacts must match exactly"
        )
    threshold = ci._threshold_fraction(containment_threshold)
    if type(minimum_shingles) is not int or minimum_shingles <= 0:
        raise DiskBackedAuditError(
            "minimum_shingles must be a positive exact integer"
        )
    if type(sample_size) is not int or sample_size <= 0:
        raise DiskBackedAuditError(
            "sample_size must be a positive exact integer"
        )

    overlaps: list[ci.ContentOverlap] = []
    exact_owners: dict[str, str] = {}
    exact_pairs: set[tuple[str, str]] = set()
    for work in works:
        digest = artifacts[work].canonical_text_sha256
        owner = exact_owners.setdefault(digest, work)
        if owner != work:
            pair = tuple(sorted((owner, work)))
            if pair not in exact_pairs:
                exact_pairs.add(pair)
                overlaps.append(
                    ci.ContentOverlap(
                        left_work=pair[0],
                        right_work=pair[1],
                        kind="exact_cross_work_chunk",
                        containment=1.0,
                        evidence=f"sha256:{digest}",
                    )
                )

    arrays = _load_shingle_map(works, artifacts)
    try:
        for left_index, left in enumerate(works):
            for right in works[left_index + 1 :]:
                a = arrays[left]
                b = arrays[right]
                if len(a) < minimum_shingles or len(b) < minimum_shingles:
                    continue
                if len(a) <= len(b):
                    short_work = left
                    long_work = right
                    short = a
                    long = b
                else:
                    short_work = right
                    long_work = left
                    short = b
                    long = a
                allowed_misses = (
                    (threshold.denominator - threshold.numerator)
                    * len(short)
                    // threshold.denominator
                )
                probe = ci._sample(
                    short,
                    max(sample_size, allowed_misses + 1),
                )
                locations = np.searchsorted(long, probe)
                in_long = (locations < len(long)) & (
                    long[np.minimum(locations, len(long) - 1)] == probe
                )
                if int((~in_long).sum()) > allowed_misses:
                    continue
                common = _intersection_count(short, long)
                if (
                    common * threshold.denominator
                    >= len(short) * threshold.numerator
                ):
                    ratio = common / len(short)
                    overlaps.append(
                        ci.ContentOverlap(
                            left_work=short_work,
                            right_work=long_work,
                            kind="word5_asymmetric_containment",
                            containment=float(ratio),
                            evidence=(
                                f"{common}/{len(short)} unique "
                                "word-5-grams"
                            ),
                        )
                    )
    finally:
        _close_shingle_map(arrays)
    return tuple(overlaps)


def audit_disk_backed(
    entries: Mapping[str, InventoryEntry],
    *,
    selected_exclusions: Sequence[str],
    scratch_parent: pathlib.Path,
    expected_full_count: int | None = None,
    expected_selected_count: int | None = None,
) -> AuditResult:
    """Build full and selected reports while reading each text exactly once."""

    if type(entries) is not dict or not entries:
        raise DiskBackedAuditError(
            "entries must be an exact non-empty dictionary"
        )
    full_ids = tuple(entries)
    if full_ids != tuple(sorted(full_ids)):
        raise DiskBackedAuditError("entries must be sorted by work_id")
    for index, (work, entry) in enumerate(entries.items()):
        _work_id(work, f"entries[{index}]")
        if type(entry) is not InventoryEntry or entry.work_id != work:
            raise DiskBackedAuditError(
                f"{work}: entry must be exactly InventoryEntry"
            )
    if expected_full_count is not None and (
        type(expected_full_count) is not int
        or expected_full_count <= 0
        or len(full_ids) != expected_full_count
    ):
        raise DiskBackedAuditError(
            "full work count differs from expected_full_count"
        )
    exclusions = tuple(
        _work_id(value, f"selected_exclusions[{index}]")
        for index, value in enumerate(selected_exclusions)
    )
    if (
        len(exclusions) != len(set(exclusions))
        or exclusions != tuple(sorted(exclusions))
    ):
        raise DiskBackedAuditError(
            "selected_exclusions must be sorted and unique"
        )
    if not set(exclusions).issubset(entries):
        raise DiskBackedAuditError(
            "selected_exclusions contains unknown work ids"
        )
    excluded = set(exclusions)
    selected_ids = tuple(work for work in full_ids if work not in excluded)
    if not selected_ids:
        raise DiskBackedAuditError("selected inventory is empty")
    if expected_selected_count is not None and (
        type(expected_selected_count) is not int
        or expected_selected_count <= 0
        or len(selected_ids) != expected_selected_count
    ):
        raise DiskBackedAuditError(
            "selected work count differs from expected_selected_count"
        )
    scratch = _require_existing_directory(
        scratch_parent,
        label="scratch parent",
    )

    work_rows: dict[str, Mapping[str, object]] = {}
    one_work_blockers: dict[str, Sequence[Mapping[str, object]]] = {}
    artifacts: dict[str, ShingleArtifact] = {}
    observed_identities: set[tuple[int, int]] = set()
    with tempfile.TemporaryDirectory(
        prefix="stylo-r1-shingles-",
        dir=scratch,
    ) as temporary:
        shingle_root = pathlib.Path(temporary)
        for ordinal, work in enumerate(full_ids):
            entry = entries[work]
            observed = _read_stable_regular(
                entry.path,
                label=f"materialized text {work}",
                expected_size=entry.byte_size,
                expected_sha256=entry.sha256,
            )
            if observed.identity in observed_identities:
                raise DiskBackedAuditError(
                    "materialization inventory aliases one inode across work ids"
                )
            observed_identities.add(observed.identity)
            payload = observed.payload
            canonical, row, blockers = tq._audit_one_work(
                payload,
                work_id=work,
                minimum_words=tq.DEFAULT_MINIMUM_WORDS,
            )
            work_rows[work] = row
            one_work_blockers[work] = blockers
            canonical_digest = _sha256_bytes(canonical.encode("utf-8"))
            shingles = ci._word_shingles([canonical], 5)
            shingle_path = shingle_root / f"{ordinal:06d}.npy"
            _save_shingles(shingle_path, shingles)
            artifacts[work] = ShingleArtifact(
                canonical_digest,
                shingle_path,
                len(shingles),
            )
            del canonical, shingles, payload
            del observed
            gc.collect()

        full_overlaps = _disk_backed_overlaps(
            full_ids,
            artifacts,
            containment_threshold=tq.DEFAULT_CONTAINMENT_THRESHOLD,
            minimum_shingles=tq.DEFAULT_MINIMUM_SHINGLES,
            sample_size=tq.DEFAULT_SAMPLE_SIZE,
        )
        selected_artifacts = {
            work: artifacts[work] for work in selected_ids
        }
        selected_overlaps = _disk_backed_overlaps(
            selected_ids,
            selected_artifacts,
            containment_threshold=tq.DEFAULT_CONTAINMENT_THRESHOLD,
            minimum_shingles=tq.DEFAULT_MINIMUM_SHINGLES,
            sample_size=tq.DEFAULT_SAMPLE_SIZE,
        )
        full_report = tq._assemble_audit_report(
            full_ids,
            work_rows=work_rows,
            one_work_blockers=one_work_blockers,
            overlaps=full_overlaps,
            minimum_words=tq.DEFAULT_MINIMUM_WORDS,
            containment_threshold=tq.DEFAULT_CONTAINMENT_THRESHOLD,
            minimum_shingles=tq.DEFAULT_MINIMUM_SHINGLES,
            sample_size=tq.DEFAULT_SAMPLE_SIZE,
        )
        selected_report = tq._assemble_audit_report(
            selected_ids,
            work_rows=work_rows,
            one_work_blockers=one_work_blockers,
            overlaps=selected_overlaps,
            minimum_words=tq.DEFAULT_MINIMUM_WORDS,
            containment_threshold=tq.DEFAULT_CONTAINMENT_THRESHOLD,
            minimum_shingles=tq.DEFAULT_MINIMUM_SHINGLES,
            sample_size=tq.DEFAULT_SAMPLE_SIZE,
        )
    return AuditResult(
        full_report,
        selected_report,
        len(full_ids),
        len(selected_ids),
    )
