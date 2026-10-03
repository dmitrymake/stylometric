"""Publish immutable directory generations through one atomic current pointer.

Legacy flat directories remain read-only. A reader resolves the pointer once
and keeps that generation, which publishers never delete or replace.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import tempfile

from .._io import exclusive_file_lock, fsync_directory, is_link, read_regular
from ..jsonio import canonical_hash, dumps_strict, loads_strict


class SnapshotPublishError(RuntimeError):
    """A directory snapshot is missing, unsafe, or inconsistent."""


STORE_SUFFIX = ".snapshots"
VERSIONS_DIRECTORY = "versions"
CURRENT_POINTER = "CURRENT.json"
GENERATION_MANIFEST = "snapshot_manifest.json"
SNAPSHOT_SCHEMA = "stylo.directory-snapshot.v1"
POINTER_SCHEMA = "stylo.directory-snapshot-pointer.v1"
_TOKEN = re.compile(r"^[0-9a-f]{64}$")

# Kept as internal aliases for the existing split publisher.
_fsync_dir = fsync_directory


def _real_directory(path: pathlib.Path) -> None:
    if is_link(path) or not path.is_dir():
        raise SnapshotPublishError(f"snapshot directory must be real: {path}")


def _fsync_tree(root: pathlib.Path) -> None:
    """Flush every staged regular file and supported directory metadata."""
    _real_directory(root)
    directories = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = pathlib.Path(dirpath)
        directories.append(directory)
        for name in dirnames:
            _real_directory(directory / name)
        for name in filenames:
            child = directory / name
            if is_link(child) or not child.is_file():
                raise SnapshotPublishError(f"non-regular staged snapshot member: {child}")
            with open(child, "r+b" if os.name == "nt" else "rb") as handle:
                os.fsync(handle.fileno())
    for directory in reversed(directories):
        _fsync_dir(directory)


def _inventory(root: pathlib.Path) -> dict:
    _real_directory(root)
    files = {}
    directories = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = pathlib.Path(dirpath)
        for name in dirnames:
            child = directory / name
            _real_directory(child)
            directories.append(child.relative_to(root).as_posix())
        for name in filenames:
            child = directory / name
            if child == root / GENERATION_MANIFEST:
                continue
            try:
                payload = read_regular(child, label="snapshot member")
            except (OSError, ValueError) as exc:
                raise SnapshotPublishError(str(exc)) from exc
            files[child.relative_to(root).as_posix()] = hashlib.sha256(payload).hexdigest()
    return {"schema_version": SNAPSHOT_SCHEMA, "directories": sorted(directories),
            "files": dict(sorted(files.items()))}


def _read_json(path: pathlib.Path) -> dict:
    try:
        return loads_strict(read_regular(path, label="snapshot metadata").decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise SnapshotPublishError(f"invalid snapshot metadata: {path}: {exc}") from exc


def _validate_generation(root: pathlib.Path, token: str) -> pathlib.Path:
    _real_directory(root)
    recorded = _read_json(root / GENERATION_MANIFEST)
    if (type(recorded) is not dict or set(recorded) != {"schema_version", "directories", "files"}
            or recorded["schema_version"] != SNAPSHOT_SCHEMA
            or canonical_hash(recorded) != token):
        raise SnapshotPublishError("snapshot generation identity mismatch")
    if _inventory(root) != recorded:
        raise SnapshotPublishError("snapshot generation inventory/hash mismatch")
    return root


def snapshot_store(target: str | pathlib.Path) -> pathlib.Path:
    target = pathlib.Path(target)
    return target.parent / (target.name + STORE_SUFFIX)


def _present(path: pathlib.Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SnapshotPublishError(f"cannot inspect snapshot path: {path}: {exc}") from exc
    return True


def resolve_directory_snapshot(target: str | pathlib.Path) -> pathlib.Path:
    """Pin one hash-verified immutable generation, or read a legacy flat root.

    Only an absent CURRENT permits legacy fallback. Malformed, unreadable,
    dangling, or symlinked pointers fail; they never expose stale flat data.
    """
    target = pathlib.Path(target)
    if is_link(target):
        raise SnapshotPublishError(f"snapshot target must not be a symlink: {target}")
    store = snapshot_store(target)
    if _present(store):
        _real_directory(store)
        pointer = store / CURRENT_POINTER
        if _present(pointer):
            record = _read_json(pointer)
            if (type(record) is not dict or set(record) != {"schema_version", "generation_id"}
                    or record["schema_version"] != POINTER_SCHEMA
                    or type(record["generation_id"]) is not str
                    or _TOKEN.fullmatch(record["generation_id"]) is None):
                raise SnapshotPublishError("snapshot CURRENT pointer identity mismatch")
            versions = store / VERSIONS_DIRECTORY
            _real_directory(versions)
            return _validate_generation(versions / record["generation_id"], record["generation_id"])
    _real_directory(target)
    return target


def _publish_current_pointer(store: pathlib.Path, token: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".CURRENT-", dir=store)
    temporary = pathlib.Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write((dumps_strict({"schema_version": POINTER_SCHEMA,
                                      "generation_id": token}, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, store / CURRENT_POINTER)
        _fsync_dir(store)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_directory_snapshot(staging: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """Publish a flushed generation; leave every prior/legacy generation intact.

    A crash before pointer replacement leaves the prior CURRENT unchanged.
    A crash after replacement exposes the complete flushed new generation.
    Unreferenced generations are retained; no publication deletes user data.
    """
    staging, target = pathlib.Path(staging), pathlib.Path(target)
    if staging.parent.absolute() != target.parent.absolute():
        raise SnapshotPublishError("staging and target must be sibling paths")
    _real_directory(staging)
    if is_link(target) or (_present(target) and not target.is_dir()):
        raise SnapshotPublishError(f"snapshot target must not be a symlink or file: {target}")
    if _present(staging / GENERATION_MANIFEST):
        raise SnapshotPublishError(f"reserved generation manifest already exists: {staging}")
    inventory = _inventory(staging)
    token = canonical_hash(inventory)
    (staging / GENERATION_MANIFEST).write_text(dumps_strict(inventory, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    _fsync_tree(staging)
    store = snapshot_store(target)
    store.mkdir(exist_ok=True)
    _real_directory(store)
    versions = store / VERSIONS_DIRECTORY
    versions.mkdir(exist_ok=True)
    _real_directory(versions)
    _fsync_dir(target.parent)
    _fsync_dir(store)
    with exclusive_file_lock(store / ".publish.lock"):
        if _present(store / CURRENT_POINTER):
            resolve_directory_snapshot(target)
        version = versions / token
        if _present(version):
            _validate_generation(version, token)
            shutil.rmtree(staging)
        else:
            os.replace(staging, version)
            _fsync_dir(versions)
            _validate_generation(version, token)
        _publish_current_pointer(store, token)
    return version


__all__ = ["SnapshotPublishError", "publish_directory_snapshot", "resolve_directory_snapshot"]
