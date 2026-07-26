"""Crash-safe publication of complete directory snapshots."""
from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import os
import pathlib
import shutil


class SnapshotPublishError(RuntimeError):
    """A staged snapshot could not be published without risking the current one."""


def _fsync_dir(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: pathlib.Path) -> None:
    """Durably flush every staged regular file and directory before publish."""

    directories: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = pathlib.Path(dirpath)
        directories.append(directory)
        for name in dirnames:
            child = directory / name
            if child.is_symlink():
                raise SnapshotPublishError(f"symlink in staged snapshot: {child}")
        for name in filenames:
            child = directory / name
            if child.is_symlink() or not child.is_file():
                raise SnapshotPublishError(
                    f"staged snapshot member is not a regular file: {child}"
                )
            fd = os.open(child, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    for directory in reversed(directories):
        _fsync_dir(directory)


@contextlib.contextmanager
def _publish_lock(target: pathlib.Path):
    lock = target.parent / f".{target.name}.publish.lock"
    if lock.is_symlink():
        raise SnapshotPublishError(f"snapshot lock must not be a symlink: {lock}")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _rename_exchange(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Atomically exchange two same-filesystem paths on Linux when available."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 2
    result = renameat2(
        at_fdcwd,
        os.fsencode(left),
        at_fdcwd,
        os.fsencode(right),
        rename_exchange,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EXDEV}:
        return False
    raise OSError(error, os.strerror(error), f"{left} <-> {right}")


def publish_directory_snapshot(staging: pathlib.Path, target: pathlib.Path) -> None:
    """Publish ``staging`` as ``target`` without exposing a partial generation.

    Both paths must be real sibling directories.  Linux ``renameat2`` exchanges
    an existing target atomically.  A conservative recoverable two-rename
    fallback is retained for filesystems without exchange support.
    """

    staging = pathlib.Path(staging)
    target = pathlib.Path(target)
    if staging.parent.absolute() != target.parent.absolute():
        raise SnapshotPublishError("staging and target must be sibling paths")
    if staging.is_symlink() or not staging.is_dir():
        raise SnapshotPublishError(f"staging must be a real directory: {staging}")
    if target.is_symlink():
        raise SnapshotPublishError(f"target must not be a symlink: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.parent / f".{target.name}.previous"

    with _publish_lock(target):
        if backup.is_symlink():
            raise SnapshotPublishError(f"snapshot backup must not be a symlink: {backup}")
        # Recover an interrupted fallback exchange before doing anything else.
        if not target.exists() and backup.is_dir():
            os.replace(backup, target)
            _fsync_dir(target.parent)
        elif target.exists() and backup.exists():
            if not backup.is_dir():
                raise SnapshotPublishError(f"snapshot backup is not a directory: {backup}")
            shutil.rmtree(backup)

        if not target.exists():
            _fsync_tree(staging)
            os.replace(staging, target)
            _fsync_dir(target.parent)
            return
        if not target.is_dir():
            raise SnapshotPublishError(f"snapshot target is not a directory: {target}")

        _fsync_tree(staging)
        if _rename_exchange(staging, target):
            _fsync_dir(target.parent)
            # ``staging`` now names the old complete generation.
            shutil.rmtree(staging)
            _fsync_dir(target.parent)
            return

        # A two-rename fallback has an unavoidable window with no current
        # target.  Fail before mutating the current generation on filesystems
        # that cannot provide an atomic directory exchange.
        raise SnapshotPublishError(
            "filesystem does not support atomic directory exchange; "
            "current snapshot was left unchanged"
        )


__all__ = [
    "SnapshotPublishError",
    "publish_directory_snapshot",
]
