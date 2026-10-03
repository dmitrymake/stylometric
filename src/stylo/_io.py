"""Portable process locks and regular-file I/O for local artifact publication.

POSIX directories are fsynced after rename. Python exposes no portable Windows
directory flush, so Windows gets flushed files and atomic pointer replacement,
but no claimed power-loss durability for directory metadata.
"""
from __future__ import annotations

import contextlib
import errno
import os
import pathlib
import stat
import threading
import time
import weakref


_THREAD_LOCKS = weakref.WeakValueDictionary()
_THREAD_LOCKS_GUARD = threading.RLock()
_ACTIVE_LOCK_FDS = set()
_HELD_LOCKS = threading.local()


def _before_fork() -> None:
    # Keep open/register and close/unregister indivisible across a fork.
    _THREAD_LOCKS_GUARD.acquire()


def _after_fork_parent() -> None:
    _THREAD_LOCKS_GUARD.release()


def _after_fork_child() -> None:
    global _THREAD_LOCKS, _THREAD_LOCKS_GUARD, _ACTIVE_LOCK_FDS, _HELD_LOCKS
    for fd in _ACTIVE_LOCK_FDS:
        # Never explicitly unlock: the descriptor shares the parent's flock.
        os.close(fd)
    _ACTIVE_LOCK_FDS = set()
    _THREAD_LOCKS = weakref.WeakValueDictionary()
    _THREAD_LOCKS_GUARD = threading.RLock()
    _HELD_LOCKS = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent,
                        after_in_child=_after_fork_child)


def is_link(path: str | pathlib.Path) -> bool:
    """Recognize symlinks and Windows junction/reparse-point paths on 3.11."""
    try:
        info = pathlib.Path(path).lstat()
    except FileNotFoundError:
        return False
    return (stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def _regular_fd(path: pathlib.Path, flags: int, mode: int = 0o600) -> int:
    if is_link(path):
        raise ValueError(f"regular file must not be a symlink: {path}")
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0), mode)
    try:
        opened = os.fstat(fd)
        current = path.lstat()
        if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode)
                or bool(getattr(current, "st_file_attributes", 0) & 0x400)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
            raise ValueError(f"path changed or is not a regular file: {path}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_regular(path: str | pathlib.Path, *, label: str = "file") -> bytes:
    """Read one regular, nonsymlink file; an atomic rename retains coherent bytes."""
    path = pathlib.Path(path)
    # An already-open regular descriptor is safe when a writer atomically
    # replaces the pointer. Do not compare its inode to a later path lookup.
    if is_link(path):
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    fd = _open_read_descriptor(path)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def _open_read_descriptor(path: pathlib.Path) -> int:
    if os.name != "nt":
        return os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                       | getattr(os, "O_NONBLOCK", 0))
    # CRT opens need not allow deletion while open. Share DELETE explicitly so
    # readers of a CURRENT pointer do not prevent its atomic replacement.
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise


def fsync_directory(path: str | pathlib.Path) -> bool:
    """Flush directory metadata where supported; return False on Windows."""
    if os.name == "nt":
        return False
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _lock_descriptor(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        # Lock a permanent byte. Never unlink the lockfile: replacing its inode
        # would let a second process acquire a different lock for the same path.
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                time.sleep(0.02)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock_descriptor(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def exclusive_file_lock(path: str | pathlib.Path):
    """Serialize writers across threads/processes; OS releases locks on exit.

    The containing directory must already exist. The lockfile stays in place
    after use and after a crash; its presence never means a lock is held.
    Nested entry for the same path is rejected before acquiring an OS lock.
    """
    path = pathlib.Path(path)
    key = os.path.normcase(str(path.resolve()))
    held = getattr(_HELD_LOCKS, "paths", None)
    if held is None:
        held = _HELD_LOCKS.paths = set()
    if key in held:
        raise RuntimeError(f"nested exclusive_file_lock is unsupported: {path}")
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.get(key)
        if thread_lock is None:
            thread_lock = threading.Lock()
            _THREAD_LOCKS[key] = thread_lock
    thread_lock.acquire()
    owner_pid = os.getpid()
    held.add(key)
    fd = None
    acquired = False
    try:
        with _THREAD_LOCKS_GUARD:
            fd = _regular_fd(path, os.O_CREAT | os.O_RDWR)
            _ACTIVE_LOCK_FDS.add(fd)
        _lock_descriptor(fd)
        acquired = True
        yield
    finally:
        if os.getpid() == owner_pid:
            try:
                if acquired:
                    _unlock_descriptor(fd)
            finally:
                try:
                    if fd is not None:
                        with _THREAD_LOCKS_GUARD:
                            _ACTIVE_LOCK_FDS.discard(fd)
                            os.close(fd)
                finally:
                    held.remove(key)
                    thread_lock.release()
