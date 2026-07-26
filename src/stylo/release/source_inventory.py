"""Fail-closed inventory for Python code and release-critical support files.

Git normally answers which files ship, but ignored Python files can still be
imported by a dirty workspace and a source archive has no ``.git`` directory at
all.  This module therefore binds the complete release Python *path set* to a
reviewed count and digest, explicitly classifies permitted local-only Python,
requires shipped paths to be tracked when Git is available, and pins immutable
evidence by content hash.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Iterable, Mapping

SCHEMA_VERSION = "stylo.executable-source-inventory.v1"
DEFAULT_MANIFEST = "release/executable_sources.json"


class SourceInventoryError(RuntimeError):
    """The source inventory itself is malformed or cannot be evaluated."""


@dataclass(frozen=True)
class SourceSnapshot:
    """Canonical identity of the set of release Python paths."""

    file_count: int
    paths_sha256: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class SourceInventoryReport:
    """All inventory violations found in one pass."""

    snapshot: SourceSnapshot
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def require_clean(self) -> None:
        if self.issues:
            raise SourceInventoryError("\n".join(self.issues))


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise SourceInventoryError(f"duplicate JSON key in source inventory: {key!r}")
        out[key] = value
    return out


def _load_manifest(path: pathlib.Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise SourceInventoryError(f"source inventory missing or symlinked: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceInventoryError(f"invalid source inventory JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SourceInventoryError("source inventory root must be an object")
    expected = {
        "schema_version",
        "python_roots",
        "release_python_file_count",
        "release_python_paths_sha256",
        "local_only_python_files",
        "required_release_files",
        "sha256_bindings",
    }
    if set(data) != expected:
        raise SourceInventoryError(
            f"source inventory keys differ: missing={sorted(expected - set(data))}, "
            f"extra={sorted(set(data) - expected)}"
        )
    if data["schema_version"] != SCHEMA_VERSION:
        raise SourceInventoryError(
            f"source inventory schema {data['schema_version']!r} != {SCHEMA_VERSION!r}"
        )
    return data


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceInventoryError(f"{label} must be a non-empty string")
    if "\\" in value:
        raise SourceInventoryError(f"{label} must use POSIX separators: {value!r}")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SourceInventoryError(f"unsafe {label}: {value!r}")
    if path.as_posix() != value:
        raise SourceInventoryError(f"non-canonical {label}: {value!r}")
    return value


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SourceInventoryError(f"{label} must be an array")
    paths = tuple(_safe_relative(item, label=label) for item in value)
    if tuple(sorted(set(paths))) != paths:
        raise SourceInventoryError(f"{label} must be sorted and duplicate-free")
    return paths


def _bindings(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SourceInventoryError("sha256_bindings must be an object")
    out: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        path = _safe_relative(raw_path, label="sha256 binding path")
        if not isinstance(raw_digest, str) or len(raw_digest) != 64:
            raise SourceInventoryError(f"invalid SHA256 binding for {path}")
        try:
            bytes.fromhex(raw_digest)
        except ValueError as exc:
            raise SourceInventoryError(f"invalid SHA256 binding for {path}") from exc
        if raw_digest != raw_digest.lower():
            raise SourceInventoryError(f"SHA256 binding must be lowercase for {path}")
        out[path] = raw_digest
    if list(out) != sorted(out):
        raise SourceInventoryError("sha256_bindings keys must be sorted")
    return out


def _path_set_digest(paths: Iterable[str]) -> str:
    """Length-prefix paths so concatenation cannot create an ambiguous set."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        raw = path.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _discover_python(repository_root: pathlib.Path,
                     roots: tuple[str, ...]) -> tuple[set[str], list[str]]:
    discovered: set[str] = set()
    issues: list[str] = []
    for relative_root in roots:
        root = repository_root / relative_root
        if root.is_symlink() or not root.is_dir():
            issues.append(f"Python source root missing, non-directory or symlinked: {relative_root}")
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(repository_root).as_posix()
            if path.is_symlink() or not path.is_file():
                issues.append(f"Python source is missing, non-file or symlinked: {relative}")
                continue
            discovered.add(relative)
    return discovered, issues


def compute_snapshot(repository_root: pathlib.Path | str, *,
                     python_roots: Iterable[str],
                     local_only_python_files: Iterable[str] = ()) -> SourceSnapshot:
    """Compute the release path-set identity independently of Git."""
    root = pathlib.Path(repository_root).resolve()
    roots = tuple(_safe_relative(item, label="Python root") for item in python_roots)
    local = set(_safe_relative(item, label="local-only Python path")
                for item in local_only_python_files)
    discovered, issues = _discover_python(root, roots)
    if issues:
        raise SourceInventoryError("\n".join(issues))
    release_paths = tuple(sorted(discovered - local))
    return SourceSnapshot(
        file_count=len(release_paths),
        paths_sha256=_path_set_digest(release_paths),
        paths=release_paths,
    )


def _git_paths(root: pathlib.Path, paths: Iterable[str]) -> set[str]:
    wanted = sorted(set(paths))
    if not wanted:
        return set()
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--", *wanted],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SourceInventoryError("cannot query Git tracked paths for source inventory") from exc
    return {
        part.decode("utf-8", "surrogateescape")
        for part in out.split(b"\0")
        if part
    }


def _git_ignored(root: pathlib.Path, path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SourceInventoryError("git executable not found") from exc
    if result.returncode not in (0, 1):
        raise SourceInventoryError(
            f"git check-ignore failed for {path}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.returncode == 0


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source_inventory(
    repository_root: pathlib.Path | str,
    *,
    manifest: pathlib.Path | str = DEFAULT_MANIFEST,
    require_git: bool = True,
) -> SourceInventoryReport:
    """Validate the reviewed source set in a checkout or a Git-free archive."""
    root = pathlib.Path(repository_root).resolve()
    manifest_relative = _safe_relative(str(manifest), label="manifest path")
    manifest_path = root / manifest_relative
    data = _load_manifest(manifest_path)

    roots = _string_list(data["python_roots"], label="python_roots")
    local = _string_list(data["local_only_python_files"], label="local_only_python_files")
    required = _string_list(data["required_release_files"], label="required_release_files")
    bindings = _bindings(data["sha256_bindings"])
    if not isinstance(data["release_python_file_count"], int) or data["release_python_file_count"] < 1:
        raise SourceInventoryError("release_python_file_count must be a positive integer")
    expected_digest = data["release_python_paths_sha256"]
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise SourceInventoryError("release_python_paths_sha256 must be a SHA256 hex string")

    local_set = set(local)
    for path in local:
        if not path.endswith(".py") or not any(
            path == source_root or path.startswith(source_root + "/") for source_root in roots
        ):
            raise SourceInventoryError(f"local-only Python path is outside python_roots: {path}")

    discovered, issues = _discover_python(root, roots)
    present_local = discovered & local_set
    unclassified_local = sorted(
        path for path in local_set if (root / path).exists() and path not in discovered
    )
    if unclassified_local:
        issues.append(
            "local-only entries exist but are not ordinary Python files: "
            + ", ".join(unclassified_local)
        )
    release_paths = tuple(sorted(discovered - local_set))
    snapshot = SourceSnapshot(
        file_count=len(release_paths),
        paths_sha256=_path_set_digest(release_paths),
        paths=release_paths,
    )
    if snapshot.file_count != data["release_python_file_count"]:
        issues.append(
            "release Python file count mismatch: "
            f"got {snapshot.file_count}, expected {data['release_python_file_count']}"
        )
    if snapshot.paths_sha256 != expected_digest:
        issues.append(
            "release Python path-set SHA256 mismatch: "
            f"got {snapshot.paths_sha256}, expected {expected_digest}"
        )

    release_required = sorted(set(release_paths) | set(required) | {manifest_relative})
    for path in release_required:
        target = root / path
        if target.is_symlink() or not target.is_file():
            issues.append(f"required release file missing, non-file or symlinked: {path}")

    for path, expected in bindings.items():
        target = root / path
        if target.is_symlink() or not target.is_file():
            issues.append(f"SHA-bound file missing, non-file or symlinked: {path}")
        else:
            got = _sha256(target)
            if got != expected:
                issues.append(f"SHA256 mismatch for {path}: got {got}, expected {expected}")

    if require_git:
        tracked = _git_paths(root, set(release_required) | local_set | set(bindings))
        untracked = sorted(path for path in release_required if path not in tracked)
        if untracked:
            issues.append("release files are not Git-tracked: " + ", ".join(untracked))
        tracked_local = sorted(present_local & tracked)
        if tracked_local:
            issues.append("local-only Python files are Git-tracked: " + ", ".join(tracked_local))
        not_ignored = sorted(path for path in present_local if not _git_ignored(root, path))
        if not_ignored:
            issues.append("local-only Python files are not ignored: " + ", ".join(not_ignored))
    elif present_local:
        issues.append(
            "local-only Python files leaked into the Git-free release archive: "
            + ", ".join(sorted(present_local))
        )

    return SourceInventoryReport(snapshot=snapshot, issues=tuple(issues))
