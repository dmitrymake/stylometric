"""The release gate accounts for every Python path in checkouts and archives."""
from __future__ import annotations

import hashlib
import pathlib
import subprocess

from stylo import jsonio
from stylo.release.source_inventory import (
    SCHEMA_VERSION,
    check_source_inventory,
    compute_snapshot,
)


def _write_tree(root: pathlib.Path, *, local: bool = False) -> None:
    for relative in ("src/stylo/__init__.py", "src/stylo/core.py", "scripts/tool.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    if local:
        path = root / "scripts/local/private_helper.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# local only\n", encoding="utf-8")
    (root / "release").mkdir()
    (root / "NOTICE").write_text("reviewed\n", encoding="utf-8")


def _write_manifest(root: pathlib.Path, *, local: bool = False) -> pathlib.Path:
    local_paths = ["scripts/local/private_helper.py"]
    snapshot = compute_snapshot(
        root,
        python_roots=["scripts", "src/stylo"],
        local_only_python_files=local_paths,
    )
    notice_sha = hashlib.sha256((root / "NOTICE").read_bytes()).hexdigest()
    manifest = root / "release/executable_sources.json"
    jsonio.dump_strict(
        {
            "schema_version": SCHEMA_VERSION,
            "python_roots": ["scripts", "src/stylo"],
            "release_python_file_count": snapshot.file_count,
            "release_python_paths_sha256": snapshot.paths_sha256,
            "local_only_python_files": local_paths,
            "required_release_files": ["NOTICE"],
            "sha256_bindings": {"NOTICE": notice_sha},
        },
        manifest,
    )
    return manifest


def _git(root: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_git_free_archive_validates_complete_release_path_set(tmp_path):
    _write_tree(tmp_path)
    _write_manifest(tmp_path)
    report = check_source_inventory(tmp_path, require_git=False)
    assert report.ok
    assert report.snapshot.file_count == 3


def test_unclassified_python_changes_reviewed_path_set_identity(tmp_path):
    _write_tree(tmp_path)
    _write_manifest(tmp_path)
    extra = tmp_path / "scripts/surprise.py"
    extra.write_text("# not reviewed\n", encoding="utf-8")
    report = check_source_inventory(tmp_path, require_git=False)
    assert not report.ok
    assert any("path-set SHA256 mismatch" in issue for issue in report.issues)


def test_checkout_requires_release_files_tracked_and_local_python_ignored(tmp_path):
    _write_tree(tmp_path, local=True)
    _write_manifest(tmp_path, local=True)
    (tmp_path / ".gitignore").write_text("scripts/local/\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".gitignore", "NOTICE", "release", "scripts/tool.py", "src")
    assert check_source_inventory(tmp_path, require_git=True).ok

    untracked = tmp_path / "scripts/new_release_tool.py"
    untracked.write_text("# untracked\n", encoding="utf-8")
    # Refreshing only the reviewed path-set identity cannot make an untracked
    # executable shippable: the independent Git check still fails it.
    _write_manifest(tmp_path, local=True)
    report = check_source_inventory(tmp_path, require_git=True)
    assert any("not Git-tracked" in issue and "new_release_tool.py" in issue for issue in report.issues)


def test_local_only_python_cannot_leak_into_archive(tmp_path):
    _write_tree(tmp_path, local=True)
    _write_manifest(tmp_path, local=True)
    report = check_source_inventory(tmp_path, require_git=False)
    assert any("leaked into the Git-free release archive" in issue for issue in report.issues)
