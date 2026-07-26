#!/usr/bin/env python3
"""Capture an immutable P0 baseline snapshot of the repository state.

Records the anchor the scientific reset freezes against, so drift after P0 is a
deliberate reviewed change rather than a silent one:

* the publish commit (HEAD, branch, commit date);
* the working-tree state (porcelain status + a hash of the tracked diff);
* the test result, measured here (not asserted) by actually running pytest;
* a SHA-256 fingerprint of every tracked artifact under ``docs/`` and ``research/``.

The snapshot file excludes itself from the fingerprint, so re-running it once it is
tracked stays stable instead of hashing its own previous contents.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stylo.jsonio import dump_strict  # noqa: E402

# Fingerprint results AND the P0 code, so an untracked change to jsonio/claims/
# hygiene/tests/CI is caught by the anchor rather than invisible behind a filename.
ARTIFACT_ROOTS = ("docs", "research", "src", "scripts", "tests", ".github", ".githooks")
_SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def _artifact_files(exclude: Path) -> list[str]:
    """Every file physically under docs/ and research/, tracked or not.

    research/ is untracked working material, so a git-only listing would miss it;
    the anchor should still fingerprint it. The snapshot excludes itself.
    """
    exclude = exclude.resolve()
    found = []
    for root in ARTIFACT_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                if name.endswith(".pyc"):
                    continue
                path = Path(dirpath) / name
                if path.resolve() == exclude:
                    continue
                found.append(os.path.relpath(path, REPO_ROOT))
    return sorted(set(found))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _collected_count() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--co", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    ).stdout
    total = 0
    for line in out.splitlines():
        if line.startswith("tests/") and line.rsplit(":", 1)[-1].strip().isdigit():
            total += int(line.rsplit(":", 1)[-1].strip())
    return total


def _run_pytest() -> dict:
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no"],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    runtime = round(time.monotonic() - start, 1)
    passed = proc.returncode == 0
    collected = _collected_count()
    summary = f"{collected} passed" if passed else f"FAILED (returncode {proc.returncode})"
    return {"all_passed": passed, "collected": collected, "summary": summary, "runtime_seconds": runtime}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" / "p0_baseline_snapshot.json"))
    parser.add_argument("--skip-tests", action="store_true", help="do not run pytest (tests block left null)")
    args = parser.parse_args()
    out_path = Path(args.out)

    porcelain = _git("status", "--porcelain").splitlines()
    modified = sum(1 for line in porcelain if line[:2].strip() == "M")
    staged = sum(1 for line in porcelain if line and line[0] in "MADRC" and line[0] != " ")
    untracked = sum(1 for line in porcelain if line.startswith("??"))
    diff_hash = hashlib.sha256(_git("diff", "HEAD").encode("utf-8")).hexdigest()

    artifacts = _artifact_files(out_path)
    artifact_hashes = {rel: _sha256_file(REPO_ROOT / rel) for rel in artifacts}

    tests = {"all_passed": None, "collected": None, "summary": "skipped", "runtime_seconds": None}
    if not args.skip_tests:
        tests = _run_pytest()

    snapshot = {
        "schema": "p0_baseline_snapshot/v2",
        "claim_status": "engineering",
        "note": (
            "Иммутабельный якорь состояния на завершение фазы P0. Хэши покрывают "
            "docs/ и research/ (кроме самого снимка). Дрейф после этой точки должен "
            "быть осознанным ревьюируемым изменением."
        ),
        "git": {
            "head_commit": _git("rev-parse", "HEAD").strip(),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "commit_date": _git("show", "-s", "--format=%cI", "HEAD").strip(),
        },
        "working_tree": {
            "modified_tracked": modified,
            "staged": staged,
            "untracked": untracked,
            "porcelain": porcelain,
            "diff_head_sha256": diff_hash,
        },
        "tests": tests,
        "artifacts": {
            "roots": list(ARTIFACT_ROOTS),
            "count": len(artifacts),
            "sha256": artifact_hashes,
        },
    }

    dump_strict(snapshot, out_path)
    print(f"wrote {out_path} | {len(artifacts)} artifact hashes | tests: {tests['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
