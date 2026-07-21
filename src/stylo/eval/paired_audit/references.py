"""Pinned A0 reference verification — SHA256 before any parse (§3.2).

The A0 reference files are verified by **pinned SHA256 before any parse** (not merely folded into the
``run_id``): ``docs/lobo_books.txt`` and ``data/ruaa_bench_v1/reference_submission_stylo.csv``. RuAA is
**additionally** checked against the frozen ``data/ruaa_bench_v1/SHA256SUMS``. Any mismatch aborts the
confirmatory preflight before the reference is opened. The returned digests are the exact values the
canonical RunPlan binds under ``a0_reference_shas``, so the RunPlan can never bind an unverified SHA.
"""
from __future__ import annotations

import hashlib
import pathlib

LOBO_BOOKS_SHA256 = "26db64475e77657eaec6db895c55bad8bcd513344584ef5a64e9a580cf9f648d"
RUAA_REFERENCE_SUBMISSION_SHA256 = "05e334f65d81aaff7ef240e7f4b5c1c9e422e050b906906482bfaa063da90db0"
RUAA_REFERENCE_BASENAME = "reference_submission_stylo.csv"


class ReferenceError(RuntimeError):
    """Fail-closed: a pinned A0 reference file is missing, a symlink, or its SHA256 does not match."""


def _sha256(path: pathlib.Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReferenceError(f"reference missing or a symlink: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pinned(path: pathlib.Path, expected: str, what: str) -> str:
    got = _sha256(path)
    if got != expected:
        raise ReferenceError(f"{what} SHA256 mismatch: got {got}, expected {expected}")
    return got


def _ruaa_sums_entry(sums_path: pathlib.Path, basename: str) -> str:
    """The SHA recorded for ``basename`` in the frozen RuAA SHA256SUMS (fail-closed if absent)."""
    if sums_path.is_symlink() or not sums_path.is_file():
        raise ReferenceError(f"RuAA SHA256SUMS missing or a symlink: {sums_path}")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if name.strip() == basename:
            return digest.strip()
    raise ReferenceError(f"{basename} not listed in {sums_path}")


def verify_a0_references(*, lobo_books: pathlib.Path | str, ruaa_reference_submission: pathlib.Path | str,
                        ruaa_sha256sums: pathlib.Path | str) -> dict:
    """Verify the pinned A0 reference files by SHA256 before any parse; returns the two verified SHAs
    for the RunPlan ``a0_reference_shas`` binding."""
    lobo_sha = _pinned(pathlib.Path(lobo_books), LOBO_BOOKS_SHA256, "docs/lobo_books.txt")
    ruaa_sha = _pinned(pathlib.Path(ruaa_reference_submission), RUAA_REFERENCE_SUBMISSION_SHA256,
                       "RuAA reference submission")
    # RuAA is additionally checked against the frozen SHA256SUMS entry
    listed = _ruaa_sums_entry(pathlib.Path(ruaa_sha256sums), RUAA_REFERENCE_BASENAME)
    if listed != RUAA_REFERENCE_SUBMISSION_SHA256:
        raise ReferenceError(
            f"RuAA SHA256SUMS lists {listed} for {RUAA_REFERENCE_BASENAME}, expected the pinned SHA")
    return {"lobo_books_txt": lobo_sha, "ruaa_reference_submission": ruaa_sha}
