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
import re

LOBO_BOOKS_SHA256 = "26db64475e77657eaec6db895c55bad8bcd513344584ef5a64e9a580cf9f648d"
LOBO_A0_CORRECT = 221
LOBO_A0_TOTAL = 251
RUAA_N_WORKS = 137
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_LOBO_STATUS_RE = re.compile(r"^\[(OK|MISS)\s*\]\s+(.+?)\s+/\s+(\S+)\s+\(rank[^:]*:\s*(\d+)\)\s*$")
_LOBO_TOP_RE = re.compile(r"^\s+топ:\s+(.+?)\s+\(")
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


# ── exact A0 reference parsing (only AFTER the SHA is verified) ───────────────
def parse_lobo_a0_reference(lobo_books: pathlib.Path | str, *, expect_correct: int = LOBO_A0_CORRECT,
                            expect_total: int = LOBO_A0_TOTAL) -> dict:
    """Verify the SHA, then safely parse ``docs/lobo_books.txt`` and assert the stylo LOBO A0 result is
    exactly ``221/251`` with per-work pred/correct/rank consistency. Returns the per-work reference."""
    path = pathlib.Path(lobo_books)
    _pinned(path, LOBO_BOOKS_SHA256, "docs/lobo_books.txt")     # SHA256 before any parse
    lines = path.read_text(encoding="utf-8").splitlines()
    per_work = []
    for i, line in enumerate(lines):
        m = _LOBO_STATUS_RE.match(line)
        if not m:
            continue
        status, author, book, rank = m.group(1), m.group(2).strip(), m.group(3), int(m.group(4))
        correct = status == "OK"
        if correct != (rank == 1):
            raise ReferenceError(f"lobo reference inconsistent: {status} but rank {rank} for {book!r}")
        top = _LOBO_TOP_RE.match(lines[i + 1]) if i + 1 < len(lines) else None
        if not top:
            raise ReferenceError(f"lobo reference missing the top-candidates line for {book!r}")
        pred = top.group(1).strip()
        if correct and pred != author:
            raise ReferenceError(f"lobo OK row pred {pred!r} != true author {author!r} for {book!r}")
        per_work.append({"book": book, "author_display": author, "correct": correct,
                         "rank": rank, "pred_display": pred})
    n_total = len(per_work)
    n_correct = sum(1 for w in per_work if w["correct"])
    if n_total != expect_total:
        raise ReferenceError(f"lobo reference has {n_total} works, expected {expect_total}")
    if n_correct != expect_correct:
        raise ReferenceError(f"lobo A0 accuracy {n_correct}/{n_total} != {expect_correct}/{expect_total}")
    return {"n_correct": n_correct, "n_total": n_total, "per_work": per_work}


def parse_ruaa_a0_reference(ruaa_reference_submission: pathlib.Path | str) -> dict:
    """Verify the SHA, then parse the RuAA A0 reference submission (exactly 137 book_id,pred_author
    rows)."""
    import csv
    path = pathlib.Path(ruaa_reference_submission)
    _pinned(path, RUAA_REFERENCE_SUBMISSION_SHA256, "RuAA reference submission")
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["book_id", "pred_author"]:
            raise ReferenceError("RuAA submission header must be exactly book_id,pred_author")
        for r in reader:
            if not r.get("book_id") or not r.get("pred_author"):
                raise ReferenceError("RuAA submission row missing book_id/pred_author")
            rows.append({"book_id": r["book_id"], "pred_author": r["pred_author"]})
    if len(rows) != RUAA_N_WORKS:
        raise ReferenceError(f"RuAA submission has {len(rows)} rows, expected {RUAA_N_WORKS}")
    if len({r["book_id"] for r in rows}) != len(rows):
        raise ReferenceError("RuAA submission has duplicate book_id")
    return {"n_rows": len(rows), "rows": rows}


def verify_ruaa_inventory(ruaa_sha256sums: pathlib.Path | str, ruaa_root: pathlib.Path | str) -> int:
    """Verify EVERY file listed in the frozen RuAA SHA256SUMS against its recorded digest (fail-closed
    on any missing/symlinked/tampered file). Returns the number of verified files."""
    sums_path = pathlib.Path(ruaa_sha256sums)
    root = pathlib.Path(ruaa_root)
    if sums_path.is_symlink() or not sums_path.is_file():
        raise ReferenceError(f"RuAA SHA256SUMS missing or a symlink: {sums_path}")
    n = 0
    for raw in sums_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, _, name = raw.partition("  ")
        digest, name = digest.strip(), name.strip()
        rel = pathlib.PurePosixPath(name)
        if not (_HEX64_RE.match(digest) and name and ".." not in rel.parts and not rel.is_absolute()):
            raise ReferenceError(f"malformed/unsafe SHA256SUMS entry: {raw[:48]!r}")
        if _sha256(root / name) != digest:                     # _sha256 fails closed on missing/symlink
            raise ReferenceError(f"RuAA inventory digest mismatch for {name}")
        n += 1
    if n == 0:
        raise ReferenceError("RuAA SHA256SUMS is empty")
    return n


def assert_a0_preflight(*, lobo_books: pathlib.Path | str,
                        ruaa_reference_submission: pathlib.Path | str | None = None,
                        ruaa_sha256sums: pathlib.Path | str | None = None,
                        ruaa_root: pathlib.Path | str | None = None,
                        require_ruaa: bool = True) -> dict:
    """The full §3.2 A0 reference preflight the runner must pass before opening any cell: SHA-pin +
    parse lobo_books (221/251), and — when the RuAA reference data is provisioned — the full RuAA
    SHA256SUMS inventory + submission parse. Missing RuAA data with ``require_ruaa`` is fatal (the
    confirmatory run needs it); the unit suite passes ``require_ruaa=False`` on a clean checkout."""
    result = {"lobo": parse_lobo_a0_reference(lobo_books)}
    paths = (ruaa_reference_submission, ruaa_sha256sums, ruaa_root)
    ruaa_present = all(p is not None and pathlib.Path(p).exists() for p in paths)
    if not ruaa_present:
        if require_ruaa:
            raise ReferenceError("RuAA reference data absent — the confirmatory A0 preflight cannot run")
        result["ruaa"] = None
        return result
    n_inventory = verify_ruaa_inventory(ruaa_sha256sums, ruaa_root)
    result["ruaa"] = {"n_inventory": n_inventory, **parse_ruaa_a0_reference(ruaa_reference_submission)}
    return result
