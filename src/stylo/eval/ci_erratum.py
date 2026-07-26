"""Author-clustered Δacc CI-sign erratum — the fail-closed, idempotent, SHA-pinned correction.

The historical headline/RuAA artifacts emitted ``vs_stylo_dacc_authorclustered_ci`` with the
opposite sign to its point estimate (``stylo − spec`` instead of ``spec − stylo``). The point,
accuracy, macro-F1, McNemar and the ``significant`` flag are unaffected; only the CI column's sign
is wrong. The exact fix is the algebraic reversal ``[lo, hi] → [-hi, -lo]``.

Design invariants (see the CI-sign erratum blocker):
  * the historical ``docs/final_comparison.{csv,txt}`` and ``docs/ruaa_bench_v1.json`` +
    ``docs/ruaa_bench_leaderboard.md`` are **immutable**: their SHA256 are pinned here and re-checked
    before AND after every run — a drift fails closed and nothing is written;
  * the correction only ever reads the FROZEN v1 inputs, so a double flip is structurally
    impossible; a value-level guard additionally rejects an already-corrected (non-positive) CI;
  * NaN/inf, ``lo > hi`` and any non-``[±x,±y]`` token fail closed;
  * corrected output goes ONLY to versioned paths (``*.v2.*`` / ``*_v1.0.1.*``) — writing a frozen
    v1 path is refused by ``assert_publish_target_not_frozen`` (future generators must not overwrite
    the frozen snapshot).
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import re
from typing import Callable, Tuple

CI_COL = "vs_stylo_dacc_authorclustered_ci"

# Frozen baseline-snapshot SHA256 (also pinned in docs/p0_baseline_snapshot.json). Immutable inputs.
FROZEN_SHA256 = {
    "final_comparison.csv": "31bba7af930685fc9862fe6b1806b3f2ba5ba21b6726e66757e4dd756a3ded6f",
    "final_comparison.txt": "5b2e6f6b6f18c87ce13183413cdec9e136b873be6674856f5780656dcda92d96",
    "ruaa_bench_v1.json": "c7228c5019e211afe6c3ee323bcf82ec3dd7a71c472e84ef0cd3bf51fb83ffc6",
    "ruaa_bench_leaderboard.md": "dcf9993b0f3a0057487a331a3c54305f1a5dc70342e86739976f37feeb760c2a",
}
# Frozen v1 → corrected versioned name. Correction NEVER writes a key of FROZEN_SHA256.
CORRECTED_OF = {
    "final_comparison.csv": "final_comparison.v2.csv",
    "final_comparison.txt": "final_comparison.v2.txt",
    "ruaa_bench_v1.json": "ruaa_bench_v1.0.1.json",
    "ruaa_bench_leaderboard.md": "ruaa_bench_leaderboard_v1.0.1.md",
}

# a signed interval token: BOTH bounds carry an explicit +/- sign (the un-signed book acc CI
# ``[0.841,0.916]`` therefore never matches and is never touched).
_SIGNED_TOKEN = re.compile(r"\[([+-]\d+(?:\.\d+)?),([+-]\d+(?:\.\d+)?)\]")


class CiErratumError(ValueError):
    """Fail-closed CI-sign erratum violation."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    return sha256_bytes(pathlib.Path(p).read_bytes())


def flip_pair(lo: float, hi: float) -> Tuple[float, float]:
    """``[lo, hi] → [-hi, -lo]`` with fail-closed validation of the ORIGINAL (un-corrected) CI."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise CiErratumError(f"non-finite CI bound: [{lo}, {hi}]")
    if lo > hi:
        raise CiErratumError(f"malformed CI (lo > hi): [{lo}, {hi}]")
    # the historical un-corrected CI is strictly positive (stylo − spec with stylo the reference);
    # a non-positive bound means the sign was already corrected — refuse the (double) flip.
    if lo <= 0 or hi <= 0:
        raise CiErratumError(f"CI already corrected / non-positive (double flip refused): [{lo}, {hi}]")
    flo, fhi = -hi, -lo
    if flo > fhi:                                   # defensive; cannot happen for 0 < lo <= hi
        raise CiErratumError("flip produced lo > hi")
    return flo, fhi


def flip_ci_string(s: str) -> str:
    """Flip a single CI cell ``'[+0.306,+0.508]' -> '[-0.508,-0.306]'`` (blank passes through)."""
    if s is None or not str(s).strip():
        return s
    m = re.fullmatch(r"\[([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\]", str(s).strip())
    if not m:
        raise CiErratumError(f"unparseable CI string: {s!r}")
    flo, fhi = flip_pair(float(m.group(1)), float(m.group(2)))
    return f"[{flo:+.3f},{fhi:+.3f}]"


def flip_signed_tokens(text: str) -> Tuple[str, int]:
    """Flip every SIGNED interval token in raw text, byte-preserving everything else. Returns
    (new_text, n_flipped). Un-signed intervals (book acc CI) are left untouched."""
    n = 0

    def _sub(m: re.Match) -> str:
        nonlocal n
        n += 1
        flo, fhi = flip_pair(float(m.group(1)), float(m.group(2)))
        return f"[{flo:+.3f},{fhi:+.3f}]"

    return _SIGNED_TOKEN.sub(_sub, text), n


def assert_publish_target_not_frozen(path) -> None:
    """Fail-closed: a (future) generator must not publish to a frozen v1 CI-artifact path.

    A frozen artifact is the canonical ``docs/<name>`` (parent dir literally named ``docs``); a
    corrected/exploratory copy under ``docs/exploratory/…`` or a versioned ``*.v2.*`` name is fine."""
    p = pathlib.Path(path)
    if p.name in FROZEN_SHA256 and p.parent.name == "docs":
        raise CiErratumError(
            f"refusing to write frozen v1 CI artifact {p.name!r}; corrected output is "
            f"{CORRECTED_OF[p.name]!r} (produced by the CI-sign erratum from the frozen snapshot)")


def verify_frozen(docs: pathlib.Path) -> None:
    """Every frozen input must exist and match its pinned SHA256 — else fail closed (no write)."""
    for name, want in FROZEN_SHA256.items():
        p = docs / name
        if not p.exists():
            raise CiErratumError(f"frozen input missing: docs/{name}")
        got = sha256_file(p)
        if got != want:
            raise CiErratumError(f"frozen input docs/{name} SHA256 {got} != pinned {want}")


# ── the four corrections ──────────────────────────────────────────────────────
def _correct_text(docs: pathlib.Path, src_name: str, expect_flips: int) -> dict:
    src = docs / src_name
    corrected, n = flip_signed_tokens(src.read_text(encoding="utf-8"))
    if n != expect_flips:
        raise CiErratumError(f"{src_name}: flipped {n} CI tokens, expected {expect_flips}")
    dst_name = CORRECTED_OF[src_name]
    assert_publish_target_not_frozen(dst_name)
    (docs / dst_name).write_text(corrected, encoding="utf-8")
    return {"original": src_name, "original_sha256": sha256_file(src),
            "corrected": dst_name, "corrected_sha256": sha256_bytes(corrected.encode("utf-8")),
            "corrected_column": CI_COL}


def _correct_ruaa_json(docs: pathlib.Path, dumps_strict, loads_strict) -> dict:
    src = docs / "ruaa_bench_v1.json"
    orig_sha = sha256_file(src)
    data = loads_strict(src.read_text(encoding="utf-8"))
    data["benchmark"] = str(data.get("benchmark", "")).replace("v1.0", "v1.0.1")
    data["supersedes"] = {"file": "ruaa_bench_v1.json", "sha256": orig_sha,
                          "reason": "author-clustered CI sign erratum"}
    flipped = 0
    for row in data.get("leaderboard", []):
        if CI_COL in row and str(row[CI_COL]).strip():
            row[CI_COL] = flip_ci_string(row[CI_COL])
            flipped += 1
    dst_name = CORRECTED_OF["ruaa_bench_v1.json"]
    assert_publish_target_not_frozen(dst_name)
    out = dumps_strict(data, indent=2) + "\n"
    (docs / dst_name).write_text(out, encoding="utf-8")
    return {"original": "ruaa_bench_v1.json", "original_sha256": orig_sha,
            "corrected": dst_name, "corrected_sha256": sha256_bytes(out.encode("utf-8")),
            "corrected_column": CI_COL, "rows_flipped": flipped}


def apply_erratum(root: pathlib.Path, dumps_strict: Callable, loads_strict: Callable) -> dict:
    """Run the full fail-closed erratum. Deterministic + idempotent: reads only the SHA-pinned
    frozen v1 inputs, writes only versioned corrected outputs, and re-verifies the frozen inputs
    are byte-unchanged afterwards. Returns the erratum record (also written to ci_sign_erratum.json)."""
    docs = pathlib.Path(root) / "docs"
    verify_frozen(docs)                                   # fail-closed BEFORE any write

    artifacts = [
        _correct_text(docs, "final_comparison.csv", expect_flips=9),
        _correct_text(docs, "final_comparison.txt", expect_flips=9),
        _correct_ruaa_json(docs, dumps_strict, loads_strict),
        _correct_text(docs, "ruaa_bench_leaderboard.md", expect_flips=6),
    ]
    # bump the corrected leaderboard's version label (content-identical bytes otherwise)
    lb = docs / CORRECTED_OF["ruaa_bench_leaderboard.md"]
    lb_text = lb.read_text(encoding="utf-8").replace("RuAA-Bench v1.0 —", "RuAA-Bench v1.0.1 —")
    lb.write_text(lb_text, encoding="utf-8")
    for a in artifacts:
        if a["corrected"] == CORRECTED_OF["ruaa_bench_leaderboard.md"]:
            a["corrected_sha256"] = sha256_bytes(lb_text.encode("utf-8"))

    verify_frozen(docs)                                   # frozen inputs must be byte-unchanged

    erratum = {
        "erratum": "author_clustered_ci_sign",
        "transform": "[lo, hi] -> [-hi, -lo] (CI now matches the point estimate spec - stylo)",
        "unchanged": ["accuracy", "macro_f1", "vs_stylo_dacc", "vs_stylo_mcnemar_p",
                      "vs_stylo_dacc_authorclustered_sig", "headline stylo accuracy 0.8805"],
        "frozen_inputs": dict(FROZEN_SHA256),
        "artifacts": artifacts,
        "note": "historical files retained as superseded for this column; consumers should use the corrected version",
    }
    (docs / "ci_sign_erratum.json").write_text(dumps_strict(erratum, indent=2) + "\n",
                                               encoding="utf-8")
    return erratum
