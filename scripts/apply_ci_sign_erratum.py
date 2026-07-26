#!/usr/bin/env python3
"""CI-sign erratum runner (fail-closed, SHA-pinned, idempotent).

The author-clustered Δacc CI was emitted with the opposite sign to its point estimate
(``stylo − spec`` instead of ``spec − stylo``). The point, accuracy, macro-F1, McNemar and the
``significant`` flag are unaffected; only the CI column's sign is wrong. The exact fix is the
algebraic reversal ``[lo, hi] → [-hi, -lo]``.

This runner does NOT regenerate or overwrite the historical artifacts. It re-checks their pinned
SHA256, emits versioned derivatives (``docs/final_comparison.v2.{csv,txt}``,
``docs/ruaa_bench_v1.0.1.json``, ``docs/ruaa_bench_leaderboard_v1.0.1.md``) and a machine-readable
erratum + corrected-SHA inventory (``docs/ci_sign_erratum.json``). All logic (SHA pins, fail-closed
validation, the frozen-path guard) lives in ``stylo.eval.ci_erratum``. Run from the repo root.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylo.eval.ci_erratum import apply_erratum  # noqa: E402
from stylo.jsonio import dumps_strict, loads_strict  # noqa: E402  (strict JSON only)


def main() -> int:
    apply_erratum(ROOT, dumps_strict, loads_strict)
    print("wrote docs/final_comparison.v2.{csv,txt}, docs/ruaa_bench_v1.0.1.json, "
          "docs/ruaa_bench_leaderboard_v1.0.1.md, docs/ci_sign_erratum.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
