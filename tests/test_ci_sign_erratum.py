"""Gate: the author-clustered Δacc CI-sign erratum is fail-closed, algebraically exact, idempotent.

The correction is the algebraic reversal ``[lo, hi] -> [-hi, -lo]`` applied ONLY to the frozen v1
snapshots (SHA-pinned). This gate proves: the frozen inputs are byte-frozen, the committed corrected
artifacts are exactly the flip of the frozen ones, a double flip is impossible, and the fail-closed
validators reject NaN/inf, lo>hi, bad format and an already-corrected sign.
"""
from __future__ import annotations

import pathlib

import pytest

from stylo import jsonio
from stylo.eval import ci_erratum as ce

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_frozen_inputs_match_pinned_sha256():
    ce.verify_frozen(DOCS)                      # exists + SHA == pin for all four frozen artifacts
    # the two SHAs named in the blocker are exactly the pins
    assert ce.FROZEN_SHA256["final_comparison.csv"] == \
        "31bba7af930685fc9862fe6b1806b3f2ba5ba21b6726e66757e4dd756a3ded6f"
    assert ce.FROZEN_SHA256["ruaa_bench_v1.json"] == \
        "c7228c5019e211afe6c3ee323bcf82ec3dd7a71c472e84ef0cd3bf51fb83ffc6"


def test_flip_is_exact_and_signed():
    assert ce.flip_ci_string("[+0.306,+0.508]") == "[-0.508,-0.306]"
    assert ce.flip_ci_string("[+0.017,+0.167]") == "[-0.167,-0.017]"
    assert ce.flip_ci_string("") == ""          # stylo row passes through
    assert ce.flip_pair(0.02, 0.15) == (-0.15, -0.02)


def test_fail_closed_on_bad_or_already_corrected_ci():
    with pytest.raises(ce.CiErratumError):      # already corrected (negative) -> double flip refused
        ce.flip_ci_string("[-0.508,-0.306]")
    with pytest.raises(ce.CiErratumError):      # non-positive bound
        ce.flip_pair(-0.1, 0.2)
    with pytest.raises(ce.CiErratumError):      # lo > hi
        ce.flip_pair(0.5, 0.2)
    with pytest.raises(ce.CiErratumError):      # NaN
        ce.flip_pair(float("nan"), 0.2)
    with pytest.raises(ce.CiErratumError):      # inf
        ce.flip_pair(float("inf"), 0.2)
    for bad in ("[0.1]", "0.1,0.2", "[a,b]", "[nan,0.2]", "[0.1,inf]"):
        with pytest.raises(ce.CiErratumError):
            ce.flip_ci_string(bad)


def test_future_generators_cannot_write_frozen_v1_paths():
    for frozen in ce.FROZEN_SHA256:
        with pytest.raises(ce.CiErratumError):
            ce.assert_publish_target_not_frozen(DOCS / frozen)
    # a versioned corrected path is allowed
    for corrected in ce.CORRECTED_OF.values():
        ce.assert_publish_target_not_frozen(DOCS / corrected)


def test_committed_corrections_are_exactly_the_flip_of_the_frozen_inputs():
    # CSV / TXT / leaderboard: byte-preserving flip of the frozen text
    for src, dst, n in (("final_comparison.csv", "final_comparison.v2.csv", 9),
                        ("final_comparison.txt", "final_comparison.v2.txt", 9)):
        want, flips = ce.flip_signed_tokens((DOCS / src).read_text(encoding="utf-8"))
        assert flips == n
        assert (DOCS / dst).read_text(encoding="utf-8") == want
    # leaderboard flip + version bump
    lb, flips = ce.flip_signed_tokens((DOCS / "ruaa_bench_leaderboard.md").read_text(encoding="utf-8"))
    assert flips == 6
    lb = lb.replace("RuAA-Bench v1.0 —", "RuAA-Bench v1.0.1 —")
    assert (DOCS / "ruaa_bench_leaderboard_v1.0.1.md").read_text(encoding="utf-8") == lb


def test_committed_ruaa_json_ci_column_is_the_flip():
    v1 = jsonio.load_strict(DOCS / "ruaa_bench_v1.json")["leaderboard"]
    v101 = jsonio.load_strict(DOCS / "ruaa_bench_v1.0.1.json")
    for a, b in zip(v1, v101["leaderboard"]):
        assert b[ce.CI_COL] == ce.flip_ci_string(a[ce.CI_COL])
    assert v101["supersedes"]["sha256"] == ce.FROZEN_SHA256["ruaa_bench_v1.json"]
    assert v101["benchmark"] == "RuAA-Bench v1.0.1"


def test_double_flip_impossible_on_corrected_files():
    # a corrected file's CIs are all negative -> re-flipping fails closed
    for dst in ("final_comparison.v2.csv", "final_comparison.v2.txt",
                "ruaa_bench_leaderboard_v1.0.1.md"):
        with pytest.raises(ce.CiErratumError):
            ce.flip_signed_tokens((DOCS / dst).read_text(encoding="utf-8"))


def test_erratum_record_inventory_matches_disk():
    rec = jsonio.load_strict(DOCS / "ci_sign_erratum.json")
    assert rec["frozen_inputs"] == ce.FROZEN_SHA256
    for art in rec["artifacts"]:
        assert art["original_sha256"] == ce.FROZEN_SHA256[art["original"]]
        assert art["corrected_sha256"] == ce.sha256_file(DOCS / art["corrected"])
        assert art["corrected"] in ce.CORRECTED_OF.values()
