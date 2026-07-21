"""Tests for the pinned A0 reference verifier (§3.2) and the frozen legacy anchor (§1.2)."""
from __future__ import annotations

import json
import pathlib

import pytest

from stylo.eval.paired_audit import references as ref
from stylo.eval.paired_audit import semantic_parity as sp

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_real_a0_references_verify_against_pinned_shas():
    # RuAA reference data lives under gitignored data/ — a clean checkout must not depend on it.
    ruaa = _ROOT / "data/ruaa_bench_v1/reference_submission_stylo.csv"
    sums = _ROOT / "data/ruaa_bench_v1/SHA256SUMS"
    if not (ruaa.is_file() and sums.is_file()):
        pytest.skip("RuAA reference data not provisioned (ignored private data)")
    out = ref.verify_a0_references(lobo_books=_ROOT / "docs/lobo_books.txt",
                                   ruaa_reference_submission=ruaa, ruaa_sha256sums=sums)
    assert out["lobo_books_txt"] == ref.LOBO_BOOKS_SHA256
    assert out["ruaa_reference_submission"] == ref.RUAA_REFERENCE_SUBMISSION_SHA256


def test_lobo_reference_alone_verifies_on_clean_checkout():
    # docs/lobo_books.txt IS committed, so its pin is verifiable without any private data
    assert ref._pinned(_ROOT / "docs/lobo_books.txt", ref.LOBO_BOOKS_SHA256, "lobo") == ref.LOBO_BOOKS_SHA256


def test_tampered_reference_rejected(tmp_path):
    bad = tmp_path / "lobo_books.txt"
    bad.write_text("tampered", encoding="utf-8")
    with pytest.raises(ref.ReferenceError):
        ref.verify_a0_references(
            lobo_books=bad,
            ruaa_reference_submission=_ROOT / "data/ruaa_bench_v1/reference_submission_stylo.csv",
            ruaa_sha256sums=_ROOT / "data/ruaa_bench_v1/SHA256SUMS")


def test_missing_reference_rejected(tmp_path):
    with pytest.raises(ref.ReferenceError):
        ref.verify_a0_references(lobo_books=tmp_path / "nope.txt",
                                 ruaa_reference_submission=tmp_path / "nope.csv",
                                 ruaa_sha256sums=tmp_path / "nope.sums")


def test_legacy_anchor_matches_screening_panel():
    panel = json.loads((_ROOT / "docs/screening_panel_v1.json").read_text(encoding="utf-8"))
    assert panel["parent_dataset_digest"] == sp.LEGACY_ANCHOR


# ── §3.2 exact A0 reference parsing ──────────────────────────────────────────
_RUAA_SUB = _ROOT / "data/ruaa_bench_v1/reference_submission_stylo.csv"
_RUAA_SUMS = _ROOT / "data/ruaa_bench_v1/SHA256SUMS"
_RUAA_ROOT = _ROOT / "data/ruaa_bench_v1"


def test_parse_lobo_a0_is_221_of_251():
    parsed = ref.parse_lobo_a0_reference(_ROOT / "docs/lobo_books.txt")
    assert parsed["n_correct"] == 221 and parsed["n_total"] == 251
    assert len(parsed["per_work"]) == 251
    # every OK row is rank 1 with pred == true author; MISS rows are rank > 1
    for w in parsed["per_work"]:
        assert (w["rank"] == 1) == w["correct"]
        if w["correct"]:
            assert w["pred_display"] == w["author_display"]


def test_parse_lobo_wrong_expectation_fails():
    with pytest.raises(ref.ReferenceError):
        ref.parse_lobo_a0_reference(_ROOT / "docs/lobo_books.txt", expect_correct=999)


def test_ruaa_inventory_verifier_logic_and_real_submission(tmp_path):
    # the inventory-verifier LOGIC is tested on synthetic data (always); the real full inventory is a
    # runner provisioning concern (data/ is gitignored and may drift, e.g. protocol.md).
    import hashlib as _h

    def _sha(p):
        return _h.sha256(p.read_bytes()).hexdigest()

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{_sha(tmp_path / 'a.txt')}  a.txt\n{_sha(tmp_path / 'b.txt')}  b.txt\n",
                    encoding="utf-8")
    assert ref.verify_ruaa_inventory(sums, tmp_path) == 2
    (tmp_path / "a.txt").write_text("TAMPERED", encoding="utf-8")
    with pytest.raises(ref.ReferenceError):
        ref.verify_ruaa_inventory(sums, tmp_path)
    # the real SHA-pinned submission parses to exactly 137 rows when the private data is provisioned
    if _RUAA_SUB.is_file():
        assert ref.parse_ruaa_a0_reference(_RUAA_SUB)["n_rows"] == 137


def test_a0_preflight_lobo_only_on_clean_checkout():
    out = ref.assert_a0_preflight(lobo_books=_ROOT / "docs/lobo_books.txt", require_ruaa=False)
    assert out["lobo"]["n_correct"] == 221 and out["ruaa"] is None


def test_a0_preflight_requires_ruaa_when_absent(tmp_path):
    with pytest.raises(ref.ReferenceError):
        ref.assert_a0_preflight(lobo_books=_ROOT / "docs/lobo_books.txt",
                                ruaa_reference_submission=tmp_path / "nope.csv",
                                ruaa_sha256sums=tmp_path / "nope.sums",
                                ruaa_root=tmp_path / "nope", require_ruaa=True)
