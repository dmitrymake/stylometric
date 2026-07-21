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


# ── §3.2 exact one-to-one A0 contract (compare by full work id, pred included, no basename) ──
# a tiny display-space reference: two authors, plus a book named "w1" under BOTH (duplicate basename)
_PARSED = {"per_work": [
    {"book": "w1", "author_display": "Автор А", "pred_display": "Автор А", "correct": True, "rank": 1},
    {"book": "w2", "author_display": "Автор А", "pred_display": "Автор Б", "correct": False, "rank": 3},
    {"book": "w1", "author_display": "Автор Б", "pred_display": "Автор Б", "correct": True, "rank": 1}]}
_MAP = {"Автор А": "aa", "Автор Б": "bb"}


def _ref_index():
    return ref.build_lobo_reference_index(_PARSED, _MAP)


def test_lobo_reference_index_keeps_duplicate_basename_distinct():
    idx = _ref_index()
    # the two "w1" books are NOT collapsed — distinct work ids under distinct authors
    assert set(idx) == {"aa/w1", "aa/w2", "bb/w1"}
    assert idx["aa/w1"]["true_author"] == "aa" and idx["bb/w1"]["true_author"] == "bb"


def test_lobo_a0_exact_match_passes():
    idx = _ref_index()
    a0 = {k: dict(v) for k, v in idx.items()}
    ref.assert_a0_matches_index(a0, idx, fields=ref.A0_LOBO_FIELDS, label="stylo lobo")


def test_lobo_a0_all_preds_wrong_but_correct_rank_match_is_rejected():
    # every prediction is forged to the wrong author while correct/rank are left intact — must fail
    idx = _ref_index()
    a0 = {k: {**v, "pred": "bb" if v["pred"] == "aa" else "aa"} for k, v in idx.items()}
    with pytest.raises(ref.ReferenceError):
        ref.assert_a0_matches_index(a0, idx, fields=ref.A0_LOBO_FIELDS, label="stylo lobo")


def test_lobo_a0_duplicate_work_id_is_fatal_at_build():
    dup = {"per_work": _PARSED["per_work"] + [_PARSED["per_work"][0]]}   # repeat aa/w1 exactly
    with pytest.raises(ref.ReferenceError):
        ref.build_lobo_reference_index(dup, _MAP)


def test_lobo_a0_missing_and_extra_rows_are_fatal():
    idx = _ref_index()
    missing = {k: v for k, v in idx.items() if k != "bb/w1"}             # a reference key absent
    with pytest.raises(ref.ReferenceError):
        ref.assert_a0_matches_index(missing, idx, fields=ref.A0_LOBO_FIELDS, label="stylo lobo")
    extra = {**idx, "cc/w9": {"true_author": "cc", "pred": "cc", "correct": True, "rank": 1}}
    with pytest.raises(ref.ReferenceError):
        ref.assert_a0_matches_index(extra, idx, fields=ref.A0_LOBO_FIELDS, label="stylo lobo")


def test_lobo_a0_row_permutation_still_matches():
    # order independence: reversing insertion order changes nothing (comparison is by work id)
    idx = _ref_index()
    shuffled = dict(reversed(list(idx.items())))
    ref.assert_a0_matches_index(shuffled, idx, fields=ref.A0_LOBO_FIELDS, label="stylo lobo")


def test_lobo_reference_unmapped_author_or_pred_is_fatal():
    with pytest.raises(ref.ReferenceError):    # pred name not in the display->slug map
        ref.build_lobo_reference_index(
            {"per_work": [{"book": "w1", "author_display": "Автор А", "pred_display": "Кто-то",
                           "correct": False, "rank": 4}]}, _MAP)


def test_ruaa_a0_exact_pred_match_and_mismatch():
    parsed = {"rows": [{"book_id": "b1", "pred_author": "aa"}, {"book_id": "b2", "pred_author": "bb"}]}
    idx = ref.build_ruaa_reference_index(parsed)
    ref.assert_a0_matches_index({"b1": {"pred": "aa"}, "b2": {"pred": "bb"}}, idx,
                                fields=ref.A0_RUAA_FIELDS, label="stylo ruaa")
    with pytest.raises(ref.ReferenceError):                              # b2 pred forged
        ref.assert_a0_matches_index({"b1": {"pred": "aa"}, "b2": {"pred": "aa"}}, idx,
                                    fields=ref.A0_RUAA_FIELDS, label="stylo ruaa")
    with pytest.raises(ref.ReferenceError):                              # duplicate book_id in reference
        ref.build_ruaa_reference_index({"rows": parsed["rows"] + [parsed["rows"][0]]})


# ── §4.2 golden-fixture inventory computed FROM DISK ──────────────────────────
def test_golden_fixture_inventory_from_disk(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("alpha", encoding="utf-8")
    b.write_text("beta", encoding="utf-8")
    d1 = ref.golden_fixture_inventory({"a": a, "b": b})
    assert d1 == ref.golden_fixture_inventory({"a": a, "b": b})          # deterministic replay
    a.write_text("ALPHA", encoding="utf-8")
    assert ref.golden_fixture_inventory({"a": a, "b": b}) != d1          # a content change re-keys it
    with pytest.raises(ref.ReferenceError):                             # empty inventory
        ref.golden_fixture_inventory({})
    with pytest.raises(ref.ReferenceError):                             # a missing golden file
        ref.golden_fixture_inventory({"missing": tmp_path / "nope.txt"})
