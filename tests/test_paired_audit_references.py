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
