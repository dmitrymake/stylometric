"""Data contract for the withdrawn author-clustered macro-F1 confidence interval.

Resampling authors changes the class set of the macro-average. The old interval
is withdrawn; the source and generated data retain a null CI and its erratum.
"""
from __future__ import annotations

import pathlib

from stylo import jsonio

ROOT = pathlib.Path(__file__).resolve().parents[1]

STATUS = "withdrawn_pending_preregistered_recompute"
SUPERSEDED = [0.6222, 0.8369]
ERRATUM_REF = "docs/macro_f1_ci_withdrawal.json"


def _authorci() -> dict:
    return jsonio.load_strict(ROOT / "docs" / "stylo_lobo_authorci.json")


def test_source_json_interval_is_json_null_not_a_magic_string():
    d = _authorci()
    # the CI KEY must be JSON null — never a magic string, never an array (no type flip, no return)
    assert d["macro_f1_authorclustered_CI"] is None


def test_source_json_withdrawal_schema():
    d = _authorci()
    assert d["macro_f1_authorclustered_interval_status"] == STATUS
    assert d["macro_f1_authorclustered_superseded_interval"] == SUPERSEDED
    assert d["macro_f1_authorclustered_erratum_ref"] == ERRATUM_REF
    # accuracy CI is UNCHANGED (only macro-F1 is withdrawn)
    assert d["accuracy_authorclustered_CI"] == [0.8116, 0.9366]


def test_erratum_record_present_and_consistent():
    p = ROOT / ERRATUM_REF
    assert p.exists(), f"missing erratum record {ERRATUM_REF}"
    rec = jsonio.load_strict(p)
    assert rec["interval_status"] == STATUS
    assert rec["superseded_interval"] == SUPERSEDED
    assert rec["affected"]["value_after_withdrawal"] is None
    assert rec["not_a_conservative_interval"] is True


def test_site_data_headline_ci_withdrawn():
    sd = jsonio.load_strict(ROOT / "site" / "src" / "generated" / "site-data.json")
    h = sd["headline"]
    assert h["macroF1CI"] is None
    assert h["macroF1CIStatus"] == STATUS
    assert h["macroF1CIErratumRef"] == ERRATUM_REF
