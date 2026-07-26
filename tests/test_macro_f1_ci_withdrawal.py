"""Gate: the author-clustered macro-F1 CI stays WITHDRAWN across every published surface.

author-clustered bootstrap resamples authors, which changes the class set of the macro-average
(a dropped-but-predicted author contributes F1=0), so the old interval is not the CI of one fixed
43-class function. It is withdrawn — not "conservative". This gate fails closed if the interval or
the "conservative" wording leaks back into the source JSON, site data, README,
or any versioned generator. Local-only paper drafts are outside the release.
"""
from __future__ import annotations

import pathlib

from stylo import jsonio

ROOT = pathlib.Path(__file__).resolve().parents[1]

STATUS = "withdrawn_pending_preregistered_recompute"
SUPERSEDED = [0.6222, 0.8369]
ERRATUM_REF = "docs/macro_f1_ci_withdrawal.json"
# the withdrawn interval must never appear rendered as an active macro-F1 CI
BANNED_INTERVAL_STRINGS = ["0.6222, 0.8369", "0.622, 0.837", "0.6222,0.8369", "0.622,0.837"]
BANNED_WORDING = "интервал консервативен"


def _authorci() -> dict:
    return jsonio.load_strict(ROOT / "docs" / "stylo_lobo_authorci.json")


def test_source_json_interval_is_json_null_not_a_magic_string():
    d = _authorci()
    # the CI KEY must be JSON null — never a magic string, never an array (no type flip, no return)
    assert d["macro_f1_authorclustered_CI"] is None
    assert not isinstance(d["macro_f1_authorclustered_CI"], (str, list))


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


def test_versioned_readme_does_not_publish_the_withdrawn_interval():
    for name in ("README.md",):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert BANNED_WORDING not in text, f"{name} reintroduced the 'conservative interval' claim"
        for frag in BANNED_INTERVAL_STRINGS:
            assert frag not in text, f"{name} renders the withdrawn macro-F1 interval {frag!r}"
        # positive: the withdrawal must be stated
        assert STATUS in text, f"{name} does not state the macro-F1 CI withdrawal status"


def test_generators_cannot_reintroduce_conservative_wording():
    # the templates themselves must not carry the banned wording (a re-run must stay clean)
    for gen in ("gen-readme.mjs", "gen-site-data.mjs"):
        src = (ROOT / "scripts" / gen).read_text(encoding="utf-8")
        assert BANNED_WORDING not in src, f"scripts/{gen} still contains the banned wording"


def test_site_sections_do_not_index_withdrawn_ci():
    # a null macroF1CI must not be indexed ([0]/[1]) by any rendered section (would crash the build)
    for sec in ("Hero.jsx", "Results.jsx", "Method.jsx"):
        src = (ROOT / "site" / "src" / "sections" / sec).read_text(encoding="utf-8")
        assert "macroF1CI[0]" not in src and "macroF1CI[1]" not in src, \
            f"{sec} still indexes the withdrawn macroF1CI array"
