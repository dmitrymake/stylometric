"""Repo-wide gate: no production writer overwrites a frozen artifact or reintroduces the withdrawn
macro-F1 CI. Scans ALL production code (src/, scripts/, log/), not just the gen-*.mjs generators —
a fast/legacy recompute (cli evaluate, log/fast_eval, log/correct_macrof1_convention) must route to
the exploratory/versioned namespace or fail closed, never to the frozen docs/ headline sources.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# frozen CI artifacts (immutable baseline and CI-sign-erratum sources), by canonical docs/ basename
FROZEN = {"final_comparison.csv", "final_comparison.txt", "ruaa_bench_v1.json", "ruaa_bench_leaderboard.md"}
HEADLINE_JSON = "stylo_lobo_authorci.json"
GUARD = "assert_publish_target_not_frozen"
WITHDRAWN_MARKER = "macro_f1_authorclustered_interval_status"
BANNED_WORDING = "интервал консервативен"
_WRITE_ATTRS = {"write_text", "write_bytes", "to_csv"}
_DUMP_FUNCS = {"dump", "dumps", "dump_strict", "dumps_strict"}


def _py_files():
    for d in ("src", "scripts", "log"):
        base = ROOT / d
        if base.exists():
            yield from base.rglob("*.py")


def _tainted_path_vars(tree, needle):
    """Names assigned a path expression whose source mentions ``needle`` (e.g. a frozen basename)."""
    tainted = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and needle in ast.unparse(n.value):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    tainted.add(t.id)
    return tainted


def _writes_path(tree, needle):
    """True iff the module has a write to a path that resolves to ``needle`` (literal or tainted var)."""
    tainted = _tainted_path_vars(tree, needle)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        if isinstance(fn, ast.Attribute) and fn.attr in _WRITE_ATTRS:
            recv = fn.value
            if (isinstance(recv, ast.Name) and recv.id in tainted) or needle in ast.unparse(n):
                return True
        fname = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
        if fname in _DUMP_FUNCS:
            for a in n.args:
                if (isinstance(a, ast.Name) and a.id in tainted) or needle in ast.unparse(a):
                    return True
    return False


def test_no_production_writer_targets_a_frozen_ci_artifact_without_the_guard():
    offenders = []
    for p in _py_files():
        if p.name == "ci_erratum.py":              # defines the guard; writes only versioned/corrected paths
            continue
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        uses_guard = GUARD in src
        for name in FROZEN:
            if _writes_path(tree, name) and not uses_guard:
                offenders.append(f"{p.relative_to(ROOT)} -> {name}")
    assert not offenders, "production writer targets a frozen CI artifact without the guard:\n" + "\n".join(offenders)


def test_headline_json_writers_carry_the_withdrawal_marker():
    offenders = []
    for p in _py_files():
        src = p.read_text(encoding="utf-8")
        if _writes_path(ast.parse(src), HEADLINE_JSON) and WITHDRAWN_MARKER not in src:
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"stylo_lobo_authorci.json writer missing the withdrawal marker: {offenders}"


def test_no_production_code_has_the_conservative_wording():
    hits = []
    files = list(_py_files()) + list((ROOT / "scripts").glob("*.mjs"))
    for p in files:
        if BANNED_WORDING in p.read_text(encoding="utf-8"):
            hits.append(str(p.relative_to(ROOT)))
    assert not hits, f"banned 'conservative interval' wording present in: {hits}"
