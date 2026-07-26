from __future__ import annotations

import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "gen-site-data.mjs"


def test_site_generator_strict_input_self_test():
    completed = subprocess.run(
        ["node", str(GENERATOR), "--self-test"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "strict-input self-test: OK" in completed.stdout


def test_site_generator_has_no_token_rewrite_or_subtree_null_allowlist():
    source = GENERATOR.read_text(encoding="utf-8")
    assert r".replace(/\bNaN\b/g" not in source
    assert "h.includes(a)" not in source
    assert "NULLABLE_PATHS.has(h)" in source
