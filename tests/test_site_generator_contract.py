from __future__ import annotations

import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "gen-site-data.mjs"
RENDER_SMOKE = ROOT / "site" / "scripts" / "check-render.mjs"


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


def test_site_build_executes_a_real_server_render_smoke():
    package = json.loads((ROOT / "site" / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["test:render"] == "node ./scripts/check-render.mjs"
    assert "npm run test:render" in package["scripts"]["build"]

    source = RENDER_SMOKE.read_text(encoding="utf-8")
    assert 'ssrLoadModule("/src/App.jsx")' in source
    assert "renderToStaticMarkup" in source


def test_method_does_not_render_the_withdrawn_macro_f1_interval():
    source = (ROOT / "site" / "src" / "sections" / "Method.jsx").read_text(encoding="utf-8")
    assert "MF1_CI" not in source
    assert "точность по авторам в диапазоне" not in source
    assert "Интервал macro-F1 отозван" in source
