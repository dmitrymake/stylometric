from __future__ import annotations

import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "gen-site-data.mjs"
README_GENERATOR = ROOT / "scripts" / "gen-readme.mjs"
RENDER_SMOKE = ROOT / "site" / "scripts" / "check-render.mjs"
HISTORICAL_NOTICE = (
    ROOT / "site" / "src" / "components" / "HistoricalHeadlineNotice.jsx"
)
PUBLIC_HEADLINE_SECTIONS = (
    "Hero.jsx",
    "Method.jsx",
    "Results.jsx",
    "Corpus.jsx",
    "Repro.jsx",
    "Conclusion.jsx",
)
PUBLIC_HEADLINE_SUPPORT_FILES = (
    ROOT / "site" / "src" / "data.js",
    ROOT / "site" / "src" / "corpus.js",
    ROOT / "site" / "src" / "segdata.js",
    HISTORICAL_NOTICE,
)


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
    assert "Исторический LOBO headline отозван" in source


def test_site_lock_contains_every_declared_optional_platform_package():
    lock = json.loads((ROOT / "site" / "package-lock.json").read_text(encoding="utf-8"))
    packages = lock["packages"]
    optional = {
        dependency
        for package in ("node_modules/esbuild", "node_modules/rollup")
        for dependency in packages[package]["optionalDependencies"]
    }
    missing = sorted(
        dependency for dependency in optional if f"node_modules/{dependency}" not in packages
    )
    assert missing == []


def test_historical_headline_status_is_visible_and_fail_closed():
    registry = json.loads(
        (
            ROOT
            / "research"
            / "evidence"
            / "ineligible_corpus_registrations_v1.json"
        ).read_text(encoding="utf-8")
    )
    site_data = json.loads(
        (ROOT / "site" / "src" / "generated" / "site-data.json").read_text(
            encoding="utf-8"
        )
    )
    headline = site_data["headline"]
    assert registry["status"] == "ineligible_for_new_scientific_runs"
    assert headline["corpusEligibilityStatus"] == registry["status"]
    assert headline["claimStatus"] == "exploratory_internal"

    notice = HISTORICAL_NOTICE.read_text(encoding="utf-8")
    for marker in (
        "Исторический LOBO headline отозван",
        "не leakage-free оценка точности",
        "не действующее",
        "Нужны новая версия корпуса и полный пересчёт",
    ):
        assert marker in notice

    for section in PUBLIC_HEADLINE_SECTIONS:
        source = (ROOT / "site" / "src" / "sections" / section).read_text(
            encoding="utf-8"
        )
        assert "HistoricalHeadlineNotice" in source


def test_public_surfaces_do_not_restore_active_ineligible_headline_claims():
    site_source = "\n".join(
        [
            *(
                (ROOT / "site" / "src" / "sections" / section).read_text(
                    encoding="utf-8"
                )
                for section in PUBLIC_HEADLINE_SECTIONS
            ),
            *(path.read_text(encoding="utf-8") for path in PUBLIC_HEADLINE_SUPPORT_FILES),
        ]
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    generator = README_GENERATOR.read_text(encoding="utf-8")
    banned = (
        "Заголовочная цифра",
        "главная цифра",
        "Перевес не случаен",
        "разница не случайна",
        "уверенно обгоняет",
        "Случайным совпадением такой разрыв не объяснить",
        "полное публикуемое число",
        "Канонический headline",
        "полный leakage-free per-book LOBO",
        "**Главный честный вывод:**",
        "Публикуемый бенчмарк",
        "публикуемом срезе",
        "точность по авторам (главная метрика)",
        "ансамбль (равновесный, leak-free)",
    )
    for phrase in banned:
        assert phrase not in site_source
        assert phrase not in readme
        assert phrase not in generator


def test_readme_is_byte_identical_to_its_fail_closed_generator():
    completed = subprocess.run(
        ["node", str(README_GENERATOR), "--check"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "README.md совпадает с генератором" in completed.stdout


def test_method_does_not_render_the_withdrawn_macro_f1_interval():
    source = (ROOT / "site" / "src" / "sections" / "Method.jsx").read_text(encoding="utf-8")
    assert "MF1_CI" not in source
    assert "точность по авторам в диапазоне" not in source
    assert "интервал macro-F1 дополнительно отозван" in source
