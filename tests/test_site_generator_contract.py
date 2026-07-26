from __future__ import annotations

import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "gen-site-data.mjs"
README_GENERATOR = ROOT / "scripts" / "gen-readme.mjs"
RENDER_SMOKE = ROOT / "site" / "scripts" / "check-render.mjs"
NO_UNDEF_GATE = ROOT / "site" / "scripts" / "check-no-undef.mjs"
SITE_INDEX = ROOT / "site" / "index.html"
PNPM_LOCK = ROOT / "site" / "pnpm-lock.yaml"
HISTORICAL_NOTICE = (
    ROOT / "site" / "src" / "components" / "HistoricalHeadlineNotice.jsx"
)
PUBLIC_HEADLINE_SECTIONS = (
    "Hero.jsx",
    "Method.jsx",
    "Results.jsx",
    "Corpus.jsx",
    "Problem.jsx",
    "Repro.jsx",
    "Conclusion.jsx",
)
PUBLIC_HEADLINE_SUPPORT_FILES = (
    GENERATOR,
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
    assert package["scripts"]["check:undef"] == "node ./scripts/check-no-undef.mjs"
    assert "npm run check:undef" in package["scripts"]["build"]
    assert package["devDependencies"]["@babel/parser"] == "7.29.7"
    assert package["devDependencies"]["@babel/traverse"] == "7.29.7"
    pnpm_lock = PNPM_LOCK.read_text(encoding="utf-8")
    for dependency in ("@babel/parser", "@babel/traverse"):
        importer = (
            f"      '{dependency}':\n"
            "        specifier: 7.29.7\n"
            "        version: 7.29.7"
        )
        assert importer in pnpm_lock

    source = RENDER_SMOKE.read_text(encoding="utf-8")
    no_undef_source = NO_UNDEF_GATE.read_text(encoding="utf-8")
    app_source = (ROOT / "site" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert 'ssrLoadModule("/src/App.jsx")' in source
    assert "renderToStaticMarkup" in source
    assert "Исторический LOBO headline отозван" in source
    assert "NONDEFAULT_FREE_IDENTIFIER" in source
    assert "initialChapter: chapter" in source
    for chapter in ("framework", "sholokhov", "ilfpetrov", "nikolai", "hohol"):
        assert f"{chapter}: [" in source
    assert "export const CHAPTER_IDS" in app_source
    assert "CHAPTER_IDS.includes(initialChapter)" in app_source
    assert "ReferencedIdentifier" in no_undef_source
    assert "scope.hasBinding" in no_undef_source
    assert "NONDEFAULT_FREE_IDENTIFIER" in no_undef_source


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
    incomplete_registry_records = sorted(
        path
        for path, record in packages.items()
        if path and ("resolved" not in record or "integrity" not in record)
    )
    assert incomplete_registry_records == []


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
        "Протокол · без подсматривания",
        "Модель не видит проверяемую книгу",
        "Отложенная книга появляется ровно один раз",
        "Канонический headline-срез",
        "HEADLINE = продакшен",
        "таблица моделей (leak-free сравнение)",
        "PD-срез (публикуемый бенчмарк)",
        "честная верхняя граница для утверждения",
        "Единственная честная единица оценки — книга",
        "Решение — проверка по целым книгам",
        "Отложенная книга ничем не помогает угадать саму себя",
        "Каждая наша цифра отвечает на все три",
        "lobo.py (leakage-free)",
        "Главный тест держит ровно одно правило",
        "не видела проверяемую книгу",
    )
    for phrase in banned:
        assert phrase not in site_source
        assert phrase not in readme
        assert phrase not in generator


def test_static_site_metadata_withdraws_headline_and_uses_production_domain():
    index = SITE_INDEX.read_text(encoding="utf-8")
    assert "leakage-free LOBO" not in index
    assert "russkykod.com" not in index
    assert index.count("https://stylometry.russkiykod.com/") == 2
    assert index.count("Исторический LOBO headline отозван") == 3


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
    assert "Следующий протокол · content-safe" in source
    assert "Весь content-компонент отложенной книги" in source
