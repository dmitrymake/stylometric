"""Freeze RuAA-Bench v1.0 — публичный воспроизводимый book-level бенчмарк
атрибуции авторства для русского языка.

Состав: авторы PUBLIC_DOMAIN_CLEAR (юр-чистый PD-срез: без реабилитационных
продлений ст. 1281 п. 5 ГК), книги с установленным источником в
docs/corpus_manifest.json, у автора остаётся >= MIN_BOOKS книг. Тексты
редистрибутируемы (Викитека/PD), поэтому пакет содержит сами тексты.

Пакет data/ruaa_bench_v1/:
  texts/<author>/<book>.txt
  manifest.json     — полный SHA256, слова, источник, год смерти автора
  protocol.md       — замороженный leak-free протокол оценки
  LICENSE           — тексты PD, метаданные CC0
  SHA256SUMS        — контрольные суммы пакета

Run: .venv/bin/python scripts/build_ruaa_bench.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylo.claims import BenchmarkRole, ClaimStatus  # noqa: E402
from stylo.corpus_tools.fetch_classics import PUBLIC_DOMAIN_CLEAR  # noqa: E402
from stylo.jsonio import dump_strict  # noqa: E402

INPUT = ROOT / "input"
OUT = ROOT / "data" / "ruaa_bench_v1"
MANIFEST_IN = ROOT / "docs" / "corpus_manifest.json"
BENCH_DOC = ROOT / "docs" / "ruaa_bench_manifest.json"
MIN_BOOKS = 2
WORD_RE = re.compile(r"[\w\-]+", re.U)

# Соавторский дуэт исключён из одноклассовой атрибуции (та же политика, что в
# headline-бенчмарке); nikolas2/sholohov не PD и в срез не входят.
EXCLUDE_AUTHORS = {"ilf-petrov"}

DEATH_YEARS = {
    "dostoevsky": 1881, "gogol": 1852, "chehov": 1904, "rozanov": 1919,
    "tolstoy": 1910, "turgenev": 1883, "kuprin": 1938, "andreev": 1919,
    "leskov": 1895, "saltykov": 1889, "garshin": 1888, "korolenko": 1921,
    "bunin": 1953, "pushkin": 1837, "lermontov": 1841, "goncharov": 1891,
    "pisemsky": 1881, "grigorovich": 1900, "prutkov": 1908,
    "serafimovich": 1949, "sevsky": 1920, "kumov": 1919, "rodionov": 1940,
    "uspensky": 1902, "reshetnikov": 1871, "zlatovratsky": 1911,
    "zasodimsky": 1912, "naumov": 1901, "veresaev": 1945,
    "novikov_priboy": 1944, "furmanov": 1926, "dmitrieva": 1928,
    "voloshin": 1932, "tolstoy_an": 1945, "bryusov": 1924, "grin": 1932,
}

PROTOCOL_MD = """# RuAA-Bench v1.0 — протокол (заморожен)

> **Статус: `reproducible_cv_legacy_not_blind` (claim_status: `exploratory_internal`).**
> Это датированный воспроизводимый CV-срез на публичном корпусе, а не blind-
> лидерборд и не научный default. `book_id = <author>/<book>` раскрывает истину,
> поэтому пакет непригоден как слепой benchmark. Взвешивание обучения —
> `chunk_weighted_training_legacy` (см. `docs/cases/work_balanced_audit/` и §5 prereg).
> Слепой пакет и work-balanced пересчёт — в v2.

Задача: закрытая атрибуция авторства на уровне книги.

- Единица оценки: книга целиком. Оценка = leave-one-book-out (LOBO):
  каждая книга ровно один раз является тестом, всё обучаемое (словари, idf,
  статистики, классификатор, калибровка) строится только на остальных книгах.
  Сплит задан самим протоколом; рандомизации и подбора по тесту нет.
- Запрещено: любое обучение/настройка на тестовой книге; использование
  внешних текстов тех же произведений; выбор конфигурации по итоговой метрике.
- Предобработка на усмотрение участника, но одинаковая для train и test
  внутри фолда; референс-пайплайн: чанк 500 слов, маскировка имён (PERSON->@).
- Метрики: accuracy (top-1), macro-F1 по авторам, top-2, per-author recall;
  интервалы: bootstrap по книгам и author-clustered bootstrap (оба 95%).
- Сабмит: CSV `book_id,pred_author` (book_id = `<author>/<book>`), опционально
  вероятности по классам. Скоринг: scripts/score_ruaa.py.

Смежные ресурсы: томский корпус атрибуции не публичен; proza_ru_hard —
сырой датасет коротких документов без замороженного протокола и baseline-
таблицы; RusProfiling — профилирование автора; RuATD — детекция
машинного текста. RuAA-Bench v1 отличает связка: замороженный leak-free
book-level протокол + манифест с полным SHA256 + baseline-таблица со
статистикой значимости (не blind).
"""

LICENSE_TXT = """Texts: public domain (Russian classics; authors died 70+ years
ago; rehabilitation-extended authors excluded per Art. 1281(5) of the Russian
Civil Code). Sources are recorded per book in manifest.json.
Metadata, manifest and protocol: CC0-1.0.
Code that builds and scores the benchmark: MIT (repository license).
"""


def main() -> int:
    manifest_in = json.loads(MANIFEST_IN.read_text(encoding="utf-8"))
    authors_in = manifest_in["authors"]

    selected = {}
    dropped = {"no_source": [], "too_few_books": [], "excluded": []}
    for author in sorted(PUBLIC_DOMAIN_CLEAR - EXCLUDE_AUTHORS):
        entry = authors_in.get(author)
        if not entry:
            continue
        books = []
        for b in entry["books"]:
            src = b.get("source") or ""
            if not src or "неизвестно" in src:
                dropped["no_source"].append(f"{author}/{b['book']}")
                continue
            books.append((b["book"], src))
        if len(books) < MIN_BOOKS:
            dropped["too_few_books"].extend(f"{author}/{b}" for b, _ in books)
            continue
        selected[author] = books
    for author in sorted(EXCLUDE_AUTHORS & set(authors_in)):
        dropped["excluded"].append(author)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "texts").mkdir(parents=True)

    bench = {
        "name": "RuAA-Bench",
        "version": "1.0",
        "claim_status": ClaimStatus.EXPLORATORY_INTERNAL.value,
        "benchmark_role": BenchmarkRole.REPRODUCIBLE_CV_LEGACY_NOT_BLIND.value,
        "training_weighting": "chunk_weighted_training_legacy",
        "task": "closed-set book-level authorship attribution, Russian",
        "n_authors": len(selected),
        "n_books": sum(len(b) for b in selected.values()),
        "legal": "PUBLIC_DOMAIN_CLEAR: смерть+70, реабилитационные продления исключены (gumilev, pilnyak)",
        "authors": {},
        "dropped": {k: sorted(v) for k, v in dropped.items()},
    }
    for author, books in selected.items():
        adir = OUT / "texts" / author
        adir.mkdir()
        entries = []
        for book, src in books:
            src_path = INPUT / author / f"{book}.txt"
            dst = adir / f"{book}.txt"
            shutil.copy2(src_path, dst)
            raw = dst.read_bytes()
            entries.append({
                "book": book,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "words": len(WORD_RE.findall(raw.decode("utf-8", "ignore"))),
                "source": src,
            })
        bench["authors"][author] = {
            "death_year": DEATH_YEARS.get(author),
            "n_books": len(entries),
            "books": entries,
        }

    dump_strict(bench, OUT / "manifest.json")
    (OUT / "protocol.md").write_text(PROTOCOL_MD, encoding="utf-8")
    (OUT / "LICENSE").write_text(LICENSE_TXT, encoding="utf-8")

    sums = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    dump_strict(bench, BENCH_DOC)
    print(f"RuAA-Bench v1.0: {bench['n_authors']} авторов, {bench['n_books']} книг")
    for k, v in bench["dropped"].items():
        print(f"  dropped {k}: {len(v)}")
    print(f"пакет: {OUT.relative_to(ROOT)} | манифест-копия: {BENCH_DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
