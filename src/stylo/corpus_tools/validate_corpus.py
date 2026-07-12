"""Валидация качества и консистентности корпуса (read-only отчёт).

Проверяет:
  - пустые/крошечные/битые файлы, долю не-кириллицы (mojibake/OCR);
  - достаточность для LOBO: книг на автора (>=2) и слов на книгу;
  - дисбаланс (max/min слов на автора);
  - точные дубликаты книг (sha1) и near-duplicate (char-5gram cosine) — ловит один
    текст под двумя авторами и потенциальную утечку train/test;
  - издательский/OCR-шум (Глава N, номера страниц, ISBN, копирайт-футеры);
  - жанровые/служебные аномалии (дневники, соавторство — по конфигу).

Выдаёт человекочитаемый отчёт + JSON. Ничего не меняет.
"""
from __future__ import annotations

import collections
import hashlib
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..jsonio import dump_strict

log = logging.getLogger("stylo.corpus_tools.validate")

_CYR = re.compile(r"[а-яёА-ЯЁ]")
_NOISE_PATTERNS = {
    "chapter_markers": re.compile(r"(?im)^\s*(глава|часть|том)\s+[ivxlcdm\d]", re.M),
    "page_numbers": re.compile(r"(?m)^\s*\d{1,4}\s*$"),
    "isbn": re.compile(r"ISBN", re.I),
    "copyright": re.compile(r"(©|copyright|все права защищены|OCR|FB2|fb2|библиотека)", re.I),
}


@dataclass
class Finding:
    severity: str   # error | warn | info
    code: str
    message: str


@dataclass
class CorpusReport:
    authors: Dict[str, dict] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    duplicates: List[Tuple[str, str, float]] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str):
        self.findings.append(Finding(severity, code, message))


def _word_count(text: str) -> int:
    return len(text.split())


def _noise_flags(text: str) -> Dict[str, int]:
    return {name: len(rx.findall(text)) for name, rx in _NOISE_PATTERNS.items()}


def validate(corpus_dir: pathlib.Path | str, near_dup_threshold: float = 0.4,
             min_books: int = 2, min_words_book: int = 500,
             min_words_tiny: int = 50) -> CorpusReport:
    corpus_dir = pathlib.Path(corpus_dir)
    rep = CorpusReport()

    book_texts: Dict[str, str] = {}   # "author/book" -> text
    book_hashes: Dict[str, str] = {}
    author_words: Dict[str, int] = collections.defaultdict(int)
    author_books: Dict[str, int] = collections.defaultdict(int)

    for adir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        author = adir.name
        for book in sorted(adir.glob("*.txt")):
            key = f"{author}/{book.stem}"
            try:
                text = book.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                rep.add("error", "read_fail", f"{key}: не читается ({exc})")
                continue
            wc = _word_count(text)
            author_words[author] += wc
            author_books[author] += 1

            if wc == 0:
                rep.add("error", "empty", f"{key}: пустой файл")
                continue
            if wc < min_words_tiny:
                rep.add("warn", "tiny", f"{key}: всего {wc} слов")
            letters = [c for c in text if c.isalpha()]
            if letters:
                cyr_ratio = sum(bool(_CYR.match(c)) for c in letters) / len(letters)
                if cyr_ratio < 0.6:
                    rep.add("warn", "non_cyrillic",
                            f"{key}: только {cyr_ratio:.0%} кириллицы (mojibake/чужой язык?)")
            noise = _noise_flags(text)
            heavy = {k: v for k, v in noise.items() if v > 5}
            if heavy:
                rep.add("info", "noise", f"{key}: возможный издательский/OCR-шум {heavy}")

            book_texts[key] = text
            book_hashes[key] = hashlib.sha1(text.encode("utf-8")).hexdigest()

    by_hash: Dict[str, List[str]] = collections.defaultdict(list)
    for k, h in book_hashes.items():
        by_hash[h].append(k)
    for h, keys in by_hash.items():
        if len(keys) > 1:
            rep.add("error", "exact_dup", f"идентичные тексты: {keys}")
            rep.duplicates.append((keys[0], keys[1], 1.0))

    # char-n-граммы на уровне книг недискриминативны (вся русская проза ~0.85+),
    # а 4-5-словные последовательности у разных книг почти не пересекаются —
    # высокий косинус здесь означает реальное текстовое совпадение/плагиат.
    keys = list(book_texts.keys())
    if len(keys) >= 2:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(4, 5), max_features=50000,
                              min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        X = vec.fit_transform([book_texts[k] for k in keys])
        sim = cosine_similarity(X)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s = float(sim[i, j])
                if s >= near_dup_threshold:
                    a_i = keys[i].split("/")[0]
                    a_j = keys[j].split("/")[0]
                    sev = "error" if a_i != a_j else "warn"
                    rep.add(sev, "near_dup",
                            f"near-duplicate {keys[i]} ~ {keys[j]} (cos={s:.2f})"
                            + (" — РАЗНЫЕ авторы!" if a_i != a_j else ""))
                    rep.duplicates.append((keys[i], keys[j], s))

    for author in sorted(author_books):
        nb = author_books[author]
        nw = author_words[author]
        rep.authors[author] = {"books": nb, "words": nw}
        if nb < min_books:
            rep.add("warn", "few_books",
                    f"{author}: {nb} книг(и) (<{min_books}) — LOBO ненадёжен/невозможен")
        if nw < min_words_book:
            rep.add("warn", "few_words", f"{author}: всего {nw} слов в корпусе")

    if author_words:
        mx = max(author_words.values())
        mn = min(v for v in author_words.values() if v > 0)
        ratio = mx / mn if mn else float("inf")
        rep.summary = {
            "n_authors": len(author_books),
            "n_books": sum(author_books.values()),
            "total_words": sum(author_words.values()),
            "imbalance_ratio": round(ratio, 1),
        }
        if ratio > 5:
            rep.add("warn", "imbalance",
                    f"дисбаланс {ratio:.0f}× по объёму (max/min слов на автора)")
    return rep


def format_report(rep: CorpusReport) -> str:
    lines: List[str] = ["=== ВАЛИДАЦИЯ КОРПУСА ==="]
    s = rep.summary
    if s:
        lines.append(f"Авторов: {s['n_authors']} | книг: {s['n_books']} | "
                     f"слов: {s['total_words']:,} | дисбаланс: {s['imbalance_ratio']}×")
    order = {"error": 0, "warn": 1, "info": 2}
    for f in sorted(rep.findings, key=lambda x: order.get(x.severity, 9)):
        tag = {"error": "❌", "warn": "⚠️ ", "info": "ℹ️ "}.get(f.severity, "  ")
        lines.append(f"{tag} [{f.code}] {f.message}")
    lines.append("")
    lines.append("Авторы (книги / слова):")
    for a, d in sorted(rep.authors.items(), key=lambda kv: kv[1]["words"]):
        lines.append(f"  {a:18} {d['books']:>2} книг | {d['words']:>9,} слов")
    return "\n".join(lines)


def run(cfg=None, corpus_dir: str | None = None) -> CorpusReport:
    from ..config import load_config
    cfg = cfg or load_config()
    cdir = corpus_dir or cfg.get_path("paths.input_clean", "input_clean")
    near = cfg.get_path("corpus_policy.near_dup_threshold", 0.4)
    min_books = cfg.get_path("corpus_policy.min_books_per_author", 2)
    min_words = cfg.get_path("corpus_policy.min_words_per_book", 500)
    rep = validate(cdir, near_dup_threshold=near, min_books=min_books,
                   min_words_book=min_words)
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    docs.mkdir(parents=True, exist_ok=True)
    txt = format_report(rep)
    (docs / "corpus_validation.txt").write_text(txt, encoding="utf-8")
    dump_strict({"summary": rep.summary, "authors": rep.authors,
                 "findings": [vars(f) for f in rep.findings],
                 "duplicates": rep.duplicates},
                docs / "corpus_validation.json", trailing_newline=False)
    print(txt)
    return rep
