"""Валидация качества и консистентности корпуса (read-only отчёт).

Проверяет:
  - пустые/крошечные/битые файлы, долю не-кириллицы (mojibake/OCR);
  - достаточность для LOBO: книг на автора (>=2) и слов на книгу;
  - дисбаланс (max/min слов на автора);
  - точные дубликаты книг (sha1) и near-duplicate (word 4–5-gram coverage) — ловит один
    текст под двумя авторами и потенциальную утечку train/test;
  - издательский/OCR-шум (Глава N, номера страниц, ISBN, копирайт-футеры);
  - жанровые/служебные аномалии (дневники, соавторство — по конфигу).

Выдаёт человекочитаемый отчёт + JSON. Ничего не меняет.
"""
from __future__ import annotations
from .._io import is_link

import collections
import hashlib
import logging
import math
import numbers
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "error")


class CorpusValidationError(RuntimeError):
    """Fatal corpus findings were recorded; downstream stages must stop."""

    def __init__(self, report: CorpusReport):
        self.report = report
        super().__init__(
            "corpus validation failed with "
            f"{len(report.errors)} error(s): "
            + ", ".join(finding.code for finding in report.errors[:5])
        )


def _word_count(text: str) -> int:
    return len(text.split())


def _noise_flags(text: str) -> Dict[str, int]:
    return {name: len(rx.findall(text)) for name, rx in _NOISE_PATTERNS.items()}


def _word_shingles(text: str) -> collections.Counter:
    tokens = re.findall(r"(?u)\b\w+\b", text.lower())
    return collections.Counter(
        tuple(tokens[start:start + width])
        for width in (4, 5)
        for start in range(len(tokens) - width + 1)
    )


def validate_threshold(value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("near_dup_threshold must be a finite number in [0, 1]")
    return float(value)


def text_overlap_coverage(left: str, right: str) -> float:
    """Existing 4–5-word multiset coverage, insensitive to case/punctuation.

    Token equality also handles identical normalized texts shorter than four
    words. A zero-shingle pair with different tokens supplies no overlap proof.
    """
    left_tokens = re.findall(r"(?u)\b\w+\b", left.lower())
    right_tokens = re.findall(r"(?u)\b\w+\b", right.lower())
    if left_tokens and left_tokens == right_tokens:
        return 1.0
    a, b = _word_shingles(left), _word_shingles(right)
    denominator = min(sum(a.values()), sum(b.values()))
    return sum((a & b).values()) / denominator if denominator else 0.0


def assert_target_isolation(report: CorpusReport, target_work_id: str, reference_work_ids) -> None:
    """Make only target/reference duplicate pairs fatal, irrespective of author."""
    references = set(reference_work_ids)
    for left, right, coverage in report.duplicates:
        other = right if left == target_work_id else left if right == target_work_id else None
        if other is not None and other in references:
            report.add("error", "target_reference_overlap",
                       f"цель {target_work_id} пересекается с эталоном {other} (coverage={coverage:.2f})")
    if any(f.code == "target_reference_overlap" for f in report.findings):
        raise CorpusValidationError(report)


def validate(corpus_dir: pathlib.Path | str, near_dup_threshold: float = 0.4,
             min_books: int = 2, min_words_book: int = 500,
             min_words_tiny: int = 50, *, work_ids=None) -> CorpusReport:
    near_dup_threshold = validate_threshold(near_dup_threshold)
    corpus_dir = pathlib.Path(corpus_dir)
    rep = CorpusReport()
    if is_link(corpus_dir) or not corpus_dir.is_dir():
        rep.add("error", "invalid_root", f"небезопасный/отсутствующий каталог: {corpus_dir}")
        return rep

    book_texts: Dict[str, str] = {}   # "author/book" -> text
    book_hashes: Dict[str, str] = {}
    author_words: Dict[str, int] = collections.defaultdict(int)
    author_books: Dict[str, int] = collections.defaultdict(int)

    author_dirs = []
    for entry in sorted(os.scandir(corpus_dir), key=lambda item: item.name):
        if is_link(entry.path):
            rep.add("error", "symlink", f"символическая ссылка запрещена: {entry.path}")
        elif entry.is_dir(follow_symlinks=False):
            author_dirs.append(pathlib.Path(entry.path))
    for adir in author_dirs:
        author = adir.name
        books = []
        for entry in sorted(os.scandir(adir), key=lambda item: item.name):
            if is_link(entry.path):
                rep.add("error", "symlink", f"символическая ссылка запрещена: {entry.path}")
            elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".txt"):
                books.append(pathlib.Path(entry.path))
        for book in books:
            key = f"{author}/{book.stem}"
            if work_ids is not None and key not in work_ids:
                continue
            try:
                text = book.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                rep.add("error", "read_fail", f"{key}: не читается как strict UTF-8 ({exc})")
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
            rep.duplicates.extend((left, right, 1.0) for index, left in enumerate(keys) for right in keys[index + 1:])

    # Count overlap against ALL word shingles, including unique ones. A shared
    # epigraph must not become the complete comparison space. The denominator
    # is the smaller document's full shingle count, retaining sensitivity to a
    # work copied inside a longer work. Multiplicities prevent a few repeated
    # phrases from representing the full length of either document.
    keys = list(book_texts.keys())
    if len(keys) >= 2:
        shingles = [_word_shingles(book_texts[key]) for key in keys]
        counts = [sum(row.values()) for row in shingles]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                denominator = min(counts[i], counts[j])
                if denominator == 0:
                    s = text_overlap_coverage(book_texts[keys[i]], book_texts[keys[j]])
                    if s == 0:
                        continue
                    overlap = 1
                else:
                    overlap = sum((shingles[i] & shingles[j]).values())
                    s = overlap / denominator
                if overlap == 0:
                    continue
                if s >= near_dup_threshold:
                    a_i = keys[i].split("/")[0]
                    a_j = keys[j].split("/")[0]
                    sev = "error" if a_i != a_j else "warn"
                    rep.add(sev, "near_dup",
                            f"near-duplicate {keys[i]} ~ {keys[j]} (shingle coverage={s:.2f})"
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


def run(
    cfg=None,
    corpus_dir: str | None = None,
    *,
    report_only: bool = False,
    target_work_id: str | None = None,
    reference_authors=None,
) -> CorpusReport:
    from ..config import load_config
    from ..pipeline._snapshot import resolve_directory_snapshot
    cfg = cfg or load_config()
    cdir = corpus_dir or cfg.get_path("paths.input_clean", "input_clean")
    cdir = resolve_directory_snapshot(cdir)
    near = validate_threshold(cfg.get_path("corpus_policy.near_dup_threshold", 0.4))
    min_books = cfg.get_path("corpus_policy.min_books_per_author", 2)
    min_words = cfg.get_path("corpus_policy.min_words_per_book", 500)
    work_ids = references = None
    if target_work_id is not None:
        if reference_authors is None:
            raise ValueError("target-aware validation requires explicit reference authors")
        references = {
            f"{author.name}/{book.stem}" for author in pathlib.Path(cdir).iterdir()
            if author.is_dir() and author.name in reference_authors
            for book in author.iterdir() if book.is_file() and book.suffix == ".txt"
            and f"{author.name}/{book.stem}" != target_work_id
        }
        work_ids = references | {target_work_id}
    rep = validate(cdir, near_dup_threshold=near, min_books=0 if work_ids is not None else min_books,
                   min_words_book=min_words, work_ids=work_ids)
    if target_work_id is not None:
        reference_counts = collections.Counter(work.split("/", 1)[0] for work in references)
        for author in sorted(reference_authors):
            count = reference_counts[author]
            if count < min_books:
                rep.add("warn", "few_books", f"{author}: {count} эталонных книг (<{min_books}) — LOBO ненадёжен/невозможен")
        try:
            assert_target_isolation(rep, target_work_id, references)
        except CorpusValidationError:
            pass  # Publish the fatal finding before stopping the workflow.
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    if is_link(docs):
        raise RuntimeError(f"docs root must not be a symlink: {docs}")
    docs.mkdir(parents=True, exist_ok=True)
    txt = format_report(rep)
    structured = {
        "summary": rep.summary,
        "authors": rep.authors,
        "findings": [vars(f) for f in rep.findings],
        "duplicates": rep.duplicates,
        "scope": {"kind": "target_and_training_references" if work_ids is not None else "all_clean_works",
                  "target_work_id": target_work_id,
                  "reference_authors": sorted(reference_authors) if reference_authors is not None else None,
                  "selected_work_ids": sorted(work_ids) if work_ids is not None else None,
                  "reference_work_ids": sorted(references) if references is not None else None},
    }
    from ..report.evidence import publish_corpus_validation

    publish_corpus_validation(
        cfg,
        corpus_root=pathlib.Path(cdir),
        text=txt,
        structured=structured,
    )
    print(txt)
    if rep.errors and not report_only:
        raise CorpusValidationError(rep)
    return rep
