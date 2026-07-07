# -*- coding: utf-8 -*-
"""Генератор docs/corpus_manifest.json — манифеста провенанса корпуса input/.

Проходит input/*/ (папка = автор, файл = книга) и для каждого файла пишет:
  book    — имя файла без .txt
  n_words — число слов (text.split())
  sha256  — первые 16 hex sha256 от байтов файла
  source  — источник текста, по приоритету:
            1) запись прежнего манифеста (author, book) — source сохраняется;
            2) configs/classics*.yaml: автор + slug(title) == имя файла
               -> "ru.wikisource.org: <страница Викитеки>";
            3) "local/неизвестно".
  genre   — только для 4 книг Крюкова (жанровая разметка кейса «Тихий Дон»).

Формат записей и верхние ключи: note, svd_basis_authors, authors, generated_by, summary.

Запуск: .venv/bin/python log/experiments/regen_manifest.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
INPUT = ROOT / "input"
MANIFEST = ROOT / "docs" / "corpus_manifest.json"

LOCAL = "local/неизвестно"

# Жанровая разметка книг Крюкова (кейс «Тихий Дон»)
KRUKOV_GENRE = {
    "na_tihom_donu": "очерки",
    "v_glubine": "очерки",
    "v_rodnih_mestah": "очерки",
    "oficersha": "повесть",
}


def _slug(title: str) -> str:
    """Тот же слаг, что у fetch_classics: имя файла из заголовка Викитеки."""
    s = re.sub(r"[^\w]+", "_", title.lower(), flags=re.U).strip("_")
    return s[:60] or "work"


def load_old_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def load_yaml_sources() -> dict[tuple[str, str], str]:
    """(author, slug(title)) -> строка источника из configs/classics*.yaml."""
    out: dict[tuple[str, str], str] = {}
    for path in sorted(glob.glob(str(ROOT / "configs" / "classics*.yaml"))):
        entries = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or []
        for e in entries:
            author = e.get("author")
            title = e.get("title")
            if not author or not title:
                continue
            src = e.get("source", title)
            out.setdefault((author, _slug(title)), f"ru.wikisource.org: {src}")
    return out


def main() -> None:
    old = load_old_manifest()
    old_sources: dict[tuple[str, str], str] = {}
    for author, rec in (old.get("authors") or {}).items():
        for b in rec.get("books", []):
            old_sources[(author, b["book"])] = b.get("source", LOCAL)

    yaml_sources = load_yaml_sources()
    svd_basis = old.get("svd_basis_authors", [])

    authors: dict[str, dict] = {}
    n_files = 0
    n_with_source = 0
    for adir in sorted(p for p in INPUT.iterdir() if p.is_dir()):
        author = adir.name
        books = []
        for f in sorted(adir.glob("*.txt")):
            book = f.stem
            data = f.read_bytes()
            source = old_sources.get((author, book)) or yaml_sources.get(
                (author, book), LOCAL
            )
            entry = {
                "book": book,
                "n_words": len(data.decode("utf-8", errors="ignore").split()),
                "sha256": hashlib.sha256(data).hexdigest()[:16],
                "source": source,
            }
            if author == "krukov" and book in KRUKOV_GENRE:
                entry["genre"] = KRUKOV_GENRE[book]
            books.append(entry)
            n_files += 1
            if source != LOCAL:
                n_with_source += 1
        # Пустая папка автора тоже фиксируется (как в прежнем манифесте).
        authors[author] = {
            "n_books": len(books),
            "in_svd_basis": author in svd_basis,
            "books": books,
        }

    manifest = {
        "note": old.get(
            "note",
            "Провенанс корпуса. Тексты под копирайтом — локально, НЕ в git.",
        ),
        "generated_by": "log/experiments/regen_manifest.py",
        "summary": {
            "n_files": n_files,
            "n_authors": sum(1 for a in authors.values() if a["n_books"]),
            "n_with_source": n_with_source,
            "coverage": f"источник установлен у {n_with_source} файлов из {n_files}",
        },
        "svd_basis_authors": svd_basis,
        "authors": authors,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"docs/corpus_manifest.json: {n_files} файлов, "
          f"{manifest['summary']['n_authors']} авторов с книгами, "
          f"источник установлен у {n_with_source}")


if __name__ == "__main__":
    main()
