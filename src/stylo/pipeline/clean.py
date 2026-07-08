"""Очистка корпуса: нормализация тире, маскировка имён (NER -> '@'), чистка мусора.

PER -> '@' единым маркером (не раздувает char-n-граммы; стабилен по кодировкам).
В bleaching NOUN намеренно разведён с '@' (см. char_ngrams).
"""
from __future__ import annotations

import logging
import pathlib
import re
from typing import List

from joblib import Parallel, delayed

from ..config import load_config
from ..nlp import load_ner

log = logging.getLogger("stylo.pipeline.clean")

NER_CHUNK_SIZE = 100_000
PER_MARK = "@"

_DASH_RE = re.compile(r"[‐-―−﹘﹣－]")
# Все виды двойных кавычек → прямая " : стиль кавычек — норма издания, не идиолект
# автора, и без унификации он протекает в char/punctuation-признаки (след источника).
# Одинарные ’‘‚‛ двусмысленны (апостроф/кавычка) и удаляются мусорным фильтром ниже.
_QUOTE_RE = re.compile(r"[«»„“”‟‹›]")
_GARBAGE_RE = re.compile(r"[^а-яА-Яa-zA-Z0-9\s\.,;:!?—\-\"@]")
_WS_RE = re.compile(r"\s+")
# остаточная вики-разметка/служебка (Викитека, lib.ru-импорт) — страховка к fetch_classics.
# Ловит и викитекстовый maintenance-футер: «Страницы с не вики-заголовками … Литература NNNN
# года Категория:Импорт lib.ru».
_WIKI_DIRT = re.compile(
    r"__[A-ZА-Я]+__"
    r"|Страницы\s+с[^\n]*"
    r"|Литература\s+\d{3,4}\s+года"
    r"|(?:Категория|Category|Импорт)\s*:?\s*\S[^\n]*"
    r"|(?:az\.)?lib\.ru\S*",
    re.U,
)


# дореформенная орфография → современная (ять, и десятеричное, фита, ижица, конечный ъ)
_PREREFORM = str.maketrans({"ѣ": "е", "Ѣ": "Е", "і": "и", "І": "И", "ї": "и", "Ї": "И",
                            "ѳ": "ф", "Ѳ": "Ф", "ѵ": "и", "Ѵ": "И"})
_FINAL_HARD = re.compile(r"ъ(?![а-яёА-ЯЁ])")  # конечный твёрдый знак (не объект/съезд)


def _depreform(text: str) -> str:
    return _FINAL_HARD.sub("", text.translate(_PREREFORM))


def normalize_dashes(text: str) -> str:
    text = _DASH_RE.sub("—", text)
    text = text.replace("--", "—")
    return re.sub(r"\s*—\s*", " — ", text)


def _mask_chunk(text: str, nlp) -> str:
    doc = nlp(text)
    out: List[str] = []
    last = 0
    for ent in doc.ents:
        if ent.label_ == "PER":
            out.append(text[last:ent.start_char])
            out.append(f" {PER_MARK} ")
            last = ent.end_char
    out.append(text[last:])
    return "".join(out)


def mask_names(text: str, nlp) -> str:
    if len(text) < NER_CHUNK_SIZE:
        return _mask_chunk(text, nlp)
    parts: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + NER_CHUNK_SIZE, n)
        if end < n:
            sp = text.rfind(" ", start, end)
            if sp != -1:
                end = sp
        piece = text[start:end]
        if piece.strip():
            parts.append(_mask_chunk(piece, nlp))
        start = end + 1
    return " ".join(parts)


def normalize(text: str, model: str, fallback: str | None) -> str:
    nlp = load_ner(model, fallback)
    text = _WIKI_DIRT.sub(" ", text)     # снять остаточную вики-разметку до NER
    text = normalize_dashes(text)
    text = mask_names(text, nlp)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = _depreform(text)              # дореформенная орфография → современная (сопоставимость)
    text = _QUOTE_RE.sub('"', text)      # унификация кавычек: «» „“ ” ‟ ‹› → "
    text = _GARBAGE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _process_file(fp: pathlib.Path, src: pathlib.Path, dst: pathlib.Path,
                  model: str, fallback: str | None) -> int:
    try:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            return 0
        clean = normalize(raw, model, fallback)
        if not clean:
            return 0
        out = dst / fp.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(clean, "utf-8")
        return 1
    except Exception as exc:  # pragma: no cover
        log.error("Ошибка очистки %s: %s", fp, exc)
        return 0


def run(cfg=None, only: list[str] | None = None) -> None:
    """Очистить input_raw -> input_clean. only=[author,...] — ограничить авторами."""
    cfg = cfg or load_config()
    src = pathlib.Path(cfg.get_path("paths.input_raw", "input"))
    dst = pathlib.Path(cfg.get_path("paths.input_clean", "input_clean"))
    model = cfg.get_path("language.spacy_model", "ru_core_news_lg")
    fallback = cfg.get_path("language.spacy_fallback", None)
    dst.mkdir(parents=True, exist_ok=True)

    files: List[pathlib.Path] = []
    for adir in sorted(src.iterdir()):
        if not adir.is_dir():
            continue
        if only and adir.name not in only:
            continue
        files.extend(sorted(adir.glob("*.txt")))

    log.info("Очистка %d файлов…", len(files))
    n_jobs = cfg.get_path("evaluation.n_jobs", -1)   # общая машина: не занимать все ядра
    res = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_process_file)(fp, src, dst, model, fallback) for fp in files
    )
    log.info("Очищено %d/%d", int(sum(res)), len(files))
