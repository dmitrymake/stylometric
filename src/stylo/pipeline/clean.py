"""Очистка корпуса: нормализация тире, маскировка имён (NER -> '@'), чистка мусора.

PER -> '@' единым маркером (не раздувает char-n-граммы; стабилен по кодировкам).
В bleaching NOUN намеренно разведён с '@' (см. char_ngrams).
"""
from __future__ import annotations

import logging
import hashlib
import pathlib
import re
import shutil
import tempfile
from typing import List

from joblib import Parallel, delayed

from ..chunking import split_text_safe
from ..config import load_config
from .._io import is_link
from ..jsonio import dump_strict, dumps_strict, loads_strict
from ..nlp import load_ner, resolved_nlp_identity
from ._snapshot import publish_directory_snapshot

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
    return "".join(_mask_chunk(piece, nlp) for piece in split_text_safe(text, NER_CHUNK_SIZE))


def mask_names_with_config(text: str, cfg=None) -> str:
    """Mask PERSON entities with the canonical configured NER resolver.

    This side-effect-free API is for explicit one-off consumers. Corpus
    publication still goes through :func:`run`.
    """

    cfg = cfg or load_config()
    return mask_names(
        text,
        load_ner(
            cfg.get_path("language.spacy_model", "ru_core_news_lg"),
            cfg.get_path("language.spacy_fallback", None),
        ),
    )


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


CLEAN_MANIFEST = "clean_manifest.json"
CLEAN_SCHEMA = "stylo.cleaned-corpus.v2"
CLEANER_CONTRACT = "stylo.clean/v2"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ner_identity(model: str, fallback: str | None) -> dict:
    # Round-trip tuple fields to their strict-JSON representation so worker,
    # in-memory and persisted identities compare without type coercion.
    return loads_strict(dumps_strict(resolved_nlp_identity(load_ner(model, fallback)).to_dict()))


def preprocessing_identity(cfg) -> dict:
    """Resolved cleaner/NER provenance, independent of the chunking settings."""
    settings = {
        "model": cfg.get_path("language.spacy_model", "ru_core_news_lg"),
        "model_version": cfg.get_path("language.spacy_model_version", None),
        "fallback": cfg.get_path("language.spacy_fallback", None),
    }
    return {
        "contract": CLEANER_CONTRACT,
        "cleaner_source_sha256": _sha256_bytes(pathlib.Path(__file__).read_bytes()),
        "text_splitter_source_sha256": _sha256_bytes(
            pathlib.Path(__file__).parents[1].joinpath("chunking.py").read_bytes()
        ),
        "ner_chunk_size": NER_CHUNK_SIZE,
        "person_marker": PER_MARK,
        "config": settings,
        "ner": _ner_identity(settings["model"], settings["fallback"]),
    }


def _read_raw_strict(fp: pathlib.Path) -> tuple[bytes, str]:
    if is_link(fp) or not fp.is_file():
        raise RuntimeError(f"raw corpus input must be a regular non-symlink file: {fp}")
    payload = fp.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"raw corpus input is not valid UTF-8: {fp}: {exc}") from exc
    if not text.strip():
        raise RuntimeError(f"raw corpus input is empty: {fp}")
    return payload, text


def _process_file(
    fp: pathlib.Path,
    src: pathlib.Path,
    dst: pathlib.Path,
    model: str,
    fallback: str | None,
    preprocessing: dict,
) -> dict[str, str]:
    payload, raw = _read_raw_strict(fp)
    if _ner_identity(model, fallback) != preprocessing["ner"]:
        raise RuntimeError(f"resolved NER identity changed during cleaning: {fp}")
    clean = normalize(raw, model, fallback)
    if not clean:
        raise RuntimeError(f"normalisation produced empty text: {fp}")
    relative = fp.relative_to(src)
    out = dst / relative
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(clean, encoding="utf-8", newline="\n")
    return {
        "source": relative.as_posix(),
        "source_sha256": _sha256_bytes(payload),
        "output_sha256": _sha256_bytes(clean.encode("utf-8")),
    }


def _raw_files(src: pathlib.Path) -> list[pathlib.Path]:
    if is_link(src) or not src.is_dir():
        raise RuntimeError(f"raw corpus root must be a real directory: {src}")
    files: list[pathlib.Path] = []
    for entry in sorted(src.iterdir()):
        if is_link(entry):
            raise RuntimeError(f"symlinked raw corpus entry rejected: {entry}")
        if not entry.is_dir():
            raise RuntimeError(
                f"raw corpus root may contain only author directories: {entry}"
            )
        author_files = 0
        for candidate in sorted(entry.iterdir()):
            if is_link(candidate):
                raise RuntimeError(f"symlinked raw corpus input rejected: {candidate}")
            if candidate.is_dir():
                raise RuntimeError(
                    "nested raw corpus directories are unsupported; expected "
                    f"author/*.txt: {candidate}"
                )
            if not candidate.is_file() or candidate.suffix != ".txt":
                raise RuntimeError(
                    f"unexpected raw corpus payload outside author/*.txt: {candidate}"
                )
            files.append(candidate)
            author_files += 1
        if author_files == 0:
            raise RuntimeError(f"raw author {entry.name!r} has no .txt works")
    if not files:
        raise RuntimeError(f"raw corpus contains no author/*.txt inputs: {src}")
    return files


def _validate_staged_snapshot(
    src: pathlib.Path,
    staging: pathlib.Path,
    entries: list[dict[str, str]],
) -> None:
    source_paths = {fp.relative_to(src).as_posix() for fp in _raw_files(src)}
    recorded_paths = {entry["source"] for entry in entries}
    output_paths = {
        fp.relative_to(staging).as_posix()
        for fp in staging.rglob("*.txt")
        if fp.is_file() and not is_link(fp)
    }
    if source_paths != recorded_paths or output_paths != source_paths:
        raise RuntimeError(
            "cleaned snapshot is not an exact raw/output bijection: "
            f"unrecorded_raw={sorted(source_paths-recorded_paths)[:3]}, "
            f"missing_output={sorted(source_paths-output_paths)[:3]}, "
            f"extra_output={sorted(output_paths-source_paths)[:3]}"
        )
    by_source = {entry["source"]: entry for entry in entries}
    for relative in sorted(source_paths):
        source_payload, _text = _read_raw_strict(src / relative)
        output = staging / relative
        if is_link(output) or not output.is_file():
            raise RuntimeError(f"cleaned output is missing/unsafe: {relative}")
        if _sha256_bytes(source_payload) != by_source[relative]["source_sha256"]:
            raise RuntimeError(f"raw input changed during cleaning: {relative}")
        if _sha256_bytes(output.read_bytes()) != by_source[relative]["output_sha256"]:
            raise RuntimeError(f"cleaned output changed during staging: {relative}")


def run(cfg=None, only: list[str] | None = None) -> None:
    """Publish an exact raw/clean snapshot as an immutable current generation.

    Partial rebuilding is intentionally unsupported: carrying outputs from a
    prior run could mix cleaner code/model/runtime generations even when the
    raw bytes are unchanged.  No per-file failure can leave stale output
    current.
    """
    if only is not None:
        raise ValueError(
            "partial clean is disabled; rebuild the complete raw corpus under "
            "one cleaner/model/runtime generation"
        )
    cfg = cfg or load_config()
    src = pathlib.Path(cfg.get_path("paths.input_raw", "input"))
    dst = pathlib.Path(cfg.get_path("paths.input_clean", "input_clean"))
    model = cfg.get_path("language.spacy_model", "ru_core_news_lg")
    fallback = cfg.get_path("language.spacy_fallback", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    files = _raw_files(src)
    preprocessing = preprocessing_identity(cfg)

    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{dst.name}.staging-", dir=dst.parent)
    )
    entries: list[dict[str, str]] = []
    try:
        log.info("Очистка %d файлов в staging snapshot…", len(files))
        n_jobs = cfg.get_path("evaluation.n_jobs", -1)
        built = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(_process_file)(fp, src, staging, model, fallback, preprocessing)
            for fp in files
        )
        entries.extend(built)
        entries.sort(key=lambda row: row["source"])
        _validate_staged_snapshot(src, staging, entries)
        dump_strict(
            {
                "schema_version": CLEAN_SCHEMA,
                "source_root": str(src.resolve()),
                "preprocessing": preprocessing,
                "files": entries,
            },
            staging / CLEAN_MANIFEST,
            sort_keys=True,
        )
        # Re-validate raw bytes immediately before the atomic exchange.
        _validate_staged_snapshot(src, staging, entries)
        if preprocessing_identity(cfg) != preprocessing:
            raise RuntimeError("cleaner/NER identity changed during snapshot construction")
        published_root = publish_directory_snapshot(staging, dst)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    log.info("Очищено и опубликовано %d/%d: %s", len(entries), len(files), published_root)
