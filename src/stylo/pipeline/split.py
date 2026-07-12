"""Нарезка очищенного корпуса на чанки по предложениям.

Раскладка результата: data/frags_train/{author}/{book}/{book}_{idx}.txt
Книги из --leave-out и каталог unknown отправляются в data/frags_unknown.
"""
from __future__ import annotations

import logging
import pathlib
import shutil
from typing import List, Sequence

from ..chunking import CombinedDoc, make_sent_chunks, sentences_for_text
from ..config import load_config
from ..jsonio import dump_strict
from ..nlp import load_sentencizer

log = logging.getLogger("stylo.pipeline.split")


def run(cfg=None, leave_out: Sequence[str] = (), clean_existing: bool = True) -> int:
    cfg = cfg or load_config()
    src = pathlib.Path(cfg.get_path("paths.input_clean", "input_clean"))
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    size = cfg.get_path("chunking.chunk_size", 500)
    min_words = cfg.get_path("chunking.min_words", 200)
    overlap = cfg.get_path("chunking.overlap", 0.0)
    unknown_name = cfg.get_path("corpus_policy.unknown_dir_name", "unknown")
    lang = cfg.get_path("language.code", "ru")

    if not src.exists():
        raise FileNotFoundError(f"Нет {src}; сначала clean.")

    train_root = data / "frags_train"
    unk_root = data / "frags_unknown"
    if clean_existing:
        for d in (train_root, unk_root):
            if d.exists():
                shutil.rmtree(d)
    train_root.mkdir(parents=True, exist_ok=True)
    unk_root.mkdir(parents=True, exist_ok=True)

    nlp = load_sentencizer(lang)
    leave = set(leave_out)
    mapping: List[dict] = []
    n_chunks_total = 0

    for adir in sorted(src.iterdir()):
        if not adir.is_dir():
            continue
        author = adir.name
        for book in sorted(adir.glob("*.txt")):
            book_id = book.stem
            try:
                raw = book.read_text("utf-8").strip()
            except Exception:
                continue
            if not raw:
                continue
            sents = sentences_for_text(raw, nlp)
            if not sents:
                continue
            chunks = make_sent_chunks(CombinedDoc(sents), size, min_words, overlap)
            if not chunks:
                log.warning("%s/%s: слишком коротка", author, book_id)
                continue

            to_unknown = author == unknown_name or book_id in leave
            base = unk_root if to_unknown else train_root
            out_dir = base / author / book_id
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, ch in enumerate(chunks):
                p = out_dir / f"{book_id}_{idx:05d}.txt"
                p.write_text(ch, "utf-8")
                mapping.append({"path": str(p), "author": author, "book": book_id,
                                "split": "unknown" if to_unknown else "train"})
            n_chunks_total += len(chunks)
            log.info("%s/%s: %d чанков -> %s", author, book_id, len(chunks),
                     "unknown" if to_unknown else "train")

    dump_strict(mapping, data / "chunk_map.json", trailing_newline=False)
    log.info("Нарезка завершена: %d чанков", n_chunks_total)
    return n_chunks_total
