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
from ..workdoc import (MANIFEST_NAME, build_work_manifest, chunker_config_hash,
                       frozen_chunker_config, sha256_text)

log = logging.getLogger("stylo.pipeline.split")


def run(cfg=None, leave_out: Sequence[str] = (), clean_existing: bool = True) -> int:
    cfg = cfg or load_config()
    src = pathlib.Path(cfg.get_path("paths.input_clean", "input_clean"))
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    # single frozen chunker config, shared with the manifest hash (no int/float drift)
    chunker = frozen_chunker_config(cfg)
    size = chunker.chunk_size
    min_words = chunker.min_words
    overlap = chunker.overlap
    unknown_name = cfg.get_path("corpus_policy.unknown_dir_name", "unknown")
    lang = chunker.language

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
    cfg_chunker_hash = chunker_config_hash(cfg)

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
            filenames = [f"{book_id}_{idx:05d}.txt" for idx in range(len(chunks))]
            for name, ch in zip(filenames, chunks):
                (out_dir / name).write_text(ch, "utf-8")
                mapping.append({"path": str(out_dir / name), "author": author, "book": book_id,
                                "split": "unknown" if to_unknown else "train"})
            # canonical chunk manifest (P1 B0): stable per-chunk identity for work_balanced.
            # provenance = sha of the exact cleaned source bytes the chunker consumed (raw).
            manifest = build_work_manifest(
                f"{author}/{book_id}", author, chunks, filenames,
                provenance_sha256=sha256_text(raw),
                chunker_config_hash=cfg_chunker_hash,
                overlap=float(overlap),
            )
            dump_strict(manifest.to_dict(), out_dir / MANIFEST_NAME, trailing_newline=False)
            n_chunks_total += len(chunks)
            log.info("%s/%s: %d чанков -> %s", author, book_id, len(chunks),
                     "unknown" if to_unknown else "train")

    dump_strict(mapping, data / "chunk_map.json", trailing_newline=False)
    log.info("Нарезка завершена: %d чанков", n_chunks_total)
    return n_chunks_total
