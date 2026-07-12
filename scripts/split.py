"""
Разбивает книги на чанки, СОХРАНЯЯ ГРАНИЦЫ ПРЕДЛОЖЕНИЙ.
Использует Spacy для сегментации.
"""
from __future__ import annotations
import sys
import json
import argparse
import logging
import pathlib
import os

from utils import make_sent_chunks
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class CombinedDoc:
    """
    Класс-обертка, имитирующий Spacy Doc для функции make_sent_chunks.
    Позволяет объединить предложения из нескольких обработанных кусков текста.
    """
    def __init__(self, sentences):
        self.sents = sentences
    
    def __len__(self):
        # make_sent_chunks может проверять длину, но использует в основном .sents
        return sum(len(s) for s in self.sents)

def parse_args():
    p = argparse.ArgumentParser(description="Split corpus into sentence-aware chunks")
    p.add_argument("--input", default="input_clean")
    p.add_argument("--chunk", type=int, default=500, help="Target token count")
    p.add_argument("--min-words", type=int, default=200)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--leave-out", nargs="*", help="Books to move to unknown")
    p.add_argument("--lang", default="ru", help="Language code (ru, en, fr)")
    return p.parse_args()

def split_text_safe(text: str, limit: int = 1_000_000) -> list[str]:
    """
    Разбивает огромный текст на куски безопасного размера для Spacy,
    стараясь не резать слова (по пробелам).
    """
    parts = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + limit, text_len)
        
        if end < text_len:
            # Ищем последний пробел, чтобы не разрезать слово
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space
        
        chunk = text[start:end]
        if chunk.strip():
            parts.append(chunk)
        
        start = end + 1 # Пропускаем пробел
        
    return parts

def process_large_text(text: str, nlp) -> list:
    """
    Обрабатывает текст любого размера, возвращает плоский список предложений (Spacy Spans).
    """
    if len(text) < 1_000_000:
        doc = nlp(text)
        return list(doc.sents)

    parts = split_text_safe(text)
    all_sents = []

    # Используем nlp.pipe для эффективности
    for doc in nlp.pipe(parts):
        all_sents.extend(list(doc.sents))
        
    return all_sents

def main():
    args = parse_args()
    
    # Устанавливаем язык ДО импорта NLP модулей
    os.environ["STYLO_LANG"] = args.lang
    
    # Отложенный импорт, чтобы применился ENV VAR
    from nlp import get_stylometry_nlp
    
    src_root = pathlib.Path(args.input)
    if not src_root.exists():
        logging.error(f"Папка {src_root} не найдена.")
        sys.exit(1)

    logging.info(f"Загрузка NLP модели для сегментации ({args.lang})...")
    nlp = get_stylometry_nlp()
    
    chunk_sz = args.chunk
    min_tokens = args.min_words
    overlap = args.overlap
    leave_out = set(args.leave_out or [])

    (pathlib.Path("data/frags_train")).mkdir(parents=True, exist_ok=True)
    (pathlib.Path("data/frags_unknown")).mkdir(parents=True, exist_ok=True)

    mapping = []
    
    # Если в модели нет tagger/parser, sentencizer должен быть добавлен явно (это делается в get_stylometry_nlp)
    pipes_to_disable = [p for p in nlp.pipe_names if p not in ["sentencizer", "senter"]]
    
    with nlp.select_pipes(disable=pipes_to_disable):
        for author_dir in sorted(src_root.iterdir()):
            if not author_dir.is_dir(): continue
            author = author_dir.name

            for book_file in sorted(author_dir.glob("*.txt")):
                book_id = book_file.stem
                try:
                    text_raw = book_file.read_text("utf-8").strip()
                except UnicodeDecodeError:
                    logging.warning(f"Ошибка кодировки: {book_file}")
                    continue
                    
                if not text_raw: 
                    continue

                try:
                    all_sentences = process_large_text(text_raw, nlp)
                except Exception as e:
                    logging.error(f"Ошибка обработки {book_id}: {e}")
                    continue

                if not all_sentences:
                    continue

                target = "unknown" if (book_id in leave_out or author == "unknown") else "train"
                out_root = pathlib.Path(f"data/frags_{target}/{author}/{book_id}")
                out_root.mkdir(parents=True, exist_ok=True)

                # make_sent_chunks ожидает объект, у которого есть свойство .sents
                dummy_doc = CombinedDoc(all_sentences)
                chunks = make_sent_chunks(dummy_doc, chunk_sz, min_tokens, overlap)

                if not chunks:
                    logging.warning(f"Книга {book_id} слишком коротка после нарезки.")
                    continue

                for idx, ch in enumerate(chunks):
                    out_path = out_root / f"{book_id}_{idx:05d}.txt"
                    out_path.write_text(ch, "utf-8")
                    mapping.append({
                        "path": str(out_path),
                        "author": author,
                        "book": book_id
                    })

                logging.info(f"{author}/{book_id}: {len(chunks)} чанков -> {target}")

    pathlib.Path("data").mkdir(exist_ok=True)
    with open("data/chunk_map.json", "w", encoding="utf-8") as f:
        f.write(dumps_strict(mapping, indent=2))

    logging.info("Нарезка по предложениям завершена.")

if __name__ == "__main__":
    main()
