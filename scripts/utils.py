import hashlib
import os
import pathlib
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ХЕШИРОВАНИЕ / КЕШИРОВАНИЕ

def calculate_corpus_hash(root_dir: str = "input_clean") -> str:
    hasher = hashlib.md5()
    root_path = pathlib.Path(root_dir)
    if not root_path.exists():
        logging.error(f"Папка {root_dir} не найдена.")
        return ""
    # Сортировка важна для детерминированности
    file_list = sorted([str(p.relative_to(root_path)) for p in root_path.rglob("*.txt")])
    for filename in file_list:
        filepath = root_path / filename
        hasher.update(filename.encode('utf-8'))
        hasher.update(str(os.path.getsize(filepath)).encode('utf-8'))
        hasher.update(str(os.path.getmtime(filepath)).encode('utf-8'))
    return hasher.hexdigest()

def check_cache(new_hash: str, cache_file: str = "data/last_corpus_hash.txt") -> bool:
    if pathlib.Path(cache_file).exists():
        with open(cache_file, "r") as f:
            old_hash = f.read().strip()
        if old_hash == new_hash:
            logging.info("Хеш корпуса не изменился. Используется кеш.")
            return True
        else:
            logging.warning("Хеш корпуса изменился. Требуется переобучение.")
            return False
    return False

def save_cache(new_hash: str, cache_file: str = "data/last_corpus_hash.txt"):
    pathlib.Path("data").mkdir(exist_ok=True)
    with open(cache_file, "w") as f:
        f.write(new_hash)

# ТОКЕНИЗАЦИЯ / ЧАНКИРОВАНИЕ

def tokenize_preserve(text: str) -> List[str]:
    """Простое разбиение по пробелам (запасной путь). Используйте Spacy где возможно."""
    return text.split()

def make_chunks(tokens: List[str], size: int, min_size: int, overlap: float = 0.0) -> List[str]:
    """Глупая нарезка по словам (запасной путь)."""
    if not tokens: return []
    step = max(1, int(size * (1.0 - overlap)))
    chunks, i, L = [], 0, len(tokens)
    while i < L:
        chunk = tokens[i: i + size]
        if len(chunk) >= min_size:
            chunks.append(" ".join(chunk))
        i += step
        if L - i < min_size and i < L: break
    return chunks

def make_sent_chunks(doc, size: int, min_size: int, overlap: float = 0.0) -> List[str]:
    """
    Умная нарезка по предложениям: чанк не обрывается посередине предложения.

    Args:
        doc: Spacy Doc object
        size: Target size in tokens (words)
        min_size: Minimum size to accept a chunk
        overlap: Percentage of overlap (0.0 - 0.9)
    """
    sentences = list(doc.sents)
    if not sentences:
        return []

    chunks = []

    step_tokens = int(size * (1.0 - overlap))
    if step_tokens < 1: step_tokens = 1

    start_sent_idx = 0

    while start_sent_idx < len(sentences):
        current_chunk_sents = []
        current_len_tokens = 0

        curr_idx = start_sent_idx

        while curr_idx < len(sentences):
            sent = sentences[curr_idx]
            sent_len = len(sent)

            if current_len_tokens + sent_len > size:
                # Гигантское предложение (> size) при пустом чанке обязаны взять,
                # иначе бесконечный цикл / пропуск текста.
                if current_len_tokens == 0:
                    current_chunk_sents.append(sent.text)
                    current_len_tokens += sent_len
                    curr_idx += 1
                    break

                # Недобор лучше перебора: не разрываем предложения.
                break

            current_chunk_sents.append(sent.text)
            current_len_tokens += sent_len
            curr_idx += 1

        if current_len_tokens >= min_size or (len(current_chunk_sents) == 1 and current_len_tokens > 0):
            chunks.append(" ".join(current_chunk_sents))

        tokens_skipped = 0
        sents_advanced = 0

        scan_idx = start_sent_idx
        while scan_idx < len(sentences):
            tokens_skipped += len(sentences[scan_idx])
            sents_advanced += 1
            if tokens_skipped >= step_tokens:
                break
            scan_idx += 1

        # Защита от зацикливания: всегда сдвигаемся хотя бы на 1 предложение
        if sents_advanced == 0:
            sents_advanced = 1

        start_sent_idx += sents_advanced

        if start_sent_idx >= len(sentences):
            break

    return chunks

# ПОИСК ФАЙЛОВ

def get_all_txt_files(root_dir: str) -> List[str]:
    txt_files = []
    root = pathlib.Path(root_dir)
    if not root.exists(): return txt_files
    txt_files.extend([str(f) for f in root.rglob("*.txt")])
    return sorted(txt_files)
