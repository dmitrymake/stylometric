"""
Очистка исходных книг:
- Нормализация тире и кавычек
- Маскировка имен (NER) для защиты от тематического переобучения
- Очистка мусора

ВАЖНОЕ РЕШЕНИЕ ПО PER:
Мы НЕ храним "<PER>" в тексте, а заменяем PERSON на один символ "@",
чтобы:
1) не раздувать char-ngrams и не создавать «шумных» подпоследовательностей,
2) иметь стабильный единый маркер во всех окружениях/кодировках.

Важно, чтобы другие механизмы маскировки НЕ использовали "@" для других
целей (в syntax_features.py NOUN-marker принудительно разведён с "@").
"""

from __future__ import annotations

import logging
import pathlib
import re
import sys
from joblib import Parallel, delayed

# Импорт NLP (будет загружен внутри процессов при первом вызове get_ner_nlp())
from scripts.nlp import get_ner_nlp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Лимит символов для одного прогона NER (защита от SpaCy Error / Memory Overflow)
NER_CHUNK_SIZE = 100_000

# Единый маркер PERSON в очищенном корпусе
PER_MARK = "@"


def normalize_dashes(text: str) -> str:
    """Приводит все виды тире к одному формату (—) и нормализует пробелы вокруг него."""
    text = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "—", text)
    text = text.replace("--", "—")
    text = re.sub(r"\s*—\s*", " — ", text)
    return text


def _process_chunk_ner(text: str, nlp) -> str:
    """
    Внутренняя функция: прогон NER по куску текста.
    PERSON -> " @ " (с пробелами), чтобы маркер не прилипал к буквам.
    """
    doc = nlp(text)
    out_parts: list[str] = []
    last_idx = 0

    for ent in doc.ents:
        if ent.label_ == "PER":
            out_parts.append(text[last_idx : ent.start_char])
            out_parts.append(f" {PER_MARK} ")
            last_idx = ent.end_char

    out_parts.append(text[last_idx:])
    return "".join(out_parts)


def mask_names(text: str) -> str:
    """
    Заменяет PERSON (имена) на единый маркер PER_MARK.
    Разбивает текст на куски, чтобы SpaCy не упал по памяти/лимиту.
    """
    nlp = get_ner_nlp()

    if len(text) < NER_CHUNK_SIZE:
        return _process_chunk_ner(text, nlp)

    parts: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + NER_CHUNK_SIZE, text_len)
        if end < text_len:
            # Ищем пробел, чтобы не разрезать слово/имя
            last_space = text.rfind(" ", start, end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end]
        if chunk.strip():
            parts.append(_process_chunk_ner(chunk, nlp))

        start = end + 1  # пропускаем пробел

    return " ".join(parts)


def normalize(text: str) -> str:
    text = normalize_dashes(text)

    text = mask_names(text)

    # Ё -> Е для устойчивости признаков
    text = text.replace("ё", "е").replace("Ё", "Е")

    # Оставляем буквы, цифры, пробелы, базовую пунктуацию и PER_MARK;
    # '@' разрешён явно, иначе маркер PERSON был бы вычищен как мусор
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s\.,;:!?—\-«»\"@]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process_file(
    file_path: pathlib.Path, src_root: pathlib.Path, dst_root: pathlib.Path
) -> int:
    """
    Обрабатывает один файл. Функция должна быть на верхнем уровне для pickle/joblib.
    Возвращает 1 в случае успеха, 0 при ошибке/пропуске.
    """
    try:
        rel_path = file_path.relative_to(src_root)
        out_path = dst_root / rel_path

        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            return 0

        clean = normalize(raw)
        if not clean:
            return 0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(clean, "utf-8")
        return 1

    except Exception as e:
        logging.error(f"Ошибка при обработке {file_path.name}: {e}")
        return 0


def main(src_dir: str, dst_dir: str):
    src = pathlib.Path(src_dir)
    dst = pathlib.Path(dst_dir)

    if not src.exists():
        logging.error(f"Источник {src} не найден")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    all_files: list[pathlib.Path] = []
    for author_dir in sorted(src.iterdir()):
        if not author_dir.is_dir():
            continue
        all_files.extend(sorted(author_dir.glob("*.txt")))

    total_files = len(all_files)
    logging.info(f"Найдено файлов: {total_files}. Запуск обработки на всех ядрах...")

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_file)(fp, src, dst) for fp in all_files
    )

    n_processed = int(sum(results))
    logging.info(f"Готово. Обработано файлов: {n_processed}/{total_files}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/clean_text.py <src_dir> <dst_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
