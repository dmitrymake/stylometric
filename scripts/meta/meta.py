"""
Модуль метаданных и глобальной конфигурации.
Автоматически выбирает настройки в зависимости от языка.
"""
from __future__ import annotations
import json
import pathlib
import os
from .config import LANG_CONFIG

# Переопределяется через export STYLO_LANG=en
CURRENT_LANG = os.getenv("STYLO_LANG", "ru")

if CURRENT_LANG not in LANG_CONFIG:
    print(f"Warning: Language '{CURRENT_LANG}' not found in config. Fallback to 'ru'.")
    CURRENT_LANG = "ru"

CFG = LANG_CONFIG[CURRENT_LANG]

# Экспорт глобальных констант
FUNCTION_WORDS = CFG['function_words']
BASE_LANG_MODEL = CFG['base_model']
ERR_LANG_MODEL = CFG['err_model']

# Настройки для алгоритмов
POS_REPLACEMENTS = CFG['pos_replacements']
HARD_VOWELS = CFG['vowels_hard']
SOFT_VOWELS = CFG['vowels_soft']

HERE = pathlib.Path(__file__).resolve().parent
JSON_PATH = HERE / "authors.json"

try:
    with open(JSON_PATH, encoding="utf-8") as f:
        AUTHOR_META: dict[str, dict[str, str]] = json.load(f)
except FileNotFoundError:
    AUTHOR_META = {"unknown": {"name": "-", "work": "-"}}

def display_name(author_id: str) -> str:
    return AUTHOR_META.get(author_id, {}).get("name", author_id)

def display_label(author_id: str) -> str:
    meta = AUTHOR_META.get(author_id)
    if not meta: return author_id
    name = meta.get("name", author_id)
    work = meta.get("work", "—")
    return f"{name} ({work})" if work and work != "—" else name
