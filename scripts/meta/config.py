"""
Конфигурация языковых параметров.
"""

# Настройки маскировки
DEFAULT_POS_REPLACEMENTS = {
    "PROPN": "^",
    "NOUN": "@",
    "NUM": "%",
    "ADJ": "&"
}

# --- ENGLISH RESOURCES ---
# Расширенный список (300+ слов), основанный на частотных словарях (Brown Corpus + Stylometry Standard).
# Список исключает контент-слова (water/oil/food/country/school/father/tree/river/mother/family/
# animal/...): они НЕСУТ ТЕМУ и ломают топик-инвариантность стиля. Канон — src/stylo/lang.py EN_FUNCTION_WORDS.
EN_FUNCTION_WORDS = {
    'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 'for', 'on',
    'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one',
    'had', 'by', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'there', 'an',
    'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out',
    'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'him', 'into', 'has', 'two',
    'more', 'no', 'way', 'could', 'my', 'than', 'been', 'who', 'its', 'now', 'did', 'get', 'may',
    'over', 'only', 'me', 'back', 'most', 'very', 'after', 'our', 'just', 'where', 'much', 'before',
    'too', 'any', 'same', 'also', 'around', 'does', 'another', 'well', 'must', 'even', 'such',
    'because', 'here', 'why', 'off', 'again', 'while', 'might', 'next', 'those', 'both', 'until',
    'us', 'without', 'once', 'should', 'every', 'between', 'own', 'below', 'under', 'few', 'along',
    'something', 'seem', 'always', 'against', 'yet', 'though', 'upon', 'within', 'whose', 'whom',
}

# --- RUSSIAN RESOURCES ---
from .function_words import RUS_FUNCTION_WORDS

# --- FRENCH RESOURCES ---
from .function_words import FR_FUNCTION_WORDS

LANG_CONFIG = {
    'ru': {
        'name': 'Russian',
        'base_model': 'ru_core_news_lg',
        'err_model': 'ru_core_news_md',
        'function_words': RUS_FUNCTION_WORDS,
        'vowels_hard': 'аоуэы',
        'vowels_soft': 'иеяёю',
        'pos_replacements': DEFAULT_POS_REPLACEMENTS
    },
    'en': {
        'name': 'English',
        'base_model': 'en_core_web_lg',
        'err_model': 'en_core_web_md',
        'function_words': EN_FUNCTION_WORDS,
        'vowels_hard': 'aou', 
        'vowels_soft': 'ei', 
        'pos_replacements': DEFAULT_POS_REPLACEMENTS
    },
    'fr': {
        'name': 'French',
        'base_model': 'fr_core_news_lg',
        'err_model': 'fr_core_news_md',
        'function_words': FR_FUNCTION_WORDS,
        'vowels_hard': 'aou',
        'vowels_soft': 'ei',
        'pos_replacements': DEFAULT_POS_REPLACEMENTS
    }
}
