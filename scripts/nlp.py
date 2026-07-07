import spacy
import logging
from meta.meta import BASE_LANG_MODEL, ERR_LANG_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_NLP_STYLO = None
_NLP_NER = None


def get_stylometry_nlp(max_length: int = 5_000_000):
    """
    Загружает или возвращает кешированную SpaCy-модель, необходимую для:
    - Лемматизации
    - POS-теггинга (Part-of-Speech)
    - Деления на предложения (Sentencizer)
    """
    global _NLP_STYLO
    if _NLP_STYLO is None:
        try:
            # Отключаем NER и TextCat, которые не нужны для стилометрии
            _NLP_STYLO = spacy.load(BASE_LANG_MODEL, disable=["ner", "textcat"])
        except OSError:
            logging.warning(
                f"Основная модель {BASE_LANG_MODEL} не найдена. Используется {ERR_LANG_MODEL}."
            )
            _NLP_STYLO = spacy.load(ERR_LANG_MODEL, disable=["ner", "textcat"])

        # Убедимся, что sentencizer добавлен, если его нет (хотя обычно он есть)
        if "sentencizer" not in _NLP_STYLO.pipe_names:
            _NLP_STYLO.add_pipe("sentencizer")
        _NLP_STYLO.max_length = max_length
        logging.info(f"SpaCy Stylometry Model ({_NLP_STYLO.meta['name']}) загружена.")

    return _NLP_STYLO


def get_ner_nlp():
    """
    Загружает или возвращает кешированную SpaCy-модель ТОЛЬКО с компонентом NER
    для маскировки имен собственных (самый долгий шаг).
    """
    global _NLP_NER
    if _NLP_NER is None:
        try:
            _NLP_NER = spacy.load(
                BASE_LANG_MODEL,
                disable=[
                    "parser",
                    "tagger",
                    "attribute_ruler",
                    "lemmatizer",
                    "sentencizer",
                    "textcat",
                ],
            )
        except OSError:
            # fallback
            _NLP_NER = spacy.load(
                ERR_LANG_MODEL,
                disable=[
                    "parser",
                    "tagger",
                    "attribute_ruler",
                    "lemmatizer",
                    "sentencizer",
                    "textcat",
                ],
            )
        logging.info(f"SpaCy NER Model ({_NLP_NER.meta['name']}) загружена.")
    return _NLP_NER
