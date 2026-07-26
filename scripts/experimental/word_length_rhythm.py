"""Experimental word-length transition entropy features."""

import numpy as np
import math
from collections import Counter
from typing import List, Tuple
import spacy

# Разбиваем слова на категории по длине (для прокси слогового ритма)
# 1: 1-2 буквы (очень короткие, служебные)
# 2: 3-5 букв (короткие, частые)
# 3: 6-8 букв (средние)
# 4: 9+ букв (длинные)


def _get_word_length_category(word_len: int) -> int:
    if word_len <= 2:
        return 1
    if word_len <= 5:
        return 2
    if word_len <= 8:
        return 3
    return 4


def calculate_word_length_entropy(doc: spacy.tokens.Doc) -> float:
    """
    Рассчитывает Энтропию Шеннона для паттернов длины слов в документе.
    Низкая энтропия = очень регулярный ритм (много слов одной длины).
    Высокая энтропия = нерегулярный ритм (равномерное чередование коротких и длинных).
    """
    length_patterns = []

    for token in doc:
        if token.is_alpha:
            length_patterns.append(_get_word_length_category(len(token.text)))

    if not length_patterns:
        return 0.0

    total = len(length_patterns)
    freq = Counter(length_patterns)
    h = 0.0
    for count in freq.values():
        p = count / total
        h -= p * math.log2(p)

    return h
