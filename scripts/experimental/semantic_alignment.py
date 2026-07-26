"""Experimental within-document sentence-length/TTR alignment features."""

import numpy as np
from typing import List, Tuple
import spacy
from scipy.stats import linregress
from collections import Counter

# Размер суб-фрагмента для расчета корреляции внутри чанка (50 слов)
SUB_CHUNK_SIZE = 50


def calculate_alignment_metrics(doc: spacy.tokens.Doc) -> Tuple[float, float, float]:
    """
    Рассчитывает мета-признаки SSA: Наклон (Slope), Пересечение (Intercept) и R-квадрат (R2) 
    корреляции между средней длиной предложения и TTR (лексическим разнообразием) 
    по 10 суб-фрагментам внутри данного Doc (чанка).
    """

    words = [t for t in doc if t.is_alpha and t.text != "<PER>"]

    if len(words) < SUB_CHUNK_SIZE * 2:
        # Недостаточно данных для надежной внутренней корреляции
        return 0.0, 0.0, 0.0

    avg_sent_lengths = []  # Прокси для Синтаксиса (Структура)
    ttr_values = []       # Прокси для Семантики (Лексика)

    for i in range(0, len(words), SUB_CHUNK_SIZE):
        sub_chunk = words[i:i + SUB_CHUNK_SIZE]

        if len(sub_chunk) >= SUB_CHUNK_SIZE * 0.8:

            V = len(Counter(t.lemma_.lower() for t in sub_chunk))
            N = len(sub_chunk)
            ttr = V / N if N > 0 else 0.0
            ttr_values.append(ttr)

            # Средняя длина слова как прокси для структуры предложения
            # (упрощение: избегаем зависимости от sentence split)
            avg_word_len = np.mean([len(t.text) for t in sub_chunk])
            avg_sent_lengths.append(avg_word_len)

    if len(avg_sent_lengths) < 3:
        # Нужно минимум 3 точки для надежной корреляции
        return 0.0, 0.0, 0.0

    # linregress даёт деление на ноль на константных данных
    if np.std(avg_sent_lengths) == 0 or np.std(ttr_values) == 0:
        return 0.0, 0.0, 0.0

    slope, intercept, r_value, p_value, std_err = linregress(
        avg_sent_lengths, ttr_values)

    r_squared = r_value**2

    return slope, intercept, r_squared
