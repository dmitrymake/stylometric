"""Вспомогательные стилометрические статистики на уровне одного spaCy Doc.

Содержит Vowel-Rhythm-Entropy и ЧЕСТНУЮ версию SSA (по реальной длине предложения,
а не по прокси «длина слова»).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np
from spacy.tokens import Doc


def vowel_rhythm_entropy(doc: Doc, hard: str, soft: str) -> float:
    """Энтропия Шеннона биграмм ритма гласных (H=твёрдые, S=мягкие).

    Ловит «певучесть»/монотонность авторской фонетики; тематически-инвариантна.
    """
    seq: List[str] = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        for ch in tok.text.lower():
            if ch in hard:
                seq.append("H")
            elif ch in soft:
                seq.append("S")
    if len(seq) < 20:
        return 0.0
    bigrams = [seq[i] + seq[i + 1] for i in range(len(seq) - 1)]
    total = len(bigrams)
    if total == 0:
        return 0.0
    h = 0.0
    for c in Counter(bigrams).values():
        p = c / total
        h -= p * math.log2(p)
    return h


def ssa_metrics(doc: Doc, min_points: int = 6) -> Tuple[float, float, float]:
    """Semantic-Syntactic Alignment (ЧЕСТНАЯ версия).

    Внутри чанка по ПРЕДЛОЖЕНИЯМ считаем пары (длина предложения в словах, TTR
    предложения) и строим регрессию TTR ~ sent_len. Возвращаем (slope, intercept, R²).

    Используется реальная длина предложения (doc.sents), а не средняя длина слова
    как прокси. Если предложений мало (< min_points) или нет вариативности —
    возвращаем нули (sweep решит, полезен ли блок).
    """
    sent_lens: List[float] = []
    ttrs: List[float] = []
    for sent in doc.sents:
        words = [t.lemma_.lower() for t in sent if t.is_alpha]
        n = len(words)
        if n < 5:
            continue
        sent_lens.append(float(n))
        ttrs.append(len(set(words)) / n)
    if len(sent_lens) < min_points:
        return 0.0, 0.0, 0.0
    x = np.asarray(sent_lens)
    y = np.asarray(ttrs)
    if x.std() == 0 or y.std() == 0:
        return 0.0, 0.0, 0.0
    # polyfit, чтобы не тянуть зависимость от scipy.stats
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    r2 = float(corr * corr)
    return float(slope), float(intercept), r2
