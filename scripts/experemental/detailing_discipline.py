import numpy as np
import math
from collections import Counter
from typing import List, Tuple
import spacy

# Порог ADJ+ADV в предложении для метки "плотное" (Dense).
# 3 детализирующих слова ~ 15% при средней длине предложения около 20 слов.
DETAIL_WORD_THRESHOLD = 3


def calculate_detailing_entropy(doc: spacy.tokens.Doc) -> float:
    """
    Рассчитывает Энтропию Шеннона по биграммам ритма детализации (Dense/Sparse).
    """

    # Dense = предложение с ADJ + ADV >= THRESHOLD (Dense=1, Sparse=0)
    rhythm_sequence = []

    for sent in doc.sents:
        detailing_words_count = 0

        for token in sent:
            if token.pos_ in {"ADJ", "ADV"}:
                detailing_words_count += 1

        if len(list(sent)) > 2:  # Игнорируем очень короткие фразы (2 слова или меньше)
            is_dense = 1 if detailing_words_count >= DETAIL_WORD_THRESHOLD else 0
            rhythm_sequence.append(is_dense)

    if len(rhythm_sequence) < 2:
        return 0.0

    # Энтропия Шеннона по биграммам ритма (00, 01, 10, 11):
    # мера непредсказуемости смены детализации.
    rhythm_bigrams = []
    for i in range(len(rhythm_sequence) - 1):
        bigram = (rhythm_sequence[i], rhythm_sequence[i+1])
        rhythm_bigrams.append(bigram)

    if not rhythm_bigrams:
        return 0.0

    total = len(rhythm_bigrams)
    freq = Counter(rhythm_bigrams)
    h = 0.0
    for count in freq.values():
        p = count / total
        h -= p * math.log2(p)

    return h
