import numpy as np
import math
from collections import Counter
from typing import List, Tuple
import spacy


def _calc_local_entropy(tokens: List[str], window_size: int = 200) -> float:
    """Расчет среднего значения энтропии токенов по скользящим окнам."""
    if len(tokens) < window_size:
        return 0.0

    entropies = []
    # Перекрытие 75% (шаг = window_size // 4)
    for i in range(0, len(tokens) - window_size + 1, window_size // 4):
        window = tokens[i: i + window_size]
        total = len(window)
        freq = Counter(window)
        h = 0.0
        for count in freq.values():
            p = count / total
            h -= p * math.log2(p)
        entropies.append(h)

    return np.mean(entropies) if entropies else 0.0


def _calc_syntactic_complexity(doc: spacy.tokens.Doc) -> float:
    """
    Рассчитывает среднюю глубину синтаксического дерева и среднюю длину зависимости
    для предложений в документе. Индекс сложности = Глубина * Длина.
    """
    depths = []
    dep_lengths = []

    for sent in doc.sents:
        if len(sent) < 3:
            continue

        max_depth = 0
        num_tokens = 0

        for token in sent:
            num_tokens += 1

            current_depth = 0
            temp_token = token
            while temp_token.head != temp_token:
                current_depth += 1
                temp_token = temp_token.head
            max_depth = max(max_depth, current_depth)

            # Длина зависимости (расстояние до головы)
            dep_lengths.append(abs(token.i - token.head.i))

        if num_tokens > 0:
            depths.append(max_depth)

    avg_depth = np.mean(depths) if depths else 0.0
    avg_dep_len = np.mean(dep_lengths) if dep_lengths else 0.0

    complexity_index = avg_depth * avg_dep_len

    return complexity_index


def get_anomaly_features(doc: spacy.tokens.Doc) -> Tuple[float, float]:
    """
    Извлекает признаки: (Средняя локальная энтропия, Индекс синтаксической сложности).
    """
    alpha_tokens = [t.text.lower() for t in doc if t.is_alpha]
    local_entropy = _calc_local_entropy(alpha_tokens)

    syntactic_complexity = _calc_syntactic_complexity(doc)

    return local_entropy, syntactic_complexity
