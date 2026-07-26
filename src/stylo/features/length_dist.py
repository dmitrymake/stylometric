"""Распределения длин слов и предложений (а не только средние).

  - гистограмма долей слов длины 1..max_word_len (+ хвост) — классика Менденхолла;
  - статистики длины предложения: std, коэф. вариации, перцентили p10/p50/p90,
    доля очень коротких (<5 слов) и очень длинных (>30 слов) предложений.

Фиксированная размерность, fit не требуется.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from .base import FeatureBlock

_SENT_NAMES = [
    "sent_std", "sent_cv", "sent_p10", "sent_p50", "sent_p90",
    "sent_frac_short", "sent_frac_long",
]


class LengthDistBlock(FeatureBlock):
    group = "length_dist"
    name = "length_dist"

    def __init__(self, max_word_len: int = 16):
        self.max_word_len = max_word_len

    def fit(self, texts, reps, groups=None):
        return self

    def _row(self, rep) -> List[float]:
        hist = rep.word_len_hist
        total = sum(hist) or 1
        word_part = [c / total for c in hist]

        sent_lens = rep.sent_lens
        if sent_lens:
            arr = np.asarray(sent_lens, dtype=float)
            mean = arr.mean()
            std = float(arr.std())
            cv = std / mean if mean else 0.0
            p10, p50, p90 = (float(x) for x in np.percentile(arr, [10, 50, 90]))
            frac_short = float(np.mean(arr < 5))
            frac_long = float(np.mean(arr > 30))
            sent_part = [std, cv, p10, p50, p90, frac_short, frac_long]
        else:
            sent_part = [0.0] * len(_SENT_NAMES)
        return word_part + sent_part

    def transform(self, texts, reps) -> csr_matrix:
        rows = [self._row(r) for r in reps]
        arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, self.n_features()), np.float32)
        return csr_matrix(arr)

    def feature_names(self) -> List[str]:
        wl = [f"len::word_{i+1}" for i in range(self.max_word_len)] + [f"len::word_{self.max_word_len+1}plus"]
        return wl + [f"len::{n}" for n in _SENT_NAMES]
