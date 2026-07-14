"""Морфологический профиль автора (через spaCy t.morph).

Нормированные частоты грамматических признаков «Feat=Value» (Case=Nom, Tense=Past,
Aspect=Imp, Number=Sing, Degree=Cmp, Mood=Ind, …). Словарь признаков фиксируется
на train. Для флективного русского это один из богатейших стилевых сигналов.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from .base import FeatureBlock
from .work_vectorizer import MIN_DF_WORKS, count_marks_presence, validate_work_ids


def _morph_items_raw(doc: Doc) -> Counter:
    """Счётчик морфопризнаков feat=value из spaCy Doc (для построения Rep)."""
    c = Counter()
    for tok in doc:
        if not tok.is_alpha:
            continue
        for feat, val in tok.morph.to_dict().items():
            c[f"{feat}={val}"] += 1
    return c


class MorphologyBlock(FeatureBlock):
    group = "morphology"
    name = "morphology"

    def __init__(self, min_df: int = 5):
        self.min_df = min_df
        self.vocab: List[str] = []

    def fit(self, texts, reps, groups=None):
        if groups is None:
            # legacy: chunk document-frequency (a prolific work can meet min_df alone)
            df = Counter()
            for r in reps:
                for key in r.morph.keys():
                    df[key] += 1
            self.vocab = sorted(k for k, c in df.items() if c >= self.min_df)
        else:
            # work_balanced: a feature counts once per WORK that actually contains it (count>0)
            work_ids = validate_work_ids(groups, len(reps))
            feat_works: dict[str, set] = defaultdict(set)
            for r, w in zip(reps, work_ids):
                for key, cnt in r.morph.items():
                    if count_marks_presence(cnt, f"morphology[{key}]"):
                        feat_works[key].add(w)
            self.vocab = sorted(k for k, ws in feat_works.items() if len(ws) >= MIN_DF_WORKS)
            if not self.vocab:
                raise ValueError(
                    f"empty vocabulary after work-DF pruning (morphology, min_df_works={MIN_DF_WORKS}); "
                    "check the corpus"
                )
        return self

    def _row(self, rep) -> List[float]:
        counts = rep.morph
        total = sum(counts.values())
        if total == 0:
            return [0.0] * len(self.vocab)
        return [counts.get(k, 0) / total for k in self.vocab]

    def transform(self, texts, reps) -> csr_matrix:
        rows = [self._row(r) for r in reps]
        arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, self.n_features()), np.float32)
        return csr_matrix(arr)

    def feature_names(self) -> List[str]:
        return [f"morph::{k}" for k in self.vocab]
