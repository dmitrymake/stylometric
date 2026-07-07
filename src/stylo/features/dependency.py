"""Признаки синтаксических зависимостей (dependency parse).

Состав:
  - нормированное распределение типов связей (t.dep_), словарь фиксируется на train;
  - агрегаты по дереву: средняя/макс дистанция головы |i-head.i|, средняя/макс глубина
    дерева, средняя ветвистость (число детей), доля листьев.

Глубокий синтаксис, тематически-инвариантен. Качество зависит от ru-парсера —
поэтому ценность проверяется sweep'ом, а не постулируется.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from .base import FeatureBlock

_AGG_NAMES = [
    "head_dist_mean", "head_dist_max",
    "tree_depth_mean", "tree_depth_max",
    "branching_mean", "leaf_ratio",
]


def _token_depth(tok) -> int:
    d = 0
    while tok.head != tok:
        tok = tok.head
        d += 1
        if d > 200:  # защита от циклов в кривом парсе
            break
    return d


class DependencyBlock(FeatureBlock):
    group = "dependency"
    name = "dependency"

    def __init__(self, min_df: int = 3):
        self.min_df = min_df
        self.dep_vocab: List[str] = []

    def fit(self, texts, reps):
        df = Counter()
        for r in reps:
            for dep in r.dep_counts.keys():
                df[dep] += 1
        self.dep_vocab = sorted(k for k, c in df.items() if c >= self.min_df)
        return self

    @staticmethod
    def _raw(doc: Doc):
        """Фолд-независимая часть из spaCy Doc: (n, dep_counts, agg). Для построения Rep."""
        n = len(doc)
        dep_counts = Counter(t.dep_ for t in doc)
        dist = [abs(t.i - t.head.i) for t in doc]
        depths = [_token_depth(t) for t in doc]
        nchild = [len(list(t.children)) for t in doc]
        leaves = sum(1 for c in nchild if c == 0)
        agg = [
            float(np.mean(dist)) if dist else 0.0,
            float(np.max(dist)) if dist else 0.0,
            float(np.mean(depths)) if depths else 0.0,
            float(np.max(depths)) if depths else 0.0,
            float(np.mean(nchild)) if nchild else 0.0,
            leaves / n if n else 0.0,
        ]
        return n, dep_counts, agg

    def _row(self, rep) -> List[float]:
        n, dep_counts, agg = rep.dep_n, rep.dep_counts, rep.dep_agg
        dep_part = [dep_counts.get(dep, 0) / n if n else 0.0 for dep in self.dep_vocab]
        return dep_part + agg

    def transform(self, texts, reps) -> csr_matrix:
        rows = [self._row(r) for r in reps]
        arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, self.n_features()), np.float32)
        return csr_matrix(arr)

    def feature_names(self) -> List[str]:
        return [f"dep::{d}" for d in self.dep_vocab] + [f"dep::{a}" for a in _AGG_NAMES]
