"""StyloVectorizer — сборка итогового вектора из включённых блоков.

Совместим со sklearn Pipeline (fit/transform принимают только тексты). Лёгкие
представления Rep берутся из RepCache (единый файл, грузится раз на процесс) —
это и есть основа быстрого leakage-free LOBO/sweep без повторной десериализации.

Пиклится безопасно: RepCache/DocCache не держат «живую» spaCy-модель.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin

from .features.base import FeatureBlock
from .features.registry import build_blocks
from .features.reps import RepCache, make_rep_cache
from .features.work_vectorizer import validate_work_ids


class StyloVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, blocks: List[FeatureBlock], rep_cache: RepCache):
        self.blocks = blocks
        self.rep_cache = rep_cache
        self._block_slices: List[Tuple[str, int, int]] = []

    @classmethod
    def from_config(cls, cfg, enabled_override: Optional[Dict[str, bool]] = None,
                    topic_strict: bool = False) -> "StyloVectorizer":
        return cls(build_blocks(cfg, enabled_override, topic_strict=topic_strict), make_rep_cache(cfg))

    def _reps(self, X: Sequence[str]):
        return self.rep_cache.get_reps(list(X))

    def fit(self, X, y=None, groups=None):
        X = list(X)
        if groups is not None:
            groups = validate_work_ids(groups, len(X))   # single B0 contract, fail-closed, pre-_reps
        reps = self._reps(X)
        for b in self.blocks:
            if groups is None:
                b.fit(X, reps)                            # exact legacy two-argument call (P0 parity)
            else:
                b.fit(X, reps, groups=groups)
        return self

    def transform(self, X) -> csr_matrix:
        reps = self._reps(X)
        parts = []
        self._block_slices = []
        col = 0
        for b in self.blocks:
            m = b.transform(X, reps)
            if not isinstance(m, csr_matrix):
                m = csr_matrix(m)
            parts.append(m)
            self._block_slices.append((b.name, col, col + m.shape[1]))
            col += m.shape[1]
        return hstack(parts, format="csr")

    def fit_transform(self, X, y=None, groups=None) -> csr_matrix:
        if groups is None:
            return self.fit(X, y).transform(X)
        return self.fit(X, y, groups=groups).transform(X)

    def feature_names(self) -> List[str]:
        names: List[str] = []
        for b in self.blocks:
            names.extend(b.feature_names())
        return names

    def block_slices(self) -> List[Tuple[str, int, int]]:
        return list(self._block_slices)
