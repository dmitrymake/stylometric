"""Базовый контракт фич-блока.

Блоки работают с предвычисленными лёгкими представлениями Rep (stylo.features.reps),
а не с тяжёлыми spaCy Doc — это даёт быстрый leakage-free LOBO/sweep (Rep грузятся
один раз на процесс из единого файла, без повторной десериализации DocBin).
"""
from __future__ import annotations

import abc
from typing import List, Sequence

from scipy.sparse import csr_matrix


class FeatureBlock(abc.ABC):
    """Абстрактный блок признаков.

    name  — уникальное имя блока (для конфигов/отчётов), напр. "char_ngrams".
    group — группа для ablation-sweep (часто == name; субблоки делят group).
    """

    name: str = "block"
    group: str = "block"

    @abc.abstractmethod
    def fit(self, texts: Sequence[str], reps: Sequence) -> "FeatureBlock":
        ...

    @abc.abstractmethod
    def transform(self, texts: Sequence[str], reps: Sequence) -> csr_matrix:
        ...

    def fit_transform(self, texts: Sequence[str], reps: Sequence) -> csr_matrix:
        return self.fit(texts, reps).transform(texts, reps)

    @abc.abstractmethod
    def feature_names(self) -> List[str]:
        ...

    def n_features(self) -> int:
        return len(self.feature_names())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r}>"
