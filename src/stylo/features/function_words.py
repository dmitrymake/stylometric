"""Частотности функциональных слов.

Два режима:
  mfw        — топ-N самых частотных слов (CountVectorizer max_features),
  fixed_list — фиксированный язык-специфичный список (RU: 405 слов из lang.py).

Возвращает сырые счётчики; нормализация (MaxAbs/Z-score) — на стороне модели.
Для НАСТОЯЩЕЙ Burrows's Delta (z-score относительных частот) см. stylo.models.delta.
"""
from __future__ import annotations

from typing import List, Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokens import Doc

from ..lang import function_words
from .base import FeatureBlock

_TOKEN_PATTERN = r"(?u)\b\w+\b"


class FunctionWordBlock(FeatureBlock):
    group = "function_words"
    name = "function_words"

    def __init__(self, mode: str = "mfw", mfw_count: int = 300, lang: str = "ru"):
        assert mode in {"mfw", "fixed_list"}
        self.mode = mode
        self.mfw_count = mfw_count
        self.lang = lang
        self._vec: CountVectorizer | None = None

    def fit(self, texts, reps):
        if self.mode == "mfw":
            self._vec = CountVectorizer(
                max_features=self.mfw_count, lowercase=True, token_pattern=_TOKEN_PATTERN,
            )
        else:
            vocab = sorted(function_words(self.lang))
            self._vec = CountVectorizer(
                vocabulary=vocab, lowercase=True, token_pattern=_TOKEN_PATTERN,
            )
        self._vec.fit(list(texts))
        return self

    def transform(self, texts, reps) -> csr_matrix:
        assert self._vec is not None, "fit перед transform"
        return self._vec.transform(list(texts))

    def feature_names(self) -> List[str]:
        assert self._vec is not None
        return [f"fw::{w}" for w in self._vec.get_feature_names_out()]
