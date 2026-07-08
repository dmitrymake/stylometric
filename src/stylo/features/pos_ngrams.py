"""POS n-граммы — синтаксический «скелет» текста.

Последовательность POS-тегов (NOUN VERB ADP …) векторизуется как word-n-граммы.
Сильный тематически-инвариантный сигнал для русской прозы.
"""
from __future__ import annotations

from typing import List, Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc

from .base import FeatureBlock


def _pos_string_raw(doc: Doc) -> str:
    """Строка POS-тегов из spaCy Doc (используется при построении Rep)."""
    return " ".join(t.pos_ for t in doc if not t.is_space)


class PosNgramBlock(FeatureBlock):
    group = "pos_ngrams"
    name = "pos_ngrams"

    def __init__(self, ngram_range=(2, 4), max_features: int = 2000, min_df: int = 3):
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.min_df = min_df
        self._vec: TfidfVectorizer | None = None

    def fit(self, texts, reps):
        self._vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)[A-Z]+",   # POS-теги — заглавные латиницей
            lowercase=False,
        )
        self._vec.fit([r.pos_str for r in reps])
        return self

    def transform(self, texts, reps) -> csr_matrix:
        assert self._vec is not None
        return self._vec.transform([r.pos_str for r in reps])

    def feature_names(self) -> List[str]:
        assert self._vec is not None
        return [f"pos::{f}" for f in self._vec.get_feature_names_out()]
