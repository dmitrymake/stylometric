"""Пунктуационные n-граммы — авторская привычка расстановки знаков.

Берём только пунктуационные токены, строим их последовательность и
векторизуем как word-n-граммы. Дёшево, тематически-нейтрально
(особенно тире/прямая речь и точка с запятой в русской прозе).
"""
from __future__ import annotations

from typing import List, Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc

from .base import FeatureBlock

_DASHES = {"—", "–", "―", "-"}


def _punct_token(text: str) -> str:
    if text in _DASHES:
        return "—"
    return text


def _punct_string_raw(doc: Doc) -> str:
    """Строка пунктуации из spaCy Doc."""
    return " ".join(_punct_token(t.text) for t in doc if t.is_punct)


class PunctNgramBlock(FeatureBlock):
    group = "punctuation_ngrams"
    name = "punctuation_ngrams"

    def __init__(self, ngram_range=(1, 3), max_features: int = 500):
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self._vec: TfidfVectorizer | None = None

    def fit(self, texts, reps):
        self._vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=2,
            sublinear_tf=True,
            token_pattern=r"\S+",
            lowercase=False,
        )
        self._vec.fit([r.punct_str for r in reps])
        return self

    def transform(self, texts, reps) -> csr_matrix:
        assert self._vec is not None
        return self._vec.transform([r.punct_str for r in reps])

    def feature_names(self) -> List[str]:
        assert self._vec is not None
        return [f"punct::{f}" for f in self._vec.get_feature_names_out()]
