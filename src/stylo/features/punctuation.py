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
from .work_vectorizer import MODE_TFIDF, WorkLevelVectorizer

_DASHES = {"—", "–", "―", "-"}
_TOKEN_PATTERN = r"\S+"
_ANALYZER = {"analyzer": "word", "token_pattern": _TOKEN_PATTERN, "lowercase": False}


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
        self._vec: TfidfVectorizer | None = None       # legacy: chunk-pooled tf-idf
        self._wv: WorkLevelVectorizer | None = None      # work_balanced: equal-work vocab/IDF

    def fit(self, texts, reps, groups=None):
        punct = [r.punct_str for r in reps]
        if groups is None:
            self._vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=2,
                sublinear_tf=True,
                token_pattern=_TOKEN_PATTERN,
                lowercase=False,
            )
            self._vec.fit(punct)
            self._wv = None
        else:
            self._wv = WorkLevelVectorizer(
                analyzer_params={**_ANALYZER, "ngram_range": self.ngram_range},
                mode=MODE_TFIDF, max_features=self.max_features, min_df_works=2, sublinear_tf=True,
            )
            self._wv.fit(punct, groups)
            self._vec = None
        return self

    def transform(self, texts, reps) -> csr_matrix:
        punct = [r.punct_str for r in reps]
        if getattr(self, "_wv", None) is not None:
            return self._wv.transform(punct)
        assert self._vec is not None
        return self._vec.transform(punct)

    def feature_names(self) -> List[str]:
        if getattr(self, "_wv", None) is not None:
            return [f"punct::{f}" for f in self._wv.feature_names()]
        assert self._vec is not None
        return [f"punct::{f}" for f in self._vec.get_feature_names_out()]
