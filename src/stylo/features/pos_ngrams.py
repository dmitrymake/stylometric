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
from .work_vectorizer import MODE_TFIDF, WorkLevelVectorizer

_TOKEN_PATTERN = r"(?u)[A-Z]+"   # POS-теги — заглавные латиницей
_ANALYZER = {"analyzer": "word", "token_pattern": _TOKEN_PATTERN, "lowercase": False}


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
        self._vec: TfidfVectorizer | None = None       # legacy: chunk-pooled tf-idf
        self._wv: WorkLevelVectorizer | None = None      # work_balanced: equal-work vocab/IDF

    def fit(self, texts, reps, groups=None):
        pos = [r.pos_str for r in reps]
        if groups is None:
            self._vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                sublinear_tf=True,
                token_pattern=_TOKEN_PATTERN,
                lowercase=False,
            )
            self._vec.fit(pos)
            self._wv = None
        else:
            self._wv = WorkLevelVectorizer(
                analyzer_params={**_ANALYZER, "ngram_range": self.ngram_range},
                mode=MODE_TFIDF, max_features=self.max_features, min_df_works=2, sublinear_tf=True,
            )
            self._wv.fit(pos, groups)
            self._vec = None
        return self

    def transform(self, texts, reps) -> csr_matrix:
        pos = [r.pos_str for r in reps]
        if getattr(self, "_wv", None) is not None:
            return self._wv.transform(pos)
        assert self._vec is not None
        return self._vec.transform(pos)

    def feature_names(self) -> List[str]:
        if getattr(self, "_wv", None) is not None:
            return [f"pos::{f}" for f in self._wv.feature_names()]
        assert self._vec is not None
        return [f"pos::{f}" for f in self._vec.get_feature_names_out()]
