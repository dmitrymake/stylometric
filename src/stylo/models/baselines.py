"""Baseline-модели для честного сравнения «нужны ли сложные фичи».

  - MajorityBaseline      : нижняя граница (всегда самый частый автор);
  - CharCosineBaseline    : char-3gram TF-IDF + косинус к центроидам (без bleaching);
  - build_bow_lr          : мешок слов + логрег.

Все принимают сырые тексты и дают predict_proba(texts) с .classes_ —
совместимо с единым LOBO-движком.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


class MajorityBaseline:
    def __init__(self):
        self.classes_ = None
        self._maj = None

    def fit(self, texts, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        vals, counts = np.unique(y, return_counts=True)
        self._maj = int(vals[np.argmax(counts)])
        return self

    def predict_proba(self, texts):
        n = len(list(texts))
        proba = np.zeros((n, len(self.classes_)))
        col = int(np.where(self.classes_ == self._maj)[0][0])
        proba[:, col] = 1.0
        return proba


class CharCosineBaseline:
    """Char-n-gram TF-IDF + косинус к центроидам авторов (классический char-baseline)."""

    def __init__(self, ngram_range=(3, 3), max_features=5000, min_df=3):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.classes_ = None
        self._vec = None
        self._centroids = None

    def fit(self, texts, y):
        y = np.asarray(y)
        self._vec = TfidfVectorizer(analyzer="char", ngram_range=self.ngram_range,
                                    max_features=self.max_features, min_df=self.min_df,
                                    sublinear_tf=True)
        X = self._vec.fit_transform(list(texts))
        self.classes_ = np.unique(y)
        self._centroids = np.vstack([np.asarray(X[y == c].mean(axis=0)).ravel()
                                     for c in self.classes_])
        return self

    def predict_proba(self, texts):
        X = self._vec.transform(list(texts))
        d = cosine_distances(X, self._centroids)
        x = -d
        x = x - x.max(axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)


def build_bow_lr() -> Pipeline:
    """Мешок слов (1-2 граммы) + логрег — простой лексический baseline."""
    return Pipeline([
        ("bow", CountVectorizer(max_features=20000, ngram_range=(1, 2),
                                token_pattern=r"(?u)\b\w+\b")),
        ("scaler", MaxAbsScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
    ])
