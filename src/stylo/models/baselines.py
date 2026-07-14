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

from ..eval.work_weighting import (CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                                   resolve_training_weighting)


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

    needs_groups = True
    ARTIFACT_SCHEMA_VERSION = 2         # v2 adds training_weighting/_wv; a versionless pickle is v1

    def __init__(self, ngram_range=(3, 3), max_features=5000, min_df=3,
                 training_weighting: str = CHUNK_WEIGHTED_LEGACY):
        self._schema_version = self.ARTIFACT_SCHEMA_VERSION
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.training_weighting = resolve_training_weighting(training_weighting)
        self.classes_ = None
        self._vec = None
        self._wv = None                       # WorkLevelVectorizer в work_balanced-ветке
        self._centroids = None

    def __setstate__(self, state):
        # migrate pre-B2 pickles (schema v1, versionless): fill attributes introduced in B2
        self.__dict__.update(state)
        sv = self.__dict__.setdefault("_schema_version", 1)
        if isinstance(sv, bool) or not isinstance(sv, int) or not (1 <= sv <= self.ARTIFACT_SCHEMA_VERSION):
            raise ValueError(f"invalid CharCosineBaseline artifact schema version {sv!r}")
        self.__dict__.setdefault("training_weighting", CHUNK_WEIGHTED_LEGACY)
        self.__dict__.setdefault("_wv", None)

    def fit(self, texts, y, groups=None):
        texts = list(texts)
        y = np.asarray(y)
        if len(texts) != len(y):
            raise ValueError("texts and y must have the same length")
        if self.training_weighting == WORK_BALANCED:
            if groups is None:
                raise ValueError("work_balanced char_cos needs per-chunk groups")
            from ..features.work_vectorizer import WorkLevelVectorizer, MODE_TFIDF
            self._wv = WorkLevelVectorizer(
                analyzer_params={"analyzer": "char", "ngram_range": self.ngram_range, "lowercase": True},
                mode=MODE_TFIDF, max_features=self.max_features, min_df_works=2, sublinear_tf=True,
            )
            self._wv.fit(texts, groups)
            X = self._wv.transform(texts)
            self._vec = None
        else:
            self._vec = TfidfVectorizer(analyzer="char", ngram_range=self.ngram_range,
                                        max_features=self.max_features, min_df=self.min_df,
                                        sublinear_tf=True)
            X = self._vec.fit_transform(texts)
            self._wv = None                   # clear any prior WB state on a legacy (re)fit
        self.classes_ = np.unique(y)
        group_X, group_y = _sparse_group_means(X, y, groups)
        self._centroids = np.vstack(
            [
                np.asarray(group_X[group_y == c].mean(axis=0)).ravel()
                for c in self.classes_
            ]
        )
        self.group_weighting_ = (
            ("work_balanced_tfidf_direction_after_within_group_mean_l2"
             if self.training_weighting == WORK_BALANCED
             else "equal_group_direction_after_within_group_mean_l2")
            if groups is not None
            else "equal_row"
        )
        return self

    def predict_proba(self, texts):
        wv = getattr(self, "_wv", None)          # getattr: pre-B2 pickles have no _wv
        vec = wv if wv is not None else self._vec
        X = vec.transform(list(texts))
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


def _sparse_group_means(X, y: np.ndarray, groups):
    """Return dense equal-weight work rows for a sparse chunk matrix."""
    if groups is None:
        return X, y
    groups = list(groups)
    if len(groups) != len(y):
        raise ValueError("groups and y must have the same length")
    rows = []
    labels = []
    for group in dict.fromkeys(groups):
        idx = np.flatnonzero([item == group for item in groups])
        group_labels = np.unique(y[idx])
        if len(group_labels) != 1:
            raise ValueError(f"group {group!r} spans multiple classes")
        row = np.asarray(X[idx].mean(axis=0)).ravel()
        rows.append(row / (np.linalg.norm(row) + 1e-12))
        labels.append(group_labels[0])
    return np.vstack(rows), np.asarray(labels, dtype=y.dtype)
