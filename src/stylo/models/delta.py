"""Burrows's Delta (классический стилометрический baseline).

Delta считается КАК В КЛАССИКЕ:
  1. N самых частотных слов (MFW) корпуса;
  2. относительная частота каждого MFW в тексте;
  3. z-нормировка по статистикам TRAIN (Burrows z-scores);
  4. профиль автора = средний z по его текстам;
  5. Delta(text, author) = средняя |z_text - z_author| (Manhattan) — меньше = ближе.
     (cosine — вариант Smith–Aldridge.)

Принимает сырые тексты (сам строит MFW-словарь на train). Leakage-free: vocab и
статистики z берутся ТОЛЬКО из train.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances

_TOKEN_PATTERN = r"(?u)\b\w+\b"


class BurrowsDelta:
    needs_groups = True

    def __init__(self, mfw_count: int = 300, metric: str = "manhattan",
                 vocabulary: List[str] | None = None):
        assert metric in {"manhattan", "cosine"}
        self.mfw_count = mfw_count
        self.metric = metric
        # если задан фиксированный словарь (напр. служебные слова) — Delta считается ТОЛЬКО по нему
        # (тема-нейтральный вариант Delta); иначе — классические top-mfw_count слов корпуса.
        self.vocabulary = list(vocabulary) if vocabulary is not None else None
        self._vec: CountVectorizer | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def _rel_freq(self, texts) -> np.ndarray:
        counts = self._vec.transform(list(texts)).toarray().astype(np.float64)
        totals = counts.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return counts / totals

    def fit(self, texts, y, groups=None):
        texts = list(texts)
        y = np.asarray(y)
        if len(texts) != len(y):
            raise ValueError("texts and y must have the same length")
        if self.vocabulary is not None:
            self._vec = CountVectorizer(vocabulary=self.vocabulary, lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        else:
            self._vec = CountVectorizer(max_features=self.mfw_count, lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        self._vec.fit(texts)
        freqs = self._rel_freq(texts)
        group_freqs, group_y = _group_means(freqs, y, groups)
        self.mean_ = group_freqs.mean(axis=0)
        self.std_ = group_freqs.std(axis=0)
        self.std_[self.std_ == 0] = 1e-9
        group_z = (group_freqs - self.mean_) / self.std_
        if self.metric == "cosine" and groups is not None:
            group_z = group_z / (
                np.linalg.norm(group_z, axis=1, keepdims=True) + 1e-12
            )

        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack(
            [group_z[group_y == c].mean(axis=0) for c in self.classes_]
        )
        self.group_weighting_ = (
            (
                "equal_group_direction_after_within_group_mean_l2"
                if self.metric == "cosine"
                else "equal_group_after_within_group_mean"
            )
            if groups is not None
            else "equal_row"
        )
        return self

    def _z(self, texts) -> np.ndarray:
        return (self._rel_freq(texts) - self.mean_) / self.std_

    def distances(self, texts) -> np.ndarray:
        """Матрица расстояний (n_texts, n_classes); меньше = ближе."""
        z = self._z(texts)
        if self.metric == "manhattan":
            # классическая Delta = средняя |z|; делим на число признаков для масштаба
            return manhattan_distances(z, self.centroids_) / z.shape[1]
        return cosine_distances(z, self.centroids_)

    def predict(self, texts) -> np.ndarray:
        d = self.distances(texts)
        return self.classes_[np.argmin(d, axis=1)]

    def predict_proba(self, texts) -> np.ndarray:
        """Псевдо-вероятности softmax(-distance) — для ансамбля/рангов."""
        d = self.distances(texts)
        x = -d
        x = x - x.max(axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)

    def feature_names(self) -> List[str]:
        return list(self._vec.get_feature_names_out()) if self._vec else []


def _group_means(values: np.ndarray, y: np.ndarray, groups):
    """Collapse chunks to equal-weight work rows, validating one label per work."""
    if groups is None:
        return values, y
    groups = list(groups)
    if len(groups) != len(y):
        raise ValueError("groups and y must have the same length")
    order = list(dict.fromkeys(groups))
    means = []
    labels = []
    for group in order:
        idx = np.flatnonzero([item == group for item in groups])
        group_labels = np.unique(y[idx])
        if len(group_labels) != 1:
            raise ValueError(f"group {group!r} spans multiple classes")
        means.append(values[idx].mean(axis=0))
        labels.append(group_labels[0])
    return np.vstack(means), np.asarray(labels, dtype=y.dtype)
