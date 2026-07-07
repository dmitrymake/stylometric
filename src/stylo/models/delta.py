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

    def fit(self, texts, y):
        y = np.asarray(y)
        if self.vocabulary is not None:
            self._vec = CountVectorizer(vocabulary=self.vocabulary, lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        else:
            self._vec = CountVectorizer(max_features=self.mfw_count, lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        self._vec.fit(list(texts))
        freqs = self._rel_freq(texts)
        self.mean_ = freqs.mean(axis=0)
        self.std_ = freqs.std(axis=0)
        self.std_[self.std_ == 0] = 1e-9
        z = (freqs - self.mean_) / self.std_

        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([z[y == c].mean(axis=0) for c in self.classes_])
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
