"""Метрики атрибуции с доверительными интервалами (bootstrap по КНИГАМ).

Единица оценки и ресэмплинга — КНИГА (одна тестовая книга = один результат LOBO).
Бутстрап по книгам, а не по чанкам: чанки внутри книги зависимы, ресэмпл по чанкам
занижал бы CI в разы.

При дисбалансе 27× accuracy вводит в заблуждение, поэтому всегда считаем и macro-F1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score


# фиксированный seed для воспроизводимости бутстрапа
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def topk_accuracy(ranks: np.ndarray, k: int) -> float:
    """ranks — 1-based ранг истинного автора в каждой книге."""
    if len(ranks) == 0:
        return 0.0
    return float(np.mean(np.asarray(ranks) <= k))


def per_author_recall(y_true: np.ndarray, y_pred: np.ndarray, authors: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for idx, a in enumerate(authors):
        mask = y_true == idx
        out[a] = float(np.mean(y_pred[mask] == idx)) if mask.any() else float("nan")
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p)] += 1
    return m


@dataclass
class CI:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.lo:.1%}, {self.hi:.1%}]"


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    n_units: int,
    iters: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> CI:
    """Бутстрап-CI. stat_fn получает массив индексов книг (ресэмпл с возвратом)
    и возвращает скаляр метрики на этой подвыборке."""
    if n_units == 0:
        return CI(0.0, 0.0, 0.0)
    rng = _rng(seed)
    point = stat_fn(np.arange(n_units))
    boot = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n_units, size=n_units)
        boot[i] = stat_fn(idx)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return CI(float(point), float(lo), float(hi))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """ECE: средневзвешенное расхождение уверенности и точности по бинам."""
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.any():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def summarize_book_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ranks: np.ndarray,
    authors: List[str],
    iters: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> Dict[str, object]:
    """Сводка метрик по книгам с CI. y_*/ranks — на уровне книг."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ranks = np.asarray(ranks)
    # macro-F1 — по тестированным классам: метки в y_true ∪ y_pred (sklearn-дефолт).
    # В LOBO single-book авторы не тестируются (run_fold → None) и в y_true не входят.
    labels = np.unique(np.concatenate([y_true, y_pred])).tolist()

    acc_ci = bootstrap_ci(lambda ix: accuracy(y_true[ix], y_pred[ix]), len(y_true), iters, level, seed)
    f1_ci = bootstrap_ci(lambda ix: macro_f1(y_true[ix], y_pred[ix], labels), len(y_true), iters, level, seed)
    top2_ci = bootstrap_ci(lambda ix: topk_accuracy(ranks[ix], 2), len(y_true), iters, level, seed)
    return {
        "n_books": int(len(y_true)),
        "accuracy": acc_ci,
        "macro_f1": f1_ci,
        "top2": top2_ci,
        "per_author_recall": per_author_recall(y_true, y_pred, authors),
    }
