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

from .metric_contract import (
    MetricContractError,
    assert_labels_in_universe,
    author_order,
    confidence_level,
    exact_nonnegative_int,
    exact_positive_int,
    finite_statistic,
    frozen_label_order,
    integer_vector,
    metric_callable,
    paired_integer_vectors,
    probability_matrix,
    random_seed,
    rank_vector,
    unknown_prediction_policy,
)


# фиксированный seed для воспроизводимости бутстрапа
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = paired_integer_vectors(y_true, y_pred)
    if truth.size == 0:
        return 0.0
    return float(np.mean(truth == pred))


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    *,
    unknown_pred: str = "reject",
) -> float:
    truth, pred = paired_integer_vectors(y_true, y_pred)
    label_order = frozen_label_order(labels)
    unknown_pred = unknown_prediction_policy(unknown_pred)
    assert_labels_in_universe(truth, label_order, name="y_true")
    if unknown_pred == "reject":
        assert_labels_in_universe(pred, label_order, name="y_pred")
    if truth.size == 0:
        return 0.0
    return float(
        f1_score(
            truth,
            pred,
            labels=list(label_order),
            average="macro",
            zero_division=0,
        )
    )


def topk_accuracy(ranks: np.ndarray, k: int) -> float:
    """ranks — 1-based ранг истинного автора в каждой книге."""
    rank_array = rank_vector(ranks)
    k = exact_positive_int(k, name="k")
    if rank_array.size == 0:
        return 0.0
    return float(np.mean(rank_array <= k))


def per_author_recall(y_true: np.ndarray, y_pred: np.ndarray, authors: List[str]) -> Dict[str, float]:
    author_ids = author_order(authors)
    truth, pred = paired_integer_vectors(y_true, y_pred)
    label_order = tuple(range(len(author_ids)))
    assert_labels_in_universe(truth, label_order, name="y_true")
    assert_labels_in_universe(pred, label_order, name="y_pred")
    out: Dict[str, float] = {}
    for idx, author in enumerate(author_ids):
        mask = truth == idx
        out[author] = (
            float(np.mean(pred[mask] == idx)) if mask.any() else float("nan")
        )
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    n = exact_positive_int(n, name="n")
    truth, pred = paired_integer_vectors(y_true, y_pred)
    label_order = tuple(range(n))
    assert_labels_in_universe(truth, label_order, name="y_true")
    assert_labels_in_universe(pred, label_order, name="y_pred")
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(truth, pred, strict=True):
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
    stat_fn = metric_callable(stat_fn, name="stat_fn")
    n_units = exact_nonnegative_int(n_units, name="n_units")
    iters = exact_positive_int(iters, name="iters")
    level = confidence_level(level)
    seed = random_seed(seed)
    if n_units == 0:
        return CI(0.0, 0.0, 0.0)
    rng = _rng(seed)
    point = finite_statistic(
        stat_fn(np.arange(n_units)),
        name="stat_fn(full)",
    )
    boot = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n_units, size=n_units)
        boot[i] = finite_statistic(
            stat_fn(idx),
            name=f"stat_fn(bootstrap[{i}])",
        )
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return CI(float(point), float(lo), float(hi))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """ECE: средневзвешенное расхождение уверенности и точности по бинам."""
    y_true = integer_vector(y_true, name="y_true")
    n_bins = exact_positive_int(n_bins, name="n_bins")
    probs = probability_matrix(probs, n_rows=len(y_true))
    assert_labels_in_universe(
        y_true,
        tuple(range(probs.shape[1])),
        name="y_true",
    )
    if y_true.size == 0:
        return 0.0
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
    author_ids = author_order(authors)
    y_true, y_pred = paired_integer_vectors(y_true, y_pred)
    ranks = rank_vector(ranks)
    if ranks.shape != y_true.shape:
        raise MetricContractError(
            "ranks and labels must have identical 1-D shape"
        )
    labels = tuple(range(len(author_ids)))
    assert_labels_in_universe(y_true, labels, name="y_true")
    assert_labels_in_universe(y_pred, labels, name="y_pred")

    acc_ci = bootstrap_ci(lambda ix: accuracy(y_true[ix], y_pred[ix]), len(y_true), iters, level, seed)
    f1_ci = bootstrap_ci(lambda ix: macro_f1(y_true[ix], y_pred[ix], labels), len(y_true), iters, level, seed)
    top2_ci = bootstrap_ci(lambda ix: topk_accuracy(ranks[ix], 2), len(y_true), iters, level, seed)
    return {
        "n_books": int(len(y_true)),
        "accuracy": acc_ci,
        "macro_f1": f1_ci,
        "top2": top2_ci,
        "per_author_recall": per_author_recall(y_true, y_pred, list(author_ids)),
    }
