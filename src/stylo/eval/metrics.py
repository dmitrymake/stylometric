"""Strict book-level point metrics and author-clustered primary uncertainty."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from .metric_contract import (
    MetricContractError,
    assert_labels_in_universe,
    author_order,
    confidence_level as validate_confidence_level,
    exact_nonnegative_int,
    exact_positive_int,
    finite_statistic,
    frozen_label_order,
    group_vector,
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
    method: str = "iid_unit_percentile_bootstrap"

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.lo:.1%}, {self.hi:.1%}]"


@dataclass(frozen=True)
class PointEstimate:
    point: float
    uncertainty: str = "point_only"

    def __str__(self) -> str:
        return f"{self.point:.1%} (point only)"


@dataclass(frozen=True)
class AuthorClusteredInferenceSpec:
    """Frozen v1 inference contract for generic exploratory LOBO summaries."""

    schema_version: str
    primary_metric: str
    primary_uncertainty: str
    sampling_unit: str
    iterations: int
    confidence_level: float
    seed: int
    macro_f1_uncertainty: str
    top2_uncertainty: str
    self_hash: str

    @staticmethod
    def _payload(
        *,
        iterations: int,
        confidence_level_value: float,
        seed: int,
    ) -> dict[str, object]:
        return {
            "schema_version": "stylo.inference.author-clustered-accuracy.v1",
            "primary_metric": "book_level_accuracy",
            "primary_uncertainty": "author_clustered_percentile_bootstrap",
            "sampling_unit": "author",
            "iterations": iterations,
            "confidence_level": confidence_level_value,
            "seed": seed,
            "macro_f1_uncertainty": "point_only",
            "top2_uncertainty": "point_only",
        }

    @classmethod
    def build(
        cls,
        *,
        iterations: int,
        confidence_level: float,
        seed: int,
    ) -> "AuthorClusteredInferenceSpec":
        iterations = exact_positive_int(iterations, name="iterations")
        confidence_level = validate_confidence_level(
            confidence_level,
            name="confidence_level",
        )
        seed = random_seed(seed)
        payload = cls._payload(
            iterations=iterations,
            confidence_level_value=confidence_level,
            seed=seed,
        )
        self_hash = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return cls(**payload, self_hash=self_hash)

    def validate(self) -> "AuthorClusteredInferenceSpec":
        if type(self) is not AuthorClusteredInferenceSpec:
            raise MetricContractError(
                "inference_spec must be exactly AuthorClusteredInferenceSpec"
            )
        rebuilt = type(self).build(
            iterations=self.iterations,
            confidence_level=self.confidence_level,
            seed=self.seed,
        )
        if self != rebuilt:
            raise MetricContractError(
                "inference_spec is noncanonical or has a self-hash mismatch"
            )
        return self

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            **self._payload(
                iterations=self.iterations,
                confidence_level_value=self.confidence_level,
                seed=self.seed,
            ),
            "self_hash": self.self_hash,
        }


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
    level = validate_confidence_level(level)
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


def author_clustered_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    groups: Sequence[object] | np.ndarray,
    *,
    inference_spec: AuthorClusteredInferenceSpec,
) -> CI:
    """Percentile CI whose draws resample authors with all their books."""

    stat_fn = metric_callable(stat_fn, name="stat_fn")
    group_array = group_vector(groups, name="book_authors")
    spec = inference_spec.validate()
    n_books = len(group_array)
    if n_books == 0:
        return CI(
            0.0,
            0.0,
            0.0,
            method=spec.primary_uncertainty,
        )
    unique_groups = np.unique(group_array)
    by_group = {
        group: np.flatnonzero(group_array == group)
        for group in unique_groups
    }
    full = np.arange(n_books)
    point = finite_statistic(stat_fn(full), name="stat_fn(full)")
    rng = _rng(spec.seed)
    boot = np.empty(spec.iterations, dtype=np.float64)
    for index in range(spec.iterations):
        draws = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        book_indexes = np.concatenate([by_group[group] for group in draws])
        boot[index] = finite_statistic(
            stat_fn(book_indexes),
            name=f"stat_fn(bootstrap[{index}])",
        )
    alpha = (1.0 - spec.confidence_level) / 2.0
    lo, hi = np.percentile(
        boot,
        [100 * alpha, 100 * (1.0 - alpha)],
    )
    return CI(
        float(point),
        float(lo),
        float(hi),
        method=spec.primary_uncertainty,
    )


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
    probability_class_order: Sequence[str],
    *,
    metric_label_order: Sequence[int],
    book_authors: Sequence[object] | np.ndarray,
    inference_spec: AuthorClusteredInferenceSpec,
) -> Dict[str, object]:
    """Summarize one frozen fold manifest without deriving metric labels."""

    author_ids = author_order(probability_class_order)
    y_true, y_pred = paired_integer_vectors(y_true, y_pred)
    ranks = rank_vector(ranks)
    if ranks.shape != y_true.shape:
        raise MetricContractError(
            "ranks and labels must have identical 1-D shape"
        )
    probability_labels = tuple(range(len(author_ids)))
    metric_labels = frozen_label_order(metric_label_order)
    if (
        any(label not in probability_labels for label in metric_labels)
        or tuple(
            label
            for label in probability_labels
            if label in frozenset(metric_labels)
        )
        != metric_labels
    ):
        raise MetricContractError(
            "metric_label_order must be an ordered subset of "
            "probability_class_order"
        )
    assert_labels_in_universe(y_true, metric_labels, name="y_true")
    assert_labels_in_universe(
        y_pred,
        probability_labels,
        name="y_pred",
    )
    if np.any(ranks > len(probability_labels)):
        raise MetricContractError(
            "ranks exceed the probability class universe"
        )
    groups = group_vector(book_authors, name="book_authors")
    if groups.shape != y_true.shape:
        raise MetricContractError(
            "book_authors and labels must have identical 1-D shape"
        )
    expected_groups = np.asarray(
        [author_ids[int(label)] for label in y_true],
        dtype=object,
    )
    if not np.array_equal(groups, expected_groups):
        raise MetricContractError(
            "book_authors must match probability_class_order[y_true]"
        )
    spec = inference_spec.validate()

    acc_ci = author_clustered_bootstrap_ci(
        lambda indexes: accuracy(y_true[indexes], y_pred[indexes]),
        groups,
        inference_spec=spec,
    )
    f1_point = PointEstimate(
        macro_f1(
            y_true,
            y_pred,
            metric_labels,
            unknown_pred="count_as_error",
        )
    )
    top2_point = PointEstimate(topk_accuracy(ranks, 2))
    return {
        "n_books": int(len(y_true)),
        "accuracy": acc_ci,
        "macro_f1": f1_point,
        "top2": top2_point,
        "per_author_recall": per_author_recall(y_true, y_pred, list(author_ids)),
        "probability_class_order": author_ids,
        "metric_label_order": metric_labels,
        "inference_spec": spec.as_dict(),
    }
