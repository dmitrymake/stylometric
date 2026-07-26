"""Strict, shared input contracts for generic metrics and significance helpers."""
from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence

import numpy as np


class MetricContractError(ValueError):
    """Metric inputs are malformed or do not match the frozen estimand."""


def exact_nonnegative_int(value, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise MetricContractError(f"{name} must be an exact non-negative int")
    return value


def exact_positive_int(value, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise MetricContractError(f"{name} must be an exact positive int")
    return value


def confidence_level(value, *, name: str = "level") -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        raise MetricContractError(
            f"{name} must be an exact finite float in (0, 1)"
        )
    return value


def random_seed(value, *, name: str = "seed") -> int:
    if type(value) is not int:
        raise MetricContractError(f"{name} must be an exact int")
    return value


def integer_vector(values, *, name: str, allow_empty: bool = True) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise MetricContractError(f"{name} must be exactly 1-D")
    if not allow_empty and array.size == 0:
        raise MetricContractError(f"{name} must not be empty")
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Integral)
        for value in raw.tolist()
    ):
        raise MetricContractError(
            f"{name} must contain only non-bool integer labels"
        )
    return np.asarray([int(value) for value in raw.tolist()], dtype=np.int64)


def boolean_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise MetricContractError(f"{name} must be exactly 1-D")
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or any(type(value) not in (bool, np.bool_) for value in raw.tolist()):
        raise MetricContractError(f"{name} must contain exact bool values")
    return np.asarray(raw, dtype=bool)


def paired_integer_vectors(
    left,
    right,
    *,
    left_name: str = "y_true",
    right_name: str = "y_pred",
) -> tuple[np.ndarray, np.ndarray]:
    left_array = integer_vector(left, name=left_name)
    right_array = integer_vector(right, name=right_name)
    if left_array.shape != right_array.shape:
        raise MetricContractError(
            f"{left_name} and {right_name} must have identical 1-D shape"
        )
    return left_array, right_array


def paired_boolean_vectors(
    left,
    right,
    *,
    left_name: str = "correct_a",
    right_name: str = "correct_b",
) -> tuple[np.ndarray, np.ndarray]:
    left_array = boolean_vector(left, name=left_name)
    right_array = boolean_vector(right, name=right_name)
    if left_array.shape != right_array.shape:
        raise MetricContractError(
            f"{left_name} and {right_name} must have identical 1-D shape"
        )
    return left_array, right_array


def frozen_label_order(labels: Sequence[int]) -> tuple[int, ...]:
    if isinstance(labels, (str, bytes)):
        raise MetricContractError("labels must be an ordered integer sequence")
    label_array = integer_vector(labels, name="labels", allow_empty=False)
    order = tuple(int(label) for label in label_array.tolist())
    if len(set(order)) != len(order):
        raise MetricContractError("labels must be unique")
    return order


def assert_labels_in_universe(
    values: np.ndarray,
    labels: Sequence[int],
    *,
    name: str,
) -> None:
    universe = frozenset(labels)
    unknown = sorted(set(values.tolist()) - universe)
    if unknown:
        raise MetricContractError(
            f"{name} contains labels outside the frozen metric order: {unknown}"
        )


def unknown_prediction_policy(value: str) -> str:
    if type(value) is not str or value not in {"reject", "count_as_error"}:
        raise MetricContractError(
            "unknown_prediction_policy must be 'reject' or 'count_as_error'"
        )
    return value


def rank_vector(values, *, name: str = "ranks") -> np.ndarray:
    ranks = integer_vector(values, name=name)
    if np.any(ranks < 1):
        raise MetricContractError(f"{name} must contain 1-based positive ranks")
    return ranks


def author_order(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise MetricContractError("authors must be an ordered sequence")
    authors = tuple(values)
    if not authors or any(type(author) is not str or not author for author in authors):
        raise MetricContractError("authors must contain non-empty exact strings")
    if len(set(authors)) != len(authors):
        raise MetricContractError("authors must be unique")
    return authors


def probability_matrix(
    probabilities,
    *,
    n_rows: int,
    name: str = "probs",
) -> np.ndarray:
    try:
        matrix = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MetricContractError(f"{name} must be a rectangular numeric matrix") from exc
    raw = np.asarray(probabilities, dtype=object)
    if matrix.ndim != 2 or raw.ndim != 2:
        raise MetricContractError(f"{name} must be exactly 2-D")
    if matrix.shape[0] != n_rows or matrix.shape[1] < 1:
        raise MetricContractError(
            f"{name} shape must be ({n_rows}, n_classes>=1)"
        )
    if any(type(value) in (bool, np.bool_) for value in raw.ravel().tolist()):
        raise MetricContractError(f"{name} must not contain bool values")
    if not np.all(np.isfinite(matrix)):
        raise MetricContractError(f"{name} must contain only finite values")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise MetricContractError(f"{name} values must lie in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise MetricContractError(f"{name} rows must sum to one")
    return matrix


def group_vector(values, *, name: str = "groups") -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise MetricContractError(f"{name} must be exactly 1-D")
    if any(
        (
            type(value) is not str
            or not value
        )
        and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Integral)
        )
        for value in array.tolist()
    ):
        raise MetricContractError(
            f"{name} must contain non-empty exact strings or non-bool integers"
        )
    return array


def finite_statistic(value, *, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise MetricContractError(f"{name} must return one finite real scalar")
    return float(value)


def metric_callable(value, *, name: str) -> Callable[[np.ndarray], float]:
    if not callable(value):
        raise MetricContractError(f"{name} must be callable")
    return value


__all__ = [
    "MetricContractError",
    "assert_labels_in_universe",
    "author_order",
    "boolean_vector",
    "confidence_level",
    "exact_nonnegative_int",
    "exact_positive_int",
    "finite_statistic",
    "frozen_label_order",
    "group_vector",
    "integer_vector",
    "metric_callable",
    "paired_boolean_vectors",
    "paired_integer_vectors",
    "probability_matrix",
    "random_seed",
    "rank_vector",
    "unknown_prediction_policy",
]
