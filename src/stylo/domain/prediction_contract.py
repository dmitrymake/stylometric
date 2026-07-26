"""Fail-closed contracts for class-indexed scores and deployment predictions."""
from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PREDICTION_CONTRACT_VERSION = "stylo.prediction_contract.v2"
PROBABILITY_SUM_ATOL = 1e-6


class PredictionContractError(ValueError):
    """A class universe or numeric prediction payload is malformed."""


@dataclass(frozen=True)
class PredictionDecision:
    """Stable top-1 and conservative true-class rank."""

    top1: int
    true_rank: int
    order: tuple[int, ...]


def validate_author_universe(authors: object) -> tuple[str, ...]:
    if (
        type(authors) is not list
        or not authors
        or any(type(author) is not str or not author.strip() for author in authors)
        or len(set(authors)) != len(authors)
    ):
        raise PredictionContractError(
            "authors must be a nonempty ordered list of unique nonblank strings"
        )
    return tuple(authors)


def validate_class_indices(
    classes: object,
    n_classes: int,
    *,
    name: str = "classes_",
) -> np.ndarray:
    if type(n_classes) is not int or n_classes < 1:
        raise PredictionContractError("n_classes must be an exact positive integer")
    arr = np.asarray(classes)
    if arr.ndim != 1:
        raise PredictionContractError(f"{name} must be one-dimensional")
    if arr.dtype.kind not in "iu" or arr.dtype.kind == "b":
        raise PredictionContractError(f"{name} must contain integer class indices")
    values = arr.astype(np.int64, copy=False)
    expected = np.arange(n_classes, dtype=np.int64)
    if not np.array_equal(values, expected):
        raise PredictionContractError(
            f"{name} must equal the complete ordered universe "
            f"{expected.tolist()}, got {values.tolist()}"
        )
    return values


def _finite_matrix(
    values: object,
    *,
    rows: int | None,
    cols: int,
    name: str,
) -> np.ndarray:
    object_view = np.asarray(values, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in object_view.flat):
        raise PredictionContractError(f"{name} must not contain booleans")
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PredictionContractError(f"{name} must be a numeric matrix") from exc
    if arr.ndim != 2:
        raise PredictionContractError(f"{name} must be two-dimensional")
    if rows is not None and arr.shape[0] != rows:
        raise PredictionContractError(
            f"{name} row count {arr.shape[0]} != expected {rows}"
        )
    if arr.shape[1] != cols:
        raise PredictionContractError(
            f"{name} width {arr.shape[1]} != class universe width {cols}"
        )
    if not np.isfinite(arr).all():
        raise PredictionContractError(f"{name} contains NaN or infinity")
    if np.any(arr <= -1.0e8):
        raise PredictionContractError(
            f"{name} contains a missing-class sentinel instead of a real score"
        )
    return arr


def validate_score_matrix(
    scores: object,
    *,
    rows: int | None,
    n_classes: int,
    name: str = "scores",
) -> np.ndarray:
    """Validate finite, class-complete decision/logit scores."""

    return _finite_matrix(scores, rows=rows, cols=n_classes, name=name)


def validate_probabilities(
    probabilities: object,
    *,
    rows: int | None,
    n_classes: int,
    name: str = "probabilities",
    atol: float = 1e-8,
) -> np.ndarray:
    arr = _finite_matrix(
        probabilities, rows=rows, cols=n_classes, name=name
    )
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise PredictionContractError(f"{name} must be within [0, 1]")
    if not np.allclose(arr.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise PredictionContractError(f"{name} rows must sum to one")
    return arr


def validate_distances(
    distances: object,
    *,
    rows: int | None,
    n_classes: int,
    name: str = "distances",
) -> np.ndarray:
    arr = _finite_matrix(distances, rows=rows, cols=n_classes, name=name)
    if np.any(arr < 0.0):
        raise PredictionContractError(f"{name} must be nonnegative")
    return arr


def validate_channel_mapping(
    channels: object,
    *,
    n_classes: int | None = None,
    rows: int | None = None,
    name: str = "channels",
) -> tuple[dict[str, np.ndarray], int, int]:
    """Validate a nonempty mapping of equally shaped finite score matrices."""

    if type(channels) is not dict or not channels:
        raise PredictionContractError(f"{name} must be a nonempty dict")
    out: dict[str, np.ndarray] = {}
    inferred_rows = rows
    inferred_classes = n_classes
    for channel, values in channels.items():
        if type(channel) is not str or not channel:
            raise PredictionContractError(f"{name} keys must be nonempty strings")
        raw = np.asarray(values)
        if raw.ndim != 2:
            raise PredictionContractError(f"{name}[{channel!r}] must be 2D")
        if inferred_rows is None:
            inferred_rows = int(raw.shape[0])
        if inferred_classes is None:
            inferred_classes = int(raw.shape[1])
        out[channel] = validate_score_matrix(
            values,
            rows=inferred_rows,
            n_classes=inferred_classes,
            name=f"{name}[{channel!r}]",
        )
    assert inferred_rows is not None and inferred_classes is not None
    if inferred_classes < 2:
        raise PredictionContractError("score matrices need at least two classes")
    return out, inferred_rows, inferred_classes


def validate_class_order(classes: object, *, n_classes: int) -> np.ndarray:
    """Compatibility spelling for the canonical exact class-index contract."""

    return validate_class_indices(classes, n_classes)


def validate_probability_matrix(
    probabilities: object,
    classes: object,
    *,
    n_classes: int,
    n_rows: int,
) -> np.ndarray:
    validate_class_indices(classes, n_classes)
    return validate_probabilities(
        probabilities,
        rows=n_rows,
        n_classes=n_classes,
        name="predict_proba",
        atol=PROBABILITY_SUM_ATOL,
    )


def validate_probability_vector(
    probabilities: Sequence[float],
    *,
    expected_width: int | None = None,
) -> np.ndarray:
    object_view = np.asarray(probabilities, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in object_view.flat):
        raise PredictionContractError("probabilities must not contain booleans")
    try:
        vector = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PredictionContractError(
            "probabilities must be a real numeric vector"
        ) from exc
    if vector.ndim != 1 or len(vector) == 0:
        raise PredictionContractError(
            "probabilities must be a non-empty one-dimensional vector"
        )
    if expected_width is not None:
        if (
            isinstance(expected_width, bool)
            or not isinstance(expected_width, numbers.Integral)
            or int(expected_width) < 1
        ):
            raise PredictionContractError(
                "expected_width must be a positive integer"
            )
        if len(vector) != int(expected_width):
            raise PredictionContractError(
                f"probability width {len(vector)} != {int(expected_width)}"
            )
    if not np.isfinite(vector).all():
        raise PredictionContractError("probabilities must contain only finite values")
    if (vector < 0.0).any() or (vector > 1.0).any():
        raise PredictionContractError("probabilities must be in [0,1]")
    total = float(vector.sum())
    if not np.isclose(total, 1.0, atol=PROBABILITY_SUM_ATOL, rtol=0.0):
        raise PredictionContractError(
            f"probabilities must sum to 1 (got {total!r})"
        )
    return vector


def stable_top1_and_worst_tie_rank(
    probabilities: Sequence[float],
    *,
    true_label: int,
    expected_width: int | None = None,
) -> PredictionDecision:
    """Use lowest-index top-1 and the worst rank within a probability tie."""

    vector = validate_probability_vector(
        probabilities, expected_width=expected_width
    )
    if isinstance(true_label, bool) or not isinstance(
        true_label, numbers.Integral
    ):
        raise PredictionContractError("true_label must be a non-bool integer")
    true_label = int(true_label)
    if not 0 <= true_label < len(vector):
        raise PredictionContractError(f"true_label must be in [0,{len(vector)})")
    order = np.argsort(-vector, kind="stable")
    return PredictionDecision(
        top1=int(order[0]),
        true_rank=int(np.count_nonzero(vector >= vector[true_label])),
        order=tuple(int(index) for index in order),
    )


def validate_prediction_record(
    *,
    probabilities: Sequence[float],
    pred_label: int,
    true_label: int,
    correct: bool,
    rank: int,
    expected_width: int | None = None,
) -> PredictionDecision:
    decision = stable_top1_and_worst_tie_rank(
        probabilities, true_label=true_label, expected_width=expected_width
    )
    width = len(decision.order)
    if isinstance(pred_label, bool) or not isinstance(
        pred_label, numbers.Integral
    ):
        raise PredictionContractError("pred_label must be a non-bool integer")
    pred_label = int(pred_label)
    if not 0 <= pred_label < width:
        raise PredictionContractError(f"pred_label must be in [0,{width})")
    if pred_label != decision.top1:
        raise PredictionContractError(
            f"pred_label {pred_label} != stable top-1 {decision.top1}"
        )
    if type(correct) is not bool:
        raise PredictionContractError("correct must be a bool")
    if correct != (pred_label == int(true_label)):
        raise PredictionContractError(
            "correct must equal (pred_label == true_label)"
        )
    if isinstance(rank, bool) or not isinstance(rank, numbers.Integral):
        raise PredictionContractError("rank must be a non-bool integer")
    if int(rank) != decision.true_rank:
        raise PredictionContractError(
            f"rank {int(rank)} != worst-tie true rank {decision.true_rank}"
        )
    return decision


__all__ = [
    "PREDICTION_CONTRACT_VERSION",
    "PROBABILITY_SUM_ATOL",
    "PredictionContractError",
    "PredictionDecision",
    "stable_top1_and_worst_tie_rank",
    "validate_author_universe",
    "validate_channel_mapping",
    "validate_class_indices",
    "validate_class_order",
    "validate_distances",
    "validate_prediction_record",
    "validate_probabilities",
    "validate_probability_matrix",
    "validate_probability_vector",
    "validate_score_matrix",
]
