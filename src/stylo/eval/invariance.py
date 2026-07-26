"""Source/edition invariance evaluation without model fitting.

The module deliberately separates two concerns:

* :func:`build_leave_one_factor_level_out` produces train/test indices that a
  caller may use to fit a model.  It never accepts or fits an estimator.
* :func:`evaluate_predictions` consumes already-produced, sample-aligned
  predictions.  It cannot accidentally expose test samples to ``fit`` because
  there is no model-facing API in the evaluator.

Metadata may be a mapping of columns, a pandas-like frame, or a sequence of
row mappings.  Typical columns are ``author``, ``work``, ``topic``, ``genre``,
``period``, ``source`` and ``edition``; any scalar metadata column can be used
as the held-out factor.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


MISSING_LEVEL = "<MISSING>"
DEFAULT_INVARIANCE_FACTORS = ("source", "edition")


@dataclasses.dataclass(frozen=True)
class PlanDiagnostics:
    """Dataset-level diagnostics for one held-out factor."""

    n_samples: int
    n_levels: int
    n_authors: int
    n_authors_spanning_levels: int
    author_level_overlap: float
    missing_factor_samples: int
    author_factor_confounded: bool
    messages: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SplitDiagnostics:
    """Feasibility and confounding diagnostics for one held-out level."""

    possible: bool
    confounded: bool
    reasons: Tuple[str, ...]
    n_train: int
    n_test: int
    n_train_labels: Optional[int]
    n_test_labels: Optional[int]
    labels_absent_from_train: Tuple[Any, ...]
    authors_absent_from_train: Tuple[Any, ...]
    test_author_overlap: float


@dataclasses.dataclass(frozen=True)
class FactorSplit:
    """One leave-one-factor-level-out split."""

    factor: str
    level: Any
    train_idx: np.ndarray
    test_idx: np.ndarray
    diagnostics: SplitDiagnostics


@dataclasses.dataclass(frozen=True)
class FactorSplitPlan:
    """All leave-one-level-out splits for a factor."""

    factor: str
    splits: Tuple[FactorSplit, ...]
    diagnostics: PlanDiagnostics

    def by_level(self) -> Dict[Any, FactorSplit]:
        return {split.level: split for split in self.splits}


@dataclasses.dataclass(frozen=True)
class PurgedFactorSplit:
    """Held-out factor×work cell with both marginals purged from training."""

    factor: str
    level: Any
    group_field: str
    group: Any
    train_idx: np.ndarray
    test_idx: np.ndarray
    purged_idx: np.ndarray
    diagnostics: SplitDiagnostics


@dataclasses.dataclass(frozen=True)
class PurgedPlanDiagnostics:
    n_samples: int
    n_levels: int
    n_groups: int
    n_splits: int
    test_coverage: float
    possible_split_coverage: float
    messages: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PurgedFactorPlan:
    factor: str
    group_field: str
    splits: Tuple[PurgedFactorSplit, ...]
    diagnostics: PurgedPlanDiagnostics

    def by_cell(self) -> Dict[tuple[Any, Any], PurgedFactorSplit]:
        return {(split.level, split.group): split for split in self.splits}


@dataclasses.dataclass(frozen=True)
class MetricEstimate:
    """Point estimate and percentile cluster-bootstrap interval."""

    point: Optional[float]
    lo: Optional[float]
    hi: Optional[float]


@dataclasses.dataclass(frozen=True)
class PredictionMetrics:
    """Metrics for an overall evaluation or one factor slice."""

    n_total: int
    n_evaluated: int
    coverage: float
    n_labels: int
    n_clusters: int
    accuracy: MetricEstimate
    macro_f1: MetricEstimate


@dataclasses.dataclass(frozen=True)
class FactorSliceEvaluation:
    factor: str
    level: Any
    metrics: PredictionMetrics
    split_diagnostics: SplitDiagnostics


@dataclasses.dataclass(frozen=True)
class FactorEvaluation:
    factor: str
    overall: PredictionMetrics
    slices: Tuple[FactorSliceEvaluation, ...]
    worst_group_accuracy: Optional[float]
    worst_group_level: Optional[Any]
    possible_split_coverage: float
    unconfounded_split_coverage: float
    diagnostics: PlanDiagnostics


@dataclasses.dataclass(frozen=True)
class InvarianceReport:
    """Overall metrics plus source/edition (or other factor) evaluations."""

    overall: PredictionMetrics
    factors: Mapping[str, FactorEvaluation]
    cluster_fields: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _hashable_scalar(value: Any, *, missing_marker: bool = False) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if missing_marker and _is_missing(value):
        return MISSING_LEVEL
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"metadata values must be scalar/hashable, got {value!r}") from exc
    return value


def _column(metadata: Any, name: str, expected_len: Optional[int] = None) -> np.ndarray:
    """Read one metadata column from common table-like representations."""
    values: Any
    if isinstance(metadata, Mapping):
        if name not in metadata:
            raise KeyError(f"metadata has no column {name!r}")
        values = metadata[name]
    elif hasattr(metadata, "columns") and name in metadata.columns:
        values = metadata[name]
    elif isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        rows = list(metadata)
        if rows and not all(isinstance(row, Mapping) for row in rows):
            raise TypeError("row-oriented metadata must be a sequence of mappings")
        try:
            values = [row[name] for row in rows]
        except KeyError as exc:
            raise KeyError(f"metadata row has no column {name!r}") from exc
    else:
        raise TypeError("metadata must be a column mapping, pandas-like frame, or rows")

    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"metadata column {name!r} must be one-dimensional")
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(
            f"metadata column {name!r} has length {len(arr)}, expected {expected_len}"
        )
    return arr


def _normalised_column(metadata: Any, name: str, expected_len: Optional[int] = None) -> np.ndarray:
    arr = _column(metadata, name, expected_len)
    return np.asarray(
        [_hashable_scalar(v, missing_marker=True) for v in arr], dtype=object
    )


def _stable_unique(values: Iterable[Any]) -> Tuple[Any, ...]:
    out = []
    seen = set()
    for raw in values:
        value = _hashable_scalar(raw)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def factor_slice_indices(metadata: Any, factor: str) -> Dict[Any, np.ndarray]:
    """Return sample indices for every level of ``factor``."""
    values = _normalised_column(metadata, factor)
    return {level: np.flatnonzero(values == level) for level in _stable_unique(values)}


def build_leave_one_factor_level_out(
    metadata: Any,
    factor: str,
    y_true: Optional[Sequence[Any]] = None,
    *,
    author_field: str = "author",
    min_train_classes: int = 2,
) -> FactorSplitPlan:
    """Build leave-one-factor-level-out indices and diagnose invalid splits.

    A split is computationally ``possible`` only when train/test are non-empty,
    train contains at least ``min_train_classes`` labels, and every test label is
    represented in train.  ``confounded`` is separate: a split may be computable
    while some held-out authors have no training work, which means factor and
    authorship effects cannot be cleanly separated.
    """
    if min_train_classes < 1:
        raise ValueError("min_train_classes must be >= 1")

    factor_values = _normalised_column(metadata, factor)
    n = len(factor_values)
    authors = _normalised_column(metadata, author_field, n)
    labels = None if y_true is None else np.asarray(y_true, dtype=object)
    if labels is not None:
        if labels.ndim != 1 or len(labels) != n:
            raise ValueError("y_true must be one-dimensional and aligned with metadata")
        if any(_is_missing(v) for v in labels):
            raise ValueError("y_true must not contain missing labels")
        labels = np.asarray([_hashable_scalar(v) for v in labels], dtype=object)

    levels = _stable_unique(factor_values)
    unique_authors = _stable_unique(authors)
    levels_by_author = {
        author: set(factor_values[authors == author].tolist()) for author in unique_authors
    }
    spanning = sum(len(v) > 1 for v in levels_by_author.values())
    author_overlap = spanning / len(unique_authors) if unique_authors else 0.0
    fully_confounded = bool(unique_authors) and spanning == 0
    missing_count = int(np.sum(factor_values == MISSING_LEVEL))

    plan_messages = []
    if len(levels) < 2:
        plan_messages.append("factor_has_fewer_than_two_levels")
    if missing_count:
        plan_messages.append("factor_contains_missing_values")
    if fully_confounded:
        plan_messages.append(f"author_fully_confounded_with_{factor}")
    elif author_overlap < 1.0:
        plan_messages.append(f"limited_author_overlap_across_{factor}_levels")

    all_idx = np.arange(n, dtype=int)
    splits = []
    for level in levels:
        test_idx = np.flatnonzero(factor_values == level)
        train_idx = all_idx[factor_values != level]
        reasons = []
        impossible = []

        if len(train_idx) == 0:
            impossible.append("empty_train")
        if len(test_idx) == 0:
            impossible.append("empty_test")
        if level == MISSING_LEVEL:
            reasons.append("held_out_level_is_missing_metadata")

        train_authors = set(authors[train_idx].tolist())
        test_authors = set(authors[test_idx].tolist())
        absent_authors = tuple(a for a in _stable_unique(authors[test_idx]) if a not in train_authors)
        author_overlap_fraction = (
            len(test_authors & train_authors) / len(test_authors) if test_authors else 0.0
        )
        if absent_authors:
            reasons.append("some_test_authors_absent_from_train")
        if test_authors and not (test_authors & train_authors):
            reasons.append("no_test_author_seen_in_train")
        if fully_confounded:
            reasons.append(f"author_fully_confounded_with_{factor}")

        n_train_labels = None
        n_test_labels = None
        absent_labels: Tuple[Any, ...] = ()
        if labels is not None:
            train_labels = set(labels[train_idx].tolist())
            test_label_values = _stable_unique(labels[test_idx])
            n_train_labels = len(train_labels)
            n_test_labels = len(test_label_values)
            absent_labels = tuple(v for v in test_label_values if v not in train_labels)
            if n_train_labels < min_train_classes:
                impossible.append("train_has_fewer_than_minimum_classes")
            if absent_labels:
                impossible.append("test_labels_absent_from_train")

        reasons.extend(impossible)
        reasons = list(dict.fromkeys(reasons))
        confounded = any(
            reason.startswith("author_fully_confounded")
            or reason in {
                "some_test_authors_absent_from_train",
                "no_test_author_seen_in_train",
                "held_out_level_is_missing_metadata",
            }
            for reason in reasons
        )
        diagnostics = SplitDiagnostics(
            possible=not impossible,
            confounded=confounded,
            reasons=tuple(reasons),
            n_train=int(len(train_idx)),
            n_test=int(len(test_idx)),
            n_train_labels=n_train_labels,
            n_test_labels=n_test_labels,
            labels_absent_from_train=absent_labels,
            authors_absent_from_train=absent_authors,
            test_author_overlap=float(author_overlap_fraction),
        )
        splits.append(
            FactorSplit(
                factor=factor,
                level=level,
                train_idx=train_idx,
                test_idx=test_idx,
                diagnostics=diagnostics,
            )
        )

    return FactorSplitPlan(
        factor=factor,
        splits=tuple(splits),
        diagnostics=PlanDiagnostics(
            n_samples=n,
            n_levels=len(levels),
            n_authors=len(unique_authors),
            n_authors_spanning_levels=spanning,
            author_level_overlap=float(author_overlap),
            missing_factor_samples=missing_count,
            author_factor_confounded=fully_confounded,
            messages=tuple(plan_messages),
        ),
    )


def build_purged_factor_group_splits(
    metadata: Any,
    factor: str,
    y_true: Sequence[Any],
    *,
    group_field: str = "work",
    author_field: str = "author",
    min_train_classes: int = 2,
) -> PurgedFactorPlan:
    """Build leakage-resistant factor×work outer splits.

    For a test cell ``factor=L AND group=G``, training contains only rows with
    ``factor!=L AND group!=G``.  Rows sharing either the held-out source/edition
    or the held-out work are purged.  This prevents a source-invariance test
    from becoming a same-work content-recognition test.
    """
    if min_train_classes < 1:
        raise ValueError("min_train_classes must be >= 1")
    factor_values = _normalised_column(metadata, factor)
    n = len(factor_values)
    groups = _normalised_column(metadata, group_field, n)
    authors = _normalised_column(metadata, author_field, n)
    labels = np.asarray(y_true, dtype=object)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("y_true must be one-dimensional and aligned with metadata")
    if any(_is_missing(value) for value in labels):
        raise ValueError("y_true must not contain missing labels")
    labels = np.asarray([_hashable_scalar(value) for value in labels], dtype=object)

    levels = _stable_unique(factor_values)
    unique_groups = _stable_unique(groups)
    all_idx = np.arange(n, dtype=int)
    splits: list[PurgedFactorSplit] = []
    test_covered = np.zeros(n, dtype=bool)
    possible_covered = np.zeros(n, dtype=bool)
    messages: list[str] = []
    if MISSING_LEVEL in levels:
        messages.append("factor_contains_missing_values")
    if MISSING_LEVEL in unique_groups:
        messages.append("group_contains_missing_values")

    for level in levels:
        cell_groups = _stable_unique(groups[factor_values == level])
        for group in cell_groups:
            test_mask = (factor_values == level) & (groups == group)
            train_mask = (factor_values != level) & (groups != group)
            test_idx = all_idx[test_mask]
            train_idx = all_idx[train_mask]
            purged_idx = all_idx[~test_mask & ~train_mask]
            test_covered[test_idx] = True

            reasons: list[str] = []
            impossible: list[str] = []
            if len(test_idx) == 0:
                impossible.append("empty_test")
            if len(train_idx) == 0:
                impossible.append("empty_train")
            if level == MISSING_LEVEL:
                reasons.append("held_out_level_is_missing_metadata")
            if group == MISSING_LEVEL:
                reasons.append("held_out_group_is_missing_metadata")

            train_authors = set(authors[train_idx].tolist())
            test_author_values = _stable_unique(authors[test_idx])
            absent_authors = tuple(a for a in test_author_values if a not in train_authors)
            if absent_authors:
                reasons.append("some_test_authors_absent_from_train")
            test_author_set = set(test_author_values)
            overlap = (
                len(test_author_set & train_authors) / len(test_author_set)
                if test_author_set
                else 0.0
            )

            train_labels = set(labels[train_idx].tolist())
            test_label_values = _stable_unique(labels[test_idx])
            absent_labels = tuple(label for label in test_label_values if label not in train_labels)
            if len(train_labels) < min_train_classes:
                impossible.append("train_has_fewer_than_minimum_classes")
            if absent_labels:
                impossible.append("test_labels_absent_from_train")
            reasons.extend(impossible)
            reasons = list(dict.fromkeys(reasons))
            confounded = any(
                reason in {
                    "held_out_level_is_missing_metadata",
                    "held_out_group_is_missing_metadata",
                    "some_test_authors_absent_from_train",
                }
                for reason in reasons
            )
            diagnostics = SplitDiagnostics(
                possible=not impossible,
                confounded=confounded,
                reasons=tuple(reasons),
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                n_train_labels=len(train_labels),
                n_test_labels=len(test_label_values),
                labels_absent_from_train=absent_labels,
                authors_absent_from_train=absent_authors,
                test_author_overlap=float(overlap),
            )
            if diagnostics.possible:
                possible_covered[test_idx] = True
            splits.append(
                PurgedFactorSplit(
                    factor=factor,
                    level=level,
                    group_field=group_field,
                    group=group,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    purged_idx=purged_idx,
                    diagnostics=diagnostics,
                )
            )

    return PurgedFactorPlan(
        factor=factor,
        group_field=group_field,
        splits=tuple(splits),
        diagnostics=PurgedPlanDiagnostics(
            n_samples=n,
            n_levels=len(levels),
            n_groups=len(unique_groups),
            n_splits=len(splits),
            test_coverage=float(np.mean(test_covered)) if n else 0.0,
            possible_split_coverage=float(np.mean(possible_covered)) if n else 0.0,
            messages=tuple(messages),
        ),
    )


def align_purged_predictions(
    plan: PurgedFactorPlan,
    predictions_by_cell: Mapping[tuple[Any, Any], Sequence[Any]],
    *,
    fill_value: Any = None,
    require_all_possible: bool = True,
) -> np.ndarray:
    """Align test outputs from independently fitted purged-cell models."""
    normalised = {
        (
            _hashable_scalar(level, missing_marker=True),
            _hashable_scalar(group, missing_marker=True),
        ): values
        for (level, group), values in predictions_by_cell.items()
    }
    known = set(plan.by_cell())
    extra = set(normalised) - known
    if extra:
        raise KeyError(f"predictions supplied for unknown cells: {sorted(extra, key=repr)}")
    out = np.full(plan.diagnostics.n_samples, fill_value, dtype=object)
    assigned = np.zeros(plan.diagnostics.n_samples, dtype=bool)
    for split in plan.splits:
        key = (split.level, split.group)
        if key in normalised and not split.diagnostics.possible:
            raise ValueError(f"predictions are forbidden for impossible cell {key!r}")
        if key not in normalised:
            if require_all_possible and split.diagnostics.possible:
                raise KeyError(f"missing predictions for possible cell {key!r}")
            continue
        pred = np.asarray(normalised[key], dtype=object)
        if pred.ndim != 1 or len(pred) != len(split.test_idx):
            raise ValueError(
                f"predictions for {key!r} have length {len(pred) if pred.ndim else 0}, "
                f"expected {len(split.test_idx)}"
            )
        if assigned[split.test_idx].any():
            raise ValueError("purged split plan has overlapping test indices")
        out[split.test_idx] = pred
        assigned[split.test_idx] = True
    return out


def align_split_predictions(
    plan: FactorSplitPlan,
    predictions_by_level: Mapping[Any, Sequence[Any]],
    *,
    n_samples: Optional[int] = None,
    fill_value: Any = None,
    require_all: bool = True,
) -> np.ndarray:
    """Align test-only predictions from independently fitted split models.

    ``predictions_by_level[level]`` must contain exactly one prediction for each
    index in that split's ``test_idx``.  The helper never accepts predictions for
    ``train_idx``, making the expected train/test boundary explicit.
    """
    n = plan.diagnostics.n_samples if n_samples is None else int(n_samples)
    if n != plan.diagnostics.n_samples:
        raise ValueError("n_samples must match the split plan")
    normalised = {
        _hashable_scalar(level, missing_marker=True): values
        for level, values in predictions_by_level.items()
    }
    known = {split.level for split in plan.splits}
    extra = set(normalised) - known
    if extra:
        raise KeyError(f"predictions supplied for unknown levels: {sorted(extra, key=repr)}")

    out = np.full(n, fill_value, dtype=object)
    assigned = np.zeros(n, dtype=bool)
    for split in plan.splits:
        if split.level in normalised and not split.diagnostics.possible:
            raise ValueError(
                f"predictions are forbidden for impossible held-out level {split.level!r}"
            )
        if split.level not in normalised:
            if require_all and split.diagnostics.possible:
                raise KeyError(f"missing predictions for held-out level {split.level!r}")
            continue
        pred = np.asarray(normalised[split.level], dtype=object)
        pred_len = pred.shape[0] if pred.ndim == 1 else 0
        if pred.ndim != 1 or pred_len != len(split.test_idx):
            raise ValueError(
                f"predictions for {split.level!r} have length {pred_len}, "
                f"expected {len(split.test_idx)}"
            )
        if assigned[split.test_idx].any():
            raise ValueError("split plan has overlapping test indices")
        out[split.test_idx] = pred
        assigned[split.test_idx] = True
    eligible = np.zeros(n, dtype=bool)
    for split in plan.splits:
        if split.diagnostics.possible:
            eligible[split.test_idx] = True
    if require_all and not assigned[eligible].all():
        raise ValueError("split predictions do not cover every feasible sample")
    return out


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any]) -> float:
    scores = []
    for label in labels:
        true_is = y_true == label
        pred_is = y_pred == label
        tp = int(np.sum(true_is & pred_is))
        fp = int(np.sum(~true_is & pred_is))
        fn = int(np.sum(true_is & ~pred_is))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores)) if scores else 0.0


def _exact_label_key(value: Any) -> tuple[type, Any]:
    value = _hashable_scalar(value)
    if _is_missing(value):
        raise ValueError("metric labels must not contain missing values")
    return type(value), value


def _metric_label_universe(
    labels: Optional[Sequence[Any]], y_true: np.ndarray
) -> Tuple[Any, ...]:
    source: Iterable[Any] = y_true if labels is None else labels
    if isinstance(source, (str, bytes)):
        raise TypeError("labels must be a sequence of scalar labels, not a string")
    ordered: list[Any] = []
    seen: set[tuple[type, Any]] = set()
    supplied = labels is not None
    for raw in source:
        value = _hashable_scalar(raw)
        key = _exact_label_key(value)
        if key in seen:
            if supplied:
                raise ValueError("metric label universe must be ordered and duplicate-free")
            continue
        seen.add(key)
        ordered.append(value)
    if not ordered:
        raise ValueError("metric label universe must be non-empty")
    observed = {_exact_label_key(value) for value in y_true}
    missing = observed - seen
    if missing:
        raise ValueError(
            "metric label universe omits observed truth labels: "
            f"{sorted((repr(value) for _kind, value in missing))!r}"
        )
    return tuple(ordered)


def _assert_plan_matches(
    supplied: FactorSplitPlan, expected: FactorSplitPlan
) -> None:
    if (
        supplied.factor != expected.factor
        or supplied.diagnostics != expected.diagnostics
        or len(supplied.splits) != len(expected.splits)
    ):
        raise ValueError("split plan does not match current metadata/truth registration")
    for left, right in zip(supplied.splits, expected.splits, strict=True):
        if (
            _exact_label_key(left.level) != _exact_label_key(right.level)
            or left.diagnostics != right.diagnostics
            or not np.array_equal(left.train_idx, right.train_idx)
            or not np.array_equal(left.test_idx, right.test_idx)
        ):
            raise ValueError("split plan does not match current metadata/truth registration")


def _cluster_keys(metadata: Any, fields: Sequence[str], n: int) -> np.ndarray:
    if not fields:
        out = np.empty(n, dtype=object)
        out[:] = [(i,) for i in range(n)]
        return out
    columns = [_normalised_column(metadata, field, n) for field in fields]
    keys = [tuple(column[i] for column in columns) for i in range(n)]
    # np.asarray(list_of_equal_tuples, dtype=object) becomes 2-D; allocate explicitly.
    out = np.empty(n, dtype=object)
    out[:] = keys
    return out


def _prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
    cluster_keys: np.ndarray,
    *,
    labels: Sequence[Any],
    bootstrap_iters: int,
    ci_level: float,
    seed: int,
) -> PredictionMetrics:
    idx = np.asarray(indices, dtype=int)
    valid = np.asarray([not _is_missing(y_pred[i]) for i in idx], dtype=bool)
    evaluated = idx[valid]
    n_total = len(idx)
    n_eval = len(evaluated)
    coverage = n_eval / n_total if n_total else 0.0
    if n_eval == 0:
        empty = MetricEstimate(None, None, None)
        return PredictionMetrics(n_total, 0, coverage, len(labels), 0, empty, empty)

    yt = y_true[evaluated]
    yp = y_pred[evaluated]
    acc_point = _accuracy(yt, yp)
    f1_point = _macro_f1(yt, yp, labels)
    unique_clusters = _stable_unique(cluster_keys[evaluated])
    n_clusters = len(unique_clusters)

    if bootstrap_iters == 0:
        acc_est = MetricEstimate(acc_point, None, None)
        f1_est = MetricEstimate(f1_point, None, None)
    else:
        local_clusters = cluster_keys[evaluated]
        by_cluster = {
            cluster: np.asarray(
                [i for i, value in enumerate(local_clusters) if value == cluster], dtype=int
            )
            for cluster in unique_clusters
        }
        rng = np.random.default_rng(seed)
        acc_boot = np.empty(bootstrap_iters, dtype=float)
        f1_boot = np.empty(bootstrap_iters, dtype=float)
        for b in range(bootstrap_iters):
            picked = rng.choice(n_clusters, size=n_clusters, replace=True)
            local_idx = np.concatenate([by_cluster[unique_clusters[j]] for j in picked])
            acc_boot[b] = _accuracy(yt[local_idx], yp[local_idx])
            f1_boot[b] = _macro_f1(yt[local_idx], yp[local_idx], labels)
        alpha = (1.0 - ci_level) / 2.0
        q = [100.0 * alpha, 100.0 * (1.0 - alpha)]
        acc_lo, acc_hi = np.percentile(acc_boot, q)
        f1_lo, f1_hi = np.percentile(f1_boot, q)
        acc_est = MetricEstimate(acc_point, float(acc_lo), float(acc_hi))
        f1_est = MetricEstimate(f1_point, float(f1_lo), float(f1_hi))

    return PredictionMetrics(
        n_total=n_total,
        n_evaluated=n_eval,
        coverage=float(coverage),
        n_labels=len(labels),
        n_clusters=n_clusters,
        accuracy=acc_est,
        macro_f1=f1_est,
    )


def evaluate_factor_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    metadata: Any,
    factor: str,
    *,
    plan: Optional[FactorSplitPlan] = None,
    author_field: str = "author",
    cluster_fields: Sequence[str] = ("author", "work"),
    labels: Optional[Sequence[Any]] = None,
    bootstrap_iters: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> FactorEvaluation:
    """Evaluate ready predictions overall and within every factor level."""
    if bootstrap_iters < 0:
        raise ValueError("bootstrap_iters must be >= 0")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    if yt.ndim != 1 or yp.ndim != 1 or len(yt) != len(yp):
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if any(_is_missing(v) for v in yt):
        raise ValueError("y_true must not contain missing labels")
    yt = np.asarray([_hashable_scalar(v) for v in yt], dtype=object)
    n = len(yt)
    cluster_keys = _cluster_keys(metadata, cluster_fields, n)
    expected_plan = build_leave_one_factor_level_out(
        metadata, factor, yt, author_field=author_field
    )
    if plan is None:
        plan = expected_plan
    else:
        _assert_plan_matches(plan, expected_plan)

    global_labels = _metric_label_universe(labels, yt)
    inferential_indices = (
        np.concatenate(
            [
                split.test_idx
                for split in plan.splits
                if split.diagnostics.possible and not split.diagnostics.confounded
            ]
        )
        if any(
            split.diagnostics.possible and not split.diagnostics.confounded
            for split in plan.splits
        )
        else np.asarray([], dtype=int)
    )
    overall = _prediction_metrics(
        yt,
        yp,
        inferential_indices,
        cluster_keys,
        labels=global_labels,
        bootstrap_iters=bootstrap_iters,
        ci_level=ci_level,
        seed=seed,
    )

    slices = []
    for i, split in enumerate(plan.splits):
        # Impossible/confounded cells remain visible through their diagnostics,
        # but they cannot contribute a point estimate or headline.  Every
        # feasible slice uses the one registered metric universe.
        indices = (
            split.test_idx
            if split.diagnostics.possible and not split.diagnostics.confounded
            else np.asarray([], dtype=int)
        )
        metrics = _prediction_metrics(
            yt,
            yp,
            indices,
            cluster_keys,
            labels=global_labels,
            bootstrap_iters=bootstrap_iters,
            ci_level=ci_level,
            seed=seed + i + 1,
        )
        slices.append(
            FactorSliceEvaluation(
                factor=factor,
                level=split.level,
                metrics=metrics,
                split_diagnostics=split.diagnostics,
            )
        )

    scored = [
        row
        for row in slices
        if row.split_diagnostics.possible
        and not row.split_diagnostics.confounded
        and row.metrics.accuracy.point is not None
    ]
    worst = min(scored, key=lambda s: s.metrics.accuracy.point) if scored else None
    possible_n = sum(
        len(split.test_idx) for split in plan.splits if split.diagnostics.possible
    )
    unconfounded_n = sum(
        len(split.test_idx)
        for split in plan.splits
        if split.diagnostics.possible and not split.diagnostics.confounded
    )
    denom = max(1, n)
    return FactorEvaluation(
        factor=factor,
        overall=overall,
        slices=tuple(slices),
        worst_group_accuracy=(None if worst is None else worst.metrics.accuracy.point),
        worst_group_level=(None if worst is None else worst.level),
        possible_split_coverage=float(possible_n / denom),
        unconfounded_split_coverage=float(unconfounded_n / denom),
        diagnostics=plan.diagnostics,
    )


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    metadata: Any,
    *,
    factors: Sequence[str] = DEFAULT_INVARIANCE_FACTORS,
    plans: Optional[Mapping[str, FactorSplitPlan]] = None,
    author_field: str = "author",
    cluster_fields: Sequence[str] = ("author", "work"),
    labels: Optional[Sequence[Any]] = None,
    bootstrap_iters: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> InvarianceReport:
    """Evaluate ready predictions by source/edition or arbitrary metadata factors.

    Predictions may contain ``None``/``NaN`` for unavailable split outputs;
    ``coverage`` reports the evaluated fraction.  Use
    :func:`align_split_predictions` to assemble test-only predictions emitted by
    independently fitted leave-one-level-out models.
    """
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    if yt.ndim != 1 or yp.ndim != 1 or len(yt) != len(yp):
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if any(_is_missing(v) for v in yt):
        raise ValueError("y_true must not contain missing labels")
    yt = np.asarray([_hashable_scalar(v) for v in yt], dtype=object)
    global_labels = _metric_label_universe(labels, yt)
    cluster_keys = _cluster_keys(metadata, cluster_fields, len(yt))
    overall = _prediction_metrics(
        yt,
        yp,
        np.arange(len(yt)),
        cluster_keys,
        labels=global_labels,
        bootstrap_iters=bootstrap_iters,
        ci_level=ci_level,
        seed=seed,
    )

    out: Dict[str, FactorEvaluation] = {}
    supplied_plans = plans or {}
    for factor in factors:
        factor_plan = supplied_plans.get(factor)
        out[factor] = evaluate_factor_predictions(
            yt,
            yp,
            metadata,
            factor,
            plan=factor_plan,
            author_field=author_field,
            cluster_fields=cluster_fields,
            labels=global_labels,
            bootstrap_iters=bootstrap_iters,
            ci_level=ci_level,
            # Keep the repeated overall interval identical to report.overall;
            # slice-level seeds are offset inside evaluate_factor_predictions.
            seed=seed,
        )
    return InvarianceReport(
        overall=overall,
        factors=out,
        cluster_fields=tuple(cluster_fields),
    )


__all__ = [
    "DEFAULT_INVARIANCE_FACTORS",
    "MISSING_LEVEL",
    "FactorEvaluation",
    "FactorSliceEvaluation",
    "FactorSplit",
    "FactorSplitPlan",
    "PurgedFactorPlan",
    "PurgedFactorSplit",
    "PurgedPlanDiagnostics",
    "InvarianceReport",
    "MetricEstimate",
    "PlanDiagnostics",
    "PredictionMetrics",
    "SplitDiagnostics",
    "align_split_predictions",
    "align_purged_predictions",
    "build_leave_one_factor_level_out",
    "build_purged_factor_group_splits",
    "evaluate_factor_predictions",
    "evaluate_predictions",
    "factor_slice_indices",
]
