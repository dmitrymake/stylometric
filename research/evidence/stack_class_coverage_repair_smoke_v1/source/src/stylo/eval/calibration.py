"""Калибровка decision-скоров в вероятности; выбор метода по OOF-NLL.

Методы: identity (softmax сырых скоров), temperature (softmax(T*score), T по
минимуму NLL на OOF), Platt (мультиномиальная LR на скорах), isotonic (OvR по
классам с нормировкой строк). Всё обучается ТОЛЬКО на OOF-скорах train-фолда;
выбор метода — тоже по OOF (тест не участвует).
"""
from __future__ import annotations

import collections.abc as cabc
import numbers
from collections import Counter, defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

Calibrator = Callable[[np.ndarray], np.ndarray]
_METHODS = ("identity", "temperature", "platt", "isotonic")


class CalibrationClassCoverageError(ValueError):
    """A learned calibration split does not represent the full score universe."""

    def __init__(self, message: str, *, stage: str, report: Dict):
        super().__init__(message)
        self.stage = stage
        self.report = report


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def _work_loss_sum(probs: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Sum over the held-out works of the mean per-chunk NLL inside each work. The pooled equal-work
    NLL is this summed across all folds and divided by the total held-out works (== W, each work
    held out exactly once) — so every work carries equal weight regardless of fold or chunk count."""
    per_chunk = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))
    buckets: Dict = defaultdict(list)
    for i, w in enumerate(groups):
        buckets[w].append(per_chunk[i])
    return float(sum(np.mean(v) for v in buckets.values()))


def _identity(scores: np.ndarray) -> np.ndarray:
    return softmax(scores, axis=1)


def fit_temperature(oof: np.ndarray, y: np.ndarray) -> Tuple[Calibrator, Dict]:
    r = minimize_scalar(lambda s: nll(softmax(np.exp(s) * oof, axis=1), y),
                        bounds=(-4.0, 4.0), method="bounded")
    T = float(np.exp(r.x))
    return (lambda sc, T=T: softmax(T * sc, axis=1)), {"scale": round(T, 4)}


def fit_platt(oof: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[Calibrator, Dict]:
    lr = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(oof, y)
    classes = lr.classes_
    n = oof.shape[1]

    def cal(sc, lr=lr, classes=classes, n=n):
        p = lr.predict_proba(sc)
        full = np.zeros((sc.shape[0], n))
        full[:, classes] = p
        return full
    return cal, {"C": 1.0}


def fit_isotonic(oof: np.ndarray, y: np.ndarray) -> Tuple[Calibrator, Dict]:
    models = []
    for j in range(oof.shape[1]):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
        iso.fit(oof[:, j], (y == j).astype(float))
        models.append(iso)

    def cal(sc, models=models):
        cols = np.column_stack([m.predict(sc[:, j]) for j, m in enumerate(models)])
        cols = np.clip(cols, 1e-9, None)
        return cols / cols.sum(axis=1, keepdims=True)
    return cal, {}


def _fit_method(m: str, oof: np.ndarray, y: np.ndarray, seed: int):
    if m == "identity":
        return _identity, {}
    if m == "temperature":
        return fit_temperature(oof, y)
    if m == "platt":
        return fit_platt(oof, y, seed)
    if m == "isotonic":
        return fit_isotonic(oof, y)
    raise ValueError(m)


def _validate_legacy_inputs(oof, y):
    """Fail-closed contract for the ungrouped calibration-selection path."""
    oof_raw = np.asarray(oof)
    if (
        oof_raw.dtype == bool
        or np.issubdtype(oof_raw.dtype, np.complexfloating)
        or not np.issubdtype(oof_raw.dtype, np.number)
    ):
        raise ValueError(
            "oof must be a real, non-bool, non-complex numeric array"
        )
    if oof_raw.ndim != 2 or oof_raw.shape[0] == 0 or oof_raw.shape[1] < 2:
        raise ValueError(
            "oof must be a non-empty 2-D score matrix with at least two classes"
        )
    if not np.isfinite(oof_raw).all():
        raise ValueError("oof has NaN/inf entries")

    n_rows, n_classes = oof_raw.shape
    y_array = np.asarray(y)
    if y_array.ndim != 1 or len(y_array) != n_rows:
        raise ValueError(f"y must be 1-D of length {n_rows}")
    if y_array.dtype == bool or not all(
        isinstance(value, numbers.Integral) and not isinstance(value, bool)
        for value in y_array.tolist()
    ):
        raise ValueError("y must be non-bool integer labels")
    expected = np.arange(n_classes)
    observed = np.unique(y_array)
    if not np.array_equal(observed, expected):
        raise CalibrationClassCoverageError(
            "legacy calibration labels do not cover every score column",
            stage="input_class_universe",
            report={
                "expected_classes": expected.tolist(),
                "observed_classes": observed.tolist(),
                "missing_classes": np.setdiff1d(
                    expected, observed, assume_unique=True,
                ).tolist(),
                "unexpected_classes": np.setdiff1d(
                    observed, expected, assume_unique=True,
                ).tolist(),
            },
        )
    # Preserve the exact arrays/dtypes consumed by the historical valid path;
    # validation must not become an implicit numerical transform.
    return oof_raw, y_array


def choose_calibrator(oof: np.ndarray, y: np.ndarray,
                      methods=("identity", "temperature", "platt", "isotonic"),
                      seed: int = 42, groups: Optional[np.ndarray] = None,
                      n_splits: int = 3) -> Tuple[Calibrator, Dict]:
    """Лучший калибратор по held-out NLL внутри OOF-строк.

    Обучаемые калибраторы (Platt, isotonic) на своих train-строках занижают NLL, поэтому выбор
    метода делается на отложенной части OOF; победитель рефитится на всех OOF-строках. Тест нигде
    не участвует.

    ``groups=None`` — legacy: одна случайная отложенная треть чанков (chunk-level).
    ``groups`` заданы (work_balanced, B3) — **group-aware**: StratifiedGroupKFold по работам, чтобы
    чанки одной работы не расщеплялись между fit/held-out (иначе book-level утечка в выборе метода).
    ``n_splits_eff = min(n_splits, min_works_per_class)``; если у класса < 2 работ или класс выпадает
    из fit-фолда — калибровка **отключается целиком** (identity), без chunk-CV фоллбэка.
    """
    # a Set/Mapping has no stable iteration order (PYTHONHASHSEED changes the chosen method) — reject;
    # a sequence/generator is materialised ordered-unique.
    if isinstance(methods, (cabc.Set, cabc.Mapping)):
        raise ValueError("methods must be an ordered sequence (list/tuple), not a set/mapping")
    methods = tuple(dict.fromkeys(methods))             # ordered-unique; materialises a generator once
    if not methods:
        raise ValueError("methods must be a non-empty collection")
    bad = [m for m in methods if m not in _METHODS]
    if bad:
        raise ValueError(f"unknown calibration methods {bad}; allowed {_METHODS}")
    if groups is None:
        oof, y = _validate_legacy_inputs(oof, y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(y))
        cut = max(1, len(y) // 3)
        val_idx, fit_idx = idx[:cut], idx[cut:]
        expected = np.arange(oof.shape[1])
        fit_classes = np.unique(y[fit_idx])
        validation_classes = np.unique(y[val_idx])
        missing_fit = np.setdiff1d(
            expected, fit_classes, assume_unique=True,
        )
        missing_validation = np.setdiff1d(
            expected, validation_classes, assume_unique=True,
        )
        if len(missing_fit) or len(missing_validation):
            raise CalibrationClassCoverageError(
                "legacy calibration selection split omits classes; learned "
                "calibration is forbidden",
                stage="selection_split",
                report={
                    "expected_classes": expected.tolist(),
                    "fit_classes": fit_classes.tolist(),
                    "validation_classes": validation_classes.tolist(),
                    "missing_fit_classes": missing_fit.tolist(),
                    "missing_validation_classes": missing_validation.tolist(),
                },
            )
        scores: Dict[str, float] = {}
        for m in methods:
            cal, _ = _fit_method(m, oof[fit_idx], y[fit_idx], seed)
            scores[m] = nll(cal(oof[val_idx]), y[val_idx])
        best = min(scores, key=scores.get)
        cal, params = _fit_method(best, oof, y, seed)
        return cal, {"method": best, "params": params, "selection": "held-out треть OOF",
                     "heldout_nll": {m: round(v, 4) for m, v in scores.items()}}
    return _choose_calibrator_grouped(oof, y, groups, methods, seed, n_splits)


def _identity_disabled(oof, y, seed, reason: str) -> Tuple[Calibrator, Dict]:
    cal, params = _fit_method("identity", oof, y, seed)
    return cal, {"method": "identity", "params": params, "group_aware": True,
                 "calibration_disabled": True, "reason": reason}


def _validate_grouped_inputs(oof, y, groups):
    """Fail-closed input contract for group-aware calibration (finite 2-D oof; 1-D non-bool integral
    y in [0, n_classes); equal non-empty lengths; validated work ids; exactly one label per work)."""
    from ..features.work_vectorizer import validate_work_ids
    # validate the RAW dtype BEFORE coercion: a float cast would silently accept bool / numeric
    # strings / drop a complex imaginary part and still emit a normal passport.
    oof_raw = np.asarray(oof)
    if (oof_raw.dtype == bool or np.issubdtype(oof_raw.dtype, np.complexfloating)
            or not np.issubdtype(oof_raw.dtype, np.number)):
        raise ValueError("oof must be a real, non-bool, non-complex numeric array")
    oof = oof_raw.astype(float)
    if oof.ndim != 2 or oof.shape[0] == 0 or oof.shape[1] == 0:
        raise ValueError("oof must be a non-empty 2-D score matrix")
    if not np.isfinite(oof).all():
        raise ValueError("oof has NaN/inf entries")
    n, n_classes = oof.shape
    y_arr = np.asarray(y)
    if y_arr.ndim != 1 or len(y_arr) != n:
        raise ValueError(f"y must be 1-D of length {n}")
    if y_arr.dtype == bool or not all(
            isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in y_arr.tolist()):
        raise ValueError("y must be non-bool integer labels")
    y_int = y_arr.astype(int)
    if y_int.min() < 0 or y_int.max() >= n_classes:
        raise ValueError(f"y labels must be within [0, {n_classes})")
    groups = np.asarray(validate_work_ids(groups, n), dtype=object)   # fail-closed on bad/short groups
    work_class: Dict = {}
    for w, c in zip(groups, y_int):
        c = int(c)
        if w in work_class and work_class[w] != c:
            raise ValueError(f"work {w!r} carries more than one label")
        work_class[w] = c
    return oof, y_int, groups, work_class, n_classes


def _choose_calibrator_grouped(oof, y, groups, methods, seed, n_splits) -> Tuple[Calibrator, Dict]:
    if isinstance(n_splits, bool) or not isinstance(n_splits, numbers.Integral) or int(n_splits) < 2:
        raise ValueError(f"n_splits must be a non-bool integer >= 2, got {n_splits!r}")
    n_splits = int(n_splits)
    oof, y, groups, work_class, _ = _validate_grouped_inputs(oof, y, groups)
    classes = np.unique(y)
    all_classes = set(classes.tolist())
    per_class_works = Counter(work_class.values())
    min_works = min(int(per_class_works.get(int(c), 0)) for c in classes)
    if min_works < 2:                                   # cannot hold a work out per class -> disable
        return _identity_disabled(oof, y, seed, f"min_works_per_class={min_works} < 2")

    n_splits_eff = min(n_splits, min_works)
    sgkf = StratifiedGroupKFold(n_splits_eff, shuffle=True, random_state=seed)
    splits = list(sgkf.split(oof, y, groups))

    # Validate the entire splitter structure BEFORE the registered class-absence
    # identity fallback.  Otherwise a malformed splitter could masquerade as a
    # scientifically allowed sparse-class condition.
    if len(splits) != n_splits_eff:
        raise ValueError(
            f"calibration splitter returned {len(splits)} folds, expected {n_splits_eff}"
        )
    val_all = np.concatenate([v for _, v in splits]) if splits else np.empty(0, dtype=int)
    n_rows = oof.shape[0]
    all_rows = np.arange(n_rows)
    if sorted(int(i) for i in val_all) != list(range(n_rows)):
        raise ValueError("calibration folds must hold out every row in validation exactly once")
    for fit_i, val_i in splits:
        fit_i = np.asarray(fit_i)
        val_i = np.asarray(val_i)
        if (
            fit_i.ndim != 1
            or val_i.ndim != 1
            or not np.issubdtype(fit_i.dtype, np.integer)
            or not np.issubdtype(val_i.dtype, np.integer)
            or len(np.unique(fit_i)) != len(fit_i)
            or len(np.unique(val_i)) != len(val_i)
            or not np.array_equal(
                np.sort(fit_i), np.setdiff1d(all_rows, val_i),
            )
        ):
            raise ValueError("calibration fold train/validation must partition all rows")
        if len(np.intersect1d(groups[fit_i], groups[val_i])):
            raise ValueError("calibration fold train/validation groups must be disjoint")

    # BOTH sides of every structurally valid split must carry all classes.  This
    # registered sparse-class condition alone may disable learned calibration.
    for fit_i, val_i in splits:
        if (set(np.unique(y[fit_i]).tolist()) != all_classes
                or set(np.unique(y[val_i]).tolist()) != all_classes):
            return _identity_disabled(oof, y, seed, "a class is absent from a calibration fold (train or val)")
    n_works_total = len({str(w) for w in groups})

    total_loss = {m: 0.0 for m in methods}
    total_works = 0
    for fit_i, val_i in splits:
        total_works += len({str(w) for w in groups[val_i]})     # each work held out exactly once
        for m in methods:
            cal, _ = _fit_method(m, oof[fit_i], y[fit_i], seed)
            total_loss[m] += _work_loss_sum(cal(oof[val_i]), y[val_i], groups[val_i])
    if total_works != n_works_total:                            # each work in exactly one validation fold
        raise ValueError("calibration works are not held out exactly once (grouped denominator invalid)")
    best = min(total_loss, key=total_loss.get)          # pooled equal-work NLL (same divisor for all)
    cal, params = _fit_method(best, oof, y, seed)
    return cal, {"method": best, "params": params, "group_aware": True, "n_splits": n_splits_eff,
                 "selection": "StratifiedGroupKFold pooled equal-work NLL",
                 "heldout_work_nll": {m: round(total_loss[m] / total_works, 4) for m in methods}}
