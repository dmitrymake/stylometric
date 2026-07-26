"""Слияние каналов: reliability-взвешивание и стекинг.

Наивное усреднение нарушает инвариант «ансамбль ≥ лучшего канала»: усредняя сильный канал
со слабыми, оно проседает. Здесь:
  • reliability_weighted — вес канала ∝ (его accuracy − случайный уровень)^power; слабые каналы глушатся.
    ВНИМАНИЕ: leak-free только если переданная accuracy получена НЕ на отчётном тест-множестве
    (напр. вложенным CV на train). Подать точность на тех же тест-книгах = test-set leak, поэтому
    headline-метрика на нём НЕ строится (см. run_benchmark.py: headline = равновесный ансамбль).
  • stacked — мета-LogisticRegression поверх softmax-вероятностей каналов (обучается на TRAIN-OOF).
"""
from __future__ import annotations
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from ..domain.prediction_contract import (
    PredictionContractError,
    validate_channel_mapping,
)


def _sm(scores):
    """Softmax only class-complete, finite scores.

    Missing classes must be rejected by the producing fold.  Turning a sentinel
    into a finite logit fabricates a probability distribution.
    """
    arr = np.asarray(scores)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise PredictionContractError("scores must be a 2D class-complete matrix")
    if not np.isfinite(arr).all():
        raise PredictionContractError("scores contain a missing/nonfinite class")
    return softmax(arr.astype(np.float64, copy=False), axis=1)


def reliability_weighted(test_scores: dict, train_acc: dict, chance: float, power: float = 2.0):
    """Взвешенное усреднение softmax каналов; вес w_c = max(0, acc_c − chance)^power.
    test_scores/train_acc — словари {channel: ...}. Возвращает (n, n_classes) вероятности."""
    checked, _, _ = validate_channel_mapping(test_scores, name="test_scores")
    names = list(checked)
    if type(train_acc) is not dict or set(train_acc) != set(names):
        raise PredictionContractError(
            "train_acc must contain exactly the score-channel names"
        )
    if (
        isinstance(chance, bool)
        or not isinstance(chance, (int, float))
        or not np.isfinite(chance)
        or not 0.0 <= float(chance) < 1.0
    ):
        raise PredictionContractError("chance must be finite and within [0, 1)")
    if (
        isinstance(power, bool)
        or not isinstance(power, (int, float))
        or not np.isfinite(power)
        or float(power) <= 0.0
    ):
        raise PredictionContractError("power must be finite and positive")
    for channel, accuracy in train_acc.items():
        if (
            isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not np.isfinite(accuracy)
            or not 0.0 <= float(accuracy) <= 1.0
        ):
            raise PredictionContractError(
                f"train_acc[{channel!r}] must be finite within [0, 1]"
            )
    w = np.array([max(0.0, train_acc[c] - chance) ** power for c in names])
    if w.sum() == 0:
        w = np.ones(len(names))
    w = w / w.sum()
    out = None
    for wi, c in zip(w, names):
        p = _sm(checked[c]) * wi
        out = p if out is None else out + p
    return out, dict(zip(names, np.round(w, 3)))


def stacked(train_oof_scores: dict, ytrain: np.ndarray, test_scores: dict, n_classes: int, seed: int = 42):
    """Стекинг: мета-LR на горизонтальной конкатенации softmax-вероятностей каналов.
    Обучается на TRAIN-OOF (leak-free), применяется к скорам каналов на тесте. Возвращает (n_test, n_classes)."""
    train_checked, n_train, width = validate_channel_mapping(
        train_oof_scores, n_classes=n_classes, name="train_oof_scores"
    )
    test_checked, _, _ = validate_channel_mapping(
        test_scores, n_classes=n_classes, name="test_scores"
    )
    names = list(train_checked)
    if set(test_checked) != set(names):
        raise PredictionContractError(
            "train and test score mappings must contain the same channels"
        )
    ytrain = np.asarray(ytrain)
    if (
        ytrain.ndim != 1
        or len(ytrain) != n_train
        or ytrain.dtype.kind not in "iu"
        or not np.array_equal(np.unique(ytrain), np.arange(width))
    ):
        raise PredictionContractError(
            "ytrain must cover the complete ordered class universe"
        )
    Xtr = np.hstack([_sm(train_checked[c]) for c in names])
    Xte = np.hstack([_sm(test_checked[c]) for c in names])
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
    meta.fit(Xtr, ytrain)
    # выровнять под полный набор классов (если в train не все)
    proba = meta.predict_proba(Xte)
    full = np.zeros((Xte.shape[0], n_classes))
    full[:, meta.classes_] = proba
    return full
