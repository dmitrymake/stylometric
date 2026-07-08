"""Калибровка decision-скоров в вероятности; выбор метода по OOF-NLL.

Методы: identity (softmax сырых скоров), temperature (softmax(T*score), T по
минимуму NLL на OOF), Platt (мультиномиальная LR на скорах), isotonic (OvR по
классам с нормировкой строк). Всё обучается ТОЛЬКО на OOF-скорах train-фолда;
выбор метода — тоже по OOF (тест не участвует).
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Calibrator = Callable[[np.ndarray], np.ndarray]


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


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


def choose_calibrator(oof: np.ndarray, y: np.ndarray,
                      methods=("identity", "temperature", "platt", "isotonic"),
                      seed: int = 42) -> Tuple[Calibrator, Dict]:
    """Лучший калибратор по held-out NLL внутри OOF-строк.

    Обучаемые калибраторы (Platt, isotonic) на своих train-строках занижают NLL,
    поэтому выбор метода делается на отложенной трети OOF; победитель затем
    рефитится на всех OOF-строках. Тест нигде не участвует.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    cut = max(1, len(y) // 3)
    val_idx, fit_idx = idx[:cut], idx[cut:]
    scores: Dict[str, float] = {}
    for m in methods:
        cal, _ = _fit_method(m, oof[fit_idx], y[fit_idx], seed)
        scores[m] = nll(cal(oof[val_idx]), y[val_idx])
    best = min(scores, key=scores.get)
    cal, params = _fit_method(best, oof, y, seed)
    passport = {"method": best, "params": params, "selection": "held-out треть OOF",
                "heldout_nll": {m: round(v, 4) for m, v in scores.items()}}
    return cal, passport
