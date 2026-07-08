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


def _sm(scores):
    """softmax по классам с защитой от -inf (отсутствующие классы в фолде)."""
    return softmax(np.where(scores < -1e8, -30.0, scores), axis=1)


def reliability_weighted(test_scores: dict, train_acc: dict, chance: float, power: float = 2.0):
    """Взвешенное усреднение softmax каналов; вес w_c = max(0, acc_c − chance)^power.
    test_scores/train_acc — словари {channel: ...}. Возвращает (n, n_classes) вероятности."""
    names = list(test_scores)
    w = np.array([max(0.0, train_acc[c] - chance) ** power for c in names])
    if w.sum() == 0:
        w = np.ones(len(names))
    w = w / w.sum()
    out = None
    for wi, c in zip(w, names):
        p = _sm(test_scores[c]) * wi
        out = p if out is None else out + p
    return out, dict(zip(names, np.round(w, 3)))


def stacked(train_oof_scores: dict, ytrain: np.ndarray, test_scores: dict, n_classes: int, seed: int = 42):
    """Стекинг: мета-LR на горизонтальной конкатенации softmax-вероятностей каналов.
    Обучается на TRAIN-OOF (leak-free), применяется к скорам каналов на тесте. Возвращает (n_test, n_classes)."""
    names = list(train_oof_scores)
    Xtr = np.hstack([_sm(train_oof_scores[c]) for c in names])
    Xte = np.hstack([_sm(test_scores[c]) for c in names])
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
    meta.fit(Xtr, ytrain)
    # выровнять под полный набор классов (если в train не все)
    proba = meta.predict_proba(Xte)
    full = np.zeros((Xte.shape[0], n_classes))
    full[:, meta.classes_] = proba
    return full
