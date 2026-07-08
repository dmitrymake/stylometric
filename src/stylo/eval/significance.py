"""Статистическая значимость различий между конфигурациями.

Сравнения ПАРНЫЕ по книгам (одни и те же тестовые книги в обоих конфигах):
  - mcnemar — точный биномиальный McNemar по парным исходам корректности. Единица —
    КНИГА; книги внутри одного автора коррелированы, поэтому этот p — book-level и
    антиконсервативная ГРАНИЦА (нижняя оценка). Кластер-робастную значимость даёт
    paired_bootstrap_diff_clustered (ресэмпл авторов, а не книг).
  - paired_bootstrap_diff — CI для разницы метрики ресэмплом книг (тоже book-level).
  - paired_bootstrap_diff_clustered — CI с ресэмплом АВТОРОВ (кластеров книг): учитывает
    внутриавторскую корреляцию, honest-версия для отчёта. Значимо, если CI не пересекает 0.
Это заменяет наивное «A 84% vs B 82%» без оценки неопределённости.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import binomtest


@dataclass
class McNemarResult:
    b: int          # A верно, B неверно
    c: int          # A неверно, B верно
    p_value: float

    def __str__(self) -> str:
        return f"McNemar b={self.b} c={self.c} p={self.p_value:.4f}"


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> McNemarResult:
    """Точный биномиальный McNemar по парным булевым исходам корректности.

    Единица — книга; при коррелированных книгах внутри автора p антиконсервативен и
    читается как ГРАНИЦА. Кластер-робастная значимость — paired_bootstrap_diff_clustered."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    disc_b = int(np.sum(a & ~b))   # A прав, B нет
    disc_c = int(np.sum(~a & b))   # B прав, A нет
    n = disc_b + disc_c
    if n == 0:
        return McNemarResult(disc_b, disc_c, 1.0)
    p = binomtest(disc_b, n, 0.5, alternative="two-sided").pvalue
    return McNemarResult(disc_b, disc_c, float(p))


@dataclass
class DiffCI:
    diff: float
    lo: float
    hi: float
    significant: bool

    def __str__(self) -> str:
        star = " *" if self.significant else ""
        return f"Δ={self.diff:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}]{star}"


def paired_bootstrap_diff(
    metric_a: Callable[[np.ndarray], float],
    metric_b: Callable[[np.ndarray], float],
    n_units: int,
    iters: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> DiffCI:
    """CI для (metric_a - metric_b) ресэмплингом ОДНИХ И ТЕХ ЖЕ книг для обоих."""
    if n_units == 0:
        return DiffCI(0.0, 0.0, 0.0, False)
    rng = np.random.default_rng(seed)
    full = np.arange(n_units)
    point = metric_a(full) - metric_b(full)
    boot = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n_units, size=n_units)
        boot[i] = metric_a(idx) - metric_b(idx)
    alpha = (1 - level) / 2
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    sig = not (lo <= 0.0 <= hi)
    return DiffCI(float(point), float(lo), float(hi), bool(sig))


def paired_bootstrap_diff_clustered(
    metric_a: Callable[[np.ndarray], float],
    metric_b: Callable[[np.ndarray], float],
    groups: np.ndarray,
    iters: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> DiffCI:
    """CI для (metric_a - metric_b) с КЛАСТЕРНЫМ ресэмплингом: единица ресэмпла — группа
    (автор), а не отдельная книга. Книги одного автора коррелированы, поэтому ресэмпл по
    книгам занижает неопределённость; ресэмпл по авторам её учитывает — это honest-версия
    значимости для разницы accuracy stylo vs baseline.

    groups: массив длины n_units с идентификатором группы (автором) каждой книги;
    metric_a/metric_b принимают массив ИНДЕКСОВ книг и возвращают скалярную метрику."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    n_units = len(groups)
    if n_units == 0 or len(uniq) == 0:
        return DiffCI(0.0, 0.0, 0.0, False)
    full = np.arange(n_units)
    point = metric_a(full) - metric_b(full)
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(iters)
    for i in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        boot[i] = metric_a(idx) - metric_b(idx)
    alpha = (1 - level) / 2
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    sig = not (lo <= 0.0 <= hi)
    return DiffCI(float(point), float(lo), float(hi), bool(sig))
