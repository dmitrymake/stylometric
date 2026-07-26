"""Cluster-valid paired inference: null-centered author-cluster bootstrap p and Holm (§3.3/§3.4).

``paired_bootstrap_diff_clustered`` returns a CI, not a p-value, so the confirmatory audit
pre-registers a proper cluster p-value here:

- **Test** — two-sided **null-centered author-cluster bootstrap** p for Δaccuracy ``spec − A0`` on
  the identical fold set. Resample **authors** with replacement ``B = 10000`` (``seed = 42``); for
  each draw compute ``Δ*``; center the bootstrap distribution at its mean (impose H0 E[Δ]=0);
  ``p = (1 + #{|Δ*_centered| ≥ |Δ_observed|}) / (1 + B)`` (**+1 correction**, so p is never 0).
- **Degenerate cases (checked in this exact order)** — (1) fewer than two unique authors → ``p = 1``;
  (2) all paired book-correctness differences are zero → ``p = 1``; (3) otherwise the general
  algorithm, **including a non-zero constant effect** (not special-cased).
- **McNemar** (book-level) is stored **diagnostic-only** and never enters a claim.

Holm–Bonferroni runs on the UNROUNDED cluster p-values of the frozen 15-member family
(``family_alpha = 0.05``; ``significant := holm_p < 0.05``); a missing/failed member invalidates the
whole family — ``m`` is never reduced.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

DEFAULT_B = 10000
DEFAULT_SEED = 42
FAMILY_ALPHA = 0.05


class PairedInferenceError(ValueError):
    """Fail-closed: paired inference inputs are misaligned or the Holm family is malformed."""


def paired_cluster_pvalue(correct_a: Sequence, correct_b: Sequence, authors: Sequence, *,
                          B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> float:
    """Two-sided null-centered author-cluster bootstrap p for Δaccuracy (a − b) over books.

    ``correct_a``/``correct_b`` are per-book correctness (0/1) on the identical fold set; ``authors``
    is the per-book author cluster label (the resampling unit). Returns a p-value in ``(0, 1]``.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    au = np.asarray([str(x) for x in authors])
    if not (a.shape == b.shape == au.shape and a.ndim == 1):
        raise PairedInferenceError("correct_a/correct_b/authors must be equal-length 1-D sequences")
    if a.size == 0:
        raise PairedInferenceError("empty comparison")
    if not (np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all()):
        raise PairedInferenceError("correctness vectors must contain only 0/1")
    if type(B) is not int or B <= 0:
        raise PairedInferenceError("B must be a positive int")
    if type(seed) is not int:
        raise PairedInferenceError("seed must be an int")
    if any(x == "" for x in au.tolist()):
        raise PairedInferenceError("author cluster ids must be non-empty")

    diff = a - b
    uniq = sorted(set(au.tolist()))
    # degenerate rules, in this exact order
    if len(uniq) < 2:                       # (1) a single author cluster cannot be resampled
        return 1.0
    if np.all(diff == 0.0):                 # (2) no paired book-correctness difference at all
        return 1.0

    # (3) general null-centered cluster bootstrap. Resampling a whole author = summing its books, so
    # Δ* = Σ(sampled authors' book-diff) / Σ(sampled authors' book-count) — a vectorized cluster mean.
    author_sum = np.array([diff[au == a_id].sum() for a_id in uniq], dtype=float)
    author_cnt = np.array([int((au == a_id).sum()) for a_id in uniq], dtype=float)
    obs = author_sum.sum() / author_cnt.sum()          # = mean(diff) over all books

    n = len(uniq)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(int(B), n))        # sample n authors with replacement, B times
    num = author_sum[draws].sum(axis=1)
    den = author_cnt[draws].sum(axis=1)                 # always >= n >= 2 > 0
    boot = num / den
    centered = boot - boot.mean()                       # impose H0: E[Δ] = 0
    extreme = int(np.sum(np.abs(centered) >= abs(obs)))
    return (1 + extreme) / (1 + int(B))                 # +1 correction — p is never 0


def mcnemar_diagnostic(correct_a: Sequence, correct_b: Sequence) -> dict:
    """Book-level exact McNemar — stored **diagnostic-only**; it never enters a claim (§3.3)."""
    from ..significance import mcnemar
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    r = mcnemar(a, b)
    return {"b": int(r.b), "c": int(r.c), "mcnemar_p_diagnostic": float(r.p_value),
            "role": "diagnostic_only"}


def holm_bonferroni(raw_pvalues: Mapping, *, m: int | None = None,
                    alpha: float = FAMILY_ALPHA) -> dict:
    """Holm–Bonferroni step-down over the UNROUNDED raw p-values.

    ``m`` (the family size) defaults to ``len(raw_pvalues)`` but is passed explicitly for a fixed
    family so it is **never reduced**. Returns ``{key: {raw_p, holm_p, significant}}`` with
    ``significant := holm_p < alpha``.
    """
    items = list(raw_pvalues.items())
    if not items:
        raise PairedInferenceError("empty p-value family")
    size = len(items) if m is None else int(m)
    if size < len(items):
        raise PairedInferenceError("m must not be smaller than the number of hypotheses")
    for _, p in items:
        if not (isinstance(p, (int, float)) and not isinstance(p, bool) and 0.0 <= float(p) <= 1.0):
            raise PairedInferenceError(f"raw p-value {p!r} is not a real number in [0,1]")

    order = sorted(range(len(items)), key=lambda i: float(items[i][1]))
    out: dict = {}
    running = 0.0
    for rank, i in enumerate(order):                    # step-down, 0-based rank
        key, p = items[i]
        adj = min(1.0, (size - rank) * float(p))        # (m - k + 1) * p_(k), k = rank + 1
        running = max(running, adj)                     # enforce monotone non-decreasing adjusted p
        out[key] = {"raw_p": float(p), "holm_p": running, "significant": running < alpha}
    return out


def holm_over_registered_family(raw_pvalues: Mapping) -> dict:
    """Holm over EXACTLY the frozen 15-member family (§3.4): a missing/extra member invalidates the
    family and ``m`` stays 15."""
    from .applicability import assert_holm_family_complete
    assert_holm_family_complete(list(raw_pvalues.keys()))
    return holm_bonferroni(raw_pvalues, m=15, alpha=FAMILY_ALPHA)
