"""The single pre-registered noninferiority headline gate (§3.5).

The **only** headline endpoint is **stylo LOBO Δaccuracy A4 − A0**, decided by an author-clustered
**percentile** bootstrap (two-sided 95% CI, ``iters = 10000``, ``seed = 42``, quantiles
``[2.5, 97.5]``) with noninferiority margin ``δ = 0.02``:

- **relabel** iff the CI **lower bound > −δ**;
- **keep_legacy** iff the CI **upper bound < −δ**;
- **inconclusive** otherwise (the CI straddles or *equals* −δ).

The decision is on the **UNROUNDED** bounds. The artifact also stores the **absolute**
author-clustered accuracy CI of A4 under the **identical** resampling settings. A1–A3 are
descriptive and out of this gate. The author-clustered macro-F1 CI stays **withdrawn** (point macro-F1
only, over the frozen ``metric_label_order``). This module computes only — it writes nothing, and in
particular never touches a headline artifact path (publication is gated separately).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

HEADLINE_ENDPOINT = "stylo_lobo_a4_minus_a0_accuracy"
DEFAULT_ITERS = 10000
DEFAULT_SEED = 42
DEFAULT_MARGIN = 0.02
DEFAULT_QUANTILES = (2.5, 97.5)


class HeadlineError(ValueError):
    """Fail-closed: headline inputs are misaligned or empty."""


def _cluster_percentile_ci(values: Sequence, authors: Sequence, *, iters: int, seed: int,
                           quantiles=DEFAULT_QUANTILES) -> dict:
    """Author-clustered percentile bootstrap of the cluster mean of ``values``.

    Resamples authors with replacement (a whole author = the sum of its books); returns
    ``{point, lo, hi}`` with ``point`` the observed cluster mean and ``[lo, hi]`` the percentile CI.
    """
    v = np.asarray(values, dtype=float)
    au = np.asarray([str(x) for x in authors])
    if not (v.shape == au.shape and v.ndim == 1):
        raise HeadlineError("values/authors must be equal-length 1-D sequences")
    if v.size == 0:
        raise HeadlineError("empty headline comparison")
    if not np.isfinite(v).all():
        raise HeadlineError("headline values must be finite")
    if type(iters) is not int or iters <= 0:
        raise HeadlineError("iters must be a positive int")
    if type(seed) is not int:
        raise HeadlineError("seed must be an int")
    q = list(quantiles)
    if len(q) != 2 or not (0 < q[0] < q[1] < 100):
        raise HeadlineError("quantiles must be two increasing values in (0,100)")
    if any(x == "" for x in au.tolist()):
        raise HeadlineError("author cluster ids must be non-empty")
    uniq = sorted(set(au.tolist()))
    point = float(v.sum() / v.size)                      # cluster mean over all books
    if len(uniq) < 2:                                    # cannot cluster-resample a single author
        return {"point": point, "lo": point, "hi": point}
    author_sum = np.array([v[au == a].sum() for a in uniq], dtype=float)
    author_cnt = np.array([int((au == a).sum()) for a in uniq], dtype=float)
    n = len(uniq)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(int(iters), n))
    boot = author_sum[draws].sum(axis=1) / author_cnt[draws].sum(axis=1)
    lo, hi = np.percentile(boot, list(quantiles))
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def author_clustered_accuracy_ci(correct: Sequence, authors: Sequence, *,
                                 iters: int = DEFAULT_ITERS, seed: int = DEFAULT_SEED,
                                 quantiles=DEFAULT_QUANTILES) -> dict:
    """Absolute author-clustered accuracy CI (percentile method)."""
    if not np.isin(np.asarray(correct, dtype=float), (0.0, 1.0)).all():
        raise HeadlineError("correctness vector must contain only 0/1")
    return _cluster_percentile_ci(correct, authors, iters=iters, seed=seed, quantiles=quantiles)


def paired_accuracy_diff_ci(correct_a: Sequence, correct_b: Sequence, authors: Sequence, *,
                            iters: int = DEFAULT_ITERS, seed: int = DEFAULT_SEED,
                            quantiles=DEFAULT_QUANTILES) -> dict:
    """Author-clustered Δaccuracy (a − b) CI (percentile method), identical resampling settings."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise HeadlineError("correct_a/correct_b must be equal length")
    if not (np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all()):
        raise HeadlineError("correctness vectors must contain only 0/1")
    return _cluster_percentile_ci(a - b, authors, iters=iters, seed=seed, quantiles=quantiles)


def headline_gate(ci_lo: float, ci_hi: float, *, margin: float = DEFAULT_MARGIN) -> str:
    """Symmetric noninferiority decision on the UNROUNDED CI bounds (boundary equality → inconclusive)."""
    if not (isinstance(margin, (int, float)) and not isinstance(margin, bool)
            and math.isfinite(margin) and margin > 0):
        raise HeadlineError("margin must be a positive finite number")
    if ci_hi < ci_lo:
        raise HeadlineError("CI upper bound is below the lower bound")
    if ci_lo > -margin:
        return "relabel"
    if ci_hi < -margin:
        return "keep_legacy"
    return "inconclusive"


def evaluate_headline(correct_a4: Sequence, correct_a0: Sequence, authors: Sequence, *,
                      margin: float = DEFAULT_MARGIN, iters: int = DEFAULT_ITERS,
                      seed: int = DEFAULT_SEED, quantiles=DEFAULT_QUANTILES) -> dict:
    """The full pre-registered headline decision: the A4−A0 difference CI, the absolute A4 accuracy CI
    (identical settings), and the gate decision on the unrounded difference bounds."""
    diff_ci = paired_accuracy_diff_ci(correct_a4, correct_a0, authors,
                                      iters=iters, seed=seed, quantiles=quantiles)
    a4_abs_ci = author_clustered_accuracy_ci(correct_a4, authors,
                                             iters=iters, seed=seed, quantiles=quantiles)
    decision = headline_gate(diff_ci["lo"], diff_ci["hi"], margin=margin)
    return {
        "endpoint": HEADLINE_ENDPOINT,
        "decision": decision,
        "margin": margin,
        "method": "author_clustered_percentile_bootstrap",
        "iters": iters, "seed": seed, "quantiles": list(quantiles),
        "diff_ci": diff_ci,                              # A4 - A0 Δaccuracy CI (drives the gate)
        "a4_abs_accuracy_ci": a4_abs_ci,                 # absolute A4 accuracy CI, identical settings
        "macro_f1_ci": "withdrawn",                      # author-clustered macro-F1 CI stays withdrawn
    }
