"""Descriptive diagnostics for authorship-model posterior matrices.

The historical version of this module treated the entropy of an arbitrary
model's posterior as ``H(A|F)``.  It then published ``H(A)-H(A|F)`` as a mutual
information lower bound and entropy transforms as unavoidable/Bayes error
floors.  Those interpretations are invalid without assumptions that the API
neither established nor tested.  In particular, a constant overconfident model
can manufacture a large apparent "information" value although its output is
independent of the true label.

Version 2 therefore exposes only quantities that the supplied model output
directly determines:

* empirical label-prior entropy;
* mean entropy of the model posterior rows;
* their explicitly named arithmetic contrast;
* empirical error and optional calibration error;
* a binary *posterior-entropy-equivalent* error, which is a descriptive
  re-expression of model confidence, never a Bayes-error bound.

No function in the v2 API estimates mutual information, a Fano lower bound, an
unavoidable error, or pair indistinguishability.  The old inferential entry
points remain as fail-closed compatibility stubs so callers cannot silently
continue publishing the withdrawn semantics.
"""
from __future__ import annotations

from typing import Any, Dict, List, NoReturn, Sequence, Tuple

import numpy as np

_EPS = 1e-12  # клиппинг постериоров для устойчивости логарифма
POSTERIOR_DIAGNOSTICS_SCHEMA = "stylo.model-posterior-diagnostics.v2"


class WithdrawnFanoSemanticsError(RuntimeError):
    """A caller requested a scientifically withdrawn inferential quantity."""


def _withdrawn(name: str) -> NoReturn:
    raise WithdrawnFanoSemanticsError(
        f"{name} was withdrawn: arbitrary model-posterior entropy does not "
        "identify conditional entropy, mutual information, or a Bayes/Fano "
        "error bound; use posterior_diagnostics_v2 for descriptive output"
    )


def _probability_matrix(
    values: np.ndarray,
    *,
    n_classes: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probability matrix must have shape (n>=1, k>=2)")
    if n_classes is not None and matrix.shape[1] != n_classes:
        raise ValueError(
            f"probability matrix has {matrix.shape[1]} columns, expected {n_classes}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("probability matrix must be finite")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("each probability row must sum to 1")
    return matrix


def _labels(y_true: np.ndarray, n_authors: int, n_rows: int) -> np.ndarray:
    if type(n_authors) is not int or n_authors < 2:
        raise ValueError("n_authors must be an integer >= 2")
    labels = np.asarray(y_true)
    if labels.ndim != 1 or n_rows < 1 or len(labels) != n_rows:
        raise ValueError(
            "y_true must be non-empty, one-dimensional, and align with probabilities"
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise TypeError("y_true must contain exact integer class indices")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= n_authors):
        raise ValueError("y_true class index is outside [0, n_authors)")
    return labels


def _row_entropy(probs: np.ndarray) -> np.ndarray:
    """Энтропия Шеннона (биты) построчно: (n,) для (n, k) постериоров."""
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log2(p)).sum(axis=1)


def prior_entropy(y_true: np.ndarray, n_authors: int) -> float:
    """H(A) по эмпирическому распределению меток авторов в выборке (биты)."""
    raw = np.asarray(y_true)
    n_rows = int(raw.shape[0]) if raw.ndim == 1 else -1
    y = _labels(raw, n_authors, n_rows)
    counts = np.bincount(y, minlength=n_authors).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mean_model_posterior_entropy_bits(prob_matrix: np.ndarray) -> float:
    """Return mean row entropy of a model probability matrix, in bits."""
    return float(np.mean(_row_entropy(_probability_matrix(prob_matrix))))


def prior_minus_model_posterior_entropy_bits(
    label_prior_entropy_bits: float,
    model_posterior_entropy_bits: float,
) -> float:
    """Return a descriptive arithmetic contrast, not mutual information."""
    values = np.asarray(
        [label_prior_entropy_bits, model_posterior_entropy_bits], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("entropy values must be finite")
    return float(label_prior_entropy_bits - model_posterior_entropy_bits)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """h(p) = -p·log2 p - (1-p)·log2(1-p), для p в [0,1] (поэлементно)."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _hinv_scalar(h_target: float) -> float:
    """Return the smaller numerical root of ``h(p) = h_target``.

    The root is only an entropy-equivalent Bernoulli probability.  It acquires
    no Bayes-error semantics from this numerical inversion.
    """
    if h_target <= 0:
        return 0.0
    if h_target >= 1.0:
        return 0.5
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _binary_entropy(np.array([mid]))[0] < h_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binary_posterior_entropy_equivalent_error(prob_AB: np.ndarray) -> float:
    """Re-express mean binary model-posterior entropy on the interval [0, .5].

    This is a model-confidence diagnostic only.  It is not the Bayes error, an
    error lower bound, or evidence that a pair is distinguishable.
    """
    matrix = _probability_matrix(prob_AB, n_classes=2)
    p = np.clip(matrix[:, 1], _EPS, 1 - _EPS)
    h2 = float(np.mean(_binary_entropy(p)))
    return _hinv_scalar(h2)


def posterior_diagnostics_v2(
    prob_matrix: np.ndarray,
    y_true: np.ndarray,
    n_authors: int,
    ece: float | None = None,
) -> Dict[str, Any]:
    """Return schema-v2 descriptive diagnostics for one model's probabilities."""
    matrix = _probability_matrix(prob_matrix, n_classes=n_authors)
    labels = _labels(y_true, n_authors, matrix.shape[0])
    prior = prior_entropy(labels, n_authors)
    posterior_entropy = mean_model_posterior_entropy_bits(matrix)
    contrast = prior_minus_model_posterior_entropy_bits(prior, posterior_entropy)
    pred = matrix.argmax(axis=1)
    empirical_err = float(1.0 - np.mean(pred == labels))
    out: Dict[str, Any] = {
        "schema_version": POSTERIOR_DIAGNOSTICS_SCHEMA,
        "semantics": "descriptive_model_posterior_only",
        "inferential_information_or_error_bound": False,
        "n_books": int(len(labels)),
        "n_authors": int(n_authors),
        "label_prior_entropy_bits": round(prior, 4),
        "maximum_label_entropy_bits": round(float(np.log2(n_authors)), 4),
        "mean_model_posterior_entropy_bits": round(posterior_entropy, 4),
        "prior_minus_model_posterior_entropy_bits": round(contrast, 4),
        "empirical_error": round(empirical_err, 4),
        "empirical_accuracy": round(1.0 - empirical_err, 4),
    }
    if ece is not None:
        if not np.isfinite(ece) or not 0.0 <= float(ece) <= 1.0:
            raise ValueError("ece must be finite and lie in [0, 1]")
        out["expected_calibration_error"] = round(float(ece), 4)
    return out


def pairwise_posterior_diagnostics_v2(
    prob_matrix: np.ndarray,
    y_true: np.ndarray,
    n_authors: int,
    pairs: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Return descriptive pair-restricted model-confidence diagnostics."""
    matrix = _probability_matrix(prob_matrix, n_classes=n_authors)
    labels = _labels(y_true, n_authors, matrix.shape[0])
    rows: List[Dict[str, Any]] = []
    for i, j in pairs:
        if type(i) is not int or type(j) is not int or i == j:
            raise ValueError("pairs must contain two distinct exact integer class indices")
        if not (0 <= i < n_authors and 0 <= j < n_authors):
            raise ValueError("pair class index is outside [0, n_authors)")
        mask = (labels == i) | (labels == j)
        if mask.sum() < 2:
            continue
        sub = matrix[mask][:, [i, j]]
        s = sub.sum(axis=1, keepdims=True)
        if np.any(s <= 0.0):
            raise ValueError("pair-restricted posterior has zero total mass")
        sub = sub / s
        entropy = mean_model_posterior_entropy_bits(sub)
        equivalent = binary_posterior_entropy_equivalent_error(sub)
        rows.append(
            {
                "schema_version": POSTERIOR_DIAGNOSTICS_SCHEMA,
                "a": int(i),
                "b": int(j),
                "n_books_pair": int(mask.sum()),
                "mean_pair_model_posterior_entropy_bits": round(entropy, 4),
                "posterior_entropy_equivalent_error": round(equivalent, 4),
                "inferential_information_or_error_bound": False,
            }
        )
    return rows


# Fail-closed compatibility stubs for the invalid v1 inferential names.
def conditional_entropy(prob_matrix: np.ndarray) -> float:
    del prob_matrix
    _withdrawn("conditional_entropy")


def mutual_information(H_A: float, H_A_given_F: float) -> float:
    del H_A, H_A_given_F
    _withdrawn("mutual_information")


def fano_floor(H_A_given_F: float, n_authors: int) -> float:
    del H_A_given_F, n_authors
    _withdrawn("fano_floor")


def binary_bayes_floor(prob_AB: np.ndarray) -> float:
    del prob_AB
    _withdrawn("binary_bayes_floor")


def fano_book_level(
    prob_matrix: np.ndarray,
    y_true: np.ndarray,
    n_authors: int,
    ece: float | None = None,
) -> Dict[str, float]:
    del prob_matrix, y_true, n_authors, ece
    _withdrawn("fano_book_level")


def pairwise_floor(
    prob_matrix: np.ndarray,
    y_true: np.ndarray,
    n_authors: int,
    pairs: Sequence[Tuple[int, int]],
) -> List[Dict]:
    del prob_matrix, y_true, n_authors, pairs
    _withdrawn("pairwise_floor")


# --- open-set / outsider: p(M_out | data) ---

def typicality_scores(prob_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-book «тичичность»: насколько постериор похож на уверенную in-set атрибуцию.

    Возвращает несколько сигналов (выше = «типичнее/увереннее» = скорее in-set):
      max_prob   — максимум постериора (главный сигнал);
      top2_mass  — суммарная масса топ-2 (устойчив к расплывчатости);
      margin     — разрыв топ1−топ2;
      neg_entropy — минус энтропия (бит).
    Outsider/замаскированный текст → низкая типичность (диффузный постериор).
    """
    p = np.clip(np.asarray(prob_matrix, dtype=np.float64), _EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    order = np.sort(p, axis=1)
    ent = _row_entropy(p)
    return {
        "max_prob": order[:, -1],
        "top2_mass": order[:, -1] + order[:, -2],
        "margin": order[:, -1] - order[:, -2],
        "neg_entropy": float(np.log2(p.shape[1])) - ent,   # 0 = равномерный, max = уверен
    }


def outsider_probability(score_in: np.ndarray, score_out: np.ndarray,
                         scores_query: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """P(M_out | score) через эмпирическое отношение плотностей (гистограммы + Лаплас).

    score: скаляр «выше = типичнее/более in-set» (напр. max_prob). По двум выборкам
    (in-set книг, outsider-книг) строим P(outsider|score) для query-книг. Возвращает
    массив p_out ∈ [0,1]. Честно: зависят от бина и от того, что считается «outsider»
    (здесь — hold-out автор, не замаскированный in-set — см. disguise в fano_disguise).
    """
    si = np.asarray(score_in, dtype=np.float64)
    so = np.asarray(score_out, dtype=np.float64)
    sq = np.asarray(scores_query, dtype=np.float64)
    lo = float(min(si.min(), so.min(), sq.min()))
    hi = float(max(si.max(), so.max(), sq.max()))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)

    def hist(s):
        h, _ = np.histogram(s, bins=edges)
        return h.astype(np.float64) + 1.0  # сглаживание Лапласа

    hin, hout = hist(si), hist(so)
    pin, pout = hin / hin.sum(), hout / hout.sum()
    prior_out = len(so) / (len(si) + len(so))
    prior_in = 1.0 - prior_out
    idx = np.clip(np.digitize(sq, edges) - 1, 0, n_bins - 1)
    num = prior_out * pout[idx]
    den = prior_in * pin[idx] + prior_out * pout[idx]
    return num / den


# ---------------------------------------------------------------------------
# Self-test: regression counterexamples (run: python -m stylo.eval.fano)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # Constant output contains no information about balanced labels, yet the
    # historical entropy contrast is large.  V2 reports the contrast only as a
    # model-output diagnostic and publishes no information/error bound.
    y = np.asarray([0, 1] * 20, dtype=int)
    constant = np.tile(np.asarray([0.999, 0.001]), (len(y), 1))
    report = posterior_diagnostics_v2(constant, y, 2)
    assert report["empirical_error"] == 0.5
    assert report["prior_minus_model_posterior_entropy_bits"] > 0.98
    assert report["inferential_information_or_error_bound"] is False
    assert not any(
        token in key.casefold()
        for key in report
        for token in ("mutual_information", "bayes", "fano", "floor")
    )

    pair_value = binary_posterior_entropy_equivalent_error(constant)
    assert pair_value < 0.01
    assert report["empirical_error"] == 0.5

    try:
        binary_bayes_floor(constant)
    except WithdrawnFanoSemanticsError:
        pass
    else:  # pragma: no cover - executable documentation
        raise AssertionError("withdrawn Bayes-floor API did not fail closed")

    print("posterior diagnostics v2 self-test OK:", report)


__all__ = [
    "POSTERIOR_DIAGNOSTICS_SCHEMA",
    "WithdrawnFanoSemanticsError",
    "binary_posterior_entropy_equivalent_error",
    "mean_model_posterior_entropy_bits",
    "outsider_probability",
    "pairwise_posterior_diagnostics_v2",
    "posterior_diagnostics_v2",
    "prior_entropy",
    "prior_minus_model_posterior_entropy_bits",
    "typicality_scores",
    # Withdrawn v1 names remain importable only to raise a typed error.
    "binary_bayes_floor",
    "conditional_entropy",
    "fano_book_level",
    "fano_floor",
    "mutual_information",
    "pairwise_floor",
]


if __name__ == "__main__":
    _self_test()
