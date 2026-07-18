"""Work-level sparse vectorizer (P1 B1, design v3 §3).

A plain ``TfidfVectorizer`` on concatenated work documents is not enough: ``max_features``
ranks by raw corpus term frequency, so a long work keeps dominating, and concatenation
invents cross-chunk n-grams. This helper instead:

1. counts each CHUNK with the block's analyzer (no cross-chunk n-grams, no cap);
2. sums chunk rows into work × feature counts;
3. prunes by **work** document-frequency (``min_df_works``);
4. ranks surviving features by the **equal-weight mean of per-work relative frequencies**
   (denominator = ALL analyzer events of the work, summed over its chunks before pruning),
   then caps to ``max_features`` with a deterministic string tie-break;
5. fits a **work-level IDF** ``ln((W+1)/(df_w+1)) + 1``.

Invariance is conditional, not general. The fitted state is invariant to a different chunk
split **only for analyzers whose multiset of events is preserved by resegmentation** — in
practice only unigram (n=1) word/token features, whose per-work counts depend solely on the
work's total tokens. Every n-gram with n>1 (word *or* char) straddles chunk boundaries, so a
re-split drops the boundary-crossing grams and changes the feature set (e.g. word-bigrams of
``["a b c"]`` vs ``["a b", "c"]`` lose ``"b c"``). Hence the guarantee holds only while the
canonical chunk boundaries stay frozen. Transform runs on CHUNK rows using only the frozen
train state.
"""
from __future__ import annotations

import collections.abc as cabc
import numbers
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer

MODE_TFIDF = "tfidf"
MODE_RELATIVE = "relative"
MODE_COUNT = "count"
_MODES = frozenset({MODE_TFIDF, MODE_RELATIVE, MODE_COUNT})

# Shared work-balanced document-frequency threshold (design v3 D1-a): a feature must occur
# in at least this many distinct works to survive. Fixed, not a public knob, so it cannot be
# silently corrupted (bool/0/-1/NaN) into a different signed estimand.
MIN_DF_WORKS = 2


def _safe_reciprocal(x: np.ndarray) -> np.ndarray:
    """1/x with 0 where x==0 (no divide-by-zero warning)."""
    return np.divide(1.0, x, out=np.zeros_like(x, dtype=np.float64), where=x > 0)


def analyzer_event_counts(analyzer_params: dict, docs: Sequence[str]) -> np.ndarray:
    """Total analyzer events (ALL tokens, before any vocabulary pruning) per chunk — the R1
    denominator. Identical to :meth:`WorkLevelVectorizer._events`; shared so the pooled A3 FunctionWord
    corner divides by the same all-events count as A4 without fitting a work-level vectorizer."""
    analyze = CountVectorizer(**analyzer_params).build_analyzer()
    return np.array([len(analyze(d)) for d in docs], dtype=np.float64)


def relative_by_events(Xc: csr_matrix, events: np.ndarray) -> csr_matrix:
    """The exact A4 relative transform: per-chunk ``selected_counts / all_analyzer_events`` as
    ``diags(1/events) @ Xc`` (same floating op order as :meth:`WorkLevelVectorizer.transform`
    MODE_RELATIVE), never densified; a zero-event chunk yields a zero row (no NaN/warning)."""
    return (diags(_safe_reciprocal(events)) @ Xc.tocsr().astype(np.float64)).tocsr()


def validate_work_ids(groups: Sequence, n_expected: Optional[int] = None) -> list[str]:
    """The single B0 work-balanced groups contract, checked before any use.

    A valid ``groups`` is a **non-empty, ordered 1-D container with positional chunk→work
    semantics** — a ``Sequence`` (list/tuple/range) or a 1-D ``ndarray``, as produced by
    ``load_work_balanced_dataset`` (``f"{author}/{work}"``) — of non-empty string work ids.
    This rejects, fail-closed:

    * a bare ``str``/``bytes`` (a scalar masquerading as a per-chunk sequence);
    * a ``Mapping`` or ``Set`` — no positional order, so ``list()`` would silently invent a
      chunk→work assignment (dict yields keys; set order is unstable);
    * an arbitrary iterator/generator (single-use, no positional guarantee) and a 2-D array;
    * an empty container (a work-balanced fit needs at least one work id);
    * any non-string / empty / ``None`` id — so ``[1, True]`` or ``[1, 1.0]`` can no longer
      silently collapse two works into one (which would change ``W`` and the IDF), and no
      flavour of NaN (python/``numpy``/``Decimal``) can slip through as its own group.

    Returns a plain ``list[str]`` (``numpy.str_`` normalised to ``str``).
    """
    if groups is None:
        raise ValueError("groups must not be None for work-balanced fitting")
    if isinstance(groups, (str, bytes)):
        raise ValueError("groups must be a 1-D sequence of work ids, not a bare string")
    if isinstance(groups, np.ndarray):
        if groups.ndim != 1:
            raise ValueError(f"groups must be a 1-D sequence, got ndim={groups.ndim}")
    elif isinstance(groups, cabc.Mapping):
        raise ValueError("groups must be an ordered sequence, not a mapping")
    elif isinstance(groups, cabc.Set):
        raise ValueError("groups must be an ordered sequence, not a set (unstable order)")
    elif not isinstance(groups, cabc.Sequence):
        raise ValueError(
            f"groups must be an ordered 1-D sequence with positional order, got {type(groups).__name__}"
        )
    ids = list(groups)
    if not ids:
        raise ValueError("empty groups: a work-balanced fit needs at least one work id")
    if n_expected is not None and len(ids) != n_expected:
        raise ValueError(f"groups length {len(ids)} != docs length {n_expected}")
    for g in ids:
        if not isinstance(g, str) or not g:
            raise ValueError(f"each work id must be a non-empty str, got {g!r}")
    return [str(g) for g in ids]


def count_marks_presence(count, label: str) -> bool:
    """Work-DF presence test for the hand-rolled dict blocks (morphology/dependency).

    The canonical producer builds integer ``Counter`` values, so the contract is strict: only
    a non-``bool`` integer is accepted — ``0`` means **absent** (a ghost ``0`` must not count
    toward the ≥2-works threshold), a positive value means present, a negative value is
    rejected. Any float/``Decimal``/non-integral value (incl. every ``numpy`` float NaN/±inf,
    which are not python ``float`` subclasses) is rejected fail-closed rather than silently
    slipping through a type-specific finiteness check.
    """
    if isinstance(count, bool) or not isinstance(count, numbers.Integral):
        raise ValueError(f"{label} count must be a non-bool integer, got {count!r}")
    if count < 0:
        raise ValueError(f"{label} count must be non-negative, got {count!r}")
    return count > 0


def _factorize(groups: Sequence[str]) -> tuple[list, np.ndarray]:
    """Stable first-seen factorization of validated string work ids (no ``np.unique``).

    ``groups`` must already have passed :func:`validate_work_ids`; deterministic first-seen
    order keeps the works×chunks indicator (and thus W/IDF) reproducible.
    """
    order: dict = {}
    inv = np.empty(len(groups), dtype=np.intp)
    for i, g in enumerate(groups):
        if g not in order:
            order[g] = len(order)
        inv[i] = order[g]
    return list(order), inv


def _group_indicator(groups: Sequence) -> tuple[list, csr_matrix]:
    """Return (first-seen work ids, works×chunks 0/1 indicator matrix)."""
    work_ids, inv = _factorize(groups)
    n_chunks = len(inv)
    G = csr_matrix((np.ones(n_chunks), (inv, np.arange(n_chunks))), shape=(len(work_ids), n_chunks))
    return work_ids, G


def _positive_int(value, name: str, *, allow_none: bool = False):
    if allow_none and value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


class WorkLevelVectorizer:
    """Fit work-balanced vocabulary/IDF on chunk docs grouped by work; transform chunk rows."""

    def __init__(
        self,
        *,
        analyzer_params: dict,
        mode: str = MODE_TFIDF,
        max_features: Optional[int] = None,
        min_df_works: int = 2,
        vocabulary: Optional[Sequence[str]] = None,
        sublinear_tf: bool = True,
    ):
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.analyzer_params = dict(analyzer_params)
        self.mode = mode
        self.max_features = _positive_int(max_features, "max_features", allow_none=True)
        self.min_df_works = _positive_int(min_df_works, "min_df_works")
        self.vocabulary = None if vocabulary is None else list(vocabulary)
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None

    # ── fit ──────────────────────────────────────────────────────────────────
    def _events(self, docs: Sequence[str]) -> np.ndarray:
        return analyzer_event_counts(self.analyzer_params, docs)

    def fit(self, docs: Sequence[str], groups: Sequence) -> "WorkLevelVectorizer":
        docs = list(docs)
        if not docs:
            raise ValueError("empty training set: work-balanced fit needs at least one chunk")
        groups = validate_work_ids(groups, len(docs))
        full = CountVectorizer(**self.analyzer_params, vocabulary=self.vocabulary)
        X = full.fit_transform(docs).tocsr()                  # chunks × features, SPARSE
        names = np.asarray(full.get_feature_names_out())

        work_ids, G = _group_indicator(groups)
        W = len(work_ids)
        work_counts = (G @ X).tocsr()                         # works × features, SPARSE (never densified)
        events = self._events(docs)
        work_events = np.asarray(G @ events).ravel()          # works vector (dense, length W)
        df_w = work_counts.getnnz(axis=0).astype(np.int64)    # per-feature #works with a nonzero count

        if self.vocabulary is None:
            # equal-work mean relative-TF per feature = (1/W) * colsum(rowscale(work_counts, 1/work_events))
            scaled = diags(_safe_reciprocal(work_events)) @ work_counts   # SPARSE works × features
            mean_rel = np.asarray(scaled.sum(axis=0)).ravel() / W          # dense length n_features
            keep = np.flatnonzero(df_w >= self.min_df_works)
            order = sorted(keep, key=lambda f: (-mean_rel[f], names[f]))
            if self.max_features is not None:
                order = order[: self.max_features]
            selected = order
        else:
            selected = list(range(len(names)))                # fixed list: keep all, no prune/cap

        selected = sorted(selected, key=lambda j: names[j])   # deterministic vocab order
        if not selected:
            raise ValueError(
                f"empty vocabulary after work-DF pruning (min_df_works={self.min_df_works}); "
                "lower min_df_works or check the corpus"
            )
        self.vocabulary_ = {names[j]: i for i, j in enumerate(selected)}
        self.idf_ = np.log((W + 1.0) / (df_w[selected] + 1.0)) + 1.0
        self._feature_names = [names[j] for j in selected]
        return self

    # ── transform ─────────────────────────────────────────────────────────────
    def transform(self, docs: Sequence[str]) -> csr_matrix:
        if not self.vocabulary_:
            raise RuntimeError("fit before transform")
        docs = list(docs)
        cv = CountVectorizer(**self.analyzer_params, vocabulary=self.vocabulary_)
        Xc = cv.transform(docs).astype(np.float64)            # chunks × selected features
        if self.mode == MODE_COUNT:
            return Xc.tocsr()
        if self.mode == MODE_RELATIVE:
            return relative_by_events(Xc, self._events(docs))
        # tfidf: (sublinear) tf * idf, then row L2
        tf = Xc.tocsr()
        if self.sublinear_tf:
            tf = tf.copy()
            tf.data = 1.0 + np.log(tf.data)
        weighted = tf @ diags(self.idf_)
        norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
        return (diags(_safe_reciprocal(norms)) @ weighted).tocsr()

    def transform_grouped(self, docs: Sequence[str], groups: Sequence) -> tuple[list, csr_matrix]:
        """One row per work: ``Σ(selected counts over work) / Σ(ALL analyzer events over work)``.

        The signed Delta estimand (design §3) sums counts and token totals across a work's chunks
        **before** dividing — not the mean of per-chunk relative frequencies (the two differ when
        chunk lengths vary). Denominator counts every analyzer token (OOV/pruned included), using
        the FROZEN train vocabulary. Work order = first-seen. Never densifies works×vocab.
        """
        if not self.vocabulary_:
            raise RuntimeError("fit before transform_grouped")
        docs = list(docs)
        groups = validate_work_ids(groups, len(docs))
        cv = CountVectorizer(**self.analyzer_params, vocabulary=self.vocabulary_)
        Xc = cv.transform(docs).astype(np.float64)            # chunks × selected feats (SPARSE)
        events = self._events(docs)                           # all analyzer tokens per chunk
        work_ids, G = _group_indicator(groups)                # works × chunks 0/1
        work_counts = (G @ Xc).tocsr()                        # works × feats, sum of selected counts
        work_events = np.asarray(G @ events).ravel()          # works vector, sum of all tokens
        rel = diags(_safe_reciprocal(work_events)) @ work_counts
        return work_ids, rel.tocsr()

    def feature_names(self) -> list[str]:
        return list(self._feature_names)
