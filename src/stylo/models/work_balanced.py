"""B2 work-balanced runtime adapters (design §§3-4).

Three sklearn-compatible pieces that carry the work-balanced estimand into the fit path without
touching ``run_fold``:

* ``WorkLevelCountTransformer`` — a cloneable ``BaseEstimator`` step wrapping the signed
  ``WorkLevelVectorizer`` (count mode) so a work-level BoW vocab can live inside a ``Pipeline``
  (the bare vectorizer is not cloneable and collides with Pipeline's positional ``y``).
* ``WorkBalancedStyloPipeline`` / ``WorkBalancedBowPipeline`` (A4) — ``Pipeline`` subclasses that
  recompute fold-local ``work_sample_weights`` (sum ``W_train``), route ONLY the two step-scoped
  params, force ``class_weight=None`` in the classifier, and pin ``enable_metadata_routing=False``
  so both ambient routing modes yield the identical estimand.

Audit-only wrappers for the paired-audit off-diagonal cells (loss/feature axes decoupled):
* ``WeightsOnly{Stylo,Bow}Pipeline`` (B4-B inc 2, A1) — the work-balanced loss on a legacy feature side;
* ``{FeatureState,RelativeFw}StyloPipeline`` / ``FeatureStateBowPipeline`` (B4-B inc 3, A2/A3) — the
  feature-state (F) and/or relative-FW (R) axis on the UNCHANGED A0 loss (canonical ``class_weight``
  frozen against ``set_params``/clone tampering).

All estimand math is delegated to already-signed B1 code — nothing new is fitted here.
"""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..eval.work_weighting import work_sample_weights
from ..features.work_vectorizer import (MODE_COUNT, WorkLevelVectorizer,
                                        validate_work_ids)


class WorkLevelCountTransformer(BaseEstimator, TransformerMixin):
    """Cloneable Pipeline step: work-level count vocabulary (delegates to WorkLevelVectorizer)."""

    def __init__(self, analyzer_params=None, max_features=None, min_df_works=2, vocabulary=None):
        self.analyzer_params = analyzer_params
        self.max_features = max_features
        self.min_df_works = min_df_works
        self.vocabulary = vocabulary

    def fit(self, X, y=None, *, groups):
        self._wv = WorkLevelVectorizer(
            analyzer_params=self.analyzer_params or {}, mode=MODE_COUNT,
            max_features=self.max_features, min_df_works=self.min_df_works,
            vocabulary=self.vocabulary, sublinear_tf=False,
        )
        self._wv.fit(X, groups)
        return self

    def fit_transform(self, X, y=None, *, groups):
        return self.fit(X, y, groups=groups).transform(X)

    def transform(self, X):
        if getattr(self, "_wv", None) is None:
            raise RuntimeError("fit before transform")
        return self._wv.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self._wv.feature_names(), dtype=object)


class _WorkBalancedPipeline(Pipeline):
    """Base for work-balanced pipelines: internal fold-local weights + fixed step routing."""

    needs_groups = True
    _GROUPS_PARAM: str = ""
    _WEIGHT_PARAM: str = ""

    def fit(self, X, y=None, *, groups, **fit_params):
        if y is None:
            raise ValueError("work_balanced fit needs y")
        X = list(X)
        groups = validate_work_ids(groups, len(X))
        reserved = {self._GROUPS_PARAM, self._WEIGHT_PARAM, "sample_weight"} & set(fit_params)
        if reserved:
            raise ValueError(f"groups/weights are computed internally; do not pass {sorted(reserved)}")
        weights = work_sample_weights(y, groups)                       # fold-local, sum = W_train
        # Force class_weight=None on the classifier step regardless of any prior set_params:
        # work weights already equalize class mass, so class_weight must not double-count.
        clf_step = self._WEIGHT_PARAM.split("__", 1)[0]
        clf = self.named_steps[clf_step]
        if "class_weight" in clf.get_params():
            clf.set_params(class_weight=None)
        routed = {self._GROUPS_PARAM: groups, self._WEIGHT_PARAM: weights, **fit_params}
        # Pin routing OFF so the step-scoped params behave identically under either ambient mode.
        with sklearn.config_context(enable_metadata_routing=False):
            return super().fit(X, y, **routed)


class WorkBalancedStyloPipeline(_WorkBalancedPipeline):
    _GROUPS_PARAM = "vectorizer__groups"
    _WEIGHT_PARAM = "classifier__sample_weight"


class WorkBalancedBowPipeline(_WorkBalancedPipeline):
    _GROUPS_PARAM = "bow__groups"
    _WEIGHT_PARAM = "lr__sample_weight"


def build_bow_lr_work_balanced() -> "WorkBalancedBowPipeline":
    """B-full work-balanced BoW: work-level count vocab + class_weight=None + fold-local weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler
    return WorkBalancedBowPipeline([
        ("bow", WorkLevelCountTransformer(
            analyzer_params={"analyzer": "word", "ngram_range": (1, 2),
                             "token_pattern": r"(?u)\b\w+\b", "lowercase": True},
            max_features=20000, min_df_works=2)),
        ("scaler", MaxAbsScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight=None, solver="lbfgs")),
    ])


# ── B4-B increment 2: weights-only A1 (work-balanced LOSS, strictly legacy A0 feature state) ──
class _WeightsOnlyPipeline(Pipeline):
    """Audit-only base: the work-balanced training loss on top of an unchanged legacy (A0) feature side.

    Identical loss wiring to ``_WorkBalancedPipeline`` — fold-local ``work_sample_weights`` (sum
    ``W_train``), forced ``class_weight=None`` on the classifier, and pinned
    ``enable_metadata_routing=False`` — but it routes ONLY the classifier ``sample_weight``. Crucially
    the work ``groups`` are NEVER routed into the vectorizer, so the fitted vocabulary/DF/IDF stay the
    exact chunk-pooled A0 projection (the vectorizer step is the plain legacy vectorizer, not the
    work-level one). Any caller param that could smuggle in the F/R axes (``…__groups``) or replace the
    signed W estimand (``…__sample_weight``) fails closed."""

    needs_groups = True
    _WEIGHT_PARAM: str = ""
    _GROUPS_RESERVED: str = ""

    def _reject_reserved(self, fit_params) -> None:
        reserved = {"sample_weight", "groups", self._WEIGHT_PARAM, self._GROUPS_RESERVED}
        hit = sorted(k for k in fit_params
                     if k in reserved or k.endswith("__sample_weight") or k.endswith("__groups"))
        if hit:
            raise ValueError(
                f"weights-only A1 computes weights internally and never fits F/R; do not pass {hit}")

    def fit(self, X, y=None, *, groups, **fit_params):
        if y is None:
            raise ValueError("weights-only A1 fit needs y")
        X = list(X)
        groups = validate_work_ids(groups, len(X))
        self._reject_reserved(fit_params)
        weights = work_sample_weights(y, groups)                       # fold-local, sum = W_train
        # Force class_weight=None regardless of any prior set_params: work weights already equalize
        # class mass, so a "balanced" class_weight would double-count (§3.1: no double weighting).
        clf_step = self._WEIGHT_PARAM.split("__", 1)[0]
        clf = self.named_steps[clf_step]
        if "class_weight" in clf.get_params():
            clf.set_params(class_weight=None)
        routed = {self._WEIGHT_PARAM: weights, **fit_params}           # groups deliberately NOT routed
        with sklearn.config_context(enable_metadata_routing=False):
            return super().fit(X, y, **routed)


class WeightsOnlyStyloPipeline(_WeightsOnlyPipeline):
    _WEIGHT_PARAM = "classifier__sample_weight"
    _GROUPS_RESERVED = "vectorizer__groups"


class WeightsOnlyBowPipeline(_WeightsOnlyPipeline):
    _WEIGHT_PARAM = "lr__sample_weight"
    _GROUPS_RESERVED = "bow__groups"


def build_bow_lr_weights_only() -> "WeightsOnlyBowPipeline":
    """A1 BoW: the EXACT legacy A0 ``CountVectorizer`` vocab/counts + fold-local work weights +
    ``class_weight=None`` (the only difference from A0 is the work-balanced loss, not the features)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler
    return WeightsOnlyBowPipeline([
        ("bow", CountVectorizer(max_features=20000, ngram_range=(1, 2),
                                token_pattern=r"(?u)\b\w+\b")),
        ("scaler", MaxAbsScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight=None, solver="lbfgs")),
    ])


# ── B4-B increment 3: feature-state A2 / relative-FW A3 (F/R axis, UNCHANGED A0 loss) ──
def _cw_token(v):
    """Canonicalize a class_weight to a comparison-safe token built ONLY from exact-type primitives —
    never invoking a user ``__eq__``. Recognizes exactly ``None`` / a plain ``str`` / a plain ``dict``
    of {exact int|str : exact int|float}; anything else (subclass, fake numeric, str-subclass) becomes a
    unique INVALID token that matches nothing, so a forged weight cannot masquerade as the A0 value."""
    if v is None:
        return ("none",)
    if type(v) is str:
        return ("str", "".join(v))                     # "".join forces a plain str, defusing subclasses
    if type(v) is dict:
        parts = []
        for k in v:
            val = v[k]
            if type(k) not in (int, str) or type(val) not in (int, float):
                return ("invalid",)                    # non-canonical key/value type -> never matches
            parts.append((type(k).__name__, k, type(val).__name__, val))
        return ("dict", tuple(sorted(parts, key=lambda p: (p[0], str(p[1])))))
    return ("invalid",)


class _FeatureAxisPipeline(Pipeline):
    """Audit-only base for A2/A3: an F (feature-state) and/or R (relative-FW) axis on top of the
    UNCHANGED A0 loss/calibration — no ``sample_weight``, the legacy ``class_weight`` from the A0
    builder. It routes ONLY the feature ``groups`` (A2, F on) or nothing (A3, F off); the R policy is
    baked into the vectorizer construction, not here. Every caller fit-param (weights, class-weight,
    groups) is rejected, AND the classifier ``class_weight`` is frozen at construction and re-checked
    at fit, so neither the signed A0 loss nor the F/R estimand can be altered via ``set_params``."""

    needs_groups = True
    _GROUPS_PARAM: str = ""
    _ROUTE_GROUPS: bool = True

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        # The A0-loss class_weight is a READ-ONLY canonical token snapshotted ONCE from the pristine step
        # at construction and stored OUTSIDE get_params (so set_params cannot reach it). ``clone`` and
        # ``pickle`` preserve the ORIGINAL token (see __sklearn_clone__), so a tamper→clone/joblib→fit is
        # caught by the fit-time token check.
        object.__setattr__(self, "_a0_cw_token", _cw_token(self._current_class_weight()))

    def _current_class_weight(self):
        clf = self.steps[-1][1]
        params = clf.get_params()
        return params["class_weight"] if "class_weight" in params else object()  # no cw -> unique -> mismatch

    def __sklearn_clone__(self):
        # a clone is UNFITTED but must carry the ORIGINAL canonical token, NOT re-derive it from the
        # (possibly tampered) cloned steps — otherwise clone would launder a class_weight change.
        cloned = type(self)([(name, sklearn.clone(est)) for name, est in self.steps],
                            memory=self.memory, verbose=self.verbose)
        object.__setattr__(cloned, "_a0_cw_token", self._a0_cw_token)
        return cloned

    def fit(self, X, y=None, *, groups, **fit_params):
        if y is None:
            raise ValueError("feature-axis A2/A3 fit needs y")
        X = list(X)
        groups = validate_work_ids(groups, len(X))
        if fit_params:                                     # A0 loss, internal routing: no caller params
            raise ValueError(
                f"A2/A3 keeps the A0 loss and routes groups internally; do not pass {sorted(fit_params)}")
        token = getattr(self, "_a0_cw_token", None)
        if token is None or _cw_token(self._current_class_weight()) != token:
            raise ValueError(
                "A2/A3 keeps the A0 loss; the classifier class_weight must equal the canonical A0 value "
                "(set_params / clone-of-tampered / forged-weight tampering rejected)")
        routed = {self._GROUPS_PARAM: groups} if self._ROUTE_GROUPS else {}
        # Pin routing OFF so the step-scoped groups param behaves identically under either ambient mode.
        with sklearn.config_context(enable_metadata_routing=False):
            return super().fit(X, y, **routed)


class FeatureStateStyloPipeline(_FeatureAxisPipeline):        # A2: F on (work-vocab), R off (raw FW)
    _GROUPS_PARAM = "vectorizer__groups"
    _ROUTE_GROUPS = True


class RelativeFwStyloPipeline(_FeatureAxisPipeline):          # A3: F off (pooled), R on (relative FW)
    _GROUPS_PARAM = "vectorizer__groups"
    _ROUTE_GROUPS = False


class FeatureStateBowPipeline(_FeatureAxisPipeline):          # A2 BoW: work-level count vocab, A0 loss
    _GROUPS_PARAM = "bow__groups"
    _ROUTE_GROUPS = True


def build_bow_lr_feature_state() -> "FeatureStateBowPipeline":
    """A2 BoW: the A4 work-level count vocabulary (F) with the A0 loss (``class_weight='balanced'``,
    no sample_weight) — features move, the loss does not."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler
    return FeatureStateBowPipeline([
        ("bow", WorkLevelCountTransformer(
            analyzer_params={"analyzer": "word", "ngram_range": (1, 2),
                             "token_pattern": r"(?u)\b\w+\b", "lowercase": True},
            max_features=20000, min_df_works=2)),
        ("scaler", MaxAbsScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),   # A0 loss
    ])
