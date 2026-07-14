"""B2 work-balanced runtime adapters (design §§3-4).

Three sklearn-compatible pieces that carry the work-balanced estimand into the fit path without
touching ``run_fold``:

* ``WorkLevelCountTransformer`` — a cloneable ``BaseEstimator`` step wrapping the signed
  ``WorkLevelVectorizer`` (count mode) so a work-level BoW vocab can live inside a ``Pipeline``
  (the bare vectorizer is not cloneable and collides with Pipeline's positional ``y``).
* ``WorkBalancedStyloPipeline`` / ``WorkBalancedBowPipeline`` — ``Pipeline`` subclasses that
  recompute fold-local ``work_sample_weights`` (sum ``W_train``), route ONLY the two step-scoped
  params, force ``class_weight=None`` in the classifier, and pin ``enable_metadata_routing=False``
  so both ambient routing modes yield the identical estimand.

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
