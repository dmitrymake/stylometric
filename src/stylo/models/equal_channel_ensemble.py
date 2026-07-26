"""Class-complete equal fusion of the established stylometric channels.

This estimator deliberately has no inner OOF matrix, learned calibration, model
selection, or meta-classifier. Under its frozen evaluation-only contract ``fit``
stores the outer training fold and every ``predict_proba`` lazily refits each
channel on that fold; it is therefore non-deployable and non-serializable. The
class-complete margins receive fixed identity-softmax and are averaged equally.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import softmax

from ..domain.work_weighting import AblationConfig, CHUNK_WEIGHTED_LEGACY
from .stacked_clf import StackedChannelClassifier


class EqualChannelEnsembleClassifier(StackedChannelClassifier):
    """Evaluation-only fixed fusion over the W/F/R-aware channel mechanics."""

    needs_groups = True

    def __init__(
        self,
        cfg,
        svc_c: float = 1.0,
        seed: int = 42,
        training_weighting: str = CHUNK_WEIGHTED_LEGACY,
        *,
        ablation: Optional[AblationConfig] = None,
    ):
        # Inheritance is mechanics-only: the overridden fit/predict path never
        # constructs OOF scores, calibrators, or a meta learner.
        super().__init__(
            cfg,
            inner_folds=2,
            svc_c=svc_c,
            meta_c=1.0,
            seed=seed,
            training_weighting=training_weighting,
            ablation=ablation,
        )
        # Do not expose the legacy stack's calibration passport or ``mode_``:
        # this estimator has neither calibration selection nor a selectable
        # fusion mode. Its sole public mechanism record is ``fusion_passport_``.
        del self.passport_
        self.fusion_passport_ = {}

    def fit(self, texts, y, groups=None):
        if groups is None:
            raise ValueError(
                "stylo_equal_channels_v1 requires groups (book id for every chunk)"
            )
        texts = list(texts)
        y = np.asarray(y)
        groups = np.asarray(groups)
        if y.ndim != 1 or groups.ndim != 1:
            raise ValueError("y and groups must be one-dimensional")
        if len(texts) == 0 or len(texts) != len(y) or len(y) != len(groups):
            raise ValueError(
                "texts, y, and groups must have the same non-zero row count"
            )

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("stylo_equal_channels_v1 requires at least two classes")
        self._classes_sorted = self.classes_
        self._train_texts = texts
        self._train_y = y
        self._train_groups = groups

        channels = self._channels()
        if not channels:
            raise ValueError("stylo_equal_channels_v1 has no enabled channels")
        self._channel_names = list(channels)
        equal_weight = 1.0 / len(self._channel_names)
        self.fusion_passport_ = {
            "schema": "stylo.equal_channel_ensemble.fusion.v1",
            "estimator_spec": "stylo_equal_channels_v1",
            "channels": list(self._channel_names),
            "axes": {
                "W": bool(self._weights_on),
                "F": bool(self._feature_on),
                "R": bool(self._relative_fw_on),
            },
            "training_weighting": self.training_weighting,
            "channel_score_transform": {
                "method": "identity_softmax",
                "temperature": 1.0,
                "learned": False,
            },
            "fusion": {
                "method": "equal_arithmetic_mean",
                "learned": False,
                "weights": {
                    name: equal_weight for name in self._channel_names
                },
            },
            "oof": {"used": False},
            "calibration": {"learned": False},
            "meta_classifier": {"present": False},
        }
        return self

    def predict_proba(self, texts) -> np.ndarray:
        if not self.fusion_passport_:
            raise ValueError("stylo_equal_channels_v1 must be fitted before prediction")
        texts = list(texts)
        channels = self._channels()
        if list(channels) != self._channel_names:
            raise RuntimeError("channel inventory changed after fit")

        n_cls = len(self.classes_)
        sample_weight = self._fold_weights(self._train_y, self._train_groups)
        channel_probabilities = []
        for name in self._channel_names:
            if self._feature_on:
                Xtr, Xte = channels[name](
                    self._train_texts, texts, self._train_groups,
                )
            else:
                Xtr, Xte = channels[name](self._train_texts, texts)
            classifier = self._svc().fit(
                Xtr, self._train_y, sample_weight=sample_weight,
            )
            scores = self._decision_full(
                classifier, Xte, n_cls, classifier.classes_,
            )
            channel_probabilities.append(softmax(scores, axis=1))

        probabilities = np.mean(
            np.stack(channel_probabilities, axis=0), axis=0,
        )
        if (
            probabilities.shape != (len(texts), n_cls)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        ):
            raise RuntimeError(
                "stylo_equal_channels_v1 produced malformed probabilities"
            )
        return probabilities
