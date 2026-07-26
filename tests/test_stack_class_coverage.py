"""Regression contract for class-complete stylometric channel fusion.

The historical stack filled absent inner-fold classes with ``-30``. A learned
Platt calibrator could then converge cleanly on that artefact and invert valid
outer-fold scores. These tests pin the repair and the separate exploratory
``stylo_equal_channels_v1`` estimator without reopening confirmatory cells.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning

from stylo.config import load_config
from stylo.eval.calibration import choose_calibrator
from stylo.eval.lobo import make_factory, make_factory_for_ablation
from stylo.eval.work_weighting import (
    CHUNK_WEIGHTED_LEGACY,
    FEATURE_STATE_ONLY_ABLATION,
    RELATIVE_FW_ONLY_ABLATION,
    WEIGHTS_ONLY_ABLATION,
    WORK_BALANCED,
)
from stylo.models import stacked_clf as sc
from stylo.models.equal_channel_ensemble import EqualChannelEnsembleClassifier


CFG = load_config()


def _three_class_panel():
    texts = np.asarray(
        ["0:a0", "0:a1", "1:b0", "1:b1", "2:c0", "2:c1"],
        dtype=object,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=int)
    groups = np.asarray(
        ["a/w0", "a/w1", "b/w0", "b/w1", "c/w0", "c/w1"],
        dtype=object,
    )
    return texts, labels, groups


def _score_matrix(texts):
    rows = []
    for text in texts:
        label = int(str(text).split(":", 1)[0])
        row = np.full(3, -1.0)
        row[label] = 2.0
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _encoded_channel(train_texts, test_texts, *_optional_groups):
    return _score_matrix(train_texts), _score_matrix(test_texts)


class _EchoDecisionClassifier:
    """Treat the synthetic channel matrix itself as the decision matrix."""

    def fit(self, matrix, labels, sample_weight=None):
        del matrix, sample_weight
        self.classes_ = np.unique(labels)
        return self

    def decision_function(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        return matrix[:, self.classes_]


class _ClassCompleteSplitter:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def split(self, matrix, labels, groups):
        del matrix, labels, groups
        return iter((
            (np.asarray([0, 2, 4]), np.asarray([1, 3, 5])),
            (np.asarray([1, 3, 5]), np.asarray([0, 2, 4])),
        ))


class _ClassIncompleteSplitter:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def split(self, matrix, labels, groups):
        del matrix, labels, groups
        return iter((
            (np.asarray([0, 2]), np.asarray([1, 3, 4, 5])),
            (np.asarray([1, 3, 4, 5]), np.asarray([0, 2])),
        ))


class _NonComplementSplitter:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def split(self, matrix, labels, groups):
        del matrix, labels, groups
        return iter((
            (np.asarray([0, 2, 4]), np.asarray([1, 3])),
            (np.asarray([1, 3, 5]), np.asarray([0, 2])),
        ))


def test_absent_class_sentinel_can_make_converged_platt_invert_valid_scores():
    labels = np.repeat(np.arange(3), 30)
    poisoned_oof = np.ones((len(labels), 3), dtype=float)
    poisoned_oof[np.arange(len(labels)), labels] = -30.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrator, passport = choose_calibrator(poisoned_oof, labels, seed=42)

    assert passport["method"] == "platt"
    assert passport["heldout_nll"]["platt"] < 0.001
    assert passport["heldout_nll"]["identity"] > 20.0
    assert not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )

    valid_outer_scores = np.full((3, 3), -1.0)
    np.fill_diagonal(valid_outer_scores, 2.0)
    calibrated = calibrator(valid_outer_scores)
    assert np.all(np.diag(calibrated) < calibrated.max(axis=1))


def test_legacy_stack_rejects_incomplete_inner_classes_before_channels(
    monkeypatch,
):
    texts, labels, groups = _three_class_panel()
    calls = {"channels": 0, "calibration": 0}

    def forbidden_channels(*args, **kwargs):
        del args, kwargs
        calls["channels"] += 1
        raise AssertionError("class-incomplete split reached channel construction")

    def forbidden_calibration(*args, **kwargs):
        del args, kwargs
        calls["calibration"] += 1
        raise AssertionError("class-incomplete scores reached calibration")

    monkeypatch.setattr(sc, "StratifiedGroupKFold", _ClassIncompleteSplitter)
    monkeypatch.setattr(sc, "make_channels", forbidden_channels)
    monkeypatch.setattr(sc, "choose_calibrator", forbidden_calibration)

    with pytest.raises(sc.StackClassCoverageError) as caught:
        sc.StackedChannelClassifier(CFG, inner_folds=2).fit(
            texts, labels, groups=groups,
        )

    assert caught.value.stage == "inner_oof"
    assert caught.value.report["class_coverage_complete"] is False
    assert calls == {"channels": 0, "calibration": 0}


def test_noncomplement_split_is_rejected_before_channels(monkeypatch):
    texts, labels, groups = _three_class_panel()
    calls = {"channels": 0}

    def forbidden_channels(*args, **kwargs):
        del args, kwargs
        calls["channels"] += 1
        raise AssertionError("invalid split reached channel construction")

    monkeypatch.setattr(sc, "StratifiedGroupKFold", _NonComplementSplitter)
    monkeypatch.setattr(sc, "make_channels", forbidden_channels)

    with pytest.raises(sc.StackClassCoverageError) as caught:
        sc.StackedChannelClassifier(CFG, inner_folds=2).fit(
            texts, labels, groups=groups,
        )

    assert caught.value.stage == "inner_oof"
    assert caught.value.report["structure_complete"] is False
    assert caught.value.report["missing_validation_row_indices"] == [4, 5]
    assert calls == {"channels": 0}


def _split_report(splits, *, groups=None):
    _, labels, default_groups = _three_class_panel()
    return sc.build_inner_split_preflight_report(
        splits,
        labels,
        np.arange(3),
        groups=default_groups if groups is None else groups,
        expected_split_count=2,
    )


def test_split_preflight_rejects_train_validation_overlap():
    report = _split_report([
        (np.asarray([0, 2, 4]), np.asarray([0, 1, 3, 5])),
        (np.asarray([1, 3, 5]), np.asarray([2, 4])),
    ])
    assert report["validation_exactly_once"] is True
    assert report["splits"][0]["overlapping_row_indices"] == [0]
    assert report["complete"] is False


def test_split_preflight_reports_duplicate_validation_index_exactly():
    report = _split_report([
        (np.asarray([0, 2, 4]), np.asarray([1, 1, 3, 5])),
        (np.asarray([1, 3, 5]), np.asarray([0, 2, 4])),
    ])
    assert report["repeated_validation_row_indices"] == [1]
    assert report["splits"][0]["validation_indices_unique"] is False
    assert report["complete"] is False


def test_split_preflight_rejects_global_validation_reuse():
    repeated_fold = (
        np.asarray([0, 2, 4]),
        np.asarray([1, 3, 5]),
    )
    report = _split_report([repeated_fold, repeated_fold])
    assert all(split["structure_complete"] for split in report["splits"])
    assert report["missing_validation_row_indices"] == [0, 2, 4]
    assert report["repeated_validation_row_indices"] == [1, 3, 5]
    assert report["complete"] is False


def test_split_preflight_rejects_same_work_on_both_sides():
    groups = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    report = _split_report(
        [
            (np.asarray([0, 2, 4]), np.asarray([1, 3, 5])),
            (np.asarray([1, 3, 5]), np.asarray([0, 2, 4])),
        ],
        groups=groups,
    )
    assert report["splits"][0]["exact_row_complement"] is True
    assert report["splits"][0]["overlapping_groups"] == ["a", "b", "c"]
    assert report["complete"] is False


def test_decision_alignment_rejects_partial_class_map_before_scoring():
    estimator = sc.StackedChannelClassifier(CFG)
    estimator.classes_ = np.asarray([0, 1, 2])
    estimator._classes_sorted = estimator.classes_

    class _PartialClassifier:
        classes_ = np.asarray([0, 2])

        def decision_function(self, matrix):
            del matrix
            raise AssertionError("partial class map reached decision scoring")

    with pytest.raises(sc.StackClassCoverageError):
        estimator._decision_full(
            _PartialClassifier(), np.zeros((1, 1)), 3, np.asarray([0, 2]),
        )


def test_decision_alignment_rejects_nonfinite_scores():
    estimator = sc.StackedChannelClassifier(CFG)
    estimator.classes_ = np.asarray([0, 1, 2])
    estimator._classes_sorted = estimator.classes_

    class _NonfiniteClassifier:
        classes_ = np.asarray([0, 1, 2])

        def decision_function(self, matrix):
            return np.full((len(matrix), 3), np.nan)

    with pytest.raises(ValueError, match="finite"):
        estimator._decision_full(
            _NonfiniteClassifier(), np.zeros((1, 1)), 3, np.arange(3),
        )


def _forbid_calibrator_fit(monkeypatch):
    import stylo.eval.calibration as calibration

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid calibration split reached _fit_method")

    monkeypatch.setattr(calibration, "_fit_method", forbidden)
    return calibration


def test_legacy_calibration_rejects_missing_fit_class_before_method(monkeypatch):
    calibration = _forbid_calibrator_fit(monkeypatch)
    labels = np.repeat(np.arange(3), 2)
    scores = np.eye(3)[labels] * 3.0

    with pytest.raises(calibration.CalibrationClassCoverageError) as caught:
        calibration.choose_calibrator(
            scores, labels, methods=("identity",), seed=4,
        )
    assert caught.value.stage == "selection_split"
    assert caught.value.report["missing_fit_classes"] == [2]


def test_legacy_calibration_rejects_missing_validation_class_before_method(
    monkeypatch,
):
    calibration = _forbid_calibrator_fit(monkeypatch)
    labels = np.repeat(np.arange(3), 2)
    scores = np.eye(3)[labels] * 3.0

    with pytest.raises(calibration.CalibrationClassCoverageError) as caught:
        calibration.choose_calibrator(
            scores, labels, methods=("identity",), seed=0,
        )
    assert caught.value.stage == "selection_split"
    assert caught.value.report["missing_fit_classes"] == []
    assert caught.value.report["missing_validation_classes"] == [0]


@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        (np.asarray([[1.0, np.nan], [0.0, 1.0]]), np.asarray([0, 1])),
        (np.eye(2), np.asarray([0.0, 1.0])),
        (np.zeros((4, 3)), np.asarray([0, 0, 1, 1])),
    ],
)
def test_legacy_calibration_validates_inputs_before_method(
    monkeypatch, scores, labels,
):
    calibration = _forbid_calibrator_fit(monkeypatch)
    with pytest.raises(ValueError):
        calibration.choose_calibrator(
            scores, labels, methods=("identity",), seed=0,
        )


@pytest.mark.parametrize("score_dtype", [np.int64, np.float32, np.float64])
def test_valid_legacy_calibration_matches_historical_path(score_dtype):
    import stylo.eval.calibration as calibration

    labels = np.tile(np.arange(3), 20)
    scores = (np.eye(3)[labels] * 3).astype(score_dtype)
    methods = ("identity", "temperature")
    seed = 7

    indices = np.random.RandomState(seed).permutation(len(labels))
    cut = max(1, len(labels) // 3)
    validation_index, fit_index = indices[:cut], indices[cut:]
    reference_losses = {}
    for method in methods:
        calibrator, _ = calibration._fit_method(
            method, scores[fit_index], labels[fit_index], seed,
        )
        reference_losses[method] = calibration.nll(
            calibrator(scores[validation_index]), labels[validation_index],
        )
    reference_method = min(reference_losses, key=reference_losses.get)
    reference_calibrator, reference_params = calibration._fit_method(
        reference_method, scores, labels, seed,
    )
    reference_passport = {
        "method": reference_method,
        "params": reference_params,
        "selection": "held-out треть OOF",
        "heldout_nll": {
            method: round(value, 4)
            for method, value in reference_losses.items()
        },
    }

    actual_calibrator, actual_passport = calibration.choose_calibrator(
        scores, labels, methods=methods, seed=seed,
    )
    assert actual_passport == reference_passport
    np.testing.assert_array_equal(
        actual_calibrator(scores), reference_calibrator(scores),
    )


def _grouped_calibration_panel():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.eye(2)[labels] * 2.0
    groups = np.asarray(
        ["a/w0", "a/w0", "a/w1", "a/w1",
         "b/w0", "b/w0", "b/w1", "b/w1"],
        dtype=object,
    )
    return scores, labels, groups


def test_grouped_split_structure_is_checked_before_identity_fallback(monkeypatch):
    import stylo.eval.calibration as calibration

    class _MalformedSplitter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def split(self, matrix, labels, groups):
            del matrix, labels, groups
            return [(
                np.asarray([2, 3, 4, 5, 6, 7]),
                np.asarray([0, 1]),
            )]

    def forbidden_identity(*args, **kwargs):
        del args, kwargs
        raise AssertionError("malformed split reached identity fallback")

    monkeypatch.setattr(calibration, "StratifiedGroupKFold", _MalformedSplitter)
    monkeypatch.setattr(calibration, "_identity_disabled", forbidden_identity)
    scores, labels, groups = _grouped_calibration_panel()
    with pytest.raises(ValueError, match="returned 1 folds"):
        calibration.choose_calibrator(
            scores, labels, groups=groups, n_splits=2,
        )


def test_structurally_valid_grouped_class_absence_keeps_identity_fallback(
    monkeypatch,
):
    import stylo.eval.calibration as calibration

    class _ClassAbsentSplitter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def split(self, matrix, labels, groups):
            del matrix, labels, groups
            return [
                (np.asarray([4, 5, 6, 7]), np.asarray([0, 1, 2, 3])),
                (np.asarray([0, 1, 2, 3]), np.asarray([4, 5, 6, 7])),
            ]

    monkeypatch.setattr(calibration, "StratifiedGroupKFold", _ClassAbsentSplitter)
    scores, labels, groups = _grouped_calibration_panel()
    _, passport = calibration.choose_calibrator(
        scores, labels, groups=groups, n_splits=2,
    )
    assert passport["method"] == "identity"
    assert passport["calibration_disabled"] is True


def test_equal_channel_estimator_uses_fixed_identity_equal_fusion(monkeypatch):
    texts, labels, groups = _three_class_panel()
    monkeypatch.setattr(
        sc,
        "make_channels",
        lambda *args, **kwargs: {
            "encoded_a": _encoded_channel,
            "encoded_b": _encoded_channel,
        },
    )
    monkeypatch.setattr(
        sc, "LinearSVC", lambda *args, **kwargs: _EchoDecisionClassifier(),
    )

    def forbidden_calibration(*args, **kwargs):
        del args, kwargs
        raise AssertionError("equal-channel estimator selected calibration")

    def forbidden_meta(*args, **kwargs):
        del args, kwargs
        raise AssertionError("equal-channel estimator constructed meta LR")

    monkeypatch.setattr(sc, "choose_calibrator", forbidden_calibration)
    monkeypatch.setattr(sc, "LogisticRegression", forbidden_meta)

    estimator = make_factory(
        "stylo_equal_channels_v1",
        CFG,
        weighting=CHUNK_WEIGHTED_LEGACY,
    )()
    estimator.fit(texts, labels, groups=groups)
    probabilities = estimator.predict_proba(texts)

    np.testing.assert_allclose(
        probabilities, softmax(_score_matrix(texts), axis=1),
        rtol=0.0, atol=1e-12,
    )
    assert not hasattr(estimator, "mode_")
    assert not hasattr(estimator, "passport_")
    passport = estimator.fusion_passport_
    assert passport["oof"] == {"used": False}
    assert passport["calibration"] == {"learned": False}
    assert passport["meta_classifier"] == {"present": False}
    assert passport["fusion"]["weights"] == {
        "encoded_a": 0.5,
        "encoded_b": 0.5,
    }

    with pytest.raises(ValueError):
        make_factory(
            "stylo_equal_channels",
            CFG,
            weighting=CHUNK_WEIGHTED_LEGACY,
        )


@pytest.mark.parametrize(
    ("weighting", "expected_axes"),
    [
        (CHUNK_WEIGHTED_LEGACY, (False, False, False)),
        (WORK_BALANCED, (True, True, True)),
    ],
)
def test_equal_channel_spec_routes_only_explicit_lobo_weighting_corners(
    weighting, expected_axes,
):
    estimator = make_factory(
        "stylo_equal_channels_v1", CFG, weighting=weighting,
    )()
    assert type(estimator) is EqualChannelEnsembleClassifier
    assert (
        estimator._weights_on,
        estimator._feature_on,
        estimator._relative_fw_on,
    ) == expected_axes


@pytest.mark.parametrize(
    "ablation",
    [
        WEIGHTS_ONLY_ABLATION,
        FEATURE_STATE_ONLY_ABLATION,
        RELATIVE_FW_ONLY_ABLATION,
    ],
)
def test_equal_channel_intermediate_axes_are_not_exposed_before_amendment(
    ablation,
):
    with pytest.raises(ValueError, match="Неизвестная модель"):
        make_factory_for_ablation(
            "stylo_equal_channels_v1", CFG, ablation=ablation,
        )


def test_equal_channel_spec_is_explicit_ece_model_not_confirmatory():
    from stylo.eval.final import DEFAULT_SPECS, ECE_SPECS
    from stylo.eval.paired_audit.applicability import MODELS

    assert "stylo_equal_channels_v1" in ECE_SPECS
    assert "stylo_equal_channels_v1" not in DEFAULT_SPECS
    assert "stylo_equal_channels_v1" not in MODELS


def test_incomplete_meta_cv_hard_fails_before_meta_lr_and_passport(monkeypatch):
    texts, labels, groups = _three_class_panel()

    class _OOFThenIncompleteMetaSplitter:
        construction_count = 0

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.phase = type(self).construction_count
            type(self).construction_count += 1

        def split(self, matrix, split_labels, split_groups):
            del matrix, split_labels, split_groups
            if self.phase == 0:
                return _ClassCompleteSplitter().split(None, None, None)
            return iter((
                (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
                (np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
                (np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
            ))

    def fixed_identity(oof, y, seed=42, groups=None, **kwargs):
        del oof, y, seed, groups, kwargs
        return (
            lambda scores: softmax(scores, axis=1),
            {"method": "identity", "params": {}},
        )

    def forbidden_lr(*args, **kwargs):
        del args, kwargs
        raise AssertionError("meta LR was constructed after failed preflight")

    monkeypatch.setattr(sc, "StratifiedGroupKFold", _OOFThenIncompleteMetaSplitter)
    monkeypatch.setattr(
        sc, "make_channels", lambda *args, **kwargs: {"encoded": _encoded_channel},
    )
    monkeypatch.setattr(
        sc, "LinearSVC", lambda *args, **kwargs: _EchoDecisionClassifier(),
    )
    monkeypatch.setattr(sc, "choose_calibrator", fixed_identity)
    monkeypatch.setattr(sc, "LogisticRegression", forbidden_lr)

    estimator = sc.StackedChannelClassifier(CFG, inner_folds=2)
    with pytest.raises(sc.StackClassCoverageError) as caught:
        estimator.fit(texts, labels, groups=groups)

    assert caught.value.stage == "meta_cv"
    assert caught.value.report["class_coverage_complete"] is False
    assert estimator.passport_ == {}
    assert not hasattr(estimator, "mode_")
    assert not hasattr(estimator, "meta_")


def test_class_complete_legacy_stack_reaches_calibration(monkeypatch):
    texts, labels, groups = _three_class_panel()
    calls = {"channels": 0, "calibration": 0}

    def channels(*args, **kwargs):
        del args, kwargs
        calls["channels"] += 1
        return {"encoded": _encoded_channel}

    def identity_and_disable_meta(oof, y, seed=42, groups=None, **kwargs):
        del seed, kwargs
        calls["calibration"] += 1
        assert groups is None
        assert oof.shape == (6, 3)
        return (
            lambda scores: softmax(scores, axis=1),
            {
                "method": "identity",
                "params": {},
                "calibration_disabled": True,
                "reason": "focused test stops before meta selection",
            },
        )

    monkeypatch.setattr(sc, "StratifiedGroupKFold", _ClassCompleteSplitter)
    monkeypatch.setattr(sc, "make_channels", channels)
    monkeypatch.setattr(
        sc, "LinearSVC", lambda *args, **kwargs: _EchoDecisionClassifier(),
    )
    monkeypatch.setattr(sc, "choose_calibrator", identity_and_disable_meta)

    estimator = sc.StackedChannelClassifier(CFG, inner_folds=2)
    estimator.fit(texts, labels, groups=groups)
    assert calls == {"channels": 1, "calibration": 1}
    assert estimator.mode_ == "equal"
    assert estimator.meta_ is None
