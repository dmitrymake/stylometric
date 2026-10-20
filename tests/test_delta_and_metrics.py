"""Настоящая Delta считается только по MFW; метрики/CI ведут себя корректно."""
import subprocess
import sys

import numpy as np
import pytest

from stylo.eval.metrics import accuracy, macro_f1, bootstrap_ci, expected_calibration_error
from stylo.eval.significance import mcnemar, paired_bootstrap_diff, paired_bootstrap_diff_clustered
from stylo.models.baselines import CharCosineBaseline
from stylo.models.delta import BurrowsDelta


def test_configuration_enums_fail_closed_under_optimized_python():
    script = """
from stylo.features.function_words import FunctionWordBlock
from stylo.models.delta import BurrowsDelta

for constructor in (
    lambda: FunctionWordBlock(mode="unknown"),
    lambda: BurrowsDelta(metric="unknown"),
):
    try:
        constructor()
    except ValueError:
        continue
    raise AssertionError("invalid configuration survived python -O")
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_delta_uses_only_mfw():
    # train: два «автора» с разной частотностью служебных слов
    a = ["и в на и в на он она его " * 5] * 4
    b = ["но что как но что как они их " * 5] * 4
    texts = a + b
    y = [0] * 4 + [1] * 4
    d = BurrowsDelta(mfw_count=10, metric="manhattan").fit(texts, y)
    # словарь Delta ограничен MFW (<=10 слов), не тысячами признаков
    assert len(d.feature_names()) <= 10
    # текст в стиле автора 0 ближе к автору 0
    assert d.predict(["и в на и в на он " * 5])[0] == 0


def test_delta_centroid_weights_works_not_chunks():
    texts = ["x"] * 9 + ["y", "y", "y"]
    y = np.array([0] * 10 + [1, 1])
    groups = [("a", "long")] * 9 + [
        ("a", "short"),
        ("b", "0"),
        ("b", "1"),
    ]

    model = BurrowsDelta(vocabulary=["x", "y"]).fit(texts, y, groups=groups)

    z = model._z(texts)
    expected_a = np.mean([z[:9].mean(axis=0), z[9:10].mean(axis=0)], axis=0)
    flat_a = z[y == 0].mean(axis=0)
    assert np.allclose(model.centroids_[0], expected_a)
    assert not np.allclose(model.centroids_[0], flat_a)
    assert model.group_weighting_ == "equal_group_after_within_group_mean"


def test_delta_two_argument_fit_preserves_legacy_row_weighting():
    texts = ["x x x y", "x y y y", "y y y y", "x x y y"]
    y = np.array([0, 0, 1, 1])

    model = BurrowsDelta(metric="cosine", vocabulary=["x", "y"]).fit(texts, y)
    z = model._z(texts)

    assert np.allclose(model.centroids_[0], z[y == 0].mean(axis=0))
    assert np.allclose(model.centroids_[1], z[y == 1].mean(axis=0))
    assert model.group_weighting_ == "equal_row"


def test_char_cosine_centroid_weights_works_not_chunks():
    texts = ["xxxx"] * 9 + ["yyyy", "yyyy", "yyyy"]
    y = np.array([0] * 10 + [1, 1])
    groups = np.array(["a-long"] * 9 + ["a-short", "b0", "b1"])

    model = CharCosineBaseline(min_df=1).fit(texts, y, groups=groups)

    X = model._vec.transform(texts)
    expected_a = np.mean(
        [
            np.asarray(X[:9].mean(axis=0)).ravel(),
            np.asarray(X[9:10].mean(axis=0)).ravel(),
        ],
        axis=0,
    )
    flat_a = np.asarray(X[y == 0].mean(axis=0)).ravel()
    assert np.allclose(model._centroids[0], expected_a)
    assert not np.allclose(model._centroids[0], flat_a)
    assert model.group_weighting_ == (
        "equal_group_direction_after_within_group_mean_l2"
    )


def test_char_cosine_two_argument_fit_preserves_legacy_row_weighting():
    texts = ["aaaa", "aaab", "bbbb", "bbba"]
    y = np.array([0, 0, 1, 1])

    model = CharCosineBaseline(min_df=1).fit(texts, y)
    X = model._vec.transform(texts)

    assert np.allclose(
        model._centroids[0], np.asarray(X[y == 0].mean(axis=0)).ravel()
    )
    assert np.allclose(
        model._centroids[1], np.asarray(X[y == 1].mean(axis=0)).ravel()
    )
    assert model.group_weighting_ == "equal_row"


def test_metrics_basic():
    truth = np.array([0, 1, 2, 0, 1])
    predicted = np.array([0, 1, 2, 1, 1])
    assert accuracy(truth, predicted) == pytest.approx(4 / 5)
    # Class F1 scores: 2/3, 4/5, 1. The fixed label universe includes absent classes.
    assert macro_f1(truth, predicted, [0, 1, 2]) == pytest.approx(37 / 45)
    assert macro_f1(truth, predicted, [0, 1, 2, 3]) == pytest.approx(37 / 60)


@pytest.mark.parametrize("values,point,low,high", [
    ([0.0, 1.0], 0.5, 0.0, 1.0),
    ([2.0, 6.0], 4.0, 2.0, 6.0),
    ([7.0, 7.0], 7.0, 7.0, 7.0),
])
def test_bootstrap_ci_bounds(values, point, low, high):
    # The four equally likely draws are (a,a), (a,b), (b,a), (b,b).
    values = np.asarray(values)
    result = bootstrap_ci(lambda ix: values[ix].mean(), 2, iters=1000, seed=1)
    assert (result.point, result.lo, result.hi) == pytest.approx((point, low, high))


def test_mcnemar_and_diff():
    a = np.ones(6, dtype=bool)
    b = np.zeros(6, dtype=bool)
    exact = mcnemar(a, b)
    assert (exact.b, exact.c, exact.p_value) == pytest.approx((6, 0, 2 / 2**6))
    assert mcnemar(b, a).p_value == exact.p_value
    assert mcnemar(a, a).p_value == 1.0
    different = paired_bootstrap_diff(lambda ix: a[ix].mean(), lambda ix: b[ix].mean(),
                                     6, iters=100, seed=1)
    assert (different.diff, different.lo, different.hi) == (1.0, 1.0, 1.0)
    assert different.significant is True
    varied = np.array([0.0, 1.0])
    paired = paired_bootstrap_diff(lambda ix: varied[ix].mean(), lambda ix: varied[ix].mean(),
                                  2, iters=100, seed=1)
    assert (paired.diff, paired.lo, paired.hi) == (0.0, 0.0, 0.0)
    assert paired.significant is False


def test_clustered_bootstrap_by_author():
    # Two authors: eight correct books and two wrong books. Drawing whole authors
    # gives means 0, .8, 1; row resampling cannot reproduce this 95% interval.
    scores = np.array([1.0] * 8 + [0.0] * 2)
    groups = np.array(["a"] * 8 + ["b"] * 2)
    result = paired_bootstrap_diff_clustered(lambda ix: scores[ix].mean(), lambda ix: 0.0,
                                            groups, iters=1000, seed=1)
    assert (result.diff, result.lo, result.hi) == pytest.approx((0.8, 0.0, 1.0))
    assert result.significant is False
    row_result = paired_bootstrap_diff(lambda ix: scores[ix].mean(), lambda ix: 0.0,
                                      len(scores), iters=1000, seed=1)
    assert row_result.lo > result.lo


def test_ece_perfect_vs_overconfident():
    truth = np.array([0, 1])
    perfect = np.eye(2)
    confident_wrong = perfect[::-1] * 0.8 + 0.1
    underconfident = perfect * 0.8 + 0.1
    assert expected_calibration_error(perfect, truth) == 0.0
    assert expected_calibration_error(confident_wrong, truth) == pytest.approx(0.9)
    assert expected_calibration_error(underconfident, truth) == pytest.approx(0.1)
    # Different bins must contribute in proportion to their sample counts.
    mixed = np.array([[0.6, 0.4], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    assert expected_calibration_error(mixed, [0, 0, 0, 1]) == pytest.approx(0.275)
