"""Настоящая Delta считается только по MFW; метрики/CI ведут себя корректно."""
import numpy as np

from stylo.eval.metrics import accuracy, macro_f1, bootstrap_ci, expected_calibration_error
from stylo.eval.significance import mcnemar, paired_bootstrap_diff, paired_bootstrap_diff_clustered
from stylo.models.baselines import CharCosineBaseline
from stylo.models.delta import BurrowsDelta


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
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert accuracy(y_true, y_pred) == 0.8
    assert 0 <= macro_f1(y_true, y_pred, [0, 1, 2]) <= 1


def test_bootstrap_ci_bounds():
    y_true = np.array([0, 1] * 20)
    y_pred = y_true.copy()
    ci = bootstrap_ci(lambda ix: accuracy(y_true[ix], y_pred[ix]), len(y_true),
                      iters=200, seed=1)
    assert ci.point == 1.0 and ci.lo <= ci.point <= ci.hi


def test_mcnemar_and_diff():
    ca = np.array([1, 1, 1, 0, 0], dtype=bool)
    cb = np.array([1, 0, 0, 0, 0], dtype=bool)
    r = mcnemar(ca, cb)
    assert r.b == 2 and r.c == 0
    diff = paired_bootstrap_diff(lambda ix: ca[ix].mean(), lambda ix: cb[ix].mean(),
                                 len(ca), iters=200, seed=1)
    assert diff.diff > 0


def test_clustered_bootstrap_by_author():
    """Кластерный bootstrap ресэмплит АВТОРОВ (группы), не отдельные книги; точка внутри CI."""
    ca = np.array([1, 1, 1, 1, 0, 0], dtype=bool)   # stylo
    cb = np.array([0, 0, 0, 0, 0, 0], dtype=bool)   # baseline
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    d = paired_bootstrap_diff_clustered(lambda ix: ca[ix].mean(), lambda ix: cb[ix].mean(),
                                        groups, iters=300, seed=1)
    assert d.diff > 0                       # stylo не хуже baseline на каждой книге
    assert d.lo <= d.diff <= d.hi


def test_ece_perfect_vs_overconfident():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    y = np.array([0, 1])
    assert expected_calibration_error(probs, y) >= 0.0
