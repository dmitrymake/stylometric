from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from stylo.corpus import Dataset
from stylo.domain import corpus_identity
from stylo.eval.lobo import build_generic_lobo_fold_manifest
from stylo.eval.metrics import (
    AuthorClusteredInferenceSpec,
    summarize_book_results,
)
from stylo.eval.significance import paired_bootstrap_diff_clustered


def _controlled_overlap(
    monkeypatch: pytest.MonkeyPatch,
    left: np.ndarray,
    right: np.ndarray,
    *,
    threshold: float,
    sample_size: int = 64,
):
    by_marker = {
        "left marker": np.asarray(left, dtype=np.uint64),
        "right marker": np.asarray(right, dtype=np.uint64),
    }

    def fake_shingles(rows, n):
        assert n == 5
        return by_marker[rows[0]]

    monkeypatch.setattr(corpus_identity, "_word_shingles", fake_shingles)
    return corpus_identity.find_cross_work_content_overlaps(
        ["left marker", "right marker"],
        ["author/left", "author/right"],
        containment_threshold=threshold,
        min_shingles=1,
        sample_size=sample_size,
    )


def test_f1_overlap_contract_v2_is_inclusive_at_exact_decimal_threshold(
    monkeypatch,
):
    left = np.arange(20, dtype=np.uint64)
    right = np.concatenate(
        [np.arange(18, dtype=np.uint64), np.asarray([100, 101], dtype=np.uint64)]
    )

    overlaps = _controlled_overlap(
        monkeypatch,
        left,
        right,
        threshold=0.90,
    )

    assert corpus_identity.CONTENT_OVERLAP_POLICY_VERSION.endswith(".v2")
    assert len(overlaps) == 1
    assert overlaps[0].evidence == "18/20 unique word-5-grams"


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.8999, True), (0.9001, False)],
)
def test_f1_overlap_contract_v2_distinguishes_below_and_above_threshold(
    monkeypatch,
    threshold,
    expected,
):
    left = np.arange(20, dtype=np.uint64)
    right = np.concatenate(
        [np.arange(18, dtype=np.uint64), np.asarray([100, 101], dtype=np.uint64)]
    )

    overlaps = _controlled_overlap(
        monkeypatch,
        left,
        right,
        threshold=threshold,
    )

    assert bool(overlaps) is expected


def test_f1_optimized_overlap_matches_exact_bruteforce_property(monkeypatch):
    rng = np.random.default_rng(24601)
    thresholds = (0.5, 0.75, 0.9, 0.95, 1.0)
    universe = np.arange(80, dtype=np.uint64)

    for _ in range(100):
        left_size = int(rng.integers(5, 41))
        right_size = int(rng.integers(left_size, 61))
        left = np.sort(rng.choice(universe, size=left_size, replace=False))
        right = np.sort(rng.choice(universe, size=right_size, replace=False))
        threshold = thresholds[int(rng.integers(0, len(thresholds)))]
        sample_size = (1, 3, 8, 64)[int(rng.integers(0, 4))]
        overlaps = _controlled_overlap(
            monkeypatch,
            left,
            right,
            threshold=threshold,
            sample_size=sample_size,
        )
        common = len(set(left.tolist()) & set(right.tolist()))
        exact_threshold = Fraction(str(threshold))
        expected = (
            common * exact_threshold.denominator
            >= left_size * exact_threshold.numerator
        )
        assert bool(overlaps) is expected


def test_f1_exact_duplicate_and_short_in_long_regressions():
    duplicate = "один и тот же дословный фрагмент " * 12
    duplicate_overlaps = corpus_identity.find_cross_work_content_overlaps(
        [duplicate, duplicate],
        ["author/story", "author/collection"],
    )
    assert [item.kind for item in duplicate_overlaps] == [
        "exact_cross_work_chunk"
    ]

    short = " ".join(f"слово{index}" for index in range(40))
    long = f"предисловие {short} послесловие"
    containment_overlaps = corpus_identity.find_cross_work_content_overlaps(
        [short, long],
        ["author/story", "author/collection"],
    )
    assert any(
        item.kind == "word5_asymmetric_containment"
        and item.containment == 1.0
        for item in containment_overlaps
    )


def _inference_spec(*, seed=42, iterations=200):
    return AuthorClusteredInferenceSpec.build(
        iterations=iterations,
        confidence_level=0.95,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("predicted", "expected_macro_f1"),
    [
        ([0, 2], 1.0),
        ([1, 2], 0.5),
        ([0, 0], 1.0 / 3.0),
    ],
)
def test_f2_fixed_metric_universe_excludes_train_only_singleton_distractor(
    predicted,
    expected_macro_f1,
):
    summary = summarize_book_results(
        np.asarray([0, 2]),
        np.asarray(predicted),
        np.asarray([1, 1]),
        probability_class_order=["aa", "singleton", "bb"],
        metric_label_order=[0, 2],
        book_authors=["aa", "bb"],
        inference_spec=_inference_spec(),
    )

    assert summary["macro_f1"].point == pytest.approx(expected_macro_f1)
    assert summary["macro_f1"].uncertainty == "point_only"


def test_f2_metric_universe_must_be_explicit_subset_of_probability_universe():
    with pytest.raises(ValueError, match="metric_label_order"):
        summarize_book_results(
            np.asarray([0, 2]),
            np.asarray([0, 2]),
            np.asarray([1, 1]),
            probability_class_order=["aa", "singleton", "bb"],
            metric_label_order=[0, 3],
            book_authors=["aa", "bb"],
            inference_spec=_inference_spec(),
        )


def test_f2_generic_fold_manifest_freezes_p_and_m_before_predictions():
    dataset = Dataset(
        texts=np.asarray(
            ["aa-1", "aa-2", "singleton", "bb-1", "bb-2"],
            dtype=object,
        ),
        y=np.asarray([0, 0, 1, 2, 2]),
        groups=np.asarray(
            ["aa/a1", "aa/a2", "singleton/s1", "bb/b1", "bb/b2"],
            dtype=object,
        ),
        authors=["aa", "singleton", "bb"],
    )

    manifest = build_generic_lobo_fold_manifest(dataset)

    assert manifest.probability_class_order == ("aa", "singleton", "bb")
    assert manifest.metric_label_order == (0, 2)
    assert [fold.work_id for fold in manifest.folds] == [
        "aa/a1",
        "aa/a2",
        "bb/b1",
        "bb/b2",
    ]
    assert manifest == build_generic_lobo_fold_manifest(dataset)


def test_f5_primary_accuracy_ci_is_deterministic_author_clustered():
    truth = np.asarray([0, 0, 0, 1])
    predicted = np.asarray([0, 0, 0, 0])
    kwargs = {
        "y_true": truth,
        "y_pred": predicted,
        "ranks": np.asarray([1, 1, 1, 2]),
        "probability_class_order": ["a", "b"],
        "metric_label_order": [0, 1],
        "book_authors": ["a", "a", "a", "b"],
        "inference_spec": _inference_spec(seed=1, iterations=1000),
    }

    first = summarize_book_results(**kwargs)
    second = summarize_book_results(**kwargs)

    assert first["accuracy"] == second["accuracy"]
    assert first["accuracy"].point == 0.75
    assert first["accuracy"].lo == 0.0
    assert first["accuracy"].hi == 1.0
    assert first["accuracy"].method == "author_clustered_percentile_bootstrap"
    assert first["top2"].uncertainty == "point_only"


def test_f5_paired_clustered_bootstrap_reuses_byte_identical_draws():
    groups = np.asarray(["a", "a", "b", "b"])
    observed_a = []
    observed_b = []

    def metric_a(indices):
        observed_a.append(indices.tobytes())
        return 0.5

    def metric_b(indices):
        observed_b.append(indices.tobytes())
        return 0.25

    paired_bootstrap_diff_clustered(
        metric_a,
        metric_b,
        groups,
        iters=40,
        level=0.95,
        seed=7,
    )

    assert observed_a == observed_b


def test_f5_inference_spec_identity_binds_seed_iterations_and_level():
    baseline = _inference_spec(seed=7, iterations=40)
    assert baseline.self_hash == _inference_spec(
        seed=7,
        iterations=40,
    ).self_hash
    assert baseline.self_hash != _inference_spec(
        seed=8,
        iterations=40,
    ).self_hash
    assert baseline.self_hash != _inference_spec(
        seed=7,
        iterations=41,
    ).self_hash
    assert baseline.self_hash != AuthorClusteredInferenceSpec.build(
        iterations=40,
        confidence_level=0.90,
        seed=7,
    ).self_hash


def test_f1_overlap_policy_version_is_bound_into_scientific_context_identity():
    from stylo.eval.provenance import (
        prepare_synthetic_scientific_evaluation,
    )

    dataset = Dataset(
        texts=np.asarray(["уникальный один", "уникальный два"], dtype=object),
        y=np.asarray([0, 1]),
        groups=np.asarray(["a/w1", "b/w1"], dtype=object),
        authors=["a", "b"],
    )
    context = prepare_synthetic_scientific_evaluation(
        dataset,
        "chunk_weighted_legacy",
    )

    assert (
        context.isolation_contract_version
        == corpus_identity.CONTENT_OVERLAP_POLICY_VERSION
    )
