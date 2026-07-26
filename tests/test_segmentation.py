import pytest

from stylo.eval.segmentation import (
    LabeledSpan,
    SegmentationDocument,
    best_corpus_label_mapping,
    boundary_metrics,
    canonicalize_spans,
    evaluate_corpus,
    evaluate_document,
    labeled_boundary_metrics,
    segment_iou_metrics,
    token_metrics,
    validate_spans,
)


def _spans(*items):
    return [LabeledSpan(*item) for item in items]


def test_validate_spans_rejects_unsorted_overlap_and_gap_when_requested():
    with pytest.raises(ValueError, match="not sorted"):
        validate_spans(_spans((5, 10, "b"), (0, 5, "a")))
    with pytest.raises(ValueError, match="overlap"):
        validate_spans(_spans((0, 6, "a"), (5, 10, "b")))
    with pytest.raises(ValueError, match="gap"):
        validate_spans(
            _spans((0, 4, "a"), (5, 10, "b")), require_contiguous=True
        )


def test_evaluation_requires_complete_common_token_partition():
    truth = _spans((0, 5, "a"), (5, 10, "b"))
    with pytest.raises(ValueError, match="full document"):
        evaluate_document(truth, _spans((1, 10, "a")))
    with pytest.raises(ValueError, match="beyond document_length"):
        evaluate_document(truth, _spans((0, 11, "a")))


def test_canonicalization_removes_artificial_same_label_boundary():
    spans = _spans((0, 2, "a"), (2, 5, "a"), (5, 8, "b"))

    assert canonicalize_spans(spans) == (
        LabeledSpan(0, 5, "a"),
        LabeledSpan(5, 8, "b"),
    )


def test_exact_segmentation_scores_one_everywhere():
    truth = _spans((0, 4, "a"), (4, 10, "b"))
    report = evaluate_document(truth, truth, document_id="doc")

    assert report.token.accuracy == 1.0
    assert report.token.macro_f1 == 1.0
    assert report.boundaries.scores.f1 == 1.0
    assert report.segments.scores.f1 == 1.0
    assert report.segments.mean_matched_iou == 1.0


def test_shifted_boundary_uses_tolerance_but_token_metric_stays_exact():
    truth = _spans((0, 5, "a"), (5, 10, "b"))
    predicted = _spans((0, 6, "a"), (6, 10, "b"))

    token = token_metrics(truth, predicted)
    strict = boundary_metrics(truth, predicted, tolerance=0)
    tolerant = boundary_metrics(truth, predicted, tolerance=1)
    labeled_tolerant = labeled_boundary_metrics(truth, predicted, tolerance=1)
    segments = segment_iou_metrics(truth, predicted, threshold=0.75)

    assert token.accuracy == pytest.approx(0.9)
    assert token.macro_f1 == pytest.approx((10 / 11 + 8 / 9) / 2)
    assert strict.scores.f1 == 0.0
    assert tolerant.scores.f1 == 1.0
    assert labeled_tolerant.scores.f1 == 1.0
    assert tolerant.matches[0].absolute_error == 1
    assert segments.scores.f1 == 1.0
    assert segments.mean_matched_iou == pytest.approx((5 / 6 + 4 / 5) / 2)
    assert segments.penalized_iou == pytest.approx((5 / 6 + 4 / 5) / 2)


def test_labeled_boundary_rejects_correct_location_with_wrong_transition():
    truth = _spans((0, 5, "a"), (5, 10, "b"))
    predicted = _spans((0, 5, "b"), (5, 10, "a"))

    unlabeled = boundary_metrics(truth, predicted)
    labeled = labeled_boundary_metrics(truth, predicted)
    document = evaluate_document(truth, predicted)
    corpus = evaluate_corpus(
        [SegmentationDocument("swapped", truth, predicted, "work")],
        bootstrap_unit="work",
        bootstrap_iters=5,
    )

    assert unlabeled.scores.f1 == 1.0
    assert labeled.scores.f1 == 0.0
    assert document.boundaries.scores.f1 == 1.0
    assert document.labeled_boundaries.scores.f1 == 0.0
    assert corpus.aggregate.boundaries.f1 == 1.0
    assert corpus.aggregate.labeled_boundaries.f1 == 0.0
    assert corpus.confidence_intervals["labeled_boundary_f1"].point == 0.0


def test_no_boundaries_is_a_perfect_boundary_result():
    one = _spans((0, 10, "a"))

    result = boundary_metrics(one, one)

    assert result.scores.true_positives == 0
    assert result.scores.precision == 1.0
    assert result.scores.recall == 1.0
    assert result.scores.f1 == 1.0


def test_permutation_safe_labels_align_anonymous_clusters_only_when_requested():
    truth = _spans((0, 5, "author_a"), (5, 10, "author_b"))
    predicted = _spans((0, 5, "cluster_7"), (5, 10, "cluster_2"))

    named = evaluate_document(truth, predicted)
    anonymous = evaluate_document(
        truth, predicted, evaluation_mode="anonymous_partition"
    )

    assert named.token.accuracy == 0.0
    assert named.segments.scores.f1 == 0.0
    assert anonymous.token.accuracy == 1.0
    assert anonymous.token.macro_f1 == 1.0
    assert anonymous.segments.scores.f1 == 1.0
    assert anonymous.label_mapping == {
        "cluster_2": "author_b",
        "cluster_7": "author_a",
    }
    assert named.evaluation_mode == "named_attribution"
    assert anonymous.evaluation_mode == "anonymous_partition"
    # Boundary locations are label-permutation invariant in either mode.
    assert named.boundaries.scores.f1 == anonymous.boundaries.scores.f1 == 1.0

    with pytest.raises(ValueError, match="forbidden for named attribution"):
        evaluate_document(truth, predicted, permutation_safe=True)


def test_segment_iou_matching_is_one_to_one_and_label_aware():
    truth = _spans((0, 4, "a"), (4, 7, "b"), (7, 10, "a"))
    predicted = _spans((0, 3, "a"), (3, 7, "b"), (7, 10, "a"))

    result = segment_iou_metrics(truth, predicted, threshold=0.7)

    assert result.scores.true_positives == 3
    assert len({match.predicted_index for match in result.matches}) == 3
    assert [match.label for match in result.matches] == ["a", "b", "a"]


def test_penalized_iou_counts_unmatched_segments_as_zero():
    truth = [LabeledSpan(index, index + 1, f"true_{index}") for index in range(10)]
    predicted = [LabeledSpan(0, 1, "true_0")] + [
        LabeledSpan(index, index + 1, f"wrong_{index}")
        for index in range(1, 10)
    ]

    segments = segment_iou_metrics(truth, predicted)
    corpus = evaluate_corpus(
        [SegmentationDocument("one_of_ten", truth, predicted, "work")],
        bootstrap_unit="work",
        bootstrap_iters=5,
    )

    assert segments.scores.true_positives == 1
    assert segments.mean_matched_iou == 1.0
    assert segments.penalized_iou == pytest.approx(0.1)
    assert segments.penalized_iou < segments.mean_matched_iou
    assert corpus.aggregate.segments.penalized_iou == pytest.approx(0.1)
    assert corpus.confidence_intervals["penalized_segment_iou"].point == pytest.approx(0.1)


def test_corpus_permutation_is_global_not_optimised_per_document():
    documents = [
        SegmentationDocument(
            "d1", _spans((0, 10, "a")), _spans((0, 10, "cluster")), "w1"
        ),
        SegmentationDocument(
            "d2", _spans((0, 10, "b")), _spans((0, 10, "cluster")), "w2"
        ),
    ]

    mapping = best_corpus_label_mapping(documents)
    report = evaluate_corpus(
        documents,
        evaluation_mode="anonymous_partition",
        bootstrap_unit="work",
        bootstrap_iters=30,
        seed=3,
    )

    assert mapping["cluster"] in {"a", "b"}
    assert report.aggregate.token_accuracy == 0.5
    assert report.label_mapping == mapping


def test_corpus_aggregates_counts_and_bootstraps_whole_works():
    good = _spans((0, 5, "a"), (5, 10, "b"))
    bad = _spans((0, 10, "b"))
    documents = [
        SegmentationDocument("d1", good, good, "novel_1"),
        SegmentationDocument("d2", good, bad, "novel_1"),
        SegmentationDocument("d3", good, good, "novel_2"),
    ]

    report = evaluate_corpus(
        documents,
        boundary_tolerance=0,
        segment_iou_threshold=0.5,
        bootstrap_unit="work",
        bootstrap_iters=100,
        ci_level=0.9,
        seed=7,
    )

    assert report.aggregate.n_documents == 3
    assert report.aggregate.n_tokens == 30
    assert report.aggregate.token_accuracy == pytest.approx(25 / 30)
    assert report.bootstrap_unit == "work"
    assert report.n_bootstrap_units == 2
    assert report.confidence_intervals["token_accuracy"].point == pytest.approx(25 / 30)
    for interval in report.confidence_intervals.values():
        assert interval.lo <= interval.point <= interval.hi


def test_single_author_false_positive_rate_is_explicit_and_bootstrapped():
    documents = [
        SegmentationDocument(
            "control_ok",
            _spans((0, 10, "a")),
            _spans((0, 10, "a")),
            "work_1",
        ),
        SegmentationDocument(
            "control_false_switch",
            _spans((0, 10, "a")),
            _spans((0, 5, "a"), (5, 10, "b")),
            "work_2",
        ),
        SegmentationDocument(
            "actual_mixture",
            _spans((0, 5, "a"), (5, 10, "b")),
            _spans((0, 5, "a"), (5, 10, "b")),
            "work_3",
        ),
    ]

    report = evaluate_corpus(documents, bootstrap_iters=50, seed=9)

    assert report.aggregate.n_single_author_controls == 2
    assert report.aggregate.single_author_false_positives == 1
    assert report.aggregate.single_author_false_positive_rate == 0.5
    assert report.aggregate.n_single_author_control_documents == 2
    assert report.aggregate.single_author_document_false_positive_rate == 0.5
    assert report.aggregate.n_single_author_control_works == 2
    assert report.aggregate.single_author_work_false_positive_rate == 0.5
    assert "single_author_false_positive_rate" in report.confidence_intervals
    assert "single_author_document_false_positive_rate" in report.confidence_intervals
    assert "single_author_work_false_positive_rate" in report.confidence_intervals


def test_single_author_document_and_work_fpr_are_distinct():
    documents = [
        SegmentationDocument(
            "chapter_ok",
            _spans((0, 10, "a")),
            _spans((0, 10, "a")),
            "same_work",
        ),
        SegmentationDocument(
            "chapter_false_switch",
            _spans((0, 10, "a")),
            _spans((0, 5, "a"), (5, 10, "b")),
            "same_work",
        ),
    ]

    report = evaluate_corpus(
        documents,
        bootstrap_unit="work",
        bootstrap_iters=20,
        seed=11,
    )

    assert report.aggregate.n_single_author_control_documents == 2
    assert report.aggregate.single_author_document_false_positive_rate == 0.5
    assert report.aggregate.n_single_author_control_works == 1
    assert report.aggregate.single_author_work_false_positive_rate == 1.0
    assert report.confidence_intervals[
        "single_author_work_false_positive_rate"
    ].point == 1.0


def test_anonymous_single_author_fpr_scores_splits_not_cluster_names():
    documents = [
        SegmentationDocument(
            "one_cluster",
            _spans((0, 10, "known_author")),
            _spans((0, 10, "cluster_9")),
            "work_1",
        ),
        SegmentationDocument(
            "spurious_split",
            _spans((0, 10, "known_author")),
            _spans((0, 5, "cluster_9"), (5, 10, "cluster_2")),
            "work_2",
        ),
    ]

    report = evaluate_corpus(
        documents,
        evaluation_mode="anonymous_partition",
        bootstrap_unit="work",
        bootstrap_iters=20,
    )

    assert report.aggregate.single_author_document_false_positive_rate == 0.5
    assert report.evaluation_mode == "anonymous_partition"


def test_explicit_work_bootstrap_requires_work_ids_and_unique_documents():
    span = _spans((0, 5, "a"))
    missing_work = [SegmentationDocument("d1", span, span)]
    duplicate_ids = [
        SegmentationDocument("d1", span, span, "w1"),
        SegmentationDocument("d1", span, span, "w2"),
    ]
    partial_work_ids = [
        SegmentationDocument("d1", span, span, "w1"),
        SegmentationDocument("d2", span, span),
    ]

    with pytest.raises(ValueError, match="requires work_id"):
        evaluate_corpus(missing_work, bootstrap_unit="work", bootstrap_iters=5)
    with pytest.raises(ValueError, match="duplicate document_id"):
        evaluate_corpus(duplicate_ids, bootstrap_iters=5)
    with pytest.raises(ValueError, match="partially missing work_id"):
        evaluate_corpus(partial_work_ids, bootstrap_iters=5)
