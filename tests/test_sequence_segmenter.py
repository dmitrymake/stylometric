import numpy as np
import pytest

from stylo.eval.segmentation import LabeledSpan, evaluate_document
from stylo.models.sequence_segmenter import (
    MixedControl,
    TokenWindow,
    calibrate_decoder_on_controls,
    decode_windows,
    tune_decoder_on_controls,
    viterbi_decode,
    windows_to_spans,
)


def test_viterbi_suppresses_isolated_label_flip():
    probs = np.asarray([
        [0.95, 0.05],
        [0.90, 0.10],
        [0.40, 0.60],  # local noise
        [0.90, 0.10],
        [0.95, 0.05],
    ])
    assert viterbi_decode(probs, transition_penalty=0.0).tolist() == [0, 0, 1, 0, 0]
    assert viterbi_decode(probs, transition_penalty=1.0).tolist() == [0, 0, 0, 0, 0]


def test_decoder_can_abstain_and_emits_full_coverage_spans():
    windows = [TokenWindow(0, 40), TokenWindow(30, 70), TokenWindow(60, 100)]
    probs = np.asarray([[0.90, 0.10], [0.36, 0.34], [0.10, 0.90]])
    spans = decode_windows(
        windows,
        probs,
        ["a", "b"],
        document_length=100,
        transition_penalty=0.0,
        abstain_threshold=0.55,
    )
    assert [(s.start, s.end, s.label) for s in spans] == [
        (0, 35, "a"),
        (35, 65, "__unknown__"),
        (65, 100, "b"),
    ]


def test_decoder_output_is_directly_accepted_by_segmentation_scorer():
    windows = [TokenWindow(0, 40), TokenWindow(30, 70), TokenWindow(60, 100)]
    probs = np.asarray([[0.95, 0.05], [0.05, 0.95], [0.05, 0.95]])

    predicted = decode_windows(
        windows,
        probs,
        ["a", "b"],
        document_length=100,
        transition_penalty=0.0,
    )
    report = evaluate_document(
        [LabeledSpan(0, 35, "a"), LabeledSpan(35, 100, "b")],
        predicted,
    )

    assert all(isinstance(span, LabeledSpan) for span in predicted)
    assert report.token.accuracy == 1.0
    assert report.boundaries.scores.f1 == 1.0


def test_windows_must_strictly_cover_document_without_gaps():
    with pytest.raises(ValueError, match="cover the full document"):
        windows_to_spans(
            [TokenWindow(10, 20), TokenWindow(20, 100)],
            [0, 1],
            ["a", "b"],
            document_length=100,
        )
    with pytest.raises(ValueError, match="uncovered token gap"):
        windows_to_spans(
            [TokenWindow(0, 20), TokenWindow(80, 100)],
            [0, 1],
            ["a", "b"],
            document_length=100,
        )
    with pytest.raises(ValueError, match="increasing starts and ends"):
        windows_to_spans(
            [TokenWindow(0, 80), TokenWindow(20, 60), TokenWindow(50, 100)],
            [0, 1, 1],
            ["a", "b"],
            document_length=100,
        )


def test_overlap_midpoint_never_assigns_tokens_outside_decoded_window():
    spans = windows_to_spans(
        [TokenWindow(0, 90), TokenWindow(80, 100)],
        [0, 1],
        ["a", "b"],
        document_length=100,
    )

    assert spans == [LabeledSpan(0, 85, "a"), LabeledSpan(85, 100, "b")]


def _mixed_small_contribution_control():
    return MixedControl(
        probs=np.asarray([
            [0.99, 0.01],
            [0.99, 0.01],
            [0.01, 0.99],
            [0.99, 0.01],
            [0.99, 0.01],
        ]),
        true_labels=[0, 0, 1, 0, 0],
        small_contribution_labels=[1],
    )


def test_control_tuning_requires_negative_and_positive_gates():
    controls = [
        np.asarray([[0.9, 0.1], [0.45, 0.55], [0.9, 0.1]]),
        np.asarray([[0.1, 0.9], [0.2, 0.8], [0.1, 0.9]]),
    ]
    result = calibrate_decoder_on_controls(
        controls,
        [0, 1],
        mixed_controls=[_mixed_small_contribution_control()],
        transition_penalties=[0.0, 0.5, 100.0],
        max_work_false_positive_rate=0.0,
        min_mixed_boundary_recall=1.0,
        min_small_contribution_recall=1.0,
    )
    assert result.transition_penalty == 0.5
    assert result.feasible is True
    assert result.failure_reason is None
    assert result.validation_scope == "development_controls_only"
    chosen = [r for r in result.rows if r.transition_penalty == result.transition_penalty][0]
    assert chosen.feasible is True
    assert chosen.work_false_positive_rate == 0.0
    assert chosen.mixed_boundary_recall == 1.0
    assert chosen.small_contribution_recall == 1.0


def test_tuning_refuses_oversmoothed_decoder_and_explains_failure():
    controls = [np.asarray([[0.9, 0.1], [0.45, 0.55], [0.9, 0.1]])]

    result = tune_decoder_on_controls(
        controls,
        [0],
        mixed_controls=[_mixed_small_contribution_control()],
        transition_penalties=[100.0],
        max_work_false_positive_rate=0.0,
        min_mixed_boundary_recall=1.0,
        min_small_contribution_recall=1.0,
    )

    assert result.feasible is False
    assert "mixed_boundary_recall" in result.failure_reason
    assert "small_contribution_recall" in result.failure_reason


def test_tuning_cannot_claim_feasibility_without_mixed_controls():
    controls = [np.asarray([[0.9, 0.1], [0.9, 0.1]])]

    with pytest.raises(ValueError, match="mixed_controls are required"):
        tune_decoder_on_controls(
            controls,
            [0],
            transition_penalties=[0.0, 1.0],
        )
