"""Regression tests for the shared stable-top1 / worst-tie-rank contract."""
from __future__ import annotations

import numpy as np
import pytest

from stylo.domain.prediction_contract import (
    PredictionContractError,
    stable_top1_and_worst_tie_rank,
    validate_prediction_record,
    validate_probability_matrix,
)


def test_tied_top_uses_lowest_index_and_worst_tie_rank():
    first = stable_top1_and_worst_tie_rank(
        [0.5, 0.5], true_label=0, expected_width=2
    )
    assert first.top1 == 0
    assert first.order == (0, 1)
    assert first.true_rank == 2

    second = stable_top1_and_worst_tie_rank(
        [0.5, 0.5], true_label=1, expected_width=2
    )
    assert second.top1 == 0
    assert second.true_rank == 2


def test_correct_tied_winner_can_have_conservative_rank_greater_than_one():
    decision = validate_prediction_record(
        probabilities=[0.5, 0.5],
        pred_label=0,
        true_label=0,
        correct=True,
        rank=2,
        expected_width=2,
    )
    assert decision.top1 == 0 and decision.true_rank == 2

    with pytest.raises(PredictionContractError, match="stable top-1"):
        validate_prediction_record(
            probabilities=[0.5, 0.5],
            pred_label=1,
            true_label=1,
            correct=True,
            rank=2,
            expected_width=2,
        )
    with pytest.raises(PredictionContractError, match="worst-tie"):
        validate_prediction_record(
            probabilities=[0.5, 0.5],
            pred_label=0,
            true_label=0,
            correct=True,
            rank=1,
            expected_width=2,
        )


def test_all_equal_vector_has_full_width_rank():
    decision = stable_top1_and_worst_tie_rank(
        [0.25, 0.25, 0.25, 0.25], true_label=3
    )
    assert decision.top1 == 0
    assert decision.true_rank == 4
    assert decision.order == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("probabilities", "classes"),
    [
        ([[0.5, 0.5], [0.2, 0.7]], [0, 1]),
        ([[0.5, 0.5], [float("nan"), float("nan")]], [0, 1]),
        ([[0.5, 0.5], [-0.1, 1.1]], [0, 1]),
        ([[0.5, 0.5], [0.5, 0.5]], [1, 0]),
        ([[0.5, 0.5], [0.5, 0.5]], [False, True]),
    ],
)
def test_probability_matrix_rejects_bad_values_or_class_order(
    probabilities, classes
):
    with pytest.raises(PredictionContractError):
        validate_probability_matrix(
            np.asarray(probabilities),
            np.asarray(classes),
            n_classes=2,
            n_rows=2,
        )


def test_probability_matrix_accepts_exact_contract():
    matrix = validate_probability_matrix(
        [[0.5, 0.5], [0.2, 0.8]],
        [0, 1],
        n_classes=2,
        n_rows=2,
    )
    np.testing.assert_allclose(matrix, [[0.5, 0.5], [0.2, 0.8]])


