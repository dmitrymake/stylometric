"""Compatibility import for the layer-neutral prediction contract.

New callers should import :mod:`stylo.domain.prediction_contract`; this module
keeps the established evaluation API without duplicating its semantics.
"""
from ..domain.prediction_contract import (
    PREDICTION_CONTRACT_VERSION,
    PROBABILITY_SUM_ATOL,
    PredictionContractError,
    PredictionDecision,
    stable_top1_and_worst_tie_rank,
    validate_author_universe,
    validate_channel_mapping,
    validate_class_indices,
    validate_class_order,
    validate_distances,
    validate_prediction_record,
    validate_probabilities,
    validate_probability_matrix,
    validate_probability_vector,
    validate_score_matrix,
)

__all__ = [
    "PREDICTION_CONTRACT_VERSION",
    "PROBABILITY_SUM_ATOL",
    "PredictionContractError",
    "PredictionDecision",
    "stable_top1_and_worst_tie_rank",
    "validate_author_universe",
    "validate_channel_mapping",
    "validate_class_indices",
    "validate_class_order",
    "validate_distances",
    "validate_prediction_record",
    "validate_probabilities",
    "validate_probability_matrix",
    "validate_probability_vector",
    "validate_score_matrix",
]
