import json
from pathlib import Path

import numpy as np
import pytest

from stylo.eval import fano


def _assert_descriptive_only(report):
    assert report["schema_version"] == fano.POSTERIOR_DIAGNOSTICS_SCHEMA
    assert report["semantics"] == "descriptive_model_posterior_only"
    assert report["inferential_information_or_error_bound"] is False
    forbidden = ("mutual_information", "bayes", "fano", "floor", "unavoidable")
    assert not any(
        token in key.casefold()
        for key in report
        for token in forbidden
    )


def test_constant_overconfident_binary_output_cannot_publish_information_or_error_bound():
    # The output is constant and therefore carries zero information about the
    # balanced truth.  Its Bayes error is 0.5, even though the model is highly
    # confident in class zero on every row.
    truth = np.asarray([0, 1] * 20, dtype=int)
    probabilities = np.tile(np.asarray([0.999, 0.001]), (len(truth), 1))

    report = fano.posterior_diagnostics_v2(probabilities, truth, 2)
    pair = fano.pairwise_posterior_diagnostics_v2(
        probabilities, truth, 2, [(0, 1)]
    )[0]

    assert report["empirical_error"] == 0.5
    assert report["prior_minus_model_posterior_entropy_bits"] > 0.98
    _assert_descriptive_only(report)
    assert pair["posterior_entropy_equivalent_error"] < 0.01
    assert pair["inferential_information_or_error_bound"] is False
    assert "indistinguishable" not in pair


def test_constant_overconfident_three_class_output_remains_descriptive():
    truth = np.tile(np.arange(3, dtype=int), 20)
    probabilities = np.tile(np.asarray([0.998, 0.001, 0.001]), (len(truth), 1))

    report = fano.posterior_diagnostics_v2(probabilities, truth, 3)

    assert report["empirical_error"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["prior_minus_model_posterior_entropy_bits"] > 1.5
    _assert_descriptive_only(report)


@pytest.mark.parametrize(
    "call",
    [
        lambda p, y: fano.conditional_entropy(p),
        lambda p, y: fano.mutual_information(1.0, 0.1),
        lambda p, y: fano.fano_floor(0.1, 3),
        lambda p, y: fano.binary_bayes_floor(p),
        lambda p, y: fano.fano_book_level(p, y, 2),
        lambda p, y: fano.pairwise_floor(p, y, 2, [(0, 1)]),
    ],
)
def test_legacy_inferential_entry_points_fail_closed(call):
    truth = np.asarray([0, 1], dtype=int)
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]])

    with pytest.raises(fano.WithdrawnFanoSemanticsError, match="withdrawn"):
        call(probabilities, truth)


def test_historical_fano_artifact_is_explicitly_withdrawn():
    path = Path(__file__).parents[1] / "docs" / "fano_frontier.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))

    assert artifact["artifact_status"] == "WITHDRAWN_INVALID_SCIENTIFIC_SEMANTICS"
    assert artifact["valid_lower_bound_fields"] == []
    assert "I_AF_bits" in artifact["invalid_inferential_fields"]
    assert artifact["withdrawal"]["historical_values_retained"] is True
    assert "Do not cite" in artifact["status"]
