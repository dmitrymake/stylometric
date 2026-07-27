from __future__ import annotations

from copy import deepcopy

import pytest

from stylo.models.stacked_clf import (
    STACK_PASSPORT_SCHEMA_V1,
    STACK_PASSPORT_SCHEMA_V2,
    STACK_SELECTION_EVIDENCE_STATUS,
    project_stack_passport_compatibility,
    validate_withdrawn_internal_selection_diagnostic,
    withdrawn_internal_selection_diagnostic,
)


def _current_passport() -> dict:
    return {
        "mode": "equal",
        "inner_oof_book_top1": withdrawn_internal_selection_diagnostic(0.8, 0.7),
        "calibration": {"char": {"method": "identity", "heldout_nll": 0.25}},
        "calibration_disabled": False,
    }


def _project(passport: dict) -> dict:
    return project_stack_passport_compatibility(
        passport,
        source_schema_version=STACK_PASSPORT_SCHEMA_V2,
        target_schema_version=STACK_PASSPORT_SCHEMA_V1,
    )


def test_current_wrapper_is_explicitly_withdrawn_and_ineligible():
    diagnostic = withdrawn_internal_selection_diagnostic(0.8, None)

    validate_withdrawn_internal_selection_diagnostic(diagnostic)
    assert diagnostic == {
        "status": STACK_SELECTION_EVIDENCE_STATUS,
        "eligible_as_unbiased_evidence": False,
        "descriptive_only": {"equal": 0.8, "stacked": None},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(status="current"),
        lambda value: value.update(eligible_as_unbiased_evidence=True),
        lambda value: value["descriptive_only"].update(equal="0.8"),
        lambda value: value["descriptive_only"].update(stacked=float("nan")),
        lambda value: value["descriptive_only"].update(extra=0.1),
    ],
)
def test_current_wrapper_rejects_payloads_that_weaken_or_malform_safety(mutation):
    diagnostic = withdrawn_internal_selection_diagnostic(0.8, 0.7)
    mutation(diagnostic)

    with pytest.raises(ValueError):
        validate_withdrawn_internal_selection_diagnostic(diagnostic)


def test_v2_to_v1_projection_changes_only_the_descriptive_wrapper():
    current = _current_passport()
    before = deepcopy(current)

    historical = _project(current)

    assert current == before
    assert historical == {
        **before,
        "inner_oof_book_top1": {"equal": 0.8, "stacked": 0.7},
    }
    assert historical["calibration"] == current["calibration"]


def test_projection_preserves_descriptive_numeric_drift_for_exact_comparison():
    current = _current_passport()
    current["inner_oof_book_top1"]["descriptive_only"]["equal"] = 0.81

    historical = _project(current)

    assert historical["inner_oof_book_top1"]["equal"] == 0.81
    assert historical["inner_oof_book_top1"] != {"equal": 0.8, "stacked": 0.7}


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (STACK_PASSPORT_SCHEMA_V1, STACK_PASSPORT_SCHEMA_V1),
        (STACK_PASSPORT_SCHEMA_V2, STACK_PASSPORT_SCHEMA_V2),
        ("stylo.stack.passport.future", STACK_PASSPORT_SCHEMA_V1),
    ],
)
def test_projection_allows_only_the_explicit_current_to_historical_edge(source, target):
    with pytest.raises(ValueError):
        project_stack_passport_compatibility(
            _current_passport(),
            source_schema_version=source,
            target_schema_version=target,
        )
