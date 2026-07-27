from __future__ import annotations

import copy
import json

import pytest

from stylo.domain.lobo_vnext import canonical_sha256
from stylo.domain.lobo_vnext_approval import (
    EXPLORATORY_CHOSEN_OPTION,
    OWNER_DECISION_SCHEMA_VERSION,
    REAL_CORPUS_EXPLORATORY_SCOPE,
    SELF_HASH_SEMANTICS,
    DecisionBindings,
    ExploratoryOwnerDecisionRecord,
    OwnerDecisionContractError,
    ReviewedEvidence,
    build_owner_decision_record,
    load_owner_decision_record,
    loads_owner_decision_record,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _bindings(**overrides: str) -> DecisionBindings:
    values = {
        "corpus_manifest_digest": _digest("corpus"),
        "content_component_manifest_digest": _digest("content"),
        "policy_manifest_digest": _digest("policies"),
        "fold_manifest_digest": _digest("folds"),
        "campaign_manifest_digest": _digest("campaign"),
        "model_role_manifest_digest": _digest("model-roles"),
        "inference_spec_digest": _digest("inference"),
        "execution_spec_digest": _digest("execution"),
    }
    values.update(overrides)
    return DecisionBindings(**values)


def _record(
    *,
    bindings: DecisionBindings | None = None,
) -> ExploratoryOwnerDecisionRecord:
    return build_owner_decision_record(
        decision_id="lobo-vnext-dry-run-2026-07-27",
        decision_revision=1,
        decision_date="2026-07-27",
        owner_id="owner:dmitrymake",
        owner_role="scientific owner",
        bindings=bindings or _bindings(),
        reviewed_evidence=(
            ReviewedEvidence(
                "research/candidates/policies.json",
                _digest("policy-evidence"),
            ),
            ReviewedEvidence(
                "research/candidates/corpus.json",
                _digest("corpus-evidence"),
            ),
        ),
        affected_contract_versions=(
            "stylo.lobo-vnext.corpus-manifest.v1",
            "stylo.lobo-vnext.execution-spec.v2",
        ),
    )


def _rehash(raw: dict[str, object]) -> None:
    raw["self_hash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )


def test_build_round_trip_binds_every_digest_and_encodes_non_authentication(
    tmp_path,
):
    record = _record()
    raw = record.to_dict()

    assert raw["schema_version"] == OWNER_DECISION_SCHEMA_VERSION
    assert raw["scope"] == REAL_CORPUS_EXPLORATORY_SCOPE
    assert raw["chosen_option"] == EXPLORATORY_CHOSEN_OPTION
    assert raw["approved_for_exploratory"] is True
    assert set(raw["bindings"]) == {
        "corpus_manifest_digest",
        "content_component_manifest_digest",
        "policy_manifest_digest",
        "fold_manifest_digest",
        "campaign_manifest_digest",
        "model_role_manifest_digest",
        "inference_spec_digest",
        "execution_spec_digest",
    }
    assert all(value is False for value in raw["safety"].values())
    assert raw["integrity_contract"] == {
        "self_hash_semantics": SELF_HASH_SEMANTICS,
        "cryptographic_owner_authentication": False,
    }
    assert [row["relative_path"] for row in raw["reviewed_evidence"]] == [
        "research/candidates/corpus.json",
        "research/candidates/policies.json",
    ]
    assert raw["self_hash"] == canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )

    text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    assert loads_owner_decision_record(text) == record
    path = tmp_path / "owner-decision.json"
    path.write_text(text, encoding="utf-8")
    assert load_owner_decision_record(path) == record


def test_every_binding_is_part_of_the_self_hash():
    baseline = _record()
    fields = tuple(_bindings().to_dict())

    changed_hashes = {
        _record(bindings=_bindings(**{field: _digest(f"changed-{field}")})).self_hash
        for field in fields
    }

    assert len(changed_hashes) == len(fields)
    assert baseline.self_hash not in changed_hashes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewed_evidence", "research/candidate.json"),
        (
            "affected_contract_versions",
            "stylo.lobo-vnext.corpus-manifest.v1",
        ),
    ],
)
def test_builder_rejects_string_as_a_sequence(field, value):
    kwargs = {
        "decision_id": "lobo-vnext-dry-run-2026-07-27",
        "decision_revision": 1,
        "decision_date": "2026-07-27",
        "owner_id": "owner:dmitrymake",
        "owner_role": "scientific owner",
        "bindings": _bindings(),
        "reviewed_evidence": [
            ReviewedEvidence("research/candidate.json", _digest("candidate"))
        ],
        "affected_contract_versions": [
            "stylo.lobo-vnext.corpus-manifest.v1"
        ],
    }
    kwargs[field] = value

    with pytest.raises(OwnerDecisionContractError, match="exact list or tuple"):
        build_owner_decision_record(**kwargs)


def test_strict_json_rejects_duplicate_keys():
    text = json.dumps(_record().to_dict(), separators=(",", ":"))
    duplicate = text.replace(
        '"decision_id":',
        '"decision_id":"forged","decision_id":',
        1,
    )

    with pytest.raises(
        OwnerDecisionContractError, match="duplicate object key"
    ):
        loads_owner_decision_record(duplicate)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_strict_json_rejects_nonfinite_numbers(constant):
    text = json.dumps(_record().to_dict(), separators=(",", ":"))
    malformed = text.replace('"decision_revision":1', f'"decision_revision":{constant}')

    with pytest.raises(OwnerDecisionContractError, match="non-finite|overflows"):
        loads_owner_decision_record(malformed)


@pytest.mark.parametrize(
    "mutation",
    ["top_extra", "top_missing", "nested_extra", "nested_missing"],
)
def test_exact_key_sets_reject_extra_and_missing_even_when_rehashed(mutation):
    raw = copy.deepcopy(_record().to_dict())
    if mutation == "top_extra":
        raw["signature"] = "not-a-supported-field"
    elif mutation == "top_missing":
        del raw["chosen_option"]
    elif mutation == "nested_extra":
        raw["bindings"]["extra_digest"] = _digest("extra")
    else:
        del raw["safety"]["headline_update_authorized"]
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match="keys must be exact"):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


def test_bool_is_not_accepted_as_integer_revision_after_valid_rehash():
    raw = copy.deepcopy(_record().to_dict())
    raw["decision_revision"] = True
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match="exact integer"):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scope", "confirmatory", "scope must be the exact literal"),
        (
            "chosen_option",
            "approve_confirmatory_execution",
            "chosen_option must be the exact literal",
        ),
        (
            "approved_for_exploratory",
            False,
            "approved_for_exploratory must be the exact literal True",
        ),
        (
            "approved_for_exploratory",
            1,
            "approved_for_exploratory must be an exact boolean",
        ),
    ],
)
def test_scope_choice_and_approval_are_exact(field, value, match):
    raw = copy.deepcopy(_record().to_dict())
    raw[field] = value
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match=match):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


@pytest.mark.parametrize(
    "flag",
    [
        "confirmatory_execution_authorized",
        "public_evidence_update_authorized",
        "headline_update_authorized",
        "frozen_evidence_mutation_authorized",
    ],
)
@pytest.mark.parametrize("value", [True, 0, "false", None])
def test_every_unsafe_authorization_flag_must_be_exact_false(flag, value):
    raw = copy.deepcopy(_record().to_dict())
    raw["safety"][flag] = value
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match=flag):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("self_hash_semantics", "signed_by_owner"),
        ("cryptographic_owner_authentication", True),
        ("cryptographic_owner_authentication", 0),
    ],
)
def test_self_hash_cannot_be_relabelled_as_owner_authentication(field, value):
    raw = copy.deepcopy(_record().to_dict())
    raw["integrity_contract"][field] = value
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match=field):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


@pytest.mark.parametrize(
    "path",
    [
        "/home/dmake/private/corpus.json",
        "C:/Users/owner/corpus.json",
        r"C:\Users\owner\corpus.json",
        "~/private/corpus.json",
        "../private/corpus.json",
        "research//candidate.json",
    ],
)
def test_reviewed_evidence_rejects_host_absolute_or_noncanonical_paths(path):
    raw = copy.deepcopy(_record().to_dict())
    raw["reviewed_evidence"][0]["relative_path"] = path
    _rehash(raw)

    with pytest.raises(
        OwnerDecisionContractError, match="canonical relative POSIX path"
    ):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


@pytest.mark.parametrize(
    ("owner_field", "value"),
    [
        ("owner_id", "/home/dmake/private"),
        ("owner_role", r"C:\Users\owner"),
        ("owner_role", "~/private"),
    ],
)
def test_owner_fields_reject_absolute_host_paths(owner_field, value):
    raw = copy.deepcopy(_record().to_dict())
    raw["owner"][owner_field] = value
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match="absolute host path"):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


@pytest.mark.parametrize(
    "value",
    [
        True,
        "A" * 64,
        "f" * 63,
        "g" * 64,
        7,
    ],
)
def test_binding_digests_are_exact_lowercase_sha256(value):
    raw = copy.deepcopy(_record().to_dict())
    raw["bindings"]["campaign_manifest_digest"] = value
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match="campaign_manifest_digest"):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


def test_reviewed_evidence_is_required_sorted_unique_and_exact():
    raw = copy.deepcopy(_record().to_dict())
    raw["reviewed_evidence"] = []
    _rehash(raw)
    with pytest.raises(OwnerDecisionContractError, match="non-empty array"):
        ExploratoryOwnerDecisionRecord.from_dict(raw)

    raw = copy.deepcopy(_record().to_dict())
    raw["reviewed_evidence"].append(copy.deepcopy(raw["reviewed_evidence"][0]))
    _rehash(raw)
    with pytest.raises(
        OwnerDecisionContractError, match="sorted and duplicate-free"
    ):
        ExploratoryOwnerDecisionRecord.from_dict(raw)

    raw = copy.deepcopy(_record().to_dict())
    raw["reviewed_evidence"].reverse()
    _rehash(raw)
    with pytest.raises(
        OwnerDecisionContractError, match="sorted and duplicate-free"
    ):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


@pytest.mark.parametrize(
    "date",
    ["2026-02-30", "2026-7-27", "27-07-2026", True, ""],
)
def test_decision_date_is_an_exact_calendar_date(date):
    raw = copy.deepcopy(_record().to_dict())
    raw["decision_date"] = date
    _rehash(raw)

    with pytest.raises(OwnerDecisionContractError, match="decision_date"):
        ExploratoryOwnerDecisionRecord.from_dict(raw)


def test_affected_contract_versions_are_required_path_free_and_unique():
    for versions, match in (
        ([], "non-empty array"),
        (
            ["stylo.lobo-vnext.corpus-manifest.v1"] * 2,
            "sorted and duplicate-free",
        ),
        (["/home/dmake/protocol.json"], "path-free canonical token"),
    ):
        raw = copy.deepcopy(_record().to_dict())
        raw["affected_contract_versions"] = versions
        _rehash(raw)
        with pytest.raises(OwnerDecisionContractError, match=match):
            ExploratoryOwnerDecisionRecord.from_dict(raw)


def test_stale_and_semantically_rehashed_tampering_are_both_rejected():
    stale = copy.deepcopy(_record().to_dict())
    stale["owner"]["owner_role"] = "different owner role"
    with pytest.raises(OwnerDecisionContractError, match="self_hash mismatch"):
        ExploratoryOwnerDecisionRecord.from_dict(stale)

    unsafe = copy.deepcopy(_record().to_dict())
    unsafe["safety"]["confirmatory_execution_authorized"] = True
    _rehash(unsafe)
    with pytest.raises(
        OwnerDecisionContractError,
        match="confirmatory_execution_authorized",
    ):
        ExploratoryOwnerDecisionRecord.from_dict(unsafe)
