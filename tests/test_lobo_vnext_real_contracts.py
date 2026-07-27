from __future__ import annotations

import copy
import json

import pytest

from stylo.domain.lobo_vnext import (
    InnerCVPlan,
    InnerFoldPlan,
    ModelSpec,
    VNextContractError,
    canonical_sha256,
)
from stylo.domain.lobo_vnext_approval import (
    DecisionBindings,
    ReviewedEvidence,
    build_owner_decision_record,
)
from stylo.domain.lobo_vnext_real import (
    BOUNDED_EXPLORATORY_AUTHORIZATION,
    REAL_EXECUTION_SPEC_SCHEMA_VERSION,
    REAL_EXPLORATORY_OUTPUT_ROOT,
    REQUIRED_RECEIPT_KINDS,
    CampaignManifest,
    IndependentDerivationReceipt,
    ModelRoleManifest,
    OutputNamespaceContract,
    RealCorpusExecutionSpec,
    RealExecutionBindings,
    inner_cv_receipt_subject_digest,
    loads_campaign_manifest,
    loads_model_role_manifest,
    loads_real_execution_spec,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _model(model_id: str, family: str) -> ModelSpec:
    return ModelSpec.build(
        model_id=model_id,
        family=family,
        features=[f"{model_id}-features.v1"],
        weighting="work_balanced",
        hyperparameters={"contract": f"{model_id}.r1"},
        seeds={"model": 42},
        requires_inner_cv=False,
        inner_cv_splits=None,
        supports_component_aware_inner_cv=False,
        approved_for_exploratory=True,
        owner_selected=True,
    )


def _models() -> tuple[ModelSpec, ModelSpec]:
    return _model("stylo", "stylo"), _model("char_cos", "char_cos")


def _inner_plan(
    model: ModelSpec,
    *,
    fold_manifest_digest: str = "",
    content_component_manifest_digest: str = "",
) -> InnerCVPlan:
    fold_digest = fold_manifest_digest or _digest("folds")
    content_digest = (
        content_component_manifest_digest or _digest("components")
    )
    fold_plan = InnerFoldPlan.build(
        fold_id="fold-a",
        fold_spec_digest=_digest("fold-a"),
        splits=(),
    )
    payload = {
        "schema_version": "stylo.lobo-vnext.inner-cv-plan.v1",
        "fold_manifest_digest": fold_digest,
        "content_component_manifest_digest": content_digest,
        "model_spec_digest": model.self_hash,
        "plans": [fold_plan.to_dict()],
    }
    return InnerCVPlan.from_dict(
        {**payload, "self_hash": canonical_sha256(payload)}
    )


def _roles() -> tuple[
    ModelRoleManifest,
    ModelSpec,
    ModelSpec,
    InnerCVPlan,
    InnerCVPlan,
]:
    primary, baseline = _models()
    primary_inner = _inner_plan(primary)
    baseline_inner = _inner_plan(baseline)
    return (
        ModelRoleManifest.build(
            primary_model_spec=primary,
            baseline_model_spec=baseline,
            primary_inner_cv_plan=primary_inner,
            baseline_inner_cv_plan=baseline_inner,
        ),
        primary,
        baseline,
        primary_inner,
        baseline_inner,
    )


def _campaign(
    roles: ModelRoleManifest | None = None,
) -> tuple[
    CampaignManifest,
    ModelRoleManifest,
    ModelSpec,
    ModelSpec,
    InnerCVPlan,
    InnerCVPlan,
]:
    if roles is None:
        roles, primary, baseline, primary_inner, baseline_inner = _roles()
    else:
        primary, baseline = _models()
        primary_inner = _inner_plan(primary)
        baseline_inner = _inner_plan(baseline)
    return (
        CampaignManifest.build(
            campaign_id="ruaa-r1-bounded-exploratory",
            fold_manifest_digest=_digest("folds"),
            inference_spec_digest=_digest("inference"),
            model_role_manifest=roles,
        ),
        roles,
        primary,
        baseline,
        primary_inner,
        baseline_inner,
    )


def _receipts(
    bindings: RealExecutionBindings | None = None,
) -> tuple[IndependentDerivationReceipt, ...]:
    selected = bindings or _bindings()
    subject_by_kind = {
        "packet_selection": selected.packet_manifest_digest,
        "content_candidates": selected.candidate_inventory_digest,
        "content_components": selected.content_component_manifest_digest,
        "folds": selected.fold_manifest_digest,
        "inner_cv": inner_cv_receipt_subject_digest(
            primary_inner_cv_plan_digest=(
                selected.primary_inner_cv_plan_digest
            ),
            baseline_inner_cv_plan_digest=(
                selected.baseline_inner_cv_plan_digest
            ),
        ),
        "config": selected.config_digest,
    }
    return tuple(
        IndependentDerivationReceipt.build(
            kind=kind,
            derivation_version=f"stylo.derive-{kind}.v1",
            expected_digest=subject_by_kind.get(
                kind, _digest(f"{kind}-subject")
            ),
            observed_digest=subject_by_kind.get(
                kind, _digest(f"{kind}-subject")
            ),
            evidence_digest=_digest(f"{kind}-evidence"),
            observation_count=index + 1,
        )
        for index, kind in enumerate(REQUIRED_RECEIPT_KINDS)
    )


def _bindings(
    *,
    campaign: CampaignManifest | None = None,
    roles: ModelRoleManifest | None = None,
    primary: ModelSpec | None = None,
    baseline: ModelSpec | None = None,
    primary_inner: InnerCVPlan | None = None,
    baseline_inner: InnerCVPlan | None = None,
) -> RealExecutionBindings:
    if (
        campaign is None
        or roles is None
        or primary is None
        or baseline is None
        or primary_inner is None
        or baseline_inner is None
    ):
        (
            campaign,
            roles,
            primary,
            baseline,
            primary_inner,
            baseline_inner,
        ) = _campaign()
    return RealExecutionBindings(
        packet_manifest_digest=_digest("packet-manifest"),
        content_policy_spec_digest=_digest("policy"),
        candidate_inventory_digest=_digest("candidates"),
        corpus_manifest_digest=_digest("corpus"),
        content_component_manifest_digest=_digest("components"),
        fold_manifest_digest=campaign.fold_manifest_digest,
        primary_inner_cv_plan_digest=primary_inner.self_hash,
        baseline_inner_cv_plan_digest=baseline_inner.self_hash,
        primary_model_spec_digest=primary.self_hash,
        baseline_model_spec_digest=baseline.self_hash,
        model_role_manifest_digest=roles.self_hash,
        inference_spec_digest=campaign.inference_spec_digest,
        campaign_manifest_digest=campaign.self_hash,
        config_digest=_digest("config"),
    )


def _execution(
) -> tuple[
    RealCorpusExecutionSpec,
    CampaignManifest,
    ModelRoleManifest,
    ModelSpec,
    ModelSpec,
    InnerCVPlan,
    InnerCVPlan,
]:
    (
        campaign,
        roles,
        primary,
        baseline,
        primary_inner,
        baseline_inner,
    ) = _campaign()
    bindings = _bindings(
        campaign=campaign,
        roles=roles,
        primary=primary,
        baseline=baseline,
        primary_inner=primary_inner,
        baseline_inner=baseline_inner,
    )
    spec = RealCorpusExecutionSpec.build(
        bindings=bindings,
        independent_receipts=_receipts(bindings),
        output_namespace=OutputNamespaceContract.build(
            namespace_id="ruaa-r1"
        ),
    )
    return (
        spec,
        campaign,
        roles,
        primary,
        baseline,
        primary_inner,
        baseline_inner,
    )


def _rehash(raw: dict[str, object]) -> None:
    raw["self_hash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )


def _rehash_execution(raw: dict[str, object]) -> None:
    _rehash(raw)


def test_model_roles_bind_exact_order_and_model_specs():
    roles, primary, baseline, primary_inner, baseline_inner = _roles()
    raw = roles.to_dict()

    assert [row["role"] for row in raw["roles"]] == [
        "primary",
        "baseline",
    ]
    assert [
        (row["model_id"], row["weighting"]) for row in raw["roles"]
    ] == [
        ("stylo", "work_balanced"),
        ("char_cos", "work_balanced"),
    ]
    assert [row["inner_cv_plan_digest"] for row in raw["roles"]] == [
        primary_inner.self_hash,
        baseline_inner.self_hash,
    ]
    assert loads_model_role_manifest(json.dumps(raw)) == roles
    assert (
        roles.assert_model_specs(
            primary_model_spec=primary,
            baseline_model_spec=baseline,
            primary_inner_cv_plan=primary_inner,
            baseline_inner_cv_plan=baseline_inner,
        )
        is roles
    )


def test_model_roles_require_two_model_specific_empty_inner_plans():
    primary, baseline = _models()
    primary_inner = _inner_plan(primary)
    baseline_inner = _inner_plan(baseline)

    with pytest.raises(VNextContractError, match="exact ModelSpec"):
        ModelRoleManifest.build(
            primary_model_spec=primary,
            baseline_model_spec=baseline,
            primary_inner_cv_plan=baseline_inner,
            baseline_inner_cv_plan=baseline_inner,
        )

    different_fold = _inner_plan(
        primary,
        fold_manifest_digest=_digest("different-folds"),
    )
    with pytest.raises(VNextContractError, match="same fold/content"):
        ModelRoleManifest.build(
            primary_model_spec=primary,
            baseline_model_spec=baseline,
            primary_inner_cv_plan=different_fold,
            baseline_inner_cv_plan=baseline_inner,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "swap",
        "wrong_primary",
        "wrong_family",
        "wrong_weighting",
        "bool_ordinal",
        "extra",
    ],
)
def test_model_roles_reject_rehashed_malformed_records(mutation):
    raw = copy.deepcopy(_roles()[0].to_dict())
    if mutation == "swap":
        raw["roles"].reverse()
    elif mutation == "wrong_primary":
        raw["roles"][0]["model_id"] = "stylo_stack"
    elif mutation == "wrong_family":
        raw["roles"][0]["family"] = "stylo_stack"
    elif mutation == "wrong_weighting":
        raw["roles"][1]["weighting"] = "chunk_weighted_legacy"
    elif mutation == "bool_ordinal":
        raw["roles"][0]["ordinal"] = False
    else:
        raw["roles"][0]["unexpected"] = "forged"
    _rehash(raw)

    with pytest.raises(
        VNextContractError,
        match="ordered exactly|exact integer|keys must be exact",
    ):
        ModelRoleManifest.from_dict(raw)


def test_model_role_digest_does_not_mask_model_spec_drift():
    roles, primary, baseline, primary_inner, baseline_inner = _roles()
    drifted = ModelSpec.build(
        model_id="stylo",
        family="stylo",
        features=["different-feature-contract.v2"],
        weighting="work_balanced",
        hyperparameters={"contract": "different"},
        seeds={"model": 42},
        requires_inner_cv=False,
        inner_cv_splits=None,
        supports_component_aware_inner_cv=False,
        approved_for_exploratory=True,
        owner_selected=True,
    )

    with pytest.raises(VNextContractError, match="differs from its role"):
        roles.assert_model_specs(
            primary_model_spec=drifted,
            baseline_model_spec=baseline,
            primary_inner_cv_plan=primary_inner,
            baseline_inner_cv_plan=baseline_inner,
        )
    assert drifted.self_hash != primary.self_hash


def test_campaign_binds_ordered_primary_and_baseline_runs():
    campaign, roles, *_ = _campaign()
    raw = campaign.to_dict()

    assert [row["role"] for row in raw["ordered_runs"]] == [
        "primary",
        "baseline",
    ]
    assert loads_campaign_manifest(json.dumps(raw)) == campaign
    assert campaign.assert_model_roles(roles) is campaign


@pytest.mark.parametrize(
    "mutation", ["swap", "missing", "digest", "inner_digest", "extra"]
)
def test_campaign_rejects_malformed_or_role_conflicting_runs(mutation):
    campaign, roles, *_ = _campaign()
    raw = copy.deepcopy(campaign.to_dict())
    if mutation == "swap":
        raw["ordered_runs"].reverse()
    elif mutation == "missing":
        raw["ordered_runs"].pop()
    elif mutation == "digest":
        raw["ordered_runs"][0]["model_spec_digest"] = _digest("forged")
    elif mutation == "inner_digest":
        raw["ordered_runs"][0]["inner_cv_plan_digest"] = _digest(
            "forged-inner-plan"
        )
    else:
        raw["ordered_runs"][0]["extra"] = False
    _rehash(raw)

    if mutation in {"digest", "inner_digest"}:
        parsed = CampaignManifest.from_dict(raw)
        with pytest.raises(VNextContractError, match="differ from ordered"):
            parsed.assert_model_roles(roles)
    else:
        with pytest.raises(
            VNextContractError,
            match="ordered primary|exactly primary|keys must be exact",
        ):
            CampaignManifest.from_dict(raw)


def test_independent_receipt_requires_exact_observed_match_and_nonbool_count():
    receipt = _receipts()[0]
    assert IndependentDerivationReceipt.from_dict(receipt.to_dict()) == receipt

    mismatch = copy.deepcopy(receipt.to_dict())
    mismatch["observed_digest"] = _digest("different-observation")
    _rehash(mismatch)
    with pytest.raises(VNextContractError, match="observed digest differs"):
        IndependentDerivationReceipt.from_dict(mismatch)

    coercible = copy.deepcopy(receipt.to_dict())
    coercible["observation_count"] = True
    _rehash(coercible)
    with pytest.raises(VNextContractError, match="exact integer"):
        IndependentDerivationReceipt.from_dict(coercible)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("root_relative_path", "/tmp/exploratory", "relative POSIX path"),
        (
            "root_relative_path",
            "docs/confirmatory/lobo_vnext",
            "exact literal",
        ),
        ("create_if_absent_only", False, "exact literal True"),
        ("confirmatory_output_authorized", True, "exact literal False"),
    ],
)
def test_output_namespace_is_exact_and_has_no_publication_authority(
    field, value, match
):
    raw = copy.deepcopy(
        OutputNamespaceContract.build(namespace_id="ruaa-r1").to_dict()
    )
    raw[field] = value
    _rehash(raw)

    with pytest.raises(VNextContractError, match=match):
        OutputNamespaceContract.from_dict(raw)


def test_execution_v2_is_strict_real_lobo_exploratory_only():
    (
        execution,
        campaign,
        roles,
        primary,
        baseline,
        primary_inner,
        baseline_inner,
    ) = _execution()
    raw = execution.to_dict()

    assert raw["schema_version"] == REAL_EXECUTION_SPEC_SCHEMA_VERSION
    assert raw["execution_mode"] == "real_corpus"
    assert raw["authorization_scope"] == BOUNDED_EXPLORATORY_AUTHORIZATION
    assert raw["evaluation_strategy"] == "lobo"
    assert raw["output_namespace"]["root_relative_path"] == (
        REAL_EXPLORATORY_OUTPUT_ROOT
    )
    assert all(value is False for value in raw["safety"].values())
    assert [row["kind"] for row in raw["independent_receipts"]] == list(
        REQUIRED_RECEIPT_KINDS
    )
    assert len(REQUIRED_RECEIPT_KINDS) == 15
    assert REQUIRED_RECEIPT_KINDS[7:11] == (
        "config",
        "primary_model_adapter",
        "baseline_model_adapter",
        "executable_sources",
    )
    assert loads_real_execution_spec(json.dumps(raw)) == execution
    assert (
        execution.assert_campaign(
            campaign_manifest=campaign,
            model_role_manifest=roles,
            primary_model_spec=primary,
            baseline_model_spec=baseline,
            primary_inner_cv_plan=primary_inner,
            baseline_inner_cv_plan=baseline_inner,
        )
        is execution
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "top_extra",
        "binding_missing",
        "receipt_extra",
        "receipt_missing",
        "receipt_reordered",
        "gkf",
        "confirmatory",
        "self_hash",
    ],
)
def test_execution_rejects_rehashed_adversarial_payloads(mutation):
    raw = copy.deepcopy(_execution()[0].to_dict())
    if mutation == "top_extra":
        raw["legacy_fallback"] = True
    elif mutation == "binding_missing":
        del raw["bindings"]["candidate_inventory_digest"]
    elif mutation == "receipt_extra":
        raw["independent_receipts"].append(
            copy.deepcopy(raw["independent_receipts"][0])
        )
    elif mutation == "receipt_missing":
        raw["independent_receipts"].pop()
    elif mutation == "receipt_reordered":
        raw["independent_receipts"][0:2] = reversed(
            raw["independent_receipts"][0:2]
        )
    elif mutation == "gkf":
        raw["evaluation_strategy"] = "GKF"
    elif mutation == "confirmatory":
        raw["safety"]["confirmatory_execution_authorized"] = True
    else:
        raw["self_hash"] = _digest("forged")
        with pytest.raises(VNextContractError, match="self_hash mismatch"):
            RealCorpusExecutionSpec.from_dict(raw)
        return
    _rehash_execution(raw)

    with pytest.raises(
        VNextContractError,
        match=(
            "keys must be exact|every required kind|GKF|exact literal False"
        ),
    ):
        RealCorpusExecutionSpec.from_dict(raw)


@pytest.mark.parametrize(
    "kind",
    [
        "content_candidates",
        "content_components",
        "folds",
        "inner_cv",
        "config",
    ],
)
def test_execution_rejects_receipt_subject_detached_from_binding(kind):
    raw = copy.deepcopy(_execution()[0].to_dict())
    receipt = next(
        row for row in raw["independent_receipts"] if row["kind"] == kind
    )
    forged = _digest(f"detached-{kind}")
    receipt["expected_digest"] = forged
    receipt["observed_digest"] = forged
    _rehash(receipt)
    _rehash(raw)

    with pytest.raises(VNextContractError, match="receipt subject mismatch"):
        RealCorpusExecutionSpec.from_dict(raw)


def test_execution_cross_check_rejects_model_specific_inner_plan_drift():
    (
        execution,
        campaign,
        roles,
        primary,
        baseline,
        primary_inner,
        baseline_inner,
    ) = _execution()
    raw = copy.deepcopy(execution.to_dict())
    raw["bindings"]["primary_inner_cv_plan_digest"] = _digest(
        "forged-primary-inner-plan"
    )
    inner_subject = inner_cv_receipt_subject_digest(
        primary_inner_cv_plan_digest=(
            raw["bindings"]["primary_inner_cv_plan_digest"]
        ),
        baseline_inner_cv_plan_digest=(
            raw["bindings"]["baseline_inner_cv_plan_digest"]
        ),
    )
    inner_receipt = next(
        row
        for row in raw["independent_receipts"]
        if row["kind"] == "inner_cv"
    )
    inner_receipt["expected_digest"] = inner_subject
    inner_receipt["observed_digest"] = inner_subject
    _rehash(inner_receipt)
    _rehash(raw)
    drifted = RealCorpusExecutionSpec.from_dict(raw)

    with pytest.raises(VNextContractError, match="primary_inner_cv_plan_digest"):
        drifted.assert_campaign(
            campaign_manifest=campaign,
            model_role_manifest=roles,
            primary_model_spec=primary,
            baseline_model_spec=baseline,
            primary_inner_cv_plan=primary_inner,
            baseline_inner_cv_plan=baseline_inner,
        )


def test_strict_loaders_reject_duplicate_keys_and_nonfinite_numbers():
    role_text = json.dumps(_roles()[0].to_dict(), separators=(",", ":"))
    duplicate = role_text.replace(
        '"schema_version":',
        '"schema_version":"forged","schema_version":',
        1,
    )
    with pytest.raises(VNextContractError, match="duplicate object key"):
        loads_model_role_manifest(duplicate)

    execution_text = json.dumps(
        _execution()[0].to_dict(), separators=(",", ":")
    )
    nonfinite = execution_text.replace('"observation_count":1', '"observation_count":NaN')
    with pytest.raises(VNextContractError, match="non-finite"):
        loads_real_execution_spec(nonfinite)


def test_external_owner_decision_binds_complete_execution_without_hash_cycle():
    execution, *_ = _execution()
    decision = build_owner_decision_record(
        decision_id="ruaa-r1-bounded-exploratory-2026-07-27",
        decision_revision=1,
        decision_date="2026-07-27",
        owner_id="owner:test",
        owner_role="scientific owner",
        bindings=DecisionBindings(
            corpus_manifest_digest=execution.bindings.corpus_manifest_digest,
            content_component_manifest_digest=(
                execution.bindings.content_component_manifest_digest
            ),
            policy_manifest_digest=(
                execution.bindings.content_policy_spec_digest
            ),
            fold_manifest_digest=execution.bindings.fold_manifest_digest,
            campaign_manifest_digest=(
                execution.bindings.campaign_manifest_digest
            ),
            model_role_manifest_digest=(
                execution.bindings.model_role_manifest_digest
            ),
            inference_spec_digest=execution.bindings.inference_spec_digest,
            execution_spec_digest=execution.self_hash,
        ),
        reviewed_evidence=(
            ReviewedEvidence(
                "research/candidates/ruaa-r1.json",
                _digest("reviewed-evidence"),
            ),
        ),
        affected_contract_versions=(
            REAL_EXECUTION_SPEC_SCHEMA_VERSION,
        ),
    )

    assert "owner_decision_digest" not in execution.to_dict()["bindings"]
    assert "owner_decision_digest" not in execution.to_dict()
    assert decision.bindings.execution_spec_digest == execution.self_hash
    assert execution.assert_owner_decision(decision) is execution


def test_owner_decision_cannot_bind_a_different_execution_contract():
    preliminary, *_ = _execution()
    decision = build_owner_decision_record(
        decision_id="ruaa-r1-bounded-exploratory-2026-07-27",
        decision_revision=1,
        decision_date="2026-07-27",
        owner_id="owner:test",
        owner_role="scientific owner",
        bindings=DecisionBindings(
            corpus_manifest_digest=preliminary.bindings.corpus_manifest_digest,
            content_component_manifest_digest=(
                preliminary.bindings.content_component_manifest_digest
            ),
            policy_manifest_digest=(
                preliminary.bindings.content_policy_spec_digest
            ),
            fold_manifest_digest=preliminary.bindings.fold_manifest_digest,
            campaign_manifest_digest=(
                preliminary.bindings.campaign_manifest_digest
            ),
            model_role_manifest_digest=(
                preliminary.bindings.model_role_manifest_digest
            ),
            inference_spec_digest=preliminary.bindings.inference_spec_digest,
            execution_spec_digest=_digest("different-execution"),
        ),
        reviewed_evidence=(
            ReviewedEvidence(
                "research/candidates/ruaa-r1.json",
                _digest("reviewed-evidence"),
            ),
        ),
        affected_contract_versions=(REAL_EXECUTION_SPEC_SCHEMA_VERSION,),
    )
    with pytest.raises(VNextContractError, match="does not bind the exact"):
        preliminary.assert_owner_decision(decision)
