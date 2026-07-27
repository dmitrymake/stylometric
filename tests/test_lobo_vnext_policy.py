from __future__ import annotations

import json

import pytest

from stylo.domain.lobo_vnext import (
    ContentCandidate,
    ContentComponent,
    ContentComponentManifest,
    VNextContractError,
    WorkIdentity,
    canonical_sha256,
)
from stylo.domain.lobo_vnext_policy import (
    AutomaticCandidateMechanisms,
    CandidateDraft,
    CandidateInventory,
    ChunkerPolicy,
    ContentPolicySpec,
    LiteralCandidateMechanism,
    ManualDisposition,
    RawByteIdentityPolicy,
    StrictUTF8Policy,
    VersionedTextPolicy,
    Word5ContainmentPolicy,
    loads_candidate_inventory,
    loads_content_policy_spec,
)


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _text_policy(label: str, disposition: str) -> VersionedTextPolicy:
    return VersionedTextPolicy.from_dict(
        {
            "policy_version": f"synthetic-{label}.v1",
            "disposition": disposition,
            "policy_document_sha256": _sha(f"{label}-document"),
        },
        label=f"{label}_policy",
    )


def _word5() -> Word5ContainmentPolicy:
    return Word5ContainmentPolicy.from_dict(
        {
            "policy_version": "synthetic-word5.v1",
            "shingle_size": 5,
            "comparison": "asymmetric_containment",
            "threshold": {"numerator": 9, "denominator": 10},
            "threshold_boundary": "inclusive",
            "min_shingles": 20,
            "sample_size": 256,
            "final_verification": "exact_intersection_authoritative",
        }
    )


def _policy(*, word5: bool = True) -> ContentPolicySpec:
    return ContentPolicySpec.build(
        policy_id="synthetic-owner-decision.v1",
        raw_byte_identity=RawByteIdentityPolicy.from_dict(
            {
                "policy_version": "synthetic-raw-identity.v1",
                "identity_fields": [
                    "relative_path",
                    "byte_size",
                    "sha256",
                ],
                "digest_algorithm": "sha256",
            }
        ),
        strict_utf8=StrictUTF8Policy.from_dict(
            {
                "policy_version": "synthetic-utf8.v1",
                "encoding": "utf-8",
                "errors": "strict",
                "bom_disposition": "reject",
            }
        ),
        canonical_row_policy=_text_policy(
            "canonical-row", "transform_versioned"
        ),
        chunker_policy=ChunkerPolicy.from_dict(
            {
                "policy_version": "synthetic-chunker.v1",
                "mode": "fixed_words",
                "policy_document_sha256": _sha("chunker-document"),
            }
        ),
        yo_e_policy=_text_policy("yo-e", "manual_required"),
        historical_orthography_policy=_text_policy(
            "historical-orthography", "preserve"
        ),
        ocr_policy=_text_policy("ocr", "manual_required"),
        markup_policy=_text_policy("markup", "transform_versioned"),
        automatic_candidates=AutomaticCandidateMechanisms(
            exact_duplicate=LiteralCandidateMechanism.from_dict(
                {
                    "policy_version": "synthetic-exact-duplicate.v1",
                    "comparison": "literal_bytes_equal",
                },
                label="automatic_candidates.exact_duplicate",
                expected_comparison="literal_bytes_equal",
            ),
            literal_contains=LiteralCandidateMechanism.from_dict(
                {
                    "policy_version": "synthetic-literal-contains.v1",
                    "comparison": "literal_byte_subsequence",
                },
                label="automatic_candidates.literal_contains",
                expected_comparison="literal_byte_subsequence",
            ),
            word5_containment=_word5() if word5 else None,
        ),
        manual_disposition_required=True,
    )


def _rehash(raw: dict) -> dict:
    raw["self_hash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    return raw


def _draft(
    candidate_id: str = "candidate-001",
    *,
    disposition: str = "unresolved",
    manual: ManualDisposition | None = None,
) -> CandidateDraft:
    return CandidateDraft.build(
        candidate_id=candidate_id,
        left_work_id="author-a/work-1",
        right_work_id="author-b/work-2",
        edge_type="word5_asymmetric_containment",
        origin="automatic",
        evidence_sha256=_sha(f"{candidate_id}-evidence"),
        disposition=disposition,
        manual_disposition=manual,
    )


def _manual(disposition: str = "same_component") -> ManualDisposition:
    return ManualDisposition.from_dict(
        {
            "decision_id": "synthetic-decision-001",
            "disposition": disposition,
            "evidence_sha256": _sha("manual-decision-evidence"),
        }
    )


def _inventory(*candidates: CandidateDraft) -> CandidateInventory:
    policy = _policy()
    return CandidateInventory.build(
        generation_id="synthetic-generation-001",
        work_identity_catalog_digest=_sha("work-identity-catalog"),
        raw_inventory_digest=_sha("raw-inventory"),
        content_policy_spec_digest=policy.self_hash,
        included_work_ids=("author-a/work-1", "author-b/work-2"),
        candidates=candidates,
    )


def test_content_policy_is_strict_self_hashed_and_all_decisions_are_explicit():
    policy = _policy()
    raw = policy.to_dict()

    assert ContentPolicySpec.from_dict(raw) == policy
    assert loads_content_policy_spec(json.dumps(raw)) == policy
    assert raw["raw_byte_identity"]["identity_fields"] == [
        "relative_path",
        "byte_size",
        "sha256",
    ]
    assert raw["strict_utf8"] == {
        "policy_version": "synthetic-utf8.v1",
        "encoding": "utf-8",
        "errors": "strict",
        "bom_disposition": "reject",
    }
    assert raw["manual_disposition_required"] is True
    assert set(raw) == {
        "schema_version",
        "policy_id",
        "raw_byte_identity",
        "strict_utf8",
        "canonical_row_policy",
        "chunker_policy",
        "yo_e_policy",
        "historical_orthography_policy",
        "ocr_policy",
        "markup_policy",
        "automatic_candidates",
        "manual_disposition_required",
        "self_hash",
    }


def test_content_policy_rejects_duplicate_missing_extra_and_coercible_fields():
    raw = _policy().to_dict()
    duplicate = '{"schema_version":"duplicate",' + json.dumps(raw)[1:]
    with pytest.raises(VNextContractError, match="duplicate object key"):
        loads_content_policy_spec(duplicate)

    for mutator, pattern in (
        (lambda item: item.pop("ocr_policy"), "keys must be exact"),
        (
            lambda item: item.update({"unexpected": "rehashed"}),
            "keys must be exact",
        ),
        (
            lambda item: item.update({"manual_disposition_required": 1}),
            "exact boolean",
        ),
        (
            lambda item: item["strict_utf8"].update({"errors": "replace"}),
            "encoding='utf-8'.*errors='strict'",
        ),
        (
            lambda item: item["raw_byte_identity"].update(
                {
                    "identity_fields": [
                        "sha256",
                        "byte_size",
                        "relative_path",
                    ]
                }
            ),
            "identity_fields must be exactly",
        ),
    ):
        changed = json.loads(json.dumps(raw))
        mutator(changed)
        _rehash(changed)
        with pytest.raises(VNextContractError, match=pattern):
            ContentPolicySpec.from_dict(changed)


def test_word5_is_explicitly_disabled_or_exact_rational_and_conservative():
    disabled = _policy(word5=False)
    assert disabled.to_dict()["automatic_candidates"]["word5_containment"] is None

    raw = _policy().to_dict()
    word5 = raw["automatic_candidates"]["word5_containment"]
    assert word5["threshold"] == {"numerator": 9, "denominator": 10}
    assert word5["threshold_boundary"] == "inclusive"
    assert word5["final_verification"] == "exact_intersection_authoritative"

    mutations = (
        ({"numerator": 0, "denominator": 1}, "exact integer >= 1"),
        ({"numerator": 11, "denominator": 10}, "interval"),
        ({"numerator": 18, "denominator": 20}, "lowest terms"),
        ({"numerator": "9", "denominator": 10}, "exact integer"),
        ({"numerator": True, "denominator": 10}, "exact integer"),
        (0.9, "exact JSON object"),
    )
    for threshold, pattern in mutations:
        changed = json.loads(json.dumps(raw))
        changed["automatic_candidates"]["word5_containment"][
            "threshold"
        ] = threshold
        _rehash(changed)
        with pytest.raises(VNextContractError, match=pattern):
            ContentPolicySpec.from_dict(changed)


def test_candidate_draft_preserves_unresolved_but_resolution_is_manual_only():
    unresolved = _draft()
    assert unresolved.disposition == "unresolved"
    assert unresolved.manual_disposition is None
    assert not unresolved.is_resolved

    with pytest.raises(
        VNextContractError, match="requires an exact manual disposition"
    ):
        _draft(disposition="same_component")

    resolved = _draft(
        disposition="same_component",
        manual=_manual("same_component"),
    )
    assert resolved.is_resolved
    assert CandidateDraft.from_dict(resolved.to_dict()) == resolved

    with pytest.raises(VNextContractError, match="disposition mismatch"):
        _draft(
            disposition="same_component",
            manual=_manual("separate_components"),
        )


def test_candidate_draft_rejects_absolute_work_ids_and_nested_extras():
    with pytest.raises(VNextContractError, match="relative work identifier"):
        CandidateDraft.build(
            candidate_id="candidate-absolute",
            left_work_id="/private/corpus/work",
            right_work_id="author-b/work-2",
            edge_type="manual",
            origin="manual",
            evidence_sha256=_sha("absolute"),
            disposition="unresolved",
            manual_disposition=None,
        )

    raw = _draft().to_dict()
    raw["manual_disposition"] = {
        **_manual().to_dict(),
        "unexpected": "field",
    }
    raw["disposition"] = "same_component"
    _rehash(raw)
    with pytest.raises(VNextContractError, match="keys must be exact"):
        CandidateDraft.from_dict(raw)


def test_candidate_inventory_derives_unresolved_and_blocks_execution_boundary():
    unresolved = _draft("candidate-002")
    resolved = _draft(
        "candidate-001",
        disposition="same_component",
        manual=_manual(),
    )
    inventory = _inventory(unresolved, resolved)

    assert tuple(row.candidate_id for row in inventory.candidates) == (
        "candidate-001",
        "candidate-002",
    )
    assert inventory.unresolved_candidate_ids == ("candidate-002",)
    assert CandidateInventory.from_dict(inventory.to_dict()) == inventory
    with pytest.raises(
        VNextContractError,
        match="unresolved candidate drafts block ContentComponentManifest",
    ):
        inventory.assert_resolved_for_component_manifest()

    fully_resolved = _inventory(resolved)
    assert fully_resolved.assert_resolved_for_component_manifest() is fully_resolved


def test_candidate_inventory_precedes_final_components_and_corpus_manifest():
    policy = _policy()
    work_catalog = {
        "generation_id": "synthetic-generation-001",
        "raw_inventory_digest": _sha("raw-inventory"),
        "works": [
            {
                "work_id": "author-a/work-1",
                "author_id": "author-a",
                "edition_id": "edition-1",
                "source_id": "source-1",
            },
            {
                "work_id": "author-b/work-2",
                "author_id": "author-b",
                "edition_id": "edition-2",
                "source_id": "source-2",
            },
        ],
    }
    inventory = CandidateInventory.build(
        generation_id=work_catalog["generation_id"],
        work_identity_catalog_digest=canonical_sha256(work_catalog),
        raw_inventory_digest=work_catalog["raw_inventory_digest"],
        content_policy_spec_digest=policy.self_hash,
        included_work_ids=("author-a/work-1", "author-b/work-2"),
        candidates=(_draft(),),
    )

    assert inventory.work_identity_catalog_digest == canonical_sha256(
        work_catalog
    )
    assert "corpus_manifest_digest" not in inventory.to_dict()
    assert inventory.unresolved_candidate_ids == ("candidate-001",)


def test_candidate_inventory_is_strict_and_policy_digest_bound():
    inventory = _inventory(_draft())
    raw = inventory.to_dict()
    duplicate = '{"schema_version":"duplicate",' + json.dumps(raw)[1:]
    with pytest.raises(VNextContractError, match="duplicate object key"):
        loads_candidate_inventory(duplicate)

    changed = json.loads(json.dumps(raw))
    changed["unresolved_candidate_ids"] = []
    _rehash(changed)
    with pytest.raises(VNextContractError, match="exactly match"):
        CandidateInventory.from_dict(changed)

    with pytest.raises(VNextContractError, match="policy digest mismatch"):
        inventory.validate(content_policy_spec=_policy(word5=False))
    assert inventory.validate(content_policy_spec=_policy()) is inventory


def test_executable_content_component_manifest_stays_resolved_only():
    works = (
        WorkIdentity(
            "author-a/work-1",
            "author-a",
            "edition-1",
            "source-1",
            "work",
            ("author-a/work-1.txt",),
        ),
        WorkIdentity(
            "author-b/work-2",
            "author-b",
            "edition-2",
            "source-2",
            "work",
            ("author-b/work-2.txt",),
        ),
    )
    unresolved = ContentCandidate(
        "candidate-existing-boundary",
        "author-a/work-1",
        "author-b/work-2",
        "manual",
        "manual",
        "unresolved",
        _sha("existing-boundary"),
    )
    with pytest.raises(
        VNextContractError, match="unresolved content candidate"
    ):
        ContentComponentManifest.build(
            automatic_candidate_policy_version="synthetic-policy.v1",
            works=works,
            components=(
                ContentComponent("component-1", ("author-a/work-1",)),
                ContentComponent("component-2", ("author-b/work-2",)),
            ),
            candidates=(unresolved,),
        )
