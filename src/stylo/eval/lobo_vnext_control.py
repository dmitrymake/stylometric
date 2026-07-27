"""Fail-closed control plane for an immutable prepared RuAA R1 packet.

The functions in this module stop at identity assembly.  They do not expose a
representation cache, model factory, fit, prediction, checkpoint, or result
writer.  A prepared packet is first reloaded from strict JSON, every file byte
and cross-manifest binding is verified, and only then may the live repository,
configuration, dependency, runtime, and thread observations be assembled into
an owner-unbound :class:`RealCorpusExecutionSpec`.
"""
from __future__ import annotations

import hashlib
import math
import os
import pathlib
import stat
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from ..config import ConfigNode
from ..domain.corpus_identity import CONTENT_OVERLAP_POLICY_VERSION
from ..domain.lobo_vnext import (
    ContentComponentManifest,
    CorpusVNextManifest,
    FoldManifest,
    InferenceSpec,
    InnerCVPlan,
    ModelSpec,
    VNextContractError,
    canonical_sha256,
    inventory_raw_files,
)
from ..domain.lobo_vnext_packet import (
    CanonicalRepresentationReceipt,
    R1PacketManifest,
    R1SourceSelectionReceipt,
    load_canonical_representation_rows,
)
from ..domain.lobo_vnext_policy import CandidateInventory, ContentPolicySpec
from ..domain.lobo_vnext_real import (
    REQUIRED_RECEIPT_KINDS,
    CampaignManifest,
    ModelRoleManifest,
    OutputNamespaceContract,
    RealCorpusExecutionSpec,
    RealExecutionBindings,
    inner_cv_receipt_subject_digest,
)
from ..jsonio import StrictJSONError, load_strict
from .lobo_vnext_models import build_r1_model_spec
from .lobo_vnext_prepare import (
    PreparedR1Packet,
    R1_AUTHOR_COUNT,
    R1_BOOTSTRAP_ITERATIONS,
    R1_BOOTSTRAP_SEED,
    R1_CHUNK_SIZE,
    R1_COLLECTION_MEMBERS,
    R1_CONFIDENCE_LEVEL,
    R1_EXCLUDED_WORK_ID,
    R1_MIN_WORDS,
    R1_OVERLAP,
    R1_PACKET_SCHEMA_VERSION,
    R1_SELECTED_BOOK_COUNT,
    build_r1_content_policy,
)
from .lobo_vnext_receipts import (
    DerivedObservation,
    assert_independent_receipts,
    build_independent_receipts,
    derive_config_and_adapter_observations,
    derive_dependency_observation,
    derive_executable_source_observation,
    derive_runtime_observation,
    derive_thread_observation,
    observation_from_canonical_value,
    observation_from_self_hashed,
)


_ARTIFACT_FILENAMES = {
    "content_policy": "manifests/content-policy.json",
    "source_selection_receipt": "manifests/source-selection.json",
    "source_candidate_inventory": (
        "manifests/source-candidate-inventory.json"
    ),
    "candidate_inventory": "manifests/candidate-inventory.json",
    "corpus_manifest": "manifests/corpus.json",
    "content_manifest": "manifests/content-components.json",
    "fold_manifest": "manifests/folds.json",
    "primary_model_spec": "manifests/model-primary.json",
    "baseline_model_spec": "manifests/model-baseline.json",
    "inference_spec": "manifests/inference.json",
    "primary_inner_cv_plan": "manifests/inner-primary.json",
    "baseline_inner_cv_plan": "manifests/inner-baseline.json",
    "model_role_manifest": "manifests/model-roles.json",
    "campaign_manifest": "manifests/campaign.json",
    "representation_receipt": "manifests/representation.json",
}
_POLICY_FILENAMES = {
    "canonicalizer": "policies/canonicalizer.json",
    "chunker": "policies/chunker.json",
    "ocr": "policies/ocr.json",
}
_EVIDENCE_KEYS = {
    "policy_version",
    "edge_type",
    "left_work_id",
    "right_work_id",
    "exact_containment_evidence",
    "reported_containment",
    "threshold",
    "threshold_boundary",
    "final_verification",
    "left_raw_paths",
    "right_raw_paths",
    "owner_selected_relation",
    "owner_selected_corpus_action",
    "evidence_sha256",
}
_HEX64 = frozenset("0123456789abcdef")


class RealControlPlaneError(VNextContractError):
    """The prepared packet or live pre-fit identity has drifted."""


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RealControlPlaneError(f"{label} must be an exact JSON object")
    actual = set(value)
    if actual != keys:
        raise RealControlPlaneError(
            f"{label} keys must be exact; "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def _exact_list(
    value: object,
    label: str,
    *,
    length: int | None = None,
) -> list[Any]:
    if type(value) is not list:
        raise RealControlPlaneError(f"{label} must be an exact JSON array")
    if length is not None and len(value) != length:
        raise RealControlPlaneError(
            f"{label} must contain exactly {length} items"
        )
    return value


def _exact_str(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise RealControlPlaneError(
            f"{label} must be an exact non-empty trimmed string"
        )
    return value


def _sha256(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if len(text) != 64 or any(char not in _HEX64 for char in text):
        raise RealControlPlaneError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return text


def _exact_int(value: object, label: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise RealControlPlaneError(
            f"{label} must be the exact integer {expected}"
        )
    return value


def _literal(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise RealControlPlaneError(
            f"{label} must be the exact literal {expected!r}"
        )


def _strict_file(root: pathlib.Path, relative_path: str, label: str) -> object:
    path = root.joinpath(*pathlib.PurePosixPath(relative_path).parts)
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise RealControlPlaneError(f"{label} is missing") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise RealControlPlaneError(f"{label} must be a regular non-symlink file")
    try:
        return load_strict(path)
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise RealControlPlaneError(f"cannot strictly load {label}: {exc}") from exc


def _scan_packet_tree(
    packet_root: str | os.PathLike[str],
) -> tuple[pathlib.Path, tuple[str, ...], tuple[str, ...]]:
    root = pathlib.Path(packet_root)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise RealControlPlaneError("prepared packet root is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RealControlPlaneError(
            "prepared packet root must be a real non-symlink directory"
        )
    root = root.resolve(strict=True)
    files: list[str] = []
    directories: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda row: row.name)
        except OSError as exc:
            raise RealControlPlaneError(
                "cannot inventory prepared packet"
            ) from exc
        for child in children:
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise RealControlPlaneError(
                    f"cannot inspect packet entry {child.path!r}"
                ) from exc
            relative = pathlib.Path(child.path).relative_to(root).as_posix()
            if stat.S_ISLNK(child_stat.st_mode):
                raise RealControlPlaneError(
                    f"symlink rejected in prepared packet: {relative}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                directories.append(relative)
                stack.append(pathlib.Path(child.path))
            elif stat.S_ISREG(child_stat.st_mode):
                files.append(relative)
            else:
                raise RealControlPlaneError(
                    f"special file rejected in prepared packet: {relative}"
                )
    return root, tuple(sorted(files)), tuple(sorted(directories))


def _expected_directories(files: Sequence[str]) -> tuple[str, ...]:
    expected: set[str] = set()
    for relative in files:
        parent = pathlib.PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected.add(parent.as_posix())
            parent = parent.parent
    return tuple(sorted(expected))


def _parse_packet_manifest(value: object) -> R1PacketManifest:
    try:
        manifest = R1PacketManifest.from_dict(value)
    except (VNextContractError, TypeError, ValueError) as exc:
        raise RealControlPlaneError(f"invalid R1 packet manifest: {exc}") from exc
    if (
        manifest.schema_version != R1_PACKET_SCHEMA_VERSION
        or manifest.selected_work_count != R1_SELECTED_BOOK_COUNT
        or manifest.generation_material.excluded_work_ids
        != (R1_EXCLUDED_WORK_ID,)
    ):
        raise RealControlPlaneError(
            "R1 packet manifest differs from the exact owner selection"
        )
    return manifest


def _validate_manifest_file_inventory(
    root: pathlib.Path,
    packet_manifest: R1PacketManifest,
    observed_files: Sequence[str],
) -> None:
    expected = ("packet.json", *(
        row.relative_path for row in packet_manifest.files
    ))
    expected = tuple(sorted(expected))
    if tuple(observed_files) != expected:
        missing = sorted(set(expected) - set(observed_files))
        extra = sorted(set(observed_files) - set(expected))
        raise RealControlPlaneError(
            f"R1 packet file inventory mismatch; missing={missing}, extra={extra}"
        )
    for row in packet_manifest.files:
        path = root.joinpath(*pathlib.PurePosixPath(row.relative_path).parts)
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RealControlPlaneError(
                f"R1 packet file is missing: {row.relative_path}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RealControlPlaneError(
                f"R1 packet file is unsafe: {row.relative_path}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if metadata.st_size != row.byte_size or digest.hexdigest() != row.sha256:
            raise RealControlPlaneError(
                f"R1 packet file bytes drifted: {row.relative_path}"
            )


def _validate_policy_documents(
    documents: Mapping[str, object],
) -> None:
    canonicalizer = _exact_object(
        documents["canonicalizer"],
        {
            "schema_version",
            "implementation_contract",
            "implementation_source_sha256",
            "encoding",
            "bom_disposition",
            "dash_normalization",
            "person_mask",
            "person_model",
            "person_model_version",
            "person_model_fallback",
            "resolved_person_model_identity",
            "yo_to_e",
            "historical_orthography_to_modern",
            "wiki_and_service_markup_cleanup",
            "quote_normalization",
            "garbage_filter",
            "whitespace_collapse",
            "ocr_correction",
        },
        "canonicalizer policy document",
    )
    expected_canonicalizer = {
        "schema_version": "stylo.lobo-vnext.canonicalizer-policy-doc.v1",
        "encoding": "utf-8",
        "bom_disposition": "reject",
        "dash_normalization": True,
        "person_mask": "@",
        "person_model": "ru_core_news_lg",
        "person_model_version": "3.8.0",
        "person_model_fallback": "ru_core_news_md",
        "yo_to_e": True,
        "historical_orthography_to_modern": True,
        "wiki_and_service_markup_cleanup": True,
        "quote_normalization": True,
        "garbage_filter": True,
        "whitespace_collapse": True,
        "ocr_correction": False,
    }
    for key, expected in expected_canonicalizer.items():
        _literal(
            canonicalizer[key],
            expected,
            f"canonicalizer policy document.{key}",
        )
    _exact_str(
        canonicalizer["implementation_contract"],
        "canonicalizer policy document.implementation_contract",
    )
    _sha256(
        canonicalizer["implementation_source_sha256"],
        "canonicalizer policy document.implementation_source_sha256",
    )
    model_identity = _exact_object(
        canonicalizer["resolved_person_model_identity"],
        {
            "requested_model",
            "resolved_model",
            "fallback_used",
            "package_version",
            "package_record_sha256",
            "spacy_version",
            "disabled_pipes",
            "active_pipes",
            "max_length",
            "identity_sha256",
        },
        "canonicalizer policy document.resolved_person_model_identity",
    )
    for key, expected in {
        "requested_model": "ru_core_news_lg",
        "resolved_model": "ru_core_news_lg",
        "fallback_used": False,
        "package_version": "3.8.0",
        "disabled_pipes": [
            "attribute_ruler",
            "lemmatizer",
            "morphologizer",
            "parser",
            "sentencizer",
            "tagger",
            "textcat",
        ],
        "active_pipes": ["tok2vec", "ner"],
        "max_length": 5_000_000,
    }.items():
        _literal(
            model_identity[key],
            expected,
            "canonicalizer policy document."
            f"resolved_person_model_identity.{key}",
        )
    _sha256(
        model_identity["package_record_sha256"],
        "canonicalizer policy document."
        "resolved_person_model_identity.package_record_sha256",
    )
    _exact_str(
        model_identity["spacy_version"],
        "canonicalizer policy document."
        "resolved_person_model_identity.spacy_version",
    )
    identity_sha = _sha256(
        model_identity["identity_sha256"],
        "canonicalizer policy document."
        "resolved_person_model_identity.identity_sha256",
    )
    if identity_sha != canonical_sha256(
        {
            key: value
            for key, value in model_identity.items()
            if key != "identity_sha256"
        }
    ):
        raise RealControlPlaneError(
            "resolved person-model identity self-hash mismatch"
        )

    chunker = _exact_object(
        documents["chunker"],
        {
            "schema_version",
            "implementation_contract",
            "implementation_source_sha256",
            "sentencizer",
            "spacy_version",
            "language",
            "sentence_aware",
            "chunk_size",
            "min_words",
            "overlap",
        },
        "chunker policy document",
    )
    expected_chunker = {
        "schema_version": "stylo.lobo-vnext.chunker-policy-doc.v1",
        "sentencizer": "spacy.blank+rule_based_sentencizer",
        "language": "ru",
        "sentence_aware": True,
        "chunk_size": R1_CHUNK_SIZE,
        "min_words": R1_MIN_WORDS,
        "overlap": R1_OVERLAP,
    }
    for key, expected in expected_chunker.items():
        _literal(chunker[key], expected, f"chunker policy document.{key}")
    _exact_str(
        chunker["implementation_contract"],
        "chunker policy document.implementation_contract",
    )
    _sha256(
        chunker["implementation_source_sha256"],
        "chunker policy document.implementation_source_sha256",
    )
    _exact_str(chunker["spacy_version"], "chunker policy document.spacy_version")

    ocr = _exact_object(
        documents["ocr"],
        {
            "schema_version",
            "disposition",
            "automatic_correction",
            "manual_correction",
        },
        "OCR policy document",
    )
    for key, expected in {
        "schema_version": "stylo.lobo-vnext.ocr-policy-doc.v1",
        "disposition": "preserve",
        "automatic_correction": False,
        "manual_correction": False,
    }.items():
        _literal(ocr[key], expected, f"OCR policy document.{key}")

def _validate_candidate_evidence(
    value: object,
    candidate_inventory: CandidateInventory,
) -> tuple[dict[str, object], ...]:
    rows = _exact_list(value, "R1 candidate evidence", length=3)
    expected_rows = tuple(
        (member, R1_EXCLUDED_WORK_ID, numerator, denominator)
        for member, (numerator, denominator) in zip(
            R1_COLLECTION_MEMBERS,
            ((2019, 2090), (5093, 5295), (3542, 3637)),
            strict=True,
        )
    )
    parsed: list[dict[str, object]] = []
    for index, (raw_value, expected) in enumerate(
        zip(rows, expected_rows, strict=True)
    ):
        left_work, right_work, numerator, denominator = expected
        raw = _exact_object(
            raw_value, _EVIDENCE_KEYS, f"R1 candidate evidence[{index}]"
        )
        evidence_sha = _sha256(
            raw["evidence_sha256"],
            f"R1 candidate evidence[{index}].evidence_sha256",
        )
        payload = {
            key: child for key, child in raw.items() if key != "evidence_sha256"
        }
        if canonical_sha256(payload) != evidence_sha:
            raise RealControlPlaneError(
                f"R1 candidate evidence[{index}] digest mismatch"
            )
        for key, expected in {
            "edge_type": "word5_asymmetric_containment",
            "left_work_id": left_work,
            "right_work_id": right_work,
            "policy_version": CONTENT_OVERLAP_POLICY_VERSION,
            "exact_containment_evidence": (
                f"{numerator}/{denominator} unique word-5-grams"
            ),
            "threshold_boundary": "inclusive",
            "final_verification": "exact_intersection_authoritative",
            "owner_selected_relation": "collection_member",
            "owner_selected_corpus_action": (
                "exclude_collection_retain_constituent"
            ),
        }.items():
            _literal(
                raw[key], expected, f"R1 candidate evidence[{index}].{key}"
            )
        containment = raw["reported_containment"]
        if (
            type(containment) is not float
            or not math.isfinite(containment)
            or containment != numerator / denominator
        ):
            raise RealControlPlaneError(
                f"R1 candidate evidence[{index}].reported_containment "
                "must equal the exact reviewed ratio"
            )
        threshold = _exact_object(
            raw["threshold"],
            {"numerator", "denominator"},
            f"R1 candidate evidence[{index}].threshold",
        )
        _exact_int(
            threshold["numerator"],
            f"R1 candidate evidence[{index}].threshold.numerator",
            9,
        )
        _exact_int(
            threshold["denominator"],
            f"R1 candidate evidence[{index}].threshold.denominator",
            10,
        )
        for path_key, work_id in (
            ("left_raw_paths", left_work),
            ("right_raw_paths", right_work),
        ):
            paths = _exact_list(
                raw[path_key],
                f"R1 candidate evidence[{index}].{path_key}",
                length=1,
            )
            _literal(
                paths[0],
                f"{work_id}.txt",
                f"R1 candidate evidence[{index}].{path_key}[0]",
            )
        parsed.append(raw)

    evidence_by_pair = {
        (row["left_work_id"], row["right_work_id"]): row[
            "evidence_sha256"
        ]
        for row in parsed
    }
    if len(candidate_inventory.candidates) != 6:
        raise RealControlPlaneError(
            "R1 candidate inventory must contain exactly six resolved records"
        )
    candidate_shape = tuple(
        (
            row.left_work_id,
            row.right_work_id,
            row.edge_type,
            row.origin,
            row.disposition,
        )
        for row in candidate_inventory.candidates
    )
    expected_shape = tuple(
        sorted(
            (
                member,
                R1_EXCLUDED_WORK_ID,
                edge_type,
                origin,
                "same_component",
            )
            for member in R1_COLLECTION_MEMBERS
            for edge_type, origin in (
                ("collection_member", "manual"),
                ("word5_asymmetric_containment", "automatic"),
            )
        )
    )
    if tuple(sorted(candidate_shape)) != expected_shape:
        raise RealControlPlaneError(
            "R1 candidate inventory differs from the reviewed three relations"
        )
    for candidate in candidate_inventory.candidates:
        expected_evidence = evidence_by_pair.get(
            (candidate.left_work_id, candidate.right_work_id)
        )
        if (
            candidate.evidence_sha256 != expected_evidence
            or candidate.manual_disposition is None
            or candidate.manual_disposition.evidence_sha256
            != candidate.evidence_sha256
        ):
            raise RealControlPlaneError(
                "R1 candidate record is not bound to exact reviewed evidence"
            )
    return tuple(parsed)


def _load_artifacts(
    root: pathlib.Path,
    packet_manifest: R1PacketManifest,
) -> dict[str, object]:
    parsers: dict[str, Callable[[object], object]] = {
        "content_policy": ContentPolicySpec.from_dict,
        "source_selection_receipt": R1SourceSelectionReceipt.from_dict,
        "source_candidate_inventory": CandidateInventory.from_dict,
        "candidate_inventory": CandidateInventory.from_dict,
        "corpus_manifest": CorpusVNextManifest.from_dict,
        "content_manifest": ContentComponentManifest.from_dict,
        "fold_manifest": FoldManifest.from_dict,
        "primary_model_spec": ModelSpec.from_dict,
        "baseline_model_spec": ModelSpec.from_dict,
        "inference_spec": InferenceSpec.from_dict,
        "primary_inner_cv_plan": InnerCVPlan.from_dict,
        "baseline_inner_cv_plan": InnerCVPlan.from_dict,
        "model_role_manifest": ModelRoleManifest.from_dict,
        "campaign_manifest": CampaignManifest.from_dict,
        "representation_receipt": CanonicalRepresentationReceipt.from_dict,
    }
    loaded: dict[str, object] = {}
    expected_hashes = {
        "source_candidate_inventory": (
            packet_manifest.source_candidate_inventory_sha256
        ),
        "candidate_inventory": packet_manifest.candidate_inventory_sha256,
        "corpus_manifest": packet_manifest.corpus_manifest_sha256,
        "content_manifest": (
            packet_manifest.content_component_manifest_sha256
        ),
        "fold_manifest": packet_manifest.fold_manifest_sha256,
        "primary_model_spec": packet_manifest.primary_model_spec_sha256,
        "baseline_model_spec": packet_manifest.baseline_model_spec_sha256,
        "inference_spec": packet_manifest.inference_spec_sha256,
        "primary_inner_cv_plan": (
            packet_manifest.primary_inner_cv_plan_sha256
        ),
        "baseline_inner_cv_plan": (
            packet_manifest.baseline_inner_cv_plan_sha256
        ),
        "model_role_manifest": packet_manifest.model_role_manifest_sha256,
        "campaign_manifest": packet_manifest.campaign_manifest_sha256,
        "representation_receipt": (
            packet_manifest.representation_receipt_sha256
        ),
    }
    for name, relative in _ARTIFACT_FILENAMES.items():
        value = _strict_file(root, relative, f"R1 artifact {name}")
        try:
            loaded[name] = parsers[name](value)
        except (VNextContractError, TypeError, ValueError) as exc:
            raise RealControlPlaneError(
                f"invalid R1 artifact {name}: {exc}"
            ) from exc
        observed_hash = getattr(loaded[name], "self_hash", None)
        if name == "content_policy":
            expected_hash = packet_manifest.generation_material.content_policy_spec_digest
        elif name == "source_selection_receipt":
            expected_hash = observed_hash
        else:
            expected_hash = expected_hashes[name]
        if observed_hash != expected_hash:
            raise RealControlPlaneError(
                f"R1 artifact {name} digest differs from packet manifest"
            )
    return loaded


def _validate_cross_bindings(
    *,
    root: pathlib.Path,
    packet_manifest: R1PacketManifest,
    artifacts: Mapping[str, object],
    documents: Mapping[str, object],
    evidence: object,
) -> None:
    policy = artifacts["content_policy"]
    source_selection = artifacts["source_selection_receipt"]
    source_candidates = artifacts["source_candidate_inventory"]
    candidates = artifacts["candidate_inventory"]
    corpus = artifacts["corpus_manifest"]
    content = artifacts["content_manifest"]
    folds = artifacts["fold_manifest"]
    primary = artifacts["primary_model_spec"]
    baseline = artifacts["baseline_model_spec"]
    inference = artifacts["inference_spec"]
    primary_inner = artifacts["primary_inner_cv_plan"]
    baseline_inner = artifacts["baseline_inner_cv_plan"]
    roles = artifacts["model_role_manifest"]
    campaign = artifacts["campaign_manifest"]
    representation = artifacts["representation_receipt"]

    if not (
        type(policy) is ContentPolicySpec
        and type(source_selection) is R1SourceSelectionReceipt
        and type(source_candidates) is CandidateInventory
        and type(candidates) is CandidateInventory
        and type(corpus) is CorpusVNextManifest
        and type(content) is ContentComponentManifest
        and type(folds) is FoldManifest
        and type(primary) is ModelSpec
        and type(baseline) is ModelSpec
        and type(inference) is InferenceSpec
        and type(primary_inner) is InnerCVPlan
        and type(baseline_inner) is InnerCVPlan
        and type(roles) is ModelRoleManifest
        and type(campaign) is CampaignManifest
        and type(representation) is CanonicalRepresentationReceipt
    ):
        raise RealControlPlaneError("R1 artifact parser returned an invalid type")

    _validate_policy_documents(documents)
    source_candidates.validate(content_policy_spec=policy)
    source_candidates.assert_resolved_for_component_manifest()
    candidates.validate(content_policy_spec=policy)
    candidates.assert_resolved_for_component_manifest()
    evidence_rows = _validate_candidate_evidence(evidence, source_candidates)

    material = packet_manifest.generation_material
    source_selection.validate_against(material)
    generation_id = packet_manifest.generation_id
    if (
        root.name != generation_id
        or corpus.generation_id != generation_id
        or source_selection.generation_id != generation_id
        or source_candidates.generation_id != generation_id
        or candidates.generation_id != generation_id
        or representation.generation_id != generation_id
    ):
        raise RealControlPlaneError(
            "R1 packet generation id differs across path/manifests"
    )
    if (
        source_candidates.raw_inventory_digest
        != material.source_raw_inventory_digest
        or source_selection.source_candidate_inventory_sha256
        != source_candidates.self_hash
        or source_selection.content_policy_spec_digest != policy.self_hash
        or source_selection.candidate_evidence_digest
        != material.candidate_evidence_digest
        or source_candidates.work_identity_catalog_digest
        != material.source_work_identity_catalog_digest
        or candidates.raw_inventory_digest
        != material.selected_raw_inventory_digest
        or candidates.work_identity_catalog_digest
        != material.selected_work_identity_catalog_digest
        or canonical_sha256(
            [row.to_dict() for row in source_candidates.candidates]
        )
        != material.source_candidate_draft_digest
        or canonical_sha256(list(evidence_rows))
        != material.candidate_evidence_digest
    ):
        raise RealControlPlaneError(
            "R1 source/selected generation material differs from preflight"
        )
    selected_ids = tuple(work.work_id for work in corpus.works)
    expected_source_ids = tuple(sorted((*selected_ids, R1_EXCLUDED_WORK_ID)))
    if (
        tuple(work.work_id for work in source_selection.source_works)
        != expected_source_ids
        or source_selection.selected_work_ids != selected_ids
        or source_selection.excluded_work_ids != (R1_EXCLUDED_WORK_ID,)
        or source_candidates.included_work_ids != expected_source_ids
        or candidates.included_work_ids != selected_ids
        or candidates.candidates
        or material.selected_work_ids != selected_ids
    ):
        raise RealControlPlaneError(
            "R1 source/selected candidate inventories differ from selection"
        )
    if (
        len(corpus.works) != R1_SELECTED_BOOK_COUNT
        or len(corpus.raw_inventory) != R1_SELECTED_BOOK_COUNT
        or len(corpus.author_ids) != R1_AUTHOR_COUNT
        or R1_EXCLUDED_WORK_ID in selected_ids
        or any(member not in selected_ids for member in R1_COLLECTION_MEMBERS)
    ):
        raise RealControlPlaneError("R1 exact owner-selected corpus drifted")
    if (
        canonical_sha256(
            [row.to_dict() for row in corpus.raw_inventory]
        )
        != material.selected_raw_inventory_digest
        or canonical_sha256([row.to_dict() for row in corpus.works])
        != material.selected_work_identity_catalog_digest
    ):
        raise RealControlPlaneError(
            "R1 selected corpus differs from exact generation material"
        )
    author_support: dict[str, int] = {}
    for work in corpus.works:
        author_support[work.author_id] = author_support.get(work.author_id, 0) + 1
    if min(author_support.values()) < 2:
        raise RealControlPlaneError("R1 corpus loses author-level LOBO support")
    corpus.assert_exploratory_authorized(synthetic_fixture=False)
    corpus.validate(content_manifest=content)
    if (
        corpus.content_policy_version != policy.self_hash
        or material.content_policy_spec_digest != policy.self_hash
        or source_candidates.content_policy_spec_digest != policy.self_hash
        or candidates.content_policy_spec_digest != policy.self_hash
        or content.automatic_candidate_policy_version != policy.self_hash
        or corpus.chunker_policy_version != policy.chunker_policy.policy_version
        or corpus.canonicalizer_policy_version
        != policy.canonical_row_policy.policy_version
    ):
        raise RealControlPlaneError(
            "R1 corpus/candidate/content policy bindings differ"
        )
    if content.candidates or any(
        len(component.work_ids) != 1 for component in content.components
    ):
        raise RealControlPlaneError(
            "R1 isolated mode requires singleton included content components"
        )
    folds.validate_against(corpus, content)
    if (
        folds.mode != "isolated"
        or len(folds.folds) != R1_SELECTED_BOOK_COUNT
        or tuple(fold.test_work_id for fold in folds.folds) != selected_ids
    ):
        raise RealControlPlaneError("R1 isolated fold manifest drifted")

    primary.assert_exploratory_authorized(synthetic_fixture=False)
    baseline.assert_exploratory_authorized(synthetic_fixture=False)
    primary_inner.validate_against(folds, corpus, content, primary)
    baseline_inner.validate_against(folds, corpus, content, baseline)
    roles.assert_model_specs(
        primary_model_spec=primary,
        baseline_model_spec=baseline,
        primary_inner_cv_plan=primary_inner,
        baseline_inner_cv_plan=baseline_inner,
    )
    campaign.assert_model_roles(roles)
    if (
        campaign.fold_manifest_digest != folds.self_hash
        or campaign.inference_spec_digest != inference.self_hash
    ):
        raise RealControlPlaneError("R1 campaign fold/inference binding drifted")
    expected_inference = (
        "book_accuracy",
        "author_clustered_percentile_bootstrap",
        ("macro_f1", "top2", "per_author"),
        "point_only",
        R1_BOOTSTRAP_SEED,
        R1_BOOTSTRAP_ITERATIONS,
        R1_CONFIDENCE_LEVEL,
        True,
        True,
    )
    observed_inference = (
        inference.primary_metric,
        inference.primary_uncertainty,
        inference.secondary_metrics,
        inference.macro_f1_uncertainty,
        inference.bootstrap_seed,
        inference.bootstrap_iterations,
        inference.confidence_level,
        inference.approved_for_exploratory,
        inference.owner_selected,
    )
    if observed_inference != expected_inference:
        raise RealControlPlaneError("R1 exact inference spec drifted")

    representation.validate(corpus_manifest=corpus)
    canonicalizer_digest = canonical_sha256(documents["canonicalizer"])
    chunker_digest = canonical_sha256(documents["chunker"])
    if (
        representation.canonicalizer_policy_document_sha256
        != canonicalizer_digest
        or representation.chunker_policy_document_sha256 != chunker_digest
    ):
        raise RealControlPlaneError(
            "R1 representation policy-document bindings drifted"
        )

    observed_raw = inventory_raw_files(root / "raw")
    if observed_raw != corpus.raw_inventory:
        raise RealControlPlaneError("R1 literal raw file inventory drifted")
    load_canonical_representation_rows(root, representation, corpus)


def load_prepared_r1_packet(
    packet_root: str | os.PathLike[str],
) -> PreparedR1Packet:
    """Strictly reload and revalidate an immutable owner-selected R1 packet."""

    root, observed_files, observed_directories = _scan_packet_tree(packet_root)
    packet_manifest = _parse_packet_manifest(
        _strict_file(root, "packet.json", "R1 packet manifest")
    )
    _validate_manifest_file_inventory(
        root, packet_manifest, observed_files
    )
    expected_directories = _expected_directories(observed_files)
    if observed_directories != expected_directories:
        missing = sorted(set(expected_directories) - set(observed_directories))
        extra = sorted(set(observed_directories) - set(expected_directories))
        raise RealControlPlaneError(
            "R1 packet directory inventory mismatch; "
            f"missing={missing}, extra={extra}"
        )
    artifacts = _load_artifacts(root, packet_manifest)
    documents = {
        name: _strict_file(root, relative, f"{name} policy document")
        for name, relative in _POLICY_FILENAMES.items()
    }
    evidence = _strict_file(
        root, "candidates/evidence.json", "R1 candidate evidence"
    )

    corpus = artifacts["corpus_manifest"]
    representation = artifacts["representation_receipt"]
    if (
        type(corpus) is not CorpusVNextManifest
        or type(representation) is not CanonicalRepresentationReceipt
    ):
        raise RealControlPlaneError("R1 corpus/representation types are invalid")
    expected_nonpacket_files = {
        "candidates/evidence.json",
        *_ARTIFACT_FILENAMES.values(),
        *_POLICY_FILENAMES.values(),
        *(f"raw/{row.relative_path}" for row in corpus.raw_inventory),
        *(row.relative_path for row in representation.rows),
    }
    manifest_files = {row.relative_path for row in packet_manifest.files}
    if manifest_files != expected_nonpacket_files:
        missing = sorted(expected_nonpacket_files - manifest_files)
        extra = sorted(manifest_files - expected_nonpacket_files)
        raise RealControlPlaneError(
            "R1 packet semantic file inventory mismatch; "
            f"missing={missing}, extra={extra}"
        )
    try:
        _validate_cross_bindings(
            root=root,
            packet_manifest=packet_manifest,
            artifacts=artifacts,
            documents=documents,
            evidence=evidence,
        )
    except RealControlPlaneError:
        raise
    except (VNextContractError, OSError, UnicodeError, ValueError) as exc:
        raise RealControlPlaneError(
            f"R1 packet cross-validation failed: {exc}"
        ) from exc

    return PreparedR1Packet(
        root=root,
        packet_manifest=packet_manifest,
        content_policy=artifacts["content_policy"],
        source_selection_receipt=artifacts[
            "source_selection_receipt"
        ],
        source_candidate_inventory=artifacts[
            "source_candidate_inventory"
        ],
        candidate_inventory=artifacts["candidate_inventory"],
        corpus_manifest=artifacts["corpus_manifest"],
        content_manifest=artifacts["content_manifest"],
        fold_manifest=artifacts["fold_manifest"],
        primary_model_spec=artifacts["primary_model_spec"],
        baseline_model_spec=artifacts["baseline_model_spec"],
        inference_spec=artifacts["inference_spec"],
        primary_inner_cv_plan=artifacts["primary_inner_cv_plan"],
        baseline_inner_cv_plan=artifacts["baseline_inner_cv_plan"],
        model_role_manifest=artifacts["model_role_manifest"],
        campaign_manifest=artifacts["campaign_manifest"],
        representation_receipt=artifacts["representation_receipt"],
    )


def _packet_object_observations(
    packet: PreparedR1Packet,
) -> dict[str, DerivedObservation]:
    raw_rows = [row.to_dict() for row in packet.corpus_manifest.raw_inventory]
    canonical_rows = [
        row.to_dict() for row in packet.representation_receipt.rows
    ]
    inner_subject = {
        "primary": packet.primary_inner_cv_plan.self_hash,
        "baseline": packet.baseline_inner_cv_plan.self_hash,
    }
    inner_digest = inner_cv_receipt_subject_digest(
        primary_inner_cv_plan_digest=packet.primary_inner_cv_plan.self_hash,
        baseline_inner_cv_plan_digest=packet.baseline_inner_cv_plan.self_hash,
    )
    rows = {
        "packet_selection": observation_from_self_hashed(
            kind="packet_selection",
            value=packet.packet_manifest,
        ),
        "raw_inventory": observation_from_canonical_value(
            kind="raw_inventory",
            value=raw_rows,
            observation_count=len(raw_rows),
        ),
        "canonical_model_rows": observation_from_canonical_value(
            kind="canonical_model_rows",
            value=canonical_rows,
            observation_count=len(canonical_rows),
        ),
        "content_candidates": observation_from_self_hashed(
            kind="content_candidates", value=packet.candidate_inventory
        ),
        "content_components": observation_from_self_hashed(
            kind="content_components", value=packet.content_manifest
        ),
        "folds": observation_from_self_hashed(
            kind="folds", value=packet.fold_manifest
        ),
        "inner_cv": observation_from_canonical_value(
            kind="inner_cv",
            value=inner_subject,
            observation_count=2,
        ),
        "representation": observation_from_self_hashed(
            kind="representation", value=packet.representation_receipt
        ),
    }
    if (
        rows["packet_selection"].digest != packet.packet_manifest.self_hash
        or rows["raw_inventory"].digest
        != canonical_sha256(raw_rows)
        or rows["canonical_model_rows"].digest
        != packet.corpus_manifest.canonical_model_row_digest
        or rows["inner_cv"].digest != inner_digest
    ):
        raise RealControlPlaneError(
            "R1 packet object observation identity mismatch"
        )
    return rows


def assemble_real_execution_spec(
    *,
    packet: PreparedR1Packet,
    cfg: ConfigNode,
    repository_root: str | os.PathLike[str],
) -> tuple[RealCorpusExecutionSpec, tuple[DerivedObservation, ...]]:
    """Assemble live ExecutionSpec v2 without an owner decision or learned work."""

    if type(packet) is not PreparedR1Packet:
        raise RealControlPlaneError("packet must be exactly PreparedR1Packet")
    if type(cfg) is not ConfigNode:
        raise RealControlPlaneError("cfg must be exactly ConfigNode")

    # Reload from bytes so callers cannot bypass immutable packet validation by
    # constructing or mutating an in-memory dataclass.
    packet = load_prepared_r1_packet(packet.root)

    # Dirty/drifted executable code blocks before live NLP/model-policy
    # resolution, dependency inspection, runtime inspection, or threading.
    executable = derive_executable_source_observation(repository_root)
    current_policy, _current_documents = build_r1_content_policy(cfg)
    if current_policy != packet.content_policy:
        raise RealControlPlaneError(
            "live R1 content policy/config differs from prepared packet"
        )
    if (
        build_r1_model_spec(role="primary", cfg=cfg)
        != packet.primary_model_spec
        or build_r1_model_spec(role="baseline", cfg=cfg)
        != packet.baseline_model_spec
    ):
        raise RealControlPlaneError(
            "live R1 model specification differs from prepared packet"
        )

    config, primary_adapter, baseline_adapter = (
        derive_config_and_adapter_observations(
            cfg=cfg,
            primary_model_spec=packet.primary_model_spec,
            baseline_model_spec=packet.baseline_model_spec,
        )
    )
    dependencies = derive_dependency_observation(repository_root)
    runtime = derive_runtime_observation()
    threads = derive_thread_observation()

    by_kind = _packet_object_observations(packet)
    by_kind.update(
        {
            executable.kind: executable,
            config.kind: config,
            primary_adapter.kind: primary_adapter,
            baseline_adapter.kind: baseline_adapter,
            dependencies.kind: dependencies,
            runtime.kind: runtime,
            threads.kind: threads,
        }
    )
    try:
        observations = tuple(by_kind[kind] for kind in REQUIRED_RECEIPT_KINDS)
    except KeyError as exc:
        raise RealControlPlaneError(
            f"missing live identity observation {exc.args[0]!r}"
        ) from exc
    if len(by_kind) != len(REQUIRED_RECEIPT_KINDS):
        raise RealControlPlaneError(
            "live identity observations contain an unexpected kind"
        )

    bindings = RealExecutionBindings(
        packet_manifest_digest=packet.packet_manifest.self_hash,
        content_policy_spec_digest=packet.content_policy.self_hash,
        candidate_inventory_digest=packet.candidate_inventory.self_hash,
        corpus_manifest_digest=packet.corpus_manifest.self_hash,
        content_component_manifest_digest=packet.content_manifest.self_hash,
        fold_manifest_digest=packet.fold_manifest.self_hash,
        primary_inner_cv_plan_digest=packet.primary_inner_cv_plan.self_hash,
        baseline_inner_cv_plan_digest=packet.baseline_inner_cv_plan.self_hash,
        primary_model_spec_digest=packet.primary_model_spec.self_hash,
        baseline_model_spec_digest=packet.baseline_model_spec.self_hash,
        model_role_manifest_digest=packet.model_role_manifest.self_hash,
        inference_spec_digest=packet.inference_spec.self_hash,
        campaign_manifest_digest=packet.campaign_manifest.self_hash,
        config_digest=config.digest,
    )
    execution = RealCorpusExecutionSpec.build(
        bindings=bindings,
        independent_receipts=build_independent_receipts(observations),
        output_namespace=OutputNamespaceContract.build(
            namespace_id=packet.campaign_manifest.campaign_id
        ),
    )
    execution.assert_campaign(
        campaign_manifest=packet.campaign_manifest,
        model_role_manifest=packet.model_role_manifest,
        primary_model_spec=packet.primary_model_spec,
        baseline_model_spec=packet.baseline_model_spec,
        primary_inner_cv_plan=packet.primary_inner_cv_plan,
        baseline_inner_cv_plan=packet.baseline_inner_cv_plan,
    )
    assert_independent_receipts(execution, observations)
    return execution, observations


__all__ = [
    "RealControlPlaneError",
    "assemble_real_execution_spec",
    "load_prepared_r1_packet",
]
