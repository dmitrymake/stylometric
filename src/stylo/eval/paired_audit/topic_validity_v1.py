"""Research-only topic-strict challenger binding and aggregate evidence.

This module deliberately has no CLI, package export, registry entry, official receipt, or real-data
execution authority.  It binds fresh A0/A4 factories and reduces transient paired probabilities to a
strict aggregate that contains no work identities, probabilities, or row-level predictions.
"""
from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from typing import Mapping

from ...domain.prediction_contract import (
    PredictionContractError,
    stable_top1_and_worst_tie_rank,
    validate_author_universe,
    validate_probability_vector,
)
from ...jsonio import artifact_self_hash, canonical_hash, loads_strict
from ...vectorizer import StyloVectorizer
from ..lobo import make_factory_for_ablation
from .applicability_v3_2 import resolve_cell_v3_2
from .evaluator_v3_2 import CONTEXT_SCHEMA, V32EvaluationContext, _validate_context

TOPIC_CHALLENGER_SCHEMA_V1 = "stylo.paired-audit.topic-strict-challenger.v1"
TOPIC_AGGREGATE_SCHEMA_V1 = "stylo.topic_validity.aggregate.v1"
TOPIC_CELLS_V1 = ("A0", "A4")
TOPIC_ARMS_V1 = ("current", "topic_strict")
_STUDY_SEAL = object()
_SOURCE_FILES = (
    ("adapter", "src/stylo/eval/paired_audit/topic_validity_v1.py"),
    ("official_evaluator", "src/stylo/eval/paired_audit/evaluator_v3_2.py"),
    ("authoritative_factory", "src/stylo/eval/lobo.py"),
    ("stylo_vectorizer", "src/stylo/vectorizer.py"),
    ("feature_registry", "src/stylo/features/registry.py"),
    ("function_word_block", "src/stylo/features/function_words.py"),
)


class TopicValidityV1Error(ValueError):
    """The research-only challenger/aggregate contract failed closed."""


@dataclass(frozen=True)
class _FoldExpectation:
    fold_index: int
    fold_identity: str
    author: str
    true_label: int


@dataclass(frozen=True)
class TopicStudyContextV1:
    cfg: object
    parent: V32EvaluationContext
    binding: dict
    folds: tuple[_FoldExpectation, ...]
    probability_order: tuple[str, ...]
    metric_order: tuple[str, ...]
    author_fold_counts: tuple[tuple[str, int], ...]
    _seal: object


def _same(value, expected) -> bool:
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        return value.keys() == expected.keys() and all(_same(value[key], expected[key]) for key in value)
    if type(value) is list:
        return len(value) == len(expected) and all(map(_same, value, expected))
    if type(value) is tuple:
        return len(value) == len(expected) and all(map(_same, value, expected))
    return value == expected


def _sha256_file(path: pathlib.Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TopicValidityV1Error(f"challenger source is missing or symlinked: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hex64(value, where: str) -> str:
    if type(value) is not str or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise TopicValidityV1Error(f"{where} must be a lowercase sha256 digest")
    return value


def _source_identities() -> dict[str, str]:
    root = pathlib.Path(__file__).resolve().parents[4]
    return {role: _sha256_file(root / relative) for role, relative in _SOURCE_FILES}


def _derive_study(cfg, context: V32EvaluationContext) -> dict:
    try:
        _validate_context(cfg, context)
    except Exception as exc:
        raise TopicValidityV1Error(f"invalid verified v3.2 context: {exc}") from exc
    manifest = context.lobo_manifest
    tested = [row for row in manifest["works"] if row["tested"]]
    if [row.get("fold_index") for row in tested] != list(range(len(tested))):
        raise TopicValidityV1Error("LOBO tested folds must be contiguous and ordered")
    try:
        probability_order = validate_author_universe(list(manifest["probability_class_order"]))
        metric_order = validate_author_universe(list(manifest["metric_label_order"]))
    except PredictionContractError as exc:
        raise TopicValidityV1Error(str(exc)) from exc
    if not set(metric_order).issubset(probability_order):
        raise TopicValidityV1Error("metric author order is outside the probability order")
    folds = []
    counts = {author: 0 for author in metric_order}
    for row in tested:
        author = row.get("author_id")
        if author not in counts or row.get("work_id", "").split("/", 1)[0] != author:
            raise TopicValidityV1Error("fold author/work identity is outside the metric universe")
        fold_index = row["fold_index"]
        fold_identity = canonical_hash([
            "stylo.topic_validity.fold.v1", fold_index, row["work_id"], author,
            row["work_content_identity"], row["content_component_identity"],
        ])
        folds.append(_FoldExpectation(fold_index, fold_identity, author, probability_order.index(author)))
        counts[author] += 1
    source_identities = _source_identities()
    identities = {
        "context_identity": context.context_identity,
        "candidate_identity": context.candidate_identity,
        "corrected_corpus_identity": context.corrected_corpus_identity,
        "corpus_manifest_identity": context.corpus_manifest_identity,
        "lobo_fold_manifest_identity": manifest["self_hash"],
        "applicability_identity": context.applicability_identity,
        "config_identity": context.config_identity,
        "protocol_identity": context.protocol_identity,
        "content_isolation_identity": context.content_isolation_identity,
        "work_identity_catalog_identity": context.work_identity_catalog_identity,
        "lobo_dataset_rows_identity": context.lobo_dataset_identity["rows_digest"],
        "lobo_row_identity": context.lobo_dataset_identity["row_identity_digest"],
        "probability_class_order_identity": canonical_hash(list(probability_order)),
        "metric_label_order_identity": canonical_hash(list(metric_order)),
        "tested_fold_order_identity": canonical_hash([fold.fold_identity for fold in folds]),
    }
    for name, value in identities.items():
        _hex64(value, f"identities.{name}")
    binding = {
        "schema": TOPIC_CHALLENGER_SCHEMA_V1,
        "status": "research_only_unexecuted",
        "confirmatory_authorized": False,
        "official_receipt_authorized": False,
        "factory_route": "stylo.eval.lobo.make_factory_for_ablation",
        "vectorizer_route": "stylo.vectorizer.StyloVectorizer.from_config",
        "model": "stylo",
        "cells": list(TOPIC_CELLS_V1),
        "arms": list(TOPIC_ARMS_V1),
        "challenger_delta": {
            "topic_strict": True,
            "function_words": "fixed_list",
            "disabled_syntax_subblocks": ["pos_ratios", "lexical_richness"],
            "relative_fw_policy": "legacy_corner_coupling",
        },
        "parent_context_schema": CONTEXT_SCHEMA,
        "identities": identities,
        "source_identities": source_identities,
        "fold_count": len(folds),
        "tested_author_count": len(metric_order),
        "probability_class_count": len(probability_order),
    }
    binding["self_hash"] = artifact_self_hash(binding)
    return {
        "binding": binding,
        "folds": tuple(folds),
        "probability_order": probability_order,
        "metric_order": metric_order,
        "author_fold_counts": tuple((author, counts[author]) for author in metric_order),
    }


def build_topic_study_context_v1(*, cfg, context: V32EvaluationContext) -> TopicStudyContextV1:
    derived = _derive_study(cfg, context)
    return TopicStudyContextV1(cfg=cfg, parent=context, _seal=_STUDY_SEAL, **derived)


def _validate_study(study: TopicStudyContextV1) -> None:
    if type(study) is not TopicStudyContextV1 or study._seal is not _STUDY_SEAL:
        raise TopicValidityV1Error("study context was not built by the v1 adapter")
    derived = _derive_study(study.cfg, study.parent)
    for key in ("binding", "folds", "probability_order", "metric_order", "author_fold_counts"):
        if not _same(getattr(study, key), derived[key]):
            raise TopicValidityV1Error(f"study context {key} drifted from verified inputs")


def _assert_selector(value, allowed: tuple[str, ...], where: str) -> str:
    if type(value) is not str or value not in allowed:
        raise TopicValidityV1Error(f"{where} must be exactly one of {list(allowed)}")
    return value


def _vectorizer_contract(estimator, *, strict: bool) -> None:
    if tuple(estimator.named_steps) != ("vectorizer", "scaler", "classifier"):
        raise TopicValidityV1Error("challenger pipeline step contract drifted")
    vectorizer = estimator.named_steps["vectorizer"]
    fw = next((block for block in vectorizer.blocks if block.name == "function_words"), None)
    syntax = next((block for block in vectorizer.blocks if block.name == "syntax"), None)
    wanted = "fixed_list" if strict else "mfw"
    if fw is None or fw.mode != wanted or fw.relative_fw is not None:
        raise TopicValidityV1Error("function-word/relative-FW challenger route drifted")
    if strict and (syntax is None or {"pos_ratios", "lexical_richness"} & set(syntax._active)):
        raise TopicValidityV1Error("topic-strict syntax subblocks were not disabled")


def make_topic_challenger_factory_v1(*, study: TopicStudyContextV1, cell: str, arm: str):
    _validate_study(study)
    cell = _assert_selector(cell, TOPIC_CELLS_V1, "cell")
    arm = _assert_selector(arm, TOPIC_ARMS_V1, "arm")
    row = resolve_cell_v3_2("stylo", cell, require_applied=True)
    authoritative = make_factory_for_ablation("stylo", study.cfg, ablation=row.ablation)

    def factory():
        _validate_study(study)
        estimator = authoritative()
        if arm == "topic_strict":
            scaler = estimator.named_steps.get("scaler")
            classifier = estimator.named_steps.get("classifier")
            estimator.set_params(vectorizer=StyloVectorizer.from_config(
                study.cfg, topic_strict=True, relative_fw=None
            ))
            if estimator.named_steps.get("scaler") is not scaler or estimator.named_steps.get("classifier") is not classifier:
                raise TopicValidityV1Error("topic-strict replacement changed a non-vectorizer step")
        _vectorizer_contract(estimator, strict=arm == "topic_strict")
        return estimator

    return factory


def _validated_predictions(study: TopicStudyContextV1, records: Mapping) -> dict:
    _validate_study(study)
    if type(records) is not dict or tuple(records) != TOPIC_CELLS_V1:
        raise TopicValidityV1Error("transient records require ordered A0/A4 cells")
    predictions = {}
    width = len(study.probability_order)
    for cell in TOPIC_CELLS_V1:
        arms = records[cell]
        if type(arms) is not dict or tuple(arms) != TOPIC_ARMS_V1:
            raise TopicValidityV1Error(f"{cell} requires ordered current/topic_strict arms")
        for arm in TOPIC_ARMS_V1:
            rows = arms[arm]
            if type(rows) is not list or len(rows) != len(study.folds):
                raise TopicValidityV1Error(f"{cell}/{arm} fold count drifted")
            values = []
            for expected, record in zip(study.folds, rows, strict=True):
                if type(record) is not dict or set(record) != {
                    "fold_index", "fold_identity", "whole_work_probabilities"
                }:
                    raise TopicValidityV1Error("transient fold record has wrong fields")
                if type(record["fold_index"]) is not int or record["fold_index"] != expected.fold_index:
                    raise TopicValidityV1Error("transient fold order/index drifted")
                if type(record["fold_identity"]) is not str or record["fold_identity"] != expected.fold_identity:
                    raise TopicValidityV1Error("transient fold identity drifted")
                probability = record["whole_work_probabilities"]
                if type(probability) is not list or any(type(value) is not float for value in probability):
                    raise TopicValidityV1Error("transient probabilities must be exact-float lists")
                try:
                    vector = validate_probability_vector(probability, expected_width=width)
                    decision = stable_top1_and_worst_tie_rank(
                        vector, true_label=expected.true_label, expected_width=width
                    )
                except PredictionContractError as exc:
                    raise TopicValidityV1Error(f"invalid transient probabilities: {exc}") from exc
                values.append(decision.top1)
            predictions[(cell, arm)] = values
    return predictions


def _digest_predictions(study, cell: str, arm: str, predictions: list[int]) -> dict:
    return {
        "count": len(predictions),
        "sha256": canonical_hash([
            "stylo.topic_validity.predictions.v1",
            study.binding["identities"]["tested_fold_order_identity"],
            cell,
            arm,
            predictions,
        ]),
    }


def _cell_aggregate(study, cell: str, predictions: dict) -> dict:
    current = predictions[(cell, "current")]
    strict = predictions[(cell, "topic_strict")]
    per_author = {author: {
        "author": author, "n_folds": 0, "both_correct": 0, "current_only_correct": 0,
        "topic_strict_only_correct": 0, "both_wrong_same_prediction": 0,
        "both_wrong_changed_prediction": 0,
    } for author in study.metric_order}
    current_correct = strict_correct = 0
    for expected, current_pred, strict_pred in zip(study.folds, current, strict, strict=True):
        row = per_author[expected.author]
        row["n_folds"] += 1
        current_ok = current_pred == expected.true_label
        strict_ok = strict_pred == expected.true_label
        current_correct += int(current_ok)
        strict_correct += int(strict_ok)
        if current_ok and strict_ok:
            category = "both_correct"
        elif current_ok:
            category = "current_only_correct"
        elif strict_ok:
            category = "topic_strict_only_correct"
        elif current_pred == strict_pred:
            category = "both_wrong_same_prediction"
        else:
            category = "both_wrong_changed_prediction"
        row[category] += 1
    if tuple((author, per_author[author]["n_folds"]) for author in study.metric_order) != study.author_fold_counts:
        raise TopicValidityV1Error("per-author fold counts drifted from the sealed study context")
    total = len(study.folds)
    return {
        "cell": cell,
        "accuracy": {
            "current": {"correct": current_correct, "total": total},
            "topic_strict": {"correct": strict_correct, "total": total},
        },
        "delta_accuracy": {
            "direction": "topic_strict_minus_current",
            "numerator": strict_correct - current_correct,
            "denominator": total,
        },
        "prediction_vector_digests": {
            arm: _digest_predictions(study, cell, arm, predictions[(cell, arm)])
            for arm in TOPIC_ARMS_V1
        },
        "per_author_transitions": [per_author[author] for author in study.metric_order],
    }


def _assemble_aggregate(study, records, implementation_source_identity, environment_lock_identity) -> dict:
    _hex64(implementation_source_identity, "implementation_source_identity")
    _hex64(environment_lock_identity, "environment_lock_identity")
    predictions = _validated_predictions(study, records)
    bindings = {
        **dict(study.binding["identities"]),
        "study_binding_identity": study.binding["self_hash"],
        "implementation_source_identity": implementation_source_identity,
        "environment_lock_identity": environment_lock_identity,
    }
    design = {
        "dataset": "lobo",
        "model": "stylo",
        "cells": list(TOPIC_CELLS_V1),
        "arms": list(TOPIC_ARMS_V1),
        "unit": "held_out_whole_work",
        "fold_count": len(study.folds),
        "tested_author_count": len(study.metric_order),
        "probability_class_count": len(study.probability_order),
        "prediction_rule": "stable_top1_lowest_index",
        "accuracy_weighting": "equal_held_out_work",
        "delta_direction": "topic_strict_minus_current",
        "transition_categories": [
            "both_correct", "current_only_correct", "topic_strict_only_correct",
            "both_wrong_same_prediction", "both_wrong_changed_prediction",
        ],
    }
    artifact = {
        "schema": TOPIC_AGGREGATE_SCHEMA_V1,
        "status": "bounded_research_aggregate_only",
        "confirmatory_authorized": False,
        "publication_authorized": False,
        "study_identity": canonical_hash(["stylo.topic_validity.study.v1", bindings, design]),
        "bindings": bindings,
        "design": design,
        "cells": [_cell_aggregate(study, cell, predictions) for cell in TOPIC_CELLS_V1],
    }
    artifact["self_hash"] = artifact_self_hash(artifact)
    return artifact


def build_topic_aggregate_v1(*, study: TopicStudyContextV1, records: Mapping,
                             implementation_source_identity: str,
                             environment_lock_identity: str) -> dict:
    return _assemble_aggregate(
        study, records, implementation_source_identity, environment_lock_identity
    )


def validate_topic_aggregate_v1(artifact: Mapping, *, study: TopicStudyContextV1, records: Mapping,
                                implementation_source_identity: str,
                                environment_lock_identity: str) -> None:
    expected = _assemble_aggregate(
        study, records, implementation_source_identity, environment_lock_identity
    )
    if type(artifact) is not dict or not _same(artifact, expected):
        raise TopicValidityV1Error("aggregate differs from independently reconstructed evidence")


def validate_topic_aggregate_json_v1(text: str, **kwargs) -> dict:
    try:
        artifact = loads_strict(text)
    except Exception as exc:
        raise TopicValidityV1Error(f"aggregate is not strict JSON: {exc}") from exc
    validate_topic_aggregate_v1(artifact, **kwargs)
    return artifact
