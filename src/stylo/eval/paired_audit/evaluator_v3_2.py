"""Unregistered v3.2 evaluator candidate: verified context -> one whole-work vote."""
from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ...domain.corpus_identity import DatasetProvenance, canonical_digest
from ...domain.prediction_contract import (
    stable_top1_and_worst_tie_rank,
    validate_probabilities,
)
from ...eval.dispatch import fit_estimator
from ...eval.lobo import make_factory_for_ablation
from ...jsonio import canonical_hash, dumps_strict
from ...workdoc import load_work_balanced_dataset
from .applicability_v3_2 import APPLICABILITY_V3_2_DIGEST, resolve_cell_v3_2
from .corrected_v3_2 import (
    FROZEN_CORPUS_CHUNKER_CONFIG_SHA256,
    FROZEN_PROTOCOL_SHA256,
    LOBO_SINGLETON_AUTHORS,
    PROTOCOL_VERSION,
    verify_v3_2_candidate,
)
from .evidence_v3_2 import (
    build_receipt_v3_2,
    fitted_state_v3_2,
    numeric_evidence,
    observe_fit_v3_2,
    ordered_strings_evidence,
    training_axis_evidence,
)
from .work_subset import derive_work_subset

CANDIDATE_IDENTITY = "ff620b05f20b81c21732014b553aa739a393c74fe344e6d9f2bd8d80996cef21"
CORRECTED_CORPUS_IDENTITY = "1a9a0779e4e578f38664fd974c7ac4565f12fb4992cf29773a34061fddee8531"
CORPUS_MANIFEST_IDENTITY = "a2dc0c4a6d3313354295a482466693d513ee8f77b297dfe4feb852104b2af3f7"
LOBO_FOLD_IDENTITY = "117b8ec9f51ef8c6359768a232660b156a16e0d124b8450e9c497b39cf4cc658"
RUAA_FOLD_IDENTITY = "d8428290c1895ea397367fbea0cab72317e5e3116176244c488d1f7c6f2b682b"
CONTEXT_SCHEMA = "paired_audit.evaluation_context.v3_2.candidate"
_CONTEXT_SEAL = object()


class V32EvaluationError(ValueError):
    """A context, fold, identity, route, prediction, or evidence contract failed closed."""


def _config_identity(cfg) -> str:
    if not hasattr(cfg, "to_dict"):
        raise V32EvaluationError("cfg must expose the resolved ConfigNode contract")
    return hashlib.sha256(dumps_strict(cfg.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


def _row_id_payload(provenance: DatasetProvenance) -> list[list]:
    return [
        [row.group, int(row.ordinal), row.text_sha256, row.work_id,
         row.provenance_sha256, row.chunker_config_hash]
        for row in provenance.row_ids
    ]


def _dataset_identity(dataset) -> dict:
    provenance = getattr(dataset, "provenance", None)
    if type(provenance) is not DatasetProvenance:
        raise V32EvaluationError("dataset lacks exact DatasetProvenance")
    recomputed = canonical_digest(
        list(dataset.texts), list(dataset.y), list(dataset.groups), list(dataset.authors),
        provenance.row_ids, loader_kind=provenance.loader_kind,
        chunker_config_hash=provenance.chunker_config_hash,
    )
    if recomputed != provenance.rows_digest:
        raise V32EvaluationError("dataset row identity/content digest changed")
    return {
        "rows_digest": provenance.rows_digest,
        "row_identity_digest": canonical_hash(_row_id_payload(provenance)),
        "n_rows": int(len(dataset.texts)),
        "authors": list(dataset.authors),
        "work_ids": sorted(set(map(str, dataset.groups))),
        "parent_rows_digest": provenance.parent_rows_digest,
        "row_selection_identity": provenance.selection_manifest_digest,
    }


def _work_rows(manifest: Mapping) -> dict[str, dict]:
    rows = manifest.get("works")
    if not isinstance(rows, list):
        raise V32EvaluationError("fold manifest has no work rows")
    return {row["work_id"]: dict(row) for row in rows}


def _manifest_identity(manifest: Mapping) -> str:
    value = manifest.get("self_hash")
    if type(value) is not str:
        raise V32EvaluationError("fold manifest lacks self_hash")
    return value


@dataclass(frozen=True)
class V32EvaluationContext:
    cfg: object
    bundle_root: pathlib.Path
    candidate_identity: str
    corrected_corpus_identity: str
    corpus_manifest_identity: str
    config_identity: str
    protocol_identity: str
    applicability_identity: str
    content_isolation_identity: str
    work_identity_catalog_identity: str
    lobo_manifest: Mapping
    ruaa_manifest: Mapping
    lobo_dataset: object
    ruaa_dataset: object
    lobo_dataset_identity: Mapping
    ruaa_dataset_identity: Mapping
    ruaa_work_selection_identity: str
    context_identity: str
    _seal: object

    def dataset(self, kind: str):
        if kind == "lobo":
            return self.lobo_dataset, self.lobo_manifest, self.lobo_dataset_identity
        if kind == "ruaa":
            return self.ruaa_dataset, self.ruaa_manifest, self.ruaa_dataset_identity
        raise V32EvaluationError(f"unknown v3.2 dataset: {kind!r}")


def _context_material(context: V32EvaluationContext) -> dict:
    return {
        "schema": CONTEXT_SCHEMA,
        "candidate_identity": context.candidate_identity,
        "corrected_corpus_identity": context.corrected_corpus_identity,
        "corpus_manifest_identity": context.corpus_manifest_identity,
        "config_identity": context.config_identity,
        "protocol_identity": context.protocol_identity,
        "applicability_identity": context.applicability_identity,
        "content_isolation_identity": context.content_isolation_identity,
        "work_identity_catalog_identity": context.work_identity_catalog_identity,
        "fold_identities": {
            "lobo": _manifest_identity(context.lobo_manifest),
            "ruaa": _manifest_identity(context.ruaa_manifest),
        },
        "datasets": {
            "lobo": dict(context.lobo_dataset_identity),
            "ruaa": dict(context.ruaa_dataset_identity),
        },
        "probability_class_orders": {
            "lobo": list(context.lobo_manifest["probability_class_order"]),
            "ruaa": list(context.ruaa_manifest["probability_class_order"]),
        },
        "metric_class_orders": {
            "lobo": list(context.lobo_manifest["metric_label_order"]),
            "ruaa": list(context.ruaa_manifest["metric_label_order"]),
        },
        "ruaa_work_selection_identity": context.ruaa_work_selection_identity,
        "ruaa_row_selection_identity": context.ruaa_dataset_identity["row_selection_identity"],
    }


def _assert_universe(kind: str, dataset, manifest: Mapping, identity: Mapping) -> None:
    authors = list(dataset.authors)
    works = sorted(set(map(str, dataset.groups)))
    tested = [row for row in manifest["works"] if row["tested"]]
    counts = (len(authors), len(works), len({row["author_id"] for row in tested}), len(tested))
    expected = (47, 252, 43, 248) if kind == "lobo" else (22, 134, 22, 134)
    if counts != expected:
        raise V32EvaluationError(f"{kind} dataset/fold counts {counts} != {expected}")
    if authors != list(manifest["probability_class_order"]):
        raise V32EvaluationError(f"{kind} probability class order differs from dataset labels")
    if len(manifest["probability_class_order"]) != (47 if kind == "lobo" else 22):
        raise V32EvaluationError(f"{kind} probability width drift")
    if len(manifest["metric_label_order"]) != (43 if kind == "lobo" else 22):
        raise V32EvaluationError(f"{kind} metric width drift")
    if works != sorted(row["work_id"] for row in manifest["works"]):
        raise V32EvaluationError(f"{kind} dataset work universe differs from fold")
    if identity["work_ids"] != works:
        raise V32EvaluationError(f"{kind} independently derived work identity drift")
    by_author = {author: 0 for author in authors}
    for work in works:
        by_author[work.split("/", 1)[0]] += 1
    if kind == "lobo":
        train_only = tuple(sorted(set(authors) - set(manifest["metric_label_order"])))
        if train_only != LOBO_SINGLETON_AUTHORS or any(by_author[a] != 1 for a in train_only):
            raise V32EvaluationError("LOBO four train-only singleton authors drift")
    elif any(by_author[row["author_id"]] < 2 for row in tested):
        raise V32EvaluationError("RuAA held-out author would disappear from train")


def build_evaluation_context_v3_2(
    *, cfg, bundle_root: pathlib.Path | str, historical_parent_root: pathlib.Path | str,
    ruaa_parent_selection: Sequence[str],
) -> V32EvaluationContext:
    """Call the exact verifier first, then load only its bundle-local corrected child."""
    config_identity = _config_identity(cfg)
    verified = verify_v3_2_candidate(
        bundle_root,
        historical_parent_root=historical_parent_root,
        ruaa_parent_selection=ruaa_parent_selection,
        config_hash=config_identity,
        protocol_sha256=FROZEN_PROTOCOL_SHA256,
    )
    candidate = verified["candidate"]
    corpus = verified["corpus_manifest"]
    lobo_manifest = verified["lobo_manifest"]
    ruaa_manifest = verified["ruaa_manifest"]
    if candidate.get("self_hash") != CANDIDATE_IDENTITY:
        raise V32EvaluationError("candidate/bundle identity is not the frozen v3.2 candidate")
    if corpus.get("corrected_content_inventory_digest") != CORRECTED_CORPUS_IDENTITY:
        raise V32EvaluationError("corrected corpus identity drift")
    if corpus.get("self_hash") != CORPUS_MANIFEST_IDENTITY:
        raise V32EvaluationError("corrected corpus manifest identity drift")
    if _manifest_identity(lobo_manifest) != LOBO_FOLD_IDENTITY:
        raise V32EvaluationError("LOBO fold identity drift")
    if _manifest_identity(ruaa_manifest) != RUAA_FOLD_IDENTITY:
        raise V32EvaluationError("RuAA fold identity drift")
    for manifest in (lobo_manifest, ruaa_manifest):
        if (manifest.get("protocol_version") != PROTOCOL_VERSION
                or manifest.get("config_hash") != config_identity
                or manifest.get("applicability_matrix_digest") != APPLICABILITY_V3_2_DIGEST):
            raise V32EvaluationError("fold config/protocol/applicability binding drift")
    corrected_root = pathlib.Path(verified["corrected_corpus_root"])
    expected_root = pathlib.Path(bundle_root) / "corrected_corpus"
    if corrected_root != expected_root:
        raise V32EvaluationError("verifier returned a non-bundle-local corrected root")
    lobo_dataset = load_work_balanced_dataset(
        corrected_root / "frags", cfg=cfg,
        input_clean_root=corrected_root / "input_clean",
        exclude_authors=(), unknown_name="unknown",
        expected_chunker_config_hash=FROZEN_CORPUS_CHUNKER_CONFIG_SHA256,
    )
    ruaa_work_ids = [row["work_id"] for row in ruaa_manifest["works"]]
    ruaa_dataset = derive_work_subset(lobo_dataset, ruaa_work_ids, expected_n_works=134)
    lobo_identity = _dataset_identity(lobo_dataset)
    ruaa_identity = _dataset_identity(ruaa_dataset)
    _assert_universe("lobo", lobo_dataset, lobo_manifest, lobo_identity)
    _assert_universe("ruaa", ruaa_dataset, ruaa_manifest, ruaa_identity)
    if ruaa_manifest["selection_digest"] == ruaa_identity["row_selection_identity"]:
        raise V32EvaluationError("RuAA work selection and row selection identities were collapsed")
    values = dict(
        cfg=cfg,
        bundle_root=pathlib.Path(bundle_root),
        candidate_identity=CANDIDATE_IDENTITY,
        corrected_corpus_identity=CORRECTED_CORPUS_IDENTITY,
        corpus_manifest_identity=CORPUS_MANIFEST_IDENTITY,
        config_identity=config_identity,
        protocol_identity=FROZEN_PROTOCOL_SHA256,
        applicability_identity=APPLICABILITY_V3_2_DIGEST,
        content_isolation_identity=corpus["content_isolation_audit_digest"],
        work_identity_catalog_identity=corpus["full_work_identity_catalog_digest"],
        lobo_manifest=lobo_manifest,
        ruaa_manifest=ruaa_manifest,
        lobo_dataset=lobo_dataset,
        ruaa_dataset=ruaa_dataset,
        lobo_dataset_identity=lobo_identity,
        ruaa_dataset_identity=ruaa_identity,
        ruaa_work_selection_identity=ruaa_manifest["selection_digest"],
        context_identity="",
        _seal=_CONTEXT_SEAL,
    )
    provisional = V32EvaluationContext(**values)
    values["context_identity"] = canonical_hash(_context_material(provisional))
    return V32EvaluationContext(**values)


def _validate_context(cfg, context: V32EvaluationContext) -> None:
    if type(context) is not V32EvaluationContext or context._seal is not _CONTEXT_SEAL:
        raise V32EvaluationError("context was not created by successful v3.2 verification")
    if cfg is not context.cfg or _config_identity(cfg) != context.config_identity:
        raise V32EvaluationError("explicit cfg differs from verified context cfg")
    if _dataset_identity(context.lobo_dataset) != dict(context.lobo_dataset_identity):
        raise V32EvaluationError("LOBO context dataset changed after verification")
    if _dataset_identity(context.ruaa_dataset) != dict(context.ruaa_dataset_identity):
        raise V32EvaluationError("RuAA context dataset changed after verification")
    if canonical_hash(_context_material(context)) != context.context_identity:
        raise V32EvaluationError("evaluation context identity changed")


def _class_alignment(classes, probability_order: Sequence[str], dataset_authors: Sequence[str]) -> list[dict]:
    values = np.asarray(classes)
    if values.ndim != 1 or len(values) == 0:
        raise V32EvaluationError("estimator classes must be a nonempty vector")
    frozen_order = list(probability_order)
    if len(set(frozen_order)) != len(frozen_order):
        raise V32EvaluationError("frozen probability class order contains duplicates")
    alignment = []
    seen = set()
    for column, raw in enumerate(values):
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, np.integer | int):
            raise V32EvaluationError("estimator class labels must be integer dataset indices")
        label = int(raw)
        if label in seen or not 0 <= label < len(dataset_authors):
            raise V32EvaluationError("estimator class label is duplicate/out of range")
        author = dataset_authors[label]
        try:
            target = frozen_order.index(author)
        except ValueError as exc:
            raise V32EvaluationError("estimator class is outside frozen probability order") from exc
        seen.add(label)
        alignment.append({"estimator_column": column, "dataset_label": label,
                          "author": author, "probability_column": target})
    if len(alignment) != len(frozen_order):
        raise V32EvaluationError(
            "estimator classes do not cover the complete frozen probability class universe"
        )
    return alignment


def evaluate_fold_v3_2(
    *, cfg, context: V32EvaluationContext, dataset_kind: str, model: str, cell: str,
    fold_index: int, work_id: str, work_content_identity: str,
    content_component_identity: str, probability_class_order: Sequence[str],
    metric_label_order: Sequence[str],
) -> dict:
    """Fit one fresh exact estimator and return one canonical in-memory work receipt."""
    _validate_context(cfg, context)
    row = resolve_cell_v3_2(model, cell, require_applied=True)  # before factory/fit
    dataset, manifest, dataset_identity = context.dataset(dataset_kind)
    if type(fold_index) is not int or type(work_id) is not str:
        raise V32EvaluationError("fold index/work ID must have exact types")
    tested = [item for item in manifest["works"] if item["tested"]]
    if not 0 <= fold_index < len(tested) or tested[fold_index]["fold_index"] != fold_index:
        raise V32EvaluationError("wrong/non-contiguous v3.2 fold index")
    fold_row = tested[fold_index]
    if fold_row["work_id"] != work_id:
        raise V32EvaluationError("fold index and full work ID disagree")
    if (fold_row["work_content_identity"] != work_content_identity
            or fold_row["content_component_identity"] != content_component_identity):
        raise V32EvaluationError("work/content identity mismatch")
    if list(probability_class_order) != list(manifest["probability_class_order"]):
        raise V32EvaluationError("probability class order mismatch")
    if list(metric_label_order) != list(manifest["metric_label_order"]):
        raise V32EvaluationError("metric class order mismatch")
    groups = np.asarray(dataset.groups, dtype=object)
    mask_test = groups == work_id
    if not mask_test.any() or set(groups[mask_test]) != {work_id}:
        raise V32EvaluationError("fold work has no exact whole-work test rows")
    mask_train = ~mask_test
    train_groups = groups[mask_train]
    if work_id in set(map(str, train_groups)):
        raise V32EvaluationError("held-out work reached train groups")
    y_train = np.asarray(dataset.y)[mask_train]
    true_values = np.unique(np.asarray(dataset.y)[mask_test])
    if len(true_values) != 1 or int(true_values[0]) not in set(map(int, y_train)):
        raise V32EvaluationError("held-out work author is absent or incoherent in train")

    # Every identity/order/leakage/applicability check above precedes factory construction.
    factory = make_factory_for_ablation(model, cfg, ablation=row.ablation)
    estimator = factory()  # fresh for every call
    train_texts = np.asarray(dataset.texts, dtype=object)[mask_train]
    test_texts = np.asarray(dataset.texts, dtype=object)[mask_test]
    with observe_fit_v3_2() as fit_trace:
        fit_estimator(estimator, train_texts, y_train, train_groups)
    classes = np.asarray(estimator.classes_)
    # The estimator may expose the complete class universe in a different column order; validate
    # probability math independently, then align those explicit classes to the frozen author order.
    alignment = _class_alignment(classes, probability_class_order, dataset.authors)
    raw = np.asarray(estimator.predict_proba(test_texts), dtype=np.float64)
    validate_probabilities(
        raw, rows=len(test_texts), n_classes=len(classes), name="predict_proba"
    )
    aligned_chunks = np.zeros((len(test_texts), len(probability_class_order)), dtype=np.float64)
    for item in alignment:
        aligned_chunks[:, item["probability_column"]] = raw[:, item["estimator_column"]]
    if not np.allclose(aligned_chunks.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise V32EvaluationError("class alignment lost probability mass")
    work_probabilities = np.round(aligned_chunks.mean(axis=0), 12)
    true_author = work_id.split("/", 1)[0]
    true_label = list(probability_class_order).index(true_author)
    decision = stable_top1_and_worst_tie_rank(
        work_probabilities, true_label=true_label,
        expected_width=len(probability_class_order),
    )
    fitted_state = fitted_state_v3_2(estimator, model=model)
    train_work_ids = list(dict.fromkeys(map(str, train_groups)))
    test_row_ids = [context.dataset(dataset_kind)[0].provenance.row_ids[index]
                    for index in np.flatnonzero(mask_test)]
    train_row_ids = [context.dataset(dataset_kind)[0].provenance.row_ids[index]
                     for index in np.flatnonzero(mask_train)]
    def row_digest(rows):
        return canonical_hash([
            [item.group, int(item.ordinal), item.text_sha256, item.work_id,
             item.provenance_sha256, item.chunker_config_hash]
            for item in rows
        ])
    receipt = build_receipt_v3_2({
        "context_identity": context.context_identity,
        "bindings": {
            "candidate": context.candidate_identity,
            "corrected_corpus": context.corrected_corpus_identity,
            "corpus_manifest": context.corpus_manifest_identity,
            "fold_manifest": manifest["self_hash"],
            "applicability": context.applicability_identity,
            "config": context.config_identity,
            "protocol": context.protocol_identity,
            "content_isolation": context.content_isolation_identity,
            "work_identity_catalog": context.work_identity_catalog_identity,
            "dataset_rows": dataset_identity["rows_digest"],
            "ruaa_work_selection": context.ruaa_work_selection_identity if dataset_kind == "ruaa" else None,
            "dataset_row_selection": dataset_identity["row_selection_identity"],
        },
        "dataset": dataset_kind,
        "model": model,
        "cell": cell,
        "fold_index": fold_index,
        "work_id": work_id,
        "work_content_identity": work_content_identity,
        "content_component_identity": content_component_identity,
        "requested_axes": row.requested_axes,
        "effective_axes": row.effective_axes,
        "train": {
            "n_rows": int(mask_train.sum()), "n_works": len(train_work_ids),
            "work_ids": ordered_strings_evidence(train_work_ids),
            "row_identity_digest": row_digest(train_row_ids),
        },
        "test": {
            "n_rows": int(mask_test.sum()), "n_works": 1,
            "work_ids": ordered_strings_evidence([work_id]),
            "row_identity_digest": row_digest(test_row_ids),
        },
        "class_orders": {
            "probability": ordered_strings_evidence(list(probability_class_order)),
            "metric": ordered_strings_evidence(list(metric_label_order)),
        },
        "factory_route": "stylo.eval.lobo.make_factory_for_ablation",
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "estimator_classes": [int(value) for value in classes],
        "class_alignment": alignment,
        "whole_work_probabilities": [float(value) for value in work_probabilities],
        "whole_work_probability_digest": numeric_evidence(work_probabilities),
        "vote": {
            "true_label": true_label,
            "pred_label": decision.top1,
            "pred_author": probability_class_order[decision.top1],
            "correct": decision.top1 == true_label,
            "rank": decision.true_rank,
        },
        "axis_evidence": training_axis_evidence(
            model=model, requested_axes=row.requested_axes, texts=train_texts,
            y=y_train, groups=train_groups, fit_trace=fit_trace,
        ),
        "actual_fitted_state": fitted_state,
    })
    return receipt


__all__ = [
    "CANDIDATE_IDENTITY", "CONTEXT_SCHEMA", "CORPUS_MANIFEST_IDENTITY",
    "CORRECTED_CORPUS_IDENTITY", "LOBO_FOLD_IDENTITY", "RUAA_FOLD_IDENTITY",
    "V32EvaluationContext", "V32EvaluationError", "build_evaluation_context_v3_2",
    "evaluate_fold_v3_2",
]
