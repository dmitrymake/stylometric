"""Saved-prediction summary contracts on an unequal-book synthetic panel."""
import copy
import importlib.util
from pathlib import Path

import pytest

from stylo.eval.paired_audit.manifest import fold_manifest_self_hash
from stylo.jsonio import artifact_self_hash, canonical_hash


PATH = Path(__file__).resolve().parents[1] / "scripts/evaluation/summarize_topic_validity.py"
SPEC = importlib.util.spec_from_file_location("topic_summary", PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _inputs():
    authors = ["alpha", "beta", "gamma"]
    truth = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    folds = []
    for i, label in enumerate(truth):
        folds.append({"work_id": f"{authors[label]}/sample_{i}", "author_id": authors[label],
                      "work_content_identity": canonical_hash(["content", i]),
                      "content_component_identity": canonical_hash(["group", i]), "tested": True, "fold_index": i})
    manifest = {"schema": "lobo_fold_manifest_v3_2", "works": folds,
                "probability_class_order": authors, "metric_label_order": authors,
                "corrected_corpus_digest": "1" * 64, "corrected_corpus_manifest_self_hash": "2" * 64}
    manifest["self_hash"] = fold_manifest_self_hash(manifest)
    fids = [canonical_hash(["stylo.topic_validity.fold.v1", i, w["work_id"], w["author_id"],
                           w["work_content_identity"], w["content_component_identity"]]) for i, w in enumerate(folds)]
    bindings = {k: "3" * 64 for k in ["study_binding_identity", "context_identity", "implementation_source_identity",
                                      "environment_lock_identity", "runtime_identity", "thread_identity"]}
    bindings.update(lobo_fold_manifest_identity=manifest["self_hash"], corrected_corpus_identity="1" * 64,
                    corpus_manifest_identity="2" * 64, probability_class_order_identity=canonical_hash(authors),
                    metric_label_order_identity=canonical_hash(authors), tested_fold_order_identity=canonical_hash(fids))
    design = {"cells": ["A0", "A4"], "arms": ["current", "topic_strict"], "fold_count": 9,
              "tested_author_count": 3, "probability_class_count": 3, "unit": "held_out_whole_work",
              "accuracy_weighting": "equal_held_out_work", "prediction_rule": "stable_top1_lowest_index"}
    vectors = {("A0", "current"): [0, 0, 1, 1, 0, 2, 0, 0, 0],
               ("A0", "topic_strict"): [0, 1, 1, 1, 1, 2, 0, 1, 0],
               ("A4", "current"): truth, ("A4", "topic_strict"): [0, 0, 1, 1, 1, 2, 2, 2, 0]}
    transitions = {
        "A0": [(1, 1, 0, 0, 0), (2, 0, 1, 0, 0), (1, 0, 0, 2, 1)],
        "A4": [(2, 0, 0, 0, 0), (3, 0, 0, 0, 0), (3, 1, 0, 0, 0)],
    }
    names = ["both_correct", "current_only_correct", "topic_strict_only_correct",
             "both_wrong_same_prediction", "both_wrong_changed_prediction"]
    cells, records = [], {}
    for cell in ["A0", "A4"]:
        accuracies, digests, records[cell] = {}, {}, {}
        for arm in ["current", "topic_strict"]:
            pred = vectors[cell, arm]
            accuracies[arm] = {"correct": sum(t == v for t, v in zip(truth, pred)), "total": 9}
            digests[arm] = {"count": 9, "sha256": canonical_hash(["stylo.topic_validity.predictions.v1",
                           bindings["tested_fold_order_identity"], cell, arm, pred])}
            records[cell][arm] = [{"fold_index": i, "fold_identity": fids[i],
                                  "whole_work_probabilities": [float(j == v) for j in range(3)]} for i, v in enumerate(pred)]
        cells.append({"cell": cell, "accuracy": accuracies, "prediction_vector_digests": digests,
                      "delta_accuracy": {"direction": "topic_strict_minus_current", "denominator": 9,
                                         "numerator": accuracies["topic_strict"]["correct"] - accuracies["current"]["correct"]},
                      "per_author_transitions": [{"author": author, "n_folds": count, **dict(zip(names, vals))}
                                                 for author, count, vals in zip(authors, [2, 3, 4], transitions[cell])]})
    aggregate = {"schema": "stylo.topic_validity.aggregate.v1", "bindings": bindings, "design": design, "cells": cells}
    aggregate["study_identity"] = canonical_hash(["stylo.topic_validity.study.v1", bindings, design])
    aggregate["self_hash"] = artifact_self_hash(aggregate)
    checkpoint = {"schema": summary.CHECKPOINT_SCHEMA, "git_commit": "4" * 40, "records": records,
                  "completed": {cell: {arm: 9 for arm in ["current", "topic_strict"]} for cell in ["A0", "A4"]}}
    checkpoint["run_identity"] = summary._checkpoint_run_identity(bindings, checkpoint["git_commit"])
    return aggregate, checkpoint, manifest


def _run(inputs):
    return summary.summarize(*inputs, inputs={role: {"sha256": "5" * 64} for role in ["aggregate", "checkpoint", "fold_manifest"]},
                             script_sha256="6" * 64)


def test_unequal_book_counts_change_author_recall_estimand():
    result = _run(_inputs())
    arm = result["arms"][0]
    assert arm["accuracy"] == pytest.approx(5 / 9)
    assert arm["macro_author_recall"] == pytest.approx((1 + 2 / 3 + 1 / 4) / 3)
    assert [r["works"] for r in arm["per_author"]] == [2, 3, 4]
    assert result["arms"][1]["accuracy"] == arm["accuracy"]
    assert result["arms"][1]["macro_author_recall"] < arm["macro_author_recall"]
    assert artifact_self_hash(result) == result["self_hash"]


def test_pairs_keep_losses_and_wrong_prediction_changes():
    result = _run(_inputs())
    assert len(result["comparisons"]) == 6
    pair = result["comparisons"][0]
    assert {k: pair[k] for k in ["corrected", "lost", "both_correct", "both_wrong_same_prediction", "both_wrong_changed_prediction"]} == {
        "corrected": 1, "lost": 1, "both_correct": 4, "both_wrong_same_prediction": 2, "both_wrong_changed_prediction": 1}
    assert pair["changed_top1"] == 3 and pair["net_correct"] == 0
    wb = next(c for c in result["comparisons"] if c["left"] == ["A0", "current"] and c["right"] == ["A4", "current"])
    assert wb["corrected"] == 4 and wb["lost"] == 0
    assert wb["authors_net_improved"] == 2


def test_modified_manifest_fails_self_hash_before_reduction():
    inputs = copy.deepcopy(_inputs())
    inputs[2]["works"][0]["work_content_identity"] = "7" * 64
    with pytest.raises(summary.SummaryError, match="manifest self hash mismatch"):
        _run(inputs)


def test_source_commit_mismatch_rejected_even_when_predictions_match():
    inputs = copy.deepcopy(_inputs())
    inputs[1]["git_commit"] = "8" * 40
    with pytest.raises(summary.SummaryError, match="source/environment binding mismatch"):
        _run(inputs)
