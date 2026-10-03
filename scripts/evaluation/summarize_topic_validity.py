#!/usr/bin/env python3
"""Reduce saved topic-validity predictions to exploratory, text-free aggregates.

Requires the separately retained checkpoint and fold metadata; does not load a corpus,
fit a model, export work titles/predictions, or modify the original aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np

from stylo.domain.prediction_contract import validate_author_universe, validate_probabilities
from stylo.eval.paired_audit.manifest import fold_manifest_self_hash
from stylo.jsonio import artifact_self_hash, canonical_hash, dump_strict, loads_strict

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "stylo.topic_validity.paired_summary.v1"
CHECKPOINT_SCHEMA = "stylo.topic_validity.checkpoint.v1"
CELLS = ("A0", "A4")
ARMS = ("current", "topic_strict")
DRAW_COUNT = 50000
SEED = 20261003


class SummaryError(ValueError):
    """The saved inputs do not support an internally bound paired summary."""


def _require(condition, message):
    if not condition:
        raise SummaryError(message)


def _checkpoint_run_identity(bindings, commit):
    return canonical_hash([
        CHECKPOINT_SCHEMA, bindings["study_binding_identity"], bindings["context_identity"],
        bindings["implementation_source_identity"], bindings["environment_lock_identity"],
        bindings["runtime_identity"], bindings["thread_identity"], commit,
    ])


def _pair_counts(left, right, truth, mask):
    lo, ro = left == truth, right == truth
    return {
        "corrected": int((mask & ~lo & ro).sum()),
        "lost": int((mask & lo & ~ro).sum()),
        "both_correct": int((mask & lo & ro).sum()),
        "both_wrong_same_prediction": int((mask & ~lo & ~ro & (left == right)).sum()),
        "both_wrong_changed_prediction": int((mask & ~lo & ~ro & (left != right)).sum()),
    }


def summarize(aggregate, checkpoint, manifest, *, inputs, script_sha256):
    """Validate source identities and return only aggregate/per-author results."""
    _require(aggregate.get("schema") == "stylo.topic_validity.aggregate.v1", "aggregate schema mismatch")
    _require(artifact_self_hash(aggregate) == aggregate.get("self_hash"), "aggregate self hash mismatch")
    _require(manifest.get("schema") == "lobo_fold_manifest_v3_2", "fold manifest schema mismatch")
    _require(fold_manifest_self_hash(manifest) == manifest.get("self_hash"), "fold manifest self hash mismatch")
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    bindings, design = aggregate["bindings"], aggregate["design"]
    _require(canonical_hash(["stylo.topic_validity.study.v1", bindings, design]) == aggregate["study_identity"],
             "aggregate study identity mismatch")
    _require(bindings["lobo_fold_manifest_identity"] == manifest["self_hash"], "fold manifest source binding mismatch")
    _require(bindings["corrected_corpus_identity"] == manifest["corrected_corpus_digest"], "corrected corpus source binding mismatch")
    _require(bindings["corpus_manifest_identity"] == manifest["corrected_corpus_manifest_self_hash"], "corpus manifest source binding mismatch")
    commit = checkpoint.get("git_commit")
    _require(isinstance(commit, str) and len(commit) == 40 and set(commit) <= set("0123456789abcdef"),
             "checkpoint source commit malformed")
    _require(checkpoint.get("run_identity") == _checkpoint_run_identity(bindings, commit),
             "checkpoint source/environment binding mismatch")
    authors = validate_author_universe(manifest["metric_label_order"])
    classes = validate_author_universe(manifest["probability_class_order"])
    _require(len(authors) >= 2 and set(authors) <= set(classes), "invalid tested author universe")
    _require(bindings["probability_class_order_identity"] == canonical_hash(list(classes)), "probability class binding mismatch")
    _require(bindings["metric_label_order_identity"] == canonical_hash(list(authors)), "metric label binding mismatch")
    folds = sorted((w for w in manifest["works"] if w["tested"]), key=lambda w: w["fold_index"])
    n, a, p = len(folds), len(authors), len(classes)
    _require(n > 0 and len({w["work_id"] for w in folds}) == n, "duplicate or absent held-out works")
    expected_ids = []
    for i, work in enumerate(folds):
        _require(work["fold_index"] == i and work["author_id"] in authors
                 and work["work_id"].split("/", 1)[0] == work["author_id"], "fold metadata order/author mismatch")
        expected_ids.append(canonical_hash([
            "stylo.topic_validity.fold.v1", i, work["work_id"], work["author_id"],
            work["work_content_identity"], work["content_component_identity"],
        ]))
    _require(bindings["tested_fold_order_identity"] == canonical_hash(expected_ids), "tested fold source binding mismatch")
    _require(design["cells"] == list(CELLS) and design["arms"] == list(ARMS)
             and design["fold_count"] == n and design["tested_author_count"] == a
             and design["probability_class_count"] == p
             and design["unit"] == "held_out_whole_work"
             and design["accuracy_weighting"] == "equal_held_out_work"
             and design["prediction_rule"] == "stable_top1_lowest_index", "aggregate design mismatch")
    _require([c["cell"] for c in aggregate["cells"]] == list(CELLS), "aggregate cells mismatch")
    truth = np.array([classes.index(w["author_id"]) for w in folds])
    author_index = np.array([authors.index(w["author_id"]) for w in folds])
    counts = np.bincount(author_index, minlength=a)
    _require((counts >= 2).all(), "tested author has fewer than two works")
    predictions = {}
    full_mask = np.ones(n, dtype=bool)
    for cell in aggregate["cells"]:
        for arm in ARMS:
            rows = sorted(checkpoint["records"][cell["cell"]][arm], key=lambda r: r["fold_index"])
            _require(len(rows) == n and checkpoint["completed"][cell["cell"]][arm] == n, "checkpoint incomplete")
            _require(all(r["fold_index"] == i and r["fold_identity"] == expected_ids[i]
                         for i, r in enumerate(rows)), "checkpoint fold identities mismatch")
            proba = validate_probabilities([r["whole_work_probabilities"] for r in rows], rows=n, n_classes=p)
            pred = proba.argmax(axis=1)  # stable lowest-index tie policy
            digest = canonical_hash(["stylo.topic_validity.predictions.v1", bindings["tested_fold_order_identity"],
                                     cell["cell"], arm, pred.tolist()])
            _require(cell["prediction_vector_digests"][arm] == {"count": n, "sha256": digest}, "prediction digest mismatch")
            _require(cell["accuracy"][arm] == {"correct": int((pred == truth).sum()), "total": n}, "canonical accuracy mismatch")
            predictions[(cell["cell"], arm)] = pred
        left, right = [predictions[(cell["cell"], arm)] for arm in ARMS]
        _require([r["author"] for r in cell["per_author_transitions"]] == list(authors), "canonical author rows mismatch")
        for i, row in enumerate(cell["per_author_transitions"]):
            pair = _pair_counts(left, right, truth, author_index == i)
            expected = {"n_folds": int(counts[i]), "both_correct": pair["both_correct"],
                        "current_only_correct": pair["lost"], "topic_strict_only_correct": pair["corrected"],
                        "both_wrong_same_prediction": pair["both_wrong_same_prediction"],
                        "both_wrong_changed_prediction": pair["both_wrong_changed_prediction"]}
            _require(all(row[k] == v for k, v in expected.items()), "canonical author transition mismatch")
        _require(cell["delta_accuracy"] == {
            "direction": "topic_strict_minus_current", "denominator": n,
            "numerator": int((right == truth).sum() - (left == truth).sum()),
        }, "canonical delta mismatch")

    correct = {key: pred == truth for key, pred in predictions.items()}
    author_correct = {key: np.bincount(author_index, weights=ok, minlength=a) for key, ok in correct.items()}
    draws = np.random.default_rng(SEED).integers(0, a, size=(DRAW_COUNT, a))
    weights = np.column_stack([(draws == i).sum(axis=1) for i in range(a)])
    denominators = weights @ counts
    arms = []
    for key, ok in correct.items():
        ac = author_correct[key]
        recall = ac / counts
        arms.append({"cell": key[0], "arm": key[1], "correct": int(ok.sum()), "total": n,
                     "accuracy": float(ok.mean()), "macro_author_recall": float(recall.mean()),
                     "perfect_authors": int((ac == counts).sum()),
                     "min_author_recall": float(recall.min()),
                     "per_author": [{"author": author, "works": int(counts[i]), "correct": int(ac[i]), "recall": float(recall[i])}
                                    for i, author in enumerate(authors)],
                     "empirical_author_reweighting_95_percentile_range": np.quantile((weights @ ac) / denominators, [.025, .975]).tolist()})
    pairs = []
    for left, right in combinations(correct, 2):
        delta = author_correct[right] - author_correct[left]
        net = int(delta.sum())
        per_author = []
        for i, author in enumerate(authors):
            pair = _pair_counts(predictions[left], predictions[right], truth, author_index == i)
            per_author.append({"author": author, "works": int(counts[i]), **pair, "net_correct": int(delta[i])})
        loo = (net - delta) / (n - counts)
        pairs.append({"left": list(left), "right": list(right), "direction": "right_minus_left",
                      **_pair_counts(predictions[left], predictions[right], truth, full_mask),
                      "changed_top1": int((predictions[left] != predictions[right]).sum()),
                      "net_correct": net, "delta_accuracy": net / n,
                      "delta_macro_author_recall": float((delta / counts).mean()),
                      "authors_net_improved": int((delta > 0).sum()), "authors_net_worsened": int((delta < 0).sum()),
                      "per_author": per_author,
                      "empirical_author_reweighting_95_percentile_range": np.quantile((weights @ delta) / denominators, [.025, .975]).tolist(),
                      "leave_one_author_out_delta_range": [float(loo.min()), float(loo.max())]})
    result = {
        "schema": SCHEMA, "status": "exploratory_saved_prediction_summary", "confirmatory": False,
        "preregistered": False, "source_commit": commit,
        "canonical_self_hash": aggregate["self_hash"], "canonical_study_identity": aggregate["study_identity"],
        "checkpoint_run_identity": checkpoint["run_identity"],
        "script_path": "scripts/evaluation/summarize_topic_validity.py", "script_sha256": script_sha256,
        "inputs": inputs,
        "validation": "Input self hashes; corpus/manifest/class/fold identities; checkpoint source/environment binding; prediction digests; every canonical accuracy, delta and author transition reproduced. Checkpoint bytes are hashed for replay; an independent transport checksum is not supplied to this interface.",
        "design": {
            "unit": "held_out_whole_work", "tested_works": n, "tested_authors": a, "probability_classes": p,
            "accuracy_weighting": "equal_held_out_work", "macro_recall_weighting": "equal_tested_author",
            "author_work_count_range": [int(counts.min()), int(counts.max())],
            "scope": "Closed-set known-author work transfer on one corrected corpus; no unseen-author, held-out-topic, edition-shift, novel-attribution or insertion-detection validation. A4 versus A0 compares the joint W/F/R package, not the causal effect of weights alone.",
            "random_seed": SEED, "draws": DRAW_COUNT, "percentiles": [0.025, 0.975],
            "uncertainty": "Sensitivity to reweighting the observed author clusters with replacement; fixed LOBO predictions, no refitting. Not a confidence interval for generalization or authorship and not a significance test. Assumes observed authors represent an exchangeable empirical population; overlapping training dependence and new-corpus shifts are not represented.",
        }, "arms": arms, "comparisons": pairs,
    }
    result["self_hash"] = artifact_self_hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Separately retained complete checkpoint JSON")
    parser.add_argument("--fold-manifest", type=Path, required=True, help="Separately retained LOBO fold metadata JSON")
    parser.add_argument("--aggregate", type=Path, default=ROOT / "research/evidence/topic_validity_lobo_v1/aggregate.json")
    parser.add_argument("--output", type=Path, required=True, help="Separate aggregate-only summary JSON")
    args = parser.parse_args(argv)
    paths = {"aggregate": args.aggregate, "checkpoint": args.checkpoint, "fold_manifest": args.fold_manifest}
    _require(args.output.resolve() not in {p.resolve() for p in paths.values()}, "output must not overwrite an input")
    # Hash the same bytes that are parsed, avoiding a second read of mutable inputs.
    payloads = {role: path.read_bytes() for role, path in paths.items()}
    values = {role: loads_strict(data.decode("utf-8")) for role, data in payloads.items()}
    inputs = {role: {"sha256": hashlib.sha256(data).hexdigest()} for role, data in payloads.items()}
    result = summarize(values["aggregate"], values["checkpoint"], values["fold_manifest"],
                       inputs=inputs, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump_strict(result, args.output, sort_keys=True)
    print(f"schema={SCHEMA} works={result['design']['tested_works']} self_hash={result['self_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
