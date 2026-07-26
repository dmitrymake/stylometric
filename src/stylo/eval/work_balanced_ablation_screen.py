"""Bounded exploratory W/F/R screening on the frozen work-level panel.

This is intentionally a small resumable research runner, not the confirmatory paired-audit control
plane. It reuses the signed ablation factories and the frozen-panel evaluator; estimator mathematics,
fold generation, publication gates, and multiple-testing machinery do not live here.
"""
from __future__ import annotations

import copy
import pathlib
import time
from typing import Callable, Iterable, Sequence

import numpy as np

from ..jsonio import artifact_self_hash, canonical_hash, dump_strict, load_strict
from .groupkfold import evaluate_frozen_panel_factory
from .lobo import make_factory_for_ablation
from .metrics import accuracy, macro_f1, topk_accuracy
from .prediction_contract import (
    PredictionContractError,
    validate_prediction_record,
)
from .provenance import (
    ScientificEvaluationContext,
    prepare_synthetic_scientific_evaluation,
    require_scientific_evaluation_context,
)
from .run_attestation import LiveRunAttestationError
from .significance import paired_bootstrap_diff_clustered
from .work_weighting import (
    AblationEquivalentError,
    AblationNotApplicableError,
    CHUNK_WEIGHTED_LEGACY,
    FEATURE_STATE_ONLY_ABLATION,
    FULL_WB_ABLATION,
    LEGACY_ABLATION,
    RELATIVE_FW_ONLY_ABLATION,
    WEIGHTS_ONLY_ABLATION,
)

STATUS = "exploratory_screening_proxy_not_confirmatory"
SCHEMA_VERSION = "b4_wfr_pilot_v1"

CELL_ABLATIONS = {
    "A0": LEGACY_ABLATION,
    "A1": WEIGHTS_ONLY_ABLATION,
    "A2": FEATURE_STATE_ONLY_ABLATION,
    "A3": RELATIVE_FW_ONLY_ABLATION,
    "A4": FULL_WB_ABLATION,
}

# The matrix is deliberately explicit: in particular, majority/A4 is not requested because the
# generic corner factory would build a duplicate baseline without a typed applicability signal.
MODEL_PLAN = (
    ("stylo", ("A0", "A1", "A2", "A3", "A4")),
    ("bow_lr", ("A0", "A1", "A2", "A3", "A4")),
    ("delta_cos:500", ("A0", "A1", "A2", "A3", "A4")),
    ("char_cos", ("A0", "A1", "A2", "A3", "A4")),
    ("majority", ("A0",)),
)

_MODEL_ORDER = {model: i for i, (model, _) in enumerate(MODEL_PLAN)}
_CELL_ORDER = {cell: i for i, cell in enumerate(CELL_ABLATIONS)}
_COMPLETE_STATUSES = frozenset({"applied", "not_applicable", "already_in_legacy", "equivalent"})


class AblationScreenArtifactError(ValueError):
    """A resumable ablation-screen artifact is inconsistent with the requested run."""


def _axes(cell: str) -> dict[str, bool]:
    ablation = CELL_ABLATIONS[cell]
    return {
        "weights": ablation.weights,
        "feature_fit": ablation.feature_fit,
        "relative_fw": ablation.relative_fw,
    }


def _normalise_selection(values: Sequence[str] | None, allowed: Iterable[str], label: str) -> set[str]:
    allowed_order = tuple(allowed)
    if values is None:
        return set(allowed_order)
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence, not a bare string")
    chosen = list(values)
    if not chosen:
        raise ValueError(f"{label} selection is empty")
    if any(type(value) is not str or not value or value.strip() != value for value in chosen):
        raise ValueError(f"{label} must contain non-empty exact strings without surrounding space")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"duplicate {label} selection: {chosen!r}")
    unknown = [value for value in chosen if value not in allowed_order]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; allowed: {list(allowed_order)}")
    return set(chosen)


def selected_plan(
    models: Sequence[str] | None = None,
    cells: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Return the canonical selected matrix, adding the required same-model A0 dependency."""
    model_set = _normalise_selection(models, (model for model, _ in MODEL_PLAN), "models")
    cell_set = _normalise_selection(cells, CELL_ABLATIONS, "cells")
    out: list[tuple[str, str]] = []
    for model, available in MODEL_PLAN:
        if model not in model_set:
            continue
        requested = [cell for cell in available if cell in cell_set]
        if requested and any(cell != "A0" for cell in requested) and "A0" not in requested:
            requested.insert(0, "A0")
        out.extend((model, cell) for cell in requested)
    if not out:
        raise ValueError("model/cell selection has no rows in the bounded ablation matrix")
    return out


def _record_order(record: dict) -> tuple[int, int]:
    return (_MODEL_ORDER[record["model"]], _CELL_ORDER[record["cell"]])


def _replace_record(artifact: dict, record: dict) -> None:
    key = (record["model"], record["cell"])
    artifact["cells"] = [
        existing for existing in artifact["cells"]
        if (existing.get("model"), existing.get("cell")) != key
    ]
    artifact["cells"].append(record)
    artifact["cells"].sort(key=_record_order)


def _find_record(artifact: dict, model: str, cell: str) -> dict | None:
    for record in artifact.get("cells", []):
        if record.get("model") == model and record.get("cell") == cell:
            return record
    return None


def _expected_metadata(
    manifest: dict,
    *,
    config_path,
    config_sha256: str,
    git_commit: str,
    git_dirty: bool,
    code_hashes: dict,
    runtime_fingerprint: dict,
    seed: int,
    bootstrap_iters: int,
    ci_level: float,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "git_commit": str(git_commit),
        "git_dirty": bool(git_dirty),
        "config": {"path": str(config_path), "sha256": str(config_sha256)},
        "panel": {
            "name": manifest["panel"],
            "self_hash": manifest["self_hash"],
            "config_hash": manifest["config_hash"],
            "parent_dataset_digest": manifest["parent_dataset_digest"],
            "k_folds": int(manifest["k_folds"]),
            "n_authors": int(manifest["n_authors"]),
            "n_works": int(manifest["n_works"]),
            "fold_sizes": [int(value) for value in manifest["fold_sizes"]],
        },
        "seed": int(seed),
        "bootstrap": {
            "kind": "paired_author_clustered_accuracy",
            "iterations": int(bootstrap_iters),
            "level": float(ci_level),
        },
        "runtime_fingerprint": copy.deepcopy(runtime_fingerprint),
        "code_hashes": copy.deepcopy(code_hashes),
        "authors": list(manifest["authors"]),
        "classes": [
            {"label": index, "author": author}
            for index, author in enumerate(manifest["authors"])
        ],
        "works": copy.deepcopy(list(manifest["works"])),
    }


def _new_artifact(metadata: dict) -> dict:
    artifact = {
        **metadata,
        "cells": [],
        "triage": {
            "label": "no_clear_signal",
            "rationale": "No completed directional comparison is available yet.",
            "scope": "engineering triage for a confirmatory run, not an authorship claim",
        },
    }
    artifact["self_hash"] = artifact_self_hash(artifact)
    return artifact


def _finite_number(value, *, field: str, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise AblationScreenArtifactError(f"{field} must be a {qualifier} number")
    return float(value)


def _assert_numeric_equal(actual, expected, *, field: str, atol: float = 1e-12) -> None:
    actual_value = _finite_number(actual, field=field)
    expected_value = _finite_number(expected, field=f"recomputed {field}")
    if not np.isclose(actual_value, expected_value, rtol=0.0, atol=atol):
        raise AblationScreenArtifactError(
            f"{field} mismatch: stored={actual_value!r}, recomputed={expected_value!r}"
        )


def _validate_applied_record(
    record: dict,
    metadata: dict,
    baseline: dict | None,
) -> None:
    """Recompute every scientific field consumed by resume/triage.

    This is the single contract used for freshly evaluated and resumed rows.
    A self-hash authenticates bytes only; it cannot make internally inconsistent
    probabilities, predictions, ranks, folds, or metrics scientifically valid.
    """

    model, cell = record["model"], record["cell"]
    for key in ("metrics", "works", "folds", "timing"):
        if key not in record:
            raise AblationScreenArtifactError(f"applied cell {model}/{cell} lacks {key}")
    if not isinstance(record["works"], list):
        raise AblationScreenArtifactError(f"applied cell {model}/{cell} works must be a list")

    expected_works = metadata["works"]
    expected_ids = [work["work_id"] for work in expected_works]
    got_ids = [
        work.get("work_id") if isinstance(work, dict) else None
        for work in record["works"]
    ]
    if got_ids != expected_ids:
        raise AblationScreenArtifactError(f"applied cell {model}/{cell} work inventory mismatch")

    authors = metadata["authors"]
    n_authors = len(authors)
    for index, (work, expected) in enumerate(
        zip(record["works"], expected_works, strict=True)
    ):
        where = f"{model}/{cell}/{expected['work_id']}"
        if not isinstance(work, dict):
            raise AblationScreenArtifactError(f"{where}: work row must be an object")
        if type(work.get("fold")) is not int or work["fold"] != int(expected["fold"]):
            raise AblationScreenArtifactError(f"{where}: fold mismatch")
        if (
            type(work.get("true_label")) is not int
            or work["true_label"] != int(expected["label"])
            or not 0 <= work["true_label"] < n_authors
        ):
            raise AblationScreenArtifactError(f"{where}: true_label mismatch")
        if work.get("true_author") != authors[work["true_label"]]:
            raise AblationScreenArtifactError(f"{where}: true_author mismatch")
        try:
            decision = validate_prediction_record(
                probabilities=work.get("probabilities"),
                pred_label=work.get("pred_label"),
                true_label=work.get("true_label"),
                correct=work.get("correct"),
                rank=work.get("rank"),
                expected_width=n_authors,
            )
        except PredictionContractError as exc:
            raise AblationScreenArtifactError(
                f"{where}: prediction contract failed: {exc}"
            ) from exc
        if work.get("pred_author") != authors[decision.top1]:
            raise AblationScreenArtifactError(f"{where}: pred_author mismatch")

    if cell != "A0" and (baseline is None or baseline.get("status") != "applied"):
        raise AblationScreenArtifactError(
            f"{model}/{cell} has no validated applied A0 dependency"
        )
    delta, clustered_ci, fold_delta = _paired_metrics(
        record,
        baseline,
        bootstrap_iters=int(metadata["bootstrap"]["iterations"]),
        ci_level=float(metadata["bootstrap"]["level"]),
        seed=int(metadata["seed"]),
    )
    pred = np.asarray([work["pred_label"] for work in record["works"]], dtype=int)
    truth = np.asarray([work["true_label"] for work in record["works"]], dtype=int)
    ranks = np.asarray([work["rank"] for work in record["works"]], dtype=int)
    recomputed_metrics = {
        "accuracy": accuracy(truth, pred),
        "macro_f1": macro_f1(truth, pred, range(n_authors)),
        "top2": topk_accuracy(ranks, 2),
        "delta_accuracy_vs_A0": delta,
    }
    metrics = record["metrics"]
    expected_metric_keys = {
        *recomputed_metrics,
        "author_clustered_bootstrap_ci",
    }
    if not isinstance(metrics, dict) or set(metrics) != expected_metric_keys:
        raise AblationScreenArtifactError(
            f"{model}/{cell}: metric field inventory mismatch"
        )
    for key, value in recomputed_metrics.items():
        _assert_numeric_equal(metrics[key], value, field=f"{model}/{cell}.metrics.{key}")
    stored_ci = metrics["author_clustered_bootstrap_ci"]
    if not isinstance(stored_ci, dict) or set(stored_ci) != {"point", "lo", "hi"}:
        raise AblationScreenArtifactError(
            f"{model}/{cell}: clustered CI field inventory mismatch"
        )
    for key in ("point", "lo", "hi"):
        _assert_numeric_equal(
            stored_ci[key],
            clustered_ci[key],
            field=f"{model}/{cell}.metrics.author_clustered_bootstrap_ci.{key}",
        )

    k_folds = int(metadata["panel"]["k_folds"])
    folds = record["folds"]
    if not isinstance(folds, list) or len(folds) != k_folds:
        raise AblationScreenArtifactError(f"{model}/{cell}: fold inventory mismatch")
    fold_fit = 0.0
    fold_predict = 0.0
    for fold_index, fold in enumerate(folds):
        where = f"{model}/{cell}.folds[{fold_index}]"
        expected_keys = {
            "fold",
            "n_works",
            "n_train_chunks",
            "n_test_chunks",
            "accuracy",
            "delta_accuracy_vs_A0",
            "fit_seconds",
            "predict_seconds",
        }
        if not isinstance(fold, dict) or set(fold) != expected_keys:
            raise AblationScreenArtifactError(f"{where}: field inventory mismatch")
        if type(fold["fold"]) is not int or fold["fold"] != fold_index:
            raise AblationScreenArtifactError(f"{where}: non-canonical fold index")
        fold_works = [work for work in record["works"] if work["fold"] == fold_index]
        if type(fold["n_works"]) is not int or fold["n_works"] != len(fold_works):
            raise AblationScreenArtifactError(f"{where}: n_works mismatch")
        for key in ("n_train_chunks", "n_test_chunks"):
            if type(fold[key]) is not int or fold[key] < 0:
                raise AblationScreenArtifactError(
                    f"{where}.{key} must be a non-negative exact integer"
                )
        _assert_numeric_equal(
            fold["accuracy"],
            float(np.mean([work["correct"] for work in fold_works])),
            field=f"{where}.accuracy",
        )
        _assert_numeric_equal(
            fold["delta_accuracy_vs_A0"],
            fold_delta[fold_index],
            field=f"{where}.delta_accuracy_vs_A0",
        )
        fold_fit += _finite_number(
            fold["fit_seconds"], field=f"{where}.fit_seconds", nonnegative=True
        )
        fold_predict += _finite_number(
            fold["predict_seconds"], field=f"{where}.predict_seconds", nonnegative=True
        )

    timing = record["timing"]
    if not isinstance(timing, dict) or set(timing) != {
        "fit_seconds",
        "predict_seconds",
        "total_seconds",
    }:
        raise AblationScreenArtifactError(f"{model}/{cell}: timing field inventory mismatch")
    fit = _finite_number(
        timing["fit_seconds"], field=f"{model}/{cell}.timing.fit_seconds", nonnegative=True
    )
    predict = _finite_number(
        timing["predict_seconds"],
        field=f"{model}/{cell}.timing.predict_seconds",
        nonnegative=True,
    )
    total = _finite_number(
        timing["total_seconds"],
        field=f"{model}/{cell}.timing.total_seconds",
        nonnegative=True,
    )
    _assert_numeric_equal(fit, fold_fit, field=f"{model}/{cell}.timing.fit_seconds")
    _assert_numeric_equal(
        predict, fold_predict, field=f"{model}/{cell}.timing.predict_seconds"
    )
    if total + 1e-12 < fit + predict:
        raise AblationScreenArtifactError(
            f"{model}/{cell}.timing.total_seconds is smaller than fit+predict"
        )


def _validate_record(
    record: dict,
    metadata: dict,
    baseline: dict | None = None,
) -> None:
    if not isinstance(record, dict):
        raise AblationScreenArtifactError("cell record must be an object")
    model, cell, status = record.get("model"), record.get("cell"), record.get("status")
    allowed_cells = dict(MODEL_PLAN).get(model)
    if allowed_cells is None or cell not in allowed_cells:
        raise AblationScreenArtifactError(f"cell record outside bounded matrix: {model!r}/{cell!r}")
    if record.get("axes") != _axes(cell):
        raise AblationScreenArtifactError(f"cell axes mismatch for {model}/{cell}")
    if status not in _COMPLETE_STATUSES | {"failed"}:
        raise AblationScreenArtifactError(f"unknown cell status {status!r} for {model}/{cell}")
    if status in {"not_applicable", "already_in_legacy", "equivalent"}:
        forbidden = {"metrics", "works", "folds", "timing"} & set(record)
        if forbidden:
            raise AblationScreenArtifactError(
                f"typed applicability row {model}/{cell} carries fake result fields {sorted(forbidden)}")
        if status == "equivalent":
            equivalent_to = record.get("equivalent_to")
            if equivalent_to not in allowed_cells or equivalent_to == cell:
                raise AblationScreenArtifactError(
                    f"equivalent row {model}/{cell} has invalid target {equivalent_to!r}"
                )
        elif record.get("reason") != status:
            raise AblationScreenArtifactError(
                f"typed applicability row {model}/{cell} reason/status mismatch"
            )
        return
    if status == "failed":
        if (
            type(record.get("error_type")) is not str
            or not record["error_type"]
            or type(record.get("error_message")) is not str
        ):
            raise AblationScreenArtifactError(f"failed cell {model}/{cell} lacks its error")
        _finite_number(
            record.get("runtime_seconds"),
            field=f"{model}/{cell}.runtime_seconds",
            nonnegative=True,
        )
        return
    _validate_applied_record(record, metadata, baseline)


def _load_or_create(output_path: pathlib.Path, metadata: dict) -> dict:
    if not output_path.exists():
        return _new_artifact(metadata)
    artifact = load_strict(output_path)
    if not isinstance(artifact, dict):
        raise AblationScreenArtifactError("existing ablation-screen output is not a JSON object")
    if artifact.get("self_hash") != artifact_self_hash(artifact):
        raise AblationScreenArtifactError("existing ablation-screen output self_hash mismatch")
    for key, expected in metadata.items():
        if artifact.get(key) != expected:
            if key == "config":
                detail = "config hash/path mismatch"
            elif key == "panel":
                detail = "panel hash/inventory mismatch"
            else:
                detail = f"resume metadata mismatch: {key}"
            raise AblationScreenArtifactError(detail)
    records = artifact.get("cells")
    if not isinstance(records, list):
        raise AblationScreenArtifactError("existing ablation-screen output cells must be a list")
    seen = set()
    baselines: dict[str, dict] = {}
    for record in records:
        baseline = (
            None
            if record.get("cell") == "A0"
            else baselines.get(str(record.get("model")))
        )
        _validate_record(record, metadata, baseline)
        key = (record["model"], record["cell"])
        if key in seen:
            raise AblationScreenArtifactError(f"duplicate cell record {key}")
        seen.add(key)
        if record.get("cell") == "A0" and record.get("status") == "applied":
            baselines[str(record["model"])] = record
    if records != sorted(records, key=_record_order):
        raise AblationScreenArtifactError("existing ablation-screen cells are not in canonical order")
    if artifact.get("triage") != _triage(artifact):
        raise AblationScreenArtifactError(
            "existing ablation-screen triage does not match recomputed records"
        )
    return artifact


def _triage(artifact: dict) -> dict[str, str]:
    applied = [record for record in artifact["cells"] if record.get("status") == "applied"]
    failed = [record for record in artifact["cells"] if record.get("status") == "failed"]
    scope = "engineering triage for a confirmatory run, not an authorship claim"
    if failed:
        cells = ", ".join(f"{r['model']}/{r['cell']}" for r in failed)
        return {
            "label": "no_clear_signal",
            "rationale": f"Screening is incomplete because cells failed: {cells}.",
            "scope": scope,
        }
    primary = next(
        (record for record in applied if record["model"] == "stylo" and record["cell"] == "A4"),
        None,
    )
    if primary is not None:
        delta = primary["metrics"]["delta_accuracy_vs_A0"]
        fold_deltas = [fold["delta_accuracy_vs_A0"] for fold in primary["folds"]]
        if delta < -0.02 and sum(value <= 0.0 for value in fold_deltas) >= 4:
            return {
                "label": "material_regression",
                "rationale": (
                    "The primary stylo/A4 proxy is more than 2 pp below A0 and the loss appears "
                    "in at least four frozen folds."
                ),
                "scope": scope,
            }
    for record in applied:
        if record["cell"] == "A0":
            continue
        metrics = record["metrics"]
        ci = metrics["author_clustered_bootstrap_ci"]
        fold_deltas = [fold["delta_accuracy_vs_A0"] for fold in record["folds"]]
        if (
            metrics["delta_accuracy_vs_A0"] > 0.0
            and ci["lo"] > 0.0
            and sum(value > 0.0 for value in fold_deltas) >= 4
        ):
            return {
                "label": "promising_directional_signal",
                "rationale": (
                    f"{record['model']}/{record['cell']} improves in at least four folds and its "
                    "author-clustered exploratory interval stays above zero."
                ),
                "scope": scope,
            }
    return {
        "label": "no_clear_signal",
        "rationale": (
            "Completed effects are small, fold-unstable, or have an author-clustered exploratory "
            "interval that does not separate from zero."
        ),
        "scope": scope,
    }


def _save(artifact: dict, output_path: pathlib.Path) -> None:
    artifact["triage"] = _triage(artifact)
    artifact["self_hash"] = artifact_self_hash(artifact)
    dump_strict(artifact, output_path, indent=2, ensure_ascii=False)


def _call_progress(progress: Callable[[dict], None] | None, event: str, **fields) -> None:
    if progress is not None:
        progress({"event": event, **fields})


def _paired_metrics(
    record: dict,
    baseline: dict | None,
    *,
    bootstrap_iters: int,
    ci_level: float,
    seed: int,
) -> tuple[float, dict[str, float], dict[int, float]]:
    current_works = record["works"]
    current_correct = np.asarray([work["correct"] for work in current_works], dtype=float)
    current_fold_acc = {
        int(fold): float(np.mean([
            work["correct"] for work in current_works if int(work["fold"]) == fold
        ]))
        for fold in sorted({int(work["fold"]) for work in current_works})
    }
    if baseline is None:
        return 0.0, {"point": 0.0, "lo": 0.0, "hi": 0.0}, {
            fold: 0.0 for fold in current_fold_acc
        }
    baseline_works = baseline["works"]
    current_ids = [work["work_id"] for work in current_works]
    baseline_ids = [work["work_id"] for work in baseline_works]
    if current_ids != baseline_ids:
        raise AblationScreenArtifactError(
            f"paired inventory mismatch for {record['model']}/{record['cell']} vs A0")
    for current, a0 in zip(current_works, baseline_works, strict=True):
        if (current["fold"], current["true_label"]) != (a0["fold"], a0["true_label"]):
            raise AblationScreenArtifactError(
                f"paired truth/fold mismatch at {current['work_id']} vs A0")
    baseline_correct = np.asarray([work["correct"] for work in baseline_works], dtype=float)
    author_groups = np.asarray([work["true_author"] for work in current_works], dtype=object)
    ci = paired_bootstrap_diff_clustered(
        lambda ix: float(current_correct[ix].mean()),
        lambda ix: float(baseline_correct[ix].mean()),
        author_groups,
        iters=bootstrap_iters,
        level=ci_level,
        seed=seed,
    )
    delta = float(current_correct.mean() - baseline_correct.mean())
    if not np.isclose(ci.diff, delta, rtol=0.0, atol=1e-15):
        raise AblationScreenArtifactError("clustered CI point does not equal paired accuracy delta")
    baseline_fold_acc = {
        int(fold): float(np.mean([
            work["correct"] for work in baseline_works if int(work["fold"]) == fold
        ]))
        for fold in current_fold_acc
    }
    fold_delta = {
        fold: float(current_fold_acc[fold] - baseline_fold_acc[fold])
        for fold in current_fold_acc
    }
    return delta, {"point": delta, "lo": float(ci.lo), "hi": float(ci.hi)}, fold_delta


def _applied_record(
    model: str,
    cell: str,
    df,
    probabilities,
    y_true,
    timing: dict,
    manifest: dict,
    baseline: dict | None,
    *,
    bootstrap_iters: int,
    ci_level: float,
    seed: int,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true)
    expected_works = list(manifest["works"])
    expected_ids = [work["work_id"] for work in expected_works]
    got_ids = [f"{row.test_author}/{row.test_book}" for row in df.itertuples()]
    if got_ids != expected_ids:
        raise AblationScreenArtifactError(f"{model}/{cell}: evaluator work order != frozen manifest")
    if probabilities.shape != (len(expected_ids), len(manifest["authors"])):
        raise AblationScreenArtifactError(
            f"{model}/{cell}: probability shape {probabilities.shape} does not match panel")
    if not np.isfinite(probabilities).all() or (probabilities < -1e-9).any():
        raise AblationScreenArtifactError(f"{model}/{cell}: invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AblationScreenArtifactError(f"{model}/{cell}: work probabilities do not sum to one")
    if y_true.tolist() != [int(work["label"]) for work in expected_works]:
        raise AblationScreenArtifactError(f"{model}/{cell}: y_true != frozen manifest")

    works = []
    for i, row in enumerate(df.itertuples()):
        expected = expected_works[i]
        if int(row.fold) != int(expected["fold"]) or int(row.true_label) != int(expected["label"]):
            raise AblationScreenArtifactError(f"{model}/{cell}/{expected['work_id']}: fold/truth mismatch")
        pred_label = int(row.pred_label)
        works.append({
            "work_id": expected["work_id"],
            "fold": int(row.fold),
            "true_label": int(row.true_label),
            "true_author": manifest["authors"][int(row.true_label)],
            "pred_label": pred_label,
            "pred_author": manifest["authors"][pred_label],
            "rank": int(row.rank),
            "correct": bool(row.correct),
            "probabilities": probabilities[i].tolist(),
        })

    record = {
        "model": model,
        "cell": cell,
        "axes": _axes(cell),
        "status": "applied",
        "works": works,
    }
    delta, clustered_ci, fold_delta = _paired_metrics(
        record,
        baseline,
        bootstrap_iters=bootstrap_iters,
        ci_level=ci_level,
        seed=seed,
    )
    pred = np.asarray([work["pred_label"] for work in works], dtype=int)
    truth = np.asarray([work["true_label"] for work in works], dtype=int)
    ranks = np.asarray([work["rank"] for work in works], dtype=int)
    record["metrics"] = {
        "accuracy": accuracy(truth, pred),
        "macro_f1": macro_f1(truth, pred, range(len(manifest["authors"]))),
        "top2": topk_accuracy(ranks, 2),
        "delta_accuracy_vs_A0": delta,
        "author_clustered_bootstrap_ci": clustered_ci,
    }
    timing_folds = {int(item["fold"]): item for item in timing.get("folds", [])}
    folds = []
    for fold in range(int(manifest["k_folds"])):
        fold_works = [work for work in works if work["fold"] == fold]
        if not fold_works or fold not in timing_folds:
            raise AblationScreenArtifactError(f"{model}/{cell}: missing fold {fold} result/timing")
        source_timing = timing_folds[fold]
        folds.append({
            "fold": fold,
            "n_works": len(fold_works),
            "n_train_chunks": int(source_timing["n_train_chunks"]),
            "n_test_chunks": int(source_timing["n_test_chunks"]),
            "accuracy": float(np.mean([work["correct"] for work in fold_works])),
            "delta_accuracy_vs_A0": float(fold_delta[fold]),
            "fit_seconds": float(source_timing["fit_seconds"]),
            "predict_seconds": float(source_timing["predict_seconds"]),
        })
    record["folds"] = folds
    record["timing"] = {
        "fit_seconds": float(timing["fit_seconds"]),
        "predict_seconds": float(timing["predict_seconds"]),
        "total_seconds": float(timing["total_seconds"]),
    }
    return record


def run_ablation_screen(
    cfg,
    dataset,
    manifest: dict,
    output_path,
    *,
    config_path,
    config_sha256: str,
    git_commit: str,
    git_dirty: bool,
    code_hashes: dict,
    runtime_fingerprint: dict,
    models: Sequence[str] | None = None,
    cells: Sequence[str] | None = None,
    seed: int = 42,
    bootstrap_iters: int = 1000,
    ci_level: float = 0.95,
    evaluator=evaluate_frozen_panel_factory,
    clock=None,
    continue_on_error: bool = True,
    progress: Callable[[dict], None] | None = None,
    attestor=None,
) -> dict:
    """Run or resume selected cells, atomically saving after every completed cell."""
    if type(seed) is not int:
        raise TypeError("seed must be an exact int")
    if type(bootstrap_iters) is not int or bootstrap_iters <= 0:
        raise ValueError("bootstrap_iters must be a positive exact int")
    if isinstance(ci_level, bool) or not isinstance(ci_level, (int, float)) or not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between zero and one")
    if not isinstance(manifest, dict) or manifest.get("self_hash") is None:
        raise TypeError("manifest must be a verified screening-panel dict")
    if type(dataset) is ScientificEvaluationContext:
        dataset = require_scientific_evaluation_context(dataset)
    else:
        dataset = prepare_synthetic_scientific_evaluation(
            dataset,
            CHUNK_WEIGHTED_LEGACY,
        )
    if dataset.weighting != CHUNK_WEIGHTED_LEGACY:
        raise AblationScreenArtifactError(
            "frozen screening panel requires the legacy dataset arm"
        )
    plan = selected_plan(models, cells)
    output_path = pathlib.Path(output_path)
    metadata = _expected_metadata(
        manifest,
        config_path=config_path,
        config_sha256=config_sha256,
        git_commit=git_commit,
        git_dirty=git_dirty,
        code_hashes=code_hashes,
        runtime_fingerprint=runtime_fingerprint,
        seed=seed,
        bootstrap_iters=bootstrap_iters,
        ci_level=float(ci_level),
    )
    artifact = _load_or_create(output_path, metadata)
    runner_clock = clock if clock is not None else time.perf_counter
    if attestor is not None:
        attestor.verify("run-start")

    for model, cell in plan:
        if attestor is not None:
            attestor.verify(f"before-{model}-{cell}")
        existing = _find_record(artifact, model, cell)
        if existing is not None and existing.get("status") in _COMPLETE_STATUSES:
            _call_progress(progress, "skip", model=model, cell=cell, status=existing["status"])
            continue
        _call_progress(progress, "start", model=model, cell=cell)
        cell_started = runner_clock()
        try:
            factory = make_factory_for_ablation(model, cfg, ablation=CELL_ABLATIONS[cell])
        except AblationNotApplicableError as exc:
            record = {
                "model": model,
                "cell": cell,
                "axes": _axes(cell),
                "status": exc.reason,
                "reason": exc.reason,
            }
            _validate_record(record, artifact)
            _replace_record(artifact, record)
            if attestor is not None:
                attestor.verify(f"before-checkpoint-{model}-{cell}")
            _save(artifact, output_path)
            _call_progress(progress, "complete", model=model, cell=cell, status=record["status"])
            continue
        except AblationEquivalentError as exc:
            record = {
                "model": model,
                "cell": cell,
                "axes": _axes(cell),
                "status": "equivalent",
                "equivalent_to": exc.equivalent_to,
            }
            _validate_record(record, artifact)
            _replace_record(artifact, record)
            if attestor is not None:
                attestor.verify(f"before-checkpoint-{model}-{cell}")
            _save(artifact, output_path)
            _call_progress(progress, "complete", model=model, cell=cell, status=record["status"])
            continue
        except LiveRunAttestationError:
            raise
        except Exception as exc:
            record = {
                "model": model,
                "cell": cell,
                "axes": _axes(cell),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "runtime_seconds": float(runner_clock() - cell_started),
            }
            _validate_record(record, artifact)
            _replace_record(artifact, record)
            if attestor is not None:
                attestor.verify(f"before-checkpoint-{model}-{cell}")
            _save(artifact, output_path)
            _call_progress(
                progress,
                "failed",
                model=model,
                cell=cell,
                status="failed",
                error_type=record["error_type"],
                error_message=record["error_message"],
            )
            if not continue_on_error:
                raise
            continue

        try:
            evaluator_kwargs = {"spec": f"{model}/{cell}"}
            if clock is not None:
                evaluator_kwargs["clock"] = clock
            df, probabilities, y_true, timing = evaluator(
                cfg, dataset, factory, manifest, **evaluator_kwargs)
            if attestor is not None:
                attestor.verify(f"after-evaluation-{model}-{cell}")
            baseline = None if cell == "A0" else _find_record(artifact, model, "A0")
            if cell != "A0" and (baseline is None or baseline.get("status") != "applied"):
                raise AblationScreenArtifactError(f"{model}/{cell} has no completed applied A0 dependency")
            record = _applied_record(
                model,
                cell,
                df,
                probabilities,
                y_true,
                timing,
                manifest,
                baseline,
                bootstrap_iters=bootstrap_iters,
                ci_level=float(ci_level),
                seed=seed,
            )
        except LiveRunAttestationError:
            raise
        except Exception as exc:
            record = {
                "model": model,
                "cell": cell,
                "axes": _axes(cell),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "runtime_seconds": float(runner_clock() - cell_started),
            }
            _validate_record(record, artifact)
            _replace_record(artifact, record)
            if attestor is not None:
                attestor.verify(f"before-checkpoint-{model}-{cell}")
            _save(artifact, output_path)
            _call_progress(
                progress,
                "failed",
                model=model,
                cell=cell,
                status="failed",
                error_type=record["error_type"],
                error_message=record["error_message"],
            )
            if not continue_on_error:
                raise
            continue

        _validate_record(record, artifact, baseline)
        _replace_record(artifact, record)
        if attestor is not None:
            attestor.verify(f"before-checkpoint-{model}-{cell}")
        _save(artifact, output_path)
        if attestor is not None:
            attestor.verify(f"after-checkpoint-{model}-{cell}")
        _call_progress(
            progress,
            "complete",
            model=model,
            cell=cell,
            status="applied",
            runtime_seconds=record["timing"]["total_seconds"],
        )

    # A pure resume whose selected cells were all skipped must preserve the exact artifact bytes.
    if attestor is not None:
        attestor.verify("before-final-return")
    return artifact


def format_compact_table(artifact: dict) -> str:
    """Format the deliberately compact exploratory result table."""
    header = (
        "model  cell  accuracy  macro_f1  dacc_vs_A0  clustered_CI  fold_deltas  runtime"
    )
    lines = [header]
    for record in artifact.get("cells", []):
        model, cell, status = record["model"], record["cell"], record["status"]
        if status == "applied":
            metrics = record["metrics"]
            ci = metrics["author_clustered_bootstrap_ci"]
            fold_deltas = ",".join(
                f"{fold['delta_accuracy_vs_A0']:+.3f}" for fold in record["folds"])
            lines.append(
                f"{model}  {cell}  {metrics['accuracy']:.4f}  {metrics['macro_f1']:.4f}  "
                f"{metrics['delta_accuracy_vs_A0']:+.4f}  [{ci['lo']:+.4f},{ci['hi']:+.4f}]  "
                f"{fold_deltas}  {record['timing']['total_seconds']:.1f}s"
            )
        elif status == "equivalent":
            lines.append(f"{model}  {cell}  equivalent_to:{record['equivalent_to']}")
        elif status in {"not_applicable", "already_in_legacy"}:
            lines.append(f"{model}  {cell}  {status}")
        else:
            lines.append(
                f"{model}  {cell}  failed:{record.get('error_type', 'Error')}  "
                f"{record.get('runtime_seconds', 0.0):.1f}s"
            )
    return "\n".join(lines)


__all__ = [
    "STATUS",
    "SCHEMA_VERSION",
    "CELL_ABLATIONS",
    "MODEL_PLAN",
    "AblationScreenArtifactError",
    "selected_plan",
    "run_ablation_screen",
    "format_compact_table",
]
