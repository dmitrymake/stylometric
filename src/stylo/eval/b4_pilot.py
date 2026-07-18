"""Bounded exploratory W/F/R screening on the frozen work-level panel.

This is intentionally a small resumable research runner, not the confirmatory B4 control plane.
It reuses the signed ablation factories and the frozen-panel evaluator; estimator mathematics,
fold generation, publication gates, and multiple-testing machinery do not live here.
"""
from __future__ import annotations

import copy
import hashlib
import pathlib
import time
from typing import Callable, Iterable, Sequence

import numpy as np

from ..jsonio import dump_strict, dumps_strict, load_strict
from .groupkfold import evaluate_frozen_panel_factory
from .lobo import make_factory_for_ablation
from .metrics import accuracy, macro_f1, topk_accuracy
from .significance import paired_bootstrap_diff_clustered
from .work_weighting import (
    AblationEquivalentError,
    AblationNotApplicableError,
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


class PilotArtifactError(ValueError):
    """A resumable pilot artifact is inconsistent with the requested run."""


def canonical_hash(obj) -> str:
    """SHA256 of strict, sorted, compact JSON."""
    payload = dumps_strict(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_self_hash(artifact: dict) -> str:
    """Canonical content hash excluding only the hash field itself."""
    if not isinstance(artifact, dict):
        raise TypeError("artifact must be a dict")
    return canonical_hash({key: value for key, value in artifact.items() if key != "self_hash"})


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
        raise ValueError("model/cell selection has no rows in the bounded pilot matrix")
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


def _validate_record(record: dict, metadata: dict) -> None:
    if not isinstance(record, dict):
        raise PilotArtifactError("cell record must be an object")
    model, cell, status = record.get("model"), record.get("cell"), record.get("status")
    allowed_cells = dict(MODEL_PLAN).get(model)
    if allowed_cells is None or cell not in allowed_cells:
        raise PilotArtifactError(f"cell record outside bounded matrix: {model!r}/{cell!r}")
    if record.get("axes") != _axes(cell):
        raise PilotArtifactError(f"cell axes mismatch for {model}/{cell}")
    if status not in _COMPLETE_STATUSES | {"failed"}:
        raise PilotArtifactError(f"unknown cell status {status!r} for {model}/{cell}")
    if status in {"not_applicable", "already_in_legacy", "equivalent"}:
        forbidden = {"metrics", "works", "folds", "timing"} & set(record)
        if forbidden:
            raise PilotArtifactError(
                f"typed applicability row {model}/{cell} carries fake result fields {sorted(forbidden)}")
        return
    if status == "failed":
        if not record.get("error_type") or "error_message" not in record:
            raise PilotArtifactError(f"failed cell {model}/{cell} lacks its error")
        return
    for key in ("metrics", "works", "folds", "timing"):
        if key not in record:
            raise PilotArtifactError(f"applied cell {model}/{cell} lacks {key}")
    expected_ids = [work["work_id"] for work in metadata["works"]]
    got_ids = [work.get("work_id") for work in record["works"]]
    if got_ids != expected_ids:
        raise PilotArtifactError(f"applied cell {model}/{cell} work inventory mismatch")
    n_authors = len(metadata["authors"])
    for work in record["works"]:
        probs = work.get("probabilities")
        if not isinstance(probs, list) or len(probs) != n_authors:
            raise PilotArtifactError(f"{model}/{cell}/{work.get('work_id')}: bad probabilities")


def _load_or_create(output_path: pathlib.Path, metadata: dict) -> dict:
    if not output_path.exists():
        return _new_artifact(metadata)
    artifact = load_strict(output_path)
    if not isinstance(artifact, dict):
        raise PilotArtifactError("existing pilot output is not a JSON object")
    if artifact.get("self_hash") != artifact_self_hash(artifact):
        raise PilotArtifactError("existing pilot output self_hash mismatch")
    for key, expected in metadata.items():
        if artifact.get(key) != expected:
            if key == "config":
                detail = "config hash/path mismatch"
            elif key == "panel":
                detail = "panel hash/inventory mismatch"
            else:
                detail = f"resume metadata mismatch: {key}"
            raise PilotArtifactError(detail)
    records = artifact.get("cells")
    if not isinstance(records, list):
        raise PilotArtifactError("existing pilot output cells must be a list")
    seen = set()
    for record in records:
        _validate_record(record, metadata)
        key = (record["model"], record["cell"])
        if key in seen:
            raise PilotArtifactError(f"duplicate cell record {key}")
        seen.add(key)
    if records != sorted(records, key=_record_order):
        raise PilotArtifactError("existing pilot cells are not in canonical order")
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
        raise PilotArtifactError(
            f"paired inventory mismatch for {record['model']}/{record['cell']} vs A0")
    for current, a0 in zip(current_works, baseline_works, strict=True):
        if (current["fold"], current["true_label"]) != (a0["fold"], a0["true_label"]):
            raise PilotArtifactError(
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
        raise PilotArtifactError("clustered CI point does not equal paired accuracy delta")
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
        raise PilotArtifactError(f"{model}/{cell}: evaluator work order != frozen manifest")
    if probabilities.shape != (len(expected_ids), len(manifest["authors"])):
        raise PilotArtifactError(
            f"{model}/{cell}: probability shape {probabilities.shape} does not match panel")
    if not np.isfinite(probabilities).all() or (probabilities < -1e-9).any():
        raise PilotArtifactError(f"{model}/{cell}: invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise PilotArtifactError(f"{model}/{cell}: work probabilities do not sum to one")
    if y_true.tolist() != [int(work["label"]) for work in expected_works]:
        raise PilotArtifactError(f"{model}/{cell}: y_true != frozen manifest")

    works = []
    for i, row in enumerate(df.itertuples()):
        expected = expected_works[i]
        if int(row.fold) != int(expected["fold"]) or int(row.true_label) != int(expected["label"]):
            raise PilotArtifactError(f"{model}/{cell}/{expected['work_id']}: fold/truth mismatch")
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
            raise PilotArtifactError(f"{model}/{cell}: missing fold {fold} result/timing")
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


def run_pilot(
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

    for model, cell in plan:
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
            _replace_record(artifact, record)
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
            _replace_record(artifact, record)
            _save(artifact, output_path)
            _call_progress(progress, "complete", model=model, cell=cell, status=record["status"])
            continue
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
            _replace_record(artifact, record)
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
            baseline = None if cell == "A0" else _find_record(artifact, model, "A0")
            if cell != "A0" and (baseline is None or baseline.get("status") != "applied"):
                raise PilotArtifactError(f"{model}/{cell} has no completed applied A0 dependency")
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
            _replace_record(artifact, record)
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

        _replace_record(artifact, record)
        _save(artifact, output_path)
        _call_progress(
            progress,
            "complete",
            model=model,
            cell=cell,
            status="applied",
            runtime_seconds=record["timing"]["total_seconds"],
        )

    # A pure resume whose selected cells were all skipped must preserve the exact artifact bytes.
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
    "PilotArtifactError",
    "canonical_hash",
    "artifact_self_hash",
    "selected_plan",
    "run_pilot",
    "format_compact_table",
]
