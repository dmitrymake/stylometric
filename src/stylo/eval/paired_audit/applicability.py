"""The frozen confirmatory applicability matrix — 21 registered cells, 15 A0 comparisons (§2.4/§3.4).

The honest applicability matrix decomposes the six model families across the five ablation cells
(A0..A4). Only 21 of the 30 grid entries are **applied** (produce a real estimand result); the rest
carry a typed applicability signal and **never** a copied metric:

- ``not_applicable`` — the axis is meaningless for the family (bow_lr A3 relative-FW; char_cos
  A1/A3; majority A1..A4);
- ``already_in_legacy`` — the axis is already active in A0 (delta_cos:500 A1: equal-work centroids);
- ``equivalent`` — the cell equals another registered cell (char_cos A2 ≡ A4: feature state is char's
  only axis).

The 15 Δaccuracy-vs-A0 comparisons form the frozen Holm family (§3.4). A **missing or failed**
required cell invalidates the whole family; ``m`` is never reduced. This module is pure registered
data plus validation — it constructs no estimators.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from ..work_weighting import (FEATURE_STATE_ONLY_ABLATION, FULL_WB_ABLATION,
                              LEGACY_ABLATION, RELATIVE_FW_ONLY_ABLATION,
                              WEIGHTS_ONLY_ABLATION)

MODELS = ("stylo", "stylo_stack", "bow_lr", "delta_cos:500", "char_cos", "majority")
CELLS = ("A0", "A1", "A2", "A3", "A4")
STATUSES = frozenset({"applied", "not_applicable", "equivalent_to"})   # the literal §4.1 status values
AXIS_STATES = frozenset({"applied", "not_applicable", "already_in_legacy"})   # §4.1 effective_axes values
_MATRIX_DIGEST_VERSION = "paired_audit.applicability.v1"

# axes per cell bound to the single source-of-truth corner constants (drift-detecting)
_CELL_ABLATION = {
    "A0": LEGACY_ABLATION, "A1": WEIGHTS_ONLY_ABLATION, "A2": FEATURE_STATE_ONLY_ABLATION,
    "A3": RELATIVE_FW_ONLY_ABLATION, "A4": FULL_WB_ABLATION,
}

# per-model per-axis behaviour (§2.1-2.5): which models already do the axis in legacy, and for which
# it is undefined. Used to compute effective_axes per §4.1.
_AXIS_ALREADY_LEGACY = {"W": {"delta_cos:500", "char_cos"}}   # equal-work centroids already in legacy
_AXIS_NOT_APPLICABLE = {"W": {"majority"}, "F": {"majority"},
                        "R": {"bow_lr", "char_cos", "majority"}}


def _effective_axis(model: str, axis: str, requested: bool) -> str:
    """The effective §4.1 state of one work-balanced axis for one (model, cell)."""
    if not requested:
        return "not_applicable"                          # the WB change for this axis was not requested
    if model in _AXIS_ALREADY_LEGACY.get(axis, ()):
        return "already_in_legacy"
    if model in _AXIS_NOT_APPLICABLE.get(axis, ()):
        return "not_applicable"
    return "applied"


def _effective_axes(model: str, cell: str) -> dict:
    a = _CELL_ABLATION[cell]
    return {"W": _effective_axis(model, "W", a.weights),
            "F": _effective_axis(model, "F", a.feature_fit),
            "R": _effective_axis(model, "R", a.relative_fw)}


_APPLIED = ("applied", None, None)
# per (model, cell): (status, reason, equivalent_to). Top-level status is one of the three §4.1 values;
# a cell that only touches an already-in-legacy or n/a axis is top-level not_applicable (its axis
# nature is recorded in effective_axes).
_STATUS: dict[str, dict[str, tuple]] = {
    "stylo": {c: _APPLIED for c in CELLS},
    "stylo_stack": {c: _APPLIED for c in CELLS},
    "bow_lr": {
        "A0": _APPLIED, "A1": _APPLIED, "A2": _APPLIED,
        "A3": ("not_applicable", "relative-FW transform is not defined for the BoW channel (R n/a)", None),
        "A4": _APPLIED,
    },
    "delta_cos:500": {
        "A0": _APPLIED,
        "A1": ("not_applicable", "Delta W is already equal-work in legacy (effective_axes.W)", None),
        "A2": _APPLIED, "A3": _APPLIED, "A4": _APPLIED,
    },
    "char_cos": {
        "A0": _APPLIED,
        "A1": ("not_applicable", "char n-gram cosine has no function-word/loss axis (W/R n/a)", None),
        "A2": ("equivalent_to", "feature-state is char_cos's only axis; A2 equals A4", "A4"),
        "A3": ("not_applicable", "char n-gram cosine has no relative-FW axis (R n/a)", None),
        "A4": _APPLIED,
    },
    "majority": {
        "A0": _APPLIED,
        "A1": ("not_applicable", "majority has no learnable loss/feature axis", None),
        "A2": ("not_applicable", "majority has no learnable loss/feature axis", None),
        "A3": ("not_applicable", "majority has no learnable loss/feature axis", None),
        "A4": ("not_applicable", "majority has no learnable loss/feature axis", None),
    },
}

# model-specific fold-local state passports (§2.6): Delta carries an equal-work centroid/scaling state,
# the stacked classifier carries a calibration passport. These are real estimator artifacts the runner
# NEVER synthesizes — the injected estimator must supply them per fold.
_MODEL_EVIDENCE = {"stylo_stack": ("stack_calibration_digest",),
                   "delta_cos:500": ("delta_state_digest",)}

# a non-applied produced record may carry ONLY these keys (whitelist) — anything else (accuracy,
# cluster_p, proba, significant, ...) is a forbidden silent metric copy (§4.1)
_NONAPPLIED_ALLOWED_KEYS = frozenset({
    "model", "cell", "status", "reason", "equivalent_to", "requested_axes", "effective_axes",
    "claim_status",
})

# the frozen per-model decomposition (§2.4): pinned so a COUNT-preserving status swap cannot pass
_EXPECTED_APPLIED = (
    ("stylo", "A0"), ("stylo", "A1"), ("stylo", "A2"), ("stylo", "A3"), ("stylo", "A4"),
    ("stylo_stack", "A0"), ("stylo_stack", "A1"), ("stylo_stack", "A2"), ("stylo_stack", "A3"),
    ("stylo_stack", "A4"),
    ("bow_lr", "A0"), ("bow_lr", "A1"), ("bow_lr", "A2"), ("bow_lr", "A4"),
    ("delta_cos:500", "A0"), ("delta_cos:500", "A2"), ("delta_cos:500", "A3"), ("delta_cos:500", "A4"),
    ("char_cos", "A0"), ("char_cos", "A4"),
    ("majority", "A0"),
)
_EXPECTED_HOLM = (
    ("stylo", "A1"), ("stylo", "A2"), ("stylo", "A3"), ("stylo", "A4"),
    ("stylo_stack", "A1"), ("stylo_stack", "A2"), ("stylo_stack", "A3"), ("stylo_stack", "A4"),
    ("bow_lr", "A1"), ("bow_lr", "A2"), ("bow_lr", "A4"),
    ("delta_cos:500", "A2"), ("delta_cos:500", "A3"), ("delta_cos:500", "A4"),
    ("char_cos", "A4"),
)


class ApplicabilityError(ValueError):
    """Fail-closed: the applicability matrix, a produced cell record, or the Holm family is invalid."""


def required_evidence_digests(model: str, cell: str) -> tuple:
    """The fold-local evidence digests an APPLIED cell must carry (§2.6/§4.1), used identically at
    checkpoint-ingress and at cell-record validation so the two contracts cannot diverge:

    - ``proba_digest`` always (the fold's real probability vector);
    - the per-applied-axis proving digest (W → ordered_weight_digest, F → vocab_digest + idf_digest,
      R → r_denominator_trace_digest);
    - the model state passport (Delta state / stack calibration).

    A non-applied cell requires none. The runner never synthesizes these — the estimator supplies them.
    """
    reg = cell_status(model, cell)
    if reg["status"] != "applied":
        return ()
    eff = reg["effective_axes"]
    req = ["proba_digest"]
    if eff["W"] == "applied":
        req.append("ordered_weight_digest")
    if eff["F"] == "applied":
        req += ["vocab_digest", "idf_digest"]
    if eff["R"] == "applied":
        req.append("r_denominator_trace_digest")
    req += list(_MODEL_EVIDENCE.get(model, ()))
    return tuple(req)


def cell_status(model: str, cell: str) -> dict:
    """The registered status of one grid cell (with requested axes and reason/equivalent_to)."""
    if model not in _STATUS or cell not in CELLS:
        raise ApplicabilityError(f"unknown grid cell ({model!r}, {cell!r})")
    status, reason, equivalent_to = _STATUS[model][cell]
    axes = _CELL_ABLATION[cell]
    return {
        "model": model, "cell": cell, "status": status, "reason": reason,
        "equivalent_to": equivalent_to,
        "requested_axes": {"W": axes.weights, "F": axes.feature_fit, "R": axes.relative_fw},
        "effective_axes": _effective_axes(model, cell),
    }


def applicability_matrix() -> list[dict]:
    """The full 30-entry grid in canonical (model, cell) order."""
    return [cell_status(m, c) for m in MODELS for c in CELLS]


def registered_cells() -> tuple[tuple[str, str], ...]:
    """The 21 applied cells (each produces a real estimand result), in canonical order."""
    return tuple((m, c) for m in MODELS for c in CELLS if _STATUS[m][c][0] == "applied")


def holm_family() -> tuple[tuple[str, str], ...]:
    """The 15 Δaccuracy-vs-A0 comparisons (applied non-A0 cells), in canonical order."""
    return tuple((m, c) for m in MODELS for c in CELLS
                 if c != "A0" and _STATUS[m][c][0] == "applied")


def applicability_matrix_digest() -> str:
    """Deterministic sha256 over the canonical grid — bound into the RunPlan (§4.2)."""
    h = hashlib.sha256()
    h.update(len(_MATRIX_DIGEST_VERSION).to_bytes(8, "big") + _MATRIX_DIGEST_VERSION.encode("utf-8"))
    for row in applicability_matrix():
        ax, ef = row["requested_axes"], row["effective_axes"]
        parts = [row["model"], row["cell"], row["status"], row["reason"] or "",
                 row["equivalent_to"] or "", str(ax["W"]), str(ax["F"]), str(ax["R"]),
                 ef["W"], ef["F"], ef["R"]]
        for p in parts:
            b = p.encode("utf-8")
            h.update(len(b).to_bytes(8, "big") + b)
    return h.hexdigest()


def assert_matrix_invariants() -> None:
    """Fail-closed unless the registry is exactly 21 applied cells and 15 A0 comparisons, every
    status is a registered kind, and no non-applied cell claims an axis-bearing result."""
    applied = registered_cells()
    if len(applied) != 21:
        raise ApplicabilityError(f"expected exactly 21 applied cells, got {len(applied)}")
    if len(holm_family()) != 15:
        raise ApplicabilityError(f"expected exactly 15 A0 comparisons, got {len(holm_family())}")
    # pin the exact per-model decomposition (a count-preserving status swap must not pass)
    if applied != _EXPECTED_APPLIED:
        raise ApplicabilityError("applied-cell set drifted from the frozen §2.4 decomposition")
    if holm_family() != _EXPECTED_HOLM:
        raise ApplicabilityError("Holm family drifted from the frozen §3.4 decomposition")
    for row in applicability_matrix():
        if row["status"] not in STATUSES:
            raise ApplicabilityError(f"cell ({row['model']},{row['cell']}) has invalid status")
        if row["status"] == "equivalent_to" and row["equivalent_to"] not in CELLS:
            raise ApplicabilityError("an equivalent_to cell must name a valid target cell")
        if row["status"] != "equivalent_to" and row["equivalent_to"] is not None:
            raise ApplicabilityError("only an equivalent_to cell may set equivalent_to")
        # effective_axes are registered §4.1 values, and the top-level status is consistent with them
        ef = row["effective_axes"]
        if any(v not in AXIS_STATES for v in ef.values()):
            raise ApplicabilityError(f"cell ({row['model']},{row['cell']}) has an unregistered axis state")
        applied_axis = any(v == "applied" for v in ef.values())
        if row["cell"] == "A0":
            if row["status"] != "applied":
                raise ApplicabilityError("every A0 cell must be applied (the legacy baseline)")
        elif row["status"] == "applied" and not applied_axis:
            raise ApplicabilityError(f"applied non-A0 cell ({row['model']},{row['cell']}) exercises no applied axis")
        elif row["status"] == "not_applicable" and applied_axis:
            raise ApplicabilityError(f"not_applicable cell ({row['model']},{row['cell']}) exercises an applied axis")


def assert_cell_record(model: str, cell: str, record: dict, *,
                       probability_class_order=None, work_universe=None) -> None:
    """Validate a produced cell record against the registry (§4.1): the status, requested_axes and
    effective_axes must equal the registry; an applied cell must carry the full evidence schema with
    valid field types; a not_applicable/equivalent_to cell must carry NO copied metric. When
    ``probability_class_order``/``work_universe`` are given, the per-work probability width and the
    per-work universe are additionally checked (the runner supplies them)."""
    reg = cell_status(model, cell)
    if record.get("status") != reg["status"]:
        raise ApplicabilityError(
            f"({model},{cell}) record status {record.get('status')!r} != registered {reg['status']!r}")
    if record.get("requested_axes") != reg["requested_axes"]:
        raise ApplicabilityError(f"({model},{cell}) requested_axes must equal the registry")
    if record.get("effective_axes") != reg["effective_axes"]:
        raise ApplicabilityError(f"({model},{cell}) effective_axes must equal the registry")
    if reg["status"] == "applied":
        _validate_applied_record(model, cell, record, probability_class_order, work_universe)
    else:
        extra = sorted(k for k in record if k not in _NONAPPLIED_ALLOWED_KEYS)
        if extra:                                    # whitelist: any non-metadata key is a metric leak
            raise ApplicabilityError(
                f"non-applied cell ({model},{cell}) must carry no metrics; unexpected keys {extra}")
        if reg["status"] == "equivalent_to" and record.get("equivalent_to") != reg["equivalent_to"]:
            raise ApplicabilityError(f"({model},{cell}) equivalent_to must be {reg['equivalent_to']!r}")


_POINT_KEYS = ("accuracy", "macro_f1", "top2", "per_author_recall")
_VS_A0_KEYS = ("dacc", "dacc_authorclustered_ci", "cluster_p", "holm_p",
               "mcnemar_p_diagnostic", "significant")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _finite_ci(ci) -> bool:
    return (isinstance(ci, (list, tuple)) and len(ci) == 2 and _finite(ci[0]) and _finite(ci[1])
            and ci[0] <= ci[1])


def _validate_applied_record(model, cell, record, probability_class_order, work_universe) -> None:
    """Fail-closed unless an applied cell carries the full §4.1 evidence schema with valid types."""
    point = record.get("point")
    if not isinstance(point, dict) or any(k not in point for k in _POINT_KEYS):
        raise ApplicabilityError(f"applied cell ({model},{cell}) point must carry {_POINT_KEYS}")
    for k in ("accuracy", "macro_f1", "top2"):
        if not _finite(point[k]):
            raise ApplicabilityError(f"applied cell ({model},{cell}) point.{k} must be a finite number")
    if not isinstance(point.get("per_author_recall"), dict):
        raise ApplicabilityError(f"applied cell ({model},{cell}) point.per_author_recall must be a dict")
    if not _finite_ci(record.get("abs_accuracy_authorclustered_ci")):
        raise ApplicabilityError(f"applied cell ({model},{cell}) needs a finite abs_accuracy CI [lo,hi]")
    if record.get("claim_status") != "exploratory_internal":
        raise ApplicabilityError(f"applied cell ({model},{cell}) claim_status must be exploratory_internal")

    pw = record.get("per_work")
    if not isinstance(pw, list) or not pw:
        raise ApplicabilityError(f"applied cell ({model},{cell}) must carry a non-empty per_work")
    work_ids = []
    for item in pw:
        if not isinstance(item, dict) or type(item.get("work_id")) is not str:
            raise ApplicabilityError(f"applied cell ({model},{cell}) per_work item needs a str work_id")
        # §8: the vector must independently support metric recompute -> true/pred labels + correct + rank
        for key in ("true_label", "pred_label", "rank"):
            if type(item.get(key)) is not int:
                raise ApplicabilityError(f"applied cell ({model},{cell}) per_work {key} must be an int")
        if type(item.get("correct")) is not bool:
            raise ApplicabilityError(f"applied cell ({model},{cell}) per_work correct must be a bool")
        proba = item.get("proba")
        if not isinstance(proba, list) or not all(_finite(v) for v in proba):
            raise ApplicabilityError(f"applied cell ({model},{cell}) per_work proba must be finite numbers")
        if probability_class_order is not None and len(proba) != len(probability_class_order):
            raise ApplicabilityError(f"applied cell ({model},{cell}) proba width != probability_class_order")
        if not (0 <= item["pred_label"] < len(proba)) or not (0 <= item["true_label"] < len(proba)):
            raise ApplicabilityError(f"applied cell ({model},{cell}) per_work label out of proba range")
        if item["correct"] != (item["pred_label"] == item["true_label"]):
            raise ApplicabilityError(f"applied cell ({model},{cell}) per_work correct != (pred==true)")
        work_ids.append(item["work_id"])
    if len(set(work_ids)) != len(work_ids):
        raise ApplicabilityError(f"applied cell ({model},{cell}) per_work has duplicate work_id")
    if work_universe is not None and set(work_ids) != set(work_universe):
        raise ApplicabilityError(f"applied cell ({model},{cell}) per_work universe != expected work set")

    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        raise ApplicabilityError(f"applied cell ({model},{cell}) must carry fold-local evidence")
    for k, v in evidence.items():
        if k.endswith("_digest") and not (isinstance(v, str) and _HEX64_RE.match(v)):
            raise ApplicabilityError(f"applied cell ({model},{cell}) evidence.{k} must be a sha256 hex digest")
    # §2.6/§4.1: an applied axis (and the model state passport) must carry its proving digest — the
    # SAME contract the runner enforces at checkpoint-ingress (single source of truth)
    missing = [k for k in required_evidence_digests(model, cell) if k not in evidence]
    if missing:
        raise ApplicabilityError(f"applied cell ({model},{cell}) evidence missing required digests {missing}")

    if cell != "A0":
        vs = record.get("vs_A0")
        if not isinstance(vs, dict) or any(k not in vs for k in _VS_A0_KEYS):
            raise ApplicabilityError(
                f"applied non-A0 cell ({model},{cell}) vs_A0 must carry {_VS_A0_KEYS}")
        if not _finite(vs["dacc"]) or not _finite_ci(vs["dacc_authorclustered_ci"]):
            raise ApplicabilityError(f"applied non-A0 cell ({model},{cell}) vs_A0 dacc/CI must be finite")
        for k in ("cluster_p", "holm_p"):
            if not (_finite(vs[k]) and 0.0 <= vs[k] <= 1.0):
                raise ApplicabilityError(f"applied non-A0 cell ({model},{cell}) vs_A0.{k} must be a p in [0,1]")
        if type(vs["significant"]) is not bool:
            raise ApplicabilityError(f"applied non-A0 cell ({model},{cell}) vs_A0.significant must be bool")


def assert_holm_family_complete(members: Iterable[tuple[str, str]]) -> None:
    """Fail-closed unless ``members`` is EXACTLY the 15 registered comparisons — a missing or extra
    member invalidates the whole family; ``m`` is never silently reduced (§3.4)."""
    got = list(members)
    if len(got) != len(set(got)):
        raise ApplicabilityError("duplicate Holm family member")
    expected = set(holm_family())
    got_set = set(got)
    missing = sorted(expected - got_set)
    extra = sorted(got_set - expected)
    if missing or extra:
        raise ApplicabilityError(
            f"Holm family incomplete/altered (missing={missing[:3]}, extra={extra[:3]}); m is fixed at 15")
