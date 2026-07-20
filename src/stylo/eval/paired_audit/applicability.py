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
from typing import Iterable

from ..work_weighting import (FEATURE_STATE_ONLY_ABLATION, FULL_WB_ABLATION,
                              LEGACY_ABLATION, RELATIVE_FW_ONLY_ABLATION,
                              WEIGHTS_ONLY_ABLATION)

MODELS = ("stylo", "stylo_stack", "bow_lr", "delta_cos:500", "char_cos", "majority")
CELLS = ("A0", "A1", "A2", "A3", "A4")
STATUSES = frozenset({"applied", "not_applicable", "already_in_legacy", "equivalent"})
_MATRIX_DIGEST_VERSION = "paired_audit.applicability.v1"

# axes per cell bound to the single source-of-truth corner constants (drift-detecting)
_CELL_ABLATION = {
    "A0": LEGACY_ABLATION, "A1": WEIGHTS_ONLY_ABLATION, "A2": FEATURE_STATE_ONLY_ABLATION,
    "A3": RELATIVE_FW_ONLY_ABLATION, "A4": FULL_WB_ABLATION,
}

_APPLIED = ("applied", None, None)
# per (model, cell): (status, reason, equivalent_to)
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
        "A1": ("already_in_legacy", "Delta centroids already aggregate equal-work (W in legacy)", None),
        "A2": _APPLIED, "A3": _APPLIED, "A4": _APPLIED,
    },
    "char_cos": {
        "A0": _APPLIED,
        "A1": ("not_applicable", "char n-gram cosine has no function-word/loss axis (W/R n/a)", None),
        "A2": ("equivalent", "feature-state is char_cos's only axis; A2 equals A4", "A4"),
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
        ax = row["requested_axes"]
        parts = [row["model"], row["cell"], row["status"], row["reason"] or "",
                 row["equivalent_to"] or "", str(ax["W"]), str(ax["F"]), str(ax["R"])]
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
        if row["status"] == "equivalent" and row["equivalent_to"] not in CELLS:
            raise ApplicabilityError("an equivalent cell must name a valid target cell")
        if row["status"] != "equivalent" and row["equivalent_to"] is not None:
            raise ApplicabilityError("only an equivalent cell may set equivalent_to")


def assert_cell_record(model: str, cell: str, record: dict) -> None:
    """Validate a produced cell record against the registry: the status must match, an applied
    non-A0 cell must carry ``vs_A0``, and a not_applicable/already_in_legacy/equivalent cell must
    carry NO copied metric (§4.1)."""
    reg = cell_status(model, cell)
    if record.get("status") != reg["status"]:
        raise ApplicabilityError(
            f"({model},{cell}) record status {record.get('status')!r} != registered {reg['status']!r}")
    if reg["status"] == "applied":
        if cell != "A0" and "vs_A0" not in record:
            raise ApplicabilityError(f"applied non-A0 cell ({model},{cell}) must carry vs_A0")
    else:
        extra = sorted(k for k in record if k not in _NONAPPLIED_ALLOWED_KEYS)
        if extra:                                    # whitelist: any non-metadata key is a metric leak
            raise ApplicabilityError(
                f"non-applied cell ({model},{cell}) must carry no metrics; unexpected keys {extra}")
        if reg["status"] == "equivalent" and record.get("equivalent_to") != reg["equivalent_to"]:
            raise ApplicabilityError(f"({model},{cell}) equivalent_to must be {reg['equivalent_to']!r}")


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
