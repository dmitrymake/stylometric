"""Frozen v3.2 applicability registry; no execution or legacy registry authority."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ...domain.work_weighting import (
    FEATURE_STATE_ONLY_ABLATION,
    FULL_WB_ABLATION,
    LEGACY_ABLATION,
    RELATIVE_FW_ONLY_ABLATION,
    WEIGHTS_ONLY_ABLATION,
)
from .corrected_v3_2 import applicability_matrix

APPLICABILITY_V3_2_DIGEST = "92c2d1cca7de759a88167ad0a0c0395ab797a6b932f3f816bcc8c37e5ed2018b"
FAMILY_ALPHA = 0.05
MODELS = ("stylo", "bow_lr", "delta_cos:500", "char_cos", "majority")
CELLS = ("A0", "A1", "A2", "A3", "A4")

_ABLATIONS = MappingProxyType({
    "A0": LEGACY_ABLATION,
    "A1": WEIGHTS_ONLY_ABLATION,
    "A2": FEATURE_STATE_ONLY_ABLATION,
    "A3": RELATIVE_FW_ONLY_ABLATION,
    "A4": FULL_WB_ABLATION,
})


class V32ApplicabilityError(ValueError):
    """A model/cell is unknown, withdrawn, or metadata-only in v3.2."""


@dataclass(frozen=True)
class V32Cell:
    model: str
    cell: str
    status: str
    reason: str
    holm_member: bool = False
    equivalent_to: str | None = None

    @property
    def applied(self) -> bool:
        return self.status == "applied"

    @property
    def requested_axes(self) -> dict[str, bool]:
        ablation = _ABLATIONS[self.cell]
        return {
            "W": ablation.weights,
            "F": ablation.feature_fit,
            "R": ablation.relative_fw,
        }

    @property
    def effective_axes(self) -> dict[str, str]:
        requested = self.requested_axes
        if self.model == "majority":
            return {axis: "not_applicable" for axis in "WFR"}
        if self.model == "char_cos":
            return {
                "W": "already_in_legacy",
                "F": "applied" if requested["F"] else "legacy",
                "R": "not_applicable",
            }
        if self.model == "delta_cos:500":
            return {
                "W": "already_in_legacy",
                "F": "applied" if requested["F"] else "legacy",
                "R": "applied" if requested["R"] else "legacy",
            }
        if self.model == "bow_lr":
            return {
                "W": "applied" if requested["W"] else "legacy",
                "F": "applied" if requested["F"] else "legacy",
                "R": "not_applicable",
            }
        return {
            axis: "applied" if requested[axis] else "legacy"
            for axis in "WFR"
        }

    @property
    def ablation(self):
        return _ABLATIONS[self.cell]


def _cell(model: str, cell: str, status: str, reason: str, *,
          holm: bool = False, equivalent_to: str | None = None) -> V32Cell:
    return V32Cell(model, cell, status, reason, holm, equivalent_to)


_GRID_ROWS = (
    # stylo
    _cell("stylo", "A0", "applied", "legacy_corner"),
    _cell("stylo", "A1", "applied", "weights_axis", holm=True),
    _cell("stylo", "A2", "applied", "feature_axis", holm=True),
    _cell("stylo", "A3", "applied", "relative_function_word_axis", holm=True),
    _cell("stylo", "A4", "applied", "full_work_balanced_corner", holm=True),
    # bow_lr
    _cell("bow_lr", "A0", "applied", "legacy_corner"),
    _cell("bow_lr", "A1", "applied", "weights_axis", holm=True),
    _cell("bow_lr", "A2", "applied", "feature_axis", holm=True),
    _cell("bow_lr", "A3", "not_applicable", "bow_has_no_relative_axis"),
    _cell("bow_lr", "A4", "applied", "full_work_balanced_corner", holm=True),
    # delta
    _cell("delta_cos:500", "A0", "applied", "legacy_corner"),
    _cell("delta_cos:500", "A1", "already_in_legacy", "equal_work_centroids_are_legacy"),
    _cell("delta_cos:500", "A2", "applied", "feature_axis", holm=True),
    _cell("delta_cos:500", "A3", "applied", "relative_axis", holm=True),
    _cell("delta_cos:500", "A4", "applied", "full_work_balanced_corner", holm=True),
    # char
    _cell("char_cos", "A0", "applied", "legacy_corner"),
    _cell("char_cos", "A1", "not_applicable", "equal_work_centroids_are_legacy"),
    _cell("char_cos", "A2", "equivalent_to", "feature_axis_equals_A4", equivalent_to="A4"),
    _cell("char_cos", "A3", "not_applicable", "char_has_no_relative_axis"),
    _cell("char_cos", "A4", "applied", "work_level_char_feature_axis", holm=True),
    # majority
    _cell("majority", "A0", "applied", "fixed_train_majority"),
    _cell("majority", "A1", "not_applicable", "no_learnable_axis"),
    _cell("majority", "A2", "not_applicable", "no_learnable_axis"),
    _cell("majority", "A3", "not_applicable", "no_learnable_axis"),
    _cell("majority", "A4", "not_applicable", "no_learnable_axis"),
)
REGISTRY_V3_2 = MappingProxyType({(row.model, row.cell): row for row in _GRID_ROWS})
APPLIED_CELLS = tuple((row.model, row.cell) for row in _GRID_ROWS if row.applied)
HOLM_FAMILY = tuple((row.model, row.cell) for row in _GRID_ROWS if row.holm_member)


def _assert_registry() -> None:
    corrected = applicability_matrix()
    corrected_applied = tuple((row["model"], row["cell"]) for row in corrected["applied_cells"])
    corrected_holm = tuple((row["model"], row["cell"]) for row in corrected["holm_family"])
    if (len(REGISTRY_V3_2), len(APPLIED_CELLS), len(HOLM_FAMILY)) != (25, 16, 11):
        raise RuntimeError("v3.2 applicability registry is not exact 25/16/11")
    if corrected_applied != APPLIED_CELLS or corrected_holm != HOLM_FAMILY:
        raise RuntimeError("v3.2 applicability registry differs from corrected preparation")
    if corrected["digest"] != APPLICABILITY_V3_2_DIGEST or corrected["family_alpha"] != FAMILY_ALPHA:
        raise RuntimeError("v3.2 applicability identity drift")


def resolve_cell_v3_2(model: str, cell: str, *, require_applied: bool = True) -> V32Cell:
    if model == "stylo_stack":
        raise V32ApplicabilityError("stylo_stack is withdrawn from v3.2 before factory/fit")
    if type(model) is not str or type(cell) is not str:
        raise V32ApplicabilityError("model/cell must be exact strings")
    try:
        row = REGISTRY_V3_2[(model, cell)]
    except KeyError as exc:
        raise V32ApplicabilityError(f"unknown v3.2 model/cell: {model!r}/{cell!r}") from exc
    if require_applied and not row.applied:
        suffix = f"; equivalent_to={row.equivalent_to}" if row.equivalent_to else ""
        raise V32ApplicabilityError(
            f"v3.2 cell {model}/{cell} is metadata-only: {row.status}/{row.reason}{suffix}"
        )
    return row


_assert_registry()

__all__ = [
    "APPLICABILITY_V3_2_DIGEST", "APPLIED_CELLS", "CELLS", "FAMILY_ALPHA",
    "HOLM_FAMILY", "MODELS", "REGISTRY_V3_2", "V32ApplicabilityError", "V32Cell",
    "resolve_cell_v3_2",
]
