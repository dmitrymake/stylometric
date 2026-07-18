"""Work-balanced training estimand primitives (P1 §4.2).

One work — one voice on the training side. Within an author every work carries equal
mass regardless of how many chunks it was split into; every author carries equal mass
regardless of how many works they contributed. The chunk-level equivalent weight is

    weight[i] = W / (A * num_works_for_author(a_i) * num_chunks_in_work(w_i))

where W = number of train works and A = number of authors. Then the whole train set sums
to W and each author sums to W/A, so the effective sample size equals a design with W
work-level rows under balanced classes. This absolute scale matters: sklearn's inverse
regularisation C interacts with the total sample_weight mass, so the weights are scaled to
W (not 1) to keep C comparable to a work-level fit (Codex audit, D3).

``chunk_weighted_legacy`` returns ``None`` (uniform sklearn weighting), reproducing the
pre-P1 headline. These are the two supported training weightings, mirroring the case
framework's ``centroid_weighting`` switch.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

WORK_BALANCED = "work_balanced"
CHUNK_WEIGHTED_LEGACY = "chunk_weighted_legacy"
SUPPORTED_TRAINING_WEIGHTINGS = frozenset({WORK_BALANCED, CHUNK_WEIGHTED_LEGACY})


class AblationNotImplementedError(NotImplementedError):
    """An intermediate (single-axis) ablation is requested before its estimator wiring exists.

    B4-B increment 1 introduces the AblationConfig plumbing WITHOUT changing any estimator math, so
    only the two corners (all-off == legacy, all-on == work_balanced) map to a runnable estimand.
    B4-B increment 2 additionally wires the weights-only corner A1 == ``(T,F,F)`` for the LR-family
    (stylo/bow_lr/stylo_stack). The remaining five boolean configs (WFR ``010, 001, 110, 101, 011``)
    are still not implemented and raise this until later B4-B increments.
    """


_APPLICABILITY_REASONS = frozenset({"already_in_legacy", "not_applicable"})


class AblationNotApplicableError(ValueError):
    """An audit-only ablation is requested for a model where that axis is not a new estimand.

    Distinct from :class:`AblationNotImplementedError` (a corner whose wiring simply does not exist
    yet): the requested cell is *fully implemented* but is not a meaningful new comparison for this
    model — its axis is already-in-legacy (Delta / char-centroid W == equal-work centroids), or it has
    no learnable loss axis at all (majority), or the axis simply does not exist for it (BoW has no
    relative-FW R). Rather than silently return an estimator that duplicates an existing cell — which
    would falsely mark the requested axis as *applied* in provenance — the factory fails closed with an
    exact ``reason`` the future runner records as the cell's applicability status.

    ``reason`` is the CELL-applicability status, one of exactly ``"already_in_legacy"`` (Delta family:
    the request would re-run the identical A0 estimand) or ``"not_applicable"`` (char_cos / majority /
    BoW-R / any other model where the axis is not a distinct estimand). ``requested`` carries the
    fresh exact :class:`AblationConfig` the caller asked for, so the runner can bind the cell.

    Note for the future applicability runner (schema NOT built here — that is a later increment): the
    per-axis *effective* status is a separate, spec-deterministic property and is not collapsed into
    ``reason``. Both char_cos and the Delta family use equal-work centroids, so their effective
    ``W`` is ``already_in_legacy`` at the axis level even though char_cos's cell status is
    ``not_applicable``; majority has no learnable loss axis at all. The runner derives
    ``effective_axes`` from that spec map, keeping ``reason`` a two-value contract.
    """

    def __init__(self, spec: str, reason: str, requested: "AblationConfig | None" = None):
        if reason not in _APPLICABILITY_REASONS:
            allowed = ", ".join(sorted(_APPLICABILITY_REASONS))
            raise ValueError(f"reason must be one of {{{allowed}}}, got {reason!r}")
        if requested is not None and type(requested) is not AblationConfig:
            raise TypeError(f"requested must be an AblationConfig or None, got {type(requested).__name__}")
        self.spec = spec
        self.reason = reason
        self.requested = requested                 # the exact requested cell (fresh AblationConfig)
        super().__init__(f"ablation {requested} for {spec!r} is not applicable: {reason}")

    def __reduce__(self):
        # default Exception pickling would call cls(message) and break the typed-field signature
        return (self.__class__, (self.spec, self.reason, self.requested))


class AblationEquivalentError(ValueError):
    """A requested audit cell is IDENTICAL to an already-signed corner for this model (no new
    estimand) — currently only ``char_cos`` A2, whose W is already equal-work, has no R axis, and
    whose only remaining axis F is exactly what A4 changes, so A2 ≡ A4. Routed fail-closed BEFORE
    construction/fit so the runner records ``equivalent_to`` rather than a silent A4-as-A2 duplicate
    (never a metrics copy, never an n/a)."""

    _EQUIVALENT_CORNERS = frozenset({"A4"})

    def __init__(self, spec: str, requested: "AblationConfig", equivalent_to: str):
        if equivalent_to not in self._EQUIVALENT_CORNERS:
            allowed = ", ".join(sorted(self._EQUIVALENT_CORNERS))
            raise ValueError(f"equivalent_to must be one of {{{allowed}}}, got {equivalent_to!r}")
        if type(requested) is not AblationConfig:
            raise TypeError(f"requested must be an AblationConfig, got {type(requested).__name__}")
        self.spec = spec
        self.requested = requested
        self.equivalent_to = equivalent_to
        super().__init__(f"ablation {requested} for {spec!r} is equivalent to {equivalent_to}")

    def __reduce__(self):
        return (self.__class__, (self.spec, self.requested, self.equivalent_to))


@dataclasses.dataclass(frozen=True)
class AblationConfig:
    """The three work-balanced knobs as independent booleans (design §2.2).

    ``weights`` — training loss / centroid aggregation is equal-work; ``feature_fit`` — learned
    vocabulary/DF/IDF is fit at work level; ``relative_fw`` — function-word / MFW features use the
    equal-work relative-frequency transform. Legacy == ``(F,F,F)``, full work_balanced == ``(T,T,T)``.
    """
    weights: bool
    feature_fit: bool
    relative_fw: bool

    def __post_init__(self):
        for f in ("weights", "feature_fit", "relative_fw"):
            if type(getattr(self, f)) is not bool:
                raise TypeError(f"AblationConfig.{f} must be a plain bool")

    @property
    def is_legacy_corner(self) -> bool:
        return not (self.weights or self.feature_fit or self.relative_fw)

    @property
    def is_full_wb_corner(self) -> bool:
        return self.weights and self.feature_fit and self.relative_fw

    @property
    def is_weights_only_corner(self) -> bool:
        """A1 == weights-only: the work-balanced loss/calibration protocol with a strictly legacy
        (A0) feature side. Runnable only for the LR-family via ``make_factory_for_ablation``."""
        return self.weights and not self.feature_fit and not self.relative_fw

    @property
    def is_feature_state_only_corner(self) -> bool:
        """A2 == feature-state only: work-level learned vocabulary/DF (F), with the legacy A0 training
        loss (W off) and the legacy raw FunctionWord/MFW transform (R off)."""
        return self.feature_fit and not self.weights and not self.relative_fw

    @property
    def is_relative_fw_only_corner(self) -> bool:
        """A3 == relative-FW only: the equal-work relative FunctionWord/MFW transform (R), on the
        pooled A0 vocabulary (F off) with the legacy A0 loss (W off)."""
        return self.relative_fw and not self.weights and not self.feature_fit

    def to_weighting(self) -> str:
        """Map a CORNER config to the runtime weighting enum; every non-corner raises.

        Deliberately corner-ONLY: only A0/A4 map to the two production ``training_weighting`` values.
        A1 (weights-only) is a real, runnable audit path but has NO production enum value — it must be
        built through ``make_factory_for_ablation`` with axis-aware internal state, never collapsed to
        ``WORK_BALANCED`` (which would falsely claim full work-balanced feature fitting)."""
        if self.is_legacy_corner:
            return CHUNK_WEIGHTED_LEGACY
        if self.is_full_wb_corner:
            return WORK_BALANCED
        raise AblationNotImplementedError(
            f"ablation {self} has no production weighting enum (only the two corners map; A1 is "
            "audit-only via make_factory_for_ablation)")

    @staticmethod
    def from_weighting(weighting: str) -> "AblationConfig":
        # fail-closed: an exact str only (no None / np.str_ / str subclass silently becoming legacy)
        if type(weighting) is not str:
            raise TypeError(f"weighting must be an exact str, got {type(weighting).__name__}")
        return FULL_WB_ABLATION if require_weighting(weighting) == WORK_BALANCED else LEGACY_ABLATION


LEGACY_ABLATION = AblationConfig(False, False, False)
FULL_WB_ABLATION = AblationConfig(True, True, True)
# A1 (B4-B increment 2): work-balanced loss/calibration, legacy (A0) feature state. Audit-only.
WEIGHTS_ONLY_ABLATION = AblationConfig(True, False, False)
# A2/A3 (B4-B increment 3): feature-state-only and relative-FW-only. Audit-only, legacy A0 loss.
FEATURE_STATE_ONLY_ABLATION = AblationConfig(False, True, False)
RELATIVE_FW_ONLY_ABLATION = AblationConfig(False, False, True)

# Runtime weighting label -> the public claim label used in result artifacts. The legacy
# runtime value maps to the P0 headline claim, which stays exactly as committed.
_CLAIM_LABELS = {
    CHUNK_WEIGHTED_LEGACY: "chunk_weighted_training_legacy",
    WORK_BALANCED: "work_balanced",
}

__all__ = [
    "WORK_BALANCED",
    "CHUNK_WEIGHTED_LEGACY",
    "SUPPORTED_TRAINING_WEIGHTINGS",
    "AblationConfig",
    "AblationNotImplementedError",
    "AblationNotApplicableError",
    "AblationEquivalentError",
    "LEGACY_ABLATION",
    "FULL_WB_ABLATION",
    "WEIGHTS_ONLY_ABLATION",
    "FEATURE_STATE_ONLY_ABLATION",
    "RELATIVE_FW_ONLY_ABLATION",
    "resolve_training_weighting",
    "to_claim_label",
    "work_author_map",
    "work_sample_weights",
    "training_sample_weights",
    "aggregate_by_work",
]


def resolve_training_weighting(value: str | None, *, default: str = CHUNK_WEIGHTED_LEGACY) -> str:
    """Return a validated training-weighting label (``None`` falls back to ``default``).

    The result — value OR default — is always validated against the enum, so an invalid
    default (e.g. a typo) is rejected rather than silently returned.
    """
    chosen = default if value is None else value
    if chosen not in SUPPORTED_TRAINING_WEIGHTINGS:
        allowed = ", ".join(sorted(SUPPORTED_TRAINING_WEIGHTINGS))
        raise ValueError(f"unknown training_weighting {chosen!r}; allowed: {allowed}")
    return chosen


def require_weighting(value: str) -> str:
    """Strict resolver for the lower runtime APIs: only the two enum values, NO None fallback.

    A ``None`` reaching make_factory/lobo/gkf/run_final/train would otherwise silently mean the
    legacy arm even under a work_balanced config (split-brain). The single toggle is resolved once
    at the CLI; everything below must receive an explicit, valid value.
    """
    if value not in SUPPORTED_TRAINING_WEIGHTINGS:
        allowed = ", ".join(sorted(SUPPORTED_TRAINING_WEIGHTINGS))
        raise ValueError(f"weighting must be explicit ({allowed}); got {value!r}")
    return value


def to_claim_label(training_weighting: str | None) -> str:
    """Public claim label for a runtime weighting (runtime != public claim string).

    ``chunk_weighted_legacy`` (runtime) -> ``chunk_weighted_training_legacy`` (P0 claim).
    """
    return _CLAIM_LABELS[resolve_training_weighting(training_weighting)]


def _check_equal_length(y: Sequence[Any], groups: Sequence[Any]) -> None:
    if len(y) != len(groups):
        raise ValueError(f"y and groups must have equal length ({len(y)} != {len(groups)})")


def work_author_map(y: Sequence[Any], groups: Sequence[Any]) -> dict[Any, Any]:
    """Map each work id to its single author, rejecting a work spanning two authors."""
    _check_equal_length(y, groups)
    mapping: dict[Any, Any] = {}
    for author, work in zip(y, groups, strict=True):
        seen = mapping.get(work, author)
        if seen != author:
            raise ValueError(f"work id {work!r} maps to multiple authors ({seen!r}, {author!r})")
        mapping[work] = author
    return mapping


def work_sample_weights(y: Sequence[Any], groups: Sequence[Any]) -> np.ndarray:
    """Per-chunk equal-author/equal-work weights summing to W (see module docstring)."""
    work_author = work_author_map(y, groups)
    chunks_in_work: dict[Any, int] = defaultdict(int)
    works_of_author: dict[Any, set] = defaultdict(set)
    for work, author in work_author.items():
        works_of_author[author].add(work)
    for work in groups:
        chunks_in_work[work] += 1
    n_works = len(work_author)          # W
    n_authors = len(works_of_author)    # A
    weights = np.empty(len(groups), dtype=float)
    for i, work in enumerate(groups):
        author = work_author[work]
        weights[i] = n_works / (n_authors * len(works_of_author[author]) * chunks_in_work[work])
    return weights


def training_sample_weights(
    y: Sequence[Any], groups: Sequence[Any], weighting: str
) -> np.ndarray | None:
    """Return per-chunk weights for ``weighting`` (``None`` == uniform legacy)."""
    if resolve_training_weighting(weighting) == CHUNK_WEIGHTED_LEGACY:
        return None
    return work_sample_weights(y, groups)


def aggregate_by_work(
    items: Iterable[Any], groups: Sequence[Any]
) -> list[tuple[Any, list[Any]]]:
    """Group ``items`` by work id, preserving first-seen work order.

    The building block for work-level feature fitting (§4.3): one aggregated bucket per
    work so learned vocabulary/DF/IDF and MFW are computed with equal weight per work.
    """
    items = list(items)
    _check_equal_length(items, groups)
    order: list[Any] = []
    buckets: dict[Any, list[Any]] = defaultdict(list)
    for item, work in zip(items, groups, strict=True):
        if work not in buckets:
            order.append(work)
        buckets[work].append(item)
    return [(work, buckets[work]) for work in order]
