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

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

WORK_BALANCED = "work_balanced"
CHUNK_WEIGHTED_LEGACY = "chunk_weighted_legacy"
SUPPORTED_TRAINING_WEIGHTINGS = frozenset({WORK_BALANCED, CHUNK_WEIGHTED_LEGACY})

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
