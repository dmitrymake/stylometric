"""Claim-status and benchmark-role vocabulary shared by every result artifact.

Two orthogonal labels travel with a machine-readable result:

``ClaimStatus`` — how strong the evidence behind a number is, from an engineering
sanity check up to an independently replicated blind result. It answers "how much
should a reader trust this?".

``BenchmarkRole`` — what a benchmark package is for. A reproducible cross-validation
snapshot on labelled public data is not a blind leaderboard, and the two must never
be conflated in documentation or tooling.

Both are ``str`` enums so they serialise directly to their string value.
"""
from __future__ import annotations

from enum import Enum

__all__ = [
    "ClaimStatus",
    "BenchmarkRole",
    "parse_claim_status",
    "parse_benchmark_role",
]


class ClaimStatus(str, Enum):
    """Evidence tier attached to a quantitative claim.

    The ladder is deliberately narrow; a number may only climb it by satisfying
    the corresponding protocol stage, never by reformulation.
    """

    #: internal wiring / integration check, not evidence about authorship.
    ENGINEERING = "engineering"
    #: reproducible internal analysis (e.g. leak-free CV on labelled public data)
    #: that is neither blind nor externally replicated.
    EXPLORATORY_INTERNAL = "exploratory_internal"
    #: curated public development/gold set whose truth the model team already knows.
    DEVELOPMENT_GOLD = "development_gold"
    #: measured on a blind package whose labels the model team never saw before scoring.
    EXTERNAL_BLIND = "external_blind"
    #: an external party reproduced the primary result from a clean checkout.
    INDEPENDENTLY_REPLICATED = "independently_replicated"


class BenchmarkRole(str, Enum):
    """What a benchmark package may be used for."""

    #: dated reproducible cross-validation snapshot on labelled public data.
    #: Not a blind leaderboard; identifiers reveal truth.
    REPRODUCIBLE_CV_LEGACY_NOT_BLIND = "reproducible_cv_legacy_not_blind"
    #: labelled public development package for grouped cross-validation.
    DEVELOPMENT_CV = "development_cv"
    #: flat opaque package scored once against escrowed truth.
    BLIND_LEADERBOARD = "blind_leaderboard"


def parse_claim_status(value: str) -> ClaimStatus:
    """Return the :class:`ClaimStatus` for ``value`` or raise ``ValueError``."""
    try:
        return ClaimStatus(value)
    except ValueError as exc:
        allowed = ", ".join(status.value for status in ClaimStatus)
        raise ValueError(f"unknown claim_status {value!r}; allowed: {allowed}") from exc


def parse_benchmark_role(value: str) -> BenchmarkRole:
    """Return the :class:`BenchmarkRole` for ``value`` or raise ``ValueError``."""
    try:
        return BenchmarkRole(value)
    except ValueError as exc:
        allowed = ", ".join(role.value for role in BenchmarkRole)
        raise ValueError(f"unknown benchmark_role {value!r}; allowed: {allowed}") from exc
