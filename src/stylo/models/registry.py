"""Authoritative model discovery and scientific-routing metadata.

Estimator construction remains in the evaluation adapter because it needs a
resolved run configuration.  All public names and capabilities live here,
however, so CLI help, exploratory defaults and the immutable confirmatory
protocol view cannot silently drift into independent registries.
"""
from __future__ import annotations

import dataclasses
import re
from types import MappingProxyType

LEGACY_WEIGHTING = "chunk_weighted_legacy"
WORK_BALANCED_WEIGHTING = "work_balanced"
_WEIGHTINGS = frozenset({LEGACY_WEIGHTING, WORK_BALANCED_WEIGHTING})
_POSITIVE_INT = re.compile(r"^[1-9][0-9]*$")


class ModelRegistryError(ValueError):
    """A model spec is unknown, malformed, or unavailable in this route."""


@dataclasses.dataclass(frozen=True)
class ModelRegistration:
    """One owned estimator family and its immutable routing capabilities."""

    key: str
    public_pattern: str
    description: str
    default_specs: tuple[str, ...] = ()
    default_order: int | None = None
    confirmatory_specs: tuple[str, ...] = ()
    confirmatory_order: int | None = None
    supported_weightings: frozenset[str] = _WEIGHTINGS
    explicit_only: bool = False
    reports_calibration: bool = False
    reference_only: bool = False
    deployment_supported: bool = True
    serialization_supported: bool = True
    lazy_final_fit: bool = False
    confirmatory_execution_eligible: bool = True
    internal_selection_evidence: bool = True
    scientific_status: str = "active"

    def matches(self, spec: str) -> bool:
        if self.public_pattern.endswith(":N"):
            prefix = self.public_pattern[:-1]
            suffix = spec[len(prefix) :] if spec.startswith(prefix) else ""
            return bool(_POSITIVE_INT.fullmatch(suffix))
        return spec == self.public_pattern


_REGISTRATIONS = (
    ModelRegistration(
        key="stylo",
        public_pattern="stylo",
        description="registered stylometric logistic-regression pipeline",
        default_specs=("stylo",),
        default_order=0,
        confirmatory_specs=("stylo",),
        confirmatory_order=0,
        reports_calibration=True,
    ),
    ModelRegistration(
        key="delta",
        public_pattern="delta:N",
        description="frozen legacy selected-mass Delta with N MFW",
        default_specs=("delta:150", "delta:300", "delta:500"),
        default_order=1,
    ),
    ModelRegistration(
        key="delta_cos",
        public_pattern="delta_cos:N",
        description="cosine Delta with N MFW",
        default_specs=("delta_cos:150", "delta_cos:300", "delta_cos:500"),
        default_order=2,
        confirmatory_specs=("delta_cos:500",),
        confirmatory_order=3,
    ),
    ModelRegistration(
        key="char_cos",
        public_pattern="char_cos",
        description="character n-gram cosine baseline",
        default_specs=("char_cos",),
        default_order=3,
        confirmatory_specs=("char_cos",),
        confirmatory_order=4,
    ),
    ModelRegistration(
        key="bow_lr",
        public_pattern="bow_lr",
        description="bag-of-words logistic-regression baseline",
        default_specs=("bow_lr",),
        default_order=4,
        confirmatory_specs=("bow_lr",),
        confirmatory_order=2,
    ),
    ModelRegistration(
        key="majority",
        public_pattern="majority",
        description="training-majority baseline",
        default_specs=("majority",),
        default_order=5,
        confirmatory_specs=("majority",),
        confirmatory_order=5,
    ),
    ModelRegistration(
        key="stylo_stack",
        public_pattern="stylo_stack",
        description="evaluation-only learned six-channel stack",
        confirmatory_specs=("stylo_stack",),
        confirmatory_order=1,
        explicit_only=True,
        reports_calibration=True,
        deployment_supported=False,
        serialization_supported=False,
        lazy_final_fit=True,
        confirmatory_execution_eligible=False,
        internal_selection_evidence=False,
        scientific_status="withdrawn_pending_nested_group_calibration",
    ),
    ModelRegistration(
        key="stylo_equal_channels_v1",
        public_pattern="stylo_equal_channels_v1",
        description="exploratory fixed equal-channel ensemble",
        explicit_only=True,
        reports_calibration=True,
        deployment_supported=False,
        serialization_supported=False,
        lazy_final_fit=True,
        scientific_status="exploratory_evaluation_only",
    ),
    ModelRegistration(
        key="bow_lr_ref_legacy",
        public_pattern="bow_lr_ref_legacy",
        description="work-balanced suite's frozen legacy reference row",
        supported_weightings=frozenset({WORK_BALANCED_WEIGHTING}),
        explicit_only=True,
        reference_only=True,
    ),
)

MODEL_REGISTRY = MappingProxyType(
    {registration.key: registration for registration in _REGISTRATIONS}
)
if len(MODEL_REGISTRY) != len(_REGISTRATIONS):  # pragma: no cover - import invariant
    raise RuntimeError("duplicate model registry key")

DEFAULT_EXPLORATORY_SPECS = tuple(
    spec
    for registration in sorted(
        (item for item in _REGISTRATIONS if item.default_order is not None),
        key=lambda item: item.default_order,
    )
    for spec in registration.default_specs
)
CONFIRMATORY_MODEL_SPECS = tuple(
    spec
    for registration in sorted(
        (item for item in _REGISTRATIONS if item.confirmatory_order is not None),
        key=lambda item: item.confirmatory_order,
    )
    for spec in registration.confirmatory_specs
)
CALIBRATION_MODEL_SPECS = frozenset(
    registration.public_pattern
    for registration in _REGISTRATIONS
    if registration.reports_calibration
)


def resolve_model_spec(spec: str) -> ModelRegistration:
    """Resolve exactly one public spec; reject bool/coercion and malformed N."""

    if type(spec) is not str or not spec:
        raise ModelRegistryError("model spec must be a non-empty exact str")
    matches = [
        registration
        for registration in _REGISTRATIONS
        if registration.matches(spec)
    ]
    if len(matches) != 1:
        raise ModelRegistryError(f"unknown or malformed model spec: {spec!r}")
    return matches[0]


def assert_model_route(
    spec: str,
    *,
    weighting: str,
    confirmatory: bool = False,
    deployment: bool = False,
    serialization: bool = False,
) -> ModelRegistration:
    """Return the registration only when the requested scientific route exists."""

    if type(weighting) is not str or weighting not in _WEIGHTINGS:
        raise ModelRegistryError(f"unknown training weighting: {weighting!r}")
    if type(confirmatory) is not bool or type(deployment) is not bool or type(serialization) is not bool:
        raise ModelRegistryError("route flags must be exact bool values")
    registration = resolve_model_spec(spec)
    if weighting not in registration.supported_weightings:
        raise ModelRegistryError(
            f"{spec!r} is unavailable for weighting={weighting!r}"
        )
    if confirmatory and spec not in registration.confirmatory_specs:
        raise ModelRegistryError(
            f"{spec!r} is not registered in the confirmatory protocol"
        )
    if confirmatory and not registration.confirmatory_execution_eligible:
        raise ModelRegistryError(
            f"{spec!r} confirmatory execution is blocked: "
            f"{registration.scientific_status}"
        )
    if deployment and not registration.deployment_supported:
        raise ModelRegistryError(
            f"{spec!r} is evaluation-only and has no deployment route"
        )
    if serialization and not registration.serialization_supported:
        raise ModelRegistryError(
            f"{spec!r} cannot be serialized under its lazy-final-fit contract"
        )
    return registration


def public_model_help() -> str:
    """Stable CLI help generated from the authoritative registrations."""

    return " | ".join(
        registration.public_pattern for registration in _REGISTRATIONS
    )


__all__ = [
    "CALIBRATION_MODEL_SPECS",
    "CONFIRMATORY_MODEL_SPECS",
    "DEFAULT_EXPLORATORY_SPECS",
    "LEGACY_WEIGHTING",
    "MODEL_REGISTRY",
    "ModelRegistration",
    "ModelRegistryError",
    "WORK_BALANCED_WEIGHTING",
    "assert_model_route",
    "public_model_help",
    "resolve_model_spec",
]
