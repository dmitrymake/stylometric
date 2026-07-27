"""Independently derived identity receipts for real LOBO-vNext execution.

The synthetic harness accepts caller-supplied receipt payloads because its
purpose is contract testing.  A real-corpus run may not do that: every receipt
in :mod:`stylo.domain.lobo_vnext_real` is rebuilt from the live objects,
workspace bytes, installed dependencies, and numerical runtime before any
representation cache or estimator factory is touched.

This module contains no corpus policy or estimator selection.  It only turns
already verified, path-independent observations into strict receipts and
compares them byte-for-byte with the owner-bound execution spec.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import stat
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import ConfigNode
from ..domain.lobo_vnext import VNextContractError, canonical_sha256
from ..domain.lobo_vnext_real import (
    REQUIRED_RECEIPT_KINDS,
    IndependentDerivationReceipt,
    RealCorpusExecutionSpec,
)
from ..jsonio import load_strict
from ..release.source_inventory import (
    DEFAULT_MANIFEST,
    SourceInventoryError,
    check_source_inventory,
)
from .lobo_vnext_models import (
    build_r1_model_adapter_receipt,
    r1_scientific_config_sha256,
)
from .paired_audit.run_plan import (
    blas_thread_fingerprint,
    runtime_fingerprint,
    verify_installed_environment,
)


OBSERVATION_SCHEMA_VERSION = "stylo.lobo-vnext.derived-observation.v1"
EXECUTABLE_CLOSURE_DERIVATION_VERSION = (
    "stylo.lobo-vnext.executable-source-closure.v1"
)
DEPENDENCY_DERIVATION_VERSION = "stylo.lobo-vnext.dependencies.v1"
RUNTIME_DERIVATION_VERSION = "stylo.lobo-vnext.runtime.v1"
THREAD_DERIVATION_VERSION = "stylo.lobo-vnext.thread-contract.v1"
OBJECT_DERIVATION_VERSION = "stylo.lobo-vnext.canonical-object.v1"

FORBIDDEN_PRIVATE_BRANCH = "archive/local-main-private-20260726"
REQUIRED_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

_HEX64 = frozenset("0123456789abcdef")


class RealReceiptError(VNextContractError):
    """A live identity observation is malformed, drifted, or self-attested."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _HEX64 for char in value)
    ):
        raise RealReceiptError(f"{label} must be 64 lowercase hex characters")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise RealReceiptError(f"{label} must be an exact positive integer")
    return value


def _canonical_relative_path(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise RealReceiptError(f"{label} must be a canonical relative path")
    pure = pathlib.PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RealReceiptError(f"{label} must be a canonical relative path")
    return value


@dataclasses.dataclass(frozen=True)
class DerivedObservation:
    """One live, path-independent observation before receipt assembly."""

    kind: str
    derivation_version: str
    digest: str
    evidence_digest: str
    observation_count: int

    def validate(self) -> "DerivedObservation":
        if type(self) is not DerivedObservation:
            raise RealReceiptError(
                "observation must be exactly DerivedObservation"
            )
        if self.kind not in REQUIRED_RECEIPT_KINDS:
            raise RealReceiptError(f"unsupported observation kind {self.kind!r}")
        if (
            type(self.derivation_version) is not str
            or not self.derivation_version
            or "/" in self.derivation_version
        ):
            raise RealReceiptError(
                "derivation_version must be a non-empty path-free token"
            )
        _require_sha256(self.digest, "observation.digest")
        _require_sha256(
            self.evidence_digest, "observation.evidence_digest"
        )
        _require_positive_int(
            self.observation_count, "observation.observation_count"
        )
        return self


def observation_from_canonical_value(
    *,
    kind: str,
    value: object,
    observation_count: int,
    derivation_version: str = OBJECT_DERIVATION_VERSION,
) -> DerivedObservation:
    """Derive a receipt observation from an exact canonical JSON value."""

    count = _require_positive_int(observation_count, "observation_count")
    digest = canonical_sha256(value)
    return DerivedObservation(
        kind=kind,
        derivation_version=derivation_version,
        digest=digest,
        evidence_digest=digest,
        observation_count=count,
    ).validate()


def observation_from_self_hashed(
    *,
    kind: str,
    value: object,
    derivation_version: str = OBJECT_DERIVATION_VERSION,
) -> DerivedObservation:
    """Validate a domain object and observe its recorded canonical self-hash."""

    to_dict = getattr(value, "to_dict", None)
    validate = getattr(value, "validate", None)
    if not callable(to_dict) or not callable(validate):
        raise RealReceiptError(
            f"{kind} observation requires a strict self-hashed domain object"
        )
    try:
        validated = validate()
        raw = to_dict()
    except Exception as exc:
        raise RealReceiptError(
            f"cannot validate {kind} domain object: {exc}"
        ) from exc
    if validated is not value or type(raw) is not dict:
        raise RealReceiptError(f"{kind} domain projection is noncanonical")
    self_hash = _require_sha256(raw.get("self_hash"), f"{kind}.self_hash")
    payload = {key: child for key, child in raw.items() if key != "self_hash"}
    if canonical_sha256(payload) != self_hash:
        raise RealReceiptError(f"{kind} self_hash is not authoritative")
    return DerivedObservation(
        kind=kind,
        derivation_version=derivation_version,
        digest=self_hash,
        evidence_digest=canonical_sha256(raw),
        observation_count=1,
    ).validate()


def _ordered_observations(
    observations: Sequence[DerivedObservation],
) -> tuple[DerivedObservation, ...]:
    if type(observations) not in (list, tuple):
        raise RealReceiptError("observations must be an exact list or tuple")
    rows = tuple(observations)
    if any(type(row) is not DerivedObservation for row in rows):
        raise RealReceiptError(
            "observations must contain exact DerivedObservation records"
        )
    for row in rows:
        row.validate()
    if tuple(row.kind for row in rows) != REQUIRED_RECEIPT_KINDS:
        raise RealReceiptError(
            "observations must contain every required kind exactly once "
            "in canonical order"
        )
    return rows


def build_independent_receipts(
    observations: Sequence[DerivedObservation],
) -> tuple[IndependentDerivationReceipt, ...]:
    """Freeze live observations into the execution-spec receipt order."""

    rows = _ordered_observations(observations)
    return tuple(
        IndependentDerivationReceipt.build(
            kind=row.kind,
            derivation_version=row.derivation_version,
            expected_digest=row.digest,
            observed_digest=row.digest,
            evidence_digest=row.evidence_digest,
            observation_count=row.observation_count,
        )
        for row in rows
    )


def assert_independent_receipts(
    execution_spec: RealCorpusExecutionSpec,
    observations: Sequence[DerivedObservation],
) -> RealCorpusExecutionSpec:
    """Rebuild every live receipt and compare it with the execution spec."""

    if type(execution_spec) is not RealCorpusExecutionSpec:
        raise RealReceiptError(
            "execution_spec must be exactly RealCorpusExecutionSpec"
        )
    execution_spec.validate()
    expected = build_independent_receipts(observations)
    if tuple(
        receipt.to_dict() for receipt in execution_spec.independent_receipts
    ) != tuple(receipt.to_dict() for receipt in expected):
        raise RealReceiptError(
            "live independent derivation receipts differ from execution spec"
        )
    return execution_spec


def _git(root: pathlib.Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RealReceiptError(f"cannot derive Git identity: {args!r}") from exc


def derive_executable_source_observation(
    repository_root: str | os.PathLike[str],
) -> DerivedObservation:
    """Hash the clean committed executable/support closure from live bytes."""

    root = pathlib.Path(repository_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise RealReceiptError("repository root must be a real directory")
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD").strip()
    if branch == FORBIDDEN_PRIVATE_BRANCH:
        raise RealReceiptError(
            f"forbidden private archive branch is never executable: {branch}"
        )
    status_text = _git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status_text.strip():
        raise RealReceiptError(
            "real LOBO-vNext execution requires a clean scientific worktree"
        )
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    _require_sha256(commit, "git commit")

    try:
        report = check_source_inventory(root, require_git=True)
        report.require_clean()
    except SourceInventoryError as exc:
        raise RealReceiptError(
            f"executable source inventory drifted: {exc}"
        ) from exc

    manifest_path = root / DEFAULT_MANIFEST
    try:
        manifest = load_strict(manifest_path)
    except Exception as exc:
        raise RealReceiptError(
            f"cannot load executable source inventory: {exc}"
        ) from exc
    if type(manifest) is not dict:
        raise RealReceiptError(
            "executable source inventory must be an exact object"
        )
    required = manifest.get("required_release_files")
    if type(required) is not list:
        raise RealReceiptError(
            "required_release_files must be an exact array"
        )
    paths = tuple(
        sorted(
            set(report.snapshot.paths)
            | {
                _canonical_relative_path(
                    item, "required_release_files[]"
                )
                for item in required
            }
            | {DEFAULT_MANIFEST}
        )
    )
    rows: list[dict[str, object]] = []
    for relative_path in paths:
        path = root.joinpath(*pathlib.PurePosixPath(relative_path).parts)
        try:
            file_stat = path.lstat()
        except OSError as exc:
            raise RealReceiptError(
                f"executable closure file missing: {relative_path}"
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(
            file_stat.st_mode
        ):
            raise RealReceiptError(
                f"executable closure entry is unsafe: {relative_path}"
            )
        rows.append(
            {
                "relative_path": relative_path,
                "byte_size": file_stat.st_size,
                "mode": format(stat.S_IMODE(file_stat.st_mode), "04o"),
                "sha256": _sha256_file(path),
            }
        )
    evidence = {
        "schema_version": EXECUTABLE_CLOSURE_DERIVATION_VERSION,
        "git_commit": commit,
        "git_branch": branch,
        "release_path_count": report.snapshot.file_count,
        "release_path_set_sha256": report.snapshot.paths_sha256,
        "files": rows,
    }
    digest = canonical_sha256(evidence)
    return DerivedObservation(
        kind="executable_sources",
        derivation_version=EXECUTABLE_CLOSURE_DERIVATION_VERSION,
        digest=digest,
        evidence_digest=digest,
        observation_count=len(rows),
    ).validate()


def derive_dependency_observation(
    repository_root: str | os.PathLike[str],
) -> DerivedObservation:
    """Verify and bind the installed canonical dependency surface."""

    try:
        contract = verify_installed_environment(
            pathlib.Path(repository_root).resolve()
        )
    except Exception as exc:
        raise RealReceiptError(
            f"installed dependency contract drifted: {exc}"
        ) from exc
    digest = canonical_sha256(contract)
    distributions = contract.get("distributions")
    if type(distributions) is not dict or not distributions:
        raise RealReceiptError(
            "dependency contract has no bound distributions"
        )
    return DerivedObservation(
        kind="dependencies",
        derivation_version=DEPENDENCY_DERIVATION_VERSION,
        digest=digest,
        evidence_digest=digest,
        observation_count=len(distributions),
    ).validate()


def derive_runtime_observation() -> DerivedObservation:
    """Bind the path-free numerical runtime identity."""

    material = runtime_fingerprint()
    if type(material) is not dict or not material:
        raise RealReceiptError("runtime fingerprint is empty")
    digest = canonical_sha256(material)
    return DerivedObservation(
        kind="runtime",
        derivation_version=RUNTIME_DERIVATION_VERSION,
        digest=digest,
        evidence_digest=digest,
        observation_count=len(material),
    ).validate()


def derive_thread_observation() -> DerivedObservation:
    """Require the exact single-thread numerical contract and bind live pools."""

    observed_env = {
        key: os.environ.get(key, "") for key in REQUIRED_THREAD_ENV
    }
    if observed_env != REQUIRED_THREAD_ENV:
        raise RealReceiptError(
            "real LOBO-vNext requires exact thread environment "
            f"{REQUIRED_THREAD_ENV!r}, got {observed_env!r}"
        )
    material = blas_thread_fingerprint()
    if type(material) is not dict or set(material) != {
        "threadpools",
        "thread_env",
    }:
        raise RealReceiptError("thread fingerprint has a malformed shape")
    pools = material["threadpools"]
    if type(pools) is not list:
        raise RealReceiptError("threadpools must be an exact array")
    for index, pool in enumerate(pools):
        if type(pool) is not dict or type(pool.get("num_threads")) is not int:
            raise RealReceiptError(
                f"threadpool[{index}] has no exact thread count"
            )
        if pool["num_threads"] != 1:
            raise RealReceiptError(
                f"threadpool[{index}] is not pinned to one thread"
            )
    material = {
        "threadpools": sorted(
            pools,
            key=lambda row: (
                str(row.get("internal_api")),
                str(row.get("version")),
                str(row.get("threading_layer")),
                str(row.get("architecture")),
            ),
        ),
        "thread_env": observed_env,
    }
    digest = canonical_sha256(material)
    return DerivedObservation(
        kind="thread_contract",
        derivation_version=THREAD_DERIVATION_VERSION,
        digest=digest,
        evidence_digest=digest,
        observation_count=max(1, len(pools)),
    ).validate()


def derive_config_and_adapter_observations(
    *,
    cfg: ConfigNode,
    primary_model_spec: object,
    baseline_model_spec: object,
) -> tuple[DerivedObservation, DerivedObservation, DerivedObservation]:
    """Rebuild the exact owner-selected config and both model adapters."""

    if type(cfg) is not ConfigNode:
        raise RealReceiptError("cfg must be exactly ConfigNode")
    config_digest = r1_scientific_config_sha256(cfg)
    primary = build_r1_model_adapter_receipt(
        primary_model_spec, cfg=cfg
    )
    baseline = build_r1_model_adapter_receipt(
        baseline_model_spec, cfg=cfg
    )
    return (
        DerivedObservation(
            kind="config",
            derivation_version=OBJECT_DERIVATION_VERSION,
            digest=config_digest,
            evidence_digest=config_digest,
            observation_count=1,
        ).validate(),
        DerivedObservation(
            kind="primary_model_adapter",
            derivation_version=OBJECT_DERIVATION_VERSION,
            digest=_require_sha256(
                primary.get("self_hash"),
                "primary adapter receipt self_hash",
            ),
            evidence_digest=canonical_sha256(primary),
            observation_count=1,
        ).validate(),
        DerivedObservation(
            kind="baseline_model_adapter",
            derivation_version=OBJECT_DERIVATION_VERSION,
            digest=_require_sha256(
                baseline.get("self_hash"),
                "baseline adapter receipt self_hash",
            ),
            evidence_digest=canonical_sha256(baseline),
            observation_count=1,
        ).validate(),
    )


def observations_by_kind(
    observations: Sequence[DerivedObservation],
) -> Mapping[str, DerivedObservation]:
    """Return the validated canonical sequence as an immutable-style mapping."""

    rows = _ordered_observations(observations)
    return {row.kind: row for row in rows}


__all__ = [
    "DEPENDENCY_DERIVATION_VERSION",
    "DerivedObservation",
    "EXECUTABLE_CLOSURE_DERIVATION_VERSION",
    "FORBIDDEN_PRIVATE_BRANCH",
    "OBJECT_DERIVATION_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "REQUIRED_THREAD_ENV",
    "RUNTIME_DERIVATION_VERSION",
    "RealReceiptError",
    "THREAD_DERIVATION_VERSION",
    "assert_independent_receipts",
    "build_independent_receipts",
    "derive_config_and_adapter_observations",
    "derive_dependency_observation",
    "derive_executable_source_observation",
    "derive_runtime_observation",
    "derive_thread_observation",
    "observation_from_canonical_value",
    "observation_from_self_hashed",
    "observations_by_kind",
]
