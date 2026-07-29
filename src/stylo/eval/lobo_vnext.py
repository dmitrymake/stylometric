"""Fail-closed, exploratory-only LOBO-vNext implementation harness.

This module is intentionally separate from the historical A0/A1/A4 runner.  It
provides a small execution surface for synthetic contract fixtures and refuses
real-corpus execution.  The useful output of the harness is the contract:

* every corpus/content/fold/model/inference input is versioned and self-hashed;
* all preflight checks finish before representation preparation, factory
  construction, or fitting;
* fold checkpoints are immutable scientific records with exact schemas;
* telemetry is kept outside the scientific artifact and its hash; and
* final validation replays the prediction and metric derivations.

It is not confirmatory evidence and cannot authorize a corpus, model family, or
inference plan on behalf of the project owner.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import os
import pathlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn

import numpy as np

from ..domain.lobo_vnext_packet import VNextTextRow
from ..domain.prediction_contract import (
    PREDICTION_CONTRACT_VERSION,
    PredictionContractError,
    stable_top1_and_worst_tie_rank,
    validate_prediction_record,
    validate_probability_matrix,
)
from ..jsonio import dumps_strict, load_strict
from . import _lobo_vnext_shared as _shared


EXECUTION_SPEC_SCHEMA_VERSION = "stylo.lobo-vnext.execution-spec.v1"
IDENTITY_RECEIPT_SCHEMA_VERSION = "stylo.lobo-vnext.identity-receipt.v1"
REPRESENTATION_RECEIPT_SCHEMA_VERSION = (
    "stylo.lobo-vnext.representation-receipt.v1"
)
RUN_IDENTITY_SCHEMA_VERSION = "stylo.lobo-vnext.run-identity.v1"
CHECKPOINT_SCHEMA_VERSION = "stylo.lobo-vnext.checkpoint.v1"
FINAL_ARTIFACT_SCHEMA_VERSION = "stylo.lobo-vnext.final-artifact.v1"
TELEMETRY_SCHEMA_VERSION = "stylo.lobo-vnext.telemetry.v1"
LEGACY_PROJECTION_SCHEMA_VERSION = (
    "stylo.lobo-vnext.legacy-read-only-projection.v1"
)

EXPLORATORY_STATUS = "exploratory_dry_run_only"
SYNTHETIC_EXECUTION_MODE = "synthetic_fixture"
REAL_EXECUTION_MODE = "real_corpus"
EXPLORATORY_AUTHORIZATION = "approved_for_exploratory"
LOBO_STRATEGY = "lobo"
SYNTHETIC_UNIFORM_ESTIMATOR = "synthetic_uniform_probe_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT_RE = re.compile(r"^[0-9A-Za-z_.-]+$")
_RECEIPT_KINDS = (
    "config",
    "executable_sources",
    "dependencies",
    "runtime",
    "thread_contract",
)


class LoboVNextError(ValueError):
    """The vNext dry-run contract is malformed or internally inconsistent."""


class VNextPreflightError(LoboVNextError):
    """Execution was rejected before representation, factory, or fit."""


class VNextCheckpointError(LoboVNextError):
    """A checkpoint namespace is corrupt, conflicting, or noncanonical."""


class VNextArtifactError(LoboVNextError):
    """A final artifact is malformed or has invalid derived values."""


@dataclasses.dataclass(frozen=True)
class VNextPreflight:
    corpus_manifest: Any
    content_manifest: Any
    fold_manifest: Any
    inner_cv_plan: Any
    model_spec: Any
    inference_spec: Any
    execution_spec: dict[str, Any]
    rows: tuple[VNextTextRow, ...]
    run_identity: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.run_identity["run_id"]


@dataclasses.dataclass(frozen=True)
class VNextRunOutcome:
    artifact: dict[str, Any]
    artifact_path: pathlib.Path
    telemetry: dict[str, Any]
    computed_folds: int
    resumed_folds: int

    @property
    def run_id(self) -> str:
        return self.artifact["run_identity"]["run_id"]


def _strict_json_tree(
    value: Any,
    *,
    path: str = "value",
    error_type: type[LoboVNextError] = LoboVNextError,
) -> None:
    _shared._strict_json_tree(
        value,
        path=path,
        error_type=error_type,
    )


def _canonical_bytes(
    value: Any,
    *,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> bytes:
    return _shared._canonical_bytes(value, error_type=error_type)


def _canonical_hash(
    value: Any,
    *,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> str:
    return _shared._canonical_hash(value, error_type=error_type)


def _self_hash(
    value: Mapping[str, Any],
    *,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> str:
    return _shared._self_hash(value, error_type=error_type)


def _require_exact_dict(
    value: Any,
    keys: set[str] | frozenset[str],
    *,
    path: str,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> dict[str, Any]:
    return _shared._require_exact_dict(
        value,
        keys,
        path=path,
        error_type=error_type,
    )


def _require_list(
    value: Any,
    *,
    path: str,
    nonempty: bool = False,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> list[Any]:
    return _shared._require_list(
        value,
        path=path,
        nonempty=nonempty,
        error_type=error_type,
    )


def _require_str(
    value: Any,
    *,
    path: str,
    nonempty: bool = True,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> str:
    return _shared._require_str(
        value,
        path=path,
        nonempty=nonempty,
        error_type=error_type,
    )


def _require_sha256(
    value: Any,
    *,
    path: str,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> str:
    return _shared._require_sha256(
        value,
        path=path,
        error_type=error_type,
    )


def _require_bool(
    value: Any,
    *,
    path: str,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> bool:
    return _shared._require_bool(
        value,
        path=path,
        error_type=error_type,
    )


def _require_int(
    value: Any,
    *,
    path: str,
    minimum: int | None = None,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> int:
    return _shared._require_int(
        value,
        path=path,
        minimum=minimum,
        error_type=error_type,
    )


def _require_float(
    value: Any,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> float:
    return _shared._require_float(
        value,
        path=path,
        minimum=minimum,
        maximum=maximum,
        error_type=error_type,
    )


def _require_string_array(
    value: Any,
    *,
    path: str,
    nonempty: bool = False,
    unique: bool = False,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> tuple[str, ...]:
    return _shared._require_string_array(
        value,
        path=path,
        nonempty=nonempty,
        unique=unique,
        error_type=error_type,
    )


def _require_self_hash(
    value: dict[str, Any],
    *,
    path: str,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> None:
    _shared._require_self_hash(
        value,
        path=path,
        error_type=error_type,
    )


def _reject_absolute_paths(
    value: Any,
    *,
    path: str,
    error_type: type[LoboVNextError] = VNextPreflightError,
) -> None:
    _shared._reject_absolute_paths(
        value,
        path=path,
        error_type=error_type,
    )


def build_identity_receipt(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build one explicit, run-bound source/config/runtime receipt."""

    if kind not in _RECEIPT_KINDS:
        raise LoboVNextError(f"unsupported identity receipt kind {kind!r}")
    if type(payload) is not dict:
        raise LoboVNextError("identity receipt payload must be an exact object")
    _strict_json_tree(payload, path=f"{kind}.payload")
    receipt = {
        "schema_version": IDENTITY_RECEIPT_SCHEMA_VERSION,
        "kind": kind,
        "payload": payload,
    }
    receipt["self_hash"] = _self_hash(receipt)
    validate_identity_receipt(receipt, expected_kind=kind)
    return receipt


def validate_identity_receipt(
    receipt: Any,
    *,
    expected_kind: str,
) -> dict[str, Any]:
    _require_exact_dict(
        receipt,
        {"schema_version", "kind", "payload", "self_hash"},
        path=f"identity_receipts.{expected_kind}",
        error_type=VNextPreflightError,
    )
    if receipt["schema_version"] != IDENTITY_RECEIPT_SCHEMA_VERSION:
        raise VNextPreflightError(
            f"identity_receipts.{expected_kind} schema mismatch"
        )
    if receipt["kind"] != expected_kind:
        raise VNextPreflightError(
            f"identity receipt kind mismatch for {expected_kind}"
        )
    if type(receipt["payload"]) is not dict:
        raise VNextPreflightError(
            f"identity_receipts.{expected_kind}.payload must be an exact object"
        )
    try:
        _strict_json_tree(
            receipt["payload"],
            path=f"identity_receipts.{expected_kind}.payload",
        )
    except LoboVNextError as exc:
        raise VNextPreflightError(str(exc)) from exc
    _require_self_hash(
        receipt,
        path=f"identity_receipts.{expected_kind}",
        error_type=VNextPreflightError,
    )
    payload = receipt["payload"]
    if payload.get("verified") is not True:
        raise VNextPreflightError(
            f"identity_receipts.{expected_kind} is not verified"
        )
    if expected_kind in {"config", "executable_sources", "runtime"}:
        if payload.get("drift_free") is not True:
            raise VNextPreflightError(
                f"identity_receipts.{expected_kind} is drifted"
            )
    if (
        expected_kind == "executable_sources"
        and payload.get("worktree_clean") is not True
    ):
        raise VNextPreflightError(
            "identity_receipts.executable_sources requires a clean worktree"
        )
    if (
        expected_kind == "thread_contract"
        and payload.get("deterministic") is not True
    ):
        raise VNextPreflightError(
            "identity_receipts.thread_contract must be deterministic"
        )
    _reject_absolute_paths(
        payload, path=f"identity_receipts.{expected_kind}.payload"
    )
    return receipt


def _raw_entry_dict(entry: Any) -> dict[str, Any]:
    return {
        "relative_path": entry.relative_path,
        "byte_size": entry.byte_size,
        "sha256": entry.sha256,
    }


def _manifest_dict(value: Any, *, path: str) -> dict[str, Any]:
    try:
        validated = value.validate()
    except Exception as exc:
        raise VNextPreflightError(f"{path} validation failed: {exc}") from exc
    if validated is not value:
        raise VNextPreflightError(f"{path}.validate() must return itself")
    try:
        raw = value.to_dict()
    except Exception as exc:
        raise VNextPreflightError(f"{path}.to_dict() failed: {exc}") from exc
    if type(raw) is not dict:
        raise VNextPreflightError(f"{path}.to_dict() must return an exact object")
    try:
        _strict_json_tree(raw, path=path)
    except LoboVNextError as exc:
        raise VNextPreflightError(str(exc)) from exc
    if raw.get("self_hash") != getattr(value, "self_hash", None):
        raise VNextPreflightError(f"{path} self-hash projection mismatch")
    return raw


def build_representation_receipt(
    corpus_manifest: Any,
    *,
    representation_policy_version: str,
) -> dict[str, Any]:
    """Bind deterministic per-text rows without reading or caching representations."""

    corpus_raw = _manifest_dict(corpus_manifest, path="corpus_manifest")
    _require_str(
        representation_policy_version,
        path="representation_policy_version",
    )
    inventory_by_path = {
        entry.relative_path: entry for entry in corpus_manifest.raw_inventory
    }
    rows: list[dict[str, Any]] = []
    for work in corpus_manifest.works:
        for ordinal, relative_path in enumerate(work.raw_paths):
            if relative_path not in inventory_by_path:
                raise VNextPreflightError(
                    f"work {work.work_id!r} raw path is absent from corpus inventory"
                )
            entry = inventory_by_path[relative_path]
            rows.append(
                {
                    "row_id": f"{work.work_id}#{ordinal:06d}",
                    "relative_path": relative_path,
                    "work_id": work.work_id,
                    "author_id": work.author_id,
                    "raw_sha256": entry.sha256,
                }
            )
    rows.sort(key=lambda item: (item["work_id"], item["relative_path"], item["row_id"]))
    if not rows:
        raise VNextPreflightError("representation receipt would contain no rows")
    row_inventory_sha256 = _canonical_hash(rows)
    receipt = {
        "schema_version": REPRESENTATION_RECEIPT_SCHEMA_VERSION,
        "corpus_manifest_sha256": corpus_raw["self_hash"],
        "canonical_model_row_digest": corpus_manifest.canonical_model_row_digest,
        "representation_policy_version": representation_policy_version,
        "deterministic_per_text_only": True,
        "rows": rows,
        "row_inventory_sha256": row_inventory_sha256,
    }
    receipt["self_hash"] = _self_hash(receipt)
    validate_representation_receipt(receipt, corpus_manifest)
    return receipt


def validate_representation_receipt(
    receipt: Any,
    corpus_manifest: Any,
) -> dict[str, Any]:
    _require_exact_dict(
        receipt,
        {
            "schema_version",
            "corpus_manifest_sha256",
            "canonical_model_row_digest",
            "representation_policy_version",
            "deterministic_per_text_only",
            "rows",
            "row_inventory_sha256",
            "self_hash",
        },
        path="representation_receipt",
        error_type=VNextPreflightError,
    )
    if receipt["schema_version"] != REPRESENTATION_RECEIPT_SCHEMA_VERSION:
        raise VNextPreflightError("representation receipt schema mismatch")
    corpus_raw = _manifest_dict(corpus_manifest, path="corpus_manifest")
    if receipt["corpus_manifest_sha256"] != corpus_raw["self_hash"]:
        raise VNextPreflightError(
            "representation receipt belongs to a different corpus manifest"
        )
    if (
        receipt["canonical_model_row_digest"]
        != corpus_manifest.canonical_model_row_digest
    ):
        raise VNextPreflightError(
            "representation receipt canonical model-row digest mismatch"
        )
    _require_str(
        receipt["representation_policy_version"],
        path="representation_receipt.representation_policy_version",
        error_type=VNextPreflightError,
    )
    if receipt["deterministic_per_text_only"] is not True:
        raise VNextPreflightError(
            "representation cache may contain only deterministic per-text state"
        )
    rows = _require_list(
        receipt["rows"],
        path="representation_receipt.rows",
        nonempty=True,
        error_type=VNextPreflightError,
    )
    for index, row in enumerate(rows):
        _require_exact_dict(
            row,
            {"row_id", "relative_path", "work_id", "author_id", "raw_sha256"},
            path=f"representation_receipt.rows[{index}]",
            error_type=VNextPreflightError,
        )
        for key in ("row_id", "relative_path", "work_id", "author_id"):
            _require_str(
                row[key],
                path=f"representation_receipt.rows[{index}].{key}",
                error_type=VNextPreflightError,
            )
        _require_sha256(
            row["raw_sha256"],
            path=f"representation_receipt.rows[{index}].raw_sha256",
            error_type=VNextPreflightError,
        )
    if rows != sorted(
        rows, key=lambda item: (item["work_id"], item["relative_path"], item["row_id"])
    ):
        raise VNextPreflightError(
            "representation receipt rows must be canonically sorted"
        )
    if len({row["row_id"] for row in rows}) != len(rows):
        raise VNextPreflightError("representation receipt has duplicate row ids")
    _require_sha256(
        receipt["row_inventory_sha256"],
        path="representation_receipt.row_inventory_sha256",
        error_type=VNextPreflightError,
    )
    if receipt["row_inventory_sha256"] != _canonical_hash(rows):
        raise VNextPreflightError(
            "representation receipt row inventory digest mismatch"
        )
    # Avoid recursive rebuilding: compare directly with the manifest projection.
    inventory_by_path = {
        entry.relative_path: entry for entry in corpus_manifest.raw_inventory
    }
    works = {work.work_id: work for work in corpus_manifest.works}
    expected_rows: list[dict[str, Any]] = []
    for work in corpus_manifest.works:
        for ordinal, relative_path in enumerate(work.raw_paths):
            entry = inventory_by_path.get(relative_path)
            if entry is None:
                raise VNextPreflightError(
                    f"work {work.work_id!r} path absent from raw inventory"
                )
            expected_rows.append(
                {
                    "row_id": f"{work.work_id}#{ordinal:06d}",
                    "relative_path": relative_path,
                    "work_id": work.work_id,
                    "author_id": work.author_id,
                    "raw_sha256": entry.sha256,
                }
            )
    expected_rows.sort(
        key=lambda item: (item["work_id"], item["relative_path"], item["row_id"])
    )
    if rows != expected_rows:
        raise VNextPreflightError(
            "representation receipt rows differ from corpus work identities"
        )
    if set(works) != {row["work_id"] for row in rows}:
        raise VNextPreflightError(
            "representation receipt does not cover every included work"
        )
    _require_self_hash(
        receipt,
        path="representation_receipt",
        error_type=VNextPreflightError,
    )
    return receipt


def build_execution_spec(
    *,
    estimator_key: str,
    identity_receipts: Mapping[str, dict[str, Any]],
    representation_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Build an explicit synthetic-only execution authorization."""

    _require_str(estimator_key, path="estimator_key")
    if type(identity_receipts) is not dict:
        identity_receipts = dict(identity_receipts)
    spec = {
        "schema_version": EXECUTION_SPEC_SCHEMA_VERSION,
        "execution_mode": SYNTHETIC_EXECUTION_MODE,
        "authorization": EXPLORATORY_AUTHORIZATION,
        "evaluation_strategy": LOBO_STRATEGY,
        "estimator_key": estimator_key,
        "identity_receipts": dict(identity_receipts),
        "representation_receipt": representation_receipt,
    }
    spec["self_hash"] = _self_hash(spec)
    return spec


def validate_execution_spec(
    spec: Any,
    *,
    corpus_manifest: Any,
    model_spec: Any,
    inference_spec: Any,
) -> dict[str, Any]:
    _require_exact_dict(
        spec,
        {
            "schema_version",
            "execution_mode",
            "authorization",
            "evaluation_strategy",
            "estimator_key",
            "identity_receipts",
            "representation_receipt",
            "self_hash",
        },
        path="execution_spec",
        error_type=VNextPreflightError,
    )
    if spec["schema_version"] != EXECUTION_SPEC_SCHEMA_VERSION:
        raise VNextPreflightError(
            "unversioned, legacy, or unsupported execution spec"
        )
    if spec["execution_mode"] == REAL_EXECUTION_MODE:
        raise VNextPreflightError(
            "real-corpus execution is not authorized by the dry-run harness"
        )
    if spec["execution_mode"] != SYNTHETIC_EXECUTION_MODE:
        raise VNextPreflightError("unsupported execution mode")
    if spec["authorization"] != EXPLORATORY_AUTHORIZATION:
        raise VNextPreflightError(
            "execution spec is not approved for exploratory use"
        )
    if spec["evaluation_strategy"] != LOBO_STRATEGY:
        if str(spec["evaluation_strategy"]).lower() in {
            "gkf",
            "groupkfold",
            "group_k_fold",
        }:
            raise VNextPreflightError("GKF is never accepted as LOBO")
        raise VNextPreflightError("evaluation strategy must be exact LOBO")
    _require_str(
        spec["estimator_key"],
        path="execution_spec.estimator_key",
        error_type=VNextPreflightError,
    )
    receipts = _require_exact_dict(
        spec["identity_receipts"],
        set(_RECEIPT_KINDS),
        path="execution_spec.identity_receipts",
        error_type=VNextPreflightError,
    )
    for kind in _RECEIPT_KINDS:
        validate_identity_receipt(receipts[kind], expected_kind=kind)
    validate_representation_receipt(
        spec["representation_receipt"], corpus_manifest
    )
    _require_self_hash(
        spec, path="execution_spec", error_type=VNextPreflightError
    )
    for name, value in (
        ("model_spec.approved_for_exploratory", model_spec.approved_for_exploratory),
        (
            "inference_spec.approved_for_exploratory",
            inference_spec.approved_for_exploratory,
        ),
    ):
        if value is not True:
            raise VNextPreflightError(f"{name} must be true")
    # Synthetic fixtures must never forge an owner decision.
    for name, value in (
        ("model_spec.owner_selected", model_spec.owner_selected),
        ("inference_spec.owner_selected", inference_spec.owner_selected),
    ):
        if value is not False:
            raise VNextPreflightError(
                f"{name} must be false for a synthetic fixture"
            )
    return spec


def load_execution_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Strict JSON ingress; semantic validation happens with the bound specs."""

    try:
        raw = load_strict(path)
    except Exception as exc:
        raise VNextPreflightError(f"cannot load execution spec: {exc}") from exc
    if type(raw) is not dict:
        raise VNextPreflightError("execution spec must be an exact JSON object")
    return raw


def _safe_relative_path(value: Any, *, path: str) -> pathlib.PurePosixPath:
    text = _require_str(value, path=path, error_type=VNextPreflightError)
    pure = pathlib.PurePosixPath(text)
    if (
        pure.is_absolute()
        or text != pure.as_posix()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise VNextPreflightError(f"{path} must be a canonical safe relative path")
    return pure


def _resolve_corpus_file(
    corpus_root: pathlib.Path,
    relative_path: str,
) -> pathlib.Path:
    pure = _safe_relative_path(
        relative_path, path=f"raw file {relative_path!r}"
    )
    if corpus_root.is_symlink() or not corpus_root.is_dir():
        raise VNextPreflightError(
            "corpus root must be an existing real directory"
        )
    current = corpus_root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise VNextPreflightError(
                f"symlinked corpus path component rejected: {relative_path}"
            )
    try:
        root_real = corpus_root.resolve(strict=True)
        file_real = current.resolve(strict=True)
    except OSError as exc:
        raise VNextPreflightError(
            f"missing raw corpus file {relative_path!r}: {exc}"
        ) from exc
    if not file_real.is_relative_to(root_real) or not current.is_file():
        raise VNextPreflightError(
            f"raw corpus path escapes root or is not a file: {relative_path}"
        )
    return current


def _load_bound_rows(
    corpus_root: pathlib.Path,
    corpus_manifest: Any,
    representation_receipt: dict[str, Any],
) -> tuple[VNextTextRow, ...]:
    """Read only receipt-bound UTF-8 rows after the whole preflight has passed."""

    inventory = {
        entry.relative_path: entry for entry in corpus_manifest.raw_inventory
    }
    rows: list[VNextTextRow] = []
    for raw in representation_receipt["rows"]:
        relative_path = raw["relative_path"]
        entry = inventory[relative_path]
        path = _resolve_corpus_file(corpus_root, relative_path)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise VNextPreflightError(
                f"cannot read raw corpus file {relative_path!r}: {exc}"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        if (
            len(payload) != entry.byte_size
            or digest != entry.sha256
            or digest != raw["raw_sha256"]
        ):
            raise VNextPreflightError(
                f"raw corpus bytes drifted for {relative_path!r}"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VNextPreflightError(
                f"raw model row is not UTF-8: {relative_path!r}"
            ) from exc
        if not text:
            raise VNextPreflightError(
                f"raw model row is empty: {relative_path!r}"
            )
        rows.append(
            VNextTextRow(
                row_id=raw["row_id"],
                relative_path=relative_path,
                work_id=raw["work_id"],
                author_id=raw["author_id"],
                text=text,
                raw_sha256=digest,
            )
        )
    return tuple(rows)


def _component_work_map(content_manifest: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for component in content_manifest.components:
        for work_id in component.work_ids:
            if work_id in mapping:
                raise VNextPreflightError(
                    f"work {work_id!r} belongs to multiple content components"
                )
            mapping[work_id] = component.component_id
    return mapping


def _validate_outer_folds(
    corpus_manifest: Any,
    content_manifest: Any,
    fold_manifest: Any,
) -> None:
    works = {work.work_id: work for work in corpus_manifest.works}
    work_ids = tuple(sorted(works))
    components = _component_work_map(content_manifest)
    if set(components) != set(work_ids):
        raise VNextPreflightError(
            "content components must assign every included work exactly once"
        )
    folds = tuple(fold_manifest.folds)
    if not folds:
        raise VNextPreflightError("fold manifest contains no folds")

    probability_order = tuple(folds[0].probability_class_order)
    metric_order = tuple(folds[0].metric_label_order)
    if probability_order != tuple(corpus_manifest.author_ids):
        raise VNextPreflightError(
            "fold probability class order differs from corpus author order"
        )
    if (
        not metric_order
        or len(set(metric_order)) != len(metric_order)
        or any(author not in probability_order for author in metric_order)
        or tuple(author for author in probability_order if author in metric_order)
        != metric_order
    ):
        raise VNextPreflightError(
            "metric label order must be a nonempty P-ordered subset"
        )
    expected_metric_order = tuple(
        author
        for author in probability_order
        if any(
            works[fold.test_work_id].author_id == author
            for fold in folds
        )
    )
    if metric_order != expected_metric_order:
        raise VNextPreflightError(
            "metric label order must be frozen from tested folds"
        )
    expected_test_works = tuple(
        work_id
        for work_id in work_ids
        if works[work_id].author_id in set(metric_order)
    )
    if tuple(fold.test_work_id for fold in folds) != expected_test_works:
        raise VNextPreflightError(
            "fold manifest must test every M-author whole work exactly once"
        )

    component_to_works: dict[str, set[str]] = {}
    for work_id, component_id in components.items():
        component_to_works.setdefault(component_id, set()).add(work_id)

    mode = fold_manifest.mode
    if mode not in {"isolated", "purged"}:
        raise VNextPreflightError("fold mode must be isolated or purged")
    for index, fold in enumerate(folds):
        if fold.probability_class_order != probability_order:
            raise VNextPreflightError(
                f"fold {fold.fold_id} probability class order drifted"
            )
        if fold.metric_label_order != metric_order:
            raise VNextPreflightError(
                f"fold {fold.fold_id} metric label order drifted"
            )
        test_work = fold.test_work_id
        expected_component = components[test_work]
        if fold.content_component_id != expected_component:
            raise VNextPreflightError(
                f"fold {fold.fold_id} content component mismatch"
            )
        component_peers = component_to_works[expected_component] - {test_work}
        if mode == "isolated":
            # Isolated execution is only valid after proving singleton components.
            if component_peers:
                raise VNextPreflightError(
                    "isolated mode requires one included work per content component"
                )
            expected_purged: tuple[str, ...] = ()
            expected_train = tuple(item for item in work_ids if item != test_work)
        else:
            expected_purged = tuple(sorted(component_peers))
            expected_train = tuple(
                item
                for item in work_ids
                if item != test_work and item not in component_peers
            )
        if tuple(fold.purged_work_ids) != expected_purged:
            raise VNextPreflightError(
                f"fold {fold.fold_id} purge inventory is not exact"
            )
        if tuple(fold.train_work_ids) != expected_train:
            raise VNextPreflightError(
                f"fold {fold.fold_id} train inventory is not exact"
            )
        if (
            test_work in fold.train_work_ids
            or test_work in fold.purged_work_ids
            or set(fold.train_work_ids) & set(fold.purged_work_ids)
        ):
            raise VNextPreflightError(
                f"fold {fold.fold_id} split inventories overlap"
            )
        train_authors = {works[work_id].author_id for work_id in expected_train}
        missing_support = [
            author for author in probability_order if author not in train_authors
        ]
        if missing_support:
            raise VNextPreflightError(
                f"fold {fold.fold_id} loses probability-class train support: "
                f"{missing_support}"
            )
        if works[test_work].author_id not in metric_order:
            raise VNextPreflightError(
                f"fold {fold.fold_id} truth is outside fixed metric labels"
            )
def _validate_inner_split(
    split: Any,
    *,
    outer_train: tuple[str, ...],
    component_by_work: Mapping[str, str],
    path: str,
) -> None:
    """Prove component-aware inner locality; never infer a work-only fallback."""

    train = tuple(split.train_work_ids)
    validation = tuple(split.validation_work_ids)
    validation_components = tuple(split.validation_component_ids)
    outer = set(outer_train)
    if not train or not validation:
        raise VNextPreflightError(f"{path} has empty train/validation inventory")
    if (
        not set(train).issubset(outer)
        or not set(validation).issubset(outer)
        or set(train) & set(validation)
    ):
        raise VNextPreflightError(f"{path} inventories are not an exact partition")
    if set(train) | set(validation) != outer:
        raise VNextPreflightError(f"{path} does not cover the outer train universe")
    derived_validation_components = {
        component_by_work[work_id] for work_id in validation
    }
    if (
        not validation_components
        or len(set(validation_components)) != len(validation_components)
        or set(validation_components) != derived_validation_components
    ):
        raise VNextPreflightError(
            f"{path} validation component receipt is not exact"
        )
    leaked = [
        work_id
        for work_id in train
        if component_by_work[work_id] in derived_validation_components
    ]
    if leaked:
        raise VNextPreflightError(
            f"{path} silently falls back to work-only grouping"
        )
    expected_validation = tuple(
        sorted(
            work_id
            for work_id in outer
            if component_by_work[work_id] in derived_validation_components
        )
    )
    if validation != expected_validation:
        raise VNextPreflightError(
            f"{path} must hold out whole content components"
        )


def _inner_cv_plan_dict(value: Any) -> dict[str, Any]:
    from ..domain.lobo_vnext import InnerCVPlan

    try:
        raw = value.to_dict()
        rebuilt = InnerCVPlan.from_dict(raw)
    except Exception as exc:
        raise VNextPreflightError(
            f"inner_cv_plan validation failed: {exc}"
        ) from exc
    if rebuilt != value or type(raw) is not dict:
        raise VNextPreflightError(
            "inner_cv_plan is noncanonical or has a projection mismatch"
        )
    try:
        _strict_json_tree(raw, path="inner_cv_plan")
    except LoboVNextError as exc:
        raise VNextPreflightError(str(exc)) from exc
    return raw


def _validate_inner_cv_plan(
    inner_cv_plan: Any,
    fold_manifest: Any,
    corpus_manifest: Any,
    content_manifest: Any,
    model_spec: Any,
) -> None:
    _inner_cv_plan_dict(inner_cv_plan)
    try:
        inner_cv_plan.validate_against(
            fold_manifest,
            corpus_manifest,
            content_manifest,
            model_spec,
        )
    except Exception as exc:
        raise VNextPreflightError(
            f"inner CV plan binding/feasibility failed: {exc}"
        ) from exc
    plans = inner_cv_plan.by_fold
    expected_fold_ids = tuple(fold.fold_id for fold in fold_manifest.folds)
    if tuple(plan.fold_id for plan in inner_cv_plan.plans) != expected_fold_ids:
        raise VNextPreflightError(
            "inner CV plan order differs from outer fold order"
        )
    component_by_work = _component_work_map(content_manifest)
    for fold in fold_manifest.folds:
        plan = plans[fold.fold_id]
        if plan.fold_spec_digest != fold.self_hash:
            raise VNextPreflightError(
                f"inner plan {fold.fold_id} outer fold digest mismatch"
            )
        expected_count = (
            model_spec.inner_cv_splits
            if model_spec.requires_inner_cv
            else 0
        )
        if len(plan.splits) != expected_count:
            raise VNextPreflightError(
                f"inner plan {fold.fold_id} split count mismatch"
            )
        for index, split in enumerate(plan.splits):
            _validate_inner_split(
                split,
                outer_train=tuple(fold.train_work_ids),
                component_by_work=component_by_work,
                path=f"inner_cv_plan.{fold.fold_id}.splits[{index}]",
            )


def _raw_inventory_digest(corpus_manifest: Any) -> str:
    return _canonical_hash(
        [_raw_entry_dict(entry) for entry in corpus_manifest.raw_inventory]
    )


def _build_run_identity(
    corpus_manifest: Any,
    content_manifest: Any,
    fold_manifest: Any,
    inner_cv_plan: Any,
    model_spec: Any,
    inference_spec: Any,
    execution_spec: dict[str, Any],
) -> dict[str, Any]:
    corpus_raw = _manifest_dict(corpus_manifest, path="corpus_manifest")
    content_raw = _manifest_dict(content_manifest, path="content_manifest")
    fold_raw = _manifest_dict(fold_manifest, path="fold_manifest")
    inner_raw = _inner_cv_plan_dict(inner_cv_plan)
    model_raw = _manifest_dict(model_spec, path="model_spec")
    inference_raw = _manifest_dict(inference_spec, path="inference_spec")
    first_fold = fold_manifest.folds[0]
    receipts = execution_spec["identity_receipts"]
    material = {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "status": EXPLORATORY_STATUS,
        "confirmatory_authorized": False,
        "execution_mode": SYNTHETIC_EXECUTION_MODE,
        "evaluation_strategy": LOBO_STRATEGY,
        "corpus": {
            "schema_version": corpus_manifest.schema_version,
            "generation_id": corpus_manifest.generation_id,
            "manifest_sha256": corpus_raw["self_hash"],
            "raw_inventory_sha256": _raw_inventory_digest(corpus_manifest),
            "canonical_model_row_digest": (
                corpus_manifest.canonical_model_row_digest
            ),
            "chunker_policy_version": corpus_manifest.chunker_policy_version,
            "canonicalizer_policy_version": (
                corpus_manifest.canonicalizer_policy_version
            ),
            "content_policy_version": corpus_manifest.content_policy_version,
        },
        "content_manifest_sha256": content_raw["self_hash"],
        "fold_manifest_sha256": fold_raw["self_hash"],
        "inner_cv_plan_sha256": inner_raw["self_hash"],
        "fold_spec_sha256": [fold.self_hash for fold in fold_manifest.folds],
        "probability_class_order": list(first_fold.probability_class_order),
        "metric_label_order": list(first_fold.metric_label_order),
        "model_spec_sha256": model_raw["self_hash"],
        "inference_spec_sha256": inference_raw["self_hash"],
        "execution_spec_sha256": execution_spec["self_hash"],
        "identity_receipt_sha256": {
            kind: receipts[kind]["self_hash"] for kind in _RECEIPT_KINDS
        },
        "representation_receipt_sha256": execution_spec[
            "representation_receipt"
        ]["self_hash"],
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "final_artifact_schema_version": FINAL_ARTIFACT_SCHEMA_VERSION,
    }
    run_id = _canonical_hash(material)
    identity = {**material, "run_id": run_id}
    identity["self_hash"] = _self_hash(identity)
    validate_run_identity(identity)
    return identity


def validate_run_identity(identity: Any) -> dict[str, Any]:
    top_keys = {
        "schema_version",
        "status",
        "confirmatory_authorized",
        "execution_mode",
        "evaluation_strategy",
        "corpus",
        "content_manifest_sha256",
        "fold_manifest_sha256",
        "inner_cv_plan_sha256",
        "fold_spec_sha256",
        "probability_class_order",
        "metric_label_order",
        "model_spec_sha256",
        "inference_spec_sha256",
        "execution_spec_sha256",
        "identity_receipt_sha256",
        "representation_receipt_sha256",
        "prediction_contract_version",
        "checkpoint_schema_version",
        "final_artifact_schema_version",
        "run_id",
        "self_hash",
    }
    _require_exact_dict(
        identity,
        top_keys,
        path="run_identity",
        error_type=VNextArtifactError,
    )
    if (
        identity["schema_version"] != RUN_IDENTITY_SCHEMA_VERSION
        or identity["status"] != EXPLORATORY_STATUS
        or identity["confirmatory_authorized"] is not False
        or identity["execution_mode"] != SYNTHETIC_EXECUTION_MODE
        or identity["evaluation_strategy"] != LOBO_STRATEGY
        or identity["prediction_contract_version"]
        != PREDICTION_CONTRACT_VERSION
        or identity["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or identity["final_artifact_schema_version"]
        != FINAL_ARTIFACT_SCHEMA_VERSION
    ):
        raise VNextArtifactError("run identity contract/version mismatch")
    corpus = _require_exact_dict(
        identity["corpus"],
        {
            "schema_version",
            "generation_id",
            "manifest_sha256",
            "raw_inventory_sha256",
            "canonical_model_row_digest",
            "chunker_policy_version",
            "canonicalizer_policy_version",
            "content_policy_version",
        },
        path="run_identity.corpus",
        error_type=VNextArtifactError,
    )
    for key in (
        "schema_version",
        "generation_id",
        "chunker_policy_version",
        "canonicalizer_policy_version",
        "content_policy_version",
    ):
        _require_str(
            corpus[key],
            path=f"run_identity.corpus.{key}",
            error_type=VNextArtifactError,
        )
    for key in (
        "manifest_sha256",
        "raw_inventory_sha256",
        "canonical_model_row_digest",
    ):
        _require_sha256(
            corpus[key],
            path=f"run_identity.corpus.{key}",
            error_type=VNextArtifactError,
        )
    for key in (
        "content_manifest_sha256",
        "fold_manifest_sha256",
        "inner_cv_plan_sha256",
        "model_spec_sha256",
        "inference_spec_sha256",
        "execution_spec_sha256",
        "representation_receipt_sha256",
        "run_id",
        "self_hash",
    ):
        _require_sha256(
            identity[key],
            path=f"run_identity.{key}",
            error_type=VNextArtifactError,
        )
    fold_hashes = _require_list(
        identity["fold_spec_sha256"],
        path="run_identity.fold_spec_sha256",
        nonempty=True,
        error_type=VNextArtifactError,
    )
    for index, digest in enumerate(fold_hashes):
        _require_sha256(
            digest,
            path=f"run_identity.fold_spec_sha256[{index}]",
            error_type=VNextArtifactError,
        )
    probability_order = _require_string_array(
        identity["probability_class_order"],
        path="run_identity.probability_class_order",
        nonempty=True,
        unique=True,
        error_type=VNextArtifactError,
    )
    metric_order = _require_string_array(
        identity["metric_label_order"],
        path="run_identity.metric_label_order",
        nonempty=True,
        unique=True,
        error_type=VNextArtifactError,
    )
    if tuple(item for item in probability_order if item in metric_order) != metric_order:
        raise VNextArtifactError(
            "run_identity.metric_label_order is not a P-ordered subset"
        )
    receipt_hashes = _require_exact_dict(
        identity["identity_receipt_sha256"],
        set(_RECEIPT_KINDS),
        path="run_identity.identity_receipt_sha256",
        error_type=VNextArtifactError,
    )
    for kind in _RECEIPT_KINDS:
        _require_sha256(
            receipt_hashes[kind],
            path=f"run_identity.identity_receipt_sha256.{kind}",
            error_type=VNextArtifactError,
        )
    _require_self_hash(
        identity, path="run_identity", error_type=VNextArtifactError
    )
    material = {
        key: item
        for key, item in identity.items()
        if key not in {"run_id", "self_hash"}
    }
    if identity["run_id"] != _canonical_hash(material):
        raise VNextArtifactError("run identity run_id mismatch")
    _reject_absolute_paths(identity, path="run_identity")
    return identity


def _split_record(fold: Any) -> dict[str, Any]:
    return {
        "fold_id": fold.fold_id,
        "test_work_id": fold.test_work_id,
        "content_component_id": fold.content_component_id,
        "train_work_ids": list(fold.train_work_ids),
        "purged_work_ids": list(fold.purged_work_ids),
        "probability_class_order": list(fold.probability_class_order),
        "metric_label_order": list(fold.metric_label_order),
    }


_CHECKPOINT_KEYS = {
    "schema_version",
    "status",
    "confirmatory_authorized",
    "run_id",
    "model_spec_sha256",
    "inner_cv_plan_sha256",
    "inner_fold_plan_sha256",
    "fold_index",
    "fold_spec_sha256",
    "split",
    "result",
    "self_hash",
}
_CHECKPOINT_SPLIT_KEYS = {
    "fold_id",
    "test_work_id",
    "content_component_id",
    "train_work_ids",
    "purged_work_ids",
    "probability_class_order",
    "metric_label_order",
}
_CHECKPOINT_RESULT_KEYS = {
    "work_id",
    "true_author_id",
    "true_label",
    "predicted_author_id",
    "predicted_label",
    "correct",
    "true_rank",
    "probabilities",
    "chunk_count",
}


def build_vnext_checkpoint(
    *,
    identity: dict[str, Any],
    model_spec: Any,
    inner_cv_plan: Any,
    fold_index: int,
    fold: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": EXPLORATORY_STATUS,
        "confirmatory_authorized": False,
        "run_id": identity["run_id"],
        "model_spec_sha256": model_spec.self_hash,
        "inner_cv_plan_sha256": inner_cv_plan.self_hash,
        "inner_fold_plan_sha256": inner_cv_plan.by_fold[
            fold.fold_id
        ].self_hash,
        "fold_index": fold_index,
        "fold_spec_sha256": fold.self_hash,
        "split": _split_record(fold),
        "result": result,
    }
    checkpoint["self_hash"] = _self_hash(checkpoint)
    validate_vnext_checkpoint(
        checkpoint,
        identity=identity,
        model_spec=model_spec,
        inner_cv_plan=inner_cv_plan,
        fold_index=fold_index,
        fold=fold,
    )
    return checkpoint


def validate_vnext_checkpoint(
    checkpoint: Any,
    *,
    identity: dict[str, Any],
    model_spec: Any,
    inner_cv_plan: Any,
    fold_index: int,
    fold: Any,
) -> dict[str, Any]:
    _require_exact_dict(
        checkpoint,
        _CHECKPOINT_KEYS,
        path="checkpoint",
        error_type=VNextCheckpointError,
    )
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise VNextCheckpointError(
            "legacy or unsupported checkpoint is read-only and not resumable"
        )
    if (
        checkpoint["status"] != EXPLORATORY_STATUS
        or checkpoint["confirmatory_authorized"] is not False
    ):
        raise VNextCheckpointError("checkpoint authority/status mismatch")
    _require_sha256(
        checkpoint["run_id"],
        path="checkpoint.run_id",
        error_type=VNextCheckpointError,
    )
    _require_sha256(
        checkpoint["model_spec_sha256"],
        path="checkpoint.model_spec_sha256",
        error_type=VNextCheckpointError,
    )
    _require_sha256(
        checkpoint["inner_cv_plan_sha256"],
        path="checkpoint.inner_cv_plan_sha256",
        error_type=VNextCheckpointError,
    )
    _require_sha256(
        checkpoint["inner_fold_plan_sha256"],
        path="checkpoint.inner_fold_plan_sha256",
        error_type=VNextCheckpointError,
    )
    inner_fold_plan = inner_cv_plan.by_fold.get(fold.fold_id)
    if inner_fold_plan is None:
        raise VNextCheckpointError("checkpoint fold has no inner-plan receipt")
    _require_sha256(
        checkpoint["fold_spec_sha256"],
        path="checkpoint.fold_spec_sha256",
        error_type=VNextCheckpointError,
    )
    if (
        checkpoint["run_id"] != identity["run_id"]
        or checkpoint["model_spec_sha256"] != model_spec.self_hash
        or checkpoint["inner_cv_plan_sha256"] != inner_cv_plan.self_hash
        or checkpoint["inner_fold_plan_sha256"] != inner_fold_plan.self_hash
        or checkpoint["fold_spec_sha256"] != fold.self_hash
    ):
        raise VNextCheckpointError("checkpoint identity binding mismatch")
    _require_int(
        checkpoint["fold_index"],
        path="checkpoint.fold_index",
        minimum=0,
        error_type=VNextCheckpointError,
    )
    if checkpoint["fold_index"] != fold_index:
        raise VNextCheckpointError("checkpoint fold index mismatch")
    split = _require_exact_dict(
        checkpoint["split"],
        _CHECKPOINT_SPLIT_KEYS,
        path="checkpoint.split",
        error_type=VNextCheckpointError,
    )
    expected_split = _split_record(fold)
    _require_exact_structure(
        split,
        expected_split,
        path="checkpoint.split",
        error_type=VNextCheckpointError,
    )
    result = _require_exact_dict(
        checkpoint["result"],
        _CHECKPOINT_RESULT_KEYS,
        path="checkpoint.result",
        error_type=VNextCheckpointError,
    )
    for key in ("work_id", "true_author_id", "predicted_author_id"):
        _require_str(
            result[key],
            path=f"checkpoint.result.{key}",
            error_type=VNextCheckpointError,
        )
    for key in ("true_label", "predicted_label"):
        _require_int(
            result[key],
            path=f"checkpoint.result.{key}",
            minimum=0,
            error_type=VNextCheckpointError,
        )
    _require_bool(
        result["correct"],
        path="checkpoint.result.correct",
        error_type=VNextCheckpointError,
    )
    _require_int(
        result["true_rank"],
        path="checkpoint.result.true_rank",
        minimum=1,
        error_type=VNextCheckpointError,
    )
    _require_int(
        result["chunk_count"],
        path="checkpoint.result.chunk_count",
        minimum=1,
        error_type=VNextCheckpointError,
    )
    probabilities = _require_list(
        result["probabilities"],
        path="checkpoint.result.probabilities",
        nonempty=True,
        error_type=VNextCheckpointError,
    )
    if len(probabilities) != len(fold.probability_class_order):
        raise VNextCheckpointError("checkpoint probability width mismatch")
    for index, probability in enumerate(probabilities):
        _require_float(
            probability,
            path=f"checkpoint.result.probabilities[{index}]",
            minimum=0.0,
            maximum=1.0,
            error_type=VNextCheckpointError,
        )
    if result["work_id"] != fold.test_work_id:
        raise VNextCheckpointError("checkpoint result work id mismatch")
    probability_order = tuple(fold.probability_class_order)
    if not 0 <= result["true_label"] < len(probability_order):
        raise VNextCheckpointError("checkpoint true label is outside P")
    if not 0 <= result["predicted_label"] < len(probability_order):
        raise VNextCheckpointError("checkpoint prediction label is outside P")
    if (
        probability_order[result["true_label"]] != result["true_author_id"]
        or probability_order[result["predicted_label"]]
        != result["predicted_author_id"]
    ):
        raise VNextCheckpointError("checkpoint author/label mapping mismatch")
    try:
        validate_prediction_record(
            probabilities=probabilities,
            pred_label=result["predicted_label"],
            true_label=result["true_label"],
            correct=result["correct"],
            rank=result["true_rank"],
            expected_width=len(probability_order),
        )
    except PredictionContractError as exc:
        raise VNextCheckpointError(
            f"checkpoint prediction contract failed: {exc}"
        ) from exc
    _require_self_hash(
        checkpoint, path="checkpoint", error_type=VNextCheckpointError
    )
    return checkpoint


def _require_exact_structure(
    observed: Any,
    expected: Any,
    *,
    path: str,
    error_type: type[LoboVNextError],
) -> None:
    _shared._require_exact_structure(
        observed,
        expected,
        path=path,
        error_type=error_type,
    )


def project_legacy_artifact_read_only(payload: Any) -> dict[str, Any]:
    """Expose only provenance metadata; never make a legacy object resumable."""

    if type(payload) is not dict:
        raise LoboVNextError("legacy payload must be an exact object")
    schema = payload.get("schema_version", payload.get("schema"))
    if type(schema) is not str or not schema:
        raise LoboVNextError("legacy payload has no descriptive schema")
    if schema in {
        CHECKPOINT_SCHEMA_VERSION,
        FINAL_ARTIFACT_SCHEMA_VERSION,
        RUN_IDENTITY_SCHEMA_VERSION,
    }:
        raise LoboVNextError("current vNext payload is not a legacy projection")
    digest = _canonical_hash(payload)
    return {
        "schema_version": LEGACY_PROJECTION_SCHEMA_VERSION,
        "legacy_schema": schema,
        "legacy_payload_sha256": digest,
        "resumable": False,
        "scientific_evidence": False,
    }


def reject_legacy_checkpoint_resume(payload: Any) -> NoReturn:
    projection = project_legacy_artifact_read_only(payload)
    raise VNextCheckpointError(
        "legacy checkpoint is read-only and cannot enter a vNext writer "
        f"({projection['legacy_schema']})"
    )


def _safe_component(value: str, *, path: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or _SAFE_COMPONENT_RE.fullmatch(value) is None
    ):
        raise VNextCheckpointError(f"{path} is not a safe path component")
    return value


def _canonical_file_bytes(
    value: dict[str, Any],
    *,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> bytes:
    return _shared._canonical_file_bytes(
        value,
        error_type=error_type,
    )


def _durable_create(
    path: pathlib.Path,
    payload: dict[str, Any],
    *,
    error_type: type[LoboVNextError] = LoboVNextError,
) -> bool:
    return _shared._durable_create(
        path,
        payload,
        error_type=error_type,
    )


def _load_json_exact(
    path: pathlib.Path,
    *,
    error_type: type[LoboVNextError],
) -> dict[str, Any]:
    return _shared._load_json_exact(path, error_type=error_type)


class _VNextCheckpointStore:
    """One immutable namespace for exactly one run/model/fold manifest."""

    def __init__(
        self,
        output_namespace: pathlib.Path,
        preflight: VNextPreflight,
    ) -> None:
        self.output_namespace = output_namespace
        self.preflight = preflight
        self.identity = preflight.run_identity
        self.run_root = output_namespace / preflight.run_id
        self.model_component = _safe_component(
            preflight.model_spec.self_hash, path="model spec hash"
        )
        self.checkpoint_dir = self.run_root / "checkpoints" / self.model_component
        self.identity_path = self.run_root / "run-identity.json"
        self.final_path = self.run_root / "scientific-result.json"

    def inspect_existing(self) -> dict[int, dict[str, Any]]:
        """Validate an existing namespace without creating or changing it.

        This read-only pass deliberately precedes representation preparation so
        a corrupt, extra, or conflicting checkpoint/final artifact cannot cause
        cache work, factory construction, or fit.
        """

        _guard_output_namespace(self.output_namespace)
        if self.run_root.is_symlink():
            raise VNextCheckpointError(
                f"unsafe checkpoint directory: {self.run_root}"
            )
        if not self.run_root.exists():
            return {}
        if not self.run_root.is_dir():
            raise VNextCheckpointError(
                f"unsafe checkpoint directory: {self.run_root}"
            )
        existing_identity = _load_json_exact(
            self.identity_path, error_type=VNextCheckpointError
        )
        try:
            validate_run_identity(existing_identity)
        except LoboVNextError as exc:
            raise VNextCheckpointError(
                f"existing run identity is invalid: {exc}"
            ) from exc
        if existing_identity != self.identity:
            raise VNextCheckpointError(
                "checkpoint namespace belongs to a conflicting run identity"
            )
        self._guard_tree()
        found = self.scan()
        if self.final_path.is_symlink():
            raise VNextArtifactError(
                f"symlinked final artifact rejected: {self.final_path}"
            )
        if self.final_path.exists():
            existing_final = _load_json_exact(
                self.final_path, error_type=VNextArtifactError
            )
            validate_vnext_final_artifact(
                existing_final, preflight=self.preflight
            )
            expected_indices = set(
                range(len(self.preflight.fold_manifest.folds))
            )
            if set(found) != expected_indices:
                raise VNextArtifactError(
                    "existing final artifact conflicts with an incomplete "
                    "checkpoint namespace"
                )
            if existing_final["checkpoints"] != [
                found[index] for index in range(len(found))
            ]:
                raise VNextArtifactError(
                    "existing final artifact conflicts with disk checkpoints"
                )
        return found

    def initialize(self) -> None:
        _guard_output_namespace(self.output_namespace)
        self.run_root.mkdir(parents=True, exist_ok=True)
        if not _durable_create(
            self.identity_path,
            self.identity,
            error_type=VNextCheckpointError,
        ):
            existing = _load_json_exact(
                self.identity_path, error_type=VNextCheckpointError
            )
            try:
                validate_run_identity(existing)
            except LoboVNextError as exc:
                raise VNextCheckpointError(
                    f"existing run identity is invalid: {exc}"
                ) from exc
            if existing != self.identity:
                raise VNextCheckpointError(
                    "checkpoint namespace belongs to a conflicting run identity"
                )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._guard_tree()

    def _guard_tree(self) -> None:
        for path in (
            self.output_namespace,
            self.run_root,
            self.run_root / "checkpoints",
            self.checkpoint_dir,
        ):
            if path.is_symlink() or not path.is_dir():
                raise VNextCheckpointError(
                    f"unsafe checkpoint directory: {path}"
                )
        allowed_run = {"run-identity.json", "checkpoints", "scientific-result.json"}
        extras = {
            entry.name
            for entry in os.scandir(self.run_root)
            if entry.name not in allowed_run
        }
        if extras:
            raise VNextCheckpointError(
                f"extra/conflicting run namespace entries: {sorted(extras)}"
            )
        checkpoint_root = self.run_root / "checkpoints"
        for entry in os.scandir(checkpoint_root):
            if entry.name != self.model_component:
                raise VNextCheckpointError(
                    f"extra/conflicting model checkpoint directory: {entry.name}"
                )

    def filename(self, fold_index: int, fold: Any) -> str:
        return f"{fold_index:06d}-{fold.self_hash[:16]}.json"

    def path(self, fold_index: int, fold: Any) -> pathlib.Path:
        return self.checkpoint_dir / self.filename(fold_index, fold)

    def scan(self) -> dict[int, dict[str, Any]]:
        expected_names = {
            self.filename(index, fold): (index, fold)
            for index, fold in enumerate(self.preflight.fold_manifest.folds)
        }
        found: dict[int, dict[str, Any]] = {}
        for entry in sorted(os.scandir(self.checkpoint_dir), key=lambda item: item.name):
            if entry.name not in expected_names:
                raise VNextCheckpointError(
                    f"extra/conflicting checkpoint file: {entry.path}"
                )
            index, fold = expected_names[entry.name]
            checkpoint = _load_json_exact(
                pathlib.Path(entry.path), error_type=VNextCheckpointError
            )
            validate_vnext_checkpoint(
                checkpoint,
                identity=self.identity,
                model_spec=self.preflight.model_spec,
                inner_cv_plan=self.preflight.inner_cv_plan,
                fold_index=index,
                fold=fold,
            )
            found[index] = checkpoint
        return found

    def save(
        self,
        checkpoint: dict[str, Any],
        *,
        fold_index: int,
        fold: Any,
    ) -> pathlib.Path:
        validate_vnext_checkpoint(
            checkpoint,
            identity=self.identity,
            model_spec=self.preflight.model_spec,
            inner_cv_plan=self.preflight.inner_cv_plan,
            fold_index=fold_index,
            fold=fold,
        )
        path = self.path(fold_index, fold)
        if not _durable_create(
            path,
            checkpoint,
            error_type=VNextCheckpointError,
        ):
            existing = _load_json_exact(
                path, error_type=VNextCheckpointError
            )
            validate_vnext_checkpoint(
                existing,
                identity=self.identity,
                model_spec=self.preflight.model_spec,
                inner_cv_plan=self.preflight.inner_cv_plan,
                fold_index=fold_index,
                fold=fold,
            )
            if existing != checkpoint:
                raise VNextCheckpointError(
                    f"conflicting immutable checkpoint already exists: {path}"
                )
        return path

    def publish_final(self, artifact: dict[str, Any]) -> pathlib.Path:
        if not _durable_create(
            self.final_path,
            artifact,
            error_type=VNextArtifactError,
        ):
            existing = _load_json_exact(
                self.final_path, error_type=VNextArtifactError
            )
            validate_vnext_final_artifact(
                existing, preflight=self.preflight
            )
            if existing != artifact:
                raise VNextArtifactError(
                    "conflicting immutable final artifact already exists"
                )
        return self.final_path


def _guard_output_namespace(path: pathlib.Path) -> None:
    _shared._guard_output_namespace(
        path,
        error_type=VNextPreflightError,
    )


def _require_vnext_domain_types(
    corpus_manifest: Any,
    content_manifest: Any,
    fold_manifest: Any,
    inner_cv_plan: Any,
    model_spec: Any,
    inference_spec: Any,
) -> None:
    from ..domain.lobo_vnext import (
        ContentComponentManifest,
        CorpusVNextManifest,
        FoldManifest,
        InferenceSpec,
        InnerCVPlan,
        ModelSpec,
    )

    expected = (
        ("corpus_manifest", corpus_manifest, CorpusVNextManifest),
        ("content_manifest", content_manifest, ContentComponentManifest),
        ("fold_manifest", fold_manifest, FoldManifest),
        ("inner_cv_plan", inner_cv_plan, InnerCVPlan),
        ("model_spec", model_spec, ModelSpec),
        ("inference_spec", inference_spec, InferenceSpec),
    )
    for name, value, expected_type in expected:
        if type(value) is not expected_type:
            raise VNextPreflightError(
                f"{name} must be exactly {expected_type.__name__}; "
                "legacy/duck-typed inputs are rejected"
            )


def _validate_model_and_inference(
    model_spec: Any,
    inference_spec: Any,
    execution_spec: dict[str, Any],
) -> None:
    if execution_spec["estimator_key"] != model_spec.model_id:
        raise VNextPreflightError(
            "execution estimator_key must equal the frozen ModelSpec model_id"
        )
    if model_spec.model_id != SYNTHETIC_UNIFORM_ESTIMATOR:
        raise VNextPreflightError(
            "the dry-run harness accepts only the explicit synthetic probe "
            "ModelSpec; historical A0/A1/A4 and real model families are blocked"
        )
    if (
        type(model_spec.requires_inner_cv) is not bool
        or type(model_spec.supports_component_aware_inner_cv) is not bool
        or (
            model_spec.inner_cv_splits is not None
            and type(model_spec.inner_cv_splits) is not int
        )
    ):
        raise VNextPreflightError("ModelSpec inner-CV fields are malformed")
    if model_spec.requires_inner_cv:
        if not model_spec.supports_component_aware_inner_cv:
            raise VNextPreflightError(
                f"ModelSpec {model_spec.model_id!r} is unsupported: "
                "component-aware inner CV is required"
            )
        if (
            type(model_spec.inner_cv_splits) is not int
            or model_spec.inner_cv_splits < 2
        ):
            raise VNextPreflightError(
                "inner-CV ModelSpec requires at least two frozen splits"
            )
    else:
        if model_spec.inner_cv_splits is not None:
            raise VNextPreflightError(
                "non-inner-CV ModelSpec must freeze inner_cv_splits=null"
            )
    if inference_spec.primary_metric != "book_accuracy":
        raise VNextPreflightError(
            "vNext primary metric must be book_accuracy"
        )
    if (
        inference_spec.primary_uncertainty
        != "author_clustered_percentile_bootstrap"
    ):
        raise VNextPreflightError(
            "vNext primary uncertainty must be author-clustered"
        )
    if inference_spec.macro_f1_uncertainty != "point_only":
        raise VNextPreflightError(
            "macro-F1 uncertainty remains point-only"
        )
    required_secondary = ("macro_f1", "top2", "per_author")
    if inference_spec.secondary_metrics != required_secondary:
        raise VNextPreflightError(
            "InferenceSpec secondary metrics are not the frozen vNext order"
        )


def preflight_lobo_vnext(
    *,
    corpus_root: str | os.PathLike[str],
    corpus_manifest: Any,
    content_manifest: Any,
    fold_manifest: Any,
    inner_cv_plan: Any,
    model_spec: Any,
    inference_spec: Any,
    execution_spec: dict[str, Any],
) -> VNextPreflight:
    """Complete every disk/split/authority check before any factory or fit."""

    _require_vnext_domain_types(
        corpus_manifest,
        content_manifest,
        fold_manifest,
        inner_cv_plan,
        model_spec,
        inference_spec,
    )
    corpus_raw = _manifest_dict(corpus_manifest, path="corpus_manifest")
    content_raw = _manifest_dict(content_manifest, path="content_manifest")
    _manifest_dict(fold_manifest, path="fold_manifest")
    _inner_cv_plan_dict(inner_cv_plan)
    _manifest_dict(model_spec, path="model_spec")
    _manifest_dict(inference_spec, path="inference_spec")
    if corpus_manifest.corpus_kind != SYNTHETIC_EXECUTION_MODE:
        raise VNextPreflightError(
            "only a corpus declared synthetic_fixture may enter this harness"
        )
    if (
        corpus_manifest.approved_for_exploratory is not True
        or corpus_manifest.owner_selected is not False
    ):
        raise VNextPreflightError(
            "synthetic corpus requires exploratory approval and owner_selected=false"
        )
    try:
        corpus_manifest.assert_exploratory_authorized(synthetic_fixture=True)
        model_spec.assert_exploratory_authorized(synthetic_fixture=True)
        inference_spec.assert_exploratory_authorized(synthetic_fixture=True)
    except Exception as exc:
        raise VNextPreflightError(
            f"exploratory authorization failed: {exc}"
        ) from exc
    if (
        corpus_manifest.content_component_manifest_digest
        != content_raw["self_hash"]
    ):
        raise VNextPreflightError(
            "corpus/content component manifest digest mismatch"
        )
    if (
        corpus_manifest.content_policy_version
        != content_manifest.automatic_candidate_policy_version
    ):
        raise VNextPreflightError(
            "corpus/content automatic policy version mismatch"
        )
    unresolved = [
        candidate.candidate_id
        for candidate in content_manifest.candidates
        if candidate.disposition == "unresolved"
    ]
    if unresolved:
        raise VNextPreflightError(
            f"unresolved content candidates block execution: {unresolved}"
        )

    root = pathlib.Path(corpus_root)
    try:
        from ..domain.lobo_vnext import verify_raw_inventory

        verified = verify_raw_inventory(root, corpus_manifest)
    except Exception as exc:
        raise VNextPreflightError(
            f"raw corpus inventory verification failed: {exc}"
        ) from exc
    if verified is not corpus_manifest:
        raise VNextPreflightError(
            "raw inventory verifier did not return the bound manifest"
        )
    try:
        from ..domain.lobo_vnext import (
            recompute_automatic_content_candidates,
        )

        recompute_automatic_content_candidates(
            root, corpus_manifest, content_manifest
        )
    except Exception as exc:
        raise VNextPreflightError(
            f"automatic content candidate preflight failed: {exc}"
        ) from exc
    validate_execution_spec(
        execution_spec,
        corpus_manifest=corpus_manifest,
        model_spec=model_spec,
        inference_spec=inference_spec,
    )
    try:
        fold_manifest.validate_against(corpus_manifest, content_manifest)
    except Exception as exc:
        raise VNextPreflightError(
            f"outer fold manifest binding failed: {exc}"
        ) from exc
    _validate_outer_folds(corpus_manifest, content_manifest, fold_manifest)
    _validate_model_and_inference(
        model_spec, inference_spec, execution_spec
    )
    _validate_inner_cv_plan(
        inner_cv_plan,
        fold_manifest,
        corpus_manifest,
        content_manifest,
        model_spec,
    )
    # Disk rows are still preflight evidence, not a representation cache.
    rows = _load_bound_rows(
        root, corpus_manifest, execution_spec["representation_receipt"]
    )
    works_with_rows = {row.work_id for row in rows}
    expected_works = {work.work_id for work in corpus_manifest.works}
    if works_with_rows != expected_works:
        raise VNextPreflightError(
            "model rows do not cover the exact included work inventory"
        )
    authors_by_work = {
        work.work_id: work.author_id for work in corpus_manifest.works
    }
    if any(
        row.author_id != authors_by_work[row.work_id]
        for row in rows
    ):
        raise VNextPreflightError(
            "model row author/work identity mismatch"
        )
    if corpus_raw["self_hash"] != execution_spec[
        "representation_receipt"
    ]["corpus_manifest_sha256"]:
        raise VNextPreflightError(
            "representation receipt corpus hash mismatch"
        )
    identity = _build_run_identity(
        corpus_manifest,
        content_manifest,
        fold_manifest,
        inner_cv_plan,
        model_spec,
        inference_spec,
        execution_spec,
    )
    return VNextPreflight(
        corpus_manifest=corpus_manifest,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        inner_cv_plan=inner_cv_plan,
        model_spec=model_spec,
        inference_spec=inference_spec,
        execution_spec=execution_spec,
        rows=rows,
        run_identity=identity,
    )


class _SyntheticUniformProbe:
    """Deliberately non-scientific estimator used only to exercise the harness."""

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        *,
        groups: np.ndarray,
        inner_splits: tuple[Any, ...],
    ) -> "_SyntheticUniformProbe":
        if (
            texts.ndim != 1
            or labels.ndim != 1
            or groups.ndim != 1
            or not (len(texts) == len(labels) == len(groups))
            or len(texts) == 0
        ):
            raise VNextPreflightError(
                "synthetic probe received malformed outer-train rows"
            )
        if set(int(label) for label in labels.tolist()) != set(
            range(self._n_classes)
        ):
            raise VNextPreflightError(
                "synthetic probe outer train lost class support"
            )
        if type(inner_splits) is not tuple:
            raise VNextPreflightError("inner split receipt was not restored")
        self.classes_ = np.arange(self._n_classes, dtype=np.int64)
        return self

    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("synthetic probe was not fit")
        return np.full(
            (len(texts), self._n_classes),
            1.0 / self._n_classes,
            dtype=np.float64,
        )


def _default_factory(model_spec: Any, fold: Any) -> _SyntheticUniformProbe:
    del model_spec
    return _SyntheticUniformProbe(len(fold.probability_class_order))


def _restore_and_validate_fold(preflight: VNextPreflight, fold_index: int) -> Any:
    fold = preflight.fold_manifest.folds[fold_index]
    try:
        from ..domain.lobo_vnext import FoldSpec

        validated = FoldSpec.from_dict(fold.to_dict())
    except Exception as exc:
        raise VNextPreflightError(
            f"worker fold restore validation failed: {exc}"
        ) from exc
    if (
        validated != fold
        or fold.self_hash
        != preflight.run_identity["fold_spec_sha256"][fold_index]
    ):
        raise VNextPreflightError(
            "worker fold restore identity mismatch"
        )
    return validated


def _restore_and_validate_inner_fold_plan(
    preflight: VNextPreflight,
    fold: Any,
) -> Any:
    from ..domain.lobo_vnext import InnerFoldPlan

    planned = preflight.inner_cv_plan.by_fold.get(fold.fold_id)
    if planned is None:
        raise VNextPreflightError(
            f"worker fold {fold.fold_id} has no inner-plan receipt"
        )
    try:
        restored = InnerFoldPlan.from_dict(planned.to_dict())
    except Exception as exc:
        raise VNextPreflightError(
            f"worker inner-plan restore validation failed: {exc}"
        ) from exc
    if (
        restored != planned
        or restored.fold_spec_digest != fold.self_hash
        or preflight.inner_cv_plan.self_hash
        != preflight.run_identity["inner_cv_plan_sha256"]
    ):
        raise VNextPreflightError(
            "worker inner-plan restore identity mismatch"
        )
    return restored


def _evaluate_fold(
    preflight: VNextPreflight,
    fold_index: int,
    factory: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    """Fit one model using only the exact outer-train receipt."""

    fold = _restore_and_validate_fold(preflight, fold_index)
    inner_fold_plan = _restore_and_validate_inner_fold_plan(preflight, fold)
    rows = preflight.rows
    train_ids = frozenset(fold.train_work_ids)
    train_rows = tuple(row for row in rows if row.work_id in train_ids)
    test_rows = tuple(row for row in rows if row.work_id == fold.test_work_id)
    forbidden = frozenset(fold.purged_work_ids) | {fold.test_work_id}
    if (
        not train_rows
        or not test_rows
        or {row.work_id for row in train_rows} != train_ids
        or any(row.work_id in forbidden for row in train_rows)
        or {row.work_id for row in test_rows} != {fold.test_work_id}
    ):
        raise VNextPreflightError(
            f"fold {fold.fold_id} failed restored outer split receipt"
        )
    probability_order = tuple(fold.probability_class_order)
    label_by_author = {
        author: index for index, author in enumerate(probability_order)
    }
    train_labels = np.asarray(
        [label_by_author[row.author_id] for row in train_rows],
        dtype=np.int64,
    )
    true_authors = {row.author_id for row in test_rows}
    if len(true_authors) != 1:
        raise VNextPreflightError(
            f"fold {fold.fold_id} test work has multiple authors"
        )
    true_author = next(iter(true_authors))
    true_label = label_by_author[true_author]

    # The first factory interaction occurs only after all global and restored
    # split checks above.
    estimator = factory(preflight.model_spec, fold)
    if estimator is None:
        raise LoboVNextError("model factory returned None")
    train_texts = np.asarray([row.text for row in train_rows], dtype=object)
    train_groups = np.asarray(
        [row.work_id for row in train_rows], dtype=object
    )
    estimator.fit(
        train_texts,
        train_labels,
        groups=train_groups,
        inner_splits=inner_fold_plan.splits,
    )
    test_texts = np.asarray([row.text for row in test_rows], dtype=object)
    raw_probabilities = estimator.predict_proba(test_texts)
    object_probabilities = np.asarray(raw_probabilities, dtype=object)
    if any(
        isinstance(value, (bool, np.bool_))
        or isinstance(value, (str, bytes))
        for value in object_probabilities.flat
    ):
        raise LoboVNextError(
            "estimator probabilities must be non-bool numeric scalars"
        )
    classes = getattr(estimator, "classes_", None)
    try:
        chunk_probabilities = validate_probability_matrix(
            raw_probabilities,
            classes,
            n_classes=len(probability_order),
            n_rows=len(test_rows),
        )
    except PredictionContractError as exc:
        raise LoboVNextError(
            f"fold {fold.fold_id} probability contract failed: {exc}"
        ) from exc
    book_probabilities = chunk_probabilities.mean(axis=0)
    decision = stable_top1_and_worst_tie_rank(
        book_probabilities,
        true_label=true_label,
        expected_width=len(probability_order),
    )
    result = {
        "work_id": fold.test_work_id,
        "true_author_id": true_author,
        "true_label": true_label,
        "predicted_author_id": probability_order[decision.top1],
        "predicted_label": decision.top1,
        "correct": bool(decision.top1 == true_label),
        "true_rank": decision.true_rank,
        "probabilities": [
            float(value) for value in book_probabilities.tolist()
        ],
        "chunk_count": len(test_rows),
    }
    checkpoint = build_vnext_checkpoint(
        identity=preflight.run_identity,
        model_spec=preflight.model_spec,
        inner_cv_plan=preflight.inner_cv_plan,
        fold_index=fold_index,
        fold=fold,
        result=result,
    )
    # The worker output is validated a second time after all learned state has
    # been discarded.
    restored_fold = _restore_and_validate_fold(preflight, fold_index)
    _restore_and_validate_inner_fold_plan(preflight, restored_fold)
    validate_vnext_checkpoint(
        checkpoint,
        identity=preflight.run_identity,
        model_spec=preflight.model_spec,
        inner_cv_plan=preflight.inner_cv_plan,
        fold_index=fold_index,
        fold=restored_fold,
    )
    return checkpoint


def _shared_inference_spec(inference_spec: Any) -> Any:
    return _shared._shared_inference_spec(
        inference_spec,
        error_type=VNextArtifactError,
    )


def _derive_metrics(
    checkpoints: Sequence[dict[str, Any]],
    fold_manifest: Any,
    inference_spec: Any,
    *,
    error_type: type[LoboVNextError] = VNextArtifactError,
) -> dict[str, Any]:
    return _shared._derive_metrics(
        checkpoints,
        fold_manifest,
        inference_spec,
        error_type=error_type,
    )


def _validate_metrics_schema(
    metrics: Any,
    *,
    error_type: type[LoboVNextError] = VNextArtifactError,
) -> dict[str, Any]:
    return _shared._validate_metrics_schema(
        metrics,
        error_type=error_type,
    )


def build_vnext_final_artifact(
    *,
    preflight: VNextPreflight,
    checkpoints: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ordered = list(checkpoints)
    metrics = _derive_metrics(
        ordered, preflight.fold_manifest, preflight.inference_spec
    )
    artifact = {
        "schema_version": FINAL_ARTIFACT_SCHEMA_VERSION,
        "status": EXPLORATORY_STATUS,
        "confirmatory_authorized": False,
        "run_identity": preflight.run_identity,
        "fold_manifest": preflight.fold_manifest.to_dict(),
        "inner_cv_plan": preflight.inner_cv_plan.to_dict(),
        "model_spec": preflight.model_spec.to_dict(),
        "inference_spec": preflight.inference_spec.to_dict(),
        "checkpoints": ordered,
        "metrics": metrics,
    }
    artifact["self_hash"] = _self_hash(artifact)
    validate_vnext_final_artifact(artifact, preflight=preflight)
    return artifact


def _load_embedded_specs(
    artifact: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    from ..domain.lobo_vnext import (
        loads_fold_manifest,
        loads_inference_spec,
        loads_inner_cv_plan,
        loads_model_spec,
    )

    encoded = lambda value: dumps_strict(  # noqa: E731 - narrow local adapter
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        return (
            loads_fold_manifest(encoded(artifact["fold_manifest"])),
            loads_inner_cv_plan(encoded(artifact["inner_cv_plan"])),
            loads_model_spec(encoded(artifact["model_spec"])),
            loads_inference_spec(encoded(artifact["inference_spec"])),
        )
    except Exception as exc:
        raise VNextArtifactError(
            f"embedded vNext spec validation failed: {exc}"
        ) from exc


def validate_vnext_final_artifact(
    artifact: Any,
    *,
    preflight: VNextPreflight | None = None,
) -> dict[str, Any]:
    _require_exact_dict(
        artifact,
        {
            "schema_version",
            "status",
            "confirmatory_authorized",
            "run_identity",
            "fold_manifest",
            "inner_cv_plan",
            "model_spec",
            "inference_spec",
            "checkpoints",
            "metrics",
            "self_hash",
        },
        path="final",
        error_type=VNextArtifactError,
    )
    if artifact["schema_version"] != FINAL_ARTIFACT_SCHEMA_VERSION:
        raise VNextArtifactError(
            "legacy or unsupported final artifact is read-only"
        )
    if (
        artifact["status"] != EXPLORATORY_STATUS
        or artifact["confirmatory_authorized"] is not False
    ):
        raise VNextArtifactError("final artifact authority/status mismatch")
    identity = validate_run_identity(artifact["run_identity"])
    (
        fold_manifest,
        inner_cv_plan,
        model_spec,
        inference_spec,
    ) = _load_embedded_specs(artifact)
    if preflight is not None:
        expected = (
            preflight.fold_manifest.to_dict(),
            preflight.inner_cv_plan.to_dict(),
            preflight.model_spec.to_dict(),
            preflight.inference_spec.to_dict(),
            preflight.run_identity,
        )
        observed = (
            artifact["fold_manifest"],
            artifact["inner_cv_plan"],
            artifact["model_spec"],
            artifact["inference_spec"],
            identity,
        )
        _require_exact_structure(
            observed,
            expected,
            path="final.preflight_bindings",
            error_type=VNextArtifactError,
        )
    if (
        fold_manifest.self_hash != identity["fold_manifest_sha256"]
        or inner_cv_plan.self_hash != identity["inner_cv_plan_sha256"]
        or model_spec.self_hash != identity["model_spec_sha256"]
        or inference_spec.self_hash != identity["inference_spec_sha256"]
    ):
        raise VNextArtifactError("final embedded spec/run identity mismatch")
    if (
        inner_cv_plan.fold_manifest_digest != fold_manifest.self_hash
        or inner_cv_plan.model_spec_digest != model_spec.self_hash
        or inner_cv_plan.content_component_manifest_digest
        != identity["content_manifest_sha256"]
    ):
        raise VNextArtifactError(
            "final inner-CV plan binding mismatch"
        )
    checkpoints = _require_list(
        artifact["checkpoints"],
        path="final.checkpoints",
        nonempty=True,
        error_type=VNextArtifactError,
    )
    if len(checkpoints) != len(fold_manifest.folds):
        raise VNextArtifactError(
            "final artifact checkpoint inventory is incomplete"
        )
    for index, (checkpoint, fold) in enumerate(
        zip(checkpoints, fold_manifest.folds, strict=True)
    ):
        try:
            validate_vnext_checkpoint(
                checkpoint,
                identity=identity,
                model_spec=model_spec,
                inner_cv_plan=inner_cv_plan,
                fold_index=index,
                fold=fold,
            )
        except VNextCheckpointError as exc:
            raise VNextArtifactError(
                f"invalid final checkpoint[{index}]: {exc}"
            ) from exc
    expected_metrics = _derive_metrics(
        checkpoints, fold_manifest, inference_spec
    )
    _validate_metrics_schema(artifact["metrics"])
    _require_exact_structure(
        artifact["metrics"],
        expected_metrics,
        path="final.metrics",
        error_type=VNextArtifactError,
    )
    _require_self_hash(
        artifact, path="final", error_type=VNextArtifactError
    )
    return artifact


def _validate_representation_builder_result(
    observed: Any,
    expected: dict[str, Any],
) -> None:
    if type(observed) is not dict:
        raise VNextPreflightError(
            "representation builder must return an exact receipt object"
        )
    _require_exact_structure(
        observed,
        expected,
        path="prepared_representation_receipt",
        error_type=VNextPreflightError,
    )


def run_lobo_vnext(
    *,
    corpus_root: str | os.PathLike[str],
    corpus_manifest: Any,
    content_manifest: Any,
    fold_manifest: Any,
    inner_cv_plan: Any,
    model_spec: Any,
    inference_spec: Any,
    execution_spec: dict[str, Any],
    output_namespace: str | os.PathLike[str],
    n_jobs: int,
    factory: Callable[[Any, Any], Any] | None = None,
    representation_builder: Callable[
        [VNextPreflight], dict[str, Any]
    ]
    | None = None,
) -> VNextRunOutcome:
    """Run or resume one synthetic vNext manifest.

    ``n_jobs`` is scheduling-only telemetry.  It deliberately does not alter
    the scientific run identity, so serial, parallel, and resumed executions
    produce byte-identical checkpoints and final artifacts.
    """

    if type(n_jobs) is not int or not 1 <= n_jobs <= 8:
        raise VNextPreflightError("n_jobs must be an exact integer in [1,8]")
    preflight = preflight_lobo_vnext(
        corpus_root=corpus_root,
        corpus_manifest=corpus_manifest,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        inner_cv_plan=inner_cv_plan,
        model_spec=model_spec,
        inference_spec=inference_spec,
        execution_spec=execution_spec,
    )
    output_path = pathlib.Path(output_namespace)
    _guard_output_namespace(output_path)
    store = _VNextCheckpointStore(output_path, preflight)
    store.inspect_existing()

    # Representation preparation is downstream of every global preflight
    # and existing-namespace rejection.  Its receipt must be byte-for-byte the
    # run-bound plan.
    if representation_builder is not None:
        prepared_receipt = representation_builder(preflight)
        _validate_representation_builder_result(
            prepared_receipt, execution_spec["representation_receipt"]
        )

    if factory is None:
        if execution_spec["estimator_key"] != SYNTHETIC_UNIFORM_ESTIMATOR:
            raise VNextPreflightError(
                "no built-in synthetic estimator matches the explicit ModelSpec"
            )
        factory = _default_factory
    elif not callable(factory):
        raise VNextPreflightError("factory must be callable")

    store.initialize()
    present = store.scan()
    pending = [
        index
        for index in range(len(fold_manifest.folds))
        if index not in present
    ]
    timings: dict[int, float] = {}

    def evaluate(index: int) -> tuple[int, dict[str, Any], float]:
        started = time.perf_counter()
        checkpoint = _evaluate_fold(preflight, index, factory)
        return index, checkpoint, float(time.perf_counter() - started)

    if n_jobs == 1:
        for index in pending:
            fold_index, checkpoint, duration = evaluate(index)
            store.save(
                checkpoint,
                fold_index=fold_index,
                fold=fold_manifest.folds[fold_index],
            )
            timings[fold_index] = duration
    elif pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_jobs,
            thread_name_prefix="stylo-lobo-vnext",
        ) as executor:
            futures = {
                executor.submit(evaluate, index): index for index in pending
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    fold_index, checkpoint, duration = future.result()
                    store.save(
                        checkpoint,
                        fold_index=fold_index,
                        fold=fold_manifest.folds[fold_index],
                    )
                    timings[fold_index] = duration
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    complete = store.scan()
    if set(complete) != set(range(len(fold_manifest.folds))):
        missing = sorted(set(range(len(fold_manifest.folds))) - set(complete))
        raise VNextCheckpointError(
            f"vNext checkpoint set is incomplete: {missing}"
        )
    checkpoints = [complete[index] for index in range(len(complete))]
    artifact = build_vnext_final_artifact(
        preflight=preflight, checkpoints=checkpoints
    )
    artifact_path = store.publish_final(artifact)
    telemetry = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "scientific_result_hashed": False,
        "run_id": preflight.run_id,
        "n_jobs": n_jobs,
        "computed_folds": len(pending),
        "resumed_folds": len(present),
        "fold_seconds": [
            {
                "fold_index": index,
                "seconds": timings[index],
            }
            for index in sorted(timings)
        ],
    }
    return VNextRunOutcome(
        artifact=artifact,
        artifact_path=artifact_path,
        telemetry=telemetry,
        computed_folds=len(pending),
        resumed_folds=len(present),
    )


def run_lobo_vnext_from_specs(
    *,
    corpus_root: str | os.PathLike[str],
    corpus_manifest_path: str | os.PathLike[str],
    content_manifest_path: str | os.PathLike[str],
    fold_manifest_path: str | os.PathLike[str],
    inner_cv_plan_path: str | os.PathLike[str],
    model_spec_path: str | os.PathLike[str],
    inference_spec_path: str | os.PathLike[str],
    execution_spec_path: str | os.PathLike[str],
    output_namespace: str | os.PathLike[str],
    n_jobs: int,
) -> dict[str, Any]:
    """The single file-oriented CLI entrypoint.

    All domain JSON uses duplicate-key-rejecting loaders.  There is no legacy
    fallback, no scientific default, and no real-corpus dispatch.
    """

    from ..domain.lobo_vnext import (
        load_content_component_manifest,
        load_corpus_vnext_manifest,
        load_fold_manifest,
        load_inference_spec,
        load_inner_cv_plan,
        load_model_spec,
    )

    try:
        corpus_manifest = load_corpus_vnext_manifest(corpus_manifest_path)
        content_manifest = load_content_component_manifest(
            content_manifest_path
        )
        fold_manifest = load_fold_manifest(fold_manifest_path)
        inner_cv_plan = load_inner_cv_plan(inner_cv_plan_path)
        model_spec = load_model_spec(model_spec_path)
        inference_spec = load_inference_spec(inference_spec_path)
        execution_spec = load_execution_spec(execution_spec_path)
    except LoboVNextError:
        raise
    except Exception as exc:
        raise VNextPreflightError(
            f"strict vNext spec loading failed: {exc}"
        ) from exc
    outcome = run_lobo_vnext(
        corpus_root=corpus_root,
        corpus_manifest=corpus_manifest,
        content_manifest=content_manifest,
        fold_manifest=fold_manifest,
        inner_cv_plan=inner_cv_plan,
        model_spec=model_spec,
        inference_spec=inference_spec,
        execution_spec=execution_spec,
        output_namespace=output_namespace,
        n_jobs=n_jobs,
    )
    return outcome.artifact


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "EXECUTION_SPEC_SCHEMA_VERSION",
    "EXPLORATORY_AUTHORIZATION",
    "EXPLORATORY_STATUS",
    "FINAL_ARTIFACT_SCHEMA_VERSION",
    "IDENTITY_RECEIPT_SCHEMA_VERSION",
    "LEGACY_PROJECTION_SCHEMA_VERSION",
    "LOBO_STRATEGY",
    "LoboVNextError",
    "REAL_EXECUTION_MODE",
    "REPRESENTATION_RECEIPT_SCHEMA_VERSION",
    "RUN_IDENTITY_SCHEMA_VERSION",
    "SYNTHETIC_EXECUTION_MODE",
    "SYNTHETIC_UNIFORM_ESTIMATOR",
    "TELEMETRY_SCHEMA_VERSION",
    "VNextArtifactError",
    "VNextCheckpointError",
    "VNextPreflight",
    "VNextPreflightError",
    "VNextRunOutcome",
    "VNextTextRow",
    "build_execution_spec",
    "build_identity_receipt",
    "build_representation_receipt",
    "build_vnext_checkpoint",
    "build_vnext_final_artifact",
    "load_execution_spec",
    "preflight_lobo_vnext",
    "project_legacy_artifact_read_only",
    "reject_legacy_checkpoint_resume",
    "run_lobo_vnext",
    "run_lobo_vnext_from_specs",
    "validate_execution_spec",
    "validate_identity_receipt",
    "validate_representation_receipt",
    "validate_run_identity",
    "validate_vnext_checkpoint",
    "validate_vnext_final_artifact",
]
