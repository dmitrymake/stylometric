"""Private implementation primitives shared by the LOBO-vNext evaluators.

The synthetic harness and the owner-bound real evaluator deliberately keep
their domain preflight, checkpoint stores, schemas, and final artifact builders
separate.  This module contains only byte-level and validation mechanics whose
behaviour must remain identical across those two boundaries.  Callers always
provide their own error class so failures stay inside the correct public error
family.
"""
from __future__ import annotations

import hashlib
import math
import os
import pathlib
import re
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..jsonio import dumps_strict, load_strict
from .metrics import AuthorClusteredInferenceSpec, summarize_book_results


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _strict_json_tree(
    value: Any,
    *,
    path: str = "value",
    error_type: type[Exception],
) -> None:
    """Reject coercible/non-finite/non-JSON values before canonical hashing."""

    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise error_type(f"{path} must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _strict_json_tree(
                item,
                path=f"{path}[{index}]",
                error_type=error_type,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise error_type(f"{path} keys must be exact strings")
            _strict_json_tree(
                item,
                path=f"{path}.{key}",
                error_type=error_type,
            )
        return
    raise error_type(
        f"{path} must contain only exact JSON scalar/container types; "
        f"got {type(value).__name__}"
    )


def _canonical_bytes(
    value: Any,
    *,
    error_type: type[Exception],
) -> bytes:
    _strict_json_tree(value, error_type=error_type)
    try:
        text = dumps_strict(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:  # defensive after strict-tree check
        raise error_type(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _canonical_hash(
    value: Any,
    *,
    error_type: type[Exception],
) -> str:
    return hashlib.sha256(
        _canonical_bytes(value, error_type=error_type)
    ).hexdigest()


def _self_hash(
    value: Mapping[str, Any],
    *,
    error_type: type[Exception],
) -> str:
    if type(value) is not dict:
        raise error_type("self-hashed value must be an exact object")
    return _canonical_hash(
        {key: item for key, item in value.items() if key != "self_hash"},
        error_type=error_type,
    )


def _require_exact_dict(
    value: Any,
    keys: set[str] | frozenset[str],
    *,
    path: str,
    error_type: type[Exception],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise error_type(f"{path} must be an exact object")
    observed = set(value)
    if observed != set(keys):
        raise error_type(
            f"{path} key mismatch: "
            f"missing={sorted(set(keys) - observed)}, "
            f"extra={sorted(observed - set(keys))}"
        )
    return value


def _require_list(
    value: Any,
    *,
    path: str,
    nonempty: bool = False,
    error_type: type[Exception],
) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = "nonempty exact" if nonempty else "exact"
        raise error_type(f"{path} must be a {qualifier} array")
    return value


def _require_str(
    value: Any,
    *,
    path: str,
    nonempty: bool = True,
    error_type: type[Exception],
) -> str:
    if type(value) is not str or (nonempty and not value):
        raise error_type(f"{path} must be an exact nonempty string")
    return value


def _require_sha256(
    value: Any,
    *,
    path: str,
    error_type: type[Exception],
) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise error_type(f"{path} must be a lowercase SHA-256")
    return value


def _require_bool(
    value: Any,
    *,
    path: str,
    error_type: type[Exception],
) -> bool:
    if type(value) is not bool:
        raise error_type(f"{path} must be an exact bool")
    return value


def _require_int(
    value: Any,
    *,
    path: str,
    minimum: int | None = None,
    error_type: type[Exception],
) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise error_type(f"{path} must be an exact integer{suffix}")
    return value


def _require_float(
    value: Any,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
    error_type: type[Exception],
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise error_type(f"{path} must be an exact finite float")
    if minimum is not None and value < minimum:
        raise error_type(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise error_type(f"{path} must be <= {maximum}")
    return value


def _require_string_array(
    value: Any,
    *,
    path: str,
    nonempty: bool = False,
    unique: bool = False,
    error_type: type[Exception],
) -> tuple[str, ...]:
    values = _require_list(
        value,
        path=path,
        nonempty=nonempty,
        error_type=error_type,
    )
    if any(type(item) is not str or not item for item in values):
        raise error_type(f"{path} must contain exact nonempty strings")
    if unique and len(set(values)) != len(values):
        raise error_type(f"{path} must not contain duplicates")
    return tuple(values)


def _require_self_hash(
    value: dict[str, Any],
    *,
    path: str,
    error_type: type[Exception],
) -> None:
    _require_sha256(
        value["self_hash"],
        path=f"{path}.self_hash",
        error_type=error_type,
    )
    try:
        expected = _self_hash(value, error_type=error_type)
    except error_type as exc:
        raise error_type(
            f"{path} is not strict canonical JSON: {exc}"
        ) from exc
    if value["self_hash"] != expected:
        raise error_type(f"{path}.self_hash mismatch")


def _require_exact_structure(
    observed: Any,
    expected: Any,
    *,
    path: str,
    error_type: type[Exception],
) -> None:
    """Compare JSON without Python's bool/int or int/float equality."""

    if type(observed) is not type(expected):
        raise error_type(
            f"{path} type mismatch: expected {type(expected).__name__}, "
            f"got {type(observed).__name__}"
        )
    if type(expected) is dict:
        _require_exact_dict(
            observed,
            set(expected),
            path=path,
            error_type=error_type,
        )
        for key in expected:
            _require_exact_structure(
                observed[key],
                expected[key],
                path=f"{path}.{key}",
                error_type=error_type,
            )
        return
    if type(expected) is list:
        if len(observed) != len(expected):
            raise error_type(
                f"{path} length mismatch: expected {len(expected)}, "
                f"got {len(observed)}"
            )
        for index, (left, right) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _require_exact_structure(
                left,
                right,
                path=f"{path}[{index}]",
                error_type=error_type,
            )
        return
    if type(expected) is float and (
        not math.isfinite(expected) or not math.isfinite(observed)
    ):
        raise error_type(f"{path} must be finite")
    if observed != expected:
        raise error_type(f"{path} value mismatch")


def _reject_absolute_paths(
    value: Any,
    *,
    path: str,
    error_type: type[Exception],
) -> None:
    """Keep run-bound receipts host-path independent."""

    if type(value) is str:
        if pathlib.PurePosixPath(value).is_absolute() or re.match(
            r"^[A-Za-z]:[\\/]", value
        ):
            raise error_type(f"{path} contains an absolute host path")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _reject_absolute_paths(
                item,
                path=f"{path}[{index}]",
                error_type=error_type,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            _reject_absolute_paths(
                item,
                path=f"{path}.{key}",
                error_type=error_type,
            )


def _canonical_file_bytes(
    value: dict[str, Any],
    *,
    error_type: type[Exception],
) -> bytes:
    return _canonical_bytes(value, error_type=error_type) + b"\n"


def _durable_create(
    path: pathlib.Path,
    payload: dict[str, Any],
    *,
    error_type: type[Exception],
) -> bool:
    """Atomically create ``path`` without overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_file_bytes(payload, error_type=error_type)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_json_exact(
    path: pathlib.Path,
    *,
    error_type: type[Exception],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise error_type(f"missing or symlinked JSON file: {path}")
    try:
        value = load_strict(path)
    except Exception as exc:
        raise error_type(f"cannot strictly load {path}: {exc}") from exc
    if type(value) is not dict:
        raise error_type(f"{path} must contain an exact JSON object")
    return value


def _guard_output_namespace(
    path: pathlib.Path,
    *,
    error_type: type[Exception],
) -> None:
    parts = tuple(part.lower() for part in path.parts)
    if not any("exploratory" in part for part in parts):
        raise error_type(
            "vNext dry-run output must use an explicitly exploratory namespace"
        )
    forbidden = {"evidence", "frozen", "confirmatory", "public", "headline"}
    if any(part in forbidden for part in parts):
        raise error_type(
            "vNext dry-run output cannot target evidence/public namespaces"
        )
    current = (
        pathlib.Path(path.anchor) if path.is_absolute() else pathlib.Path()
    )
    for part in path.parts:
        if path.is_absolute() and part == path.anchor:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise error_type(
                f"symlinked exploratory output path rejected: {current}"
            )


def _shared_inference_spec(
    inference_spec: Any,
    *,
    error_type: type[Exception],
) -> AuthorClusteredInferenceSpec:
    try:
        return AuthorClusteredInferenceSpec.build(
            iterations=inference_spec.bootstrap_iterations,
            confidence_level=inference_spec.confidence_level,
            seed=inference_spec.bootstrap_seed,
        )
    except Exception as exc:
        raise error_type(
            f"cannot construct author-clustered inference contract: {exc}"
        ) from exc


def _derive_metrics(
    checkpoints: Sequence[dict[str, Any]],
    fold_manifest: Any,
    inference_spec: Any,
    *,
    error_type: type[Exception],
) -> dict[str, Any]:
    folds = tuple(fold_manifest.folds)
    if len(checkpoints) != len(folds):
        raise error_type(
            "cannot derive metrics from an incomplete checkpoint set"
        )
    probability_order = tuple(folds[0].probability_class_order)
    metric_order = tuple(folds[0].metric_label_order)
    label_by_author = {
        author: index for index, author in enumerate(probability_order)
    }
    metric_indices = tuple(label_by_author[author] for author in metric_order)
    truth = np.asarray(
        [checkpoint["result"]["true_label"] for checkpoint in checkpoints],
        dtype=np.int64,
    )
    predicted = np.asarray(
        [
            checkpoint["result"]["predicted_label"]
            for checkpoint in checkpoints
        ],
        dtype=np.int64,
    )
    ranks = np.asarray(
        [checkpoint["result"]["true_rank"] for checkpoint in checkpoints],
        dtype=np.int64,
    )
    book_authors = [
        checkpoint["result"]["true_author_id"] for checkpoint in checkpoints
    ]
    try:
        summary = summarize_book_results(
            truth,
            predicted,
            ranks,
            probability_order,
            metric_label_order=metric_indices,
            book_authors=book_authors,
            inference_spec=_shared_inference_spec(
                inference_spec,
                error_type=error_type,
            ),
        )
    except Exception as exc:
        raise error_type(f"metric derivation failed: {exc}") from exc
    per_author: list[dict[str, Any]] = []
    for author_id in metric_order:
        label = label_by_author[author_id]
        mask = truth == label
        if not mask.any():
            raise error_type(
                f"metric label {author_id!r} has no tested books"
            )
        recall = float(np.mean(predicted[mask] == label))
        per_author.append(
            {
                "author_id": author_id,
                "n_test_books": int(mask.sum()),
                "recall": recall,
            }
        )
    accuracy_ci = summary["accuracy"]
    macro_f1 = summary["macro_f1"]
    top2 = summary["top2"]
    metrics = {
        "n_books": len(checkpoints),
        "primary_accuracy": {
            "point": float(accuracy_ci.point),
            "lo": float(accuracy_ci.lo),
            "hi": float(accuracy_ci.hi),
            "method": accuracy_ci.method,
        },
        "macro_f1": {
            "point": float(macro_f1.point),
            "uncertainty": macro_f1.uncertainty,
        },
        "top2": {
            "point": float(top2.point),
            "uncertainty": top2.uncertainty,
        },
        "per_author_diagnostics": per_author,
        "probability_class_order": list(probability_order),
        "metric_label_order": list(metric_order),
        "inference_spec_sha256": inference_spec.self_hash,
    }
    _validate_metrics_schema(metrics, error_type=error_type)
    return metrics


def _validate_metrics_schema(
    metrics: Any,
    *,
    error_type: type[Exception],
) -> dict[str, Any]:
    _require_exact_dict(
        metrics,
        {
            "n_books",
            "primary_accuracy",
            "macro_f1",
            "top2",
            "per_author_diagnostics",
            "probability_class_order",
            "metric_label_order",
            "inference_spec_sha256",
        },
        path="final.metrics",
        error_type=error_type,
    )
    _require_int(
        metrics["n_books"],
        path="final.metrics.n_books",
        minimum=1,
        error_type=error_type,
    )
    accuracy_record = _require_exact_dict(
        metrics["primary_accuracy"],
        {"point", "lo", "hi", "method"},
        path="final.metrics.primary_accuracy",
        error_type=error_type,
    )
    for key in ("point", "lo", "hi"):
        _require_float(
            accuracy_record[key],
            path=f"final.metrics.primary_accuracy.{key}",
            minimum=0.0,
            maximum=1.0,
            error_type=error_type,
        )
    if not (
        accuracy_record["lo"]
        <= accuracy_record["point"]
        <= accuracy_record["hi"]
    ):
        raise error_type("primary accuracy point must lie inside its CI")
    if (
        accuracy_record["method"]
        != "author_clustered_percentile_bootstrap"
    ):
        raise error_type(
            "primary accuracy uses a non-clustered uncertainty method"
        )
    for metric_name in ("macro_f1", "top2"):
        record = _require_exact_dict(
            metrics[metric_name],
            {"point", "uncertainty"},
            path=f"final.metrics.{metric_name}",
            error_type=error_type,
        )
        _require_float(
            record["point"],
            path=f"final.metrics.{metric_name}.point",
            minimum=0.0,
            maximum=1.0,
            error_type=error_type,
        )
        if record["uncertainty"] != "point_only":
            raise error_type(
                f"final.metrics.{metric_name} must remain point-only"
            )
    diagnostics = _require_list(
        metrics["per_author_diagnostics"],
        path="final.metrics.per_author_diagnostics",
        nonempty=True,
        error_type=error_type,
    )
    for index, diagnostic in enumerate(diagnostics):
        _require_exact_dict(
            diagnostic,
            {"author_id", "n_test_books", "recall"},
            path=f"final.metrics.per_author_diagnostics[{index}]",
            error_type=error_type,
        )
        _require_str(
            diagnostic["author_id"],
            path=f"final.metrics.per_author_diagnostics[{index}].author_id",
            error_type=error_type,
        )
        _require_int(
            diagnostic["n_test_books"],
            path=(
                f"final.metrics.per_author_diagnostics[{index}].n_test_books"
            ),
            minimum=1,
            error_type=error_type,
        )
        _require_float(
            diagnostic["recall"],
            path=f"final.metrics.per_author_diagnostics[{index}].recall",
            minimum=0.0,
            maximum=1.0,
            error_type=error_type,
        )
    probability_order = _require_string_array(
        metrics["probability_class_order"],
        path="final.metrics.probability_class_order",
        nonempty=True,
        unique=True,
        error_type=error_type,
    )
    metric_order = _require_string_array(
        metrics["metric_label_order"],
        path="final.metrics.metric_label_order",
        nonempty=True,
        unique=True,
        error_type=error_type,
    )
    if (
        tuple(item for item in probability_order if item in metric_order)
        != metric_order
    ):
        raise error_type(
            "final metric label order is not a P-ordered subset"
        )
    if [item["author_id"] for item in diagnostics] != list(metric_order):
        raise error_type(
            "per-author diagnostics do not use exact M order"
        )
    _require_sha256(
        metrics["inference_spec_sha256"],
        path="final.metrics.inference_spec_sha256",
        error_type=error_type,
    )
    return metrics
