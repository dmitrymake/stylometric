#!/usr/bin/env python3
"""Resume the interrupted B4 true-LOBO run with kernel version fields non-binding.

This is a deliberately narrow recovery wrapper.  The original run started before a reboot and
bound ``platform``/``release`` from ``uname`` into its immutable identity.  The user explicitly
directed that those two OS-kernel fields must not force a full recomputation.  We therefore:

* build the ordinary current identity with the unchanged, hashed production runner;
* require exact equality with the saved identity after replacing only the two kernel fields;
* keep the saved identity/checkpoints byte-for-byte and run the ordinary evaluator;
* record the actual runtime and every recovery attempt in an adjacent atomic audit sidecar.

No numerical-library, thread, code, config, corpus, cache, class-order, or work-order drift is
accepted.  The sidecar makes the mixed kernel observation explicit instead of pretending that the
new folds ran under the pre-reboot kernel.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import os
import pathlib
import sys
from typing import Any, Sequence


# Match the deterministic environment contract before importing NumPy/SciPy/sklearn indirectly.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_b4_true_lobo as base  # noqa: E402
from stylo.eval.b4_pilot import artifact_self_hash, canonical_hash  # noqa: E402
from stylo.eval.b4_true_lobo import RUN_SCHEMA_VERSION, validate_run_identity  # noqa: E402
from stylo.jsonio import dump_strict, load_strict  # noqa: E402


SCHEMA_VERSION = "b4_true_lobo_kernel_compat_audit_v1"
NONBINDING_RUNTIME_FIELDS = ("platform", "release")
OUTPUT_PATH = (
    ROOT / "docs" / "exploratory" / "work_balanced" /
    "b4_true_lobo_a0_a1_a4_v1.json"
)
CHECKPOINT_ROOT = OUTPUT_PATH.with_suffix(".checkpoints")
RUN_PATH = CHECKPOINT_ROOT / "RUN.json"
AUDIT_PATH = OUTPUT_PATH.with_suffix(".kernel_compat_audit.json")
LOCK_PATH = OUTPUT_PATH.with_suffix(".kernel_compat.lock")


class KernelCompatibilityError(RuntimeError):
    """The interrupted run differs in something other than the allowed kernel fields."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash_identity(identity: dict[str, Any]) -> None:
    body = {
        key: value for key, value in identity.items() if key not in {"run_id", "self_hash"}
    }
    identity["run_id"] = canonical_hash(body)
    identity["self_hash"] = artifact_self_hash(identity)


def _diff_paths(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            paths.extend(_diff_paths(left_item, right_item, f"{path}[{index}]"))
        return paths
    return [] if left == right else [path]


def require_kernel_only_compatibility(
    saved: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return a drift report or fail if anything besides two kernel strings changed."""
    validate_run_identity(saved)
    validate_run_identity(candidate)
    saved_runtime = saved.get("runtime_fingerprint")
    current_runtime = candidate.get("runtime_fingerprint")
    if not isinstance(saved_runtime, dict) or not isinstance(current_runtime, dict):
        raise KernelCompatibilityError("both identities must contain runtime_fingerprint objects")
    if set(saved_runtime) != set(current_runtime):
        raise KernelCompatibilityError("runtime_fingerprint field inventory changed")
    for field in NONBINDING_RUNTIME_FIELDS:
        if type(saved_runtime.get(field)) is not str or type(current_runtime.get(field)) is not str:
            raise KernelCompatibilityError(f"runtime_fingerprint.{field} must be a string")

    normalised = copy.deepcopy(candidate)
    for field in NONBINDING_RUNTIME_FIELDS:
        normalised["runtime_fingerprint"][field] = saved_runtime[field]
    _rehash_identity(normalised)
    if normalised != saved:
        differences = _diff_paths(saved, normalised)
        preview = differences[:12]
        suffix = " ..." if len(differences) > len(preview) else ""
        raise KernelCompatibilityError(
            "non-kernel run identity drift detected: " + ", ".join(preview) + suffix
        )

    observed = {
        field: {"saved": saved_runtime[field], "current": current_runtime[field]}
        for field in NONBINDING_RUNTIME_FIELDS
        if saved_runtime[field] != current_runtime[field]
    }
    return {
        "compatible": True,
        "nonbinding_fields": [
            f"runtime_fingerprint.{field}" for field in NONBINDING_RUNTIME_FIELDS
        ],
        "observed_drift": observed,
        "candidate_run_id_before_kernel_normalisation": candidate["run_id"],
        "normalised_identity_equals_saved": True,
    }


def _checkpoint_counts() -> dict[str, int]:
    return {
        cell: len(list((CHECKPOINT_ROOT / cell).glob("*.json")))
        for cell in base.CELL_ORDER
    }


def _load_audit(saved: dict[str, Any]) -> dict[str, Any]:
    if not AUDIT_PATH.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "active",
            "policy": {
                "decision": "kernel_platform_and_release_are_nonbinding",
                "authority": "explicit_user_direction_after_interrupted_run",
                "nonbinding_fields": [
                    f"runtime_fingerprint.{field}" for field in NONBINDING_RUNTIME_FIELDS
                ],
                "all_other_identity_fields": "must_match_exactly",
            },
            "saved_run": {
                "run_id": saved["run_id"],
                "self_hash": saved["self_hash"],
                "runtime_fingerprint": copy.deepcopy(saved["runtime_fingerprint"]),
                "thread_fingerprint": copy.deepcopy(saved["thread_fingerprint"]),
            },
            "output_path": str(OUTPUT_PATH.resolve()),
            "checkpoint_root": str(CHECKPOINT_ROOT.resolve()),
            "runtime_observations": [],
            "attempts": [],
        }
    audit = load_strict(AUDIT_PATH)
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("self_hash") != artifact_self_hash(audit)
        or audit.get("saved_run", {}).get("run_id") != saved["run_id"]
        or audit.get("saved_run", {}).get("self_hash") != saved["self_hash"]
    ):
        raise KernelCompatibilityError("existing kernel compatibility audit is invalid")
    return audit


def _save_audit(audit: dict[str, Any]) -> None:
    audit["self_hash"] = artifact_self_hash(audit)
    dump_strict(audit, AUDIT_PATH, indent=2, ensure_ascii=False)


def _record_attempt(
    saved: dict[str, Any],
    candidate: dict[str, Any],
    report: dict[str, Any],
    *,
    n_jobs: int,
) -> int:
    audit = _load_audit(saved)
    observation = {
        "observed_at_utc": _utc_now(),
        "runtime_fingerprint": copy.deepcopy(candidate["runtime_fingerprint"]),
        "thread_fingerprint": copy.deepcopy(candidate["thread_fingerprint"]),
        "candidate_run_id_before_kernel_normalisation": candidate["run_id"],
        "compatibility": copy.deepcopy(report),
    }
    audit["runtime_observations"].append(observation)
    attempt = {
        "started_at_utc": _utc_now(),
        "status": "active",
        "n_jobs": int(n_jobs),
        "checkpoint_counts_at_start": _checkpoint_counts(),
        "wrapper": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": _sha256_file(pathlib.Path(__file__).resolve()),
        },
        "hashed_runner": {
            "path": str(pathlib.Path(base.__file__).resolve()),
            "sha256": _sha256_file(pathlib.Path(base.__file__).resolve()),
        },
        "runtime_observation_index": len(audit["runtime_observations"]) - 1,
    }
    audit["attempts"].append(attempt)
    audit["status"] = "active"
    _save_audit(audit)
    return len(audit["attempts"]) - 1


def _finish_attempt(index: int, *, status: str, error: BaseException | None = None) -> None:
    saved = load_strict(RUN_PATH)
    audit = _load_audit(saved)
    if not 0 <= index < len(audit["attempts"]):
        raise KernelCompatibilityError("kernel compatibility attempt index disappeared")
    attempt = audit["attempts"][index]
    attempt["status"] = status
    attempt["finished_at_utc"] = _utc_now()
    attempt["checkpoint_counts_at_finish"] = _checkpoint_counts()
    if error is not None:
        attempt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    if status == "complete" and OUTPUT_PATH.is_file():
        attempt["final_output_sha256"] = _sha256_file(OUTPUT_PATH)
        audit["status"] = "complete"
        audit["completed_at_utc"] = attempt["finished_at_utc"]
    else:
        audit["status"] = "interrupted_or_failed"
    _save_audit(audit)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the one interrupted B4 true-LOBO run while treating only Linux kernel "
            "platform/release strings as non-binding."
        )
    )
    parser.add_argument("--n-jobs", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if type(args.n_jobs) is not int or args.n_jobs <= 0:
        raise ValueError("--n-jobs must be a positive integer")
    if not RUN_PATH.is_file():
        raise FileNotFoundError(f"saved interrupted run is missing: {RUN_PATH}")
    saved = load_strict(RUN_PATH)
    validate_run_identity(saved)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise KernelCompatibilityError("another kernel-compatible LOBO resume is active") from exc

        original_builder = base.build_run_identity
        attempt_index: int | None = None

        def compatible_builder(**kwargs):
            nonlocal attempt_index
            candidate = original_builder(**kwargs)
            report = require_kernel_only_compatibility(saved, candidate)
            attempt_index = _record_attempt(
                saved, candidate, report, n_jobs=args.n_jobs
            )
            print(
                "kernel_compatibility "
                f"saved_run_id={saved['run_id']} "
                f"ignored_fields={','.join(report['nonbinding_fields'])} "
                f"observed_drift={report['observed_drift']}",
                flush=True,
            )
            return copy.deepcopy(saved)

        base.build_run_identity = compatible_builder
        forwarded = [
            "--config", str(base.DEFAULT_CONFIG),
            "--cells", "A0,A1,A4",
            "--n-jobs", str(args.n_jobs),
            "--output", str(OUTPUT_PATH),
        ]
        try:
            result = base.main(forwarded)
        except BaseException as exc:
            if attempt_index is not None:
                _finish_attempt(attempt_index, status="failed", error=exc)
            raise
        finally:
            base.build_run_identity = original_builder
        if attempt_index is None:
            raise KernelCompatibilityError("production runner never built a candidate identity")
        _finish_attempt(attempt_index, status="complete")
        return result


if __name__ == "__main__":
    raise SystemExit(main())
