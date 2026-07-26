#!/usr/bin/env python3
"""Run the bounded real-corpus smoke for the stack class-coverage repair.

This is an exploratory diagnostic, not a confirmatory runner.  It evaluates the
fixed ``stylo_equal_channels_v1`` estimator on five pre-registered LOBO folds
selected independently of its results.  Confirmatory checkpoints, run plans,
statistics, and headline artifacts are never read as resumable state or written.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pathlib
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Mapping


WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPECTED_REPAIR_SHA = "07a8df82c48b62d88dd65bb262dbb3c4931b0500"
EXPECTED_AUDIT_CORPUS_DIGEST = (
    "15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029"
)
EXPECTED_CORPUS_MANIFEST_SELF_HASH = (
    "8d39132b8b7732af0d39112a4884947caa1125e24adae2281ab8f5f6d4287705"
)
EXPECTED_SEMANTIC_DIGEST = (
    "bd8720e2b4f7f8d4f49b13da02619993da82b528f45360f0da2054d7137246ad"
)
EXPECTED_DATASET_DIGEST = (
    "1c17d2c6e9623466bae200e2cfd3bd9c7a1535cc5ac65e9239b32623df124b9e"
)
EXPECTED_LOBO_MANIFEST_SELF_HASH = (
    "358bbfc069c782a33ba5ddee40c36c7005f8a0eed8f8ae38b6db673730932dbc"
)
EXPECTED_LOBO_MANIFEST_FILE_SHA256 = (
    "3bce8103189ba0665189700cee15168082a90511d7f9c7072e8fee2cd392935f"
)
EXPECTED_PROBABILITY_ORDER_DIGEST = (
    "d944f9f376283f6ee8202ce8bc9928cb3026b8406f134e32c80c67b0ffd2bb89"
)
EXPECTED_REPRESENTATION_RECEIPT_SELF_HASH = (
    "558f7b9fefc2e6661fbd264acb9e72f1bbd3471ffecc0a9a251f9bd6319f55b0"
)
EXPECTED_REPRESENTATION_CACHE_SHA256 = (
    "48c8e8494909e71944a52a7c45c3274ee2baca75e5650463929b75b6203d3348"
)
EXPECTED_COUNTS = {
    "chunks": 23_226,
    "works": 255,
    "authors": 47,
    "tested_works": 251,
    "tested_authors": 43,
}
PANEL = (
    (0, "akunun/azazel", 10),
    (15, "bulgakov/rokovie_yaytca", 7),
    (31, "chehov/дуэль", 13),
    (46, "dostoevsky/идиот", 5),
    (62, "garshin/медведи", 22),
)
SCHEMA = "stylo.stack_class_coverage_repair.smoke.v1"
MODEL_SPEC = "stylo_equal_channels_v1"
WEIGHTING = "chunk_weighted_legacy"


class SmokeError(RuntimeError):
    """The smoke input, mechanism, or result violated its frozen contract."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    return _json_hash({key: item for key, item in value.items() if key != "self_hash"})


def _registered_self_hash(value: Mapping[str, Any]) -> str:
    """Match the project's registered ``dumps_strict(..., sort_keys=True)`` hash."""
    body = {key: item for key, item in value.items() if key != "self_hash"}
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(payload: dict[str, Any], path: pathlib.Path) -> None:
    payload["updated_at"] = _utc_now()
    payload["self_hash"] = _self_hash(payload)
    data = (_canonical_json(payload) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise SmokeError(f"{path}: non-finite JSON constant {token}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SmokeError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    if not isinstance(value, dict):
        raise SmokeError(f"{path}: expected a JSON object")
    return value


def _git(code_root: pathlib.Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(code_root), *arguments],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _verify_environment() -> None:
    required = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    wrong = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if wrong or sys.flags.hash_randomization != 0:
        raise SmokeError(
            f"deterministic execution environment mismatch: values={wrong}, "
            f"hash_randomization={sys.flags.hash_randomization}"
        )


def _verify_output_path(path: pathlib.Path) -> pathlib.Path:
    path = path.resolve()
    expected_parent = (
        WORKSPACE_ROOT / "docs" / "exploratory" / "work_balanced"
    ).resolve()
    try:
        relative = path.relative_to(WORKSPACE_ROOT.resolve())
        path.relative_to(expected_parent)
    except ValueError as exc:
        raise SmokeError(
            f"output must stay below {expected_parent}, got {path}"
        ) from exc
    if path.suffix != ".json":
        raise SmokeError("smoke output must be a .json file")
    ignored = subprocess.run(
        ["git", "-C", str(WORKSPACE_ROOT), "check-ignore", "-q", "--", str(relative)],
        check=False,
    )
    if ignored.returncode != 0:
        raise SmokeError(f"smoke output is not ignored by git: {relative}")
    if "paired_audit_confirmatory" in path.as_posix():
        raise SmokeError("smoke output must not use the confirmatory namespace")
    return path


def _install_clean_repair(code_root: pathlib.Path) -> dict[str, Any]:
    code_root = code_root.resolve()
    observed_sha = _git(code_root, "rev-parse", "HEAD")
    if observed_sha != EXPECTED_REPAIR_SHA:
        raise SmokeError(
            f"repair checkout SHA {observed_sha} != frozen {EXPECTED_REPAIR_SHA}"
        )
    dirty = _git(code_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise SmokeError("repair execution checkout is not clean")
    source_root = code_root / "src"
    if not source_root.is_dir():
        raise SmokeError(f"repair source root is missing: {source_root}")
    sys.path.insert(0, str(source_root))
    source_paths = (
        "src/stylo/eval/calibration.py",
        "src/stylo/eval/lobo.py",
        "src/stylo/models/stacked_clf.py",
        "src/stylo/models/equal_channel_ensemble.py",
    )
    return {
        "git_commit": observed_sha,
        "git_dirty": False,
        "source_hashes": {
            relative: _sha256_file(code_root / relative) for relative in source_paths
        },
        "driver_sha256": _sha256_file(pathlib.Path(__file__).resolve()),
    }


def _verify_import_locations(code_root: pathlib.Path, modules) -> None:
    for module in modules:
        module_path = pathlib.Path(module.__file__).resolve()
        try:
            module_path.relative_to(code_root.resolve())
        except ValueError as exc:
            raise SmokeError(
                f"module {module.__name__} imported outside clean repair checkout: "
                f"{module_path}"
            ) from exc


def _verify_hashed_object(payload: Mapping[str, Any], expected: str, label: str) -> None:
    observed = _registered_self_hash(payload)
    if payload.get("self_hash") != expected or observed != expected:
        raise SmokeError(
            f"{label} self-hash mismatch: declared={payload.get('self_hash')}, "
            f"recomputed={observed}, expected={expected}"
        )


def _load_reference_checkpoint(
    checkpoint_root: pathlib.Path,
    model: str,
    fold_index: int,
    work_id: str,
    expected_manifest_digest: str,
    *,
    dumps_strict,
) -> dict[str, Any]:
    directory = checkpoint_root / "lobo" / model / "A0"
    matches = sorted(directory.glob(f"{fold_index:04d}-*.json"))
    if len(matches) != 1:
        raise SmokeError(
            f"expected exactly one {model} checkpoint for fold {fold_index}, "
            f"found {len(matches)}"
        )
    path = matches[0]
    record = _load_json(path)
    body = {key: value for key, value in record.items() if key != "self_hash"}
    observed_self_hash = hashlib.sha256(
        dumps_strict(body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if record.get("self_hash") != observed_self_hash:
        raise SmokeError(f"forensic checkpoint self-hash mismatch: {path}")
    expected_identity = {
        "dataset": "lobo",
        "model": model,
        "cell": "A0",
        "fold_index": fold_index,
        "work_id": work_id,
    }
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            raise SmokeError(
                f"{path}: {field}={record.get(field)!r} != {expected!r}"
            )
    if (
        record.get("bindings", {}).get("dataset_digest") != EXPECTED_DATASET_DIGEST
        or record.get("bindings", {}).get("fold_manifest_digest")
        != expected_manifest_digest
    ):
        raise SmokeError(f"{path}: forensic checkpoint binding mismatch")
    result = record.get("result")
    if not isinstance(result, dict):
        raise SmokeError(f"{path}: missing result")
    probability = result.get("probabilities")
    if not isinstance(probability, list) or len(probability) != EXPECTED_COUNTS["authors"]:
        raise SmokeError(f"{path}: malformed probability vector")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        for value in probability
    ):
        raise SmokeError(f"{path}: invalid probabilities")
    if not math.isclose(sum(probability), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SmokeError(f"{path}: probabilities are not normalized")
    true_label = result.get("true_label")
    pred_label = result.get("pred_label")
    expected_pred = max(range(len(probability)), key=probability.__getitem__)
    expected_rank = sum(value >= probability[true_label] for value in probability)
    if (
        pred_label != expected_pred
        or result.get("rank") != expected_rank
        or result.get("correct") != (pred_label == true_label)
    ):
        raise SmokeError(f"{path}: internally inconsistent result")
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "self_hash": record["self_hash"],
        "run_id": record["run_id"],
        "result": {
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": result["correct"],
            "rank": result["rank"],
            "probabilities": probability,
        },
    }


def _prepare_inputs(args, code_identity):
    import numpy as np

    from stylo.config import load_config
    from stylo.eval.paired_audit import corpus as audit_corpus
    from stylo.eval.paired_audit import manifest as fold_manifest
    from stylo.eval.paired_audit import run_plan
    from stylo.eval import lobo
    from stylo.jsonio import dumps_strict

    _verify_import_locations(
        args.code_root,
        (audit_corpus, fold_manifest, run_plan, lobo),
    )

    data_root = args.data_root.resolve()
    cfg = load_config(
        args.code_root / "configs" / "default.yaml",
        overrides={"paths.data": str(data_root)},
    )
    published_root = audit_corpus.resolve_current_root(data_root / "audit_corpus")
    if published_root.name != EXPECTED_AUDIT_CORPUS_DIGEST:
        raise SmokeError(
            f"audit corpus digest {published_root.name} != "
            f"{EXPECTED_AUDIT_CORPUS_DIGEST}"
        )
    corpus_manifest_path = published_root / audit_corpus.CORPUS_MANIFEST_NAME
    corpus_manifest = _load_json(corpus_manifest_path)
    _verify_hashed_object(
        corpus_manifest,
        EXPECTED_CORPUS_MANIFEST_SELF_HASH,
        "audit corpus manifest",
    )
    if (
        corpus_manifest.get("audit_corpus_digest") != EXPECTED_AUDIT_CORPUS_DIGEST
        or corpus_manifest.get("n_chunks") != EXPECTED_COUNTS["chunks"]
        or corpus_manifest.get("n_works") != EXPECTED_COUNTS["works"]
    ):
        raise SmokeError("audit corpus manifest count/digest drift")

    dataset = audit_corpus.load_audit_dataset(published_root, cfg)
    semantic_digest = audit_corpus.verify_audit_dataset(dataset)
    if semantic_digest != EXPECTED_SEMANTIC_DIGEST:
        raise SmokeError(
            f"semantic digest {semantic_digest} != {EXPECTED_SEMANTIC_DIGEST}"
        )
    if dataset.provenance.rows_digest != EXPECTED_DATASET_DIGEST:
        raise SmokeError(
            f"dataset digest {dataset.provenance.rows_digest} != "
            f"{EXPECTED_DATASET_DIGEST}"
        )

    groups = np.asarray(dataset.groups, dtype=object)
    labels = np.asarray(dataset.y)
    if (
        len(groups) != EXPECTED_COUNTS["chunks"]
        or len(set(groups.tolist())) != EXPECTED_COUNTS["works"]
        or len(dataset.authors) != EXPECTED_COUNTS["authors"]
        or sorted(set(int(value) for value in labels))
        != list(range(EXPECTED_COUNTS["authors"]))
    ):
        raise SmokeError("loaded audit dataset universe drift")
    for group, label in zip(groups, labels, strict=True):
        if str(group).split("/", 1)[0] != dataset.authors[int(label)]:
            raise SmokeError(f"dataset label/work mismatch at {group!r}")

    manifest_path = args.fold_manifest.resolve()
    if _sha256_file(manifest_path) != EXPECTED_LOBO_MANIFEST_FILE_SHA256:
        raise SmokeError("LOBO manifest file SHA-256 drift")
    manifest = _load_json(manifest_path)
    fold_manifest.verify_manifest_self_hash(manifest)
    fold_manifest.assert_lobo_universe(manifest)
    fold_manifest.assert_manifest_consistent_with_dataset(manifest, dataset)
    rebuilt = fold_manifest.build_fold_manifest(
        "lobo",
        dataset,
        parent_dataset_digest=manifest["parent_dataset_digest"],
        algorithm=fold_manifest.REGISTERED_ALGORITHM["lobo"],
        seed=fold_manifest.REGISTERED_SEED,
        config_hash=manifest["config_hash"],
    )
    fold_manifest.verify_manifest_matches_rebuilt(manifest, rebuilt, universe=True)
    if (
        manifest["self_hash"] != EXPECTED_LOBO_MANIFEST_SELF_HASH
        or manifest["dataset_digest"] != EXPECTED_DATASET_DIGEST
        or run_plan.class_order_digest(manifest["probability_class_order"])
        != EXPECTED_PROBABILITY_ORDER_DIGEST
    ):
        raise SmokeError("LOBO manifest binding drift")

    tested_by_fold = {
        row["fold_index"]: row
        for row in manifest["works"]
        if row["tested"]
    }
    for fold_index, work_id, _old_rank in PANEL:
        if tested_by_fold.get(fold_index, {}).get("work_id") != work_id:
            raise SmokeError(
                f"frozen smoke fold {fold_index} is not {work_id!r}"
            )

    receipt_path = args.representation_receipt.resolve()
    receipt = _load_json(receipt_path)
    _verify_hashed_object(
        receipt,
        EXPECTED_REPRESENTATION_RECEIPT_SELF_HASH,
        "representation-cache receipt",
    )
    if (
        receipt.get("status") != "PASS"
        or receipt.get("corpus", {}).get("content_digest")
        != EXPECTED_AUDIT_CORPUS_DIGEST
        or receipt.get("datasets", {}).get("lobo", {}).get("dataset_digest")
        != EXPECTED_DATASET_DIGEST
    ):
        raise SmokeError("representation-cache receipt binding drift")
    cache_path = args.representation_cache.resolve()
    cache_sha = _sha256_file(cache_path)
    if cache_sha != EXPECTED_REPRESENTATION_CACHE_SHA256:
        raise SmokeError(
            f"representation cache SHA {cache_sha} != "
            f"{EXPECTED_REPRESENTATION_CACHE_SHA256}"
        )

    references = {}
    for fold_index, work_id, expected_old_rank in PANEL:
        old_stack = _load_reference_checkpoint(
            args.forensic_checkpoint_root,
            "stylo_stack",
            fold_index,
            work_id,
            manifest["self_hash"],
            dumps_strict=dumps_strict,
        )
        core = _load_reference_checkpoint(
            args.forensic_checkpoint_root,
            "stylo",
            fold_index,
            work_id,
            manifest["self_hash"],
            dumps_strict=dumps_strict,
        )
        if old_stack["result"]["correct"] or old_stack["result"]["rank"] != expected_old_rank:
            raise SmokeError(
                f"broken-stack reference drift at fold {fold_index}"
            )
        if not core["result"]["correct"] or core["result"]["rank"] != 1:
            raise SmokeError(f"core-stylo reference drift at fold {fold_index}")
        references[str(fold_index)] = {
            "broken_stack": old_stack,
            "core_stylo": core,
        }

    input_binding = {
        "code": code_identity,
        "corpus": {
            "root": str(published_root),
            "content_digest": published_root.name,
            "manifest_path": str(corpus_manifest_path.resolve()),
            "manifest_self_hash": corpus_manifest["self_hash"],
            "semantic_digest": semantic_digest,
        },
        "dataset": {
            "rows_digest": dataset.provenance.rows_digest,
            **EXPECTED_COUNTS,
        },
        "fold_manifest": {
            "path": str(manifest_path),
            "file_sha256": _sha256_file(manifest_path),
            "self_hash": manifest["self_hash"],
            "probability_class_order_digest": EXPECTED_PROBABILITY_ORDER_DIGEST,
        },
        "representation_cache": {
            "path": str(cache_path),
            "file_sha256": cache_sha,
            "receipt_path": str(receipt_path),
            "receipt_self_hash": receipt["self_hash"],
        },
    }
    return cfg, dataset, manifest, references, input_binding, lobo


def _run_legacy_negative_control(cfg, dataset, lobo_module) -> dict[str, Any]:
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY
    from stylo.models.stacked_clf import StackClassCoverageError

    fold_index, work_id, _old_rank = PANEL[0]
    factory = lobo_module.make_factory(
        "stylo_stack",
        cfg,
        weighting=CHUNK_WEIGHTED_LEGACY,
    )
    estimator_box = {}

    def capture_factory():
        estimator = factory()
        estimator_box["estimator"] = estimator
        return estimator

    try:
        lobo_module.run_fold(
            dataset.texts,
            dataset.y,
            dataset.groups,
            len(dataset.authors),
            list(dataset.authors),
            work_id,
            capture_factory,
            5,
        )
    except StackClassCoverageError as exc:
        report = exc.report
        estimator = estimator_box.get("estimator")
        forbidden_fitted_state = {
            name: hasattr(estimator, name)
            for name in (
                "mode_",
                "meta_",
                "meta_split_preflight_",
                "_calibrators",
            )
        }
        passed = (
            exc.stage == "inner_oof"
            and isinstance(report, dict)
            and report.get("complete") is False
            and report.get("class_coverage_complete") is False
            and not any(forbidden_fitted_state.values())
        )
        result = {
            "fold_index": fold_index,
            "work_id": work_id,
            "expected_exception": "StackClassCoverageError",
            "observed_exception": type(exc).__name__,
            "stage": exc.stage,
            "preflight": {
                key: report.get(key)
                for key in (
                    "expected_split_count",
                    "split_count",
                    "incomplete_split_count",
                    "split_count_complete",
                    "validation_exactly_once",
                    "class_coverage_complete",
                    "structure_complete",
                    "complete",
                )
            },
            "missing_train_classes_by_split": [
                item.get("missing_train_classes") for item in report.get("splits", [])
            ],
            "forbidden_fitted_state_present": forbidden_fitted_state,
            "passed": passed,
        }
        if not passed:
            raise SmokeError(f"legacy negative control failed: {result}")
        return result
    raise SmokeError(
        "legacy stylo_stack unexpectedly produced a prediction instead of "
        "failing closed at inner_oof"
    )


def _mechanism_passport_checks(passport: Any, estimator: Any) -> dict[str, Any]:
    if not isinstance(passport, dict):
        return {"passed": False, "reason": "fusion passport is not an object"}
    expected_channels = [
        "char (2-5)",
        "word (1-2)",
        "syntax (dep+pos+syn)",
        "dependency",
        "function_words",
        "morphology",
    ]
    channels = passport.get("channels")
    weights = passport.get("fusion", {}).get("weights")
    expected_weight = 1.0 / 6.0
    checks = {
        "schema": passport.get("schema")
        == "stylo.equal_channel_ensemble.fusion.v1",
        "estimator_spec": passport.get("estimator_spec") == MODEL_SPEC,
        "exact_six_channels": channels == expected_channels,
        "legacy_axes": passport.get("axes") == {"W": False, "F": False, "R": False},
        "legacy_training_weighting": passport.get("training_weighting") == WEIGHTING,
        "identity_softmax": passport.get("channel_score_transform")
        == {"method": "identity_softmax", "temperature": 1.0, "learned": False},
        "equal_mean": passport.get("fusion", {}).get("method")
        == "equal_arithmetic_mean",
        "fusion_not_learned": passport.get("fusion", {}).get("learned") is False,
        "equal_weights": (
            isinstance(weights, dict)
            and isinstance(channels, list)
            and set(weights) == set(channels)
            and all(
                math.isclose(
                    float(weights[name]),
                    expected_weight,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for name in channels
            )
        ),
        "no_oof": passport.get("oof") == {"used": False},
        "no_calibration": passport.get("calibration") == {"learned": False},
        "no_meta": passport.get("meta_classifier") == {"present": False},
        "no_legacy_passport": not hasattr(estimator, "passport_"),
        "no_mode": not hasattr(estimator, "mode_"),
        "no_fitted_meta": not hasattr(estimator, "meta_"),
        "no_meta_preflight": not hasattr(estimator, "meta_split_preflight_"),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _evaluate_equal_fold(
    cfg,
    dataset,
    lobo_module,
    fold_index: int,
    work_id: str,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY

    groups = np.asarray(dataset.groups, dtype=object)
    labels = np.asarray(dataset.y)
    heldout = groups == work_id
    train_groups = set(groups[~heldout].tolist())
    full_groups = set(groups.tolist())
    split_checks = {
        "heldout_chunk_count": int(heldout.sum()),
        "heldout_work_absent_from_train": work_id not in train_groups,
        "train_is_exact_work_complement": train_groups
        == full_groups - {work_id},
        "train_retains_all_47_classes": sorted(
            set(int(value) for value in labels[~heldout].tolist())
        )
        == list(range(EXPECTED_COUNTS["authors"])),
    }
    if not all(
        value is True
        for key, value in split_checks.items()
        if key != "heldout_chunk_count"
    ):
        raise SmokeError(f"{work_id}: outer split integrity failed")

    base_factory = lobo_module.make_factory(
        MODEL_SPEC,
        cfg,
        weighting=CHUNK_WEIGHTED_LEGACY,
    )
    estimator_box = {}

    def capture_factory():
        estimator = base_factory()
        estimator_box["estimator"] = estimator
        return estimator

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        row = lobo_module.run_fold(
            dataset.texts,
            dataset.y,
            dataset.groups,
            len(dataset.authors),
            list(dataset.authors),
            work_id,
            capture_factory,
            5,
        )
    elapsed = time.perf_counter() - started
    if row is None:
        raise SmokeError(f"{work_id}: equal-channel fold returned no result")
    estimator = estimator_box["estimator"]
    probability = np.asarray(row.pop("_prob"), dtype=np.float64)
    true_label = int(labels[heldout][0])
    pred_label = int(np.argmax(probability))
    rank = int(np.sum(probability >= probability[true_label]))
    correct = pred_label == true_label
    classes = np.asarray(estimator.classes_)
    probability_checks = {
        "shape_47": probability.shape == (EXPECTED_COUNTS["authors"],),
        "classes_0_to_46": (
            classes.ndim == 1
            and classes.tolist() == list(range(EXPECTED_COUNTS["authors"]))
        ),
        "finite": bool(np.isfinite(probability).all()),
        "nonnegative": bool((probability >= 0.0).all()),
        "normalized_atol_1e_12": bool(
            np.allclose(probability.sum(), 1.0, rtol=0.0, atol=1e-12)
        ),
        "true_label_recomputed": int(row["true_label"]) == true_label,
        "pred_label_recomputed": int(row["pred_label"]) == pred_label,
        "rank_recomputed": int(row["rank"]) == rank,
        "correct_recomputed": bool(row["correct"]) == correct,
    }
    mechanism = _mechanism_passport_checks(
        getattr(estimator, "fusion_passport_", None),
        estimator,
    )
    warning_rows = [
        {
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in captured
    ]
    convergence_warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in captured
    )
    mechanical_pass = (
        all(probability_checks.values())
        and mechanism["passed"]
        and convergence_warning_count == 0
    )
    if not mechanical_pass:
        raise SmokeError(
            f"{work_id}: new estimator mechanical gate failed: "
            f"probability={probability_checks}, mechanism={mechanism}, "
            f"convergence_warnings={convergence_warning_count}"
        )

    other_max = float(np.max(np.delete(probability, true_label)))
    result = {
        "fold_index": fold_index,
        "work_id": work_id,
        "true_author": dataset.authors[true_label],
        "pred_author": dataset.authors[pred_label],
        "result": {
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": correct,
            "rank": rank,
            "probabilities": probability.tolist(),
        },
        "diagnostics": {
            "true_class_probability": float(probability[true_label]),
            "true_class_margin": float(probability[true_label] - other_max),
            "entropy_nats": float(
                -np.sum(
                    probability[probability > 0.0]
                    * np.log(probability[probability > 0.0])
                )
            ),
            "wall_seconds": float(elapsed),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "split_checks": split_checks,
        "probability_checks": probability_checks,
        "fusion_passport": estimator.fusion_passport_,
        "mechanism_checks": mechanism,
        "warnings": warning_rows,
        "convergence_warning_count": convergence_warning_count,
        "mechanical_pass": mechanical_pass,
    }
    del estimator_box["estimator"]
    del estimator
    gc.collect()
    return result


def _validate_resumed_fold(row: Mapping[str, Any], fold_index: int, work_id: str) -> None:
    if (
        row.get("fold_index") != fold_index
        or row.get("work_id") != work_id
        or row.get("mechanical_pass") is not True
    ):
        raise SmokeError(f"saved fold {fold_index} identity/mechanical gate mismatch")
    result = row.get("result")
    probability = result.get("probabilities") if isinstance(result, dict) else None
    if not isinstance(probability, list) or len(probability) != EXPECTED_COUNTS["authors"]:
        raise SmokeError(f"saved fold {fold_index} probability vector malformed")
    if (
        any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0
            for value in probability)
        or not math.isclose(sum(probability), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise SmokeError(f"saved fold {fold_index} probabilities invalid")
    true_label = result.get("true_label")
    pred_label = max(range(len(probability)), key=probability.__getitem__)
    rank = sum(value >= probability[true_label] for value in probability)
    if (
        result.get("pred_label") != pred_label
        or result.get("rank") != rank
        or result.get("correct") != (pred_label == true_label)
    ):
        raise SmokeError(f"saved fold {fold_index} result is inconsistent")


def _contract() -> dict[str, Any]:
    return {
        "scope": {
            "kind": "exploratory_real_corpus_smoke",
            "model": MODEL_SPEC,
            "cell": "A0",
            "training_weighting": WEIGHTING,
            "seed": 42,
            "svc_c": 1.0,
            "confirmatory_execution": False,
            "headline_decision": False,
        },
        "selection": {
            "rule": (
                "five equally spaced order statistics of the pre-existing "
                "interrupted forensic prefix 0..62"
            ),
            "rule_formula": "floor(j*(63-1)/4), j=0..4",
            "selected_before_new_results": True,
            "folds": [
                {"fold_index": index, "work_id": work_id}
                for index, work_id, _rank in PANEL
            ],
            "staging": "first fold, then remaining four only after mechanical PASS",
            "replacement_or_retuning_allowed": False,
        },
        "reference_panel": {
            "core_stylo": {"top1": 5, "n": 5, "ranks": [1, 1, 1, 1, 1]},
            "broken_stack": {
                "top1": 0,
                "n": 5,
                "ranks": [rank for _index, _work_id, rank in PANEL],
            },
        },
        "mechanical_gate": {
            "legacy_stack": "StackClassCoverageError(stage=inner_oof)",
            "probability_width": 47,
            "normalization_atol": 1e-12,
            "class_order": "0..46",
            "fusion": "six identity-softmax channels, equal 1/6 arithmetic mean",
            "learned_oof_calibration_meta": False,
            "convergence_warnings_allowed": 0,
        },
        "diagnostic_gate": {
            "required_completed_folds": 5,
            "minimum_top1": 4,
            "minimum_top2": 4,
            "required_median_rank": 1,
            "maximum_rank": 3,
            "minimum_paired_top1_gains_vs_broken": 4,
            "maximum_paired_top1_losses_vs_broken": 0,
            "maximum_top1_deficits_vs_core": 1,
        },
        "excluded_inferences": [
            "full-corpus accuracy estimate",
            "macro-F1",
            "ECE",
            "confidence interval",
            "p-value",
            "confirmatory claim",
            "headline decision",
        ],
        "pass_consequence": (
            "licenses a full exploratory LOBO evaluation only; it does not "
            "resume or alter confirmatory execution"
        ),
    }


def _build_decision(
    rows: list[dict[str, Any]],
    references: Mapping[str, Any],
    *,
    complete: bool,
) -> dict[str, Any]:
    ranks = [int(row["result"]["rank"]) for row in rows]
    top1 = sum(bool(row["result"]["correct"]) for row in rows)
    top2 = sum(rank <= 2 for rank in ranks)
    gains = 0
    losses = 0
    rank_improvements = 0
    core_deficits = 0
    for row in rows:
        reference = references[str(row["fold_index"])]
        new_correct = bool(row["result"]["correct"])
        old_correct = bool(reference["broken_stack"]["result"]["correct"])
        core_correct = bool(reference["core_stylo"]["result"]["correct"])
        gains += int(new_correct and not old_correct)
        losses += int(old_correct and not new_correct)
        core_deficits += int(core_correct and not new_correct)
        rank_improvements += int(
            int(row["result"]["rank"])
            < int(reference["broken_stack"]["result"]["rank"])
        )
    metrics = {
        "completed_folds": len(rows),
        "top1_correct": top1,
        "top2_correct": top2,
        "ranks": ranks,
        "median_rank": statistics.median(ranks) if ranks else None,
        "maximum_rank": max(ranks) if ranks else None,
        "paired_top1_gains_vs_broken": gains,
        "paired_top1_losses_vs_broken": losses,
        "rank_improvements_vs_broken": rank_improvements,
        "top1_deficits_vs_core": core_deficits,
        "total_wall_seconds": sum(
            float(row["diagnostics"]["wall_seconds"]) for row in rows
        ),
    }
    if not complete:
        return {
            "status": "PARTIAL_MECHANICAL_PASS",
            "metrics": metrics,
            "diagnostic_gate_evaluated": False,
            "reason": "the pre-registered five-fold panel is incomplete",
        }
    checks = {
        "all_five_completed": len(rows) == 5,
        "top1_at_least_4_of_5": top1 >= 4,
        "top2_at_least_4_of_5": top2 >= 4,
        "median_rank_is_1": statistics.median(ranks) == 1,
        "maximum_rank_at_most_3": max(ranks) <= 3,
        "at_least_four_paired_top1_gains": gains >= 4,
        "no_paired_top1_losses": losses == 0,
        "at_most_one_top1_deficit_vs_core": core_deficits <= 1,
        "every_rank_improves_vs_broken": rank_improvements == 5,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "metrics": metrics,
        "checks": checks,
        "diagnostic_gate_evaluated": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=WORKSPACE_ROOT / "data",
    )
    parser.add_argument(
        "--fold-manifest",
        type=pathlib.Path,
        default=(
            WORKSPACE_ROOT
            / "data"
            / "paired_audit_preparation"
            / "ca8523f77173b1285546823fc20e25d4fbbd7a076c9b13db875c4319f3c21ce7"
            / "lobo_fold_manifest_v1.json"
        ),
    )
    parser.add_argument(
        "--forensic-checkpoint-root",
        type=pathlib.Path,
        default=(
            WORKSPACE_ROOT
            / "data"
            / "paired_audit_confirmatory_checkpoints"
            / "c24b79b3"
        ),
    )
    parser.add_argument(
        "--representation-receipt",
        type=pathlib.Path,
        default=(
            WORKSPACE_ROOT
            / "data"
            / "paired_audit_representation_cache"
            / EXPECTED_REPRESENTATION_RECEIPT_SELF_HASH
            / "representation_cache_receipt.json"
        ),
    )
    parser.add_argument(
        "--representation-cache",
        type=pathlib.Path,
        default=(
            WORKSPACE_ROOT
            / "data"
            / "reps_ru_core_news_lg_3.8.0_f7be72cf9a96.pkl"
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            WORKSPACE_ROOT
            / "docs"
            / "exploratory"
            / "work_balanced"
            / "stylo_equal_channels_repair_smoke_v1.json"
        ),
    )
    parser.add_argument(
        "--max-new-folds",
        type=int,
        default=0,
        help="0 means all missing folds; use 1 for the registered first stage",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    args.code_root = args.code_root.resolve()
    args.data_root = args.data_root.resolve()
    args.forensic_checkpoint_root = args.forensic_checkpoint_root.resolve()
    if args.max_new_folds < 0:
        raise SmokeError("--max-new-folds must be non-negative")
    _verify_environment()
    output = _verify_output_path(args.output)
    code_identity = _install_clean_repair(args.code_root)

    cfg, dataset, manifest, references, input_binding, lobo_module = _prepare_inputs(
        args,
        code_identity,
    )
    contract = _contract()

    if output.exists():
        if not args.resume:
            raise SmokeError(
                f"output exists; pass --resume after inspecting it: {output}"
            )
        artifact = _load_json(output)
        if artifact.get("self_hash") != _self_hash(artifact):
            raise SmokeError("existing smoke artifact self-hash mismatch")
        if (
            artifact.get("schema") != SCHEMA
            or artifact.get("contract") != contract
            or artifact.get("input_binding") != input_binding
            or artifact.get("references") != references
        ):
            raise SmokeError("existing smoke artifact identity/binding mismatch")
    else:
        artifact = {
            "schema": SCHEMA,
            "status": "running",
            "started_at": _utc_now(),
            "contract": contract,
            "input_binding": input_binding,
            "references": references,
            "legacy_negative_control": None,
            "fold_results": [],
            "decision": None,
            "confirmatory_namespace_written": False,
        }
        _atomic_write(artifact, output)

    if artifact["legacy_negative_control"] is None:
        print(
            _canonical_json(
                {
                    "event": "legacy_negative_control_started",
                    "fold_index": 0,
                    "work_id": PANEL[0][1],
                }
            ),
            flush=True,
        )
        artifact["legacy_negative_control"] = _run_legacy_negative_control(
            cfg,
            dataset,
            lobo_module,
        )
        _atomic_write(artifact, output)
        print(
            _canonical_json(
                {
                    "event": "legacy_negative_control_passed",
                    "stage": artifact["legacy_negative_control"]["stage"],
                }
            ),
            flush=True,
        )
    elif artifact["legacy_negative_control"].get("passed") is not True:
        raise SmokeError("saved legacy negative control is not PASS")

    saved_by_fold = {
        int(row["fold_index"]): row for row in artifact["fold_results"]
    }
    if len(saved_by_fold) != len(artifact["fold_results"]):
        raise SmokeError("existing smoke artifact contains duplicate fold results")
    for fold_index, work_id, _old_rank in PANEL:
        if fold_index in saved_by_fold:
            _validate_resumed_fold(saved_by_fold[fold_index], fold_index, work_id)

    missing = [
        (fold_index, work_id)
        for fold_index, work_id, _old_rank in PANEL
        if fold_index not in saved_by_fold
    ]
    if args.max_new_folds:
        missing = missing[: args.max_new_folds]
    for fold_index, work_id in missing:
        print(
            _canonical_json(
                {
                    "event": "equal_channel_fold_started",
                    "fold_index": fold_index,
                    "work_id": work_id,
                }
            ),
            flush=True,
        )
        try:
            fold_result = _evaluate_equal_fold(
                cfg,
                dataset,
                lobo_module,
                fold_index,
                work_id,
            )
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["failure"] = {
                "fold_index": fold_index,
                "work_id": work_id,
                "exception": type(exc).__name__,
                "message": str(exc),
            }
            _atomic_write(artifact, output)
            raise
        artifact["fold_results"].append(fold_result)
        artifact["fold_results"].sort(key=lambda row: int(row["fold_index"]))
        _atomic_write(artifact, output)
        print(
            _canonical_json(
                {
                    "event": "equal_channel_fold_completed",
                    "fold_index": fold_index,
                    "work_id": work_id,
                    "correct": fold_result["result"]["correct"],
                    "rank": fold_result["result"]["rank"],
                    "wall_seconds": fold_result["diagnostics"]["wall_seconds"],
                }
            ),
            flush=True,
        )

    complete = len(artifact["fold_results"]) == len(PANEL)
    artifact["decision"] = _build_decision(
        artifact["fold_results"],
        references,
        complete=complete,
    )
    artifact["status"] = (
        "passed"
        if artifact["decision"]["status"] == "PASS"
        else "partial_pass"
        if artifact["decision"]["status"] == "PARTIAL_MECHANICAL_PASS"
        else "failed"
    )
    artifact.pop("failure", None)
    _atomic_write(artifact, output)
    print(
        _canonical_json(
            {
                "event": "smoke_stage_completed",
                "artifact": str(output),
                "status": artifact["status"],
                "decision": artifact["decision"],
                "self_hash": artifact["self_hash"],
            }
        ),
        flush=True,
    )
    return 0 if artifact["status"] in {"passed", "partial_pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
