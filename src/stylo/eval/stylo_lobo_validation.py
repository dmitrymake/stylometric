"""Resumable 47-class true-LOBO validation for the signed stylo cells.

This module is deliberately narrower than the future confirmatory paired-audit control plane. It evaluates
only ``stylo`` A0/A4/A1 on one disk-verified legacy corpus, keeps the four singleton authors in
every train fold, gates all non-A0 work on exact parity with the pinned legacy book report, and
stores one ordinary atomic checkpoint per held-out work.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import pathlib
import re
import resource
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from ..jsonio import (
    artifact_self_hash,
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
)
from ..lang import display_name
from .dispatch import fit_estimator
from .lobo import make_factory_for_ablation, run_fold
from .metrics import accuracy, macro_f1, topk_accuracy
from .provenance import (
    reverify_scientific_context_from_disk,
    require_disk_verified_scientific_context,
    require_scientific_evaluation_context,
)
from .significance import paired_bootstrap_diff_clustered
from .work_weighting import FULL_WB_ABLATION, LEGACY_ABLATION, WEIGHTS_ONLY_ABLATION


STATUS = "true_lobo_target_protocol_validation_not_external_replication"
SCHEMA_VERSION = "b4_true_lobo_a0_a1_a4_v2"
CHECKPOINT_SCHEMA_VERSION = "b4_true_lobo_checkpoint_v2"
LEGACY_RUN_SCHEMA_VERSION = "b4_true_lobo_run_v1"
OLDER_RUN_SCHEMA_VERSION = "stylo_lobo_validation_run_v2"
PREVIOUS_RUN_SCHEMA_VERSION = "stylo_lobo_validation_run_v3"
RUN_SCHEMA_VERSION = "stylo_lobo_validation_run_v4"
GATE_SCHEMA_VERSION = "b4_true_lobo_a0_gate_v2"
LEGACY_RUNTIME_SCHEMA_VERSION = "b4_true_lobo_runtime_v1"
RUNTIME_SCHEMA_VERSION = "stylo_lobo_runtime_v2"
DISK_AUTHORITY_MODE = "disk_verified"
SYNTHETIC_AUTHORITY_MODE = "synthetic_test"
PRODUCTION_EVALUATOR_ID = (
    "stylo.eval.stylo_lobo_validation.evaluate_cell_fold"
)
SYNTHETIC_EVALUATOR_ID = "injected_synthetic_test_evaluator"

try:
    import fcntl
except ImportError:  # pragma: no cover - canonical runner is POSIX
    fcntl = None

RUNTIME_BINDING_FIELDS = frozenset({
    "python",
    "python_implementation",
    "python_compiler",
    "system",
    "machine",
    "processor",
    "libc",
    "numpy",
    "scipy",
    "sklearn",
    "spacy",
    "joblib",
    "threadpoolctl",
    "spacy_model",
})
REFERENCE_SHA256 = "26db64475e77657eaec6db895c55bad8bcd513344584ef5a64e9a580cf9f648d"
REFERENCE_CORRECT = 221
REFERENCE_TESTED = 251
REFERENCE_ACCURACY = 0.8804780876494024

EXPECTED_AUTHORS = 47
EXPECTED_WORKS = 255
EXPECTED_TESTED_AUTHORS = 43
EXPECTED_TESTED_WORKS = 251
EXPECTED_SINGLETON_AUTHORS = ("goncharov", "grigorovich", "reshetnikov", "voloshin")

# Canonical scientific/scheduling order: signed primary A4 is completed before secondary A1.
CELL_ORDER = ("A0", "A4", "A1")
MAX_TRUE_LOBO_WORKERS = 8
CELL_ABLATIONS = {
    "A0": LEGACY_ABLATION,
    "A4": FULL_WB_ABLATION,
    "A1": WEIGHTS_ONLY_ABLATION,
}


class TrueLoboError(ValueError):
    """The target-protocol run, checkpoint set, or result is internally inconsistent."""


class A0ParityError(TrueLoboError):
    """The frozen A0 result did not reproduce the pinned 221/251 reference exactly."""


class CheckpointError(TrueLoboError):
    """A checkpoint directory or checkpoint payload cannot be safely resumed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _axes(cell: str) -> dict[str, bool]:
    try:
        ablation = CELL_ABLATIONS[cell]
    except KeyError as exc:
        raise TrueLoboError(f"unsupported true-LOBO cell {cell!r}") from exc
    return {
        "weights": bool(ablation.weights),
        "feature_fit": bool(ablation.feature_fit),
        "relative_fw": bool(ablation.relative_fw),
    }


def _checked_label(value: Any, n_authors: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TrueLoboError(f"{field} must be a non-bool integer")
    label = int(value)
    if not 0 <= label < n_authors:
        raise TrueLoboError(f"{field}={label} outside [0,{n_authors})")
    return label


def derive_inventory(dataset, *, enforce_target: bool = False) -> dict[str, Any]:
    """Build the one immutable 47-class work universe and its tested 43-class subset.

    Labels are never compacted: the metric order is expressed in the original full-universe label
    space, so the four singleton authors remain valid prediction targets without entering macro-F1.
    """
    dataset = require_scientific_evaluation_context(dataset)
    texts = np.asarray(dataset.texts, dtype=object)
    y = np.asarray(dataset.y)
    groups = np.asarray(dataset.groups, dtype=object)
    authors = list(dataset.authors)
    if not (len(texts) == len(y) == len(groups)) or not len(texts):
        raise TrueLoboError("dataset texts/y/groups must have one non-empty common length")
    if len(set(authors)) != len(authors) or not all(type(author) is str for author in authors):
        raise TrueLoboError("dataset authors must be unique exact strings")

    work_labels: dict[str, int] = {}
    work_chunks: Counter[str] = Counter()
    for index, (group_value, label_value) in enumerate(zip(groups, y, strict=True)):
        if type(group_value) is not str or "/" not in group_value:
            raise TrueLoboError(f"groups[{index}] is not an exact author/work string")
        label = _checked_label(label_value, len(authors), field=f"y[{index}]")
        author = group_value.split("/", 1)[0]
        if authors[label] != author:
            raise TrueLoboError(
                f"groups[{index}] author {author!r} != authors[y] {authors[label]!r}")
        prior = work_labels.setdefault(group_value, label)
        if prior != label:
            raise TrueLoboError(f"work {group_value!r} has multiple truth labels")
        work_chunks[group_value] += 1

    ordered_work_ids = sorted(work_labels)
    works_per_author = Counter(work_id.split("/", 1)[0] for work_id in ordered_work_ids)
    work_universe = []
    tested_inventory = []
    for work_index, work_id in enumerate(ordered_work_ids):
        label = work_labels[work_id]
        item = {
            "work_index": work_index,
            "work_id": work_id,
            "true_label": label,
            "true_author": authors[label],
            "n_chunks": int(work_chunks[work_id]),
        }
        work_universe.append(item)
        if works_per_author[item["true_author"]] >= 2:
            tested_inventory.append({**item, "fold_index": len(tested_inventory)})

    metric_labels = [
        label for label, author in enumerate(authors) if works_per_author[author] >= 2
    ]
    singleton_labels = [
        label for label, author in enumerate(authors) if works_per_author[author] == 1
    ]
    singleton_work_ids = [
        item["work_id"] for item in work_universe if item["true_label"] in singleton_labels
    ]
    if any(works_per_author[authors[label]] < 2 for label in metric_labels):
        raise TrueLoboError("a tested author would disappear from one of its train folds")

    inventory = {
        "n_chunks": int(len(texts)),
        "probability_class_order": [
            {"label": label, "author": author} for label, author in enumerate(authors)
        ],
        "metric_label_order": [
            {"label": label, "author": authors[label]} for label in metric_labels
        ],
        "work_universe": work_universe,
        "tested_inventory": tested_inventory,
        "singleton_train_only": [
            {
                "label": label,
                "author": authors[label],
                "work_id": next(
                    item["work_id"] for item in work_universe if item["true_label"] == label
                ),
            }
            for label in singleton_labels
        ],
    }
    inventory["probability_order_sha256"] = canonical_hash(
        inventory["probability_class_order"])
    inventory["metric_order_sha256"] = canonical_hash(inventory["metric_label_order"])
    inventory["work_universe_sha256"] = canonical_hash(inventory["work_universe"])
    inventory["tested_inventory_sha256"] = canonical_hash(inventory["tested_inventory"])

    if enforce_target:
        observed = (
            len(authors), len(work_universe), len(metric_labels), len(tested_inventory),
            tuple(item["author"] for item in inventory["singleton_train_only"]),
        )
        expected = (
            EXPECTED_AUTHORS, EXPECTED_WORKS, EXPECTED_TESTED_AUTHORS,
            EXPECTED_TESTED_WORKS, EXPECTED_SINGLETON_AUTHORS,
        )
        if observed != expected:
            raise TrueLoboError(f"target inventory drift: expected {expected!r}, got {observed!r}")
    return inventory


_REFERENCE_HEADER = "=== LOBO: топ-кандидаты по каждой книге (leakage-free) ==="
_REFERENCE_ROW_RE = re.compile(
    r"^\[(OK  |MISS)\] (.+?) / (.+?)  \(rank истинного автора: ([1-9][0-9]*)\)$")
_REFERENCE_TOP_RE = re.compile(r"^        топ: (.+)$")
_REFERENCE_CANDIDATE_RE = re.compile(r"^(.+?) \((-?(?:[0-9]+(?:\.[0-9]+)?))\)$")


def load_pinned_a0_reference(
    path: str | pathlib.Path,
    inventory: dict[str, Any],
    *,
    expected_sha256: str = REFERENCE_SHA256,
    expected_correct: int = REFERENCE_CORRECT,
) -> dict[str, Any]:
    """Hash the frozen report before decoding/parsing, then bind every reference row."""
    path = pathlib.Path(path)
    raw = path.read_bytes()
    observed_sha = _sha256_bytes(raw)
    if observed_sha != expected_sha256:
        raise A0ParityError(
            f"A0 reference SHA256 mismatch: expected {expected_sha256}, got {observed_sha}")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise A0ParityError("pinned A0 reference is not UTF-8") from exc
    if not lines or lines[0] != _REFERENCE_HEADER or (len(lines) - 1) % 2:
        raise A0ParityError("pinned A0 reference has a malformed header/record count")

    class_order = inventory["probability_class_order"]
    display_to_author: dict[str, str] = {}
    for item in class_order:
        rendered = display_name(item["author"])
        if rendered in display_to_author:
            raise A0ParityError(f"display-name mapping is not bijective: {rendered!r}")
        display_to_author[rendered] = item["author"]
    author_to_label = {item["author"]: int(item["label"]) for item in class_order}

    records = []
    for offset in range(1, len(lines), 2):
        row_match = _REFERENCE_ROW_RE.fullmatch(lines[offset])
        top_match = _REFERENCE_TOP_RE.fullmatch(lines[offset + 1])
        if row_match is None or top_match is None:
            raise A0ParityError(f"malformed A0 reference record at line {offset + 1}")
        marker, true_display, book_id, rank_text = row_match.groups()
        candidate_parts = top_match.group(1).split(", ")
        if len(candidate_parts) != min(5, len(class_order)):
            raise A0ParityError(f"wrong A0 candidate count at line {offset + 2}")
        candidates = []
        for candidate_part in candidate_parts:
            candidate_match = _REFERENCE_CANDIDATE_RE.fullmatch(candidate_part)
            if candidate_match is None:
                raise A0ParityError(f"malformed A0 candidate at line {offset + 2}")
            candidate_display, score_text = candidate_match.groups()
            if candidate_display not in display_to_author:
                raise A0ParityError(
                    f"unknown candidate display name in A0 reference: {candidate_display!r}")
            score = float(score_text)
            if not math.isfinite(score) or score < 0.0:
                raise A0ParityError(f"invalid A0 candidate score at line {offset + 2}")
            candidates.append((candidate_display, score))
        if len({candidate[0] for candidate in candidates}) != len(candidates):
            raise A0ParityError(f"duplicate A0 candidate at line {offset + 2}")
        if any(left[1] < right[1] for left, right in zip(candidates, candidates[1:])):
            raise A0ParityError(f"A0 candidates are not score-sorted at line {offset + 2}")
        if not candidates:
            raise A0ParityError(f"malformed A0 top candidate at line {offset + 2}")
        pred_display = candidates[0][0]
        if true_display not in display_to_author or pred_display not in display_to_author:
            raise A0ParityError(
                f"unknown display name in A0 reference: {true_display!r}/{pred_display!r}")
        true_author = display_to_author[true_display]
        pred_author = display_to_author[pred_display]
        correct = marker == "OK  "
        if correct != (true_author == pred_author):
            raise A0ParityError(f"A0 status/top-1 disagree at {true_author}/{book_id}")
        rank = int(rank_text)
        if correct != (rank == 1):
            raise A0ParityError(f"A0 status/rank disagree at {true_author}/{book_id}")
        records.append({
            "fold_index": len(records),
            "work_id": f"{true_author}/{book_id}",
            "true_label": author_to_label[true_author],
            "true_author": true_author,
            "pred_label": author_to_label[pred_author],
            "pred_author": pred_author,
            "rank": rank,
            "correct": correct,
        })

    expected_inventory = inventory["tested_inventory"]
    if len(records) != len(expected_inventory):
        raise A0ParityError(
            f"A0 reference has {len(records)} rows, expected {len(expected_inventory)}")
    if len({row["work_id"] for row in records}) != len(records):
        raise A0ParityError("A0 reference contains duplicate works")
    for reference, expected in zip(records, expected_inventory, strict=True):
        for field in ("fold_index", "work_id", "true_label", "true_author"):
            if reference[field] != expected[field]:
                raise A0ParityError(
                    f"A0 reference inventory mismatch at fold {expected['fold_index']}: {field}")
    correct_count = sum(row["correct"] for row in records)
    if correct_count != expected_correct:
        raise A0ParityError(
            f"A0 reference correctness drift: {correct_count}/{len(records)} != "
            f"{expected_correct}/{len(records)}")
    return {
        "path": str(path.resolve()),
        "sha256": observed_sha,
        "expected_correct": int(expected_correct),
        "expected_tested": len(records),
        "expected_accuracy": float(expected_correct / len(records)),
        "records": records,
        "records_sha256": canonical_hash(records),
    }


class _TimedEstimator:
    """Transparent timing proxy used inside the existing signed ``run_fold`` math."""

    def __init__(self, estimator, wall_clock: Callable[[], float], cpu_clock: Callable[[], float]):
        self.estimator = estimator
        self.needs_groups = bool(getattr(estimator, "needs_groups", False))
        self.wall_clock = wall_clock
        self.cpu_clock = cpu_clock
        self.fit_wall_seconds = 0.0
        self.predict_wall_seconds = 0.0
        self.fit_cpu_seconds = 0.0
        self.predict_cpu_seconds = 0.0

    def fit(self, texts, y, *, groups=None):
        wall_started = self.wall_clock()
        cpu_started = self.cpu_clock()
        fit_estimator(self.estimator, texts, y, groups)
        self.fit_cpu_seconds = float(self.cpu_clock() - cpu_started)
        self.fit_wall_seconds = float(self.wall_clock() - wall_started)
        self.classes_ = np.asarray(self.estimator.classes_)
        return self

    def predict_proba(self, texts):
        wall_started = self.wall_clock()
        cpu_started = self.cpu_clock()
        probabilities = self.estimator.predict_proba(texts)
        self.predict_cpu_seconds = float(self.cpu_clock() - cpu_started)
        self.predict_wall_seconds = float(self.wall_clock() - wall_started)
        return probabilities


def evaluate_true_lobo_fold(
    cfg,
    dataset,
    fold_spec: dict[str, Any],
    factory: Callable[[], Any],
    *,
    wall_clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
) -> dict[str, Any]:
    """Production fold evaluator requiring a disk-verified context."""

    dataset = require_disk_verified_scientific_context(dataset)
    return _evaluate_true_lobo_fold_validated(
        cfg,
        dataset,
        fold_spec,
        factory,
        wall_clock=wall_clock,
        cpu_clock=cpu_clock,
    )


def evaluate_synthetic_true_lobo_fold(
    cfg,
    dataset,
    fold_spec: dict[str, Any],
    factory: Callable[[], Any],
    *,
    wall_clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
) -> dict[str, Any]:
    """Explicit non-production fold seam for isolated integration tests."""

    dataset = require_scientific_evaluation_context(dataset)
    if dataset.disk_verified:
        raise TrueLoboError(
            "synthetic true-LOBO fold requires a synthetic context"
        )
    return _evaluate_true_lobo_fold_validated(
        cfg,
        dataset,
        fold_spec,
        factory,
        wall_clock=wall_clock,
        cpu_clock=cpu_clock,
    )


def _evaluate_true_lobo_fold_validated(
    cfg,
    dataset,
    fold_spec: dict[str, Any],
    factory: Callable[[], Any],
    *,
    wall_clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
) -> dict[str, Any]:
    """Evaluate one full-work fold via the existing LOBO implementation and add split/timing proof."""
    work_id = fold_spec["work_id"]
    groups = np.asarray(dataset.groups, dtype=object)
    y = np.asarray(dataset.y)
    mask_test = groups == work_id
    if int(mask_test.sum()) != int(fold_spec["n_chunks"]):
        raise TrueLoboError(f"{work_id}: test chunk count differs from frozen inventory")
    train_groups = sorted(set(groups[~mask_test].tolist()))
    full_groups = sorted(set(groups.tolist()))
    expected_train_groups = [group for group in full_groups if group != work_id]
    if train_groups != expected_train_groups:
        raise TrueLoboError(f"{work_id}: train work inventory is not the exact complement")
    train_labels = sorted(set(int(value) for value in y[~mask_test].tolist()))
    if train_labels != list(range(len(dataset.authors))):
        raise TrueLoboError(f"{work_id}: train fold does not retain every full-universe class")

    timed: list[_TimedEstimator] = []

    def timed_factory():
        wrapped = _TimedEstimator(factory(), wall_clock, cpu_clock)
        timed.append(wrapped)
        return wrapped

    total_wall_started = wall_clock()
    total_cpu_started = cpu_clock()
    with threadpool_limits(limits=1):
        row = run_fold(
            np.asarray(dataset.texts, dtype=object),
            y,
            groups,
            len(dataset.authors),
            list(dataset.authors),
            work_id,
            timed_factory,
            int(cfg.get_path("evaluation.top_k_candidates", 5)),
        )
    total_cpu = float(cpu_clock() - total_cpu_started)
    total_wall = float(wall_clock() - total_wall_started)
    if row is None or len(timed) != 1:
        raise TrueLoboError(f"{work_id}: fold did not construct exactly one fresh estimator")
    timer = timed[0]
    probability = np.asarray(row.pop("_prob"), dtype=np.float64)
    if probability.shape != (len(dataset.authors),):
        raise TrueLoboError(f"{work_id}: work probability vector has wrong width")

    singleton_ids = [
        group for group in full_groups
        if sum(candidate.split("/", 1)[0] == group.split("/", 1)[0] for candidate in full_groups) == 1
    ]
    return {
        "fold_index": int(fold_spec["fold_index"]),
        "work_id": work_id,
        "true_label": int(row["true_label"]),
        "true_author": str(row["test_author"]),
        "split": {
            "n_train_chunks": int((~mask_test).sum()),
            "n_test_chunks": int(mask_test.sum()),
            "n_train_works": len(train_groups),
            "n_train_authors": len(train_labels),
            "train_work_inventory_sha256": canonical_hash(train_groups),
            "singleton_work_ids_present": singleton_ids,
        },
        "result": {
            "pred_label": int(row["pred_label"]),
            "pred_author": str(row["pred_author"]),
            "rank": int(row["rank"]),
            "correct": bool(row["correct"]),
            "probabilities": probability.tolist(),
        },
        "timing": {
            "fit_wall_seconds": float(timer.fit_wall_seconds),
            "predict_wall_seconds": float(timer.predict_wall_seconds),
            "total_wall_seconds": total_wall,
            "fit_cpu_seconds": float(timer.fit_cpu_seconds),
            "predict_cpu_seconds": float(timer.predict_cpu_seconds),
            "total_cpu_seconds": total_cpu,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }


def evaluate_cell_fold(cfg, dataset, cell: str, fold_spec: dict[str, Any]) -> dict[str, Any]:
    """Picklable outer-fold worker: build one signed ablation factory and evaluate one work."""
    dataset = reverify_scientific_context_from_disk(cfg, dataset)
    factory = make_factory_for_ablation("stylo", cfg, ablation=CELL_ABLATIONS[cell])
    return evaluate_true_lobo_fold(cfg, dataset, fold_spec, factory)


def _validate_runtime_binding(runtime_fingerprint: dict[str, Any]) -> None:
    """Validate the exact numerical runtime identity; OS release is deliberately absent."""
    if not isinstance(runtime_fingerprint, dict):
        raise TrueLoboError("runtime_fingerprint must be an object")
    observed = set(runtime_fingerprint)
    if observed != RUNTIME_BINDING_FIELDS:
        missing = sorted(RUNTIME_BINDING_FIELDS - observed)
        extra = sorted(observed - RUNTIME_BINDING_FIELDS)
        raise TrueLoboError(
            f"runtime_fingerprint field mismatch: missing={missing}, extra={extra}"
        )
    libc = runtime_fingerprint["libc"]
    if (
        not isinstance(libc, dict)
        or set(libc) != {"name", "version"}
        or any(type(libc[key]) is not str for key in ("name", "version"))
    ):
        raise TrueLoboError("runtime_fingerprint.libc must contain exact name/version strings")
    spacy_model = runtime_fingerprint["spacy_model"]
    expected_model_fields = {"name", "lang", "version", "spacy_version", "spacy_git_version"}
    if not isinstance(spacy_model, dict) or set(spacy_model) != expected_model_fields:
        raise TrueLoboError("runtime_fingerprint.spacy_model field inventory mismatch")


def _runtime_schema_for_identity(identity: dict[str, Any]) -> str:
    schema = identity.get("schema_version")
    if schema in {
        RUN_SCHEMA_VERSION,
        PREVIOUS_RUN_SCHEMA_VERSION,
        OLDER_RUN_SCHEMA_VERSION,
    }:
        return RUNTIME_SCHEMA_VERSION
    if schema == LEGACY_RUN_SCHEMA_VERSION:
        return LEGACY_RUNTIME_SCHEMA_VERSION
    raise CheckpointError(f"unsupported run identity schema: {schema!r}")


def _run_id_material(identity: dict[str, Any]) -> dict[str, Any]:
    """Return scientific identity fields, excluding relocation-only display paths."""

    body = {
        key: copy.deepcopy(value)
        for key, value in identity.items()
        if key not in {"run_id", "self_hash"}
    }
    if identity.get("schema_version") in {
        RUN_SCHEMA_VERSION,
        PREVIOUS_RUN_SCHEMA_VERSION,
    }:
        if not isinstance(body.get("config"), dict) or not isinstance(
            body.get("representation_cache"), dict
        ):
            raise CheckpointError(
                "relocatable run identity lacks config/cache objects"
            )
        # ``threadpoolctl`` reports the absolute shared-library filepath.  Its
        # version/API/thread properties remain binding, but install/check-out
        # location is display-only just like config/cache paths.
        def strip_display(value):
            if isinstance(value, dict):
                return {
                    key: strip_display(item)
                    for key, item in value.items()
                    if key not in {"display_path", "filepath"}
                }
            if isinstance(value, list):
                return [strip_display(item) for item in value]
            return value

        body = strip_display(body)
    return body


def build_run_identity(*, dataset, **kwargs) -> dict[str, Any]:
    """Build a production identity from a live disk-verification authority."""

    dataset = require_disk_verified_scientific_context(dataset)
    return _build_run_identity(dataset=dataset, **kwargs)


def build_synthetic_run_identity(*, dataset, **kwargs) -> dict[str, Any]:
    """Build an explicitly non-production identity for isolated tests."""

    dataset = require_scientific_evaluation_context(dataset)
    if dataset.disk_verified:
        raise TrueLoboError(
            "synthetic run identity requires a synthetic scientific context"
        )
    return _build_run_identity(dataset=dataset, **kwargs)


def _build_run_identity(
    *,
    dataset,
    inventory: dict[str, Any],
    config: dict[str, Any],
    code_hashes: dict[str, str],
    git_commit: str,
    git_dirty: bool,
    runtime_fingerprint: dict[str, Any],
    thread_fingerprint: dict[str, Any],
    representation_cache: dict[str, Any],
    reference_sha256: str = REFERENCE_SHA256,
    seed: int = 42,
    bootstrap_iters: int = 10_000,
    ci_level: float = 0.95,
    noninferiority_margin: float = 0.02,
) -> dict[str, Any]:
    """Build the immutable scientific identity that every checkpoint binds explicitly."""
    dataset = require_scientific_evaluation_context(dataset)
    if type(seed) is not int or type(bootstrap_iters) is not int or bootstrap_iters <= 0:
        raise TrueLoboError("seed/bootstrap_iters must be exact positive integers")
    if not 0.0 < float(ci_level) < 1.0 or float(noninferiority_margin) <= 0.0:
        raise TrueLoboError("invalid CI level or noninferiority margin")
    _validate_runtime_binding(runtime_fingerprint)
    dataset_digest = dataset.rows_digest
    if type(dataset_digest) is not str or len(dataset_digest) != 64:
        raise TrueLoboError("verified dataset must carry a canonical rows_digest")
    code_tree_sha256 = canonical_hash(code_hashes)
    required_config_fields = {"path", "sha256", "resolved_sha256"}
    if not isinstance(config, dict) or set(config) != required_config_fields:
        raise TrueLoboError(
            f"config must contain exactly {sorted(required_config_fields)}"
        )
    if any(type(config[key]) is not str or not config[key] for key in required_config_fields):
        raise TrueLoboError("config fingerprint has invalid fields")
    required_cache_fields = {"path", "size_bytes", "sha256", "rep_version"}
    if not isinstance(representation_cache, dict) or set(representation_cache) != required_cache_fields:
        raise TrueLoboError(
            f"representation_cache must contain exactly {sorted(required_cache_fields)}")
    if (
        type(representation_cache["path"]) is not str
        or type(representation_cache["size_bytes"]) is not int
        or representation_cache["size_bytes"] <= 0
        or type(representation_cache["sha256"]) is not str
        or len(representation_cache["sha256"]) != 64
        or type(representation_cache["rep_version"]) is not str
    ):
        raise TrueLoboError("representation_cache fingerprint has invalid fields")
    body = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": STATUS,
        "evaluation_authority": {
            "mode": (
                DISK_AUTHORITY_MODE
                if dataset.disk_verified
                else SYNTHETIC_AUTHORITY_MODE
            ),
            "context_rows_digest": dataset_digest,
            "evaluator": (
                PRODUCTION_EVALUATOR_ID
                if dataset.disk_verified
                else SYNTHETIC_EVALUATOR_ID
            ),
        },
        "git_commit": str(git_commit),
        "git_dirty": bool(git_dirty),
        "config": {
            "display_path": str(config["path"]),
            "sha256": str(config["sha256"]),
            "resolved_sha256": str(config["resolved_sha256"]),
        },
        "code_hashes": copy.deepcopy(code_hashes),
        "code_tree_sha256": code_tree_sha256,
        "dataset": {
            "rows_digest": dataset_digest,
            "n_chunks": int(inventory["n_chunks"]),
            "n_authors": len(inventory["probability_class_order"]),
            "n_works": len(inventory["work_universe"]),
            "n_tested_authors": len(inventory["metric_label_order"]),
            "n_tested_works": len(inventory["tested_inventory"]),
        },
        "bindings": {
            "config_sha256": str(config["sha256"]),
            "resolved_config_sha256": str(config["resolved_sha256"]),
            "code_tree_sha256": code_tree_sha256,
            "probability_order_sha256": inventory["probability_order_sha256"],
            "metric_order_sha256": inventory["metric_order_sha256"],
            "work_universe_sha256": inventory["work_universe_sha256"],
            "tested_inventory_sha256": inventory["tested_inventory_sha256"],
            "reference_sha256": str(reference_sha256),
            "representation_cache_sha256": representation_cache["sha256"],
        },
        "probability_class_order": copy.deepcopy(inventory["probability_class_order"]),
        "metric_label_order": copy.deepcopy(inventory["metric_label_order"]),
        "work_universe": copy.deepcopy(inventory["work_universe"]),
        "tested_inventory": copy.deepcopy(inventory["tested_inventory"]),
        "singleton_train_only": copy.deepcopy(inventory["singleton_train_only"]),
        "cells": list(CELL_ORDER),
        "statistics": {
            "paired_bootstrap": "author_clustered_percentile_accuracy",
            "iterations": bootstrap_iters,
            "seed": seed,
            "level": float(ci_level),
            "noninferiority_margin": float(noninferiority_margin),
        },
        "runtime_fingerprint": copy.deepcopy(runtime_fingerprint),
        "thread_fingerprint": copy.deepcopy(thread_fingerprint),
        "representation_cache": {
            "display_path": str(representation_cache["path"]),
            "size_bytes": int(representation_cache["size_bytes"]),
            "sha256": str(representation_cache["sha256"]),
            "rep_version": str(representation_cache["rep_version"]),
        },
    }
    body["run_id"] = canonical_hash(_run_id_material(body))
    body["self_hash"] = artifact_self_hash(body)
    return body


def validate_run_identity(identity: dict[str, Any]) -> None:
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version")
        not in {
            RUN_SCHEMA_VERSION,
            PREVIOUS_RUN_SCHEMA_VERSION,
            OLDER_RUN_SCHEMA_VERSION,
            LEGACY_RUN_SCHEMA_VERSION,
        }
    ):
        raise CheckpointError("invalid true-LOBO run identity")
    if identity.get("status") != STATUS:
        raise CheckpointError("true-LOBO run status mismatch")
    if identity.get("self_hash") != artifact_self_hash(identity):
        raise CheckpointError("true-LOBO run identity self_hash mismatch")
    if identity.get("run_id") != canonical_hash(_run_id_material(identity)):
        raise CheckpointError("true-LOBO run_id mismatch")
    if identity["schema_version"] == RUN_SCHEMA_VERSION:
        config = identity.get("config")
        cache = identity.get("representation_cache")
        authority = identity.get("evaluation_authority")
        if (
            not isinstance(config, dict)
            or set(config) != {"display_path", "sha256", "resolved_sha256"}
            or type(config["display_path"]) is not str
            or not isinstance(cache, dict)
            or set(cache)
            != {"display_path", "size_bytes", "sha256", "rep_version"}
            or type(cache["display_path"]) is not str
            or not isinstance(authority, dict)
            or set(authority)
            != {"mode", "context_rows_digest", "evaluator"}
            or authority["mode"]
            not in {DISK_AUTHORITY_MODE, SYNTHETIC_AUTHORITY_MODE}
            or type(authority["context_rows_digest"]) is not str
            or len(authority["context_rows_digest"]) != 64
            or authority["evaluator"]
            != (
                PRODUCTION_EVALUATOR_ID
                if authority["mode"] == DISK_AUTHORITY_MODE
                else SYNTHETIC_EVALUATOR_ID
            )
        ):
            raise CheckpointError(
                "v4 display/content/authority binding fields are invalid"
            )
    if identity["schema_version"] in {
        RUN_SCHEMA_VERSION,
        PREVIOUS_RUN_SCHEMA_VERSION,
        OLDER_RUN_SCHEMA_VERSION,
    }:
        _validate_runtime_binding(identity.get("runtime_fingerprint"))


def _validate_context_against_run_identity(
    dataset,
    identity: dict[str, Any],
) -> None:
    """Re-derive every corpus/inventory binding before checkpoint mutation."""

    dataset = require_scientific_evaluation_context(dataset)
    if identity.get("schema_version") != RUN_SCHEMA_VERSION:
        raise CheckpointError(
            "legacy true-LOBO identity lacks evaluation authority and is not resumable"
        )
    expected_authority = {
        "mode": (
            DISK_AUTHORITY_MODE
            if dataset.disk_verified
            else SYNTHETIC_AUTHORITY_MODE
        ),
        "context_rows_digest": dataset.rows_digest,
        "evaluator": (
            PRODUCTION_EVALUATOR_ID
            if dataset.disk_verified
            else SYNTHETIC_EVALUATOR_ID
        ),
    }
    if identity.get("evaluation_authority") != expected_authority:
        raise CheckpointError(
            "current scientific context authority does not match run identity"
        )
    inventory = derive_inventory(dataset)
    expected_dataset = {
        "rows_digest": dataset.rows_digest,
        "n_chunks": int(inventory["n_chunks"]),
        "n_authors": len(inventory["probability_class_order"]),
        "n_works": len(inventory["work_universe"]),
        "n_tested_authors": len(inventory["metric_label_order"]),
        "n_tested_works": len(inventory["tested_inventory"]),
    }
    if identity.get("dataset") != expected_dataset:
        raise CheckpointError(
            "current scientific context does not match run dataset identity"
        )
    for field in (
        "probability_class_order",
        "metric_label_order",
        "work_universe",
        "tested_inventory",
        "singleton_train_only",
    ):
        if identity.get(field) != inventory[field]:
            raise CheckpointError(
                f"current scientific context does not match {field}"
            )
    digest_bindings = {
        "probability_order_sha256": inventory["probability_order_sha256"],
        "metric_order_sha256": inventory["metric_order_sha256"],
        "work_universe_sha256": inventory["work_universe_sha256"],
        "tested_inventory_sha256": inventory["tested_inventory_sha256"],
    }
    bindings = identity.get("bindings")
    if not isinstance(bindings, dict) or any(
        bindings.get(field) != expected
        for field, expected in digest_bindings.items()
    ):
        raise CheckpointError(
            "current scientific context inventory digests do not match run identity"
        )


def _expected_split(identity: dict[str, Any], fold_spec: dict[str, Any]) -> dict[str, Any]:
    universe = identity["work_universe"]
    train_ids = [item["work_id"] for item in universe if item["work_id"] != fold_spec["work_id"]]
    singletons = [item["work_id"] for item in identity["singleton_train_only"]]
    return {
        "n_train_chunks": int(identity["dataset"]["n_chunks"] - fold_spec["n_chunks"]),
        "n_test_chunks": int(fold_spec["n_chunks"]),
        "n_train_works": len(universe) - 1,
        "n_train_authors": int(identity["dataset"]["n_authors"]),
        "train_work_inventory_sha256": canonical_hash(train_ids),
        "singleton_work_ids_present": singletons,
    }


def build_checkpoint(
    identity: dict[str, Any],
    cell: str,
    fold_spec: dict[str, Any],
    evaluated: dict[str, Any],
) -> dict[str, Any]:
    for field in ("fold_index", "work_id", "true_label", "true_author"):
        expected = fold_spec[field]
        if evaluated.get(field) != expected:
            raise CheckpointError(
                f"worker result identity mismatch for {cell}/{fold_spec['work_id']}: {field}")
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": STATUS,
        "run_id": identity["run_id"],
        "evaluation_authority": copy.deepcopy(
            identity["evaluation_authority"]
        ),
        "bindings": copy.deepcopy(identity["bindings"]),
        "cell": cell,
        "axes": _axes(cell),
        "fold_index": int(fold_spec["fold_index"]),
        "work_id": fold_spec["work_id"],
        "true_label": int(fold_spec["true_label"]),
        "true_author": fold_spec["true_author"],
        "split": copy.deepcopy(evaluated["split"]),
        "result": copy.deepcopy(evaluated["result"]),
        "timing": copy.deepcopy(evaluated["timing"]),
    }
    checkpoint["self_hash"] = artifact_self_hash(checkpoint)
    validate_checkpoint(checkpoint, identity, cell, fold_spec)
    return checkpoint


def validate_checkpoint(
    checkpoint: dict[str, Any],
    identity: dict[str, Any],
    cell: str,
    fold_spec: dict[str, Any],
) -> None:
    if not isinstance(checkpoint, dict):
        raise CheckpointError("checkpoint must be an object")
    if checkpoint.get("self_hash") != artifact_self_hash(checkpoint):
        raise CheckpointError(f"checkpoint self_hash mismatch for {cell}/{fold_spec['work_id']}")
    expected_scalars = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": STATUS,
        "run_id": identity["run_id"],
        "evaluation_authority": identity["evaluation_authority"],
        "bindings": identity["bindings"],
        "cell": cell,
        "axes": _axes(cell),
        "fold_index": int(fold_spec["fold_index"]),
        "work_id": fold_spec["work_id"],
        "true_label": int(fold_spec["true_label"]),
        "true_author": fold_spec["true_author"],
        "split": _expected_split(identity, fold_spec),
    }
    for field, expected in expected_scalars.items():
        if checkpoint.get(field) != expected:
            raise CheckpointError(
                f"checkpoint {cell}/{fold_spec['work_id']} metadata mismatch: {field}")

    result = checkpoint.get("result")
    if not isinstance(result, dict):
        raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} lacks result")
    n_authors = int(identity["dataset"]["n_authors"])
    probabilities = np.asarray(result.get("probabilities"), dtype=np.float64)
    if probabilities.shape != (n_authors,):
        raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} probability width mismatch")
    if not np.isfinite(probabilities).all() or (probabilities < -1e-9).any():
        raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} has invalid probabilities")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6, rtol=0.0):
        raise CheckpointError(
            f"checkpoint {cell}/{fold_spec['work_id']} probabilities do not sum to one")
    order = np.argsort(-probabilities, kind="stable")
    pred_label = _checked_label(result.get("pred_label"), n_authors, field="pred_label")
    true_label = int(fold_spec["true_label"])
    rank = int((probabilities >= probabilities[true_label]).sum())
    expected_pred_author = identity["probability_class_order"][pred_label]["author"]
    if pred_label != int(order[0]) or result.get("pred_author") != expected_pred_author:
        raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} prediction mismatch")
    if result.get("rank") != rank or result.get("correct") is not (pred_label == true_label):
        raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} rank/correct mismatch")
    timing = checkpoint.get("timing")
    timing_fields = (
        "fit_wall_seconds", "predict_wall_seconds", "total_wall_seconds",
        "fit_cpu_seconds", "predict_cpu_seconds", "total_cpu_seconds", "peak_rss_kib",
    )
    if not isinstance(timing, dict) or any(field not in timing for field in timing_fields):
        raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} timing mismatch")
    for field in timing_fields[:-1]:
        value = timing[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} invalid timing {field}")
    if type(timing["peak_rss_kib"]) is not int or timing["peak_rss_kib"] < 0:
        raise CheckpointError(f"checkpoint {cell}/{fold_spec['work_id']} invalid peak RSS")


def checkpoint_filename(fold_spec: dict[str, Any]) -> str:
    digest = hashlib.sha256(fold_spec["work_id"].encode("utf-8")).hexdigest()[:16]
    return f"{int(fold_spec['fold_index']):04d}-{digest}.json"


def _fsync_directory(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_create_json(path: pathlib.Path, value: dict[str, Any]) -> bool:
    """Publish strict JSON atomically without overwrite.

    The unique temp is fully fsynced, then hard-linked into the canonical name.
    ``os.link`` is the atomic create-if-absent commit point: concurrent writers
    cannot replace each other.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        dumps_strict(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
    )
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise CheckpointError(
                    f"refusing immutable JSON target that is symlink/non-file: {path}"
                )
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _durable_replace_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    """Durably replace a mutable ledger while preserving the prior file on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        dumps_strict(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
    )
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _exclusive_ledger_lock(path: pathlib.Path):
    lock = path.parent / f".{path.name}.lock"
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | nofollow, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class CheckpointStore:
    """Stable output-adjacent checkpoint root bound to exactly one run identity."""

    def __init__(self, root: str | pathlib.Path, identity: dict[str, Any]):
        validate_run_identity(identity)
        self.root = pathlib.Path(root)
        self.identity = identity
        if self.root.is_symlink():
            raise CheckpointError(f"checkpoint root is a symlink: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        run_path = self.root / "RUN.json"
        if not _durable_create_json(run_path, identity):
            existing = load_strict(run_path)
            validate_run_identity(existing)
            if (
                existing["run_id"] != identity["run_id"]
                or _run_id_material(existing) != _run_id_material(identity)
            ):
                raise CheckpointError("checkpoint root belongs to different run metadata")
            # Preserve the originally published display metadata so a relocated
            # resume reassembles byte-identically while sharing the same
            # content-only run_id.
            self.identity = existing
        for cell in CELL_ORDER:
            directory = self.root / cell
            if directory.is_symlink():
                raise CheckpointError(f"checkpoint cell directory is a symlink: {directory}")
            directory.mkdir(exist_ok=True)
        allowed_root_entries = {
            "RUN.json", "A0_GATE.json", "RUNTIME.json", *CELL_ORDER,
        }
        extras = sorted(
            path.name for path in self.root.iterdir()
            if not path.name.startswith(".") and path.name not in allowed_root_entries
        )
        if extras:
            raise CheckpointError(f"extra/conflicting checkpoint-root entries: {extras}")

    def path_for(self, cell: str, fold_spec: dict[str, Any]) -> pathlib.Path:
        return self.root / cell / checkpoint_filename(fold_spec)

    def scan_cell(self, cell: str) -> dict[int, dict[str, Any]]:
        expected = {
            checkpoint_filename(fold): fold for fold in self.identity["tested_inventory"]
        }
        directory = self.root / cell
        observed_files = {path.name: path for path in directory.glob("*.json")}
        extras = sorted(set(observed_files) - set(expected))
        if extras:
            raise CheckpointError(f"extra/conflicting {cell} checkpoint files: {extras}")
        found: dict[int, dict[str, Any]] = {}
        for name, path in observed_files.items():
            fold = expected[name]
            checkpoint = load_strict(path)
            validate_checkpoint(checkpoint, self.identity, cell, fold)
            index = int(fold["fold_index"])
            if index in found:
                raise CheckpointError(f"duplicate {cell} checkpoint fold {index}")
            found[index] = checkpoint
        return found

    def save(self, checkpoint: dict[str, Any]) -> pathlib.Path:
        cell = checkpoint["cell"]
        fold = self.identity["tested_inventory"][int(checkpoint["fold_index"])]
        validate_checkpoint(checkpoint, self.identity, cell, fold)
        path = self.path_for(cell, fold)
        if not _durable_create_json(path, checkpoint):
            existing = load_strict(path)
            validate_checkpoint(existing, self.identity, cell, fold)
            if existing != checkpoint:
                raise CheckpointError(f"refusing to overwrite conflicting checkpoint {path}")
            return path
        return path

    def has_variant_checkpoints(self) -> bool:
        return any(self.scan_cell(cell) for cell in ("A4", "A1"))

    def load_runtime(self) -> dict[str, Any]:
        path = self.root / "RUNTIME.json"
        expected_schema = _runtime_schema_for_identity(self.identity)
        if not path.exists():
            value = {
                "schema_version": expected_schema,
                "run_id": self.identity["run_id"],
                "n_jobs_history": [],
                "per_cell_execution_wall_seconds": {cell: 0.0 for cell in CELL_ORDER},
            }
            value["self_hash"] = artifact_self_hash(value)
            return value
        value = load_strict(path)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != expected_schema
            or value.get("run_id") != self.identity["run_id"]
            or value.get("self_hash") != artifact_self_hash(value)
        ):
            raise CheckpointError("runtime ledger mismatch")
        return value

    def add_runtime(self, cell: str, seconds: float, n_jobs: int) -> None:
        path = self.root / "RUNTIME.json"
        with _exclusive_ledger_lock(path):
            value = self.load_runtime()
            if n_jobs not in value["n_jobs_history"]:
                value["n_jobs_history"].append(int(n_jobs))
            value["per_cell_execution_wall_seconds"][cell] = float(
                value["per_cell_execution_wall_seconds"][cell]
                + max(0.0, float(seconds))
            )
            value["self_hash"] = artifact_self_hash(value)
            _durable_replace_json(path, value)

    def gate_path(self) -> pathlib.Path:
        return self.root / "A0_GATE.json"

    def load_gate(self) -> dict[str, Any] | None:
        path = self.gate_path()
        if not path.exists():
            return None
        value = load_strict(path)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != GATE_SCHEMA_VERSION
            or value.get("run_id") != self.identity["run_id"]
            or value.get("self_hash") != artifact_self_hash(value)
        ):
            raise CheckpointError("A0 gate receipt mismatch")
        return value

    def save_gate(self, receipt: dict[str, Any]) -> None:
        path = self.gate_path()
        if not _durable_create_json(path, receipt):
            existing = self.load_gate()
            if existing != receipt:
                raise CheckpointError("refusing to overwrite conflicting A0 gate receipt")
            return


def assemble_cell(
    identity: dict[str, Any],
    cell: str,
    checkpoints: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    tested = identity["tested_inventory"]
    missing = [int(fold["fold_index"]) for fold in tested if int(fold["fold_index"]) not in checkpoints]
    if missing:
        raise CheckpointError(f"cannot assemble incomplete {cell}: missing folds {missing[:10]}")
    works = []
    for fold in tested:
        index = int(fold["fold_index"])
        checkpoint = checkpoints[index]
        validate_checkpoint(checkpoint, identity, cell, fold)
        works.append({
            "fold_index": index,
            "work_id": fold["work_id"],
            "true_label": int(fold["true_label"]),
            "true_author": fold["true_author"],
            **copy.deepcopy(checkpoint["result"]),
            "split": copy.deepcopy(checkpoint["split"]),
            "timing": copy.deepcopy(checkpoint["timing"]),
            "checkpoint_self_hash": checkpoint["self_hash"],
        })
    truth = np.asarray([work["true_label"] for work in works], dtype=int)
    pred = np.asarray([work["pred_label"] for work in works], dtype=int)
    ranks = np.asarray([work["rank"] for work in works], dtype=int)
    # Macro-F1 is bound to the frozen tested-author estimand.  A prediction of
    # a train-only singleton is still an error for the tested true class, but
    # the singleton is not added as a zero-support class to the macro average.
    metric_labels = [
        int(item["label"])
        for item in identity["metric_label_order"]
    ]
    per_author = {}
    for item in identity["metric_label_order"]:
        label = int(item["label"])
        mask = truth == label
        per_author[item["author"]] = float(np.mean(pred[mask] == label))
    timing = {
        "fit_wall_seconds_sum": float(sum(work["timing"]["fit_wall_seconds"] for work in works)),
        "predict_wall_seconds_sum": float(
            sum(work["timing"]["predict_wall_seconds"] for work in works)),
        "fold_wall_seconds_sum": float(sum(work["timing"]["total_wall_seconds"] for work in works)),
        "fit_cpu_seconds_sum": float(sum(work["timing"]["fit_cpu_seconds"] for work in works)),
        "predict_cpu_seconds_sum": float(
            sum(work["timing"]["predict_cpu_seconds"] for work in works)),
        "total_cpu_seconds": float(sum(work["timing"]["total_cpu_seconds"] for work in works)),
        "peak_rss_kib_max": int(max(work["timing"]["peak_rss_kib"] for work in works)),
    }
    record = {
        "cell": cell,
        "axes": _axes(cell),
        "status": "applied",
        "works": works,
        "metrics": {
            "correct": int(np.sum(truth == pred)),
            "n_tested": len(works),
            "accuracy": accuracy(truth, pred),
            "macro_f1": macro_f1(
                truth,
                pred,
                metric_labels,
                unknown_pred="count_as_error",
            ),
            "top2": topk_accuracy(ranks, 2),
            "per_author_recall": per_author,
        },
        "timing": timing,
    }
    record["result_sha256"] = canonical_hash({
        "cell": cell,
        "works": [
            {
                key: work[key]
                for key in (
                    "fold_index", "work_id", "true_label", "pred_label", "rank", "correct",
                    "probabilities",
                )
            }
            for work in works
        ],
        "metrics": record["metrics"],
    })
    return record


def verify_a0_parity(
    a0: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    works = a0.get("works") or []
    expected = reference["records"]
    correct = int(sum(work["correct"] for work in works))
    if len(works) != int(reference["expected_tested"]) or correct != int(reference["expected_correct"]):
        raise A0ParityError(
            f"A0 frozen accuracy mismatch: {correct}/{len(works)} != "
            f"{reference['expected_correct']}/{reference['expected_tested']}")
    mismatches = []
    for observed, pinned in zip(works, expected, strict=True):
        for field in (
            "fold_index", "work_id", "true_label", "true_author",
            "pred_label", "pred_author", "rank", "correct",
        ):
            if observed[field] != pinned[field]:
                mismatches.append({
                    "fold_index": pinned["fold_index"],
                    "work_id": pinned["work_id"],
                    "field": field,
                    "expected": pinned[field],
                    "observed": observed[field],
                })
    if mismatches:
        raise A0ParityError(f"A0 per-work frozen parity mismatch: {mismatches[:3]!r}")
    if a0["metrics"]["accuracy"] != float(correct / len(works)):
        raise A0ParityError("A0 stored point accuracy does not equal its integer count")
    receipt = {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "passed",
        "run_id": None,
        "reference_sha256": reference["sha256"],
        "reference_records_sha256": reference["records_sha256"],
        "a0_result_sha256": a0["result_sha256"],
        "checkpoint_self_hashes_sha256": canonical_hash(
            [work["checkpoint_self_hash"] for work in works]),
        "correct": correct,
        "n_tested": len(works),
        "accuracy": float(correct / len(works)),
        "per_work_exact_matches": len(works),
    }
    return receipt


def _paired_signature(cell: dict[str, Any]) -> list[tuple[int, str, int, str]]:
    return [
        (int(work["fold_index"]), work["work_id"], int(work["true_label"]), work["true_author"])
        for work in cell["works"]
    ]


def paired_analysis(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    iterations: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
    include_leave_one_author_out: bool = False,
) -> dict[str, Any]:
    """Return the paired author-clustered accuracy comparison ``left − right``."""
    if _paired_signature(left) != _paired_signature(right):
        raise TrueLoboError(f"paired inventory mismatch: {left['cell']} vs {right['cell']}")
    left_correct = np.asarray([work["correct"] for work in left["works"]], dtype=float)
    right_correct = np.asarray([work["correct"] for work in right["works"]], dtype=float)
    authors = np.asarray([work["true_author"] for work in left["works"]], dtype=object)
    ci = paired_bootstrap_diff_clustered(
        lambda idx: float(left_correct[idx].mean()),
        lambda idx: float(right_correct[idx].mean()),
        authors,
        iters=iterations,
        level=level,
        seed=seed,
    )
    point = float(left_correct.mean() - right_correct.mean())
    if not np.isclose(point, ci.diff, rtol=0.0, atol=1e-15):
        raise TrueLoboError("paired clustered CI point differs from the direct accuracy delta")
    gains = [
        work["work_id"] for work, lc, rc in zip(left["works"], left_correct, right_correct, strict=True)
        if lc > rc
    ]
    losses = [
        work["work_id"] for work, lc, rc in zip(left["works"], left_correct, right_correct, strict=True)
        if lc < rc
    ]
    prediction_disagreements = [
        lwork["work_id"]
        for lwork, rwork in zip(left["works"], right["works"], strict=True)
        if lwork["pred_label"] != rwork["pred_label"]
    ]
    tested_authors = list(dict.fromkeys(work["true_author"] for work in left["works"]))
    by_author = {}
    for author in tested_authors:
        author_work_ids = {
            work["work_id"] for work in left["works"] if work["true_author"] == author
        }
        author_gains = [work_id for work_id in gains if work_id in author_work_ids]
        author_losses = [work_id for work_id in losses if work_id in author_work_ids]
        author_pred = [
            work_id for work_id in prediction_disagreements if work_id in author_work_ids
        ]
        by_author[author] = {
            "gains": author_gains,
            "losses": author_losses,
            "correctness_disagreements": author_gains + author_losses,
            "prediction_disagreements": author_pred,
            "net_correct": len(author_gains) - len(author_losses),
        }
    changed = [author for author, item in by_author.items() if item["correctness_disagreements"]]
    max_abs_net = max((abs(item["net_correct"]) for item in by_author.values()), default=0)
    result = {
        "left_cell": left["cell"],
        "right_cell": right["cell"],
        "delta_accuracy": point,
        "author_clustered_percentile_ci": {
            "point": float(ci.diff),
            "lo": float(ci.lo),
            "hi": float(ci.hi),
            "iterations": int(iterations),
            "seed": int(seed),
            "level": float(level),
        },
        "gains": gains,
        "losses": losses,
        "correctness_disagreements": gains + losses,
        "prediction_disagreements": prediction_disagreements,
        "gains_count": len(gains),
        "losses_count": len(losses),
        "by_author": by_author,
        "author_concentration": {
            "authors_with_correctness_change": changed,
            "n_authors_with_correctness_change": len(changed),
            "max_abs_net_correct_per_author": max_abs_net,
        },
    }
    if include_leave_one_author_out:
        loao = []
        for excluded in tested_authors:
            keep = authors != excluded
            if not keep.any():
                raise TrueLoboError("leave-one-author-out comparison has no remaining works")
            loao.append({
                "excluded_author": excluded,
                "n_remaining_works": int(keep.sum()),
                "delta_accuracy": float(left_correct[keep].mean() - right_correct[keep].mean()),
            })
        result["leave_one_author_out"] = loao
    result["self_hash"] = artifact_self_hash(result)
    return result


def primary_a4_gate(lo: float, hi: float, margin: float = 0.02) -> str:
    """Apply the signed strict-boundary noninferiority rule to unrounded CI limits."""
    values = (lo, hi, margin)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values):
        raise TrueLoboError("primary gate inputs must be finite numbers")
    if lo > hi or margin <= 0:
        raise TrueLoboError("primary gate has an invalid interval/margin")
    boundary = -float(margin)
    if lo > boundary:
        return "noninferior"
    if hi < boundary:
        return "inferior"
    return "inconclusive"


def _a0_receipt(identity: dict[str, Any], a0: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    receipt = verify_a0_parity(a0, reference)
    receipt["run_id"] = identity["run_id"]
    receipt["evaluation_authority"] = copy.deepcopy(
        identity["evaluation_authority"]
    )
    receipt["self_hash"] = artifact_self_hash(receipt)
    return receipt


def _validate_gate_receipt(
    receipt: dict[str, Any],
    identity: dict[str, Any],
    a0: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    expected = _a0_receipt(identity, a0, reference)
    if receipt != expected:
        raise CheckpointError("A0 gate receipt no longer matches its checkpoints/reference")


def _checkpoint_inventory(
    identity: dict[str, Any],
    by_cell: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    result = {"expected_per_cell": len(identity["tested_inventory"]), "cells": {}}
    for cell in CELL_ORDER:
        checkpoints = by_cell[cell]
        result["cells"][cell] = {
            "count": len(checkpoints),
            "files": [
                {
                    "fold_index": index,
                    "work_id": identity["tested_inventory"][index]["work_id"],
                    "filename": checkpoint_filename(identity["tested_inventory"][index]),
                    "self_hash": checkpoints[index]["self_hash"],
                }
                for index in sorted(checkpoints)
            ],
        }
    result["self_hash"] = artifact_self_hash(result)
    return result


def assemble_artifact(
    identity: dict[str, Any],
    cells: dict[str, dict[str, Any]],
    gate_receipt: dict[str, Any],
    checkpoints: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    stats = identity["statistics"]
    a1_a0 = paired_analysis(
        cells["A1"], cells["A0"], iterations=stats["iterations"], seed=stats["seed"],
        level=stats["level"], include_leave_one_author_out=True)
    a4_a0 = paired_analysis(
        cells["A4"], cells["A0"], iterations=stats["iterations"], seed=stats["seed"],
        level=stats["level"], include_leave_one_author_out=True)
    a1_a4 = paired_analysis(
        cells["A1"], cells["A4"], iterations=stats["iterations"], seed=stats["seed"],
        level=stats["level"], include_leave_one_author_out=False)
    primary_ci = a4_a0["author_clustered_percentile_ci"]
    gate = {
        "estimand": "stylo_A4_minus_A0_accuracy",
        "margin": float(stats["noninferiority_margin"]),
        "ci_lo": float(primary_ci["lo"]),
        "ci_hi": float(primary_ci["hi"]),
        "decision": primary_a4_gate(
            float(primary_ci["lo"]), float(primary_ci["hi"]),
            float(stats["noninferiority_margin"])),
        "scope": "exploratory artifact only; no headline mutation",
    }
    total_cpu = float(sum(cells[cell]["timing"]["total_cpu_seconds"] for cell in CELL_ORDER))
    fold_wall_sum = float(
        sum(cells[cell]["timing"]["fold_wall_seconds_sum"] for cell in CELL_ORDER))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "run_id": identity["run_id"],
        "attestation": {
            key: copy.deepcopy(identity[key])
            for key in (
                "git_commit", "git_dirty", "config", "code_hashes", "code_tree_sha256",
                "dataset", "bindings", "runtime_fingerprint", "thread_fingerprint",
                "representation_cache", "statistics", "evaluation_authority",
            )
        },
        "probability_class_order": copy.deepcopy(identity["probability_class_order"]),
        "metric_label_order": copy.deepcopy(identity["metric_label_order"]),
        "work_universe": copy.deepcopy(identity["work_universe"]),
        "tested_inventory": copy.deepcopy(identity["tested_inventory"]),
        "singleton_train_only": copy.deepcopy(identity["singleton_train_only"]),
        "a0_frozen_parity": copy.deepcopy(gate_receipt),
        "cells": [copy.deepcopy(cells[cell]) for cell in CELL_ORDER],
        "comparisons": {
            "A1_minus_A0": a1_a0,
            "A4_minus_A0": a4_a0,
            "A1_minus_A4": a1_a4,
        },
        "primary_A4_noninferiority_gate": gate,
        "checkpoint_inventory": _checkpoint_inventory(identity, checkpoints),
        "runtime": {
            "total_cpu_seconds": total_cpu,
            "total_wall_seconds": fold_wall_sum,
            "fold_wall_seconds_sum": fold_wall_sum,
            "total_wall_definition": (
                "sum of measured per-fold worker wall seconds; resume-invariant and distinct from "
                "parallel orchestrator elapsed time"
            ),
            "per_cell": {
                cell: {
                    "total_cpu_seconds": float(cells[cell]["timing"]["total_cpu_seconds"]),
                    "wall_seconds": float(cells[cell]["timing"]["fold_wall_seconds_sum"]),
                    "fold_wall_seconds_sum": float(
                        cells[cell]["timing"]["fold_wall_seconds_sum"]),
                }
                for cell in CELL_ORDER
            },
            "orchestrator_telemetry": (
                "operational RUNTIME.json sidecar; intentionally excluded from the canonical "
                "artifact so interrupted and uninterrupted assembly are identical"
            ),
            "peak_rss_kib_max": int(max(
                cells[cell]["timing"]["peak_rss_kib_max"] for cell in CELL_ORDER)),
        },
        "interpretation_scope": (
            "target-protocol validation on the known corpus; not independent external replication, "
            "not an authorship proof, and not a headline/publication artifact"
        ),
    }
    artifact["self_hash"] = artifact_self_hash(artifact)
    return artifact


def _validate_final_artifact(artifact: dict[str, Any], identity: dict[str, Any]) -> None:
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("status") != STATUS
        or artifact.get("run_id") != identity["run_id"]
        or artifact.get("attestation", {}).get("evaluation_authority")
        != identity["evaluation_authority"]
        or artifact.get("self_hash") != artifact_self_hash(artifact)
    ):
        raise TrueLoboError("existing final true-LOBO artifact is invalid or belongs to another run")


def _progress(progress: Callable[[dict[str, Any]], None] | None, event: str, **fields) -> None:
    if progress is not None:
        progress({"event": event, **fields})


def _attest(attestor, stage: str) -> None:
    if attestor is not None:
        attestor.verify(stage)


def _evaluate_with_live_attestation(
    evaluator,
    cfg,
    dataset,
    cell: str,
    fold: dict[str, Any],
    attestor,
) -> dict[str, Any]:
    label = f"{cell}/fold-{fold['fold_index']}"
    dataset = require_scientific_evaluation_context(dataset)
    _attest(attestor, f"before-{label}")
    result = evaluator(cfg, dataset, cell, fold)
    require_scientific_evaluation_context(dataset)
    _attest(attestor, f"after-{label}")
    return result


def _compute_missing(
    cfg,
    dataset,
    identity: dict[str, Any],
    store: CheckpointStore,
    cell: str,
    *,
    n_jobs: int,
    evaluator: Callable[[Any, Any, str, dict[str, Any]], dict[str, Any]],
    max_new_folds: int | None,
    progress: Callable[[dict[str, Any]], None] | None,
    clock: Callable[[], float],
    attestor=None,
) -> dict[int, dict[str, Any]]:
    existing = store.scan_cell(cell)
    missing = [
        fold for fold in identity["tested_inventory"] if int(fold["fold_index"]) not in existing
    ]
    if max_new_folds is not None:
        missing = missing[:max_new_folds]
    if not missing:
        _progress(progress, "cell_resume_complete", cell=cell, completed=len(existing))
        return existing
    _progress(
        progress, "cell_start", cell=cell, completed=len(existing), pending=len(missing), n_jobs=n_jobs)
    last = clock()
    if n_jobs == 1:
        results: Iterable[dict[str, Any]] = (
            _evaluate_with_live_attestation(
                evaluator, cfg, dataset, cell, fold, attestor
            )
            for fold in missing
        )
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator_unordered",
            pre_dispatch=n_jobs,
            batch_size=1,
            verbose=10,
        )(
            delayed(_evaluate_with_live_attestation)(
                evaluator, cfg, dataset, cell, fold, attestor
            )
            for fold in missing
        )
    for evaluated in results:
        index = int(evaluated["fold_index"])
        if not 0 <= index < len(identity["tested_inventory"]):
            raise CheckpointError(f"worker returned out-of-range fold {index}")
        fold = identity["tested_inventory"][index]
        if evaluated["work_id"] != fold["work_id"]:
            raise CheckpointError("worker returned a fold/work identity mismatch")
        checkpoint = build_checkpoint(identity, cell, fold, evaluated)
        _attest(attestor, f"before-checkpoint-{cell}-{index}")
        store.save(checkpoint)
        now = clock()
        store.add_runtime(cell, now - last, n_jobs)
        last = now
        existing[index] = checkpoint
        _progress(
            progress,
            "fold_complete",
            cell=cell,
            fold_index=index,
            work_id=fold["work_id"],
            correct=checkpoint["result"]["correct"],
            completed=len(existing),
            expected=len(identity["tested_inventory"]),
            wall_seconds=checkpoint["timing"]["total_wall_seconds"],
            cpu_seconds=checkpoint["timing"]["total_cpu_seconds"],
            peak_rss_kib=checkpoint["timing"]["peak_rss_kib"],
        )
    now = clock()
    store.add_runtime(cell, now - last, n_jobs)
    return store.scan_cell(cell)


def run_true_lobo(
    cfg,
    dataset,
    identity: dict[str, Any],
    reference: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Production true-LOBO entrypoint requiring disk-verified authority."""

    dataset = require_disk_verified_scientific_context(dataset)
    if kwargs.get("evaluator", evaluate_cell_fold) is not evaluate_cell_fold:
        raise TrueLoboError(
            "production true-LOBO runner does not accept an injected evaluator"
        )
    if kwargs.get("clock", time.perf_counter) is not time.perf_counter:
        raise TrueLoboError(
            "production true-LOBO runner does not accept an injected clock"
        )
    return _run_true_lobo_validated(
        cfg,
        dataset,
        identity,
        reference,
        **kwargs,
    )


def run_synthetic_true_lobo(
    cfg,
    dataset,
    identity: dict[str, Any],
    reference: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Explicit non-production resumability seam for isolated tests."""

    dataset = require_scientific_evaluation_context(dataset)
    if dataset.disk_verified:
        raise TrueLoboError(
            "synthetic true-LOBO runner requires a synthetic context"
        )
    if kwargs.get("evaluator", evaluate_cell_fold) is evaluate_cell_fold:
        raise TrueLoboError(
            "synthetic true-LOBO runner requires an injected test evaluator"
        )
    return _run_true_lobo_validated(
        cfg,
        dataset,
        identity,
        reference,
        **kwargs,
    )


def _run_true_lobo_validated(
    cfg,
    dataset,
    identity: dict[str, Any],
    reference: dict[str, Any],
    *,
    output_path: str | pathlib.Path,
    checkpoint_root: str | pathlib.Path,
    n_jobs: int,
    cells: Sequence[str] = ("A0", "A1", "A4"),
    smoke_only: bool = False,
    evaluator: Callable[[Any, Any, str, dict[str, Any]], dict[str, Any]] = evaluate_cell_fold,
    progress: Callable[[dict[str, Any]], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    attestor=None,
) -> dict[str, Any]:
    """Run/resume A0, gate it exactly, then schedule only A4 and A1 and assemble the artifact."""
    if (
        type(n_jobs) is not int
        or n_jobs <= 0
        or n_jobs > MAX_TRUE_LOBO_WORKERS
    ):
        raise TrueLoboError(
            f"n_jobs must be a positive exact integer <= {MAX_TRUE_LOBO_WORKERS}"
        )
    requested = list(cells)
    if len(set(requested)) != len(requested) or any(cell not in CELL_ORDER for cell in requested):
        raise TrueLoboError(f"cells must be unique values from {CELL_ORDER!r}")
    if "A0" not in requested:
        raise TrueLoboError("A0 is a mandatory dependency")
    if not smoke_only and set(requested) != set(CELL_ORDER):
        raise TrueLoboError("the complete target validation requires exactly A0,A1,A4")
    dataset = require_scientific_evaluation_context(dataset)
    validate_run_identity(identity)
    _validate_context_against_run_identity(dataset, identity)
    _attest(attestor, "run-start")
    store = CheckpointStore(checkpoint_root, identity)
    identity = store.identity
    output_path = pathlib.Path(output_path)

    a0_checkpoints = store.scan_cell("A0")
    existing_gate = store.load_gate()
    if store.has_variant_checkpoints() and existing_gate is None:
        raise CheckpointError("A1/A4 checkpoints exist without a valid A0 gate receipt")

    if output_path.exists():
        _attest(attestor, "before-existing-final-validation")
        by_cell = {cell: store.scan_cell(cell) for cell in CELL_ORDER}
        if any(len(by_cell[cell]) != len(identity["tested_inventory"]) for cell in CELL_ORDER):
            raise CheckpointError("final artifact exists but checkpoint inventory is incomplete")
        a0 = assemble_cell(identity, "A0", by_cell["A0"])
        if existing_gate is None:
            raise CheckpointError("final artifact exists without A0 gate receipt")
        _validate_gate_receipt(existing_gate, identity, a0, reference)
        artifact = load_strict(output_path)
        _validate_final_artifact(artifact, identity)
        rebuilt_cells = {
            cell: (a0 if cell == "A0" else assemble_cell(identity, cell, by_cell[cell]))
            for cell in CELL_ORDER
        }
        rebuilt = assemble_artifact(identity, rebuilt_cells, existing_gate, by_cell)
        if artifact != rebuilt:
            raise TrueLoboError(
                "existing final artifact does not exactly reassemble from its checkpoints")
        _attest(attestor, "after-existing-final-validation")
        return artifact

    a0_checkpoints = _compute_missing(
        cfg, dataset, identity, store, "A0", n_jobs=n_jobs, evaluator=evaluator,
        max_new_folds=1 if smoke_only else None, progress=progress, clock=clock,
        attestor=attestor)
    if smoke_only:
        completed = sorted(a0_checkpoints)
        latest = a0_checkpoints[completed[-1]]
        return {
            "status": "smoke_complete_checkpoint_saved",
            "run_id": identity["run_id"],
            "cell": "A0",
            "completed_a0_checkpoints": len(a0_checkpoints),
            "fold_index": int(latest["fold_index"]),
            "work_id": latest["work_id"],
            "timing": copy.deepcopy(latest["timing"]),
        }
    a0 = assemble_cell(identity, "A0", a0_checkpoints)
    gate_receipt = _a0_receipt(identity, a0, reference)
    if existing_gate is not None:
        _validate_gate_receipt(existing_gate, identity, a0, reference)
    else:
        _attest(attestor, "before-a0-gate")
        store.save_gate(gate_receipt)
    _progress(
        progress,
        "a0_gate_passed",
        correct=a0["metrics"]["correct"],
        n_tested=a0["metrics"]["n_tested"],
        accuracy=a0["metrics"]["accuracy"],
    )

    cell_records = {"A0": a0}
    checkpoint_sets = {"A0": a0_checkpoints}
    for cell in CELL_ORDER[1:]:
        if cell not in requested:
            continue
        checkpoints = _compute_missing(
            cfg, dataset, identity, store, cell, n_jobs=n_jobs, evaluator=evaluator,
            max_new_folds=None, progress=progress, clock=clock,
            attestor=attestor)
        checkpoint_sets[cell] = checkpoints
        cell_records[cell] = assemble_cell(identity, cell, checkpoints)

    if set(cell_records) != set(CELL_ORDER):
        raise CheckpointError("refusing final assembly before all A0/A4/A1 cells are complete")
    artifact = assemble_artifact(
        identity, cell_records, gate_receipt, checkpoint_sets)
    _attest(attestor, "before-final-publication")
    # Canonical key ordering keeps the published artifact byte-identical after
    # loading an immutable, sort-keyed RUN.json during resume.
    dump_strict(artifact, output_path, indent=2, ensure_ascii=False, sort_keys=True)
    loaded = load_strict(output_path)
    _validate_final_artifact(loaded, identity)
    _attest(attestor, "after-final-publication")
    return loaded


def format_compact_table(artifact: dict[str, Any]) -> str:
    comparisons = artifact["comparisons"]
    lines = [
        "cell  correct/251  accuracy  macro_f1  top2  dacc_vs_A0  clustered_CI  "
        "gains/losses  CPUh  wall"
    ]
    for cell in CELL_ORDER:
        record = next(item for item in artifact["cells"] if item["cell"] == cell)
        metrics = record["metrics"]
        if cell == "A0":
            delta = 0.0
            ci = {"lo": 0.0, "hi": 0.0}
            gains = losses = 0
        else:
            comparison = comparisons[f"{cell}_minus_A0"]
            delta = comparison["delta_accuracy"]
            ci = comparison["author_clustered_percentile_ci"]
            gains = comparison["gains_count"]
            losses = comparison["losses_count"]
        cpu_hours = record["timing"]["total_cpu_seconds"] / 3600.0
        wall = record["timing"]["fold_wall_seconds_sum"]
        lines.append(
            f"{cell}  {metrics['correct']}/{metrics['n_tested']}  {metrics['accuracy']:.4f}  "
            f"{metrics['macro_f1']:.4f}  {metrics['top2']:.4f}  {delta:+.4f}  "
            f"[{ci['lo']:+.4f},{ci['hi']:+.4f}]  {gains}/{losses}  {cpu_hours:.2f}  {wall:.1f}s"
        )
    return "\n".join(lines)


__all__ = [
    "STATUS",
    "SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "LEGACY_RUN_SCHEMA_VERSION",
    "OLDER_RUN_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "GATE_SCHEMA_VERSION",
    "LEGACY_RUNTIME_SCHEMA_VERSION",
    "RUNTIME_SCHEMA_VERSION",
    "DISK_AUTHORITY_MODE",
    "SYNTHETIC_AUTHORITY_MODE",
    "PRODUCTION_EVALUATOR_ID",
    "SYNTHETIC_EVALUATOR_ID",
    "RUNTIME_BINDING_FIELDS",
    "REFERENCE_SHA256",
    "REFERENCE_CORRECT",
    "REFERENCE_TESTED",
    "REFERENCE_ACCURACY",
    "CELL_ORDER",
    "MAX_TRUE_LOBO_WORKERS",
    "CELL_ABLATIONS",
    "TrueLoboError",
    "A0ParityError",
    "CheckpointError",
    "derive_inventory",
    "load_pinned_a0_reference",
    "evaluate_true_lobo_fold",
    "evaluate_synthetic_true_lobo_fold",
    "evaluate_cell_fold",
    "build_run_identity",
    "build_synthetic_run_identity",
    "validate_run_identity",
    "build_checkpoint",
    "validate_checkpoint",
    "checkpoint_filename",
    "CheckpointStore",
    "assemble_cell",
    "verify_a0_parity",
    "paired_analysis",
    "primary_a4_gate",
    "assemble_artifact",
    "run_true_lobo",
    "run_synthetic_true_lobo",
    "format_compact_table",
]
