"""Per-fold immutable checkpoint store with strict resume semantics (§4.3/§7).

A stack cell is 251 folds and too costly to redo, so checkpoints are **per-fold**. Each checkpoint is
an immutable object keyed exactly by ``(dataset, model, cell, fold)``, published **atomically without
overwrite**, carrying a **self-hash** and binding the ``run_id`` plus the dataset/manifest/class-order
digests and the fold-local evidence. Resume semantics:

- a **valid** existing checkpoint (self-hash + identity + bound digests all match) is **skipped**;
- a **missing** fold *during a run* is **pending** — it is computed (missing is not fatal mid-run);
- a **corrupt / conflicting / extra** checkpoint is **always fatal**.

``COMPLETE`` is declared only when **every** applicable fold is present, so a missing fold is fatal
**only at COMPLETE assembly**, never mid-run. Before and after each fold the runner re-verifies the
bound digests via :meth:`CheckpointStore.verify_bindings`.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
from typing import Iterable, Mapping

from ...jsonio import dump_strict, dumps_strict, load_strict
from ...pipeline.bundle import (BundleError, _real_within, _safe_name,
                                _verify_real_dir_chain)

_CHECKPOINT_SCHEMA = "paired_audit.checkpoint.v1"
_DATASETS = ("lobo", "ruaa")
_REQUIRED_BINDING_KEYS = ("dataset_digest", "fold_manifest_digest",
                          "probability_class_order_digest", "metric_label_order_digest")
_IDENTITY_KEYS = ("run_id", "dataset", "model", "cell", "fold_index", "work_id")
_FILENAME_RE = re.compile(r"^\d{4}-[0-9a-f]{16}\.json$")


class CheckpointError(RuntimeError):
    """Fail-closed: a checkpoint is corrupt, conflicting, extra, or the resume identity mismatches."""


def _slug(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_.-]", "_", name)
    if not _safe_name(slug):
        raise CheckpointError(f"unsafe path component derived from {name!r}")
    return slug


def _self_hash(record: Mapping) -> str:
    body = {k: v for k, v in record.items() if k != "self_hash"}
    return hashlib.sha256(dumps_strict(body, sort_keys=True).encode("utf-8")).hexdigest()


class CheckpointStore:
    """An immutable per-fold checkpoint store bound to exactly one ``run_id`` and its per-dataset
    binding digests."""

    def __init__(self, root: pathlib.Path | str, run_id: str, dataset_bindings: Mapping[str, Mapping]):
        self.root = pathlib.Path(root)
        if not (isinstance(run_id, str) and run_id):
            raise CheckpointError("run_id must be a non-empty string")
        self.run_id = run_id
        self.dataset_bindings: dict[str, dict] = {}
        for ds in _DATASETS:
            if ds not in dataset_bindings:
                raise CheckpointError(f"dataset_bindings missing {ds!r}")
            b = dataset_bindings[ds]
            if set(b) != set(_REQUIRED_BINDING_KEYS) or any(not b[k] for k in _REQUIRED_BINDING_KEYS):
                raise CheckpointError(f"{ds} bindings must carry non-empty {_REQUIRED_BINDING_KEYS}")
            self.dataset_bindings[ds] = dict(b)

    # ── paths ────────────────────────────────────────────────────────────────
    def _cell_dir(self, dataset: str, model: str, cell: str) -> pathlib.Path:
        if dataset not in _DATASETS:
            raise CheckpointError(f"unknown dataset {dataset!r}")
        return self.root / _slug(dataset) / _slug(model) / _slug(cell)

    def checkpoint_filename(self, fold_index: int, work_id: str) -> str:
        if type(fold_index) is not int or fold_index < 0:
            raise CheckpointError("fold_index must be a non-negative int")
        if fold_index > 9999:                            # keep the 4-digit filename namespace exact
            raise CheckpointError("fold_index exceeds the 4-digit checkpoint namespace (max 9999)")
        tag = hashlib.sha256(str(work_id).encode("utf-8")).hexdigest()[:16]
        return f"{fold_index:04d}-{tag}.json"

    def checkpoint_path(self, dataset: str, model: str, cell: str, fold_index: int,
                        work_id: str) -> pathlib.Path:
        return self._cell_dir(dataset, model, cell) / self.checkpoint_filename(fold_index, work_id)

    # ── identity / binding checks ─────────────────────────────────────────────
    def verify_bindings(self, record: Mapping) -> None:
        """Fail-closed unless the record binds this store's run_id and the exact digests for its
        dataset (used before and after each fold, and on every resume load)."""
        for k in _IDENTITY_KEYS:
            if k not in record:
                raise CheckpointError(f"checkpoint missing identity key {k!r}")
        if record["run_id"] != self.run_id:
            raise CheckpointError("checkpoint run_id does not match the store")
        ds = record["dataset"]
        if ds not in self.dataset_bindings:
            raise CheckpointError(f"checkpoint has unknown dataset {ds!r}")
        if record.get("bindings") != self.dataset_bindings[ds]:
            raise CheckpointError(f"checkpoint bindings differ from the {ds} run bindings")

    def _build(self, dataset: str, model: str, cell: str, fold_index: int, work_id: str,
               result: Mapping, fold_local_evidence: Mapping) -> dict:
        if dataset not in self.dataset_bindings:
            raise CheckpointError(f"unknown dataset {dataset!r}")
        record = {
            "schema": _CHECKPOINT_SCHEMA,
            "run_id": self.run_id,
            "dataset": dataset,
            "model": model,
            "cell": cell,
            "fold_index": int(fold_index),
            "work_id": str(work_id),
            "bindings": dict(self.dataset_bindings[dataset]),
            "result": dict(result),
            "fold_local_evidence": dict(fold_local_evidence),
        }
        record["self_hash"] = _self_hash(record)
        return record

    # ── save / load ────────────────────────────────────────────────────────────
    def save(self, dataset: str, model: str, cell: str, fold_index: int, work_id: str, *,
             result: Mapping, fold_local_evidence: Mapping) -> pathlib.Path:
        """Publish one fold checkpoint atomically without overwrite. An existing identical checkpoint
        is idempotently reused; an existing DIFFERENT checkpoint at the same identity is fatal."""
        record = self._build(dataset, model, cell, fold_index, work_id,
                             result, fold_local_evidence)
        self.verify_bindings(record)
        path = self.checkpoint_path(dataset, model, cell, fold_index, work_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _verify_real_dir_chain(path.parent)
        except BundleError as exc:
            raise CheckpointError(f"unsafe checkpoint path chain: {exc}") from exc
        if path.exists() or path.is_symlink():
            existing = self._load_path(path)
            if existing != record:
                raise CheckpointError(f"conflicting checkpoint already exists at {path}")
            return path                                  # identical — idempotent
        # atomic write (dump_strict: leading-dot mkstemp + os.replace); path does not exist here
        dump_strict(record, path, trailing_newline=True)
        return path

    def _load_path(self, path: pathlib.Path) -> dict:
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f"checkpoint missing or a symlink: {path}")
        if not _real_within(path, self.root, must_file=True):
            raise CheckpointError(f"checkpoint escapes the store root: {path}")
        try:
            record = load_strict(path)
        except Exception as exc:
            raise CheckpointError(f"corrupt checkpoint (unreadable): {path}: {exc}") from exc
        if not isinstance(record, dict) or record.get("schema") != _CHECKPOINT_SCHEMA:
            raise CheckpointError(f"corrupt checkpoint (bad schema): {path}")
        if record.get("self_hash") != _self_hash(record):
            raise CheckpointError(f"corrupt checkpoint (self-hash mismatch): {path}")
        self.verify_bindings(record)                     # conflicting identity/bindings -> fatal
        return record

    # ── scan / resume / complete ──────────────────────────────────────────────
    def scan_cell(self, dataset: str, model: str, cell: str) -> dict[int, dict]:
        """Load every checkpoint under a cell dir; a corrupt/conflicting/misfiled one is fatal.
        Returns ``{fold_index: record}``."""
        cell_dir = self._cell_dir(dataset, model, cell)
        out: dict[int, dict] = {}
        if not cell_dir.exists():
            return out
        for entry in sorted(os.scandir(cell_dir), key=lambda e: e.name):
            if entry.name.startswith("."):
                continue                                 # transient atomic-write temp (crash orphan)
            if not _FILENAME_RE.match(entry.name):
                raise CheckpointError(f"extra/unexpected file in checkpoint dir: {entry.path}")
            record = self._load_path(pathlib.Path(entry.path))
            if record["dataset"] != dataset or record["model"] != model or record["cell"] != cell:
                raise CheckpointError(f"misfiled checkpoint (identity != dir): {entry.path}")
            expected_name = self.checkpoint_filename(record["fold_index"], record["work_id"])
            if entry.name != expected_name:
                raise CheckpointError(f"checkpoint filename does not match its identity: {entry.path}")
            if record["fold_index"] in out:
                raise CheckpointError(f"duplicate fold_index in {cell_dir}")
            out[record["fold_index"]] = record
        return out

    def resume_cell(self, dataset: str, model: str, cell: str,
                    expected_folds: Iterable[tuple[int, str]]) -> dict:
        """Return ``{present: {fold_index: record}, pending: [(fold_index, work_id)]}``.

        A present valid checkpoint that is NOT in the expected fold set is an **extra** checkpoint and
        is fatal; a missing expected fold is **pending** (computed), not fatal mid-run.
        """
        expected = list(expected_folds)
        expected_index = {int(fi): str(wid) for fi, wid in expected}
        if len(expected_index) != len(expected):
            raise CheckpointError("expected_folds has duplicate fold indices")
        present = self.scan_cell(dataset, model, cell)
        for fi, record in present.items():
            if fi not in expected_index or record["work_id"] != expected_index[fi]:
                raise CheckpointError(f"extra/unexpected checkpoint fold_index={fi} in {dataset}/{model}/{cell}")
        pending = [(fi, wid) for fi, wid in expected_index.items() if fi not in present]
        return {"present": present, "pending": sorted(pending)}

    def assert_cell_complete(self, dataset: str, model: str, cell: str,
                             expected_folds: Iterable[tuple[int, str]]) -> dict[int, dict]:
        """Fail-closed unless EVERY expected fold is present, valid, and there is no extra — a missing
        fold is fatal only here (COMPLETE assembly). Returns the present records."""
        state = self.resume_cell(dataset, model, cell, expected_folds)
        if state["pending"]:
            raise CheckpointError(
                f"cell {dataset}/{model}/{cell} incomplete: {len(state['pending'])} folds pending")
        return state["present"]
