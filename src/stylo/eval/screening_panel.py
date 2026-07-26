"""screening_panel_v1 — the FROZEN 5-fold work assignment for the GKF screening proxy.

The GKF proxy (``eval/groupkfold.py``) is a fast ranking screen for the ablation sweep. To make
every sweep case comparable and reproducible it must run on ONE frozen fold assignment, not a
per-run ``StratifiedGroupKFold(shuffle=True)`` draw. This module derives, freezes and verifies that
assignment:

  * **panel** = the authors of the canonical corpus with >= 2 works (single-work authors cannot be
    tested in any leave-out CV). For the committed corpus that is exactly **43 authors / 251 works**.
  * **folds** = a deterministic per-author rotating round-robin over each author's sorted works
    (``fold = (author_rank + work_rank) % k``). This guarantees every work lands in exactly one
    test fold and — because an author with >= 2 works occupies >= 2 distinct folds — every TRAIN
    fold contains all 43 classes.
  * the manifest binds author/work IDs, the per-work fold, the parent (disk-anchored) dataset
    digest, the algorithm/seed/config hash and a self-hash, so a drifted corpus or a tampered
    manifest fails closed.

``verify_result_against_panel`` compares each engine's evaluated (work, label, fold) set against
the canonical manifest — for the primary model AND every baseline row — before any metric is read.
"""
from __future__ import annotations

import collections
import hashlib
import numbers
import pathlib
from typing import Dict, List, Tuple

from ..jsonio import dumps_strict, loads_strict

PANEL_NAME = "screening_panel_v1"
ALGORITHM = "rotating_round_robin_per_author"
K_FOLDS = 5
SEED = 42
MANIFEST_PATH = "docs/screening_panel_v1.json"


class ScreeningPanelError(ValueError):
    """Fail-closed screening-panel contract violation."""


# ── derivation ─────────────────────────────────────────────────────────────────
def _works_by_author(dataset) -> Dict[str, List[str]]:
    works = sorted({str(g) for g in dataset.groups.tolist()})
    by_author: Dict[str, List[str]] = collections.defaultdict(list)
    for w in works:
        by_author[w.split("/", 1)[0]].append(w)
    return {a: sorted(ws) for a, ws in by_author.items()}


def panel_authors_and_works(dataset) -> Tuple[List[str], Dict[str, List[str]]]:
    """The panel = authors with >= 2 works (sorted); returns (authors, works_by_author)."""
    by_author = _works_by_author(dataset)
    authors = sorted(a for a, ws in by_author.items() if len(ws) >= 2)
    return authors, {a: by_author[a] for a in authors}


def assign_folds(authors: List[str], works_by_author: Dict[str, List[str]], k: int) -> Dict[str, int]:
    """Deterministic per-author rotating round-robin: ``fold = (author_rank + work_rank) % k``."""
    fold: Dict[str, int] = {}
    for ai, a in enumerate(authors):
        for wj, w in enumerate(works_by_author[a]):
            fold[w] = (ai + wj) % k
    return fold


def _canonical(obj) -> bytes:
    # strict, sorted, compact — a stable content-hash payload (no NaN/Infinity)
    return dumps_strict(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _config_hash(algorithm: str, k: int, seed: int) -> str:
    return hashlib.sha256(_canonical({"algorithm": algorithm, "k_folds": k, "seed": seed})).hexdigest()


def _self_hash(manifest: dict) -> str:
    payload = {k: v for k, v in manifest.items() if k != "self_hash"}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def build_manifest(dataset, *, k: int = K_FOLDS, seed: int = SEED) -> dict:
    """Build the frozen manifest from a disk-anchored dataset (deterministic; no RNG)."""
    prov = getattr(dataset, "provenance", None)
    parent_digest = getattr(prov, "rows_digest", None)
    if not parent_digest:
        raise ScreeningPanelError("dataset has no provenance rows_digest to anchor the panel")
    authors, works_by_author = panel_authors_and_works(dataset)
    fold = assign_folds(authors, works_by_author, k)
    a2i = {a: i for i, a in enumerate(authors)}
    works = []
    for a in authors:
        for w in works_by_author[a]:
            works.append({"work_id": w, "author": a, "label": a2i[a], "fold": fold[w]})
    works.sort(key=lambda r: r["work_id"])
    fold_sizes = [sum(1 for r in works if r["fold"] == f) for f in range(k)]
    manifest = {
        "panel": PANEL_NAME,
        "algorithm": ALGORITHM,
        "k_folds": k,
        "seed": seed,
        "n_authors": len(authors),
        "n_works": len(works),
        "authors": authors,                       # index == label
        "parent_dataset_digest": parent_digest,   # disk-anchored full-corpus digest
        "config_hash": _config_hash(ALGORITHM, k, seed),
        "fold_sizes": fold_sizes,
        "works": works,
    }
    manifest["self_hash"] = _self_hash(manifest)
    return manifest


# ── verification ───────────────────────────────────────────────────────────────
def verify_manifest(manifest: dict) -> None:
    """Fail-closed integrity + invariant check of a loaded manifest (no dataset needed)."""
    if not isinstance(manifest, dict) or manifest.get("panel") != PANEL_NAME:
        raise ScreeningPanelError("not a screening_panel_v1 manifest")
    k = manifest["k_folds"]
    if manifest["algorithm"] != ALGORITHM or not isinstance(k, int) or k < 2:
        raise ScreeningPanelError("bad algorithm/k_folds")
    if manifest["config_hash"] != _config_hash(ALGORITHM, k, manifest["seed"]):
        raise ScreeningPanelError("config_hash mismatch")
    if manifest["self_hash"] != _self_hash(manifest):
        raise ScreeningPanelError("self_hash mismatch (manifest tampered)")
    authors = manifest["authors"]
    if len(set(authors)) != len(authors) or manifest["n_authors"] != len(authors):
        raise ScreeningPanelError("authors not unique / n_authors mismatch")
    works = manifest["works"]
    ids = [w["work_id"] for w in works]
    if len(set(ids)) != len(ids) or manifest["n_works"] != len(ids):
        raise ScreeningPanelError("work ids not unique / n_works mismatch")
    a2i = {a: i for i, a in enumerate(authors)}
    per_fold_authors: Dict[int, set] = {f: set() for f in range(k)}
    for w in works:
        f, lbl = w["fold"], w["label"]
        if type(f) is not int or not (0 <= f < k):
            raise ScreeningPanelError(f"work {w['work_id']}: bad fold {f!r}")
        if type(lbl) is bool or type(lbl) is not int:      # exact non-bool integer label
            raise ScreeningPanelError(f"work {w['work_id']}: label must be a non-bool int")
        if a2i.get(w["author"]) != lbl or w["work_id"].split("/", 1)[0] != w["author"]:
            raise ScreeningPanelError(f"work {w['work_id']}: authors[label] != author(work_id)")
        per_fold_authors[f].add(w["author"])
    all_authors = set(authors)
    for f in range(k):
        if not per_fold_authors[f]:
            raise ScreeningPanelError(f"test fold {f} is empty")
        # TRAIN fold f = all authors EXCEPT those whose works are entirely in test fold f.
        train_authors = {a for a in authors if any(w["fold"] != f for w in works if w["author"] == a)}
        if train_authors != all_authors:
            raise ScreeningPanelError(f"train fold {f} does not cover all {len(all_authors)} classes")
    # declared fold_sizes must equal the actual per-fold work counts (no lied-about sizes)
    actual_sizes = [sum(1 for w in works if w["fold"] == f) for f in range(k)]
    if list(manifest.get("fold_sizes", [])) != actual_sizes:
        raise ScreeningPanelError(
            f"fold_sizes {manifest.get('fold_sizes')!r} != actual per-fold counts {actual_sizes}")


def manifest_docs_path(cfg) -> pathlib.Path:
    return pathlib.Path(cfg.get_path("paths.docs", "docs")) / "screening_panel_v1.json"


def load_manifest_file(path, *, verify: bool = True) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise ScreeningPanelError(f"missing screening panel manifest: {p}")
    manifest = loads_strict(p.read_text(encoding="utf-8"))
    if verify:
        verify_manifest(manifest)
    return manifest


def load_manifest(root, *, verify: bool = True) -> dict:
    return load_manifest_file(pathlib.Path(root) / MANIFEST_PATH, verify=verify)


def build_panel_subset(dataset, manifest: dict):
    """Provenance-preserving subset of the full corpus restricted to the panel works, relabelled to
    the manifest's 43-author space (via ``derive_dataset`` — no hand-built Dataset)."""
    from .provenance import derive_dataset
    work_ids = {w["work_id"] for w in manifest["works"]}
    groups = [str(g) for g in dataset.groups.tolist()]
    missing = work_ids - set(groups)
    if missing:
        raise ScreeningPanelError(f"panel works absent from dataset: {sorted(missing)[:5]}")
    idx = [i for i, g in enumerate(groups) if g in work_ids]
    sub = derive_dataset(dataset, idx)
    if list(sub.authors) != list(manifest["authors"]):
        raise ScreeningPanelError("panel subset authors != manifest authors")
    return sub


def verify_result_against_panel(df, manifest: dict) -> None:
    """Compare an evaluated result frame against the CANONICAL panel manifest (not just a reference
    model): the exact set of works, each work's integer label and fold, and authors[label]==author.
    Applied to the primary model AND every baseline row before any metric is read."""
    if "fold" not in list(getattr(df, "columns", [])):     # the fold column is MANDATORY (no silent default)
        raise ScreeningPanelError("result frame must carry a 'fold' column")
    authors = manifest["authors"]
    expected = {w["work_id"]: (w["label"], w["fold"]) for w in manifest["works"]}
    seen = set()
    for r in df.itertuples():
        work_id = f"{r.test_author}/{r.test_book}"
        if work_id not in expected:
            raise ScreeningPanelError(f"result work {work_id} not in the frozen panel")
        if work_id in seen:
            raise ScreeningPanelError(f"result work {work_id} evaluated more than once")
        seen.add(work_id)
        lbl = r.true_label
        # exact non-bool Integral label (numpy ints ok; a bool or a float 2.0 is refused)
        if isinstance(lbl, bool) or not isinstance(lbl, numbers.Integral):
            raise ScreeningPanelError(f"{work_id}: true_label must be a non-bool Integral, got {lbl!r}")
        li = int(lbl)
        exp_label, exp_fold = expected[work_id]
        if li != exp_label:
            raise ScreeningPanelError(f"{work_id}: label {li} != canonical {exp_label}")
        if authors[li] != r.test_author:
            raise ScreeningPanelError(f"{work_id}: authors[label]={authors[li]!r} != {r.test_author!r}")
        rf = r.fold
        if isinstance(rf, bool) or not isinstance(rf, numbers.Integral) or int(rf) != exp_fold:
            raise ScreeningPanelError(f"{work_id}: fold {rf!r} != canonical {exp_fold}")
    if seen != set(expected):
        raise ScreeningPanelError(
            f"result covers {len(seen)} works, canonical panel has {len(expected)}")
