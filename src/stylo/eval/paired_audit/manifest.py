"""LOBO and RuAA fold-manifest builder, verifier, and frozen-universe invariants (§1.6/§12).

Each dataset manifest records the work/author IDs, the per-work fold, the ``probability_class_order``
(LOBO 47 / RuAA 22), the frozen ``metric_label_order`` (LOBO 43 tested / RuAA 22), the parent-dataset
digest, the RuAA selection digest, the algorithm/seed/config hash, and a self-hash. The confirmatory
runner **rebuilds the expected manifest from disk and requires exact equality** with the committed one
— it never self-signs a passed manifest.

Frozen LOBO universe: 47 train-pool authors / 255 train works; 43 tested authors / 251 tested folds;
the four single-work authors (``goncharov, grigorovich, reshetnikov, voloshin``) stay train-only.
Frozen RuAA universe: exactly 137 whole works / 22 authors, probability and metric orders both 22.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Optional

from ...jsonio import dumps_strict

LOBO_SCHEMA = "lobo_fold_manifest_v1"
RUAA_SCHEMA = "ruaa_fold_manifest_v1"
_KIND_SCHEMA = {"lobo": LOBO_SCHEMA, "ruaa": RUAA_SCHEMA}

# frozen expected universe counts (§1.1/§1.5)
LOBO_UNIVERSE = {"n_train_authors": 47, "n_train_works": 255, "n_tested_authors": 43,
                 "n_tested_works": 251, "n_singleton_train_only": 4}
RUAA_UNIVERSE = {"n_authors": 22, "n_works": 137}
# the four single-work authors kept train-only in the frozen LOBO universe (§1.1)
LOBO_SINGLETON_AUTHORS = ("goncharov", "grigorovich", "reshetnikov", "voloshin")


class FoldManifestError(ValueError):
    """Fail-closed: a fold manifest is malformed, its self-hash is wrong, or the universe is off."""


def fold_manifest_self_hash(body) -> str:
    """sha256 over the manifest body with ``self_hash`` removed (canonical strict-JSON)."""
    payload = {k: v for k, v in body.items() if k != "self_hash"}
    return hashlib.sha256(dumps_strict(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _works_and_authors(dataset):
    groups = [str(g) for g in dataset.groups]
    works = sorted(set(groups))
    work_author = {w: w.split("/", 1)[0] for w in works}
    return works, work_author, Counter(work_author.values())


def build_fold_manifest(dataset_kind: str, dataset, *, parent_dataset_digest: str, algorithm: str,
                        seed: int, config_hash: str, selection_digest: Optional[str] = None) -> dict:
    """Build a fold manifest from a dataset. LOBO leaves single-work authors train-only; RuAA is a
    whole-work panel (every selected work is tested)."""
    if dataset_kind not in _KIND_SCHEMA:
        raise FoldManifestError(f"unknown dataset_kind {dataset_kind!r}")
    if not parent_dataset_digest:
        raise FoldManifestError("parent_dataset_digest is required")
    if dataset_kind == "ruaa" and not selection_digest:
        raise FoldManifestError("RuAA manifest requires a selection_digest (three-digest binding)")
    if dataset_kind == "ruaa":
        # close the three-digest binding at build: the supplied selection_digest must equal the
        # derived subset's own provenance selection digest when the parent carries one.
        prov_sel = getattr(getattr(dataset, "provenance", None), "selection_manifest_digest", None)
        if prov_sel is not None and prov_sel != selection_digest:
            raise FoldManifestError("RuAA selection_digest != the derived subset's provenance digest")

    works, work_author, awc = _works_and_authors(dataset)
    authors = sorted(set(work_author.values()))
    if dataset_kind == "lobo":
        tested_works = sorted(w for w in works if awc[work_author[w]] > 1)   # singletons train-only
        tested_authors = sorted(a for a in authors if awc[a] > 1)
    else:
        tested_works = list(works)                                          # whole-work panel
        tested_authors = list(authors)
    fold_of = {w: i for i, w in enumerate(tested_works)}
    work_rows = [{"work_id": w, "author_id": work_author[w],
                  "fold_index": fold_of.get(w), "tested": w in fold_of} for w in works]

    body = {
        "schema": _KIND_SCHEMA[dataset_kind],
        "dataset_kind": dataset_kind,
        "n_train_authors": len(authors),
        "n_train_works": len(works),
        "n_tested_authors": len(tested_authors),
        "n_tested_works": len(tested_works),
        "probability_class_order": authors,
        "metric_label_order": tested_authors,
        "works": work_rows,
        "parent_dataset_digest": parent_dataset_digest,
        "algorithm": algorithm,
        "seed": int(seed),
        "config_hash": config_hash,
    }
    if dataset_kind == "ruaa":
        body["selection_digest"] = selection_digest
    body["self_hash"] = fold_manifest_self_hash(body)
    return body


def verify_manifest_self_hash(manifest) -> None:
    if not isinstance(manifest, dict) or "self_hash" not in manifest:
        raise FoldManifestError("manifest is not a dict with a self_hash")
    if manifest["self_hash"] != fold_manifest_self_hash(manifest):
        raise FoldManifestError("fold manifest self-hash mismatch (tampered)")


def verify_manifest_matches_rebuilt(committed, rebuilt) -> None:
    """The runner requires the committed manifest to EXACTLY equal a manifest freshly rebuilt from
    disk (it never self-signs the committed one), then runs the frozen-universe validator for its
    schema. Both self-hashes are checked before the equality compare."""
    verify_manifest_self_hash(committed)
    verify_manifest_self_hash(rebuilt)
    if committed != rebuilt:
        raise FoldManifestError("committed fold manifest does not match the disk-rebuilt manifest")
    schema = committed.get("schema")
    if schema == LOBO_SCHEMA:
        assert_lobo_universe(committed)
    elif schema == RUAA_SCHEMA:
        assert_ruaa_universe(committed)
    else:
        raise FoldManifestError(f"unknown manifest schema {schema!r}")


def _recompute(manifest, schema) -> dict:
    """Recompute EVERY count from ``works`` (never trust the declared ``n_*``) after strict per-work
    type and consistency checks; require the declared fields to equal the recomputed values."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("works"), list) or not manifest["works"]:
        raise FoldManifestError("manifest must be a dict with a non-empty works list")
    if manifest.get("schema") != schema:
        raise FoldManifestError(f"expected schema {schema!r}, got {manifest.get('schema')!r}")
    verify_manifest_self_hash(manifest)
    for w in manifest["works"]:
        if not isinstance(w, dict):
            raise FoldManifestError("each work row must be a dict")
        if type(w.get("work_id")) is not str or type(w.get("author_id")) is not str:
            raise FoldManifestError("work_id/author_id must be str")
        if type(w.get("tested")) is not bool:
            raise FoldManifestError("tested must be a bool")
        fi = w.get("fold_index")
        if not (fi is None or (type(fi) is int and fi >= 0)):
            raise FoldManifestError("fold_index must be a non-negative int or null")
        if (fi is not None) != w["tested"]:
            raise FoldManifestError("fold_index must be set iff the work is tested")
        if w["author_id"] != w["work_id"].split("/", 1)[0]:
            raise FoldManifestError("author_id must equal the work_id prefix")
    work_ids = [w["work_id"] for w in manifest["works"]]
    if len(set(work_ids)) != len(work_ids):
        raise FoldManifestError("duplicate work_id in manifest")
    authors = sorted({w["author_id"] for w in manifest["works"]})
    tested_authors = sorted({w["author_id"] for w in manifest["works"] if w["tested"]})
    rc = {"n_train_works": len(manifest["works"]), "n_train_authors": len(authors),
          "n_tested_works": sum(1 for w in manifest["works"] if w["tested"]),
          "n_tested_authors": len(tested_authors), "authors": authors, "tested_authors": tested_authors}
    for k in ("n_train_works", "n_train_authors", "n_tested_works", "n_tested_authors"):
        if manifest.get(k) != rc[k]:
            raise FoldManifestError(f"{k}={manifest.get(k)} != {rc[k]} recomputed from works")
    folds = sorted(w["fold_index"] for w in manifest["works"] if w["tested"])
    if folds != list(range(len(folds))):
        raise FoldManifestError("tested fold indices must be contiguous 0..n-1")
    if manifest.get("probability_class_order") != authors:
        raise FoldManifestError("probability_class_order must equal the sorted train authors")
    if manifest.get("metric_label_order") != tested_authors:
        raise FoldManifestError("metric_label_order must equal the sorted tested authors")
    return rc


def assert_lobo_universe(manifest) -> None:
    """Fail-closed unless the LOBO manifest matches the frozen 47/255/43/251 universe with exactly the
    four registered single-work train-only authors — every count recomputed from ``works``."""
    rc = _recompute(manifest, LOBO_SCHEMA)
    if (rc["n_train_authors"], rc["n_train_works"]) != (47, 255):
        raise FoldManifestError(f"LOBO train universe {rc['n_train_authors']}/{rc['n_train_works']} != 47/255")
    if (rc["n_tested_authors"], rc["n_tested_works"]) != (43, 251):
        raise FoldManifestError(f"LOBO tested universe {rc['n_tested_authors']}/{rc['n_tested_works']} != 43/251")
    train_only_authors = sorted(set(rc["authors"]) - set(rc["tested_authors"]))
    if train_only_authors != sorted(LOBO_SINGLETON_AUTHORS):
        raise FoldManifestError(f"LOBO train-only authors {train_only_authors} != the 4 registered singletons")
    awc = Counter(w["author_id"] for w in manifest["works"])
    if any(awc[a] != 1 for a in train_only_authors):
        raise FoldManifestError("each train-only singleton author must own exactly one work")


def assert_ruaa_universe(manifest) -> None:
    """Fail-closed unless the RuAA manifest is exactly 137 whole works / 22 authors (recomputed from
    ``works``), both orders 22, every work tested, with a bound selection digest."""
    rc = _recompute(manifest, RUAA_SCHEMA)
    if rc["n_train_works"] != RUAA_UNIVERSE["n_works"] or rc["n_train_authors"] != RUAA_UNIVERSE["n_authors"]:
        raise FoldManifestError(f"RuAA universe {rc['n_train_authors']}/{rc['n_train_works']} != 22/137")
    if rc["n_tested_works"] != RUAA_UNIVERSE["n_works"]:
        raise FoldManifestError("RuAA is a whole-work panel: every work must be tested")
    if not manifest.get("selection_digest"):
        raise FoldManifestError("RuAA manifest must bind a selection_digest")
