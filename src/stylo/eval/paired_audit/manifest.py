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
    disk (it never self-signs the committed one). Both self-hashes are checked, then exact equality."""
    verify_manifest_self_hash(committed)
    verify_manifest_self_hash(rebuilt)
    if committed != rebuilt:
        raise FoldManifestError("committed fold manifest does not match the disk-rebuilt manifest")


def _common_checks(manifest, schema) -> None:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("works"), list):
        raise FoldManifestError("manifest must be a dict with a works list")
    if manifest.get("schema") != schema:
        raise FoldManifestError(f"expected schema {schema!r}, got {manifest.get('schema')!r}")
    verify_manifest_self_hash(manifest)
    # 'fold_index iff tested' BEFORE any sort, so a tested row with fold_index=None fails typed
    if any((w.get("fold_index") is not None) != bool(w.get("tested")) for w in manifest["works"]):
        raise FoldManifestError("fold_index must be set iff the work is tested")
    tested = [w for w in manifest["works"] if w["tested"]]
    if sorted(w["fold_index"] for w in tested) != list(range(len(tested))):
        raise FoldManifestError("tested fold indices must be contiguous 0..n-1")
    # class orders must equal the actual author sets (not just the right length)
    authors = sorted({w["author_id"] for w in manifest["works"]})
    tested_authors = sorted({w["author_id"] for w in manifest["works"] if w["tested"]})
    if manifest.get("probability_class_order") != authors:
        raise FoldManifestError("probability_class_order must equal the sorted train authors")
    if manifest.get("metric_label_order") != tested_authors:
        raise FoldManifestError("metric_label_order must equal the sorted tested authors")


def assert_lobo_universe(manifest) -> None:
    """Fail-closed unless the LOBO manifest matches the frozen 47/255/43/251 universe with exactly
    four single-work train-only authors."""
    _common_checks(manifest, LOBO_SCHEMA)
    for key, want in LOBO_UNIVERSE.items():
        if key == "n_singleton_train_only":
            continue
        if manifest.get(key) != want:
            raise FoldManifestError(f"LOBO {key}={manifest.get(key)} != {want}")
    if len(manifest["probability_class_order"]) != 47:
        raise FoldManifestError("LOBO probability_class_order must be 47-wide")
    if len(manifest["metric_label_order"]) != 43:
        raise FoldManifestError("LOBO metric_label_order must be 43-wide")
    train_only = [w for w in manifest["works"] if not w["tested"]]
    train_only_authors = {w["author_id"] for w in train_only}
    if len(train_only) != 4 or len(train_only_authors) != 4:
        raise FoldManifestError("LOBO must have exactly four single-work train-only authors")
    awc = Counter(w["author_id"] for w in manifest["works"])
    if any(awc[a] != 1 for a in train_only_authors):
        raise FoldManifestError("each train-only author must own exactly one work")


def assert_ruaa_universe(manifest) -> None:
    """Fail-closed unless the RuAA manifest is exactly 137 whole works / 22 authors, both orders 22,
    with a bound selection digest and every work tested."""
    _common_checks(manifest, RUAA_SCHEMA)
    if manifest.get("n_train_works") != RUAA_UNIVERSE["n_works"]:
        raise FoldManifestError(f"RuAA must hold exactly {RUAA_UNIVERSE['n_works']} works")
    if manifest.get("n_train_authors") != RUAA_UNIVERSE["n_authors"]:
        raise FoldManifestError(f"RuAA must hold exactly {RUAA_UNIVERSE['n_authors']} authors")
    if len(manifest["probability_class_order"]) != 22 or len(manifest["metric_label_order"]) != 22:
        raise FoldManifestError("RuAA probability and metric orders must both be 22-wide")
    if manifest.get("n_tested_works") != RUAA_UNIVERSE["n_works"]:
        raise FoldManifestError("RuAA is a whole-work panel: every work must be tested")
    if not manifest.get("selection_digest"):
        raise FoldManifestError("RuAA manifest must bind a selection_digest")
    if not all(w["tested"] for w in manifest["works"]):
        raise FoldManifestError("RuAA whole-work panel: every work must be tested")
