"""Audit-only dataset verifier, immutable audit-corpus builder, and published-root loader (§1.3/1.4).

255 per-work ``manifest.json`` cannot be laid down atomically inside the live ``data/frags_train``,
so audit-corpus preparation instead builds a **complete, immutable audit-corpus root** — a separate
content-addressed directory ``data/audit_corpus/<digest>/`` holding the selected works (chunks +
manifests) and their cleaned sources — published **whole and atomically** only after the §1.2 legacy
anchor + semantic-parity pass (with a per-chunk exact byte/filename compare against the source). The
live ``data/frags_train`` is never mutated. The confirmatory ``load_work_balanced_dataset`` reads the
published immutable root, never the live corpus.

Atomicity, symlink rejection, and path containment reuse the audited primitives of
:mod:`stylo.pipeline.bundle`. A partial root is never valid; an existing immutable root is never
silently overwritten; a same-name-different-content root is a fatal conflict.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile
from typing import Dict, Iterable, Optional, Sequence

from ...jsonio import dump_strict, dumps_strict, load_strict
from ...pipeline.bundle import (_real_within, _safe_name, _sha256_file, _verify_real_dir_chain)
from ...workdoc import (MANIFEST_NAME, chunker_config_hash, load_work_manifest)
from ..provenance import WORK_BALANCED_MANIFEST
from .semantic_parity import (assert_semantic_parity, dataset_semantic_digest,
                              verify_legacy_anchor, LEGACY_ANCHOR)

FRAGS_SUBDIR = "frags"
INPUT_CLEAN_SUBDIR = "input_clean"
CORPUS_MANIFEST_NAME = "corpus_manifest.json"
CURRENT_NAME = "current.json"
_CORPUS_DIGEST_VERSION = "paired_audit.corpus.v1"
_CONTENT_SUBDIRS = (FRAGS_SUBDIR, INPUT_CLEAN_SUBDIR)


class AuditCorpusError(RuntimeError):
    """Fail-closed: the audit corpus is partial, conflicting, drifted, or escapes its root."""


# ── audit-only dataset verifier (§1.4) ───────────────────────────────────────
def verify_audit_dataset(dataset) -> str:
    """The audit-only dataset↔estimand contract (§1.4): the dataset MUST be the
    work-balanced-manifest dataset (``dataset_contract=work_balanced_manifest``), independent of
    which estimator cell (A0..A4) runs on it. Asserts the declared contract axis and recomputes the
    loader-bound provenance digest over the CURRENT arrays, so a mutated/relabeled Dataset cannot
    pose as the audit dataset. Disk-anchored forgery resistance (a self-consistent but off-disk
    forged Dataset) is provided separately by
    :func:`stylo.eval.provenance.verify_dataset_against_disk` in the confirmatory runner. Returns
    the loader-agnostic semantic digest for downstream binding.
    """
    from ..provenance import DatasetProvenance, canonical_digest

    prov = getattr(dataset, "provenance", None)
    if not isinstance(prov, DatasetProvenance):
        raise AuditCorpusError("audit dataset carries no DatasetProvenance")
    if prov.loader_kind != WORK_BALANCED_MANIFEST:
        raise AuditCorpusError(
            f"audit dataset must be work_balanced_manifest, got loader_kind={prov.loader_kind!r}"
        )
    recomputed = canonical_digest(
        [str(t) for t in dataset.texts], [int(v) for v in dataset.y],
        [str(g) for g in dataset.groups], list(dataset.authors), prov.row_ids,
        loader_kind=prov.loader_kind, chunker_config_hash=prov.chunker_config_hash,
    )
    if recomputed != prov.rows_digest:
        raise AuditCorpusError("audit dataset rows_digest mismatch — mutated, relabeled or forged")
    return dataset_semantic_digest(dataset)


# ── content digest over the immutable-root subtrees ──────────────────────────
def _walk_real_files(base: pathlib.Path) -> list[pathlib.Path]:
    """All files under ``base``; a symlinked file OR a symlinked subdirectory is fatal (a symlinked
    subtree substitution must be rejected outright, not merely inferred from a digest change)."""
    out: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        for d in dirnames:
            if (pathlib.Path(dirpath) / d).is_symlink():
                raise AuditCorpusError(
                    f"symlinked directory in audit corpus (rejected): {pathlib.Path(dirpath) / d}")
        for name in filenames:
            p = pathlib.Path(dirpath) / name
            if p.is_symlink():
                raise AuditCorpusError(f"symlinked file in audit corpus (rejected): {p}")
            out.append(p)
    return out


def _tree_content_digest(root: pathlib.Path, subdirs: Sequence[str]) -> str:
    """Deterministic sha256 over ``(posix relpath, file sha256)`` for every file under each subdir.

    Excludes the top-level ``corpus_manifest.json``/``current.json`` (which record this digest).
    """
    h = hashlib.sha256()
    h.update(len(_CORPUS_DIGEST_VERSION).to_bytes(8, "big") + _CORPUS_DIGEST_VERSION.encode("utf-8"))
    for subdir in sorted(subdirs):
        base = root / subdir
        if not base.exists():
            raise AuditCorpusError(f"audit corpus missing required subtree: {subdir}")
        rows = []
        for p in _walk_real_files(base):
            rel = p.relative_to(root).as_posix()
            rows.append((rel, _sha256_file(p)))
        for rel, digest in sorted(rows):
            h.update(len(rel).to_bytes(8, "big") + rel.encode("utf-8"))
            h.update(bytes.fromhex(digest))
    return h.hexdigest()


def _self_hash(body: Dict) -> str:
    return hashlib.sha256(dumps_strict(body, sort_keys=True).encode("utf-8")).hexdigest()


# ── selection validation ─────────────────────────────────────────────────────
def _validate_selection(work_ids: Iterable[str], available: Sequence[str]) -> list[str]:
    """Return the sorted requested work ids after an exact-set check against ``available``.

    A duplicate, a missing work, or an extra work not present in the source is a hard fail.
    """
    if isinstance(work_ids, (str, bytes)):
        raise AuditCorpusError("work_ids must be an iterable of work ids, not a string")
    requested = list(work_ids)
    if any(type(w) is not str for w in requested):
        raise AuditCorpusError("every work id must be exactly str")
    if len(set(requested)) != len(requested):
        raise AuditCorpusError("duplicate work id in selection")
    avail = set(available)
    missing = sorted(set(requested) - avail)
    if missing:
        raise AuditCorpusError(f"selection references works absent from the source: {missing[:3]}")
    return sorted(requested)


# ── copy one work into the staging root, proving byte/filename equality ──────
def _copy_work(source_frags_root: pathlib.Path, input_clean_root: pathlib.Path,
               work_id: str, expected_chash: str, staging: pathlib.Path) -> Dict:
    author, book = work_id.split("/", 1)
    src_wdir = source_frags_root / author / book
    manifest, _texts = load_work_manifest(
        src_wdir, input_clean_root=input_clean_root, expected_chunker_config_hash=expected_chash,
    )  # validates structure, provenance, byte hashes, ordinals BEFORE any copy

    dest_wdir = staging / FRAGS_SUBDIR / author / book
    dest_wdir.mkdir(parents=True, exist_ok=False)
    # copy the exact manifest bytes (keep it byte-identical to the source)
    shutil.copyfile(src_wdir / MANIFEST_NAME, dest_wdir / MANIFEST_NAME)
    for entry in manifest.chunks:
        src_chunk = src_wdir / entry.path
        if src_chunk.is_symlink():
            raise AuditCorpusError(f"{work_id}: source chunk is a symlink: {entry.path}")
        dst_chunk = dest_wdir / entry.path
        shutil.copyfile(src_chunk, dst_chunk)
        if _sha256_file(src_chunk) != _sha256_file(dst_chunk):
            raise AuditCorpusError(f"{work_id}: chunk byte copy mismatch for {entry.path}")

    src_clean = input_clean_root / author / f"{book}.txt"
    dst_clean = staging / INPUT_CLEAN_SUBDIR / author / f"{book}.txt"
    dst_clean.parent.mkdir(parents=True, exist_ok=True)
    if src_clean.is_symlink():
        raise AuditCorpusError(f"{work_id}: cleaned source is a symlink")
    shutil.copyfile(src_clean, dst_clean)
    if _sha256_file(src_clean) != _sha256_file(dst_clean):
        raise AuditCorpusError(f"{work_id}: cleaned-source byte copy mismatch")

    return {
        "work_id": work_id,
        "author_id": author,
        "provenance_sha256": manifest.provenance_sha256,
        "chunker_config_hash": manifest.chunker_config_hash,
        "n_chunks": len(manifest.chunks),
        "chunk_paths": [c.path for c in manifest.chunks],
    }


def _published_root_matches(versioned: pathlib.Path, digest: str) -> bool:
    """True iff an existing immutable root is a real contained dir whose recomputed content digest
    equals its own name AND whose corpus manifest self-hash and recorded digest agree."""
    if not _real_within(versioned, versioned.parent, must_dir=True):
        return False
    if versioned.name != digest:
        return False
    # exact top-level inventory: no smuggled extra file/dir and no symlink at the root
    top = list(os.scandir(versioned))
    if {e.name for e in top} != {FRAGS_SUBDIR, INPUT_CLEAN_SUBDIR, CORPUS_MANIFEST_NAME}:
        return False
    if any(e.is_symlink() for e in top):
        return False
    try:
        if _tree_content_digest(versioned, _CONTENT_SUBDIRS) != digest:
            return False
    except AuditCorpusError:
        return False
    manifest_path = versioned / CORPUS_MANIFEST_NAME
    if not _real_within(manifest_path, versioned, must_file=True):
        return False
    try:
        body = load_strict(manifest_path)
    except Exception:
        return False
    recorded = dict(body)
    self_hash = recorded.pop("self_hash", None)
    if self_hash != _self_hash(recorded):
        return False
    return recorded.get("audit_corpus_digest") == digest


def verify_published_corpus(published_root: pathlib.Path | str) -> Dict:
    """Fail-closed verification of an immutable audit-corpus root; returns its corpus manifest."""
    published_root = pathlib.Path(published_root)
    _verify_real_dir_chain(published_root)
    if not published_root.is_dir():
        raise AuditCorpusError(f"published audit corpus is not a directory: {published_root}")
    digest = published_root.name
    if not _published_root_matches(published_root, digest):
        raise AuditCorpusError(
            f"published audit corpus is partial, tampered, or conflicting: {published_root}"
        )
    return load_strict(published_root / CORPUS_MANIFEST_NAME)


def build_audit_corpus(
    *,
    source_frags_root: pathlib.Path | str,
    input_clean_root: pathlib.Path | str,
    cfg,
    audit_parent: pathlib.Path | str,
    work_ids: Optional[Iterable[str]] = None,
    legacy_anchor: str = LEGACY_ANCHOR,
    exclude_authors: Iterable[str] = (),
    unknown_name: str = "unknown",
    expected_n_works: Optional[int] = None,
) -> pathlib.Path:
    """Build and atomically publish a whole immutable audit-corpus root; returns its path.

    Proves the §1.2 legacy anchor and semantic parity over the source, copies the selected works
    byte-for-byte into ``audit_parent/<digest>/``, publishes it whole and atomically (conflict on a
    same-name-different-content root is fatal), then re-loads and re-proves parity + per-chunk byte
    equality from the published root. ``data/frags_train`` is never mutated.
    """
    from ...corpus import load_dataset
    from ...workdoc import load_work_balanced_dataset

    source_frags_root = pathlib.Path(source_frags_root)
    input_clean_root = pathlib.Path(input_clean_root)
    audit_parent = pathlib.Path(audit_parent)
    expected_chash = chunker_config_hash(cfg)

    # 1. legacy load of the source; 2. anchor pin; 3. WB load; 4. semantic parity (whole source)
    legacy_full = load_dataset(source_frags_root, exclude_authors=exclude_authors,
                               unknown_name=unknown_name)
    verify_legacy_anchor(legacy_full, expected=legacy_anchor)
    wb_full = load_work_balanced_dataset(source_frags_root, cfg=cfg,
                                         input_clean_root=input_clean_root,
                                         exclude_authors=exclude_authors, unknown_name=unknown_name)
    semantic_digest = assert_semantic_parity(legacy_full, wb_full)

    available = sorted({str(g) for g in wb_full.groups})
    selected = _validate_selection(work_ids, available) if work_ids is not None else available
    is_full = set(selected) == set(available)
    if expected_n_works is not None and len(selected) != expected_n_works:
        raise AuditCorpusError(
            f"selection has {len(selected)} works, expected {expected_n_works}"
        )

    _verify_real_dir_chain(audit_parent)
    audit_parent.mkdir(parents=True, exist_ok=True)
    _verify_real_dir_chain(audit_parent)

    staging = pathlib.Path(tempfile.mkdtemp(dir=audit_parent, prefix=".staging_"))
    published_root: Optional[pathlib.Path] = None
    newly_published = False
    try:
        (staging / FRAGS_SUBDIR).mkdir()
        (staging / INPUT_CLEAN_SUBDIR).mkdir()
        work_records = [
            _copy_work(source_frags_root, input_clean_root, w, expected_chash, staging)
            for w in selected
        ]
        digest = _tree_content_digest(staging, _CONTENT_SUBDIRS)

        body = {
            "schema": _CORPUS_DIGEST_VERSION,
            "audit_corpus_digest": digest,
            "source_semantic_parity_digest": semantic_digest,   # parity over the FULL source universe
            "legacy_anchor": legacy_anchor,
            "legacy_anchor_is_full_universe": is_full,
            "chunker_config_hash": expected_chash,
            "n_works": len(selected),
            "n_chunks": sum(r["n_chunks"] for r in work_records),
            "works": work_records,
        }
        body["self_hash"] = _self_hash(body)
        dump_strict(body, staging / CORPUS_MANIFEST_NAME, trailing_newline=True)

        versioned = audit_parent / digest
        if versioned.exists() or versioned.is_symlink():
            if not _published_root_matches(versioned, digest):
                raise AuditCorpusError(
                    f"immutable audit root {digest} already exists with different/partial content "
                    "(refusing to overwrite — fatal identity conflict)"
                )
            shutil.rmtree(staging)          # identical complete root already published — reuse
        else:
            os.replace(staging, versioned)  # whole immutable root, single atomic rename
            newly_published = True
        staging = None
        published_root = versioned
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    # re-load from the published root and re-prove parity + per-chunk byte/filename equality BEFORE
    # the pointer is made resolvable (§1.3: the root is published whole only AFTER the equality
    # proof passes; a build that fails the proof must never leave a resolvable pointer). If the proof
    # fails on a root we just created, remove it so the audit_parent keeps no uncertified root.
    try:
        _reverify_published_root(published_root, cfg, legacy_anchor if is_full else None,
                                 semantic_digest if is_full else None,
                                 source_frags_root, input_clean_root, selected)
    except Exception:
        if newly_published and _real_within(published_root, audit_parent, must_dir=True):
            shutil.rmtree(published_root, ignore_errors=True)
        raise

    # only now flip the pointer atomically
    tmp_ptr = pathlib.Path(tempfile.mktemp(dir=audit_parent, prefix=".current_"))
    dump_strict({"schema": _CORPUS_DIGEST_VERSION, "version": published_root.name},
                tmp_ptr, trailing_newline=True)
    os.replace(tmp_ptr, audit_parent / CURRENT_NAME)
    return published_root


def _reverify_published_root(published_root: pathlib.Path, cfg, legacy_anchor: Optional[str],
                             expected_semantic: Optional[str], source_frags_root: pathlib.Path,
                             input_clean_root: pathlib.Path, selected: Sequence[str]) -> None:
    """Re-load both arms from the immutable root and prove parity, anchor (full only), and exact
    per-chunk byte/filename equality vs the source."""
    verify_published_corpus(published_root)
    from ...corpus import load_dataset
    from ...workdoc import load_work_balanced_dataset

    root_frags = published_root / FRAGS_SUBDIR
    root_clean = published_root / INPUT_CLEAN_SUBDIR
    legacy_ds = load_dataset(root_frags)
    wb_ds = load_work_balanced_dataset(root_frags, cfg=cfg, input_clean_root=root_clean)
    parity = assert_semantic_parity(legacy_ds, wb_ds)
    if expected_semantic is not None and parity != expected_semantic:
        raise AuditCorpusError("published-root parity digest differs from the source parity digest")
    if legacy_anchor is not None:
        verify_legacy_anchor(legacy_ds, expected=legacy_anchor)

    expected_chash = chunker_config_hash(cfg)
    for work_id in selected:
        author, book = work_id.split("/", 1)
        src_wdir = source_frags_root / author / book
        dst_wdir = root_frags / author / book
        manifest, _ = load_work_manifest(dst_wdir, input_clean_root=root_clean,
                                         expected_chunker_config_hash=expected_chash)
        src_names = {c.path for c in load_work_manifest(
            src_wdir, input_clean_root=input_clean_root,
            expected_chunker_config_hash=expected_chash)[0].chunks}
        if {c.path for c in manifest.chunks} != src_names:
            raise AuditCorpusError(f"{work_id}: published chunk filenames differ from source")
        for entry in manifest.chunks:
            if _sha256_file(src_wdir / entry.path) != _sha256_file(dst_wdir / entry.path):
                raise AuditCorpusError(f"{work_id}: published chunk bytes differ from source")


# ── published-root loader (§1.3) ─────────────────────────────────────────────
def load_audit_dataset(published_root: pathlib.Path | str, cfg, *, weighting: str,
                       exclude_authors: Iterable[str] = (), unknown_name: str = "unknown"):
    """Load the audit dataset from a VERIFIED immutable root (never the live corpus).

    ``weighting`` selects the arm: the legacy A0 estimand loads via the recursive loader, the
    work-balanced arm via the manifest loader; both read the same immutable ``frags``/``input_clean``
    subtrees so the arms stay byte-identical.
    """
    from ...corpus import load_dataset
    from ...workdoc import load_work_balanced_dataset
    from ..work_weighting import WORK_BALANCED, resolve_training_weighting

    published_root = pathlib.Path(published_root)
    verify_published_corpus(published_root)
    root_frags = published_root / FRAGS_SUBDIR
    root_clean = published_root / INPUT_CLEAN_SUBDIR
    if resolve_training_weighting(weighting) == WORK_BALANCED:
        return load_work_balanced_dataset(root_frags, cfg=cfg, input_clean_root=root_clean,
                                          exclude_authors=exclude_authors, unknown_name=unknown_name)
    return load_dataset(root_frags, exclude_authors=exclude_authors, unknown_name=unknown_name)


def resolve_current_root(audit_parent: pathlib.Path | str) -> pathlib.Path:
    """Resolve the immutable root named by the atomic ``current.json`` pointer (verified)."""
    audit_parent = pathlib.Path(audit_parent)
    _verify_real_dir_chain(audit_parent)
    ptr = audit_parent / CURRENT_NAME
    if not _real_within(ptr, audit_parent, must_file=True):
        raise AuditCorpusError("current pointer missing, a symlink, or escapes the audit parent")
    pointer = load_strict(ptr)
    if pointer.get("schema") != _CORPUS_DIGEST_VERSION:
        raise AuditCorpusError("current pointer schema mismatch")
    version = pointer.get("version")
    if not isinstance(version, str) or not _safe_name(version):   # rejects '', '.', '..', separators
        raise AuditCorpusError("current pointer version is not a safe token")
    root = audit_parent / version
    if not _real_within(root, audit_parent, must_dir=True):
        raise AuditCorpusError("pointer target missing, a symlink, or escapes the audit parent")
    verify_published_corpus(root)
    return root
