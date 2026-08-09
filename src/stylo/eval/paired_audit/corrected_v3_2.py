"""Preparation-only corrected corpus and fold contract for paired-audit v3.2.

Nothing here imports an evaluator or creates a RunPlan.  It reads the exact
registered v3.1 evidence root, derives a new content-addressed child with the
three adjudicated exclusions, runs a corpus-wide content isolation audit, and
writes unapproved v3.2 fold manifests to a caller-owned output root.
"""
from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import os
import pathlib
import re
import shutil
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from ...jsonio import dump_strict, dumps_strict, load_strict
from ...pipeline.bundle import _verify_real_dir_chain
from ...workdoc import MANIFEST_NAME, load_work_manifest
from ...domain.corpus_identity import find_cross_work_content_overlaps
from . import corpus as historical_corpus


PROTOCOL_VERSION = "paired_audit_protocol_v3_2"
IDENTITY_CONTRACT_VERSION = "paired_audit.full_work_identity.v3_2"
HISTORICAL_PARENT_DIGEST = "15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029"
HISTORICAL_PARENT_SCHEMA = "paired_audit.corpus.v1"
CORPUS_SCHEMA = "paired_audit.corrected_corpus.v3_2"
LOBO_SCHEMA = "lobo_fold_manifest_v3_2"
RUAA_SCHEMA = "ruaa_fold_manifest_v3_2"
CANDIDATE_SCHEMA = "paired_audit.local_candidate_preparation.v3_2"
BUNDLE_STORAGE_CONTRACT_VERSION = "paired_audit.preparation_bundle.v1"
BUNDLE_PARENT_NAME = "paired_audit_v3_2_bundles"
BUNDLE_INVENTORY_NAME = "bundle_inventory_v1.json"
CORRECTED_CORPUS_DIR = "corrected_corpus"
CORRECTED_MANIFEST_NAME = "corrected_corpus_manifest_v3_2.json"
SHA256SUMS_NAME = "SHA256SUMS"
PUBLISH_LOCK_NAME = ".publish.lock"
PREPARATION_STATUS = "local_candidate_preparation_pending_review"

EXCLUSIONS = (
    ("turgenev/записки_охотника", "collection_umbrella_content_component"),
    ("serafimovich/у_нас_и_у_них", "adjudicated_authorship_mismatch"),
    ("sevsky/дон_на_костылях", "adjudicated_source_quality_exclusion"),
)
EXCLUDED_WORK_IDS = tuple(row[0] for row in EXCLUSIONS)
LOBO_SINGLETON_AUTHORS = ("goncharov", "grigorovich", "reshetnikov", "voloshin")
EXPECTED_LOBO_BASENAME_COLLISIONS = (
    ("bunin/деревня", "grigorovich/деревня"),
    ("radov/rasskazi", "zoshenko/rasskazi"),
)
PROVENANCE_LIMITATIONS = (
    {
        "work_ids": ["radov/rasskazi", "zoshenko/rasskazi"],
        "upstream_source_label": "local/неизвестно",
        "statement": "source-label limitation only; not a basename or content-overlap collision",
    },
)
LOBO_COUNTS = (47, 252, 43, 248)
RUAA_COUNTS = (22, 134)
RUAA_PARENT_COUNTS = (22, 137)
FOLD_ALGORITHM = {"lobo": "leave_one_work_out", "ruaa": "whole_work"}
FOLD_SEED = 42
HISTORICAL_WORK_COUNT = 255

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE = re.compile(r"^[^/\\\x00]+$")


class CorrectedCorpusError(RuntimeError):
    """A v3.2 preparation identity, content, or output contract failed closed."""


def _digest(namespace: str, value: object) -> str:
    data = dumps_strict(value, sort_keys=True).encode("utf-8")
    encoded = namespace.encode("utf-8")
    return hashlib.sha256(len(encoded).to_bytes(8, "big") + encoded + data).hexdigest()


def _self_hash(value: Mapping) -> str:
    return hashlib.sha256(dumps_strict(dict(value), sort_keys=True).encode("utf-8")).hexdigest()


def _file_hash(path: pathlib.Path) -> str:
    _regular(path, "hashed payload")
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CorrectedCorpusError(f"hashed payload changed after lstat: {path}")
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _mode(path: pathlib.Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):04o}"


def _normalised_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def full_work_id(value: object) -> str:
    """The sole v3.2 work key: exact NFC UTF-8 ``author_id/work_slug``.

    A bare basename cannot reach a join, selection, fold, reference, or lookup.
    """
    if type(value) is not str:
        raise CorrectedCorpusError("work identity must be an exact str")
    normalized = unicodedata.normalize("NFC", value)
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CorrectedCorpusError("work identity must be UTF-8 encodable") from exc
    if value != normalized or normalized.count("/") != 1:
        raise CorrectedCorpusError("work identity must be NFC full author_id/work_slug, never a basename")
    author, slug = normalized.split("/", 1)
    if not _SAFE.fullmatch(author) or not _SAFE.fullmatch(slug) or author in (".", "..") or slug in (".", ".."):
        raise CorrectedCorpusError("unsafe full work identity")
    return normalized


def resolve_full_work(catalog: Mapping[str, Mapping], value: object) -> Mapping:
    """Fail closed for a bare or unknown key; this intentionally has no basename branch."""
    key = full_work_id(value)
    try:
        return catalog[key]
    except KeyError as exc:
        raise CorrectedCorpusError(f"unknown full work identity: {key}") from exc


def exclusion_policy() -> dict:
    body = {
        "schema": "paired_audit.exclusion_policy.v3_2",
        "protocol_version": PROTOCOL_VERSION,
        "exclusions": [{"work_id": work, "reason": reason} for work, reason in EXCLUSIONS],
    }
    return {**body, "digest": _digest(body["schema"], body)}


def applicability_matrix() -> dict:
    applied = (
        ("stylo", "A0"), ("stylo", "A1"), ("stylo", "A2"), ("stylo", "A3"), ("stylo", "A4"),
        ("bow_lr", "A0"), ("bow_lr", "A1"), ("bow_lr", "A2"), ("bow_lr", "A4"),
        ("delta_cos:500", "A0"), ("delta_cos:500", "A2"), ("delta_cos:500", "A3"), ("delta_cos:500", "A4"),
        ("char_cos", "A0"), ("char_cos", "A4"), ("majority", "A0"),
    )
    holm = tuple(row for row in applied if row[1] != "A0")
    if len(applied) != 16 or len(holm) != 11 or any(model == "stylo_stack" for model, _ in applied):
        raise CorrectedCorpusError("v3.2 applicability matrix invariant failed")
    body = {
        "schema": "paired_audit.applicability.v3_2",
        "protocol_version": PROTOCOL_VERSION,
        "applied_cells": [{"model": model, "cell": cell} for model, cell in applied],
        "holm_family": [{"model": model, "cell": cell} for model, cell in holm],
        "family_alpha": 0.05,
        "stylo_stack": "withdrawn_no_replacement",
    }
    return {**body, "digest": _digest(body["schema"], body)}


def _regular(path: pathlib.Path, label: str) -> os.stat_result:
    """Require one real, single-link regular file using exactly one ``lstat``."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise CorrectedCorpusError(f"missing or unreadable file in {label}: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CorrectedCorpusError(f"symlink, hardlink, or special file in {label}: {path}")
    return info


def _directory(path: pathlib.Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CorrectedCorpusError(f"missing or unreadable directory in {label}: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise CorrectedCorpusError(f"symlink or non-directory in {label}: {path}")


def _require_mode(path: pathlib.Path, expected: int, label: str) -> None:
    actual = stat.S_IMODE(path.lstat().st_mode)
    if actual != expected:
        raise CorrectedCorpusError(
            f"mode drift in {label}: {path} has {actual:04o}, expected {expected:04o}"
        )


def _created_dir(path: pathlib.Path, *, parents: bool = False) -> None:
    """Create a staging/output directory and set its creation-time canonical mode."""
    path.mkdir(mode=0o755, parents=parents, exist_ok=False)
    os.chmod(path, 0o755)


def _created_file_mode(path: pathlib.Path) -> None:
    """Set mode only for a file just created by this preparation."""
    os.chmod(path, 0o644)


def _entries(path: pathlib.Path, label: str) -> list[os.DirEntry]:
    _directory(path, label)
    return sorted(os.scandir(path), key=lambda item: item.name)


def _work_record(root: pathlib.Path, author: str, slug: str) -> dict:
    work_id = full_work_id(f"{author}/{slug}")
    work_root = root / "frags" / author / slug
    source = root / "input_clean" / author / f"{slug}.txt"
    _regular(source, f"source {work_id}")
    _regular(work_root / MANIFEST_NAME, f"manifest {work_id}")
    try:
        manifest, texts = load_work_manifest(work_root, input_clean_root=root / "input_clean")
    except Exception as exc:
        raise CorrectedCorpusError(f"invalid work manifest for {work_id}: {exc}") from exc
    if manifest.work_id != work_id or manifest.author_id != author:
        raise CorrectedCorpusError(f"manifest author/work identity mismatch for {work_id}")
    chunks = []
    for entry, text in zip(manifest.chunks, texts, strict=True):
        path = work_root / entry.path
        _regular(path, f"chunk {work_id}")
        chunks.append({
            "path": entry.path,
            "span_ordinal": entry.span_ordinal,
            "byte_sha256": _file_hash(path),
            "text_sha256": entry.text_sha256,
            "normalized_sha256": _normalised_hash(text),
            "text": text,
        })
    source_text = source.read_text(encoding="utf-8")
    identity = {
        "schema": IDENTITY_CONTRACT_VERSION,
        "work_id": work_id,
        "manifest_sha256": _file_hash(work_root / MANIFEST_NAME),
        "source_sha256": _file_hash(source),
        "chunks": [{key: row[key] for key in row if key != "text"} for row in chunks],
    }
    return {
        "work_id": work_id,
        "author_id": author,
        "work_slug": slug,
        "manifest_sha256": identity["manifest_sha256"],
        "source_sha256": identity["source_sha256"],
        "source_normalized_sha256": _normalised_hash(source_text),
        "work_content_identity": _digest("paired_audit.work_content.v3_2", identity),
        "content_component_identity": _digest(
            "paired_audit.content_component.v3_2",
            {"source": _normalised_hash(source_text), "chunks": [row["normalized_sha256"] for row in chunks]},
        ),
        "chunks": chunks,
    }


def _validate_corpus_structure(root: pathlib.Path, *, child: bool) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate the complete path/type/link shape before parsing any manifest."""
    try:
        _verify_real_dir_chain(root)
    except Exception as exc:
        raise CorrectedCorpusError(f"unsafe corpus root path chain: {root}") from exc
    manifest_name = CORRECTED_MANIFEST_NAME if child else "corpus_manifest.json"
    expected = {"frags", "input_clean", manifest_name}
    top = _entries(root, "corpus root")
    if {entry.name for entry in top} != expected:
        raise CorrectedCorpusError("corpus root has missing or extra top-level members")
    _directory(root / "frags", "frags")
    _directory(root / "input_clean", "input_clean")
    _regular(root / manifest_name, "corpus manifest")
    if child:
        _require_mode(root, 0o755, "corrected corpus root")
        _require_mode(root / "frags", 0o755, "frags")
        _require_mode(root / "input_clean", 0o755, "input_clean")
        _require_mode(root / manifest_name, 0o644, "corrected corpus manifest")

    frag_authors = _entries(root / "frags", "frags")
    clean_authors = _entries(root / "input_clean", "input_clean")
    if [entry.name for entry in frag_authors] != [entry.name for entry in clean_authors]:
        raise CorrectedCorpusError("frags/input_clean author inventories differ")
    works: list[tuple[str, tuple[str, ...]]] = []
    for entry in frag_authors:
        author = entry.name
        if not _SAFE.fullmatch(author) or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise CorrectedCorpusError("unsafe author entry")
        frag_author = pathlib.Path(entry.path)
        clean_author = root / "input_clean" / author
        _directory(frag_author, f"frags author {author}")
        _directory(clean_author, f"input_clean author {author}")
        if child:
            _require_mode(frag_author, 0o755, f"frags author {author}")
            _require_mode(clean_author, 0o755, f"input_clean author {author}")
        expected_sources = set()
        slugs = []
        for book in _entries(frag_author, f"author {author}"):
            slug = book.name
            if not _SAFE.fullmatch(slug) or book.is_symlink() or not book.is_dir(follow_symlinks=False):
                raise CorrectedCorpusError("unsafe work entry")
            work_dir = pathlib.Path(book.path)
            if child:
                _require_mode(work_dir, 0o755, f"work {author}/{slug}")
            files = _entries(work_dir, f"work {author}/{slug}")
            names = {item.name for item in files}
            if MANIFEST_NAME not in names or len(names) < 2 or any(
                name != MANIFEST_NAME and (not _SAFE.fullmatch(name) or not name.endswith(".txt"))
                for name in names
            ):
                raise CorrectedCorpusError("work has an extra, missing, or unsafe member")
            for item in files:
                path = pathlib.Path(item.path)
                _regular(path, f"work {author}/{slug}")
                if child:
                    _require_mode(path, 0o644, f"work {author}/{slug}")
            expected_sources.add(f"{slug}.txt")
            slugs.append(slug)
        clean_entries = _entries(clean_author, f"input_clean {author}")
        if {entry.name for entry in clean_entries} != expected_sources:
            raise CorrectedCorpusError("input_clean has missing, extra, symlinked, or special members")
        for source_entry in clean_entries:
            source = pathlib.Path(source_entry.path)
            _regular(source, f"input_clean {author}")
            if child:
                _require_mode(source, 0o644, f"input_clean {author}")
        works.append((author, tuple(slugs)))
    return tuple(works)


def _scan_root(root: pathlib.Path, *, child: bool) -> tuple[dict, ...]:
    works = _validate_corpus_structure(root, child=child)
    records = []
    for author, slugs in works:
        for slug in slugs:
            records.append(_work_record(root, author, slug))
    records.sort(key=lambda row: row["work_id"])
    if len({row["work_id"] for row in records}) != len(records):
        raise CorrectedCorpusError("duplicate full work_id is fatal")
    return tuple(records)


def verify_historical_parent(root: pathlib.Path | str) -> tuple[dict, tuple[dict, ...]]:
    root = pathlib.Path(root)
    if root.name != HISTORICAL_PARENT_DIGEST:
        raise CorrectedCorpusError("historical parent digest identity-first rejection")
    records = _scan_root(root, child=False)
    try:
        manifest = historical_corpus.verify_published_corpus(root)
    except Exception as exc:
        raise CorrectedCorpusError(f"historical parent verification failed: {exc}") from exc
    if manifest.get("schema") != HISTORICAL_PARENT_SCHEMA or manifest.get("audit_corpus_digest") != HISTORICAL_PARENT_DIGEST:
        raise CorrectedCorpusError("historical parent schema/digest mismatch")
    if len(records) != HISTORICAL_WORK_COUNT or manifest.get("n_works") != HISTORICAL_WORK_COUNT:
        raise CorrectedCorpusError(
            f"historical parent must contain exactly {HISTORICAL_WORK_COUNT} works"
        )
    catalog = work_identity_catalog(records)
    return {
        "historical_parent_digest": HISTORICAL_PARENT_DIGEST,
        "historical_parent_manifest_self_hash": manifest.get("self_hash"),
        "historical_parent_manifest_sha256": _file_hash(root / "corpus_manifest.json"),
        "full_work_identity_catalog_digest": catalog["digest"],
    }, records


def work_identity_catalog(records: Sequence[Mapping]) -> dict:
    rows = []
    for row in sorted(records, key=lambda item: item["work_id"]):
        work_id = full_work_id(row["work_id"])
        if row["author_id"] != work_id.split("/", 1)[0] or row["work_slug"] != work_id.split("/", 1)[1]:
            raise CorrectedCorpusError("author-prefix/work-slug mismatch is fatal")
        rows.append({
            "work_id": work_id,
            "author_id": row["author_id"], "work_slug": row["work_slug"],
            "work_content_identity": row["work_content_identity"],
            "content_component_identity": row["content_component_identity"],
        })
    if len({row["work_id"] for row in rows}) != len(rows):
        raise CorrectedCorpusError("duplicate full work ID in identity catalog")
    body = {"schema": IDENTITY_CONTRACT_VERSION, "work_ids": rows}
    return {**body, "digest": _digest(IDENTITY_CONTRACT_VERSION, body)}


def basename_collision_inventory(records: Sequence[Mapping], *, expected: Sequence[Sequence[str]]) -> dict:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        work_id = full_work_id(row["work_id"])
        groups[work_id.split("/", 1)[1]].append(work_id)
    found = tuple(sorted(tuple(sorted(ids)) for ids in groups.values() if len(ids) > 1))
    expected_normalized = tuple(sorted(tuple(sorted(full_work_id(item) for item in ids)) for ids in expected))
    if found != expected_normalized:
        raise CorrectedCorpusError(
            f"unexpected/missing/third basename collision component: found={found!r}, expected={expected_normalized!r}"
        )
    body = {
        "schema": "paired_audit.basename_collision_inventory.v3_2",
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "components": [list(ids) for ids in found],
        "diagnostic_only": True,
        "content_isolation_allowlist": [],
    }
    return {**body, "digest": _digest(body["schema"], body)}


def content_isolation_audit(records: Sequence[Mapping], *, lobo_tested_work_ids: Sequence[str] = (),
                            ruaa_work_ids: Sequence[str] = ()) -> dict:
    """Corpus-wide exact/normalized/component audit; basename is deliberately absent here."""
    exact: dict[str, set[str]] = defaultdict(set)
    normalized: dict[str, set[str]] = defaultdict(set)
    texts, groups = [], []
    for record in records:
        for chunk in record["chunks"]:
            exact[chunk["byte_sha256"]].add(record["work_id"])
            normalized[chunk["normalized_sha256"]].add(record["work_id"])
            texts.append(chunk["text"])
            groups.append(record["work_id"])
        exact[record["source_sha256"]].add(record["work_id"])
        normalized[record["source_normalized_sha256"]].add(record["work_id"])
    exact_rows = [{"sha256": digest, "work_ids": sorted(owner)} for digest, owner in sorted(exact.items()) if len(owner) > 1]
    normalized_rows = [{"sha256": digest, "work_ids": sorted(owner)} for digest, owner in sorted(normalized.items()) if len(owner) > 1]
    overlaps = find_cross_work_content_overlaps(texts, groups)
    component_rows = [{
        "left_work": item.left_work, "right_work": item.right_work, "kind": item.kind,
        "containment": item.containment, "evidence": item.evidence,
    } for item in overlaps]
    work_ids = {row["work_id"] for row in records}
    lobo_tested = set(lobo_tested_work_ids) or set(_lobo_tested(records))
    ruaa = set(ruaa_work_ids)
    if not lobo_tested <= work_ids or not ruaa <= work_ids:
        raise CorrectedCorpusError("train/test leakage audit work identity escaped the corrected corpus")
    body = {
        "schema": "paired_audit.content_isolation_audit.v3_2",
        "exact_duplicate_bytes": exact_rows,
        "normalized_text_duplicates": normalized_rows,
        "cross_work_chunk_component_overlap": component_rows,
        "collection_member_overlap": component_rows,
        "train_test_content_leakage": {
            "lobo": {"tested_works": len(lobo_tested), "train_test_pairs_checked": len(lobo_tested) * (len(work_ids) - 1), "findings": []},
            "ruaa": {"tested_works": len(ruaa), "train_test_pairs_checked": len(ruaa) * max(len(ruaa) - 1, 0), "findings": []},
        },
    }
    if exact_rows or normalized_rows or component_rows:
        raise CorrectedCorpusError("content-isolation audit found an overlap; basename diagnostics cannot allowlist it")
    return {**body, "digest": _digest(body["schema"], body)}


def _ruaa_selection(parent_selection: Iterable[str], available: set[str]) -> tuple[list[str], dict]:
    values = [full_work_id(value) for value in parent_selection]
    if values != sorted(values) or len(values) != len(set(values)) or not set(values) <= available:
        raise CorrectedCorpusError("RuAA historical selection must be sorted unique full work IDs from parent")
    if (len({value.split('/', 1)[0] for value in values}), len(values)) != RUAA_PARENT_COUNTS:
        raise CorrectedCorpusError("historical RuAA selection must be 22 authors / 137 works")
    if any(values.count(value) != 1 for value in EXCLUDED_WORK_IDS):
        raise CorrectedCorpusError("each registered exclusion must occur exactly once in historical RuAA selection")
    child = [value for value in values if value not in EXCLUDED_WORK_IDS]
    if (len({value.split('/', 1)[0] for value in child}), len(child)) != RUAA_COUNTS:
        raise CorrectedCorpusError("corrected RuAA selection must be 22 authors / 134 works")
    parent_digest = _digest("paired_audit.ruaa_historical_selection.v3_2", values)
    return child, {
        "historical_selection_digest": parent_digest,
        "selection_digest": _digest("paired_audit.ruaa_selection.v3_2", {"parent": parent_digest, "work_ids": child}),
    }


def _lobo_tested(records: Sequence[Mapping]) -> list[str]:
    counts = Counter(row["author_id"] for row in records)
    return sorted(row["work_id"] for row in records if counts[row["author_id"]] > 1)


def _assert_counts(records: Sequence[Mapping], lobo_tested: Sequence[str], ruaa: Sequence[str]) -> None:
    counts = (len({row["author_id"] for row in records}), len(records),
              len({value.split("/", 1)[0] for value in lobo_tested}), len(lobo_tested))
    if counts != LOBO_COUNTS:
        raise CorrectedCorpusError(f"LOBO counts {counts} != {LOBO_COUNTS}")
    singleton = sorted(set(row["author_id"] for row in records) - {value.split("/", 1)[0] for value in lobo_tested})
    if tuple(singleton) != LOBO_SINGLETON_AUTHORS:
        raise CorrectedCorpusError("LOBO singleton authors drift")
    if (len({value.split('/', 1)[0] for value in ruaa}), len(ruaa)) != RUAA_COUNTS:
        raise CorrectedCorpusError("RuAA counts drift")


def _copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    _regular(source, "historical parent copy source")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        source_info = os.fstat(source_fd)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
            raise CorrectedCorpusError("historical parent copy source changed after lstat")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        try:
            destination_info = os.fstat(destination_fd)
            if not stat.S_ISREG(destination_info.st_mode) or destination_info.st_nlink != 1:
                raise CorrectedCorpusError("staging copy destination is not a new single-link file")
            while True:
                block = os.read(source_fd, 1 << 20)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fchmod(destination_fd, 0o644)
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)
    if _file_hash(source) != _file_hash(destination):
        raise CorrectedCorpusError("byte copy mismatch")


def _inventory(root: pathlib.Path) -> tuple[list[dict], str]:
    """Pure read-only child-content inventory (the historical digest domain)."""
    rows = []
    for subtree in ("frags", "input_clean"):
        _directory(root / subtree, "child inventory")
        for current, dirs, files in os.walk(root / subtree, followlinks=False):
            directory = pathlib.Path(current)
            _directory(directory, "child inventory")
            for name in dirs:
                path = directory / name
                _directory(path, "child inventory")
            for name in files:
                path = directory / name
                _regular(path, "generated child")
                rows.append({"path": path.relative_to(root).as_posix(), "sha256": _file_hash(path), "mode": _mode(path)})
    rows.sort(key=lambda row: row["path"])
    return rows, _digest("paired_audit.content_inventory.v3_2", rows)


def _recursive_inventory(root: pathlib.Path, *, excluded: Iterable[str] = ()) -> list[dict]:
    """Pure exact recursive inventory, including directories and rejecting every unsafe inode."""
    excluded_set = set(excluded)
    _directory(root, "bundle root")
    rows: list[dict] = []
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        directory = pathlib.Path(current)
        _directory(directory, "bundle inventory")
        relative_dir = directory.relative_to(root).as_posix()
        if relative_dir != ".":
            rows.append({"path": relative_dir + "/", "type": "directory", "mode": _mode(directory)})
        dirs.sort()
        files.sort()
        for name in dirs:
            _directory(directory / name, "bundle inventory")
        for name in files:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            info = _regular(path, "bundle inventory")
            if relative not in excluded_set:
                rows.append({
                    "path": relative, "type": "file", "mode": _mode(path),
                    "size": info.st_size, "sha256": _file_hash(path),
                })
    rows.sort(key=lambda row: row["path"])
    return rows


def _canonical_sums(root: pathlib.Path) -> str:
    rows = _recursive_inventory(root, excluded=(SHA256SUMS_NAME,))
    files = [row for row in rows if row["type"] == "file"]
    return "".join(f"{row['sha256']}  {row['path']}\n" for row in files)


def _write_json(value: Mapping, path: pathlib.Path) -> None:
    dump_strict(value, path, trailing_newline=True)
    _created_file_mode(path)


def _corpus_manifest(*, root: pathlib.Path, records: Sequence[Mapping], parent_identity: Mapping,
                     policy: Mapping, catalog: Mapping, basename: Mapping, isolation: Mapping,
                     config_hash: str, protocol_sha256: str) -> dict:
    inventory, digest = _inventory(root)
    body = {
        "schema": CORPUS_SCHEMA, "protocol_version": PROTOCOL_VERSION,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "historical_parent": dict(parent_identity), "exclusion_policy": dict(policy),
        "corrected_content_inventory": inventory, "corrected_content_inventory_digest": digest,
        "full_work_identity_catalog_digest": catalog["digest"],
        "basename_collision_audit_digest": basename["digest"],
        "content_isolation_audit_digest": isolation["digest"],
        "provenance_limitations": list(PROVENANCE_LIMITATIONS),
        "config_hash": config_hash, "protocol_sha256": protocol_sha256, "n_works": len(records),
    }
    body["self_hash"] = _self_hash(body)
    return body


def _assemble_child(parent: pathlib.Path, root: pathlib.Path, records: Sequence[Mapping], *,
                    parent_identity: Mapping, policy: Mapping, catalog: Mapping, basename: Mapping,
                    isolation: Mapping, config_hash: str, protocol_sha256: str,
                    fault_inject: Callable[[str], None] | None) -> dict:
    _created_dir(root)
    _created_dir(root / "frags")
    _created_dir(root / "input_clean")
    injected = False
    created_authors: set[str] = set()
    for row in records:
        author = row["author_id"]
        if author not in created_authors:
            _created_dir(root / "frags" / author)
            _created_dir(root / "input_clean" / author)
            created_authors.add(author)
        source = parent / "frags" / author / row["work_slug"]
        destination = root / "frags" / author / row["work_slug"]
        _created_dir(destination)
        _copy(source / MANIFEST_NAME, destination / MANIFEST_NAME)
        for chunk in row["chunks"]:
            _copy(source / chunk["path"], destination / chunk["path"])
        _copy(parent / "input_clean" / author / f"{row['work_slug']}.txt",
              root / "input_clean" / author / f"{row['work_slug']}.txt")
        if not injected:
            injected = True
            _fault(fault_inject, "during_child_assembly")
    body = _corpus_manifest(
        root=root, records=records, parent_identity=parent_identity, policy=policy,
        catalog=catalog, basename=basename, isolation=isolation, config_hash=config_hash,
        protocol_sha256=protocol_sha256,
    )
    _write_json(body, root / CORRECTED_MANIFEST_NAME)
    return body


def _verify_child(root: pathlib.Path, expected: Mapping) -> tuple[dict, ...]:
    records = _scan_root(root, child=True)
    manifest = load_strict(root / CORRECTED_MANIFEST_NAME)
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise CorrectedCorpusError("historical/v3.1 corpus schema rejected identity-first")
    body = dict(manifest)
    self_hash = body.pop("self_hash", None)
    if self_hash != _self_hash(body) or manifest != expected:
        raise CorrectedCorpusError("corrected corpus manifest tamper/conflict")
    inventory, digest = _inventory(root)
    if digest != manifest["corrected_content_inventory_digest"] or inventory != manifest["corrected_content_inventory"]:
        raise CorrectedCorpusError("corrected corpus bytes/modes inventory drift")
    return records


def _fold(kind: str, records: Sequence[Mapping], corpus: Mapping, *, selection_digest: str, config_hash: str) -> dict:
    records = tuple(sorted(records, key=lambda row: row["work_id"]))
    authors = sorted({row["author_id"] for row in records})
    tested = _lobo_tested(records) if kind == "lobo" else [row["work_id"] for row in records]
    tested_authors = sorted({row.split("/", 1)[0] for row in tested})
    index = {work: position for position, work in enumerate(tested)}
    rows = [{
        "work_id": row["work_id"], "author_id": row["author_id"],
        "work_content_identity": row["work_content_identity"], "content_component_identity": row["content_component_identity"],
        "tested": row["work_id"] in index, "fold_index": index.get(row["work_id"]),
    } for row in records]
    body = {
        "schema": LOBO_SCHEMA if kind == "lobo" else RUAA_SCHEMA, "protocol_version": PROTOCOL_VERSION,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION, "dataset_kind": kind,
        "historical_parent_digest": corpus["historical_parent"]["historical_parent_digest"],
        "exclusion_policy_digest": corpus["exclusion_policy"]["digest"],
        "corrected_corpus_digest": corpus["corrected_content_inventory_digest"],
        "corrected_corpus_manifest_self_hash": corpus["self_hash"],
        "full_work_identity_catalog_digest": corpus["full_work_identity_catalog_digest"],
        "basename_collision_audit_digest": corpus["basename_collision_audit_digest"],
        "content_isolation_audit_digest": corpus["content_isolation_audit_digest"],
        "selection_digest": selection_digest, "algorithm": FOLD_ALGORITHM[kind], "seed": FOLD_SEED,
        "config_hash": config_hash, "applicability_matrix_digest": applicability_matrix()["digest"],
        "n_train_authors": len(authors), "n_train_works": len(rows),
        "n_tested_authors": len(tested_authors), "n_tested_works": len(tested),
        "probability_class_order": authors, "metric_label_order": tested_authors, "works": rows,
    }
    body["self_hash"] = _self_hash(body)
    return body


def assert_v3_2_fold_manifest(manifest: Mapping, *, kind: str,
                              expected: Mapping | None = None) -> None:
    expected_schema = LOBO_SCHEMA if kind == "lobo" else RUAA_SCHEMA if kind == "ruaa" else None
    if expected_schema is None or not isinstance(manifest, Mapping) or manifest.get("schema") != expected_schema:
        raise CorrectedCorpusError("historical/v3.1 fold schema rejected identity-first")
    body = dict(manifest)
    self_hash = body.pop("self_hash", None)
    if manifest.get("protocol_version") != PROTOCOL_VERSION or self_hash != _self_hash(body):
        raise CorrectedCorpusError("v3.2 fold protocol/self-hash mismatch")
    if expected is None:
        raise CorrectedCorpusError("fold acceptance requires trusted-input reconstruction")
    if dict(manifest) != dict(expected):
        raise CorrectedCorpusError(f"{kind} fold differs from trusted-input reconstruction")
    identities = ("historical_parent_digest", "exclusion_policy_digest", "corrected_corpus_digest",
                  "corrected_corpus_manifest_self_hash", "full_work_identity_catalog_digest",
                  "basename_collision_audit_digest", "content_isolation_audit_digest", "selection_digest",
                  "config_hash", "applicability_matrix_digest")
    if any(not isinstance(manifest.get(key), str) or not _HEX64.fullmatch(manifest[key]) for key in identities):
        raise CorrectedCorpusError("fold identity binding missing or malformed")
    if manifest["historical_parent_digest"] != HISTORICAL_PARENT_DIGEST or manifest["exclusion_policy_digest"] != exclusion_policy()["digest"]:
        raise CorrectedCorpusError("fold parent/policy binding drift")
    rows = manifest.get("works")
    if not isinstance(rows, list) or not rows:
        raise CorrectedCorpusError("fold work rows missing")
    ids = [full_work_id(row.get("work_id")) if isinstance(row, Mapping) else None for row in rows]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise CorrectedCorpusError("fold duplicate/unsorted full work IDs")
    for row in rows:
        if row["author_id"] != row["work_id"].split("/", 1)[0]:
            raise CorrectedCorpusError("fold author-prefix mismatch")
        if row["tested"] != (row["fold_index"] is not None):
            raise CorrectedCorpusError("fold test index mismatch")
    authors = sorted({row["author_id"] for row in rows})
    tested = [row for row in rows if row["tested"]]
    if [row["fold_index"] for row in tested] != list(range(len(tested))):
        raise CorrectedCorpusError("fold indices must be contiguous")
    counts = (len(authors), len(rows), len({row["author_id"] for row in tested}), len(tested))
    expected_counts = LOBO_COUNTS if kind == "lobo" else (*RUAA_COUNTS, *RUAA_COUNTS)
    if counts != expected_counts:
        raise CorrectedCorpusError("fold count universe mismatch")
    if manifest["probability_class_order"] != authors or manifest["metric_label_order"] != sorted({row["author_id"] for row in tested}):
        raise CorrectedCorpusError("fold label order mismatch")


def _fault(callback: Callable[[str], None] | None, point: str) -> None:
    if callback is not None:
        callback(point)


def _trusted_derivation(parent: pathlib.Path, ruaa_parent_selection: Sequence[str]) -> dict:
    parent_identity, all_records = verify_historical_parent(parent)
    parent_ids = [row["work_id"] for row in all_records]
    if any(parent_ids.count(value) != 1 for value in EXCLUDED_WORK_IDS):
        raise CorrectedCorpusError("the exact three registered exclusions cannot derive the corrected child")
    records = tuple(row for row in all_records if row["work_id"] not in EXCLUDED_WORK_IDS)
    if len(records) != HISTORICAL_WORK_COUNT - len(EXCLUDED_WORK_IDS):
        raise CorrectedCorpusError("corrected child count differs from exact exclusions")
    ruaa, ruaa_selection = _ruaa_selection(ruaa_parent_selection, set(parent_ids))
    lobo_tested = _lobo_tested(records)
    _assert_counts(records, lobo_tested, ruaa)
    catalog = work_identity_catalog(records)
    basename_lobo = basename_collision_inventory(records, expected=EXPECTED_LOBO_BASENAME_COLLISIONS)
    ruaa_set = set(ruaa)
    ruaa_records = tuple(row for row in records if row["work_id"] in ruaa_set)
    basename_ruaa = basename_collision_inventory(ruaa_records, expected=())
    basename = {
        "schema": "paired_audit.basename_collision_audit.v3_2",
        "lobo": basename_lobo, "ruaa": basename_ruaa,
    }
    basename["digest"] = _digest(basename["schema"], basename)
    isolation = content_isolation_audit(
        records, lobo_tested_work_ids=lobo_tested, ruaa_work_ids=ruaa,
    )
    return {
        "parent_identity": parent_identity, "all_records": all_records, "records": records,
        "policy": exclusion_policy(), "catalog": catalog, "basename": basename,
        "isolation": isolation, "ruaa": ruaa, "ruaa_records": ruaa_records,
        "ruaa_selection": ruaa_selection,
    }


def _file_map(root: pathlib.Path) -> dict[str, dict]:
    rows = _recursive_inventory(root, excluded=("candidate.json", SHA256SUMS_NAME))
    return {
        row["path"]: {key: row[key] for key in ("sha256", "size", "mode")}
        for row in rows if row["type"] == "file"
    }


def _candidate_body(*, corpus: Mapping, lobo: Mapping, ruaa: Mapping, isolation: Mapping,
                    basename: Mapping, parent_identity: Mapping, ruaa_selection: Mapping,
                    inventory: Mapping, files: Mapping) -> dict:
    body = {
        "schema": CANDIDATE_SCHEMA,
        "storage_contract_version": BUNDLE_STORAGE_CONTRACT_VERSION,
        "protocol_version": PROTOCOL_VERSION, "status": PREPARATION_STATUS,
        "review_state": "independent_manifest_review_required",
        "freeze_status": "unapproved_no_freeze_root_pin",
        "production_evaluator_status": "unregistered",
        "confirmatory_execution_status": "hard_disabled",
        "headline_status": "not_authorized", "publication_status": "not_authorized",
        "bundle_layout": {
            "corrected_corpus_relative_root": CORRECTED_CORPUS_DIR,
            "exact_inventory": BUNDLE_INVENTORY_NAME, "sha256sums": SHA256SUMS_NAME,
            "root_mode": "0755",
        },
        "historical_parent": dict(parent_identity),
        "corrected_corpus": {
            "relative_root": CORRECTED_CORPUS_DIR,
            "digest": corpus["corrected_content_inventory_digest"],
            "self_hash": corpus["self_hash"],
        },
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "full_work_identity_catalog_digest": corpus["full_work_identity_catalog_digest"],
        "basename_collision_audit_digest": basename["digest"],
        "content_isolation_audit_digest": isolation["digest"],
        "ruaa_selection": dict(ruaa_selection), "applicability": applicability_matrix(),
        "folds": {"lobo_self_hash": lobo["self_hash"], "ruaa_self_hash": ruaa["self_hash"]},
        "exact_inventory_self_hash": inventory["self_hash"], "files": dict(files),
    }
    body["self_hash"] = _self_hash(body)
    return body


_BUNDLE_TOP = {
    CORRECTED_CORPUS_DIR, "lobo_fold_manifest_v3_2.json", "ruaa_fold_manifest_v3_2.json",
    "content_isolation_audit_v3_2.json", "basename_collision_audit_v3_2.json",
    BUNDLE_INVENTORY_NAME, "candidate.json", SHA256SUMS_NAME,
}


def _verify_bundle_shape(root: pathlib.Path) -> None:
    try:
        _verify_real_dir_chain(root)
    except Exception as exc:
        raise CorrectedCorpusError(f"unsafe bundle path chain: {root}") from exc
    entries = _entries(root, "preparation bundle")
    if {entry.name for entry in entries} != _BUNDLE_TOP:
        raise CorrectedCorpusError("bundle has missing or unexpected members")
    _require_mode(root, 0o755, "bundle root")
    _directory(root / CORRECTED_CORPUS_DIR, "bundle-local corrected corpus")
    for name in _BUNDLE_TOP - {CORRECTED_CORPUS_DIR}:
        _regular(root / name, f"bundle payload {name}")
    for row in _recursive_inventory(root):
        expected = "0755" if row["type"] == "directory" else "0644"
        if row["mode"] != expected:
            raise CorrectedCorpusError(
                f"mode drift in bundle member {row['path']}: {row['mode']} != {expected}"
            )
    _validate_corpus_structure(root / CORRECTED_CORPUS_DIR, child=True)


def verify_v3_2_candidate(bundle_root: pathlib.Path | str, *,
                          historical_parent_root: pathlib.Path | str,
                          ruaa_parent_selection: Iterable[str], config_hash: str,
                          protocol_sha256: str, require_basename: bool = True,
                          fault_inject: Callable[[str], None] | None = None) -> dict:
    """Pure disk-backed reconstruction and exact verification of one complete bundle."""
    if not _HEX64.fullmatch(config_hash) or not _HEX64.fullmatch(protocol_sha256):
        raise CorrectedCorpusError("config/protocol hashes must be SHA256")
    root = pathlib.Path(bundle_root)
    parent = pathlib.Path(historical_parent_root)
    selection = tuple(ruaa_parent_selection)
    _verify_bundle_shape(root)  # path/type/link/mode checks happen before every parse

    candidate = load_strict(root / "candidate.json")
    if not isinstance(candidate, dict) or candidate.get("schema") != CANDIDATE_SCHEMA:
        raise CorrectedCorpusError("old or malformed candidate schema rejected identity-first")
    if candidate.get("storage_contract_version") != BUNDLE_STORAGE_CONTRACT_VERSION:
        raise CorrectedCorpusError("old unsafe preparation storage contract rejected")
    candidate_body = dict(candidate)
    candidate_hash = candidate_body.pop("self_hash", None)
    if candidate_hash != _self_hash(candidate_body):
        raise CorrectedCorpusError("candidate self-hash mismatch")
    if require_basename and root.name != candidate_hash:
        raise CorrectedCorpusError("bundle directory basename differs from committed candidate digest")

    inventory = load_strict(root / BUNDLE_INVENTORY_NAME)
    payload_rows = _recursive_inventory(
        root, excluded=(BUNDLE_INVENTORY_NAME, "candidate.json", SHA256SUMS_NAME),
    )
    inventory_body = {
        "schema": "paired_audit.preparation_bundle_inventory.v1",
        "storage_contract_version": BUNDLE_STORAGE_CONTRACT_VERSION,
        "root_mode": "0755", "entries": payload_rows,
    }
    inventory_body["self_hash"] = _self_hash(inventory_body)
    if inventory != inventory_body:
        raise CorrectedCorpusError("exact bundle inventory mismatch")
    files = _file_map(root)
    if candidate.get("files") != files:
        raise CorrectedCorpusError("candidate recursive file map mismatch")
    expected_sums = _canonical_sums(root)
    if (root / SHA256SUMS_NAME).read_bytes() != expected_sums.encode("utf-8"):
        raise CorrectedCorpusError("non-canonical or stale SHA256SUMS")

    trusted = _trusted_derivation(parent, selection)
    child_root = root / CORRECTED_CORPUS_DIR
    expected_corpus = _corpus_manifest(
        root=child_root, records=trusted["records"], parent_identity=trusted["parent_identity"],
        policy=trusted["policy"], catalog=trusted["catalog"], basename=trusted["basename"],
        isolation=trusted["isolation"], config_hash=config_hash, protocol_sha256=protocol_sha256,
    )
    child_records = _verify_child(child_root, expected_corpus)
    if child_records != trusted["records"]:
        raise CorrectedCorpusError("corrected child is not the exact trusted three-exclusion derivation")

    loaded_basename = load_strict(root / "basename_collision_audit_v3_2.json")
    loaded_isolation = load_strict(root / "content_isolation_audit_v3_2.json")
    if loaded_basename != trusted["basename"] or loaded_isolation != trusted["isolation"]:
        raise CorrectedCorpusError("audit differs from trusted-input reconstruction")
    lobo_expected = _fold(
        "lobo", trusted["records"], expected_corpus,
        selection_digest=_digest(
            "paired_audit.lobo_selection.v3_2", [row["work_id"] for row in trusted["records"]]
        ), config_hash=config_hash,
    )
    ruaa_expected = _fold(
        "ruaa", trusted["ruaa_records"], expected_corpus,
        selection_digest=trusted["ruaa_selection"]["selection_digest"], config_hash=config_hash,
    )
    lobo_loaded = load_strict(root / "lobo_fold_manifest_v3_2.json")
    ruaa_loaded = load_strict(root / "ruaa_fold_manifest_v3_2.json")
    assert_v3_2_fold_manifest(lobo_loaded, kind="lobo", expected=lobo_expected)
    _fault(fault_inject, "verifying_ruaa_fold")
    assert_v3_2_fold_manifest(ruaa_loaded, kind="ruaa", expected=ruaa_expected)
    expected_candidate = _candidate_body(
        corpus=expected_corpus, lobo=lobo_expected, ruaa=ruaa_expected,
        isolation=trusted["isolation"], basename=trusted["basename"],
        parent_identity=trusted["parent_identity"], ruaa_selection=trusted["ruaa_selection"],
        inventory=inventory_body, files=files,
    )
    if candidate != expected_candidate:
        raise CorrectedCorpusError("candidate differs from trusted-input reconstruction")
    return {
        "bundle_root": root, "candidate_root": root, "corrected_corpus_root": child_root,
        "candidate": candidate, "corpus_manifest": expected_corpus,
        "lobo_manifest": lobo_expected, "ruaa_manifest": ruaa_expected,
        "isolation_audit": trusted["isolation"], "basename_audit": trusted["basename"],
        "reused": True,
    }


@contextlib.contextmanager
def _publish_lock(parent: pathlib.Path) -> Iterator[None]:
    path = parent / PUBLISH_LOCK_NAME
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
    except FileExistsError:
        fd = os.open(path, flags)
    try:
        if created:
            os.fchmod(fd, 0o644)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o644:
            raise CorrectedCorpusError("publish lock is symlinked, hardlinked, special, or mode-drifted")
        fcntl.flock(fd, fcntl.LOCK_EX)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CorrectedCorpusError("publish lock inode changed while acquiring lock")
        yield
    finally:
        os.close(fd)


def _rename_noreplace(source: pathlib.Path, destination: pathlib.Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise CorrectedCorpusError("atomic no-clobber renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _ensure_directory(path: pathlib.Path) -> None:
    try:
        _verify_real_dir_chain(path)
    except Exception as exc:
        raise CorrectedCorpusError(f"unsafe output path chain: {path}") from exc
    if path.exists():
        _directory(path, "output root")
        return
    path.mkdir(parents=True, mode=0o755)
    try:
        _verify_real_dir_chain(path)
    except Exception as exc:
        raise CorrectedCorpusError(f"unsafe output path chain after creation: {path}") from exc


def prepare_corrected_v3_2(*, historical_parent_root: pathlib.Path | str,
                           output_root: pathlib.Path | str,
                           ruaa_parent_selection: Iterable[str], config_hash: str,
                           protocol_sha256: str,
                           fault_inject: Callable[[str], None] | None = None) -> dict:
    """Build, fully verify, then atomically publish one unapproved preparation bundle."""
    if not _HEX64.fullmatch(config_hash) or not _HEX64.fullmatch(protocol_sha256):
        raise CorrectedCorpusError("config/protocol hashes must be SHA256")
    parent = pathlib.Path(historical_parent_root)
    output = pathlib.Path(output_root)
    selection = tuple(ruaa_parent_selection)
    _ensure_directory(output)
    bundle_parent = output / BUNDLE_PARENT_NAME
    if not bundle_parent.exists():
        _created_dir(bundle_parent)
    else:
        _directory(bundle_parent, "bundle parent")
        _verify_real_dir_chain(bundle_parent)
    existing = [pathlib.Path(entry.path) for entry in _entries(bundle_parent, "bundle parent")
                if not entry.name.startswith(".")]
    if existing:
        if len(existing) != 1:
            raise CorrectedCorpusError("bundle parent has multiple existing destinations")
        return verify_v3_2_candidate(
            existing[0], historical_parent_root=parent, ruaa_parent_selection=selection,
            config_hash=config_hash, protocol_sha256=protocol_sha256,
        )
    trusted = _trusted_derivation(parent, selection)
    stage_parent = pathlib.Path(tempfile.mkdtemp(prefix=".staging_v3_2_", dir=bundle_parent))
    os.chmod(stage_parent, 0o755)
    work_root = stage_parent / "bundle"
    published = False
    try:
        corpus_root = work_root / CORRECTED_CORPUS_DIR
        _created_dir(work_root)
        corpus = _assemble_child(
            parent, corpus_root, trusted["records"], parent_identity=trusted["parent_identity"],
            policy=trusted["policy"], catalog=trusted["catalog"], basename=trusted["basename"],
            isolation=trusted["isolation"], config_hash=config_hash, protocol_sha256=protocol_sha256,
            fault_inject=fault_inject,
        )
        _fault(fault_inject, "building_lobo_fold")
        lobo_manifest = _fold(
            "lobo", trusted["records"], corpus,
            selection_digest=_digest(
                "paired_audit.lobo_selection.v3_2", [row["work_id"] for row in trusted["records"]]
            ), config_hash=config_hash,
        )
        _fault(fault_inject, "building_ruaa_fold")
        ruaa_manifest = _fold(
            "ruaa", trusted["ruaa_records"], corpus,
            selection_digest=trusted["ruaa_selection"]["selection_digest"], config_hash=config_hash,
        )
        values = {
            "lobo_fold_manifest_v3_2.json": lobo_manifest,
            "ruaa_fold_manifest_v3_2.json": ruaa_manifest,
            "content_isolation_audit_v3_2.json": trusted["isolation"],
            "basename_collision_audit_v3_2.json": trusted["basename"],
        }
        _fault(fault_inject, "writing_audits")
        for name, value in values.items():
            _write_json(value, work_root / name)
        payload_rows = _recursive_inventory(work_root)
        inventory = {
            "schema": "paired_audit.preparation_bundle_inventory.v1",
            "storage_contract_version": BUNDLE_STORAGE_CONTRACT_VERSION,
            "root_mode": "0755", "entries": payload_rows,
        }
        inventory["self_hash"] = _self_hash(inventory)
        _write_json(inventory, work_root / BUNDLE_INVENTORY_NAME)
        files = _file_map(work_root)
        candidate = _candidate_body(
            corpus=corpus, lobo=lobo_manifest, ruaa=ruaa_manifest,
            isolation=trusted["isolation"], basename=trusted["basename"],
            parent_identity=trusted["parent_identity"], ruaa_selection=trusted["ruaa_selection"],
            inventory=inventory, files=files,
        )
        _fault(fault_inject, "writing_candidate")
        _write_json(candidate, work_root / "candidate.json")
        _fault(fault_inject, "writing_sha256sums")
        sums_path = work_root / SHA256SUMS_NAME
        sums_path.write_text(_canonical_sums(work_root), encoding="utf-8")
        _created_file_mode(sums_path)
        stage = work_root.rename(stage_parent / candidate["self_hash"])
        verified = verify_v3_2_candidate(
            stage, historical_parent_root=parent, ruaa_parent_selection=selection,
            config_hash=config_hash, protocol_sha256=protocol_sha256,
            fault_inject=fault_inject,
        )
        destination = bundle_parent / candidate["self_hash"]
        _fault(fault_inject, "before_final_rename")
        collision = False
        with _publish_lock(bundle_parent):
            if destination.exists() or destination.is_symlink():
                collision = True
            else:
                _fault(fault_inject, "final_rename")
                _rename_noreplace(stage, destination)
                published = True
        if collision:
            existing_result = verify_v3_2_candidate(
                destination, historical_parent_root=parent, ruaa_parent_selection=selection,
                config_hash=config_hash, protocol_sha256=protocol_sha256,
            )
            shutil.rmtree(stage)
            existing_result["reused"] = True
            return existing_result
        # No fallible scientific or filesystem verification is allowed after this rename.
        verified.update({
            "bundle_root": destination, "candidate_root": destination,
            "corrected_corpus_root": destination / CORRECTED_CORPUS_DIR,
            "reused": False,
        })
        return verified
    finally:
        if not published and stage_parent.exists():
            shutil.rmtree(stage_parent, ignore_errors=True)
        elif published and stage_parent.exists():
            try:
                stage_parent.rmdir()
            except OSError:
                pass


PARITY_CONTRACT_VERSION = "paired_audit.preparation_bundle_parity.v1"


def _parity_bundle_root(value: pathlib.Path | str) -> pathlib.Path:
    root = pathlib.Path(value)
    if (root / "candidate.json").exists() or (root / "candidate.json").is_symlink():
        return root
    parent = root / BUNDLE_PARENT_NAME
    _directory(parent, "parity bundle parent")
    candidates = [pathlib.Path(entry.path) for entry in _entries(parent, "parity bundle parent")
                  if not entry.name.startswith(".")]
    if len(candidates) != 1:
        raise CorrectedCorpusError("parity root must contain exactly one digest-named bundle")
    return candidates[0]


def tree_bytes_modes(root: pathlib.Path | str) -> tuple[dict[str, tuple], str]:
    """Canonical parity over one bundle root and every descendant path/type/size/mode/byte hash."""
    bundle = _parity_bundle_root(root)
    _require_mode(bundle, 0o755, "parity bundle root")
    inventory = _recursive_inventory(bundle)
    rows: dict[str, tuple] = {}
    for row in inventory:
        if row["type"] == "directory":
            rows[row["path"]] = ("directory", row["mode"])
        else:
            rows[row["path"]] = (row["sha256"], row["mode"], row["size"])
    domain = {
        "schema": PARITY_CONTRACT_VERSION,
        "root": "digest-named preparation bundle (root itself excluded)",
        "namespace": "bundle-relative POSIX paths; directories carry a trailing slash",
        "included_members": "every descendant directory and regular file, including candidate, exact inventory, and SHA256SUMS",
        "mode_policy": {"bundle_root": "0755", "directories": "0755", "regular_files": "0644"},
        "entries": rows,
    }
    return rows, _digest(PARITY_CONTRACT_VERSION, domain)


def assert_preparation_parity(first: pathlib.Path | str, second: pathlib.Path | str) -> dict:
    left, left_digest = tree_bytes_modes(first)
    right, right_digest = tree_bytes_modes(second)
    if left != right or left_digest != right_digest:
        raise CorrectedCorpusError("independent preparations differ in paths, bytes, or modes")
    return {
        "paths_types_sizes_modes_bytes_digest": left_digest,
        "paths_bytes_modes_digest": left_digest,
        "contract": PARITY_CONTRACT_VERSION,
        "root": "digest-named preparation bundle",
        "n_entries": len(left),
    }


__all__ = [
    "BUNDLE_STORAGE_CONTRACT_VERSION", "CorrectedCorpusError", "EXCLUDED_WORK_IDS",
    "HISTORICAL_PARENT_DIGEST", "IDENTITY_CONTRACT_VERSION", "PARITY_CONTRACT_VERSION",
    "PREPARATION_STATUS", "assert_preparation_parity", "assert_v3_2_fold_manifest", "applicability_matrix",
    "basename_collision_inventory", "content_isolation_audit", "full_work_id", "prepare_corrected_v3_2",
    "resolve_full_work", "tree_bytes_modes", "verify_historical_parent", "verify_v3_2_candidate",
    "work_identity_catalog",
]
