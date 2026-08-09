"""Preparation-only corrected corpus and fold contract for paired-audit v3.2.

Nothing here imports an evaluator or creates a RunPlan.  It reads the exact
registered v3.1 evidence root, derives a new content-addressed child with the
three adjudicated exclusions, runs a corpus-wide content isolation audit, and
writes unapproved v3.2 fold manifests to a caller-owned output root.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

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
    if path.is_symlink() or not path.is_file():
        raise CorrectedCorpusError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
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


def _regular(path: pathlib.Path, label: str) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CorrectedCorpusError(f"symlink or special file in {label}: {path}")


def _directory(path: pathlib.Path, label: str) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CorrectedCorpusError(f"symlink or non-directory in {label}: {path}")


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


def _scan_root(root: pathlib.Path, *, child: bool) -> tuple[dict, ...]:
    expected = {"frags", "input_clean", "corrected_corpus_manifest_v3_2.json" if child else "corpus_manifest.json"}
    if {entry.name for entry in _entries(root, "corpus root")} != expected:
        raise CorrectedCorpusError("corpus root has missing or extra top-level members")
    frag_authors = _entries(root / "frags", "frags")
    clean_authors = _entries(root / "input_clean", "input_clean")
    if [entry.name for entry in frag_authors] != [entry.name for entry in clean_authors]:
        raise CorrectedCorpusError("frags/input_clean author inventories differ")
    records = []
    for entry in frag_authors:
        author = entry.name
        if not _SAFE.fullmatch(author) or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise CorrectedCorpusError("unsafe author entry")
        expected_sources = set()
        for book in _entries(pathlib.Path(entry.path), f"author {author}"):
            slug = book.name
            if not _SAFE.fullmatch(slug) or book.is_symlink() or not book.is_dir(follow_symlinks=False):
                raise CorrectedCorpusError("unsafe work entry")
            files = _entries(pathlib.Path(book.path), f"work {author}/{slug}")
            if not files or files[0].name == "":
                raise CorrectedCorpusError("empty work directory")
            if {item.name for item in files} != {MANIFEST_NAME, *[item.name for item in files if item.name.endswith('.txt')]}:
                raise CorrectedCorpusError("work has an extra, missing, or unsafe member")
            if any(item.is_symlink() or not item.is_file(follow_symlinks=False) for item in files):
                raise CorrectedCorpusError("work has a symlink, directory, or special member")
            expected_sources.add(f"{slug}.txt")
            records.append(_work_record(root, author, slug))
        clean_entries = _entries(root / "input_clean" / author, f"input_clean {author}")
        if {entry.name for entry in clean_entries} != expected_sources or any(
            entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in clean_entries
        ):
            raise CorrectedCorpusError("input_clean has missing, extra, symlinked, or special members")
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
    if len(records) != 255 or manifest.get("n_works") != 255:
        raise CorrectedCorpusError("historical parent must contain exactly 255 works")
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)
    if _file_hash(source) != _file_hash(destination):
        raise CorrectedCorpusError("byte copy mismatch")


def _inventory(root: pathlib.Path) -> tuple[list[dict], str]:
    rows = []
    for subtree in ("frags", "input_clean"):
        for current, dirs, files in os.walk(root / subtree, followlinks=False):
            directory = pathlib.Path(current)
            if directory.is_symlink():
                raise CorrectedCorpusError("symlink in generated child")
            os.chmod(directory, 0o755)
            for name in dirs:
                path = directory / name
                if path.is_symlink():
                    raise CorrectedCorpusError("symlink in generated child")
                os.chmod(path, 0o755)
            for name in files:
                path = directory / name
                _regular(path, "generated child")
                os.chmod(path, 0o644)
                rows.append({"path": path.relative_to(root).as_posix(), "sha256": _file_hash(path), "mode": _mode(path)})
    rows.sort(key=lambda row: row["path"])
    return rows, _digest("paired_audit.content_inventory.v3_2", rows)


def _write_child(parent: pathlib.Path, output: pathlib.Path, records: Sequence[Mapping], *, parent_identity: Mapping,
                 policy: Mapping, catalog: Mapping, basename: Mapping, isolation: Mapping, config_hash: str,
                 protocol_sha256: str) -> tuple[pathlib.Path, dict]:
    target_parent = output / "corrected_audit_corpus_v3_2"
    target_parent.mkdir(parents=True, exist_ok=True)
    _verify_real_dir_chain(target_parent)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=".staging_v3_2_", dir=target_parent))
    try:
        (stage / "frags").mkdir()
        (stage / "input_clean").mkdir()
        for row in records:
            source = parent / "frags" / row["author_id"] / row["work_slug"]
            destination = stage / "frags" / row["author_id"] / row["work_slug"]
            destination.mkdir(parents=True)
            _copy(source / MANIFEST_NAME, destination / MANIFEST_NAME)
            for chunk in row["chunks"]:
                _copy(source / chunk["path"], destination / chunk["path"])
            _copy(parent / "input_clean" / row["author_id"] / f"{row['work_slug']}.txt",
                  stage / "input_clean" / row["author_id"] / f"{row['work_slug']}.txt")
        inventory, digest = _inventory(stage)
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
        dump_strict(body, stage / "corrected_corpus_manifest_v3_2.json", trailing_newline=True)
        os.chmod(stage / "corrected_corpus_manifest_v3_2.json", 0o644)
        destination = target_parent / digest
        if destination.exists() or destination.is_symlink():
            _verify_child(destination, body)
            shutil.rmtree(stage)
        else:
            os.replace(stage, destination)
        stage = None
        _verify_child(destination, body)
        return destination, body
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _verify_child(root: pathlib.Path, expected: Mapping) -> None:
    if {entry.name for entry in _entries(root, "corrected child")} != {"frags", "input_clean", "corrected_corpus_manifest_v3_2.json"}:
        raise CorrectedCorpusError("partial/extraneous corrected child")
    manifest = load_strict(root / "corrected_corpus_manifest_v3_2.json")
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise CorrectedCorpusError("historical/v3.1 corpus schema rejected identity-first")
    body = dict(manifest)
    self_hash = body.pop("self_hash", None)
    if self_hash != _self_hash(body) or manifest != expected:
        raise CorrectedCorpusError("corrected corpus manifest tamper/conflict")
    inventory, digest = _inventory(root)
    if root.name != digest or inventory != manifest["corrected_content_inventory"]:
        raise CorrectedCorpusError("corrected corpus bytes/modes inventory drift")


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


def assert_v3_2_fold_manifest(manifest: Mapping, *, kind: str) -> None:
    expected_schema = LOBO_SCHEMA if kind == "lobo" else RUAA_SCHEMA if kind == "ruaa" else None
    if expected_schema is None or not isinstance(manifest, Mapping) or manifest.get("schema") != expected_schema:
        raise CorrectedCorpusError("historical/v3.1 fold schema rejected identity-first")
    body = dict(manifest)
    self_hash = body.pop("self_hash", None)
    if manifest.get("protocol_version") != PROTOCOL_VERSION or self_hash != _self_hash(body):
        raise CorrectedCorpusError("v3.2 fold protocol/self-hash mismatch")
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
    expected_counts = LOBO_COUNTS if kind == "lobo" else (22, 134, 22, 134)
    if counts != expected_counts:
        raise CorrectedCorpusError("fold count universe mismatch")
    if manifest["probability_class_order"] != authors or manifest["metric_label_order"] != sorted({row["author_id"] for row in tested}):
        raise CorrectedCorpusError("fold label order mismatch")


def _write_candidate(output: pathlib.Path, *, corpus_root: pathlib.Path, corpus: Mapping, lobo: Mapping, ruaa: Mapping,
                     isolation: Mapping, basename: Mapping, parent_identity: Mapping, ruaa_selection: Mapping) -> pathlib.Path:
    parent = output / "paired_audit_v3_2_preparation"
    parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=".staging_v3_2_", dir=parent))
    try:
        values = {
            "lobo_fold_manifest_v3_2.json": lobo, "ruaa_fold_manifest_v3_2.json": ruaa,
            "content_isolation_audit_v3_2.json": isolation, "basename_collision_audit_v3_2.json": basename,
        }
        for name, value in values.items():
            dump_strict(value, stage / name, trailing_newline=True)
            os.chmod(stage / name, 0o644)
        body = {
            "schema": CANDIDATE_SCHEMA, "protocol_version": PROTOCOL_VERSION, "status": PREPARATION_STATUS,
            "review_state": "independent_manifest_review_required", "freeze_status": "unapproved_no_freeze_root_pin",
            "production_evaluator_status": "unregistered", "confirmatory_execution_status": "hard_disabled",
            "headline_status": "not_authorized", "publication_status": "not_authorized",
            "historical_parent": dict(parent_identity), "corrected_corpus": {
                "relative_root": str(corpus_root.relative_to(output)), "digest": corpus_root.name, "self_hash": corpus["self_hash"],
            }, "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "full_work_identity_catalog_digest": corpus["full_work_identity_catalog_digest"],
            "basename_collision_audit_digest": basename["digest"], "content_isolation_audit_digest": isolation["digest"],
            "ruaa_selection": dict(ruaa_selection), "applicability": applicability_matrix(),
            "folds": {"lobo_self_hash": lobo["self_hash"], "ruaa_self_hash": ruaa["self_hash"]},
        }
        body["files"] = {name: {"sha256": _file_hash(stage / name), "mode": _mode(stage / name)} for name in sorted(values)}
        body["self_hash"] = _self_hash(body)
        dump_strict(body, stage / "candidate.json", trailing_newline=True)
        os.chmod(stage / "candidate.json", 0o644)
        payloads = sorted([stage / "candidate.json", *[stage / name for name in values]], key=lambda path: path.name)
        (stage / "SHA256SUMS").write_text("".join(f"{_file_hash(path)}  {path.name}\n" for path in payloads), encoding="utf-8")
        os.chmod(stage / "SHA256SUMS", 0o644)
        destination = parent / body["self_hash"]
        if destination.exists():
            if any(_file_hash(destination / path.name) != _file_hash(path) or _mode(destination / path.name) != _mode(path) for path in [*payloads, stage / "SHA256SUMS"]):
                raise CorrectedCorpusError("candidate resume conflict")
            shutil.rmtree(stage)
        else:
            os.replace(stage, destination)
        stage = None
        return destination
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def prepare_corrected_v3_2(*, historical_parent_root: pathlib.Path | str, output_root: pathlib.Path | str,
                           ruaa_parent_selection: Iterable[str], config_hash: str, protocol_sha256: str) -> dict:
    """Create/revalidate one deterministic unapproved candidate.  It never writes ``data/``."""
    if not _HEX64.fullmatch(config_hash) or not _HEX64.fullmatch(protocol_sha256):
        raise CorrectedCorpusError("config/protocol hashes must be SHA256")
    parent = pathlib.Path(historical_parent_root)
    output = pathlib.Path(output_root)
    parent_identity, all_records = verify_historical_parent(parent)
    policy = exclusion_policy()
    records = tuple(row for row in all_records if row["work_id"] not in EXCLUDED_WORK_IDS)
    if len(records) != 252 or set(all_records[i]["work_id"] for i in range(len(all_records))) & set(EXCLUDED_WORK_IDS) != set(EXCLUDED_WORK_IDS):
        raise CorrectedCorpusError("the exact three registered exclusions cannot derive the corrected child")
    ruaa, ruaa_selection = _ruaa_selection(ruaa_parent_selection, {row["work_id"] for row in all_records})
    lobo = _lobo_tested(records)
    _assert_counts(records, lobo, ruaa)
    catalog = work_identity_catalog(records)
    basename_lobo = basename_collision_inventory(records, expected=EXPECTED_LOBO_BASENAME_COLLISIONS)
    ruaa_records = tuple(row for row in records if row["work_id"] in set(ruaa))
    basename_ruaa = basename_collision_inventory(ruaa_records, expected=())
    basename = {"schema": "paired_audit.basename_collision_audit.v3_2", "lobo": basename_lobo, "ruaa": basename_ruaa}
    basename["digest"] = _digest(basename["schema"], basename)
    isolation = content_isolation_audit(records, lobo_tested_work_ids=lobo, ruaa_work_ids=ruaa)
    corpus_root, corpus = _write_child(parent, output, records, parent_identity=parent_identity, policy=policy,
                                       catalog=catalog, basename=basename, isolation=isolation, config_hash=config_hash,
                                       protocol_sha256=protocol_sha256)
    _verify_child(corpus_root, corpus)
    lobo_manifest = _fold("lobo", records, corpus, selection_digest=_digest("paired_audit.lobo_selection.v3_2", [row["work_id"] for row in records]), config_hash=config_hash)
    ruaa_manifest = _fold("ruaa", ruaa_records, corpus, selection_digest=ruaa_selection["selection_digest"], config_hash=config_hash)
    assert_v3_2_fold_manifest(lobo_manifest, kind="lobo")
    assert_v3_2_fold_manifest(ruaa_manifest, kind="ruaa")
    candidate = _write_candidate(output, corpus_root=corpus_root, corpus=corpus, lobo=lobo_manifest, ruaa=ruaa_manifest,
                                 isolation=isolation, basename=basename, parent_identity=parent_identity,
                                 ruaa_selection=ruaa_selection)
    return {"corrected_corpus_root": corpus_root, "candidate_root": candidate, "corpus_manifest": corpus,
            "lobo_manifest": lobo_manifest, "ruaa_manifest": ruaa_manifest, "isolation_audit": isolation,
            "basename_audit": basename}


def tree_bytes_modes(root: pathlib.Path | str) -> tuple[dict[str, tuple[str, str]], str]:
    root = pathlib.Path(root)
    rows: dict[str, tuple[str, str]] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        directory = pathlib.Path(current)
        if directory.is_symlink():
            raise CorrectedCorpusError("symlink in parity tree")
        for name in dirs:
            path = directory / name
            _directory(path, "parity tree")
            rows[path.relative_to(root).as_posix() + "/"] = ("directory", _mode(path))
        for name in files:
            path = directory / name
            _regular(path, "parity tree")
            rows[path.relative_to(root).as_posix()] = (_file_hash(path), _mode(path))
    return rows, _digest("paired_audit.preparation_parity.v3_2", rows)


def assert_preparation_parity(first: pathlib.Path | str, second: pathlib.Path | str) -> dict:
    left, left_digest = tree_bytes_modes(first)
    right, right_digest = tree_bytes_modes(second)
    if left != right or left_digest != right_digest:
        raise CorrectedCorpusError("independent preparations differ in paths, bytes, or modes")
    return {"paths_bytes_modes_digest": left_digest, "n_entries": len(left)}


__all__ = [
    "CorrectedCorpusError", "EXCLUDED_WORK_IDS", "HISTORICAL_PARENT_DIGEST", "IDENTITY_CONTRACT_VERSION",
    "PREPARATION_STATUS", "assert_preparation_parity", "assert_v3_2_fold_manifest", "applicability_matrix",
    "basename_collision_inventory", "content_isolation_audit", "full_work_id", "prepare_corrected_v3_2",
    "resolve_full_work", "tree_bytes_modes", "verify_historical_parent", "work_identity_catalog",
]
