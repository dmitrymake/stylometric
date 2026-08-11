"""Preparation-only corrected corpus and fold contract for paired-audit v3.2.

Nothing here imports an evaluator or creates a RunPlan.  It reads the exact
registered v3.1 evidence root, derives a new content-addressed child with the
three adjudicated exclusions, runs a corpus-wide content isolation audit, and
writes unapproved v3.2 fold manifests to a caller-owned output root.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import pathlib
import re
import secrets
import shutil
import stat
import unicodedata
from collections import Counter, defaultdict
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from ...jsonio import dump_strict, dumps_strict, loads_strict
from ...workdoc import MANIFEST_NAME, WorkManifest, canonical_chunk_text, sha256_text
from ...domain.corpus_identity import find_cross_work_content_overlaps


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
_HISTORICAL_CORPUS_DIGEST_VERSION = "paired_audit.corpus.v1"


class CorrectedCorpusError(RuntimeError):
    """A v3.2 preparation identity, content, or output contract failed closed."""


def _without_symlink_components(path: pathlib.Path | str) -> pathlib.Path:
    """Return an absolute local path only when no existing component is a symlink."""
    target = pathlib.Path(path).absolute()
    current = pathlib.Path(target.anchor)
    for component in target.parts[1:]:
        current /= component
        if current.is_symlink():
            raise CorrectedCorpusError(f"symlink path component is not allowed: {current}")
    return target


@contextlib.contextmanager
def _open_or_create_directory_path(
    path: pathlib.Path | str, *, leaf_mode: int = 0o755
) -> Iterator[pathlib.Path]:
    """Create/open one trusted local directory for a cooperative single writer."""
    target = _without_symlink_components(path)
    if target == pathlib.Path(target.anchor):
        raise CorrectedCorpusError("output root cannot be the filesystem root")
    try:
        target.mkdir(mode=leaf_mode, parents=True, exist_ok=True)
    except OSError as exc:
        raise CorrectedCorpusError(f"unavailable output directory: {target}") from exc
    _without_symlink_components(target)
    if not target.is_dir():
        raise CorrectedCorpusError(f"output path is not a real directory: {target}")
    yield target


@contextlib.contextmanager
def _open_directory_path(path: pathlib.Path | str) -> Iterator[pathlib.Path]:
    """Validate and expose one ordinary local directory path."""
    target = _without_symlink_components(path)
    try:
        is_directory = target.is_dir()
    except OSError as exc:
        raise CorrectedCorpusError(f"unavailable directory path: {target}") from exc
    if not is_directory:
        raise CorrectedCorpusError(f"path is not a real directory: {target}")
    yield target


def _read_file_from_dir(
    parent_fd: pathlib.Path, name: str, *, label: str, expected: os.stat_result | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    target = parent_fd / name
    try:
        before = target.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise CorrectedCorpusError(f"symlink or special file in {label}: {name}")
        if expected is not None and (
            expected.st_size != before.st_size
            or stat.S_IMODE(expected.st_mode) != stat.S_IMODE(before.st_mode)
        ):
            raise CorrectedCorpusError(f"file differs from validated shape in {label}: {name}")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            raise CorrectedCorpusError(
                f"mode drift in {label}: {name} has {stat.S_IMODE(before.st_mode):04o}, "
                f"expected {expected_mode:04o}"
            )
        payload = target.read_bytes()
        if len(payload) != before.st_size:
            raise CorrectedCorpusError(f"file size changed while reading in {label}: {name}")
        return payload, before
    except CorrectedCorpusError:
        raise
    except OSError as exc:
        raise CorrectedCorpusError(f"missing or unsafe file in {label}: {name}") from exc


def _open_relative_parent(root_fd: pathlib.Path, parts: Sequence[str], *, label: str):
    if not parts or any(not _SAFE.fullmatch(part) or part in (".", "..") for part in parts):
        raise CorrectedCorpusError(f"unsafe relative file path in {label}")
    current = root_fd.joinpath(*parts[:-1])
    if current.is_symlink() or not current.is_dir():
        raise CorrectedCorpusError(f"non-directory component in {label}")
    yield current, parts[-1]


_open_relative_parent = contextlib.contextmanager(_open_relative_parent)


def _read_relative(
    root_fd: pathlib.Path, relative: str, *, label: str, expected: os.stat_result | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    parts = tuple(pathlib.PurePosixPath(relative).parts)
    with _open_relative_parent(root_fd, parts, label=label) as (parent_fd, name):
        return _read_file_from_dir(
            parent_fd, name, label=label, expected=expected, expected_mode=expected_mode,
        )


def _read_stable_path(path: pathlib.Path | str, *, expected: os.stat_result | None = None,
                      expected_mode: int | None = None) -> bytes:
    target = pathlib.Path(path)
    with _open_directory_path(target.parent) as parent_fd:
        payload, _ = _read_file_from_dir(
            parent_fd, target.name, label="stable path read", expected=expected,
            expected_mode=expected_mode,
        )
        return payload


def read_stable_bytes(path: pathlib.Path | str, *, expected_mode: int | None = None) -> bytes:
    """Read one regular local file under the cooperative filesystem contract."""
    return _read_stable_path(path, expected_mode=expected_mode)


def _parse_json_bytes(payload: bytes, *, label: str, canonical: bool = False):
    try:
        text = payload.decode("utf-8")
        value = loads_strict(text)
    except Exception as exc:
        raise CorrectedCorpusError(f"invalid strict UTF-8 JSON in {label}") from exc
    if canonical and payload != (dumps_strict(value, indent=2) + "\n").encode("utf-8"):
        raise CorrectedCorpusError(f"non-canonical JSON encoding in {label}")
    return value


def load_stable_json(path: pathlib.Path | str, *, canonical: bool = False):
    return _parse_json_bytes(read_stable_bytes(path), label=str(path), canonical=canonical)


@dataclasses.dataclass(frozen=True)
class _FileSnapshot:
    info: os.stat_result
    sha256: str
    payload: bytes | None


@dataclasses.dataclass(frozen=True)
class _TreeSnapshot:
    root_info: os.stat_result
    directories: Mapping[str, os.stat_result]
    files: Mapping[str, _FileSnapshot]


def _snapshot_tree_fd(root_fd: pathlib.Path, *, label: str) -> _TreeSnapshot:
    directories: dict[str, os.stat_result] = {"": root_fd.lstat()}
    files: dict[str, _FileSnapshot] = {}

    def visit(directory: pathlib.Path, prefix: str) -> None:
        names = sorted(path.name for path in directory.iterdir())
        for name in names:
            if not _SAFE.fullmatch(name) or name in (".", ".."):
                raise CorrectedCorpusError(f"unsafe entry name in {label}: {name!r}")
            relative = f"{prefix}/{name}" if prefix else name
            target = directory / name
            before = target.lstat()
            if stat.S_ISDIR(before.st_mode):
                directories[relative] = before
                visit(target, relative)
            elif stat.S_ISREG(before.st_mode):
                payload, captured = _read_file_from_dir(
                    directory, name, label=label, expected=before,
                )
                cache = payload if name.endswith(".json") or name == SHA256SUMS_NAME else None
                files[relative] = _FileSnapshot(
                    captured, hashlib.sha256(payload).hexdigest(), cache,
                )
            else:
                raise CorrectedCorpusError(f"symlink or special entry in {label}: {relative}")

    if not stat.S_ISDIR(directories[""].st_mode):
        raise CorrectedCorpusError(f"root is not a directory in {label}")
    visit(root_fd, "")
    return _TreeSnapshot(directories[""], directories, files)


def _snapshot_path(root: pathlib.Path, *, label: str) -> _TreeSnapshot:
    with _open_directory_path(root) as root_fd:
        return _snapshot_tree_fd(root_fd, label=label)


def _digest(namespace: str, value: object) -> str:
    data = dumps_strict(value, sort_keys=True).encode("utf-8")
    encoded = namespace.encode("utf-8")
    return hashlib.sha256(len(encoded).to_bytes(8, "big") + encoded + data).hexdigest()


def _self_hash(value: Mapping) -> str:
    return hashlib.sha256(dumps_strict(dict(value), sort_keys=True).encode("utf-8")).hexdigest()


def _file_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(read_stable_bytes(path)).hexdigest()


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


def _created_dir(path: pathlib.Path, *, parents: bool = False) -> None:
    """Create a staging/output directory and set its creation-time canonical mode."""
    path.mkdir(mode=0o755, parents=parents, exist_ok=False)
    os.chmod(path, 0o755)


def _created_file_mode(path: pathlib.Path) -> None:
    """Set mode only for a file just created by this preparation."""
    os.chmod(path, 0o644)


def _immediate_children(values: Iterable[str], parent: str) -> set[str]:
    prefix = f"{parent}/" if parent else ""
    return {
        value[len(prefix):] for value in values
        if value != parent and value.startswith(prefix)
        and value[len(prefix):] and "/" not in value[len(prefix):]
    }


def _snapshot_payload(snapshot: _TreeSnapshot, relative: str, *, label: str) -> bytes:
    try:
        payload = snapshot.files[relative].payload
    except KeyError as exc:
        raise CorrectedCorpusError(f"missing file in {label}: {relative}") from exc
    if payload is None:
        raise CorrectedCorpusError(f"internal stable-read cache missing in {label}: {relative}")
    return payload


def _work_record(root_fd: int, snapshot: _TreeSnapshot, author: str, slug: str,
                 *, prefix: str = "") -> dict:
    work_id = full_work_id(f"{author}/{slug}")
    base = f"{prefix}/" if prefix else ""
    work_root = f"{base}frags/{author}/{slug}"
    manifest_path = f"{work_root}/{MANIFEST_NAME}"
    source_path = f"{base}input_clean/{author}/{slug}.txt"
    try:
        manifest_bytes = _snapshot_payload(snapshot, manifest_path, label=f"manifest {work_id}")
        manifest = WorkManifest.from_dict(
            _parse_json_bytes(manifest_bytes, label=f"manifest {work_id}")
        )
    except Exception as exc:
        raise CorrectedCorpusError(f"invalid work manifest for {work_id}: {exc}") from exc
    if manifest.work_id != work_id or manifest.author_id != author:
        raise CorrectedCorpusError(f"manifest author/work identity mismatch for {work_id}")
    if manifest.overlap != 0.0:
        raise CorrectedCorpusError(f"manifest overlap must be zero for {work_id}")
    if [entry.span_ordinal for entry in manifest.chunks] != list(range(len(manifest.chunks))):
        raise CorrectedCorpusError(f"manifest span ordinals are not contiguous for {work_id}")
    listed = [entry.path for entry in manifest.chunks]
    if len(listed) != len(set(listed)):
        raise CorrectedCorpusError(f"duplicate chunk path in manifest for {work_id}")

    source_bytes, _ = _read_relative(
        root_fd, source_path, label=f"source {work_id}",
        expected=snapshot.files[source_path].info,
        expected_mode=0o644 if prefix else None,
    )
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorrectedCorpusError(f"source is not valid UTF-8 for {work_id}") from exc
    if sha256_text(source_text.strip()) != manifest.provenance_sha256:
        raise CorrectedCorpusError(f"manifest provenance mismatch for {work_id}")

    chunks = []
    for entry in manifest.chunks:
        relative = f"{work_root}/{entry.path}"
        chunk_bytes, _ = _read_relative(
            root_fd, relative, label=f"chunk {work_id}",
            expected=snapshot.files[relative].info,
            expected_mode=0o644 if prefix else None,
        )
        try:
            text = canonical_chunk_text(chunk_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise CorrectedCorpusError(f"chunk is not valid UTF-8 for {work_id}") from exc
        if not text or sha256_text(text) != entry.text_sha256:
            raise CorrectedCorpusError(f"chunk text hash mismatch for {work_id}/{entry.path}")
        chunks.append({
            "path": entry.path,
            "span_ordinal": entry.span_ordinal,
            "byte_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
            "text_sha256": entry.text_sha256,
            "normalized_sha256": _normalised_hash(text),
            "text": text,
        })
    identity = {
        "schema": IDENTITY_CONTRACT_VERSION,
        "work_id": work_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
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


def _validate_corpus_structure(snapshot: _TreeSnapshot, *, child: bool,
                               prefix: str = "") -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate the complete captured path/type shape before parsing."""
    base = f"{prefix}/" if prefix else ""
    manifest_name = CORRECTED_MANIFEST_NAME if child else "corpus_manifest.json"
    expected = {f"{base}frags", f"{base}input_clean"}
    expected_file = f"{base}{manifest_name}"
    if (_immediate_children(snapshot.directories, prefix) != {"frags", "input_clean"}
            or _immediate_children(snapshot.files, prefix) != {manifest_name}):
        raise CorrectedCorpusError("corpus root has missing or extra top-level members")
    if child:
        root_info = snapshot.directories[prefix]
        if any(stat.S_IMODE(snapshot.directories[path].st_mode) != 0o755 for path in expected):
            raise CorrectedCorpusError("mode drift in corrected corpus directory")
        if stat.S_IMODE(root_info.st_mode) != 0o755:
            raise CorrectedCorpusError("mode drift in corrected corpus root")
        if stat.S_IMODE(snapshot.files[expected_file].info.st_mode) != 0o644:
            raise CorrectedCorpusError("mode drift in corrected corpus manifest")

    frag_parent = f"{base}frags"
    clean_parent = f"{base}input_clean"
    frag_authors = sorted(_immediate_children(snapshot.directories, frag_parent))
    clean_authors = sorted(_immediate_children(snapshot.directories, clean_parent))
    if frag_authors != clean_authors:
        raise CorrectedCorpusError("frags/input_clean author inventories differ")
    works: list[tuple[str, tuple[str, ...]]] = []
    for author in frag_authors:
        if not _SAFE.fullmatch(author) or author in (".", ".."):
            raise CorrectedCorpusError("unsafe author entry")
        frag_author = f"{frag_parent}/{author}"
        clean_author = f"{clean_parent}/{author}"
        if child:
            if any(stat.S_IMODE(snapshot.directories[path].st_mode) != 0o755
                   for path in (frag_author, clean_author)):
                raise CorrectedCorpusError(f"mode drift in author directory {author}")
        if _immediate_children(snapshot.files, frag_author):
            raise CorrectedCorpusError("frags author directory contains files")
        if _immediate_children(snapshot.directories, clean_author):
            raise CorrectedCorpusError("input_clean author directory contains directories")
        slugs = sorted(_immediate_children(snapshot.directories, frag_author))
        expected_sources = {f"{slug}.txt" for slug in slugs}
        if _immediate_children(snapshot.files, clean_author) != expected_sources:
            raise CorrectedCorpusError("input_clean has missing, extra, symlinked, or special members")
        for slug in slugs:
            if not _SAFE.fullmatch(slug) or slug in (".", ".."):
                raise CorrectedCorpusError("unsafe work entry")
            work_dir = f"{frag_author}/{slug}"
            if child:
                if stat.S_IMODE(snapshot.directories[work_dir].st_mode) != 0o755:
                    raise CorrectedCorpusError(f"mode drift in work {author}/{slug}")
            if _immediate_children(snapshot.directories, work_dir):
                raise CorrectedCorpusError("work has a nested directory")
            names = _immediate_children(snapshot.files, work_dir)
            if MANIFEST_NAME not in names or len(names) < 2 or any(
                name != MANIFEST_NAME and (not _SAFE.fullmatch(name) or not name.endswith(".txt"))
                for name in names
            ):
                raise CorrectedCorpusError("work has an extra, missing, or unsafe member")
            if child:
                if any(stat.S_IMODE(snapshot.files[f"{work_dir}/{name}"].info.st_mode) != 0o644
                       for name in names):
                    raise CorrectedCorpusError(f"mode drift in work {author}/{slug}")
        if child and any(
            stat.S_IMODE(snapshot.files[f"{clean_author}/{name}"].info.st_mode) != 0o644
            for name in expected_sources
        ):
            raise CorrectedCorpusError(f"mode drift in input_clean {author}")
        works.append((author, tuple(slugs)))
    return tuple(works)


def _scan_root(root_fd: int, snapshot: _TreeSnapshot, *, child: bool,
               prefix: str = "") -> tuple[dict, ...]:
    works = _validate_corpus_structure(snapshot, child=child, prefix=prefix)
    records = []
    for author, slugs in works:
        for slug in slugs:
            records.append(_work_record(root_fd, snapshot, author, slug, prefix=prefix))
    records.sort(key=lambda row: row["work_id"])
    if len({row["work_id"] for row in records}) != len(records):
        raise CorrectedCorpusError("duplicate full work_id is fatal")
    return tuple(records)


def _historical_tree_digest(snapshot: _TreeSnapshot) -> str:
    digest = hashlib.sha256()
    namespace = _HISTORICAL_CORPUS_DIGEST_VERSION.encode("utf-8")
    digest.update(len(namespace).to_bytes(8, "big") + namespace)
    for subtree in ("frags", "input_clean"):
        prefix = subtree + "/"
        for relative in sorted(path for path in snapshot.files if path.startswith(prefix)):
            encoded = relative.encode("utf-8")
            # Historical v3.1 deliberately retains its original character-count framing.
            digest.update(len(relative).to_bytes(8, "big") + encoded)
            digest.update(bytes.fromhex(snapshot.files[relative].sha256))
    return digest.hexdigest()


def _verify_historical_parent_with_snapshot(
    root: pathlib.Path,
) -> tuple[dict, tuple[dict, ...], _TreeSnapshot]:
    root = pathlib.Path(root)
    if root.name != HISTORICAL_PARENT_DIGEST:
        raise CorrectedCorpusError("historical parent digest identity-first rejection")
    with _open_directory_path(root) as root_fd:
        snapshot = _snapshot_tree_fd(root_fd, label="historical corpus")
        records = _scan_root(root_fd, snapshot, child=False)
    manifest = _parse_json_bytes(
        _snapshot_payload(snapshot, "corpus_manifest.json", label="historical manifest"),
        label="historical manifest",
    )
    if not isinstance(manifest, dict):
        raise CorrectedCorpusError("historical parent manifest must be an object")
    manifest_body = dict(manifest)
    if manifest_body.pop("self_hash", None) != _self_hash(manifest_body):
        raise CorrectedCorpusError("historical parent manifest self-hash mismatch")
    if _historical_tree_digest(snapshot) != HISTORICAL_PARENT_DIGEST:
        raise CorrectedCorpusError("historical parent content digest mismatch")
    if manifest.get("schema") != HISTORICAL_PARENT_SCHEMA or manifest.get("audit_corpus_digest") != HISTORICAL_PARENT_DIGEST:
        raise CorrectedCorpusError("historical parent schema/digest mismatch")
    if len(records) != HISTORICAL_WORK_COUNT or manifest.get("n_works") != HISTORICAL_WORK_COUNT:
        raise CorrectedCorpusError(
            f"historical parent must contain exactly {HISTORICAL_WORK_COUNT} works"
        )
    catalog = work_identity_catalog(records)
    identity = {
        "historical_parent_digest": HISTORICAL_PARENT_DIGEST,
        "historical_parent_manifest_self_hash": manifest.get("self_hash"),
        "historical_parent_manifest_sha256": snapshot.files["corpus_manifest.json"].sha256,
        "full_work_identity_catalog_digest": catalog["digest"],
    }
    return identity, records, snapshot


def verify_historical_parent(root: pathlib.Path | str) -> tuple[dict, tuple[dict, ...]]:
    identity, records, _ = _verify_historical_parent_with_snapshot(pathlib.Path(root))
    return identity, records


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


def _write_new_file(destination: pathlib.Path, payload: bytes, *, mode: int = 0o644) -> None:
    if destination.exists() or destination.is_symlink():
        raise CorrectedCorpusError(f"staging file already exists: {destination}")
    try:
        destination.write_bytes(payload)
        os.chmod(destination, mode)
    except OSError as exc:
        raise CorrectedCorpusError(f"cannot create staging file: {destination}") from exc
    info = destination.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != len(payload):
        raise CorrectedCorpusError("staging destination is not the expected regular file")


def _copy(source: pathlib.Path, destination: pathlib.Path, *,
          expected: _FileSnapshot | None = None) -> None:
    payload = _read_stable_path(source, expected=expected.info if expected is not None else None)
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected.sha256:
        raise CorrectedCorpusError("historical parent copy source changed since verification")
    _write_new_file(destination, payload)
    if hashlib.sha256(payload).hexdigest() != _file_hash(destination):
        raise CorrectedCorpusError("byte copy mismatch")


def _inventory(root: pathlib.Path, *, snapshot: _TreeSnapshot | None = None,
               prefix: str = "") -> tuple[list[dict], str]:
    """Pure read-only child-content inventory (the historical digest domain)."""
    snapshot = snapshot or _snapshot_path(root, label="child inventory")
    base = f"{prefix}/" if prefix else ""
    rows = []
    for relative, record in snapshot.files.items():
        if relative.startswith(f"{base}frags/") or relative.startswith(f"{base}input_clean/"):
            local = relative[len(base):]
            rows.append({
                "path": local, "sha256": record.sha256,
                "mode": f"{stat.S_IMODE(record.info.st_mode):04o}",
            })
    rows.sort(key=lambda row: row["path"])
    return rows, _digest("paired_audit.content_inventory.v3_2", rows)


def _recursive_inventory(root: pathlib.Path, *, excluded: Iterable[str] = (),
                         snapshot: _TreeSnapshot | None = None,
                         prefix: str = "") -> list[dict]:
    """Pure exact recursive inventory of regular files and directories."""
    excluded_set = set(excluded)
    snapshot = snapshot or _snapshot_path(root, label="bundle inventory")
    base = f"{prefix}/" if prefix else ""
    rows: list[dict] = []
    for relative, info in snapshot.directories.items():
        if relative == prefix:
            continue
        if prefix and not relative.startswith(base):
            continue
        local = relative[len(base):]
        rows.append({
            "path": local + "/", "type": "directory",
            "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        })
    for relative, record in snapshot.files.items():
        if prefix and not relative.startswith(base):
            continue
        local = relative[len(base):]
        if local not in excluded_set:
            rows.append({
                "path": local, "type": "file",
                "mode": f"{stat.S_IMODE(record.info.st_mode):04o}",
                "size": record.info.st_size, "sha256": record.sha256,
            })
    rows.sort(key=lambda row: row["path"])
    return rows


def _canonical_sums(root: pathlib.Path, *, snapshot: _TreeSnapshot | None = None,
                    prefix: str = "") -> str:
    rows = _recursive_inventory(
        root, excluded=(SHA256SUMS_NAME,), snapshot=snapshot, prefix=prefix,
    )
    files = [row for row in rows if row["type"] == "file"]
    return "".join(f"{row['sha256']}  {row['path']}\n" for row in files)


def _write_json(value: Mapping, path: pathlib.Path) -> None:
    dump_strict(value, path, trailing_newline=True)
    _created_file_mode(path)


def _corpus_manifest(*, root: pathlib.Path, records: Sequence[Mapping], parent_identity: Mapping,
                     policy: Mapping, catalog: Mapping, basename: Mapping, isolation: Mapping,
                     config_hash: str, protocol_sha256: str,
                     snapshot: _TreeSnapshot | None = None, prefix: str = "") -> dict:
    inventory, digest = _inventory(root, snapshot=snapshot, prefix=prefix)
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
                    parent_snapshot: _TreeSnapshot,
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
        manifest_relative = f"frags/{author}/{row['work_slug']}/{MANIFEST_NAME}"
        _copy(
            source / MANIFEST_NAME, destination / MANIFEST_NAME,
            expected=parent_snapshot.files[manifest_relative],
        )
        for chunk in row["chunks"]:
            relative = f"frags/{author}/{row['work_slug']}/{chunk['path']}"
            _copy(
                source / chunk["path"], destination / chunk["path"],
                expected=parent_snapshot.files[relative],
            )
        source_relative = f"input_clean/{author}/{row['work_slug']}.txt"
        _copy(
            parent / source_relative, root / source_relative,
            expected=parent_snapshot.files[source_relative],
        )
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


def _verify_child_snapshot(root_fd: int, snapshot: _TreeSnapshot, expected: Mapping,
                           *, prefix: str) -> tuple[dict, ...]:
    records = _scan_root(root_fd, snapshot, child=True, prefix=prefix)
    manifest_relative = f"{prefix}/{CORRECTED_MANIFEST_NAME}" if prefix else CORRECTED_MANIFEST_NAME
    manifest = _parse_json_bytes(
        _snapshot_payload(snapshot, manifest_relative, label="corrected corpus manifest"),
        label="corrected corpus manifest", canonical=True,
    )
    if not isinstance(manifest, dict):
        raise CorrectedCorpusError("corrected corpus manifest must be an object")
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise CorrectedCorpusError("historical/v3.1 corpus schema rejected identity-first")
    body = dict(manifest)
    self_hash = body.pop("self_hash", None)
    if self_hash != _self_hash(body) or manifest != expected:
        raise CorrectedCorpusError("corrected corpus manifest tamper/conflict")
    inventory, digest = _inventory(pathlib.Path("."), snapshot=snapshot, prefix=prefix)
    if digest != manifest["corrected_content_inventory_digest"] or inventory != manifest["corrected_content_inventory"]:
        raise CorrectedCorpusError("corrected corpus bytes/modes inventory drift")
    return records


def _verify_child(root_or_fd, snapshot_or_expected, expected: Mapping | None = None,
                  *, prefix: str = CORRECTED_CORPUS_DIR) -> tuple[dict, ...]:
    """Snapshot verifier plus a compatibility path wrapper used by focused unit tests."""
    if expected is not None:
        return _verify_child_snapshot(root_or_fd, snapshot_or_expected, expected, prefix=prefix)
    root = pathlib.Path(root_or_fd)
    with _open_directory_path(root) as root_fd:
        snapshot = _snapshot_tree_fd(root_fd, label="corrected child")
        return _verify_child_snapshot(root_fd, snapshot, snapshot_or_expected, prefix="")


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
    parent_identity, all_records, parent_snapshot = _verify_historical_parent_with_snapshot(parent)
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
        "ruaa_selection": ruaa_selection, "parent_snapshot": parent_snapshot,
    }


def _file_map(root: pathlib.Path, *, snapshot: _TreeSnapshot | None = None,
              prefix: str = "") -> dict[str, dict]:
    rows = _recursive_inventory(
        root, excluded=("candidate.json", SHA256SUMS_NAME), snapshot=snapshot, prefix=prefix,
    )
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


def _verify_bundle_shape(snapshot: _TreeSnapshot) -> None:
    if (_immediate_children(snapshot.directories, "")
            | _immediate_children(snapshot.files, "")) != _BUNDLE_TOP:
        raise CorrectedCorpusError("bundle has missing or unexpected members")
    if _immediate_children(snapshot.directories, "") != {CORRECTED_CORPUS_DIR}:
        raise CorrectedCorpusError("bundle top-level type mismatch")
    if stat.S_IMODE(snapshot.root_info.st_mode) != 0o755:
        raise CorrectedCorpusError("mode drift in bundle root")
    for row in _recursive_inventory(pathlib.Path("."), snapshot=snapshot):
        expected = "0755" if row["type"] == "directory" else "0644"
        if row["mode"] != expected:
            raise CorrectedCorpusError(
                f"mode drift in bundle member {row['path']}: {row['mode']} != {expected}"
            )
    _validate_corpus_structure(snapshot, child=True, prefix=CORRECTED_CORPUS_DIR)


def _require_literal_child_binding(candidate: Mapping) -> None:
    layout = candidate.get("bundle_layout")
    corrected = candidate.get("corrected_corpus")
    left = layout.get("corrected_corpus_relative_root") if type(layout) is dict else None
    right = corrected.get("relative_root") if type(corrected) is dict else None
    if (type(left) is not str or type(right) is not str
            or left != CORRECTED_CORPUS_DIR or right != CORRECTED_CORPUS_DIR
            or left != right):
        raise CorrectedCorpusError(
            "candidate corrected-corpus child binding must be the exact literal corrected_corpus"
        )


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
    with _open_directory_path(root) as root_fd:
        candidate_bytes, candidate_info = _read_file_from_dir(
            root_fd, "candidate.json", label="candidate envelope", expected_mode=0o644,
        )
        candidate = _parse_json_bytes(
            candidate_bytes, label="candidate envelope", canonical=True,
        )
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
        _require_literal_child_binding(candidate)

        # No child listing, parse, hash, or descent occurs before the exact envelope gate above.
        snapshot = _snapshot_tree_fd(root_fd, label="preparation bundle")
        captured_candidate = snapshot.files.get("candidate.json")
        if (captured_candidate is None
                or candidate_info.st_size != captured_candidate.info.st_size
                or captured_candidate.sha256 != hashlib.sha256(candidate_bytes).hexdigest()):
            raise CorrectedCorpusError("candidate changed after envelope verification")
        _verify_bundle_shape(snapshot)

        inventory = _parse_json_bytes(
            _snapshot_payload(snapshot, BUNDLE_INVENTORY_NAME, label="exact inventory"),
            label="exact inventory", canonical=True,
        )
        payload_rows = _recursive_inventory(
            root, excluded=(BUNDLE_INVENTORY_NAME, "candidate.json", SHA256SUMS_NAME),
            snapshot=snapshot,
        )
        inventory_body = {
            "schema": "paired_audit.preparation_bundle_inventory.v1",
            "storage_contract_version": BUNDLE_STORAGE_CONTRACT_VERSION,
            "root_mode": "0755", "entries": payload_rows,
        }
        inventory_body["self_hash"] = _self_hash(inventory_body)
        if inventory != inventory_body:
            raise CorrectedCorpusError("exact bundle inventory mismatch")
        files = _file_map(root, snapshot=snapshot)
        if candidate.get("files") != files:
            raise CorrectedCorpusError("candidate recursive file map mismatch")
        expected_sums = _canonical_sums(root, snapshot=snapshot)
        if _snapshot_payload(snapshot, SHA256SUMS_NAME, label="SHA256SUMS") != expected_sums.encode("utf-8"):
            raise CorrectedCorpusError("non-canonical or stale SHA256SUMS")

        trusted = _trusted_derivation(parent, selection)
        child_root = root / CORRECTED_CORPUS_DIR
        expected_corpus = _corpus_manifest(
            root=child_root, records=trusted["records"], parent_identity=trusted["parent_identity"],
            policy=trusted["policy"], catalog=trusted["catalog"], basename=trusted["basename"],
            isolation=trusted["isolation"], config_hash=config_hash, protocol_sha256=protocol_sha256,
            snapshot=snapshot, prefix=CORRECTED_CORPUS_DIR,
        )
        child_records = _verify_child(root_fd, snapshot, expected_corpus)
        if child_records != trusted["records"]:
            raise CorrectedCorpusError("corrected child is not the exact trusted three-exclusion derivation")

        loaded_basename = _parse_json_bytes(
            _snapshot_payload(snapshot, "basename_collision_audit_v3_2.json", label="basename audit"),
            label="basename audit", canonical=True,
        )
        loaded_isolation = _parse_json_bytes(
            _snapshot_payload(snapshot, "content_isolation_audit_v3_2.json", label="content audit"),
            label="content audit", canonical=True,
        )
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
        lobo_loaded = _parse_json_bytes(
            _snapshot_payload(snapshot, "lobo_fold_manifest_v3_2.json", label="LOBO fold"),
            label="LOBO fold", canonical=True,
        )
        ruaa_loaded = _parse_json_bytes(
            _snapshot_payload(snapshot, "ruaa_fold_manifest_v3_2.json", label="RuAA fold"),
            label="RuAA fold", canonical=True,
        )
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


def _rename_noreplace(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Publish for a cooperative single writer without overwriting a destination."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    source.rename(destination)


def _ensure_directory(path: pathlib.Path) -> None:
    with _open_or_create_directory_path(path):
        pass


def _existing_destinations(parent: pathlib.Path) -> list[pathlib.Path]:
    with _open_directory_path(parent) as directory:
        return sorted(
            (path for path in directory.iterdir() if not path.name.startswith(".")),
            key=lambda path: path.name,
        )


def _create_stage_parent(parent: pathlib.Path) -> pathlib.Path:
    with _open_directory_path(parent) as directory:
        for _ in range(128):
            name = f".staging_v3_2_{secrets.token_hex(12)}"
            stage = directory / name
            try:
                stage.mkdir(mode=0o700)
            except FileExistsError:
                continue
            os.chmod(stage, 0o700)
            return stage
    raise CorrectedCorpusError("could not allocate a unique hidden staging directory")


def prepare_corrected_v3_2(*, historical_parent_root: pathlib.Path | str,
                           output_root: pathlib.Path | str,
                           ruaa_parent_selection: Iterable[str], config_hash: str,
                           protocol_sha256: str,
                           fault_inject: Callable[[str], None] | None = None) -> dict:
    """Build, verify, then publish one unapproved single-writer preparation bundle."""
    if not _HEX64.fullmatch(config_hash) or not _HEX64.fullmatch(protocol_sha256):
        raise CorrectedCorpusError("config/protocol hashes must be SHA256")
    parent = pathlib.Path(historical_parent_root)
    output = pathlib.Path(output_root)
    selection = tuple(ruaa_parent_selection)
    _ensure_directory(output)
    bundle_parent = output / BUNDLE_PARENT_NAME
    _ensure_directory(bundle_parent)
    existing = _existing_destinations(bundle_parent)
    if existing:
        if len(existing) != 1:
            raise CorrectedCorpusError("bundle parent has multiple existing destinations")
        return verify_v3_2_candidate(
            existing[0], historical_parent_root=parent, ruaa_parent_selection=selection,
            config_hash=config_hash, protocol_sha256=protocol_sha256,
        )
    trusted = _trusted_derivation(parent, selection)
    stage_parent = _create_stage_parent(bundle_parent)
    work_root = stage_parent / "bundle"
    published = False
    try:
        corpus_root = work_root / CORRECTED_CORPUS_DIR
        _created_dir(work_root)
        corpus = _assemble_child(
            parent, corpus_root, trusted["records"], parent_identity=trusted["parent_identity"],
            policy=trusted["policy"], catalog=trusted["catalog"], basename=trusted["basename"],
            isolation=trusted["isolation"], config_hash=config_hash, protocol_sha256=protocol_sha256,
            parent_snapshot=trusted["parent_snapshot"],
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
        _write_new_file(sums_path, _canonical_sums(work_root).encode("utf-8"))
        stage = work_root.rename(stage_parent / candidate["self_hash"])
        verified = verify_v3_2_candidate(
            stage, historical_parent_root=parent, ruaa_parent_selection=selection,
            config_hash=config_hash, protocol_sha256=protocol_sha256,
            fault_inject=fault_inject,
        )
        destination = bundle_parent / candidate["self_hash"]
        _fault(fault_inject, "before_final_rename")
        collision = False
        _fault(fault_inject, "final_rename")
        try:
            _rename_noreplace(stage, destination)
            published = True
        except FileExistsError:
            collision = True
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
    with _open_directory_path(root) as directory:
        candidate_path = directory / "candidate.json"
        candidate = (
            candidate_path.lstat()
            if candidate_path.exists() or candidate_path.is_symlink()
            else None
        )
        if candidate is not None:
            if not stat.S_ISREG(candidate.st_mode):
                raise CorrectedCorpusError("parity candidate is symlinked or special")
            return root
    parent = root / BUNDLE_PARENT_NAME
    with _open_directory_path(parent) as directory:
        candidates = sorted(
            (path for path in directory.iterdir() if not path.name.startswith(".")),
            key=lambda path: path.name,
        )
    if len(candidates) != 1:
        raise CorrectedCorpusError("parity root must contain exactly one digest-named bundle")
    return candidates[0]


def tree_bytes_modes(root: pathlib.Path | str) -> tuple[dict[str, tuple], str]:
    """Canonical parity over one bundle root and every descendant path/type/size/mode/byte hash."""
    bundle = _parity_bundle_root(root)
    snapshot = _snapshot_path(bundle, label="preparation parity")
    if stat.S_IMODE(snapshot.root_info.st_mode) != 0o755:
        raise CorrectedCorpusError("mode drift in parity bundle root")
    inventory = _recursive_inventory(bundle, snapshot=snapshot)
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
    "load_stable_json", "read_stable_bytes", "resolve_full_work",
    "tree_bytes_modes", "verify_historical_parent", "verify_v3_2_candidate",
    "work_identity_catalog",
]
