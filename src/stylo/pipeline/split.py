"""Build and atomically publish complete cleaned-corpus chunk snapshots."""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import logging
import os
import pathlib
import re
import shutil
import stat
import tempfile
from typing import Sequence

from ..chunking import CombinedDoc, make_sent_chunks, sentences_for_text
from ..config import load_config
from ..jsonio import canonical_hash, dump_strict, dumps_strict, loads_strict
from ..nlp import load_sentencizer
from ..workdoc import (
    MANIFEST_NAME,
    build_work_manifest,
    chunker_config_hash,
    frozen_chunker_config,
    sha256_text,
)
from ._snapshot import _fsync_dir, _fsync_tree

log = logging.getLogger("stylo.pipeline.split")

SPLIT_SCHEMA = "stylo.fragment-snapshot.v3"
SPLIT_MANIFEST = "split_manifest.json"
GENERATION_MANIFEST = "generation_manifest.json"
SNAPSHOT_DIRECTORY = "fragment_snapshots"
VERSIONS_DIRECTORY = "versions"
CURRENT_POINTER = "CURRENT.json"
POINTER_SCHEMA = "stylo.fragment-snapshot-pointer.v1"
PUBLISH_LOCK = ".publish.lock"
_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")


class FragmentSnapshotError(RuntimeError):
    """A fragment generation or its single current pointer is invalid."""


@dataclasses.dataclass(frozen=True)
class FragmentSnapshot:
    """One coherently resolved train/unknown/map fragment generation."""

    generation_id: str
    root: pathlib.Path
    train_root: pathlib.Path
    unknown_root: pathlib.Path
    chunk_map: pathlib.Path | None
    versioned: bool


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_nofollow(path: pathlib.Path, *, label: str) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise FragmentSnapshotError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise FragmentSnapshotError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _safe_generation_relative(value: object) -> str:
    if type(value) is not str or not value:
        raise FragmentSnapshotError("generation file path must be a nonempty string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise FragmentSnapshotError(f"unsafe generation file path: {value!r}")
    return value


def _generation_files(root: pathlib.Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = pathlib.Path(dirpath)
        for name in dirnames:
            child = directory / name
            if child.is_symlink():
                raise FragmentSnapshotError(f"symlink in fragment generation: {child}")
        for name in filenames:
            child = directory / name
            if child.name == GENERATION_MANIFEST and child.parent == root:
                continue
            if child.is_symlink() or not child.is_file():
                raise FragmentSnapshotError(
                    f"fragment generation member is not a regular file: {child}"
                )
            relative = child.relative_to(root).as_posix()
            files[relative] = _sha256_file(child)
    return dict(sorted(files.items()))


def _generation_token(generation: dict, files: dict[str, str]) -> str:
    return canonical_hash(
        {
            "schema_version": SPLIT_SCHEMA,
            "generation": generation,
            "files": files,
        }
    )


def _validate_generation(
    generation_root: pathlib.Path,
    *,
    expected_token: str,
) -> FragmentSnapshot:
    if (
        not _TOKEN_RE.fullmatch(expected_token)
        or generation_root.is_symlink()
        or not generation_root.is_dir()
    ):
        raise FragmentSnapshotError(
            f"fragment generation missing, unsafe, or malformed: {generation_root}"
        )
    manifest_path = generation_root / GENERATION_MANIFEST
    try:
        manifest = loads_strict(
            _read_regular_nofollow(
                manifest_path, label="fragment generation manifest"
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise FragmentSnapshotError(f"invalid fragment generation manifest: {exc}") from exc
    if type(manifest) is not dict or set(manifest) != {
        "schema_version",
        "generation_id",
        "generation",
        "files",
    }:
        raise FragmentSnapshotError("fragment generation manifest field mismatch")
    if (
        manifest["schema_version"] != SPLIT_SCHEMA
        or manifest["generation_id"] != expected_token
        or type(manifest["generation"]) is not dict
        or type(manifest["files"]) is not dict
    ):
        raise FragmentSnapshotError("fragment generation identity mismatch")
    registered: dict[str, str] = {}
    for raw_path, digest in manifest["files"].items():
        relative = _safe_generation_relative(raw_path)
        if (
            type(digest) is not str
            or _TOKEN_RE.fullmatch(digest) is None
            or relative in registered
        ):
            raise FragmentSnapshotError("fragment generation file digest map is malformed")
        registered[relative] = digest
    registered = dict(sorted(registered.items()))
    if _generation_token(manifest["generation"], registered) != expected_token:
        raise FragmentSnapshotError("fragment generation token does not bind its manifest")
    observed = _generation_files(generation_root)
    if observed != registered:
        raise FragmentSnapshotError("fragment generation inventory/hash mismatch")
    train_root = generation_root / "frags_train"
    unknown_root = generation_root / "frags_unknown"
    chunk_map = generation_root / "chunk_map.json"
    if (
        train_root.is_symlink()
        or not train_root.is_dir()
        or unknown_root.is_symlink()
        or not unknown_root.is_dir()
        or chunk_map.is_symlink()
        or not chunk_map.is_file()
    ):
        raise FragmentSnapshotError("fragment generation endpoints are incomplete")
    return FragmentSnapshot(
        generation_id=expected_token,
        root=generation_root,
        train_root=train_root,
        unknown_root=unknown_root,
        chunk_map=chunk_map,
        versioned=True,
    )


def _load_pointer(pointer_path: pathlib.Path) -> str:
    try:
        pointer = loads_strict(
            _read_regular_nofollow(pointer_path, label="fragment CURRENT pointer").decode(
                "utf-8"
            )
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise FragmentSnapshotError(f"invalid fragment CURRENT pointer: {exc}") from exc
    if (
        type(pointer) is not dict
        or set(pointer) != {"schema_version", "generation_id"}
        or pointer["schema_version"] != POINTER_SCHEMA
        or type(pointer["generation_id"]) is not str
        or _TOKEN_RE.fullmatch(pointer["generation_id"]) is None
    ):
        raise FragmentSnapshotError("fragment CURRENT pointer field mismatch")
    return pointer["generation_id"]


def resolve_fragment_snapshot(
    data_root: str | pathlib.Path,
    *,
    require_versioned: bool = False,
) -> FragmentSnapshot:
    """Resolve one pointer once and validate its complete immutable generation.

    A read-only legacy fallback is retained for a checkout that has not yet run
    the v3 splitter.  Canonical split publication never mutates those legacy
    sibling roots.
    """

    data = pathlib.Path(data_root)
    if data.is_symlink() or not data.is_dir():
        raise FragmentSnapshotError(f"data root must be a real directory: {data}")
    publication_root = data / SNAPSHOT_DIRECTORY
    pointer_path = publication_root / CURRENT_POINTER
    if pointer_path.exists() or pointer_path.is_symlink():
        if publication_root.is_symlink() or not publication_root.is_dir():
            raise FragmentSnapshotError("fragment publication root is unsafe")
        token = _load_pointer(pointer_path)
        versions = publication_root / VERSIONS_DIRECTORY
        if versions.is_symlink() or not versions.is_dir():
            raise FragmentSnapshotError("fragment versions root is unsafe")
        return _validate_generation(versions / token, expected_token=token)
    if require_versioned:
        raise FragmentSnapshotError("no versioned fragment CURRENT pointer is published")
    train_root = data / "frags_train"
    unknown_root = data / "frags_unknown"
    if (
        train_root.is_symlink()
        or not train_root.is_dir()
        or unknown_root.is_symlink()
        or not unknown_root.is_dir()
    ):
        raise FragmentSnapshotError(
            "neither a versioned fragment snapshot nor complete legacy roots exist"
        )
    chunk_map = data / "chunk_map.json"
    if chunk_map.is_symlink():
        raise FragmentSnapshotError("legacy chunk map must not be a symlink")
    return FragmentSnapshot(
        generation_id="legacy-unversioned",
        root=data,
        train_root=train_root,
        unknown_root=unknown_root,
        chunk_map=chunk_map if chunk_map.is_file() else None,
        versioned=False,
    )


def _publish_current_pointer(
    publication_root: pathlib.Path,
    generation_id: str,
) -> None:
    pointer_fd, pointer_name = tempfile.mkstemp(
        prefix=".CURRENT.", dir=publication_root
    )
    os.close(pointer_fd)
    pointer_tmp = pathlib.Path(pointer_name)
    try:
        pointer_tmp.write_text(
            dumps_strict(
                {
                    "schema_version": POINTER_SCHEMA,
                    "generation_id": generation_id,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with open(pointer_tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(pointer_tmp, publication_root / CURRENT_POINTER)
        _fsync_dir(publication_root)
    finally:
        if pointer_tmp.exists():
            pointer_tmp.unlink()


def _publish_generation(
    staging: pathlib.Path,
    publication_root: pathlib.Path,
    *,
    generation: dict,
) -> str:
    files = _generation_files(staging)
    token = _generation_token(generation, files)
    dump_strict(
        {
            "schema_version": SPLIT_SCHEMA,
            "generation_id": token,
            "generation": generation,
            "files": files,
        },
        staging / GENERATION_MANIFEST,
        sort_keys=True,
    )
    _fsync_tree(staging)
    versions = publication_root / VERSIONS_DIRECTORY
    lock_path = publication_root / PUBLISH_LOCK
    lock_fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise FragmentSnapshotError("fragment publication lock is not a regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        version = versions / token
        if version.exists() or version.is_symlink():
            _validate_generation(version, expected_token=token)
            shutil.rmtree(staging)
        else:
            os.replace(staging, version)
            _fsync_dir(versions)
        _publish_current_pointer(publication_root, token)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return token


def _source_books(src: pathlib.Path) -> list[tuple[pathlib.Path, bytes, str]]:
    if src.is_symlink() or not src.is_dir():
        raise RuntimeError(f"clean corpus root must be a real directory: {src}")
    books: list[tuple[pathlib.Path, bytes, str]] = []
    for author_entry in sorted(os.scandir(src), key=lambda item: item.name):
        if author_entry.is_symlink():
            raise RuntimeError(f"symlinked clean-corpus entry rejected: {author_entry.path}")
        if not author_entry.is_dir(follow_symlinks=False):
            continue
        author_dir = pathlib.Path(author_entry.path)
        author_books = 0
        for book_entry in sorted(os.scandir(author_dir), key=lambda item: item.name):
            if book_entry.is_symlink():
                raise RuntimeError(f"symlinked clean source rejected: {book_entry.path}")
            if book_entry.is_dir(follow_symlinks=False):
                raise RuntimeError(
                    f"nested clean-corpus directory is unsupported: {book_entry.path}"
                )
            if not (
                book_entry.is_file(follow_symlinks=False)
                and book_entry.name.endswith(".txt")
            ):
                continue
            path = pathlib.Path(book_entry.path)
            payload = path.read_bytes()
            try:
                text = payload.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"clean source is not strict UTF-8: {path}") from exc
            if not text:
                raise RuntimeError(f"clean source is empty: {path}")
            books.append((path, payload, text))
            author_books += 1
        if author_books == 0:
            raise RuntimeError(f"clean author {author_dir.name!r} has no .txt works")
    if not books:
        raise RuntimeError(f"clean corpus contains no author/*.txt works: {src}")
    return books


def _source_receipt(
    src: pathlib.Path,
    books: list[tuple[pathlib.Path, bytes, str]],
) -> list[dict[str, str]]:
    return [
        {
            "source": path.relative_to(src).as_posix(),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload, _text in books
    ]


def _assert_sources_unchanged(
    src: pathlib.Path,
    receipt: list[dict[str, str]],
) -> None:
    current = _source_books(src)
    if _source_receipt(src, current) != receipt:
        raise RuntimeError("clean corpus changed during chunk snapshot construction")


def run(cfg=None, leave_out: Sequence[str] = (), clean_existing: bool = True) -> int:
    """Chunk every expected clean work, validate a bijection, then publish.

    ``clean_existing=False`` formerly overlaid new files on an unknown old
    generation.  That mode cannot prove completeness and is now rejected.
    """

    if clean_existing is not True:
        raise ValueError("split only supports complete atomic replacement snapshots")
    cfg = cfg or load_config()
    src = pathlib.Path(cfg.get_path("paths.input_clean", "input_clean"))
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    if data.is_symlink():
        raise RuntimeError(f"data root must not be a symlink: {data}")
    data.mkdir(parents=True, exist_ok=True)

    chunker = frozen_chunker_config(cfg)
    unknown_name = cfg.get_path("corpus_policy.unknown_dir_name", "unknown")
    if isinstance(leave_out, (str, bytes)):
        raise ValueError("leave_out must be a sequence of exact work ids, not a string")
    leave = set(leave_out)
    if any(type(work_id) is not str or not work_id for work_id in leave):
        raise ValueError("leave_out must contain nonempty strings")

    books = _source_books(src)
    receipt = _source_receipt(src, books)
    known_work_ids = {
        f"{path.parent.name}/{path.stem}" for path, _payload, _text in books
    }
    unknown_leave = leave - known_work_ids
    if unknown_leave:
        raise RuntimeError(f"leave-out work ids not present in clean corpus: {sorted(unknown_leave)}")

    publication_root = data / SNAPSHOT_DIRECTORY
    if publication_root.is_symlink():
        raise RuntimeError(f"fragment publication root must not be a symlink: {publication_root}")
    publication_root.mkdir(exist_ok=True)
    versions = publication_root / VERSIONS_DIRECTORY
    if versions.is_symlink():
        raise RuntimeError(f"fragment versions root must not be a symlink: {versions}")
    versions.mkdir(exist_ok=True)
    generation_stage = pathlib.Path(
        tempfile.mkdtemp(prefix=".staging-", dir=versions)
    )
    train_stage = generation_stage / "frags_train"
    unknown_stage = generation_stage / "frags_unknown"
    train_stage.mkdir()
    unknown_stage.mkdir()
    mapping: list[dict[str, str]] = []
    produced_works: set[str] = set()
    cfg_hash = chunker_config_hash(cfg)
    nlp = load_sentencizer(chunker.language)

    try:
        for book, _payload, raw in books:
            author = book.parent.name
            book_id = book.stem
            work_id = f"{author}/{book_id}"
            sentences = sentences_for_text(raw, nlp)
            if not sentences:
                raise RuntimeError(f"{work_id}: sentencizer produced no sentences")
            chunks = make_sent_chunks(
                CombinedDoc(sentences),
                chunker.chunk_size,
                chunker.min_words,
                chunker.overlap,
            )
            if not chunks:
                raise RuntimeError(
                    f"{work_id}: chunker produced no chunks; every clean work is required"
                )

            to_unknown = author == unknown_name or work_id in leave
            stage_root = unknown_stage if to_unknown else train_stage
            published_root = "frags_unknown" if to_unknown else "frags_train"
            out_dir = stage_root / author / book_id
            out_dir.mkdir(parents=True, exist_ok=False)
            filenames = [f"{book_id}_{idx:05d}.txt" for idx in range(len(chunks))]
            for name, chunk in zip(filenames, chunks, strict=True):
                (out_dir / name).write_text(chunk, encoding="utf-8")
                mapping.append(
                    {
                        "path": f"{published_root}/{author}/{book_id}/{name}",
                        "author": author,
                        "book": book_id,
                        "split": "unknown" if to_unknown else "train",
                    }
                )
            manifest = build_work_manifest(
                work_id,
                author,
                chunks,
                filenames,
                provenance_sha256=sha256_text(raw),
                chunker_config_hash=cfg_hash,
                overlap=float(chunker.overlap),
            )
            dump_strict(
                manifest.to_dict(),
                out_dir / MANIFEST_NAME,
                trailing_newline=False,
            )
            produced_works.add(work_id)

        expected_works = {
            f"{path.parent.name}/{path.stem}" for path, _payload, _text in books
        }
        if produced_works != expected_works:
            raise RuntimeError(
                "split snapshot is not an exact clean-work bijection: "
                f"missing={sorted(expected_works-produced_works)}, "
                f"extra={sorted(produced_works-expected_works)}"
            )
        _assert_sources_unchanged(src, receipt)
        generation = {
            "schema_version": SPLIT_SCHEMA,
            "chunker_config_hash": cfg_hash,
            "source_files": receipt,
            "works": sorted(produced_works),
            "n_chunks": len(mapping),
        }
        dump_strict(generation, train_stage / SPLIT_MANIFEST, sort_keys=True)
        dump_strict(generation, unknown_stage / SPLIT_MANIFEST, sort_keys=True)
        _assert_sources_unchanged(src, receipt)

        dump_strict(
            mapping,
            generation_stage / "chunk_map.json",
            trailing_newline=False,
        )
        _publish_generation(
            generation_stage,
            publication_root,
            generation=generation,
        )
        generation_stage = None
    finally:
        if generation_stage is not None and generation_stage.exists():
            shutil.rmtree(generation_stage)

    log.info("Нарезка завершена: %d чанков / %d работ", len(mapping), len(books))
    return len(mapping)
