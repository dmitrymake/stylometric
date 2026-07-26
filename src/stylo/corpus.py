"""Единая загрузка корпуса фрагментов.

Ожидаемая раскладка (после split): {frags_root}/{author}/{book_id}/{chunk}.txt
Группа LOBO = "author/book_id".
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

from .domain.corpus_identity import (
    LEGACY_RECURSIVE,
    CorpusPolicyProvenance,
    RowIdentity,
    build_provenance,
)
from .jsonio import load_strict


@dataclasses.dataclass
class Dataset:
    texts: np.ndarray      # dtype=object, тексты чанков
    y: np.ndarray          # dtype=int, индекс автора
    groups: np.ndarray     # dtype=object, "author/book_id"
    authors: List[str]     # отсортированный список id авторов (индекс == метка)
    provenance: object = None  # DatasetProvenance binds rows to the selected weighting contract

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_authors(self) -> int:
        return len(self.authors)

    def book_to_author(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for g, lbl in zip(self.groups, self.y):
            out.setdefault(str(g), int(lbl))
        return out


class CorpusLoadError(ValueError):
    """The on-disk corpus is incomplete, malformed, or unsafe."""


def iter_fragment_files(frags_root: pathlib.Path) -> List[pathlib.Path]:
    return sorted(p for p in frags_root.rglob("*.txt") if p.is_file())


def infer_author_book(fp: pathlib.Path, frags_root: pathlib.Path) -> Tuple[str, str]:
    """Return the strict ``author/work`` identity for a canonical chunk path."""
    parts = fp.relative_to(frags_root).parts
    if len(parts) == 3:
        return parts[0], parts[1]
    raise CorpusLoadError(
        f"chunk must have layout author/work/chunk.txt, got {fp}"
    )


def list_authors(frags_root: pathlib.Path, exclude_unknown: str = "unknown",
                 exclude: Sequence[str] = ()) -> List[str]:
    excl = set(exclude) | {exclude_unknown}
    authors: list[str] = []
    for entry in sorted(os.scandir(frags_root), key=lambda item: item.name):
        if entry.is_symlink():
            raise CorpusLoadError(f"symlinked corpus entry rejected: {entry.path}")
        if entry.is_dir(follow_symlinks=False) and entry.name not in excl:
            authors.append(entry.name)
        elif entry.name.endswith(".txt"):
            raise CorpusLoadError(
                f"stray chunk at corpus root; expected author/work/chunk.txt: {entry.path}"
            )
    return authors


def _strict_legacy_inventory(
    root: pathlib.Path,
    authors: Sequence[str],
) -> list[tuple[pathlib.Path, str, str]]:
    """Inventory an exact, symlink-free author/work/chunk hierarchy.

    A work manifest is optional for historical corpora.  When present, its
    chunk list and text hashes are authoritative, making deletion/rename or
    substitution detectable before model fitting.
    """

    inventory: list[tuple[pathlib.Path, str, str]] = []
    for author in authors:
        author_dir = root / author
        work_entries = sorted(os.scandir(author_dir), key=lambda item: item.name)
        work_dirs: list[pathlib.Path] = []
        for entry in work_entries:
            if entry.is_symlink():
                raise CorpusLoadError(f"symlinked work entry rejected: {entry.path}")
            if entry.is_dir(follow_symlinks=False):
                work_dirs.append(pathlib.Path(entry.path))
            elif entry.name.endswith(".txt"):
                raise CorpusLoadError(
                    f"stray author-level chunk rejected: {entry.path}"
                )
        if not work_dirs:
            raise CorpusLoadError(f"author {author!r} has no work directories")

        author_rows = 0
        for work_dir in work_dirs:
            entries = sorted(os.scandir(work_dir), key=lambda item: item.name)
            chunk_paths: list[pathlib.Path] = []
            for entry in entries:
                if entry.is_symlink():
                    raise CorpusLoadError(f"symlinked chunk entry rejected: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    raise CorpusLoadError(
                        f"nested directory below work is not allowed: {entry.path}"
                    )
                if entry.name.endswith(".txt"):
                    chunk_paths.append(pathlib.Path(entry.path))
            if not chunk_paths:
                raise CorpusLoadError(f"work {author}/{work_dir.name} has no chunks")

            manifest_path = work_dir / "manifest.json"
            expected_hashes: dict[str, str] | None = None
            if manifest_path.exists():
                if manifest_path.is_symlink() or not manifest_path.is_file():
                    raise CorpusLoadError(
                        f"unsafe work manifest: {manifest_path}"
                    )
                try:
                    raw = load_strict(manifest_path)
                    chunks = raw["chunks"]
                    if type(raw) is not dict or type(chunks) is not list or not chunks:
                        raise TypeError("manifest/chunks schema")
                    expected_hashes = {}
                    for chunk in chunks:
                        if (
                            type(chunk) is not dict
                            or set(chunk) != {"span_ordinal", "text_sha256", "path"}
                            or type(chunk["path"]) is not str
                            or pathlib.PurePath(chunk["path"]).name != chunk["path"]
                            or type(chunk["text_sha256"]) is not str
                        ):
                            raise TypeError("manifest chunk schema")
                        expected_hashes[chunk["path"]] = chunk["text_sha256"]
                    if len(expected_hashes) != len(chunks):
                        raise TypeError("duplicate manifest chunk path")
                except Exception as exc:
                    raise CorpusLoadError(
                        f"malformed work manifest {manifest_path}: {exc}"
                    ) from exc
                actual_names = {path.name for path in chunk_paths}
                if actual_names != set(expected_hashes):
                    raise CorpusLoadError(
                        f"manifest/file mismatch for {author}/{work_dir.name}: "
                        f"missing={sorted(set(expected_hashes) - actual_names)}, "
                        f"extra={sorted(actual_names - set(expected_hashes))}"
                    )

            for path in chunk_paths:
                try:
                    decoded = path.read_bytes().decode("utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise CorpusLoadError(
                        f"unreadable/non-UTF-8 chunk {path}: {exc}"
                    ) from exc
                canonical = decoded.strip()
                if not canonical:
                    raise CorpusLoadError(f"empty chunk rejected: {path}")
                if expected_hashes is not None:
                    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    if digest != expected_hashes[path.name]:
                        raise CorpusLoadError(
                            f"manifest text hash mismatch for {path}"
                        )
                inventory.append((path, canonical, f"{author}/{work_dir.name}"))
                author_rows += 1
        if author_rows == 0:
            raise CorpusLoadError(f"author {author!r} contributes zero rows")
    return inventory


def load_dataset(
    frags_root: pathlib.Path | str,
    exclude_authors: Iterable[str] = (),
    unknown_name: str = "unknown",
) -> Dataset:
    """Загрузить все train-фрагменты (исключая каталог unknown и exclude_authors)."""
    frags_root = pathlib.Path(frags_root)
    if frags_root.is_symlink():
        raise CorpusLoadError(f"corpus root must not be a symlink: {frags_root}")
    if not frags_root.is_dir():
        raise FileNotFoundError(f"Каталог фрагментов не найден: {frags_root}")

    if isinstance(exclude_authors, (str, bytes)):     # a bare string would become a set of letters
        raise ValueError("exclude_authors must be an iterable of author ids, not a string")
    exclude_authors = tuple(exclude_authors)          # materialize ONCE (avoid generator exhaustion)
    excl: Set[str] = set(exclude_authors)
    authors = list_authors(frags_root, exclude_unknown=unknown_name, exclude=excl)
    if len(authors) < 2:
        raise ValueError(f"Нужно минимум 2 автора, найдено: {authors}")
    auth2idx = {a: i for i, a in enumerate(authors)}

    texts: List[str] = []
    y: List[int] = []
    groups: List[str] = []
    row_ids: List = []
    _ordinals: dict[str, int] = {}

    inventory = _strict_legacy_inventory(frags_root, authors)
    for fp, txt, group in inventory:
        author, book_id = group.split("/", 1)
        texts.append(txt)
        y.append(auth2idx[author])
        groups.append(group)
        ordinal = _ordinals.get(group, 0)
        _ordinals[group] = ordinal + 1
        row_ids.append(RowIdentity(group=group, ordinal=ordinal,
                                   text_sha256=hashlib.sha256(txt.encode("utf-8")).hexdigest()))

    if len(texts) < 10:
        raise ValueError(f"Слишком мало фрагментов после загрузки: {len(texts)}")

    prov = build_provenance(
        loader_kind=LEGACY_RECURSIVE,
        texts=texts, y=y, groups=groups, authors=authors, row_ids=row_ids,
        frags_root=str(frags_root.resolve()),
        corpus_policy=CorpusPolicyProvenance.build(excl, unknown_name),
    )
    return Dataset(
        texts=np.asarray(texts, dtype=object),
        y=np.asarray(y, dtype=int),
        groups=np.asarray(groups, dtype=object),
        authors=authors,
        provenance=prov,
    )


def load_unknown(frags_root: pathlib.Path | str, unknown_name: str = "unknown") -> List[str]:
    """Загрузить тексты спорного автора (каталог unknown) для атрибуции."""
    root = pathlib.Path(frags_root) / unknown_name
    if root.is_symlink():
        raise CorpusLoadError(f"unknown root must not be a symlink: {root}")
    if not root.exists():
        return []
    out: List[str] = []
    for fp in iter_fragment_files(root):
        if fp.is_symlink():
            raise CorpusLoadError(f"symlinked unknown fragment rejected: {fp}")
        try:
            txt = fp.read_bytes().decode("utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise CorpusLoadError(f"invalid unknown fragment {fp}: {exc}") from exc
        if not txt:
            raise CorpusLoadError(f"empty unknown fragment rejected: {fp}")
        out.append(txt)
    return out
