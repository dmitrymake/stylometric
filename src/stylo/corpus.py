"""Единая загрузка корпуса фрагментов.

Ожидаемая раскладка (после split): {frags_root}/{author}/{book_id}/{chunk}.txt
Группа LOBO = "author/book_id".
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np


@dataclasses.dataclass
class Dataset:
    texts: np.ndarray      # dtype=object, тексты чанков
    y: np.ndarray          # dtype=int, индекс автора
    groups: np.ndarray     # dtype=object, "author/book_id"
    authors: List[str]     # отсортированный список id авторов (индекс == метка)
    provenance: object = None  # DatasetProvenance (B2) — набор данных привязан к weighting-арме

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


def iter_fragment_files(frags_root: pathlib.Path) -> List[pathlib.Path]:
    return sorted(p for p in frags_root.rglob("*.txt") if p.is_file())


def infer_author_book(fp: pathlib.Path, frags_root: pathlib.Path) -> Tuple[str, str]:
    """(author, book_id) из пути frags_root/author/book_id/chunk.txt с фолбэками."""
    parts = fp.relative_to(frags_root).parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return parts[0], fp.stem
    return "unknown", fp.stem


def list_authors(frags_root: pathlib.Path, exclude_unknown: str = "unknown",
                 exclude: Sequence[str] = ()) -> List[str]:
    excl = set(exclude) | {exclude_unknown}
    return sorted(
        p.name for p in frags_root.iterdir()
        if p.is_dir() and p.name not in excl
    )


def load_dataset(
    frags_root: pathlib.Path | str,
    exclude_authors: Iterable[str] = (),
    unknown_name: str = "unknown",
) -> Dataset:
    """Загрузить все train-фрагменты (исключая каталог unknown и exclude_authors)."""
    frags_root = pathlib.Path(frags_root)
    if not frags_root.exists():
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

    for fp in iter_fragment_files(frags_root):
        author, book_id = infer_author_book(fp, frags_root)
        if author not in auth2idx:
            continue
        try:
            txt = fp.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not txt:
            continue
        group = f"{author}/{book_id}"
        texts.append(txt)
        y.append(auth2idx[author])
        groups.append(group)
        ordinal = _ordinals.get(group, 0)
        _ordinals[group] = ordinal + 1
        import hashlib as _hl
        from .eval.provenance import RowIdentity
        row_ids.append(RowIdentity(group=group, ordinal=ordinal,
                                   text_sha256=_hl.sha256(txt.encode("utf-8")).hexdigest()))

    if len(texts) < 10:
        raise ValueError(f"Слишком мало фрагментов после загрузки: {len(texts)}")

    from .eval.provenance import (LEGACY_RECURSIVE, CorpusPolicyProvenance,
                                  build_provenance)
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
    if not root.exists():
        return []
    out: List[str] = []
    for fp in iter_fragment_files(root):
        try:
            txt = fp.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if txt:
            out.append(txt)
    return out
