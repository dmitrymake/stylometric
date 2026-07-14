"""Canonical WorkDocument, chunk manifest and the single manifest-driven loader (P1 B0).

The work-balanced estimand needs one canonical, verified view of the training data so the
gate and the model never diverge. Every work folder carries a strict-JSON ``manifest.json``
recording, per chunk, its ordinal and the sha256 of the **canonical (model-representation)
text** — the whitespace-stripped chunk the model actually consumes — plus the work-level
provenance and chunker-config hashes and the chunker overlap. The canonical identity is

    (work_id, provenance_sha256, chunker_config_hash, span_ordinal, text_sha256)

Two identical texts at *different* ordinals are kept; a repeat of one full identity fails
closed. :func:`load_work_balanced_dataset` is the ONLY loader the ``work_balanced`` path
uses — it returns exactly the validated manifest texts, in manifest order, so a nested /
stray / symlinked / non-UTF-8 / whitespace-only file can never be trained on while the gate
reports clean. ``chunk_weighted_legacy`` keeps using :func:`corpus.load_dataset`; both read
the same stripped text, so P0 is byte-reproducible.

Contract scope (design v3 §1 + B0 audits): ordinal-only spans, valid only at
``overlap == 0`` (non-zero fails closed). Provenance is mandatory. Symlinks are rejected at
every author/work/source/chunk component and paths must resolve inside the corpus root.
The chunker-config hash binds a single frozen typed config extractor (shared with the
splitter), the sentencizer (``spacy.blank`` + rule-based sentencizer, keyed by the spaCy
version), and an explicit normalization/name-masking contract version; the exact cleaned
source bytes are additionally bound via ``provenance_sha256``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import pathlib
import re
from typing import Any, Iterable, Sequence

import spacy

from .jsonio import dumps_strict, load_strict

CHUNKER_ALGORITHM = "stylo.sent_chunks/v1"
NORMALIZATION_CONTRACT = "stylo.clean/v1"  # dash-normalize + NER PER->@ + garbage strip (pipeline/clean.py)
MANIFEST_NAME = "manifest.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

__all__ = [
    "CHUNKER_ALGORITHM",
    "NORMALIZATION_CONTRACT",
    "MANIFEST_NAME",
    "ChunkEntry",
    "WorkManifest",
    "ChunkerConfig",
    "ManifestError",
    "sha256_text",
    "canonical_chunk_text",
    "source_provenance_sha256",
    "frozen_chunker_config",
    "chunker_config_hash",
    "build_work_manifest",
    "chunk_identity",
    "validate_work_manifest",
    "load_work_manifest",
    "load_work_balanced_dataset",
    "resolve_dataset",
]


class ManifestError(ValueError):
    """A work manifest is missing, malformed, stale, unsafe or inconsistent with its files."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_chunk_text(text: str) -> str:
    """The exact representation the model consumes (matches corpus.load_dataset .strip())."""
    return text.strip()


def source_provenance_sha256(path: str | pathlib.Path) -> str:
    """Provenance hash of a cleaned source book — the exact bytes the chunker consumed."""
    return sha256_text(pathlib.Path(path).read_text(encoding="utf-8").strip())


# ── frozen typed chunker config (shared by splitter and hash) ────────────────
@dataclasses.dataclass(frozen=True)
class ChunkerConfig:
    chunk_size: int
    min_words: int
    overlap: float
    language: str
    masking_model: str
    masking_model_version: str
    masking_fallback: str


def _req_str_cfg(cfg, path: str) -> str:
    value = cfg.get_path(path, None)
    if type(value) is not str:
        raise ManifestError(f"{path} must be a string, got {value!r}")
    return value


def frozen_chunker_config(cfg) -> ChunkerConfig:
    """Validated, coercion-free chunker + masking settings used by the splitter and hash.

    Rejects non-int sizes (200 and 200.5 must not collide), non-finite / out-of-range
    overlap, and non-string model fields (so None/NaN cannot sanitise to the same null).
    ``overlap`` is canonicalised so ``-0.0`` and ``0.0`` hash identically.
    """
    chunk_size = cfg.get_path("chunking.chunk_size", 500)
    min_words = cfg.get_path("chunking.min_words", 200)
    overlap = cfg.get_path("chunking.overlap", 0.0)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ManifestError(f"chunking.chunk_size must be a positive int, got {chunk_size!r}")
    if type(min_words) is not int or min_words <= 0:
        raise ManifestError(f"chunking.min_words must be a positive int, got {min_words!r}")
    if type(overlap) is bool or not isinstance(overlap, (int, float)) or not math.isfinite(overlap) or not (0.0 <= overlap < 1.0):
        raise ManifestError(f"chunking.overlap must be a finite float in [0,1), got {overlap!r}")
    overlap = 0.0 if overlap == 0.0 else float(overlap)  # normalise -0.0 -> 0.0
    return ChunkerConfig(
        chunk_size, min_words, overlap,
        language=_req_str_cfg(cfg, "language.code"),
        masking_model=_req_str_cfg(cfg, "language.spacy_model"),
        masking_model_version=_req_str_cfg(cfg, "language.spacy_model_version"),
        masking_fallback=_req_str_cfg(cfg, "language.spacy_fallback"),
    )


def chunker_config_hash(cfg) -> str:
    """Typed hash of the actual chunking + sentencizer + normalization/masking dependencies."""
    cc = frozen_chunker_config(cfg)
    payload = {
        "algorithm": CHUNKER_ALGORITHM,
        "sentencizer": "spacy.blank+rule_based_sentencizer",
        "spacy_version": spacy.__version__,
        "language": cc.language,
        "chunk_size": cc.chunk_size,
        "min_words": cc.min_words,
        "overlap": cc.overlap,
        "normalization_contract": NORMALIZATION_CONTRACT,
        "masking_model": cc.masking_model,
        "masking_model_version": cc.masking_model_version,
        "masking_fallback": cc.masking_fallback,
    }
    return sha256_text(dumps_strict(payload, sort_keys=True))


# ── strict typed field readers (no coercion) ─────────────────────────────────
def _req_str(d: dict, key: str) -> str:
    value = d[key]
    if type(value) is not str:
        raise ManifestError(f"{key!r} must be a string, got {type(value).__name__}")
    return value


def _req_sha256(d: dict, key: str) -> str:
    value = _req_str(d, key)
    if not _SHA256_RE.fullmatch(value):
        raise ManifestError(f"{key!r} must be 64 lowercase hex chars")
    return value


def _req_int(d: dict, key: str) -> int:
    value = d[key]
    if type(value) is not int:  # rejects bool and float
        raise ManifestError(f"{key!r} must be an int, got {type(value).__name__}")
    return value


def _req_float(d: dict, key: str) -> float:
    value = d[key]
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ManifestError(f"{key!r} must be a number")
    return float(value)


def _is_safe_basename(name: str) -> bool:
    return (
        isinstance(name, str)
        and name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and name == pathlib.PurePosixPath(name).name
    )


@dataclasses.dataclass(frozen=True)
class ChunkEntry:
    span_ordinal: int
    text_sha256: str
    path: str


@dataclasses.dataclass(frozen=True)
class WorkManifest:
    work_id: str
    author_id: str
    provenance_sha256: str
    chunker_config_hash: str
    overlap: float
    chunks: tuple[ChunkEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "author_id": self.author_id,
            "provenance_sha256": self.provenance_sha256,
            "chunker_config_hash": self.chunker_config_hash,
            "overlap": self.overlap,
            "chunks": [dataclasses.asdict(c) for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "WorkManifest":
        if not isinstance(raw, dict):
            raise ManifestError("manifest must be a JSON object")
        allowed = {"work_id", "author_id", "provenance_sha256", "chunker_config_hash", "overlap", "chunks"}
        if set(raw) != allowed:
            raise ManifestError(f"manifest keys must be exactly {sorted(allowed)}, got {sorted(raw)}")
        chunks_raw = raw["chunks"]
        if not isinstance(chunks_raw, list) or not chunks_raw:
            raise ManifestError("manifest 'chunks' must be a non-empty list")
        entries = []
        for c in chunks_raw:
            if not isinstance(c, dict) or set(c) != {"span_ordinal", "text_sha256", "path"}:
                raise ManifestError("each chunk needs exactly span_ordinal, text_sha256, path")
            path = _req_str(c, "path")
            if not _is_safe_basename(path):
                raise ManifestError(f"unsafe chunk path {path!r}")
            entries.append(ChunkEntry(_req_int(c, "span_ordinal"), _req_sha256(c, "text_sha256"), path))
        return cls(
            work_id=_req_str(raw, "work_id"),
            author_id=_req_str(raw, "author_id"),
            provenance_sha256=_req_sha256(raw, "provenance_sha256"),
            chunker_config_hash=_req_sha256(raw, "chunker_config_hash"),
            overlap=_req_float(raw, "overlap"),
            chunks=tuple(entries),
        )


def build_work_manifest(
    work_id: str,
    author_id: str,
    chunk_texts: Sequence[str],
    filenames: Sequence[str],
    *,
    provenance_sha256: str,
    chunker_config_hash: str,
    overlap: float,
) -> WorkManifest:
    """Build a manifest hashing the CANONICAL (stripped) text; rejects empty chunks."""
    entries = []
    for i, (text, name) in enumerate(zip(chunk_texts, filenames, strict=True)):
        canonical = canonical_chunk_text(text)
        if not canonical:
            raise ManifestError(f"{work_id}: chunk {name} is empty after normalization")
        entries.append(ChunkEntry(span_ordinal=i, text_sha256=sha256_text(canonical), path=name))
    return WorkManifest(work_id, author_id, provenance_sha256, chunker_config_hash, float(overlap), tuple(entries))


def chunk_identity(manifest: WorkManifest, entry: ChunkEntry) -> tuple:
    return (
        manifest.work_id,
        manifest.provenance_sha256,
        manifest.chunker_config_hash,
        entry.span_ordinal,
        entry.text_sha256,
    )


def _walk_txt_no_follow(root: pathlib.Path) -> list[pathlib.Path]:
    """All *.txt under root without following symlinked directories (symlink files kept)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            if name.endswith(".txt"):
                found.append(pathlib.Path(dirpath) / name)
    return found


def _require_no_symlink(path: pathlib.Path, what: str) -> None:
    if path.is_symlink():
        raise ManifestError(f"{what} is a symlink (rejected): {path}")


def _resolve_within_no_symlink(root: str | pathlib.Path, *parts: str, what: str) -> pathlib.Path:
    """Return root/parts, rejecting a symlink at ANY component and paths escaping root.

    Checks the whole chain (root, root/part0, …) via ``is_symlink`` (lstat) so a symlinked
    author dir cannot redirect the leaf source file outside the corpus.
    """
    root = pathlib.Path(root)
    if root.is_symlink():
        raise ManifestError(f"{what}: root is a symlink (rejected): {root}")
    cur = root
    for part in parts:
        cur = cur / part
        if cur.is_symlink():
            raise ManifestError(f"{what}: symlinked path component (rejected): {cur}")
    if not cur.resolve().is_relative_to(root.resolve()):
        raise ManifestError(f"{what}: escapes root: {cur}")
    return cur


def validate_work_manifest(
    manifest: WorkManifest,
    work_dir: str | pathlib.Path,
    *,
    author_id: str,
    work_id: str,
    input_clean_root: str | pathlib.Path,
    expected_chunker_config_hash: str | None = None,
) -> list[str]:
    """Full validation; returns the canonical (stripped) chunk texts in manifest order.

    Provenance is mandatory (``input_clean_root`` is required). Raises ``ManifestError`` on
    any discrepancy.
    """
    work_dir = pathlib.Path(work_dir)

    if manifest.author_id != author_id:
        raise ManifestError(f"{work_id}: author_id {manifest.author_id!r} != directory {author_id!r}")
    if manifest.work_id != work_id:
        raise ManifestError(f"{work_id}: work_id {manifest.work_id!r} != directory-derived {work_id!r}")
    if manifest.overlap != 0.0:
        raise ManifestError(f"{work_id}: work_balanced requires overlap==0 (got {manifest.overlap})")
    if expected_chunker_config_hash is not None and manifest.chunker_config_hash != expected_chunker_config_hash:
        raise ManifestError(f"{work_id}: stale manifest (chunker_config_hash mismatch)")

    identities = [chunk_identity(manifest, c) for c in manifest.chunks]
    if len(set(identities)) != len(identities):
        raise ManifestError(f"{work_id}: duplicate canonical chunk identity")
    if [c.span_ordinal for c in manifest.chunks] != list(range(len(manifest.chunks))):
        raise ManifestError(f"{work_id}: span ordinals must be contiguous 0..n-1 in order")

    listed = [c.path for c in manifest.chunks]
    if len(set(listed)) != len(listed):
        raise ManifestError(f"{work_id}: duplicate chunk path in manifest")

    # structure: no symlink .txt anywhere, no nested .txt, exact bijection
    all_txt = _walk_txt_no_follow(work_dir)
    symlinked = [p for p in all_txt if p.is_symlink()]
    if symlinked:
        raise ManifestError(f"{work_id}: symlinked chunk file(s) rejected: {sorted(p.name for p in symlinked)[:3]}")
    nested = [p for p in all_txt if p.parent != work_dir]
    if nested:
        raise ManifestError(f"{work_id}: nested .txt files not allowed: {sorted(p.name for p in nested)[:3]}")
    on_disk = {p.name for p in all_txt if p.parent == work_dir}
    if on_disk != set(listed):
        raise ManifestError(
            f"{work_id}: manifest/file mismatch "
            f"(extra {sorted(on_disk - set(listed))[:3]}, missing {sorted(set(listed) - on_disk)[:3]})"
        )

    # provenance against the cleaned source (mandatory); whole path chain must be symlink-free
    author, book = work_id.split("/", 1)
    src = _resolve_within_no_symlink(input_clean_root, author, f"{book}.txt", what=f"{work_id}: cleaned source")
    if not src.is_file():
        raise ManifestError(f"{work_id}: cleaned source not found for provenance check: {src}")
    if source_provenance_sha256(src) != manifest.provenance_sha256:
        raise ManifestError(f"{work_id}: provenance_sha256 mismatch vs {src}")

    # canonical text: recompute over valid UTF-8, reject empty, hash the model representation
    texts: list[str] = []
    for c in manifest.chunks:
        fp = work_dir / c.path
        _require_no_symlink(fp, f"{work_id}: chunk {c.path}")
        try:
            decoded = fp.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestError(f"{work_id}: chunk {c.path} is not valid UTF-8: {exc}") from exc
        canonical = canonical_chunk_text(decoded)
        if not canonical:
            raise ManifestError(f"{work_id}: chunk {c.path} is empty after normalization")
        if sha256_text(canonical) != c.text_sha256:
            raise ManifestError(f"{work_id}: text_sha256 mismatch for {c.path}")
        texts.append(canonical)
    return texts


def load_work_manifest(
    work_dir: str | pathlib.Path,
    *,
    input_clean_root: str | pathlib.Path,
    expected_chunker_config_hash: str | None = None,
) -> tuple[WorkManifest, list[str]]:
    """Read + validate one work manifest; returns (manifest, canonical texts in order)."""
    work_dir = pathlib.Path(work_dir)
    author_id = work_dir.parent.name
    work_id = f"{author_id}/{work_dir.name}"
    path = work_dir / MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise ManifestError(f"missing/unsafe {MANIFEST_NAME} in {work_dir} (required for work_balanced)")
    manifest = WorkManifest.from_dict(load_strict(path))
    texts = validate_work_manifest(
        manifest, work_dir, author_id=author_id, work_id=work_id,
        input_clean_root=input_clean_root, expected_chunker_config_hash=expected_chunker_config_hash,
    )
    return manifest, texts


def load_work_balanced_dataset(
    frags_root: str | pathlib.Path,
    *,
    cfg,
    input_clean_root: str | pathlib.Path | None = None,
    exclude_authors: Iterable[str] = (),
    unknown_name: str = "unknown",
):
    """The single canonical loader for the ``work_balanced`` path.

    Builds a :class:`corpus.Dataset` from exactly the validated manifest texts (in manifest
    order). Rejects symlinked author/work dirs and paths escaping the corpus root, stray
    .txt directly under an author dir, and enforces global work-id uniqueness and full
    author coverage. ``input_clean_root`` defaults to ``cfg.paths.input_clean`` (provenance
    is mandatory).
    """
    from .corpus import Dataset  # local import to avoid a cycle

    root = pathlib.Path(frags_root)
    if not root.exists():
        raise ManifestError(f"frags root not found: {root}")
    root_resolved = root.resolve()
    if input_clean_root is None:
        input_clean_root = cfg.get_path("paths.input_clean", "input_clean")
    expected_hash = chunker_config_hash(cfg)
    if isinstance(exclude_authors, (str, bytes)):     # a bare string would become a set of letters
        raise ManifestError("exclude_authors must be an iterable of author ids, not a string")
    exclude_authors = tuple(exclude_authors)          # materialize ONCE (a generator is consumed twice)
    excl = set(exclude_authors) | {unknown_name}

    def _real_subdirs(parent: pathlib.Path, what: str) -> list[pathlib.Path]:
        """Real (non-symlink) subdirectories, contained in the corpus root.

        Any symlink entry — dir or file — is rejected outright (not silently skipped),
        so a symlinked author/work dir cannot smuggle in out-of-tree data.
        """
        out = []
        for e in sorted(os.scandir(parent), key=lambda e: e.name):
            if e.is_symlink():
                raise ManifestError(f"symlinked {what} rejected: {e.path}")
            if e.is_dir(follow_symlinks=False):
                p = pathlib.Path(e.path)
                if not p.resolve().is_relative_to(root_resolved):
                    raise ManifestError(f"{what} escapes corpus root: {e.path}")
                out.append(p)
        return out

    authors: list[str] = []
    for adir in _real_subdirs(root, "author dir"):
        if adir.name in excl:
            continue  # exclusion BEFORE any stray/structure check on this author
        stray = [e.name for e in os.scandir(adir) if e.is_file(follow_symlinks=False) and e.name.endswith(".txt")]
        if stray:
            raise ManifestError(f"{adir.name}: stray .txt directly under author dir: {sorted(stray)[:3]}")
        authors.append(adir.name)

    authors = sorted(authors)
    if len(authors) < 2:
        raise ManifestError(f"need >=2 authors, found {authors}")
    auth2idx = {a: i for i, a in enumerate(authors)}

    from .eval.provenance import RowIdentity

    texts: list[str] = []
    y: list[int] = []
    groups: list[str] = []
    seen_work_ids: set[str] = set()
    used_paths: list[str] = []
    row_ids: list = []

    for author in authors:
        adir = root / author
        work_dirs = _real_subdirs(adir, f"work dir under {author}")
        if not work_dirs:
            raise ManifestError(f"author {author!r} has no works (no observations for the class)")
        for wdir in work_dirs:
            work_id = f"{author}/{wdir.name}"
            if work_id in seen_work_ids:
                raise ManifestError(f"duplicate work_id across corpus: {work_id}")
            seen_work_ids.add(work_id)
            manifest, chunk_texts = load_work_manifest(
                wdir, input_clean_root=input_clean_root, expected_chunker_config_hash=expected_hash,
            )
            for entry, text in zip(manifest.chunks, chunk_texts, strict=True):
                texts.append(text)
                y.append(auth2idx[author])
                groups.append(work_id)
                used_paths.append(str(wdir / entry.path))
                row_ids.append(RowIdentity(
                    group=work_id, ordinal=entry.span_ordinal, text_sha256=entry.text_sha256,
                    work_id=work_id, provenance_sha256=manifest.provenance_sha256,
                    chunker_config_hash=manifest.chunker_config_hash,
                ))

    if len(texts) < 10:
        raise ManifestError(f"too few fragments after canonical load: {len(texts)}")

    import numpy as np

    y_arr = np.array(y, dtype=int)
    if set(y_arr.tolist()) != set(range(len(authors))):
        raise ManifestError("every author must contribute at least one observation")

    from .eval.provenance import (WORK_BALANCED_MANIFEST, CorpusPolicyProvenance,
                                  build_provenance)
    manifest_hash = hashlib.sha256(
        b"".join(f"{r.work_id}\x00{r.provenance_sha256}\x00".encode() for r in row_ids)
    ).hexdigest()
    prov = build_provenance(
        loader_kind=WORK_BALANCED_MANIFEST,
        texts=texts, y=y, groups=groups, authors=authors, row_ids=row_ids,
        frags_root=str(root_resolved),
        corpus_policy=CorpusPolicyProvenance.build(exclude_authors, unknown_name),
        chunker_config_hash=expected_hash,
        manifest_hash=manifest_hash,
    )
    dataset = Dataset(
        texts=np.array(texts, dtype=object),
        y=y_arr,
        groups=np.array(groups, dtype=object),
        authors=authors,
        provenance=prov,
    )
    dataset._manifest_paths = tuple(used_paths)  # type: ignore[attr-defined]
    return dataset


def resolve_dataset(
    cfg,
    training_weighting: str,
    frags_root: str | pathlib.Path,
    *,
    exclude_authors: Iterable[str] = (),
    unknown_name: str = "unknown",
):
    """Single dataset dispatcher: canonical loader for ``work_balanced``, legacy otherwise."""
    from .corpus import load_dataset
    from .eval.work_weighting import WORK_BALANCED, resolve_training_weighting

    if resolve_training_weighting(training_weighting) == WORK_BALANCED:
        return load_work_balanced_dataset(
            frags_root, cfg=cfg, exclude_authors=exclude_authors, unknown_name=unknown_name,
        )
    return load_dataset(frags_root, exclude_authors=exclude_authors, unknown_name=unknown_name)
