"""B2 dataset provenance — the fail-closed contract binding a Dataset to a weighting arm.

A `DatasetProvenance` proves *how* a `Dataset` was built (loader kind, full per-row identity,
corpus policy, chunker/normalization hash) with a **versioned, length-prefixed canonical digest**
over `texts + y + groups + authors + row identities`. Length-prefixing removes every field-
boundary/encoding ambiguity; `authors` is inside the digest because permuting authors reassigns
every `y`. The guard recomputes the digest over the *current* arrays, so a hand-built or mutated
Dataset cannot pose as manifest-validated (B0 gate). See research/P1_B2_MODEL_WIRING_DESIGN.md §2.
"""
from __future__ import annotations

import collections.abc as cabc
import dataclasses
import enum
import hashlib
import numbers
import re
from typing import Optional, Sequence

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

from .work_weighting import (CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                             resolve_training_weighting)

DIGEST_VERSION = "b2.prov.v2"   # v2: digest now binds loader_kind + chunker_config_hash
LEGACY_RECURSIVE = "legacy_recursive"
WORK_BALANCED_MANIFEST = "work_balanced_manifest"

# loader_kind required by each weighting arm (both-sided guard)
_KIND_FOR_WEIGHTING = {
    CHUNK_WEIGHTED_LEGACY: LEGACY_RECURSIVE,
    WORK_BALANCED: WORK_BALANCED_MANIFEST,
}


class VariantRole(str, enum.Enum):
    """Per-leaderboard-row role. Distinct from the 5-value ClaimStatus (which is NOT extended)."""
    PRIMARY = "primary"
    REFERENCE = "reference"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED_NOT_IMPLEMENTED = "blocked_not_implemented"


class UnsupportedVariantError(RuntimeError):
    """Raised by preflight for a requested (spec, weighting) that B2-core will not run."""


class ProvenanceError(ValueError):
    """Fail-closed dataset/weighting contract violation."""


@dataclasses.dataclass(frozen=True)
class CorpusPolicyProvenance:
    """Immutable corpus policy (NOT a dict) — part of the data contract."""
    exclude_from_benchmark: tuple[str, ...]      # sorted, de-duplicated
    unknown_dir_name: str

    @staticmethod
    def build(exclude_from_benchmark: Sequence[str], unknown_dir_name: str) -> "CorpusPolicyProvenance":
        # strict types: a bare string exclude would silently become a set of letters; a non-str
        # unknown_dir_name would load a bogus author directory.
        if isinstance(exclude_from_benchmark, (str, bytes)):
            raise ProvenanceError("exclude_from_benchmark must be a list of author ids, not a string")
        excl = list(exclude_from_benchmark or ())
        if not all(isinstance(a, str) for a in excl):
            raise ProvenanceError("exclude_from_benchmark entries must all be str")
        if not isinstance(unknown_dir_name, str):
            raise ProvenanceError("unknown_dir_name must be str")
        return CorpusPolicyProvenance(tuple(sorted(set(excl))), unknown_dir_name)


@dataclasses.dataclass(frozen=True)
class RowIdentity:
    """Per-row identity. WB rows carry the full B0 identity chain; legacy rows carry a subset."""
    group: str
    ordinal: int
    text_sha256: str
    work_id: Optional[str] = None
    provenance_sha256: Optional[str] = None
    chunker_config_hash: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DataContract:
    """The data-side contract compared at run entry (NOT the full model-config: ablation-safe)."""
    frags_root: str                              # resolved absolute path string
    corpus_policy: CorpusPolicyProvenance
    chunker_config_hash: Optional[str]           # None for the legacy recursive loader
    canonical_digest: Optional[str] = None       # WB only: digest independently recomputed from disk


@dataclasses.dataclass(frozen=True)
class DatasetProvenance:
    digest_version: str
    loader_kind: str
    row_ids: tuple                               # tuple[RowIdentity]
    authors: tuple                               # tuple[str], index == label
    n_rows: int
    chunker_config_hash: Optional[str]
    corpus_policy: CorpusPolicyProvenance
    frags_root: str
    rows_digest: str
    manifest_hash: Optional[str] = None
    config_id: Optional[str] = None
    parent_rows_digest: Optional[str] = None          # subset only: the disk-anchored parent digest
    selection_manifest_digest: Optional[str] = None   # subset only: sha256 of ordered selected ids

    def contract(self) -> DataContract:
        return DataContract(self.frags_root, self.corpus_policy, self.chunker_config_hash)


# ── canonical digest ──────────────────────────────────────────────────────────
def _lp(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def _s(x) -> bytes:
    return b"" if x is None else str(x).encode("utf-8")


def canonical_digest(texts: Sequence, y: Sequence, groups: Sequence,
                     authors: Sequence, row_ids: Sequence[RowIdentity],
                     *, loader_kind: str, chunker_config_hash: Optional[str]) -> str:
    """Versioned length-prefixed sha256 over loader_kind + chunker hash + texts+y+groups+authors
    + row identities, in order. Binding ``loader_kind``/``chunker_config_hash`` means relabeling a
    Dataset's arm (e.g. legacy→work_balanced) invalidates the digest recompute."""
    n = len(texts)
    if not (len(y) == len(groups) == len(row_ids) == n):
        raise ProvenanceError("texts/y/groups/row_ids length mismatch")
    h = hashlib.sha256()
    h.update(_lp(DIGEST_VERSION.encode()))
    h.update(_lp(_s(loader_kind)))
    h.update(_lp(_s(chunker_config_hash)))
    h.update(n.to_bytes(8, "big"))
    for i in range(n):
        h.update(_lp(_s(texts[i])))
        h.update(int(y[i]).to_bytes(8, "big", signed=True))
        h.update(_lp(_s(groups[i])))
        ri = row_ids[i]
        for part in (ri.group, ri.ordinal, ri.text_sha256, ri.work_id,
                     ri.provenance_sha256, ri.chunker_config_hash):
            h.update(_lp(_s(part)))
    for a in authors:
        h.update(_lp(_s(a)))
    return h.hexdigest()


def _str_or_none(x, name: str):
    if x is not None and type(x) is not str:
        raise ProvenanceError(f"{name} must be an exact str or None")
    return x


def _canonical_ri(ri) -> tuple:
    """Exact-type canonical identity tuple — rejects RowIdentity subclasses / non-str/int fields /
    fields with overridden equality, so comparisons are never on polymorphic objects."""
    if type(ri) is not RowIdentity:
        raise ProvenanceError("row identity must be exactly RowIdentity")
    if type(ri.group) is not str or type(ri.ordinal) is not int or type(ri.text_sha256) is not str:
        raise ProvenanceError("row identity group/ordinal/text_sha256 have wrong exact types")
    return (ri.group, ri.ordinal, ri.text_sha256,
            _str_or_none(ri.work_id, "work_id"),
            _str_or_none(ri.provenance_sha256, "provenance_sha256"),
            _str_or_none(ri.chunker_config_hash, "chunker_config_hash"))


def _validate_provenance_schema(prov) -> None:
    """Single exact-type schema gate for a DatasetProvenance (no dataclass subclasses / poly fields)."""
    if type(prov) is not DatasetProvenance:
        raise ProvenanceError("provenance must be exactly DatasetProvenance")
    if type(prov.digest_version) is not str or prov.digest_version != DIGEST_VERSION:
        raise ProvenanceError("bad digest_version")
    if prov.loader_kind not in (LEGACY_RECURSIVE, WORK_BALANCED_MANIFEST):
        raise ProvenanceError("bad loader_kind")
    if type(prov.rows_digest) is not str or not _HEX64.fullmatch(prov.rows_digest):
        raise ProvenanceError("rows_digest must be an exact sha256 hex string")
    if type(prov.n_rows) is not int or type(prov.frags_root) is not str:
        raise ProvenanceError("n_rows/frags_root wrong exact types")
    if type(prov.authors) is not tuple or not all(type(a) is str for a in prov.authors):
        raise ProvenanceError("authors must be a tuple of exact str")
    if type(prov.corpus_policy) is not CorpusPolicyProvenance:
        raise ProvenanceError("corpus_policy must be exactly CorpusPolicyProvenance")
    if (type(prov.corpus_policy.exclude_from_benchmark) is not tuple
            or not all(type(a) is str for a in prov.corpus_policy.exclude_from_benchmark)
            or type(prov.corpus_policy.unknown_dir_name) is not str):
        raise ProvenanceError("corpus_policy fields wrong exact types")
    if type(prov.row_ids) is not tuple or len(prov.row_ids) != prov.n_rows:
        raise ProvenanceError("row_ids must be a tuple of length n_rows")
    for ri in prov.row_ids:
        _canonical_ri(ri)                         # validates each identity's exact types
    for f, name in ((prov.parent_rows_digest, "parent_rows_digest"),
                    (prov.selection_manifest_digest, "selection_manifest_digest"),
                    (prov.chunker_config_hash, "chunker_config_hash"),
                    (prov.manifest_hash, "manifest_hash"), (prov.config_id, "config_id")):
        _str_or_none(f, name)


def _author_of_group(group: str) -> str:
    if type(group) is not str:
        raise ProvenanceError("group must be an exact str")
    return group.split("/", 1)[0]


def _as_label(lbl, i: int) -> int:
    """Exact non-bool integral label — no float/bool coercion (0.5/False must not become 0)."""
    if isinstance(lbl, bool) or not isinstance(lbl, numbers.Integral):
        raise ProvenanceError(f"y[{i}]={lbl!r} must be a non-bool integer label")
    return int(lbl)


def _validate_semantics(y: Sequence, groups: Sequence, authors: Sequence) -> None:
    n_auth = len(authors)
    if len(set(authors)) != n_auth:
        raise ProvenanceError("authors must be unique")
    for i, (lbl, g) in enumerate(zip(y, groups)):
        li = _as_label(lbl, i)
        if not (0 <= li < n_auth):
            raise ProvenanceError(f"y[{i}]={li} out of range [0,{n_auth})")
        if authors[li] != _author_of_group(g):
            raise ProvenanceError(
                f"row {i}: authors[y]={authors[li]!r} != author of groups[i]={_author_of_group(g)!r}"
            )


def _validate_self_consistency(texts, y, groups, authors, row_ids,
                               loader_kind, chunker_config_hash) -> None:
    """The single self-consistency gate: identity fields must bind to the actual arrays.

    Beyond semantics + the structural identity gate, this ties each row's ``text_sha256`` to the
    real text (blocks laundering a mutated parent through ``derive_dataset``), its ``group`` to
    ``groups[i]``, its ``work_id`` to the group, and its ``chunker_config_hash`` to the dataset's.
    """
    n = len(texts)
    if not (len(y) == len(groups) == len(row_ids) == n):
        raise ProvenanceError("texts/y/groups/row_ids length mismatch")
    _validate_semantics(y, groups, authors)
    _require_identity_for_kind(loader_kind, row_ids)
    if not all(type(a) is str for a in authors):     # exact str: a subclass could hash/vectorize oddly
        raise ProvenanceError("authors must all be exactly str")
    for i, ri in enumerate(row_ids):
        if type(texts[i]) is not str:
            raise ProvenanceError(f"row {i}: text must be exactly str (no int/subclass coercion)")
        if type(groups[i]) is not str:
            raise ProvenanceError(f"row {i}: group must be exactly str")
        if isinstance(ri.ordinal, bool) or not isinstance(ri.ordinal, numbers.Integral) or ri.ordinal < 0:
            raise ProvenanceError(f"row {i}: span ordinal must be a non-negative int")
        if type(ri.text_sha256) is not str or not _HEX64.match(ri.text_sha256):    # exact-type, not str()
            raise ProvenanceError(f"row {i}: text_sha256 is not a sha256 hex digest")
        if hashlib.sha256(texts[i].encode("utf-8")).hexdigest() != ri.text_sha256:
            raise ProvenanceError(f"row {i}: text_sha256 does not match the actual text")
        if type(ri.group) is not str or ri.group != groups[i]:
            raise ProvenanceError(f"row {i}: row_id.group != groups[i]")
        if loader_kind == WORK_BALANCED_MANIFEST:
            if type(ri.work_id) is not str or ri.work_id != groups[i]:
                raise ProvenanceError(f"row {i}: work_id != group")
            if type(ri.provenance_sha256) is not str or not _HEX64.match(ri.provenance_sha256):
                raise ProvenanceError(f"row {i}: provenance_sha256 is not a sha256 hex digest")
            if type(ri.chunker_config_hash) is not str or ri.chunker_config_hash != chunker_config_hash:
                raise ProvenanceError(f"row {i}: row chunker hash != dataset chunker hash")


def _require_identity_for_kind(loader_kind: str, row_ids: Sequence[RowIdentity]) -> None:
    """Structural gate: WB rows must carry full manifest identity; legacy rows must not claim it.

    Blocks forging a work_balanced provenance over raw legacy arrays (which have no manifest
    identity) and forging legacy rows that carry fake manifest fields.
    """
    if loader_kind == WORK_BALANCED_MANIFEST:
        for ri in row_ids:
            if ri.work_id is None or ri.provenance_sha256 is None or ri.chunker_config_hash is None:
                raise ProvenanceError(
                    "work_balanced provenance requires per-row manifest identity "
                    "(work_id/provenance_sha256/chunker_config_hash)"
                )
    else:  # legacy_recursive rows must not masquerade as manifest-validated
        for ri in row_ids:
            if ri.work_id is not None or ri.provenance_sha256 is not None or ri.chunker_config_hash is not None:
                raise ProvenanceError("legacy_recursive rows must not carry manifest identity")


def _verify_stored_provenance(texts, y, groups, authors, prov: "DatasetProvenance") -> None:
    """Full re-verification of a STORED provenance against the current arrays (guard + subset)."""
    _validate_provenance_schema(prov)             # exact-type schema BEFORE any comparison
    if prov.n_rows != len(texts):
        raise ProvenanceError("row count changed since load (mutated Dataset)")
    if tuple(authors) != tuple(prov.authors):
        raise ProvenanceError("authors changed since load")
    _validate_self_consistency(texts, y, groups, authors, prov.row_ids,
                               prov.loader_kind, prov.chunker_config_hash)
    recomputed = canonical_digest(texts, y, groups, authors, prov.row_ids,
                                  loader_kind=prov.loader_kind,
                                  chunker_config_hash=prov.chunker_config_hash)
    if recomputed != prov.rows_digest:
        raise ProvenanceError("rows_digest mismatch — Dataset was mutated, relabeled or forged")


def _selection_digest(row_ids: Sequence[RowIdentity]) -> str:
    """sha256 over the ordered canonical row identities — the subset's selection manifest."""
    h = hashlib.sha256()
    h.update(_lp(b"selection.v1"))
    for ri in row_ids:
        for part in _canonical_ri(ri):
            h.update(_lp(_s(part)))
    return h.hexdigest()


def _require_subsequence(child: Sequence[RowIdentity], parent: Sequence[RowIdentity]) -> None:
    """Each child row identity must appear in the parent, order-preserving (a true selection).

    Comparison is on canonical exact-type tuples — never a polymorphic RowIdentity ``==``."""
    pit = (_canonical_ri(p) for p in parent)
    for c in child:
        target = _canonical_ri(c)
        for p in pit:
            if p == target:
                break
        else:
            raise ProvenanceError("subset rows are not an ordered subsequence of the disk parent")


def build_provenance(
    *,
    loader_kind: str,
    texts: Sequence,
    y: Sequence,
    groups: Sequence,
    authors: Sequence,
    row_ids: Sequence[RowIdentity],
    frags_root: str,
    corpus_policy: CorpusPolicyProvenance,
    chunker_config_hash: Optional[str] = None,
    manifest_hash: Optional[str] = None,
    config_id: Optional[str] = None,
    parent_rows_digest: Optional[str] = None,
    selection_manifest_digest: Optional[str] = None,
) -> DatasetProvenance:
    if loader_kind not in (LEGACY_RECURSIVE, WORK_BALANCED_MANIFEST):
        raise ProvenanceError(f"unknown loader_kind {loader_kind!r}")
    _validate_self_consistency(texts, y, groups, authors, row_ids, loader_kind, chunker_config_hash)
    digest = canonical_digest(texts, y, groups, authors, row_ids,
                              loader_kind=loader_kind, chunker_config_hash=chunker_config_hash)
    return DatasetProvenance(
        digest_version=DIGEST_VERSION,
        loader_kind=loader_kind,
        row_ids=tuple(row_ids),
        authors=tuple(authors),
        n_rows=len(texts),
        chunker_config_hash=chunker_config_hash,
        corpus_policy=corpus_policy,
        frags_root=str(frags_root),
        rows_digest=digest,
        manifest_hash=manifest_hash,
        config_id=config_id,
        parent_rows_digest=parent_rows_digest,
        selection_manifest_digest=selection_manifest_digest,
    )


# NOTE: the caller-supplied-contract guard was removed — a Dataset's provenance is authoritative
# ONLY after verify_dataset_against_disk re-derives the anchor from the actual on-disk corpus
# (see below). No self-anchored DataContract is ever trusted by the runtime engines.


@dataclasses.dataclass(frozen=True)
class RunContract:
    """The FROZEN run-config identity of the corpus to verify against (never self-declared)."""
    frags_root: str                          # resolved absolute path
    corpus_policy: CorpusPolicyProvenance

    @staticmethod
    def build(frags_root, exclude, unknown) -> "RunContract":
        import pathlib as _pl
        return RunContract(str(_pl.Path(frags_root).resolve()),
                           CorpusPolicyProvenance.build(exclude, unknown))


def _load_frozen_corpus(cfg, contract: "RunContract", weighting: str):
    """Load the corpus named by the FROZEN run-contract (not by the passed Dataset) from disk."""
    excl = list(contract.corpus_policy.exclude_from_benchmark)
    unknown = contract.corpus_policy.unknown_dir_name
    try:
        if weighting == WORK_BALANCED:
            from ..workdoc import load_work_balanced_dataset
            return load_work_balanced_dataset(contract.frags_root, cfg=cfg, exclude_authors=excl,
                                              unknown_name=unknown)
        from ..corpus import load_dataset
        return load_dataset(contract.frags_root, exclude_authors=excl, unknown_name=unknown)
    except Exception as exc:
        raise ProvenanceError(f"could not load the frozen-contract corpus from disk: {exc}") from exc


def verify_dataset_against_disk(cfg, dataset, weighting: str, contract: "RunContract") -> str:
    """The authoritative run-entry gate: bind a Dataset to the weighting arm AND to the corpus the
    FROZEN run-contract names on disk. The dataset's declared (root, policy) must equal the frozen
    contract BEFORE any load; the anchor is then re-derived by loading the frozen corpus (both
    arms), so neither a self-declared corpus redirect nor a self-anchored digest is trusted. A
    derived subset (RuAA) is verified by chaining ``parent_rows_digest`` to the disk parent + an
    ordered ``selection_manifest_digest``."""
    weighting = resolve_training_weighting(weighting)
    prov: Optional[DatasetProvenance] = getattr(dataset, "provenance", None)
    if not isinstance(prov, DatasetProvenance):
        raise ProvenanceError("Dataset has no DatasetProvenance; refusing (hand-built Dataset?)")
    if prov.loader_kind != _KIND_FOR_WEIGHTING[weighting]:
        raise ProvenanceError(
            f"{weighting} requires loader_kind={_KIND_FOR_WEIGHTING[weighting]!r}, got {prov.loader_kind!r}")
    # the dataset's declared contract MUST equal the frozen run-contract, checked BEFORE any load
    if prov.frags_root != contract.frags_root:
        raise ProvenanceError(f"frags_root {prov.frags_root!r} != frozen contract {contract.frags_root!r}")
    if prov.corpus_policy != contract.corpus_policy:
        raise ProvenanceError("corpus policy != frozen run-contract (exclude/unknown differ)")
    _verify_stored_provenance(dataset.texts, dataset.y, dataset.groups, dataset.authors, prov)
    disk = _load_frozen_corpus(cfg, contract, weighting).provenance   # anchor = frozen corpus on disk
    if prov.selection_manifest_digest is not None:              # a derived subset (RuAA)
        if prov.parent_rows_digest != disk.rows_digest:
            raise ProvenanceError("subset parent_rows_digest != disk parent digest")
        if _selection_digest(prov.row_ids) != prov.selection_manifest_digest:
            raise ProvenanceError("subset selection_manifest_digest mismatch")
        _require_subsequence(prov.row_ids, disk.row_ids)
    else:                                                       # a full dataset
        if prov.rows_digest != disk.rows_digest:
            raise ProvenanceError("dataset digest != frozen on-disk digest (fabricated/non-disk)")
    return weighting


# ── atomic subset derivation ──────────────────────────────────────────────────
def derive_dataset(parent, indices: Sequence[int]):
    """The ONLY provenance-preserving subset builder (replaces manual Dataset construction).

    Rows are selected only from a parent that already carries valid provenance; indices are
    validated (unique, in range); authors/y are recomputed (present-only, relabelled) and the
    provenance digest is recomputed. Returns a Dataset with a fresh DatasetProvenance.
    """
    import numpy as np

    from ..corpus import Dataset

    prov: Optional[DatasetProvenance] = getattr(parent, "provenance", None)
    if not isinstance(prov, DatasetProvenance):
        raise ProvenanceError("derive_dataset requires a parent with DatasetProvenance")
    if prov.loader_kind == WORK_BALANCED_MANIFEST:
        # a WB subset would need its own disk-derived canonical anchor — not supported in B2-core
        raise ProvenanceError("work_balanced subsetting is not supported (needs a disk-derived anchor)")
    # FULL stored-provenance re-verification of the parent (not just self-consistency) — a mutated
    # parent with a stale stored digest must not launder a child.
    _verify_stored_provenance(parent.texts, parent.y, parent.groups, parent.authors, prov)
    n = len(parent.texts)
    if isinstance(indices, (str, bytes)) or not isinstance(indices, (cabc.Sequence, np.ndarray)):
        raise ProvenanceError("indices must be an ordered 1-D sequence (no set/generator/mapping)")
    if isinstance(indices, np.ndarray) and indices.ndim != 1:
        raise ProvenanceError("indices must be 1-D")
    idx = []
    for k in indices:
        if isinstance(k, bool) or not isinstance(k, numbers.Integral):
            raise ProvenanceError(f"index {k!r} must be a non-bool integer")
        idx.append(int(k))
    if any(i < 0 or i >= n for i in idx):
        raise ProvenanceError("subset index out of range")
    if len(set(idx)) != len(idx):
        raise ProvenanceError("subset indices must be unique")

    sub_texts = [parent.texts[i] for i in idx]
    sub_groups = [str(parent.groups[i]) for i in idx]
    sub_row_ids = [prov.row_ids[i] for i in idx]
    present = sorted({_author_of_group(g) for g in sub_groups})
    a2i = {a: i for i, a in enumerate(present)}
    sub_y = [a2i[_author_of_group(g)] for g in sub_groups]

    child_prov = build_provenance(
        loader_kind=prov.loader_kind,
        texts=sub_texts, y=sub_y, groups=sub_groups, authors=present, row_ids=sub_row_ids,
        frags_root=prov.frags_root, corpus_policy=prov.corpus_policy,
        chunker_config_hash=prov.chunker_config_hash,
        manifest_hash=prov.manifest_hash, config_id=prov.config_id,
        parent_rows_digest=prov.rows_digest,                     # chain to the disk-anchored parent
        selection_manifest_digest=_selection_digest(sub_row_ids),
    )
    ds = Dataset(
        texts=np.array(sub_texts, dtype=object),
        y=np.array(sub_y, dtype=int),
        groups=np.array(sub_groups, dtype=object),
        authors=present,
    )
    ds.provenance = child_prov  # type: ignore[attr-defined]
    return ds


def safe_exploratory_dir(base, *subparts):
    """Create + return base/<subparts> for exploratory output, fail-closed against symlink escape.

    Every path component must be a real (non-symlink) directory; the result must resolve strictly
    inside ``base`` and never equal ``base`` — so a ``work_balanced -> docs`` symlink cannot make a
    work_balanced run overwrite the legacy headline files.
    """
    import pathlib as _pl
    base = _pl.Path(base)
    if base.is_symlink():
        raise ProvenanceError(f"exploratory base is a symlink: {base}")
    base.mkdir(parents=True, exist_ok=True)
    cur = base
    for part in subparts:
        cur = cur / part
        if cur.is_symlink():
            raise ProvenanceError(f"symlink in exploratory output path: {cur}")
        cur.mkdir(exist_ok=True)
    rp = cur.resolve()
    if rp == base.resolve() or not rp.is_relative_to(base.resolve()):
        raise ProvenanceError("exploratory output dir escapes or equals the base root")
    return cur


def safe_write_text(path, text: str) -> None:
    """TOCTOU-safe atomic write that NEVER follows a symlink (a WB output file symlinked onto a
    legacy headline file must not be overwritten). All operations are relative to a directory fd
    opened with ``O_NOFOLLOW`` (so the parent cannot be a symlink); the temp is created with
    ``O_CREAT|O_EXCL|O_NOFOLLOW`` (race-safe, no ``mktemp``); the target is ``lstat``-checked and
    ``os.replace``'d relative to that fd."""
    import os as _os
    import pathlib as _pl
    import secrets as _secrets
    import stat as _stat
    path = _pl.Path(path)
    name = path.name
    if "/" in name or name in ("", ".", ".."):
        raise ProvenanceError(f"unsafe output filename: {name!r}")
    nofollow = getattr(_os, "O_NOFOLLOW", 0)
    dfd = _os.open(path.parent, _os.O_RDONLY | _os.O_DIRECTORY | nofollow)   # parent must not be a symlink
    try:
        try:
            st = _os.lstat(name, dir_fd=dfd)
            if not _stat.S_ISREG(st.st_mode):        # existing target must be a plain file, not a symlink
                raise ProvenanceError(f"refusing to overwrite a symlink/non-regular output: {name!r}")
        except FileNotFoundError:
            pass
        tmpname = None
        for _ in range(64):
            cand = f".w_{_secrets.token_hex(8)}"
            try:
                fd = _os.open(cand, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY | nofollow, 0o600, dir_fd=dfd)
                tmpname = cand
                break
            except FileExistsError:
                continue
        if tmpname is None:
            raise ProvenanceError("could not create a unique temp output file")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                _os.fsync(fh.fileno())
            _os.replace(tmpname, name, src_dir_fd=dfd, dst_dir_fd=dfd)   # atomic, relative to dir fd
        except BaseException:
            try:
                _os.unlink(tmpname, dir_fd=dfd)
            except OSError:
                pass
            raise
    finally:
        _os.close(dfd)


def safe_write_batch(dirpath, name_to_text: dict) -> None:
    """All-or-nothing generation of a set of outputs in one dir: every temp is written first (a
    write failure aborts before ANY output is replaced), then all are atomically swapped in — so a
    failure on the second file never leaves a new file beside stale siblings."""
    import os as _os
    import pathlib as _pl
    import secrets as _secrets
    import stat as _stat
    dirpath = _pl.Path(dirpath)
    nofollow = getattr(_os, "O_NOFOLLOW", 0)
    dfd = _os.open(dirpath, _os.O_RDONLY | _os.O_DIRECTORY | nofollow)
    tmps: list[tuple[str, str]] = []          # (tmpname, finalname)
    try:
        for name, text in name_to_text.items():
            if "/" in name or name in ("", ".", ".."):
                raise ProvenanceError(f"unsafe output filename: {name!r}")
            try:
                st = _os.lstat(name, dir_fd=dfd)
                if not _stat.S_ISREG(st.st_mode):
                    raise ProvenanceError(f"refusing to overwrite symlink/non-regular: {name!r}")
            except FileNotFoundError:
                pass
            fd = None
            for _ in range(64):
                cand = f".w_{_secrets.token_hex(8)}"
                try:
                    fd = _os.open(cand, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY | nofollow, 0o600, dir_fd=dfd)
                    break
                except FileExistsError:
                    continue
            if fd is None:
                raise ProvenanceError("could not create a unique temp output file")
            with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                _os.fsync(fh.fileno())
            tmps.append((cand, name))
        for tmpname, name in tmps:            # phase 2: swap all in (all temps already written)
            _os.replace(tmpname, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        tmps = []
    finally:
        for tmpname, _ in tmps:               # a failure before phase 2 completes -> clean up temps
            try:
                _os.unlink(tmpname, dir_fd=dfd)
            except OSError:
                pass
        _os.close(dfd)


# ── headline-output guard ─────────────────────────────────────────────────────
def assert_headline_write_allowed(weighting: str) -> None:
    """Fail-closed: only the legacy arm may write headline artifacts (docs/final_comparison.* …)."""
    if resolve_training_weighting(weighting) != CHUNK_WEIGHTED_LEGACY:
        raise ProvenanceError(
            "refusing to write headline artifacts under a non-legacy weighting; "
            "work_balanced output must go to the exploratory namespace"
        )
