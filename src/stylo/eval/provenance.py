"""Dataset provenance — the fail-closed contract binding a Dataset to a weighting arm.

A `DatasetProvenance` proves *how* a `Dataset` was built (loader kind, full per-row identity,
corpus policy, chunker/normalization hash) with a **versioned, length-prefixed canonical digest**
over `texts + y + groups + authors + row identities`. Length-prefixing removes every field-
boundary/encoding ambiguity; `authors` is inside the digest because permuting authors reassigns
every `y`. The guard recomputes the digest over the *current* arrays, so a hand-built or mutated
Dataset cannot pose as manifest-validated. See research/work_balanced/model_routing.md §2.
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

from ..domain import corpus_identity as _corpus_identity
from .work_weighting import (CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                             resolve_training_weighting)

DIGEST_VERSION = _corpus_identity.DIGEST_VERSION
LEGACY_RECURSIVE = _corpus_identity.LEGACY_RECURSIVE
WORK_BALANCED_MANIFEST = _corpus_identity.WORK_BALANCED_MANIFEST
ProvenanceError = _corpus_identity.ProvenanceError
CorpusPolicyProvenance = _corpus_identity.CorpusPolicyProvenance
RowIdentity = _corpus_identity.RowIdentity
DataContract = _corpus_identity.DataContract
DatasetProvenance = _corpus_identity.DatasetProvenance

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
    """Raised when model-routing preflight rejects a requested (spec, weighting)."""


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


# Compatibility facade: corpus loaders now depend only on the inward domain
# contract.  Keep the historical evaluation import path, but make every
# exported type/function the exact same object rather than a duplicate class.
DIGEST_VERSION = _corpus_identity.DIGEST_VERSION
LEGACY_RECURSIVE = _corpus_identity.LEGACY_RECURSIVE
WORK_BALANCED_MANIFEST = _corpus_identity.WORK_BALANCED_MANIFEST
ProvenanceError = _corpus_identity.ProvenanceError
CorpusPolicyProvenance = _corpus_identity.CorpusPolicyProvenance
RowIdentity = _corpus_identity.RowIdentity
DataContract = _corpus_identity.DataContract
DatasetProvenance = _corpus_identity.DatasetProvenance
canonical_digest = _corpus_identity.canonical_digest
build_provenance = _corpus_identity.build_provenance
_canonical_ri = _corpus_identity._canonical_ri
_validate_provenance_schema = _corpus_identity._validate_provenance_schema
_author_of_group = _corpus_identity._author_of_group
_validate_semantics = _corpus_identity._validate_semantics
_validate_self_consistency = _corpus_identity._validate_self_consistency
_verify_stored_provenance = _corpus_identity._verify_stored_provenance
_selection_digest = _corpus_identity._selection_digest
_require_subsequence = _corpus_identity._require_subsequence


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
        # A WB subset would need its own disk-derived canonical anchor, which this contract lacks.
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


_BATCH_MANIFEST_SCHEMA = "stylo.atomic-batch-manifest.v1"
_BATCH_POINTER_SCHEMA = "stylo.atomic-batch-pointer.v1"


def _batch_publication_id(names) -> str:
    """Derive a stable namespace from the exact output-name set."""
    from ..jsonio import canonical_hash

    return "batch-" + canonical_hash(sorted(names))[:16]


def _safe_publication_id(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or re.fullmatch(r"[0-9A-Za-z_.-]+", value) is None
    ):
        raise ProvenanceError(f"unsafe batch publication id: {value!r}")
    return value


def _fsync_dir(path) -> None:
    import os as _os

    fd = _os.open(path, _os.O_RDONLY | getattr(_os, "O_DIRECTORY", 0))
    try:
        _os.fsync(fd)
    finally:
        _os.close(fd)


def _remove_private_tree(path, *, _staging_root: bool = True) -> None:
    """Remove only a writer-owned hidden staging tree."""
    import pathlib as _pl

    path = _pl.Path(path)
    if _staging_root and not path.name.startswith(".staging-"):
        raise ProvenanceError(f"refusing to clean non-staging path: {path}")
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_private_tree(child, _staging_root=False)
        else:
            child.unlink()
    path.rmdir()


def _validate_batch_generation(generation, *, publication_id: str) -> dict:
    import hashlib as _hashlib
    import pathlib as _pl

    from ..jsonio import artifact_self_hash, load_strict

    generation = _pl.Path(generation)
    if generation.is_symlink() or not generation.is_dir():
        raise ProvenanceError(f"batch generation is missing/symlinked: {generation}")
    manifest_path = generation / "MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ProvenanceError(f"batch manifest is missing/symlinked: {manifest_path}")
    manifest = load_strict(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _BATCH_MANIFEST_SCHEMA
        or manifest.get("publication_id") != publication_id
        or manifest.get("generation_id") != generation.name
        or manifest.get("self_hash") != artifact_self_hash(manifest)
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise ProvenanceError("batch generation manifest is invalid")
    allowed = {"MANIFEST.json", *manifest["files"]}
    observed = {path.name for path in generation.iterdir()}
    if observed != allowed:
        raise ProvenanceError(
            f"batch generation file inventory mismatch: {sorted(observed)} != "
            f"{sorted(allowed)}"
        )
    for name, binding in manifest["files"].items():
        if (
            "/" in name
            or name in {"", ".", "..", "MANIFEST.json"}
            or not isinstance(binding, dict)
            or set(binding) != {"sha256", "size_bytes"}
            or type(binding["sha256"]) is not str
            or len(binding["sha256"]) != 64
            or type(binding["size_bytes"]) is not int
            or binding["size_bytes"] < 0
        ):
            raise ProvenanceError(f"invalid batch file binding for {name!r}")
        path = generation / name
        if path.is_symlink() or not path.is_file():
            raise ProvenanceError(f"batch member is missing/symlinked: {name!r}")
        payload = path.read_bytes()
        if (
            len(payload) != binding["size_bytes"]
            or _hashlib.sha256(payload).hexdigest() != binding["sha256"]
        ):
            raise ProvenanceError(f"batch member digest mismatch: {name!r}")
    return manifest


def resolve_published_batch(
    dirpath,
    *,
    publication_id: str,
    expected_names=None,
) -> dict:
    """Resolve one complete generation through its single atomic pointer."""
    import pathlib as _pl

    from ..jsonio import artifact_self_hash, load_strict

    publication_id = _safe_publication_id(publication_id)
    root = _pl.Path(dirpath) / ".stylo-batches" / publication_id
    pointer_path = root / "CURRENT.json"
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise ProvenanceError(f"batch pointer is missing/symlinked: {pointer_path}")
    pointer = load_strict(pointer_path)
    if (
        not isinstance(pointer, dict)
        or pointer.get("schema_version") != _BATCH_POINTER_SCHEMA
        or pointer.get("publication_id") != publication_id
        or type(pointer.get("generation_id")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", pointer["generation_id"]) is None
        or pointer.get("self_hash") != artifact_self_hash(pointer)
    ):
        raise ProvenanceError("batch pointer is invalid")
    generation = root / "generations" / pointer["generation_id"]
    manifest = _validate_batch_generation(
        generation,
        publication_id=publication_id,
    )
    if pointer.get("manifest_self_hash") != manifest["self_hash"]:
        raise ProvenanceError("batch pointer/manifest binding mismatch")
    names = set(manifest["files"])
    if expected_names is not None and names != set(expected_names):
        raise ProvenanceError(
            f"resolved batch names {sorted(names)} != expected {sorted(expected_names)}"
        )
    return {name: generation / name for name in sorted(names)}


def safe_write_batch(
    dirpath,
    name_to_text: dict,
    *,
    publication_id: str | None = None,
) -> dict:
    """Publish a complete immutable generation behind one atomic pointer.

    Individual sibling replacement cannot be a filesystem transaction.  This
    API therefore writes and fsyncs a content-addressed generation first, then
    commits it with exactly one atomic ``CURRENT.json`` replacement.  A failure
    or crash before that replacement leaves the prior generation resolvable;
    an orphan generation is harmless and can be reused by a retry.
    """
    import hashlib as _hashlib
    import os as _os
    import pathlib as _pl
    import secrets as _secrets
    import stat as _stat

    from ..jsonio import artifact_self_hash, canonical_hash, dumps_strict

    if not isinstance(name_to_text, dict) or not name_to_text:
        raise ProvenanceError("batch outputs must be a non-empty mapping")
    names = list(name_to_text)
    if len(set(names)) != len(names):
        raise ProvenanceError("batch output names must be unique")
    for name, text in name_to_text.items():
        if (
            type(name) is not str
            or "/" in name
            or name in ("", ".", "..", "MANIFEST.json")
        ):
            raise ProvenanceError(f"unsafe output filename: {name!r}")
        if type(text) is not str:
            raise ProvenanceError(f"batch output {name!r} must be text")
    publication_id = _safe_publication_id(
        publication_id or _batch_publication_id(names)
    )

    dirpath = _pl.Path(dirpath)
    nofollow = getattr(_os, "O_NOFOLLOW", 0)
    dfd = _os.open(
        dirpath,
        _os.O_RDONLY | getattr(_os, "O_DIRECTORY", 0) | nofollow,
    )
    try:
        # Preserve the old symlink/non-regular preflight: a stable-name alias
        # must not coexist with a published generation and mislead consumers.
        for name in names:
            try:
                st = _os.lstat(name, dir_fd=dfd)
                if not _stat.S_ISREG(st.st_mode):
                    raise ProvenanceError(
                        f"refusing batch publication beside symlink/non-regular: {name!r}"
                    )
            except FileNotFoundError:
                pass
    finally:
        _os.close(dfd)

    batch_root = dirpath / ".stylo-batches"
    publication_root = batch_root / publication_id
    generations = publication_root / "generations"
    for path in (batch_root, publication_root, generations):
        existed = path.exists()
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ProvenanceError(f"batch publication path is unsafe: {path}")
        path.mkdir(exist_ok=True)
        if not existed:
            _fsync_dir(path.parent)

    payloads = {
        name: text.encode("utf-8")
        for name, text in sorted(name_to_text.items())
    }
    bindings = {
        name: {
            "sha256": _hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in payloads.items()
    }
    generation_body = {
        "schema_version": _BATCH_MANIFEST_SCHEMA,
        "publication_id": publication_id,
        "files": bindings,
    }
    generation_id = canonical_hash(generation_body)
    manifest = {
        **generation_body,
        "generation_id": generation_id,
    }
    manifest["self_hash"] = artifact_self_hash(manifest)
    generation = generations / generation_id

    if not generation.exists():
        staging = generations / f".staging-{_secrets.token_hex(16)}"
        staging.mkdir(mode=0o700)
        try:
            for name, payload in payloads.items():
                fd = _os.open(
                    staging / name,
                    _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY | nofollow,
                    0o600,
                )
                with _os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    _os.fsync(handle.fileno())
            manifest_payload = (
                dumps_strict(
                    manifest,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            fd = _os.open(
                staging / "MANIFEST.json",
                _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY | nofollow,
                0o600,
            )
            with _os.fdopen(fd, "wb") as handle:
                handle.write(manifest_payload)
                handle.flush()
                _os.fsync(handle.fileno())
            _fsync_dir(staging)
            try:
                _os.rename(staging, generation)
                _fsync_dir(generations)
            except OSError:
                if not generation.is_dir():
                    raise
                _remove_private_tree(staging)
        except BaseException:
            if staging.exists():
                _remove_private_tree(staging)
            raise
    published_manifest = _validate_batch_generation(
        generation,
        publication_id=publication_id,
    )
    if published_manifest != manifest:
        raise ProvenanceError(
            f"content-addressed batch generation conflict: {generation_id}"
        )

    pointer = {
        "schema_version": _BATCH_POINTER_SCHEMA,
        "publication_id": publication_id,
        "generation_id": generation_id,
        "manifest_self_hash": manifest["self_hash"],
    }
    pointer["self_hash"] = artifact_self_hash(pointer)
    pointer_payload = (
        dumps_strict(pointer, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    tmp = publication_root / f".CURRENT.{_secrets.token_hex(16)}.tmp"
    fd = _os.open(
        tmp,
        _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY | nofollow,
        0o600,
    )
    try:
        with _os.fdopen(fd, "wb") as handle:
            handle.write(pointer_payload)
            handle.flush()
            _os.fsync(handle.fileno())
        _os.replace(tmp, publication_root / "CURRENT.json")
        _fsync_dir(publication_root)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return resolve_published_batch(
        dirpath,
        publication_id=publication_id,
        expected_names=names,
    )


# ── headline-output guard ─────────────────────────────────────────────────────
def assert_headline_write_allowed(weighting: str) -> None:
    """Fail-closed: only the legacy arm may write headline artifacts (docs/final_comparison.* …)."""
    if resolve_training_weighting(weighting) != CHUNK_WEIGHTED_LEGACY:
        raise ProvenanceError(
            "refusing to write headline artifacts under a non-legacy weighting; "
            "work_balanced output must go to the exploratory namespace"
        )
