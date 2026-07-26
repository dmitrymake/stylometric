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
import threading
import weakref
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


SCIENTIFIC_ISOLATION_CONTRACT_VERSION = (
    "stylo.cross-work-content-isolation.word5.v1"
)
_SCIENTIFIC_CONTEXT_SEAL = object()


class _DiskVerificationAuthority:
    """Identity-only receipt minted after an actual on-disk corpus comparison."""


@dataclasses.dataclass(frozen=True, init=False, eq=False)
class ScientificEvaluationContext:
    """Read-only, dataset-bound authority required by scientific CV kernels.

    The constructor is intentionally application-internal. Contexts are made
    only by the preparation functions below, after the relevant content gate
    has passed. Exact-type checks at raw kernels prevent a bare ``Dataset``
    from accidentally bypassing that gate.
    """

    texts: object
    y: object
    groups: object
    authors: tuple[str, ...]
    provenance: object
    weighting: str
    rows_digest: str
    isolation_receipt_sha256: str
    isolation_contract_version: str
    disk_verified: bool
    _seal: object = dataclasses.field(repr=False, compare=False)
    _disk_authority: object = dataclasses.field(repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_authors(self) -> int:
        return len(self.authors)

    def book_to_author(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for group, label in zip(self.groups, self.y, strict=True):
            out.setdefault(str(group), int(label))
        return out

    def __reduce__(self):
        # NumPy's default pickle restores arrays as writeable. Route every
        # process boundary through the validating restore helper so joblib
        # workers retain the sealed read-only contract.
        return (
            _restore_scientific_evaluation_context,
            (
                self.texts,
                self.y,
                self.groups,
                self.authors,
                self.provenance,
                self.weighting,
                self.rows_digest,
                self.isolation_receipt_sha256,
                self.isolation_contract_version,
            ),
        )


_SCIENTIFIC_CONTEXT_REGISTRY: weakref.WeakKeyDictionary = (
    weakref.WeakKeyDictionary()
)
_SCIENTIFIC_CONTEXT_REGISTRY_LOCK = threading.RLock()
_DISK_AUTHORITY_REGISTRY: weakref.WeakKeyDictionary = (
    weakref.WeakKeyDictionary()
)
_VERIFIED_DATASET_REGISTRY: dict[int, tuple] = {}
_WORKER_REVERIFIED_CONTEXTS: dict[tuple, ScientificEvaluationContext] = {}


def _restore_scientific_evaluation_context(
    texts,
    y,
    groups,
    authors,
    provenance,
    weighting,
    rows_digest,
    isolation_receipt_sha256,
    isolation_contract_version,
) -> ScientificEvaluationContext:
    # A process boundary never trusts the serialized seal.  Rebuild through
    # structural/provenance/content validation as a synthetic transport
    # context. Production workers must independently re-verify it against disk
    # with their run config before invoking a production evaluator.
    return _build_scientific_evaluation_context(
        texts=texts,
        y=y,
        groups=groups,
        authors=authors,
        provenance=provenance,
        weighting=weighting,
        rows_digest=rows_digest,
        isolation_receipt_sha256=isolation_receipt_sha256,
        isolation_contract_version=isolation_contract_version,
        disk_authority=None,
    )


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
    authors = tuple(dataset.authors)
    receipt = _scientific_rows_receipt(
        dataset.texts,
        dataset.y,
        dataset.groups,
        authors,
    )
    fingerprint = _scientific_payload_fingerprint(
        texts=dataset.texts,
        y=dataset.y,
        groups=dataset.groups,
        authors=authors,
        provenance=prov,
        weighting=weighting,
        rows_digest=prov.rows_digest,
        isolation_receipt_sha256=receipt,
        isolation_contract_version=SCIENTIFIC_ISOLATION_CONTRACT_VERSION,
    )
    authority = _DiskVerificationAuthority()
    dataset_key = id(dataset)

    def cleanup(reference, *, key=dataset_key):
        with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
            current = _VERIFIED_DATASET_REGISTRY.get(key)
            if current is not None and current[0] is reference:
                _VERIFIED_DATASET_REGISTRY.pop(key, None)

    reference = weakref.ref(dataset, cleanup)
    with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
        _DISK_AUTHORITY_REGISTRY[authority] = fingerprint
        _VERIFIED_DATASET_REGISTRY[dataset_key] = (
            reference,
            fingerprint,
            authority,
        )
    return weighting


def _scientific_rows_receipt(texts, y, groups, authors) -> str:
    """Path-free receipt for the exact frozen arrays authorized by the gate."""

    digest = hashlib.sha256()
    digest.update(b"stylo.scientific-evaluation-rows.v1")
    for values in (texts, y, groups, authors):
        digest.update(len(values).to_bytes(8, "big"))
        for value in values:
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _provenance_fingerprint(provenance) -> tuple | None:
    """Immutable value snapshot of every provenance/audit field."""

    if provenance is None:
        return None
    _validate_provenance_schema(provenance)
    return (
        provenance.digest_version,
        provenance.loader_kind,
        tuple(_canonical_ri(row_id) for row_id in provenance.row_ids),
        provenance.authors,
        provenance.n_rows,
        provenance.chunker_config_hash,
        (
            provenance.corpus_policy.exclude_from_benchmark,
            provenance.corpus_policy.unknown_dir_name,
        ),
        provenance.frags_root,
        provenance.rows_digest,
        provenance.manifest_hash,
        provenance.config_id,
        provenance.parent_rows_digest,
        provenance.selection_manifest_digest,
    )


def _scientific_payload_fingerprint(
    *,
    texts,
    y,
    groups,
    authors,
    provenance,
    weighting,
    rows_digest,
    isolation_receipt_sha256,
    isolation_contract_version,
) -> tuple:
    return (
        _scientific_rows_receipt(texts, y, groups, authors),
        tuple(authors),
        weighting,
        rows_digest,
        isolation_receipt_sha256,
        isolation_contract_version,
        _provenance_fingerprint(provenance),
    )


def _require_disk_authority_for_dataset(
    dataset,
    weighting: str,
) -> _DiskVerificationAuthority:
    provenance = getattr(dataset, "provenance", None)
    authors = tuple(dataset.authors)
    rows_digest = getattr(provenance, "rows_digest", None)
    receipt = _scientific_rows_receipt(
        dataset.texts,
        dataset.y,
        dataset.groups,
        authors,
    )
    current = _scientific_payload_fingerprint(
        texts=dataset.texts,
        y=dataset.y,
        groups=dataset.groups,
        authors=authors,
        provenance=provenance,
        weighting=weighting,
        rows_digest=rows_digest,
        isolation_receipt_sha256=receipt,
        isolation_contract_version=SCIENTIFIC_ISOLATION_CONTRACT_VERSION,
    )
    with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
        record = _VERIFIED_DATASET_REGISTRY.get(id(dataset))
        if (
            record is None
            or record[0]() is not dataset
            or record[1] != current
            or _DISK_AUTHORITY_REGISTRY.get(record[2]) != current
        ):
            raise ProvenanceError(
                "disk-verified context requires a current registered "
                "verify_dataset_against_disk receipt"
            )
        return record[2]


def _scientific_context_fingerprint(
    context: ScientificEvaluationContext,
) -> tuple:
    """Return the immutable state registered for one in-process capability."""

    return (
        _scientific_payload_fingerprint(
            texts=context.texts,
            y=context.y,
            groups=context.groups,
            authors=context.authors,
            provenance=context.provenance,
            weighting=context.weighting,
            rows_digest=context.rows_digest,
            isolation_receipt_sha256=context.isolation_receipt_sha256,
            isolation_contract_version=context.isolation_contract_version,
        ),
        context.disk_verified,
        id(context._disk_authority),
    )


def _build_scientific_evaluation_context(
    *,
    texts,
    y,
    groups,
    authors,
    provenance,
    weighting,
    rows_digest,
    isolation_receipt_sha256,
    isolation_contract_version,
    disk_authority,
) -> ScientificEvaluationContext:
    """Validate, content-gate, freeze and register one context capability.

    This is also the pickle restore boundary.  Serialized seals are never
    trusted: every restored payload passes the exact content-isolation gate
    again before it becomes an accepted in-process capability.
    """

    import numpy as np

    if type(authors) is not tuple or not all(
        type(author) is str for author in authors
    ):
        raise ProvenanceError("scientific context authors must be a tuple of exact str")
    if type(weighting) is not str:
        raise ProvenanceError("scientific context weighting must be an exact str")
    weighting = resolve_training_weighting(weighting)
    if (
        type(rows_digest) is not str
        or not _HEX64.fullmatch(rows_digest)
    ):
        raise ProvenanceError("scientific context rows_digest must be a sha256")
    if (
        type(isolation_receipt_sha256) is not str
        or not _HEX64.fullmatch(isolation_receipt_sha256)
    ):
        raise ProvenanceError(
            "scientific context isolation receipt must be a sha256"
        )
    if (
        type(isolation_contract_version) is not str
        or isolation_contract_version
        != SCIENTIFIC_ISOLATION_CONTRACT_VERSION
    ):
        raise ProvenanceError("scientific context isolation contract mismatch")
    if (
        disk_authority is not None
        and type(disk_authority) is not _DiskVerificationAuthority
    ):
        raise ProvenanceError("scientific context disk authority is malformed")

    raw_arrays = tuple(np.asarray(values) for values in (texts, y, groups))
    if any(values.ndim != 1 for values in raw_arrays):
        raise ProvenanceError("scientific context arrays must be one-dimensional")
    raw_texts, raw_y, raw_groups = raw_arrays
    if not all(type(value) is str for value in raw_texts):
        raise ProvenanceError("scientific context texts must contain exact str")
    if not all(type(value) is str for value in raw_groups):
        raise ProvenanceError("scientific context groups must contain exact str")
    normalized_y = [
        _as_label(value, index) for index, value in enumerate(raw_y)
    ]
    frozen_arrays = (
        np.array(raw_texts, dtype=object, copy=True),
        np.array(normalized_y, dtype=int, copy=True),
        np.array(raw_groups, dtype=object, copy=True),
    )
    if not (
        len(frozen_arrays[0])
        == len(frozen_arrays[1])
        == len(frozen_arrays[2])
    ):
        raise ProvenanceError("scientific context length mismatch")
    _validate_semantics(frozen_arrays[1], frozen_arrays[2], authors)

    receipt = _scientific_rows_receipt(
        frozen_arrays[0],
        frozen_arrays[1],
        frozen_arrays[2],
        authors,
    )
    if receipt != isolation_receipt_sha256:
        raise ProvenanceError("scientific context isolation receipt mismatch")

    _corpus_identity.assert_cross_work_content_isolation(
        frozen_arrays[0],
        frozen_arrays[2],
    )
    if provenance is None:
        if disk_authority is not None:
            raise ProvenanceError(
                "disk-verified scientific context requires exact provenance"
            )
        if rows_digest != receipt:
            raise ProvenanceError(
                "synthetic scientific context rows_digest mismatch"
            )
    else:
        if type(provenance) is not DatasetProvenance:
            raise ProvenanceError(
                "scientific context provenance must be exactly DatasetProvenance"
            )
        _verify_stored_provenance(
            frozen_arrays[0],
            frozen_arrays[1],
            frozen_arrays[2],
            authors,
            provenance,
        )
        if rows_digest != provenance.rows_digest:
            raise ProvenanceError(
                "scientific context rows_digest != provenance rows_digest"
            )

    payload_fingerprint = _scientific_payload_fingerprint(
        texts=frozen_arrays[0],
        y=frozen_arrays[1],
        groups=frozen_arrays[2],
        authors=authors,
        provenance=provenance,
        weighting=weighting,
        rows_digest=rows_digest,
        isolation_receipt_sha256=isolation_receipt_sha256,
        isolation_contract_version=isolation_contract_version,
    )
    if disk_authority is None:
        disk_verified = False
    else:
        with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
            authorized = _DISK_AUTHORITY_REGISTRY.get(disk_authority)
        if authorized != payload_fingerprint:
            raise ProvenanceError(
                "scientific context lacks a matching disk-verification receipt"
            )
        disk_verified = True

    for values in frozen_arrays:
        values.setflags(write=False)

    context = object.__new__(ScientificEvaluationContext)
    for name, value in (
        ("texts", frozen_arrays[0]),
        ("y", frozen_arrays[1]),
        ("groups", frozen_arrays[2]),
        ("authors", authors),
        ("provenance", provenance),
        ("weighting", weighting),
        ("rows_digest", rows_digest),
        ("isolation_receipt_sha256", isolation_receipt_sha256),
        ("isolation_contract_version", isolation_contract_version),
        ("disk_verified", disk_verified),
        ("_seal", _SCIENTIFIC_CONTEXT_SEAL),
        ("_disk_authority", disk_authority),
    ):
        object.__setattr__(context, name, value)
    with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
        _SCIENTIFIC_CONTEXT_REGISTRY[context] = (
            _scientific_context_fingerprint(context)
        )
    return context


def _freeze_scientific_context(
    dataset,
    weighting: str,
    *,
    disk_verified: bool,
) -> ScientificEvaluationContext:
    if type(disk_verified) is not bool:
        raise ProvenanceError("disk_verified must be an exact bool")
    weighting = resolve_training_weighting(weighting)
    disk_authority = (
        _require_disk_authority_for_dataset(dataset, weighting)
        if disk_verified
        else None
    )
    authors = tuple(dataset.authors)
    provenance = getattr(dataset, "provenance", None)
    rows_digest = getattr(provenance, "rows_digest", None)
    if type(rows_digest) is not str:
        rows_digest = _scientific_rows_receipt(
            dataset.texts,
            dataset.y,
            dataset.groups,
            authors,
        )
    return _build_scientific_evaluation_context(
        texts=dataset.texts,
        y=dataset.y,
        groups=dataset.groups,
        authors=authors,
        provenance=provenance,
        weighting=weighting,
        rows_digest=rows_digest,
        isolation_receipt_sha256=_scientific_rows_receipt(
            dataset.texts,
            dataset.y,
            dataset.groups,
            authors,
        ),
        isolation_contract_version=SCIENTIFIC_ISOLATION_CONTRACT_VERSION,
        disk_authority=disk_authority,
    )


def require_scientific_evaluation_context(
    context,
) -> ScientificEvaluationContext:
    """Validate an already registered context, including explicit test contexts."""

    import numpy as np

    if (
        type(context) is not ScientificEvaluationContext
        or getattr(context, "_seal", None) is not _SCIENTIFIC_CONTEXT_SEAL
        or context.isolation_contract_version
        != SCIENTIFIC_ISOLATION_CONTRACT_VERSION
    ):
        raise ProvenanceError(
            "scientific evaluator requires a sealed "
            "ScientificEvaluationContext"
        )
    with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
        registered = _SCIENTIFIC_CONTEXT_REGISTRY.get(context)
    if registered is None:
        raise ProvenanceError(
            "scientific evaluator requires a registered sealed context"
        )
    if any(
        type(values) is not np.ndarray or values.ndim != 1
        for values in (context.texts, context.y, context.groups)
    ):
        raise ProvenanceError("scientific evaluation context arrays are malformed")
    if any(
        values.flags.writeable
        for values in (context.texts, context.y, context.groups)
    ):
        raise ProvenanceError("scientific evaluation context arrays are mutable")
    if not (
        len(context.texts)
        == len(context.y)
        == len(context.groups)
    ):
        raise ProvenanceError("scientific evaluation context length mismatch")
    if type(context.authors) is not tuple or not all(
        type(author) is str for author in context.authors
    ):
        raise ProvenanceError("scientific evaluation context authors are malformed")
    if not all(type(value) is str for value in context.texts):
        raise ProvenanceError("scientific evaluation context texts are malformed")
    if not all(type(value) is str for value in context.groups):
        raise ProvenanceError("scientific evaluation context groups are malformed")
    _validate_semantics(context.y, context.groups, context.authors)
    if (
        type(context.weighting) is not str
        or resolve_training_weighting(context.weighting) != context.weighting
    ):
        raise ProvenanceError("scientific evaluation context weighting is malformed")
    if type(context.disk_verified) is not bool:
        raise ProvenanceError(
            "scientific evaluation context disk_verified is malformed"
        )
    if (
        (context.disk_verified and type(context._disk_authority)
         is not _DiskVerificationAuthority)
        or (not context.disk_verified and context._disk_authority is not None)
    ):
        raise ProvenanceError(
            "scientific evaluation context disk authority is malformed"
        )
    receipt = _scientific_rows_receipt(
        context.texts,
        context.y,
        context.groups,
        context.authors,
    )
    if (
        type(context.isolation_receipt_sha256) is not str
        or receipt != context.isolation_receipt_sha256
    ):
        raise ProvenanceError(
            "scientific evaluation context isolation receipt mismatch"
        )
    if context.provenance is None:
        if context.disk_verified or context.rows_digest != receipt:
            raise ProvenanceError(
                "synthetic scientific evaluation context binding mismatch"
            )
    else:
        if type(context.provenance) is not DatasetProvenance:
            raise ProvenanceError(
                "scientific evaluation context provenance is malformed"
            )
        _verify_stored_provenance(
            context.texts,
            context.y,
            context.groups,
            context.authors,
            context.provenance,
        )
        if context.rows_digest != context.provenance.rows_digest:
            raise ProvenanceError(
                "scientific evaluation context rows_digest mismatch"
            )
    payload_fingerprint = _scientific_payload_fingerprint(
        texts=context.texts,
        y=context.y,
        groups=context.groups,
        authors=context.authors,
        provenance=context.provenance,
        weighting=context.weighting,
        rows_digest=context.rows_digest,
        isolation_receipt_sha256=context.isolation_receipt_sha256,
        isolation_contract_version=context.isolation_contract_version,
    )
    if context.disk_verified:
        with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
            authorized = _DISK_AUTHORITY_REGISTRY.get(
                context._disk_authority
            )
        if authorized != payload_fingerprint:
            raise ProvenanceError(
                "scientific evaluation context disk authority is stale"
            )
    if _scientific_context_fingerprint(context) != registered:
        raise ProvenanceError(
            "scientific evaluation context changed after authorization"
        )
    return context


def require_disk_verified_scientific_context(
    context,
) -> ScientificEvaluationContext:
    """Require the production scientific authority, never the synthetic seam."""

    context = require_scientific_evaluation_context(context)
    if not context.disk_verified:
        raise ProvenanceError(
            "production scientific evaluator requires a disk-verified context"
        )
    return context


def prepare_scientific_evaluation(
    cfg,
    dataset,
    weighting: str,
) -> ScientificEvaluationContext:
    """Disk-bind, isolate and freeze one dataset before scientific execution.

    The frozen run contract is derived here from ``cfg``; callers cannot
    supply a self-selected contract.
    """

    from .dispatch import frozen_run_contract

    weighting = verify_dataset_against_disk(
        cfg,
        dataset,
        weighting,
        frozen_run_contract(cfg),
    )
    return _freeze_scientific_context(
        dataset,
        weighting,
        disk_verified=True,
    )


def reverify_scientific_context_from_disk(
    cfg,
    context,
) -> ScientificEvaluationContext:
    """Re-establish production authority after a process/pickle boundary.

    Serialized contexts are deliberately restored without disk authority.
    Workers use the trusted run config to repeat the full on-disk comparison;
    a small process-local cache avoids reloading the same corpus for every fold.
    """

    context = require_scientific_evaluation_context(context)
    if context.disk_verified:
        return require_disk_verified_scientific_context(context)
    if type(context.provenance) is not DatasetProvenance:
        raise ProvenanceError(
            "disk-verified re-verification requires exact dataset provenance"
        )
    try:
        config_body = cfg.to_dict()
    except Exception as exc:
        raise ProvenanceError(
            "disk re-verification requires a resolved run configuration"
        ) from exc
    from ..jsonio import dumps_strict

    config_sha256 = hashlib.sha256(
        dumps_strict(config_body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload = _scientific_payload_fingerprint(
        texts=context.texts,
        y=context.y,
        groups=context.groups,
        authors=context.authors,
        provenance=context.provenance,
        weighting=context.weighting,
        rows_digest=context.rows_digest,
        isolation_receipt_sha256=context.isolation_receipt_sha256,
        isolation_contract_version=context.isolation_contract_version,
    )
    cache_key = (config_sha256, payload)
    with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
        cached = _WORKER_REVERIFIED_CONTEXTS.get(cache_key)
    if cached is not None:
        return require_disk_verified_scientific_context(cached)

    import numpy as np

    from ..corpus import Dataset

    dataset = Dataset(
        texts=np.array(context.texts, dtype=object, copy=True),
        y=np.array(context.y, dtype=int, copy=True),
        groups=np.array(context.groups, dtype=object, copy=True),
        authors=list(context.authors),
        provenance=context.provenance,
    )
    verified = prepare_scientific_evaluation(
        cfg,
        dataset,
        context.weighting,
    )
    verified_payload = _scientific_payload_fingerprint(
        texts=verified.texts,
        y=verified.y,
        groups=verified.groups,
        authors=verified.authors,
        provenance=verified.provenance,
        weighting=verified.weighting,
        rows_digest=verified.rows_digest,
        isolation_receipt_sha256=verified.isolation_receipt_sha256,
        isolation_contract_version=verified.isolation_contract_version,
    )
    if verified_payload != payload:
        raise ProvenanceError(
            "disk re-verification changed the scientific context payload"
        )
    with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
        if len(_WORKER_REVERIFIED_CONTEXTS) >= 4:
            _WORKER_REVERIFIED_CONTEXTS.clear()
        _WORKER_REVERIFIED_CONTEXTS[cache_key] = verified
    return verified


def prepare_derived_scientific_evaluation(
    parent_context,
    dataset,
) -> ScientificEvaluationContext:
    """Authorize a provenance-preserving subset of an already verified parent."""

    parent = require_disk_verified_scientific_context(parent_context)
    return _prepare_derived_scientific_evaluation(parent, dataset)


def prepare_synthetic_derived_scientific_evaluation(
    parent_context,
    dataset,
) -> ScientificEvaluationContext:
    """Explicit non-production subset seam for isolated integration tests."""

    parent = require_scientific_evaluation_context(parent_context)
    if parent.disk_verified:
        raise ProvenanceError(
            "synthetic derived preparation requires a synthetic parent"
        )
    return _prepare_derived_scientific_evaluation(parent, dataset)


def _prepare_derived_scientific_evaluation(
    parent,
    dataset,
) -> ScientificEvaluationContext:
    parent_prov = parent.provenance
    if type(parent_prov) is not DatasetProvenance:
        raise ProvenanceError(
            "derived scientific evaluation requires a provenance-bound parent"
        )
    prov = getattr(dataset, "provenance", None)
    if type(prov) is not DatasetProvenance:
        raise ProvenanceError("derived scientific dataset has no provenance")
    if (
        prov.parent_rows_digest != parent.rows_digest
        or prov.frags_root != parent_prov.frags_root
        or prov.corpus_policy != parent_prov.corpus_policy
        or prov.loader_kind != parent_prov.loader_kind
        or prov.chunker_config_hash != parent_prov.chunker_config_hash
        or prov.manifest_hash != parent_prov.manifest_hash
        or prov.config_id != parent_prov.config_id
    ):
        raise ProvenanceError(
            "derived scientific dataset is not bound to the verified parent"
        )
    _verify_stored_provenance(
        dataset.texts,
        dataset.y,
        dataset.groups,
        dataset.authors,
        prov,
    )
    if (
        prov.selection_manifest_digest is None
        or _selection_digest(prov.row_ids)
        != prov.selection_manifest_digest
    ):
        raise ProvenanceError(
            "derived scientific dataset selection manifest mismatch"
        )
    _require_subsequence(prov.row_ids, parent_prov.row_ids)
    if not parent.disk_verified:
        return _freeze_scientific_context(
            dataset,
            parent.weighting,
            disk_verified=False,
        )

    # The already disk-verified parent is the authority for this exact ordered
    # subsequence. Mint the child receipt only here, after the complete
    # provenance-chain and row-identity checks above; callers cannot request
    # this state through a flag or serialized context.
    authors = tuple(dataset.authors)
    receipt = _scientific_rows_receipt(
        dataset.texts,
        dataset.y,
        dataset.groups,
        authors,
    )
    payload_fingerprint = _scientific_payload_fingerprint(
        texts=dataset.texts,
        y=dataset.y,
        groups=dataset.groups,
        authors=authors,
        provenance=prov,
        weighting=parent.weighting,
        rows_digest=prov.rows_digest,
        isolation_receipt_sha256=receipt,
        isolation_contract_version=SCIENTIFIC_ISOLATION_CONTRACT_VERSION,
    )
    authority = _DiskVerificationAuthority()
    with _SCIENTIFIC_CONTEXT_REGISTRY_LOCK:
        _DISK_AUTHORITY_REGISTRY[authority] = payload_fingerprint
    return _build_scientific_evaluation_context(
        texts=dataset.texts,
        y=dataset.y,
        groups=dataset.groups,
        authors=authors,
        provenance=prov,
        weighting=parent.weighting,
        rows_digest=prov.rows_digest,
        isolation_receipt_sha256=receipt,
        isolation_contract_version=SCIENTIFIC_ISOLATION_CONTRACT_VERSION,
        disk_authority=authority,
    )


def prepare_synthetic_scientific_evaluation(
    dataset,
    weighting: str,
) -> ScientificEvaluationContext:
    """Content-gated context for synthetic integration tests and injected evaluators."""

    weighting = resolve_training_weighting(weighting)
    return _freeze_scientific_context(
        dataset,
        weighting,
        disk_verified=False,
    )


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
