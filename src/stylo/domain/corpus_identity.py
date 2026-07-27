"""Cross-work content-identity checks used before scientific evaluation.

A work id is not an independence boundary when the same text is also present in
another registered work (for example, a short story and a collection containing
that story).  This module deliberately has no dependency on corpus loaders or
evaluation code so every runner can apply the same fail-closed contract.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import numbers
import re
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Optional

import numpy as np

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UINT64_MASK = (1 << 64) - 1
_SHINGLE_FACTORS = np.asarray(
    [
        0x9E3779B185EBCA87,
        0xC2B2AE3D27D4EB4F,
        0x165667B19E3779F9,
        0x85EBCA77C2B2AE63,
        0x27D4EB2F165667C5,
    ],
    dtype=np.uint64,
)

DIGEST_VERSION = "b2.prov.v2"
LEGACY_RECURSIVE = "legacy_recursive"
WORK_BALANCED_MANIFEST = "work_balanced_manifest"
CONTENT_OVERLAP_POLICY_VERSION = (
    "stylo.cross-work-content-isolation.word5.v2"
)


class ProvenanceError(ValueError):
    """Fail-closed corpus identity or weighting-contract violation."""


@dataclasses.dataclass(frozen=True)
class CorpusPolicyProvenance:
    """Immutable corpus-selection policy bound into a dataset identity."""

    exclude_from_benchmark: tuple[str, ...]
    unknown_dir_name: str

    @staticmethod
    def build(
        exclude_from_benchmark: Sequence[str],
        unknown_dir_name: str,
    ) -> "CorpusPolicyProvenance":
        if isinstance(exclude_from_benchmark, (str, bytes)):
            raise ProvenanceError(
                "exclude_from_benchmark must be a list of author ids, not a string"
            )
        excl = list(exclude_from_benchmark or ())
        if not all(type(author) is str for author in excl):
            raise ProvenanceError(
                "exclude_from_benchmark entries must all be exact str"
            )
        if type(unknown_dir_name) is not str:
            raise ProvenanceError("unknown_dir_name must be an exact str")
        return CorpusPolicyProvenance(
            tuple(sorted(set(excl))),
            unknown_dir_name,
        )


@dataclasses.dataclass(frozen=True)
class RowIdentity:
    """Neutral per-row identity shared by corpus loaders and evaluators."""

    group: str
    ordinal: int
    text_sha256: str
    work_id: Optional[str] = None
    provenance_sha256: Optional[str] = None
    chunker_config_hash: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DataContract:
    """The immutable data-side contract checked at run entry."""

    frags_root: str
    corpus_policy: CorpusPolicyProvenance
    chunker_config_hash: Optional[str]
    canonical_digest: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DatasetProvenance:
    """Canonical identity of the exact rows exposed by a corpus loader."""

    digest_version: str
    loader_kind: str
    row_ids: tuple
    authors: tuple
    n_rows: int
    chunker_config_hash: Optional[str]
    corpus_policy: CorpusPolicyProvenance
    frags_root: str
    rows_digest: str
    manifest_hash: Optional[str] = None
    config_id: Optional[str] = None
    parent_rows_digest: Optional[str] = None
    selection_manifest_digest: Optional[str] = None

    def contract(self) -> DataContract:
        return DataContract(
            self.frags_root,
            self.corpus_policy,
            self.chunker_config_hash,
        )


def _lp(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _s(value) -> bytes:
    return b"" if value is None else str(value).encode("utf-8")


def canonical_digest(
    texts: Sequence,
    y: Sequence,
    groups: Sequence,
    authors: Sequence,
    row_ids: Sequence[RowIdentity],
    *,
    loader_kind: str,
    chunker_config_hash: Optional[str],
) -> str:
    """Return the versioned digest over rows, labels and identity metadata."""

    n_rows = len(texts)
    if not (len(y) == len(groups) == len(row_ids) == n_rows):
        raise ProvenanceError("texts/y/groups/row_ids length mismatch")
    digest = hashlib.sha256()
    digest.update(_lp(DIGEST_VERSION.encode()))
    digest.update(_lp(_s(loader_kind)))
    digest.update(_lp(_s(chunker_config_hash)))
    digest.update(n_rows.to_bytes(8, "big"))
    for index in range(n_rows):
        digest.update(_lp(_s(texts[index])))
        digest.update(int(y[index]).to_bytes(8, "big", signed=True))
        digest.update(_lp(_s(groups[index])))
        row_id = row_ids[index]
        for part in (
            row_id.group,
            row_id.ordinal,
            row_id.text_sha256,
            row_id.work_id,
            row_id.provenance_sha256,
            row_id.chunker_config_hash,
        ):
            digest.update(_lp(_s(part)))
    for author in authors:
        digest.update(_lp(_s(author)))
    return digest.hexdigest()


def _str_or_none(value, name: str):
    if value is not None and type(value) is not str:
        raise ProvenanceError(f"{name} must be an exact str or None")
    return value


def _canonical_ri(row_id) -> tuple:
    if type(row_id) is not RowIdentity:
        raise ProvenanceError("row identity must be exactly RowIdentity")
    if (
        type(row_id.group) is not str
        or type(row_id.ordinal) is not int
        or type(row_id.text_sha256) is not str
    ):
        raise ProvenanceError(
            "row identity group/ordinal/text_sha256 have wrong exact types"
        )
    return (
        row_id.group,
        row_id.ordinal,
        row_id.text_sha256,
        _str_or_none(row_id.work_id, "work_id"),
        _str_or_none(row_id.provenance_sha256, "provenance_sha256"),
        _str_or_none(row_id.chunker_config_hash, "chunker_config_hash"),
    )


def _validate_provenance_schema(provenance) -> None:
    if type(provenance) is not DatasetProvenance:
        raise ProvenanceError("provenance must be exactly DatasetProvenance")
    if (
        type(provenance.digest_version) is not str
        or provenance.digest_version != DIGEST_VERSION
    ):
        raise ProvenanceError("bad digest_version")
    if provenance.loader_kind not in (
        LEGACY_RECURSIVE,
        WORK_BALANCED_MANIFEST,
    ):
        raise ProvenanceError("bad loader_kind")
    if (
        type(provenance.rows_digest) is not str
        or not _HEX64.fullmatch(provenance.rows_digest)
    ):
        raise ProvenanceError("rows_digest must be an exact sha256 hex string")
    if (
        type(provenance.n_rows) is not int
        or type(provenance.frags_root) is not str
    ):
        raise ProvenanceError("n_rows/frags_root wrong exact types")
    if (
        type(provenance.authors) is not tuple
        or not all(type(author) is str for author in provenance.authors)
    ):
        raise ProvenanceError("authors must be a tuple of exact str")
    if type(provenance.corpus_policy) is not CorpusPolicyProvenance:
        raise ProvenanceError(
            "corpus_policy must be exactly CorpusPolicyProvenance"
        )
    if (
        type(provenance.corpus_policy.exclude_from_benchmark) is not tuple
        or not all(
            type(author) is str
            for author in provenance.corpus_policy.exclude_from_benchmark
        )
        or type(provenance.corpus_policy.unknown_dir_name) is not str
    ):
        raise ProvenanceError("corpus_policy fields wrong exact types")
    if (
        type(provenance.row_ids) is not tuple
        or len(provenance.row_ids) != provenance.n_rows
    ):
        raise ProvenanceError("row_ids must be a tuple of length n_rows")
    for row_id in provenance.row_ids:
        _canonical_ri(row_id)
    for value, name in (
        (provenance.parent_rows_digest, "parent_rows_digest"),
        (provenance.selection_manifest_digest, "selection_manifest_digest"),
        (provenance.chunker_config_hash, "chunker_config_hash"),
        (provenance.manifest_hash, "manifest_hash"),
        (provenance.config_id, "config_id"),
    ):
        _str_or_none(value, name)


def _author_of_group(group: str) -> str:
    if type(group) is not str:
        raise ProvenanceError("group must be an exact str")
    return group.split("/", 1)[0]


def _as_label(label, index: int) -> int:
    if isinstance(label, bool) or not isinstance(label, numbers.Integral):
        raise ProvenanceError(
            f"y[{index}]={label!r} must be a non-bool integer label"
        )
    return int(label)


def _validate_semantics(
    y: Sequence,
    groups: Sequence,
    authors: Sequence,
) -> None:
    n_authors = len(authors)
    if len(set(authors)) != n_authors:
        raise ProvenanceError("authors must be unique")
    for index, (label, group) in enumerate(zip(y, groups, strict=True)):
        label_index = _as_label(label, index)
        if not 0 <= label_index < n_authors:
            raise ProvenanceError(
                f"y[{index}]={label_index} out of range [0,{n_authors})"
            )
        if authors[label_index] != _author_of_group(group):
            raise ProvenanceError(
                f"row {index}: authors[y]={authors[label_index]!r} != "
                f"author of groups[i]={_author_of_group(group)!r}"
            )


def _require_identity_for_kind(
    loader_kind: str,
    row_ids: Sequence[RowIdentity],
) -> None:
    if loader_kind == WORK_BALANCED_MANIFEST:
        for row_id in row_ids:
            if (
                row_id.work_id is None
                or row_id.provenance_sha256 is None
                or row_id.chunker_config_hash is None
            ):
                raise ProvenanceError(
                    "work_balanced provenance requires per-row manifest "
                    "identity (work_id/provenance_sha256/chunker_config_hash)"
                )
    else:
        for row_id in row_ids:
            if (
                row_id.work_id is not None
                or row_id.provenance_sha256 is not None
                or row_id.chunker_config_hash is not None
            ):
                raise ProvenanceError(
                    "legacy_recursive rows must not carry manifest identity"
                )


def _validate_self_consistency(
    texts,
    y,
    groups,
    authors,
    row_ids,
    loader_kind,
    chunker_config_hash,
) -> None:
    n_rows = len(texts)
    if not (len(y) == len(groups) == len(row_ids) == n_rows):
        raise ProvenanceError("texts/y/groups/row_ids length mismatch")
    _validate_semantics(y, groups, authors)
    _require_identity_for_kind(loader_kind, row_ids)
    if not all(type(author) is str for author in authors):
        raise ProvenanceError("authors must all be exactly str")
    for index, row_id in enumerate(row_ids):
        if type(texts[index]) is not str:
            raise ProvenanceError(
                f"row {index}: text must be exactly str (no int/subclass coercion)"
            )
        if type(groups[index]) is not str:
            raise ProvenanceError(f"row {index}: group must be exactly str")
        if (
            isinstance(row_id.ordinal, bool)
            or not isinstance(row_id.ordinal, numbers.Integral)
            or row_id.ordinal < 0
        ):
            raise ProvenanceError(
                f"row {index}: span ordinal must be a non-negative int"
            )
        if (
            type(row_id.text_sha256) is not str
            or not _HEX64.fullmatch(row_id.text_sha256)
        ):
            raise ProvenanceError(
                f"row {index}: text_sha256 is not a sha256 hex digest"
            )
        if (
            hashlib.sha256(texts[index].encode("utf-8")).hexdigest()
            != row_id.text_sha256
        ):
            raise ProvenanceError(
                f"row {index}: text_sha256 does not match the actual text"
            )
        if (
            type(row_id.group) is not str
            or row_id.group != groups[index]
        ):
            raise ProvenanceError(f"row {index}: row_id.group != groups[i]")
        if loader_kind == WORK_BALANCED_MANIFEST:
            if type(row_id.work_id) is not str or row_id.work_id != groups[index]:
                raise ProvenanceError(f"row {index}: work_id != group")
            if (
                type(row_id.provenance_sha256) is not str
                or not _HEX64.fullmatch(row_id.provenance_sha256)
            ):
                raise ProvenanceError(
                    f"row {index}: provenance_sha256 is not a sha256 hex digest"
                )
            if (
                type(row_id.chunker_config_hash) is not str
                or row_id.chunker_config_hash != chunker_config_hash
            ):
                raise ProvenanceError(
                    f"row {index}: row chunker hash != dataset chunker hash"
                )


def _verify_stored_provenance(
    texts,
    y,
    groups,
    authors,
    provenance: DatasetProvenance,
) -> None:
    _validate_provenance_schema(provenance)
    if provenance.n_rows != len(texts):
        raise ProvenanceError("row count changed since load (mutated Dataset)")
    if tuple(authors) != provenance.authors:
        raise ProvenanceError("authors changed since load")
    _validate_self_consistency(
        texts,
        y,
        groups,
        authors,
        provenance.row_ids,
        provenance.loader_kind,
        provenance.chunker_config_hash,
    )
    recomputed = canonical_digest(
        texts,
        y,
        groups,
        authors,
        provenance.row_ids,
        loader_kind=provenance.loader_kind,
        chunker_config_hash=provenance.chunker_config_hash,
    )
    if recomputed != provenance.rows_digest:
        raise ProvenanceError(
            "rows_digest mismatch — Dataset was mutated, relabeled or forged"
        )


def _selection_digest(row_ids: Sequence[RowIdentity]) -> str:
    digest = hashlib.sha256()
    digest.update(_lp(b"selection.v1"))
    for row_id in row_ids:
        for part in _canonical_ri(row_id):
            digest.update(_lp(_s(part)))
    return digest.hexdigest()


def _require_subsequence(
    child: Sequence[RowIdentity],
    parent: Sequence[RowIdentity],
) -> None:
    parent_iterator = (_canonical_ri(row_id) for row_id in parent)
    for child_row_id in child:
        target = _canonical_ri(child_row_id)
        for parent_row_id in parent_iterator:
            if parent_row_id == target:
                break
        else:
            raise ProvenanceError(
                "subset rows are not an ordered subsequence of the disk parent"
            )


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
    _validate_self_consistency(
        texts,
        y,
        groups,
        authors,
        row_ids,
        loader_kind,
        chunker_config_hash,
    )
    rows_digest = canonical_digest(
        texts,
        y,
        groups,
        authors,
        row_ids,
        loader_kind=loader_kind,
        chunker_config_hash=chunker_config_hash,
    )
    return DatasetProvenance(
        digest_version=DIGEST_VERSION,
        loader_kind=loader_kind,
        row_ids=tuple(row_ids),
        authors=tuple(authors),
        n_rows=len(texts),
        chunker_config_hash=chunker_config_hash,
        corpus_policy=corpus_policy,
        frags_root=str(frags_root),
        rows_digest=rows_digest,
        manifest_hash=manifest_hash,
        config_id=config_id,
        parent_rows_digest=parent_rows_digest,
        selection_manifest_digest=selection_manifest_digest,
    )


@dataclasses.dataclass(frozen=True)
class ContentOverlap:
    """One disallowed relationship between two distinct registered works."""

    left_work: str
    right_work: str
    kind: str
    containment: float
    evidence: str


class ContentIsolationError(ValueError):
    """Raised when train/test work ids are not content-independent."""

    def __init__(self, overlaps: Sequence[ContentOverlap]):
        self.overlaps = tuple(overlaps)
        sample = "; ".join(
            f"{o.left_work} -> {o.right_work}: {o.kind} "
            f"({o.containment:.2%}, {o.evidence})"
            for o in self.overlaps[:8]
        )
        more = "" if len(self.overlaps) <= 8 else f"; +{len(self.overlaps) - 8} more"
        super().__init__(
            "cross-work content isolation failed; the corpus is ineligible for "
            f"work-held-out evaluation: {sample}{more}"
        )


def _token_ids(text: str, cache: dict[str, int]) -> np.ndarray:
    """Stable 64-bit token ids for one row (row boundaries stay explicit)."""

    ids: list[int] = []
    if type(text) is not str:
        raise TypeError("corpus texts must be exact strings")
    for token in _WORD_RE.findall(text.casefold()):
        token_id = cache.get(token)
        if token_id is None:
            token_id = int.from_bytes(
                hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            cache[token] = token_id
        ids.append(token_id)
    return np.asarray(ids, dtype=np.uint64)


def _word_shingles(texts: Sequence[str], n: int) -> np.ndarray:
    if n != 5:
        raise ValueError("the registered content-isolation contract uses word 5-grams")
    cache: dict[str, int] = {}
    rows: list[np.ndarray] = []
    for text in texts:
        ids = _token_ids(text, cache)
        width = len(ids) - n + 1
        if width <= 0:
            continue
        mixed = np.zeros(width, dtype=np.uint64)
        for offset, factor in enumerate(_SHINGLE_FACTORS):
            mixed ^= ids[offset : offset + width] * factor
        rows.append(mixed)
    if not rows:
        return np.asarray([], dtype=np.uint64)
    return np.unique(np.concatenate(rows))


def _sample(values: np.ndarray, size: int) -> np.ndarray:
    if len(values) <= size:
        return values
    indexes = np.linspace(0, len(values) - 1, num=size, dtype=np.int64)
    return values[indexes]


def _threshold_fraction(value: object) -> Fraction:
    """Canonical decimal/rational threshold used by overlap contract v2."""

    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("containment_threshold must be finite")
        threshold = Fraction(str(value))
    elif type(value) is int:
        threshold = Fraction(value, 1)
    elif type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("containment_threshold must be finite")
        threshold = Fraction(value)
    elif type(value) is Fraction:
        threshold = value
    else:
        raise TypeError(
            "containment_threshold must be an exact int, float, Decimal, "
            "or Fraction"
        )
    if not Fraction(0, 1) < threshold <= Fraction(1, 1):
        raise ValueError("containment_threshold must be in (0, 1]")
    return threshold


def find_cross_work_content_overlaps(
    texts: Sequence[str] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    *,
    containment_threshold: float = 0.90,
    min_shingles: int = 20,
    sample_size: int = 64,
) -> tuple[ContentOverlap, ...]:
    """Return exact-chunk and asymmetric short-in-long overlaps.

    Repetition inside one registered work is allowed.  Across distinct works,
    an exact canonical chunk is always disallowed.  Work-level word-5-gram
    containment is checked with a cheap deterministic sample and then an exact
    set intersection for candidates, which keeps the 23k-chunk corpus gate
    practical without weakening the final threshold.
    """

    if len(texts) != len(groups):
        raise ValueError("texts and groups must have equal length")
    threshold = _threshold_fraction(containment_threshold)
    if type(min_shingles) is not int or min_shingles <= 0:
        raise ValueError("min_shingles must be a positive exact integer")
    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive exact integer")

    work_texts: dict[str, list[str]] = defaultdict(list)
    exact_owners: dict[str, str] = {}
    overlaps: list[ContentOverlap] = []
    exact_pairs: set[tuple[str, str]] = set()
    for raw_text, raw_group in zip(texts, groups, strict=True):
        if type(raw_text) is not str:
            raise TypeError("corpus texts must be exact strings")
        group = str(raw_group)
        canonical = raw_text.strip()
        if not canonical:
            raise ValueError(f"{group}: empty canonical corpus row")
        work_texts[group].append(canonical)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        owner = exact_owners.setdefault(digest, group)
        if owner != group:
            pair = tuple(sorted((owner, group)))
            if pair not in exact_pairs:
                exact_pairs.add(pair)
                overlaps.append(
                    ContentOverlap(
                        left_work=pair[0],
                        right_work=pair[1],
                        kind="exact_cross_work_chunk",
                        containment=1.0,
                        evidence=f"sha256:{digest}",
                    )
                )

    shingles = {
        work: _word_shingles(rows, 5) for work, rows in sorted(work_texts.items())
    }
    works = sorted(shingles)
    for left_index, left in enumerate(works):
        for right in works[left_index + 1 :]:
            a, b = shingles[left], shingles[right]
            if len(a) < min_shingles or len(b) < min_shingles:
                continue
            if len(a) <= len(b):
                short_work, long_work, short, long = left, right, a, b
            else:
                short_work, long_work, short, long = right, left, b, a
            # Reject through the prefilter only after observing more misses
            # than a >=threshold overlap could possibly contain.  This bound
            # makes the filter exact rather than heuristic.
            allowed_misses = (
                (threshold.denominator - threshold.numerator) * len(short)
                // threshold.denominator
            )
            probe = _sample(short, max(sample_size, allowed_misses + 1))
            locations = np.searchsorted(long, probe)
            in_long = (locations < len(long)) & (
                long[np.minimum(locations, len(long) - 1)] == probe
            )
            if int((~in_long).sum()) > allowed_misses:
                continue
            common = int(np.intersect1d(short, long, assume_unique=True).size)
            if (
                common * threshold.denominator
                >= len(short) * threshold.numerator
            ):
                ratio = common / len(short)
                overlaps.append(
                    ContentOverlap(
                        left_work=short_work,
                        right_work=long_work,
                        kind="word5_asymmetric_containment",
                        containment=float(ratio),
                        evidence=f"{common}/{len(short)} unique word-5-grams",
                    )
                )
    return tuple(overlaps)


def assert_cross_work_content_isolation(
    texts: Sequence[str] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    **kwargs,
) -> None:
    """Fail closed unless every registered work is content-independent."""

    overlaps = find_cross_work_content_overlaps(texts, groups, **kwargs)
    if overlaps:
        raise ContentIsolationError(overlaps)


__all__ = [
    "DIGEST_VERSION",
    "LEGACY_RECURSIVE",
    "WORK_BALANCED_MANIFEST",
    "CONTENT_OVERLAP_POLICY_VERSION",
    "CorpusPolicyProvenance",
    "DataContract",
    "DatasetProvenance",
    "ProvenanceError",
    "RowIdentity",
    "ContentIsolationError",
    "ContentOverlap",
    "assert_cross_work_content_isolation",
    "build_provenance",
    "canonical_digest",
    "find_cross_work_content_overlaps",
]
