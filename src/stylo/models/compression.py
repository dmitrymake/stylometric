"""Research-only conditional LZMA compression scores for explicit reference works.

Ryabko and Savina (2021), section 3, define C(reference + target) - C(reference)
with no separator and select LZMA as their compressor:
https://pmc.ncbi.nlm.nih.gov/articles/PMC8534409/

This baseline uses fixed raw LZMA1 options and exact, strict UTF-8 input. It does
not reproduce the paper's preprocessing, sample sizes, parameter selection or
statistical tests. Smaller scores indicate better conditional compression;
scores are diagnostic, can be negative, and are neither a distance metric nor
an authorship probability. There is no authorship decision or calibrated gate.

Candidate scores are arithmetic means over independently supplied works, each
with weight 1 / n_works. Reference texts are never concatenated across works.
Equal work weights do not remove sensitivity to reference/target length, the
finite dictionary, topic, quotations, spelling, edition or text order. Work IDs
and exact-content exclusions cannot establish content independence: callers
must independently check provenance, alternate editions and overlapping text.
Results are reproducible for a fixed Python/liblzma runtime; different liblzma
versions may produce different compressed lengths. No corpus or registry is
accessed and no files are written.
"""
from __future__ import annotations

import hashlib
import lzma
import math
from collections.abc import Sequence
from dataclasses import dataclass, field


ALGORITHM_VERSION = "stylo.compression.lzma1-conditional.v1"
SCORE_UNIT = "compressed_bytes_per_target_utf8_byte"


def compressor_spec() -> dict:
    """Return a detached description of the fixed algorithm and filter options."""
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "format": "raw",
        "filter": "LZMA1",
        "dict_size": 8 * 1024 * 1024,
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": "normal",
        "nice_len": 64,
        "mf": "bt4",
        "depth": 0,
    }


@dataclass(frozen=True, slots=True)
class ConditionalCompressionScore:
    reference_utf8_bytes: int
    target_utf8_bytes: int
    compressed_reference_bytes: int
    compressed_joint_bytes: int
    conditional_bytes: int
    bytes_per_target_byte: float
    algorithm_version: str = ALGORITHM_VERSION
    score_unit: str = SCORE_UNIT


@dataclass(frozen=True, slots=True)
class ReferenceWork:
    """Caller-supplied candidate/work identities and one complete reference text.

    ``work_id`` must identify the underlying work globally, across candidates
    and editions, rather than a fragment or filename. This is a metadata claim,
    checked for collisions but not independently verified by this module.
    """

    candidate_id: str
    work_id: str
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class CandidateCompressionScore:
    candidate_id: str
    n_reference_works: int
    mean_bytes_per_target_byte: float
    algorithm_version: str = ALGORITHM_VERSION
    score_unit: str = SCORE_UNIT
    aggregation: str = "equal_work_mean"
    diagnostic_only: bool = True


def _utf8(text: str, name: str) -> bytes:
    if type(text) is not str:
        raise TypeError(f"{name} must be an exact str")
    if not text or not text.strip():
        raise ValueError(f"{name} must contain non-whitespace text")
    # No stripping, normalization, newline conversion, separator or BOM added.
    return text.encode("utf-8", errors="strict")


def _identifier(value: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without outer whitespace")
    return value


def _compressed_length(data: bytes) -> int:
    return len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=[{
        "id": lzma.FILTER_LZMA1,
        "dict_size": 8 * 1024 * 1024,
        "lc": 3, "lp": 0, "pb": 2,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 64, "mf": lzma.MF_BT4, "depth": 0,
    }]))


def _score_bytes(reference: bytes, target: bytes) -> ConditionalCompressionScore:
    reference_length = _compressed_length(reference)
    joint_length = _compressed_length(reference + target)
    conditional = joint_length - reference_length
    return ConditionalCompressionScore(
        reference_utf8_bytes=len(reference), target_utf8_bytes=len(target),
        compressed_reference_bytes=reference_length,
        compressed_joint_bytes=joint_length,
        conditional_bytes=conditional, bytes_per_target_byte=conditional / len(target),
    )


def conditional_compression_score(reference_text: str, target_text: str) -> ConditionalCompressionScore:
    """Compute C(reference + target) - C(reference), normalized by target bytes.

    This mathematical pair scorer makes no reference-independence claim. Use
    ``rank_candidate_references`` for panel, work-ID and exact-content gates.
    """
    reference = _utf8(reference_text, "reference_text")
    target = _utf8(target_text, "target_text")
    return _score_bytes(reference, target)


def rank_candidate_references(
    target_text: str, *, target_work_id: str, candidate_ids: Sequence[str],
    references: Sequence[ReferenceWork],
) -> tuple[CandidateCompressionScore, ...]:
    """Return ascending diagnostic scores with explicit equal-work aggregation.

    Every panel candidate needs at least one reference. Reject duplicate work
    IDs, target work IDs, exact target/reference content and duplicate reference
    content before any compression. Ties use candidate ID order for reproducible
    presentation; that order conveys no extra evidence about authorship.
    """
    target = _utf8(target_text, "target_text")
    _identifier(target_work_id, "target_work_id")
    if not isinstance(candidate_ids, Sequence) or isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a sequence of candidate IDs")
    panel = tuple(_identifier(value, "candidate_id") for value in candidate_ids)
    if not panel or len(set(panel)) != len(panel):
        raise ValueError("candidate_ids must be non-empty and unique")
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise TypeError("references must be a sequence of ReferenceWork records")
    by_candidate: dict[str, list[tuple[str, bytes]]] = {value: [] for value in panel}
    work_ids: set[str] = set()
    content_hashes: set[bytes] = set()
    for reference in references:
        if type(reference) is not ReferenceWork:
            raise TypeError("references must contain exact ReferenceWork records")
        candidate = _identifier(reference.candidate_id, "reference candidate_id")
        work = _identifier(reference.work_id, "reference work_id")
        if candidate not in by_candidate:
            raise ValueError("reference candidate_id is outside the explicit panel")
        if work == target_work_id:
            raise ValueError("target_work_id must be excluded from all references")
        if work in work_ids:
            raise ValueError("reference work_id collision across reference records")
        work_ids.add(work)
        encoded = _utf8(reference.text, "reference text")
        if encoded == target:
            raise ValueError("reference content exactly matches the target")
        digest = hashlib.sha256(encoded).digest()
        if digest in content_hashes:
            raise ValueError("duplicate reference content under distinct work IDs")
        content_hashes.add(digest)
        by_candidate[candidate].append((work, encoded))
    if any(not records for records in by_candidate.values()):
        raise ValueError("every candidate requires at least one reference work")

    results = []
    for candidate in sorted(by_candidate):
        # Canonical work order also removes floating-point summation dependence
        # on the order in which the caller supplies the references.
        records = sorted(by_candidate[candidate])
        scores = [_score_bytes(text, target).bytes_per_target_byte for _work, text in records]
        mean = math.fsum(scores) / len(records)
        if not math.isfinite(mean):
            raise ValueError("conditional compression mean must be finite")
        results.append(CandidateCompressionScore(candidate, len(records), mean))
    return tuple(sorted(results, key=lambda result: (
        result.mean_bytes_per_target_byte, result.candidate_id,
    )))
