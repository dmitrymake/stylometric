"""Stable, evaluation-independent scientific data contracts."""

from .corpus_identity import (
    CONTENT_OVERLAP_POLICY_VERSION,
    DIGEST_VERSION,
    LEGACY_RECURSIVE,
    WORK_BALANCED_MANIFEST,
    ContentIsolationError,
    ContentOverlap,
    CorpusPolicyProvenance,
    DataContract,
    DatasetProvenance,
    ProvenanceError,
    RowIdentity,
    assert_cross_work_content_isolation,
    build_provenance,
    canonical_digest,
    find_cross_work_content_overlaps,
)

__all__ = [
    "CONTENT_OVERLAP_POLICY_VERSION",
    "DIGEST_VERSION",
    "LEGACY_RECURSIVE",
    "WORK_BALANCED_MANIFEST",
    "ContentIsolationError",
    "ContentOverlap",
    "CorpusPolicyProvenance",
    "DataContract",
    "DatasetProvenance",
    "ProvenanceError",
    "RowIdentity",
    "assert_cross_work_content_isolation",
    "build_provenance",
    "canonical_digest",
    "find_cross_work_content_overlaps",
]
