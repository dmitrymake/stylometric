"""Deterministic preparation of the owner-selected RuAA R1 vNext packet.

This is a preparation surface, not an evaluator.  It verifies the legacy RuAA
bundle only as public source evidence, resolves the three owner-reviewed
collection-member candidates, copies the selected 136 literal works into a new
immutable generation, and derives canonical clean/chunk rows.  No
representation cache, estimator factory, fit, prediction, or public evidence
writer is reachable from this module.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import spacy

from ..chunking import CombinedDoc, make_sent_chunks, sentences_for_text
from ..config import ConfigNode
from ..domain.corpus_identity import (
    CONTENT_OVERLAP_POLICY_VERSION,
    find_cross_work_content_overlaps,
)
from ..domain.lobo_vnext import (
    ContentComponent,
    ContentComponentManifest,
    CorpusVNextManifest,
    InferenceSpec,
    InnerCVPlan,
    ModelSpec,
    RawInventoryEntry,
    VNextContractError,
    WorkIdentity,
    build_corpus_vnext_manifest,
    build_fold_manifest,
    build_inner_cv_plan,
    canonical_sha256,
    inventory_raw_files,
)
from ..domain.lobo_vnext_packet import (
    CanonicalRepresentationReceipt,
    CanonicalRowEntry,
    PacketFileEntry,
    R1GenerationMaterial,
    R1PacketManifest,
    R1SourceSelectionReceipt,
    R1_GENERATION_MATERIAL_SCHEMA_VERSION,
    R1_PACKET_MANIFEST_SCHEMA_VERSION,
)
from ..domain.lobo_vnext_policy import (
    AutomaticCandidateMechanisms,
    CandidateDraft,
    CandidateInventory,
    ChunkerPolicy,
    ContentPolicySpec,
    LiteralCandidateMechanism,
    ManualDisposition,
    RawByteIdentityPolicy,
    StrictUTF8Policy,
    VersionedTextPolicy,
    Word5ContainmentPolicy,
)
from ..domain.lobo_vnext_real import (
    CampaignManifest,
    ModelRoleManifest,
)
from ..jsonio import dump_strict, loads_strict
from ..nlp import load_ner, load_sentencizer, resolved_nlp_identity
from ..pipeline.clean import normalize
from ..workdoc import CHUNKER_ALGORITHM, NORMALIZATION_CONTRACT
from .lobo_vnext_models import build_r1_model_spec


R1_PACKET_SCHEMA_VERSION = R1_PACKET_MANIFEST_SCHEMA_VERSION
R1_SOURCE_NAME = "RuAA-Bench"
R1_SOURCE_VERSION = "1.0"
R1_SOURCE_MANIFEST_SHA256 = (
    "bc3f95bf09032d8cbbef4bca25ff4acee1d537df085c2ec6a941e865b149767d"
)
R1_SOURCE_BOOK_COUNT = 137
R1_SELECTED_BOOK_COUNT = 136
R1_AUTHOR_COUNT = 22
R1_EXCLUDED_WORK_ID = "turgenev/записки_охотника"
R1_COLLECTION_MEMBERS = (
    "turgenev/бирюк",
    "turgenev/певцы",
    "turgenev/хорь_и_калиныч",
)
R1_WORD5_THRESHOLD = Fraction(9, 10)
R1_WORD5_MIN_SHINGLES = 20
R1_WORD5_SAMPLE_SIZE = 64
R1_CHUNK_SIZE = 500
R1_MIN_WORDS = 200
R1_OVERLAP = 0.0
R1_BOOTSTRAP_SEED = 42
R1_BOOTSTRAP_ITERATIONS = 10_000
R1_CONFIDENCE_LEVEL = 0.95

_SOURCE_MANIFEST_KEYS = {
    "name",
    "version",
    "claim_status",
    "benchmark_role",
    "training_weighting",
    "task",
    "n_authors",
    "n_books",
    "legal",
    "authors",
    "dropped",
}
_HEX64 = frozenset("0123456789abcdef")


class R1PacketPreparationError(VNextContractError):
    """The R1 source or deterministic packet preparation is unsafe."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _HEX64 for char in value)
    ):
        raise R1PacketPreparationError(
            f"{label} must be 64 lowercase hex characters"
        )
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise R1PacketPreparationError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _safe_source_root(value: str | os.PathLike[str]) -> pathlib.Path:
    root = pathlib.Path(value)
    _reject_symlink_components(root, label="R1 source root")
    if root.is_symlink() or not root.is_dir():
        raise R1PacketPreparationError(
            "R1 source root must be a real directory"
        )
    return root.resolve(strict=True)


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise R1PacketPreparationError(
                f"{label} must not contain symlink components: {component}"
            )


def _source_catalog(
    source_root: pathlib.Path,
    manifest_path: pathlib.Path,
) -> tuple[
    tuple[WorkIdentity, ...],
    dict[str, dict[str, object]],
    tuple[RawInventoryEntry, ...],
    str,
]:
    _reject_symlink_components(
        manifest_path, label="legacy source manifest"
    )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise R1PacketPreparationError(
            "legacy source manifest must be a regular non-symlink file"
        )
    try:
        raw_manifest_bytes = manifest_path.read_bytes()
        manifest = loads_strict(raw_manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise R1PacketPreparationError(
            f"cannot strictly load legacy source manifest: {exc}"
        ) from exc
    if _sha256_bytes(raw_manifest_bytes) != R1_SOURCE_MANIFEST_SHA256:
        raise R1PacketPreparationError(
            "legacy source manifest bytes differ from the exact R1 evidence"
        )
    if type(manifest) is not dict or set(manifest) != _SOURCE_MANIFEST_KEYS:
        raise R1PacketPreparationError(
            "legacy source manifest has an unexpected exact schema"
        )
    expected_scalars = {
        "name": R1_SOURCE_NAME,
        "version": R1_SOURCE_VERSION,
        "claim_status": "exploratory_internal",
        "benchmark_role": "reproducible_cv_legacy_not_blind",
        "n_authors": R1_AUTHOR_COUNT,
        "n_books": R1_SOURCE_BOOK_COUNT,
    }
    for key, expected in expected_scalars.items():
        observed = manifest[key]
        if type(observed) is not type(expected) or observed != expected:
            raise R1PacketPreparationError(
                f"legacy source manifest {key!r} differs from R1 evidence"
            )
    authors = manifest["authors"]
    if type(authors) is not dict or len(authors) != R1_AUTHOR_COUNT:
        raise R1PacketPreparationError(
            "legacy source manifest authors are malformed"
        )
    inventory = inventory_raw_files(source_root)
    inventory_by_path = {row.relative_path: row for row in inventory}
    works: list[WorkIdentity] = []
    metadata: dict[str, dict[str, object]] = {}
    expected_paths: set[str] = set()
    for author_id in sorted(authors):
        author = authors[author_id]
        if type(author) is not dict or set(author) != {
            "death_year",
            "n_books",
            "books",
        }:
            raise R1PacketPreparationError(
                f"legacy author record is malformed: {author_id!r}"
            )
        books = author["books"]
        n_books = _exact_int(
            author["n_books"], f"authors.{author_id}.n_books", minimum=1
        )
        if type(books) is not list or len(books) != n_books:
            raise R1PacketPreparationError(
                f"legacy author book inventory mismatch: {author_id!r}"
            )
        seen_books: set[str] = set()
        for book in books:
            if type(book) is not dict or set(book) != {
                "book",
                "sha256",
                "words",
                "source",
            }:
                raise R1PacketPreparationError(
                    f"legacy book record is malformed: {author_id!r}"
                )
            book_id = book["book"]
            if (
                type(book_id) is not str
                or not book_id
                or "/" in book_id
                or book_id in seen_books
            ):
                raise R1PacketPreparationError(
                    f"legacy book id is malformed: {author_id!r}/{book_id!r}"
                )
            seen_books.add(book_id)
            work_id = f"{author_id}/{book_id}"
            relative_path = f"{work_id}.txt"
            expected_paths.add(relative_path)
            expected_sha = _exact_sha(
                book["sha256"], f"authors.{author_id}.{book_id}.sha256"
            )
            entry = inventory_by_path.get(relative_path)
            if entry is None or entry.sha256 != expected_sha:
                raise R1PacketPreparationError(
                    f"legacy source bytes differ for {work_id!r}"
                )
            if _exact_int(
                book["words"], f"authors.{author_id}.{book_id}.words", minimum=1
            ) < 1:
                raise AssertionError("unreachable positive word count")
            source = book["source"]
            if type(source) is not str or not source:
                raise R1PacketPreparationError(
                    f"legacy source description is empty: {work_id!r}"
                )
            kind = (
                "collection"
                if work_id == R1_EXCLUDED_WORK_ID
                else "work"
            )
            works.append(
                WorkIdentity.from_dict(
                    {
                        "work_id": work_id,
                        "author_id": author_id,
                        "edition_id": (
                            f"ruaa-v1:{author_id}:{book_id}:"
                            f"{expected_sha[:16]}"
                        ),
                        "source_id": (
                            "ruaa-v1-source:"
                            f"{canonical_sha256({'source': source})}"
                        ),
                        "work_kind": kind,
                        "raw_paths": [relative_path],
                    }
                )
            )
            metadata[work_id] = dict(book)
    if (
        len(works) != R1_SOURCE_BOOK_COUNT
        or set(inventory_by_path) != expected_paths
    ):
        raise R1PacketPreparationError(
            "legacy source missing/extra file inventory"
        )
    return (
        tuple(sorted(works, key=lambda row: row.work_id)),
        metadata,
        inventory,
        _sha256_bytes(raw_manifest_bytes),
    )


def _policy_documents(cfg: ConfigNode) -> dict[str, dict[str, object]]:
    if type(cfg) is not ConfigNode:
        raise R1PacketPreparationError("cfg must be exactly ConfigNode")
    required = {
        "language.code": "ru",
        "language.spacy_model": "ru_core_news_lg",
        "language.spacy_model_version": "3.8.0",
        "language.spacy_fallback": "ru_core_news_md",
        "chunking.chunk_size": R1_CHUNK_SIZE,
        "chunking.min_words": R1_MIN_WORDS,
        "chunking.overlap": R1_OVERLAP,
        "chunking.sentence_aware": True,
    }
    for path, expected in required.items():
        observed = cfg.get_path(path)
        if type(observed) is not type(expected) or observed != expected:
            raise R1PacketPreparationError(
                f"R1 config requires {path}={expected!r}, got {observed!r}"
            )
    requested_model = cfg.get_path("language.spacy_model")
    fallback_model = cfg.get_path("language.spacy_fallback")
    try:
        ner = load_ner(requested_model, fallback_model)
        ner_identity = resolved_nlp_identity(ner)
    except Exception as exc:
        raise R1PacketPreparationError(
            f"cannot resolve and verify the R1 NER pipeline: {exc}"
        ) from exc
    if ner_identity.package_version != cfg.get_path(
        "language.spacy_model_version"
    ):
        raise R1PacketPreparationError(
            "resolved R1 NER package version differs from the frozen config"
        )
    ner_identity_payload = {
        **ner_identity.to_dict(),
        "disabled_pipes": list(ner_identity.disabled_pipes),
        "active_pipes": list(ner_identity.active_pipes),
    }
    clean_source = pathlib.Path(__file__).resolve().parents[1] / "pipeline/clean.py"
    chunk_source = pathlib.Path(__file__).resolve().parents[1] / "chunking.py"
    documents = {
        "canonicalizer": {
            "schema_version": "stylo.lobo-vnext.canonicalizer-policy-doc.v1",
            "implementation_contract": NORMALIZATION_CONTRACT,
            "implementation_source_sha256": _sha256_file(clean_source),
            "encoding": "utf-8",
            "bom_disposition": "reject",
            "dash_normalization": True,
            "person_mask": "@",
            "person_model": "ru_core_news_lg",
            "person_model_version": "3.8.0",
            "person_model_fallback": "ru_core_news_md",
            "resolved_person_model_identity": ner_identity_payload,
            "yo_to_e": True,
            "historical_orthography_to_modern": True,
            "wiki_and_service_markup_cleanup": True,
            "quote_normalization": True,
            "garbage_filter": True,
            "whitespace_collapse": True,
            "ocr_correction": False,
        },
        "chunker": {
            "schema_version": "stylo.lobo-vnext.chunker-policy-doc.v1",
            "implementation_contract": CHUNKER_ALGORITHM,
            "implementation_source_sha256": _sha256_file(chunk_source),
            "sentencizer": "spacy.blank+rule_based_sentencizer",
            "spacy_version": spacy.__version__,
            "language": "ru",
            "sentence_aware": True,
            "chunk_size": R1_CHUNK_SIZE,
            "min_words": R1_MIN_WORDS,
            "overlap": R1_OVERLAP,
        },
        "ocr": {
            "schema_version": "stylo.lobo-vnext.ocr-policy-doc.v1",
            "disposition": "preserve",
            "automatic_correction": False,
            "manual_correction": False,
        },
    }
    return documents


def build_r1_content_policy(
    cfg: ConfigNode,
) -> tuple[ContentPolicySpec, dict[str, dict[str, object]]]:
    documents = _policy_documents(cfg)
    clean_digest = canonical_sha256(documents["canonicalizer"])
    chunk_digest = canonical_sha256(documents["chunker"])
    ocr_digest = canonical_sha256(documents["ocr"])
    transform_clean = lambda version: VersionedTextPolicy.from_dict(  # noqa: E731
        {
            "policy_version": version,
            "disposition": "transform_versioned",
            "policy_document_sha256": clean_digest,
        },
        label=version,
    )
    policy = ContentPolicySpec.build(
        policy_id="stylo.lobo-vnext.ruaa-r1-content.v1",
        raw_byte_identity=RawByteIdentityPolicy.from_dict(
            {
                "policy_version": "stylo.lobo-vnext.raw-bytes.v1",
                "identity_fields": [
                    "relative_path",
                    "byte_size",
                    "sha256",
                ],
                "digest_algorithm": "sha256",
            }
        ),
        strict_utf8=StrictUTF8Policy.from_dict(
            {
                "policy_version": "stylo.lobo-vnext.strict-utf8.v1",
                "encoding": "utf-8",
                "errors": "strict",
                "bom_disposition": "reject",
            }
        ),
        canonical_row_policy=transform_clean("stylo.clean.v1"),
        chunker_policy=ChunkerPolicy.from_dict(
            {
                "policy_version": "stylo.sent-chunks.v1",
                "mode": "external_versioned",
                "policy_document_sha256": chunk_digest,
            }
        ),
        yo_e_policy=transform_clean("stylo.clean.yo-to-e.v1"),
        historical_orthography_policy=transform_clean(
            "stylo.clean.depreform.v1"
        ),
        ocr_policy=VersionedTextPolicy.from_dict(
            {
                "policy_version": "stylo.ocr.preserve.v1",
                "disposition": "preserve",
                "policy_document_sha256": ocr_digest,
            },
            label="ocr_policy",
        ),
        markup_policy=transform_clean(
            "stylo.clean.markup-and-per-mask.v1"
        ),
        automatic_candidates=AutomaticCandidateMechanisms.from_dict(
            {
                "exact_duplicate": {
                    "policy_version": (
                        "stylo.lobo-vnext.literal-duplicate.v1"
                    ),
                    "comparison": "literal_bytes_equal",
                },
                "literal_contains": {
                    "policy_version": (
                        "stylo.lobo-vnext.literal-contains.v1"
                    ),
                    "comparison": "literal_byte_subsequence",
                },
                "word5_containment": {
                    "policy_version": CONTENT_OVERLAP_POLICY_VERSION,
                    "shingle_size": 5,
                    "comparison": "asymmetric_containment",
                    "threshold": {"numerator": 9, "denominator": 10},
                    "threshold_boundary": "inclusive",
                    "min_shingles": R1_WORD5_MIN_SHINGLES,
                    "sample_size": R1_WORD5_SAMPLE_SIZE,
                    "final_verification": (
                        "exact_intersection_authoritative"
                    ),
                },
            }
        ),
        manual_disposition_required=True,
    )
    return policy, documents


def _read_source_texts(
    source_root: pathlib.Path,
    works: Sequence[WorkIdentity],
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    groups: list[str] = []
    for work in works:
        if len(work.raw_paths) != 1:
            raise R1PacketPreparationError(
                "R1 source works must each have exactly one raw path"
            )
        path = source_root.joinpath(
            *pathlib.PurePosixPath(work.raw_paths[0]).parts
        )
        payload = path.read_bytes()
        if payload.startswith(b"\xef\xbb\xbf"):
            raise R1PacketPreparationError(
                f"R1 BOM-reject policy failed for {work.work_id!r}"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise R1PacketPreparationError(
                f"R1 source is not strict UTF-8: {work.work_id!r}"
            ) from exc
        if not text.strip():
            raise R1PacketPreparationError(
                f"R1 source work is empty: {work.work_id!r}"
            )
        texts.append(text)
        groups.append(work.work_id)
    return texts, groups


def _candidate_drafts(
    source_root: pathlib.Path,
    works: Sequence[WorkIdentity],
) -> tuple[tuple[CandidateDraft, ...], tuple[dict[str, object], ...]]:
    texts, groups = _read_source_texts(source_root, works)
    overlaps = find_cross_work_content_overlaps(
        texts,
        groups,
        containment_threshold=R1_WORD5_THRESHOLD,
        min_shingles=R1_WORD5_MIN_SHINGLES,
        sample_size=R1_WORD5_SAMPLE_SIZE,
    )
    observed_pairs = tuple(
        (row.left_work, row.right_work, row.kind) for row in overlaps
    )
    expected_pairs = tuple(
        (
            member,
            R1_EXCLUDED_WORK_ID,
            "word5_asymmetric_containment",
        )
        for member in R1_COLLECTION_MEMBERS
    )
    if observed_pairs != expected_pairs:
        raise R1PacketPreparationError(
            "R1 automatic candidate inventory differs from the "
            f"owner-reviewed three pairs: {observed_pairs!r}"
        )
    work_by_id = {work.work_id: work for work in works}
    drafts: list[CandidateDraft] = []
    evidence_rows: list[dict[str, object]] = []
    for overlap in overlaps:
        left = work_by_id[overlap.left_work]
        right = work_by_id[overlap.right_work]
        evidence = {
            "policy_version": CONTENT_OVERLAP_POLICY_VERSION,
            "edge_type": overlap.kind,
            "left_work_id": overlap.left_work,
            "right_work_id": overlap.right_work,
            "exact_containment_evidence": overlap.evidence,
            "reported_containment": overlap.containment,
            "threshold": {"numerator": 9, "denominator": 10},
            "threshold_boundary": "inclusive",
            "final_verification": "exact_intersection_authoritative",
            "left_raw_paths": list(left.raw_paths),
            "right_raw_paths": list(right.raw_paths),
            "owner_selected_relation": "collection_member",
            "owner_selected_corpus_action": (
                "exclude_collection_retain_constituent"
            ),
        }
        evidence_sha = canonical_sha256(evidence)
        evidence_rows.append({**evidence, "evidence_sha256": evidence_sha})
        decision_id = (
            "R1.collection-member."
            + canonical_sha256(
                {
                    "left": overlap.left_work,
                    "right": overlap.right_work,
                }
            )[:24]
        )
        manual = ManualDisposition.from_dict(
            {
                "decision_id": decision_id,
                "disposition": "same_component",
                "evidence_sha256": evidence_sha,
            }
        )
        drafts.append(
            CandidateDraft.build(
                candidate_id=f"R1.word5.{evidence_sha[:32]}",
                left_work_id=overlap.left_work,
                right_work_id=overlap.right_work,
                edge_type="word5_asymmetric_containment",
                origin="automatic",
                evidence_sha256=evidence_sha,
                disposition="same_component",
                manual_disposition=manual,
            )
        )
        drafts.append(
            CandidateDraft.build(
                candidate_id=f"R1.collection-member.{evidence_sha[:32]}",
                left_work_id=overlap.left_work,
                right_work_id=overlap.right_work,
                edge_type="collection_member",
                origin="manual",
                evidence_sha256=evidence_sha,
                disposition="same_component",
                manual_disposition=manual,
            )
        )
    return (
        tuple(sorted(drafts, key=lambda row: row.candidate_id)),
        tuple(evidence_rows),
    )


def _selected_works(
    works: Sequence[WorkIdentity],
) -> tuple[WorkIdentity, ...]:
    selected = tuple(
        work for work in works if work.work_id != R1_EXCLUDED_WORK_ID
    )
    if (
        len(selected) != R1_SELECTED_BOOK_COUNT
        or any(
            member not in {work.work_id for work in selected}
            for member in R1_COLLECTION_MEMBERS
        )
    ):
        raise R1PacketPreparationError("R1 exact 136-work selection drifted")
    counts: dict[str, int] = {}
    for work in selected:
        counts[work.author_id] = counts.get(work.author_id, 0) + 1
    if len(counts) != R1_AUTHOR_COUNT or min(counts.values()) < 2:
        raise R1PacketPreparationError(
            "R1 selection loses author-level LOBO support"
        )
    return selected


def _copy_selected_raw(
    *,
    source_root: pathlib.Path,
    raw_root: pathlib.Path,
    works: Sequence[WorkIdentity],
) -> None:
    raw_root.mkdir(parents=True, exist_ok=False)
    for work in works:
        relative_path = work.raw_paths[0]
        source = source_root.joinpath(
            *pathlib.PurePosixPath(relative_path).parts
        )
        target = raw_root.joinpath(
            *pathlib.PurePosixPath(relative_path).parts
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source.read_bytes() != target.read_bytes():
            raise R1PacketPreparationError(
                f"literal raw copy drifted for {work.work_id!r}"
            )


def _canonical_rows(
    *,
    cfg: ConfigNode,
    raw_root: pathlib.Path,
    packet_root: pathlib.Path,
    works: Sequence[WorkIdentity],
) -> tuple[CanonicalRowEntry, ...]:
    model = cfg.get_path("language.spacy_model")
    fallback = cfg.get_path("language.spacy_fallback")
    sentencizer = load_sentencizer("ru")
    rows: list[CanonicalRowEntry] = []
    for work in works:
        source_relative = work.raw_paths[0]
        source_path = raw_root.joinpath(
            *pathlib.PurePosixPath(source_relative).parts
        )
        source_bytes = source_path.read_bytes()
        raw_text = source_bytes.decode("utf-8", errors="strict")
        clean_text = normalize(raw_text, model, fallback)
        if not clean_text or clean_text != clean_text.strip():
            raise R1PacketPreparationError(
                f"canonicalizer produced an invalid row for {work.work_id!r}"
            )
        sentences = sentences_for_text(clean_text, sentencizer)
        if not sentences:
            raise R1PacketPreparationError(
                f"sentencizer produced no sentences for {work.work_id!r}"
            )
        chunks = make_sent_chunks(
            CombinedDoc(sentences),
            R1_CHUNK_SIZE,
            R1_MIN_WORDS,
            R1_OVERLAP,
        )
        if not chunks:
            raise R1PacketPreparationError(
                f"chunker produced no rows for {work.work_id!r}"
            )
        for ordinal, raw_chunk in enumerate(chunks):
            chunk = raw_chunk.strip()
            if not chunk:
                raise R1PacketPreparationError(
                    f"chunker produced an empty row for {work.work_id!r}"
                )
            relative = (
                f"canonical_rows/{work.work_id}/{ordinal:06d}.txt"
            )
            output = packet_root.joinpath(
                *pathlib.PurePosixPath(relative).parts
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = chunk.encode("utf-8")
            output.write_bytes(payload)
            rows.append(
                CanonicalRowEntry.from_dict(
                    {
                        "row_id": f"{work.work_id}#{ordinal:06d}",
                        "relative_path": relative,
                        "work_id": work.work_id,
                        "author_id": work.author_id,
                        "ordinal": ordinal,
                        "source_relative_path": source_relative,
                        "source_raw_sha256": _sha256_bytes(source_bytes),
                        "canonical_byte_size": len(payload),
                        "canonical_sha256": _sha256_bytes(payload),
                        "word_count": len(chunk.split()),
                    }
                )
            )
    return tuple(
        sorted(rows, key=lambda row: (row.work_id, row.ordinal, row.relative_path))
    )


@dataclasses.dataclass(frozen=True)
class PreparedR1Packet:
    root: pathlib.Path
    packet_manifest: R1PacketManifest
    content_policy: ContentPolicySpec
    source_selection_receipt: R1SourceSelectionReceipt
    source_candidate_inventory: CandidateInventory
    candidate_inventory: CandidateInventory
    corpus_manifest: CorpusVNextManifest
    content_manifest: ContentComponentManifest
    fold_manifest: object
    primary_model_spec: ModelSpec
    baseline_model_spec: ModelSpec
    inference_spec: InferenceSpec
    primary_inner_cv_plan: InnerCVPlan
    baseline_inner_cv_plan: InnerCVPlan
    model_role_manifest: ModelRoleManifest
    campaign_manifest: CampaignManifest
    representation_receipt: CanonicalRepresentationReceipt


_ARTIFACT_FILENAMES = {
    "content_policy": "manifests/content-policy.json",
    "source_selection_receipt": "manifests/source-selection.json",
    "source_candidate_inventory": (
        "manifests/source-candidate-inventory.json"
    ),
    "candidate_inventory": "manifests/candidate-inventory.json",
    "corpus_manifest": "manifests/corpus.json",
    "content_manifest": "manifests/content-components.json",
    "fold_manifest": "manifests/folds.json",
    "primary_model_spec": "manifests/model-primary.json",
    "baseline_model_spec": "manifests/model-baseline.json",
    "inference_spec": "manifests/inference.json",
    "primary_inner_cv_plan": "manifests/inner-primary.json",
    "baseline_inner_cv_plan": "manifests/inner-baseline.json",
    "model_role_manifest": "manifests/model-roles.json",
    "campaign_manifest": "manifests/campaign.json",
    "representation_receipt": "manifests/representation.json",
}


def _write_json(root: pathlib.Path, relative: str, value: object) -> None:
    target = root.joinpath(*pathlib.PurePosixPath(relative).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    to_dict = getattr(value, "to_dict", None)
    raw = to_dict() if callable(to_dict) else value
    dump_strict(raw, target, trailing_newline=True)


def _inventory_packet_files(root: pathlib.Path) -> tuple[PacketFileEntry, ...]:
    if root.is_symlink() or not root.is_dir():
        raise R1PacketPreparationError(
            "staged R1 packet root must be a real directory"
        )
    rows: list[PacketFileEntry] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        for child in sorted(os.scandir(directory), key=lambda row: row.name):
            metadata = child.stat(follow_symlinks=False)
            path = pathlib.Path(child.path)
            if stat.S_ISLNK(metadata.st_mode):
                raise R1PacketPreparationError(
                    f"symlink rejected in staged R1 packet: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                stack.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise R1PacketPreparationError(
                    f"special file rejected in staged R1 packet: {path}"
                )
            relative = path.relative_to(root).as_posix()
            if relative == "packet.json":
                raise R1PacketPreparationError(
                    "packet.json unexpectedly existed before publication"
                )
            rows.append(
                PacketFileEntry(
                    relative,
                    metadata.st_size,
                    _sha256_file(path),
                )
            )
    return tuple(sorted(rows, key=lambda row: row.relative_path))


def prepare_r1_packet(
    *,
    source_root: str | os.PathLike[str],
    legacy_source_manifest: str | os.PathLike[str],
    output_parent: str | os.PathLike[str],
    cfg: ConfigNode,
) -> PreparedR1Packet:
    """Prepare the exact owner-selected packet without constructing a model."""

    source = _safe_source_root(source_root)
    manifest_path = pathlib.Path(legacy_source_manifest)
    output = pathlib.Path(output_parent)
    _reject_symlink_components(output, label="packet output parent")
    if output.is_symlink():
        raise R1PacketPreparationError("packet output parent is symlinked")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    lowered_parts = tuple(part.lower() for part in output.parts)
    required_namespace = ("exploratory", "lobo_vnext", "packets")
    if not any(
        lowered_parts[index : index + len(required_namespace)]
        == required_namespace
        for index in range(
            len(lowered_parts) - len(required_namespace) + 1
        )
    ):
        raise R1PacketPreparationError(
            "R1 packets must stay in an explicit "
            "exploratory/lobo_vnext/packets namespace"
        )
    works, _metadata, source_inventory, source_manifest_sha = _source_catalog(
        source, manifest_path
    )
    policy, documents = build_r1_content_policy(cfg)
    stage = pathlib.Path(
        tempfile.mkdtemp(prefix=".r1-source-snapshot.", dir=output)
    )
    try:
        snapshot_root = stage / ".source_snapshot"
        _copy_selected_raw(
            source_root=source,
            raw_root=snapshot_root,
            works=works,
        )
        snapshot_inventory = inventory_raw_files(snapshot_root)
        if snapshot_inventory != source_inventory:
            raise R1PacketPreparationError(
                "R1 source changed while the immutable snapshot was created"
            )
        source_candidate_drafts, candidate_evidence = _candidate_drafts(
            snapshot_root, works
        )
        selected = _selected_works(works)
        source_inventory_digest = canonical_sha256(
            [row.to_dict() for row in snapshot_inventory]
        )
        source_work_catalog_digest = canonical_sha256(
            [work.to_dict() for work in works]
        )
        selected_paths = {
            relative_path
            for work in selected
            for relative_path in work.raw_paths
        }
        selected_inventory = tuple(
            row
            for row in snapshot_inventory
            if row.relative_path in selected_paths
        )
        if (
            len(selected_inventory) != len(selected_paths)
            or {row.relative_path for row in selected_inventory}
            != selected_paths
        ):
            raise R1PacketPreparationError(
                "R1 selected raw inventory does not exactly match "
                "selected works"
            )
        selected_inventory_digest = canonical_sha256(
            [row.to_dict() for row in selected_inventory]
        )
        selected_work_catalog_digest = canonical_sha256(
            [work.to_dict() for work in selected]
        )
        if any(
            candidate.left_work_id != R1_EXCLUDED_WORK_ID
            and candidate.right_work_id != R1_EXCLUDED_WORK_ID
            for candidate in source_candidate_drafts
        ):
            raise R1PacketPreparationError(
                "R1 selected works retain an automatic content candidate"
            )
        generation_material = R1GenerationMaterial.from_dict(
            {
                "schema_version": (
                    R1_GENERATION_MATERIAL_SCHEMA_VERSION
                ),
                "source_manifest_sha256": source_manifest_sha,
                "source_raw_inventory_digest": source_inventory_digest,
                "source_work_identity_catalog_digest": (
                    source_work_catalog_digest
                ),
                "selected_raw_inventory_digest": (
                    selected_inventory_digest
                ),
                "selected_work_identity_catalog_digest": (
                    selected_work_catalog_digest
                ),
                "content_policy_spec_digest": policy.self_hash,
                "selected_work_ids": [
                    work.work_id for work in selected
                ],
                "excluded_work_ids": [R1_EXCLUDED_WORK_ID],
                "source_candidate_draft_digest": canonical_sha256(
                    [row.to_dict() for row in source_candidate_drafts]
                ),
                "candidate_evidence_digest": canonical_sha256(
                    list(candidate_evidence)
                ),
            }
        )
        generation_id = generation_material.generation_id
        source_candidate_inventory = CandidateInventory.build(
            generation_id=generation_id,
            work_identity_catalog_digest=source_work_catalog_digest,
            raw_inventory_digest=source_inventory_digest,
            content_policy_spec_digest=policy.self_hash,
            included_work_ids=tuple(work.work_id for work in works),
            candidates=source_candidate_drafts,
        )
        source_candidate_inventory.validate(content_policy_spec=policy)
        source_candidate_inventory.assert_resolved_for_component_manifest()
        source_selection_receipt = R1SourceSelectionReceipt.build(
            generation_material=generation_material,
            source_candidate_inventory_sha256=(
                source_candidate_inventory.self_hash
            ),
            source_raw_inventory=snapshot_inventory,
            source_works=works,
        )
        candidate_inventory = CandidateInventory.build(
            generation_id=generation_id,
            work_identity_catalog_digest=selected_work_catalog_digest,
            raw_inventory_digest=selected_inventory_digest,
            content_policy_spec_digest=policy.self_hash,
            included_work_ids=tuple(work.work_id for work in selected),
            candidates=(),
        )
        candidate_inventory.validate(content_policy_spec=policy)
        candidate_inventory.assert_resolved_for_component_manifest()
        packet_root = output / generation_id
        if packet_root.exists():
            raise R1PacketPreparationError(
                f"immutable R1 packet already exists: {packet_root}"
            )
        raw_root = stage / "raw"
        _copy_selected_raw(
            source_root=snapshot_root, raw_root=raw_root, works=selected
        )
        if inventory_raw_files(raw_root) != selected_inventory:
            raise R1PacketPreparationError(
                "R1 selected raw copy differs from its immutable snapshot"
            )
        shutil.rmtree(snapshot_root)
        content_manifest = ContentComponentManifest.build(
            automatic_candidate_policy_version=policy.self_hash,
            works=selected,
            components=tuple(
                ContentComponent(
                    f"component-{canonical_sha256({'work_id': work.work_id})}",
                    (work.work_id,),
                )
                for work in selected
            ),
            candidates=(),
        )
        content_manifest.validate(works=selected)
        if (
            len(content_manifest.components) != len(selected)
            or any(
                len(component.work_ids) != 1
                for component in content_manifest.components
            )
        ):
            raise R1PacketPreparationError(
                "R1 isolated mode requires one singleton component per work"
            )
        # This is the last pre-representation boundary.  The raw snapshot,
        # policy, complete source candidate screen, selected candidate
        # inventory, work identities, components, and author support have all
        # been frozen and validated.  No cache/factory/fit is reachable here.
        canonical_rows = _canonical_rows(
            cfg=cfg,
            raw_root=raw_root,
            packet_root=stage,
            works=selected,
        )
        canonical_digest = canonical_sha256(
            [row.to_dict() for row in canonical_rows]
        )
        corpus_manifest = build_corpus_vnext_manifest(
            raw_root,
            corpus_kind="real_corpus",
            generation_id=generation_id,
            approved_for_exploratory=True,
            owner_selected=True,
            author_ids=tuple(
                sorted({work.author_id for work in selected})
            ),
            works=selected,
            canonical_model_row_digest=canonical_digest,
            chunker_policy_version=policy.chunker_policy.policy_version,
            canonicalizer_policy_version=(
                policy.canonical_row_policy.policy_version
            ),
            content_policy_version=policy.self_hash,
            content_component_manifest_digest=content_manifest.self_hash,
        )
        if (
            canonical_sha256(
                [row.to_dict() for row in corpus_manifest.raw_inventory]
            )
            != selected_inventory_digest
        ):
            raise R1PacketPreparationError(
                "copied R1 raw inventory differs from the selected source bytes"
            )
        content_manifest.validate(works=corpus_manifest.works)
        fold_manifest = build_fold_manifest(
            corpus_manifest, content_manifest, mode="isolated"
        )
        primary_model = build_r1_model_spec(role="primary", cfg=cfg)
        baseline_model = build_r1_model_spec(role="baseline", cfg=cfg)
        inference = InferenceSpec.build(
            primary_metric="book_accuracy",
            primary_uncertainty="author_clustered_percentile_bootstrap",
            secondary_metrics=("macro_f1", "top2", "per_author"),
            macro_f1_uncertainty="point_only",
            bootstrap_seed=R1_BOOTSTRAP_SEED,
            bootstrap_iterations=R1_BOOTSTRAP_ITERATIONS,
            confidence_level=R1_CONFIDENCE_LEVEL,
            approved_for_exploratory=True,
            owner_selected=True,
        )
        primary_inner = build_inner_cv_plan(
            fold_manifest,
            corpus_manifest,
            content_manifest,
            primary_model,
        )
        baseline_inner = build_inner_cv_plan(
            fold_manifest,
            corpus_manifest,
            content_manifest,
            baseline_model,
        )
        model_roles = ModelRoleManifest.build(
            primary_model_spec=primary_model,
            baseline_model_spec=baseline_model,
            primary_inner_cv_plan=primary_inner,
            baseline_inner_cv_plan=baseline_inner,
        )
        campaign = CampaignManifest.build(
            campaign_id=f"ruaa-r1-{generation_id[:24]}",
            fold_manifest_digest=fold_manifest.self_hash,
            inference_spec_digest=inference.self_hash,
            model_role_manifest=model_roles,
        )
        representation = CanonicalRepresentationReceipt.build(
            generation_id=generation_id,
            corpus_manifest_sha256=corpus_manifest.self_hash,
            canonicalizer_policy_document_sha256=canonical_sha256(
                documents["canonicalizer"]
            ),
            chunker_policy_document_sha256=canonical_sha256(
                documents["chunker"]
            ),
            rows=canonical_rows,
        )
        representation.validate(corpus_manifest=corpus_manifest)
        objects = {
            "content_policy": policy,
            "source_selection_receipt": source_selection_receipt,
            "source_candidate_inventory": source_candidate_inventory,
            "candidate_inventory": candidate_inventory,
            "corpus_manifest": corpus_manifest,
            "content_manifest": content_manifest,
            "fold_manifest": fold_manifest,
            "primary_model_spec": primary_model,
            "baseline_model_spec": baseline_model,
            "inference_spec": inference,
            "primary_inner_cv_plan": primary_inner,
            "baseline_inner_cv_plan": baseline_inner,
            "model_role_manifest": model_roles,
            "campaign_manifest": campaign,
            "representation_receipt": representation,
        }
        for name, relative in _ARTIFACT_FILENAMES.items():
            _write_json(stage, relative, objects[name])
        for name, document in documents.items():
            _write_json(stage, f"policies/{name}.json", document)
        _write_json(stage, "candidates/evidence.json", list(candidate_evidence))
        packet_manifest = R1PacketManifest.build(
            generation_material=generation_material,
            source_candidate_inventory_sha256=(
                source_candidate_inventory.self_hash
            ),
            candidate_inventory_sha256=candidate_inventory.self_hash,
            corpus_manifest_sha256=corpus_manifest.self_hash,
            content_component_manifest_sha256=content_manifest.self_hash,
            fold_manifest_sha256=fold_manifest.self_hash,
            primary_model_spec_sha256=primary_model.self_hash,
            baseline_model_spec_sha256=baseline_model.self_hash,
            inference_spec_sha256=inference.self_hash,
            primary_inner_cv_plan_sha256=primary_inner.self_hash,
            baseline_inner_cv_plan_sha256=baseline_inner.self_hash,
            model_role_manifest_sha256=model_roles.self_hash,
            campaign_manifest_sha256=campaign.self_hash,
            representation_receipt_sha256=representation.self_hash,
            files=_inventory_packet_files(stage),
        )
        _write_json(stage, "packet.json", packet_manifest)
        try:
            os.rename(stage, packet_root)
        except FileExistsError as exc:
            raise R1PacketPreparationError(
                f"immutable R1 packet conflict: {packet_root}"
            ) from exc
        return PreparedR1Packet(
            root=packet_root,
            packet_manifest=packet_manifest,
            content_policy=policy,
            source_selection_receipt=source_selection_receipt,
            source_candidate_inventory=source_candidate_inventory,
            candidate_inventory=candidate_inventory,
            corpus_manifest=corpus_manifest,
            content_manifest=content_manifest,
            fold_manifest=fold_manifest,
            primary_model_spec=primary_model,
            baseline_model_spec=baseline_model,
            inference_spec=inference,
            primary_inner_cv_plan=primary_inner,
            baseline_inner_cv_plan=baseline_inner,
            model_role_manifest=model_roles,
            campaign_manifest=campaign,
            representation_receipt=representation,
        )
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


__all__ = [
    "PreparedR1Packet",
    "R1PacketPreparationError",
    "R1_AUTHOR_COUNT",
    "R1_BOOTSTRAP_ITERATIONS",
    "R1_BOOTSTRAP_SEED",
    "R1_CHUNK_SIZE",
    "R1_COLLECTION_MEMBERS",
    "R1_CONFIDENCE_LEVEL",
    "R1_EXCLUDED_WORK_ID",
    "R1_MIN_WORDS",
    "R1_OVERLAP",
    "R1_PACKET_SCHEMA_VERSION",
    "R1_SELECTED_BOOK_COUNT",
    "R1_SOURCE_BOOK_COUNT",
    "R1_SOURCE_MANIFEST_SHA256",
    "R1_WORD5_MIN_SHINGLES",
    "R1_WORD5_SAMPLE_SIZE",
    "R1_WORD5_THRESHOLD",
    "build_r1_content_policy",
    "prepare_r1_packet",
]
