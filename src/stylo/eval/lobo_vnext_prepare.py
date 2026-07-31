"""Deterministic preparation of the selected RuAA R1 acquisition packet.

This is a preparation surface, not an evaluator.  It exactly reloads the
immutable selected-134 acquisition, derives provider-bound work identities,
reruns the packet content-candidate screen, copies literal bytes, and derives
canonical clean/chunk rows.  No authorization, representation cache, estimator
factory, fit, prediction, or public evidence writer is reachable here.
"""
from __future__ import annotations

import ctypes
import dataclasses
import errno
import hashlib
import os
import pathlib
import shutil
import stat
import tempfile
from collections.abc import Sequence
from fractions import Fraction

import spacy

from ..chunking import CombinedDoc, make_sent_chunks, sentences_for_text
from ..config import ConfigNode
from ..corpus_tools.ruaa_r1_acquisition import (
    ACQUISITION_RECEIPT_NAME,
    AUDIT_REPORT_NAME,
    MANIFEST_NAME,
    MaterializedR1Acquisition,
    R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V3,
    R1_ACQUISITION_RECEIPT_SCHEMA_VERSION_V2,
    R1_EXCLUDED_WORK_IDS_V3,
    R1AcquisitionManifest,
    R1AcquisitionReceipt,
    load_materialized_r1_acquisition,
)
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
    build_fold_manifest,
    build_inner_cv_plan,
    canonical_sha256,
    inventory_raw_files,
)
from ..domain.lobo_vnext_packet import (
    CanonicalRepresentationReceipt,
    CanonicalRowEntry,
    PacketFileEntry,
    R1AcquisitionBinding,
    R1CorpusGenerationMaterial,
    R1PacketManifest,
    R1_PACKET_MANIFEST_SCHEMA_VERSION,
)
from ..domain.lobo_vnext_policy import (
    AutomaticCandidateMechanisms,
    CandidateInventory,
    ChunkerPolicy,
    ContentPolicySpec,
    RawByteIdentityPolicy,
    StrictUTF8Policy,
    VersionedTextPolicy,
    Word5ContainmentPolicy,
)
from ..domain.lobo_vnext_real import (
    CampaignManifest,
    ModelRoleManifest,
)
from ..jsonio import dump_strict, dumps_strict
from ..nlp import load_ner, load_sentencizer, resolved_nlp_identity
from ..pipeline.clean import normalize
from ..workdoc import CHUNKER_ALGORITHM, NORMALIZATION_CONTRACT
from .lobo_vnext_models import build_r1_model_spec


R1_PACKET_SCHEMA_VERSION = R1_PACKET_MANIFEST_SCHEMA_VERSION
R1_ACQUISITION_GENERATION_ID = (
    "7a930a56390ff8e310bfba75e35d028c3f260a2311a1f469dc687d235923ce4c"
)
R1_ACQUISITION_MANIFEST_SELF_HASH = (
    "7f6f8efba31c1c99d7b124708bb5331d5e5803a3ad9ebbafae21a6959646f209"
)
R1_ACQUISITION_RECEIPT_SELF_HASH = (
    "b745fbf1badc340042bcfc16757c4cbc8cd40d77f18f1e317849c0e134918e58"
)
R1_SELECTED_AUDIT_FILE_SHA256 = (
    "a6887053a928a687c4fc12607515fdb10a5aa99d3912e054d14edc5410e5408b"
)
R1_SELECTED_AUDIT_SELF_HASH = (
    "233326b39ef7dcf3593a3bd5607ef64dde72620d6cbe20f2a5f786cf32440780"
)
R1_RAW_INVENTORY_DIGEST = (
    "840bcc1d6f0c1521c9f2808773d298c720b70b8aad54e60aeb56cebc0fe76d65"
)
R1_WORK_IDENTITY_CATALOG_DIGEST = (
    "b3d0c30fea4668c8a6db2011d7cf61f390149cf42e7c65e76c7506e6d0941e2a"
)
R1_WORK_COUNT = 134
R1_AUTHOR_COUNT = 22
R1_UPSTREAM_EXCLUDED_WORK_IDS = R1_EXCLUDED_WORK_IDS_V3
R1_WORD5_THRESHOLD = Fraction(9, 10)
R1_WORD5_MIN_SHINGLES = 20
R1_WORD5_SAMPLE_SIZE = 64
R1_CHUNK_SIZE = 500
R1_MIN_WORDS = 200
R1_OVERLAP = 0.0
R1_BOOTSTRAP_SEED = 42
R1_BOOTSTRAP_ITERATIONS = 10_000
R1_CONFIDENCE_LEVEL = 0.95

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


def _publish_directory_no_replace(
    source: pathlib.Path,
    target: pathlib.Path,
) -> None:
    """Atomically publish one directory without replacing any target."""

    if os.name != "posix":
        raise R1PacketPreparationError(
            "atomic no-clobber packet publication is unavailable"
        )
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise R1PacketPreparationError(
            "atomic no-clobber packet publication is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,  # AT_FDCWD
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise R1PacketPreparationError(
            f"immutable R1 packet conflict: {target}"
        )
    raise R1PacketPreparationError(
        "atomic no-clobber packet publication failed: "
        f"{os.strerror(error_number)}"
    )


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise R1PacketPreparationError(
                f"{label} must not contain symlink components: {component}"
            )


def _validate_canonical_acquisition(
    acquisition: MaterializedR1Acquisition,
) -> None:
    if (
        type(acquisition) is not MaterializedR1Acquisition
        or type(acquisition.manifest) is not R1AcquisitionManifest
        or type(acquisition.receipt) is not R1AcquisitionReceipt
    ):
        raise R1PacketPreparationError(
            "packet input must be exactly MaterializedR1Acquisition"
        )
    manifest = acquisition.manifest
    receipt = acquisition.receipt
    audit = acquisition.audit_report.to_dict()
    observed = {
        "manifest_schema": manifest.schema_version,
        "receipt_schema": receipt.schema_version,
        "generation_id": manifest.generation_id,
        "manifest_self_hash": manifest.self_hash,
        "receipt_self_hash": receipt.self_hash,
        "audit_file_sha256": _sha256_file(
            acquisition.root / AUDIT_REPORT_NAME
        ),
        "audit_self_hash": audit.get("self_hash"),
        "work_count": len(receipt.raw_inventory),
        "excluded_work_ids": tuple(
            row.work_id for row in manifest.exclusions
        ),
    }
    expected = {
        "manifest_schema": R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V3,
        "receipt_schema": R1_ACQUISITION_RECEIPT_SCHEMA_VERSION_V2,
        "generation_id": R1_ACQUISITION_GENERATION_ID,
        "manifest_self_hash": R1_ACQUISITION_MANIFEST_SELF_HASH,
        "receipt_self_hash": R1_ACQUISITION_RECEIPT_SELF_HASH,
        "audit_file_sha256": R1_SELECTED_AUDIT_FILE_SHA256,
        "audit_self_hash": R1_SELECTED_AUDIT_SELF_HASH,
        "work_count": R1_WORK_COUNT,
        "excluded_work_ids": R1_UPSTREAM_EXCLUDED_WORK_IDS,
    }
    if observed != expected:
        raise R1PacketPreparationError(
            "R1 acquisition differs from canonical selected-134 identity"
        )
    if (
        receipt.generation_id != manifest.generation_id
        or receipt.included_work_ids != manifest.included_work_ids
        or audit.get("status") != "passed"
        or audit.get("blocking_findings") != []
        or audit.get("cross_work_overlaps") != []
        or audit.get("work_count") != R1_WORK_COUNT
    ):
        raise R1PacketPreparationError(
            "R1 acquisition quality/receipt boundary is not passed and exact"
        )


def _provider_work_specs(
    manifest: R1AcquisitionManifest,
) -> dict[str, tuple[str, object]]:
    rows: dict[str, list[tuple[str, object]]] = {}

    def add(kind: str, work_id: str, spec: object) -> None:
        rows.setdefault(work_id, []).append((kind, spec))

    for spec in manifest.wikisource_campaign.works:
        add("wikisource", spec.work_id, spec)
    add("feb", manifest.feb_work_spec.work_id, manifest.feb_work_spec)
    reviewed = manifest.reviewed_text_campaign
    if reviewed is not None:
        for spec in reviewed.works:
            add("reviewed-text", spec.work_id, spec)
    ambiguous = sorted(
        work_id for work_id, providers in rows.items()
        if len(providers) != 1
    )
    if ambiguous or set(rows) != set(manifest.included_work_ids):
        raise R1PacketPreparationError(
            "R1 provider work-spec inventory is ambiguous or incomplete"
        )
    known = {"wikisource", "feb", "reviewed-text"}
    resolved = {
        work_id: providers[0] for work_id, providers in rows.items()
    }
    if any(kind not in known for kind, _spec in resolved.values()):
        raise R1PacketPreparationError("unknown R1 provider kind")
    return resolved


def _acquisition_catalog(
    acquisition: MaterializedR1Acquisition,
) -> tuple[tuple[WorkIdentity, ...], tuple[RawInventoryEntry, ...]]:
    providers = _provider_work_specs(acquisition.manifest)
    raw_rows = tuple(
        RawInventoryEntry(
            row.relative_path,
            row.byte_size,
            row.sha256,
        )
        for row in acquisition.receipt.raw_inventory
    )
    raw_by_work = {
        row.work_id: row for row in acquisition.receipt.raw_inventory
    }
    works: list[WorkIdentity] = []
    for work_id in acquisition.manifest.included_work_ids:
        pure = pathlib.PurePosixPath(work_id)
        if (
            pure.as_posix() != work_id
            or len(pure.parts) != 2
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise R1PacketPreparationError(
                f"R1 acquisition work_id is malformed: {work_id!r}"
            )
        author_id = pure.parts[0]
        raw = raw_by_work.get(work_id)
        if raw is None or raw.relative_path != f"raw/{work_id}.txt":
            raise R1PacketPreparationError(
                f"R1 raw identity is missing for {work_id!r}"
            )
        provider_kind, provider_spec = providers[work_id]
        if provider_kind not in {"wikisource", "feb", "reviewed-text"}:
            raise R1PacketPreparationError(
                f"unknown R1 provider kind for {work_id!r}"
            )
        to_dict = getattr(provider_spec, "to_dict", None)
        if not callable(to_dict):
            raise R1PacketPreparationError(
                f"R1 provider work spec is not canonical for {work_id!r}"
            )
        provider_identity = canonical_sha256(to_dict())
        works.append(
            WorkIdentity.from_dict(
                {
                    "work_id": work_id,
                    "author_id": author_id,
                    "edition_id": (
                        "stylo.lobo-vnext.ruaa-r1-raw-edition.v1:"
                        f"sha256:{raw.sha256}"
                    ),
                    "source_id": (
                        "stylo.lobo-vnext.ruaa-r1-provider-work-spec.v1:"
                        f"{provider_kind}:{provider_identity}"
                    ),
                    "work_kind": "work",
                    "raw_paths": [raw.relative_path],
                }
            )
        )
    ordered = tuple(sorted(works, key=lambda row: row.work_id))
    if (
        len(ordered) != R1_WORK_COUNT
        or tuple(work.work_id for work in ordered)
        != acquisition.manifest.included_work_ids
        or tuple(row.relative_path for row in raw_rows)
        != tuple(sorted(row.relative_path for row in raw_rows))
    ):
        raise R1PacketPreparationError(
            "R1 acquisition work/raw catalog is noncanonical"
        )
    counts: dict[str, int] = {}
    for work in ordered:
        counts[work.author_id] = counts.get(work.author_id, 0) + 1
    if len(counts) != R1_AUTHOR_COUNT or min(counts.values()) < 2:
        raise R1PacketPreparationError(
            "R1 acquisition loses author-level LOBO support"
        )
    if set(R1_UPSTREAM_EXCLUDED_WORK_IDS) & {
        work.work_id for work in ordered
    }:
        raise R1PacketPreparationError(
            "R1 upstream exclusion leaked into selected acquisition"
        )
    return ordered, raw_rows


def _require_exact_catalog_digests(
    *,
    works: Sequence[WorkIdentity],
    raw_inventory: Sequence[RawInventoryEntry],
) -> tuple[str, str]:
    """Stop exact R1 catalog drift before content screening or NLP loading."""

    raw_inventory_digest = canonical_sha256(
        [row.to_dict() for row in raw_inventory]
    )
    work_catalog_digest = canonical_sha256(
        [work.to_dict() for work in works]
    )
    if raw_inventory_digest != R1_RAW_INVENTORY_DIGEST:
        raise R1PacketPreparationError(
            "R1 raw inventory digest differs from the exact production pin"
        )
    if work_catalog_digest != R1_WORK_IDENTITY_CATALOG_DIGEST:
        raise R1PacketPreparationError(
            "R1 WorkIdentity catalog digest differs from the exact production pin"
        )
    return raw_inventory_digest, work_catalog_digest


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
            "schema_version": "stylo.lobo-vnext.canonicalizer-policy-doc.v2",
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


def _screen_selected_content(
    source_root: pathlib.Path,
    works: Sequence[WorkIdentity],
) -> None:
    texts, groups = _read_source_texts(source_root, works)
    overlaps = find_cross_work_content_overlaps(
        texts,
        groups,
        containment_threshold=R1_WORD5_THRESHOLD,
        min_shingles=R1_WORD5_MIN_SHINGLES,
        sample_size=R1_WORD5_SAMPLE_SIZE,
    )
    if overlaps:
        sample = tuple(
            (row.left_work, row.right_work, row.kind)
            for row in overlaps
        )
        raise R1PacketPreparationError(
            "R1 selected-134 post-selection content screen found "
            f"candidates: {sample!r}"
        )


def _copy_acquisition_raw(
    *,
    source_root: pathlib.Path,
    packet_root: pathlib.Path,
    works: Sequence[WorkIdentity],
) -> None:
    raw_root = packet_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=False)
    for work in works:
        relative_path = work.raw_paths[0]
        source = source_root.joinpath(
            *pathlib.PurePosixPath(relative_path).parts
        )
        target = packet_root.joinpath(
            *pathlib.PurePosixPath(relative_path).parts
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source.read_bytes() != target.read_bytes():
            raise R1PacketPreparationError(
                f"literal raw copy drifted for {work.work_id!r}"
            )


def _packet_raw_inventory(
    packet_root: pathlib.Path,
) -> tuple[RawInventoryEntry, ...]:
    return tuple(
        RawInventoryEntry(
            f"raw/{row.relative_path}",
            row.byte_size,
            row.sha256,
        )
        for row in inventory_raw_files(packet_root / "raw")
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
    acquisition_binding: R1AcquisitionBinding
    corpus_generation_material: R1CorpusGenerationMaterial
    content_policy: ContentPolicySpec
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
_ACQUISITION_FILENAMES = {
    MANIFEST_NAME: f"acquisition/{MANIFEST_NAME}",
    ACQUISITION_RECEIPT_NAME: f"acquisition/{ACQUISITION_RECEIPT_NAME}",
    AUDIT_REPORT_NAME: f"acquisition/{AUDIT_REPORT_NAME}",
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
    acquisition_root: str | os.PathLike[str],
    output_parent: str | os.PathLike[str],
    cfg: ConfigNode,
) -> PreparedR1Packet:
    """Prepare the exact selected-134 packet without constructing a model."""

    try:
        acquisition = load_materialized_r1_acquisition(acquisition_root)
    except (OSError, UnicodeError, ValueError) as exc:
        raise R1PacketPreparationError(
            f"R1 acquisition failed exact validation: {exc}"
        ) from exc
    _validate_canonical_acquisition(acquisition)
    works, raw_inventory = _acquisition_catalog(acquisition)
    raw_inventory_digest, work_catalog_digest = (
        _require_exact_catalog_digests(
            works=works,
            raw_inventory=raw_inventory,
        )
    )
    _screen_selected_content(acquisition.root, works)
    policy, documents = build_r1_content_policy(cfg)
    binding = R1AcquisitionBinding.build(
        acquisition_generation_id=acquisition.manifest.generation_id,
        acquisition_manifest_self_hash=acquisition.manifest.self_hash,
        acquisition_receipt_self_hash=acquisition.receipt.self_hash,
        selected_audit_file_sha256=_sha256_file(
            acquisition.root / AUDIT_REPORT_NAME
        ),
        selected_audit_self_hash=acquisition.audit_report.self_hash,
        raw_inventory_digest=raw_inventory_digest,
        work_identity_catalog_digest=work_catalog_digest,
        upstream_excluded_work_ids=R1_UPSTREAM_EXCLUDED_WORK_IDS,
        content_policy_spec_digest=policy.self_hash,
        work_count=len(works),
        author_count=len({work.author_id for work in works}),
    )
    corpus_generation_material = R1CorpusGenerationMaterial.build(
        acquisition_binding_self_hash=binding.self_hash,
    )
    corpus_generation_id = corpus_generation_material.self_hash
    candidate_inventory = CandidateInventory.build(
        generation_id=corpus_generation_id,
        work_identity_catalog_digest=work_catalog_digest,
        raw_inventory_digest=raw_inventory_digest,
        content_policy_spec_digest=policy.self_hash,
        included_work_ids=tuple(work.work_id for work in works),
        candidates=(),
    )
    candidate_inventory.validate(content_policy_spec=policy)
    candidate_inventory.assert_resolved_for_component_manifest()

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
    stage = pathlib.Path(
        tempfile.mkdtemp(prefix=".r1-acquisition-packet.", dir=output)
    )
    try:
        _copy_acquisition_raw(
            source_root=acquisition.root,
            packet_root=stage,
            works=works,
        )
        if _packet_raw_inventory(stage) != raw_inventory:
            raise R1PacketPreparationError(
                "R1 packet raw copy differs from validated acquisition"
            )
        acquisition_values = {
            MANIFEST_NAME: acquisition.manifest.to_dict(),
            ACQUISITION_RECEIPT_NAME: acquisition.receipt.to_dict(),
            AUDIT_REPORT_NAME: acquisition.audit_report.to_dict(),
        }
        for source_name, packet_relative in _ACQUISITION_FILENAMES.items():
            source_path = acquisition.root / source_name
            target_path = stage.joinpath(
                *pathlib.PurePosixPath(packet_relative).parts
            )
            canonical_bytes = (
                dumps_strict(
                    acquisition_values[source_name],
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            if source_path.read_bytes() != canonical_bytes:
                raise R1PacketPreparationError(
                    f"R1 acquisition evidence is noncanonical: {source_name}"
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            if target_path.read_bytes() != canonical_bytes:
                raise R1PacketPreparationError(
                    f"R1 acquisition evidence copy drifted: {source_name}"
                )
        content_manifest = ContentComponentManifest.build(
            automatic_candidate_policy_version=policy.self_hash,
            works=works,
            components=tuple(
                ContentComponent(
                    f"component-{canonical_sha256({'work_id': work.work_id})}",
                    (work.work_id,),
                )
                for work in works
            ),
            candidates=(),
        )
        content_manifest.validate(works=works)
        if (
            len(content_manifest.components) != len(works)
            or any(
                len(component.work_ids) != 1
                for component in content_manifest.components
            )
        ):
            raise R1PacketPreparationError(
                "R1 isolated mode requires one singleton component per work"
            )
        # This is the last pre-representation boundary.  The exact acquisition,
        # zero-candidate screen, policy, identities, components, and author
        # support are frozen.  No authorization/cache/factory/fit is reachable.
        canonical_rows = _canonical_rows(
            cfg=cfg,
            raw_root=stage,
            packet_root=stage,
            works=works,
        )
        canonical_digest = canonical_sha256(
            [row.to_dict() for row in canonical_rows]
        )
        corpus_manifest = CorpusVNextManifest.build(
            corpus_kind="real_corpus",
            generation_id=corpus_generation_id,
            approved_for_exploratory=True,
            owner_selected=True,
            raw_inventory=raw_inventory,
            author_ids=tuple(
                sorted({work.author_id for work in works})
            ),
            works=works,
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
            != raw_inventory_digest
        ):
            raise R1PacketPreparationError(
                "copied R1 raw inventory differs from acquisition binding"
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
            campaign_id=f"ruaa-r1-{corpus_generation_id}",
            fold_manifest_digest=fold_manifest.self_hash,
            inference_spec_digest=inference.self_hash,
            model_role_manifest=model_roles,
        )
        representation = CanonicalRepresentationReceipt.build(
            generation_id=corpus_generation_id,
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
        packet_manifest = R1PacketManifest.build(
            acquisition_binding=binding,
            corpus_generation_material=corpus_generation_material,
            content_policy_spec_sha256=policy.self_hash,
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
        packet_root = output / packet_manifest.packet_generation_id
        if packet_root.exists() or packet_root.is_symlink():
            raise R1PacketPreparationError(
                f"immutable R1 packet already exists: {packet_root}"
            )
        _write_json(stage, "packet.json", packet_manifest)
        _publish_directory_no_replace(stage, packet_root)
        return PreparedR1Packet(
            root=packet_root,
            packet_manifest=packet_manifest,
            acquisition_binding=binding,
            corpus_generation_material=corpus_generation_material,
            content_policy=policy,
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
    "R1_ACQUISITION_GENERATION_ID",
    "R1_ACQUISITION_MANIFEST_SELF_HASH",
    "R1_ACQUISITION_RECEIPT_SELF_HASH",
    "R1_AUTHOR_COUNT",
    "R1_BOOTSTRAP_ITERATIONS",
    "R1_BOOTSTRAP_SEED",
    "R1_CHUNK_SIZE",
    "R1_CONFIDENCE_LEVEL",
    "R1_MIN_WORDS",
    "R1_OVERLAP",
    "R1_PACKET_SCHEMA_VERSION",
    "R1_RAW_INVENTORY_DIGEST",
    "R1_SELECTED_AUDIT_FILE_SHA256",
    "R1_SELECTED_AUDIT_SELF_HASH",
    "R1_UPSTREAM_EXCLUDED_WORK_IDS",
    "R1_WORK_IDENTITY_CATALOG_DIGEST",
    "R1_WORK_COUNT",
    "R1_WORD5_MIN_SHINGLES",
    "R1_WORD5_SAMPLE_SIZE",
    "R1_WORD5_THRESHOLD",
    "build_r1_content_policy",
    "prepare_r1_packet",
]
