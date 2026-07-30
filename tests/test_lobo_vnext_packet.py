from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylo.domain.lobo_vnext import (
    LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION,
    ContentComponent,
    ContentComponentManifest,
    WorkIdentity,
    build_corpus_vnext_manifest,
    canonical_sha256,
)
from stylo.domain.lobo_vnext_packet import (
    CanonicalRepresentationReceipt,
    CanonicalRowEntry,
    PacketFileEntry,
    R1AcquisitionBinding,
    R1PacketManifest,
    R1_ACQUISITION_BINDING_SCHEMA_VERSION,
    R1_PACKET_MANIFEST_SCHEMA_VERSION,
    VNextPacketError,
    load_canonical_representation_rows,
    loads_canonical_representation_receipt,
)


_UPSTREAM_EXCLUSIONS = (
    "serafimovich/у_нас_и_у_них",
    "sevsky/дон_на_костылях",
    "turgenev/записки_охотника",
)


def _work(work_id: str) -> WorkIdentity:
    author, leaf = work_id.split("/", 1)
    return WorkIdentity.from_dict(
        {
            "work_id": work_id,
            "author_id": author,
            "edition_id": f"edition-{leaf}",
            "source_id": f"source-{leaf}",
            "work_kind": "work",
            "raw_paths": [f"{work_id}.txt"],
        }
    )


def _packet(tmp_path: Path):
    raw_root = tmp_path / "raw"
    packet_root = tmp_path / "packet"
    works = (_work("a/a1"), _work("a/a2"), _work("b/b1"), _work("b/b2"))
    for work in works:
        path = raw_root / work.raw_paths[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"literal source {work.work_id}\n", encoding="utf-8")
    content = ContentComponentManifest.build(
        automatic_candidate_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        works=works,
        components=tuple(
            ContentComponent(f"component-{work.work_id}", (work.work_id,))
            for work in works
        ),
        candidates=(),
    )
    entries: list[CanonicalRowEntry] = []
    for work in works:
        text = f"canonical model row {work.work_id}"
        relative = f"canonical_rows/{work.work_id}/000000.txt"
        output = packet_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        source = raw_root / work.raw_paths[0]
        source_bytes = source.read_bytes()
        canonical_bytes = text.encode("utf-8")
        entries.append(
            CanonicalRowEntry.from_dict(
                {
                    "row_id": f"{work.work_id}#000000",
                    "relative_path": relative,
                    "work_id": work.work_id,
                    "author_id": work.author_id,
                    "ordinal": 0,
                    "source_relative_path": work.raw_paths[0],
                    "source_raw_sha256": __import__("hashlib").sha256(
                        source_bytes
                    ).hexdigest(),
                    "canonical_byte_size": len(canonical_bytes),
                    "canonical_sha256": __import__("hashlib").sha256(
                        canonical_bytes
                    ).hexdigest(),
                    "word_count": len(text.split()),
                }
            )
        )
    entries.sort(key=lambda row: (row.work_id, row.ordinal, row.relative_path))
    row_digest = canonical_sha256([entry.to_dict() for entry in entries])
    corpus = build_corpus_vnext_manifest(
        raw_root,
        corpus_kind="real_corpus",
        generation_id="test-real-generation",
        approved_for_exploratory=True,
        owner_selected=True,
        author_ids=("a", "b"),
        works=works,
        canonical_model_row_digest=row_digest,
        chunker_policy_version="stylo.sent-chunks.v1",
        canonicalizer_policy_version="stylo.clean.v1",
        content_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        content_component_manifest_digest=content.self_hash,
    )
    receipt = CanonicalRepresentationReceipt.build(
        generation_id=corpus.generation_id,
        corpus_manifest_sha256=corpus.self_hash,
        canonicalizer_policy_document_sha256=canonical_sha256(
            {"policy": "clean"}
        ),
        chunker_policy_document_sha256=canonical_sha256(
            {"policy": "chunker"}
        ),
        rows=entries,
    )
    receipt.validate(corpus_manifest=corpus)
    return raw_root, packet_root, corpus, receipt


def _rehash(raw: dict) -> dict:
    raw["self_hash"] = canonical_sha256(
        {key: child for key, child in raw.items() if key != "self_hash"}
    )
    return raw


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _acquisition_binding() -> R1AcquisitionBinding:
    return R1AcquisitionBinding.build(
        generation_id=_digest("acquisition-generation"),
        acquisition_manifest_self_hash=_digest("acquisition-manifest"),
        acquisition_receipt_self_hash=_digest("acquisition-receipt"),
        selected_audit_file_sha256=_digest("selected-audit-file"),
        selected_audit_self_hash=_digest("selected-audit-self"),
        raw_inventory_digest=_digest("raw-inventory"),
        work_identity_catalog_digest=_digest("work-identities"),
        upstream_excluded_work_ids=_UPSTREAM_EXCLUSIONS,
        content_policy_spec_digest=_digest("content-policy"),
        post_selection_candidate_inventory_sha256=_digest(
            "candidate-inventory"
        ),
        work_count=134,
        author_count=22,
    )


def _r1_packet_manifest() -> R1PacketManifest:
    binding = _acquisition_binding()
    return R1PacketManifest.build(
        acquisition_binding=binding,
        candidate_inventory_sha256=(
            binding.post_selection_candidate_inventory_sha256
        ),
        corpus_manifest_sha256=_digest("corpus"),
        content_component_manifest_sha256=_digest("content-components"),
        fold_manifest_sha256=_digest("folds"),
        primary_model_spec_sha256=_digest("primary-model"),
        baseline_model_spec_sha256=_digest("baseline-model"),
        inference_spec_sha256=_digest("inference"),
        primary_inner_cv_plan_sha256=_digest("primary-inner"),
        baseline_inner_cv_plan_sha256=_digest("baseline-inner"),
        model_role_manifest_sha256=_digest("model-roles"),
        campaign_manifest_sha256=_digest("campaign"),
        representation_receipt_sha256=_digest("representation"),
        files=(
            PacketFileEntry(
                "manifests/acquisition-manifest.json",
                1,
                _digest("acquisition-manifest-file"),
            ),
        ),
    )


def test_acquisition_binding_is_strict_self_hashed_selected_134_contract():
    binding = _acquisition_binding()

    assert binding.schema_version == R1_ACQUISITION_BINDING_SCHEMA_VERSION
    assert binding.work_count == 134
    assert binding.author_count == 22
    assert binding.upstream_excluded_work_ids == _UPSTREAM_EXCLUSIONS
    assert (
        binding.post_selection_candidate_inventory_sha256
        == _digest("candidate-inventory")
    )
    assert R1AcquisitionBinding.from_dict(binding.to_dict()) == binding
    assert binding.validate() is binding


def test_acquisition_binding_rejects_schema_shape_hash_and_count_drift():
    base = _acquisition_binding().to_dict()

    legacy = json.loads(json.dumps(base))
    legacy["schema_version"] = (
        "stylo.lobo-vnext.ruaa-r1-acquisition-binding.v0"
    )
    _rehash(legacy)
    with pytest.raises(VNextPacketError, match="legacy or unsupported"):
        R1AcquisitionBinding.from_dict(legacy)

    extra = json.loads(json.dumps(base))
    extra["compatibility_mode"] = False
    _rehash(extra)
    with pytest.raises(VNextPacketError, match="keys must be exact"):
        R1AcquisitionBinding.from_dict(extra)

    tampered = json.loads(json.dumps(base))
    tampered["raw_inventory_digest"] = "0" * 64
    with pytest.raises(VNextPacketError, match="self_hash mismatch"):
        R1AcquisitionBinding.from_dict(tampered)

    unsorted = json.loads(json.dumps(base))
    unsorted["upstream_excluded_work_ids"].reverse()
    _rehash(unsorted)
    with pytest.raises(VNextPacketError, match="sorted and unique"):
        R1AcquisitionBinding.from_dict(unsorted)

    bool_count = json.loads(json.dumps(base))
    bool_count["work_count"] = True
    _rehash(bool_count)
    with pytest.raises(VNextPacketError, match="exact integer"):
        R1AcquisitionBinding.from_dict(bool_count)


def test_packet_v3_binds_acquisition_and_post_selection_inventory():
    packet = _r1_packet_manifest()

    assert packet.schema_version == R1_PACKET_MANIFEST_SCHEMA_VERSION
    assert packet.schema_version.endswith(".v3")
    assert packet.generation_id == packet.acquisition_binding.generation_id
    assert packet.selected_work_count == 134
    assert packet.confirmatory_authorized is False
    bound_candidates = (
        packet.acquisition_binding.post_selection_candidate_inventory_sha256
    )
    assert (
        packet.candidate_inventory_sha256
        == bound_candidates
    )
    assert R1PacketManifest.from_dict(packet.to_dict()) == packet
    assert packet.validate() is packet

    with pytest.raises(
        VNextPacketError,
        match="candidate inventory differs from acquisition binding",
    ):
        R1PacketManifest.build(
            acquisition_binding=packet.acquisition_binding,
            candidate_inventory_sha256=_digest("wrong-candidates"),
            corpus_manifest_sha256=packet.corpus_manifest_sha256,
            content_component_manifest_sha256=(
                packet.content_component_manifest_sha256
            ),
            fold_manifest_sha256=packet.fold_manifest_sha256,
            primary_model_spec_sha256=packet.primary_model_spec_sha256,
            baseline_model_spec_sha256=packet.baseline_model_spec_sha256,
            inference_spec_sha256=packet.inference_spec_sha256,
            primary_inner_cv_plan_sha256=(
                packet.primary_inner_cv_plan_sha256
            ),
            baseline_inner_cv_plan_sha256=(
                packet.baseline_inner_cv_plan_sha256
            ),
            model_role_manifest_sha256=(
                packet.model_role_manifest_sha256
            ),
            campaign_manifest_sha256=packet.campaign_manifest_sha256,
            representation_receipt_sha256=(
                packet.representation_receipt_sha256
            ),
            files=packet.files,
        )


def test_packet_v3_rejects_rehashed_cross_binding_and_shape_drift():
    base = _r1_packet_manifest().to_dict()

    generation = json.loads(json.dumps(base))
    generation["generation_id"] = "0" * 64
    _rehash(generation)
    with pytest.raises(
        VNextPacketError,
        match="generation_id differs from acquisition",
    ):
        R1PacketManifest.from_dict(generation)

    candidates = json.loads(json.dumps(base))
    candidates["candidate_inventory_sha256"] = "0" * 64
    _rehash(candidates)
    with pytest.raises(
        VNextPacketError,
        match="candidate inventory differs from acquisition binding",
    ):
        R1PacketManifest.from_dict(candidates)

    count = json.loads(json.dumps(base))
    count["selected_work_count"] = 133
    _rehash(count)
    with pytest.raises(VNextPacketError, match="selected work count mismatch"):
        R1PacketManifest.from_dict(count)

    extra = json.loads(json.dumps(base))
    extra["legacy_generation_material"] = {}
    _rehash(extra)
    with pytest.raises(VNextPacketError, match="keys must be exact"):
        R1PacketManifest.from_dict(extra)


def test_packet_v2_schema_is_explicitly_rejected_as_legacy():
    historical = _r1_packet_manifest().to_dict()
    historical["schema_version"] = "stylo.lobo-vnext.ruaa-r1-packet.v2"
    _rehash(historical)

    with pytest.raises(VNextPacketError, match="legacy or unsupported"):
        R1PacketManifest.from_dict(historical)


def test_canonical_rows_are_separate_from_literal_source_identity(tmp_path):
    raw_root, packet_root, corpus, receipt = _packet(tmp_path)

    rows = load_canonical_representation_rows(
        packet_root, receipt, corpus
    )

    assert len(rows) == 4
    assert rows[0].text.startswith("canonical model row")
    assert rows[0].raw_sha256 == receipt.rows[0].source_raw_sha256
    assert (
        receipt.canonical_model_row_digest
        == corpus.canonical_model_row_digest
    )
    assert raw_root not in [
        Path(row.relative_path) for row in receipt.rows
    ]


def test_raw_byte_mutation_requires_a_new_corpus_and_receipt_namespace(tmp_path):
    raw_root, packet_root, corpus, receipt = _packet(tmp_path)
    path = raw_root / corpus.works[0].raw_paths[0]
    path.write_bytes(path.read_bytes() + b" ")

    # Canonical model rows remain numerically identical, but the old literal
    # source identity is no longer valid.
    assert (
        load_canonical_representation_rows(packet_root, receipt, corpus)[0].text
        == "canonical model row a/a1"
    )
    from stylo.domain.lobo_vnext import VNextContractError, verify_raw_inventory

    with pytest.raises(VNextContractError, match="byte size|SHA-256"):
        verify_raw_inventory(raw_root, corpus)


@pytest.mark.parametrize("mutation", ["missing", "extra", "tamper", "symlink"])
def test_row_directory_rejects_missing_extra_tampered_and_symlinked(
    tmp_path, mutation
):
    _raw_root, packet_root, corpus, receipt = _packet(tmp_path)
    target = packet_root / receipt.rows[0].relative_path
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (packet_root / "canonical_rows/extra.txt").write_text(
            "extra", encoding="utf-8"
        )
    elif mutation == "tamper":
        target.write_text("tampered", encoding="utf-8")
    else:
        target.unlink()
        target.symlink_to(packet_root / receipt.rows[1].relative_path)

    with pytest.raises(VNextPacketError):
        load_canonical_representation_rows(packet_root, receipt, corpus)


def test_receipt_rejects_extra_missing_duplicate_and_bool_as_int(tmp_path):
    _raw_root, _packet_root, _corpus, receipt = _packet(tmp_path)
    base = receipt.to_dict()

    extra = json.loads(json.dumps(base))
    extra["unexpected"] = "rehashed"
    _rehash(extra)
    with pytest.raises(VNextPacketError, match="keys must be exact"):
        loads_canonical_representation_receipt(json.dumps(extra))

    missing = json.loads(json.dumps(base))
    del missing["n_rows"]
    _rehash(missing)
    with pytest.raises(VNextPacketError, match="keys must be exact"):
        loads_canonical_representation_receipt(json.dumps(missing))

    duplicate = (
        '{"schema_version":"duplicate",'
        + json.dumps(base, ensure_ascii=False)[1:]
    )
    with pytest.raises(VNextPacketError, match="duplicate object key"):
        loads_canonical_representation_receipt(duplicate)

    bool_count = json.loads(json.dumps(base))
    bool_count["rows"][0]["canonical_byte_size"] = True
    _rehash(bool_count)
    with pytest.raises(VNextPacketError, match="exact integer"):
        loads_canonical_representation_receipt(json.dumps(bool_count))


def test_receipt_rejects_noncontiguous_ordinals_and_wrong_source_binding(
    tmp_path,
):
    _raw_root, _packet_root, corpus, receipt = _packet(tmp_path)
    ordinal = receipt.to_dict()
    ordinal["rows"][0]["ordinal"] = 2
    ordinal["row_inventory_sha256"] = canonical_sha256(ordinal["rows"])
    ordinal["canonical_model_row_digest"] = ordinal["row_inventory_sha256"]
    _rehash(ordinal)
    with pytest.raises(VNextPacketError, match="ordinals"):
        loads_canonical_representation_receipt(json.dumps(ordinal))

    source = receipt.to_dict()
    source["rows"][0]["source_raw_sha256"] = "0" * 64
    source["row_inventory_sha256"] = canonical_sha256(source["rows"])
    source["canonical_model_row_digest"] = source["row_inventory_sha256"]
    corpus_raw = corpus.to_dict()
    corpus_raw["canonical_model_row_digest"] = source[
        "canonical_model_row_digest"
    ]
    _rehash(corpus_raw)
    from stylo.domain.lobo_vnext import CorpusVNextManifest

    changed_corpus = CorpusVNextManifest.from_dict(corpus_raw)
    source["corpus_manifest_sha256"] = changed_corpus.self_hash
    _rehash(source)
    changed = loads_canonical_representation_receipt(json.dumps(source))
    with pytest.raises(VNextPacketError, match="source SHA"):
        changed.validate(corpus_manifest=changed_corpus)


def test_corpus_binding_rejects_generation_and_model_row_digest_drift(tmp_path):
    _raw_root, _packet_root, corpus, receipt = _packet(tmp_path)
    raw = corpus.to_dict()
    raw["canonical_model_row_digest"] = "0" * 64
    _rehash(raw)
    from stylo.domain.lobo_vnext import CorpusVNextManifest

    drifted = CorpusVNextManifest.from_dict(raw)
    with pytest.raises(VNextPacketError, match="corpus digest"):
        receipt.validate(corpus_manifest=drifted)
