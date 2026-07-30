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
    R1CorpusGenerationMaterial,
    R1PacketGenerationMaterial,
    R1PacketManifest,
    R1_ACQUISITION_BINDING_SCHEMA_VERSION,
    R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION,
    R1_PACKET_GENERATION_MATERIAL_SCHEMA_VERSION,
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


def _acquisition_binding(
    *,
    acquisition_generation_id: str | None = None,
    raw_inventory_digest: str | None = None,
    work_identity_catalog_digest: str | None = None,
    content_policy_spec_digest: str | None = None,
) -> R1AcquisitionBinding:
    return R1AcquisitionBinding.build(
        acquisition_generation_id=(
            acquisition_generation_id or _digest("acquisition-generation")
        ),
        acquisition_manifest_self_hash=_digest("acquisition-manifest"),
        acquisition_receipt_self_hash=_digest("acquisition-receipt"),
        selected_audit_file_sha256=_digest("selected-audit-file"),
        selected_audit_self_hash=_digest("selected-audit-self"),
        raw_inventory_digest=(
            raw_inventory_digest or _digest("raw-inventory")
        ),
        work_identity_catalog_digest=(
            work_identity_catalog_digest or _digest("work-identities")
        ),
        upstream_excluded_work_ids=_UPSTREAM_EXCLUSIONS,
        content_policy_spec_digest=(
            content_policy_spec_digest or _digest("content-policy")
        ),
        work_count=134,
        author_count=22,
    )


def _corpus_generation_material(
    binding: R1AcquisitionBinding,
) -> R1CorpusGenerationMaterial:
    return R1CorpusGenerationMaterial.build(
        acquisition_binding_self_hash=binding.self_hash
    )


def _r1_packet_manifest(
    *,
    binding: R1AcquisitionBinding | None = None,
    candidate_inventory_sha256: str | None = None,
    primary_model_spec_sha256: str | None = None,
    inference_spec_sha256: str | None = None,
    files: tuple[PacketFileEntry, ...] | None = None,
) -> R1PacketManifest:
    binding = binding or _acquisition_binding()
    corpus_material = _corpus_generation_material(binding)
    return R1PacketManifest.build(
        acquisition_binding=binding,
        corpus_generation_material=corpus_material,
        content_policy_spec_sha256=binding.content_policy_spec_digest,
        candidate_inventory_sha256=(
            candidate_inventory_sha256 or _digest("candidate-inventory")
        ),
        corpus_manifest_sha256=_digest("corpus"),
        content_component_manifest_sha256=_digest("content-components"),
        fold_manifest_sha256=_digest("folds"),
        primary_model_spec_sha256=(
            primary_model_spec_sha256 or _digest("primary-model")
        ),
        baseline_model_spec_sha256=_digest("baseline-model"),
        inference_spec_sha256=(
            inference_spec_sha256 or _digest("inference")
        ),
        primary_inner_cv_plan_sha256=_digest("primary-inner"),
        baseline_inner_cv_plan_sha256=_digest("baseline-inner"),
        model_role_manifest_sha256=_digest("model-roles"),
        campaign_manifest_sha256=_digest("campaign"),
        representation_receipt_sha256=_digest("representation"),
        files=files or (
            PacketFileEntry(
                "manifests/acquisition-manifest.json",
                1,
                _digest("acquisition-manifest-file"),
            ),
        ),
    )


def test_acquisition_binding_is_strict_self_hashed_selected_134_contract():
    binding = _acquisition_binding()
    wire = binding.to_dict()

    assert binding.schema_version == R1_ACQUISITION_BINDING_SCHEMA_VERSION
    assert binding.schema_version.endswith(".v2")
    assert binding.acquisition_generation_id == _digest(
        "acquisition-generation"
    )
    assert binding.work_count == 134
    assert binding.author_count == 22
    assert binding.upstream_excluded_work_ids == _UPSTREAM_EXCLUSIONS
    assert "generation_id" not in wire
    assert "post_selection_candidate_inventory_sha256" not in wire
    assert R1AcquisitionBinding.from_dict(binding.to_dict()) == binding
    assert binding.validate() is binding


def test_acquisition_binding_rejects_schema_shape_hash_and_count_drift():
    base = _acquisition_binding().to_dict()

    legacy = json.loads(json.dumps(base))
    legacy["schema_version"] = (
        "stylo.lobo-vnext.ruaa-r1-acquisition-binding.v1"
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


def test_corpus_generation_material_is_versioned_self_hash_and_drift_sensitive():
    binding = _acquisition_binding()
    material = _corpus_generation_material(binding)

    assert (
        material.schema_version
        == R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION
    )
    assert material.acquisition_binding_self_hash == binding.self_hash
    assert material.corpus_generation_id == material.self_hash
    assert material.self_hash == canonical_sha256(
        {
            "schema_version": (
                R1_CORPUS_GENERATION_MATERIAL_SCHEMA_VERSION
            ),
            "acquisition_binding_self_hash": binding.self_hash,
        }
    )
    assert R1CorpusGenerationMaterial.from_dict(material.to_dict()) == material
    assert material.validate() is material

    changed_bindings = (
        _acquisition_binding(
            raw_inventory_digest=_digest("changed-raw-inventory")
        ),
        _acquisition_binding(
            work_identity_catalog_digest=_digest("changed-work-identities")
        ),
        _acquisition_binding(
            content_policy_spec_digest=_digest("changed-content-policy")
        ),
    )
    assert all(
        _corpus_generation_material(changed).self_hash != material.self_hash
        for changed in changed_bindings
    )


def test_corpus_generation_material_rejects_schema_shape_and_hash_drift():
    base = _corpus_generation_material(_acquisition_binding()).to_dict()

    legacy = json.loads(json.dumps(base))
    legacy["schema_version"] = (
        "stylo.lobo-vnext.ruaa-r1-corpus-generation-material.v0"
    )
    _rehash(legacy)
    with pytest.raises(VNextPacketError, match="legacy or unsupported"):
        R1CorpusGenerationMaterial.from_dict(legacy)

    extra = json.loads(json.dumps(base))
    extra["candidate_inventory_sha256"] = _digest("candidate-inventory")
    _rehash(extra)
    with pytest.raises(VNextPacketError, match="keys must be exact"):
        R1CorpusGenerationMaterial.from_dict(extra)

    tampered = json.loads(json.dumps(base))
    tampered["acquisition_binding_self_hash"] = "0" * 64
    with pytest.raises(VNextPacketError, match="self_hash mismatch"):
        R1CorpusGenerationMaterial.from_dict(tampered)


def test_packet_v4_has_three_unambiguous_identities_and_exact_material():
    packet = _r1_packet_manifest()
    wire = packet.to_dict()

    assert packet.schema_version == R1_PACKET_MANIFEST_SCHEMA_VERSION
    assert packet.schema_version.endswith(".v4")
    assert packet.acquisition_generation_id == (
        packet.acquisition_binding.acquisition_generation_id
    )
    assert packet.corpus_generation_id == (
        packet.corpus_generation_material.self_hash
    )
    assert packet.packet_generation_id == canonical_sha256(
        packet.packet_generation_material.to_dict()
    )
    assert (
        packet.packet_generation_material.schema_version
        == R1_PACKET_GENERATION_MATERIAL_SCHEMA_VERSION
    )
    assert (
        packet.packet_generation_material.packet_schema_version
        == R1_PACKET_MANIFEST_SCHEMA_VERSION
    )
    assert (
        packet.packet_generation_material.acquisition_binding_self_hash
        == packet.acquisition_binding.self_hash
    )
    assert (
        packet.packet_generation_material.content_policy_spec_sha256
        == packet.acquisition_binding.content_policy_spec_digest
    )
    assert packet.selected_work_count == 134
    assert packet.confirmatory_authorized is False
    assert "generation_id" not in wire
    assert "file_inventory_sha256" not in wire
    assert (
        "candidate_inventory_sha256"
        not in packet.acquisition_binding.to_dict()
    )
    assert (
        R1PacketGenerationMaterial.from_dict(
            packet.packet_generation_material.to_dict()
        )
        == packet.packet_generation_material
    )
    assert R1PacketManifest.from_dict(packet.to_dict()) == packet
    assert packet.validate() is packet


def test_packet_generation_material_rejects_schema_shape_and_digest_drift():
    base = _r1_packet_manifest().packet_generation_material.to_dict()

    legacy = json.loads(json.dumps(base))
    legacy["schema_version"] = (
        "stylo.lobo-vnext.ruaa-r1-packet-generation-material.v0"
    )
    with pytest.raises(VNextPacketError, match="legacy or unsupported"):
        R1PacketGenerationMaterial.from_dict(legacy)

    packet_v3 = json.loads(json.dumps(base))
    packet_v3["packet_schema_version"] = (
        "stylo.lobo-vnext.ruaa-r1-packet.v3"
    )
    with pytest.raises(VNextPacketError, match="packet schema is unsupported"):
        R1PacketGenerationMaterial.from_dict(packet_v3)

    extra = json.loads(json.dumps(base))
    extra["self_hash"] = _digest("forbidden-self-hash")
    with pytest.raises(VNextPacketError, match="keys must be exact"):
        R1PacketGenerationMaterial.from_dict(extra)

    malformed = json.loads(json.dumps(base))
    malformed["candidate_inventory_sha256"] = "not-a-digest"
    with pytest.raises(VNextPacketError, match="64 lowercase hex"):
        R1PacketGenerationMaterial.from_dict(malformed)


@pytest.mark.parametrize(
    ("argument", "changed_digest"),
    (
        ("candidate_inventory_sha256", _digest("changed-candidates")),
        ("primary_model_spec_sha256", _digest("changed-primary-model")),
        ("inference_spec_sha256", _digest("changed-inference")),
    ),
)
def test_downstream_digest_changes_only_packet_generation(
    argument,
    changed_digest,
):
    first = _r1_packet_manifest()
    second = _r1_packet_manifest(**{argument: changed_digest})

    assert (
        first.acquisition_binding.acquisition_generation_id
        == second.acquisition_binding.acquisition_generation_id
    )
    assert first.corpus_generation_id == second.corpus_generation_id
    assert first.packet_generation_id != second.packet_generation_id
    assert first.self_hash != second.self_hash


def test_file_inventory_changes_packet_generation_but_not_corpus_generation():
    first = _r1_packet_manifest()
    second = _r1_packet_manifest(
        files=(
            PacketFileEntry(
                "manifests/acquisition-manifest.json",
                2,
                _digest("changed-acquisition-manifest-file"),
            ),
        )
    )

    assert first.corpus_generation_id == second.corpus_generation_id
    assert first.file_inventory_sha256 != second.file_inventory_sha256
    assert first.packet_generation_id != second.packet_generation_id


def test_packet_v4_rejects_rehashed_cross_binding_and_shape_drift():
    base = _r1_packet_manifest().to_dict()

    tampered = json.loads(json.dumps(base))
    tampered["packet_generation_id"] = "0" * 64
    with pytest.raises(VNextPacketError, match="self_hash mismatch"):
        R1PacketManifest.from_dict(tampered)

    corpus = json.loads(json.dumps(base))
    corpus["corpus_generation_id"] = "0" * 64
    _rehash(corpus)
    with pytest.raises(
        VNextPacketError,
        match="corpus_generation_id differs from material",
    ):
        R1PacketManifest.from_dict(corpus)

    corpus_binding = json.loads(json.dumps(base))
    corpus_material = corpus_binding["corpus_generation_material"]
    corpus_material["acquisition_binding_self_hash"] = "0" * 64
    _rehash(corpus_material)
    _rehash(corpus_binding)
    with pytest.raises(
        VNextPacketError,
        match="corpus generation material differs from acquisition binding",
    ):
        R1PacketManifest.from_dict(corpus_binding)

    content_policy = json.loads(json.dumps(base))
    packet_material = content_policy["packet_generation_material"]
    packet_material["content_policy_spec_sha256"] = "0" * 64
    content_policy["packet_generation_id"] = canonical_sha256(packet_material)
    _rehash(content_policy)
    with pytest.raises(
        VNextPacketError,
        match="content policy differs from acquisition binding",
    ):
        R1PacketManifest.from_dict(content_policy)

    inventory = json.loads(json.dumps(base))
    packet_material = inventory["packet_generation_material"]
    packet_material["file_inventory_sha256"] = "0" * 64
    inventory["packet_generation_id"] = canonical_sha256(packet_material)
    _rehash(inventory)
    with pytest.raises(VNextPacketError, match="file inventory digest mismatch"):
        R1PacketManifest.from_dict(inventory)

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


@pytest.mark.parametrize("legacy_version", ("v2", "v3"))
def test_packet_v2_and_v3_schemas_are_explicitly_rejected_as_legacy(
    legacy_version,
):
    historical = _r1_packet_manifest().to_dict()
    historical["schema_version"] = (
        f"stylo.lobo-vnext.ruaa-r1-packet.{legacy_version}"
    )
    historical.pop("packet_generation_material")

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
