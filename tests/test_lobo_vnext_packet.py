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
    VNextPacketError,
    load_canonical_representation_rows,
    loads_canonical_representation_receipt,
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
