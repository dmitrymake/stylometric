"""Adversarial identity/content contracts for corrected paired-audit v3.2."""
from __future__ import annotations

import copy

import pytest

from stylo.eval.paired_audit import corrected_v3_2 as v3


def _record(work_id: str, text: str, *, author: str | None = None) -> dict:
    author_id, slug = work_id.split("/", 1)
    author_id = author if author is not None else author_id
    byte = __import__("hashlib").sha256(text.encode()).hexdigest()
    normalized = __import__("hashlib").sha256(" ".join(text.casefold().split()).encode()).hexdigest()
    return {
        "work_id": work_id, "author_id": author_id, "work_slug": slug,
        "manifest_sha256": "1" * 64, "source_sha256": byte, "source_normalized_sha256": normalized,
        "work_content_identity": "2" * 64, "content_component_identity": "3" * 64,
        "chunks": [{"path": "c.txt", "span_ordinal": 0, "byte_sha256": byte,
                    "text_sha256": byte, "normalized_sha256": normalized, "text": text}],
    }


def test_adjudicated_basename_pairs_round_trip_by_full_id_only():
    records = [_record("radov/rasskazi", "radov unique words"), _record("zoshenko/rasskazi", "zoshenko unique words"),
               _record("bunin/деревня", "bunin unique words"), _record("grigorovich/деревня", "grigorovich unique words")]
    catalog = {row["work_id"]: row for row in records}
    assert v3.resolve_full_work(catalog, "radov/rasskazi")["author_id"] == "radov"
    assert v3.resolve_full_work(catalog, "zoshenko/rasskazi")["author_id"] == "zoshenko"
    assert v3.basename_collision_inventory(records, expected=v3.EXPECTED_LOBO_BASENAME_COLLISIONS)["components"] == [
        ["bunin/деревня", "grigorovich/деревня"], ["radov/rasskazi", "zoshenko/rasskazi"]
    ]
    for bare in ("rasskazi", "деревня"):
        with pytest.raises(v3.CorrectedCorpusError, match="never a basename"):
            v3.resolve_full_work(catalog, bare)


def test_duplicate_full_id_and_author_prefix_mismatch_are_fatal():
    record = _record("radov/rasskazi", "one distinct text")
    with pytest.raises(v3.CorrectedCorpusError, match="duplicate"):
        v3.work_identity_catalog([record, copy.deepcopy(record)])
    forged = copy.deepcopy(record)
    forged["author_id"] = "zoshenko"
    with pytest.raises(v3.CorrectedCorpusError, match="author-prefix"):
        v3.work_identity_catalog([forged])


def test_collision_inventory_rejects_missing_unexpected_or_third_member():
    records = [_record("radov/rasskazi", "one"), _record("zoshenko/rasskazi", "two"), _record("other/rasskazi", "three")]
    with pytest.raises(v3.CorrectedCorpusError, match="unexpected/missing/third"):
        v3.basename_collision_inventory(records, expected=(("radov/rasskazi", "zoshenko/rasskazi"),))


def test_content_overlap_inside_adjudicated_pair_remains_fatal():
    records = [_record("radov/rasskazi", "the exact component is shared across works"),
               _record("zoshenko/rasskazi", "the exact component is shared across works")]
    with pytest.raises(v3.CorrectedCorpusError, match="content-isolation"):
        v3.content_isolation_audit(records)


def test_old_fold_schema_is_rejected_before_any_universe_interpretation():
    with pytest.raises(v3.CorrectedCorpusError, match="identity-first"):
        v3.assert_v3_2_fold_manifest({"schema": "lobo_fold_manifest_v1", "works": []}, kind="lobo")


def test_tampered_child_digest_is_fatal_and_never_creates_current_pointer(tmp_path):
    root = tmp_path / "staging"
    (root / "frags").mkdir(parents=True)
    (root / "input_clean").mkdir()
    inventory, digest = v3._inventory(root)
    root = root.rename(tmp_path / digest)
    body = {
        "schema": v3.CORPUS_SCHEMA,
        "protocol_version": v3.PROTOCOL_VERSION,
        "corrected_content_inventory": inventory,
        "corrected_content_inventory_digest": digest,
    }
    body["self_hash"] = v3._self_hash(body)
    from stylo.jsonio import dump_strict

    manifest_path = root / "corrected_corpus_manifest_v3_2.json"
    dump_strict(body, manifest_path, trailing_newline=True)
    body["corrected_content_inventory_digest"] = "0" * 64
    dump_strict(body, manifest_path, trailing_newline=True)
    with pytest.raises(v3.CorrectedCorpusError, match="tamper"):
        v3._verify_child(root, body)
    assert not (tmp_path / "current.json").exists()


def test_full_work_id_rejects_non_nfc_and_bare_values():
    with pytest.raises(v3.CorrectedCorpusError):
        v3.full_work_id("деревня")
    with pytest.raises(v3.CorrectedCorpusError):
        v3.full_work_id("bunin/cafe\u0301")
