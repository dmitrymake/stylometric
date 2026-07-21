"""Consolidated §9 fail-closed coverage sweep for the confirmatory paired-audit control plane.

This module is the single coverage manifest for the protocol §9 fail-closed catalog. Most scenarios
are proven in the per-module suites; this file both DOCUMENTS the mapping and adds the two remaining
gap-fill proofs (duplicate chunk identity at the builder, and publisher crash-orphan recovery). No
test here touches the real closed corpus.

§9 scenario → covering test:
  - legacy-anchor mismatch .............. test_paired_audit_corpus::test_legacy_anchor_mismatch_aborts_build
  - semantic parity mismatch ............ test_paired_audit_corpus::test_parity_mismatch_between_different_corpora
  - row-order parity invariant .......... test_paired_audit_corpus::test_row_order_parity_invariant_catches_reordering
  - changed text bytes .................. test_paired_audit_corpus::test_wb_manifest_guard_catches_byte_mutation_directly
  - missing/extra work .................. test_paired_audit_corpus::test_missing_work_selection_fails / TestWorkSubset
  - missing/extra chunk ................. test_paired_audit_corpus::test_wb_manifest_guard_catches_stray_chunk_directly
  - duplicate work/chunk identity ....... this::test_duplicate_chunk_identity_aborts_build
  - bad corpus-manifest self-hash ....... test_paired_audit_corpus::test_corpus_manifest_self_hash_tamper_rejected
  - partial audit-root .................. test_paired_audit_corpus::test_partial_root_never_valid
  - conflicting immutable root .......... test_paired_audit_corpus::test_conflicting_immutable_root_is_fatal
  - wrong selection digest .............. test_paired_audit_corpus::test_forged_selection_digest_rejected_at_disk_verify
  - wrong parent digest ................. test_paired_audit_corpus::test_forged_parent_digest_rejected_at_disk_verify
  - incomplete/reordered/non-whole RuAA . test_paired_audit_corpus::TestWorkSubset (missing/extra/duplicate/whole-work)
  - wrong LOBO/RuAA class order ......... test_paired_audit_manifest::test_bogus_class_order_contents_rejected
  - RunPlan canonicalization ............ test_paired_audit_control_plane::test_build_run_plan_and_stable_run_id
  - any binding change → new run_id ..... test_paired_audit_control_plane::test_run_id_changes_on_any_binding_change
  - no kernel strings in identity ....... test_paired_audit_control_plane::test_runtime_fingerprint_binds_stack_omits_kernel
  - valid checkpoint resume ............. test_paired_audit_checkpoints::test_valid_resume_skips_present_and_pends_missing
  - corrupt/conflicting/extra ckpt ...... test_paired_audit_checkpoints::TestFailClosed
  - missing ckpt pending until COMPLETE . test_paired_audit_checkpoints::test_assert_complete_success_and_incomplete_fatal
  - incomplete COMPLETE fatal ........... test_paired_audit_checkpoints::test_assert_complete_success_and_incomplete_fatal
  - path traversal / headline write ..... test_paired_audit_publisher::TestPathGuard
  - atomic publisher crash/failure ...... this::test_publisher_recovers_from_staging_orphan
  - content-addressed archive verify .... test_paired_audit_publisher::test_publish_and_load_round_trip
  - self-hash tampering ................. test_paired_audit_publisher::test_tampered_summary_self_hash_rejected
  - cluster-p degenerate cases .......... test_paired_audit_inference::TestClusterPValue (rule1/rule2)
  - constant non-zero cluster effect .... test_paired_audit_inference::test_rule3_constant_nonzero_effect_not_special_cased
  - fixed 15-comparison Holm family ..... test_paired_audit_inference::test_full_family_runs_with_m15
  - missing Holm member invalidates ..... test_paired_audit_inference::test_missing_or_extra_member_invalidates_family
  - headline boundary equality .......... test_paired_audit_headline::test_boundary_equality_is_inconclusive
  - A0/A4 golden replay ................. tests/test_work_balanced_ablation_goldens.py (+ opt-in live replay,
                                          WORK_BALANCED_LIVE_GOLDEN_REPLAY=1)
  - production-default invariants ....... tests/test_work_balanced_ablation_config.py
"""
from __future__ import annotations

import pathlib

import pytest

from stylo import workdoc as wd
from stylo.config import load_config
from stylo.jsonio import dump_strict, load_strict
from stylo.workdoc import chunker_config_hash
from stylo.eval.paired_audit import corpus as ac
from stylo.eval.paired_audit import publisher as pub
from stylo.eval.paired_audit import semantic_parity as sp

CFG = load_config()
_CHASH = chunker_config_hash(CFG)


def _toy_wb(tmp: pathlib.Path):
    """A minimal valid work-balanced corpus (>=10 chunks, >=2 authors)."""
    frags, ic = tmp / "frags", tmp / "clean"
    spec = {"alpha": ["a1", "a2"], "beta": ["b1", "b2"], "gamma": ["g1", "g2"]}
    for author, books in spec.items():
        for book in books:
            src = ic / author / f"{book}.txt"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(f"source {author} {book}", encoding="utf-8")
            wdir = frags / author / book
            wdir.mkdir(parents=True, exist_ok=True)
            texts = [f"{author} {book} chunk {i} words" for i in range(3)]
            names = [f"c_{i:03d}.txt" for i in range(3)]
            for nm, tx in zip(names, texts):
                (wdir / nm).write_text(tx, encoding="utf-8")
            m = wd.build_work_manifest(f"{author}/{book}", author, texts, names,
                                       provenance_sha256=wd.source_provenance_sha256(src),
                                       chunker_config_hash=_CHASH, overlap=0.0)
            dump_strict(m.to_dict(), wdir / wd.MANIFEST_NAME, trailing_newline=False)
    return frags, ic


def test_duplicate_chunk_identity_aborts_build(tmp_path):
    frags, ic = _toy_wb(tmp_path)
    from stylo.corpus import load_dataset
    anchor = load_dataset(frags).provenance.rows_digest
    # inject a duplicate chunk entry into one work's manifest (non-contiguous ordinals / dup path)
    mpath = frags / "alpha" / "a1" / wd.MANIFEST_NAME
    raw = load_strict(mpath)
    raw["chunks"].append(dict(raw["chunks"][0]))            # duplicate the first chunk identity
    dump_strict(raw, mpath, trailing_newline=False)
    with pytest.raises((wd.ManifestError, sp.SemanticParityError)):
        ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                              audit_parent=tmp_path / "audit", legacy_anchor=anchor)


def _summary():
    return {"claim_status": "exploratory_internal", "run_id": "a" * 32, "decision": "inconclusive"}


def _vectors():
    return {"lobo/stylo/A4": [{"work_id": "auth/w1", "proba": [0.1, 0.9]}]}


def test_publisher_recovers_from_staging_orphan(tmp_path):
    # simulate a crash that left a staging dir before os.replace — publish must still succeed cleanly
    versions = tmp_path / pub.ARCHIVE_DIRNAME / pub.VERSIONS_DIR
    versions.mkdir(parents=True)
    orphan = versions / ".staging_orphan"
    orphan.mkdir()
    (orphan / "half_written.json").write_text("{", encoding="utf-8")
    published = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
    loaded = pub.load_published_audit(tmp_path)
    assert loaded["version"] == published["version"]
    assert orphan.exists()                                  # orphan is inert, never referenced
