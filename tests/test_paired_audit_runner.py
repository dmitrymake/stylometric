"""Synthetic end-to-end test of the confirmatory runner (mandatory check D).

Builds a toy immutable audit corpus + manifests, injects a deterministic dummy estimator, runs the
WHOLE chain (references -> dataset -> manifests -> matrix -> RunPlan -> cells -> checkpoints ->
COMPLETE -> metrics -> cluster-p -> Holm -> headline -> publisher) in smoke mode, and round-trips the
published artifact. NEVER touches the real corpus.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

from stylo import workdoc as wd
from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.jsonio import dump_strict
from stylo.workdoc import chunker_config_hash, load_work_balanced_dataset
from stylo.eval.paired_audit import corpus as ac
from stylo.eval.paired_audit import manifest as mf
from stylo.eval.paired_audit import publisher as pub
from stylo.eval.paired_audit import runner as rn
from stylo.eval.paired_audit.work_subset import derive_work_subset

CFG = load_config()
_CHASH = chunker_config_hash(CFG)

# the runner obtains the git commit LIVE (blocker #9: a caller cannot spoof a clean state), so the
# end-to-end test needs a real git repo; a bare `git archive` snapshot has no .git and skips.
_HAS_GIT = rn.rp.git_commit_info().get("git_commit") is not None
_needs_git = pytest.mark.skipif(not _HAS_GIT, reason="runner e2e needs a git repo for the live commit binding")


def _make_corpus(tmp: pathlib.Path):
    frags, ic = tmp / "frags", tmp / "clean"
    # 4 multi-work authors (2 works each) + 2 single-work authors (train-only for LOBO)
    spec = {"aa": ["w1", "w2"], "bb": ["w1", "w2"], "cc": ["w1", "w2"], "dd": ["w1", "w2"],
            "ss": ["only"], "tt": ["only"]}
    for author, books in spec.items():
        for book in books:
            src = ic / author / f"{book}.txt"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(f"source {author} {book}", encoding="utf-8")
            wdir = frags / author / book
            wdir.mkdir(parents=True, exist_ok=True)
            texts = [f"{author} {book} chunk {i} more words here" for i in range(3)]
            names = [f"c_{i:03d}.txt" for i in range(3)]
            for nm, tx in zip(names, texts):
                (wdir / nm).write_text(tx, encoding="utf-8")
            m = wd.build_work_manifest(f"{author}/{book}", author, texts, names,
                                       provenance_sha256=wd.source_provenance_sha256(src),
                                       chunker_config_hash=_CHASH, overlap=0.0)
            dump_strict(m.to_dict(), wdir / wd.MANIFEST_NAME, trailing_newline=False)
    return frags, ic


def _dummy_evaluator(dataset, ds_obj, model, cell, fold_index, work_id, ablation):
    prob_order = sorted(set(str(g).split("/", 1)[0] for g in ds_obj.groups))
    width = len(prob_order)
    author = str(work_id).split("/", 1)[0]
    true_label = prob_order.index(author)
    # deterministic, cell-dependent correctness so cluster-p is non-degenerate
    h = int(hashlib.sha256(f"{work_id}:{model}:{cell}".encode()).hexdigest(), 16)
    correct = (h % 4) != 0
    pred_label = true_label if correct else (true_label + 1) % width
    proba = [0.05] * width
    proba[pred_label] = 1.0 - 0.05 * (width - 1)
    return {"pred_label": pred_label, "correct": correct, "rank": 1 if correct else 2,
            "probabilities": proba}


@_needs_git
def test_synthetic_end_to_end_runner(tmp_path):
    frags, ic = _make_corpus(tmp_path)
    anchor = load_dataset(frags).provenance.rows_digest
    audit_root = ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                       audit_parent=tmp_path / "audit", legacy_anchor=anchor)

    lobo_ds = ac.load_audit_dataset(audit_root, CFG)
    ruaa_work_ids = ["aa/w1", "aa/w2", "bb/w1", "bb/w2"]      # a whole-work RuAA subset (2 authors)
    ruaa_ds = derive_work_subset(lobo_ds, ruaa_work_ids)

    lobo_m = mf.build_fold_manifest("lobo", lobo_ds, parent_dataset_digest=lobo_ds.provenance.rows_digest,
                                    algorithm="leave_one_work_out", seed=42, config_hash="c" * 64)
    ruaa_m = mf.build_fold_manifest("ruaa", ruaa_ds, parent_dataset_digest=lobo_ds.provenance.rows_digest,
                                    algorithm="whole_work", seed=42, config_hash="c" * 64,
                                    selection_digest=ruaa_ds.provenance.selection_manifest_digest)

    out = rn.run_paired_audit(
        audit_root=audit_root, cfg=CFG,
        committed_lobo_manifest=lobo_m, committed_ruaa_manifest=ruaa_m,
        ruaa_work_ids=ruaa_work_ids,
        checkpoint_root=tmp_path / "ck", docs_root=tmp_path / "docs",
        evaluator=_dummy_evaluator,
        a0_references={"lobo_books": pathlib.Path.cwd() / "docs/lobo_books.txt"},
        tolerances={}, golden_fixture_inventory_sha="f" * 64, run_kind="smoke")

    assert len(out["run_id"]) == 64
    # published + round-trips through the verified loader
    loaded = pub.load_published_audit(tmp_path / "docs")
    assert loaded["version"] == out["published"]["version"]
    assert loaded["summary"]["run_id"] == out["run_id"]
    assert loaded["summary"]["headline"]["endpoint"] == "stylo_lobo_a4_minus_a0_accuracy"
    # both datasets, 30 cells each, and the 15-member Holm family per dataset
    assert set(loaded["summary"]["cells"]) == {"lobo", "ruaa"}
    assert len(loaded["summary"]["cells"]["lobo"]) == 30
    assert len(loaded["summary"]["holm"]["ruaa"]) == 15


@_needs_git
def test_runner_resumes_from_checkpoints(tmp_path):
    # a second run over the same checkpoint root reuses every fold (idempotent resume) and republishes
    frags, ic = _make_corpus(tmp_path)
    anchor = load_dataset(frags).provenance.rows_digest
    audit_root = ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                       audit_parent=tmp_path / "audit", legacy_anchor=anchor)
    lobo_ds = ac.load_audit_dataset(audit_root, CFG)
    ruaa_work_ids = ["aa/w1", "aa/w2", "bb/w1", "bb/w2"]
    ruaa_ds = derive_work_subset(lobo_ds, ruaa_work_ids)
    lobo_m = mf.build_fold_manifest("lobo", lobo_ds, parent_dataset_digest=lobo_ds.provenance.rows_digest,
                                    algorithm="leave_one_work_out", seed=42, config_hash="c" * 64)
    ruaa_m = mf.build_fold_manifest("ruaa", ruaa_ds, parent_dataset_digest=lobo_ds.provenance.rows_digest,
                                    algorithm="whole_work", seed=42, config_hash="c" * 64,
                                    selection_digest=ruaa_ds.provenance.selection_manifest_digest)
    kw = dict(audit_root=audit_root, cfg=CFG, committed_lobo_manifest=lobo_m,
              committed_ruaa_manifest=ruaa_m, ruaa_work_ids=ruaa_work_ids,
              checkpoint_root=tmp_path / "ck", docs_root=tmp_path / "docs",
              evaluator=_dummy_evaluator, a0_references={"lobo_books": pathlib.Path.cwd() / "docs/lobo_books.txt"},
              tolerances={}, golden_fixture_inventory_sha="f" * 64, run_kind="smoke")
    a = rn.run_paired_audit(**kw)
    b = rn.run_paired_audit(**kw)                             # resumes; same run_id + version
    assert a["run_id"] == b["run_id"]
    assert a["published"]["version"] == b["published"]["version"]
