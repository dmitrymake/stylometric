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
from stylo.eval.paired_audit import applicability as ap
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
    # supply the REAL fold-local evidence this applied cell requires (the runner never synthesizes it)
    evidence = {key: hashlib.sha256(f"{key}:{model}:{cell}:{work_id}".encode()).hexdigest()
                for key in ap.required_evidence_digests(model, cell)}
    for pk in ap.required_evidence_passports(model, cell):
        evidence[pk] = {"calibration_disabled": False, "mode": "sigmoid", "meta": {}}
    return {"pred_label": pred_label, "correct": correct, "rank": 1 if correct else 2,
            "probabilities": proba, "evidence": evidence}


def test_a0_index_resolves_pred_and_rejects_out_of_range():
    # the runner resolves the pred CLASS INDEX to its author slug and keys by the full work id
    prob_order = ["aa", "bb"]
    a = {"works": ["aa/w1", "bb/w1"], "correct": [1, 0], "ranks": [1, 3], "preds": [0, 0]}
    idx = rn._a0_index("lobo", a, prob_order)
    assert idx == {"aa/w1": {"true_author": "aa", "pred": "aa", "correct": True, "rank": 1},
                   "bb/w1": {"true_author": "bb", "pred": "aa", "correct": False, "rank": 3}}
    with pytest.raises(rn.RunnerError):                            # pred index out of range
        rn._a0_index("lobo", {"works": ["aa/w1"], "correct": [1], "ranks": [1], "preds": [9]}, prob_order)


def test_a0_reference_mismatch_is_fatal():
    # §3.2: exact one-to-one vs the pinned reference index, pred included, keyed by full work id
    prob_order = ["aa", "bb"]
    ref_index = {"aa/b1": {"true_author": "aa", "pred": "aa", "correct": True, "rank": 1},
                 "bb/b2": {"true_author": "bb", "pred": "aa", "correct": False, "rank": 5}}
    ok = {"works": ["aa/b1", "bb/b2"], "correct": [1, 0], "ranks": [1, 5], "preds": [0, 0]}
    rn._assert_a0_matches_reference("lobo", ok, prob_order, ref_index)      # matches -> ok
    forged_pred = {"works": ["aa/b1", "bb/b2"], "correct": [1, 0], "ranks": [1, 5], "preds": [0, 1]}
    with pytest.raises(rn.RunnerError):                            # bb/b2 pred forged to bb
        rn._assert_a0_matches_reference("lobo", forged_pred, prob_order, ref_index)
    short = {"works": ["aa/b1"], "correct": [1], "ranks": [1], "preds": [0]}
    with pytest.raises(rn.RunnerError):                            # missing reference work
        rn._assert_a0_matches_reference("lobo", short, prob_order, ref_index)


def test_fold_evidence_required_and_hex():
    # stylo A1 exercises W -> requires proba_digest + ordered_weight_digest, each sha256-hex
    ok = {"proba_digest": "1" * 64, "ordered_weight_digest": "2" * 64}
    rn._assert_fold_evidence("stylo", "A1", ok)
    with pytest.raises(rn.RunnerError):                            # missing ordered_weight_digest
        rn._assert_fold_evidence("stylo", "A1", {"proba_digest": "1" * 64})
    with pytest.raises(rn.RunnerError):                            # non-hex digest
        rn._assert_fold_evidence("stylo", "A1", {"proba_digest": "nope", "ordered_weight_digest": "2" * 64})
    # stylo_stack A4 additionally requires the stack calibration passport
    with pytest.raises(rn.RunnerError):
        rn._assert_fold_evidence("stylo_stack", "A4",
                                 {"proba_digest": "1" * 64, "ordered_weight_digest": "2" * 64,
                                  "vocab_digest": "3" * 64, "idf_digest": "4" * 64,
                                  "r_denominator_trace_digest": "5" * 64})   # no stack_calibration_digest


def test_evidence_aggregates_real_fold_digests_not_synthetic():
    recs = {0: {"work_id": "aa/w1", "fold_local_evidence": {"proba_digest": "1" * 64,
                                                            "ordered_weight_digest": "2" * 64}},
            1: {"work_id": "bb/w1", "fold_local_evidence": {"proba_digest": "3" * 64,
                                                            "ordered_weight_digest": "4" * 64}}}
    ev1 = rn._aggregate_evidence("stylo", "A1", recs)
    assert set(ev1) == {"proba_digest", "ordered_weight_digest"}
    # changing a fold's REAL weight digest changes the aggregate (byte/digest propagation)
    recs[0]["fold_local_evidence"]["ordered_weight_digest"] = "9" * 64
    ev2 = rn._aggregate_evidence("stylo", "A1", recs)
    assert ev2["ordered_weight_digest"] != ev1["ordered_weight_digest"]
    assert ev2["proba_digest"] == ev1["proba_digest"]             # untouched digest is stable
    # a missing required fold digest is fatal (no synthesis)
    del recs[1]["fold_local_evidence"]["proba_digest"]
    with pytest.raises(rn.RunnerError):
        rn._aggregate_evidence("stylo", "A1", recs)


def _built_corpus(tmp_path):
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
    return audit_root, lobo_ds, ruaa_work_ids, lobo_m, ruaa_m


def test_reattest_detects_disk_mutation(tmp_path):
    # §4/R3.5: the re-attestation re-hashes the WHOLE corpus tree, re-verifies the dataset arrays and
    # re-derives the manifest FROM DISK every fold — never an in-memory object against itself
    audit_root, lobo_ds, ruaa_ids, lobo_m, ruaa_m = _built_corpus(tmp_path)
    ruaa_ds = derive_work_subset(lobo_ds, ruaa_ids)
    cm = ac.verify_published_corpus(audit_root)
    plan = {"execution_source_sha256": rn.rp.execution_source_sha256(None),
            "config_id": rn.rp.config_id(CFG),
            "env_lock_sha256": rn.rp.env_lock_sha256(None),
            "corpus_chain": {"legacy_anchor": cm["legacy_anchor"],
                             "semantic_parity_digest": cm["source_semantic_parity_digest"]},
            "lobo": {"dataset_digest": lobo_ds.provenance.rows_digest,
                     "fold_manifest_digest": lobo_m["self_hash"]},
            "ruaa": {"dataset_digest": ruaa_ds.provenance.rows_digest,
                     "fold_manifest_digest": ruaa_m["self_hash"]}}
    ctx = {"plan": plan, "audit_root": audit_root, "cfg": CFG,
           "committed": {"lobo": lobo_m, "ruaa": ruaa_m},
           "datasets": {"lobo": lobo_ds, "ruaa": ruaa_ds}, "confirmatory": False,
           "repo_root": None, "src_root": None}
    rn._reattest(ctx)                                             # matches disk + data + manifest -> ok
    lm = plan["lobo"]
    for over in ({"config_id": "0" * 64},
                 {"corpus_chain": {"legacy_anchor": "0" * 64, "semantic_parity_digest": "1" * 64}},
                 {"lobo": {**lm, "dataset_digest": "0" * 64}},        # dataset digest drift
                 {"lobo": {**lm, "fold_manifest_digest": "0" * 64}}):  # manifest digest drift
        with pytest.raises(rn.RunnerError):
            rn._reattest({**ctx, "plan": {**plan, **over}})
    # a physical tamper of the on-disk corpus manifest is caught (surfaced as a runner fail-close)
    mpath = audit_root / ac.CORPUS_MANIFEST_NAME
    mpath.write_text(mpath.read_text(encoding="utf-8").replace(cm["legacy_anchor"], "0" * 64),
                     encoding="utf-8")
    with pytest.raises(rn.RunnerError):
        rn._reattest(ctx)


def test_forged_manifest_caught_by_nontautological_rebuild(tmp_path):
    # §5: the rebuild sources algorithm/seed/parent from REGISTERED constants + the actual dataset, so
    # a committed manifest that lies about them (and re-self-hashes) still fails the equality check
    audit_root, lobo_ds, _ruaa_ids, _lm, _rm = _built_corpus(tmp_path)
    real_parent = lobo_ds.provenance.rows_digest
    forged_algo = mf.build_fold_manifest("lobo", lobo_ds, parent_dataset_digest=real_parent,
                                         algorithm="whole_work", seed=42, config_hash="c" * 64)
    rebuilt = rn._rebuild_manifest("lobo", lobo_ds, real_parent, CFG, forged_algo, confirmatory=False)
    with pytest.raises(mf.FoldManifestError):
        mf.verify_manifest_matches_rebuilt(forged_algo, rebuilt, universe=False)
    forged_parent = mf.build_fold_manifest("lobo", lobo_ds, parent_dataset_digest="0" * 64,
                                           algorithm="leave_one_work_out", seed=42, config_hash="c" * 64)
    rebuilt2 = rn._rebuild_manifest("lobo", lobo_ds, real_parent, CFG, forged_parent, confirmatory=False)
    with pytest.raises(mf.FoldManifestError):
        mf.verify_manifest_matches_rebuilt(forged_parent, rebuilt2, universe=False)


def _run_smoke(tmp_path):
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
        committed_lobo_manifest=lobo_m, committed_ruaa_manifest=ruaa_m, ruaa_work_ids=ruaa_work_ids,
        checkpoint_root=tmp_path / "ck", docs_root=tmp_path / "docs", evaluator=_dummy_evaluator,
        a0_references={"lobo_books": pathlib.Path.cwd() / "docs/lobo_books.txt"},
        tolerances={q: {"atol": 1e-9, "rtol": 0, "dtype": "float64"}
                    for q in rn.rp.REGISTERED_TOLERANCE_QUANTITIES}, run_kind="smoke")
    return out, lobo_m, ruaa_m


@_needs_git
def test_result_audit_catches_tampered_metric(tmp_path):
    # §8: the independent auditor re-passes the audited candidate but rejects a tampered metric
    import copy
    from stylo.eval.paired_audit import result_audit as ra
    out, _lm, _rm = _run_smoke(tmp_path)
    summary, vectors, plan = out["summary"], out["per_work_vectors"], out["summary"]["run_plan"]
    assert out["result_audit"]["passed"] is True
    ra.audit_results(summary, vectors, plan)                     # the audited candidate re-passes
    bad = copy.deepcopy(summary)
    bad["cells"]["lobo"]["stylo/A0"]["point"]["accuracy"] += 0.3
    with pytest.raises(ra.ResultAuditError):
        ra.audit_results(bad, vectors, plan)
    worse = copy.deepcopy(summary)                               # a tampered cluster p is also caught
    worse["cells"]["lobo"]["stylo/A4"]["vs_A0"]["cluster_p"] = 0.999999
    with pytest.raises(ra.ResultAuditError):
        ra.audit_results(worse, vectors, plan)


@_needs_git
def test_synthetic_end_to_end_runner(tmp_path):
    out, lobo_m, ruaa_m = _run_smoke(tmp_path)

    assert len(out["run_id"]) == 64
    # a smoke run publishes ONLY to the transient run namespace (no committed production artifact)
    assert not (tmp_path / "docs" / pub.SUMMARY_NAME).exists()
    loaded = pub.load_published_audit(tmp_path / "docs", run_kind="smoke", run_id=out["run_id"])
    assert loaded["version"] == out["published"]["version"]
    assert loaded["summary"]["run_id"] == out["run_id"]
    assert loaded["summary"]["headline"]["endpoint"] == "stylo_lobo_a4_minus_a0_accuracy"
    # both datasets, 30 cells each, and the 15-member Holm family per dataset
    assert set(loaded["summary"]["cells"]) == {"lobo", "ruaa"}
    assert len(loaded["summary"]["cells"]["lobo"]) == 30
    assert len(loaded["summary"]["holm"]["ruaa"]) == 15
    # byte/digest propagation evaluator -> checkpoint -> artifact: the published cell evidence is
    # EXACTLY the aggregate of the real per-fold evidence stored in the checkpoints (not synthesized)
    from stylo.eval.paired_audit.checkpoints import CheckpointStore
    store = CheckpointStore(tmp_path / "ck", out["run_id"],
                            {"lobo": rn._bindings_for(lobo_m), "ruaa": rn._bindings_for(ruaa_m)})
    recs = store.scan_cell("lobo", "stylo", "A1")
    art_ev = loaded["summary"]["cells"]["lobo"]["stylo/A1"]["evidence"]
    assert art_ev == rn._aggregate_evidence("stylo", "A1", recs)
    assert set(art_ev) == {"proba_digest", "ordered_weight_digest"} and all(len(v) == 64 for v in art_ev.values())


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
              tolerances={q: {"atol": 1e-9, "rtol": 0, "dtype": "float64"}
                    for q in rn.rp.REGISTERED_TOLERANCE_QUANTITIES}, run_kind="smoke")
    a = rn.run_paired_audit(**kw)
    b = rn.run_paired_audit(**kw)                             # resumes; same run_id + version
    assert a["run_id"] == b["run_id"]
    assert a["published"]["version"] == b["published"]["version"]
