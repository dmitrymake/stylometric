"""Confirmatory paired-audit runner/orchestrator (synthetic fixtures / dry preflight until §11).

One chain, no critical link left to a future caller: pinned A0 references -> immutable disk-verified
audit dataset -> rebuilt-vs-committed manifests -> universe checks -> exact applicability matrix ->
live RunPlan -> per-fold estimator execution (an injected evaluator) -> before/after-fold binding
verification -> checkpoint resume -> all-cell COMPLETE -> metrics -> cluster p-values -> the fixed Holm
family -> headline -> validated publisher.

Until §11 the runner runs ONLY on synthetic fixtures or a dry preflight (``run_kind`` in
{smoke, dry_preflight}); it NEVER opens the real corpus. The single genuinely-external step is the
per-fold estimator, provided as an ``evaluator`` callable — the confirmatory run injects the real
estimator (make_factory_for_ablation), the synthetic tests inject a deterministic dummy.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Mapping

from ...jsonio import dumps_strict
from ...eval.metrics import macro_f1
from ..work_weighting import (FEATURE_STATE_ONLY_ABLATION, FULL_WB_ABLATION, LEGACY_ABLATION,
                              RELATIVE_FW_ONLY_ABLATION, WEIGHTS_ONLY_ABLATION)
from . import applicability as ap
from . import corpus as ac
from . import headline as hl
from . import inference as inf
from . import manifest as mf
from . import publisher as pub
from . import references as refmod
from . import run_plan as rp
from . import semantic_parity
from .checkpoints import CheckpointStore, dataset_bindings
from .work_subset import derive_work_subset

_DATASETS = ("lobo", "ruaa")
_CELL_ABLATION = {"A0": LEGACY_ABLATION, "A1": WEIGHTS_ONLY_ABLATION, "A2": FEATURE_STATE_ONLY_ABLATION,
                  "A3": RELATIVE_FW_ONLY_ABLATION, "A4": FULL_WB_ABLATION}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class RunnerError(RuntimeError):
    """Fail-closed: the confirmatory chain broke a contract (a bad preflight, manifest, binding, or
    incomplete run)."""


def _author_of(work_id: str) -> str:
    return str(work_id).split("/", 1)[0]


def _run_contract_digest(dataset) -> str:
    prov = dataset.provenance
    body = {"frags_root": prov.frags_root,
            "exclude_from_benchmark": list(prov.corpus_policy.exclude_from_benchmark),
            "unknown_dir_name": prov.corpus_policy.unknown_dir_name}
    return hashlib.sha256(dumps_strict(body, sort_keys=True).encode("utf-8")).hexdigest()


def _expected_folds(manifest) -> list[tuple[int, str]]:
    return sorted((w["fold_index"], w["work_id"]) for w in manifest["works"] if w["tested"])


def _bindings_for(manifest) -> dict:
    return dataset_bindings(manifest["dataset_digest"], manifest["parent_dataset_digest"],
                            manifest["self_hash"], manifest["probability_class_order"],
                            manifest["metric_label_order"])


def _rebuild_manifest(dataset_kind: str, dataset, parent_digest: str, cfg, committed: Mapping, *,
                      confirmatory: bool):
    """Rebuild the expected manifest from the on-disk dataset, sourcing the algorithm/seed from the
    REGISTERED constants and the parent/child/selection digests from the ACTUAL datasets — never from
    the committed manifest, so verifying those fields is non-tautological. A confirmatory run
    additionally pins ``config_hash`` to the independently computed config id (a smoke/dry run keeps the
    committed value so synthetic fixtures round-trip)."""
    config_hash = rp.config_id(cfg) if confirmatory else committed["config_hash"]
    selection_digest = None
    if dataset_kind == "ruaa":
        selection_digest = getattr(getattr(dataset, "provenance", None),
                                   "selection_manifest_digest", None)
    return mf.build_fold_manifest(
        dataset_kind, dataset,
        parent_dataset_digest=parent_digest,
        algorithm=mf.REGISTERED_ALGORITHM[dataset_kind], seed=mf.REGISTERED_SEED,
        config_hash=config_hash, selection_digest=selection_digest)


# ── the chain ────────────────────────────────────────────────────────────────
def run_paired_audit(*, audit_root, cfg, committed_lobo_manifest: Mapping,
                     committed_ruaa_manifest: Mapping, ruaa_work_ids, checkpoint_root, docs_root,
                     evaluator: Callable, a0_references: Mapping, tolerances: Mapping,
                     golden_fixture_inventory_sha: str, run_kind: str = "smoke",
                     lobo_author_display_map: Mapping | None = None,
                     repo_root=None, src_root=None) -> dict:
    """Run the whole confirmatory chain on the published immutable ``audit_root`` and publish the
    verified summary. Returns ``{run_id, summary, per_work_vectors, published}``."""
    if run_kind not in rp.REGISTERED_RUN_KINDS:
        raise RunnerError(f"unknown run_kind {run_kind!r}")
    confirmatory = run_kind == "confirmatory"
    if confirmatory:
        # a confirmatory run obtains git/source/env fingerprints from the real tree itself — a caller
        # cannot substitute a fake clean state via repo_root/src_root.
        repo_root = src_root = None

    # 1. pinned A0 references (SHA before parse; RuAA required only for a confirmatory run)
    preflight = refmod.assert_a0_preflight(require_ruaa=confirmatory, **a0_references)

    # 2. immutable disk-verified audit dataset(s); A0..A4 all run on the WB-manifest dataset
    corpus_manifest = ac.verify_published_corpus(audit_root)
    if confirmatory and corpus_manifest.get("legacy_anchor") != semantic_parity.LEGACY_ANCHOR:
        raise RunnerError("audit corpus legacy_anchor != the frozen LEGACY_ANCHOR")
    lobo_ds = ac.load_audit_dataset(audit_root, cfg)
    ac.verify_audit_dataset(lobo_ds)
    ruaa_ds = derive_work_subset(lobo_ds, ruaa_work_ids)
    datasets = {"lobo": lobo_ds, "ruaa": ruaa_ds}

    # 3. rebuilt-vs-committed manifests + universe checks. The rebuild sources algorithm/seed from the
    # REGISTERED constants and the parent/child/selection digests from the ACTUAL on-disk datasets, and
    # the committed manifest is additionally checked against the actual dataset labels — so verifying
    # those fields is non-tautological (a forged manifest cannot self-certify).
    parent_digest = lobo_ds.provenance.rows_digest
    committed = {"lobo": committed_lobo_manifest, "ruaa": committed_ruaa_manifest}
    manifests = {}
    for ds in _DATASETS:
        rebuilt = _rebuild_manifest(ds, datasets[ds], parent_digest, cfg, committed[ds],
                                    confirmatory=confirmatory)
        mf.assert_manifest_consistent_with_dataset(committed[ds], datasets[ds])
        mf.verify_manifest_matches_rebuilt(committed[ds], rebuilt, universe=confirmatory)
        manifests[ds] = committed[ds]

    # 4. exact applicability matrix
    ap.assert_matrix_invariants()

    # 4b. resolve the production evaluator identity and bind it into the run_id (§4.2): a confirmatory
    # run requires a REGISTERED EvaluatorSpec whose source bytes / import identity / estimator config /
    # mechanism passport all fold into the plan; a bare smoke callable is wrapped as a NON-registered
    # identity (usable only under smoke/dry, never confirmatory).
    if confirmatory and not isinstance(evaluator, rp.EvaluatorSpec):
        raise RunnerError("a confirmatory run requires a registered EvaluatorSpec, not a bare callable")
    eval_spec = evaluator if isinstance(evaluator, rp.EvaluatorSpec) else rp.EvaluatorSpec(
        name="smoke_dummy", fn=evaluator, estimator_config={"smoke": True},
        mechanism_passport={"smoke": True})
    eval_identity = rp.evaluator_identity(eval_spec, confirmatory=confirmatory)
    eval_fn = eval_spec.fn

    # 5. live RunPlan
    plan = _build_run_plan(datasets, manifests, corpus_manifest, run_kind=run_kind,
                           a0_references=a0_references, tolerances=tolerances,
                           golden_fixture_inventory_sha=golden_fixture_inventory_sha,
                           evaluator_identity=eval_identity,
                           cfg=cfg, repo_root=repo_root, src_root=src_root)
    run_id = rp.run_id(plan)

    # 6. checkpoint store bound to the run_id + per-dataset class-order/manifest digests
    store = CheckpointStore(checkpoint_root, run_id,
                            {ds: _bindings_for(manifests[ds]) for ds in _DATASETS})

    # 7. per-fold estimator execution with before/after DISK re-attestation + checkpoint resume
    reattest_ctx = {"plan": plan, "audit_root": audit_root, "cfg": cfg, "committed": manifests,
                    "repo_root": repo_root, "src_root": src_root}
    _run_all_cells(store, datasets, manifests, eval_fn, reattest_ctx)

    # 8. all-cell COMPLETE
    expected = {(ds, m, c): _expected_folds(manifests[ds])
                for ds in _DATASETS for (m, c) in ap.registered_cells()}
    present = store.assert_run_complete(expected)

    # 9-13. metrics, cluster p, Holm, headline; assemble the verified summary + per-work vectors.
    # For a confirmatory run the stylo A0 result must match the pinned reference EXACTLY, keyed by the
    # full work id and comparing pred (LOBO: true_author/pred/correct/rank; RuAA: pred). The display->slug
    # map is a required §11 provisioning input so the pred comparison can never be silently skipped.
    a0_confirm = None
    if confirmatory:
        if lobo_author_display_map is None:
            raise RunnerError("confirmatory run requires lobo_author_display_map to compare A0 predictions")
        a0_confirm = {
            "lobo": refmod.build_lobo_reference_index(preflight["lobo"], dict(lobo_author_display_map)),
            "ruaa": refmod.build_ruaa_reference_index(preflight["ruaa"])}
    summary, per_work_vectors = _assemble(present, manifests, run_id, plan, a0_confirm=a0_confirm)

    # 14. validated publisher
    published = pub.publish_audit(summary, per_work_vectors, docs_root=docs_root)
    return {"run_id": run_id, "summary": summary, "per_work_vectors": per_work_vectors,
            "published": published}


def _build_run_plan(datasets, manifests, corpus_manifest, *, run_kind, a0_references, tolerances,
                    golden_fixture_inventory_sha, evaluator_identity, cfg, repo_root, src_root) -> dict:
    # the A0 references were SHA-verified in the preflight; bind the pinned digests here (the two
    # call sites of a0_references have incompatible signatures, so never re-call verify_a0_references).
    a0_shas = {"lobo_books_txt": refmod.LOBO_BOOKS_SHA256,
               "ruaa_reference_submission": refmod.RUAA_REFERENCE_SUBMISSION_SHA256}
    git = rp.git_commit_info(repo_root)
    per_ds = {}
    for ds in _DATASETS:
        m = manifests[ds]
        entry = {"dataset_digest": datasets[ds].provenance.rows_digest,
                 "fold_manifest_digest": m["self_hash"],
                 "probability_class_order": m["probability_class_order"],
                 "metric_label_order": m["metric_label_order"],
                 "run_contract_digest": _run_contract_digest(datasets[ds])}
        if ds == "ruaa":
            entry["selection_digest"] = m["selection_digest"]
        per_ds[ds] = entry
    return rp.build_run_plan(
        run_kind=run_kind, audit_version=rp.AUDIT_VERSION,
        git_commit=git["git_commit"], git_dirty=git["git_dirty"],
        execution_source_sha256=rp.execution_source_sha256(src_root),
        env_lock_sha256=rp.env_lock_sha256(repo_root), config_id=rp.config_id(cfg),
        runtime_fingerprint=rp.runtime_fingerprint(),
        blas_thread_fingerprint=rp.blas_thread_fingerprint(),
        applicability_matrix_digest=ap.applicability_matrix_digest(),
        a0_reference_shas=a0_shas, evaluator_identity=evaluator_identity, tolerances=dict(tolerances),
        corpus_chain={"legacy_anchor": corpus_manifest["legacy_anchor"],
                      "semantic_parity_digest": corpus_manifest["source_semantic_parity_digest"]},
        golden_fixture_inventory_sha=golden_fixture_inventory_sha,
        lobo=per_ds["lobo"], ruaa=per_ds["ruaa"])


def _run_all_cells(store: CheckpointStore, datasets, manifests, evaluator, ctx) -> None:
    for (model, cell) in ap.registered_cells():
        ablation = _CELL_ABLATION[cell]
        for ds in _DATASETS:
            manifest = manifests[ds]
            expected = _expected_folds(manifest)
            _reattest(ctx, full=True)                              # before the cell: FULL disk re-attest
            _reverify_bindings(store, ds, manifest)
            state = store.resume_cell(ds, model, cell, expected)
            for fold_index, work_id in state["pending"]:
                _reattest(ctx, full=False)                         # before the fold: light disk re-attest
                res = evaluator(ds, datasets[ds], model, cell, fold_index, work_id, ablation)
                evidence = res.get("evidence")
                if not isinstance(evidence, Mapping):
                    raise RunnerError(f"{ds}/{model}/{cell} fold {work_id}: estimator supplied no "
                                      f"fold-local evidence (synthesis is forbidden)")
                _assert_fold_evidence(model, cell, evidence)       # real axis/state digests, no fallback
                store.save(ds, model, cell, fold_index, work_id,
                           result={"pred_label": int(res["pred_label"]),
                                   "correct": bool(res["correct"]), "rank": int(res["rank"]),
                                   "probabilities": [float(p) for p in res["probabilities"]]},
                           fold_local_evidence=dict(evidence))
                _reattest(ctx, full=False)                         # after the fold
            _reverify_bindings(store, ds, manifest)
            _reattest(ctx, full=True)                              # after the cell: FULL disk re-attest


def _reverify_bindings(store: CheckpointStore, dataset: str, manifest) -> None:
    if store.dataset_bindings[dataset] != _bindings_for(manifest):
        raise RunnerError(f"{dataset} checkpoint bindings drifted from the manifest mid-run")


def _reattest(ctx: Mapping, *, full: bool) -> None:
    """Re-derive the actual code/config/env/corpus/manifest digests FROM DISK and compare them to the
    frozen RunPlan (never an in-memory object against itself). Runs before and after every fold and
    cell; any mid-run mutation fails the run closed. ``full`` re-hashes the whole immutable corpus
    tree (cell boundary); otherwise the corpus manifest is re-attested cheaply (fold boundary)."""
    plan = ctx["plan"]
    if rp.execution_source_sha256(ctx["src_root"]) != plan["execution_source_sha256"]:
        raise RunnerError("code source tree changed mid-run (execution_source drift)")
    if rp.config_id(ctx["cfg"]) != plan["config_id"]:
        raise RunnerError("config changed mid-run (config_id drift)")
    if rp.env_lock_sha256(ctx["repo_root"]) != plan["env_lock_sha256"]:
        raise RunnerError("environment lock changed mid-run (env_lock drift)")
    try:
        cm = (ac.verify_published_corpus if full else ac.verify_corpus_manifest_light)(ctx["audit_root"])
    except ac.AuditCorpusError as exc:                          # surface a disk tamper as a fail-close
        raise RunnerError(f"corpus re-attestation failed mid-run: {exc}") from exc
    if {"legacy_anchor": cm.get("legacy_anchor"),
        "semantic_parity_digest": cm.get("source_semantic_parity_digest")} != plan["corpus_chain"]:
        raise RunnerError("corpus chain (anchor/parity) changed mid-run")
    for ds in _DATASETS:
        if ctx["committed"][ds]["self_hash"] != plan[ds]["fold_manifest_digest"]:
            raise RunnerError(f"{ds} fold-manifest digest drifted from the plan mid-run")


def _digest(res) -> str:
    return hashlib.sha256(dumps_strict(res, sort_keys=True).encode("utf-8")).hexdigest()


def _assert_fold_evidence(model: str, cell: str, evidence: Mapping) -> None:
    """The estimator MUST supply the real fold-local evidence digests this applied cell requires
    (§2.6); the runner never synthesizes them. Each required digest must be a sha256 hex string."""
    for key in ap.required_evidence_digests(model, cell):
        v = evidence.get(key)
        if not (isinstance(v, str) and _HEX64_RE.match(v)):
            raise RunnerError(f"{model}/{cell} fold-local evidence.{key} must be a sha256 hex digest "
                              f"(got {v!r})")


def _aggregate_evidence(model: str, cell: str, recs: Mapping) -> dict:
    """Aggregate the REAL per-fold evidence stored in the checkpoints into the cell-level evidence.

    Each required digest becomes a sha256 over the sorted ``(work_id, fold_digest)`` pairs, so the
    published cell evidence is a pure function of the estimator's stored fold digests — a changed fold
    digest provably changes the artifact. Nothing is re-synthesized from works/correct/probas.
    """
    out = {}
    for key in ap.required_evidence_digests(model, cell):
        parts = []
        for f in sorted(recs):
            v = recs[f]["fold_local_evidence"].get(key)
            if not (isinstance(v, str) and _HEX64_RE.match(v)):
                raise RunnerError(f"{model}/{cell} checkpoint {recs[f]['work_id']} missing/invalid "
                                  f"evidence.{key}")
            parts.append([recs[f]["work_id"], v])
        out[key] = _digest(["evidence", key, sorted(parts)])
    return out


def _a0_index(dataset_kind: str, a: dict, prob_order) -> dict:
    """Build the exact A0 result index keyed by the FULL work id (never a basename), resolving the
    predicted class index to its author slug. Duplicate work ids are fatal (via ``index_from_records``);
    an out-of-range pred index is fatal here rather than silently wrapping."""
    width = len(prob_order)

    def gen():
        for i, w in enumerate(a["works"]):
            pr = int(a["preds"][i])
            if not (0 <= pr < width):
                raise RunnerError(f"stylo {dataset_kind} A0 pred index {pr} out of range [0,{width}) for {w!r}")
            pred_author = prob_order[pr]
            if dataset_kind == "lobo":
                yield str(w), {"true_author": _author_of(w), "pred": pred_author,
                               "correct": bool(a["correct"][i]), "rank": int(a["ranks"][i])}
            else:
                yield str(w), {"pred": pred_author}
    return refmod.index_from_records(gen(), label=f"stylo {dataset_kind} A0")


def _assert_a0_matches_reference(dataset_kind: str, a: dict, prob_order, reference_index) -> None:
    """§3.2: exact one-to-one comparison of the stylo A0 result against the pinned reference index,
    keyed by the full work id and comparing pred explicitly (so all-wrong preds with matching
    correct/rank are rejected)."""
    fields = refmod.A0_LOBO_FIELDS if dataset_kind == "lobo" else refmod.A0_RUAA_FIELDS
    try:
        refmod.assert_a0_matches_index(_a0_index(dataset_kind, a, prob_order), reference_index,
                                       fields=fields, label=f"stylo {dataset_kind}")
    except refmod.ReferenceError as exc:                            # surface as a runner fail-close
        raise RunnerError(str(exc)) from exc


def _assemble(present, manifests, run_id, plan, *, a0_confirm=None) -> tuple[dict, dict]:
    """Assemble the verified summary + per-work vectors from the COMPLETE checkpoints."""
    cells_out = {ds: {} for ds in _DATASETS}
    holm_out = {ds: {} for ds in _DATASETS}
    per_work_vectors = {}
    headline_arms = {}
    iters, seed = plan["stats"]["bootstrap_iters"], plan["stats"]["seed"]
    quantiles = plan["stats"]["quantiles"]
    margin = plan["stats"]["noninferiority_margin"]

    for ds in _DATASETS:
        m = manifests[ds]
        prob_order = m["probability_class_order"]
        metric_idx = [prob_order.index(a) for a in m["metric_label_order"]]
        # gather per-cell fold arrays
        cell_arrays = {}
        for (model, cell) in ap.registered_cells():
            recs = present[(ds, model, cell)]                       # {fold_index: checkpoint}
            folds = sorted(recs)
            correct = [1 if recs[f]["result"]["correct"] else 0 for f in folds]
            ranks = [recs[f]["result"]["rank"] for f in folds]
            authors = [_author_of(recs[f]["work_id"]) for f in folds]
            works = [recs[f]["work_id"] for f in folds]
            probas = [recs[f]["result"]["probabilities"] for f in folds]
            preds = [recs[f]["result"]["pred_label"] for f in folds]
            trues = [prob_order.index(a) if a in prob_order else -1 for a in authors]
            cell_arrays[(model, cell)] = dict(correct=correct, ranks=ranks, authors=authors,
                                              works=works, probas=probas, preds=preds, trues=trues,
                                              evidence=_aggregate_evidence(model, cell, recs))
        if a0_confirm is not None and a0_confirm.get(ds) is not None:   # §3.2: exact A0 vs pinned reference
            _assert_a0_matches_reference(ds, cell_arrays[("stylo", "A0")], prob_order, a0_confirm[ds])
        # per-cell records + Holm + headline
        B = plan["stats"]["bootstrap_B"]
        raw_ps = {}
        for (model, cell) in ap.registered_cells():
            a = cell_arrays[(model, cell)]
            point = _point_metrics(a, metric_idx)
            reg = ap.cell_status(model, cell)
            abs_ci = hl.author_clustered_accuracy_ci(a["correct"], a["authors"], iters=iters, seed=seed,
                                                     quantiles=quantiles)
            rec = {"status": "applied", "requested_axes": reg["requested_axes"],
                   "effective_axes": reg["effective_axes"],
                   "point": point, "per_work": _per_work(a),
                   "abs_accuracy_authorclustered_ci": [abs_ci["lo"], abs_ci["hi"]],
                   "evidence": a["evidence"],                       # aggregated from real checkpoints
                   "claim_status": "exploratory_internal"}
            if cell != "A0":
                base = cell_arrays[(model, "A0")]
                dacc_ci = hl.paired_accuracy_diff_ci(a["correct"], base["correct"], a["authors"],
                                                     iters=iters, seed=seed, quantiles=quantiles)
                cp = inf.paired_cluster_pvalue(a["correct"], base["correct"], a["authors"], B=B, seed=seed)
                mc = inf.mcnemar_diagnostic(a["correct"], base["correct"])
                rec["vs_A0"] = {"dacc": point["accuracy"] - _point_metrics(base, metric_idx)["accuracy"],
                                "dacc_authorclustered_ci": [dacc_ci["lo"], dacc_ci["hi"]],
                                "cluster_p": cp, "holm_p": 1.0, "significant": False,
                                "mcnemar_p_diagnostic": mc["mcnemar_p_diagnostic"]}
                raw_ps[(model, cell)] = cp
            cells_out[ds][f"{model}/{cell}"] = rec
            per_work_vectors[f"{ds}/{model}/{cell}"] = _per_work(a)
        # non-applied cells: metadata records only
        for m_name in ap.MODELS:
            for c in ap.CELLS:
                if ap.cell_status(m_name, c)["status"] != "applied":
                    reg = ap.cell_status(m_name, c)
                    r = {"status": reg["status"], "requested_axes": reg["requested_axes"],
                         "effective_axes": reg["effective_axes"], "claim_status": "exploratory_internal"}
                    if reg["equivalent_to"] is not None:
                        r["equivalent_to"] = reg["equivalent_to"]
                    cells_out[ds][f"{m_name}/{c}"] = r
        # Holm over the fixed 15-member family
        holm = inf.holm_over_registered_family(raw_ps)
        for (model, cell), hp in holm.items():
            holm_out[ds][f"{model}/{cell}"] = hp
            vs = cells_out[ds][f"{model}/{cell}"]["vs_A0"]
            vs["holm_p"], vs["significant"] = hp["holm_p"], hp["significant"]
        headline_arms[ds] = cell_arrays

    # headline: stylo LOBO A4 - A0 only, fully bound to the run-id RunPlan stats
    la = headline_arms["lobo"]
    head = hl.evaluate_headline(la[("stylo", "A4")]["correct"], la[("stylo", "A0")]["correct"],
                                la[("stylo", "A4")]["authors"], margin=margin, iters=iters, seed=seed,
                                quantiles=quantiles)
    summary = {"run_id": run_id, "claim_status": "exploratory_internal", "cells": cells_out,
               "holm": holm_out, "headline": {"endpoint": head["endpoint"], "decision": head["decision"],
                                              "diff_ci": head["diff_ci"], "margin": head["margin"]},
               "attestation": {"git_commit": plan["git_commit"], "run_kind": plan["run_kind"],
                               "audit_version": plan["audit_version"]},
               "run_id_source": "canonical_run_plan_sha256"}
    return summary, per_work_vectors


def _point_metrics(a, metric_idx) -> dict:
    import numpy as np
    n = len(a["correct"]) or 1
    acc = sum(a["correct"]) / n
    top2 = sum(1 for r in a["ranks"] if r <= 2) / n
    f1 = float(macro_f1(np.array(a["trues"]), np.array(a["preds"]), metric_idx)) if a["trues"] else 0.0
    recall = {}
    for au in sorted(set(a["authors"])):
        idx = [i for i, x in enumerate(a["authors"]) if x == au]
        recall[au] = sum(a["correct"][i] for i in idx) / len(idx)
    return {"accuracy": acc, "macro_f1": f1, "top2": top2, "per_author_recall": recall}


def _per_work(a) -> list:
    return [{"work_id": w, "pred_label": p, "rank": r, "proba": pr}
            for w, p, r, pr in zip(a["works"], a["preds"], a["ranks"], a["probas"])]
