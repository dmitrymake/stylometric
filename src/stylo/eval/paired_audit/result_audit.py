"""Independent result-auditor (§8): recompute every published metric from the per-work vectors.

The runner assembles a COMPLETE candidate summary; this module is a SEPARATE code path that
reconstructs the per-cell arrays from the published per-work vectors (each carries the true/pred
labels, correctness, rank and probability vector) and INDEPENDENTLY recomputes accuracy, macro-F1,
top-2, per-author recall, the paired Δaccuracy, the author-clustered CIs, the null-centered cluster
p-values, the Holm family and the headline difference CI — then asserts each equals the candidate
within the run's registered tolerances. The publisher accepts only a candidate that passes this audit,
and the headline DECISION is stamped as a distinct later stage from the audited numbers.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from ...eval.metrics import macro_f1
from ...jsonio import dumps_strict
from . import headline as hl
from . import inference as inf
from .applicability import holm_family, registered_cells
from .checkpoints import proba_digest as _proba_digest

_DATASETS = ("lobo", "ruaa")
# which registered tolerance each recomputed quantity is compared under
_ACCURACY, _DELTA, _PVALUE, _CI = "accuracy", "delta_accuracy", "cluster_pvalue", "ci_endpoint"
_PROBA_SUM_TOL = 1e-9
# the frozen confirmatory universe (§1.1/§12): a 2-author toy fixture must NEVER pass as production
_FROZEN_UNIVERSE = {"lobo": {"n_prob": 47, "n_metric": 43, "n_tested_works": 251},
                    "ruaa": {"n_prob": 22, "n_metric": 22, "n_tested_works": 137}}


class ResultAuditError(RuntimeError):
    """Fail-closed: an independently recomputed metric does not match the candidate summary."""


def _tol(plan, quantity):
    t = plan.get("tolerances", {}).get(quantity)
    return (float(t["atol"]), float(t["rtol"])) if isinstance(t, dict) else (0.0, 0.0)


def _close(got, exp, tol) -> bool:
    atol, rtol = tol
    return math.isfinite(got) and math.isfinite(exp) and math.isclose(got, exp, rel_tol=rtol, abs_tol=atol)


def _require(cond, msg):
    if not cond:
        raise ResultAuditError(msg)


def _arrays(vec):
    """Reconstruct the per-cell arrays from a per-work vector, sorted by work_id (metrics + the
    author-clustered bootstrap are row-order invariant, and sorting aligns the paired A0 arm)."""
    rows = sorted(vec, key=lambda v: v["work_id"])
    return {
        "works": [v["work_id"] for v in rows],
        "correct": [1 if v["correct"] else 0 for v in rows],
        "authors": [str(v["work_id"]).split("/", 1)[0] for v in rows],
        "ranks": [int(v["rank"]) for v in rows],
        "trues": [int(v["true_label"]) for v in rows],
        "preds": [int(v["pred_label"]) for v in rows],
        "probas": [v["proba"] for v in rows],
    }


def _validate_fold_coherence(a, prob_order, label):
    """Re-validate EVERY per-work row from scratch at the publish boundary (the auditor never trusts
    the vectors): the probability vector must be finite, in [0,1], sum to 1, of the class-order width;
    pred==argmax; rank consistent with the true label's position; correct==(pred==true); a correct fold
    has rank 1; and — decisively — the true_label MUST equal ``prob_order.index(work_id author)`` (a
    permuted true_label that contradicts the author in the work_id is fatal)."""
    width = len(prob_order)
    n = len(a["works"])
    for i in range(n):
        w, proba = a["works"][i], a["probas"][i]
        pred, true, rank, correct = a["preds"][i], a["trues"][i], a["ranks"][i], a["correct"][i]
        if not isinstance(proba, list) or len(proba) != width:
            raise ResultAuditError(f"{label} {w}: proba width {len(proba) if isinstance(proba, list) else '?'} != {width}")
        for p in proba:
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not (0.0 <= p <= 1.0):
                raise ResultAuditError(f"{label} {w}: proba must be finite numbers in [0,1] (got {proba})")
        if abs(sum(proba) - 1.0) > _PROBA_SUM_TOL:
            raise ResultAuditError(f"{label} {w}: proba must sum to 1 (got {sum(proba)})")
        if not (0 <= pred < width) or proba[pred] != max(proba):
            raise ResultAuditError(f"{label} {w}: pred_label is not the argmax of the proba vector")
        if not (0 <= true < width):
            raise ResultAuditError(f"{label} {w}: true_label {true} out of range")
        author = str(w).split("/", 1)[0]
        if author not in prob_order or true != prob_order.index(author):
            raise ResultAuditError(f"{label} {w}: true_label {true} != the class-order index of author {author!r}")
        if bool(correct) != (pred == true):
            raise ResultAuditError(f"{label} {w}: correct != (pred==true)")
        expected_rank = 1 + sum(1 for p in proba if p > proba[true])
        if rank != expected_rank:
            raise ResultAuditError(f"{label} {w}: rank {rank} != the true label's rank {expected_rank}")
        if correct and rank != 1:
            raise ResultAuditError(f"{label} {w}: a correct fold must have rank 1")


def _cell_proba_digest(a) -> str:
    """Independently recompute the cell-level proba_digest aggregate from the per-work probability
    vectors (each hashed by the canonical :func:`proba_digest`), matching the runner's aggregation — so
    a published proba_digest that does not correspond to the actual probability vectors is fatal."""
    parts = sorted([str(w), _proba_digest(proba)] for w, proba in zip(a["works"], a["probas"]))
    body = ["evidence", "proba_digest", parts]
    return hashlib.sha256(dumps_strict(body, sort_keys=True).encode("utf-8")).hexdigest()


def _assert_frozen_universe(ds, prob_order, metric_order, works):
    fu = _FROZEN_UNIVERSE[ds]
    n_tested = len(set(works))
    if (len(prob_order), len(metric_order), n_tested) != (fu["n_prob"], fu["n_metric"], fu["n_tested_works"]):
        raise ResultAuditError(
            f"{ds} universe {len(prob_order)}/{len(metric_order)}/{n_tested} authors/tested-authors/"
            f"tested-works != the frozen {fu['n_prob']}/{fu['n_metric']}/{fu['n_tested_works']}")


def _point(a, metric_idx):
    n = len(a["correct"]) or 1
    acc = sum(a["correct"]) / n
    top2 = sum(1 for r in a["ranks"] if r <= 2) / n
    f1 = float(macro_f1(np.array(a["trues"]), np.array(a["preds"]), metric_idx)) if a["trues"] else 0.0
    recall = {}
    for au in sorted(set(a["authors"])):
        idx = [i for i, x in enumerate(a["authors"]) if x == au]
        recall[au] = sum(a["correct"][i] for i in idx) / len(idx)
    return {"accuracy": acc, "macro_f1": f1, "top2": top2, "per_author_recall": recall}


def audit_results(summary, per_work_vectors, plan) -> dict:
    """Independently recompute every metric/CI/p-value/Holm/headline from the per-work vectors and
    assert it matches the candidate ``summary`` within the run's registered tolerances. Returns the
    audited headline (endpoint + diff CI + decision) for the separate headline-decision stage; raises
    :class:`ResultAuditError` on any disagreement."""
    stats = plan["stats"]
    iters, seed = stats["bootstrap_iters"], stats["seed"]
    quantiles, margin, B = stats["quantiles"], stats["noninferiority_margin"], stats["bootstrap_B"]
    tol_acc, tol_delta = _tol(plan, _ACCURACY), _tol(plan, _DELTA)
    tol_p, tol_ci = _tol(plan, _PVALUE), _tol(plan, _CI)
    confirmatory = plan.get("run_kind") == "confirmatory"

    for ds in _DATASETS:
        uni = summary["universes"][ds]
        prob_order, metric_order = uni["probability_class_order"], uni["metric_label_order"]
        metric_idx = [prob_order.index(a) for a in metric_order]
        cells = summary["cells"][ds]
        arrays = {}
        for (model, cell) in registered_cells():
            key = f"{ds}/{model}/{cell}"
            _require(key in per_work_vectors, f"missing per-work vector {key}")
            arrays[(model, cell)] = _arrays(per_work_vectors[key])
        # a confirmatory run must be the FROZEN universe (a 2-author toy fixture is not production-ready)
        if confirmatory:
            _assert_frozen_universe(ds, prob_order, metric_order, arrays[("stylo", "A0")]["works"])

        raw_ps = {}
        for (model, cell) in registered_cells():
            a = arrays[(model, cell)]
            rec = cells[f"{model}/{cell}"]
            # re-validate every per-work row (invalid proba, permuted true_label, bad rank -> fatal)
            _validate_fold_coherence(a, prob_order, f"{ds}/{model}/{cell}")
            _require(rec.get("evidence", {}).get("proba_digest") == _cell_proba_digest(a),
                     f"{ds}/{model}/{cell} evidence.proba_digest != the recomputed proba-vector aggregate")
            point = _point(a, metric_idx)
            for k in ("accuracy", "macro_f1", "top2"):
                _require(_close(point[k], rec["point"][k], tol_acc),
                         f"{ds}/{model}/{cell} {k} recompute {point[k]} != {rec['point'][k]}")
            pub_recall = rec["point"].get("per_author_recall", {})
            _require(isinstance(pub_recall, dict) and set(pub_recall) == set(point["per_author_recall"]),
                     f"{ds}/{model}/{cell} per_author_recall author set != recompute (a ghost author?)")
            for au, r in point["per_author_recall"].items():
                _require(_close(r, pub_recall.get(au, float("nan")), tol_acc),
                         f"{ds}/{model}/{cell} recall[{au}] mismatch")
            abs_ci = hl.author_clustered_accuracy_ci(a["correct"], a["authors"], iters=iters, seed=seed,
                                                     quantiles=quantiles)
            for got, exp in zip((abs_ci["lo"], abs_ci["hi"]), rec["abs_accuracy_authorclustered_ci"]):
                _require(_close(got, exp, tol_ci), f"{ds}/{model}/{cell} abs CI recompute mismatch")

            if cell != "A0":
                base = arrays[(model, "A0")]
                _require(a["works"] == base["works"], f"{ds}/{model}/{cell} work set != its A0 arm")
                dacc = point["accuracy"] - _point(base, metric_idx)["accuracy"]
                _require(_close(dacc, rec["vs_A0"]["dacc"], tol_delta),
                         f"{ds}/{model}/{cell} dacc recompute {dacc} != {rec['vs_A0']['dacc']}")
                dacc_ci = hl.paired_accuracy_diff_ci(a["correct"], base["correct"], a["authors"],
                                                     iters=iters, seed=seed, quantiles=quantiles)
                for got, exp in zip((dacc_ci["lo"], dacc_ci["hi"]), rec["vs_A0"]["dacc_authorclustered_ci"]):
                    _require(_close(got, exp, tol_ci), f"{ds}/{model}/{cell} dacc CI recompute mismatch")
                cp = inf.paired_cluster_pvalue(a["correct"], base["correct"], a["authors"], B=B, seed=seed)
                _require(_close(cp, rec["vs_A0"]["cluster_p"], tol_p),
                         f"{ds}/{model}/{cell} cluster_p recompute {cp} != {rec['vs_A0']['cluster_p']}")
                mc = inf.mcnemar_diagnostic(a["correct"], base["correct"])   # diagnostic-only, but published
                _require(_close(mc["mcnemar_p_diagnostic"], rec["vs_A0"].get("mcnemar_p_diagnostic",
                                                                            float("nan")), tol_p),
                         f"{ds}/{model}/{cell} mcnemar_p_diagnostic recompute mismatch")
                raw_ps[(model, cell)] = cp

        # Holm over the independently recomputed cluster p-values must match the published Holm family
        holm = inf.holm_over_registered_family(raw_ps)
        published = summary["holm"][ds]
        _require(set(f"{m}/{c}" for (m, c) in holm) == set(published),
                 f"{ds} Holm family keys differ from the published set")
        for (model, cell), hp in holm.items():
            pub_hp = published[f"{model}/{cell}"]
            _require(_close(hp["holm_p"], pub_hp["holm_p"], tol_p),
                     f"{ds} Holm p mismatch for {model}/{cell}")
            _require(_close(hp["raw_p"], pub_hp.get("raw_p", float("nan")), tol_p),
                     f"{ds} Holm raw_p mismatch for {model}/{cell}")
            _require(bool(hp["significant"]) == bool(pub_hp["significant"]),
                     f"{ds} Holm significance mismatch for {model}/{cell}")
            # Holm<->cell consistency: the cell's vs_A0 must carry the same Holm verdict
            vs = summary["cells"][ds][f"{model}/{cell}"]["vs_A0"]
            _require(_close(vs["holm_p"], hp["holm_p"], tol_p) and bool(vs["significant"]) == bool(hp["significant"]),
                     f"{ds} cell vs_A0 Holm verdict != the Holm family for {model}/{cell}")

    # headline: recompute the stylo LOBO A4-A0 difference CI and gate decision from the vectors
    la4 = arrays_for(summary, per_work_vectors, "lobo", "stylo", "A4")
    la0 = arrays_for(summary, per_work_vectors, "lobo", "stylo", "A0")
    diff_ci = hl.paired_accuracy_diff_ci(la4["correct"], la0["correct"], la4["authors"],
                                         iters=iters, seed=seed, quantiles=quantiles)
    head = summary["headline"]
    for got, exp in zip((diff_ci["lo"], diff_ci["hi"]), (head["diff_ci"]["lo"], head["diff_ci"]["hi"])):
        _require(_close(got, exp, tol_ci), "headline diff CI recompute mismatch")
    decision = hl.headline_gate(diff_ci["lo"], diff_ci["hi"], margin=margin)
    return {"passed": True, "auditor": "independent_recompute_v1",
            "headline": {"endpoint": hl.HEADLINE_ENDPOINT, "diff_ci": diff_ci, "decision": decision,
                         "margin": margin}}


def arrays_for(summary, per_work_vectors, ds, model, cell):
    return _arrays(per_work_vectors[f"{ds}/{model}/{cell}"])
