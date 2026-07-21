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

import math

import numpy as np

from ...eval.metrics import macro_f1
from . import headline as hl
from . import inference as inf
from .applicability import holm_family, registered_cells

_DATASETS = ("lobo", "ruaa")
# which registered tolerance each recomputed quantity is compared under
_ACCURACY, _DELTA, _PVALUE, _CI = "accuracy", "delta_accuracy", "cluster_pvalue", "ci_endpoint"


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
    }


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

        raw_ps = {}
        for (model, cell) in registered_cells():
            a = arrays[(model, cell)]
            rec = cells[f"{model}/{cell}"]
            point = _point(a, metric_idx)
            for k in ("accuracy", "macro_f1", "top2"):
                _require(_close(point[k], rec["point"][k], tol_acc),
                         f"{ds}/{model}/{cell} {k} recompute {point[k]} != {rec['point'][k]}")
            for au, r in point["per_author_recall"].items():
                _require(_close(r, rec["point"]["per_author_recall"].get(au, float("nan")), tol_acc),
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
