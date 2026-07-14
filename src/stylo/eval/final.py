"""Финальное сравнение: полный leakage-free LOBO для продакшен-модели и baseline-ов.

Sweep (GKF) отвечает «какие блоки работают». Здесь — ИТОГОВЫЕ цифры в отчёт:
полный LOBO для stylo + классических baseline-ов, с CI, парной значимостью против stylo
(McNemar по книгам — антиконсервативная граница; author-clustered bootstrap разницы
accuracy — кластер-робастная honest-значимость), и калибровкой (ECE) для stylo.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ..claims import ClaimStatus
from ..corpus import Dataset
from ..lang import display_name
from .dispatch import frozen_run_contract
from .lobo import _lobo_run, make_factory
from .metrics import (accuracy, expected_calibration_error, macro_f1,
                      summarize_book_results)
from .provenance import (UnsupportedVariantError, VariantRole,
                         verify_dataset_against_disk)
from .significance import mcnemar, paired_bootstrap_diff_clustered
from .work_weighting import (CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                             require_weighting, resolve_training_weighting)

log = logging.getLogger("stylo.eval.final")

DEFAULT_SPECS = ["stylo", "delta:150", "delta:300", "delta:500",
                 "delta_cos:150", "delta_cos:300", "delta_cos:500",
                 "char_cos", "bow_lr", "majority"]

# stylo_stack is intentionally explicit-only: it is a slow calibration/channel
# fusion experiment, not the default headline model.
ECE_SPECS = {"stylo", "stylo_stack"}


def _fold_manifest(df: pd.DataFrame):
    """Ordered (book_id, true_label) manifest — the exact evaluated fold set for paired stats."""
    man = [(f"{r.test_author}/{r.test_book}", int(r.true_label)) for r in df.itertuples()]
    ids = [b for b, _ in man]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate book id in a LOBO result — cannot form a fold manifest")
    return tuple(man)


def _correct_by_book(df: pd.DataFrame) -> Dict[str, bool]:
    return {f"{r.test_author}/{r.test_book}": bool(r.correct) for r in df.itertuples()}


def _variant_role(spec: str, weighting: str) -> str:
    if weighting == CHUNK_WEIGHTED_LEGACY:
        return VariantRole.PRIMARY.value
    if spec == "majority":
        return VariantRole.NOT_APPLICABLE.value
    if spec == "bow_lr_ref_legacy":
        return VariantRole.REFERENCE.value
    if spec == "stylo_stack":
        return VariantRole.BLOCKED_NOT_IMPLEMENTED.value
    return VariantRole.PRIMARY.value


def _estimator_weighting(spec: str, weighting: str) -> str:
    # majority (chunk-count prior) and the frozen reference row never adopt work weights
    if spec in ("majority", "bow_lr_ref_legacy"):
        return CHUNK_WEIGHTED_LEGACY
    return weighting


def _provenance_block(dataset: Dataset, weighting: str, specs: List[str]) -> Dict:
    prov = getattr(dataset, "provenance", None)
    return {
        "suite_weighting": weighting,
        "dataset_contract": getattr(prov, "loader_kind", None),
        "rows_digest": getattr(prov, "rows_digest", None),
        "claim_status": ClaimStatus.EXPLORATORY_INTERNAL.value,
        "per_spec": {
            spec: {
                "estimator_training_weighting": _estimator_weighting(spec, weighting),
                "variant_role": _variant_role(spec, weighting),
                "claim_status": ClaimStatus.EXPLORATORY_INTERNAL.value,
            }
            for spec in specs
        },
    }


def run_final(cfg, dataset: Dataset, specs: List[str] | None = None,
              n_jobs=None, *, weighting: str) -> Dict:
    weighting = require_weighting(weighting)
    weighting = verify_dataset_against_disk(cfg, dataset, weighting, frozen_run_contract(cfg))
    specs = list(specs or DEFAULT_SPECS)
    if len(set(specs)) != len(specs):
        raise ValueError(f"duplicate specs in the run-plan: {specs}")
    # BoW frozen reference is a work_balanced-only row: auto-added to the WB suite, forbidden legacy
    if weighting == WORK_BALANCED:
        if "bow_lr_ref_legacy" not in specs:
            specs.append("bow_lr_ref_legacy")
    elif "bow_lr_ref_legacy" in specs:
        raise UnsupportedVariantError("bow_lr_ref_legacy is forbidden in the legacy arm")
    if "stylo" not in specs:
        raise ValueError("run_final requires 'stylo' (the paired-comparison reference)")
    # preflight the WHOLE run-plan BEFORE the first fit: block variants + build every factory so a
    # bad spec fails before any expensive LOBO runs (no partial suite left behind)
    for spec in specs:
        if spec == "stylo_stack" and weighting == WORK_BALANCED:
            raise UnsupportedVariantError(
                "stylo_stack under work_balanced is blocked in B2-core (B2a wires it; B3 calibration)")
        make_factory(spec, cfg, weighting=weighting)()   # actually INSTANTIATE (catches e.g. delta.metric=bogus)
    iters = cfg.get_path("evaluation.bootstrap_iters", 1000)
    level = cfg.get_path("evaluation.ci_level", 0.95)
    seed = cfg.get_path("seed", 42)

    results: Dict[str, Dict] = {}
    for spec in specs:
        log.info("final LOBO[%s]: %s", weighting, spec)
        df, probs, ytrue = _lobo_run(cfg, dataset, spec, None, 0, n_jobs, weighting)   # verified once above
        summ = summarize_book_results(df["true_label"].to_numpy(), df["pred_label"].to_numpy(),
                                      df["rank"].to_numpy(), dataset.authors,
                                      iters=iters, level=level, seed=seed)
        ece = expected_calibration_error(probs, ytrue) if spec in ECE_SPECS else None
        results[spec] = {"df": df, "summary": summ, "ece": ece, "probs": probs, "ytrue": ytrue}

    ref = results["stylo"]
    ref_manifest = _fold_manifest(ref["df"])          # exact ordered (book, true_label) fold set
    ref_correct = _correct_by_book(ref["df"])
    n_authors = dataset.n_authors

    rows = []
    for spec, r in results.items():
        s = r["summary"]
        row = {"model": spec, "accuracy": s["accuracy"].point,
               "acc_ci": f"[{s['accuracy'].lo:.3f},{s['accuracy'].hi:.3f}]",
               "macro_f1": s["macro_f1"].point, "top2": s["top2"].point,
               "ece": r["ece"]}
        if spec != "stylo":
            # paired stats require the EXACT identical ordered fold manifest (book + true_label),
            # else point and CI would use different denominators — fail-closed.
            if _fold_manifest(r["df"]) != ref_manifest:
                raise ValueError(
                    f"paired comparison {spec} vs stylo has a non-identical fold manifest — refusing")
            cur = _correct_by_book(r["df"])
            books = [b for b, _ in ref_manifest]
            ca = np.array([ref_correct[b] for b in books])       # stylo
            cb = np.array([cur[b] for b in books])               # spec
            mc = mcnemar(ca, cb)
            # McNemar по книгам антиконсервативен (книги внутри автора коррелированы) — это
            # ГРАНИЦА. Кластер-робастная значимость: CI разницы accuracy с ресэмплом АВТОРОВ.
            # point и CI считаются как spec − stylo на ОДНОМ наборе (один знак и знаменатель).
            authors_common = np.array([b.split("/", 1)[0] for b in books])
            dci = paired_bootstrap_diff_clustered(
                lambda idx: float(cb[idx].mean()), lambda idx: float(ca[idx].mean()),   # spec − stylo
                authors_common, iters=iters, level=level, seed=seed)
            row["vs_stylo_dacc"] = dci.diff                      # SAME sign/denominator as the CI
            row["vs_stylo_mcnemar_p"] = mc.p_value
            row["vs_stylo_dacc_authorclustered_ci"] = f"[{dci.lo:+.3f},{dci.hi:+.3f}]"
            row["vs_stylo_dacc_authorclustered_sig"] = dci.significant
        else:
            row["vs_stylo_dacc"] = 0.0
            row["vs_stylo_mcnemar_p"] = float("nan")
            row["vs_stylo_dacc_authorclustered_ci"] = ""
            row["vs_stylo_dacc_authorclustered_sig"] = ""
        rows.append(row)

    table = pd.DataFrame(rows)
    # provenance is a SEPARATE block (never columns on the byte-parity headline table)
    return {"table": table, "results": results,
            "weighting": weighting, "provenance": _provenance_block(dataset, weighting, specs)}


def format_final(table: pd.DataFrame, results: Dict) -> str:
    lines = ["=== ФИНАЛЬНОЕ СРАВНЕНИЕ (полный leakage-free LOBO, book-level) ==="]
    hdr = (f"{'model':<12} {'acc':>7} {'acc 95% CI':>16} {'macroF1':>8} {'top2':>6} "
           f"{'Δvs_stylo':>10} {'McNemar_p':>10} {'Δacc authCl 95%CI':>20} {'ECE':>6}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in table.itertuples():
        p = "" if (isinstance(r.vs_stylo_mcnemar_p, float) and np.isnan(r.vs_stylo_mcnemar_p)) else f"{r.vs_stylo_mcnemar_p:.4f}"
        ece = "" if r.ece is None else f"{r.ece:.3f}"
        dacc = "" if r.model == "stylo" else f"{r.vs_stylo_dacc:+.3f}"
        clci = getattr(r, "vs_stylo_dacc_authorclustered_ci", "") or ""
        lines.append(f"{r.model:<12} {r.accuracy:>7.3f} {r.acc_ci:>16} {r.macro_f1:>8.3f} "
                     f"{r.top2:>6.3f} {dacc:>10} {p:>10} {clci:>20} {ece:>6}")
    lines.append("")
    lines.append("Per-author recall (stylo):")
    pa = results["stylo"]["summary"]["per_author_recall"]
    for a, v in sorted(pa.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -1)):
        lines.append(f"  {display_name(a):24} {v:.2f}" if v == v else f"  {display_name(a):24} n/a")
    return "\n".join(lines)
