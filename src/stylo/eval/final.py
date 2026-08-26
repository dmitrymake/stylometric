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
from ..jsonio import canonical_hash
from ..lang import display_name
from ..models.registry import (
    CALIBRATION_MODEL_SPECS,
    DEFAULT_EXPLORATORY_SPECS,
)
from .lobo import (
    _lobo_run,
    build_generic_lobo_fold_manifest,
    make_factory,
    validate_generic_lobo_result,
)
from .metrics import (
    AuthorClusteredInferenceSpec,
    accuracy,
    expected_calibration_error,
    macro_f1,
    summarize_book_results,
)
from .provenance import (
    UnsupportedVariantError,
    VariantRole,
    prepare_scientific_evaluation,
)
from .significance import mcnemar, paired_bootstrap_diff_clustered
from ..domain.work_weighting import (CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                             require_weighting, resolve_training_weighting)

log = logging.getLogger("stylo.eval.final")
TOPOLOGY_ROLE = "exploratory_model_comparison_compatibility_module"

DEFAULT_SPECS = list(DEFAULT_EXPLORATORY_SPECS)

# Both channel-fusion estimators are intentionally explicit-only, not default
# headline models. The legacy stack remains listed so historical runs retain
# their ECE contract; its class-coverage preflight now fails closed.
ECE_SPECS = set(CALIBRATION_MODEL_SPECS)


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
    # stylo_stack is fully routed for work-balanced training and group-aware calibration.
    return VariantRole.PRIMARY.value


def _estimator_weighting(spec: str, weighting: str) -> str:
    # majority (chunk-count prior) and the frozen reference row never adopt work weights
    if spec in ("majority", "bow_lr_ref_legacy"):
        return CHUNK_WEIGHTED_LEGACY
    return weighting


def _provenance_block(
    dataset,
    weighting: str,
    specs: List[str],
    *,
    fold_manifest,
    inference_spec: AuthorClusteredInferenceSpec,
) -> Dict:
    prov = getattr(dataset, "provenance", None)
    identity = {
        "schema_version": "stylo.generic-final-run-identity.v2",
        "rows_digest": getattr(prov, "rows_digest", None),
        "training_weighting": weighting,
        "overlap_policy_version": dataset.isolation_contract_version,
        "fold_manifest_self_hash": fold_manifest.self_hash,
        "model_specs": list(specs),
        "inference_spec_self_hash": inference_spec.self_hash,
    }
    run_id = canonical_hash(identity)
    return {
        "suite_weighting": weighting,
        "dataset_contract": getattr(prov, "loader_kind", None),
        "rows_digest": getattr(prov, "rows_digest", None),
        "overlap_policy_version": dataset.isolation_contract_version,
        "fold_manifest": fold_manifest.as_dict(),
        "inference_spec": inference_spec.as_dict(),
        "run_identity": {**identity, "run_id": run_id},
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
    context = prepare_scientific_evaluation(
        cfg,
        dataset,
        weighting,
    )
    dataset = context
    weighting = context.weighting
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
    iters = cfg.get_path("evaluation.bootstrap_iters", 1000)
    level = cfg.get_path("evaluation.ci_level", 0.95)
    seed = cfg.get_path("seed", 42)
    inference_spec = AuthorClusteredInferenceSpec.build(
        iterations=iters,
        confidence_level=level,
        seed=seed,
    )
    fold_manifest = build_generic_lobo_fold_manifest(context)
    # preflight the WHOLE run-plan BEFORE the first fit: build every factory so a bad spec fails
    # before any expensive LOBO runs (no partial suite left behind)
    for spec in specs:
        make_factory(spec, cfg, weighting=weighting)()   # actually INSTANTIATE (catches e.g. delta.metric=bogus)

    results: Dict[str, Dict] = {}
    for spec in specs:
        log.info("final LOBO[%s]: %s", weighting, spec)
        df, probs, ytrue = _lobo_run(
            cfg,
            context,
            spec,
            None,
            0,
            n_jobs,
            fold_manifest=fold_manifest,
        )  # provenance + content isolation verified once above
        validate_generic_lobo_result(df, fold_manifest)
        summ = summarize_book_results(
            df["true_label"].to_numpy(),
            df["pred_label"].to_numpy(),
            df["rank"].to_numpy(),
            probability_class_order=fold_manifest.probability_class_order,
            metric_label_order=fold_manifest.metric_label_order,
            book_authors=fold_manifest.book_authors,
            inference_spec=inference_spec,
        )
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
            "weighting": weighting, "provenance": _provenance_block(
                dataset,
                weighting,
                specs,
                fold_manifest=fold_manifest,
                inference_spec=inference_spec,
            )}


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
