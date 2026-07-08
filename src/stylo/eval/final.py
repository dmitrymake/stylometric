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

from ..corpus import Dataset
from ..lang import display_name
from .lobo import lobo_evaluate
from .metrics import (accuracy, expected_calibration_error, macro_f1,
                      summarize_book_results)
from .significance import mcnemar, paired_bootstrap_diff_clustered

log = logging.getLogger("stylo.eval.final")

DEFAULT_SPECS = ["stylo", "delta:150", "delta:300", "delta:500",
                 "delta_cos:150", "delta_cos:300", "delta_cos:500",
                 "char_cos", "bow_lr", "majority"]

# stylo_stack is intentionally explicit-only: it is a slow calibration/channel
# fusion experiment, not the default headline model.
ECE_SPECS = {"stylo", "stylo_stack"}


def _correct_map(df: pd.DataFrame) -> Dict[str, bool]:
    return {f"{r.test_author}/{r.test_book}": bool(r.correct) for r in df.itertuples()}


def run_final(cfg, dataset: Dataset, specs: List[str] | None = None,
              n_jobs=None) -> Dict:
    specs = specs or DEFAULT_SPECS
    iters = cfg.get_path("evaluation.bootstrap_iters", 1000)
    level = cfg.get_path("evaluation.ci_level", 0.95)
    seed = cfg.get_path("seed", 42)

    results: Dict[str, Dict] = {}
    for spec in specs:
        log.info("final LOBO: %s", spec)
        df, probs, ytrue = lobo_evaluate(cfg, dataset, spec=spec, n_jobs=n_jobs)
        summ = summarize_book_results(df["true_label"].to_numpy(), df["pred_label"].to_numpy(),
                                      df["rank"].to_numpy(), dataset.authors,
                                      iters=iters, level=level, seed=seed)
        ece = expected_calibration_error(probs, ytrue) if spec in ECE_SPECS else None
        results[spec] = {"df": df, "summary": summ, "ece": ece, "probs": probs, "ytrue": ytrue}

    ref = results["stylo"]
    ref_correct = _correct_map(ref["df"])
    n_authors = dataset.n_authors

    rows = []
    for spec, r in results.items():
        s = r["summary"]
        row = {"model": spec, "accuracy": s["accuracy"].point,
               "acc_ci": f"[{s['accuracy'].lo:.3f},{s['accuracy'].hi:.3f}]",
               "macro_f1": s["macro_f1"].point, "top2": s["top2"].point,
               "ece": r["ece"]}
        if spec != "stylo":
            cur = _correct_map(r["df"])
            common = sorted(set(ref_correct) & set(cur))
            ca = np.array([ref_correct[b] for b in common])
            cb = np.array([cur[b] for b in common])
            mc = mcnemar(ca, cb)
            row["vs_stylo_dacc"] = s["accuracy"].point - ref["summary"]["accuracy"].point
            row["vs_stylo_mcnemar_p"] = mc.p_value
            # McNemar по книгам антиконсервативен (книги внутри автора коррелированы) — это
            # ГРАНИЦА. Кластер-робастная значимость: CI разницы accuracy с ресэмплом АВТОРОВ.
            authors_common = np.array([b.split("/", 1)[0] for b in common])
            dci = paired_bootstrap_diff_clustered(
                lambda idx: float(ca[idx].mean()), lambda idx: float(cb[idx].mean()),
                authors_common, iters=iters, level=level, seed=seed)
            row["vs_stylo_dacc_authorclustered_ci"] = f"[{dci.lo:+.3f},{dci.hi:+.3f}]"
            row["vs_stylo_dacc_authorclustered_sig"] = dci.significant
        else:
            row["vs_stylo_dacc"] = 0.0
            row["vs_stylo_mcnemar_p"] = float("nan")
            row["vs_stylo_dacc_authorclustered_ci"] = ""
            row["vs_stylo_dacc_authorclustered_sig"] = ""
        rows.append(row)

    table = pd.DataFrame(rows)
    return {"table": table, "results": results}


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
