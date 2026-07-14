"""Оркестратор ablation/feature-sweep — честный ответ «какие фичи работают».

Режимы:
  leave_one_out  : baseline = все включённые блоки; по очереди отключаем каждый блок
                   (и субблоки syntax) — значимое падение метрики => блок «работает».
  add_one_in     : от пустого набора добавляем по одному блоку — маржинальная ценность.
  baselines      : majority / delta:150,300,500 / char_cos / bow_lr — для сравнения.

Скрининг идёт быстрым прокси GroupKFold; финальные цифры — полным LOBO (strategy='lobo').
Для каждой строки: точность и macro-F1 с CI, Δ к reference и McNemar p (парно по книгам).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..corpus import Dataset
from ..features.registry import BLOCK_ORDER
from ..features.syntax import SUBBLOCK_ORDER
from .groupkfold import _gkf_run
from .lobo import _lobo_run
from .work_weighting import CHUNK_WEIGHTED_LEGACY
from .metrics import accuracy, macro_f1, summarize_book_results
from .significance import mcnemar

log = logging.getLogger("stylo.eval.sweep")


@dataclass
class EvalCase:
    label: str
    spec: str = "stylo"
    enabled_override: Optional[Dict[str, bool]] = None
    subblock_override: Optional[Dict[str, bool]] = None


def _clone_with(cfg, dotted_overrides: Dict[str, object]):
    from ..config import ConfigNode
    raw = cfg.to_dict()
    for k, v in dotted_overrides.items():
        node = raw
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return ConfigNode(raw)


def _evaluate_case(cfg, dataset: Dataset, case: EvalCase, strategy: str = "gkf",
                  n_jobs: Optional[int] = None, *, weighting: str, panel=None) -> Dict:
    eval_cfg = cfg
    if case.subblock_override:
        eval_cfg = _clone_with(cfg, {f"features.syntax.subblocks.{k}": v
                                     for k, v in case.subblock_override.items()})
    if strategy == "gkf":
        df, probs, ytrue = _gkf_run(eval_cfg, dataset, case.spec, case.enabled_override, None, weighting, panel)
    elif strategy == "lobo":
        df, probs, ytrue = _lobo_run(eval_cfg, dataset, case.spec, case.enabled_override,
                                     0, n_jobs, weighting)
    else:
        raise ValueError(f"unknown sweep strategy {strategy!r} (expected 'gkf' or 'lobo')")
    summ = summarize_book_results(df["true_label"].to_numpy(), df["pred_label"].to_numpy(),
                                  df["rank"].to_numpy(), dataset.authors,
                                  iters=cfg.get_path("evaluation.bootstrap_iters", 1000),
                                  level=cfg.get_path("evaluation.ci_level", 0.95),
                                  seed=cfg.get_path("seed", 42))
    return {"label": case.label, "df": df, "probs": probs, "summary": summ}


from .final import _correct_by_book, _fold_manifest  # noqa: E402  (shared exact-manifest helpers)


def default_cases(cfg) -> Tuple[EvalCase, List[EvalCase], List[EvalCase]]:
    """(full, leave_one_out_cases, baseline_cases) по включённым в конфиге блокам."""
    feats = cfg.get_path("features")
    enabled = [b for b in BLOCK_ORDER
               if feats.get_path(b) is not None and feats.get_path(b).get("enabled", False)]
    full = EvalCase("full(all-enabled)")

    loo: List[EvalCase] = []
    for b in enabled:
        loo.append(EvalCase(f"-{b}", enabled_override={b: False}))
    if "syntax" in enabled:
        subs = cfg.get_path("features.syntax.subblocks").to_dict()
        for s in SUBBLOCK_ORDER:
            if subs.get(s):
                loo.append(EvalCase(f"-syntax.{s}", subblock_override={s: False}))

    baselines = [
        EvalCase("base:majority", spec="majority"),
        EvalCase("base:delta-150", spec="delta:150"),
        EvalCase("base:delta-300", spec="delta:300"),
        EvalCase("base:delta-500", spec="delta:500"),
        EvalCase("base:char_cos", spec="char_cos"),
        EvalCase("base:bow_lr", spec="bow_lr"),
    ]
    return full, loo, baselines


def run_sweep(cfg, dataset: Dataset, strategy: str = "gkf",
              include_baselines: bool = True, n_jobs: Optional[int] = None,
              *, weighting: str) -> Dict:
    """Полный ablation-sweep. Возвращает {'table': DataFrame, 'cases': {...}}.

    При strategy='gkf' каждый конфиг — отдельная (внутри последовательная) GKF-оценка,
    поэтому КОНФИГИ считаются параллельно (joblib) — это снимает основное узкое место.
    При strategy='lobo' внутренняя LOBO уже параллельна по фолдам, поэтому конфиги
    считаем последовательно (без вложенного параллелизма).
    """
    if strategy not in ("gkf", "lobo"):                # validate BEFORE any fit (no silent LOBO)
        raise ValueError(f"unknown sweep strategy {strategy!r} (expected 'gkf' or 'lobo')")
    from .dispatch import frozen_run_contract
    from .provenance import verify_dataset_against_disk
    weighting = verify_dataset_against_disk(cfg, dataset, weighting, frozen_run_contract(cfg))
    # legacy GKF screening runs on the FROZEN screening_panel_v1 (43 authors / 251 works): every
    # sweep case sees the SAME folds, and each result is checked against the canonical manifest.
    panel = None
    if strategy == "gkf":
        from .groupkfold import bind_screening_panel
        dataset, panel = bind_screening_panel(cfg, dataset, weighting)
    full, loo, baselines = default_cases(cfg)
    cases = [full] + loo + (baselines if include_baselines else [])

    if strategy == "gkf":
        from joblib import Parallel, delayed
        njobs = n_jobs if n_jobs is not None else cfg.get_path("evaluation.n_jobs", -1)
        log.info("sweep[gkf/%s]: %d конфигов параллельно (n_jobs=%s, panel=%s)",
                 weighting, len(cases), njobs, panel is not None)
        out = Parallel(n_jobs=njobs, pre_dispatch="2*n_jobs")(
            delayed(_evaluate_case)(cfg, dataset, case, "gkf", 1, weighting=weighting, panel=panel)
            for case in cases
        )
        results = {case.label: r for case, r in zip(cases, out)}
    else:
        results = {}
        for case in cases:
            log.info("sweep[lobo/%s]: %s", weighting, case.label)
            results[case.label] = _evaluate_case(cfg, dataset, case, "lobo", n_jobs, weighting=weighting)

    ref = results[full.label]
    ref_manifest = _fold_manifest(ref["df"])          # ordered (book, true_label) — exact fold set
    ref_correct = _correct_by_book(ref["df"])
    ref_acc = ref["summary"]["accuracy"].point

    table_rows = []
    for label, r in results.items():
        s = r["summary"]
        row = {
            "config": label,
            "n_books": s["n_books"],
            "accuracy": s["accuracy"].point,
            "acc_ci": f"[{s['accuracy'].lo:.3f},{s['accuracy'].hi:.3f}]",
            "macro_f1": s["macro_f1"].point,
            "f1_ci": f"[{s['macro_f1'].lo:.3f},{s['macro_f1'].hi:.3f}]",
            "top2": s["top2"].point,
            "delta_acc_vs_full": s["accuracy"].point - ref_acc,
        }
        # McNemar только для блок-ablation (парно по ТОЧНО тому же fold-manifest, что и full)
        if label.startswith("-"):
            if _fold_manifest(r["df"]) != ref_manifest:
                raise ValueError(f"ablation {label} has a non-identical fold manifest vs full — refusing")
            cur = _correct_by_book(r["df"])
            books = [b for b, _ in ref_manifest]
            mc = mcnemar(np.array([ref_correct[b] for b in books]),
                         np.array([cur[b] for b in books]))
            row["mcnemar_p"] = mc.p_value
        else:
            row["mcnemar_p"] = float("nan")
        table_rows.append(row)

    table = pd.DataFrame(table_rows)
    table["_order"] = table["config"].apply(lambda c: 0 if c == full.label else (2 if c.startswith("base:") else 1))
    table = table.sort_values(["_order", "delta_acc_vs_full"]).drop(columns="_order").reset_index(drop=True)
    return {"table": table, "cases": results, "reference": full.label}


def format_sweep_table(table: pd.DataFrame) -> str:
    lines = []
    header = f"{'config':<22} {'acc':>7} {'acc_ci':>16} {'macroF1':>8} {'top2':>6} {'Δacc':>7} {'McNemar_p':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in table.itertuples():
        p = "" if (isinstance(r.mcnemar_p, float) and np.isnan(r.mcnemar_p)) else f"{r.mcnemar_p:.4f}"
        sig = " *" if (p and r.mcnemar_p < 0.05) else ""
        lines.append(f"{r.config:<22} {r.accuracy:>7.3f} {r.acc_ci:>16} {r.macro_f1:>8.3f} "
                     f"{r.top2:>6.3f} {r.delta_acc_vs_full:>+7.3f} {p:>10}{sig}")
    return "\n".join(lines)
