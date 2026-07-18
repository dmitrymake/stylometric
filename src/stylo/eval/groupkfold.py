"""Быстрый прокси-оценщик: StratifiedGroupKFold по книгам (k фолдов; полный LOBO — ~100).

Используется в sweep для СКРИНИНГА блоков (минуты; полный LOBO — часы). Книги не дробятся
между train/test (groups=book) — утечки book-level нет. Это лишь ранжирующий прокси:
финальные цифры в отчёт идут ТОЛЬКО из полного LOBO (eval/lobo.py).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..corpus import Dataset
from .dispatch import fit_estimator, frozen_run_contract
from .lobo import _align_proba, _validate_proba, make_factory
from .provenance import verify_dataset_against_disk
from .work_weighting import CHUNK_WEIGHTED_LEGACY, require_weighting

log = logging.getLogger("stylo.eval.groupkfold")


def gkf_evaluate(
    cfg,
    dataset: Dataset,
    spec: str = "stylo",
    enabled_override: Optional[Dict[str, bool]] = None,
    k: Optional[int] = None,
    *,
    weighting: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """k-fold прокси. Публичный вход всегда проверяет dataset против cfg-выведенного контракта."""
    weighting = require_weighting(weighting)
    weighting = verify_dataset_against_disk(cfg, dataset, weighting, frozen_run_contract(cfg))
    return _gkf_run(cfg, dataset, spec, enabled_override, k, weighting)


def _gkf_run(cfg, dataset, spec, enabled_override, k, weighting, panel=None):
    """Internal GKF worker — NO verification (caller must have verified). Not a public API.

    When a frozen ``screening_panel_v1`` manifest is supplied, the dataset is ALREADY the panel
    subset and evaluation uses the frozen folds (every sweep case sees identical folds), with each
    result checked against the canonical manifest. Otherwise the legacy per-run StratifiedGroupKFold
    proxy is used (toy datasets / non-canonical corpora / the work_balanced arm)."""
    if panel is not None:
        return _gkf_run_panel(cfg, dataset, spec, enabled_override, weighting, panel)
    if k is None:
        k = cfg.get_path("evaluation.groupkfold_k", 5)
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)

    texts, y, groups = dataset.texts, dataset.y, dataset.groups
    n_authors, authors = dataset.n_authors, dataset.authors
    factory = make_factory(spec, cfg, enabled_override, weighting=weighting)

    # fail-closed preflight: every class needs >=2 works for any leak-free CV; else a class can be
    # absent from a train fold (majority then invents plausible rows for an unlearnable class).
    book_author = dataset.book_to_author()
    per_author_books: Dict[int, int] = {}
    for lbl in book_author.values():
        per_author_books[lbl] = per_author_books.get(lbl, 0) + 1
    min_works = min(per_author_books.values())
    if min_works < 2:
        raise ValueError(
            f"GKF requires every class to have >=2 works; got min_works_per_class={min_works}")
    k_eff = max(2, min(k, min_works))

    sgkf = StratifiedGroupKFold(n_splits=k_eff, shuffle=True,
                                random_state=cfg.get_path("seed", 42))
    all_classes = set(np.unique(y).tolist())
    splits = list(sgkf.split(texts, y, groups))
    for tr, te in splits:                       # every split's TRAIN fold must contain all classes
        if set(np.unique(y[tr]).tolist()) != all_classes:
            raise ValueError("GKF split leaves a class absent from train — refusing (fail-closed)")
    rows = []
    prob_by_book: Dict[str, np.ndarray] = {}

    for tr, te in splits:
        est = factory()
        fit_estimator(est, texts[tr], y[tr], groups[tr])   # единый dispatch (groups iff needs_groups)
        proba = np.asarray(est.predict_proba(texts[te]))
        _validate_proba(proba, est.classes_, n_authors, len(te))   # fail-closed (same as LOBO)
        classes_ = np.asarray(est.classes_)
        te_groups = groups[te]
        for g in np.unique(te_groups):
            m = te_groups == g
            full = _align_proba(proba[m], classes_, n_authors)
            prob_by_book[str(g)] = full

    for g, full in prob_by_book.items():
        true_label = book_author[g]
        order = np.argsort(-full, kind="stable")  # равные → меньший индекс (как argmax/predict), без смещения к старшему
        top1 = int(order[0])
        rank = int((full >= full[true_label]).sum())  # tie-aware худший ранг: классы с prob >= p_true (включая сам класс); при вырожденных скорах (majority: нули) ничья не даёт ложного «2-го места»
        author_id, book_id = g.split("/", 1)
        rows.append({
            "test_author": author_id, "test_book": book_id,
            "true_label": true_label, "pred_label": top1,
            "pred_author": authors[top1], "correct": bool(top1 == true_label),
            "rank": rank, "confidence": float(full[true_label]),
            "top_candidates": [(authors[int(i)], float(full[int(i)])) for i in order[:top_k]],
            "_prob": full,
        })

    df = pd.DataFrame(rows)
    prob_matrix = np.vstack([r["_prob"] for r in rows])
    df = df.drop(columns=["_prob"])
    y_true = df["true_label"].to_numpy()
    log.info("GKF[%s]: %d книг, k=%d", spec, len(df), k_eff)
    return df, prob_matrix, y_true


def bind_screening_panel(cfg, dataset, weighting):
    """Return (panel_subset_dataset, manifest) for the frozen screening panel.

    For the LEGACY arm the panel is MANDATORY and self-verifying is not enough: the committed
    ``docs/screening_panel_v1.json`` must exist, pass its integrity checks, AND be byte-for-byte the
    manifest **rebuilt from the disk-verified corpus** (``build_manifest``, k=5, seed=42). A missing,
    tampered, truncated or stale manifest is a HARD FAILURE — never a silent fall back to the dynamic
    ``StratifiedGroupKFold`` proxy. Non-legacy arms use the dynamic proxy (return ``dataset, None``)."""
    from .screening_panel import (build_manifest, build_panel_subset,
                                  load_manifest_file, manifest_docs_path)
    if weighting != CHUNK_WEIGHTED_LEGACY:
        return dataset, None
    committed = load_manifest_file(manifest_docs_path(cfg))   # raises on missing / integrity failure
    expected = build_manifest(dataset)                        # rebuilt from the disk-verified corpus
    if committed != expected:                                 # exact canonical equality — no self-signed panel
        from .screening_panel import ScreeningPanelError
        raise ScreeningPanelError(
            "committed docs/screening_panel_v1.json != manifest rebuilt from the corpus "
            "(truncated / tampered / stale panel or corpus drift); regenerate via "
            "scripts/build_screening_panel.py — refusing to screen on a self-signed panel")
    return build_panel_subset(dataset, committed), committed


def _gkf_run_panel(cfg, dataset, spec, enabled_override, weighting, manifest):
    """GKF over the FROZEN screening_panel_v1 folds; ``dataset`` is already the panel subset."""
    factory = make_factory(spec, cfg, enabled_override, weighting=weighting)
    df, prob_matrix, y_true, _timing = evaluate_frozen_panel_factory(
        cfg, dataset, factory, manifest, spec=spec)
    return df, prob_matrix, y_true


def evaluate_frozen_panel_factory(
    cfg,
    dataset,
    factory,
    manifest,
    *,
    spec="injected",
    clock=None,
):
    """Evaluate a fresh estimator from ``factory`` on each frozen screening-panel fold.

    This is the reusable injected-factory seam for exploratory evaluations.  It deliberately keeps
    the frozen-panel validation, class alignment, one-average-per-work aggregation and deterministic
    row order identical to :func:`_gkf_run_panel`, while exposing fold-level wall-clock timings.
    """
    from .screening_panel import (ScreeningPanelError, verify_result_against_panel)
    if clock is None:
        clock = time.perf_counter
    total_started = clock()
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)
    authors = list(dataset.authors)
    n_authors = dataset.n_authors
    if authors != list(manifest["authors"]) or n_authors != manifest["n_authors"]:
        raise ScreeningPanelError("panel subset author space != canonical manifest")
    work_fold = {w["work_id"]: w["fold"] for w in manifest["works"]}
    groups = dataset.groups
    if {str(g) for g in np.unique(groups)} != set(work_fold):
        raise ScreeningPanelError("panel subset works != canonical manifest works")
    row_fold = np.array([work_fold[str(g)] for g in groups], dtype=int)
    k = manifest["k_folds"]
    actual_sizes = [int(np.unique(groups[row_fold == f]).size) for f in range(k)]   # actual per-fold works
    if actual_sizes != list(manifest["fold_sizes"]):
        raise ScreeningPanelError(f"actual fold_sizes {actual_sizes} != manifest {manifest['fold_sizes']}")
    texts, y = dataset.texts, dataset.y

    prob_by_work: Dict[str, np.ndarray] = {}
    fold_timings = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for f in range(k):
        te = np.flatnonzero(row_fold == f)
        tr = np.flatnonzero(row_fold != f)
        est = factory()
        fit_started = clock()
        fit_estimator(est, texts[tr], y[tr], groups[tr])
        fold_fit_seconds = float(clock() - fit_started)
        predict_started = clock()
        proba = np.asarray(est.predict_proba(texts[te]))
        fold_predict_seconds = float(clock() - predict_started)
        fit_seconds += fold_fit_seconds
        predict_seconds += fold_predict_seconds
        _validate_proba(proba, est.classes_, n_authors, len(te))   # fail-closed (same as LOBO)
        classes_ = np.asarray(est.classes_)
        te_groups = groups[te]
        fold_timings.append({
            "fold": int(f),
            "n_train_chunks": int(len(tr)),
            "n_test_chunks": int(len(te)),
            "n_test_works": int(np.unique(te_groups).size),
            "fit_seconds": fold_fit_seconds,
            "predict_seconds": fold_predict_seconds,
        })
        for g in np.unique(te_groups):
            m = te_groups == g
            prob_by_work[str(g)] = _align_proba(proba[m], classes_, n_authors)

    book_label = {w["work_id"]: w["label"] for w in manifest["works"]}
    rows = []
    for g in sorted(prob_by_work):                       # sorted → deterministic, manifest-aligned
        full = prob_by_work[g]
        true_label = book_label[g]
        order = np.argsort(-full, kind="stable")
        top1 = int(order[0])
        rank = int((full >= full[true_label]).sum())
        author_id, book_id = g.split("/", 1)
        rows.append({
            "test_author": author_id, "test_book": book_id,
            "true_label": int(true_label), "pred_label": top1,
            "pred_author": authors[top1], "correct": bool(top1 == true_label),
            "rank": rank, "fold": int(work_fold[g]), "confidence": float(full[true_label]),
            "top_candidates": [(authors[int(i)], float(full[int(i)])) for i in order[:top_k]],
            "_prob": full,
        })
    df = pd.DataFrame(rows)
    verify_result_against_panel(df, manifest)            # canonical-manifest check (primary + baselines)
    prob_matrix = np.vstack([r["_prob"] for r in rows])
    df = df.drop(columns=["_prob"])
    y_true = df["true_label"].to_numpy()
    log.info("GKF-panel[%s]: %d works, k=%d (%s)", spec, len(df), k, manifest["panel"])
    timing = {
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "total_seconds": float(clock() - total_started),
        "folds": fold_timings,
    }
    return df, prob_matrix, y_true, timing
