"""Быстрый прокси-оценщик: StratifiedGroupKFold по книгам (k фолдов; полный LOBO — ~100).

Используется в sweep для СКРИНИНГА блоков (минуты; полный LOBO — часы). Книги не дробятся
между train/test (groups=book) — утечки book-level нет. Это лишь ранжирующий прокси:
финальные цифры в отчёт идут ТОЛЬКО из полного LOBO (eval/lobo.py).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..corpus import Dataset
from .lobo import _align_proba, make_factory

log = logging.getLogger("stylo.eval.groupkfold")


def gkf_evaluate(
    cfg,
    dataset: Dataset,
    spec: str = "stylo",
    enabled_override: Optional[Dict[str, bool]] = None,
    k: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """k-fold прокси. Возвращает (df книг, prob_matrix, y_true) как у lobo_evaluate."""
    if k is None:
        k = cfg.get_path("evaluation.groupkfold_k", 5)
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)

    texts, y, groups = dataset.texts, dataset.y, dataset.groups
    n_authors, authors = dataset.n_authors, dataset.authors
    factory = make_factory(spec, cfg, enabled_override)

    # число фолдов не больше, чем мин. число книг у автора (иначе StratifiedGroupKFold падает)
    book_author = dataset.book_to_author()
    per_author_books: Dict[int, int] = {}
    for lbl in book_author.values():
        per_author_books[lbl] = per_author_books.get(lbl, 0) + 1
    k_eff = max(2, min(k, min(per_author_books.values())))

    sgkf = StratifiedGroupKFold(n_splits=k_eff, shuffle=True,
                                random_state=cfg.get_path("seed", 42))
    rows = []
    prob_by_book: Dict[str, np.ndarray] = {}

    for tr, te in sgkf.split(texts, y, groups):
        est = factory()
        est.fit(texts[tr], y[tr])
        proba = np.asarray(est.predict_proba(texts[te]))
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
