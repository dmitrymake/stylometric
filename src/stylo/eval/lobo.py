"""Leakage-free LOBO (Leave-One-Book-Out) — ЕДИНСТВЕННЫЙ честный движок оценки.

Инвариант отсутствия утечки: всё, что обучается (векторизатор, IDF, MFW-словарь,
z-статистики, классификатор), обучается ТОЛЬКО на train-фолде (все книги, кроме одной
тестовой). Тестовая книга не видна на этапе fit.

Один движок обслуживает и продакшен-модель ('stylo'), и baseline-ы (delta/char_cos/
bow_lr/majority) — все дают predict_proba(texts) и .classes_, выровненные на общий
набор авторов. Голосование по книге — усреднение вероятностей чанков (soft voting).

Скорость: spaCy-доки берутся из прогретого DocCache (в фолдах только чтение, без
повторного разбора). Фолды считаются параллельно (joblib).
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..corpus import Dataset
from ..lang import display_name
from ..models.baselines import CharCosineBaseline, MajorityBaseline, build_bow_lr
from ..models.delta import BurrowsDelta
from ..features.reps import make_rep_cache
from ..models.lr import make_full_pipeline
from ..vectorizer import StyloVectorizer

# BLAS-потоки ограничиваем, чтобы не конфликтовать с joblib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

log = logging.getLogger("stylo.eval.lobo")


def make_factory(spec: str, cfg, enabled_override: Optional[Dict[str, bool]] = None) -> Callable:
    """spec -> фабрика свежего эстиматора (fit/predict_proba/classes_)."""
    if spec == "stylo":
        return lambda: make_full_pipeline(cfg, StyloVectorizer.from_config(cfg, enabled_override))
    if spec.startswith("delta:"):
        n = int(spec.split(":", 1)[1])
        metric = cfg.get_path("delta.metric", "manhattan")
        return lambda: BurrowsDelta(n, metric)
    if spec.startswith("delta_cos:"):
        # Cosine Delta (Smith–Aldridge / Evert et al. 2017): та же MFW-z-механика,
        # угол вместо Manhattan — устойчив к разреженному хвосту MFW на коротких чанках
        n = int(spec.split(":", 1)[1])
        return lambda: BurrowsDelta(n, "cosine")
    if spec == "char_cos":
        return lambda: CharCosineBaseline()
    if spec == "bow_lr":
        return lambda: build_bow_lr()
    if spec == "majority":
        return lambda: MajorityBaseline()
    if spec == "stylo_stack":
        from ..models.stacked_clf import StackedChannelClassifier
        st = cfg.get_path("evaluation.stacking", {}) or {}
        get = st.get if hasattr(st, "get") else (lambda *_: None)
        return lambda: StackedChannelClassifier(
            cfg,
            inner_folds=get("inner_folds", 3) or 3,
            svc_c=get("svc_c", 1.0) or 1.0,
            meta_c=get("meta_c", 1.0) or 1.0,
            seed=cfg.get_path("seed", 42),
        )
    raise ValueError(f"Неизвестная модель: {spec}")


def _align_proba(proba: np.ndarray, classes_: np.ndarray, n_authors: int) -> np.ndarray:
    """Усреднить вероятности чанков и выровнять на полный набор авторов."""
    mean = proba.mean(axis=0)
    full = np.zeros(n_authors, dtype=np.float64)
    for j, c in enumerate(classes_):
        full[int(c)] = mean[j]
    return full


def run_fold(
    texts: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_authors: int,
    authors: List[str],
    test_group: str,
    factory: Callable,
    top_k: int,
) -> Optional[Dict]:
    mask_test = groups == test_group
    if not mask_test.any():
        return None
    mask_train = ~mask_test
    y_train = y[mask_train]
    true_label = int(y[mask_test][0])
    if true_label not in set(y_train.tolist()):
        # у автора единственная книга — LOBO невозможен
        return None

    est = factory()
    if getattr(est, "needs_groups", False):
        # стекинг: inner-CV по книгам train-фолда (leak-free: тестовой книги нет)
        est.fit(texts[mask_train], y_train, groups=groups[mask_train])
    else:
        est.fit(texts[mask_train], y_train)
    proba = est.predict_proba(texts[mask_test])
    full = _align_proba(np.asarray(proba), np.asarray(est.classes_), n_authors)

    order = np.argsort(-full, kind="stable")  # равные → меньший индекс (как argmax/predict), без смещения к старшему
    top1 = int(order[0])
    rank = int((full >= full[true_label]).sum())  # tie-aware худший ранг: классы с prob >= p_true (включая сам класс); при вырожденных скорах (majority: нули) ничья не даёт ложного «2-го места»
    top_candidates = [(authors[int(i)], float(full[int(i)])) for i in order[:top_k]]

    author_id, book_id = test_group.split("/", 1)
    return {
        "test_author": author_id,
        "test_book": book_id,
        "true_label": true_label,
        "pred_label": top1,
        "pred_author": authors[top1],
        "correct": bool(top1 == true_label),
        "rank": rank,
        "confidence": float(full[true_label]),
        "n_chunks": int(mask_test.sum()),
        "top_candidates": top_candidates,
        "_prob": full,
    }


def lobo_evaluate(
    cfg,
    dataset: Dataset,
    spec: str = "stylo",
    enabled_override: Optional[Dict[str, bool]] = None,
    max_books: int = 0,
    n_jobs: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Прогнать LOBO для одной модели. Возвращает (df_книг, prob_matrix, y_true_книг)."""
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)
    if n_jobs is None:
        n_jobs = cfg.get_path("evaluation.n_jobs", -1)

    books = sorted(set(dataset.groups.tolist()))
    if max_books and max_books > 0:
        books = books[:max_books]
    log.info("LOBO[%s]: %d книг, n_jobs=%s", spec, len(books), n_jobs)

    factory = make_factory(spec, cfg, enabled_override)

    # Прогреть rep-кэш ОДИН раз в родителе перед параллельными фолдами: фолд-НЕЗАВИСИМые
    # представления (bleach/pos/punct/dep/morph/syntax/длины) строятся один раз и пишутся в
    # единый файл; воркеры их только читают, spaCy в фолдах не вызывается. Без этого прогрева
    # холодный кэш заставляет КАЖДЫЙ воркер строить представления заново на каждом фолде —
    # это главная причина многочасовых прогонов. Leak-free сохранён: Rep не зависит от меток.
    # Идемпотентно (при полном кэше — быстрый no-op).
    try:
        make_rep_cache(cfg).warm(dataset.texts, n_process=cfg.get_path("language.parse_n_process", 4))
    except Exception as exc:  # pragma: no cover — на отсутствии spaCy падать в per-fold путь
        log.warning("rep-кэш не прогрет (%s) — фолды построят представления на лету", exc)

    # verbose=10 → joblib печатает прогресс «Done N out of M» (иначе после старта LOBO — тишина до конца;
    # с CalibratedClassifierCV(cv=3) внутри stylo один фолд ~30 CPU-мин, прогон видеть НАДО).
    res = Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs", verbose=10)(
        delayed(run_fold)(dataset.texts, dataset.y, dataset.groups, dataset.n_authors,
                          dataset.authors, g, factory, top_k)
        for g in books
    )
    rows = [r for r in res if r is not None]
    if not rows:
        raise RuntimeError(f"LOBO[{spec}] не дал результатов (мало книг на автора?)")

    prob_matrix = np.vstack([r.pop("_prob") for r in rows])
    df = pd.DataFrame(rows)
    y_true = df["true_label"].to_numpy()
    return df, prob_matrix, y_true


def format_top_candidates(top_candidates: List[Tuple[str, float]]) -> str:
    return ", ".join(f"{display_name(a)} ({s:.3f})" for a, s in top_candidates)


def write_book_report(df: pd.DataFrame, path) -> None:
    """Отчёт по каждой книге с топ-N претендентов."""
    import pathlib
    lines = ["=== LOBO: топ-кандидаты по каждой книге (leakage-free) ==="]
    for r in df.sort_values(["test_author", "test_book"]).itertuples():
        mark = "OK  " if r.correct else "MISS"
        lines.append(
            f"[{mark}] {display_name(r.test_author)} / {r.test_book}  "
            f"(rank истинного автора: {r.rank})\n"
            f"        топ: {format_top_candidates(r.top_candidates)}"
        )
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")
