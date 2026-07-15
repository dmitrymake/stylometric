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
from ..models.lr import make_full_pipeline, make_logreg, make_scaler
from ..vectorizer import StyloVectorizer
from .dispatch import fit_estimator, frozen_run_contract
from .provenance import UnsupportedVariantError, verify_dataset_against_disk
from .work_weighting import (CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                             require_weighting, resolve_training_weighting)

# BLAS-потоки ограничиваем, чтобы не конфликтовать с joblib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

log = logging.getLogger("stylo.eval.lobo")


def make_factory_for_ablation(spec: str, cfg, *, ablation,
                              enabled_override: Optional[Dict[str, bool]] = None) -> Callable:
    """B4-B increment 1: the factory-routing entrypoint for the paired audit. An ``AblationConfig`` is
    mapped to the weighting enum (the two corners only; an intermediate raises
    AblationNotImplementedError) and the estimator is built by the unchanged ``make_factory`` — so the
    A0/A4 corners reached this way reproduce the frozen goldens (no estimator math changes).

    Fail-closed: ``ablation`` is keyword-only and must be **exactly** an ``AblationConfig`` (never a
    duck-typed / subclass object whose ``to_weighting`` could route A4 axes to a legacy estimator), its
    three axis fields are re-verified as plain bools, and the weighting is computed from a **freshly
    constructed** ``AblationConfig`` via the **class** method — so an instance whose ``to_weighting``
    was shadowed (``object.__setattr__``) cannot route A4 axes to a legacy model."""
    from .work_weighting import AblationConfig
    if type(ablation) is not AblationConfig:
        raise TypeError(f"ablation must be exactly an AblationConfig, got {type(ablation).__name__}")
    for f in ("weights", "feature_fit", "relative_fw"):
        if type(getattr(ablation, f)) is not bool:
            raise TypeError(f"ablation.{f} must be a plain bool")
    fresh = AblationConfig(ablation.weights, ablation.feature_fit, ablation.relative_fw)   # clean, no shadowed attr
    weighting = AblationConfig.to_weighting(fresh)              # the class method, never an instance override
    return make_factory(spec, cfg, enabled_override, weighting=weighting)


def make_factory(spec: str, cfg, enabled_override: Optional[Dict[str, bool]] = None,
                 *, weighting: str) -> Callable:
    """spec + resolved weighting -> фабрика свежего эстиматора (fit/predict_proba/classes_).

    The weighting enum (single toggle, resolved once upstream) is passed explicitly; the estimand
    is baked into the returned estimator. Legacy arm reproduces the pre-B2 estimators exactly.
    """
    weighting = require_weighting(weighting)   # strict: None is NOT a silent legacy fallback
    wb = weighting == WORK_BALANCED
    if spec == "stylo":
        if wb:
            from ..models.work_balanced import WorkBalancedStyloPipeline
            return lambda: WorkBalancedStyloPipeline([
                ("vectorizer", StyloVectorizer.from_config(cfg, enabled_override)),
                ("scaler", make_scaler(cfg)),
                ("classifier", make_logreg(cfg, class_weight=None)),
            ])
        return lambda: make_full_pipeline(cfg, StyloVectorizer.from_config(cfg, enabled_override))
    if spec.startswith("delta:"):
        n = int(spec.split(":", 1)[1])
        metric = cfg.get_path("delta.metric", "manhattan")
        return lambda: BurrowsDelta(n, metric, training_weighting=weighting)
    if spec.startswith("delta_cos:"):
        # Cosine Delta (Smith–Aldridge / Evert et al. 2017): та же MFW-z-механика,
        # угол вместо Manhattan — устойчив к разреженному хвосту MFW на коротких чанках
        n = int(spec.split(":", 1)[1])
        return lambda: BurrowsDelta(n, "cosine", training_weighting=weighting)
    if spec == "char_cos":
        return lambda: CharCosineBaseline(training_weighting=weighting)
    if spec == "bow_lr":
        if wb:
            from ..models.work_balanced import build_bow_lr_work_balanced
            return lambda: build_bow_lr_work_balanced()
        return lambda: build_bow_lr()
    if spec == "bow_lr_ref_legacy":
        # frozen historical reference row — WB-only (forbidden in the legacy arm, where it would
        # duplicate bow_lr and break byte-parity)
        if not wb:
            raise UnsupportedVariantError("bow_lr_ref_legacy is a work_balanced-only reference row")
        return lambda: build_bow_lr()
    if spec == "majority":
        return lambda: MajorityBaseline()
    if spec == "stylo_stack":
        # B3: work_balanced stack is wired end-to-end (feature+loss=B2a, group-aware calibration=B3).
        from ..models.stacked_clf import StackedChannelClassifier
        st = cfg.get_path("evaluation.stacking", {}) or {}
        get = st.get if hasattr(st, "get") else (lambda *_: None)
        return lambda: StackedChannelClassifier(
            cfg,
            inner_folds=get("inner_folds", 3) or 3,
            svc_c=get("svc_c", 1.0) or 1.0,
            meta_c=get("meta_c", 1.0) or 1.0,
            seed=cfg.get_path("seed", 42),
            training_weighting=weighting,
        )
    raise ValueError(f"Неизвестная модель: {spec}")


def _validate_proba(proba: np.ndarray, classes_: np.ndarray, n_authors: int, n_rows: int) -> None:
    """Fail-closed on a malformed probability output (NaN/negative/unnormalised rows, bad classes)."""
    import numbers
    proba = np.asarray(proba, dtype=np.float64)
    classes_ = np.asarray(classes_)
    if proba.ndim != 2 or proba.shape[0] != n_rows or proba.shape[1] != len(classes_):
        raise ValueError(f"predict_proba shape {proba.shape} != ({n_rows}, {len(classes_)})")
    if not np.isfinite(proba).all() or (proba < -1e-9).any():
        raise ValueError("predict_proba has NaN/inf/negative entries")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("predict_proba rows do not sum to 1")
    # classes_ must be a 1-D vector of EXACT non-bool integers, equal (ordered) to range(n_authors)
    if classes_.ndim != 1 or len(classes_) != n_authors:
        raise ValueError(f"classes_ must be 1-D of length n_authors={n_authors}, got {classes_.shape}")
    cl = classes_.tolist()
    for c in cl:
        if isinstance(c, bool) or not isinstance(c, numbers.Integral):
            raise ValueError(f"classes_ must be non-bool integers, got {c!r}")
    if [int(c) for c in cl] != list(range(n_authors)):
        raise ValueError(f"classes_ must equal range({n_authors}), got {cl}")


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
    # единый dispatch: groups маршрутизируются iff estimator их требует (LOBO/GKF/train — одинаково)
    fit_estimator(est, texts[mask_train], y_train, groups[mask_train])
    proba = np.asarray(est.predict_proba(texts[mask_test]))
    _validate_proba(proba, est.classes_, n_authors, int(mask_test.sum()))   # fail-closed
    full = _align_proba(proba, np.asarray(est.classes_), n_authors)

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
    *,
    weighting: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Прогнать LOBO для одной модели. Возвращает (df_книг, prob_matrix, y_true_книг).

    Публичный вход ВСЕГДА проверяет dataset против контракта, выведенного ТОЛЬКО из ``cfg``
    (никакого caller-supplied contract — иначе Dataset+rogue-контракт обошли бы cfg-корпус)."""
    weighting = require_weighting(weighting)
    weighting = verify_dataset_against_disk(cfg, dataset, weighting, frozen_run_contract(cfg))
    return _lobo_run(cfg, dataset, spec, enabled_override, max_books, n_jobs, weighting)


def _lobo_run(cfg, dataset, spec, enabled_override, max_books, n_jobs, weighting):
    """Internal LOBO worker — NO verification (caller must have verified). Not a public API."""
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)
    if n_jobs is None:
        n_jobs = cfg.get_path("evaluation.n_jobs", -1)

    books = sorted(set(dataset.groups.tolist()))
    if max_books and max_books > 0:
        books = books[:max_books]
    log.info("LOBO[%s/%s]: %d книг, n_jobs=%s", spec, weighting, len(books), n_jobs)

    factory = make_factory(spec, cfg, enabled_override, weighting=weighting)

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


def format_book_report(df: pd.DataFrame) -> str:
    """Отчёт по каждой книге с топ-N претендентов (как строка)."""
    lines = ["=== LOBO: топ-кандидаты по каждой книге (leakage-free) ==="]
    for r in df.sort_values(["test_author", "test_book"]).itertuples():
        mark = "OK  " if r.correct else "MISS"
        lines.append(
            f"[{mark}] {display_name(r.test_author)} / {r.test_book}  "
            f"(rank истинного автора: {r.rank})\n"
            f"        топ: {format_top_candidates(r.top_candidates)}"
        )
    return "\n".join(lines)


def write_book_report(df: pd.DataFrame, path) -> None:
    """Отчёт по каждой книге с топ-N претендентов."""
    import pathlib
    pathlib.Path(path).write_text(format_book_report(df), encoding="utf-8")
