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

import dataclasses
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..corpus import Dataset
from ..jsonio import canonical_hash
from ..lang import display_name
from ..models.baselines import CharCosineBaseline, MajorityBaseline, build_bow_lr
from ..models.delta import BurrowsDelta
from ..models.registry import (
    ModelRegistryError,
    assert_model_route,
    resolve_model_spec,
)
from ..features.reps import make_rep_cache
from ..models.lr import make_full_pipeline, make_logreg, make_scaler
from ..vectorizer import StyloVectorizer
from .dispatch import fit_estimator
from ..domain.prediction_contract import (
    stable_top1_and_worst_tie_rank,
    validate_probability_matrix,
)
from .provenance import (
    UnsupportedVariantError,
    prepare_scientific_evaluation,
    require_disk_verified_scientific_context,
)
from ..domain.work_weighting import (AblationNotImplementedError, CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                             require_weighting, resolve_training_weighting)

# BLAS-потоки ограничиваем, чтобы не конфликтовать с joblib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

log = logging.getLogger("stylo.eval.lobo")
MAX_GENERIC_LOBO_WORKERS = 8
GENERIC_LOBO_FOLD_MANIFEST_VERSION = "stylo.generic-lobo-fold-manifest.v2"


@dataclasses.dataclass(frozen=True)
class GenericLoboFold:
    work_id: str
    true_label: int
    true_author: str


@dataclasses.dataclass(frozen=True)
class GenericLoboFoldManifest:
    """Frozen generic LOBO test universe built before cache/factory/fit."""

    schema_version: str
    probability_class_order: tuple[str, ...]
    metric_label_order: tuple[int, ...]
    folds: tuple[GenericLoboFold, ...]
    self_hash: str

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "probability_class_order": list(self.probability_class_order),
            "metric_label_order": list(self.metric_label_order),
            "folds": [
                {
                    "work_id": fold.work_id,
                    "true_label": fold.true_label,
                    "true_author": fold.true_author,
                }
                for fold in self.folds
            ],
        }

    def validate(self) -> "GenericLoboFoldManifest":
        if type(self) is not GenericLoboFoldManifest:
            raise ValueError(
                "fold_manifest must be exactly GenericLoboFoldManifest"
            )
        rebuilt = _build_generic_lobo_fold_manifest_from_payload(
            self._payload()
        )
        if rebuilt != self:
            raise ValueError(
                "generic LOBO fold manifest is noncanonical or rehashed"
            )
        return self

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    @property
    def book_authors(self) -> tuple[str, ...]:
        return tuple(fold.true_author for fold in self.folds)


def _canonical_manifest_hash(payload: dict[str, object]) -> str:
    return canonical_hash(payload)


def _build_generic_lobo_fold_manifest_from_payload(
    payload: dict[str, object],
) -> GenericLoboFoldManifest:
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "probability_class_order",
        "metric_label_order",
        "folds",
    }:
        raise ValueError("generic LOBO fold manifest payload has wrong keys")
    if payload["schema_version"] != GENERIC_LOBO_FOLD_MANIFEST_VERSION:
        raise ValueError("generic LOBO fold manifest schema mismatch")
    raw_probability_order = payload["probability_class_order"]
    if (
        type(raw_probability_order) is not list
        or not raw_probability_order
        or any(
            type(author) is not str or not author
            for author in raw_probability_order
        )
        or len(set(raw_probability_order)) != len(raw_probability_order)
    ):
        raise ValueError("probability_class_order is malformed")
    raw_metric_order = payload["metric_label_order"]
    if (
        type(raw_metric_order) is not list
        or not raw_metric_order
        or any(type(label) is not int for label in raw_metric_order)
        or len(set(raw_metric_order)) != len(raw_metric_order)
    ):
        raise ValueError("metric_label_order is malformed")
    probability_labels = tuple(range(len(raw_probability_order)))
    if (
        any(label not in probability_labels for label in raw_metric_order)
        or [
            label
            for label in probability_labels
            if label in frozenset(raw_metric_order)
        ]
        != raw_metric_order
    ):
        raise ValueError(
            "metric_label_order must be a P-ordered subset"
        )
    raw_folds = payload["folds"]
    if type(raw_folds) is not list or not raw_folds:
        raise ValueError("generic LOBO folds must be a nonempty list")
    folds: list[GenericLoboFold] = []
    for raw_fold in raw_folds:
        if type(raw_fold) is not dict or set(raw_fold) != {
            "work_id",
            "true_label",
            "true_author",
        }:
            raise ValueError("generic LOBO fold has wrong keys")
        work_id = raw_fold["work_id"]
        true_label = raw_fold["true_label"]
        true_author = raw_fold["true_author"]
        if (
            type(work_id) is not str
            or not work_id
            or type(true_label) is not int
            or true_label not in probability_labels
            or type(true_author) is not str
            or true_author != raw_probability_order[true_label]
        ):
            raise ValueError("generic LOBO fold is malformed")
        folds.append(GenericLoboFold(work_id, true_label, true_author))
    if [fold.work_id for fold in folds] != sorted(
        fold.work_id for fold in folds
    ) or len({fold.work_id for fold in folds}) != len(folds):
        raise ValueError("generic LOBO folds must have unique sorted work ids")
    if (
        tuple(
            label
            for label in probability_labels
            if any(fold.true_label == label for fold in folds)
        )
        != tuple(raw_metric_order)
    ):
        raise ValueError(
            "metric_label_order does not match the frozen tested folds"
        )
    canonical_payload = {
        "schema_version": payload["schema_version"],
        "probability_class_order": list(raw_probability_order),
        "metric_label_order": list(raw_metric_order),
        "folds": [
            {
                "work_id": fold.work_id,
                "true_label": fold.true_label,
                "true_author": fold.true_author,
            }
            for fold in folds
        ],
    }
    return GenericLoboFoldManifest(
        schema_version=GENERIC_LOBO_FOLD_MANIFEST_VERSION,
        probability_class_order=tuple(raw_probability_order),
        metric_label_order=tuple(raw_metric_order),
        folds=tuple(folds),
        self_hash=_canonical_manifest_hash(canonical_payload),
    )


def build_generic_lobo_fold_manifest(
    dataset,
    *,
    max_books: int = 0,
) -> GenericLoboFoldManifest:
    """Freeze P, M and exact tested work order without observing predictions."""

    if type(max_books) is not int or max_books < 0:
        raise ValueError("max_books must be an exact non-negative integer")
    authors = tuple(dataset.authors)
    if (
        not authors
        or any(type(author) is not str or not author for author in authors)
        or len(set(authors)) != len(authors)
    ):
        raise ValueError("dataset authors are malformed")
    groups = np.asarray(dataset.groups, dtype=object)
    labels = np.asarray(dataset.y, dtype=object)
    if groups.ndim != 1 or labels.ndim != 1 or groups.shape != labels.shape:
        raise ValueError("dataset groups/y must be aligned one-dimensional arrays")
    if any(type(group) is not str or not group for group in groups.tolist()):
        raise ValueError("dataset groups must contain exact nonempty strings")
    if any(
        isinstance(label, (bool, np.bool_))
        or not isinstance(label, (int, np.integer))
        for label in labels.tolist()
    ):
        raise ValueError("dataset labels must contain non-bool integers")
    normalized_labels = np.asarray(
        [int(label) for label in labels.tolist()],
        dtype=np.int64,
    )
    work_to_label: dict[str, int] = {}
    for group, raw_label in zip(groups.tolist(), normalized_labels, strict=True):
        label = int(raw_label)
        if not 0 <= label < len(authors):
            raise ValueError("dataset label lies outside probability class order")
        existing = work_to_label.setdefault(group, label)
        if existing != label:
            raise ValueError(f"work {group!r} contains multiple truth labels")
        if group.split("/", 1)[0] != authors[label]:
            raise ValueError(
                f"work {group!r} does not match its registered author"
            )
    ordered_works = sorted(work_to_label)
    selected_works = (
        ordered_works[:max_books] if max_books > 0 else ordered_works
    )
    work_counts = {
        label: sum(
            int(work_label == label)
            for work_label in work_to_label.values()
        )
        for label in range(len(authors))
    }
    folds = [
        GenericLoboFold(
            work_id=work_id,
            true_label=work_to_label[work_id],
            true_author=authors[work_to_label[work_id]],
        )
        for work_id in selected_works
        if work_counts[work_to_label[work_id]] >= 2
    ]
    if not folds:
        raise ValueError("generic LOBO fold manifest has no feasible tested works")
    metric_order = [
        label
        for label in range(len(authors))
        if any(fold.true_label == label for fold in folds)
    ]
    return _build_generic_lobo_fold_manifest_from_payload(
        {
            "schema_version": GENERIC_LOBO_FOLD_MANIFEST_VERSION,
            "probability_class_order": list(authors),
            "metric_label_order": metric_order,
            "folds": [
                {
                    "work_id": fold.work_id,
                    "true_label": fold.true_label,
                    "true_author": fold.true_author,
                }
                for fold in folds
            ],
        }
    )


def validate_generic_lobo_result(
    frame: pd.DataFrame,
    manifest: GenericLoboFoldManifest,
) -> None:
    manifest = manifest.validate()
    expected = tuple(
        (fold.work_id, fold.true_label)
        for fold in manifest.folds
    )
    required_columns = {
        "test_author",
        "test_book",
        "true_label",
        "pred_label",
        "rank",
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError("generic LOBO result is missing required columns")
    observed = tuple(
        (
            f"{row.test_author}/{row.test_book}",
            int(row.true_label),
        )
        for row in frame.itertuples()
    )
    if observed != expected:
        raise ValueError(
            "generic LOBO result does not match its frozen fold manifest"
        )


def _bounded_lobo_workers(cfg, requested: Optional[int]) -> int:
    """Resolve joblib workers under an explicit process/memory amplification cap."""

    raw = (
        cfg.get_path("evaluation.n_jobs", MAX_GENERIC_LOBO_WORKERS)
        if requested is None
        else requested
    )
    if type(raw) is not int or raw == 0 or raw < -1:
        raise ValueError("LOBO n_jobs must be -1 or a positive exact integer")
    configured_cap = cfg.get_path(
        "evaluation.max_parallel_folds",
        MAX_GENERIC_LOBO_WORKERS,
    )
    if (
        type(configured_cap) is not int
        or configured_cap <= 0
        or configured_cap > MAX_GENERIC_LOBO_WORKERS
    ):
        raise ValueError(
            "evaluation.max_parallel_folds must be a positive exact integer "
            f"<= {MAX_GENERIC_LOBO_WORKERS}"
        )
    cpu_cap = max(1, int(os.cpu_count() or 1))
    cap = min(configured_cap, MAX_GENERIC_LOBO_WORKERS, cpu_cap)
    wanted = cap if raw == -1 else raw
    effective = min(wanted, cap)
    if wanted != effective:
        log.warning(
            "LOBO n_jobs=%s capped to %s (configured/absolute/cpu resource bound)",
            wanted,
            effective,
        )
    return effective


def make_factory_for_ablation(spec: str, cfg, *, ablation,
                              enabled_override: Optional[Dict[str, bool]] = None) -> Callable:
    """Factory-routing entrypoint for the paired W/F/R audit.

    Three runnable ablations:

    * **A0** (all axes off) and **A4** (all axes on) map to the two production weighting enums and are
      built by the unchanged ``make_factory`` — so the corners reached this way reproduce the frozen
      goldens byte-for-byte (no estimator math changes);
    * **A1** (weights-only == ``(T,F,F)``) is an audit-only path: the work-balanced loss/calibration
      protocol on a strictly legacy (A0) feature side, wired ONLY for the LR-family
      (``stylo``/``bow_lr``/``stylo_stack``) via axis-aware estimators. A1 has NO production weighting
      enum (``to_weighting`` stays corner-only), so it is never collapsed to ``WORK_BALANCED``.

    A1 for a model where W is not a new loss axis fails closed with ``AblationNotApplicableError`` and an
    exact ``reason`` (Delta → ``already_in_legacy``; char_cos/majority/other → ``not_applicable``); the
    five remaining intermediates raise ``AblationNotImplementedError``.

    Fail-closed: ``ablation`` is keyword-only and must be **exactly** an ``AblationConfig`` (never a
    duck-typed / subclass object whose ``to_weighting`` could route A4 axes to a legacy estimator), its
    three axis fields are re-verified as plain bools, and every downstream decision is taken from a
    **freshly constructed** ``AblationConfig`` via **class** methods — so an instance whose axis
    properties or ``to_weighting`` were shadowed (``object.__setattr__``) cannot mis-route the axes."""
    from ..domain.work_weighting import AblationConfig
    if type(ablation) is not AblationConfig:
        raise TypeError(f"ablation must be exactly an AblationConfig, got {type(ablation).__name__}")
    for f in ("weights", "feature_fit", "relative_fw"):
        if type(getattr(ablation, f)) is not bool:
            raise TypeError(f"ablation.{f} must be a plain bool")
    fresh = AblationConfig(ablation.weights, ablation.feature_fit, ablation.relative_fw)   # clean, no shadowed attr
    if AblationConfig.is_legacy_corner.fget(fresh) or AblationConfig.is_full_wb_corner.fget(fresh):
        weighting = AblationConfig.to_weighting(fresh)         # class method, never an instance override
        return make_factory(spec, cfg, enabled_override, weighting=weighting)
    for cell in ("is_weights_only_corner", "is_feature_state_only_corner", "is_relative_fw_only_corner"):
        if getattr(AblationConfig, cell).fget(fresh):
            return _make_audit_factory(spec, cfg, enabled_override, fresh)   # A1 / A2 / A3
    # the remaining intermediates (WFR 110/101/011) have no estimator wiring yet
    raise AblationNotImplementedError(
        f"ablation {fresh} has no estimator wiring yet (runnable: A0/A4 corners and audit A1/A2/A3)")


def _delta_n_or_raise(spec: str) -> int:
    """A well-formed positive int after the colon; a malformed/bare delta spec is a plain ValueError
    (never laundered into a clean applicability status)."""
    n = spec.split(":", 1)[1]
    if not (n.isdigit() and int(n) > 0):
        raise ValueError(f"invalid delta spec {spec!r}")
    return int(n)


def _make_audit_factory(spec: str, cfg, enabled_override: Optional[Dict[str, bool]],
                        ablation) -> Callable:
    """Build the audit-only A1 (weights) / A2 (feature-state) / A3 (relative-FW) estimator, or fail
    closed with the exact typed applicability signal recorded by the evaluation runner.

    Loss/calibration: A1 is work-balanced (W on); A2/A3 keep the exact A0 loss (W off). Feature side:
    A1 is exact legacy A0; A2 routes work-level feature fitting (F); A3 applies the pooled relative-FW
    transform (R). Non-applicable cells raise ``AblationNotApplicableError`` (exact reason + requested)
    or, for char_cos A2 which duplicates A4, ``AblationEquivalentError('A4')``."""
    from ..domain.work_weighting import (AblationEquivalentError, AblationNotApplicableError,
                                 CHUNK_WEIGHTED_LEGACY)
    a1 = ablation.is_weights_only_corner
    a2 = ablation.is_feature_state_only_corner
    a3 = ablation.is_relative_fw_only_corner

    if spec == "stylo":
        from ..models.work_balanced import (FeatureStateStyloPipeline, RelativeFwStyloPipeline,
                                            WeightsOnlyStyloPipeline)
        if a1:
            return lambda: WeightsOnlyStyloPipeline([
                ("vectorizer", StyloVectorizer.from_config(cfg, enabled_override)),   # legacy A0 vec
                ("scaler", make_scaler(cfg)),
                ("classifier", make_logreg(cfg, class_weight=None))])                 # W on
        if a2:
            return lambda: FeatureStateStyloPipeline([
                ("vectorizer", StyloVectorizer.from_config(cfg, enabled_override, relative_fw=False)),
                ("scaler", make_scaler(cfg)),
                ("classifier", make_logreg(cfg))])                                   # A0 loss (canonical)
        return lambda: RelativeFwStyloPipeline([                                      # a3
            ("vectorizer", StyloVectorizer.from_config(cfg, enabled_override, relative_fw=True)),
            ("scaler", make_scaler(cfg)),
            ("classifier", make_logreg(cfg))])                                       # A0 loss (canonical)
    if spec == "bow_lr":
        from ..models.work_balanced import build_bow_lr_feature_state, build_bow_lr_weights_only
        if a1:
            return lambda: build_bow_lr_weights_only()
        if a2:
            return lambda: build_bow_lr_feature_state()
        raise AblationNotApplicableError(spec, "not_applicable", ablation)            # a3: BoW has no R
    if spec == "stylo_stack":
        from ..models.stacked_clf import StackedChannelClassifier
        return lambda: StackedChannelClassifier(cfg, **_stacked_kwargs(cfg),
                                                training_weighting=CHUNK_WEIGHTED_LEGACY, ablation=ablation)
    if spec.startswith("delta:") or spec.startswith("delta_cos:"):
        n = _delta_n_or_raise(spec)                                                   # plain ValueError if bad
        if a1:
            raise AblationNotApplicableError(spec, "already_in_legacy", ablation)     # W == equal-work already
        metric = "cosine" if spec.startswith("delta_cos:") else cfg.get_path("delta.metric", "manhattan")
        from ..models.delta import BurrowsDelta
        return lambda: BurrowsDelta(n, metric, ablation=ablation)                     # A2/A3 F×R grid
    if spec == "char_cos":
        if a2:
            raise AblationEquivalentError(spec, ablation, "A4")   # char A2 ≡ A4 (W legacy, no R, only F)
        raise AblationNotApplicableError(spec, "not_applicable", ablation)            # A1/A3
    if spec in ("majority", "bow_lr_ref_legacy"):
        raise AblationNotApplicableError(spec, "not_applicable", ablation)
    raise ValueError(f"Неизвестная модель: {spec}")


def _stacked_kwargs(cfg) -> Dict:
    """Shared stack hyper-params (identical for A0/A4/A1 — only the axis wiring differs)."""
    st = cfg.get_path("evaluation.stacking", {}) or {}
    get = st.get if hasattr(st, "get") else (lambda *_: None)
    return dict(
        inner_folds=get("inner_folds", 3) or 3,
        svc_c=get("svc_c", 1.0) or 1.0,
        meta_c=get("meta_c", 1.0) or 1.0,
        seed=cfg.get_path("seed", 42),
    )


def _equal_channel_kwargs(cfg) -> Dict:
    """Only hyperparameters that affect the fixed equal-channel estimator."""
    st = cfg.get_path("evaluation.stacking", {}) or {}
    get = st.get if hasattr(st, "get") else (lambda *_: None)
    return {
        "svc_c": get("svc_c", 1.0) or 1.0,
        "seed": cfg.get_path("seed", 42),
    }


def make_factory(spec: str, cfg, enabled_override: Optional[Dict[str, bool]] = None,
                 *, weighting: str) -> Callable:
    """spec + resolved weighting -> фабрика свежего эстиматора (fit/predict_proba/classes_).

    The weighting enum (single toggle, resolved once upstream) is passed explicitly; the estimand
    is baked into the returned estimator. The legacy arm reproduces the frozen estimators exactly.
    """
    weighting = require_weighting(weighting)   # strict: None is NOT a silent legacy fallback
    wb = weighting == WORK_BALANCED
    registration = resolve_model_spec(spec)
    try:
        assert_model_route(spec, weighting=weighting)
    except ModelRegistryError as exc:
        raise UnsupportedVariantError(str(exc)) from exc
    if registration.key == "stylo":
        if wb:
            from ..models.work_balanced import WorkBalancedStyloPipeline
            return lambda: WorkBalancedStyloPipeline([
                ("vectorizer", StyloVectorizer.from_config(cfg, enabled_override)),
                ("scaler", make_scaler(cfg)),
                ("classifier", make_logreg(cfg, class_weight=None)),
            ])
        return lambda: make_full_pipeline(cfg, StyloVectorizer.from_config(cfg, enabled_override))
    if registration.key == "delta":
        n = int(spec.split(":", 1)[1])
        metric = cfg.get_path("delta.metric", "manhattan")
        return lambda: BurrowsDelta(n, metric, training_weighting=weighting)
    if registration.key == "delta_cos":
        # Cosine Delta (Smith–Aldridge / Evert et al. 2017): та же MFW-z-механика,
        # угол вместо Manhattan — устойчив к разреженному хвосту MFW на коротких чанках
        n = int(spec.split(":", 1)[1])
        return lambda: BurrowsDelta(n, "cosine", training_weighting=weighting)
    if registration.key == "char_cos":
        return lambda: CharCosineBaseline(training_weighting=weighting)
    if registration.key == "bow_lr":
        if wb:
            from ..models.work_balanced import build_bow_lr_work_balanced
            return lambda: build_bow_lr_work_balanced()
        return lambda: build_bow_lr()
    if registration.key == "bow_lr_ref_legacy":
        # frozen historical reference row — WB-only (forbidden in the legacy arm, where it would
        # duplicate bow_lr and break byte-parity)
        if not wb:
            raise UnsupportedVariantError("bow_lr_ref_legacy is a work_balanced-only reference row")
        return lambda: build_bow_lr()
    if registration.key == "majority":
        return lambda: MajorityBaseline()
    if registration.key == "stylo_stack":
        # The work-balanced stack wires feature fitting, loss weighting, and group-aware calibration.
        from ..models.stacked_clf import StackedChannelClassifier
        return lambda: StackedChannelClassifier(cfg, **_stacked_kwargs(cfg), training_weighting=weighting)
    if registration.key == "stylo_equal_channels_v1":
        from ..models.equal_channel_ensemble import EqualChannelEnsembleClassifier
        return lambda: EqualChannelEnsembleClassifier(
            cfg, **_equal_channel_kwargs(cfg), training_weighting=weighting,
        )
    raise AssertionError(f"registry entry has no factory adapter: {registration.key}")


def _validate_proba(proba: np.ndarray, classes_: np.ndarray, n_authors: int, n_rows: int) -> None:
    """Compatibility wrapper around the one versioned prediction contract."""
    validate_probability_matrix(
        proba, classes_, n_classes=n_authors, n_rows=n_rows
    )


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

    decision = stable_top1_and_worst_tie_rank(
        full, true_label=true_label, expected_width=n_authors
    )
    order = decision.order
    top1 = decision.top1
    rank = decision.true_rank
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
    context = prepare_scientific_evaluation(
        cfg,
        dataset,
        weighting,
    )
    return _lobo_run(
        cfg,
        context,
        spec,
        enabled_override,
        max_books,
        n_jobs,
    )


def _lobo_run(
    cfg,
    context,
    spec,
    enabled_override,
    max_books,
    n_jobs,
    *,
    fold_manifest: GenericLoboFoldManifest | None = None,
):
    """Internal LOBO worker.

    A bare Dataset is rejected: callers must pass the sealed result of
    :func:`prepare_scientific_evaluation`. This is not a public API.
    """
    dataset = require_disk_verified_scientific_context(context)
    weighting = dataset.weighting
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)
    n_jobs = _bounded_lobo_workers(cfg, n_jobs)

    live_manifest = build_generic_lobo_fold_manifest(
        dataset,
        max_books=max_books,
    )
    if fold_manifest is not None:
        fold_manifest = fold_manifest.validate()
        if fold_manifest != live_manifest:
            raise ValueError(
                "supplied generic LOBO fold manifest does not match "
                "the sealed dataset"
            )
    else:
        fold_manifest = live_manifest
    books = [fold.work_id for fold in fold_manifest.folds]
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

    # verbose=10 → joblib печатает прогресс «Done N out of M» (иначе после старта LOBO — тишина
    # до конца; многократные channel/meta fits в экспериментальном stack могут быть долгими).
    # Rep caches can be hundreds of MiB.  Never queue twice the active fold
    # count: bounded pre-dispatch avoids needless serialization/RSS pressure.
    res = Parallel(n_jobs=n_jobs, pre_dispatch=n_jobs, verbose=10)(
        delayed(run_fold)(dataset.texts, dataset.y, dataset.groups, dataset.n_authors,
                          dataset.authors, g, factory, top_k)
        for g in books
    )
    if any(row is None for row in res):
        raise RuntimeError(
            f"LOBO[{spec}] produced a runtime skip outside the frozen manifest"
        )
    rows = list(res)

    prob_matrix = np.vstack([r.pop("_prob") for r in rows])
    df = pd.DataFrame(rows)
    validate_generic_lobo_result(df, fold_manifest)
    df.attrs["generic_lobo_fold_manifest"] = fold_manifest.as_dict()
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
