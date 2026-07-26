"""OOF-стекинг каналов внутри train-фолда LOBO.

Внутри fit: StratifiedGroupKFold(inner_folds) ПО КНИГАМ train-фолда даёт
out-of-fold decision-скоры каждого канала (chunk-level). На этих OOF:
  1) выбирается калибратор канала (identity/temperature/Platt/isotonic,
     held-out NLL — stylo.models.calibration.choose_calibrator);
  2) выбирается режим слияния (равновесный vs мета-LR стекинг) по book-level
     top-1 на OOF: для стекинга — вложенный 3-fold CV по книгам над OOF-матрицей.
Тестовая книга LOBO нигде не участвует: полный fit каналов и скоринг теста
происходят в predict_proba (векторизаторы каналов фолд-локальны).

Инвариант leak-free тот же, что у lobo.py: всё обучаемое живёт внутри train.
Это evaluation-only lazy-final-fit estimator: он удерживает raw train rows,
переобучает финальные каналы на каждом predict и намеренно не сериализуется.
Его internal mode-selection score отозван как unbiased evidence, поскольку
глобально откалиброванные OOF features повторно используют labels meta-fold.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from .calibration import choose_calibrator
from ..domain.work_weighting import (AblationConfig, CHUNK_WEIGHTED_LEGACY,
                                   FULL_WB_ABLATION, LEGACY_ABLATION, WORK_BALANCED,
                                   resolve_training_weighting, work_sample_weights)
from .channels import make_channels

log = logging.getLogger("stylo.models.stacked")


class StackClassCoverageError(ValueError):
    """The legacy stack cannot construct class-complete OOF or decision scores.

    ``report`` is attached so callers can persist the exact failed split
    inventory without parsing the human-readable message.
    """

    def __init__(self, message: str, *, stage: str, report: Optional[Dict] = None):
        super().__init__(message)
        self.stage = stage
        self.report = report


class EvaluationOnlyEstimatorError(RuntimeError):
    """A lazy evaluation estimator reached serialization or deployment."""


STACK_SELECTION_EVIDENCE_STATUS = (
    "withdrawn_biased_global_calibration_reuse"
)


def withdrawn_internal_selection_diagnostic(
    equal_accuracy: float,
    stacked_accuracy: Optional[float],
) -> Dict:
    """Package legacy mode-selection scores so they cannot pose as evidence."""

    values = (equal_accuracy, stacked_accuracy)
    if any(
        value is not None
        and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (float, np.floating))
            or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
        )
        for value in values
    ):
        raise ValueError("internal selection accuracies must be finite floats in [0,1]")
    return {
        "status": STACK_SELECTION_EVIDENCE_STATUS,
        "eligible_as_unbiased_evidence": False,
        "descriptive_only": {
            "equal": round(float(equal_accuracy), 4),
            "stacked": (
                round(float(stacked_accuracy), 4)
                if stacked_accuracy is not None
                else None
            ),
        },
    }


def _plain_scalar(value):
    """Return a JSON-friendly scalar without changing its equality semantics."""
    return value.item() if isinstance(value, np.generic) else value


def build_inner_split_preflight_report(
    splits, y: np.ndarray, expected_classes: np.ndarray, *,
    groups: Optional[np.ndarray] = None,
    expected_split_count: Optional[int] = None,
) -> Dict:
    """Pure class-coverage and structure inventory for every supplied split."""
    labels = np.asarray(y)
    expected = np.asarray(expected_classes)
    if labels.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got {labels.shape}")
    if expected.ndim != 1 or len(expected) == 0:
        raise ValueError(
            f"expected_classes must be a non-empty 1-D array, got {expected.shape}"
        )
    if len(np.unique(expected)) != len(expected):
        raise ValueError("expected_classes contains duplicates")
    if expected_split_count is not None and (
        isinstance(expected_split_count, bool)
        or not isinstance(expected_split_count, (int, np.integer))
        or int(expected_split_count) < 1
    ):
        raise ValueError("expected_split_count must be a positive integer")
    expected_split_count = (
        int(expected_split_count) if expected_split_count is not None else None
    )
    group_array = None
    if groups is not None:
        group_array = np.asarray(groups)
        if group_array.ndim != 1 or len(group_array) != len(labels):
            raise ValueError(
                f"groups must be one-dimensional with {len(labels)} rows"
            )

    split_reports = []
    validation_counts = np.zeros(len(labels), dtype=np.int64)
    all_indices = np.arange(len(labels), dtype=np.int64)
    for split_index, (train_index, validation_index) in enumerate(splits):
        train_index = np.asarray(train_index)
        validation_index = np.asarray(validation_index)
        if train_index.ndim != 1 or validation_index.ndim != 1:
            raise ValueError(f"split {split_index} indices must be one-dimensional")
        if not np.issubdtype(train_index.dtype, np.integer) \
                or not np.issubdtype(validation_index.dtype, np.integer):
            raise ValueError(f"split {split_index} indices must be integers")
        if ((train_index < 0).any() or (train_index >= len(labels)).any()
                or (validation_index < 0).any() or (validation_index >= len(labels)).any()):
            raise ValueError(f"split {split_index} contains an out-of-range row index")

        train_unique = np.unique(train_index)
        validation_unique = np.unique(validation_index)
        train_indices_unique = len(train_unique) == len(train_index)
        validation_indices_unique = len(validation_unique) == len(validation_index)
        overlap = np.intersect1d(train_unique, validation_unique, assume_unique=True)
        combined = np.union1d(train_unique, validation_unique)
        train_validation_disjoint = len(overlap) == 0
        exact_row_complement = (
            train_indices_unique
            and validation_indices_unique
            and train_validation_disjoint
            and np.array_equal(combined, all_indices)
        )
        np.add.at(validation_counts, validation_index, 1)

        overlapping_groups = []
        groups_disjoint = True
        if group_array is not None:
            train_groups = np.unique(group_array[train_unique])
            validation_groups = np.unique(group_array[validation_unique])
            overlapping_groups = [
                _plain_scalar(value) for value in np.intersect1d(
                    train_groups, validation_groups, assume_unique=True,
                )
            ]
            groups_disjoint = len(overlapping_groups) == 0

        observed = np.unique(labels[train_index])
        missing = np.setdiff1d(expected, observed, assume_unique=True)
        unexpected = np.setdiff1d(observed, expected, assume_unique=True)
        class_coverage_complete = len(missing) == 0 and len(unexpected) == 0
        split_structure_complete = exact_row_complement and groups_disjoint
        split_reports.append({
            "split_index": split_index,
            "train_row_count": int(len(train_index)),
            "validation_row_count": int(len(validation_index)),
            "observed_train_classes": [
                _plain_scalar(value) for value in observed
            ],
            "missing_train_classes": [
                _plain_scalar(value) for value in missing
            ],
            "unexpected_train_classes": [
                _plain_scalar(value) for value in unexpected
            ],
            "class_coverage_complete": class_coverage_complete,
            "train_indices_unique": train_indices_unique,
            "validation_indices_unique": validation_indices_unique,
            "train_validation_disjoint": train_validation_disjoint,
            "overlapping_row_indices": [int(value) for value in overlap],
            "exact_row_complement": exact_row_complement,
            "groups_disjoint": groups_disjoint,
            "overlapping_groups": overlapping_groups,
            "structure_complete": split_structure_complete,
            "complete": class_coverage_complete and split_structure_complete,
        })

    missing_validation = np.flatnonzero(validation_counts == 0)
    repeated_validation = np.flatnonzero(validation_counts > 1)
    validation_exactly_once = (
        len(missing_validation) == 0 and len(repeated_validation) == 0
    )
    split_count_complete = (
        expected_split_count is None
        or len(split_reports) == expected_split_count
    )
    class_coverage_complete = all(
        item["class_coverage_complete"] for item in split_reports
    )
    structure_complete = (
        bool(split_reports)
        and split_count_complete
        and validation_exactly_once
        and all(item["structure_complete"] for item in split_reports)
    )
    incomplete_count = sum(not item["complete"] for item in split_reports)
    return {
        "schema": "stylo.stack.inner_split_preflight.v1",
        "expected_classes": [_plain_scalar(value) for value in expected],
        "expected_split_count": expected_split_count,
        "split_count": len(split_reports),
        "incomplete_split_count": incomplete_count,
        "split_count_complete": split_count_complete,
        "validation_exactly_once": validation_exactly_once,
        "missing_validation_row_indices": [
            int(value) for value in missing_validation
        ],
        "repeated_validation_row_indices": [
            int(value) for value in repeated_validation
        ],
        "class_coverage_complete": class_coverage_complete,
        "structure_complete": structure_complete,
        "complete": class_coverage_complete and structure_complete,
        "splits": split_reports,
    }


def _book_level(probs: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        out[g].append(i)
    return {g: probs[idx].mean(axis=0) for g, idx in out.items()}


def _book_top1(probs: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    hits, total = 0, 0
    for g, mean_p in _book_level(probs, groups).items():
        yb = y[groups == g][0]
        hits += int(mean_p.argmax() == yb)
        total += 1
    return hits / max(1, total)


class StackedChannelClassifier:
    """Evaluation-only lazy stack for one held-out-work scoring call."""

    needs_groups = True
    evaluation_only = True
    lazy_final_fit = True
    deployment_supported = False
    serialization_supported = False
    internal_selection_evidence = False

    def __init__(self, cfg, inner_folds: int = 3, svc_c: float = 1.0,
                 meta_c: float = 1.0, seed: int = 42,
                 training_weighting: str = CHUNK_WEIGHTED_LEGACY,
                 *, ablation: Optional[AblationConfig] = None):
        self.cfg = cfg
        self.inner_folds = inner_folds
        self.svc_c = svc_c
        self.meta_c = meta_c
        self.seed = seed
        # The work-balanced stack decomposes into three independent axes — W (loss weights +
        # class_weight=None + group-aware calibration), F (channel vectorizers fit at work level via the
        # 3-arg ChannelFn call) and R (the FunctionWord channel's all-event relative transform). The two
        # PRODUCTION corners keep W==F==R: legacy A0 (all off, byte-for-byte the legacy stack) and
        # work_balanced A4 (all on). The audit-only intermediates come via an explicit ``ablation``:
        # A1 (W on, F/R off), A2 (F on, W/R off, A0 loss), A3 (R on, W/F off, A0 loss). ``ablation`` is
        # keyword-only and audit-only; production callers pass only ``training_weighting``.
        #
        # ``self._axes`` is the SINGLE canonical source of the ablation state; the public
        # ``training_weighting`` and ``ablation_`` are read-only views derived from it (properties,
        # no independently mutable copies), so provenance can never drift from the math.
        resolved = resolve_training_weighting(training_weighting)
        # Defuse a stateful ``str``-subclass that could pass the membership check as work_balanced yet
        # flip a later comparison to forge the public label (constructor-time split-brain): after
        # resolution the label must be an EXACT canonical str, never a polymorphic object.
        if type(resolved) is not str:
            raise TypeError(f"training_weighting must resolve to an exact str, got {type(resolved).__name__}")
        is_wb_label = resolved == WORK_BALANCED
        if ablation is None:
            axes = FULL_WB_ABLATION if is_wb_label else LEGACY_ABLATION
        else:
            if type(ablation) is not AblationConfig:
                raise TypeError(f"ablation must be exactly an AblationConfig, got {type(ablation).__name__}")
            for f in ("weights", "feature_fit", "relative_fw"):
                if type(getattr(ablation, f)) is not bool:
                    raise TypeError(f"ablation.{f} must be a plain bool")
            axes = AblationConfig(ablation.weights, ablation.feature_fit, ablation.relative_fw)
            # the ablation override wires the audit intermediates A1/A2/A3; the A0/A4 corners still go
            # through the enum (ablation=None), never a runnable-label override.
            if not (axes.is_weights_only_corner or axes.is_feature_state_only_corner
                    or axes.is_relative_fw_only_corner):
                raise ValueError(f"stack ablation override supports only audit cells A1/A2/A3, got {axes}")
            # split-brain guard: A1's feature side is legacy, so it must NOT carry the full-WB label —
            # pairing ablation=A1 with training_weighting=work_balanced would falsely claim full
            # work-balanced feature fitting in provenance while the math is W-on/F-off.
            if is_wb_label:
                raise ValueError(
                    "weights-only A1 has a legacy feature side; do not pair ablation=A1 with "
                    "training_weighting=work_balanced (split-brain provenance)")
        self._axes = axes                              # THE single canonical axis source
        self.passport_: Dict = {}

    def __getstate__(self):
        """Fail closed before retained raw training rows enter an artifact."""

        raise EvaluationOnlyEstimatorError(
            f"{type(self).__name__} is evaluation-only: fit retains raw training rows and "
            "predict_proba performs a lazy final fit; serialization/deployment "
            "is forbidden"
        )

    # -- read-only provenance (derived from the single ``_axes`` source) ------
    @property
    def ablation_(self) -> AblationConfig:
        """Authoritative axis provenance (W/F/R booleans); read-only, no independent copy."""
        return self._axes

    @property
    def training_weighting(self) -> str:
        """Public production label, DERIVED from the axes: the full-WB corner maps to the
        work_balanced enum, every other axis state (A0 and audit-only A1) to the legacy label."""
        return WORK_BALANCED if self._axes.is_full_wb_corner else CHUNK_WEIGHTED_LEGACY

    # -- внутренние помощники ------------------------------------------------
    @property
    def _weights_on(self) -> bool:
        """W axis: fold-local sample weights, class_weight=None, group-aware calibration."""
        return self._axes.weights

    @property
    def _feature_on(self) -> bool:
        """F axis: channel vectorizers fit at work level (3-arg ChannelFn call routing groups)."""
        return self._axes.feature_fit

    @property
    def _relative_fw_on(self) -> bool:
        """R axis: the FunctionWord channel uses the all-event relative transform (FW-only)."""
        return self._axes.relative_fw

    def _channels(self):
        """Build the channel set with the R policy. A0/A1/A4 keep the legacy corner coupling
        (R follows F, byte-exact goldens) via the EXACT legacy ``make_channels(cfg)`` call; A2/A3 — the
        only cells where R and F diverge — pass the explicit FW R policy to the sole FunctionWord
        channel. Keeping the A0/A1/A4 call kwarg-free preserves compatibility with any 2-arg-only
        ``make_channels`` (older callers / test doubles)."""
        relative_fw = None if self._feature_on == self._relative_fw_on else self._relative_fw_on
        if relative_fw is None:
            return make_channels(self.cfg)
        return make_channels(self.cfg, relative_fw=relative_fw)

    def _class_weight(self):
        return None if self._weights_on else "balanced"  # legacy=balanced, W-on=None (+ sample_weight)

    def _fold_weights(self, y_sub, groups_sub):
        """Fold-local per-chunk sample weights (sum == that fit's W_train); None when W is off."""
        return work_sample_weights(y_sub, groups_sub) if self._weights_on else None

    def _svc(self):
        return LinearSVC(C=self.svc_c, class_weight=self._class_weight(), max_iter=3000,
                         random_state=self.seed)

    def _decision_full(self, clf, X, n_classes: int, class_map: np.ndarray) -> np.ndarray:
        expected = np.asarray(self._classes_sorted)
        class_map = np.asarray(class_map)
        if n_classes != len(expected):
            raise ValueError(
                f"n_classes={n_classes} does not match fitted class universe {len(expected)}"
            )
        if class_map.ndim != 1:
            raise StackClassCoverageError(
                f"decision class map must be one-dimensional, got {class_map.shape}",
                stage="decision_function",
            )
        unique_map = np.unique(class_map)
        missing = np.setdiff1d(expected, unique_map, assume_unique=True)
        unexpected = np.setdiff1d(unique_map, expected, assume_unique=True)
        if len(unique_map) != len(class_map) or len(missing) or len(unexpected):
            report = {
                "expected_classes": [_plain_scalar(value) for value in expected],
                "observed_classes": [_plain_scalar(value) for value in class_map],
                "missing_classes": [_plain_scalar(value) for value in missing],
                "unexpected_classes": [
                    _plain_scalar(value) for value in unexpected
                ],
                "duplicate_class_count": int(len(class_map) - len(unique_map)),
            }
            raise StackClassCoverageError(
                "stylo_stack decision_function class coverage is incomplete; "
                "sentinel score filling is forbidden",
                stage="decision_function",
                report=report,
            )

        d = np.asarray(clf.decision_function(X))
        if (
            d.dtype == bool
            or np.issubdtype(d.dtype, np.complexfloating)
            or not np.issubdtype(d.dtype, np.number)
            or not np.isfinite(d).all()
        ):
            raise ValueError(
                "decision_function must return finite, real numeric scores"
            )
        if d.ndim == 1:  # бинарный случай
            d = np.column_stack([-d, d])
        if d.ndim != 2 or d.shape != (X.shape[0], len(class_map)):
            raise ValueError(
                "decision_function shape "
                f"{d.shape} != ({X.shape[0]}, {len(class_map)})"
            )
        full = np.empty((X.shape[0], n_classes), dtype=d.dtype)
        for j, c in enumerate(class_map):
            full[:, int(np.searchsorted(self._classes_sorted, c))] = d[:, j]
        return full

    # -- sklearn-подобный интерфейс -------------------------------------------
    def fit(self, texts, y, groups=None):
        if groups is None:
            raise ValueError("stylo_stack требует groups (book id каждого чанка)")
        texts = list(texts)
        y = np.asarray(y)
        groups = np.asarray(groups)
        self.classes_ = np.unique(y)
        self._classes_sorted = self.classes_
        n_cls = len(self.classes_)
        y_local = np.searchsorted(self.classes_, y)

        skf = StratifiedGroupKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        splits = list(skf.split(np.zeros(len(y)), y, groups))
        self.inner_split_preflight_ = build_inner_split_preflight_report(
            splits, y, self.classes_, groups=groups,
            expected_split_count=self.inner_folds,
        )
        if not self.inner_split_preflight_["complete"]:
            raise StackClassCoverageError(
                "stylo_stack is withdrawn for this fold: the inner OOF split "
                "preflight found incomplete class coverage or invalid train/"
                "validation structure; no sentinel or fallback scores are permitted",
                stage="inner_oof",
                report=self.inner_split_preflight_,
            )

        # Construct channels only after every inner split has passed the global
        # preflight, so a bad split cannot produce a partial channel fit.
        channels = self._channels()

        oof: Dict[str, np.ndarray] = {}
        for name, fn in channels.items():
            # NaN is a fail-closed construction marker, never a model score.
            scores = np.full((len(y), n_cls), np.nan, dtype=np.float64)
            for tr_i, te_i in splits:
                tr_t = [texts[i] for i in tr_i]
                te_t = [texts[i] for i in te_i]
                # F axis: strict 2-arg legacy call (any legacy channel) unless work-level feature
                # fitting is on, which routes the fold's work groups (A1 keeps this legacy).
                Xtr, Xte = fn(tr_t, te_t, groups[tr_i]) if self._feature_on else fn(tr_t, te_t)
                clf = self._svc().fit(Xtr, y[tr_i],
                                      sample_weight=self._fold_weights(y[tr_i], groups[tr_i]))
                scores[te_i] = self._decision_full(clf, Xte, n_cls, clf.classes_)
            invalid_rows = np.flatnonzero(~np.isfinite(scores).all(axis=1))
            if len(invalid_rows):
                raise StackClassCoverageError(
                    f"inner OOF channel {name!r} left rows unscored or non-finite",
                    stage="inner_oof_scores",
                    report={
                        "channel": name,
                        "invalid_row_indices": [
                            int(index) for index in invalid_rows
                        ],
                    },
                )
            oof[name] = scores

        # калибратор каждого канала — по held-out NLL внутри OOF
        self._calibrators = {}
        cal_passports = {}
        oof_probs = {}
        for name in channels:
            # Group-aware calibrator selection whenever the W axis is on (works held out whole;
            # disabled fail-closed if a class has < 2 works). W off passes groups=None (chunk-level,
            # unchanged). A1 keeps this group-aware because W is on even though the features are legacy.
            cal, passport = choose_calibrator(
                oof[name], y_local, seed=self.seed,
                groups=(groups if self._weights_on else None))
            self._calibrators[name] = cal
            cal_passports[name] = passport
            oof_probs[name] = cal(oof[name])

        # режим слияния по book-level top-1 на OOF
        eq_oof = np.mean(list(oof_probs.values()), axis=0)
        eq_acc = _book_top1(eq_oof, y_local, groups)
        Moof = np.hstack([oof_probs[n] for n in channels])
        # Calibration contract: when calibration is disabled (a class has < 2 works, so no held-out work
        # per class), the stack falls back to identity calibrators + an EQUAL-weight ensemble — no
        # meta-CV / meta-LR selection may run and mode_ cannot become "stacked".
        calibration_disabled = any(p.get("calibration_disabled") for p in cal_passports.values())
        self.meta_split_preflight_ = None
        if calibration_disabled:
            self.mode_ = "equal"
            self.meta_ = None
            stack_acc = None
        else:
            stack_hits, stack_total = 0, 0
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=self.seed)
            meta_splits = list(inner.split(np.zeros(len(y)), y, groups))
            self.meta_split_preflight_ = build_inner_split_preflight_report(
                meta_splits, y, self.classes_, groups=groups,
                expected_split_count=3,
            )
            if not self.meta_split_preflight_["complete"]:
                raise StackClassCoverageError(
                    "stylo_stack is withdrawn for this fold: the meta-CV split "
                    "preflight found incomplete class coverage or invalid train/"
                    "validation structure; fallback would change the registered "
                    "estimand",
                    stage="meta_cv",
                    report=self.meta_split_preflight_,
                )
            for tr_i, te_i in meta_splits:
                meta = LogisticRegression(C=self.meta_c, max_iter=2000,
                                          class_weight=self._class_weight(), random_state=self.seed)
                meta.fit(Moof[tr_i], y_local[tr_i], sample_weight=self._fold_weights(y[tr_i], groups[tr_i]))
                proba = np.zeros((len(te_i), n_cls))
                proba[:, meta.classes_] = meta.predict_proba(Moof[te_i])
                for g, mean_p in _book_level(proba, groups[te_i]).items():
                    yb = y_local[groups == g][0]
                    stack_hits += int(mean_p.argmax() == yb)
                    stack_total += 1
            stack_acc = stack_hits / max(1, stack_total)
            self.mode_ = "stacked" if stack_acc > eq_acc else "equal"
            self.meta_ = None
            if self.mode_ == "stacked":
                self.meta_ = LogisticRegression(
                    C=self.meta_c, max_iter=2000, class_weight=self._class_weight(),
                    random_state=self.seed).fit(Moof, y_local,
                                                sample_weight=self._fold_weights(y, groups))
        self._train_texts = texts
        self._train_y = y
        self._train_groups = groups
        self._channel_names = list(channels)
        self.selection_evidence_status_ = STACK_SELECTION_EVIDENCE_STATUS
        self.passport_ = {
            "mode": self.mode_,
            "inner_oof_book_top1": withdrawn_internal_selection_diagnostic(
                float(eq_acc),
                float(stack_acc) if stack_acc is not None else None,
            ),
            "calibration": cal_passports,
            "calibration_disabled": calibration_disabled,
        }
        return self

    def predict_proba(self, texts) -> np.ndarray:
        texts = list(texts)
        channels = self._channels()
        n_cls = len(self.classes_)
        te_probs = {}
        sw = self._fold_weights(self._train_y, self._train_groups)
        for name in self._channel_names:
            # F axis: strict 2-arg legacy call (any legacy channel) unless work-level feature fitting
            # is on, which routes the full-train work groups (A1 keeps this legacy).
            if self._feature_on:
                Xtr, Xte = channels[name](self._train_texts, texts, self._train_groups)
            else:
                Xtr, Xte = channels[name](self._train_texts, texts)
            clf = self._svc().fit(Xtr, self._train_y, sample_weight=sw)
            scores = self._decision_full(clf, Xte, n_cls, clf.classes_)
            te_probs[name] = self._calibrators[name](scores)
        if self.mode_ == "equal":
            return np.mean(list(te_probs.values()), axis=0)
        Mte = np.hstack([te_probs[n] for n in self._channel_names])
        proba = np.zeros((len(texts), n_cls))
        proba[:, self.meta_.classes_] = self.meta_.predict_proba(Mte)
        return proba
