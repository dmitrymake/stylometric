"""OOF-стекинг каналов внутри train-фолда LOBO.

Внутри fit: StratifiedGroupKFold(inner_folds) ПО КНИГАМ train-фолда даёт
out-of-fold decision-скоры каждого канала (chunk-level). На этих OOF:
  1) выбирается калибратор канала (identity/temperature/Platt/isotonic,
     held-out NLL — stylo.eval.calibration.choose_calibrator);
  2) выбирается режим слияния (равновесный vs мета-LR стекинг) по book-level
     top-1 на OOF: для стекинга — вложенный 3-fold CV по книгам над OOF-матрицей.
Тестовая книга LOBO нигде не участвует: полный fit каналов и скоринг теста
происходят в predict_proba (векторизаторы каналов фолд-локальны).

Инвариант leak-free тот же, что у lobo.py: всё обучаемое живёт внутри train.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from ..eval.calibration import choose_calibrator
from ..eval.work_weighting import (AblationConfig, CHUNK_WEIGHTED_LEGACY,
                                   FULL_WB_ABLATION, LEGACY_ABLATION, WORK_BALANCED,
                                   resolve_training_weighting, work_sample_weights)
from .channels import make_channels

log = logging.getLogger("stylo.models.stacked")


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
    """fit(texts, y, groups) / predict_proba(texts) / classes_ (локальные индексы)."""

    needs_groups = True

    def __init__(self, cfg, inner_folds: int = 3, svc_c: float = 1.0,
                 meta_c: float = 1.0, seed: int = 42,
                 training_weighting: str = CHUNK_WEIGHTED_LEGACY,
                 *, ablation: Optional[AblationConfig] = None):
        self.cfg = cfg
        self.inner_folds = inner_folds
        self.svc_c = svc_c
        self.meta_c = meta_c
        self.seed = seed
        # B2a/B3/B4: the stack decomposes into three independent axes — W (loss weights +
        # class_weight=None + group-aware calibration), F (channel vectorizers fit at work level via the
        # 3-arg ChannelFn call) and R (the FunctionWord channel's all-event relative transform). The two
        # PRODUCTION corners keep W==F==R: legacy A0 (all off, byte-for-byte the pre-B2a stack) and
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
        d = clf.decision_function(X)
        if d.ndim == 1:  # бинарный случай
            d = np.column_stack([-d, d])
        full = np.full((X.shape[0], n_classes), -30.0)
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

        channels = self._channels()
        skf = StratifiedGroupKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        splits = list(skf.split(np.zeros(len(y)), y, groups))

        oof: Dict[str, np.ndarray] = {}
        for name, fn in channels.items():
            scores = np.full((len(y), n_cls), -30.0)
            for tr_i, te_i in splits:
                tr_t = [texts[i] for i in tr_i]
                te_t = [texts[i] for i in te_i]
                # F axis: strict 2-arg legacy call (any pre-B2a channel) unless work-level feature
                # fitting is on, which routes the fold's work groups (A1 keeps this legacy).
                Xtr, Xte = fn(tr_t, te_t, groups[tr_i]) if self._feature_on else fn(tr_t, te_t)
                clf = self._svc().fit(Xtr, y[tr_i],
                                      sample_weight=self._fold_weights(y[tr_i], groups[tr_i]))
                scores[te_i] = self._decision_full(clf, Xte, n_cls, clf.classes_)
            oof[name] = scores

        # калибратор каждого канала — по held-out NLL внутри OOF
        self._calibrators = {}
        cal_passports = {}
        oof_probs = {}
        for name in channels:
            # B3: group-aware calibrator selection whenever the W axis is on (works held out whole;
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
        # B3 signed contract: when calibration is disabled (a class has < 2 works, so no held-out work
        # per class), the stack falls back to identity calibrators + an EQUAL-weight ensemble — no
        # meta-CV / meta-LR selection may run and mode_ cannot become "stacked".
        calibration_disabled = any(p.get("calibration_disabled") for p in cal_passports.values())
        if calibration_disabled:
            self.mode_ = "equal"
            self.meta_ = None
            stack_acc = None
        else:
            stack_hits, stack_total = 0, 0
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=self.seed)
            for tr_i, te_i in inner.split(np.zeros(len(y)), y, groups):
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
        self.passport_ = {
            "mode": self.mode_,
            "inner_oof_book_top1": {"equal": round(eq_acc, 4),
                                    "stacked": (round(stack_acc, 4) if stack_acc is not None else None)},
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
            # F axis: strict 2-arg legacy call (any pre-B2a channel) unless work-level feature fitting
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
