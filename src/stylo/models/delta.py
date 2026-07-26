"""Frozen legacy selected-mass Delta compatibility estimator.

The public ``delta:N`` identifier is immutable because historical artifacts and
the A0 protocol bind it.  It is *not* branded as canonical Burrows's Delta:
  1. N самых частотных слов (MFW) корпуса;
  2. relative frequency divides by the selected-MFW mass, not all tokens;
  3. z-нормировка по статистикам TRAIN (Burrows z-scores);
  4. профиль автора = средний z по его текстам;
  5. Delta(text, author) = средняя |z_text - z_author| (Manhattan) — меньше = ближе.
     (cosine — вариант Smith–Aldridge.)

Принимает сырые тексты (сам строит MFW-словарь на train). Leakage-free: vocab и
статистики z берутся ТОЛЬКО из train.
"""
from __future__ import annotations

import numbers
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances

from ..domain.work_weighting import (AblationConfig, CHUNK_WEIGHTED_LEGACY,
                                   SUPPORTED_TRAINING_WEIGHTINGS, WORK_BALANCED,
                                   resolve_training_weighting)
from ..features.work_vectorizer import (MODE_COUNT, MODE_RELATIVE, WorkLevelVectorizer, _group_indicator,
                                        analyzer_event_counts, relative_by_events, validate_work_ids)

_TOKEN_PATTERN = r"(?u)\b\w+\b"
_ANALYZER = {"analyzer": "word", "token_pattern": _TOKEN_PATTERN, "lowercase": True}


class BurrowsDelta:
    """Compatibility class for the frozen ``delta:N`` selected-mass family."""

    needs_groups = True
    PUBLIC_DISPLAY_NAME = "Frozen legacy selected-mass Delta"
    FREQUENCY_DENOMINATOR = "sum_selected_mfw_counts"
    # v3 adds the audit-only ``_ablation`` for the A2/A3 F×R grid; v2 added
    # training_weighting/_wv; a versionless pickle is v1. A0/A4 external state is byte-identical.
    ARTIFACT_SCHEMA_VERSION = 3

    def __init__(self, mfw_count: int = 300, metric: str = "manhattan",
                 vocabulary: List[str] | None = None,
                 training_weighting: str = CHUNK_WEIGHTED_LEGACY,
                 *, ablation: "AblationConfig | None" = None):
        assert metric in {"manhattan", "cosine"}
        self._schema_version = self.ARTIFACT_SCHEMA_VERSION
        self.mfw_count = mfw_count
        self.metric = metric
        # если задан фиксированный словарь (напр. служебные слова) — Delta считается ТОЛЬКО по нему
        # (тема-нейтральный вариант Delta); иначе — классические top-mfw_count слов корпуса.
        self.vocabulary = list(vocabulary) if vocabulary is not None else None
        resolved = resolve_training_weighting(training_weighting)
        # canonical exact str: a stateful str-subclass could pass the membership check as work_balanced
        # yet flip a later comparison to forge the label (constructor-time split-brain).
        if type(resolved) is not str:
            raise TypeError(f"training_weighting must resolve to an exact str, got {type(resolved).__name__}")
        self.training_weighting = WORK_BALANCED if resolved == WORK_BALANCED else CHUNK_WEIGHTED_LEGACY
        # W is already-in-legacy for Delta (equal-work centroids), so the only axes are F and R.
        # ``ablation=None`` -> the two production corners via training_weighting (A0 legacy F0R0, A4
        # work_balanced F1R1). An explicit A2/A3 AblationConfig selects the audit off-diagonal cells.
        self._ablation = self._validate_ablation(ablation)
        self._vec: CountVectorizer | None = None
        self._wv = None                      # WorkLevelVectorizer в work_balanced/F1-ветке
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self._validate_state(check_fitted=False)   # label/ablation coherence at construction

    @staticmethod
    def _validate_ablation(ablation):
        if ablation is None:
            return None
        if type(ablation) is not AblationConfig:
            raise TypeError(f"ablation must be an AblationConfig or None, got {type(ablation).__name__}")
        fresh = AblationConfig(ablation.weights, ablation.feature_fit, ablation.relative_fw)
        if not (fresh.is_feature_state_only_corner or fresh.is_relative_fw_only_corner):
            raise ValueError(f"Delta ablation override supports only A2/A3, got {fresh}")
        return fresh

    @property
    def ablation_(self):
        """Exact authoritative F×R provenance (W already-in-legacy). A2/A3 carry their explicit cell;
        the production corners derive it from the weighting (A4 == full-WB, else A0 legacy)."""
        from ..domain.work_weighting import FULL_WB_ABLATION, LEGACY_ABLATION
        self._validate_state(check_fitted=True)              # re-check on every provenance read
        ab = getattr(self, "_ablation", None)
        if ab is not None:
            return ab
        return FULL_WB_ABLATION if self.training_weighting == WORK_BALANCED else LEGACY_ABLATION

    def _validate_state(self, *, check_fitted: bool):
        """The SINGLE exact-type state validator (load / fit / predict / provenance). Checks the label
        is a canonical enum str, the ablation is None or an exact A2/A3 (plain-bool fields, legacy
        label), and — when ``check_fitted`` — that the fitted (ablation, label, vectorizer, mode,
        z-state) tuple is one of the historically/currently allowed shapes; any mixed/partial/forged
        state is rejected rather than silently changing probabilities."""
        ab = getattr(self, "_ablation", None)
        tw = getattr(self, "training_weighting", None)
        if type(tw) is not str or (tw != CHUNK_WEIGHTED_LEGACY and tw != WORK_BALANCED):
            raise ValueError(f"Delta training_weighting must be a canonical enum str, got {tw!r}")
        if ab is not None:
            if type(ab) is not AblationConfig:
                raise ValueError(f"Delta _ablation must be None or an AblationConfig, got {type(ab).__name__}")
            for f in ("weights", "feature_fit", "relative_fw"):     # catch a corrupt non-bool field
                if type(getattr(ab, f)) is not bool:
                    raise ValueError(f"Delta _ablation.{f} must be a plain bool")
            if not (ab.is_feature_state_only_corner or ab.is_relative_fw_only_corner):
                raise ValueError(f"Delta _ablation must be an A2/A3 cell, got {ab}")
            if tw == WORK_BALANCED:                                 # A2/A3 carry a legacy label
                raise ValueError("Delta A2/A3 carry a legacy label; work_balanced is split-brain")
        if not check_fitted:
            return
        vec, wv = getattr(self, "_vec", None), getattr(self, "_wv", None)
        if vec is not None and wv is not None:
            raise ValueError("Delta has both a pooled _vec and a work _wv (incoherent state)")
        fitted = (vec is not None) or (wv is not None)
        if not fitted:
            if getattr(self, "mean_", None) is not None:
                raise ValueError("Delta has a z-state but no vectorizer (incoherent)")
            return
        for a in ("mean_", "std_", "centroids_", "classes_"):       # a fitted Delta needs its z-state
            if getattr(self, a, None) is None:
                raise ValueError(f"fitted Delta missing {a}")
        wv_mode = getattr(wv, "mode", None)
        if ab is None:                                              # production corner A0 / A4
            if tw == WORK_BALANCED:                                 # A4: MODE_RELATIVE work vectorizer
                if wv is None or wv_mode != MODE_RELATIVE:
                    raise ValueError("A4 must be a fitted MODE_RELATIVE _wv")
            elif vec is None:                                       # A0: pooled _vec
                raise ValueError("A0 must be a fitted pooled _vec")
        elif ab.is_feature_state_only_corner:                       # A2: MODE_COUNT work vectorizer
            if wv is None or wv_mode != MODE_COUNT:
                raise ValueError("A2 must be a fitted MODE_COUNT _wv")
        elif vec is None:                                           # A3: pooled _vec
            raise ValueError("A3 must be a fitted pooled _vec")

    def __setstate__(self, state):
        # version-aware, fail-closed migration; a single validator binds the whole state machine.
        self.__dict__.update(state)
        sv = self.__dict__.setdefault("_schema_version", 1)
        if isinstance(sv, bool) or not isinstance(sv, int) or not (1 <= sv <= self.ARTIFACT_SCHEMA_VERSION):
            raise ValueError(f"invalid BurrowsDelta artifact schema version {sv!r}")
        self.__dict__.setdefault("training_weighting", CHUNK_WEIGHTED_LEGACY)
        self.__dict__.setdefault("_wv", None)
        self.__dict__.setdefault("vocabulary", None)
        if sv <= 2:
            # v1/v2 predate the F×R grid -> a production corner. FORCE (not setdefault) _ablation=None so
            # an injected _ablation on an old-schema artifact cannot forge an audit cell. The historically
            # allowed A0/A4 tuple is then validated below (bogus label / opposite tuple rejected).
            self.__dict__["_ablation"] = None
        elif "_ablation" not in state:
            # v3 MUST carry _ablation explicitly (a stripped/absent field is a corrupt artifact that
            # would silently load an A2/A3 as a production corner).
            raise ValueError("corrupt v3 BurrowsDelta artifact: missing _ablation")
        self._validate_state(check_fitted=True)

    # -- per-chunk PREDICT frequency (R axis) --------------------------------
    def _rel_freq(self, texts) -> np.ndarray:
        """Compatibility transform; A0 deliberately uses selected-MFW mass."""
        if getattr(self, "_ablation", None) is not None:   # A2/A3 explicit F×R
            return self._grid_rel_freq(texts)
        if getattr(self, "_wv", None) is not None:   # getattr: legacy pickles may have no _wv
            # work_balanced (A4): per-chunk count/ALL-tokens over the work-selected vocab
            return self._wv.transform(list(texts)).toarray().astype(np.float64)
        counts = self._vec.transform(list(texts)).toarray().astype(np.float64)
        totals = counts.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return counts / totals

    def _selected_counts(self, texts) -> np.ndarray:
        """Raw selected-MFW counts per chunk (dense) using the fitted F0/F1 vocabulary."""
        if getattr(self, "_wv", None) is not None:         # F1: WorkLevelVectorizer(mode=COUNT)
            return self._wv.transform(list(texts)).toarray().astype(np.float64)
        return self._vec.transform(list(texts)).toarray().astype(np.float64)

    def _grid_rel_freq(self, texts) -> np.ndarray:
        """A2/A3 per-chunk frequency. R0 = selected / Σselected; R1 = selected / ALL analyzer events
        (OOV/pruned included). A zero-denominator chunk yields a zero row (stays a work vote)."""
        counts = self._selected_counts(texts)
        if self._ablation.relative_fw:                     # R1
            events = analyzer_event_counts(_ANALYZER, list(texts)).copy()
            events[events == 0] = 1.0                      # zero-event chunk -> zero row (0/1)
            return counts / events[:, None]
        totals = counts.sum(axis=1, keepdims=True)         # R0
        totals[totals == 0] = 1.0
        return counts / totals

    def fit(self, texts, y, groups=None):
        texts = list(texts)
        y = np.asarray(y)
        if len(texts) != len(y):
            raise ValueError("texts and y must have the same length")
        # re-check provenance coherence at fit: defeats a post-construction ``est.training_weighting =
        # WORK_BALANCED`` that would forge a full-WB label over A2/A3 math.
        self._validate_state(check_fitted=False)
        if getattr(self, "_ablation", None) is not None:
            return self._fit_grid(texts, y, groups)
        if self.training_weighting == WORK_BALANCED:
            return self._fit_work_balanced(texts, y, groups)
        if self.vocabulary is not None:
            self._vec = CountVectorizer(vocabulary=self.vocabulary, lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        else:
            self._vec = CountVectorizer(max_features=self.mfw_count, lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        self._vec.fit(texts)
        self._wv = None                       # clear any prior WB state on a legacy (re)fit
        freqs = self._rel_freq(texts)
        group_freqs, group_y = _group_means(freqs, y, groups)
        self.mean_ = group_freqs.mean(axis=0)
        self.std_ = group_freqs.std(axis=0)
        self.std_[self.std_ == 0] = 1e-9
        group_z = (group_freqs - self.mean_) / self.std_
        if self.metric == "cosine" and groups is not None:
            group_z = group_z / (
                np.linalg.norm(group_z, axis=1, keepdims=True) + 1e-12
            )

        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack(
            [group_z[group_y == c].mean(axis=0) for c in self.classes_]
        )
        self.group_weighting_ = (
            (
                "equal_group_direction_after_within_group_mean_l2"
                if self.metric == "cosine"
                else "equal_group_after_within_group_mean"
            )
            if groups is not None
            else "equal_row"
        )
        return self

    def _fit_work_balanced(self, texts, y, groups):
        """Delta on work-balanced feature state (design §1): work-level MFW vocab + Σcounts/Σevents
        z-input, then the unchanged z-mean/std/centroid math on ONE row per work."""
        from ..features.work_vectorizer import WorkLevelVectorizer, MODE_RELATIVE, validate_work_ids
        if groups is None:
            raise ValueError("work_balanced Delta needs per-chunk groups")
        g = validate_work_ids(groups, len(y))
        for i, yi in enumerate(y):                     # exact integral labels (no float/bool merge)
            if isinstance(yi, (bool, np.bool_)) or not isinstance(yi, numbers.Integral):
                raise ValueError(f"work_balanced Delta labels must be non-bool integers, got y[{i}]={yi!r}")
        if self.vocabulary is not None:
            self._wv = WorkLevelVectorizer(analyzer_params=_ANALYZER, mode=MODE_RELATIVE,
                                           vocabulary=self.vocabulary)
        else:
            self._wv = WorkLevelVectorizer(analyzer_params=_ANALYZER, mode=MODE_RELATIVE,
                                           max_features=self.mfw_count, min_df_works=2)
        self._wv.fit(texts, g)
        self._vec = None
        work_ids, rel_rows = self._wv.transform_grouped(texts, g)      # one row per work, Σ/Σ
        group_freqs = rel_rows.toarray()
        lbl: dict = {}
        for gi, yi in zip(g, y):
            yi = int(yi)
            if gi in lbl and lbl[gi] != yi:
                raise ValueError(f"work {gi!r} spans multiple classes")
            lbl[gi] = yi
        group_y = np.array([lbl[w] for w in work_ids], dtype=y.dtype)
        self.mean_ = group_freqs.mean(axis=0)
        self.std_ = group_freqs.std(axis=0)
        self.std_[self.std_ == 0] = 1e-9
        group_z = (group_freqs - self.mean_) / self.std_
        if self.metric == "cosine":
            group_z = group_z / (np.linalg.norm(group_z, axis=1, keepdims=True) + 1e-12)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([group_z[group_y == c].mean(axis=0) for c in self.classes_])
        self.group_weighting_ = (
            "work_balanced_sumcounts_over_sumtokens_z"
            + ("_l2" if self.metric == "cosine" else "")
        )
        return self

    def _fit_grid(self, texts, y, groups):
        """A2/A3 off-diagonal cells: F (pooled vs work-rank/DF vocab) decoupled from R (mean of chunk
        selected/Σselected vs per-work Σselected/Σall-events). W stays already-in-legacy — one z-input
        row per work, unchanged z-mean/std/centroid math."""
        feature_state = self._ablation.feature_fit
        relative = self._ablation.relative_fw
        if groups is None:
            raise ValueError("A2/A3 Delta needs per-chunk groups")
        g = validate_work_ids(groups, len(y))
        for i, yi in enumerate(y):                     # exact integral labels (no float/bool merge)
            if isinstance(yi, (bool, np.bool_)) or not isinstance(yi, numbers.Integral):
                raise ValueError(f"A2/A3 Delta labels must be non-bool integers, got y[{i}]={yi!r}")
        # F axis — vocabulary (raw counts either way; R is applied separately below)
        if feature_state:                              # F1: work-rank / work-DF prune
            kw = ({"vocabulary": self.vocabulary} if self.vocabulary is not None
                  else {"max_features": self.mfw_count, "min_df_works": 2})
            self._wv = WorkLevelVectorizer(analyzer_params=_ANALYZER, mode=MODE_COUNT, **kw)
            self._wv.fit(texts, g)
            self._vec = None
        else:                                          # F0: pooled CountVectorizer(max_features|vocab)
            if self.vocabulary is not None:
                self._vec = CountVectorizer(vocabulary=self.vocabulary, lowercase=True,
                                            token_pattern=_TOKEN_PATTERN)
            else:
                self._vec = CountVectorizer(max_features=self.mfw_count, lowercase=True,
                                            token_pattern=_TOKEN_PATTERN)
            self._vec.fit(texts)
            self._wv = None
        # R axis — one train z-input row per work
        if relative:                                   # R1: per-work Σselected / Σall-events
            group_freqs, group_y = self._grid_work_rows_r1(texts, g, y)
        else:                                          # R0: mean over chunks of chunk selected/Σselected
            group_freqs, group_y = _group_means(self._grid_rel_freq(texts), y, g)
        self.mean_ = group_freqs.mean(axis=0)
        self.std_ = group_freqs.std(axis=0)
        self.std_[self.std_ == 0] = 1e-9
        group_z = (group_freqs - self.mean_) / self.std_
        if self.metric == "cosine":
            group_z = group_z / (np.linalg.norm(group_z, axis=1, keepdims=True) + 1e-12)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([group_z[group_y == c].mean(axis=0) for c in self.classes_])
        self.group_weighting_ = (f"delta_grid_F{int(feature_state)}R{int(relative)}"
                                 + ("_l2" if self.metric == "cosine" else ""))
        return self

    def _grid_work_rows_r1(self, texts, g, y):
        """One row per work = Σselected counts / Σall analyzer events (the A4 denominator formula, on
        whichever F0/F1 vocabulary). Work order = first-seen (matches ``_group_means``)."""
        counts = self._selected_counts(texts)          # chunks × selected (dense)
        events = analyzer_event_counts(_ANALYZER, list(texts))
        work_ids, gmat = _group_indicator(g)           # works × chunks 0/1
        work_counts = np.asarray(gmat @ counts)        # Σselected per work
        work_events = np.asarray(gmat @ events).ravel()  # Σall events per work
        we = work_events.copy()
        we[we == 0] = 1.0                              # zero-event work -> zero row (stays a vote)
        group_freqs = work_counts / we[:, None]
        lbl: dict = {}
        for gi, yi in zip(g, y):
            yi = int(yi)
            if gi in lbl and lbl[gi] != yi:
                raise ValueError(f"work {gi!r} spans multiple classes")
            lbl[gi] = yi
        group_y = np.array([lbl[w] for w in work_ids], dtype=y.dtype)
        return group_freqs, group_y

    def _z(self, texts) -> np.ndarray:
        return (self._rel_freq(texts) - self.mean_) / self.std_

    def distances(self, texts) -> np.ndarray:
        """Матрица расстояний (n_texts, n_classes); меньше = ближе."""
        self._validate_state(check_fitted=True)   # predict path: reject an incoherent/tampered state
        z = self._z(texts)
        if self.metric == "manhattan":
            # классическая Delta = средняя |z|; делим на число признаков для масштаба
            return manhattan_distances(z, self.centroids_) / z.shape[1]
        return cosine_distances(z, self.centroids_)

    def predict(self, texts) -> np.ndarray:
        d = self.distances(texts)
        return self.classes_[np.argmin(d, axis=1)]

    def predict_proba(self, texts) -> np.ndarray:
        """Псевдо-вероятности softmax(-distance) — для ансамбля/рангов."""
        d = self.distances(texts)
        x = -d
        x = x - x.max(axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)

    def feature_names(self) -> List[str]:
        if getattr(self, "_wv", None) is not None:
            return list(self._wv.feature_names())
        return list(self._vec.get_feature_names_out()) if self._vec else []


def _group_means(values: np.ndarray, y: np.ndarray, groups):
    """Collapse chunks to equal-weight work rows, validating one label per work."""
    if groups is None:
        return values, y
    groups = list(groups)
    if len(groups) != len(y):
        raise ValueError("groups and y must have the same length")
    order = list(dict.fromkeys(groups))
    means = []
    labels = []
    for group in order:
        idx = np.flatnonzero([item == group for item in groups])
        group_labels = np.unique(y[idx])
        if len(group_labels) != 1:
            raise ValueError(f"group {group!r} spans multiple classes")
        means.append(values[idx].mean(axis=0))
        labels.append(group_labels[0])
    return np.vstack(means), np.asarray(labels, dtype=y.dtype)
