"""Частотности функциональных слов.

Два режима словаря:
  mfw        — топ-N самых частотных слов (CountVectorizer max_features),
  fixed_list — фиксированный язык-специфичный список (RU: 405 слов из lang.py).

Две независимые оси (B4-B increment 3):
  F (feature state) — словарь: pooled ``CountVectorizer`` (F0) vs work-level
      ``WorkLevelVectorizer`` (F1); маршрутизируется наличием ``groups`` на ``fit``.
  R (transform)     — сырые счётчики (R0) vs относительная частота
      ``selected / ВСЕ события анализатора`` (R1, знаменатель — все токены до prune);
      задаётся ``relative_fw``.

``relative_fw=None`` — LEGACY corner coupling (R = F): байт-в-байт A0 (F0R0, сырые)
и A4 (F1R1, ``WorkLevelVectorizer`` relative). Явный bool — audit-путь A2 (F1R0) и
A3 (F0R1). Нормализация (MaxAbs/Z) — на стороне модели. Для НАСТОЯЩЕЙ Burrows's Delta
см. ``stylo.models.delta``.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokens import Doc  # noqa: F401 (kept: reps carry spaCy Docs by contract)

from ..lang import function_words
from .base import FeatureBlock
from .work_vectorizer import (MODE_COUNT, MODE_RELATIVE, WorkLevelVectorizer,
                              analyzer_event_counts, relative_by_events)

_TOKEN_PATTERN = r"(?u)\b\w+\b"
_ANALYZER = {"analyzer": "word", "token_pattern": _TOKEN_PATTERN, "lowercase": True}


class FunctionWordBlock(FeatureBlock):
    group = "function_words"
    name = "function_words"

    def __init__(self, mode: str = "mfw", mfw_count: int = 300, lang: str = "ru",
                 relative_fw: bool | None = None):
        assert mode in {"mfw", "fixed_list"}
        if relative_fw is not None and type(relative_fw) is not bool:
            raise TypeError(f"relative_fw must be a plain bool or None, got {type(relative_fw).__name__}")
        self.mode = mode
        self.mfw_count = mfw_count
        self.lang = lang
        # None -> legacy corner coupling (R follows F); bool -> explicit audit R policy (A2/A3).
        self.relative_fw = relative_fw
        self._vec: CountVectorizer | None = None       # F0: pooled vocab
        self._wv: WorkLevelVectorizer | None = None      # F1: work-level vocab
        self.feature_fit_: bool | None = None            # effective F (set at fit)
        self.relative_fw_: bool | None = None            # effective R (set at fit)

    def __setstate__(self, state):
        super().__setstate__(state)
        # An artifact is pre-increment-3 iff ``relative_fw`` is absent from BOTH the top-level state AND
        # the constructor snapshot (``_ctor_state``) — stripping only the top-level field on an inc-3
        # artifact must NOT downgrade it to legacy (that would silently flip R). Then the full fitted
        # coherence table (below) rejects any partial/incoherent state.
        ctor = self.__dict__.get("_ctor_state") or {}
        is_inc3 = ("relative_fw" in self.__dict__) or ("relative_fw" in ctor)
        if not is_inc3:
            self.__dict__["relative_fw"] = None
            self.__dict__.setdefault("feature_fit_", None)
            self.__dict__.setdefault("relative_fw_", None)
        elif "relative_fw" not in self.__dict__:
            self.__dict__["relative_fw"] = ctor.get("relative_fw")   # reconstruct from the ctor snapshot
        self._assert_fw_coherent()

    def _assert_fw_coherent(self):
        """Exact-type coherence over (relative_fw, feature_fit_, relative_fw_, _vec, _wv, mode)."""
        rf = self.__dict__.get("relative_fw")
        ff_, rf_ = self.__dict__.get("feature_fit_"), self.__dict__.get("relative_fw_")
        vec, wv = self.__dict__.get("_vec"), self.__dict__.get("_wv")
        if rf is not None and type(rf) is not bool:
            raise ValueError(f"corrupt FunctionWord relative_fw {rf!r}")
        if vec is not None and wv is not None:
            raise ValueError("FunctionWord has both a pooled _vec and a work _wv (incoherent)")
        fitted = (vec is not None) or (wv is not None)
        if not fitted:
            return
        wv_mode = getattr(wv, "mode", None)
        if rf is None:                                          # legacy corner coupling: A0 (_vec) / A4 (_wv rel)
            if wv is not None and wv_mode != MODE_RELATIVE:
                raise ValueError("legacy FunctionWord _wv must be MODE_RELATIVE (A4)")
            if type(rf_) is bool and rf_ is not (wv is not None):   # if a bool is recorded it must match
                raise ValueError("legacy FunctionWord effective relative_fw_ is incoherent")
        else:                                                   # explicit inc-3 A2/A3 (or explicit corner)
            if type(ff_) is not bool or type(rf_) is not bool:
                raise ValueError("fitted explicit FunctionWord needs plain-bool effective axes")
            if rf_ is not rf:                                   # the fitted R must match the policy
                raise ValueError("fitted FunctionWord relative_fw_ != relative_fw policy")
            if ff_:                                             # F1: work vocabulary (MODE_COUNT)
                if wv is None or wv_mode != MODE_COUNT:
                    raise ValueError("explicit F1 FunctionWord needs a MODE_COUNT _wv")
            elif vec is None:                                   # F0: pooled vocabulary
                raise ValueError("explicit F0 FunctionWord needs a pooled _vec")

    # -- vocabulary (F axis) --------------------------------------------------
    def _fit_pooled_vocab(self, texts):
        if self.mode == "mfw":
            self._vec = CountVectorizer(max_features=self.mfw_count, lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        else:
            self._vec = CountVectorizer(vocabulary=sorted(function_words(self.lang)), lowercase=True,
                                        token_pattern=_TOKEN_PATTERN)
        self._vec.fit(list(texts))
        self._wv = None

    def _fit_work_vocab(self, texts, groups, mode):
        kw = ({"max_features": self.mfw_count, "min_df_works": 2} if self.mode == "mfw"
              else {"vocabulary": sorted(function_words(self.lang))})
        self._wv = WorkLevelVectorizer(analyzer_params=_ANALYZER, mode=mode, **kw)
        self._wv.fit(list(texts), groups)
        self._vec = None

    def fit(self, texts, reps, groups=None):
        feature_on = groups is not None
        relative_fw = getattr(self, "relative_fw", None)
        relative_on = feature_on if relative_fw is None else relative_fw
        self.feature_fit_ = feature_on
        self.relative_fw_ = relative_on
        if relative_fw is None:
            # LEGACY corner coupling — byte-exact prior behavior. A4 keeps MODE_RELATIVE (F1R1 in one).
            if feature_on:
                self._fit_work_vocab(texts, groups, MODE_RELATIVE)
            else:
                self._fit_pooled_vocab(texts)
        else:
            # explicit audit F×R: vocab per F (work-level counts vs pooled), R applied at transform.
            if feature_on:
                self._fit_work_vocab(texts, groups, MODE_COUNT)
            else:
                self._fit_pooled_vocab(texts)
        return self

    # -- transform (R axis) ---------------------------------------------------
    def _selected_counts(self, texts) -> csr_matrix:
        """Raw selected counts from the fitted vocab. NB: only valid when ``_wv`` (if present) is a
        MODE_COUNT vectorizer; the legacy MODE_RELATIVE ``_wv`` is handled directly in ``transform``."""
        if getattr(self, "_wv", None) is not None:
            return self._wv.transform(list(texts))         # MODE_COUNT -> raw selected counts
        assert self._vec is not None, "fit перед transform"
        return self._vec.transform(list(texts)).astype(np.float64).tocsr()

    def transform(self, texts, reps) -> csr_matrix:
        # Dispatch on FITTED state only (never the mutable ``relative_fw`` constructor param): a legacy
        # MODE_RELATIVE ``_wv`` already yields the A4 relative transform; every other corner reads raw
        # selected counts and applies the fitted R policy ``relative_fw_``. This makes transform
        # invariant to any post-fit mutation of ``relative_fw`` (no double division, no branch flip).
        wv = getattr(self, "_wv", None)
        if wv is not None and getattr(wv, "mode", None) == MODE_RELATIVE:
            return wv.transform(list(texts))               # legacy A4 (F1R1 in one)
        if getattr(self, "relative_fw_", False) is not True:
            # R0: raw selected counts, preserving the legacy dtype (A0 _vec stays int64, byte-exact;
            # incl. an old A0 pickle whose relative_fw_ is None).
            return wv.transform(list(texts)) if wv is not None else self._vec.transform(list(texts))
        # R1: selected / all analyzer events
        return relative_by_events(self._selected_counts(texts), analyzer_event_counts(_ANALYZER, list(texts)))

    def feature_names(self) -> List[str]:
        if getattr(self, "_wv", None) is not None:
            return [f"fw::{w}" for w in self._wv.feature_names()]
        assert self._vec is not None
        return [f"fw::{w}" for w in self._vec.get_feature_names_out()]
