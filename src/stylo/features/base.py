"""Базовый контракт фич-блока.

Блоки работают с предвычисленными лёгкими представлениями Rep (stylo.features.reps),
а не с тяжёлыми spaCy Doc — это даёт быстрый leakage-free LOBO/sweep (Rep грузятся
один раз на процесс из единого файла, без повторной десериализации DocBin).
"""
from __future__ import annotations

import abc
import copy
import functools
from typing import List, Optional, Sequence

from scipy.sparse import csr_matrix


class FeatureBlock(abc.ABC):
    """Абстрактный блок признаков.

    name  — уникальное имя блока (для конфигов/отчётов), напр. "char_ngrams".
    group — группа для ablation-sweep (часто == name; субблоки делят group).

    ``fit`` принимает опциональный ``groups`` (per-chunk work id, P1 §4.3): если он
    передан, обучаемое состояние (словарь/DF/IDF/MFW) фитится с равным весом на работу;
    ``groups=None`` — прежнее chunk-pooled поведение (chunk_weighted_legacy, P0).

    Блоки cloneable как sklearn-эстиматоры: ``sklearn.clone`` возвращает UNFITTED копию
    (никакого fitted state) с ПОЛНОСТЬЮ независимым состоянием — клон внутри Pipeline/
    калибровки не переносит обучение между фолдами и не делит изменяемые контейнеры
    (списки/словари) с оригиналом или другими клонами. Реализовано через deep-снимок
    конструкторного состояния (см. ниже) — без ограничения «__init__ не меняет параметры».
    """

    name: str = "block"
    group: str = "block"
    SCHEMA_VERSION = 2                                # v2 adds _wv/_ctor_state; a versionless pickle = v1

    def __setstate__(self, state):
        # migrate a pre-B2 pickled block (inside an old model.pkl): fill B1/B2 attrs, reject a future
        # schema, and never silently reuse fitted state as an unfitted clone (see __sklearn_clone__).
        self.__dict__.update(state)
        sv = self.__dict__.setdefault("_schema_version", 1)
        if isinstance(sv, bool) or not isinstance(sv, int) or not (1 <= sv <= FeatureBlock.SCHEMA_VERSION):
            raise ValueError(f"invalid feature-block artifact schema version {sv!r}")
        self.__dict__.setdefault("_wv", None)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None:
            return

        @functools.wraps(init)                       # keep the real __init__ signature/introspection
        def __init__(self, *args, **kw):
            init(self, *args, **kw)
            self.__dict__.setdefault("_schema_version", FeatureBlock.SCHEMA_VERSION)
            # Deep, detached snapshot of the pristine constructor state so the original may
            # mutate freely without leaking into future clones. Always overwrite: for a
            # subclass whose __init__ calls super().__init__(), the OUTERMOST __init__
            # completes last and thus captures the full (base + subclass) state.
            snap = {k: v for k, v in self.__dict__.items() if k != "_ctor_state"}
            self.__dict__["_ctor_state"] = copy.deepcopy(snap)

        cls.__init__ = __init__

    def __sklearn_clone__(self) -> "FeatureBlock":
        """Return an UNFITTED copy: constructor state only, deep-copied for full isolation."""
        state = getattr(self, "_ctor_state", None)
        if state is not None:
            fresh = self.__class__.__new__(self.__class__)
            snap = copy.deepcopy(state)
            fresh.__dict__.update(snap)
            fresh.__dict__["_ctor_state"] = copy.deepcopy(snap)
            return fresh
        # Pre-B2 pickle without a snapshot: rebuild a genuinely UNFITTED instance by re-running
        # __init__ from the constructor params (stored as same-named attrs) — never copy the
        # fitted dict (that would leak a fitted _vec/vocab into the "clone").
        import inspect
        params = [p for p in inspect.signature(self.__class__.__init__).parameters if p != "self"]
        kwargs = {p: self.__dict__[p] for p in params if p in self.__dict__}
        return self.__class__(**kwargs)

    @abc.abstractmethod
    def fit(self, texts: Sequence[str], reps: Sequence,
            groups: Optional[Sequence] = None) -> "FeatureBlock":
        ...

    @abc.abstractmethod
    def transform(self, texts: Sequence[str], reps: Sequence) -> csr_matrix:
        ...

    def fit_transform(self, texts: Sequence[str], reps: Sequence,
                      groups: Optional[Sequence] = None) -> csr_matrix:
        if groups is None:
            return self.fit(texts, reps).transform(texts, reps)   # exact legacy call (P0 parity)
        return self.fit(texts, reps, groups=groups).transform(texts, reps)

    @abc.abstractmethod
    def feature_names(self) -> List[str]:
        ...

    def n_features(self) -> int:
        return len(self.feature_names())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r}>"
