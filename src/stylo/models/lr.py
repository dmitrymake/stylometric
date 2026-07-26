"""Логистическая регрессия и сборка пайплайна.

Старый публичный ``make_classifier`` сохранён для import-совместимости, но его
калибровочная ветка не имела work-group контракта и теперь fail-closed. Активная
научная калибровка маршрутизируется только через явно group-aware evaluation API.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


_FROM_CFG = "__from_cfg__"


class UngroupedCalibrationError(ValueError):
    """The compatibility classifier factory cannot calibrate independent works."""


def make_logreg(cfg, class_weight=_FROM_CFG) -> LogisticRegression:
    c = cfg.get_path("model.classifier", {})
    cw = ((c.get("class_weight", "balanced") if hasattr(c, "get") else "balanced")
          if class_weight is _FROM_CFG else class_weight)
    return LogisticRegression(
        max_iter=c.get("max_iter", 2000) if hasattr(c, "get") else 2000,
        class_weight=cw,
        solver=c.get("solver", "lbfgs") if hasattr(c, "get") else "lbfgs",
        C=c.get("C", 1.0) if hasattr(c, "get") else 1.0,
    )


def make_scaler(cfg):
    kind = cfg.get_path("model.scaler", "maxabs")
    if kind == "maxabs":
        return MaxAbsScaler()
    raise ValueError(f"Неизвестный scaler: {kind}")


def make_classifier(cfg, calibrate: bool | None = None):
    """Import-compatible classifier factory over already-vectorized X.

    ``calibrate=None`` reads ``model.calibration.enabled``. The uncalibrated
    scaler+LR pipeline remains compatible; any learned-calibration request fails
    closed because this factory has no work-group input or split contract.
    """
    base = Pipeline([("scaler", make_scaler(cfg)), ("lr", make_logreg(cfg))])
    do_cal = cfg.get_path("model.calibration.enabled", False) if calibrate is None else calibrate
    if type(do_cal) is not bool:
        raise TypeError(
            f"calibrate must resolve to a plain bool, got {type(do_cal).__name__}"
        )
    if do_cal:
        raise UngroupedCalibrationError(
            "make_classifier learned calibration is retired: this compatibility "
            "factory has no work-group split contract; use an explicitly "
            "group-aware, versioned evaluation route"
        )
    return base


def make_full_pipeline(cfg, vectorizer) -> Pipeline:
    """Полный продакшен-пайплайн: StyloVectorizer -> Scaler -> LR (без калибровки,
    калибровку держим отдельно, чтобы не усложнять сериализацию)."""
    return Pipeline([
        ("vectorizer", vectorizer),
        ("scaler", make_scaler(cfg)),
        ("classifier", make_logreg(cfg)),
    ])
