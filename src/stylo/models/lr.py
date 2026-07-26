"""Логистическая регрессия (+опциональная калибровка) и сборка пайплайна.

Калибровка (isotonic) применяется ВНУТРИ train-фолда — это важно: калибровать на
всём корпусе = утечка. CalibratedClassifierCV делает внутренний CV по train.
"""
from __future__ import annotations

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


_FROM_CFG = "__from_cfg__"


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
    """Классификатор, работающий на УЖЕ векторизованном X (scaler внутри).

    calibrate=None -> берём из конфига (model.calibration.enabled).
    """
    base = Pipeline([("scaler", make_scaler(cfg)), ("lr", make_logreg(cfg))])
    do_cal = cfg.get_path("model.calibration.enabled", False) if calibrate is None else calibrate
    if not do_cal:
        return base
    method = cfg.get_path("model.calibration.method", "isotonic")
    return CalibratedClassifierCV(base, method=method, cv=3)


def make_full_pipeline(cfg, vectorizer) -> Pipeline:
    """Полный продакшен-пайплайн: StyloVectorizer -> Scaler -> LR (без калибровки,
    калибровку держим отдельно, чтобы не усложнять сериализацию)."""
    return Pipeline([
        ("vectorizer", vectorizer),
        ("scaler", make_scaler(cfg)),
        ("classifier", make_logreg(cfg)),
    ])
