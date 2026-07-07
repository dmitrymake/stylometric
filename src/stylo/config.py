"""Единый источник истины конфигурации.

Загружает configs/default.yaml в объект с атрибутным и dict-доступом.
Любой скрипт получает параметры отсюда — больше никаких копий VECTORIZER_PARAMS.

Использование:
    from stylo.config import load_config
    cfg = load_config()                      # configs/default.yaml
    cfg.chunking.chunk_size                   # 500
    cfg["features"]["char_ngrams"]["max_features"]
    cfg = load_config(overrides={"features.char_ngrams.bleach": False})
"""
from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, Mapping, Optional

import yaml


def _project_root() -> pathlib.Path:
    # src/stylo/config.py -> подняться к корню репозитория
    return pathlib.Path(__file__).resolve().parents[2]


DEFAULT_CONFIG_PATH = _project_root() / "configs" / "default.yaml"


class ConfigNode(Mapping):
    """Обёртка над dict с атрибутным доступом и неизменяемым контрактом чтения.

    Поддерживает cfg.a.b.c, cfg["a"]["b"], итерацию ключей и .get(path).
    """

    __slots__ = ("_d",)

    def __init__(self, d: Dict[str, Any]):
        object.__setattr__(self, "_d", d)

    def __getitem__(self, key: str) -> Any:
        val = self._d[key]
        return ConfigNode(val) if isinstance(val, dict) else val

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __getattr__(self, name: str) -> Any:
        # Приватные/dunder-имена (в т.ч. _d при распиковке) НЕ ищем в данных —
        # иначе рекурсия при pickle/copy в loky-воркерах.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    # Pickle: для передачи cfg в joblib/loky-воркеры.
    def __getstate__(self) -> Dict[str, Any]:
        return self._d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, "_d", state)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """cfg.get_path('features.char_ngrams.max_features')."""
        node: Any = self._d
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return ConfigNode(node) if isinstance(node, dict) else node

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConfigNode({list(self._d.keys())})"


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"Override path conflicts with scalar: {dotted}")
    node[parts[-1]] = value


def _coerce(value: str) -> Any:
    """Грубое приведение строковых CLI-override к типам (true/false/int/float)."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(
    path: Optional[pathlib.Path | str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigNode:
    """Загрузить YAML-конфиг с опциональными dot-path override.

    overrides: {"features.char_ngrams.bleach": False, ...}
               значения-строки приводятся к типам (для CLI --set k=v).
    """
    cfg_path = pathlib.Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    if overrides:
        for k, v in overrides.items():
            _set_dotted(raw, k, _coerce(v) if isinstance(v, str) else v)

    return ConfigNode(raw)


def parse_set_overrides(pairs: Optional[list[str]]) -> Dict[str, Any]:
    """Преобразовать ['a.b=1', 'c=true'] в dict для load_config(overrides=...)."""
    out: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set ожидает key=value, получено: {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out
