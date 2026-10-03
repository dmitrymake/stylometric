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
import hashlib
import importlib.resources
import pathlib
from typing import Any, Dict, Mapping, Optional

import yaml


# The default is package data, not a repository-relative file. This keeps
# ``load_config()`` usable from wheels, sdists and git archives alike.
DEFAULT_CONFIG_PATH = importlib.resources.files("stylo.resources").joinpath("default.yaml")


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


def with_overrides(cfg: "ConfigNode", dotted_overrides: Dict[str, Any]) -> "ConfigNode":
    """Return a cfg-clone with dotted-path overrides — the sanctioned way to build a trusted
    cfg with explicitly-allowed root/policy (e.g. the RuAA full-corpus benchmark contract)."""
    raw = cfg.to_dict()
    for k, v in dotted_overrides.items():
        _set_dotted(raw, k, copy.deepcopy(v))
    return ConfigNode(raw)


def artifact_config_id(cfg: "ConfigNode") -> str:
    """Bind model/computation settings independently of output and presentation.

    A bundle token is derived from the training configuration and supplied after
    training. Including it in that configuration's digest creates a circular
    identity. View-only top-k and report directory also do not change fitted
    scores. Features, panel, processing, data paths and weighting stay bound.
    """
    from .jsonio import dumps_strict

    raw = cfg.to_dict()
    if isinstance(raw.get("evaluation"), dict):
        raw["evaluation"].pop("top_k_candidates", None)
    if isinstance(raw.get("paths"), dict):
        raw["paths"].pop("docs", None)
    deployment = raw.get("deployment")
    if isinstance(deployment, dict):
        deployment.pop("expected_bundle_token", None)
        if not deployment:
            raw.pop("deployment")
    return hashlib.sha256(dumps_strict(raw, sort_keys=True).encode("utf-8")).hexdigest()


def deployment_candidates(cfg: "ConfigNode") -> tuple[str, ...]:
    """Require an explicit deployment panel, independent of benchmark exclusions."""
    authors = cfg.get_path("deployment.candidate_authors")
    unknown = cfg.get_path("corpus_policy.unknown_dir_name", "unknown")
    if (
        type(authors) is not list
        or len(authors) < 2
        or any(
            type(author) is not str or not author or author != author.strip()
            or author in {".", "..", unknown} or "/" in author or "\\" in author
            for author in authors
        )
        or len(set(authors)) != len(authors)
    ):
        raise ValueError(
            "deployment.candidate_authors must explicitly list at least two unique "
            "author IDs (excluding unknown); use a YAML list or repeated --candidate-author"
        )
    return tuple(sorted(authors))


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    if type(dotted) is not str or not dotted or any(not part for part in dotted.split(".")):
        raise ValueError("Override path must contain non-empty dotted keys")
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"Override path conflicts with scalar: {dotted}")
    node[parts[-1]] = value


def _merge_config(defaults: dict, supplied: dict) -> dict:
    """Overlay nested mappings; explicit scalars, lists and null replace defaults."""
    merged = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if type(key) is not str or not key:
            raise ValueError("Configuration keys must be non-empty strings")
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        elif isinstance(value, dict):
            merged[key] = _merge_config({}, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


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
    """Overlay an optional YAML file on packaged defaults, then dot-path overrides.

    overrides: {"features.char_ngrams.bleach": False, ...}
               значения-строки приводятся к типам (для CLI --set k=v).
    """
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if type(raw) is not dict:
        raise ValueError("Packaged configuration must be a YAML mapping")
    if path is not None:
        with pathlib.Path(path).open("r", encoding="utf-8") as fh:
            supplied = yaml.safe_load(fh)
        if type(supplied) is not dict:
            raise ValueError("Configuration file must contain a YAML mapping")
        raw = _merge_config(raw, supplied)

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
        if not key.strip():
            raise ValueError("--set requires a non-empty key")
        out[key.strip()] = val.strip()
    return out
