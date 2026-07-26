"""Strict JSON loading for benchmark manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import BenchmarkManifest
from .validator import ManifestValidationError, validate_manifest


class _DuplicateKeyError(ValueError):
    pass


class _InvalidConstantError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise _InvalidConstantError(f"non-finite number {value!r} is not valid JSON")


def loads_manifest(text: str) -> BenchmarkManifest:
    """Parse and validate a manifest from strict JSON text.

    Python's JSON decoder normally accepts duplicate keys and the non-standard
    values ``NaN``/``Infinity``.  Benchmark manifests reject both because they
    undermine deterministic interchange and hashing.
    """

    if type(text) is not str:
        raise TypeError("manifest JSON must be text")
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(
            [f"$: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]
        ) from exc
    except (_DuplicateKeyError, _InvalidConstantError) as exc:
        raise ManifestValidationError([f"$: {exc}"]) from exc
    return validate_manifest(raw)


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Read a UTF-8 JSON file and return a validated manifest."""

    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestValidationError(
            [f"$: {manifest_path} is not valid UTF-8: {exc}"]
        ) from exc
    return loads_manifest(text)


__all__ = ["load_manifest", "loads_manifest"]
