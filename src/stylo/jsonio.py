"""Strict, deterministic JSON I/O for reproducible artifacts.

``NaN``, ``Infinity`` and ``-Infinity`` are not valid JSON, yet Python's decoder
accepts them and its encoder emits them by default. Every committed artifact must
round-trip through a strict parser, so this module provides the single writer the
project uses for machine-readable results.

On write, non-finite floats are serialised as ``null``: in this codebase they only
ever mark an undefined or not-applicable metric (an ECE for a non-probabilistic
model, a self-comparison p-value), never a real value. ``allow_nan=False`` then
guards against anything slipping through. On read, non-finite constants and
duplicate object keys are rejected so that a malformed artifact fails loudly
instead of poisoning a downstream hash.

The writer mirrors the strict parser already used for benchmark manifests
(:mod:`stylo.benchmarks.loader`) and scoring bundles.

Determinism is byte-for-byte for a given input tree: dict key order is preserved
(insertion order), sets are sorted, and numpy scalars/arrays are coerced to native
Python. If the caller builds a dict from an unordered source, pass ``sort_keys=True``
for order-independent output.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "StrictJSONError",
    "dumps_strict",
    "dump_strict",
    "loads_strict",
    "load_strict",
    "sanitise",
]


class StrictJSONError(ValueError):
    """Raised when JSON text violates the strict interchange contract."""


def _to_python_scalar(value: Any) -> Any:
    """Best-effort conversion of numpy scalars/arrays to plain Python objects."""
    # Arrays with shape -> nested lists (preserve shape, incl. singleton [x]).
    # .item() would collapse a 1-element array to a bare scalar, losing the shape.
    if getattr(value, "ndim", 0) and hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (ValueError, TypeError):
            pass
    # numpy scalar -> python builtin by dtype kind. item()/tolist() keep the numpy
    # type for np.float128 (Python float lacks 128-bit precision), so coerce here.
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    try:
        if kind == "f":
            # A finite longdouble can exceed float64 range: float() silently yields
            # inf, which would otherwise be written as null (silent data loss). Only
            # a genuinely non-finite value becomes null; an overflow is raised.
            try:
                import numpy as _np
                originally_finite = bool(_np.isfinite(value))
            except Exception:
                originally_finite = None
            reduced = float(value)
            if math.isfinite(reduced):
                return reduced
            if originally_finite:
                raise StrictJSONError(f"numpy float {value!r} exceeds JSON float64 range")
            return None
        if kind in ("i", "u"):
            return int(value)
        if kind == "b":
            return bool(value)
    except StrictJSONError:
        raise
    except (ValueError, TypeError):
        pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            reduced = value.item()
        except (ValueError, TypeError):
            reduced = value
        # Hand back whatever item() produced (a container from a 0-d object array,
        # or a still-numpy scalar wrapped inside one) so sanitise re-processes it.
        # The recursion guard in sanitise stops if the type is unchanged.
        if reduced is not value:
            return reduced
    return value


def sanitise(value: Any) -> Any:
    """Recursively coerce a value into strict-JSON-safe Python.

    Non-finite floats become ``None``; numpy scalars/arrays become native
    Python; mappings and sequences are rebuilt so the result is a plain tree.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, sub in value.items():
            str_key = key if isinstance(key, str) else str(key)
            if str_key in result:
                raise StrictJSONError(
                    f"dict keys collide to {str_key!r} after string coercion; "
                    "refusing to drop a value silently"
                )
            result[str_key] = sanitise(sub)
        return result
    if isinstance(value, set):
        # sort for output determinism across interpreter runs (PYTHONHASHSEED)
        return [sanitise(sub) for sub in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [sanitise(sub) for sub in value]
    converted = _to_python_scalar(value)
    # Stop if the value could not be reduced to a different type; recursing on the
    # same numpy type (e.g. an un-coercible longdouble) would loop forever.
    if converted is value or type(converted) is type(value):
        return value
    return sanitise(converted)


def dumps_strict(
    obj: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialise ``obj`` to strict JSON text (no NaN/Infinity).

    A faithful drop-in for :func:`json.dumps`: same defaults (``indent=None`` gives
    compact output) and the same ``separators`` argument, so a canonical/compact
    content-hash payload keeps its exact bytes. :func:`dump_strict` pretty-prints
    files with ``indent=2``.
    """
    return json.dumps(
        sanitise(obj),
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
        allow_nan=False,
    )


def dump_strict(
    obj: Any,
    path: str | os.PathLike[str],
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    trailing_newline: bool = True,
) -> Path:
    """Atomically write ``obj`` as strict JSON to ``path`` and return the path."""
    text = dumps_strict(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    if trailing_newline:
        text += "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        # mkstemp forces 0600; restore the usual umask-respecting file mode so an
        # artifact stays world-readable like a normal write_text() would produce.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise StrictJSONError(f"non-finite number {value!r} is not valid JSON")


def _parse_finite_float(value: str) -> float:
    # Catches magnitudes like 1e999 that json parses to inf via parse_float
    # without ever emitting the NaN/Infinity tokens parse_constant guards.
    number = float(value)
    if not math.isfinite(number):
        raise StrictJSONError(f"number {value!r} overflows to a non-finite float")
    return number


def loads_strict(text: str) -> Any:
    """Parse strict JSON text, rejecting NaN/Infinity and duplicate keys."""
    if not isinstance(text, str):
        raise TypeError("JSON payload must be text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_non_finite,
            parse_float=_parse_finite_float,
        )
    except json.JSONDecodeError as exc:
        raise StrictJSONError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except _DuplicateKeyError as exc:
        raise StrictJSONError(str(exc)) from exc


def load_strict(path: str | os.PathLike[str]) -> Any:
    """Read a UTF-8 file and parse it as strict JSON."""
    return loads_strict(Path(path).read_text(encoding="utf-8"))
