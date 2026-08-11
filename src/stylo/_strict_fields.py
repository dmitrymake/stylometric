"""Neutral exact-field checks for internal content-addressed contracts.

Schemas and scientific semantics stay in their owning modules.  This helper only
implements the repeated JSON container/scalar checks and raises the caller's
domain-specific exception with its existing wording.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_UNSET = object()
_HEX64 = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ExactFieldReader:
    error: type[Exception]
    object_type_message: str = "must be an exact JSON object"
    detailed_object_keys: bool = True
    string_policy: str = "nul_free"
    hash_message: str = "must be 64 lowercase hexadecimal characters"
    default_minimum: int | None = 0

    def object(
        self, value: object, keys: set[str] | frozenset[str], label: str
    ) -> dict[str, Any]:
        if type(value) is not dict:
            raise self.error(f"{label} {self.object_type_message}")
        expected = set(keys)
        actual = set(value)
        if actual != expected:
            detail = (
                f"; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
                if self.detailed_object_keys
                else ""
            )
            raise self.error(f"{label} keys must be exact{detail}")
        return value

    def array(
        self, value: object, label: str, *, nonempty: bool = False
    ) -> list[Any]:
        if type(value) is not list or (nonempty and not value):
            qualifier = " non-empty" if nonempty else ""
            raise self.error(f"{label} must be an exact{qualifier} array")
        return value

    def string(self, value: object, label: str) -> str:
        policy = self.string_policy
        base_invalid = type(value) is not str or not value
        if policy == "nonempty":
            if base_invalid:
                raise self.error(f"{label} must be an exact non-empty string")
        elif policy == "nul_free":
            if base_invalid or "\x00" in value:
                raise self.error(
                    f"{label} must be an exact non-empty NUL-free string"
                )
        elif policy == "nul_separate":
            if base_invalid:
                raise self.error(f"{label} must be an exact non-empty string")
            if "\x00" in value:
                raise self.error(f"{label} must not contain NUL")
        elif policy == "trimmed_nul":
            if base_invalid or value != value.strip() or "\x00" in value:
                raise self.error(
                    f"{label} must be an exact non-empty trimmed string"
                )
        elif policy in {"trimmed_control", "trimmed_control_separate"}:
            trimmed_invalid = base_invalid or value != value.strip()
            has_control = not base_invalid and any(
                ord(char) < 32 or ord(char) == 127 for char in value
            )
            if trimmed_invalid or (policy == "trimmed_control" and has_control):
                raise self.error(
                    f"{label} must be an exact non-empty trimmed string"
                )
            if has_control:
                raise self.error(f"{label} contains a control character")
        else:  # pragma: no cover - construction-time programming error
            raise RuntimeError(f"unknown exact-field string policy: {policy}")
        return value

    def integer(
        self, value: object, label: str, *, minimum: int | None | object = _UNSET
    ) -> int:
        lower = self.default_minimum if minimum is _UNSET else minimum
        if type(value) is not int or (lower is not None and value < lower):
            suffix = f" >= {lower}" if lower is not None else ""
            raise self.error(f"{label} must be an exact integer{suffix}")
        return value

    def sha256(self, value: object, label: str) -> str:
        text = self.string(value, label)
        if len(text) != 64 or any(char not in _HEX64 for char in text):
            raise self.error(f"{label} {self.hash_message}")
        return text
