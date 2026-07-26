"""Evaluation-independent span contracts for mixed-authorship models."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabeledSpan:
    """A contiguous author-labelled token span using ``[start, end)`` offsets."""

    start: int
    end: int
    label: str

    @property
    def length(self) -> int:
        return self.end - self.start


Span = LabeledSpan

__all__ = ["LabeledSpan", "Span"]
