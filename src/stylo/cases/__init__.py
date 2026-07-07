"""Case-gate framework for historical authorship hypotheses.

The public entry points are intentionally small: load a case spec, run the
gate/attribution protocol, and rank generated passports. Existing bespoke case
scripts can migrate here incrementally.
"""
from __future__ import annotations

from .framework import (CasePassport, CaseSpec, dossier_markdown, load_case_spec,
                        load_passport, passport_markdown, rank_passports, run_case,
                        write_passport)

__all__ = [
    "CasePassport",
    "CaseSpec",
    "load_case_spec",
    "load_passport",
    "dossier_markdown",
    "passport_markdown",
    "rank_passports",
    "run_case",
    "write_passport",
]
