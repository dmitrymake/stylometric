"""CLI helpers for `stylo case ...`."""
from __future__ import annotations

import pathlib
from typing import Iterable, List

from ..jsonio import dump_strict
from .framework import (dossier_markdown, load_case_spec, load_passport, passport_markdown,
                        rank_passports, run_case, write_passport)


def run_spec(spec_path: str, out: str | None = None) -> dict:
    spec = load_case_spec(spec_path)
    passport = run_case(spec)
    data = passport.to_dict()
    if out:
        write_passport(passport, out)
    return data


def load_or_run_many(paths: Iterable[str]) -> List[dict]:
    rows = []
    for raw in paths:
        p = pathlib.Path(raw)
        if p.suffix.lower() == ".json":
            obj = load_passport(p)
            if "evidence_score" in obj and "case_id" in obj:
                rows.append(obj)
                continue
        rows.append(run_case(load_case_spec(p)).to_dict())
    return rows


def rank(paths: Iterable[str], out: str | None = None) -> List[dict]:
    rows = rank_passports(load_or_run_many(paths))
    if out:
        dump_strict(rows, out)
    return rows


def report(paths: Iterable[str], out: str | None = None) -> str:
    md = passport_markdown(load_or_run_many(paths))
    if out:
        p = pathlib.Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md + "\n", encoding="utf-8")
    return md


def dossier(paths: Iterable[str], out: str | None = None) -> str:
    md = dossier_markdown(load_or_run_many(paths))
    if out:
        p = pathlib.Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
    return md
