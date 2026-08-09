#!/usr/bin/env python3
"""Prepare an unapproved paired-audit v3.2 corrected corpus/fold candidate only."""
from __future__ import annotations

import argparse
import hashlib
import pathlib

from stylo.config import load_config
from stylo.eval.paired_audit.corrected_v3_2 import (
    HISTORICAL_PARENT_DIGEST,
    assert_preparation_parity,
    prepare_corrected_v3_2,
)
from stylo.eval.paired_audit.run_plan import config_id
from stylo.jsonio import load_strict


def _sha256(path: pathlib.Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing regular required file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ruaa_selection(path: pathlib.Path) -> list[str]:
    raw = load_strict(path)
    authors = raw.get("authors") if isinstance(raw, dict) else None
    if not isinstance(authors, dict):
        raise RuntimeError("RuAA selection evidence must contain authors")
    work_ids = []
    for author, record in authors.items():
        books = record.get("books") if isinstance(record, dict) else None
        if type(author) is not str or not isinstance(books, list) or record.get("n_books") != len(books):
            raise RuntimeError("malformed RuAA selection evidence")
        for book in books:
            slug = book.get("book") if isinstance(book, dict) else None
            if type(slug) is not str or not slug:
                raise RuntimeError("malformed RuAA work selection")
            work_ids.append(f"{author}/{slug}")
    work_ids.sort()
    if len(authors) != 22 or len(work_ids) != 137 or len(work_ids) != len(set(work_ids)):
        raise RuntimeError("RuAA selection evidence must be exactly 22 authors / 137 works")
    return work_ids


def prepare(repo: pathlib.Path, output: pathlib.Path, *, parent: pathlib.Path | None = None,
            ruaa_manifest: pathlib.Path | None = None) -> dict:
    repository = repo.resolve()
    cfg = load_config(repository / "configs" / "default.yaml")
    return prepare_corrected_v3_2(
        historical_parent_root=parent or repository / "data" / "audit_corpus" / HISTORICAL_PARENT_DIGEST,
        output_root=output.resolve(),
        ruaa_parent_selection=_ruaa_selection(ruaa_manifest or repository / "data" / "ruaa_bench_v1" / "manifest.json"),
        config_hash=config_id(cfg),
        protocol_sha256=_sha256(repository / "research" / "work_balanced" / "paired_audit_protocol.md"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--historical-parent-root", type=pathlib.Path)
    parser.add_argument("--ruaa-selection-manifest", type=pathlib.Path)
    parser.add_argument("--compare-with", type=pathlib.Path)
    args = parser.parse_args()
    result = prepare(args.repo_root, args.output_root, parent=args.historical_parent_root,
                     ruaa_manifest=args.ruaa_selection_manifest)
    if args.compare_with:
        parity = assert_preparation_parity(args.output_root.resolve(), args.compare_with.resolve())
        print(f"parity={parity['paths_bytes_modes_digest']} entries={parity['n_entries']}")
    print(result["candidate_root"])


if __name__ == "__main__":
    main()
