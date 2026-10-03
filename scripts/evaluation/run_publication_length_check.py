#!/usr/bin/env python3
"""Post-hoc development-only sensitivity to the first two included chunks/work."""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import time
import warnings

import numpy as np

import run_publication_comparison as primary

ARMS = ("stylo_A0_current", "stylo_A4_current", "cosine_delta_300")


def development_prefix(catalog, root, cfg):
    authors = primary.validate_catalog(catalog)  # metadata validation only
    nlp = primary.load_sentencizer(cfg.language.code)
    texts, labels, groups, lengths, prefix_tokens = [], [], [], {}, []
    for row in catalog["works"]:
        if row["role"] != "development":
            continue  # locked bodies are never opened
        payload = (root / (row["work_id"] + ".txt")).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["text_sha256"]:
            raise ValueError("development input checksum differs from catalog")
        clean = primary.normalize(payload.decode("utf-8"), cfg.language.spacy_model,
                                  cfg.language.spacy_fallback)
        chunks = primary.make_sent_chunks(
            primary.CombinedDoc(primary.sentences_for_text(clean, nlp)),
            cfg.chunking.chunk_size, cfg.chunking.min_words, overlap=0.0,
        )
        if len(chunks) < 2:
            raise ValueError("each development work needs two included chunks")
        lengths[row["work_id"]] = len(chunks)
        texts.extend(chunks[:2])
        labels.extend([authors.index(row["author_id"])] * 2)
        groups.extend([row["work_id"]] * 2)
        prefix_tokens.append(len(nlp.make_doc(" ".join(chunks[:2]))))
    return np.asarray(texts, dtype=object), np.asarray(labels), np.asarray(groups), authors, lengths, prefix_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--text-root", type=pathlib.Path, required=True)
    parser.add_argument("--settings", type=pathlib.Path, default=pathlib.Path("configs/publication_comparison.json"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    paths = [pathlib.Path(primary.__file__), *sorted((repo / "src/stylo").rglob("*.py")), pathlib.Path(__file__)]
    hashes = {str(p.resolve().relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    settings = primary.load_strict(args.settings)
    cfg = primary.load_config(repo / "configs/default.yaml", overrides=settings["overrides"])
    catalog_bytes = args.catalog.read_bytes()
    started = time.monotonic()
    texts, y, groups, authors, lengths, token_counts = development_prefix(
        primary.loads_strict(catalog_bytes.decode("utf-8")), args.text_root, cfg,
    )
    records = {arm: [] for arm in ARMS}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for arm in ARMS:
            for work in lengths:
                train = groups != work
                factory = lambda: primary.make_estimator(arm, cfg, int(train.sum()), len(lengths) - 1, settings)
                row = primary.run_fold(texts, y, groups, len(authors), authors, work, factory, 1)
                if row is None:
                    raise ValueError("unevaluable development fold")
                records[arm].append({"work_id": work, "author": row["test_author"], "correct": row["correct"]})
            print(primary.json.dumps({"completed_arm": arm}), flush=True)
    grouped = {}
    for label, lower, upper in (("up_to_10", 0, 10), ("11_to_30", 11, 30), ("31_or_more", 31, float("inf"))):
        subset = {arm: [r for r in rows if lower <= lengths[r["work_id"]] <= upper] for arm, rows in records.items()}
        grouped[label] = {"works": len(subset[ARMS[0]]),
                          "correct": {arm: sum(r["correct"] for r in rows) for arm, rows in subset.items()},
                          "paired_against_A0": {arm: primary.paired_counts(subset[ARMS[0]], subset[arm]) for arm in ARMS[1:]}}
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.resolve().relative_to(repo))] for p in paths):
        raise ValueError("length-check source changed during execution")
    output = {"schema": "stylo.publication-length-check.v1", "design": "post-hoc development-only first-two-included-chunks sensitivity; no tuning or locked reading/scoring",
              "arms": list(ARMS), "fits": len(ARMS) * len(lengths), "works": len(lengths), "chunks": len(texts),
              "prefix_tokens_per_work": {"min": min(token_counts), "max": max(token_counts)},
              "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
              "settings_sha256": hashlib.sha256(args.settings.read_bytes()).hexdigest(),
              "settings": settings, "effective_config": cfg.to_dict(),
              "environment": {"python": primary.platform.python_version(),
                              **{name: primary.importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "spacy")}},
              "source_hashes": hashes, "source_sha256": primary.canonical_hash(hashes),
              "results": primary.summarize(records, authors), "by_original_chunk_count": grouped,
              "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
              "elapsed_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(primary.dumps_strict(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
