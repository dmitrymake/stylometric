#!/usr/bin/env python3
"""Fixed-parameter, whole-work comparison on an attested prose panel.

The locked test is excluded from every development fit. Outputs contain aggregates,
not literary text, chunk predictions, or work-level predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import time
import warnings
from collections import Counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from stylo.chunking import CombinedDoc, make_sent_chunks, sentences_for_text
from stylo.config import load_config
from stylo.domain.prediction_contract import stable_top1_and_worst_tie_rank
from stylo.domain.work_weighting import (
    CHUNK_WEIGHTED_LEGACY, RELATIVE_FW_ONLY_ABLATION, WORK_BALANCED,
)
from stylo.eval.dispatch import fit_estimator
from stylo.eval.lobo import _align_proba, _validate_proba, make_factory, run_fold
from stylo.jsonio import canonical_hash, dumps_strict, load_strict, loads_strict
from stylo.models.delta import BurrowsDelta
from stylo.nlp import load_sentencizer
from stylo.pipeline.clean import normalize
from stylo.vectorizer import StyloVectorizer

ARMS = (
    "stylo_A0_current", "stylo_A4_current", "stylo_A4_current_mass_control",
    "stylo_A0_topic_strict", "stylo_A4_topic_strict",
    "delta_300", "cosine_delta_300", "char_tfidf_lr",
)


class WholeWorkDelta(BurrowsDelta):
    """Apply the existing Delta transform to the concatenated included chunks.

    The evaluator passes chunks of one work at a time. Repeating its single score
    preserves the existing chunk-matrix contract without changing work weighting.
    """

    def predict_proba(self, texts):
        texts = list(texts)
        scores = super().predict_proba([" ".join(texts)])
        return np.repeat(scores, len(texts), axis=0)


def make_estimator(arm, cfg, n_train_chunks, n_train_works, settings):
    if arm not in ARMS:
        raise ValueError("unknown comparison arm")
    if arm.startswith("stylo_"):
        wb = "_A4_" in arm
        estimator = make_factory(
            "stylo", cfg, weighting=WORK_BALANCED if wb else CHUNK_WEIGHTED_LEGACY,
        )()
        if arm.endswith("topic_strict"):
            estimator.steps[0] = (
                "vectorizer", StyloVectorizer.from_config(cfg, topic_strict=True),
            )
        if arm.endswith("mass_control"):
            # C*W = C0*N: same average-loss regularization as A0, unchanged A4
            # relative weights, feature fit, scaler, solver and representation.
            estimator.named_steps["classifier"].set_params(
                C=cfg.get_path("model.classifier.C") * n_train_chunks / n_train_works,
            )
        return estimator
    if arm in {"delta_300", "cosine_delta_300"}:
        # Existing A3 route uses all analyzer tokens as denominator and train-only
        # pooled MFW selection; the frozen delta:N identifier uses selected mass.
        return WholeWorkDelta(
            settings["delta_mfw"], "cosine" if arm.startswith("cosine") else "manhattan",
            ablation=RELATIVE_FW_ONLY_ABLATION,
        )
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char", ngram_range=tuple(settings["char_ngram_range"]),
            max_features=settings["char_max_features"], min_df=settings["char_min_df"],
            sublinear_tf=True, lowercase=True,
        )),
        ("classifier", LogisticRegression(
            C=cfg.get_path("model.classifier.C"), solver="lbfgs",
            max_iter=cfg.get_path("model.classifier.max_iter"), class_weight="balanced",
        )),
    ])


def validate_catalog(catalog):
    works = catalog["works"]
    ids = [row["work_id"] for row in works]
    if not works or len(set(ids)) != len(ids):
        raise ValueError("catalog needs distinct works")
    for row in works:
        if row["role"] not in {"development", "locked_test"}:
            raise ValueError("role must be development or locked_test")
        parts = row["work_id"].split("/")
        if len(parts) != 2 or parts[0] != row["author_id"] or any(
            part in {"", ".", ".."} for part in parts
        ):
            raise ValueError("work_id must be author_id/slug")
    authors = sorted({row["author_id"] for row in works})
    if len(authors) < 2:
        raise ValueError("comparison needs at least two authors")
    dev = Counter(row["author_id"] for row in works if row["role"] == "development")
    locked = Counter(row["author_id"] for row in works if row["role"] == "locked_test")
    if any(dev[a] < 2 or locked[a] < 1 for a in authors):
        raise ValueError("each author needs >=2 development works and >=1 locked work")
    hashes = [row["text_sha256"] for row in works]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate input bytes cannot be independent works")
    return authors


def load_panel(catalog, text_root, cfg):
    authors = validate_catalog(catalog)
    nlp = load_sentencizer(cfg.get_path("language.code"))
    texts, labels, groups, roles, counts = [], [], [], [], []
    cleaned_hashes = set()
    for row in catalog["works"]:
        path = text_root / (row["work_id"] + ".txt")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["text_sha256"]:
            raise ValueError("input text checksum differs from catalog")
        cleaned = normalize(payload.decode("utf-8"), cfg.language.spacy_model,
                            cfg.language.spacy_fallback)
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if digest in cleaned_hashes:
            raise ValueError("duplicate normalized works")
        cleaned_hashes.add(digest)
        chunks = make_sent_chunks(
            CombinedDoc(sentences_for_text(cleaned, nlp)), cfg.chunking.chunk_size,
            cfg.chunking.min_words, overlap=0.0,
        )
        if not chunks:
            raise ValueError("a catalog work has no eligible chunks")
        texts.extend(chunks)
        labels.extend([authors.index(row["author_id"])] * len(chunks))
        groups.extend([row["work_id"]] * len(chunks))
        roles.extend([row["role"]] * len(chunks))
        counts.append({"author": row["author_id"], "role": row["role"],
                       "chunks": len(chunks), "tokens": len(nlp.make_doc(cleaned)),
                       "included_tokens": len(nlp.make_doc(" ".join(chunks)))})
    return (np.asarray(texts, dtype=object), np.asarray(labels),
            np.asarray(groups), np.asarray(roles), authors, counts)


def paired_counts(base, candidate):
    if [r["work_id"] for r in base] != [r["work_id"] for r in candidate]:
        raise ValueError("paired decisions must use identical ordered works")
    counts = Counter(
        "both_correct" if a["correct"] and b["correct"] else
        "gained" if b["correct"] else "lost" if a["correct"] else "both_wrong"
        for a, b in zip(base, candidate)
    )
    return {key: counts[key] for key in ("both_correct", "gained", "lost", "both_wrong")}


def summarize(records, authors):
    result = {}
    for arm, rows in records.items():
        per_author = {}
        for author in authors:
            selected = [row for row in rows if row["author"] == author]
            correct = sum(row["correct"] for row in selected)
            per_author[author] = {"correct": correct, "total": len(selected)}
        result[arm] = {
            "correct": sum(row["correct"] for row in rows), "total": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "macro_author_recall": float(np.mean([
                row["correct"] / row["total"] for row in per_author.values()
            ])), "per_author": per_author,
        }
    paired = {}
    for arm in records:
        if arm == "stylo_A0_current":
            continue
        paired[arm] = paired_counts(records["stylo_A0_current"], records[arm])
    summary = {"metrics": result, "paired_against_stylo_A0_current": paired}
    for key, candidate in (("mass_control_against_A4_current", "stylo_A4_current_mass_control"),
                           ("topic_strict_against_A4_current", "stylo_A4_topic_strict")):
        if "stylo_A4_current" in records and candidate in records:
            summary[key] = paired_counts(records["stylo_A4_current"], records[candidate])
    return summary


def evaluate_panel(texts, y, groups, roles, authors, cfg, settings, arms=ARMS):
    development = roles == "development"
    dev_texts, dev_y, dev_groups = texts[development], y[development], groups[development]
    dev_works = list(dict.fromkeys(dev_groups))
    locked_works = list(dict.fromkeys(groups[~development]))
    records = {split: {arm: [] for arm in arms} for split in ("development_lobo", "locked_test")}
    fit_details = []
    for arm in arms:
        for work in dev_works:
            train = dev_groups != work
            estimator_factory = lambda: make_estimator(
                arm, cfg, int(train.sum()), len(dev_works) - 1, settings,
            )
            row = run_fold(dev_texts, dev_y, dev_groups, len(authors), authors,
                           work, estimator_factory, 1)
            if row is None:
                raise ValueError("a whole-work development fold is not evaluable")
            records["development_lobo"][arm].append({
                "work_id": work, "author": row["test_author"], "correct": row["correct"],
            })
        estimator = make_estimator(arm, cfg, len(dev_texts), len(dev_works), settings)
        started = time.monotonic()
        fit_estimator(estimator, dev_texts, dev_y, dev_groups)
        fit_details.append({"arm": arm, "locked_training_chunks": len(dev_texts),
                            "locked_training_works": len(dev_works),
                            "locked_fit_seconds": time.monotonic() - started,
                            "C": (float(estimator.named_steps["classifier"].C)
                                  if hasattr(estimator, "named_steps") else None)})
        for work in locked_works:
            mask = groups == work
            probabilities = np.asarray(estimator.predict_proba(texts[mask]))
            _validate_proba(probabilities, estimator.classes_, len(authors), int(mask.sum()))
            scores = _align_proba(probabilities, estimator.classes_, len(authors))
            true = int(y[mask][0])
            decision = stable_top1_and_worst_tie_rank(scores, true_label=true,
                                                    expected_width=len(authors))
            records["locked_test"][arm].append({
                "work_id": work, "author": authors[true], "correct": decision.top1 == true,
            })
        print(json.dumps({"completed_arm": arm}), flush=True)
    return {split: summarize(rows, authors) for split, rows in records.items()}, fit_details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--text-root", type=pathlib.Path, required=True)
    parser.add_argument("--settings", type=pathlib.Path,
                        default=pathlib.Path("configs/publication_comparison.json"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    source_paths = [pathlib.Path(__file__).resolve(), *sorted((repo / "src/stylo").rglob("*.py"))]
    source_hashes = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    settings = load_strict(args.settings)
    cfg = load_config(pathlib.Path("configs/default.yaml"), overrides=settings["overrides"])
    catalog_bytes = args.catalog.read_bytes()
    catalog = loads_strict(catalog_bytes.decode("utf-8"))
    started = time.monotonic()
    panel = load_panel(catalog, args.text_root, cfg)
    texts, y, groups, roles, authors, counts = panel
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        results, fits = evaluate_panel(texts, y, groups, roles, authors, cfg, settings)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path.relative_to(repo))]
           for path in source_paths):
        raise ValueError("comparison source changed during execution")
    output = {
        "schema": "stylo.publication-comparison.v1",
        "design": {"arms": list(ARMS), "selection": "fixed parameters; no tuning",
                   "development": "whole-work LOBO", "test": "locked new-work same-platform test",
                   "test_fit": "development works only; one final fit per arm",
                   "work_score": {
                       "stylo_and_char": "mean chunk probability; stable top-1",
                       "delta": "Delta of concatenated included chunks; stable top-1",
                   },
                   "loss_mass_control": "A4 C*=C0*N_train/W_train",
                   "settings": settings, "effective_config": cfg.to_dict()},
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "source_sha256": canonical_hash(source_hashes),
        "source_hashes": source_hashes,
        "environment": {"python": platform.python_version(),
                        "thread_environment": {key: os.environ.get(key) for key in
                                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")},
                        **{name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "scikit-learn", "spacy", "ru-core-news-lg")}},
        "panel": {"authors": authors, "works": len(counts), "chunks": len(texts),
                  "per_author_role": [
                      {"author": author, "role": role,
                       "works": sum(c["author"] == author and c["role"] == role for c in counts),
                       "chunks": sum(c["chunks"] for c in counts if c["author"] == author and c["role"] == role),
                       "tokens": sum(c["tokens"] for c in counts if c["author"] == author and c["role"] == role),
                       "included_tokens": sum(c["included_tokens"] for c in counts if c["author"] == author and c["role"] == role)}
                      for author in authors for role in ("development", "locked_test")]},
        "results": results, "final_fits": fits,
        "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(dumps_strict(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "works": len(counts),
                      "chunks": len(texts), "warnings": len(caught)}))


if __name__ == "__main__":
    main()
