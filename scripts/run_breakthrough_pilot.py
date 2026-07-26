#!/usr/bin/env python3
"""Run the integration-only end-to-end breakthrough research pilot."""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from stylo.benchmarks import load_manifest, score_files, verify_manifest_artifacts
from stylo.eval.invariance import (
    align_purged_predictions,
    build_purged_factor_group_splits,
    evaluate_predictions,
)
from stylo.models.invariant import PairedInvariantAuthorshipModel
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402


def _read_documents(manifest, root: pathlib.Path):
    documents = [
        document
        for document in manifest.documents
        if document.split != "blind" and document.author_label is not None
    ]
    texts = np.asarray(
        [(root / str(document.text_path)).read_text(encoding="utf-8") for document in documents],
        dtype=object,
    )
    labels = np.asarray([document.author_label for document in documents], dtype=object)
    metadata = {
        "author": labels.copy(),
        "work": np.asarray([document.work for document in documents], dtype=object),
        "source": np.asarray([document.source.source_id for document in documents], dtype=object),
        "edition": np.asarray([document.edition for document in documents], dtype=object),
        "period": np.asarray([document.period for document in documents], dtype=object),
        "genre": np.asarray([document.genre for document in documents], dtype=object),
        "topic": np.asarray([document.topic for document in documents], dtype=object),
        "register": np.asarray([document.register for document in documents], dtype=object),
    }
    return texts, labels, metadata


def _baseline_predict(train_texts, train_y, test_texts):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=8_000,
        sublinear_tf=True,
    )
    Xtr = vectorizer.fit_transform(train_texts)
    Xte = vectorizer.transform(test_texts)
    model = LogisticRegression(
        max_iter=2_000, class_weight="balanced", random_state=42
    ).fit(Xtr, train_y)
    return model.predict(Xte)


def _evaluate_factor(texts, labels, metadata, factor: str, bootstrap_iters: int):
    plan = build_purged_factor_group_splits(
        metadata,
        factor,
        labels,
        group_field="work",
        author_field="author",
    )
    baseline_cells = {}
    invariant_cells = {}
    residualizer_diagnostics = []
    failures = []
    for split in plan.splits:
        if not split.diagnostics.possible:
            failures.append(
                {
                    "cell": [split.level, split.group],
                    "reasons": list(split.diagnostics.reasons),
                }
            )
            continue
        train, test = split.train_idx, split.test_idx
        key = (split.level, split.group)
        baseline_cells[key] = _baseline_predict(texts[train], labels[train], texts[test])
        nuisance = np.asarray(
            [f"{metadata['source'][i]}|{metadata['edition'][i]}" for i in train],
            dtype=object,
        )
        model = PairedInvariantAuthorshipModel(
            min_df=1,
            max_features=8_000,
            embedding_dim=32,
            variance_threshold=0.90,
            max_nuisance_rank=8,
            random_state=42,
        ).fit(
            texts[train],
            labels[train],
            work_ids=metadata["work"][train],
            nuisance_ids=nuisance,
        )
        invariant_cells[key] = model.predict(texts[test])
        residualizer_diagnostics.append(dataclasses.asdict(model.diagnostics_))

    baseline_pred = align_purged_predictions(
        plan, baseline_cells, require_all_possible=True
    )
    invariant_pred = align_purged_predictions(
        plan, invariant_cells, require_all_possible=True
    )
    baseline_report = evaluate_predictions(
        labels,
        baseline_pred,
        metadata,
        factors=(factor,),
        bootstrap_iters=bootstrap_iters,
        seed=42,
    )
    invariant_report = evaluate_predictions(
        labels,
        invariant_pred,
        metadata,
        factors=(factor,),
        bootstrap_iters=bootstrap_iters,
        seed=42,
    )
    return {
        "plan": dataclasses.asdict(plan.diagnostics),
        "failures": failures,
        "baseline_char_lr": baseline_report.to_dict(),
        "paired_edition_residualizer": invariant_report.to_dict(),
        "residualizer_fits": residualizer_diagnostics,
    }


def run(package: pathlib.Path, bootstrap_iters: int = 100) -> dict:
    manifest_path = package / "manifest.json"
    manifest = load_manifest(manifest_path)
    artifacts = verify_manifest_artifacts(manifest, package)
    reference = score_files(
        manifest,
        manifest_path,
        package / "truth.synthetic-public.json",
        package / "submission.reference.json",
        artifact_root=package,
        synthetic_integration_only=True,
        segmentation_bootstrap_unit="document",
        bootstrap_iters=bootstrap_iters,
        seed=42,
    )
    texts, labels, metadata = _read_documents(manifest, package)
    factors = {
        factor: _evaluate_factor(texts, labels, metadata, factor, bootstrap_iters)
        for factor in ("source", "edition")
    }
    return {
        "status": "integration_control_only",
        "scientific_claim_allowed": False,
        "dataset": dataclasses.asdict(manifest.dataset),
        "artifact_report": artifacts.to_dict(),
        "reference_blind_score": reference.to_dict(),
        "purged_factor_experiments": factors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = run(pathlib.Path(args.package), bootstrap_iters=args.bootstrap_iters)
    rendered = dumps_strict(report, ensure_ascii=False, indent=2)
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
