#!/usr/bin/env python3
"""Exploratory real-text edition-invariance pilot with work-purged evaluation."""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from stylo.benchmarks import load_manifest, verify_manifest_artifacts
from stylo.eval.invariance import (
    align_purged_predictions,
    build_purged_factor_group_splits,
    evaluate_predictions,
)
from stylo.models.invariant import (
    PairedEditionResidualizer,
    PairedInvariantAuthorshipModel,
)
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402


def _load(panel: pathlib.Path):
    manifest = load_manifest(panel / "manifest.json")
    artifacts = verify_manifest_artifacts(manifest, panel)
    documents = [document for document in manifest.documents if document.author_label]
    texts = np.asarray(
        [(panel / str(document.text_path)).read_text(encoding="utf-8") for document in documents],
        dtype=object,
    )
    labels = np.asarray([document.author_label for document in documents], dtype=object)
    metadata = {
        "author": labels.copy(),
        "work": np.asarray([document.work for document in documents], dtype=object),
        "source": np.asarray([document.source.source_id for document in documents], dtype=object),
        "edition": np.asarray([document.edition for document in documents], dtype=object),
        "period": np.asarray([document.period for document in documents], dtype=object),
    }
    return manifest, artifacts, texts, labels, metadata


def _classifier(seed=42):
    return LogisticRegression(
        max_iter=2_000, class_weight="balanced", random_state=seed
    )


def _scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def identifiability_audit(labels, metadata):
    """Report factors that deterministically encode the nominal author label."""
    authors = np.asarray(labels, dtype=object)
    factors = {}
    for name in ("period", "source", "edition"):
        if name not in metadata:
            continue
        values = np.asarray(metadata[name], dtype=object)
        observed = [value for value in values if value not in (None, "")]
        if len(observed) != len(values):
            factors[name] = {
                "complete": False,
                "author_determined_by_factor": False,
                "factor_determined_by_author": False,
                "perfect_confound": False,
            }
            continue
        by_factor = {
            str(value): sorted({str(author) for author in authors[values == value]})
            for value in sorted(set(values), key=str)
        }
        by_author = {
            str(author): sorted({str(value) for value in values[authors == author]})
            for author in sorted(set(authors), key=str)
        }
        factor_to_author = all(len(group) == 1 for group in by_factor.values())
        author_to_factor = all(len(group) == 1 for group in by_author.values())
        factors[name] = {
            "complete": True,
            "author_determined_by_factor": factor_to_author,
            "factor_determined_by_author": author_to_factor,
            "perfect_confound": factor_to_author and author_to_factor,
            "authors_by_factor": by_factor,
            "factors_by_author": by_author,
        }
    return {
        "factors": factors,
        "perfect_author_confounds": sorted(
            name for name, report in factors.items() if report["perfect_confound"]
        ),
    }


def work_holdout_representation_probe(texts, labels, metadata):
    """Measure author retention and edition recoverability on unseen works."""
    baseline_author = np.empty(len(texts), dtype=object)
    baseline_edition = np.empty(len(texts), dtype=object)
    embedded_author = np.empty(len(texts), dtype=object)
    embedded_edition = np.empty(len(texts), dtype=object)
    invariant_author = np.empty(len(texts), dtype=object)
    invariant_edition = np.empty(len(texts), dtype=object)
    diagnostics = []
    for fold_no, work in enumerate(sorted(set(metadata["work"]))):
        test = np.flatnonzero(metadata["work"] == work)
        train = np.flatnonzero(metadata["work"] != work)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=20_000,
            sublinear_tf=True,
        )
        Xtr = vectorizer.fit_transform(texts[train])
        Xte = vectorizer.transform(texts[test])
        baseline_author[test] = _classifier(100 + fold_no).fit(
            Xtr, labels[train]
        ).predict(Xte)
        baseline_edition[test] = _classifier(200 + fold_no).fit(
            Xtr, metadata["edition"][train]
        ).predict(Xte)

        residualizer = PairedEditionResidualizer(
            embedding_dim=64,
            variance_threshold=0.90,
            max_nuisance_rank=8,
            random_state=300 + fold_no,
        ).fit(Xtr, metadata["work"][train], metadata["edition"][train])
        Etr = residualizer.embedding_transform(Xtr)
        Ete = residualizer.embedding_transform(Xte)
        Ztr = residualizer.transform(Xtr)
        Zte = residualizer.transform(Xte)
        embedded_author[test] = _classifier(350 + fold_no).fit(
            Etr, labels[train]
        ).predict(Ete)
        embedded_edition[test] = _classifier(375 + fold_no).fit(
            Etr, metadata["edition"][train]
        ).predict(Ete)
        invariant_author[test] = _classifier(400 + fold_no).fit(
            Ztr, labels[train]
        ).predict(Zte)
        invariant_edition[test] = _classifier(500 + fold_no).fit(
            Ztr, metadata["edition"][train]
        ).predict(Zte)
        diagnostics.append(dataclasses.asdict(residualizer.diagnostics_))

    return {
        "unit": "leave_one_work_out",
        "n_works": len(set(metadata["work"])),
        "baseline": {
            "author": _scores(labels, baseline_author),
            "edition_probe": _scores(metadata["edition"], baseline_edition),
        },
        "svd_only_control": {
            "author": _scores(labels, embedded_author),
            "edition_probe": _scores(metadata["edition"], embedded_edition),
        },
        "paired_residualizer": {
            "author": _scores(labels, invariant_author),
            "edition_probe": _scores(metadata["edition"], invariant_edition),
        },
        "fold_diagnostics": diagnostics,
    }


def purged_edition_attribution(texts, labels, metadata, bootstrap_iters=200):
    plan = build_purged_factor_group_splits(
        metadata, "edition", labels, group_field="work", author_field="author"
    )
    baseline_cells = {}
    embedded_cells = {}
    invariant_cells = {}
    fit_diagnostics = []
    for fold_no, split in enumerate(plan.splits):
        if not split.diagnostics.possible:
            continue
        train, test = split.train_idx, split.test_idx
        key = (split.level, split.group)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=20_000,
            sublinear_tf=True,
        )
        Xtr = vectorizer.fit_transform(texts[train])
        baseline_cells[key] = _classifier(600 + fold_no).fit(
            Xtr, labels[train]
        ).predict(vectorizer.transform(texts[test]))

        model = PairedInvariantAuthorshipModel(
            min_df=1,
            max_features=20_000,
            embedding_dim=64,
            variance_threshold=0.90,
            max_nuisance_rank=8,
            random_state=700 + fold_no,
        ).fit(
            texts[train],
            labels[train],
            work_ids=metadata["work"][train],
            nuisance_ids=metadata["edition"][train],
        )
        embedded_train = model.residualizer_.embedding_transform(
            model.vectorizer_.transform(texts[train])
        )
        embedded_test = model.residualizer_.embedding_transform(
            model.vectorizer_.transform(texts[test])
        )
        embedded_cells[key] = _classifier(650 + fold_no).fit(
            embedded_train, labels[train]
        ).predict(embedded_test)
        invariant_cells[key] = model.predict(texts[test])
        fit_diagnostics.append(dataclasses.asdict(model.diagnostics_))

    baseline = align_purged_predictions(plan, baseline_cells)
    embedded = align_purged_predictions(plan, embedded_cells)
    invariant = align_purged_predictions(plan, invariant_cells)

    def score_purged(predictions):
        # These predictions come from factor×work-purged cell fits.  Ordinary
        # leave-one-edition diagnostics describe different train/test sets, so
        # only metrics are computed here; the actual cell geometry is reported
        # from ``plan`` below.
        report = evaluate_predictions(
            labels,
            predictions,
            metadata,
            factors=(),
            bootstrap_iters=bootstrap_iters,
            seed=42,
        ).to_dict()
        by_edition = {}
        for level in sorted(set(metadata["edition"])):
            idx = np.flatnonzero(metadata["edition"] == level)
            subset_metadata = {key: values[idx] for key, values in metadata.items()}
            by_edition[str(level)] = evaluate_predictions(
                labels[idx],
                predictions[idx],
                subset_metadata,
                factors=(),
                bootstrap_iters=bootstrap_iters,
                seed=42,
            ).to_dict()["overall"]
        report["by_edition"] = by_edition
        report["worst_edition_accuracy"] = min(
            row["accuracy"]["point"] for row in by_edition.values()
        )
        return report

    possible = [split for split in plan.splits if split.diagnostics.possible]
    if not possible:
        raise ValueError("edition×work plan has no feasible purged cells")
    return {
        "plan": dataclasses.asdict(plan.diagnostics),
        "actual_cell_fit_sizes": {
            "min_train": min(len(split.train_idx) for split in possible),
            "max_train": max(len(split.train_idx) for split in possible),
            "min_test": min(len(split.test_idx) for split in possible),
            "max_test": max(len(split.test_idx) for split in possible),
        },
        "baseline_char_lr": score_purged(baseline),
        "svd_only_control": score_purged(embedded),
        "paired_residualizer": score_purged(invariant),
        "fit_diagnostics": fit_diagnostics,
    }


def run(panel: pathlib.Path, bootstrap_iters=200) -> dict:
    manifest, artifacts, texts, labels, metadata = _load(panel)
    return {
        "status": "exploratory_internal_real_text_pilot",
        "confirmatory_claim_allowed": False,
        "limitations": [
            "two authors only",
            "author is perfectly confounded with period: Chekhov=1890s and Gogol=1830s",
            "local_clean upstream provenance unknown",
            "Wikisource main/version2 plus local are not a fully crossed source×edition design",
            "edition labels represent acquisition/version pipelines; source and edition effects are not causally separable",
            "aligned passages were selected using all three unlabeled realizations and exclude rewritten content",
            "bootstrap intervals are descriptive over six selected works, not population-level inference",
            "there is no untouched validation split in this feasibility panel",
            "model and protocol were developed before this result",
        ],
        "dataset": dataclasses.asdict(manifest.dataset),
        "n_documents": len(texts),
        "n_benchmark_tokens": sum(
            document.n_tokens for document in artifacts.documents
        ),
        "identifiability_audit": identifiability_audit(labels, metadata),
        "work_holdout_representation_probe": work_holdout_representation_probe(
            texts, labels, metadata
        ),
        "purged_edition_work_attribution": purged_edition_attribution(
            texts, labels, metadata, bootstrap_iters=bootstrap_iters
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = run(pathlib.Path(args.panel), args.bootstrap_iters)
    rendered = dumps_strict(report, ensure_ascii=False, indent=2)
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
