"""SOCIOLIT-lite benchmark on public Russian term-frequency matrices.

This does not run the full Stylo pipeline: the SOCIOLIT full texts are not
publicly redistributed. It evaluates reproducible lexical baselines on the
public `TfMatrixRU.json` and `authorsRU.json` artifacts from:
https://github.com/DDPronin/Rank-Turbulence-Delta
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time
import urllib.request
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import LinearSVC


ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
DATA_DIR = ROOT / "data" / "external" / "sociolit"
DEFAULT_TF = DATA_DIR / "TfMatrixRU.json"
DEFAULT_AUTHORS = DATA_DIR / "authorsRU.json"
DEFAULT_OUT = ROOT / "docs" / "sociolit_tf.json"
DEFAULT_PRED = ROOT / "docs" / "sociolit_tf_predictions.csv"

URLS = {
    "tf": "https://media.githubusercontent.com/media/DDPronin/Rank-Turbulence-Delta/main/data/TfMatrixRU.json",
    "authors": "https://raw.githubusercontent.com/DDPronin/Rank-Turbulence-Delta/main/data/authorsRU.json",
}

SEED = 42


def log(*args: object) -> None:
    print(*args, flush=True)


def download(url: str, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    log(f"download {url} -> {path}")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(path)


def ensure_inputs(tf_path: pathlib.Path, authors_path: pathlib.Path, do_download: bool) -> None:
    if do_download:
        if not tf_path.exists():
            download(URLS["tf"], tf_path)
        if not authors_path.exists():
            download(URLS["authors"], authors_path)
    missing = [str(p) for p in (tf_path, authors_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SOCIOLIT-lite input files: "
            + ", ".join(missing)
            + ". Run with --download or fetch them with curl."
        )
    with tf_path.open("r", encoding="utf-8", errors="ignore") as fh:
        head = fh.read(80)
    if head.startswith("version https://git-lfs"):
        raise ValueError(
            f"{tf_path} is a Git LFS pointer, not the matrix. Download from {URLS['tf']}."
        )


def load_authors(path: pathlib.Path) -> list[str]:
    raw = json.loads(path.read_text("utf-8"))
    if isinstance(raw, dict):
        return [raw[str(i)] for i in range(len(raw))]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise TypeError(f"Unsupported authorsRU.json shape: {type(raw).__name__}")


def load_matrix(path: pathlib.Path, max_features: int) -> tuple[np.ndarray, list[str], list[str]]:
    log(f"loading matrix: {path}")
    df = pd.read_json(path)
    if "Word" not in df.columns:
        raise ValueError("TfMatrixRU.json must contain a Word column")
    n_features = min(max_features, len(df))
    feature_df = df.iloc[:n_features].copy()
    feature_names = feature_df["Word"].astype(str).tolist()
    doc_cols = [c for c in feature_df.columns if c not in {"Count", "Word"}]
    X = feature_df[doc_cols].T.to_numpy(dtype=np.float32, copy=True)
    return X, [str(c) for c in doc_cols], feature_names


def zscore_global(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, dtype=np.float64)
    std = X.std(axis=0, dtype=np.float64)
    std[std < 1e-12] = 1.0
    return ((X - mean) / std).astype(np.float32, copy=False)


def nn_cosine_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1)
    norms[norms < 1e-12] = 1.0
    Xn = X / norms[:, None]
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    return y[np.argmax(sim, axis=1)]


def nn_l1_predict(X: np.ndarray, y: np.ndarray, n_jobs: int | None) -> np.ndarray:
    dist = pairwise_distances(X, metric="manhattan", n_jobs=n_jobs)
    np.fill_diagonal(dist, np.inf)
    return y[np.argmin(dist, axis=1)]


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    recalls = []
    for lab in labels:
        mask = y_true == lab
        recalls.append(float(np.mean(y_pred[mask] == lab)))
    recalls_arr = np.asarray(recalls)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "zero_recall_authors": int(np.sum(recalls_arr == 0.0)),
        "median_author_recall": round(float(np.median(recalls_arr)), 6),
        "min_author_recall": round(float(np.min(recalls_arr)), 6),
    }


def run_linear_svm(X: np.ndarray, y: np.ndarray, requested_folds: int) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.bincount(y)
    min_count = int(counts[counts > 0].min())
    if requested_folds <= 1:
        cv: Any = LeaveOneOut()
        protocol = "leave-one-out"
        actual_folds = len(y)
    else:
        actual_folds = min(requested_folds, min_count)
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
        protocol = f"stratified-{actual_folds}-fold"
    clf = make_pipeline(
        MaxAbsScaler(),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=SEED),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    meta = {
        "protocol": protocol,
        "requested_folds": requested_folds,
        "actual_folds": actual_folds,
        "note": "Feature matrix and MFW selection are public derived SOCIOLIT artifacts; this is not a raw-text full-pipeline run.",
    }
    return pred, meta


def top_error_pairs(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], limit: int = 15) -> list[dict[str, Any]]:
    pairs = Counter((int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p)
    out = []
    for (true_i, pred_i), n in pairs.most_common(limit):
        out.append({"true": classes[true_i], "pred": classes[pred_i], "n": int(n)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=pathlib.Path, default=DEFAULT_TF)
    ap.add_argument("--authors", type=pathlib.Path, default=DEFAULT_AUTHORS)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--predictions", type=pathlib.Path, default=DEFAULT_PRED)
    ap.add_argument("--max-features", type=int, default=10_000)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--skip-l1", action="store_true", help="skip Manhattan distance; cosine is much faster")
    ap.add_argument("--skip-svm", action="store_true")
    ap.add_argument("--svm-folds", type=int, default=3, help="<=1 means leave-one-out; default is 3")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    ensure_inputs(args.tf, args.authors, args.download)
    X, doc_ids, feature_names = load_matrix(args.tf, args.max_features)
    author_names = load_authors(args.authors)
    if len(author_names) != X.shape[0]:
        raise ValueError(f"authors length {len(author_names)} != matrix documents {X.shape[0]}")

    enc = LabelEncoder()
    y = enc.fit_transform(author_names)
    classes = [str(c) for c in enc.classes_]
    labels = np.arange(len(classes))
    counts = Counter(author_names)

    log(
        f"SOCIOLIT-lite: docs={X.shape[0]} authors={len(classes)} "
        f"features={X.shape[1]} min_docs/author={min(counts.values())}"
    )
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}

    log("running raw relative-frequency cosine NN")
    pred = nn_cosine_predict(X, y)
    predictions["nn_raw_cosine"] = pred
    results["nn_raw_cosine"] = metric_summary(y, pred, labels)

    log("running global-z cosine NN (paper-style/transductive)")
    Z = zscore_global(X)
    pred = nn_cosine_predict(Z, y)
    predictions["nn_global_z_cosine"] = pred
    results["nn_global_z_cosine"] = metric_summary(y, pred, labels)

    if not args.skip_l1:
        log("running global-z L1 NN (paper-style/transductive)")
        pred = nn_l1_predict(Z, y, args.n_jobs)
        predictions["nn_global_z_l1"] = pred
        results["nn_global_z_l1"] = metric_summary(y, pred, labels)

    svm_meta = None
    if not args.skip_svm:
        log(f"running LinearSVC CV (requested folds={args.svm_folds})")
        pred, svm_meta = run_linear_svm(X, y, args.svm_folds)
        predictions["linear_svm_tf"] = pred
        results["linear_svm_tf"] = metric_summary(y, pred, labels)
        results["linear_svm_tf"]["cv"] = svm_meta

    best_key = max(results, key=lambda k: results[k]["balanced_accuracy"])
    report = {
        "dataset": "SOCIOLIT-lite public TF matrix (Russian)",
        "source": {
            "repository": "https://github.com/DDPronin/Rank-Turbulence-Delta",
            "paper": "https://arxiv.org/abs/2604.19499",
            "tf_matrix": URLS["tf"],
            "authors": URLS["authors"],
        },
        "protocol": {
            "kind": "lexical/TF benchmark on public derived artifacts",
            "not_full_stylo_pipeline": True,
            "documents": int(X.shape[0]),
            "authors": int(len(classes)),
            "features_used": int(X.shape[1]),
            "feature_order": "first rows of TfMatrixRU.json, sorted by corpus Count in the source notebook",
            "top_features_preview": feature_names[:20],
            "labels": "authorsRU.json order aligned to matrix document columns",
            "primary_metric": "balanced_accuracy",
            "caveats": [
                "Full SOCIOLIT texts are not publicly redistributed, so syntax/morphology/cleaning channels are not evaluated here.",
                "Global z-score nearest-neighbor metrics are marked paper-style/transductive because standardization sees the full matrix.",
                "The supervised SVM uses public precomputed feature selection from the released matrix; treat it as SOCIOLIT-lite, not a strict raw-text benchmark.",
            ],
            "svm": svm_meta,
        },
        "results": results,
        "best_by_balanced_accuracy": {"method": best_key, **results[best_key]},
        "top_error_pairs": {
            method: top_error_pairs(y, pred, classes) for method, pred in predictions.items()
        },
        "runtime_sec": round(float(time.time() - t0), 3),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_strict(report, ensure_ascii=False, indent=2), "utf-8")

    pred_rows = {"doc_id": doc_ids, "author": author_names}
    for method, pred in predictions.items():
        pred_rows[f"pred_{method}"] = [classes[int(i)] for i in pred]
        pred_rows[f"correct_{method}"] = (pred == y).tolist()
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(args.predictions, index=False)

    log("\n%-24s %8s %9s %8s %5s" % ("method", "acc", "bal_acc", "macroF1", "zero"))
    for method, metrics in sorted(results.items(), key=lambda kv: kv[1]["balanced_accuracy"], reverse=True):
        log(
            "%-24s %8.3f %9.3f %8.3f %5d"
            % (
                method,
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
                metrics["zero_recall_authors"],
            )
        )
    log(f"\nsaved {args.out.relative_to(ROOT)}")
    log(f"saved {args.predictions.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
