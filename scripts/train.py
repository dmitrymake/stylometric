from __future__ import annotations

import argparse
import logging
import os
import pathlib
from typing import Dict, List, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("=== Train (Unified Pipeline: StyloVectorizer + LR + Delta assets) ===")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train stylometry models (unified pipeline)."
    )
    p.add_argument("--lang", default="ru", help="Language code: ru|en|fr")
    p.add_argument(
        "--train-root", default="data/frags_train", help="Path to train fragments"
    )
    p.add_argument("--outdir", default="data", help="Artifacts output dir")
    return p.parse_args()


def iter_train_files(train_root: pathlib.Path) -> List[pathlib.Path]:
    # Expected layout:
    # data/frags_train/{author}/{book_id}/{chunk_file}.txt
    return sorted([p for p in train_root.rglob("*.txt") if p.is_file()])


def infer_author_book(fp: pathlib.Path, train_root: pathlib.Path) -> Tuple[str, str]:
    """
    Infer (author, book_id) from path.
    Expected: train_root/author/book_id/chunk.txt
    Fallbacks are handled but you should keep the canonical structure.
    """
    rel = fp.relative_to(train_root)
    parts = rel.parts

    if len(parts) >= 3:
        author = parts[0]
        book_id = parts[1]
        return author, book_id

    # Fallback: train_root/author/chunk.txt
    if len(parts) == 2:
        author = parts[0]
        book_id = fp.stem
        return author, book_id

    # Very rare fallback
    return "unknown", fp.stem


def main() -> None:
    args = parse_args()

    # IMPORTANT: set language before importing meta/meta-dependent modules
    os.environ["STYLO_LANG"] = args.lang

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_root = pathlib.Path(args.train_root)
    if not train_root.exists():
        logging.error(f"Train directory not found: {train_root}")
        raise SystemExit(1)

    # Robust import (works both with PYTHONPATH=./scripts and as package)
    try:
        from syntax_features import StyloVectorizer
    except ImportError:
        from scripts.syntax_features import StyloVectorizer  # type: ignore

    files = iter_train_files(train_root)
    if not files:
        logging.error(f"No .txt files found in {train_root}")
        raise SystemExit(1)

    # Collect authors from directory names to ensure stable indexing
    authors = sorted(
        [p.name for p in train_root.iterdir() if p.is_dir() and p.name != "unknown"]
    )
    if len(authors) < 2:
        logging.error("Need at least 2 authors in train set.")
        raise SystemExit(1)

    auth2idx: Dict[str, int] = {a: i for i, a in enumerate(authors)}

    texts: List[str] = []
    labels: List[int] = []
    groups: List[str] = []

    logging.info(f"Loading train fragments ({len(files)} files)...")
    for fp in files:
        try:
            txt = fp.read_text(encoding="utf-8").strip()
        except Exception as e:
            logging.warning(f"Read error: {fp} ({e})")
            continue

        if not txt:
            continue

        author, book_id = infer_author_book(fp, train_root)
        if author not in auth2idx:
            # Skip unknown or non-listed authors from train
            continue

        texts.append(txt)
        labels.append(auth2idx[author])
        groups.append(f"{author}/{book_id}")

    if len(texts) < 10:
        logging.error("Too few training samples after filtering.")
        raise SystemExit(1)

    y = np.asarray(labels, dtype=int)
    groups_arr = np.asarray(groups, dtype=object)

    # Canonical vectorizer (single source of truth).
    # SSA + Vowel Rhythm are already included inside SyntaxFeatureExtractor
    # (see syntax_features.py) -> no extra manual feature concat here.
    VECTORIZER_PARAMS = dict(
        char_ngram_range=(3, 5),
        max_char_features=5000,
        char_min_df=3,
        use_char=True,
        use_func=True,
        use_mfw=True,
        mfw_count=300,
        use_syntax=True,
        auto_bleach=True,
    )

    vec = StyloVectorizer(**VECTORIZER_PARAMS)

    # MaxAbsScaler is used to stay sparse-friendly and stable.
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
    )

    pipe = Pipeline(
        steps=[
            ("vectorizer", vec),
            ("scaler", MaxAbsScaler()),
            ("classifier", clf),
        ]
    )

    logging.info("Fitting main pipeline (vectorizer + LR)...")
    pipe.fit(texts, y)

    # Save main model
    joblib.dump(pipe, outdir / "model.pkl")
    logging.info(f"Saved: {outdir / 'model.pkl'}")

    # Also save fitted vectorizer separately (for backward compatibility of other scripts)
    joblib.dump(vec, outdir / "vectorizer.pkl")
    joblib.dump(vec, outdir / "vectorizer_fitted.pkl")
    logging.info(f"Saved: {outdir / 'vectorizer.pkl'} and vectorizer_fitted.pkl")

    # IMPORTANT: keep sparse matrix (do NOT toarray()).
    logging.info("Transforming train set to feature matrix...")
    X = vec.transform(texts)
    if not issparse(X):
        # should not happen, but keep safe
        X = csr_matrix(X)

    joblib.dump(
        {
            "X_transformed": X,  # SPARSE CSR
            "labels": y,
            "groups": groups_arr,
            "authors": authors,
        },
        outdir / "train_vectors.pkl",
    )
    logging.info(f"Saved: {outdir / 'train_vectors.pkl'} (sparse)")

    # Delta assets: distances want z-scored features.
    # StandardScaler(with_mean=False) supports sparse.
    logging.info("Fitting Delta scaler (Z-score, sparse-safe)...")
    delta_scaler = StandardScaler(with_mean=False)
    X_z = delta_scaler.fit_transform(X)

    joblib.dump(delta_scaler, outdir / "scaler_delta.pkl")
    logging.info(f"Saved: {outdir / 'scaler_delta.pkl'}")

    # Centroids (authors x features) computed in Z-space
    logging.info("Computing centroids in Z-space...")
    n_auth = len(authors)
    centroids: List[np.ndarray] = []

    for lbl in range(n_auth):
        mask = y == lbl
        if not np.any(mask):
            logging.warning(
                f"Author '{authors[lbl]}' has 0 samples, centroid will be zeros."
            )
            c = np.zeros(X_z.shape[1], dtype=np.float32)
            centroids.append(c)
            continue

        work_means = []
        for group in dict.fromkeys(groups_arr[mask].tolist()):
            work_mean = X_z[mask & (groups_arr == group)].mean(axis=0)
            if issparse(work_mean):
                work_mean = work_mean.toarray()
            work_mean = np.asarray(work_mean).ravel()
            work_means.append(work_mean / (np.linalg.norm(work_mean) + 1e-12))
        c = np.mean(work_means, axis=0).astype(np.float32)
        centroids.append(c)

    centroids_arr = np.vstack(centroids)
    np.save(outdir / "centroids.npy", centroids_arr)
    np.save(outdir / "authors.npy", np.asarray(authors, dtype=object))
    logging.info(f"Saved: {outdir / 'centroids.npy'} shape={centroids_arr.shape}")
    logging.info(f"Saved: {outdir / 'authors.npy'}")

    logging.info("Training complete.")


if __name__ == "__main__":
    main()
