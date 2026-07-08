"""
LOBO-валидация (Leave-One-Book-Out) — leakage-free.

Гарантии честности:
1) Векторизатор (StyloVectorizer) обучается ТОЛЬКО на train-части каждого фолда.
   Иначе возникает утечка (IDF/max_features/min_df и т.п. «подсматривают» тестовую книгу).
2) LR и Delta считаются в одном и том же feature-space, созданном на train-фолде.
3) Группы (books) берутся из структуры data/frags_train/{author}/{book_id}/*.
4) STYLO_LANG ставится до импортов meta (--lang).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("=== LOBO VALIDATION (Leakage-Free, Unified) ===")

# Ограничиваем потоки numpy/BLAS, чтобы не конфликтовать с joblib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free LOBO validation")
    p.add_argument("--lang", default="ru", help="Language code: ru|en|fr")
    p.add_argument(
        "--train-root", default="data/frags_train", help="Train fragments root"
    )
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs")
    p.add_argument(
        "--max-books", type=int, default=0, help="Limit number of books (0=all)"
    )
    p.add_argument("--outdir", default="docs", help="Where to write reports")
    return p.parse_args()


def iter_train_fragments(train_root: pathlib.Path) -> List[pathlib.Path]:
    return sorted([p for p in train_root.rglob("*.txt") if p.is_file()])


def infer_author_book(fp: pathlib.Path, train_root: pathlib.Path) -> Tuple[str, str]:
    """
    Expected layout:
      train_root/author/book_id/chunk.txt
    """
    rel = fp.relative_to(train_root)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return parts[0], fp.stem
    return "unknown", fp.stem


def load_dataset(
    train_root: pathlib.Path, authors: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    auth2idx = {a: i for i, a in enumerate(authors)}

    texts: List[str] = []
    y: List[int] = []
    groups: List[str] = []

    files = iter_train_fragments(train_root)
    for fp in files:
        try:
            txt = fp.read_text(encoding="utf-8").strip()
        except Exception:
            continue

        if not txt:
            continue

        author, book_id = infer_author_book(fp, train_root)
        if author not in auth2idx:
            continue

        texts.append(txt)
        y.append(auth2idx[author])
        groups.append(f"{author}/{book_id}")

    return (
        np.asarray(texts, dtype=object),
        np.asarray(y, dtype=int),
        np.asarray(groups, dtype=object),
    )


def create_book_groups(labels: np.ndarray, groups: np.ndarray) -> Dict[str, List[int]]:
    bg: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        bg[str(g)].append(i)
    return bg


def compute_centroids_sparse_mean(
    X, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centroids for each class label present in y_train.
    Returns:
      centroids: (n_auth_in_train, n_features) dense
      centroid_labels: (n_auth_in_train,)
    """
    unique = np.unique(y_train)
    centroids: List[np.ndarray] = []
    centroid_labels: List[int] = []

    for lbl in unique:
        mask = y_train == lbl
        Xm = X[mask]
        c = Xm.mean(axis=0)
        if issparse(c):
            c = c.toarray()
        c = np.asarray(c).ravel()
        centroids.append(c)
        centroid_labels.append(int(lbl))

    return np.vstack(centroids), np.asarray(centroid_labels, dtype=int)


def run_fold(
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    book_groups: Dict[str, List[int]],
    authors: List[str],
    test_group_id: str,
    vectorizer_params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    test_indices = book_groups[test_group_id]

    mask_test = np.zeros(labels.shape[0], dtype=bool)
    mask_test[test_indices] = True
    mask_train = ~mask_test

    y_train = labels[mask_train]
    y_test = labels[mask_test]
    test_auth_idx = int(y_test[0])

    # Если у автора всего 1 книга, LOBO невозможен (он пропадёт из train)
    if test_auth_idx not in y_train:
        return None

    # leakage-free vectorizer: fit только внутри train-фолда
    try:
        from syntax_features import StyloVectorizer
    except ImportError:
        from scripts.syntax_features import StyloVectorizer  # type: ignore

    vec = StyloVectorizer(**vectorizer_params)

    X_train = vec.fit_transform(texts[mask_train])
    X_test = vec.transform(texts[mask_test])

    # METHOD 1: Logistic Regression
    lr = make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_test)
    mean_prob = probs.mean(axis=0)

    sorted_idx_lr = np.argsort(mean_prob)[::-1]
    top1_lr = int(sorted_idx_lr[0])
    rank_lr = int(np.where(sorted_idx_lr == test_auth_idx)[0][0] + 1)

    # METHOD 2: Cosine Delta (Z-score на TRAIN-фолде)
    delta_scaler = StandardScaler(with_mean=False)
    X_train_z = delta_scaler.fit_transform(X_train)
    X_test_z = delta_scaler.transform(X_test)

    centroids, centroid_labels = compute_centroids_sparse_mean(X_train_z, y_train)

    dists = cosine_distances(X_test_z, centroids)  # (n_chunks, n_auth_train)
    mean_dists = dists.mean(axis=0)

    sorted_local = np.argsort(mean_dists)
    top1_local = int(sorted_local[0])
    top1_delta_global = int(centroid_labels[top1_local])

    true_loc = np.where(centroid_labels == test_auth_idx)[0]
    if len(true_loc) > 0:
        true_loc_idx = int(true_loc[0])
        rank_delta = int(np.where(sorted_local == true_loc_idx)[0][0] + 1)
    else:
        rank_delta = 999

    author_id, book_id = test_group_id.split("/", 1)

    return {
        "test_author": authors[test_auth_idx],
        "test_book": book_id,
        "true_label": int(test_auth_idx),
        # LR
        "lr_pred": int(top1_lr),
        "lr_correct": bool(top1_lr == test_auth_idx),
        "lr_rank": int(rank_lr),
        "lr_conf": float(mean_prob[test_auth_idx]),
        # Delta
        "delta_pred": int(top1_delta_global),
        "delta_correct": bool(top1_delta_global == test_auth_idx),
        "delta_rank": int(rank_delta),
        # meta
        "n_chunks": int(len(test_indices)),
        "group_id": test_group_id,
        "author_id": author_id,
    }


def main() -> None:
    args = parse_args()

    # IMPORTANT: set before importing meta/meta-dependent modules
    os.environ["STYLO_LANG"] = args.lang
    from meta.meta import display_name  # noqa: WPS433

    train_root = pathlib.Path(args.train_root)
    if not train_root.exists():
        logging.error(f"Train root not found: {train_root}")
        raise SystemExit(1)

    authors = sorted(
        [p.name for p in train_root.iterdir() if p.is_dir() and p.name != "unknown"]
    )
    if len(authors) < 2:
        logging.error("Need at least 2 authors in data/frags_train.")
        raise SystemExit(1)

    logging.info(f"Language: {args.lang}")
    logging.info(f"Authors ({len(authors)}): {', '.join(authors)}")
    logging.info("Loading dataset...")

    texts, labels, groups = load_dataset(train_root, authors)
    if len(texts) < 50:
        logging.warning(f"Very small dataset: {len(texts)} fragments.")

    book_groups = create_book_groups(labels, groups)
    unique_books = sorted(book_groups.keys())

    if args.max_books and args.max_books > 0:
        unique_books = unique_books[: args.max_books]
        logging.warning(f"Limiting books to {len(unique_books)} due to --max-books")

    logging.info(f"Books in LOBO: {len(unique_books)}")

    vectorizer_params = dict(
        char_ngram_range=(3, 5),
        max_char_features=5000,
        char_min_df=3,
        use_char=True,
        use_func=True,
        use_mfw=True,
        mfw_count=300,
        use_syntax=False,   # выключено для выполнимости LOBO (spaCy-парс ×282 фолда = зависание); char≫word — доминирующий сигнал
        auto_bleach=False,  # bleaching тоже парсит per-fold; топик-контролируемое число — отдельно через GKF5
    )

    logging.info("Running LOBO folds...")
    parallel_res = Parallel(n_jobs=args.n_jobs, verbose=5, pre_dispatch="2*n_jobs")(
        delayed(run_fold)(
            texts,
            labels,
            groups,
            book_groups,
            authors,
            gid,
            vectorizer_params,
        )
        for gid in unique_books
    )
    results = [r for r in parallel_res if r is not None]

    if not results:
        logging.error("No LOBO results (possibly each author has only 1 book).")
        raise SystemExit(1)

    df = pd.DataFrame(results)

    lr_acc = df["lr_correct"].mean()
    lr_top2 = (df["lr_rank"] <= 2).mean()

    delta_acc = df["delta_correct"].mean()
    delta_top2 = (df["delta_rank"] <= 2).mean()

    logging.info("=== LOBO RESULTS (Leakage-Free) ===")
    logging.info(f"LR Accuracy:    {lr_acc:.3%} (Top-2: {lr_top2:.3%})")
    logging.info(f"Delta Accuracy: {delta_acc:.3%} (Top-2: {delta_top2:.3%})")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "experiments_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== LOBO EXPERIMENTS (Leakage-Free, Unified) ===\n")
        f.write(f"Дата: {datetime.datetime.now():%d.%m.%Y %H:%M}\n")
        f.write(f"Язык: {args.lang}\n\n")
        f.write("METRICS:\n")
        f.write(f"  LR Accuracy:    {lr_acc:.3%} (Top-2: {lr_top2:.3%})\n")
        f.write(f"  Delta Accuracy: {delta_acc:.3%} (Top-2: {delta_top2:.3%})\n\n")
        f.write("DETAILED BY AUTHOR (LR / Delta):\n")
        for auth in authors:
            auth_idx = authors.index(auth)
            sub = df[df["true_label"] == auth_idx]
            if len(sub) == 0:
                continue
            lr_sub = sub["lr_correct"].mean()
            delta_sub = sub["delta_correct"].mean()
            f.write(
                f"  {display_name(auth):20} -> "
                f"LR: {lr_sub:6.1%} | Delta: {delta_sub:6.1%} ({len(sub)} books)\n"
            )

    df.to_csv(outdir / "experiments_detailed.csv", index=False)
    logging.info(
        f"Saved: {outdir / 'experiments_summary.txt'} and experiments_detailed.csv"
    )


if __name__ == "__main__":
    main()
