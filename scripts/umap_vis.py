"""
UMAP-визуализация стилометрических векторов.

Использует единые артефакты, которые создает train.py:
   - data/vectorizer_fitted.pkl (StyloVectorizer)
   - data/scaler_delta.pkl (StandardScaler(with_mean=False))
   - data/authors.npy

STYLO_LANG выставляется ДО импорта meta.meta (язык влияет на загрузку
языкозависимых модулей в момент импорта).
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib

import joblib
import numpy as np
import seaborn as sns
import umap
from scipy.sparse import issparse
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("pymorphy3").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UMAP visualization for stylometry vectors")
    p.add_argument("--lang", default="ru", help="Language code: ru|en|fr")
    p.add_argument(
        "--datadir", default="data", help="Artifacts dir (vectorizer/scaler/authors)"
    )
    p.add_argument("--max-train-per-author", type=int, default=200)
    p.add_argument("--unknown-step", type=int, default=3)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument(
        "--metric", default="cosine", help="UMAP metric (cosine is recommended)"
    )
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def read_text_safe(fp: pathlib.Path) -> str:
    try:
        return fp.read_text("utf-8").strip()
    except Exception:
        return ""


def to_dense(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def main() -> None:
    args = parse_args()

    # IMPORTANT: set language before importing meta/meta-dependent modules
    os.environ["STYLO_LANG"] = args.lang
    from meta.meta import display_label  # noqa: WPS433 (import after env)

    logging.info("=== UMAP visualization (Unified, Z-Score) ===")
    logging.info(f"Language: {args.lang}")

    datadir = pathlib.Path(args.datadir)
    vec_path = datadir / "vectorizer_fitted.pkl"
    scaler_path = datadir / "scaler_delta.pkl"
    authors_path = datadir / "authors.npy"

    if not vec_path.exists() or not scaler_path.exists() or not authors_path.exists():
        logging.error(
            "Не найдены артефакты (vectorizer_fitted.pkl / scaler_delta.pkl / authors.npy)."
        )
        logging.error("Запустите scripts/train.py")
        raise SystemExit(1)

    vec = joblib.load(vec_path)
    scaler = joblib.load(scaler_path)
    authors = np.load(authors_path, allow_pickle=True)

    # Collect TRAIN
    train_root = pathlib.Path("data/frags_train")
    X_train_raw: list[str] = []
    train_labels: list[str] = []

    for author in authors:
        author = str(author)
        path = train_root / author
        if not path.exists():
            continue

        files = sorted(path.rglob("*.txt"))[: args.max_train_per_author]
        for fp in files:
            txt = read_text_safe(fp)
            if txt:
                X_train_raw.append(txt)
                train_labels.append(author)

    logging.info(f"Train fragments: {len(X_train_raw)}")

    # Collect UNKNOWN
    unk_root = pathlib.Path("data/frags_unknown")
    if not unk_root.exists():
        unk_root = pathlib.Path("data/frags_unknown_plain")

    unk_files = sorted(unk_root.rglob("*.txt"))
    if len(unk_files) > 300 and args.unknown_step > 1:
        unk_files = unk_files[:: args.unknown_step]

    X_unk_raw = [read_text_safe(fp) for fp in unk_files if fp.is_file()]
    X_unk_raw = [t for t in X_unk_raw if t]

    logging.info(f"Unknown fragments: {len(X_unk_raw)}")

    if not X_train_raw:
        logging.error(
            "Нет train данных для UMAP (data/frags_train пуст или не совпадает с authors.npy)."
        )
        raise SystemExit(1)

    # Vectorize + Z-score
    X_all = X_train_raw + X_unk_raw
    X_vec = vec.transform(X_all)
    X_z = scaler.transform(X_vec)

    # UMAP: зачастую надежнее на dense (на вашем объеме данных это безопасно)
    X_umap = to_dense(X_z).astype(np.float32, copy=False)

    n_train = len(X_train_raw)
    rng = check_random_state(args.random_state)

    # UMAP
    logging.info(
        f"UMAP: metric={args.metric}, n_neighbors={args.n_neighbors}, min_dist={args.min_dist}"
    )
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=rng,
        n_jobs=-1,
    )
    emb = reducer.fit_transform(X_umap)

    emb_train = emb[:n_train]
    emb_unk = emb[n_train:]

    # Plot
    plt.figure(figsize=(14, 10))
    palette = sns.color_palette("bright", len(authors))

    for i, a in enumerate(authors):
        a = str(a)
        mask = np.array([lbl == a for lbl in train_labels], dtype=bool)
        if not mask.any():
            continue
        pts = emb_train[mask]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            label=display_label(a),
            color=palette[i],
            alpha=0.6,
            s=60,
            edgecolors="none",
        )

    if len(emb_unk) > 0:
        plt.scatter(
            emb_unk[:, 0],
            emb_unk[:, 1],
            marker="*",
            s=220,
            color="#2c3e50",
            edgecolors="white",
            linewidths=1.5,
            label=display_label("unknown"),
            zorder=10,
        )

    plt.title("UMAP Projection (Unified vectors, Z-Score)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    os.makedirs("docs", exist_ok=True)
    outfile = pathlib.Path("docs/umap.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    logging.info(f"UMAP saved: {outfile}")


if __name__ == "__main__":
    main()
