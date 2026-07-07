"""
K-Means кластеризация стилометрических векторов.

Контракт:
- --lang: STYLO_LANG выставляется ДО импортов meta.meta.
- Артефакты train.py:
  - data/vectorizer_fitted.pkl (StyloVectorizer)
  - data/scaler_delta.pkl (StandardScaler(with_mean=False), Z-score)
  - data/authors.npy
- KMeans требует dense => для ограниченного числа точек делаем .toarray().
- Легенда: кластеры подсвечиваются доминирующим автором (если доля > 0.4).
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from collections import Counter

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy.sparse import issparse
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("pymorphy3").setLevel(logging.WARNING)
logging.info("=== Clustering analysis (K-Means, Unified) ===")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KMeans clustering for stylometry vectors")
    p.add_argument("--lang", default="ru", help="Language code: ru|en|fr")
    p.add_argument(
        "--datadir", default="data", help="Artifacts dir (vectorizer/scaler/authors)"
    )
    p.add_argument("--max-train-per-author", type=int, default=250)
    p.add_argument("--max-unknown", type=int, default=300)
    p.add_argument("--k-offset", type=int, default=2, help="k = n_authors + k_offset")
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

    datadir = pathlib.Path(args.datadir)
    vec_path = datadir / "vectorizer_fitted.pkl"
    scaler_path = datadir / "scaler_delta.pkl"
    authors_path = datadir / "authors.npy"

    if not vec_path.exists() or not scaler_path.exists() or not authors_path.exists():
        logging.error(
            "Файлы моделей не найдены (vectorizer_fitted.pkl / scaler_delta.pkl / authors.npy)."
        )
        logging.error("Сначала запустите scripts/train.py")
        raise SystemExit(1)

    vec = joblib.load(vec_path)
    scaler = joblib.load(scaler_path)
    authors = np.load(authors_path, allow_pickle=True)

    # Read TRAIN
    train_root = pathlib.Path("data/frags_train")
    train_texts: list[str] = []
    train_labels: list[str] = []

    for a in authors:
        a = str(a)
        files = sorted((train_root / a).rglob("*.txt"))[: args.max_train_per_author]
        for fp in files:
            txt = read_text_safe(fp)
            if txt:
                train_texts.append(txt)
                train_labels.append(a)

    # Read UNKNOWN
    unk_root = pathlib.Path("data/frags_unknown")
    if not unk_root.exists():
        unk_root = pathlib.Path("data/frags_unknown_plain")

    unk_files = sorted([p for p in unk_root.rglob("*.txt") if p.is_file()])
    if args.max_unknown and len(unk_files) > args.max_unknown:
        unk_files = unk_files[: args.max_unknown]

    unk_texts = [read_text_safe(fp) for fp in unk_files]
    unk_texts = [t for t in unk_texts if t]

    # Vectorize + Z-score
    X_all = train_texts + unk_texts
    if not X_all:
        logging.error("Нет данных для кластеризации.")
        raise SystemExit(1)

    logging.info(f"Vectorizing total fragments: {len(X_all)}")
    X_vec = vec.transform(X_all)
    X_z = scaler.transform(X_vec)

    n_train = len(train_texts)

    # KMeans needs dense
    X_kmeans = to_dense(X_z).astype(np.float32, copy=False)

    # UMAP for visualization
    reducer = umap.UMAP(
        n_neighbors=20, min_dist=0.05, metric="cosine", random_state=args.random_state
    )
    emb = reducer.fit_transform(X_kmeans)
    emb_unk = emb[n_train:]

    # KMeans
    k = int(len(authors) + args.k_offset)
    logging.info(f"Running KMeans: k={k}")
    km = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
    km_labels = km.fit_predict(X_kmeans)

    km_train = km_labels[:n_train]

    # Plot
    plt.figure(figsize=(14, 10))
    unique_clusters = np.sort(np.unique(km_labels))
    colors = cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        mask_all = km_labels == cluster_id
        mask_train_in_cluster = km_train == cluster_id

        label_text = f"Clust {cluster_id}"

        train_indices = np.where(mask_train_in_cluster)[0]
        if len(train_indices) > 0:
            cluster_authors = [train_labels[idx] for idx in train_indices]
            if cluster_authors:
                most_common, count = Counter(cluster_authors).most_common(1)[0]
                if count / len(cluster_authors) > 0.4:
                    label_text += f" (~{display_label(most_common)})"

        plt.scatter(
            emb[mask_all, 0],
            emb[mask_all, 1],
            color=colors[i],
            label=label_text,
            s=40,
            alpha=0.6,
            edgecolors="none",
        )

    if len(emb_unk) > 0:
        plt.scatter(
            emb_unk[:, 0],
            emb_unk[:, 1],
            color="red",
            marker="*",
            s=160,
            edgecolors="white",
            linewidths=1.5,
            label=display_label("unknown"),
            zorder=10,
        )

    plt.title(f"K-Means Clustering (k={k}) on Unified Z-Scored Vectors")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("docs", exist_ok=True)
    out = pathlib.Path("docs/clusters_kmeans.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
