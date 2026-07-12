"""
Расчет коэффициента консистентности (однородности стиля).
Memory Safe: работает с sparse матрицами без полного разворачивания.
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import logging
import seaborn as sns
from scipy.spatial.distance import cosine
from collections import defaultdict
from scipy.sparse import issparse
from meta.meta import display_label, display_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_vec_dense(X, idx):
    """Извлекает вектор по индексу и делает его плотным (1D array)."""
    v = X[idx]
    if issparse(v):
        v = v.toarray()
    return np.asarray(v).flatten()

def main():
    logging.info("=== АНАЛИЗ КОНСИСТЕНТНОСТИ СТИЛЯ (LOBO METHOD) ===")

    try:
        loaded_data = joblib.load("data/train_vectors.pkl")
        X = loaded_data["X_transformed"] # SPARSE MATRIX
        y = loaded_data["labels"]
        groups = loaded_data["groups"]
        authors = loaded_data["authors"]
    except FileNotFoundError:
        logging.error("Нет данных. Запустите train.py.")
        exit(1)

    # Не делаем X.toarray() целиком! Это убьет RAM на больших данных.
    
    # Индексация
    author_indices = defaultdict(list)
    book_indices = defaultdict(list)

    for i, (lbl, grp) in enumerate(zip(y, groups)):
        author_indices[lbl].append(i)
        book_indices[grp].append(i)

    consistency_results = []
    author_stats = []
    
    unique_authors = np.unique(y)

    for auth_idx in unique_authors:
        auth_name = authors[auth_idx]
        all_auth_idxs = author_indices[auth_idx]
        if not all_auth_idxs: continue
            
        auth_books = sorted(list(set(groups[all_auth_idxs])))
        # Равный train-side вес книг: сначала среднее чанков каждой книги,
        # затем среднее книг автора. Длинная книга не доминирует в LOBO-centroid.
        book_means = {}
        for book_grp in auth_books:
            book_mean = X[book_indices[book_grp]].mean(axis=0)
            if issparse(book_mean):
                book_mean = book_mean.toarray()
            book_mean = np.asarray(book_mean).flatten()
            book_means[book_grp] = book_mean / (np.linalg.norm(book_mean) + 1e-12)
        total_work_sum = np.sum(list(book_means.values()), axis=0)
        distances_buffer = [] 

        for book_grp in auth_books:
            b_idxs = book_indices[book_grp]
            
            book_submatrix = X[b_idxs]
            # LOBO Centroid
            remainder_count = len(auth_books) - 1
            if remainder_count <= 0:
                continue
            
            lobo_centroid = (total_work_sum - book_means[book_grp]) / remainder_count
            
            # Тут придется идти циклом или батчами, т.к. cosine в scipy требует 1D
            # Для скорости можно использовать sklearn cosine_distances(book_submatrix, lobo_centroid)
            from sklearn.metrics.pairwise import cosine_distances
            
            # lobo_centroid shape (n,) -> reshape (1, n)
            centroid_2d = lobo_centroid.reshape(1, -1)
            
            # book_submatrix (sparse) vs centroid (dense) - работает эффективно
            dists = cosine_distances(book_submatrix, centroid_2d).flatten()
            
            mean_d = np.mean(dists)
            std_d = np.std(dists)
            
            consistency_results.append({
                "author": auth_name,
                "book": book_grp.split("/")[-1],
                "mean_dist": mean_d,
                "std_dist": std_d,
                "distances": dists
            })
            distances_buffer.extend(dists)

        if distances_buffer:
            author_stats.append({
                "author": auth_name,
                "global_mean": np.mean(distances_buffer),
                "global_std": np.std(distances_buffer)
            })

    # Визуализация и сохранение
    if not consistency_results:
        logging.error("Нет данных.")
        exit(0)

    # Сохранение отчета
    df_stats = pd.DataFrame(consistency_results)
    df_stats["Author"] = df_stats["author"].apply(display_name)
    df_stats = df_stats[["Author", "book", "mean_dist", "std_dist"]].sort_values(by=["Author", "mean_dist"])
    
    df_auth_stats = pd.DataFrame(author_stats).sort_values(by="global_mean")
    
    pathlib.Path("docs").mkdir(exist_ok=True)
    with open("docs/consistency_stats.txt", "w", encoding="utf-8") as f:
        f.write("=== CONSISTENCY REPORT (LOBO) ===\n")
        f.write(df_stats.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n=== AUTHOR STABILITY ===\n")
        f.write(df_auth_stats.to_markdown(index=False, floatfmt=".4f"))

    # График
    plot_data = []
    for r in consistency_results:
        for d in r["distances"]:
            plot_data.append({"Author": display_label(r["author"]), "Distance": d})
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Author', y='Distance', data=pd.DataFrame(plot_data), showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.title("Стилистическая дистанция (LOBO)")
    plt.tight_layout()
    plt.savefig("docs/consistency_boxplot.png", dpi=150)
    plt.close()
    
    logging.info("Готово.")

if __name__ == "__main__":
    main()
