import os
import joblib
import numpy as np
import pandas as pd
from meta import display_label, display_name
from collections import defaultdict
from scipy.sparse import issparse

print("=== АНАЛИЗ СТРУКТУРНОЙ АНОМАЛЬНОСТИ ПО КНИГАМ ===")

try:
    loaded_data = joblib.load("data/train_vectors.pkl")
    X_transformed = loaded_data["X_transformed"]
    labels = loaded_data["labels"]
    groups = loaded_data["groups"]  # group: author/book_id
    authors = loaded_data["authors"]  # list of author names

except FileNotFoundError:
    print("❌ Не найден data/train_vectors.pkl. Сначала запустите train.py.")
    exit(1)


# Определяем индексы признаков "Локальная энтропия" и "Синтаксическая сложность".
# Мы полагаемся на то, что SyntaxFeatureExtractor добавляет эти 2 признака в конец вектора.

TOTAL_SYNTAX_FEATURES = 18  # Общее количество признаков в SyntaxFeatureExtractor (если добавлены все)
EXPECTED_ANOMALY_FEATURES = 2  # (Local Entropy, Syntactic Complexity)

if X_transformed.shape[1] < EXPECTED_ANOMALY_FEATURES:
    print(f"❌ Недостаточно признаков ({X_transformed.shape[1]}). Возможно, синтаксические признаки не были рассчитаны.")
    exit(1)

LOCAL_ENTROPY_COL = X_transformed.shape[1] - 2
SYNTACTIC_COMPLEXITY_COL = X_transformed.shape[1] - 1


# Группировка данных по книгам: group -> [index1, index2, ...]
book_groups_map = defaultdict(list)
for i, group_id in enumerate(groups):
    book_groups_map[group_id].append(i)

results = []

print(f"Анализ {len(book_groups_map)} книг...")

for group_id, indices in book_groups_map.items():
    X_book = X_transformed[indices]

    # .tocsr() для надежного среза по столбцам на разреженной матрице
    if issparse(X_book):
        X_book = X_book.tocsr()

    X_anomaly = X_book[:, [LOCAL_ENTROPY_COL, SYNTACTIC_COMPLEXITY_COL]]

    mean_entropy = np.mean(X_anomaly[:, 0])
    mean_complexity = np.mean(X_anomaly[:, 1])

    parts = group_id.split("/")
    author_id = parts[0]
    book_name = parts[1]
    
    results.append({
        "author_id": author_id,
        "author_name": display_name(author_id),
        "book_id": book_name,
        "local_entropy_mean": float(mean_entropy),
        "syntactic_complexity_mean": float(mean_complexity),
        "n_chunks": len(indices)
    })
    
df = pd.DataFrame(results)

df_sorted = df.sort_values(by="author_name")

os.makedirs("docs", exist_ok=True)

df_sorted.to_csv("docs/anomaly_stats_by_book.csv",
                 index=False, float_format="%.4f")

print("\n=== РЕЗУЛЬТАТЫ ПО КНИГАМ ===")
print("\nТОП-10 КНИГ ПО СИНТАКСИЧЕСКОЙ СЛОЖНОСТИ:")
print(df_sorted.nlargest(10, 'syntactic_complexity_mean')[[
    "author_name", "book_id", "syntactic_complexity_mean", "n_chunks"
]].to_markdown(index=False, floatfmt=".4f"))

print("\nФАЙЛ СОХРАНЕН: docs/anomaly_stats_by_book.csv (детализация по всем книгам)")
