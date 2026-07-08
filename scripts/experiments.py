"""
Автоматические LOBO-эксперименты (Leave-One-Book-Out валидация).
"""

from __future__ import annotations
import pathlib
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import datetime
from collections import defaultdict
import gc
import re

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from syntax_features import StyloVectorizer
from meta.meta import display_name
from utils import tokenize_preserve, make_chunks

import os
# Ограничиваем потоки numpy, чтобы отдать ядра joblib
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("=== LOBO EXPERIMENTS (PARALLEL LOADING) ===")

# Конфигурация
CLEAN_ROOT = pathlib.Path("input_clean")

VECTORIZER_PARAMS = {
    "char_ngram_range": (3, 5),
    "max_char_features": 5000,
    "char_min_df": 3,
    "use_char": True,
    "use_func": True,
    "use_mfw": True,
    "mfw_count": 300,
    "use_syntax": True
}

def process_book(auth_name, auth_idx, book_path):
    """
    Читает одну книгу, токенизирует и режет на чанки.
    """
    try:
        text_raw = book_path.read_text("utf-8").strip()
        if not text_raw:
            return None

        words = tokenize_preserve(text_raw)
        chunks = make_chunks(words, size=500, min_size=200, overlap=0.0)

        if not chunks:
            return None

        book_id = book_path.stem
        return {
            "chunks": chunks,
            "label": auth_idx,
            "group_id": f"{auth_name}/{book_id}"
        }
    except Exception as e:
        logging.warning(f"Ошибка при чтении {book_path}: {e}")
        return None


# 1. Сбор и загрузка данных
logging.info("1. Параллельное чтение и нарезка текстов...")

authors = sorted([d.name for d in CLEAN_ROOT.iterdir()
                 if d.is_dir() and d.name != "unknown"])
if not authors:
    logging.error("Нет авторов в input_clean!")
    exit(1)

auth2idx = {a: i for i, a in enumerate(authors)}

tasks = []
for auth in authors:
    auth_dir = CLEAN_ROOT / auth
    for book_path in auth_dir.glob("*.txt"):
        tasks.append((auth, auth2idx[auth], book_path))

logging.info(f"Найдено книг: {len(tasks)}. Запуск воркеров...")

results_raw = Parallel(n_jobs=-1, verbose=1)(
    delayed(process_book)(auth, idx, path) for auth, idx, path in tasks
)

texts = []
labels = []
groups = []
BOOK_GROUPS = defaultdict(list)

for res in results_raw:
    if res is None:
        continue

    start_idx = len(texts)

    texts.extend(res['chunks'])

    n = len(res['chunks'])
    labels.extend([res['label']] * n)
    groups.extend([res['group_id']] * n)

    end_idx = len(texts)

    BOOK_GROUPS[res['group_id']] = list(range(start_idx, end_idx))

texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype=int)
groups = np.array(groups)

logging.info(f"Загружено: {len(texts)} чанков. Готово к векторизации.")

# 2. Глобальная векторизация
logging.info("2. Векторизация...")

vec = StyloVectorizer(**VECTORIZER_PARAMS)
X_sparse = vec.fit_transform(texts)
logging.info(f"Размерность матрицы: {X_sparse.shape}")

del texts
gc.collect()

# 3. LOBO цикл
unique_groups = sorted(BOOK_GROUPS.keys())
logging.info(f"3. Запуск экспериментов ({len(unique_groups)} книг)...")


def run_fold(test_group_id):
    test_indices = BOOK_GROUPS[test_group_id]

    mask_test = np.zeros(X_sparse.shape[0], dtype=bool)
    mask_test[test_indices] = True
    mask_train = ~mask_test

    X_train = X_sparse[mask_train]
    y_train = labels[mask_train]
    X_test = X_sparse[mask_test]

    test_auth_idx = labels[test_indices[0]]

    # Если автора нет в трейне (например, у него была всего 1 книга)
    if test_auth_idx not in y_train:
        return None

    clf = make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", solver='lbfgs')
    )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    mean_prob = probs.mean(axis=0)

    sorted_idx = np.argsort(mean_prob)[::-1]
    top1 = sorted_idx[0]
    true_label = test_auth_idx
    rank = np.where(sorted_idx == true_label)[0][0] + 1

    return {
        "test_author": authors[true_label],
        "test_book": test_group_id.split("/", 1)[1],
        "true_label": int(true_label),
        "pred_label": int(top1),
        "pred_author": authors[top1],
        "rank": int(rank),
        "top1_correct": top1 == true_label,
        "confidence": float(mean_prob[true_label]),
        "n_chunks": len(test_indices)
    }


parallel_res = Parallel(n_jobs=-1, verbose=5, pre_dispatch='2*n_jobs')(
    delayed(run_fold)(gid) for gid in unique_groups
)
results = [r for r in parallel_res if r is not None]

if not results:
    logging.error("Нет результатов!")
    exit(1)

df = pd.DataFrame(results)

# 4. Статистика и отчеты
df["rank"] = df["rank"].astype(int)

accuracy = accuracy_score(df["true_label"], df["pred_label"])
rank1 = (df["rank"] == 1).mean()
rank2 = (df["rank"] <= 2).mean()

logging.info(f"Accuracy (Top-1): {accuracy:.3%}")
logging.info(f"Top-2 Accuracy:   {rank2:.3%}")

os.makedirs("docs", exist_ok=True)

with open("docs/experiments_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"=== LOBO EXPERIMENTS (OPTIMIZED + PARALLEL LOAD) ===\n")
    f.write(f"Дата: {datetime.datetime.now():%d.%m.%Y %H:%M}\n")
    f.write(f"Accuracy (Top-1): {accuracy:.3%}\n")
    f.write(f"Top-2 Accuracy:   {rank2:.3%}\n\n")
    for auth in authors:
        sub = df[df["test_author"] == auth]
        if len(sub) == 0:
            continue
        acc = sub["top1_correct"].mean()
        f.write(f"  {display_name(auth):20} → {acc:.3%} ({len(sub)} книг)\n")

df.to_csv("docs/experiments_detailed.csv", index=False)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="rank", hue="rank", legend=False, palette="viridis")

plt.title("Распределение рангов (LOBO)")
plt.savefig("docs/rank_distribution.png", dpi=150)
plt.close()

try:
    conf = pd.crosstab(
        df["test_author"].map(display_name),
        df["pred_author"].map(display_name),
        normalize="index"
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf, annot=True, fmt=".2f",
                cmap="YlGnBu", cbar_kws={'label': 'Доля'})
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.savefig("docs/confusion_heatmap.png", dpi=150)
    plt.close()
except:
    pass

logging.info("Готово!")
