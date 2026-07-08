import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import logging
from meta import display_label

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info("=== Entropy analysis ===")


def tokenize_ru(text):
    return re.findall(r"[а-яёА-ЯЁ]+", text.lower())


def entropy(tokens):
    if not tokens:
        return 0.0
    total = len(tokens)
    freq = Counter(tokens)
    h = 0.0
    for cnt in freq.values():
        p = cnt / total
        h -= p * math.log2(p)
    return h


def get_all_txt_files(root_dir):
    txt_files = []
    if not os.path.exists(root_dir):
        return txt_files
    for dp, _, fs in os.walk(root_dir):
        for f in fs:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(dp, f))
    return sorted(txt_files)


datasets = {}

if os.path.exists("data/frags_unknown"):
    datasets["unknown"] = "data/frags_unknown"
elif os.path.exists("data/frags_unknown_plain"):
    datasets["unknown"] = "data/frags_unknown_plain"

train_root = "data/frags_train"
if os.path.exists(train_root):
    for a in sorted(os.listdir(train_root)):
        path = os.path.join(train_root, a)
        if os.path.isdir(path):
            datasets[a] = path

results = {}
raw_values = {}

logging.info("Сбор значений энтропии...")

for label, path in datasets.items():
    files = get_all_txt_files(path)
    if not files:
        logging.warning(f"Нет данных для {label}")
        continue

    values = []
    for f in files:
        try:
            text = open(f, encoding="utf-8").read()
            if not text.strip():
                continue
            tok = tokenize_ru(text)
            if tok:
                e = entropy(tok)
                values.append(e)
        except Exception as ex:
            logging.warning(f"Ошибка {f}: {ex}")

    if values:
        raw_values[label] = values
        results[label] = np.mean(values)
        logging.info(f"  {display_label(label)}: {results[label]:.3f}")

if not results:
    logging.error("Нет данных для визуализации!")
    exit(1)

os.makedirs("docs", exist_ok=True)

# Текстовый отчет
out = ["=== ENTROPY STATISTICS REPORT ===\n"]
for label in raw_values:
    vals = np.array(raw_values[label])
    out.append(display_label(label))
    out.append(f"  Mean: {np.mean(vals):.4f} | Median: {np.median(vals):.4f}")
    out.append("")

with open("docs/entropy_stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))

# График
plt.figure(figsize=(12, 6))
names = [display_label(k) for k in results.keys()]
vals = [results[k] for k in results.keys()]
colors = ["#e74c3c" if "unknown" in k else "#9b59b6" for k in results.keys()]

plt.bar(names, vals, color=colors, edgecolor='black')
plt.title("Лексическая энтропия (разнообразие словаря)")
plt.ylabel("H (бит на слово)")
plt.xticks(rotation=45, ha='right')
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("docs/entropy.png", dpi=150)
logging.info("Сохранено: docs/entropy.png")
