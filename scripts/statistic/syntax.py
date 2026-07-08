import os
import spacy
import numpy as np
import matplotlib.pyplot as plt
from meta.meta import display_label
from scripts.nlp import get_stylometry_nlp
from utils import get_all_txt_files
import pathlib


datasets = {}

# 1. Unknown
if os.path.exists("data/frags_unknown"):
    datasets["unknown"] = "data/frags_unknown"
elif os.path.exists("data/frags_unknown_plain"):
    datasets["unknown"] = "data/frags_unknown_plain"

# 2. Train authors
train_root = "data/frags_train"
if os.path.exists(train_root):
    for a in sorted(os.listdir(train_root)):
        path = os.path.join(train_root, a)
        if os.path.isdir(path):
            datasets[a] = path

labels = []
data = []

print("Сбор статистики по синтаксису...")

# Загружаем NLP только здесь, чтобы избежать глобальной загрузки
nlp_pipe = get_stylometry_nlp()

for label, folder in datasets.items():
    files = get_all_txt_files(folder)
    if not files:
        print(f"⚠️ Нет файлов для {label}")
        continue

    texts = [open(f, encoding="utf-8").read() for f in files]
    lengths = []

    for doc in nlp_pipe.pipe(texts, batch_size=40):
        for sent in doc.sents:
            words = [t for t in sent if t.is_alpha]
            if words:
                lengths.append(len(words))

    if lengths:
        labels.append(display_label(label))
        data.append(lengths)
        print(f"  OK: {label} ({len(lengths)} предложений)")

if not data:
    print("Нет данных для построения графика!")
    exit(1)

plt.figure(figsize=(10, 6))
plt.boxplot(data, tick_labels=labels, showmeans=True)
plt.xticks(rotation=45, ha='right')
plt.title("Распределение длины предложений (в словах)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/syntax.png", dpi=150)
print("График сохранен: docs/syntax.png")

with open("docs/syntax_stats.txt", "w", encoding="utf-8") as f:
    f.write("=== SYNTAX SUMMARY ===\n")
    for label, lengths in zip(labels, data):
        f.write(f"{label}\n")
        f.write(f"  mean: {np.mean(lengths):.2f}\n")
        f.write(f"  median: {np.median(lengths):.2f}\n")
        f.write(f"  count: {len(lengths)}\n\n")
