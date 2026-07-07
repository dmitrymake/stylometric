import os
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from meta import display_label
from scripts.nlp import get_stylometry_nlp
from utils import get_all_txt_files
from meta import BASE_LANG_MODEL, ERR_LANG_MODEL

datasets = {}

train_root = "data/frags_train"
if os.path.exists(train_root):
    for a in sorted(os.listdir(train_root)):
        path = os.path.join(train_root, a)
        if os.path.isdir(path):
            datasets[a] = path

ratios = {}
print("Сбор статистики POS...")

nlp_pipe = get_stylometry_nlp()

for label, folder in datasets.items():
    files = get_all_txt_files(folder)
    if not files:
        continue

    texts = [open(f, encoding="utf-8").read() for f in files]

    c = Counter()
    for doc in nlp_pipe.pipe(texts, batch_size=40):
        for t in doc:
            if t.is_alpha:
                c[t.pos_] += 1

    adj = c["ADJ"]
    verb = c["VERB"]
    ratio = adj / verb if verb else 0
    ratios[label] = ratio
    print(f"  OK: {label} (ADJ/VERB = {ratio:.2f})")

if not ratios:
    print("Нет данных POS!")
    exit(1)

plt.figure(figsize=(10, 6))
labels_sorted = sorted(ratios.keys())
vals = [ratios[k] for k in labels_sorted]
names = [display_label(k) for k in labels_sorted]

plt.bar(names, vals, color="#3498db")
plt.xticks(rotation=45, ha='right')
plt.title("Соотношение Прилагательных к Глаголам (ADJ / VERB)")
plt.tight_layout()

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/pos.png", dpi=150)
print("График сохранен: docs/pos.png")

with open("docs/pos_stats.txt", "w", encoding="utf-8") as f:
    f.write("=== POS STATISTICS ===\n\n")
    for label, ratio in ratios.items():
        f.write(f"{display_label(label)}: ADJ/VERB = {ratio:.4f}\n")
