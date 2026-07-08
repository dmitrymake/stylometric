import os
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from meta import display_label
from scripts.nlp import get_stylometry_nlp
from utils import get_all_txt_files
from meta import BASE_LANG_MODEL, ERR_LANG_MODEL


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

FEATURES = ["Case", "Number", "Aspect", "Tense", "Mood", "Degree"]
results = {}

print("Анализ морфологии...")

nlp_pipe = get_stylometry_nlp()

for label, folder in datasets.items():
    files = get_all_txt_files(folder)
    if not files:
        continue

    texts = [open(f, encoding="utf-8").read() for f in files]
    counters = {feat: Counter() for feat in FEATURES}

    for doc in nlp_pipe.pipe(texts, batch_size=40):
        for t in doc:
            if not t.is_alpha:
                continue
            morph = t.morph
            for feat in FEATURES:
                val = morph.get(feat)
                if val:
                    counters[feat][val[0]] += 1

    results[label] = counters
    print(f"  OK: {label}")

os.makedirs("docs", exist_ok=True)
if not results:
    print("Нет данных!")
    exit(1)
# Графики
for feat in FEATURES:
    plt.figure(figsize=(12, 6))
    xs, ys = [], []
    for label in results:
        cnt = results[label][feat]
        total = sum(cnt.values())
        xs.append(display_label(label))
        ys.append(total)

    plt.bar(xs, ys, color="#2ecc71")
    plt.title(f"Морфология: {feat} (абсолютная частота)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"docs/morpho_{feat}.png", dpi=150)
    plt.close()

print("Графики морфологии сохранены.")
