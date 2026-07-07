"""Дообученный нейро-baseline: fine-tune ruBERT-tiny2 на Proza.ru (тот же сплит, что LUAR/char-SVM).

Сплит идентичен scripts/run_luar_proza.py (50 авторов, 50/50, seed=42): числа сопоставимы
с docs/luar_proza.json (char-SVM 0.881, frozen ruBERT-tiny2 0.682, LUAR-MUD 0.387).
Модель: cointegrated/rubert-tiny2 + классификационная голова (50 классов); выбор эпохи —
по валидации, выделенной ИЗ TRAIN (тест не участвует в выборе). CPU, потоки зажаты
(общая машина). Выход: docs/neuro_finetune_proza.json.

Run: nice -n 10 .venv/bin/python scripts/run_finetune_rubert_proza.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import pathlib
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SEED = 42
ROOT = pathlib.Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "external" / "proza_ru_hard.parquet"
NAUTH, PER = 50, 60
MODEL = "cointegrated/rubert-tiny2"
MAX_LEN = 512
BATCH = 16
EPOCHS = 6
LR = 5e-5
VAL_FRAC = 0.15


def log(*a):
    print(*a, flush=True)


def load_split():
    """Бит-в-бит копия load_split из scripts/run_luar_proza.py (один и тот же сплит)."""
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    lens = df["fullText"].apply(len)
    top = df.assign(n=lens).sort_values("n", ascending=False).head(NAUTH)
    rng = np.random.RandomState(SEED)
    texts, labels = [], []
    for _, row in top.iterrows():
        arr = [str(t) for t in row["fullText"] if len(str(t).split()) >= 30]
        if len(arr) > PER:
            arr = [arr[i] for i in sorted(rng.choice(len(arr), PER, replace=False))]
        for t in arr:
            texts.append(t)
            labels.append(row["authorIDs"])
    A = sorted(set(labels))
    aidx = {a: i for i, a in enumerate(A)}
    y = np.array([aidx[a] for a in labels])
    tr = np.zeros(len(y), bool)
    for a in range(len(A)):
        idx = np.where(y == a)[0]
        rng.shuffle(idx)
        tr[idx[: len(idx) // 2]] = True
    return texts, y, tr, len(A)


def main():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score

    torch.set_num_threads(6)
    torch.manual_seed(SEED)

    texts, y, tr, nA = load_split()
    te = ~tr
    log(f"Proza.ru: authors={nA} texts={len(texts)} (train={tr.sum()}/test={te.sum()})")

    # валидация — из train (стратифицированно), тест не участвует в выборе эпохи
    rng = np.random.RandomState(SEED)
    tr_idx = np.where(tr)[0]
    val_mask = np.zeros(len(y), bool)
    for a in range(nA):
        ai = tr_idx[y[tr_idx] == a]
        rng.shuffle(ai)
        k = max(1, int(len(ai) * VAL_FRAC))
        val_mask[ai[:k]] = True
    fit_mask = tr & ~val_mask
    log(f"fit={fit_mask.sum()} val={val_mask.sum()} test={te.sum()}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=nA)

    def encode(idx_mask):
        idx = np.where(idx_mask)[0]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"],
                             torch.tensor(y[idx], dtype=torch.long))

    dl_fit = DataLoader(encode(fit_mask), batch_size=BATCH, shuffle=True)
    dl_val = DataLoader(encode(val_mask), batch_size=BATCH * 2)
    dl_te = DataLoader(encode(te), batch_size=BATCH * 2)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    @torch.no_grad()
    def evaluate(dl):
        model.eval()
        preds, gold = [], []
        for ids, am, yy in dl:
            out = model(input_ids=ids, attention_mask=am)
            preds.append(out.logits.argmax(-1).numpy())
            gold.append(yy.numpy())
        p, g = np.concatenate(preds), np.concatenate(gold)
        return float(accuracy_score(g, p)), float(f1_score(g, p, average="macro"))

    best_val, best_state, best_epoch = -1.0, None, -1
    history = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for ids, am, yy in dl_fit:
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=am, labels=yy)
            out.loss.backward()
            opt.step()
            tot += float(out.loss)
        va, vf = evaluate(dl_val)
        history.append({"epoch": ep, "train_loss": round(tot / len(dl_fit), 4),
                        "val_top1": round(va, 4), "val_macro_f1": round(vf, 4)})
        log(f"эпоха {ep}: loss={tot/len(dl_fit):.4f} val top1={va:.4f} f1={vf:.4f} ({time.time()-t0:.0f}s)")
        if va > best_val:
            best_val, best_epoch = va, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ta, tf_ = evaluate(dl_te)
    log(f"ТЕСТ (эпоха {best_epoch} по валидации): top1={ta:.4f} macro-F1={tf_:.4f}")

    prev = json.loads((ROOT / "docs" / "luar_proza.json").read_text())["results_top1_macroF1"]
    out = {
        "dataset": "Proza.ru hard 50auth 50/50 seed42 (сплит идентичен docs/luar_proza.json)",
        "model": MODEL,
        "protocol": (f"fine-tune классификационной головы и всех весов; max_len={MAX_LEN}, batch={BATCH}, "
                     f"lr={LR}, до {EPOCHS} эпох; выбор эпохи по валидации {VAL_FRAC:.0%} из train "
                     "(тест в выборе не участвует); CPU"),
        "best_epoch": best_epoch,
        "history": history,
        "test_top1": round(ta, 3),
        "test_macro_f1": round(tf_, 3),
        "reference_same_split": prev,
    }
    (ROOT / "docs" / "neuro_finetune_proza.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    log("saved docs/neuro_finetune_proza.json")


if __name__ == "__main__":
    main()
