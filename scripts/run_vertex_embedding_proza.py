"""Замороженный SOTA-класса нейро-baseline: gemini-embedding-001 (через VertexAI) на Proza.ru.

Сплит идентичен scripts/run_luar_proza.py (50 авторов, 50/50, seed=42) — числа сопоставимы
с docs/luar_proza.json и docs/neuro_finetune_proza.json. Эмбеддинги (3072-dim,
task_type=CLASSIFICATION, вход обрезается до MAX_CHARS) + LinearSVC.
Идентификатор проекта читается из конфигурации gcloud и в артефакты не пишется.
Кэш эмбеддингов: data/emb_proza_gemini.npy (вне git).

Run: .venv/bin/python scripts/run_vertex_embedding_proza.py
"""
from __future__ import annotations
import json
import pathlib
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

SEED = 42
ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
PARQUET = ROOT / "data" / "external" / "proza_ru_hard.parquet"
CACHE = ROOT / "data" / "emb_proza_gemini.npy"
NAUTH, PER = 50, 60
MODEL = "gemini-embedding-001"
MAX_CHARS = 6000
WORKERS = 8


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


def _gcloud(args):
    return subprocess.check_output(["gcloud"] + args, text=True).strip()


def embed_all(texts):
    if CACHE.exists():
        E = np.load(CACHE)
        if len(E) == len(texts):
            log("  кэш эмбеддингов найден")
            return E
    proj = _gcloud(["config", "get-value", "project"])
    url = (f"https://us-central1-aiplatform.googleapis.com/v1/projects/{proj}"
           f"/locations/us-central1/publishers/google/models/{MODEL}:predict")
    tok = {"v": _gcloud(["auth", "application-default", "print-access-token"]), "t": time.time()}

    def token():
        if time.time() - tok["t"] > 1800:
            tok["v"] = _gcloud(["auth", "application-default", "print-access-token"])
            tok["t"] = time.time()
        return tok["v"]

    def one(i_text):
        i, text = i_text
        body = dumps_strict({"instances": [{"content": text[:MAX_CHARS],
                                          "task_type": "CLASSIFICATION"}]},
                            ensure_ascii=True).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, data=body, headers={
                    "Authorization": f"Bearer {token()}",
                    "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.load(r)
                return i, np.asarray(d["predictions"][0]["embeddings"]["values"], dtype=np.float32)
            except Exception as exc:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    E = np.zeros((len(texts), 3072), dtype=np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, (i, v) in enumerate(ex.map(one, enumerate(texts)), 1):
            E[i] = v
            if k % 200 == 0:
                log(f"  {k}/{len(texts)} ({time.time()-t0:.0f}s)")
    np.save(CACHE, E)
    log(f"  эмбеддинги готовы {E.shape} за {time.time()-t0:.0f}s")
    return E


def main():
    texts, y, tr, nA = load_split()
    te = ~tr
    log(f"Proza.ru: authors={nA} texts={len(texts)} (train={tr.sum()}/test={te.sum()})")
    E = normalize(embed_all(texts))
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=SEED).fit(E[tr], y[tr])
    p = clf.predict(E[te])
    top1 = round(float(accuracy_score(y[te], p)), 3)
    f1 = round(float(f1_score(y[te], p, average="macro")), 3)
    log(f"gemini-embedding-001 + LinearSVC: top1={top1} macroF1={f1}")

    prev = json.loads((ROOT / "docs" / "luar_proza.json").read_text())["results_top1_macroF1"]
    out = {
        "dataset": "Proza.ru hard 50auth 50/50 seed42 (сплит идентичен docs/luar_proza.json)",
        "model": f"{MODEL} (через VertexAI, замороженные эмбеддинги 3072-dim, task_type=CLASSIFICATION, вход до {MAX_CHARS} символов)",
        "classifier": "LinearSVC C=1.0 на L2-нормированных эмбеддингах",
        "test_top1": top1,
        "test_macro_f1": f1,
        "reference_same_split": prev,
    }
    (ROOT / "docs" / "vertex_embedding_proza.json").write_text(
        dumps_strict(out, ensure_ascii=False, indent=2), "utf-8")
    log("saved docs/vertex_embedding_proza.json")


if __name__ == "__main__":
    main()
