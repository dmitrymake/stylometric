"""Decisive test: does LUAR (contrastive author-embedding, EN-trained) beat char-SVM on RUSSIAN Proza.ru?
Reuses identical 50-author 50/50 split (seed=42). Thread-pinned for shared box.
Run:  OMP_NUM_THREADS=4 .venv/bin/python scripts/run_luar_proza.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys, json, time, pathlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score
import warnings; warnings.filterwarnings("ignore")
SEED = 42
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
PARQUET = ROOT / "data" / "external" / "proza_ru_hard.parquet"
NAUTH, PER = 50, 60
def log(*a): print(*a, flush=True)

def load_split():
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
        for t in arr: texts.append(t); labels.append(row["authorIDs"])
    A = sorted(set(labels)); aidx = {a: i for i, a in enumerate(A)}
    y = np.array([aidx[a] for a in labels])
    tr = np.zeros(len(y), bool)
    for a in range(len(A)):
        idx = np.where(y == a)[0]; rng.shuffle(idx); tr[idx[: len(idx)//2]] = True
    return texts, y, tr, len(A)

def evalm(Xtr, Xte, ytr, yte):
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=SEED).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return round(float(accuracy_score(yte, p)), 3), round(float(f1_score(yte, p, average="macro")), 3)

def luar_encode(texts):
    cf = ROOT / "data" / "emb_proza_LUAR-MUD.npy"
    if cf.exists():
        log("  LUAR cache hit"); return np.load(cf)
    import torch
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    t = time.time()
    m = SentenceTransformer("gabrielloiseau/LUAR-MUD-sentence-transformers", device="cpu")
    E = m.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(cf, E); log(f"  LUAR encoded {E.shape} in {time.time()-t:.0f}s")
    return E

def main():
    texts, y, tr, nA = load_split()
    te = ~tr; ytr, yte = y[tr], y[te]
    log(f"Proza.ru: authors={nA} texts={len(texts)} (train={tr.sum()}/test={te.sum()})")
    res = {}

    # char-SVM reference
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 5), n_features=2**18, alternate_sign=False, norm=None)
    tf = TfidfTransformer(sublinear_tf=True)
    Xtr = tf.fit_transform(hv.transform([texts[i] for i in np.where(tr)[0]]))
    Xte = tf.transform(hv.transform([texts[i] for i in np.where(te)[0]]))
    res["char-SVM"] = evalm(Xtr, Xte, ytr, yte); log(f"  char-SVM: {res['char-SVM']}")
    Pchar_tr, Pchar_te = Xtr, Xte  # keep for fusion

    # rubert-tiny2 (cached)
    rb = ROOT / "data" / "emb_proza_cointegrated_rubert-tiny2.npy"
    Erb = None
    if rb.exists():
        Erb = normalize(np.load(rb))
        res["ruBERT-tiny2"] = evalm(Erb[tr], Erb[te], ytr, yte); log(f"  ruBERT-tiny2: {res['ruBERT-tiny2']}")

    # LUAR — the actual contrastive author-embedding SOTA
    E = luar_encode(texts); En = normalize(E)
    res["LUAR-MUD (author-emb)"] = evalm(En[tr], En[te], ytr, yte); log(f"  LUAR-MUD: {res['LUAR-MUD (author-emb)']}")

    # late-fusion char + LUAR (decision-function z-scored concat via LinearSVC on stacked probas)
    from scipy.sparse import hstack, csr_matrix
    Xtr_f = hstack([Pchar_tr, csr_matrix(En[tr])]).tocsr()
    Xte_f = hstack([Pchar_te, csr_matrix(En[te])]).tocsr()
    res["char+LUAR concat"] = evalm(Xtr_f, Xte_f, ytr, yte); log(f"  char+LUAR concat: {res['char+LUAR concat']}")

    log("\n=== RESULT (top-1 / macro-F1) ===")
    for k, v in sorted(res.items(), key=lambda x: -x[1][0]):
        log(f"  {k:28} {v[0]:.3f} / {v[1]:.3f}")
    (ROOT / "docs" / "luar_proza.json").write_text(
        dumps_strict({"dataset": "Proza.ru hard 50auth 50/50 seed42", "results_top1_macroF1": {k: list(v) for k, v in res.items()}}, ensure_ascii=False, indent=2), "utf-8")
    log("\nsaved docs/luar_proza.json")

if __name__ == "__main__":
    main()
