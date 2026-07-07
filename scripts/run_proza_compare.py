"""Сравнение нашего решения с СОВРЕМЕННЫМИ инструментами на русском Proza.ru (тот же 50-авторский сплит).
Methods:
  • char-SVM (классический baseline ≈ stylo/Burrows-style)
  • LUAR (gabrielloiseau/LUAR-MUD-sentence-transformers) — SOTA universal author embedding
  • ruBERT-sentence (DeepPavlov/rubert-base-cased-sentence) — русские нейро-эмбеддинги
  • (наш полный ансамбль — из docs/proza_ru.json, считается отдельно scripts/run_proza_ru.py)
Все — LinearSVC, ИДЕНТИЧНЫЙ train/test (seed=42). Эмбеддинги кешируются.

Запуск:  python scripts/run_proza_compare.py
"""
from __future__ import annotations
import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score
import warnings; warnings.filterwarnings("ignore")
SEED = 42
ROOT = pathlib.Path(__file__).resolve().parents[1]
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

def emb_cache(model_id, texts, st=True):
    cf = ROOT / "data" / f"emb_proza_{model_id.replace('/','_')}.npy"
    if cf.exists(): return np.load(cf)
    t = time.time()
    if st:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_id, device="cpu")
        E = m.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    else:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModel.from_pretrained(model_id).eval()
        torch.set_num_threads(4); out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=256, return_tensors="pt")
                h = mdl(**enc).last_hidden_state; mask = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((h*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy())
        E = np.vstack(out)
    E = E.astype(np.float32); np.save(cf, E); log(f"  {model_id} эмбеддинги {E.shape} ({time.time()-t:.0f}s)")
    return E

def evalm(Xtr, Xte, ytr, yte):
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=SEED).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return round(float(accuracy_score(yte, p)), 3), round(float(f1_score(yte, p, average="macro")), 3)

def main():
    texts, y, tr, nA = load_split()
    te = ~tr; ytr, yte = y[tr], y[te]
    log(f"Proza.ru сравнение: авторов={nA} текстов={len(texts)} (train={tr.sum()}/test={te.sum()})")
    res = {}

    # 1. char-SVM (классический baseline ≈ stylo)
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 5), n_features=2**18, alternate_sign=False, norm=None)
    tf = TfidfTransformer(sublinear_tf=True); X = tf.fit_transform(hv.transform([texts[i] for i in np.where(tr)[0]]))
    Xt = tf.transform(hv.transform([texts[i] for i in np.where(te)[0]]))
    res["char-SVM (≈stylo)"] = evalm(X, Xt, ytr, yte)
    log(f"  char-SVM: {res['char-SVM (≈stylo)']}")

    # 2-4. нейро-модели (русские — первыми; LUAR англ./медленный — последним)
    for tag, mid, st in [("ruBERT-tiny2 (рус.)", "cointegrated/rubert-tiny2", False),
                         ("ruBERT-sentence (рус.)", "DeepPavlov/rubert-base-cased-sentence", False),
                         ("LUAR (SOTA author-emb, англ.)", "gabrielloiseau/LUAR-MUD-sentence-transformers", True)]:
        try:
            E = normalize(emb_cache(mid, texts, st))
            res[tag] = evalm(E[tr], E[te], ytr, yte); log(f"  {tag}: {res[tag]}")
        except Exception as e:
            log(f"  {tag}: ПРОПУЩЕН ({str(e)[:70]})"); res[tag] = None

    # наш полный ансамбль — из docs/proza_ru.json
    pj = ROOT / "docs" / "proza_ru.json"
    ours = json.loads(pj.read_text())["channels"].get("АНСАМБЛЬ") if pj.exists() else None
    if ours: res["НАШ ансамбль (char+word+синтаксис+fw+морф)"] = (ours["top1"], ours["macro_f1"])

    log("\n=== СРАВНЕНИЕ на Proza.ru (top-1 / macro-F1) ===")
    for k, v in sorted({k: v for k, v in res.items() if v}.items(), key=lambda x: -x[1][0]):
        log(f"  {k:46} {v[0]:.3f} / {v[1]:.3f}")
    out = {"dataset": "Proza.ru hard (50 авторов, 50/50 split)", "comparison": {k: (list(v) if v else None) for k, v in res.items()},
           "note": "Сравнение нашего решения с современными инструментами (LUAR SOTA author-emb, ruBERT) и классическим baseline (char-SVM≈stylo) на ИДЕНТИЧНОМ русском сплите."}
    (ROOT / "docs" / "proza_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    log("\n✓ saved docs/proza_compare.json")

if __name__ == "__main__":
    main()
