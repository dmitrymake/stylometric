"""Честное сравнение char-SVM (≈stylo) с ГОТОВЫМИ современными нейро authorship/style-эмбеддерами
на ИДЕНТИЧНОМ (но УРЕЗАННОМ для скорости на CPU) русском сплите Proza.ru hard.

Сравнение берёт СИЛЬНЫЕ готовые многоязычные AA/style-модели (XLM-R-large, контрастивные),
реально покрывающие русский — не слабое нейро (ruBERT-tiny2 zero-shot, англ. LUAR 0.387):
  • Blablablab/multilingual-style-representation — multilingual authorship representation (arXiv 2509.16531)
  • Hieuman/erlas                              — authorship representation (от автора датасета proza_ru_hard)
  • StyleDistance/mstyledistance               — multilingual style embeddings (arXiv 2502.15168) [опц., медленная]

XLM-R-large на CPU дорог (~3-6 с/текст), потому:
  • PER урезан (env PER, по умолч. 14 текстов/автор) — char-SVM ПЕРЕсчитывается на ТОМ ЖЕ подмножестве (честно).
  • MAXLEN=128, потоков 8 (вежливо к параллельному /loop на общей машине).
  • Эмбеддинги кешируются per-model; json дописывается ПОСЛЕ КАЖДОЙ модели (частичный результат не теряется).

Запуск:  .venv/bin/python scripts/experemental/run_ru_neural_compare.py
         PER=20 MODELS=blabla,erlas .venv/bin/python scripts/experemental/run_ru_neural_compare.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, json, time, pathlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score
import warnings; warnings.filterwarnings("ignore")

SEED = 42
ROOT = pathlib.Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "external" / "proza_ru_hard.parquet"
NAUTH = 50
PER = int(os.environ.get("PER", "14"))      # текстов/автор (урезано для CPU; char-SVM считается на этом же срезе)
MAXLEN = int(os.environ.get("MAXLEN", "128"))
OUT = ROOT / "docs" / "proza_neural_compare.json"
def log(*a): print(*a, flush=True)

ALL_MODELS = {
    "blabla": ("multilingual-style-rep (XLM-R-L, AA)", "Blablablab/multilingual-style-representation"),
    "erlas":  ("Hieuman/erlas (AA-rep)",                "Hieuman/erlas"),
    "mstyle": ("mStyleDistance (XLM-R-L, style)",       "StyleDistance/mstyledistance"),
}
WANT = os.environ.get("MODELS", "blabla,erlas").split(",")

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

def emb_cache_st(model_id, texts):
    cf = ROOT / "data" / f"emb_proza_{model_id.replace('/','_')}_p{PER}_m{MAXLEN}.npy"
    if cf.exists(): return np.load(cf)
    from sentence_transformers import SentenceTransformer
    t = time.time()
    m = SentenceTransformer(model_id, device="cpu", trust_remote_code=True)
    try: m.max_seq_length = MAXLEN
    except Exception: pass
    E = m.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(cf, E); log(f"  {model_id}: эмбеддинги {E.shape} за {time.time()-t:.0f}s")
    return E

def evalm(Xtr, Xte, ytr, yte):
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=SEED).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return round(float(accuracy_score(yte, p)), 3), round(float(f1_score(yte, p, average="macro")), 3)

def dump(res, nA, ntexts):
    out = {
        "dataset": f"Proza.ru hard (Hieuman/proza_ru_hard); {nA} авторов, PER={PER} (урезано для CPU), 50/50, seed=42",
        "protocol": f"LinearSVC поверх L2-norm эмбеддингов; char-SVM = char_wb(2-5)+TFIDF; MAXLEN={MAXLEN}; char-SVM ПЕРЕсчитан на ТОМ ЖЕ срезе",
        "n_texts": ntexts,
        "comparison_top1_macroF1": {k: (list(v) if v else None) for k, v in res.items()},
        "note": "Сильные готовые многоязычные AA/style-эмбеддеры (XLM-R-L, контрастивные) vs char-stylo на русском. PER урезан — числа чуть шумнее полного (PER=60) среза, где char-SVM=0.881; смотреть на РАЗРЫВ, не на абсолют.",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")

def main():
    import torch; torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    texts, y, tr, nA = load_split()
    te = ~tr; ytr, yte = y[tr], y[te]
    log(f"Proza.ru: авторов={nA} текстов={len(texts)} PER={PER} MAXLEN={MAXLEN} (train={tr.sum()}/test={te.sum()}); потоков={os.environ['OMP_NUM_THREADS']}")
    res = {}

    # char-SVM на ЭТОМ ЖЕ урезанном срезе (честный якорь; на полном PER=60 он = 0.881)
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 5), n_features=2**18, alternate_sign=False, norm=None)
    tf = TfidfTransformer(sublinear_tf=True)
    Xc = tf.fit_transform(hv.transform([texts[i] for i in np.where(tr)[0]]))
    Xct = tf.transform(hv.transform([texts[i] for i in np.where(te)[0]]))
    res["char-SVM (≈stylo)"] = evalm(Xc, Xct, ytr, yte); log(f"  char-SVM: {res['char-SVM (≈stylo)']}"); dump(res, nA, len(texts))

    best_E = None; best_tag = None
    for key in WANT:
        if key not in ALL_MODELS: log(f"  (неизвестная модель '{key}', пропуск)"); continue
        tag, mid = ALL_MODELS[key]
        try:
            E = emb_cache_st(mid, texts); En = normalize(E)
            res[tag] = evalm(En[tr], En[te], ytr, yte); log(f"  {tag}: {res[tag]}")
            if best_tag is None or res[tag][0] > res[best_tag][0]: best_E, best_tag = En, tag
        except Exception as e:
            log(f"  {tag}: ПРОПУЩЕН ({type(e).__name__}: {str(e)[:90]})"); res[tag] = None
        dump(res, nA, len(texts))  # дописываем после каждой модели

    # char ⊕ лучший нейро — даёт ли готовое нейро что-то СВЕРХ char-n-грамм
    if best_E is not None:
        from scipy.sparse import hstack, csr_matrix
        Xtr2 = hstack([Xc, csr_matrix(best_E[tr])]).tocsr(); Xte2 = hstack([Xct, csr_matrix(best_E[te])]).tocsr()
        res[f"char ⊕ {best_tag}"] = evalm(Xtr2, Xte2, ytr, yte); log(f"  char ⊕ {best_tag}: {res[f'char ⊕ {best_tag}']}"); dump(res, nA, len(texts))

    log("\n=== СРАВНЕНИЕ на Proza.ru hard (top-1 / macro-F1), один срез/протокол ===")
    for k, v in sorted({k: v for k, v in res.items() if v}.items(), key=lambda x: -x[1][0]):
        log(f"  {k:42} {v[0]:.3f} / {v[1]:.3f}")
    log(f"\n✓ saved {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
