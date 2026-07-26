"""КОНСИСТЕНТНОСТЬ ШОЛОХОВА: насколько внутренне когерентен его корпус в сравнении с авторами,
чьё авторство НЕОСПОРИМО. Если Шолохов писал сам — его книги должны быть так же «близки» друг к другу,
как у бесспорного одиночки; если корпус склеен из рук — он должен быть аномально РАЗНОРОДНЫМ
(как коллективный Козьма Прутков).

Метрики (на чистом ансамбле dep+pos+syntax, нейтральный leak-free базис):
  1. within-author силуэт (book-balanced, 2-кластер): НИЗКО = когерентный одиночка, ВЫСОКО = распадается.
  2. self-consistency: доля held-out книг автора, атрибутируемых обратно САМОМУ себе (в панели авторов).

Запуск:  python scripts/run_consistency.py
"""
from __future__ import annotations
import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings; warnings.filterwarnings("ignore")
from stylo.config import load_config
from stylo.features.reps import make_rep_cache
from stylo.vectorizer import StyloVectorizer
def log(*a): print(*a, flush=True)
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "frags_train"; RNG = np.random.RandomState(42)
CLEAN = ["dependency", "pos_ngrams", "syntax"]
def books(a):
    d = DATA / a
    return {b.name: [f.read_text("utf-8") for f in sorted(b.glob("*.txt"))] for b in sorted(d.iterdir())} if d.exists() else {}
def cap(ch, n=60): return ch if len(ch) <= n else [ch[i] for i in sorted(RNG.choice(len(ch), n, replace=False))]
def bal(bk, bud=200):
    items = [t for ch in bk.values() for t in ch]
    return items if len(items) <= bud else [items[i] for i in RNG.choice(len(items), bud, replace=False)]
cfg = load_config()
def ov(bl):
    o = {k: False for k in ["char_ngrams","function_words","syntax","pos_ngrams","punctuation_ngrams","dependency","morphology","length_dist","embeddings"]}
    for b in bl: o[b] = True
    return o

# панель: Шолохов + НЕОСПОРИМЫЕ одиночки + коллектив (Прутков) как верхний маркер
INDISPUTABLE = ["bunin","dostoevsky","turgenev","chehov","gogol","tolstoy","kuprin","gorky",
                "grin","tolstoy_an","babel","pilnyak","platonov","nabokov","leskov","saltykov"]
PANEL = ["sholohov"] + [a for a in INDISPUTABLE if (DATA/a).exists()] + (["prutkov"] if (DATA/"prutkov").exists() else [])
BASIS_AUTH = [a for a in ["chehov","gogol","nabokov","leskov","saltykov","kuprin","gazdanov"] if (DATA/a).exists()]

allbk = {a: books(a) for a in set(PANEL + BASIS_AUTH)}
basis_texts = [t for a in BASIS_AUTH for t in bal(allbk[a])]
warm = basis_texts + [t for a in PANEL for ch in allbk[a].values() for t in cap(ch, 60)]
t = time.time(); make_rep_cache(cfg).warm(warm, n_process=2); log(f"rep-кэш прогрет {time.time()-t:.0f}s")
vb = StyloVectorizer.from_config(cfg, enabled_override=ov(CLEAN)); Xb = vb.fit_transform(basis_texts)
mas = MaxAbsScaler().fit(Xb); Xs = mas.transform(Xb); nc = max(2, min(40, Xs.shape[1]-1, Xs.shape[0]-1))
svd = TruncatedSVD(nc, random_state=42).fit(Xs); ss = StandardScaler().fit(svd.transform(Xs))
def B(texts): return ss.transform(svd.transform(mas.transform(vb.transform(list(texts)))))

def author_centroid(bk):
    """Equal-book centroid: chunks -> one book row, then books -> author."""
    rows = [B(cap(ch, 60)).mean(0) for ch in bk.values() if ch]
    return np.mean(rows, axis=0) if rows else None

# центроиды авторов (для self-consistency)
cent = {a: author_centroid(allbk[a]) for a in PANEL}
def within_sil(bk):
    rows = [B(cap(ch)) for ch in bk.values() if len(ch) >= 4]
    if len(rows) < 2: return None
    Z = np.vstack(rows); return float(silhouette_score(Z, KMeans(2, n_init=20, random_state=0).fit_predict(Z)))
def self_rate(a):
    bk = allbk[a]; ok = 0; tot = 0
    for b, ch in bk.items():
        if len(ch) < 4: continue
        z = B(cap(ch)).mean(0)
        # ближайший центроид среди панели, но центроид САМОГО автора пересчитан БЕЗ этой книги
        cself = author_centroid({k: v for k, v in bk.items() if k != b}) if len(bk) > 1 else cent[a]
        dists = {a2: (np.linalg.norm(z - (cself if a2 == a else cent[a2]))) for a2 in PANEL}
        ok += (min(dists, key=dists.get) == a); tot += 1
    return (ok, tot)

rows = []
log("\n%-14s %10s %14s %8s" % ("автор", "силуэт↓", "книг→к себе", "книг"))
for a in PANEL:
    s = within_sil(allbk[a])
    if s is None: continue
    ok, tot = self_rate(a)
    rows.append({"author": a, "within_silhouette": round(s, 4), "self_rate": f"{ok}/{tot}", "n_books": tot})
    log("%-14s %10.4f %14s %8d" % (a, s, f"{ok}/{tot}", tot))

rows_sorted = sorted(rows, key=lambda r: r["within_silhouette"])
sh = next(r for r in rows_sorted if r["author"] == "sholohov")
rank = rows_sorted.index(sh) + 1
singles = [r for r in rows_sorted if r["author"] not in ("sholohov", "prutkov")]
sing_sils = [r["within_silhouette"] for r in singles]
prut = next((r for r in rows_sorted if r["author"] == "prutkov"), None)
log(f"\nШолохов: силуэт {sh['within_silhouette']} (ранг {rank}/{len(rows_sorted)} по разнородности; {rank-1} одиночек когерентнее, {len(rows_sorted)-rank} разнороднее)")
log(f"диапазон неоспоримых одиночек: [{min(sing_sils):.3f}–{max(sing_sils):.3f}], медиана {np.median(sing_sils):.3f}")
if prut: log(f"коллектив Прутков (контроль): {prut['within_silhouette']} (× {prut['within_silhouette']/sh['within_silhouette']:.1f} от Шолохова)")
verdict = ("В НОРМЕ одиночек — Шолохов когерентен как бесспорный автор; «много рук» не подтверждается"
           if sh["within_silhouette"] <= np.percentile(sing_sils, 75) else
           "ВЫШЕ нормы — повышенная разнородность, требует объяснения (жанр/период?)")
log(f"ВЫВОД: {verdict}")

out = {"metric": "within-author силуэт (book-balanced, чистый ансамбль dep+pos+syntax) — НИЗКО=когерентный одиночка",
       "train_centroid_weighting": "equal_book_after_within_book_chunk_mean",
       "sholokhov_silhouette": sh["within_silhouette"], "sholokhov_rank": rank, "n_panel": len(rows_sorted),
       "indisputable_range": [round(min(sing_sils), 3), round(max(sing_sils), 3)], "indisputable_median": round(float(np.median(sing_sils)), 3),
       "collective_prutkov": prut["within_silhouette"] if prut else None,
       "verdict": verdict, "per_author": rows_sorted}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "consistency.json").write_text(dumps_strict(out, ensure_ascii=False, indent=2), "utf-8")
log("\n✓ saved docs/consistency.json")
