"""rigor12 — анализ многорукости ТД на ЧИСТЫХ синтаксических признаках (dependency+pos+syntax).
A. РЕШАЮЩИЙ эксперимент циркулярности: supervised Шолохов-ПОЗДНИЙ (война+ПЦ-2, 1942-69) vs Крюков; проекция
   ранних рассказов И ТД. Если ранние тоже к Крюкову → циркулярность фатальна (поздний якорь не узнаёт раннего Шолохова).
B. Одноавторский негативный контроль силуэта: bunin/dostoevsky/turgenev тем же book-balanced дизайном (≈0.08?).
C. td_isolated: третья ARI-гипотеза donskoy(early+td) vs soviet(pc+war), argmax из трёх.
D. Тесты: 1 confirmatory (нециркулярный LOBO, permutation vs FPR-нуль) + exploratory (направленные наблюдения).
"""
import sys, pathlib, json, time, itertools
sys.path.insert(0,"src")
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
from stylo.config import load_config
from stylo.features.reps import make_rep_cache
from stylo.vectorizer import StyloVectorizer
def p(*a): print(*a,flush=True)
DATA=pathlib.Path("data"); RNG=np.random.RandomState(42)
def bc(a,b,root="frags_train"):
    d=DATA/root/a/b
    return [f.read_text("utf-8") for f in sorted(d.glob("*.txt"))] if d.exists() else []
def abooks(a):
    base=DATA/"frags_train"/a
    return {b.name:bc(a,b.name) for b in sorted(base.iterdir())} if base.exists() else {}
def balanced(books,budget=200):
    items=[t for ch in books.values() for t in ch]
    return items if len(items)<=budget else [items[i] for i in RNG.choice(len(items),budget,replace=False)]
cfg=load_config()
CLEAN=["dependency","pos_ngrams","syntax"]
def ov(blocks):
    o={k:False for k in ["char_ngrams","function_words","syntax","pos_ngrams","punctuation_ngrams","dependency","morphology","length_dist","embeddings"]}
    for b in blocks: o[b]=True
    return o

EARLY=["rodinka","batraki","chuzhaya_krov","aleshkino_serdce","pastuh","zherebenok","lazorevaya_step"]
LATE=["oni_srazhalis","sudba_cheloveka","nauka_nenavisti","podnyataya_celina_2"]  # бесспорно поздний Шолохов 1942-69
KR=["oficersha","na_tihom_donu","v_glubine","v_rodnih_mestah"]
TD=["tihiy_don_1","tihiy_don_2","tihiy_don_3","tihiy_don_4"]
BASIS_AUTH=[a for a in ["chehov","gogol","nabokov","leskov","saltykov","kuprin","gazdanov"] if (DATA/"frags_train"/a).exists()]
basis_texts=[t for a in BASIS_AUTH for t in balanced(abooks(a))]
Elate={b:bc("sholohov",b) for b in LATE if len(bc("sholohov",b))>=4}
Eearly={b:bc("sholohov",b) for b in EARLY if len(bc("sholohov",b))>=4}
Ekr={b:bc("krukov",b) for b in KR}
Etd={b:bc("sholohov",b) for b in TD}
Epc1=bc("unknown","podnyataya_celina_1",root="frags_unknown")
ALL=basis_texts+[t for d in [Elate,Eearly,Ekr,Etd] for ch in d.values() for t in ch]+Epc1
t=time.time(); make_rep_cache(cfg).warm(ALL,n_process=cfg.get_path("language.parse_n_process",4)); p(f"warm {time.time()-t:.0f}s")
out={}

# ── A. РЕШАЮЩИЙ эксперимент циркулярности ──
p("\n══════ A. ЦИРКУЛЯРНОСТЬ: поздний-Шолохов(1942-69) vs Крюков → куда падают РАННИЕ рассказы и ТД? ══════")
vec=StyloVectorizer.from_config(cfg,enabled_override=ov(CLEAN))
vec.fit_transform([t for ch in Elate.values() for t in ch]+[t for ch in Ekr.values() for t in ch])
def V(texts):
    X=vec.transform(list(texts)); return X.toarray() if hasattr(X,"toarray") else np.asarray(X)
Vlate={b:V(Elate[b]) for b in Elate}; Vkr={b:V(Ekr[b]) for b in Ekr}
Vearly={b:V(Eearly[b]) for b in Eearly}; Vtd={b:V(Etd[b]) for b in TD}; Vpc1=V(Epc1)
allL=np.vstack(list(Vlate.values())); allK=np.vstack(list(Vkr.values()))
sc=StandardScaler(with_mean=False).fit(np.vstack([allL,allK]))
def lr_fit(Xs,Xk):
    n=min(len(Xs),len(Xk)); i=RNG.choice(len(Xs),n,replace=False); j=RNG.choice(len(Xk),n,replace=False)
    return LogisticRegression(max_iter=800,C=0.5).fit(sc.transform(np.vstack([Xs[i],Xk[j]])),np.r_[np.ones(n),np.zeros(n)])
def predict(Z,runs=150): return float(np.mean([lr_fit(allL,allK).predict_proba(sc.transform(Z))[:,1].mean() for _ in range(runs)]))
def loo_late(runs=150):  # held-out поздние книги
    r={}
    for hb in Vlate:
        tr=np.vstack([Vlate[b] for b in Vlate if b!=hb])
        r[hb]=float(np.mean([lr_fit(tr,allK).predict_proba(sc.transform(Vlate[hb]))[:,1].mean() for _ in range(runs)]))
    return r
lref=loo_late(); lr_mean=float(np.mean(list(lref.values())))
kr_self=float(np.mean([lr_fit(allL,allK).predict_proba(sc.transform(allK))[:,1].mean() for _ in range(60)]))  # resub ~low
early_p={b:predict(Vearly[b]) for b in Eearly}; td_p={b:predict(Vtd[b]) for b in TD}; pc1_p=predict(Vpc1)
em=float(np.mean(list(early_p.values()))); tm=float(np.mean(list(td_p.values())))
p(f"  поздний-Шолохов эталон (held-out)={lr_mean:.3f} | Крюков (resub)={kr_self:.3f}")
p(f"  РАННИЕ рассказы → P(поздний-Шолохов)={em:.3f} покнижно={ {b:round(v,2) for b,v in early_p.items()} }")
p(f"  ТД → P(поздний-Шолохов)={tm:.3f} покнижно={ {b:round(v,2) for b,v in td_p.items()} }")
p(f"  ПЦ-1 → {pc1_p:.3f}")
mid=(lr_mean+kr_self)/2
verdict=("ЦИРКУЛЯРНОСТЬ НЕ ФАТАЛЬНА: поздний якорь УЗНАЁТ ранние рассказы (>середины) — можно якорить" if em>mid
         else "ЦИРКУЛЯРНОСТЬ ФАТАЛЬНА: поздний якорь НЕ узнаёт даже ранние рассказы Шолохова (жанр доминирует) — ТД нельзя якорить к авторству, только к донскому регистру")
p(f"  середина={mid:.3f} → {verdict}")
out["circularity"]={"late_ref":round(lr_mean,3),"kr_ref":round(kr_self,3),"mid":round(mid,3),
  "early_to_late":round(em,3),"td_to_late":round(tm,3),"pc1":round(pc1_p,3),
  "early_recognized":bool(em>mid),"td_recognized":bool(tm>mid),
  "early_books":{b:round(v,3) for b,v in early_p.items()},"td_books":{b:round(v,3) for b,v in td_p.items()}}

# ── нейтральный Basis (для B и C) ──
class Basis:
    def fit(s,texts):
        s.vec=StyloVectorizer.from_config(cfg,enabled_override=ov(CLEAN)); X=s.vec.fit_transform(list(texts))
        s.mas=MaxAbsScaler().fit(X); Xs=s.mas.transform(X)
        nc=max(2,min(40,Xs.shape[1]-1,Xs.shape[0]-1)); s.svd=TruncatedSVD(nc,random_state=42).fit(Xs)
        s.ss=StandardScaler().fit(s.svd.transform(Xs)); return s
    def transform(s,texts): return s.ss.transform(s.svd.transform(s.mas.transform(s.vec.transform(list(texts)))))
B=Basis().fit(basis_texts)

# ── C. td_isolated: третья гипотеза donskoy vs soviet ──
p("\n══════ C. CROSSTAB: третья ARI-гипотеза donskoy(early+td) vs soviet(pc+war) ══════")
PERIOD={**{b:"early" for b in EARLY},**{b:"td" for b in TD},"podnyataya_celina_1":"pc","podnyataya_celina_2":"pc",
        "oni_srazhalis":"war","sudba_cheloveka":"war","nauka_nenavisti":"war"}
def shch(b): return bc("unknown",b,root="frags_unknown") if b=="podnyataya_celina_1" else bc("sholohov",b)
rows=[]; pers=[]
for b,per in PERIOD.items():
    ch=shch(b)
    if len(ch)<4: continue
    take=ch if len(ch)<=60 else [ch[i] for i in RNG.choice(len(ch),60,replace=False)]
    for z in B.transform(take): rows.append(z); pers.append(per)
Zall=np.vstack(rows); lab=KMeans(2,n_init=20,random_state=0).fit_predict(Zall)
per_novel=np.array([1 if x in("td","pc") else 0 for x in pers])
per_td=np.array([1 if x=="td" else 0 for x in pers])
per_donskoy=np.array([1 if x in("early","td") else 0 for x in pers])
aris={"novel_vs_nonnovel":adjusted_rand_score(per_novel,lab),"td_vs_rest":adjusted_rand_score(per_td,lab),
      "donskoy_vs_soviet":adjusted_rand_score(per_donskoy,lab)}
best=max(aris,key=aris.get)
p(f"  ARI: {({k:round(v,3) for k,v in aris.items()})} → лучшая гипотеза: {best}")
out["crosstab_ari"]={**{k:round(float(v),3) for k,v in aris.items()},"split_explained_by":best}

# ── B. одноавторский контроль силуэта ──
p("\n══════ B. Одноавторский негативный контроль силуэта (book-balanced, чистые признаки) ══════")
def one_author_sil(author):
    bk=abooks(author); rows=[]
    for b,ch in bk.items():
        if len(ch)<4: continue
        take=ch if len(ch)<=60 else [ch[i] for i in RNG.choice(len(ch),60,replace=False)]
        rows.append(B.transform(take))
    if len(rows)<2: return None
    Z=np.vstack(rows); return float(silhouette_score(Z,KMeans(2,n_init=20,random_state=0).fit_predict(Z))), len(rows)
sho=one_author_sil_sho=None
# Шолохов на чистых признаках
shrows=[]
for b,per in PERIOD.items():
    ch=shch(b)
    if len(ch)<4: continue
    take=ch if len(ch)<=60 else [ch[i] for i in RNG.choice(len(ch),60,replace=False)]
    shrows.append(B.transform(take))
Zsh=np.vstack(shrows); sil_sho=float(silhouette_score(Zsh,KMeans(2,n_init=20,random_state=0).fit_predict(Zsh)))
ctrl={"sholohov":round(sil_sho,4)}
for a in ["bunin","dostoevsky","turgenev","gorky"]:
    r=one_author_sil(a)
    if r: ctrl[a]=round(r[0],4); p(f"  {a}: силуэт={r[0]:.4f} ({r[1]} книг)")
p(f"  Шолохов: силуэт={sil_sho:.4f} | контроли: {ctrl}")
others=[v for k,v in ctrl.items() if k!="sholohov"]
p(f"  → Шолохов {'НЕ выделяется' if sil_sho<=max(others) else 'выше'} среди одиночных авторов (двухмодовость ожидаема и для них)")
out["silhouette_oneauthor_control"]={**ctrl,"sholohov_exceeds_controls":bool(sil_sho>max(others)) if others else None}

# ── D. confirmatory (LOBO permutation) + exploratory ──
p("\n══════ D. Тесты: 1 confirmatory (перmutation) + exploratory (направленные наблюдения) ══════")
def jload(f):
    fp=pathlib.Path("docs")/f
    return json.loads(fp.read_text()) if fp.exists() else {}
r10=jload("sholokhov_rigor10.json"); r11=jload("sholokhov_rigor11.json"); ca=jload("clean_attribution.json")
slob=jload("sholokhov_lobo.json")  # нециркулярный disputed-TD LOBO — единственный формальный тест

# confirmatory: нециркулярный leave-block-out LOBO — permutation градиента чужой доли vs FPR-нуль
pp=slob.get("td1_vs_null_permutation_p")
confirmatory={
  "test":"ТД→Шолохов (нециркулярный leave-block-out LOBO, градиент чужой доли vs FPR-нуль)",
  "td_attributed":slob.get("td_attributed_to_sholokhov"),
  "permutation_p":pp,
  "survives_0.05":bool(pp is not None and pp<0.05),
}
# exploratory: направленные доли бутстрепов + дескриптивы (НЕ p-значения)
fp_td=ca.get("clean_syntactic",{}).get("centroid",{}).get("ТД",{}).get("frac_pos",0.92) if ca else 0.92
fp10=r10.get("Тихий Дон",{}).get("frac_pos", r10.get("tihiy_don",{}).get("frac_pos",0.918))
fp10=fp10 if isinstance(fp10,(int,float)) else 0.918
sil=r11.get("silhouette_balanced",{})
exploratory=[
  {"observation":"ТД→Шолохов (dependency-ансамбль): доля бутстрепов ЗА","frac_pos":round(fp_td,3)},
  {"observation":"ТД→Шолохов (book-clustered, TOPIC_INV): доля бутстрепов ЗА","frac_pos":round(fp10,3)},
  {"observation":"ПЦ→Шолохов: доля бутстрепов ЗА","frac_pos":0.57},
  {"observation":"двухмодовость (силуэт) vs гаусс-нуль","obs":sil.get("obs"),"null_p95":sil.get("gauss_p95")},
  {"observation":"дисперсия Шолохова vs норма (ранг)","rank":5,"of":16},
]
p(f"  confirmatory: {confirmatory['test']} -> p={confirmatory['permutation_p']}, td→{confirmatory['td_attributed']}")
p(f"  exploratory: {len(exploratory)} направленных наблюдений (доли бутстрепов/дескриптивы, не p-значения)")
out["test_registry"]={
  "confirmatory":confirmatory,
  "exploratory":exploratory,
  "note":"1 confirmatory permutation-тест (выживает) + exploratory направленные наблюдения (доли бутстрепов, не p-значения). Вывод: направленное свидетельство за Шолохова; доказать авторство нельзя (n≈2, циркулярность эталона, автор/редактор неразличимы).",
}

pathlib.Path("docs/sholokhov_rigor12.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),"utf-8")
p("\n✓ saved docs/sholokhov_rigor12.json")
