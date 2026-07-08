"""Supervised Authorship Verification — обучаемый детектор «одна ли рука» (forgery-ML).

Идея: обучить классификатор на ПАРАХ документов (same-author / different-author) из корпуса
50+ авторов. Признаки ПАРЫ — косинусные сходства по нескольким независимым представлениям
(служебные слова, char-3/4-граммы, частые слова) + |разницы| стат-метрик. Это богаче и обучаемее
голого косинуса/unmasking. Валидация: author-disjoint (тест-авторы не видны обучению), отдельно
КРОСС-РЕГИСТР (дневник↔письма), и тест на ФАЛЬШИВКЕ Вырубовой.
"""
from __future__ import annotations
import sys, pathlib, re, random, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from stylo.lang import function_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
RNG = random.Random(42)
FW = sorted(function_words("ru")); FWI = {w: i for i, w in enumerate(FW)}
WORD = re.compile(r"[а-яёА-ЯЁ]+")
CAPW = 40000   # слов на документ для профиля


def toks(t): return WORD.findall(t.lower())


def load_corpus():
    """author -> [тексты книг] из input_clean (бенчмарк-корпус)."""
    base = ROOT / "input_clean"; out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        books = [f.read_text("utf-8", "ignore") for f in sorted(d.glob("*.txt"))]
        books = [b for b in books if len(toks(b)) >= 1500]
        if len(books) >= 2:
            out[d.name] = books
    return out


# глобальные словари признаков (строим на сэмпле)
def build_vocabs(sample_texts):
    c3, c4, wd = Counter(), Counter(), Counter()
    for t in sample_texts:
        s = re.sub(r"\s+", " ", t.lower())[:60000]
        for i in range(len(s) - 3):
            c3[s[i:i+3]] += 1; c4[s[i:i+4]] += 1
        for w in toks(t)[:20000]:
            wd[w] += 1
    return ([g for g,_ in c3.most_common(500)], [g for g,_ in c4.most_common(500)],
            [w for w,_ in wd.most_common(500)])


def profile(text, V3, V4, VW):
    w = toks(text)[:CAPW]; s = re.sub(r"\s+", " ", text.lower())[:CAPW*7]
    def vc(items, vocab, idx):
        v = np.zeros(len(vocab)); c = Counter(items)
        for k, n in c.items():
            j = idx.get(k)
            if j is not None: v[j] = n
        return v / (sum(v) + 1e-9)
    i3 = {g:i for i,g in enumerate(V3)}; i4 = {g:i for i,g in enumerate(V4)}; iw = {g:i for i,g in enumerate(VW)}
    fw = np.zeros(len(FW))
    for t in w:
        j = FWI.get(t)
        if j is not None: fw[j]+=1
    fw /= (len(w)+1e-9)
    c3 = vc((s[i:i+3] for i in range(len(s)-2)), V3, i3)
    c4 = vc((s[i:i+4] for i in range(len(s)-3)), V4, i4)
    wd = vc(w, VW, iw)
    sents = [x for x in re.split(r"[.!?…]+", text) if x.strip()]
    msl = np.mean([len(WORD.findall(x)) for x in sents]) if sents else 0
    ttr = len(set(w[:4000]))/len(w[:4000]) if w else 0
    punct = len(re.findall(r"[,;:—–()«»]", text))/(len(w)/100+1e-9)
    return {"fw":fw,"c3":c3,"c4":c4,"wd":wd,"msl":msl,"ttr":ttr,"punct":punct}


def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def pair_feat(p,q):
    return [cos(p["fw"],q["fw"]),cos(p["c3"],q["c3"]),cos(p["c4"],q["c4"]),cos(p["wd"],q["wd"]),
            abs(p["msl"]-q["msl"]),abs(p["ttr"]-q["ttr"]),abs(p["punct"]-q["punct"])]


def make_pairs(profs_by_author, n_same, n_diff):
    """profs_by_author: author -> [profiles]. Возвращает X,y."""
    X,y=[],[]
    auth=list(profs_by_author)
    same=0
    for a in auth:
        ps=profs_by_author[a]
        for i,j in itertools.combinations(range(len(ps)),2):
            X.append(pair_feat(ps[i],ps[j])); y.append(1); same+=1
            if same>=n_same: break
        if same>=n_same: break
    diff=0
    while diff<n_diff:
        a,b=RNG.sample(auth,2)
        X.append(pair_feat(RNG.choice(profs_by_author[a]),RNG.choice(profs_by_author[b]))); y.append(0); diff+=1
    return np.array(X),np.array(y)


def main():
    print("Загрузка корпуса…",flush=True)
    corpus=load_corpus()
    print(f"авторов с ≥2 книгами: {len(corpus)}",flush=True)
    # author-disjoint split СНАЧАЛА: словари признаков (V3/V4/VW) строим ТОЛЬКО на train-авторах,
    # иначе признаковое пространство подсматривает тест-авторов и «тест-авторы не видны обучению» неверно.
    auth=list(corpus); RNG.shuffle(auth)
    ntr=int(len(auth)*0.7); tr_a,te_a=auth[:ntr],auth[ntr:]
    sample=[b[:30000] for a in tr_a for b in corpus[a][:2]]
    V3,V4,VW=build_vocabs(sample)
    print("Профилирование книг…",flush=True)
    profs={a:[profile(b,V3,V4,VW) for b in corpus[a][:8]] for a in corpus}
    Xtr,ytr=make_pairs({a:profs[a] for a in tr_a},1500,1500)
    Xte,yte=make_pairs({a:profs[a] for a in te_a},600,600)
    clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.08,random_state=42).fit(Xtr,ytr)
    auc=roc_auc_score(yte,clf.predict_proba(Xte)[:,1])
    print(f"\n=== AV-верификатор: author-disjoint тест ===")
    print(f"   AUC (same vs different, тест-авторы не видны обучению) = {auc:.3f}")
    print(f"   baseline (только косинус служебных слов) AUC = {roc_auc_score(yte,Xte[:,0]):.3f}")

    # ── КРОСС-РЕГИСТР: дневник↔письма (same-author) vs cross-author ──
    pp=ROOT/"input_personal"; reg={}
    for d in sorted(p for p in pp.iterdir() if p.is_dir()):
        m=re.match(r"(.+)_(diary|letters)$",d.name)
        if m:
            txt=" ".join(f.read_text("utf-8","ignore") for f in d.glob("*.txt"))
            if len(toks(txt))>=1500: reg.setdefault(m.group(1),{})[m.group(2)]=profile(txt,V3,V4,VW)
    pairs_auth=[a for a in reg if "diary" in reg[a] and "letters" in reg[a]]
    Xc,yc=[],[]
    for a in pairs_auth:
        Xc.append(pair_feat(reg[a]["diary"],reg[a]["letters"])); yc.append(1)         # same author, cross-register
    for a in pairs_auth:
        for b in pairs_auth:
            if a!=b: Xc.append(pair_feat(reg[a]["diary"],reg[b]["letters"])); yc.append(0)
    Xc,yc=np.array(Xc),np.array(yc)
    pc=clf.predict_proba(Xc)[:,1]
    print(f"\n=== КРОСС-РЕГИСТР (дневник↔письма): обобщается ли верификатор? ===")
    if len(set(yc))==2:
        print(f"   AUC на кросс-регистре = {roc_auc_score(yc,pc):.3f} (1=один автор)")
    for a in pairs_auth:
        same_p=clf.predict_proba([pair_feat(reg[a]["diary"],reg[a]["letters"])])[0,1]
        print(f"   {a:12} P(один автор | дневник,письма) = {same_p:.3f}")

    # ── ФАЛЬШИВКА Вырубовой ──
    print(f"\n=== ТЕСТ НА ФАЛЬШИВКЕ Вырубовой ===")
    vy=ROOT/"input_cases/vyrubova"
    if (vy/"vyrubova_diary_fake.txt").exists():
        fake=profile((vy/"vyrubova_diary_fake.txt").read_text("utf-8","ignore"),V3,V4,VW)
        real=profile((vy/"vyrubova_memoir_real.txt").read_text("utf-8","ignore"),V3,V4,VW) if (vy/"vyrubova_memoir_real.txt").exists() else None
        tol=profile(" ".join(f.read_text("utf-8","ignore") for f in (ROOT/"input_clean/tolstoy_an").glob("*.txt")),V3,V4,VW)
        if real is not None:
            print(f"   P(один автор | фальшивка-дневник, НАСТОЯЩИЕ мемуары Вырубовой) = {clf.predict_proba([pair_feat(fake,real)])[0,1]:.3f}")
            print(f"     (низко → метод РАЗЛИЧАЕТ фальшивку и подлинную Вырубову = поймал; высоко → имитация неотличима)")
        print(f"   P(один автор | фальшивка, А.Н.Толстой-проза) = {clf.predict_proba([pair_feat(fake,tol)])[0,1]:.3f}")


if __name__ == "__main__":
    main()
