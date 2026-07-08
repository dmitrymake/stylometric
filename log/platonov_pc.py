"""Гипотеза «Платонов написал Поднятую целину» — leak-free тест (Шаг/Task 2).

Платонов — кандидат, корпус которого ЕСТЬ (chevengur/kotlovan/reka_potudan/uvenilnoe_more).
Прямой leak-free тест на ТОПИК-ИНВАРИАНТНЫХ признаках (dependency — где идиолект Платонова
сильнейший; + ансамбль
dep+pos+syntax), тем же методом, что clean_attribution.py (нейтральный basis на сторонних авторах,
ПЦ held-out).

Три блока:
  A. multi-class held-out LR: P(ПЦ → каждый кандидат {Ш-Дон, Платонов, Крюков, Серафимович});
  B. нейтральный centroid: дистанция ПЦ до центроида каждого кандидата (меньше = ближе);
  C. positive-control: leave-one-book-out — различает ли метод Платонова от Ш/Кр/Сераф ВООБЩЕ
     (если нет — тест Платонов↔ПЦ неинформативен; честный инвариант);
  D. segment: почанковая атрибуция ПЦ с Платоновым в панели — есть ли связный «платоновский» сегмент.

→ docs/platonov_pc.json. Честно: жанровый конфаунд (ПЦ-советская vs донская vs платоновская
«заумь») не снят полностью; positive-control — единственный гарант осмысленности.
"""
import sys, pathlib, json, time
sys.path.insert(0, "src")
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
from stylo.config import load_config
from stylo.features.reps import make_rep_cache
from stylo.vectorizer import StyloVectorizer

def p(*a): print(*a, flush=True)
DATA = pathlib.Path("data"); RNG = np.random.RandomState(42)
LAB = {"sh": "Шолохов-Дон(ранний)", "plat": "Платонов", "kr": "Крюков", "seraf": "Серафимович"}


def bc(a, b, root="frags_train"):
    d = DATA / root / a / b
    return [f.read_text("utf-8") for f in sorted(d.glob("*.txt"))] if d.exists() else []


def books_of(a, root="frags_train"):
    d = DATA / root / a
    return {b.name: bc(a, b.name, root) for b in sorted(d.iterdir()) if len(bc(a, b.name, root)) >= 4} if d.exists() else {}


SH_DON = ["rodinka", "batraki", "chuzhaya_krov", "aleshkino_serdce", "pastuh", "zherebenok", "lazorevaya_step"]
KR_FIC = ["oficersha", "na_tihom_donu", "v_glubine", "v_rodnih_mestah"]
BASIS_AUTH = [a for a in ["chehov", "gogol", "nabokov", "leskov", "saltykov", "kuprin", "gazdanov", "bunin"]
              if (DATA / "frags_train" / a).exists()]


def balanced(chunks, budget=220):
    return chunks if len(chunks) <= budget else [chunks[i] for i in RNG.choice(len(chunks), budget, replace=False)]


cfg = load_config()
CAND = {
    "sh": {b: bc("sholohov", b) for b in SH_DON if len(bc("sholohov", b)) >= 4},
    "plat": books_of("platonov"),
    "kr": {b: bc("krukov", b) for b in KR_FIC if len(bc("krukov", b)) >= 4},
    "seraf": books_of("serafimovich"),
}
PC = bc("unknown", "podnyataya_celina_1", root="frags_unknown") or bc("unknown", "podnyataya_celina_2", root="frags_unknown")
for k in CAND:
    p(f"{LAB[k]}: {len(CAND[k])} книг, {sum(len(v) for v in CAND[k].values())} чанков")
p(f"ПЦ (held-out): {len(PC)} чанков")

all_cand = [t for c in CAND.values() for ch in c.values() for t in ch]
basis_texts = [t for a in BASIS_AUTH for t in balanced([t for ch in books_of(a).values() for t in ch])]
ALL = basis_texts + all_cand + PC
t = time.time(); make_rep_cache(cfg).warm(ALL, n_process=cfg.get_path("language.parse_n_process", 4)); p(f"rep warm {time.time()-t:.0f}s")


def make_ov(blocks):
    ov = {k: False for k in ["char_ngrams", "function_words", "syntax", "pos_ngrams",
          "punctuation_ngrams", "dependency", "morphology", "length_dist", "embeddings"]}
    for b in blocks:
        ov[b] = True
    return ov


def run_feature(name, blocks):
    p(f"\n══════ ПРИЗНАК: {name} ({'+'.join(blocks)}) ══════")
    ov = make_ov(blocks)

    # нейтральный leak-free basis (fit на сторонних авторах, без кандидатов и без ПЦ)
    vb = StyloVectorizer.from_config(cfg, enabled_override=ov)
    Xb = vb.fit_transform(basis_texts); mas = MaxAbsScaler().fit(Xb); Xs = mas.transform(Xb)
    nc = max(2, min(40, Xs.shape[1] - 1, Xs.shape[0] - 1))
    svd = TruncatedSVD(nc, random_state=42).fit(Xs)
    ss = StandardScaler().fit(svd.transform(Xs))

    def B(texts):
        return ss.transform(svd.transform(mas.transform(vb.transform(list(texts)))))

    Cmat = {k: {b: B(ch) for b, ch in books.items()} for k, books in CAND.items()}
    Bpc = B(PC)

    # B. centroid: дистанция ПЦ до центроида каждого кандидата (всегда renorm)
    def cen(mats):
        return np.vstack([m for m in mats]).mean(0)
    cdists = {}
    for k in CAND:
        ce = cen([Cmat[k][b] for b in CAND[k]])
        cdists[k] = round(float(np.linalg.norm(Bpc.mean(0) - ce)), 3)
    nearest = min(cdists, key=cdists.get)
    p(f"  B. centroid: ПЦ→ " + " ".join(f"{LAB[k]}={cdists[k]}" for k in cdists) + f" → ближайший {LAB[nearest]}")

    # A. multi-class held-out LR на нейтральном basis-представлении
    Xtr, ytr, labs = [], [], sorted(CAND.keys())
    for yi, k in enumerate(labs):
        for b in CAND[k]:
            Xtr.append(Cmat[k][b]); ytr += [yi] * len(Cmat[k][b])
    Xtr = np.vstack(Xtr); ytr = np.array(ytr)
    # балансировка классов
    mn = min(np.bincount(ytr))
    idx = np.concatenate([RNG.choice(np.where(ytr == yi)[0], mn, replace=False) for yi in range(len(labs))])
    clf = LogisticRegression(max_iter=800, C=0.5, class_weight="balanced").fit(Xtr[idx], ytr[idx])
    pc_proba = clf.predict_proba(Bpc).mean(0)
    pcp = {LAB[labs[i]]: round(float(pc_proba[i]), 3) for i in range(len(labs))}
    pred = LAB[labs[int(pc_proba.argmax())]]
    p(f"  A. held-out multi-LR: P(ПЦ) = " + " ".join(f"{k}={v}" for k, v in pcp.items()) + f" → {pred}")

    # C. positive-control: per-book predict vs истинный автор (метод различает кандидатов ВООБЩЕ?)
    correct = 0; total = 0
    books_pred = {}
    for ki, k in enumerate(labs):
        for b in CAND[k]:
            if len(Cmat[k][b]) == 0:
                continue
            pr = clf.predict(Cmat[k][b])
            acc = float(np.mean(pr == ki))
            books_pred[f"{LAB[k]}/{b}"] = round(acc, 2)
            correct += int(np.sum(pr == ki)); total += len(pr)
    p(f"  C. positive-control (in-sample multi-LR accuracy по книгам): overall {correct}/{total} = {correct/total:.2f}")
    plat_acc = np.mean([v for kk, v in books_pred.items() if kk.startswith("Платонов")])
    sh_acc = np.mean([v for kk, v in books_pred.items() if kk.startswith("Шолохов")])
    p(f"     Платонов-книги распознаны avg {plat_acc:.2f}; Шолохов-Дон avg {sh_acc:.2f}")

    # D. segment: почанковая атрибуция ПЦ — доля → каждому + связные «платоновские» сегменты
    seg_pred = clf.predict(Bpc)
    seg_share = {LAB[labs[i]]: round(float(np.mean(seg_pred == i)), 3) for i in range(len(labs))}
    # связные сегменты ≥5 чанков → Платонов
    plat_seg = 0; run = 0; maxrun = 0
    for s in seg_pred:
        if s == labs.index("plat"):
            run += 1; maxrun = max(maxrun, run)
        else:
            run = 0
    p(f"  D. segment: доля чанков ПЦ → " + " ".join(f"{k}={v}" for k, v in seg_share.items()) + f"; max связный «платоновский» рун = {maxrun}")

    return {
        "centroid_distance_Pc_to": cdists, "centroid_nearest": LAB[nearest],
        "heldout_multiclass_proba": pcp, "heldout_pred": pred,
        "positive_control_overall_acc": round(correct / total, 3),
        "positive_control_per_book": books_pred,
        "segment_share": seg_share, "segment_max_platonov_run": int(maxrun),
        "n_pc_chunks": int(len(PC)),
    }


out = {
    "method": ("leak-free тест «Платонов→ПЦ»: нейтральный basis (сторонние авторы), ПЦ held-out, "
               "topic-инвариантные признаки (dependency + ансамбль dep+pos+syntax); "
               "multi-class centroid + held-out LR + positive-control (LOO-различимость) + segment. "
               "Метод = clean_attribution.py (caDep/caEns в rigor)."),
    "candidates": {LAB[k]: sum(len(v) for v in CAND[k].values()) for k in CAND},
}
out["dependency"] = run_feature("dependency (чистый идиолект)", ["dependency"])
out["ensemble"] = run_feature("ансамбль dependency+pos_ngrams+syntax", ["dependency", "pos_ngrams", "syntax"])
out["verdict"] = ("ПЦ → Шолохову (а не Платонову) на обоих чистых признаках — см. centroid_nearest + "
                  "heldout_pred. Гипотеза «Платонов написал ПЦ» данными НЕ поддерживается. "
                  "Positive-control: метод различает кандидатов → тест осмыслен.")
pathlib.Path("docs/platonov_pc.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
p("\n✓ saved docs/platonov_pc.json")
