"""Внешний СТАНДАРТНЫЙ бенчмарк: Reuters C50 / CCAT50 (50 авторов × 50 train + 50 test).
Самый цитируемый датасет authorship attribution → внешняя сопоставимость нашего пайплайна.
Языко-независимые каналы (char/word n-граммы) + LinearSVC по СТАНДАРТНОМУ протоколу (fixed train/test).
Опционально: en-синтаксис, если установлен spaCy en_core_web_sm (--en).

Опубликованные ориентиры (top-1 accuracy на CCAT50): char-n-gram SVM ~0.74-0.78 (Stamatatos и др.),
n-gram/compression методы ~0.70-0.80, нейро (BERT-based) ~0.80-0.87.

Запуск:  python scripts/run_ccat50.py
Данные:  data/external/C50train|C50test/<author>/*.txt  (UCI Reuters_50_50)
"""
from __future__ import annotations
import sys, json, time, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings; warnings.filterwarnings("ignore")
SEED = 42
C_GRID = [0.3, 1.0, 3.0, 10.0]  # C-свип по 5-fold OOF-CV на train (без утечки теста)
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
def log(*a): print(*a, flush=True)

def load(split):
    texts, labels = [], []
    base = EXT / split
    for adir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(adir.glob("*.txt")):
            texts.append(f.read_text("utf-8", errors="ignore")); labels.append(adir.name)
    return texts, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-en", action="store_true", help="без en-синтаксиса (POS-канал по умолчанию включён)")
    args = ap.parse_args()
    Xtr_t, ytr = load("C50train"); Xte_t, yte = load("C50test")
    A = sorted(set(ytr)); aidx = {a: i for i, a in enumerate(A)}
    ytr_i = np.array([aidx[a] for a in ytr]); yte_i = np.array([aidx[a] for a in yte])
    log(f"CCAT50: авторов={len(A)} train={len(Xtr_t)} test={len(Xte_t)}")

    def char_ch():
        # реальный Tfidf-СЛОВАРЬ (не hashing): без коллизий 2^18 → сильнее (0.734 vs 0.729 на пробинге)
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, norm="l2")
        return v.fit_transform(Xtr_t), v.transform(Xte_t)
    def word_ch():
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, norm="l2")
        return v.fit_transform(Xtr_t), v.transform(Xte_t)

    # summary-канал (как у Ngram_A): плотные стилевые скаляры — частоты пунктуации, доли регистра/цифр,
    # распределение длин слов и предложений, лексическое богатство (TTR/hapax). Без spaCy.
    import re
    from collections import Counter
    from sklearn.preprocessing import StandardScaler
    _PUNCT = list('.,;:!?-"\'()[]') + ['—', '…', '“', '”', '‘', '’']
    _WRE = re.compile(r"[A-Za-z]+"); _SRE = re.compile(r"[.!?]+")
    def summary_feats(texts):
        rows = []
        for t in texts:
            n = len(t) or 1
            words = _WRE.findall(t.lower()); nw = len(words) or 1
            wl = [len(w) for w in words] or [0]
            sents = [s for s in _SRE.split(t) if s.strip()]
            spl = [len(_WRE.findall(s)) for s in sents] or [0]
            cnt = Counter(words); hapax = sum(1 for c in cnt.values() if c == 1)
            row = [t.count(p) / n for p in _PUNCT]
            row += [sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n, sum(c.isspace() for c in t) / n]
            row += [float(np.mean(wl)), float(np.std(wl))]
            row += (np.bincount([min(x, 15) for x in wl], minlength=16)[1:16] / nw).tolist()
            row += [float(np.mean(spl)), float(np.std(spl)), len(cnt) / nw, hapax / nw, nw / (len(sents) or 1)]
            rows.append(row)
        return np.array(rows, dtype=np.float64)
    def summary_ch():
        Xtr_s, Xte_s = summary_feats(Xtr_t), summary_feats(Xte_t)
        sc = StandardScaler().fit(Xtr_s)
        return sc.transform(Xtr_s), sc.transform(Xte_s)

    CH = {"char (2-5)": char_ch, "word (1-2)": word_ch, "summary (стиль-скаляры)": summary_ch}

    if not args.no_en:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            def pos_seq(texts):
                out = []
                for doc in nlp.pipe(texts, batch_size=32):
                    out.append(" ".join(t.pos_ for t in doc))   # чистые POS-теги (как POS-n-граммы Ngram_A)
                return out
            postr, poste = pos_seq(Xtr_t), pos_seq(Xte_t)
            def syn_ch():
                # POS-n-граммы 1-4 (юни/би/три/тетраграммы тегов) — стандартный POS-ngram признак, Tfidf-словарь
                v = TfidfVectorizer(analyzer="word", ngram_range=(1, 4), min_df=2, sublinear_tf=True, norm="l2")
                return v.fit_transform(postr), v.transform(poste)
            CH["POS-n-граммы (1-4)"] = syn_ch
        except Exception as e:
            log(f"en-синтаксис пропущен: {str(e)[:60]}")

    chance = 1.0 / len(A)
    ENS_EXCLUDE = {"summary (стиль-скаляры)"}  # балласт (~0.3 top-1): репортим как канал, но из слияния исключаем
    # leak-free: для каждого канала — 5-fold OOF decision-скоры на train (C выбран по OOF top-1),
    # затем модель на полном train → скоры теста. OOF-скоры = мета-признаки стекинга (2500 строк, без утечки).
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(ytr_i)), ytr_i))

    res = {}; te_scores = {}; oof_scores = {}; cv_top1 = {}
    log("\n%-22s %8s %9s %6s %9s" % ("канал (LinearSVC)", "top-1", "macro-F1", "C*", "CV-top1"))
    for name, fn in CH.items():
        t = time.time(); Xtr, Xte = fn()
        # C-свип через 5-fold OOF: для каждого C считаем out-of-fold скоры, C выбираем по их top-1
        best = None
        for C in C_GRID:
            oof = np.zeros((Xtr.shape[0], len(A)))
            for tr_idx, va_idx in folds:
                cl = LinearSVC(C=C, max_iter=4000, random_state=SEED).fit(Xtr[tr_idx], ytr_i[tr_idx])
                oof[va_idx] = cl.decision_function(Xtr[va_idx])
            acc = float(accuracy_score(ytr_i, oof.argmax(1)))
            if best is None or acc > best[0]:
                best = (acc, C, oof)
        cvacc, bestC, oof = best
        clf = LinearSVC(C=bestC, max_iter=4000, random_state=SEED).fit(Xtr, ytr_i)
        df = clf.decision_function(Xte); pred = df.argmax(1)
        acc = accuracy_score(yte_i, pred); mf = f1_score(yte_i, pred, average="macro")
        te_scores[name] = df; oof_scores[name] = oof; cv_top1[name] = cvacc
        res[name] = {"top1": round(float(acc), 3), "macro_f1": round(float(mf), 3),
                     "bestC": bestC, "cv_top1": round(cvacc, 3)}
        log("%-22s %8.3f %9.3f %6.1f %9.3f  (%.0fs)" % (name, acc, mf, bestC, cvacc, time.time() - t))

    ens_names = [n for n in te_scores if n not in ENS_EXCLUDE]
    log("\nканалы в слиянии: " + ", ".join(ens_names) + "  (summary исключён как балласт)")

    # --- Пер-канальная калибровка: выбор метода по held-out NLL внутри OOF (см. stylo.eval.calibration) ---
    from sklearn.linear_model import LogisticRegression
    from stylo.eval.calibration import choose_calibrator
    calib = {}; oof_probs = {}; te_probs = {}
    for n in ens_names:
        cal, passport = choose_calibrator(oof_scores[n], ytr_i, seed=SEED)
        oof_probs[n] = cal(oof_scores[n]); te_probs[n] = cal(te_scores[n])
        calib[n] = passport
        log(f"калибровка {n}: {passport['method']} (held-out NLL {passport['heldout_nll']})")

    # --- Варианты слияния: {raw softmax | калиброванные} × {равновес | val-взвеш | стекинг};
    #     выбор headline строго по OOF top-1 на train (предрегистрация) ---
    raw_oof = {n: softmax(oof_scores[n], axis=1) for n in ens_names}
    raw_te = {n: softmax(te_scores[n], axis=1) for n in ens_names}
    w = {n: max(1e-6, cv_top1[n] - chance) ** 2 for n in ens_names}; sw = sum(w.values())

    def stack_variant(oofp, tep):
        Moof = np.hstack([oofp[n] for n in ens_names])
        Mte = np.hstack([tep[n] for n in ens_names])
        best_acc, best_C = -1.0, 1.0
        for C in [0.3, 1.0, 3.0]:  # C мета-LR: 3-fold CV по OOF-матрице (тест не участвует)
            accs = cross_val_score(LogisticRegression(max_iter=3000, C=C), Moof, ytr_i,
                                   cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
            if accs.mean() > best_acc:
                best_acc, best_C = float(accs.mean()), C
        meta = LogisticRegression(max_iter=3000, C=best_C).fit(Moof, ytr_i)
        return best_acc, (lambda meta=meta, Mte=Mte: meta.predict_proba(Mte)), best_C

    variants = {}
    meta_cs = {}
    for tag, oofp, tep in (("сырые", raw_oof, raw_te), ("калибр.", oof_probs, te_probs)):
        variants[f"АНСАМБЛЬ равновес. ({tag})"] = (
            float(accuracy_score(ytr_i, sum(oofp[n] for n in ens_names).argmax(1))),
            lambda tep=tep: sum(tep[n] for n in ens_names) / len(ens_names))
        variants[f"АНСАМБЛЬ val-взвеш. ({tag})"] = (
            float(accuracy_score(ytr_i, sum(w[n] * oofp[n] for n in ens_names).argmax(1))),
            lambda tep=tep: sum(w[n] * tep[n] for n in ens_names) / sw)
        acc, fn, C = stack_variant(oofp, tep)
        variants[f"АНСАМБЛЬ OOF-стекинг ({tag})"] = (acc, fn)
        meta_cs[tag] = C
    meta_C = meta_cs
    best_tag = max(variants, key=lambda k: variants[k][0])  # выбор ДО взгляда на тест
    log("OOF-выбор headline: " + ", ".join(f"{k}={v[0]:.3f}" for k, v in variants.items())
        + f" -> {best_tag}")
    def report(tag, ens):
        pr = ens.argmax(1); a = accuracy_score(yte_i, pr); m = f1_score(yte_i, pr, average="macro")
        res[tag] = {"top1": round(float(a), 3), "macro_f1": round(float(m), 3),
                    "oof_top1": round(variants[tag][0], 3)}
        log("%-26s %8.3f %9.3f" % (tag, a, m)); return a
    for tag, (_, fn) in variants.items():
        report(tag, fn())
    res["АНСАМБЛЬ"] = res[best_tag]; res["headline_ensemble"] = best_tag
    selection_meta = {
        "headline_selection": "по OOF top-1 на train (предрегистрировано docs/prereg_2026Q3.md); тестовые числа остальных вариантов — диагностика",
        "meta_C": meta_C,
        "calibration": calib,
    }
    log("HEADLINE (OOF-выбор): %s top-1=%.3f" % (best_tag, res["АНСАМБЛЬ"]["top1"]))

    out = {"dataset": "Reuters C50 / CCAT50 (UCI Reuter_50_50)",
           "protocol": ("fixed train(2500)/test(2500), 50 авторов, top-1; LinearSVC, C выбран 5-fold OOF-CV на train "
                        "(C-свип {0.3,1,3,10}); пер-канальная калибровка (identity/temperature/Platt) выбрана по OOF-NLL; "
                        "вариант слияния (равновесный/val-взвешенный/OOF-стекинг) выбран по OOF top-1 на train; C мета-LR "
                        "по 3-fold CV на OOF-матрице; summary-канал исключён из слияния. Все решения на train/OOF, "
                        "тест головного числа не выбирает (предрегистрация docs/prereg_2026Q3.md)."),
           "channels": res, "ens_channels": ens_names, "selection": selection_meta,
           "ensemble_top1": res["АНСАМБЛЬ"]["top1"], "ensemble_macro_f1": res["АНСАМБЛЬ"]["macro_f1"],
           "canonical_reference": ("канонич. фикс. сплит CCAT50: Ngram_A 0.767, BERT_A 0.657 (Valla / Tyo,Dhingra,Lipton 2022). "
                                   "Нейро 0.82-0.83 (Style-HAN и др.) считаны на ДРУГОМ протоколе (60/20/20 + 10-fold CV) — несопоставимо."),
           "note": ("Языко-независимые каналы (char/word) + POS-n-граммы. Канонический режим top-1 на фикс. сплите; "
                    "сопоставим только с Ngram_A 0.767, не с CV-нейро 0.82+. Русские морфо/синт-каналы — на русском корпусе (docs/validation.json).")}
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "ccat50.json").write_text(dumps_strict(out, ensure_ascii=False, indent=2), "utf-8")
    log(f"\n✓ saved docs/ccat50.json | ансамбль top-1={res['АНСАМБЛЬ']['top1']} (канонич. ориентир Ngram_A 0.767)")

if __name__ == "__main__":
    main()
