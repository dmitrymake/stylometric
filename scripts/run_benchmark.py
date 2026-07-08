"""Воспроизводимый бенчмарк атрибуции авторства (версионируется в репозитории).

Гарантии валидности бенчмарка:
  • Один классификатор (LinearSVC) для ВСЕХ каналов — честное сравнение признаков.
  • Векторизаторы/idf/scaler фитятся ВНУТРИ train-фолда (leak-free), не на всех текстах.
  • StratifiedGroupKFold(5) ПО КНИГАМ — тест-книги не видны обучению.
  • Headline-метрика = macro-F1 (микро top-1/top-3 рядом); per-author recall + confusion.
  • Ансамбль — равновесное усреднение softmax (без подбора веса по тесту).
  • Детерминизм: фиксированный seed; результат пишется в docs/validation.json.

Запуск:  python scripts/run_benchmark.py            (все каналы)
         python scripts/run_benchmark.py --fast    (без DSP, быстрее)
Требует:  собранный корпус data/frags_train/<author>/<book>/*.txt (см. README: fetch → clean → chunk).
"""
from __future__ import annotations
import sys, os, json, time, math, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from collections import defaultdict, Counter
from scipy.special import softmax
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import warnings; warnings.filterwarnings("ignore")
from stylo.config import load_config
from stylo.features.reps import make_rep_cache
from stylo.vectorizer import StyloVectorizer
from stylo.corpus_tools.fetch_classics import PUBLIC_DOMAIN_CLEAR  # единый источник истины по юр-чистому PD (минус реабилитационные продления)

SEED = 42
CAP = 35           # макс. чанков с книги (баланс по объёму)
MIN_CHUNKS = 3     # мин. чанков в книге
MIN_BOOKS = 2      # мин. книг у автора (иначе LOBO/GroupKFold невозможен)
# ilf-petrov — соавторский дуэт; nikolas2 — дневники Николая II (не проза, не PD) → вне прозаического headline
EXCLUDE = {"ilf-petrov", "nikolas2", "sholohov"}  # sholohov исключён из headline: авторство ОСПАРИВАЕТСЯ — держать спорного автора нормальным классом некорректно (см. кейс Шолохова отдельно)
# PD-only подмножество для ПУБЛИКУЕМОГО/воспроизводимого числа (умершие >70 лет; см. fetch_classics.PUBLIC_DOMAIN)
PD_ONLY = "--pd-only" in sys.argv
DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "frags_train"
DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
RNG = np.random.RandomState(SEED)

def log(*a): print(*a, flush=True)

def load_corpus():
    authors = {}
    for adir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        a = adir.name
        if a in EXCLUDE:
            continue
        if PD_ONLY and a not in PUBLIC_DOMAIN_CLEAR:  # публикуемое число — только юр-чистые редистрибутируемые PD-авторы (без реабилитационных продлений)
            continue
        books = {}
        for bdir in sorted(p for p in adir.iterdir() if p.is_dir()):
            files = sorted(bdir.glob("*.txt"))
            if len(files) < MIN_CHUNKS:
                continue
            if len(files) > CAP:
                files = [files[i] for i in sorted(RNG.choice(len(files), CAP, replace=False))]
            books[bdir.name] = [f.read_text("utf-8") for f in files]
        if len(books) >= MIN_BOOKS:
            authors[a] = books
    return authors

# ── DSP: профиль словообразовательных суффиксов (stateless, без fit → без утечки) ──
SUF = sorted(["ость","ение","ание","ние","тель","ник","ниц","ист","изм","ация","ция","ство","еств","чик","щик",
              "арь","ач","ёж","льник","очк","ушк","ишк","ёнок","онок","знь","оват","еват","еньк","оньк","аст",
              "ив","лив","чив","еск","чат","альн","ова","ыва","ива","ничать","ировать","ствова","ани","ени"],
             key=len, reverse=True)
_DSP_NLP = [None]
def dsp_matrix(texts):
    import spacy, hashlib, pickle
    cf = pathlib.Path(__file__).resolve().parents[1] / "data" / "dsp_bench_cache.pkl"
    cache = pickle.loads(cf.read_bytes()) if cf.exists() else {}
    h = lambda s: hashlib.sha1(s.encode()).hexdigest()
    todo = [t for t in texts if h(t) not in cache]
    if todo:
        if _DSP_NLP[0] is None:
            _DSP_NLP[0] = spacy.load("ru_core_news_lg", disable=["parser", "ner"])
        for doc, txt in zip(_DSP_NLP[0].pipe(todo, batch_size=64), todo):
            types = set(tk.lemma_.lower() for tk in doc if tk.pos_ in {"NOUN","VERB","ADJ"} and tk.lemma_.isalpha() and len(tk.lemma_) > 3)
            prof = {s: 0 for s in SUF}; m = 0
            for l in types:
                for s in SUF:
                    if l.endswith(s) and len(l) > len(s) + 1:
                        prof[s] += 1; m += 1; break
            N = len(types) + 1; c = np.array([prof[s] for s in SUF], float)
            pp = c / (c.sum() + 1); ent = -sum(x*math.log2(x) for x in pp if x > 0) / math.log2(len(SUF))
            cache[h(txt)] = np.concatenate([c / N, [m / N, ent]])
        cf.write_bytes(pickle.dumps(cache))
    return np.array([cache[h(t)] for t in texts])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="без DSP-канала (быстрее)")
    ap.add_argument("--pd-only", action="store_true", help="только public-domain авторы (публикуемое/воспроизводимое число)")
    args = ap.parse_args()
    log(f"режим корпуса: {'PD-only (публикуемое)' if PD_ONLY else 'полный исследовательский (вкл. копирайтных, локально)'}")

    authors = load_corpus()
    A = sorted(authors); aidx = {a: i for i, a in enumerate(A)}
    items = [(a, b, ch) for a in A for b, ch in authors[a].items()]
    texts = [t for _, _, ch in items for t in ch]
    ychunk = np.array([aidx[a] for a, _, ch in items for _ in ch])
    gchunk = np.array([bi for bi, (_, _, ch) in enumerate(items) for _ in ch])
    ybook = np.array([aidx[a] for a, _, _ in items])
    book_chunks = []
    off = 0
    for _, _, ch in items:
        book_chunks.append(list(range(off, off + len(ch)))); off += len(ch)
    log(f"корпус: авторов={len(A)} книг={len(items)} чанков={len(texts)}")

    cfg = load_config()
    t = time.time(); make_rep_cache(cfg).warm(texts, n_process=cfg.get_path("language.parse_n_process", 4))
    log(f"rep-кэш прогрет {time.time()-t:.0f}s")

    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(ychunk)), ychunk, gchunk))

    # ── каналы: единый источник — stylo.models.channels (fit ТОЛЬКО на train);
    #    DSP остаётся локальным (тяжёлый spaCy-lg кэш) ──
    from stylo.models.channels import make_channels
    def ch_dsp(tr, te):
        Etr = dsp_matrix(tr); Ete = dsp_matrix(te)
        sc = StandardScaler().fit(Etr); return sc.transform(Etr), sc.transform(Ete)

    CHANNELS = make_channels(cfg)
    if not args.fast:
        CHANNELS["DSP (suffixes)"] = ch_dsp

    def channel_book_scores(chan_fn):
        """OOF decision_function по чанкам (fit-within-fold), затем mean по книге → (n_books, n_classes)."""
        dfc = np.full((len(ychunk), len(A)), np.nan)
        for tr_i, te_i in splits:
            tr_txt = [texts[i] for i in tr_i]; te_txt = [texts[i] for i in te_i]
            Xtr, Xte = chan_fn(tr_txt, te_txt)
            clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED).fit(Xtr, ychunk[tr_i])
            df = clf.decision_function(Xte); pres = clf.classes_
            if df.ndim == 1:
                dfc[te_i, pres[1]] = df; dfc[te_i, pres[0]] = -df
            else:
                for j, c in enumerate(pres): dfc[te_i, c] = df[:, j]
        bs = np.full((len(items), len(A)), -1e9)
        for k, ii in enumerate(book_chunks):
            m = np.nanmean(dfc[np.array(ii)], axis=0); bs[k] = np.where(np.isnan(m), -1e9, m)
        return bs

    def metrics(scores):
        pred = scores.argmax(1)
        top1 = float(np.mean(pred == ybook))
        top3 = float(np.mean([ybook[k] in scores[k].argsort()[::-1][:3] for k in range(len(ybook))]))
        mf1 = float(f1_score(ybook, pred, average="macro"))
        return top1, top3, mf1, pred

    log("\n%-24s %8s %8s %9s" % ("канал (один классификатор SVM)", "top-1", "top-3", "macro-F1"))
    chan_scores = {}; res = {}
    for name, fn in CHANNELS.items():
        t = time.time(); bs = channel_book_scores(fn); chan_scores[name] = bs
        t1, t3, mf, _ = metrics(bs); res[name] = {"top1": round(t1, 3), "top3": round(t3, 3), "macro_f1": round(mf, 3)}
        log("%-24s %8.3f %8.3f %9.3f  (%.0fs)" % (name, t1, t3, mf, time.time() - t))

    # ── ансамбль: равновесный + reliability-взвешенный (вес ∝ (OOF-acc − chance)^p) ──
    from stylo.eval.ensemble import reliability_weighted
    chance0 = 1.0 / len(A)
    ens = sum(softmax(np.where(bs < -1e8, -30, bs), axis=1) for bs in chan_scores.values()) / len(chan_scores)
    t1, t3, mf, _ = metrics(ens)
    res["АНСАМБЛЬ (равновес.)"] = {"top1": round(t1, 3), "top3": round(t3, 3), "macro_f1": round(mf, 3)}
    log("%-24s %8.3f %8.3f %9.3f" % ("АНСАМБЛЬ (равновес.)", t1, t3, mf))
    # reliability-взвешенный ансамбль — ТОЛЬКО ДИАГНОСТИКА: его веса = OOF-точность каждого канала
    # НА ТЕХ ЖЕ тест-книгах, по которым потом отчитывается ансамбль → это test-set leak. Поэтому
    # headline'ом он быть НЕ может; оставляем как справочный вариант, честно помечая источник весов.
    oof_test_acc = {n: res[n]["top1"] for n in chan_scores}   # это TEST-OOF точность, не train
    diag_leaked = {}  # reliability^p — НЕ в channels: его веса выведены по тесту (leak), держим ОТДЕЛЬНО
    for p_ in [2.0, 4.0, 6.0]:
        ensw, _ = reliability_weighted(chan_scores, oof_test_acc, chance0, power=p_)
        wt1, wt3, wmf, _ = metrics(ensw)
        diag_leaked[f"reliability^{int(p_)}"] = {"top1": round(wt1, 3), "top3": round(wt3, 3), "macro_f1": round(wmf, 3)}
        log("%-24s %8.3f %8.3f %9.3f" % (f"reliability^{int(p_)} (диагностика)", wt1, wt3, wmf))
    # HEADLINE = РАВНОВЕСНЫЙ ансамбль: единственный leak-free вариант (веса не зависят от теста).
    headline_key = "АНСАМБЛЬ (равновес.)"
    _, _, _, pred = metrics(ens)
    t1, t3, mf = res[headline_key]["top1"], res[headline_key]["top3"], res[headline_key]["macro_f1"]
    log("HEADLINE: %s (macro-F1 %.3f) — равновесный, leak-free (веса не зависят от теста)" % (headline_key, mf))

    # per-author recall + confusion + author-clustered CI на macro-F1
    byA = defaultdict(lambda: [0, 0]); conf = Counter()
    for k in range(len(ybook)):
        tr = A[ybook[k]]; pr = A[pred[k]]; byA[tr][1] += 1; byA[tr][0] += (tr == pr)
        if tr != pr: conf[(tr, pr)] += 1
    recalls = {a: round(c / n, 2) for a, (c, n) in sorted(byA.items(), key=lambda x: x[1][0] / x[1][1])}
    low = [a for a, r in recalls.items() if r <= 0.5]
    uniq = list(set(ybook)); f1s = []
    for _ in range(2000):
        sa = RNG.choice(len(uniq), len(uniq), replace=True)
        sel = np.concatenate([np.where(ybook == uniq[i])[0] for i in sa])
        f1s.append(f1_score(ybook[sel], pred[sel], average="macro"))
    ci = [round(float(np.percentile(f1s, 2.5)), 3), round(float(np.percentile(f1s, 97.5)), 3)]
    chance = 1.0 / len(A)
    log(f"\nmacro-F1 ансамбля author-clustered 95% CI {ci} | случайный (1/{len(A)})={chance:.3f}")
    log(f"низкий recall (≤0.5): {low or 'нет'}")
    log(f"топ-путаниц: {[f'{a}->{b}x{c}' for (a,b),c in conf.most_common(8)]}")

    hb = res[headline_key]  # равновесный headline-ансамбль (leak-free: веса не зависят от теста)
    out = {
        "method": "LinearSVC (один для всех каналов) + StratifiedGroupKFold(5) book-level, векторизаторы fit ВНУТРИ фолда (leak-free); headline-ансамбль = РАВНОВЕСНОЕ усреднение softmax каналов (веса не зависят от теста, leak-free); reliability^p — справочная диагностика с весами, выведенными по тесту",
        "corpus_mode": "pd_only" if PD_ONLY else "full_research",
        "corpus_note": ("классики, умершие >70 лет назад; тексты докачиваются по URL-манифесту для локальной валидации — "
                        "у Гумилёва и Пильняка охрана в РФ продлена после реабилитации (ст. 1281 п. 5 ГК), "
                        "downstream-редистрибуция — ответственность пользователя"
                        if PD_ONLY else "полный исследовательский корпус ВКЛЮЧАЕТ копирайтных/живых авторов — НЕ редистрибутируемо; для публикуемого числа используйте --pd-only"),
        "headline_ensemble": headline_key,
        "seed": SEED, "cap_chunks_per_book": CAP, "n_authors": len(A), "n_books": len(items), "n_chunks": len(texts),
        "chance_micro": round(chance, 4),
        "headline_macro_f1": hb["macro_f1"],
        "macro_f1_authorclustered_CI": ci,
        "ensemble_top1": hb["top1"], "ensemble_top3": hb["top3"],
        "channels": res, "per_author_recall": recalls, "low_recall_authors": low,
        "_diagnostic_test_leaked": {"_warning": "веса этих ансамблей выведены по ТЕСТУ (leak) — НЕ headline, только справочно", **diag_leaked},
        "top_confusions": [f"{a}->{b}x{c}" for (a, b), c in conf.most_common(12)], "authors": A,
        "notes": [
            "Честное сравнение: ВСЕ каналы под одним классификатором (LinearSVC); смена nearest-centroid→SVM сама по себе даёт прибавку — это вклад классификатора, не признаков.",
            "Топик-инвариантный синтаксис НЕ превосходит char-n-граммы по точности (trade-off, не преимущество по accuracy).",
            "Бенчмарк на собственном корпусе, НЕ на стандартном PAN/RusProfiling — claim 'SOTA' не делается.",
            "headline-ансамбль = РАВНОВЕСНЫЙ (веса не зависят от теста, leak-free); reliability^p — справочная диагностика, её веса выведены по тесту, поэтому headline'ом она быть не может.",
            "Воспроизводимо: python scripts/run_benchmark.py [--pd-only] (требует собранного корпуса; seed фиксирован).",
        ],
    }
    DOCS.mkdir(exist_ok=True)
    fname = "validation_pd.json" if PD_ONLY else "validation.json"
    (DOCS / fname).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    log(f"\n✓ saved docs/{fname} (воспроизводимо из scripts/run_benchmark.py{' --pd-only' if PD_ONLY else ''})")

if __name__ == "__main__":
    main()
