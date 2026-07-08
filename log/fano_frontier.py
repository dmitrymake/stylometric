"""Detection-frontier vs information-theoretic пол (Шаг 1).

Один прогон собирает:
  1. leakage-free OOF-матрицу постериоров (GKF5, быстрый прокси LOBO);
  2. Fano-пол на уровне книг: H(A), H(A|F), I(A;F), fano_floor_Pe, эмпирическая
     ошибка и зазор (сколько «живого» места над полом);
  3. per-pair Bayes-пол неразличимости P_e(A,B) для набора пар авторов;
  4. синтетический splice-детектор: склейка host+intruder при нескольких mix_ratio,
     фиксирующий, ловит ли детектор известную чужую руку;
  5. ОВЕРЛЕЙ: эмпирическая детекция vs per-pair пол — рушится ли детекция там,
     где пара становится информационно неразличимой (≈0.5).

КЛЮЧЕВОЙ инсайт: если детекция чужой руки коллапсирует ровно у пол (P_e(A,B)→0.5),
это эмпирическое подтверждение тезиса «успешная маскировка/имитация = floor-феномен»
(нельзя обнаружить вставку между информационно неразличимыми авторами).

Первый прогон — --spec bow_lr (минуты, валидация + инсайт). Честные числа — --spec stylo
(медленнее из-за CalibratedClassifierCV). ECE сообщается рядом — при ECE≈0.3 оценка
H(A|F) смещена (Boenninghoff 2021); gap_* интерпретируем с этой оговоркой.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np

from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.features.reps import make_rep_cache
from stylo.models.lr import make_full_pipeline
from stylo.vectorizer import StyloVectorizer
from stylo.eval.groupkfold import gkf_evaluate
from stylo.eval.lobo import make_factory
from stylo.eval.metrics import expected_calibration_error
from stylo.eval.fano import fano_book_level, pairwise_floor
from stylo.eval.segment import (chunk_probs, restrict_renorm, rolling_mean,
                                detect_segments, foreign_fraction)

ROOT = pathlib.Path(__file__).resolve().parents[1]
WIN, CONF, MIN_RUN = 5, 0.6, 3

# пары host↔intruder. SIMILAR = близкие манеры (донская школа и т.п.) — там пол выше.
SIMILAR = [("krukov", "serafimovich"), ("bunin", "kuprin"), ("chehov", "bunin"),
           ("tolstoy", "tolstoy_an")]   # tolstoy/tolstoy_an — одно семейство стиля
DISSIMILAR = [("dostoevsky", "sorokin"), ("chehov", "prokhanov"), ("turgenev", "mamleev"),
              ("gogol", "nabokov"), ("bunin", "radov"), ("tolstoy", "victor_erofeev")]


def build_books(ds):
    """(author,book)->[chunk texts в порядке]; by_author: author->[(book,txts)]."""
    books = {}
    for t, g in zip(ds.texts, ds.groups):
        books.setdefault(g, []).append(t)
    by_author = {}
    for g, txts in books.items():
        a = g.split("/", 1)[0]
        by_author.setdefault(a, []).append((g, txts))
    return books, by_author


def biggest_book(by_author, a):
    bs = sorted(by_author.get(a, []), key=lambda kv: len(kv[1]), reverse=True)
    return bs[0][1] if bs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="bow_lr", help="OOF-спецификация: bow_lr (быстро) | stylo (честно)")
    ap.add_argument("--n-random-pairs", type=int, default=18, help="случайных пар авторов для фронтира")
    ap.add_argument("--mix-ratios", default="25,50,75", help="%% чужой руки в склейке")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-detection", action="store_true",
                    help="только GKF5+пол+all-pairs (для медленного stylo без детекционной сетки)")
    a = ap.parse_args()

    cfg = load_config()
    excl = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=excl)
    # дроп single-book авторов (как канонический benchmark 43 авт): иначе их 1 книга
    # всегда в test без train-примера → всегда misclassified → завышает error/H(A|F).
    from collections import Counter
    ba = ds.book_to_author()
    single = {ds.authors[a] for a, c in Counter(ba.values()).items() if c < 2}
    if single:
        print(f"дроп single-book авторов: {sorted(single)}", flush=True)
        ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=excl | single)
    make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    authors = list(ds.authors)
    K = ds.n_authors
    aset = set(authors)
    books, by_author = build_books(ds)
    print(f"корпус: {K} авторов, {len(books)} книг; spec={a.spec}", flush=True)

    # --- 1+2. OOF-матрица (GKF5) + Fano-пол ---
    print(f"\n— GKF5 OOF для spec={a.spec} (leak-free)…", flush=True)
    df, prob_matrix, y_true = gkf_evaluate(cfg, ds, spec=a.spec)
    ece = expected_calibration_error(prob_matrix, y_true)
    floor = fano_book_level(prob_matrix, y_true, K, ece=ece)
    print("  Fano-пол (уровень книг):", json.dumps(floor, ensure_ascii=False), flush=True)
    # дамп OOF-матрицы для повторного использования (без ре-рана GKF5)
    np.savez(ROOT / "data" / f"fano_oof_{a.spec}.npz",
             prob_matrix=prob_matrix, y_true=y_true, authors=np.array(authors))

    # --- 3. per-pair Bayes-пол + сбор пар для фронтира ---
    authors_with_books = [x for x in authors if by_author.get(x) and biggest_book(by_author, x)
                          and len(biggest_book(by_author, x)) >= MIN_RUN]
    named = []
    for label, pairs in (("similar", SIMILAR), ("dissimilar", DISSIMILAR)):
        for h, intr in pairs:
            if h in aset and intr in aset and h in authors_with_books and intr in authors_with_books:
                named.append((h, intr, label))
    rng = np.random.default_rng(a.seed)
    pool = [(x, y) for i, x in enumerate(authors_with_books) for y in authors_with_books[i + 1:]]
    rng.shuffle(pool)
    rnd = [(h, intr, "random") for h, intr in pool[:a.n_random_pairs]
           if (h, intr) not in {(p[0], p[1]) for p in named}]
    all_pairs = named + rnd

    def idx(h, intr): return (authors.index(h), authors.index(intr))
    pair_floor = pairwise_floor(prob_matrix, y_true, K, [idx(h, intr) for h, intr, _ in all_pairs])
    pf_map = {(p["a"], p["b"]): p["bayes_floor_Pe"] for p in pair_floor}
    print(f"\n— per-pair Bayes-пол для {len(all_pairs)} пар; "
          f"неразличимых (Pe≥0.45): {sum(1 for p in pair_floor if p['indistinguishable'])}", flush=True)

    # --- 3b. ALL-pairs Pe: есть ли ВООБЩЕ натуральная пара у пола? ---
    awb_idx = [authors.index(x) for x in authors_with_books]
    all_ij = [(awb_idx[i], awb_idx[j])
              for i in range(len(awb_idx)) for j in range(i + 1, len(awb_idx))]
    all_pf = pairwise_floor(prob_matrix, y_true, K, all_ij)
    all_pf_sorted = sorted(all_pf, key=lambda p: -p["bayes_floor_Pe"])
    hist = {}
    for lo, hi in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.51)]:
        c = sum(1 for p in all_pf if lo <= p["bayes_floor_Pe"] < hi)
        hist[f"[{lo:.2f},{hi:.2f})"] = c
    n_total = len(all_pf)
    n_ge035 = sum(1 for p in all_pf if p["bayes_floor_Pe"] >= 0.35)
    print(f"  ALL-pairs: {n_total} пар; у пола (Pe≥0.35): {n_ge035}; "
          f"макс Pe={all_pf_sorted[0]['bayes_floor_Pe'] if all_pf_sorted else 'NA'}", flush=True)
    print("  гистограмма Pe:", hist, flush=True)
    print("  топ-12 самых конфузных пар:", flush=True)
    for p in all_pf_sorted[:12]:
        print(f"    {authors[p['a']]:18} + {authors[p['b']]:18}  Pe={p['bayes_floor_Pe']:.3f}", flush=True)

    # --- 4+5. splice-детектор по сетке mix_ratio ---
    frontier, overlay = [], []
    if a.skip_detection:
        print("— --skip-detection: детекционная сетка пропущена (только пол + all-pairs)", flush=True)
    else:
        print(f"— обучение in-sample детектора (spec={a.spec}, для синтетических склеек)…", flush=True)
        pipe = make_factory(a.spec, cfg)()
        pipe.fit(ds.texts, ds.y)   # in-sample калибровка детектора — ОК для recall на известных склейках
        mix_ratios = [int(x) for x in a.mix_ratios.split(",")]
        for h, intr, label in all_pairs:
            ta = biggest_book(by_author, h); tb = biggest_book(by_author, intr)
            if not ta or not tb or len(ta) < MIN_RUN or len(tb) < MIN_RUN:
                continue
            n = min(len(ta), len(tb))
            base = ta[:n]
            pe_ab = pf_map.get(idx(h, intr))
            for pct in mix_ratios:
                k = int(round(n * pct / 100))
                mixed = (base[: n - k] + tb[:k]) if k else base
                probs = chunk_probs(mixed, pipe, authors)
                sub, _ = restrict_renorm(probs, authors, [h, intr])
                sub = rolling_mean(sub, WIN)
                segs = detect_segments(sub, [h, intr], h, conf=CONF, min_run=MIN_RUN)
                ff = foreign_fraction(sub, [h, intr], h)
                # «чужой» сегмент после точки склейки = попадание
                splice = len(mixed) - k
                hit = any(s.start >= splice - WIN and s.author != h for s in segs)
                frontier.append({"host": h, "intruder": intr, "kind": label,
                                 "pair_bayes_floor_Pe": pe_ab, "mix_ratio_pct": pct,
                                 "detected": bool(hit), "foreign_fraction": round(ff, 3)})
            print(f"  [{label}] {h}+{intr}: Pe(A,B)={pe_ab}", flush=True)

        # сводка оверлея: средняя детекция по бинам pair_bayes_floor
        bins = [(0.0, 0.15, "очень различимы"), (0.15, 0.35, "различимы"),
                (0.35, 0.45, "на грани"), (0.45, 0.51, "неразличимы (≈пол)")]
        for lo, hi, name in bins:
            cells = [f for f in frontier if f["pair_bayes_floor_Pe"] is not None
                     and lo <= f["pair_bayes_floor_Pe"] < hi]
            if cells:
                overlay.append({"bin": name, "range": [lo, hi], "n_cells": len(cells),
                                "mean_detection": round(float(np.mean([c["detected"] for c in cells])), 3),
                                "mean_foreign_fraction": round(float(np.mean([c["foreign_fraction"] for c in cells])), 3)})

    out = {
        "status": ("exploratory; НЕ headline-ready: Fano-пол (Pe≈0.256) ВЫШЕ эмпирической ошибки (≈0.188) → gap<0, "
                   "нижняя граница невалидна. Причина: H(A|F) из некалиброванных OOF-постериоров (ECE≈0.24). "
                   "Не цитировать как результат до within-fold калибровки и sanity-гейта floor≤empirical_error."),
        "method": ("Fano/Bayes information-пол (Harrison&Yener ISIT'25 framing) + синтетический splice-детектор; "
                   "OOF=GKF leak-free (k=min(5,min_books); при olesha=2 книг k=2); overlay = эмпирическая детекция vs per-pair Bayes-пол"),
        "spec": a.spec,
        "fano_floor_book_level": floor,
        "ece_calibration": round(ece, 4),
        "all_pairs_distribution": {
            "n_pairs": n_total,
            "n_at_floor_ge035": n_ge035,
            "max_pair_Pe": all_pf_sorted[0]["bayes_floor_Pe"] if all_pf_sorted else None,
            "histogram": hist,
            "top_confusable": [{"a": authors[p["a"]], "b": authors[p["b"]],
                                "bayes_floor_Pe": p["bayes_floor_Pe"]} for p in all_pf_sorted[:15]],
            "interpretation": ("если max_pair_Pe ≪ 0.45 и n_at_floor_ge035≈0 — НИКАКАЯ натуральная пара "
                               "авторов не у information-пола; пол не ограничивает натуральную AA, "
                               "а вступает в силу лишь в вырожденных/циркулярных режимах (n≈2, жанр≡автор)"),
        },
        "calibration_caveat": "H(A|F) из сырых OOF-постериоров; при ECE≈0.3 (см.) оценка смещена — "
                              "gap интерпретировать осторожно; в Шаге 2 — within-fold изотония (Boenninghoff 2021).",
        "pairwise_floor": pair_floor,
        "detection_frontier": frontier,
        "overlay_detection_vs_floor": overlay,
        "headline_check": ("детекция чужой руки коллапсирует к 0 там, где pair_bayes_floor→0.5 "
                           "(неразличимая пара) ⇒ splice/имитация между неразличимыми авторами = floor-феномен"),
    }
    p = ROOT / "docs" / "fano_frontier.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved {p}", flush=True)
    print("\n=== ОВЕРЛЕЙ (детекция vs Bayes-пол) ===", flush=True)
    for o in overlay:
        print(f"  {o['bin']:<26} [{o['range'][0]:.2f},{o['range'][1]:.2f})  n={o['n_cells']:>3}  "
              f"det={o['mean_detection']:.2f}  ff={o['mean_foreign_fraction']:.2f}", flush=True)


if __name__ == "__main__":
    main()
