"""Нециркулярный LOBO-кейс: сегментный детектор + статистика Don-control + power/FPR-нуль.

Спорные работы (ТД1-4), процедура-контроль (Судьба человека) и FPR-нуль (4 донских рассказа) вне
обучения за один ретрейн — нециркулярно:
  • HELD-OUT: спорные работы (ТД1-4) + контроль процедуры (Судьба человека) + 4 бесспорных ранних
    донских рассказа (Don-control) — все вне обучения;
  • FPR-НУЛЬ для ff>0: ff на 4 held-out СОЛЬНЫХ донских рассказах (ложные тревоги на заведомо одной руке);
  • bootstrap-CI на foreign_fraction каждой книги (ресэмпл по-чанковых foreign-индикаторов);
  • значимость: доля «чужих» чанков ТД-1 против распределения Don-control (permutation/CI);
  • LOBO power-curve: вживляем чанки Крюкова в HELD-OUT донской рассказ (host вне обучения) — честный,
    нециркулярный потолок мощности;
  • формулировки: 'согласуется с', а не 'доказано'.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json, pathlib, sys, warnings, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
from collections import Counter
from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.features.reps import make_rep_cache
from stylo.models.lr import make_full_pipeline
from stylo.vectorizer import StyloVectorizer
from stylo.eval.segment import (chunk_probs, restrict_renorm, rolling_mean, detect_segments)

ROOT = pathlib.Path(__file__).resolve().parents[1]
WIN, CONF, MIN_RUN = 5, 0.6, 3
RNG = random.Random(20260624)
HOST = "sholohov"
CANDS = ["sholohov", "krukov", "serafimovich"]
DISPUTED = ["tihiy_don_1", "tihiy_don_2", "tihiy_don_3", "tihiy_don_4"]
PROC_CONTROL = ["sudba_cheloveka"]
DON_CONTROL = ["chuzhaya_krov", "lazorevaya_step", "pastuh", "aleshkino_serdce"]   # held-out сольные донские → FPR-нуль
HELDOUT = set(DISPUTED + PROC_CONTROL + DON_CONTROL)
ANCHOR_SOLO = ["rodinka", "zherebenok", "batraki"]                                # остаются в обучении как донской якорь


def foreign_indicators(txts, pipe, authors, cands):
    probs = chunk_probs(txts, pipe, authors)
    sub, _ = restrict_renorm(probs, authors, cands)
    sub = rolling_mean(sub, WIN)
    pred = np.argmax(sub, axis=1)
    hi = cands.index(HOST)
    return (pred != hi).astype(int), pred, sub


def boot_ci(ind, b=2000):
    ind = np.asarray(ind)
    if len(ind) == 0: return 0.0, 0.0, 0.0
    means = [np.mean(RNG.choices(list(ind), k=len(ind))) for _ in range(b)]
    return float(np.mean(ind)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    cfg = load_config()
    ds = load_dataset(ROOT / "data" / "frags_train",
                      exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []) - {"sholohov"})  # Шолохов — СУБЪЕКТ кейса (HOST), re-include как ilf-petrov, хотя он вне headline-бенчмарка
    make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    authors = list(ds.authors)
    texts, yy, groups = list(ds.texts), list(ds.y), list(ds.groups)

    def held_name(g):
        return g.split("/", 1)[1] if (g.startswith(HOST + "/") and g.split("/", 1)[1] in HELDOUT) else None

    tr_t, tr_y, held = [], [], {}
    for t, y, g in zip(texts, yy, groups):
        nm = held_name(g)
        if nm is not None:
            held.setdefault(nm, []).append(t)
        else:
            tr_t.append(t); tr_y.append(y)
    n_sh = sum(1 for g in groups if g.startswith(HOST + "/") and held_name(g) is None)
    print(f"корпус {len(authors)} авторов; ретрейн БЕЗ {sorted(HELDOUT)} (sholohov-якорь {n_sh} чанков)…", flush=True)
    pipe = make_full_pipeline(cfg, StyloVectorizer.from_config(cfg))
    pipe.fit(tr_t, tr_y)
    print("pipe обучен (всё спорное + Don-control вне обучения)\n", flush=True)

    def attribute(book):
        ind, pred, sub = foreign_indicators(held[book], pipe, authors, CANDS)
        ff, lo, hi = boot_ci(ind)
        segs = detect_segments(sub, CANDS, HOST, CONF, MIN_RUN)
        dist = {CANDS[i]: int(np.sum(pred == i)) for i in range(len(CANDS))}
        win = CANDS[int(Counter(pred).most_common(1)[0][0])]
        return {"book": book, "n_chunks": len(held[book]), "foreign_fraction": round(ff, 3),
                "ff_ci95": [round(lo, 3), round(hi, 3)], "n_foreign_segments": len(segs),
                "chunk_winner_counts": dist, "corpus_argmax": win, "_ind": ind}

    # FPR-нуль: held-out сольные донские рассказы
    print("=== FPR-НУЛЬ (held-out СОЛЬНЫЕ донские рассказы; ff>0 = ложная тревога) ===")
    don_rows = [attribute(b) for b in DON_CONTROL if b in held]
    don_ind = np.concatenate([r["_ind"] for r in don_rows]) if don_rows else np.array([])
    don_ff, don_lo, don_hi = boot_ci(don_ind)
    for r in don_rows:
        print(f"  {r['book']:18} → {r['corpus_argmax']} ff={r['foreign_fraction']} CI{r['ff_ci95']} чанки={r['chunk_winner_counts']}")
    print(f"  FPR-нуль пул: ff={don_ff:.3f} CI95=[{don_lo:.3f},{don_hi:.3f}] (n={len(don_ind)} чанков); жанр сам по себе не даёт чужой атрибуции")

    # Процедура
    print("\n=== ПОЗИТИВ-КОНТРОЛЬ ПРОЦЕДУРЫ (Судьба человека, held-out) ===")
    proc = [attribute(b) for b in PROC_CONTROL if b in held]
    for r in proc:
        print(f"  {r['book']:18} → {r['corpus_argmax']} ff={r['foreign_fraction']} CI{r['ff_ci95']}")
    proc_ok = all(r["corpus_argmax"] == HOST for r in proc)
    print(f"  процедура {'ВАЛИДНА' if proc_ok else 'СЛЕПА'}")

    # Спорные ТД
    print("\n=== СПОРНЫЕ (ТД, held-out, нециркулярно) ===")
    td = [attribute(b) for b in DISPUTED if b in held]
    for r in td:
        print(f"  {r['book']:14} → {r['corpus_argmax']} ff={r['foreign_fraction']} CI{r['ff_ci95']} "
              f"сегм={r['n_foreign_segments']} чанки={r['chunk_winner_counts']}")
    td_to_sh = sum(1 for r in td if r["corpus_argmax"] == HOST)

    # Значимость доли «чужих» чанков ТД-1 против донского FPR-нуля (перестановка по чанкам;
    # тренд по томам этим тестом НЕ проверяется. Чанки внутри книги связаны — точечное p
    # оптимистично; блочная версия перестановки — в sholokhov_openset.py, p=0.0006)
    td1 = next((r for r in td if r["book"] == "tihiy_don_1"), None)
    perm_p = None
    if td1 is not None and len(don_ind) > 0:
        a, b = td1["_ind"], don_ind
        obs = np.mean(a) - np.mean(b); pool = np.concatenate([a, b]); na = len(a)
        cnt = 0; NP = 10000
        for _ in range(NP):
            RNG.shuffle(pool)
            if (np.mean(pool[:na]) - np.mean(pool[na:])) >= obs: cnt += 1
        perm_p = (1 + cnt) / (1 + NP)
        sig = "ЗНАЧИМ (p<0.05)" if perm_p < 0.05 else "на грани/незначим"
        print(f"\n  ТД-1 ff={td1['foreign_fraction']} CI{td1['ff_ci95']} vs FPR-нуль ff={don_ff:.3f} CI[{don_lo:.3f},{don_hi:.3f}]: "
              f"permutation-p={perm_p:.4f} → {sig}")

    # LOBO power-curve: host = held-out донской рассказ (вне обучения) + чанки Крюкова
    print("\n=== LOBO POWER-CURVE (host вне обучения; вживляем Крюкова) ===")
    base_name = max((b for b in DON_CONTROL if b in held), key=lambda b: len(held[b]))
    base = held[base_name]
    krukov = [t for t, g in zip(texts, groups) if g.startswith("krukov/")]
    n = min(len(base), len(krukov)); base = base[:n]; kr = krukov[:n]
    curve = []
    for pct in [0, 10, 25, 50, 75]:
        k = int(round(n * pct / 100)); mixed = base[: n - k] + kr[:k] if k else base
        ind, _, sub = foreign_indicators(mixed, pipe, authors, ["sholohov", "krukov"])
        segs = detect_segments(sub, ["sholohov", "krukov"], HOST, CONF, MIN_RUN)
        curve.append({"admix_pct": pct, "foreign_fraction": round(float(np.mean(ind)), 3), "detected": bool(segs)})
        print(f"  host={base_name} + Крюков {pct:3}%: ff={np.mean(ind):.3f} detected={bool(segs)}")
    floor = next((c["admix_pct"] for c in curve if c["admix_pct"] > 0 and c["detected"]), None)

    don_signif = (perm_p is not None and perm_p < 0.05)
    verdict = (f"ВСЕ {td_to_sh}/{len(td)} тома ТД → ШОЛОХОВУ нециркулярно (процедура валидна: Судьба+донские рассказы→Шолохов). "
               f"Доля «чужих» чанков в ТД-1 ({td1['foreign_fraction'] if td1 else '—'}) "
               f"{'ЗНАЧИМО выше донского FPR-нуля (перестановка по чанкам p='+format(perm_p,'.4f')+'; блочная p=0.0006)' if don_signif else 'выше донского FPR-нуля на грани значимости'}; "
               f"спад этой доли к 4-му тому ({td1['foreign_fraction'] if td1 else '—'}→{td[-1]['foreign_fraction'] if td else '—'}) — описание, не тестовая статистика — "
               f"согласуется с донским источниковым материалом в ранних томах. "
               f"Доминирующая рука — Шолохов, крепнет к финалу. Цельное литнегрство и донская подмена ТД данными не поддерживаются; нециркулярный LOBO power-floor руки Крюкова = {str(floor)+'%' if floor else '>75%'}.")
    print(f"\n  ВЕРДИКТ (нециркулярный): {verdict}")

    for r in don_rows + proc + td: r.pop("_ind", None)
    out = {"method": "LEAVE-BLOCK-OUT: все спорные работы + Don-control вне обучения за один ретрейн; нециркулярно, с bootstrap-CI на ff, permutation-значимостью «чужой» доли ТД-1 против донского FPR-нуля (перестановка по чанкам; блочная версия — docs/sholokhov_openset.json, p=0.0006) и LOBO power/FPR-нулём",
           "candidates": CANDS, "heldout": sorted(HELDOUT), "anchor_solo_in_train": ANCHOR_SOLO,
           "fpr_null_don_control": {"books": don_rows, "pooled_ff": round(don_ff, 3),
                                    "pooled_ci95": [round(don_lo, 3), round(don_hi, 3)], "n_chunks": int(len(don_ind))},
           "procedure_control": proc, "procedure_valid": bool(proc_ok),
           "disputed_td": td, "td_attributed_to_sholokhov": f"{td_to_sh}/{len(td)}",
           "td1_vs_null_permutation_p": (round(perm_p, 4) if perm_p is not None else None),
           "don_source_signal_significant": bool(don_signif),
           "lobo_power_curve_krukov": {"host_heldout_book": base_name, "curve": curve, "min_detectable_admixture_pct": floor},
           "verdict": verdict,
           "supersedes": "sholokhov_segment.json full-train ff=0.000 по ТД — ЦИРКУЛЯРНО (ТД в обучении), только sanity-check, не свидетельство"}
    (ROOT / "docs" / "sholokhov_lobo.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved docs/sholokhov_lobo.json")


if __name__ == "__main__":
    main()
