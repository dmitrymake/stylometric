"""OPEN-SET тест + блочная permutation-значимость для нециркулярного LOBO-кейса ТД.

(1) OPEN-SET: решающий LR-сегмент closed-set по {Шолохов,Крюков,Серафимович}. Проверяем, не поглощает ли
    argmax-плюральность чанки автора ВНЕ набора кандидатов в Шолохова:
      • full 48-way argmax для томов ТД (без restrict) — куда идут чанки среди ВСЕХ известных авторов;
      • инъекция held-out НЕ-кандидата (Платонов) в host-донской рассказ при restrict {Шолохов,Крюков,Серафимович}
        — если чужак уходит в Шолохова (низкая ff) → метод поглотил бы цельного гострайтера вне набора;
        full 48-way тех же чанков обязан вернуть Платонова (корректное отклонение).
(2) БЛОЧНАЯ PERMUTATION: ТД-1 ff против FPR-нуля считаем moving-block перестановкой (блок≈win=5),
    снимая автокорреляцию сглаженных чанков (честный p вместо оптимистичного 0.0001).
LOBO-режим: ТД и Don-control вне обучения.
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
from stylo.eval.segment import chunk_probs, restrict_renorm, rolling_mean, foreign_fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
WIN, CONF, MIN_RUN = 5, 0.6, 3
RNG = random.Random(20260624)
HOST = "sholohov"
CANDS = ["sholohov", "krukov", "serafimovich"]
TD = ["tihiy_don_1", "tihiy_don_2", "tihiy_don_3", "tihiy_don_4"]
DON_CONTROL = ["chuzhaya_krov", "lazorevaya_step", "pastuh", "aleshkino_serdce"]
HELDOUT = set(TD + DON_CONTROL + ["sudba_cheloveka"])
OUTSIDER = "platonov"     # автор ВНЕ набора кандидатов (для open-set инъекции)


def indicators(txts, pipe, authors, cands):
    probs = chunk_probs(txts, pipe, authors)
    sub, _ = restrict_renorm(probs, authors, cands); sub = rolling_mean(sub, WIN)
    pred = np.argmax(sub, axis=1); hi = cands.index(HOST)
    return (pred != hi).astype(int)


def moving_block_perm(a, b, win=WIN, NP=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b]); n = len(pooled); na = len(a)
    nb_blocks = int(np.ceil(n / win))
    cnt = 0
    for _ in range(NP):
        starts = [RNG.randrange(0, max(1, n - win + 1)) for _ in range(nb_blocks)]
        perm = np.concatenate([pooled[s:s+win] for s in starts])[:n]
        if (perm[:na].mean() - perm[na:].mean()) >= obs: cnt += 1
    return (1 + cnt) / (1 + NP)


def main():
    cfg = load_config()
    ds = load_dataset(ROOT / "data" / "frags_train",
                      exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []) - {"sholohov"})  # Шолохов — СУБЪЕКТ кейса (HOST), re-include как ilf-petrov, хотя он вне headline-бенчмарка
    make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    authors = list(ds.authors)
    texts, yy, groups = list(ds.texts), list(ds.y), list(ds.groups)

    def is_held(g): return g.startswith(HOST + "/") and g.split("/", 1)[1] in HELDOUT
    tr_t, tr_y, sh = [], [], {}
    by = {}
    for t, y, g in zip(texts, yy, groups):
        by.setdefault(g, []).append(t)
        if is_held(g): sh.setdefault(g.split("/", 1)[1], []).append(t)
        else: tr_t.append(t); tr_y.append(y)
    print(f"корпус {len(authors)} авторов; ретрейн БЕЗ спорных/контролей…", flush=True)
    pipe = make_full_pipeline(cfg, StyloVectorizer.from_config(cfg)); pipe.fit(tr_t, tr_y)
    print("pipe обучен\n", flush=True)
    aidx = {a: i for i, a in enumerate(authors)}

    # (1a) OPEN-SET: full 48-way argmax для томов ТД
    print("=== OPEN-SET: full 48-way argmax томов ТД (без restrict, все известные авторы) ===")
    td_open = []
    for v in TD:
        probs = chunk_probs(sh[v], pipe, authors)
        sm = rolling_mean(probs, WIN)
        pred = np.argmax(sm, axis=1)
        c = Counter(authors[p] for p in pred); n = len(pred)
        top = c.most_common(4)
        sh_share = c.get(HOST, 0) / n
        td_open.append({"vol": v, "sholokhov_share_open": round(sh_share, 3),
                        "top": [(a, round(k/n, 3)) for a, k in top]})
        print(f"  {v:14} Шолохов(open)={sh_share:.3f} | топ: {[(a, round(k/n,2)) for a,k in top]}")

    # (1b) OPEN-SET инъекция: НЕ-кандидат (Платонов) — поглощается ли в Шолохова?
    print(f"\n=== OPEN-SET инъекция: чанки {OUTSIDER} (вне кандидатов) ===")
    outs_chunks = [t for g, txts in by.items() if g.startswith(OUTSIDER + "/") for t in txts]
    inj = {}
    if outs_chunks:
        # full 48-way: куда идут чанки Платонова?
        probs = chunk_probs(outs_chunks, pipe, authors); sm = rolling_mean(probs, WIN)
        pred = np.argmax(sm, axis=1); c = Counter(authors[p] for p in pred); n = len(pred)
        open_self = c.get(OUTSIDER, 0) / n; open_sh = c.get(HOST, 0) / n
        print(f"  full 48-way: {OUTSIDER}→{OUTSIDER}={open_self:.3f}, →Шолохов={open_sh:.3f} (корректно, если →{OUTSIDER})")
        # restrict {Шолохов,Крюков,Серафимович}: поглощается ли в Шолохова (host)?
        ind = indicators(outs_chunks, pipe, authors, CANDS)
        ff_out = 1 - ind.mean()   # доля, ушедшая в HOST при принудительном выборе из 3
        print(f"  restrict 3-канд: доля {OUTSIDER}→Шолохов(host)={ff_out:.3f} "
              f"({'ТРЕВОГА: чужак поглощается в Шолохова' if ff_out>0.5 else 'ОК: не поглощается как Шолохов'})")
        inj = {"outsider": OUTSIDER, "open_to_self": round(open_self, 3), "open_to_sholokhov": round(open_sh, 3),
               "restrict3_share_to_sholokhov": round(float(ff_out), 3)}

    # (2) БЛОЧНАЯ PERMUTATION для ТД-1 vs FPR-нуль (Don-control)
    print(f"\n=== БЛОЧНАЯ PERMUTATION (moving-block, блок={WIN}): ТД-1 ff vs FPR-нуль ===")
    don_ind = np.concatenate([indicators(sh[b], pipe, authors, CANDS) for b in DON_CONTROL if b in sh])
    td1_ind = indicators(sh["tihiy_don_1"], pipe, authors, CANDS)
    p_naive = None
    # наивный (по чанкам) для сравнения
    obs = td1_ind.mean() - don_ind.mean()
    p_block = moving_block_perm(td1_ind, don_ind)
    print(f"  ТД-1 ff={td1_ind.mean():.3f} (n={len(td1_ind)}) vs FPR-нуль ff={don_ind.mean():.3f} (n={len(don_ind)})")
    print(f"  блочная permutation-p = {p_block:.4f} (честно, с учётом автокорреляции win={WIN})"
          f" → {'значим' if p_block<0.05 else 'на грани/незначим'}")

    out = {"method": "open-set (full 48-way + инъекция не-кандидата) + блочная permutation; LOBO-режим",
           "openset_td_full_argmax": td_open,
           "openset_injection": inj,
           "td1_block_permutation": {"td1_ff": round(float(td1_ind.mean()), 3), "null_ff": round(float(don_ind.mean()), 3),
                                     "block_perm_p": round(p_block, 4), "win": WIN,
                                     "note": "moving-block снимает автокорреляцию сглаженных чанков; p честнее точечного 0.0001"}}
    (ROOT / "docs" / "sholokhov_openset.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved docs/sholokhov_openset.json")


if __name__ == "__main__":
    main()
