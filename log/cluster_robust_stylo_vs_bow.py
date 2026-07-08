"""Шаг 6 (prod-ready): author-clustered проверка stylo vs BoW.

Headline p (stylo>BoW, McNemar p=0.0003) считан по LOBO-книгам как iid-пары.
Но книги одного автора скоррелированы по исходу (43 кластера, ~6 книг/автор) ->
внутри-авторская корреляция ЗАВЫШАЕТ значимость. Здесь:
  1) flat McNemar на OOF (sanity, то же направление);
  2) AUTHOR-CLUSTERED paired bootstrap разности accuracy (ресэмпл АВТОРОВ, не книг):
     значимо, если 95% CI разности не пересекает 0;
  3) cluster-robust p-подобное: доля автор-ресэмплов, где BoW >= stylo.

Свежий ОДИНАКОВЫЙ GKF5-OOF для stylo и bow_lr на одном датасете (выровнены).
Регим: OOF=GKF5 (фактически GKF2, olesha=2 книги), НЕ LOBO. Это GKF-диагностический тест
кластер-робастности stylo-vs-BoW — НЕ прямой тест LOBO-headline (для него нужны LOBO per-book
для обоих спеков, которые не сохранены). LOBO Δacc (0.073) МЕНЬШЕ GKF Δacc (0.112), поэтому
результат GKF-режима не переносится на LOBO автоматически — только как диагностика.
"""
from __future__ import annotations
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import binomtest
from collections import Counter

from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.features.reps import make_rep_cache
from stylo.eval.groupkfold import gkf_evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_BOOT = 10000
SEED = 42


def aligned_oof():
    """Свежий ОДИНаковый GKF5-OOF для stylo и bow_lr на одном датасете."""
    cfg = load_config()
    excl = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=excl)
    ba = ds.book_to_author()  # book_id -> author_idx
    single = {a for a, c in Counter(ba.values()).items() if c < 2}
    if single:
        single_names = sorted(ds.authors[a] for a in single)
        print(f"дроп single-book авторов: {single_names}", flush=True)
        ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=excl | set(ds.authors[a] for a in single))
    make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    print(f"корпус: {ds.n_authors} авт, {len(set(ds.groups.tolist()))} книг -> GKF OOF", flush=True)
    df_s, P_s, y_s = gkf_evaluate(cfg, ds, spec="stylo")
    df_b, P_b, y_b = gkf_evaluate(cfg, ds, spec="bow_lr")
    assert len(y_s) == len(y_b) and np.array_equal(y_s, y_b), "OOF не выровнены после общего прогона"
    # автор на книгу — из df (test_author), оба df в одном порядке книг
    assert list(df_s["test_book"]) == list(df_b["test_book"]), "порядок книг разошёлся"
    return P_s, P_b, y_s, df_s["test_author"].to_numpy()


def main():
    P_s, P_b, y, book_authors = aligned_oof()
    uniq = np.array(sorted(set(book_authors.tolist())))
    n_books = len(y)
    n_authors = len(uniq)
    print(f"книг={n_books} | кластеров(авторов)={n_authors} | книг/автор={n_books/n_authors:.2f}", flush=True)

    c_s = (P_s.argmax(1) == y)
    c_b = (P_b.argmax(1) == y)
    acc_s, acc_b = float(c_s.mean()), float(c_b.mean())
    print(f"accuracy stylo={acc_s:.4f} bow={acc_b:.4f} Δ={acc_s-acc_b:+.4f}", flush=True)

    # 1) flat McNemar (iid-предположение headline)
    b = int(np.sum(c_s & ~c_b))
    c = int(np.sum(~c_s & c_b))
    p_flat = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue) if (b + c) else 1.0
    print(f"[FLAT iid] McNemar b={b} c={c} p={p_flat:.6g}  (headline LOBO p≈0.0003 — др. регим)", flush=True)

    diff = c_s.astype(float) - c_b.astype(float)
    book_idx_by_author = {au: np.where(book_authors == au)[0] for au in uniq}

    # 2) author-clustered paired bootstrap разности accuracy
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([book_idx_by_author[uniq[j]] for j in sa])
        boots[i] = diff[sel].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    median = float(np.median(boots))
    sig = not (lo <= 0.0 <= hi)
    p_cluster = float(np.mean(boots <= 0.0))
    print(f"[AUTHOR-CLUSTERED] Δaccuracy: медиана={median:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  значимо(0∉CI)={sig}", flush=True)
    print(f"[AUTHOR-CLUSTERED] p-подобное P(Δ<=0)={p_cluster:.4f}  (доля автор-ресэмплов, где BoW≥stylo)", flush=True)

    out = {
        "method": "author-clustered paired bootstrap Δaccuracy(stylo−bow) на свежем GKF5-OOF (оба спека одним прогоном); ресэмпл АВТОРОВ, не книг",
        "regime_caveat": "GKF5(факт.GKF2)-OOF, НЕ LOBO. GKF-диагностика кластер-робастности stylo-vs-BoW; НЕ прямой тест LOBO-headline (нужны LOBO per-book обоих спеков, не сохранены). LOBO Δacc(0.073) < GKF Δacc(0.112) — переноса на LOBO нет.",
        "n_books": int(n_books), "n_authors": int(n_authors), "books_per_author": round(n_books / n_authors, 2),
        "acc_stylo": round(acc_s, 4), "acc_bow": round(acc_b, 4), "delta_accuracy": round(float(acc_s - acc_b), 4),
        "flat_mcnemar": {"b": b, "c": c, "p": round(p_flat, 6),
                         "note": "iid-предположение (как в headline); headline LOBO p≈0.0003"},
        "author_clustered": {"delta_median": round(median, 4),
                             "ci95": [round(float(lo), 4), round(float(hi), 4)],
                             "significant_0_not_in_ci": bool(sig),
                             "p_like_P_delta_le_0": round(p_cluster, 4),
                             "n_boot": N_BOOT},
        "verdict": ("stylo>BoW выживает авторскую кластеризацию В GKF-РЕГИМЕ (CI не пересекает 0). Диагностика кластер-робастности; НЕ прямой тест LOBO-headline."
                    if sig and p_cluster < 0.05 else
                    "stylo>BoW НЕ выживает авторскую кластеризацию в GKF-режиме — нужна проверка в LOBO-режиме"),
    }
    out_path = ROOT / "docs" / "cluster_robust_stylo_vs_bow.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ {out_path}", flush=True)
    print(f"VERDICT: {out['verdict']}", flush=True)


if __name__ == "__main__":
    main()
