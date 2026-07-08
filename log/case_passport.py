"""Forensic-паспорт на исторических кейсах — ИЛЛЮСТРАЦИЯ (Шаг 3, не валидация).

К каждому спорному тексту прикладываем passport-линзу:
  • типичность (max_prob, entropy, margin) почанковых soft-vote постериоров;
  • z-score типичности vs РАСПРЕДЕЛЕНИЕ корпорных книг (та же in-sample модель —
    относительное сравнение честно, абсолютная уверенность in-sample завышена);
  • топ-автор + запас (margin) над вторым;
  • вердикт: «genuine/атрибутируем» (высокая типичность, чёткий margin) vs
    «at-floor/диффузный» (низкая типичность, низкий margin — жанр/регистр доминируют,
    рука не видна).

Контраст-иллюстрация принципа «псевдоним ловится, мистификация/подделка — нет»:
  • Вырубова-фальшивка (disguised)  — ожидаемо at-floor;
  • Чехонте held-out Чехов (genuine под псевдонимом) — ожидаемо genuine→chehov.

ЧЕСТНО: n=единицы кейсов ⇒ это ИЛЛЮСТРАЦИЯ паспорта (naturalistic floor-якорь),
НЕ статистическая валидация детектора. z-score in-sample — относительная мера.
"""
from __future__ import annotations
import json, pathlib, sys, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np

from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.features.reps import make_rep_cache
from stylo.eval.lobo import make_factory, _align_proba
from stylo.eval.fano import typicality_scores

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK_WORDS = 500

# кейсы: (имя, путь к mystery, ожидание)
CASES = [
    ("vyrubova_diary_fake", "input_cases/vyrubova/vyrubova_diary_fake.txt", "disguised (подделка)"),
    ("vyrubova_memoir_real", "input_cases/vyrubova/vyrubova_memoir_real.txt", "outsider (настоящая Вырубова)"),
    ("cherubina_poems", "input_cases/cherubina/cherubina_poems.txt", "псевдоним/мистификация"),
    ("chekhonte_heldout_chehov", "input_cases/chekhonte/mystery_heldout_chehov_1884_85.txt", "genuine (Чехов под псевд.)"),
    ("prutkov_poems", "input_cases/prutkov_hands/mystery_prutkov_poems", "коллективный"),
]


def chunk_text(txt, words=CHUNK_WORDS):
    w = txt.split()
    return [" ".join(w[i:i + words]) for i in range(0, len(w), words)] if w else []


def load_mystery(path):
    p = pathlib.Path(path)
    if p.is_dir():
        out = []
        for f in sorted(p.glob("*.txt")):
            out.extend(chunk_text(f.read_text("utf-8", "ignore")))
        return out
    return chunk_text(p.read_text("utf-8", "ignore"))


def main():
    cfg = load_config()
    excl = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=excl)
    make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    authors = list(ds.authors)
    K = ds.n_authors
    texts, y, groups = ds.texts, ds.y, ds.groups
    book_author = ds.book_to_author()
    print(f"корпус: {K} авт; fit bow_lr in-sample…", flush=True)

    est = make_factory("bow_lr", cfg)()
    est.fit(texts, y)
    classes = list(est.classes_)

    def softvote(chunks):
        if not chunks:
            return None
        pr = np.asarray(est.predict_proba(chunks))
        return _align_proba(pr, np.asarray(classes), K)

    # референс: типичность КОРПУСНЫХ книг из leak-free OOF (data/fano_oof_bow_lr.npz).
    # ЧЕСТНО: mystery предсказывается ПОЛНОЙ in-sample моделью, OOF-референс — фолд-моделями
    # (4/5) — шкалы чуть разные, НО обе out-of-sample (не in-sample corpus=0.94, что ломало z).
    ref = {"max_prob": [], "entropy_bits": [], "margin": []}
    oof_path = ROOT / "data" / "fano_oof_bow_lr.npz"
    if oof_path.exists():
        z = np.load(oof_path, allow_pickle=True)
        pm = z["prob_matrix"]
        t = typicality_scores(pm)
        ref["max_prob"] = t["max_prob"]
        ref["entropy_bits"] = float(np.log2(K)) - t["neg_entropy"]
        s = np.sort(pm, axis=1)
        ref["margin"] = s[:, -1] - s[:, -2]
    else:
        print("  ВНИМАНИЕ: OOF-референс не найден — in-sample корпус (сверх-уверенный, z сломан)", flush=True)
    for k in ref:
        ref[k] = np.array(ref[k])
    print(f"  OOF-референс: {len(ref['max_prob'])} корп. книг; median max_prob={np.median(ref['max_prob']):.3f}", flush=True)

    def z(score, arr):
        sd = float(np.std(arr)) or 1e-9
        return round(float((score - np.mean(arr)) / sd), 2)

    results = []
    for name, path, expectation in CASES:
        chunks = load_mystery(path)
        if len(chunks) < 3:
            print(f"  пропуск {name} (мало чанков: {len(chunks)})", flush=True)
            continue
        full = softvote(chunks)
        order = np.argsort(-full, kind="stable")
        top1, top2 = authors[int(order[0])], authors[int(order[1])]
        margin = float(full[order[0]] - full[order[1]])
        maxp = float(full.max())
        ent = float(-np.sum(np.where(full > 0, full * np.log2(np.clip(full, 1e-12, 1)), 0)))
        zmax = z(maxp, ref["max_prob"])
        zent = z(ent, ref["entropy_bits"])
        zmar = z(margin, ref["margin"])
        # вердикт: genuine если типичность выше медианы корпуса И margin чёткий
        genuine = (maxp >= np.median(ref["max_prob"])) and (margin >= np.median(ref["margin"]))
        results.append({
            "case": name, "expectation": expectation, "n_chunks": len(chunks),
            "top_author": top1, "top2_author": top2, "top_share": round(float(full[order[0]]), 3),
            "margin_top1_top2": round(margin, 3),
            "typicality": {"max_prob": round(maxp, 3), "entropy_bits": round(ent, 3),
                           "z_max_prob_vs_corpus": zmax, "z_entropy_vs_corpus": zent, "z_margin_vs_corpus": zmar},
            "verdict": "genuine/атрибутируем" if genuine else "at-floor/диффузный (жанр-регистр доминирует)",
        })
        print(f"  {name:26} [{expectation}]: top={top1}({full[order[0]]:.2f}) margin={margin:.2f} "
              f"z_max={zmax:+.2f} z_ent={zent:+.2f} → {results[-1]['verdict']}", flush=True)

    out = {
        "method": ("forensic-паспорт на кейсах: typicity soft-vote постериоров vs leak-free OOF корпус-референс; "
                   "вердикт genuine (типичность≥медиана + чёткий margin) vs at-floor (диффузный). ИЛЛЮСТРАЦИЯ, n≈5"),
        "corpus_reference": {"n_books": int(len(ref["max_prob"])),
                             "median_max_prob": round(float(np.median(ref["max_prob"])), 3),
                             "median_margin": round(float(np.median(ref["margin"])), 3)},
        "cases": results,
        "register_confound": ("ВАЖНО: все кейсы — ДРУГОЙ РЕГИСТР (стихи Черубины/Пруткова, дневник Вырубовой) "
                              "vs ПРОЗА-корпус. Типичность паспорта конфаундит регистр: genuine автор в чужом "
                              "регистре выглядит атипичным (z<0 у ВСЕХ, включая genuine Чехов-контроль). ⇒ "
                              "корпус-typicality-паспорт НЕ переносится на регистр-смещённые кейсы; для них "
                              "нужен register-matched attribute_case (docs/cases/*.json, УЖЕ сделано). Паспорт "
                              "применим к floor/outsider-characterization (Шаги 1-2), НЕ к adjudication кейсов."),
        "principle_test": ("контраст genuine-vs-disguised НЕ виден через корпус-typicality (регистр доминирует); "
                           "виден только через register-matched attribute_case (docs/cases)"),
    }
    p = ROOT / "docs" / "case_passport.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved {p}", flush=True)


if __name__ == "__main__":
    main()
