"""Delta-протокол в headline-LOBO: cosine-вариант и книжный (канонический) режим.

Контекст: в docs/final_comparison.csv Burrows Delta падает 0.489 (150 MFW) → 0.196
(500 MFW). Причина — протокол «500-токенные чанки × Manhattan-центроиды»: слова
хвоста MFW (ранги 301–500) присутствуют в малой доле чанков, их почанковые z —
шум разреженности, топящий среднее |z|. Здесь три измерения на том же leak-free LOBO
(та же выборка, те же исключения, что headline):

  1) контроль протокола: delta:150 — обязан воспроизвести строку delta:150 из final_comparison.csv;
  2) delta_cos:150/300/500 — Cosine Delta (Smith–Aldridge / Evert et al. 2017) на тех же z;
  3) книжный Delta (книга = один образец, конкатенация чанков) 150/300/500/1000 MFW —
     режим, каноничный для литературы по Delta;
  4) разреженность хвоста MFW по чанкам — измерение механизма;
  5) McNemar + author-clustered bootstrap разницы accuracy против stylo
     (per-book результаты stylo — из docs/lobo_books.txt, LOBO не перепрогоняется).

Выход: docs/delta_cosine_lobo.json.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from joblib import Parallel, delayed

from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.eval.lobo import lobo_evaluate
from stylo.eval.metrics import summarize_book_results
from stylo.eval.significance import mcnemar, paired_bootstrap_diff_clustered
from stylo.lang import display_name

DOCS = ROOT / "docs"
OUT = DOCS / "delta_cosine_lobo.json"
N_JOBS = 8
SEED = 42
BOOT_ITERS = 1000

_WORD = re.compile(r"\w+", re.UNICODE)

HDR = re.compile(r"^\[(OK|MISS)\s*\]\s+(.+?)\s*/\s*(\S+)\s+\(rank истинного автора:\s*(\d+)\)")


def parse_stylo_books(authors: list[str]) -> dict[str, bool]:
    """per-book correct stylo из persisted docs/lobo_books.txt → {author_id/book_id: correct}."""
    disp2id = {display_name(a): a for a in authors}
    assert len(disp2id) == len(authors), "display_name не уникальны"
    out: dict[str, bool] = {}
    for line in (DOCS / "lobo_books.txt").read_text(encoding="utf-8").splitlines():
        h = HDR.match(line)
        if not h:
            continue
        status, disp, book = h.group(1), h.group(2).strip(), h.group(3)
        if disp not in disp2id:
            continue
        out[f"{disp2id[disp]}/{book}"] = (status == "OK")
    return out


def chunk_specs(cfg, ds):
    """Чанковые Delta-спеки: матричный путь, математически эквивалентный BurrowsDelta в
    lobo_evaluate. Счётчики слов чанков предвычисляются ОДИН раз (не зависят от меток);
    в каждом фолде MFW = топ-N по суммарной частоте train-чанков (tie-break по алфавиту,
    как в CountVectorizer), mean/std — только по train. Leak-free сохранён.
    Эквивалентность проверяется контролем: delta:150 против строки в final_comparison.csv."""
    import pandas as pd
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    Xr = vec.fit_transform(ds.texts.tolist()).tocsr()
    Xc = Xr.tocsc()
    words = vec.get_feature_names_out()
    books = sorted(set(ds.groups.tolist()))
    b2a = ds.book_to_author()
    print(f"[chunk] counts-матрица: {Xr.shape}", flush=True)

    def run_fold(test_book: str, mfw: int, metric: str):
        mask_test = ds.groups == test_book
        mask_train = ~mask_test
        true_label = b2a[test_book]
        y_train = ds.y[mask_train]
        if true_label not in set(y_train.tolist()):
            return None
        # MFW: топ-N по суммарной частоте train (сортировка (-freq, слово) = CountVectorizer)
        freq = np.asarray(Xr[mask_train].sum(axis=0)).ravel()
        order = np.lexsort((words, -freq))[:mfw]
        cnts = Xc[:, order].toarray().astype(np.float64)
        # нормирование как в models/delta.py::_rel_freq — на сумму счётчиков ПО СЛОВАРЮ MFW,
        # не на длину чанка (иначе контроль delta:150 не воспроизводит канонический CSV)
        row_tot = cnts.sum(axis=1, keepdims=True)
        row_tot[row_tot == 0] = 1.0
        sub = cnts / row_tot
        tr = sub[mask_train]
        mean, std = tr.mean(axis=0), tr.std(axis=0)
        std[std == 0] = 1e-9
        Z = (sub - mean) / std
        classes = np.unique(y_train)
        cents = np.vstack([Z[mask_train][y_train == c].mean(axis=0) for c in classes])
        zt = Z[mask_test]
        if metric == "manhattan":
            d = np.abs(zt[:, None, :] - cents[None, :, :]).mean(axis=2)
        else:
            a = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-12)
            b = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
            d = 1.0 - a @ b.T
        # как в lobo: softmax(-d) на чанк -> среднее по чанкам -> выравнивание на всех авторов
        x = -d - (-d).max(axis=1, keepdims=True)
        ex = np.exp(x)
        p = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
        full = np.zeros(ds.n_authors)
        full[classes] = p.mean(axis=0)
        top1 = int(np.argsort(-full, kind="stable")[0])
        rank = int((full >= full[true_label]).sum())
        return {"test_author": test_book.split("/", 1)[0], "test_book": test_book.split("/", 1)[1],
                "true_label": true_label, "pred_label": top1, "rank": rank,
                "correct": bool(top1 == true_label)}

    res = {}
    for spec, (mfw, metric) in {
        "delta:150": (150, "manhattan"),
        "delta_cos:150": (150, "cosine"),
        "delta_cos:300": (300, "cosine"),
        "delta_cos:500": (500, "cosine"),
    }.items():
        rows = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(run_fold)(b, mfw, metric) for b in books)
        rows = [r for r in rows if r is not None]
        df = pd.DataFrame(rows)
        acc = float(df["correct"].mean())
        res[spec] = df
        print(f"[chunk] {spec}: acc={acc:.4f} ({int(df['correct'].sum())}/{len(df)})", flush=True)
    return res


def book_level_delta(ds, mfw_list=(150, 300, 500, 1000)):
    """Книжный Delta: книга = один образец. Leak-free: MFW-словарь и mean/std —
    только по train-книгам каждого фолда. Токенизация как в models/delta.py
    (lowercase, \\b\\w+\\b), предвычисляется один раз (счётчики не зависят от меток)."""
    books = sorted(set(ds.groups.tolist()))
    b2a = ds.book_to_author()
    counters = {}
    for b in books:
        mask = ds.groups == b
        cnt = Counter()
        for t in ds.texts[mask]:
            cnt.update(_WORD.findall(t.lower()))
        counters[b] = cnt
    print(f"[book] токенизировано книг: {len(books)}", flush=True)

    def run_fold(test_book: str, mfw: int):
        train = [b for b in books if b != test_book]
        true_label = b2a[test_book]
        if sum(1 for b in train if b2a[b] == true_label) == 0:
            return None  # single-book автор — как в штатном LOBO
        total = Counter()
        for b in train:
            total.update(counters[b])
        vocab = [w for w, _ in total.most_common(mfw)]
        vidx = {w: i for i, w in enumerate(vocab)}

        def rel(b):
            c = counters[b]
            n = sum(c.values()) or 1
            v = np.zeros(len(vocab))
            for w, k in c.items():
                j = vidx.get(w)
                if j is not None:
                    v[j] = k / n
            return v

        X = np.vstack([rel(b) for b in train])
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1e-9
        Z = (X - mean) / std
        classes = sorted({b2a[b] for b in train})
        y = np.array([b2a[b] for b in train])
        cents = np.vstack([Z[y == c].mean(axis=0) for c in classes])
        zt = (rel(test_book) - mean) / std
        d = np.abs(zt[None, :] - cents).mean(axis=1)
        pred = classes[int(np.argmin(d))]
        return {"book": test_book, "true": true_label, "pred": pred, "correct": pred == true_label}

    out = {}
    for mfw in mfw_list:
        rows = Parallel(n_jobs=N_JOBS)(delayed(run_fold)(b, mfw) for b in books)
        rows = [r for r in rows if r is not None]
        acc = float(np.mean([r["correct"] for r in rows]))
        out[mfw] = {"rows": rows, "accuracy": acc, "n_books": len(rows)}
        print(f"[book] delta_book:{mfw}: acc={acc:.4f} ({sum(r['correct'] for r in rows)}/{len(rows)})", flush=True)
    return out


def mfw_tail_presence(ds, ranks=((1, 150), (151, 300), (301, 500))):
    """Доля чанков, содержащих хотя бы K=1 слово данного диапазона MFW-рангов,
    и медианная доля слов диапазона, встретившихся в чанке."""
    total = Counter()
    tokenized = [set(_WORD.findall(t.lower())) for t in ds.texts]
    for t in ds.texts:
        total.update(_WORD.findall(t.lower()))
    ranked = [w for w, _ in total.most_common(500)]
    out = {}
    for lo, hi in ranks:
        band = set(ranked[lo - 1:hi])
        cover = np.array([len(band & s) / len(band) for s in tokenized])
        out[f"mfw_{lo}_{hi}"] = {
            "median_share_of_band_present_in_chunk": float(np.median(cover)),
            "mean_share": float(cover.mean()),
        }
    return out


def vs_stylo(df, stylo_correct, iters=BOOT_ITERS):
    cur = {f"{r.test_author}/{r.test_book}": bool(r.correct) for r in df.itertuples()}
    common = sorted(set(stylo_correct) & set(cur))
    ca = np.array([stylo_correct[b] for b in common])
    cb = np.array([cur[b] for b in common])
    mc = mcnemar(ca, cb)
    authors = np.array([b.split("/", 1)[0] for b in common])
    dci = paired_bootstrap_diff_clustered(
        lambda idx: float(ca[idx].mean()), lambda idx: float(cb[idx].mean()),
        authors, iters=iters, level=0.95, seed=SEED)
    return {
        "n_common_books": len(common),
        "mcnemar_p": mc.p_value,
        "dacc_stylo_minus_this": dci.diff,
        "dacc_authorclustered_ci95": [dci.lo, dci.hi],
        "dacc_authorclustered_significant": dci.significant,
    }


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    excl = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=excl)
    print(f"датасет: {len(ds.authors)} авторов, {len(set(ds.groups.tolist()))} книг, {len(ds)} чанков", flush=True)

    stylo_correct = parse_stylo_books(ds.authors)
    acc_stylo = float(np.mean(list(stylo_correct.values())))
    import csv
    with open(DOCS / "final_comparison.csv", encoding="utf-8") as fh:
        canon = {r["model"]: float(r["accuracy"]) for r in csv.DictReader(fh)}
    print(f"stylo из lobo_books.txt: {len(stylo_correct)} книг, acc={acc_stylo:.4f} (канон {canon['stylo']:.4f})", flush=True)
    assert abs(acc_stylo - canon["stylo"]) < 1e-3, "парсер lobo_books.txt разошёлся с каноном"

    presence = mfw_tail_presence(ds)
    print(f"[механизм] {json.dumps(presence, ensure_ascii=False)}", flush=True)

    chunks = chunk_specs(cfg, ds)
    books = book_level_delta(ds)

    result = {
        "method": ("Delta в headline-протоколе (leak-free per-book LOBO, те же исключения, что headline): "
                   "контроль delta:150; Cosine Delta (тот же MFW-z, угол вместо Manhattan) 150/300/500; "
                   "книжный Delta (книга = один образец, Manhattan) 150/300/500/1000. "
                   "Значимость против stylo — McNemar по книгам (антиконсервативная граница) и "
                   "author-clustered bootstrap разницы accuracy (ресэмпл авторов)."),
        "control_delta150_expected_from_final_csv": canon.get("delta:150"),
        "mechanism_mfw_tail_sparsity_in_chunks": presence,
        "chunk_level": {},
        "book_level": {},
    }
    for spec, df in chunks.items():
        summ = summarize_book_results(df["true_label"].to_numpy(), df["pred_label"].to_numpy(),
                                      df["rank"].to_numpy(), ds.authors,
                                      iters=BOOT_ITERS, level=0.95, seed=SEED)
        result["chunk_level"][spec] = {
            "accuracy": summ["accuracy"].point,
            "acc_ci95_books": [summ["accuracy"].lo, summ["accuracy"].hi],
            "macro_f1": summ["macro_f1"].point,
            "n_books": summ["n_books"],
            "vs_stylo": vs_stylo(df, stylo_correct),
        }
    for mfw, r in books.items():
        import pandas as pd
        df = pd.DataFrame([{"test_author": x["book"].split("/", 1)[0],
                            "test_book": x["book"].split("/", 1)[1],
                            "correct": x["correct"]} for x in r["rows"]])
        result["book_level"][f"delta_book:{mfw}"] = {
            "accuracy": r["accuracy"],
            "n_books": r["n_books"],
            "vs_stylo": vs_stylo(df, stylo_correct),
        }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записано: {OUT}", flush=True)


if __name__ == "__main__":
    main()
