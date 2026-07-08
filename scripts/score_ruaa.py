"""Score a RuAA-Bench v1.0 submission.

Submission: CSV with header `book_id,pred_author` (book_id = `<author>/<book>`),
optionally extra columns named after authors with class probabilities (then
top-2 is also scored). Every benchmark book must be predicted exactly once.

Usage:
  .venv/bin/python scripts/score_ruaa.py predictions.csv [--bench data/ruaa_bench_v1]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 42
ITERS = 1000
LEVEL = 0.95


def bootstrap_ci(values_fn, index_pool, iters=ITERS, level=LEVEL, seed=SEED):
    rng = np.random.RandomState(seed)
    stats = []
    pool = np.asarray(index_pool)
    for _ in range(iters):
        take = rng.choice(len(pool), size=len(pool), replace=True)
        stats.append(values_fn(pool[take]))
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def macro_f1(y_true, y_pred, authors):
    f1s = []
    for a in authors:
        tp = int(np.sum((y_true == a) & (y_pred == a)))
        fp = int(np.sum((y_true != a) & (y_pred == a)))
        fn = int(np.sum((y_true == a) & (y_pred != a)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--bench", default=str(ROOT / "data" / "ruaa_bench_v1"))
    args = ap.parse_args()

    bench = json.loads((pathlib.Path(args.bench) / "manifest.json").read_text("utf-8"))
    truth = {f"{a}/{b['book']}": a
             for a, e in bench["authors"].items() for b in e["books"]}
    authors = sorted(bench["authors"])

    preds = {}
    probs = {}
    with open(args.submission, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        prob_cols = [c for c in (reader.fieldnames or []) if c in set(authors)]
        for row in reader:
            bid = row["book_id"].strip()
            preds[bid] = row["pred_author"].strip()
            if prob_cols:
                probs[bid] = {c: float(row[c] or 0.0) for c in prob_cols}

    missing = sorted(set(truth) - set(preds))
    extra = sorted(set(preds) - set(truth))
    if missing or extra:
        print(f"ОШИБКА: пропущено {len(missing)} книг, лишних {len(extra)}", file=sys.stderr)
        for x in (missing + extra)[:10]:
            print("  ", x, file=sys.stderr)
        return 1

    books = sorted(truth)
    y_true = np.array([truth[b] for b in books])
    y_pred = np.array([preds[b] for b in books])
    correct = (y_true == y_pred).astype(float)

    acc = float(correct.mean())
    mf1 = macro_f1(y_true, y_pred, authors)
    idx = np.arange(len(books))
    acc_ci = bootstrap_ci(lambda t: correct[t].mean(), idx)
    book_authors = y_true
    uniq = np.unique(book_authors)

    def clustered_acc(author_take):
        mask = np.concatenate([np.where(book_authors == a)[0] for a in author_take])
        return correct[mask].mean()
    acc_ci_cl = bootstrap_ci(clustered_acc, uniq)

    report = {
        "benchmark": f"{bench['name']} v{bench['version']}",
        "n_books": len(books), "n_authors": len(authors),
        "accuracy": round(acc, 4),
        "accuracy_ci95_book_bootstrap": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
        "accuracy_ci95_author_clustered": [round(acc_ci_cl[0], 4), round(acc_ci_cl[1], 4)],
        "macro_f1": round(mf1, 4),
        "per_author_recall": {
            a: round(float(correct[y_true == a].mean()), 3) for a in authors},
    }
    if probs:
        top2 = 0
        for i, b in enumerate(books):
            order = sorted(probs[b], key=probs[b].get, reverse=True)[:2]
            top2 += int(y_true[i] in order)
        report["top2"] = round(top2 / len(books), 4)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
