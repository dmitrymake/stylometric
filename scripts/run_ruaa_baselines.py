"""Официальная baseline-таблица RuAA-Bench v1.0.

Гоняет тот же leak-free LOBO-движок (stylo.eval.final.run_final) на срезе
корпуса, ограниченном книгами бенчмарка (docs/ruaa_bench_manifest.json):
классика (Delta/char-cos/BoW), полный stylo и стек stylo_stack — одна таблица,
одни фолды, клaстер-робастная значимость против stylo уже внутри run_final.

Выход: docs/ruaa_bench_v1.json + docs/ruaa_bench_leaderboard.md +
эталонный сабмит data/ruaa_bench_v1/reference_submission_stylo.csv (для
smoke-проверки score_ruaa.py).

Run (ночной, часы): nice -n 10 .venv/bin/python scripts/run_ruaa_baselines.py
Быстрая проверка: --specs stylo --max-authors 5
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylo.config import load_config  # noqa: E402
from stylo.corpus import load_dataset, Dataset  # noqa: E402
from stylo.eval.final import run_final, format_final  # noqa: E402

BENCH_DOC = ROOT / "docs" / "ruaa_bench_manifest.json"
OUT_JSON = ROOT / "docs" / "ruaa_bench_v1.json"
OUT_MD = ROOT / "docs" / "ruaa_bench_leaderboard.md"
REF_CSV = ROOT / "data" / "ruaa_bench_v1" / "reference_submission_stylo.csv"

DEFAULT_SPECS = ["stylo", "stylo_stack", "delta:300", "delta_cos:500",
                 "char_cos", "bow_lr", "majority"]


def bench_subset(ds: Dataset, bench: dict, max_authors: int = 0) -> Dataset:
    keep_books = {f"{a}/{b['book']}"
                  for a, e in bench["authors"].items() for b in e["books"]}
    if max_authors:
        keep_authors = sorted(bench["authors"])[:max_authors]
        keep_books = {k for k in keep_books if k.split("/", 1)[0] in keep_authors}
    mask = np.array([g in keep_books for g in ds.groups])
    authors = sorted({g.split("/", 1)[0] for g in ds.groups[mask]})
    a2i = {a: i for i, a in enumerate(authors)}
    y = np.array([a2i[g.split("/", 1)[0]] for g in ds.groups[mask]])
    sub = Dataset(texts=ds.texts[mask], y=y, groups=ds.groups[mask], authors=authors)
    missing = keep_books - set(sub.groups.tolist())
    if missing:
        print(f"ВНИМАНИЕ: {len(missing)} книг бенчмарка нет во frags_train: "
              f"{sorted(missing)[:6]}...")
    return sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=",".join(DEFAULT_SPECS))
    ap.add_argument("--max-authors", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    specs = [s for s in args.specs.split(",") if s]

    bench = json.loads(BENCH_DOC.read_text("utf-8"))
    cfg = load_config()
    full = load_dataset(ROOT / "data" / "frags_train")
    ds = bench_subset(full, bench, args.max_authors)
    print(f"RuAA-срез: авторов={ds.n_authors} книг={len(set(ds.groups.tolist()))} "
          f"чанков={len(ds)}")

    out = run_final(cfg, ds, specs=specs, n_jobs=args.n_jobs)
    table, results = out["table"], out["results"]
    print(format_final(table, results))

    rows = table.to_dict(orient="records")
    OUT_JSON.write_text(json.dumps({
        "benchmark": f"{bench['name']} v{bench['version']}",
        "n_authors": ds.n_authors,
        "n_books": len(set(ds.groups.tolist())),
        "protocol": "leak-free LOBO book-level (см. data/ruaa_bench_v1/protocol.md)",
        "prereg": "docs/prereg_2026Q3.md",
        "leaderboard": rows,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    md = ["# RuAA-Bench v1.0 — официальные baseline-ы",
          "",
          "| модель | accuracy | 95% CI (книги) | macro-F1 | top-2 | Δacc vs stylo (author-clustered CI) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['accuracy']:.3f} | {r['acc_ci']} | "
                  f"{r['macro_f1']:.3f} | {r['top2']:.3f} | "
                  f"{r.get('vs_stylo_dacc_authorclustered_ci', '')} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    # эталонный сабмит для smoke score_ruaa.py — предсказания stylo
    ref = results.get("stylo")
    if ref is not None:
        REF_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(REF_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["book_id", "pred_author"])
            for r in ref["df"].itertuples():
                w.writerow([f"{r.test_author}/{r.test_book}", r.pred_author])
        print(f"эталонный сабмит: {REF_CSV.relative_to(ROOT)}")

    print(f"wrote {OUT_JSON.relative_to(ROOT)} | {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
