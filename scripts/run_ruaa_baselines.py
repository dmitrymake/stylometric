"""RuAA-Bench v1.0 baseline-таблица (reproducible_cv_legacy_not_blind).

Гоняет тот же leak-free LOBO-движок (stylo.eval.final.run_final) на срезе
корпуса, ограниченном книгами бенчмарка (docs/ruaa_bench_manifest.json):
классика (Delta/char-cos/BoW), полный stylo и калибровочный stylo_stack —
одна таблица, одни фолды, клaстер-робастная значимость против stylo уже
внутри run_final. Набор по умолчанию воспроизводит закоммиченный
docs/ruaa_bench_v1.json (7 строк); stylo_stack — медленный, поэтому для быстрой
проверки его исключают вручную (--specs stylo --max-authors 5).

Выход: docs/ruaa_bench_v1.json + docs/ruaa_bench_leaderboard.md +
эталонный сабмит data/ruaa_bench_v1/reference_submission_stylo.csv (для
smoke-проверки score_ruaa.py).

Run (ночной, часы): nice -n 10 .venv/bin/python scripts/run_ruaa_baselines.py
Быстрая проверка (без stylo_stack): --specs stylo,delta:300,char_cos --max-authors 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylo.claims import BenchmarkRole, ClaimStatus  # noqa: E402
from stylo.config import load_config  # noqa: E402
from stylo.corpus import load_dataset, Dataset  # noqa: E402
from stylo.eval.final import run_final, format_final  # noqa: E402
from stylo.jsonio import dump_strict  # noqa: E402

RUAA_V1_NOTE = (
    "Датированный воспроизводимый CV-срез на публичном корпусе; это не blind-"
    "лидерборд и не научный default. Идентификаторы раскрывают истину. "
    "Слепой пакет и work-balanced пересчёт — в v2. null в ece/vs_stylo_mcnemar_p "
    "означает неприменимость метрики (модель без вероятностей / сравнение модели с собой)."
)

BENCH_DOC = ROOT / "docs" / "ruaa_bench_manifest.json"
OUT_JSON = ROOT / "docs" / "ruaa_bench_v1.json"
OUT_MD = ROOT / "docs" / "ruaa_bench_leaderboard.md"
REF_CSV = ROOT / "data" / "ruaa_bench_v1" / "reference_submission_stylo.csv"


def _record_checksum(path: pathlib.Path) -> None:
    """Add/update ``path``'s entry in the package SHA256SUMS.

    build_ruaa_bench.py writes SHA256SUMS before this reference submission exists,
    so the file would otherwise be an unchecksummed package member.
    """
    sums_path = path.parent / "SHA256SUMS"
    rel = path.relative_to(path.parent).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries = {}
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if line:
                sha, _, name = line.partition("  ")
                entries[name] = sha
    entries[rel] = digest
    body = "\n".join(f"{entries[name]}  {name}" for name in sorted(entries))
    sums_path.write_text(body + "\n", encoding="utf-8")

# Order matches the committed docs/ruaa_bench_v1.json so a default rerun
# reproduces all 7 rows (incl. the slow calibration model stylo_stack).
DEFAULT_SPECS = ["stylo", "stylo_stack", "delta:300", "delta_cos:500",
                 "char_cos", "bow_lr", "majority"]


def bench_subset(ds: Dataset, bench: dict, max_authors: int = 0) -> Dataset:
    from stylo.eval.provenance import derive_dataset
    keep_books = {f"{a}/{b['book']}"
                  for a, e in bench["authors"].items() for b in e["books"]}
    if max_authors:
        keep_authors = sorted(bench["authors"])[:max_authors]
        keep_books = {k for k in keep_books if k.split("/", 1)[0] in keep_authors}
    idx = [i for i, g in enumerate(ds.groups) if g in keep_books]
    # provenance-preserving subset (no hand-built Dataset on a provenance-bearing path)
    sub = derive_dataset(ds, idx)
    missing = keep_books - set(sub.groups.tolist())
    if missing:
        print(f"ВНИМАНИЕ: {len(missing)} книг бенчмарка нет во frags_train: "
              f"{sorted(missing)[:6]}...")
    return sub, sorted(missing)


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
    ds, missing = bench_subset(full, bench, args.max_authors)
    print(f"RuAA-срез: авторов={ds.n_authors} книг={len(set(ds.groups.tolist()))} "
          f"чанков={len(ds)}")

    # Canonical publish ONLY for a complete, whole-benchmark run; a quick/partial run goes to the
    # exploratory namespace and never overwrites the canonical package / reference / SHA256SUMS.
    # canonical ONLY for the literal full spec list and no author cap (a permuted list or
    # --max-authors=-1 must NOT be treated as canonical)
    quick = (specs != list(DEFAULT_SPECS)) or (args.max_authors != 0)
    if missing and not quick:
        print(f"FATAL: {len(missing)} книг бенчмарка отсутствуют — canonical publish невозможен.")
        return 2
    canonical = not quick and not missing
    global OUT_JSON, OUT_MD, REF_CSV
    if not canonical:
        expl = ROOT / "docs" / "exploratory" / "ruaa"; expl.mkdir(parents=True, exist_ok=True)
        OUT_JSON = expl / "ruaa_bench_v1.json"
        OUT_MD = expl / "ruaa_bench_leaderboard.md"
        REF_CSV = expl / "reference_submission_stylo.csv"
        print(f"РЕЖИМ: exploratory (quick={quick}, missing={len(missing)}) → {expl.relative_to(ROOT)}")
    else:
        # docs/ruaa_bench_v1.json is a FROZEN P0 snapshot (its author-clustered CI sign is corrected
        # only via the versioned docs/ruaa_bench_v1.0.1.json — see scripts/apply_ci_sign_erratum.py).
        # A canonical re-run must not overwrite the frozen v1 paths; publish a new benchmark version.
        from stylo.eval.ci_erratum import assert_publish_target_not_frozen
        assert_publish_target_not_frozen(OUT_JSON)
        assert_publish_target_not_frozen(OUT_MD)

    # RuAA is pinned to the legacy estimand; the subset (built via derive_dataset) chains to the
    # disk-anchored full corpus, so run_final's internal disk verification accepts it — no
    # caller-supplied self-anchor.
    from stylo.config import with_overrides
    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY
    # RuAA evaluates exactly the bench authors, so its frozen contract is the FULL corpus with NO
    # benchmark exclusions — expressed as a trusted cfg-clone (not a caller-supplied contract). The
    # subset chains to that disk-anchored parent inside run_final's own verification.
    bench_cfg = with_overrides(cfg, {"corpus_policy.exclude_from_benchmark": []})
    out = run_final(bench_cfg, ds, specs=specs, n_jobs=args.n_jobs, weighting=CHUNK_WEIGHTED_LEGACY)
    table, results = out["table"], out["results"]
    print(format_final(table, results))

    rows = table.to_dict(orient="records")
    dump_strict({
        "benchmark": f"{bench['name']} v{bench['version']}",
        "claim_status": ClaimStatus.EXPLORATORY_INTERNAL.value,
        "benchmark_role": BenchmarkRole.REPRODUCIBLE_CV_LEGACY_NOT_BLIND.value,
        "training_weighting": "chunk_weighted_training_legacy",
        "note": RUAA_V1_NOTE,
        "n_authors": ds.n_authors,
        "n_books": len(set(ds.groups.tolist())),
        "protocol": "leak-free LOBO book-level (см. data/ruaa_bench_v1/protocol.md)",
        "prereg": "docs/prereg_2026Q3.md",
        "leaderboard": rows,
    }, OUT_JSON)

    md = ["# RuAA-Bench v1.0 — baseline-ы (`reproducible_cv_legacy_not_blind`)",
          "",
          "> Датированный воспроизводимый CV-срез на публичном корпусе, **не blind-лидерборд**",
          "> и не научный default. Взвешивание — `chunk_weighted_training_legacy`; work-balanced",
          "> пересчёт и слепой пакет — в v2. `null` в столбцах = метрика неприменима.",
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
        _record_checksum(REF_CSV)
        print(f"эталонный сабмит: {REF_CSV.relative_to(ROOT)}")

    print(f"wrote {OUT_JSON.relative_to(ROOT)} | {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
