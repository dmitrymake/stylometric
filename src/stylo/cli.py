"""Единая точка входа `stylo` (console-script). Подкоманды:

  stylo validate-corpus      — отчёт о качестве корпуса
  stylo clean                — очистка input -> input_clean (NER-маскировка)
  stylo split [--leave-out]  — нарезка на чанки по предложениям
  stylo warm                 — прогрев DocBin-кеша spaCy
  stylo train                — обучение продакшен-модели
  stylo lobo [--model spec]  — честный LOBO одной модели
  stylo sweep [--lobo]       — ablation-sweep «что работает»
  stylo predict              — атрибуция unknown
  stylo fetch-classics       — докачка public-domain классиков
  stylo report               — собрать отчёт
  stylo case run|rank|report|dossier — паспорта исторических кейсов

Глобально: --config PATH, --set key=value (повторяемо).
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import load_config, parse_set_overrides
from .jsonio import dump_strict, dumps_strict


def _add_global(p: argparse.ArgumentParser):
    p.add_argument("--config", default=None, help="Путь к YAML-конфигу")
    p.add_argument("--set", action="append", default=[], metavar="key=value",
                   help="Переопределить параметр конфига (повторяемо)")


def _cfg(args):
    return load_config(args.config, overrides=parse_set_overrides(args.set))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="stylo", description="Стилометрия авторства")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["validate-corpus", "clean", "warm", "train", "predict", "report",
                 "fetch-classics", "evaluate"]:
        sp = sub.add_parser(name)
        _add_global(sp)

    sp_split = sub.add_parser("split"); _add_global(sp_split)
    sp_split.add_argument("--leave-out", nargs="*", default=[])

    sp_lobo = sub.add_parser("lobo"); _add_global(sp_lobo)
    sp_lobo.add_argument("--model", default="stylo", help="stylo|delta:N|char_cos|bow_lr|majority")
    sp_lobo.add_argument("--max-books", type=int, default=0)

    sp_sweep = sub.add_parser("sweep"); _add_global(sp_sweep)
    sp_sweep.add_argument("--lobo", action="store_true", help="финал полным LOBO (медленно)")
    sp_sweep.add_argument("--no-baselines", action="store_true")

    sp_case = sub.add_parser("case"); _add_global(sp_case)
    case_sub = sp_case.add_subparsers(dest="case_cmd", required=True)
    sp_case_run = case_sub.add_parser("run")
    sp_case_run.add_argument("spec", help="YAML/JSON case spec")
    sp_case_run.add_argument("--out", default=None, help="Куда записать паспорт JSON")
    sp_case_rank = case_sub.add_parser("rank")
    sp_case_rank.add_argument("paths", nargs="+", help="Case specs или уже готовые passport JSON")
    sp_case_rank.add_argument("--out", default=None, help="Куда записать ранжированный JSON")
    sp_case_report = case_sub.add_parser("report")
    sp_case_report.add_argument("paths", nargs="+", help="Case specs или passport JSON")
    sp_case_report.add_argument("--out", default=None, help="Куда записать markdown-таблицу")
    sp_case_dossier = case_sub.add_parser("dossier")
    sp_case_dossier.add_argument("paths", nargs="+", help="Case specs или passport JSON")
    sp_case_dossier.add_argument("--out", default=None, help="Куда записать markdown-досье")

    sp_benchmark = sub.add_parser(
        "benchmark", help="SPOOF-RU/IDIOSHIFT-RU manifest, artifacts and blind scoring"
    )
    _add_global(sp_benchmark)
    benchmark_sub = sp_benchmark.add_subparsers(dest="benchmark_cmd", required=True)
    benchmark_sub.add_parser("schema", help="Print the canonical JSON Schema")
    sp_benchmark_validate = benchmark_sub.add_parser("validate")
    sp_benchmark_validate.add_argument("manifest")
    sp_benchmark_validate.add_argument(
        "--root", default=None, help="Also verify packaged text paths/hashes under ROOT"
    )
    sp_benchmark_score = benchmark_sub.add_parser("score")
    sp_benchmark_score.add_argument("manifest")
    sp_benchmark_score.add_argument("truth")
    sp_benchmark_score.add_argument("submission")
    sp_benchmark_score.add_argument("--boundary-tolerance", type=int, default=0)
    sp_benchmark_score.add_argument("--segment-iou", type=float, default=0.5)
    sp_benchmark_score.add_argument("--bootstrap-iters", type=int, default=1000)
    sp_benchmark_score.add_argument("--seed", type=int, default=42)
    sp_benchmark_score.add_argument(
        "--root", default=None, help="Verify exact text artifacts and truth offsets under ROOT"
    )
    sp_benchmark_score.add_argument("--out", default=None)

    sp_invariance = sub.add_parser(
        "invariance", help="Evaluate precomputed OOF predictions across nuisance factors"
    )
    _add_global(sp_invariance)
    sp_invariance.add_argument("table", help="CSV with true_label, pred_label and metadata")
    sp_invariance.add_argument(
        "--factors", nargs="+", default=["source", "edition"],
        help="Metadata columns to slice and diagnose",
    )
    sp_invariance.add_argument("--author-field", default="author")
    sp_invariance.add_argument("--cluster-fields", nargs="+", default=["author", "work"])
    sp_invariance.add_argument("--bootstrap-iters", type=int, default=1000)
    sp_invariance.add_argument("--seed", type=int, default=42)
    sp_invariance.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    cfg = _cfg(args)

    if args.cmd == "validate-corpus":
        from .corpus_tools import validate_corpus
        validate_corpus.run(cfg)
    elif args.cmd == "clean":
        from .pipeline import clean
        clean.run(cfg)
    elif args.cmd == "split":
        from .pipeline import split
        split.run(cfg, leave_out=args.leave_out)
    elif args.cmd == "warm":
        from .corpus import load_dataset
        from .features.reps import make_rep_cache
        import pathlib
        ds = load_dataset(pathlib.Path(cfg.get_path("paths.data", "data")) / "frags_train")
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    elif args.cmd == "train":
        from .pipeline import train
        train.run(cfg)
    elif args.cmd == "predict":
        from .pipeline import predict
        predict.run(cfg)
    elif args.cmd == "lobo":
        from .corpus import load_dataset
        from .features.reps import make_rep_cache
        from .eval.lobo import lobo_evaluate, format_top_candidates
        from .eval.metrics import summarize_book_results
        import pathlib
        ds = load_dataset(pathlib.Path(cfg.get_path("paths.data", "data")) / "frags_train",
                          exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []))
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
        df, _, _ = lobo_evaluate(cfg, ds, spec=args.model, max_books=args.max_books)
        s = summarize_book_results(df["true_label"].to_numpy(), df["pred_label"].to_numpy(),
                                   df["rank"].to_numpy(), ds.authors,
                                   iters=cfg.get_path("evaluation.bootstrap_iters", 1000),
                                   seed=cfg.get_path("seed", 42))
        print(f"\nLOBO[{args.model}]: acc={s['accuracy']} macroF1={s['macro_f1']} top2={s['top2']}")
    elif args.cmd == "sweep":
        from .corpus import load_dataset
        from .features.reps import make_rep_cache
        from .eval.sweep import run_sweep, format_sweep_table
        import pathlib
        ds = load_dataset(pathlib.Path(cfg.get_path("paths.data", "data")) / "frags_train",
                          exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []))
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
        sw = run_sweep(cfg, ds, strategy="lobo" if args.lobo else "gkf",
                       include_baselines=not args.no_baselines)
        table = format_sweep_table(sw["table"])
        print(table)
        import pathlib as _pl
        docs = _pl.Path(cfg.get_path("paths.docs", "docs")); docs.mkdir(parents=True, exist_ok=True)
        (docs / "sweep_table.txt").write_text(table, encoding="utf-8")
        sw["table"].to_csv(docs / "sweep_table.csv", index=False)
    elif args.cmd == "evaluate":
        import pathlib
        from .corpus import load_dataset
        from .features.reps import make_rep_cache
        from .eval.lobo import lobo_evaluate, write_book_report
        from .eval.final import run_final, format_final
        ds = load_dataset(pathlib.Path(cfg.get_path("paths.data", "data")) / "frags_train",
                          exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []))
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
        out = run_final(cfg, ds)
        txt = format_final(out["table"], out["results"])
        print(txt)
        docs = pathlib.Path(cfg.get_path("paths.docs", "docs")); docs.mkdir(parents=True, exist_ok=True)
        (docs / "final_comparison.txt").write_text(txt, encoding="utf-8")
        out["table"].to_csv(docs / "final_comparison.csv", index=False)
        write_book_report(out["results"]["stylo"]["df"], docs / "lobo_books.txt")
    elif args.cmd == "fetch-classics":
        from .corpus_tools import fetch_classics
        fetch_classics.run(cfg)
    elif args.cmd == "report":
        from .report import build
        build.run(cfg)
    elif args.cmd == "case":
        from .cases import cli as case_cli
        if args.case_cmd == "run":
            data = case_cli.run_spec(args.spec, out=args.out)
            print(dumps_strict(data, indent=2))
        elif args.case_cmd == "rank":
            data = case_cli.rank(args.paths, out=args.out)
            print(dumps_strict(data, indent=2))
        elif args.case_cmd == "report":
            print(case_cli.report(args.paths, out=args.out))
        elif args.case_cmd == "dossier":
            print(case_cli.dossier(args.paths, out=args.out))
        else:  # pragma: no cover
            parser.error(f"Неизвестная case-команда {args.case_cmd}")
    elif args.cmd == "benchmark":
        import dataclasses
        import pathlib
        from .benchmarks import (
            MANIFEST_SCHEMA,
            load_manifest,
            score_files,
            verify_manifest_artifacts,
        )

        if args.benchmark_cmd == "schema":
            data = MANIFEST_SCHEMA
        elif args.benchmark_cmd == "validate":
            manifest = load_manifest(args.manifest)
            data = {
                "valid": True,
                "dataset": dataclasses.asdict(manifest.dataset),
                "task_types": list(manifest.task_types),
                "n_documents": len(manifest.documents),
                "split_counts": {
                    split: sum(document.split == split for document in manifest.documents)
                    for split in sorted({document.split for document in manifest.documents})
                },
            }
            if args.root:
                data["artifacts"] = verify_manifest_artifacts(manifest, args.root).to_dict()
        elif args.benchmark_cmd == "score":
            manifest = load_manifest(args.manifest)
            score = score_files(
                manifest,
                args.manifest,
                args.truth,
                args.submission,
                artifact_root=args.root,
                boundary_tolerance=args.boundary_tolerance,
                segment_iou_threshold=args.segment_iou,
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed,
            )
            data = score.to_dict()
        else:  # pragma: no cover
            parser.error(f"Неизвестная benchmark-команда {args.benchmark_cmd}")
        rendered = dumps_strict(data, indent=2)
        if getattr(args, "out", None):
            dump_strict(data, args.out)
        print(rendered)
    elif args.cmd == "invariance":
        import pathlib
        import pandas as pd
        from .eval.invariance import evaluate_predictions

        table = pd.read_csv(args.table)
        required = {"true_label", "pred_label", args.author_field, *args.cluster_fields, *args.factors}
        missing = sorted(required - set(table.columns))
        if missing:
            parser.error(f"CSV не содержит обязательные колонки: {missing}")
        metadata = {column: table[column].to_numpy(dtype=object) for column in table.columns}
        report = evaluate_predictions(
            table["true_label"].to_numpy(dtype=object),
            table["pred_label"].to_numpy(dtype=object),
            metadata,
            factors=args.factors,
            author_field=args.author_field,
            cluster_fields=args.cluster_fields,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
        )
        rendered = dumps_strict(report.to_dict(), indent=2)
        if args.out:
            dump_strict(report.to_dict(), args.out)
        print(rendered)
    else:  # pragma: no cover
        parser.error(f"Неизвестная команда {args.cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
