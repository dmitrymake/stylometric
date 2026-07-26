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
from .models.registry import public_model_help


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

    simple_commands = {}
    for name in ["validate-corpus", "clean", "warm", "train", "predict", "report",
                 "fetch-classics", "evaluate"]:
        sp = sub.add_parser(name)
        _add_global(sp)
        simple_commands[name] = sp
    simple_commands["predict"].add_argument(
        "--model-bundle-token",
        required=False,
        help="Trusted content token printed by `stylo train`; required unless pinned in config",
    )
    simple_commands["validate-corpus"].add_argument(
        "--report-only",
        action="store_true",
        help="Write the advisory report but do not fail on severity=error findings",
    )

    sp_pre = sub.add_parser("preflight"); _add_global(sp_pre)
    sp_pre.add_argument("--stages", default="", help="comma-separated run-plan stages to validate")

    sp_split = sub.add_parser("split"); _add_global(sp_split)
    sp_split.add_argument("--leave-out", nargs="*", default=[])

    sp_lobo = sub.add_parser("lobo"); _add_global(sp_lobo)
    sp_lobo.add_argument(
        "--model",
        default="stylo",
        help=public_model_help(),
    )
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
    sp_benchmark_score.add_argument(
        "--truth-sha256",
        required=True,
        help="Independently published SHA-256 commitment of the exact truth file",
    )
    sp_benchmark_score.add_argument("--boundary-tolerance", type=int, default=0)
    sp_benchmark_score.add_argument("--segment-iou", type=float, default=0.5)
    sp_benchmark_score.add_argument("--bootstrap-iters", type=int, default=1000)
    sp_benchmark_score.add_argument("--seed", type=int, default=42)
    sp_benchmark_score.add_argument(
        "--segmentation-bootstrap-unit",
        choices=("work", "document"),
        default=None,
        help="Required when a registered segmentation endpoint is scored",
    )
    sp_benchmark_score.add_argument(
        "--root",
        required=True,
        help="Required benchmark package root for exact artifact/hash/offset verification",
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
        validate_corpus.run(cfg, report_only=args.report_only)
    elif args.cmd == "clean":
        from .pipeline import clean
        clean.run(cfg)
    elif args.cmd == "split":
        from .pipeline import split
        split.run(cfg, leave_out=args.leave_out)
    elif args.cmd == "warm":
        from .corpus import load_dataset
        from .dataset import resolve_fragment_roots
        from .features.reps import make_rep_cache
        ds = load_dataset(resolve_fragment_roots(cfg).train_root)
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    elif args.cmd == "preflight":
        # validate the WHOLE run-plan before any mutation: work_balanced cannot run predict/deploy
        from .eval.provenance import UnsupportedVariantError
        from .domain.work_weighting import CHUNK_WEIGHTED_LEGACY, resolve_training_weighting
        w = resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))
        stages = [s for s in args.stages.split(",") if s]
        if "predict" in stages and w != CHUNK_WEIGHTED_LEGACY:
            raise UnsupportedVariantError(
                f"run-plan includes 'predict' but weighting={w} has no supported deployment "
                "inference path — "
                "run the exploratory stages individually")
        print(f"preflight OK: weighting={w} stages={stages}")
    elif args.cmd == "train":
        from .pipeline import train
        from .domain.work_weighting import resolve_training_weighting
        receipt = train.run(
            cfg,
            weighting=resolve_training_weighting(
                cfg.get_path("evaluation.training_weighting")
            ),
        )
        print(
            dumps_strict(
                {
                    "bundle_version": receipt["bundle_version"],
                    "bundle_token": receipt["bundle_token"],
                    "training_weighting": receipt["training_weighting"],
                },
                sort_keys=True,
            )
        )
    elif args.cmd == "predict":
        from .pipeline import predict
        predict.run(cfg, expected_bundle_token=args.model_bundle_token)
    elif args.cmd == "lobo":
        from .features.reps import make_rep_cache
        from .eval.lobo import lobo_evaluate, format_top_candidates
        from .eval.metrics import summarize_book_results
        from .domain.work_weighting import resolve_training_weighting
        from .dataset import resolve_dataset
        weighting = resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))
        ds = resolve_dataset(cfg, weighting,
                             exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []),
                             unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"))
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
        df, _, _ = lobo_evaluate(cfg, ds, spec=args.model, max_books=args.max_books, weighting=weighting)
        s = summarize_book_results(df["true_label"].to_numpy(), df["pred_label"].to_numpy(),
                                   df["rank"].to_numpy(), ds.authors,
                                   iters=cfg.get_path("evaluation.bootstrap_iters", 1000),
                                   seed=cfg.get_path("seed", 42))
        print(f"\nLOBO[{args.model}]: acc={s['accuracy']} macroF1={s['macro_f1']} top2={s['top2']}")
    elif args.cmd == "sweep":
        from .features.reps import make_rep_cache
        from .eval.sweep import run_sweep, format_sweep_table
        from .domain.work_weighting import CHUNK_WEIGHTED_LEGACY, resolve_training_weighting
        from .dataset import resolve_dataset
        weighting = resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))
        ds = resolve_dataset(cfg, weighting,
                             exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []),
                             unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"))
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
        sw = run_sweep(cfg, ds, strategy="lobo" if args.lobo else "gkf",
                       include_baselines=not args.no_baselines, weighting=weighting)
        table = format_sweep_table(sw["table"])
        print(table)
        import hashlib as _hl
        import pathlib as _pl
        from .eval.provenance import safe_exploratory_dir, safe_write_batch
        from .pipeline.train import _attestation
        strategy = "lobo" if args.lobo else "gkf"
        docs = _pl.Path(cfg.get_path("paths.docs", "docs"))
        if weighting != CHUNK_WEIGHTED_LEGACY:
            docs = safe_exploratory_dir(docs, "exploratory", "work_balanced")   # symlink-safe
        else:
            docs.mkdir(parents=True, exist_ok=True)
        # v2: the GKF/LOBO proxy routes groups to needs_groups baselines (cross-engine estimand
        # uniformity). This group-aware routing differs from the historical sweep, so the output
        # uses a versioned name and never silently overwrites sweep_table.*.
        csv_text = sw["table"].to_csv(index=False)
        _h = lambda t: _hl.sha256(t.encode("utf-8")).hexdigest()
        prov_json = dumps_strict({
            "schema_version": "stylo.sweep.v2.provenance",
            "training_weighting": weighting, "strategy": strategy,
            "dataset_contract": getattr(ds.provenance, "loader_kind", None),
            "rows_digest": getattr(ds.provenance, "rows_digest", None),
            "attestation": _attestation(cfg),
            "files": {"sweep_table.v2.csv": _h(csv_text), "sweep_table.v2.txt": _h(table)},
            "note": f"v2 ({strategy}): proxy routes groups to needs_groups baselines",
        }, indent=2) + "\n"
        published = safe_write_batch(
            docs,
            {
                "sweep_table.v2.txt": table,
                "sweep_table.v2.csv": csv_text,
                "sweep_table.v2.provenance.json": prov_json,
            },
            publication_id="sweep-table-v2",
        )
        print(f"atomic sweep generation → {published['sweep_table.v2.txt'].parent}")
    elif args.cmd == "evaluate":
        import pathlib
        from .features.reps import make_rep_cache
        from .eval.lobo import write_book_report
        from .eval.final import run_final, format_final
        from .eval.provenance import assert_headline_write_allowed, safe_exploratory_dir
        from .domain.work_weighting import CHUNK_WEIGHTED_LEGACY, resolve_training_weighting
        from .jsonio import dump_strict
        from .dataset import resolve_dataset
        weighting = resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))
        ds = resolve_dataset(cfg, weighting,
                             exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []),
                             unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"))
        make_rep_cache(cfg).warm(list(ds.texts), n_process=cfg.get_path("language.parse_n_process", 4))
        out = run_final(cfg, ds, weighting=weighting)
        txt = format_final(out["table"], out["results"])
        print(txt)
        docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
        if weighting == CHUNK_WEIGHTED_LEGACY:
            assert_headline_write_allowed(weighting)   # fail-closed: only legacy writes headline
            # docs/final_comparison.* and docs/lobo_books.txt are FROZEN legacy baseline and
            # CI-sign-erratum sources. A fresh legacy recompute is written to the exploratory
            # namespace and NEVER overwrites the frozen headline (the canonical committed artifact).
            from .eval.ci_erratum import assert_publish_target_not_frozen
            rec = safe_exploratory_dir(docs, "exploratory", "legacy_recompute")
            (rec / "final_comparison.txt").write_text(txt, encoding="utf-8")
            out["table"].to_csv(rec / "final_comparison.csv", index=False)
            write_book_report(out["results"]["stylo"]["df"], rec / "lobo_books.txt")
            for _name in ("final_comparison.csv", "final_comparison.txt"):
                assert_publish_target_not_frozen(rec / _name)   # belt-and-suspenders: exploratory, not frozen docs/
            print(f"legacy recompute → {rec} (frozen docs/final_comparison.* untouched)")
        else:
            import hashlib as _hl
            from .eval.lobo import format_book_report
            from .eval.provenance import safe_write_batch
            from .pipeline.train import _attestation
            wbdir = safe_exploratory_dir(docs, "exploratory", "work_balanced")   # symlink-safe dir
            # WB CSV carries the five per-row provenance fields (byte-parity does not apply here)
            prov = out["provenance"]; per = prov["per_spec"]
            wb_table = out["table"].copy()
            wb_table["suite_weighting"] = prov["suite_weighting"]
            wb_table["dataset_contract"] = prov["dataset_contract"]
            wb_table["estimator_training_weighting"] = wb_table["model"].map(
                lambda m: per.get(m, {}).get("estimator_training_weighting"))
            wb_table["variant_role"] = wb_table["model"].map(lambda m: per.get(m, {}).get("variant_role"))
            wb_table["claim_status"] = wb_table["model"].map(lambda m: per.get(m, {}).get("claim_status"))
            csv_text = wb_table.to_csv(index=False)
            books_text = format_book_report(out["results"]["stylo"]["df"])
            outputs = {n: _hl.sha256(t.encode("utf-8")).hexdigest()
                       for n, t in {"final_comparison.txt": txt, "final_comparison.csv": csv_text,
                                    "lobo_books.txt": books_text}.items()}
            run_prov = {**prov, "attestation": _attestation(cfg), "output_sha256": outputs}
            published = safe_write_batch(
                wbdir,
                {
                    "final_comparison.txt": txt,
                    "final_comparison.csv": csv_text,
                    "lobo_books.txt": books_text,
                    "run_provenance.json": dumps_strict(run_prov, indent=2) + "\n",
                },
                publication_id="work-balanced-final-v1",
            )
            print(f"atomic evaluation generation → {published['run_provenance.json'].parent}")
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
                expected_truth_sha256=args.truth_sha256,
                boundary_tolerance=args.boundary_tolerance,
                segment_iou_threshold=args.segment_iou,
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed,
                segmentation_bootstrap_unit=args.segmentation_bootstrap_unit,
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
