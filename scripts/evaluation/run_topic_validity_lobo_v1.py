#!/usr/bin/env python3
"""Preflight, time, or execute the approved aggregate-only LOBO topic-validity study."""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import tempfile
import time

from stylo.config import load_config
from stylo.eval.paired_audit.evaluator_v3_2 import (
    CANDIDATE_IDENTITY,
    LOBO_FOLD_IDENTITY,
    build_evaluation_context_v3_2,
)
from stylo.eval.paired_audit.run_plan import (
    blas_thread_fingerprint,
    env_lock_sha256,
    execution_source_sha256,
    git_commit_info,
    runtime_fingerprint,
    verify_installed_environment,
)
from stylo.eval.paired_audit.topic_validity_v1 import (
    TOPIC_ARMS_V1,
    TOPIC_CELLS_V1,
    build_topic_aggregate_v1,
    build_topic_study_context_v1,
    evaluate_topic_fold_v1,
    validate_topic_aggregate_v1,
)
from stylo.jsonio import canonical_hash, dumps_strict, load_strict

EXPECTED_OUTPUT = pathlib.Path("research/evidence/topic_validity_lobo_v1/aggregate.json")
THREAD_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


class TopicRunV1Error(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--timing-probe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--bundle-root", type=pathlib.Path, required=True)
    parser.add_argument("--historical-parent-root", type=pathlib.Path, required=True)
    parser.add_argument("--ruaa-selection-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def _selection(path: pathlib.Path) -> list[str]:
    raw = load_strict(path)
    authors = raw.get("authors") if type(raw) is dict else None
    if type(authors) is not dict:
        raise TopicRunV1Error("RuAA selection manifest has no exact authors mapping")
    work_ids = []
    for author, record in authors.items():
        books = record.get("books") if type(record) is dict else None
        if type(author) is not str or type(books) is not list or record.get("n_books") != len(books):
            raise TopicRunV1Error("malformed RuAA selection manifest")
        for book in books:
            slug = book.get("book") if type(book) is dict else None
            if type(slug) is not str or not slug:
                raise TopicRunV1Error("malformed RuAA selection work")
            work_ids.append(f"{author}/{slug}")
    work_ids.sort()
    if len(authors) != 22 or len(work_ids) != 137 or len(set(work_ids)) != 137:
        raise TopicRunV1Error("RuAA selection must be exactly 22 authors / 137 works")
    return work_ids


def _script_sha256() -> str:
    path = pathlib.Path(__file__).resolve()
    if path.is_symlink() or not path.is_file():
        raise TopicRunV1Error("runner source is missing or symlinked")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identities(repo: pathlib.Path, study) -> dict[str, str]:
    verify_installed_environment(repo)
    observed = {key: os.environ.get(key) for key in THREAD_ENV}
    if observed != THREAD_ENV:
        raise TopicRunV1Error(f"thread environment must equal {THREAD_ENV}; got {observed}")
    runtime = runtime_fingerprint()
    threads = blas_thread_fingerprint()
    return {
        "implementation_source_identity": canonical_hash([
            "stylo.topic_validity.execution.v1",
            execution_source_sha256(),
            _script_sha256(),
            study.binding["self_hash"],
        ]),
        "environment_lock_identity": env_lock_sha256(repo),
        "runtime_identity": canonical_hash(runtime),
        "thread_identity": canonical_hash(threads),
    }


def _load_study(args):
    repo = pathlib.Path(__file__).resolve().parents[2]
    git = git_commit_info(repo)
    if git["git_dirty"] or not git["git_commit"]:
        raise TopicRunV1Error("topic-validity preflight requires a clean committed checkout")
    cfg = load_config(repo / "configs/default.yaml")
    context = build_evaluation_context_v3_2(
        cfg=cfg,
        bundle_root=args.bundle_root.resolve(),
        historical_parent_root=args.historical_parent_root.resolve(),
        ruaa_parent_selection=_selection(args.ruaa_selection_manifest.resolve()),
    )
    study = build_topic_study_context_v1(cfg=cfg, context=context)
    if context.candidate_identity != CANDIDATE_IDENTITY:
        raise TopicRunV1Error("candidate identity drift")
    if context.lobo_manifest["self_hash"] != LOBO_FOLD_IDENTITY:
        raise TopicRunV1Error("LOBO fold identity drift")
    if (len(study.folds), len(study.metric_order), len(study.probability_order)) != (248, 43, 47):
        raise TopicRunV1Error("official topic-validity universe must be exactly 248/43/47")
    return repo, study, _identities(repo, study), git["git_commit"]


def _write_new_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise TopicRunV1Error("aggregate output already exists; refusing overwrite")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(dumps_strict(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _safe_preflight(study, identities, commit) -> None:
    print("preflight=ok", f"commit={commit}",
          f"context={study.binding['identities']['context_identity']}",
          f"binding={study.binding['self_hash']}", f"folds={len(study.folds)}",
          f"authors={len(study.metric_order)}", f"classes={len(study.probability_order)}",
          f"runtime={identities['runtime_identity']}", f"threads={identities['thread_identity']}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    repo_hint = pathlib.Path(__file__).resolve().parents[2]
    if (args.preflight_only or args.timing_probe) and args.output is not None:
        raise TopicRunV1Error("preflight/timing modes do not accept --output")
    if args.execute and (
        args.output is None or args.output.resolve() != (repo_hint / EXPECTED_OUTPUT).resolve()
    ):
        raise TopicRunV1Error(f"--output must be exactly {EXPECTED_OUTPUT.as_posix()}")
    repo, study, identities, commit = _load_study(args)
    _safe_preflight(study, identities, commit)
    if args.preflight_only:
        return 0
    if args.timing_probe:
        started = time.monotonic()
        evaluate_topic_fold_v1(study=study, cell="A0", arm="current", fold_index=0)
        print(f"timing_probe=ok fits=1 seconds={time.monotonic() - started:.3f}")
        return 0
    expected_output = (repo / EXPECTED_OUTPUT).resolve()
    if expected_output.exists() or expected_output.is_symlink():
        raise TopicRunV1Error("aggregate output already exists; execution is create-once")

    records = {cell: {arm: [] for arm in TOPIC_ARMS_V1} for cell in TOPIC_CELLS_V1}
    total = len(study.folds) * len(TOPIC_CELLS_V1) * len(TOPIC_ARMS_V1)
    completed = 0
    started = time.monotonic()
    for cell in TOPIC_CELLS_V1:
        for arm in TOPIC_ARMS_V1:
            for fold_index in range(len(study.folds)):
                records[cell][arm].append(evaluate_topic_fold_v1(
                    study=study, cell=cell, arm=arm, fold_index=fold_index
                ))
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(
                        f"progress={completed}/{total}",
                        f"cell={cell}",
                        f"arm={arm}",
                        f"elapsed_seconds={time.monotonic() - started:.1f}",
                        flush=True,
                    )
    aggregate = build_topic_aggregate_v1(study=study, records=records, **identities)
    validate_topic_aggregate_v1(aggregate, study=study, records=records, **identities)
    _write_new_json(expected_output, aggregate)
    loaded = load_strict(expected_output)
    validate_topic_aggregate_v1(loaded, study=study, records=records, **identities)
    for cell in loaded["cells"]:
        current = cell["accuracy"]["current"]
        strict = cell["accuracy"]["topic_strict"]
        delta = cell["delta_accuracy"]
        print(
            f"aggregate cell={cell['cell']}",
            f"current={current['correct']}/{current['total']}",
            f"topic_strict={strict['correct']}/{strict['total']}",
            f"delta={delta['numerator']}/{delta['denominator']}",
        )
    print(f"aggregate=ok self_hash={loaded['self_hash']} elapsed_seconds={time.monotonic() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
