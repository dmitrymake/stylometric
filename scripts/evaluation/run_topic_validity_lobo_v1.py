#!/usr/bin/env python3
"""Preflight, time, or execute the LOBO topic-validity study with resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import pathlib
import signal
import tempfile
import time

from stylo.config import load_config
from stylo.eval.paired_audit.evaluator_v3_2 import (CANDIDATE_IDENTITY, LOBO_FOLD_IDENTITY,
                                                    build_evaluation_context_v3_2)
from stylo.eval.paired_audit.run_plan import (
    blas_thread_fingerprint,
    env_lock_sha256,
    execution_source_sha256,
    git_commit_info,
    runtime_fingerprint,
    verify_installed_environment,
)
from stylo.eval.paired_audit.topic_validity_v1 import (TOPIC_ARMS_V1, TOPIC_CELLS_V1,
                                                       build_topic_aggregate_v1,
                                                       build_topic_study_context_v1,
                                                       evaluate_topic_fold_v1,
                                                       validate_topic_aggregate_v1)
from stylo.eval.prediction_contract import stable_top1_and_worst_tie_rank
from stylo.features.reps import make_rep_cache
from stylo.jsonio import canonical_hash, dumps_strict, load_strict

EXPECTED_OUTPUT = pathlib.Path("research/evidence/topic_validity_lobo_v1/aggregate.json")
DEFAULT_CHECKPOINT = pathlib.Path("research/local/topic_validity_lobo_v1.checkpoint.json")
CHECKPOINT_SCHEMA = "stylo.topic_validity.checkpoint.v1"
THREAD_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1"}
DEFAULT_WORKERS = 8
SAVE_EVERY = 10
POLL_SECONDS = 5.0
_WORKER_STUDY = None
_STOP_REQUESTED = False


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
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None,
                        help="resumable progress file; defaults to the ignored local path")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore and overwrite an existing checkpoint")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cells", default=",".join(TOPIC_CELLS_V1),
                        help="comma-separated subset of the study cells")
    parser.add_argument("--arms", default=",".join(TOPIC_ARMS_V1),
                        help="comma-separated subset of the study arms")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="stop and keep progress after this many hours; 0 disables the limit")
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


def _selectors(raw: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    chosen = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not chosen or any(item not in allowed for item in chosen) or len(set(chosen)) != len(chosen):
        raise TopicRunV1Error(f"--{label} must be a unique subset of {list(allowed)}")
    return tuple(item for item in allowed if item in chosen)


def _load_study(args):
    repo = pathlib.Path(__file__).resolve().parents[2]
    git = git_commit_info(repo)
    if not git["git_commit"]:
        raise TopicRunV1Error("topic-validity execution requires a committed checkout")
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
    return repo, study, _identities(repo, study), git


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


def _replace_json(path: pathlib.Path, value: dict) -> None:
    """Atomically write or overwrite a resumable progress file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(dumps_strict(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def _run_identity(study, identities: dict[str, str], commit: str) -> str:
    return canonical_hash([
        CHECKPOINT_SCHEMA,
        study.binding["self_hash"],
        study.binding["identities"]["context_identity"],
        identities["implementation_source_identity"],
        identities["environment_lock_identity"],
        commit,
    ])


def _empty_records() -> dict:
    return {cell: {arm: [] for arm in TOPIC_ARMS_V1} for cell in TOPIC_CELLS_V1}


def _load_checkpoint(path: pathlib.Path, run_identity: str) -> dict:
    if not path.exists():
        return _empty_records()
    stored = load_strict(path)
    if type(stored) is not dict or stored.get("schema") != CHECKPOINT_SCHEMA:
        raise TopicRunV1Error(f"{path} is not a topic-validity checkpoint")
    if stored.get("run_identity") != run_identity:
        raise TopicRunV1Error(
            f"{path} belongs to a different study, source tree or environment; "
            "pass --fresh to discard it"
        )
    records = _empty_records()
    for cell in TOPIC_CELLS_V1:
        for arm in TOPIC_ARMS_V1:
            rows = stored.get("records", {}).get(cell, {}).get(arm, [])
            if type(rows) is not list:
                raise TopicRunV1Error("checkpoint records are malformed")
            records[cell][arm] = sorted(rows, key=lambda row: row["fold_index"])
    return records


def _save_checkpoint(path: pathlib.Path, run_identity: str, records: dict, *,
                     commit: str, elapsed: float) -> None:
    _replace_json(path, {
        "schema": CHECKPOINT_SCHEMA,
        "run_identity": run_identity,
        "git_commit": commit,
        "elapsed_seconds": round(elapsed, 1),
        "completed": {cell: {arm: len(records[cell][arm]) for arm in TOPIC_ARMS_V1}
                      for cell in TOPIC_CELLS_V1},
        "records": records,
    })


def _pending(records: dict, study, cells: tuple[str, ...], arms: tuple[str, ...]) -> list[tuple]:
    tasks = []
    for cell in cells:
        for arm in arms:
            done = {row["fold_index"] for row in records[cell][arm]}
            tasks.extend((cell, arm, index) for index in range(len(study.folds))
                         if index not in done)
    return tasks


def _summarise(study, records: dict) -> None:
    """Report top-1 accuracy for every complete arm and the delta once a cell has both."""
    width = len(study.probability_order)
    scores = {}
    for cell in TOPIC_CELLS_V1:
        for arm in TOPIC_ARMS_V1:
            rows = records[cell][arm]
            if not rows:
                continue
            by_index = {row["fold_index"]: row for row in rows}
            correct = 0
            for expected in study.folds:
                row = by_index.get(expected.fold_index)
                if row is None:
                    continue
                decision = stable_top1_and_worst_tie_rank(
                    row["whole_work_probabilities"], true_label=expected.true_label,
                    expected_width=width,
                )
                correct += int(decision["top1_correct"])
            scores[(cell, arm)] = (correct, len(rows))
            print(f"partial cell={cell} arm={arm} top1={correct}/{len(rows)}"
                  f" ({correct / len(rows):.4f})", flush=True)
    for cell in TOPIC_CELLS_V1:
        current, strict = scores.get((cell, "current")), scores.get((cell, "topic_strict"))
        if current and strict and current[1] == strict[1] == len(study.folds):
            delta = (strict[0] - current[0]) / len(study.folds)
            print(f"partial cell={cell} delta_topic_strict_minus_current={delta:+.4f}", flush=True)


def _warm_representations(study) -> None:
    started = time.monotonic()
    workers = min(8, max(1, int(study.cfg.get_path("language.parse_n_process", 1))))
    created = make_rep_cache(study.cfg).warm(
        list(study.parent.lobo_dataset.texts), n_process=workers, batch_size=32
    )
    print(f"representation_warm=ok rows={len(study.parent.lobo_dataset.texts)} created={created} "
          f"workers={workers} seconds={time.monotonic() - started:.1f}", flush=True)


def _worker_fold(task):
    cell, arm, fold_index = task
    if _WORKER_STUDY is None:
        raise TopicRunV1Error("fork worker has no inherited sealed study")
    return cell, arm, evaluate_topic_fold_v1(
        study=_WORKER_STUDY, cell=cell, arm=arm, fold_index=fold_index
    )


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"stop_requested signal={signum}; finishing in-flight fits and saving progress",
          flush=True)


def _sorted_in_place(records: dict) -> None:
    for cell in TOPIC_CELLS_V1:
        for arm in TOPIC_ARMS_V1:
            records[cell][arm].sort(key=lambda row: row["fold_index"])


def _serial_records(study, records: dict, tasks: list[tuple], *, started: float,
                    deadline: float | None, save) -> str:
    """Single-process path: same stop, deadline and checkpoint semantics without a pool."""
    reason, done = "complete", 0
    try:
        for cell, arm, fold_index in tasks:
            if _STOP_REQUESTED:
                reason = "stop_requested"
                break
            if deadline is not None and time.monotonic() >= deadline:
                reason = "time_limit"
                break
            records[cell][arm].append(
                evaluate_topic_fold_v1(study=study, cell=cell, arm=arm, fold_index=fold_index)
            )
            done += 1
            if done % SAVE_EVERY == 0:
                save(records)
                print(f"progress={done}/{len(tasks)} cell={cell} arm={arm} "
                      f"elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
    finally:
        _sorted_in_place(records)
        save(records)
    return reason


def _collect_records(study, records: dict, tasks: list[tuple], *, workers: int, started: float,
                     deadline: float | None, save) -> str:
    """Run pending fits, checkpointing as results arrive. Returns why the loop ended."""
    global _WORKER_STUDY, _STOP_REQUESTED
    if not tasks:
        return "complete"
    if workers < 1:
        raise TopicRunV1Error("--workers must be at least one")
    if workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        raise TopicRunV1Error("multi-process execution requires the fork start method")
    _STOP_REQUESTED = False
    previous = {number: signal.signal(number, _request_stop)
                for number in (signal.SIGINT, signal.SIGTERM)}
    if workers == 1:
        try:
            return _serial_records(study, records, tasks, started=started, deadline=deadline,
                                   save=save)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    _WORKER_STUDY = study
    pool = multiprocessing.get_context("fork").Pool(processes=workers)
    reason, done = "complete", 0
    try:
        iterator = pool.imap(_worker_fold, tasks, chunksize=1)
        while done < len(tasks):
            if _STOP_REQUESTED:
                reason = "stop_requested"
                break
            if deadline is not None and time.monotonic() >= deadline:
                reason = "time_limit"
                break
            try:
                cell, arm, record = iterator.next(timeout=POLL_SECONDS)
            except multiprocessing.TimeoutError:
                continue
            records[cell][arm].append(record)
            done += 1
            if done % SAVE_EVERY == 0:
                save(records)
                print(f"progress={done}/{len(tasks)} cell={cell} arm={arm} "
                      f"elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
        pool.terminate()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        _WORKER_STUDY = None
        for number, handler in previous.items():
            signal.signal(number, handler)
        _sorted_in_place(records)
        save(records)
    return reason


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    repo_hint = pathlib.Path(__file__).resolve().parents[2]
    if (args.preflight_only or args.timing_probe) and args.output is not None:
        raise TopicRunV1Error("preflight/timing modes do not accept --output")
    if args.execute and (
        args.output is None or args.output.resolve() != (repo_hint / EXPECTED_OUTPUT).resolve()
    ):
        raise TopicRunV1Error(f"--output must be exactly {EXPECTED_OUTPUT.as_posix()}")
    cells = _selectors(args.cells, TOPIC_CELLS_V1, "cells")
    arms = _selectors(args.arms, TOPIC_ARMS_V1, "arms")
    repo, study, identities, git = _load_study(args)
    print("preflight=ok", f"commit={git['git_commit']}", f"dirty={git['git_dirty']}",
          f"context={study.binding['identities']['context_identity']}",
          f"binding={study.binding['self_hash']}", f"folds={len(study.folds)}",
          f"authors={len(study.metric_order)}", f"classes={len(study.probability_order)}",
          f"runtime={identities['runtime_identity']}", f"threads={identities['thread_identity']}")
    if args.preflight_only:
        return 0
    if args.timing_probe:
        _warm_representations(study)
        started = time.monotonic()
        evaluate_topic_fold_v1(study=study, cell=cells[0], arm=arms[0], fold_index=0)
        print(f"timing_probe=ok fits=1 seconds={time.monotonic() - started:.3f}")
        return 0

    expected_output = (repo / EXPECTED_OUTPUT).resolve()
    if expected_output.exists() or expected_output.is_symlink():
        raise TopicRunV1Error("aggregate output already exists; execution is create-once")
    checkpoint = (args.checkpoint or (repo / DEFAULT_CHECKPOINT)).resolve()
    run_identity = _run_identity(study, identities, git["git_commit"])
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()
    records = _load_checkpoint(checkpoint, run_identity)
    restored = sum(len(records[cell][arm]) for cell in TOPIC_CELLS_V1 for arm in TOPIC_ARMS_V1)
    started = time.monotonic()
    deadline = started + args.max_hours * 3600 if args.max_hours > 0 else None
    print(f"checkpoint={checkpoint} restored_fits={restored} run_identity={run_identity}"
          f" cells={list(cells)} arms={list(arms)} workers={args.workers}"
          f" max_hours={args.max_hours or 'none'}", flush=True)

    _warm_representations(study)
    tasks = _pending(records, study, cells, arms)
    print(f"pending_fits={len(tasks)}", flush=True)
    reason = _collect_records(
        study, records, tasks, workers=args.workers, started=started, deadline=deadline,
        save=lambda current: _save_checkpoint(
            checkpoint, run_identity, current, commit=git["git_commit"],
            elapsed=time.monotonic() - started,
        ),
    )
    _summarise(study, records)
    total = sum(len(records[cell][arm]) for cell in TOPIC_CELLS_V1 for arm in TOPIC_ARMS_V1)
    complete = total == len(TOPIC_CELLS_V1) * len(TOPIC_ARMS_V1) * len(study.folds)
    if not complete:
        print(f"execution={reason} completed_fits={total} checkpoint_kept={checkpoint}"
              f" elapsed_seconds={time.monotonic() - started:.1f}"
              " — rerun the same command to resume", flush=True)
        return 0

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
