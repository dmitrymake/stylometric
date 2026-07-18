#!/usr/bin/env python3
"""Run the bounded B4 W/F/R exploratory pilot on the frozen screening panel.

This entrypoint deliberately loads and verifies the legacy recursive corpus once, binds the
committed ``screening_panel_v1`` once, and then lets :mod:`stylo.eval.b4_pilot` inject every
ablation factory into those identical folds.  In particular, A4 is never used to select a second
dataset or a second fold assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import platform
import subprocess
import sys
from typing import Any, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylo.config import load_config  # noqa: E402
from stylo.eval.b4_pilot import (SCHEMA_VERSION, STATUS, format_compact_table,  # noqa: E402
                                 run_pilot)
from stylo.eval.dispatch import frozen_run_contract  # noqa: E402
from stylo.eval.groupkfold import bind_screening_panel, evaluate_frozen_panel_factory  # noqa: E402
from stylo.eval.provenance import verify_dataset_against_disk  # noqa: E402
from stylo.eval.screening_panel import verify_manifest  # noqa: E402
from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY  # noqa: E402
from stylo.features.reps import make_rep_cache  # noqa: E402
from stylo.jsonio import dumps_strict, load_strict  # noqa: E402
from stylo.workdoc import resolve_dataset  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT = ROOT / "docs" / "exploratory" / "work_balanced" / "b4_wfr_pilot_v1.json"
OUTPUT_ROOT = ROOT / "docs" / "exploratory" / "work_balanced"
_THREAD_ENV = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(value: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_metadata() -> tuple[str, bool]:
    commit = _git_output("rev-parse", "HEAD")
    # The pilot itself is intentionally uncommitted.  Record only tracked source drift here; the
    # exact uncommitted pilot/evaluator bytes are independently bound by ``code_hashes`` below.
    tracked = _git_output("status", "--porcelain=v1", "--untracked-files=no")
    return commit, bool(tracked)


def _code_hashes() -> dict[str, str]:
    # Bind every project Python dependency that a factory may import lazily on a later cell.  A
    # partial list would allow an interrupted artifact to mix cells from two dirty code states.
    paths = [pathlib.Path(__file__).resolve()]
    paths.extend(sorted((ROOT / "src" / "stylo").rglob("*.py")))
    paths.extend((ROOT / "pyproject.toml", ROOT / "requirements.lock"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pilot code-hash input missing: {missing}")
    return {path.relative_to(ROOT).as_posix(): _sha256_file(path) for path in paths}


def _runtime_fingerprint(cfg) -> dict[str, Any]:
    import numpy as np
    import scipy
    import scipy.linalg
    import sklearn
    import spacy
    import threadpoolctl

    # Ensure the NumPy and SciPy BLAS pools are loaded before taking the fingerprint.
    np.dot(np.ones((2, 2)), np.ones((2, 2)))
    scipy.linalg.svd(np.ones((2, 2)))

    model_name = cfg.get_path("language.spacy_model", "ru_core_news_lg")
    model_info = spacy.info(model_name)
    pools = []
    for pool in threadpoolctl.threadpool_info():
        pools.append(
            {
                key: pool.get(key)
                for key in (
                    "user_api",
                    "internal_api",
                    "prefix",
                    "filepath",
                    "version",
                    "num_threads",
                    "threading_layer",
                    "architecture",
                )
            }
        )
    pools.sort(
        key=lambda item: (
            str(item.get("internal_api")),
            str(item.get("prefix")),
            str(item.get("filepath")),
        )
    )
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "spacy": spacy.__version__,
        "spacy_model": {
            key: model_info.get(key)
            for key in ("name", "lang", "version", "spacy_version", "spacy_git_version")
        },
        "threadpool": pools,
        "thread_environment": {key: os.environ.get(key) for key in _THREAD_ENV},
    }


def _require_ignored_output(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"pilot output must resolve inside the repository: {resolved}") from exc
    try:
        resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"pilot output must stay inside {OUTPUT_ROOT.relative_to(ROOT)}: {resolved}"
        ) from exc
    if resolved.suffix != ".json":
        raise ValueError(f"pilot output must be a .json file: {resolved}")
    checked = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", "--", relative.as_posix()],
        check=False,
    )
    if checked.returncode != 0:
        raise ValueError(
            f"pilot output is not ignored by git; refusing to write a review/result artifact: {relative}"
        )
    if resolved.exists():
        existing = load_strict(resolved)
        if (
            not isinstance(existing, dict)
            or existing.get("status") != STATUS
            or existing.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError(f"existing output is not a {SCHEMA_VERSION} pilot artifact: {resolved}")
    return resolved


def _comma_list(value: str | None) -> list[str] | None:
    """Split only; the evaluator owns the strict allow-list and duplicate validation."""
    return None if value is None else value.split(",")


def _progress(event: dict[str, Any]) -> None:
    print(
        "progress "
        + dumps_strict(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _has_failed_status(value: Any) -> bool:
    """Return true only for an explicit failed cell; typed applicability statuses are successful."""
    if isinstance(value, dict):
        if value.get("status") == "failed":
            return True
        return any(_has_failed_status(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_failed_status(child) for child in value)
    return False


def _triage(artifact: dict[str, Any]) -> tuple[str, str]:
    raw = artifact.get("triage") or artifact.get("triage_conclusion") or {}
    if isinstance(raw, str):
        return raw, str(artifact.get("triage_rationale", ""))
    if isinstance(raw, dict):
        label = raw.get("label") or raw.get("conclusion") or raw.get("status") or "no_clear_signal"
        rationale = raw.get("rationale") or raw.get("reason") or "exploratory screening only"
        return str(label), str(rationale)
    return "no_clear_signal", "exploratory screening only"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B4 W/F/R exploratory proxy on screening_panel_v1."
    )
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", help="strict comma-separated model ids")
    parser.add_argument("--cells", help="strict comma-separated cell ids (A0..A4)")
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=None,
        help="author-clustered bootstrap iterations (default: evaluation.bootstrap_iters)",
    )
    parser.add_argument(
        "--no-warm-cache",
        action="store_true",
        help="skip the one-time representation-cache warmup (focused smoke/testing only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = _repo_path(args.config)
    output_path = _require_ignored_output(_repo_path(args.output))
    cfg = load_config(config_path)
    models = _comma_list(args.models)
    cells = _comma_list(args.cells)

    data_root = _repo_path(cfg.get_path("paths.data", "data")) / "frags_train"
    dataset = resolve_dataset(
        cfg,
        CHUNK_WEIGHTED_LEGACY,
        data_root,
        exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []),
        unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"),
    )
    verified = verify_dataset_against_disk(
        cfg,
        dataset,
        CHUNK_WEIGHTED_LEGACY,
        frozen_run_contract(cfg, data_root),
    )
    if verified != CHUNK_WEIGHTED_LEGACY:
        raise RuntimeError(f"unexpected verified weighting: {verified!r}")

    # Bind exactly once under the legacy corpus contract.  Every A0--A4 factory below receives this
    # same subset and manifest; A4 never triggers a work-balanced loader or dynamic split.
    panel_dataset, manifest = bind_screening_panel(cfg, dataset, CHUNK_WEIGHTED_LEGACY)
    verify_manifest(manifest)
    if manifest["n_authors"] != 43 or manifest["n_works"] != 251:
        raise RuntimeError(
            "screening_panel_v1 inventory drift: "
            f"expected 43 authors / 251 works, got {manifest['n_authors']} / {manifest['n_works']}"
        )

    if not args.no_warm_cache and (models is None or "stylo" in models):
        warmed = make_rep_cache(cfg).warm(
            list(panel_dataset.texts),
            n_process=cfg.get_path("language.parse_n_process", 4),
        )
        print(f"representation_cache_warmed={warmed}", flush=True)

    git_commit, git_dirty = _git_metadata()
    bootstrap_iters = (
        args.bootstrap_iters
        if args.bootstrap_iters is not None
        else cfg.get_path("evaluation.bootstrap_iters", 1000)
    )
    artifact = run_pilot(
        cfg,
        panel_dataset,
        manifest,
        output_path,
        config_path=str(config_path),
        config_sha256=_sha256_file(config_path),
        git_commit=git_commit,
        git_dirty=git_dirty,
        code_hashes=_code_hashes(),
        runtime_fingerprint=_runtime_fingerprint(cfg),
        models=models,
        cells=cells,
        seed=cfg.get_path("seed", 42),
        bootstrap_iters=bootstrap_iters,
        ci_level=cfg.get_path("evaluation.ci_level", 0.95),
        evaluator=evaluate_frozen_panel_factory,
        continue_on_error=True,
        progress=_progress,
    )
    print(format_compact_table(artifact))
    label, rationale = _triage(artifact)
    print(f"triage: {label} — {rationale}")
    return 1 if _has_failed_status(artifact) else 0


if __name__ == "__main__":
    raise SystemExit(main())
