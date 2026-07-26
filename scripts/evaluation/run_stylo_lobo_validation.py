#!/usr/bin/env python3
"""Run or resume the bounded 47-class stylo A0/A4/A1 per-book LOBO validation."""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import platform
import subprocess
import sys
from typing import Any, Sequence


# Thread defaults are installed before importing NumPy/SciPy/sklearn.  ``PYTHONHASHSEED`` is
# intentionally NOT set here: Python reads it before interpreter initialization, so an in-process
# setdefault would create a false-green environment string while hash randomization stayed enabled.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.config import load_config  # noqa: E402
from stylo.jsonio import canonical_hash, dumps_strict, load_strict  # noqa: E402
from stylo.eval.stylo_lobo_validation import (  # noqa: E402
    CELL_ORDER,
    MAX_TRUE_LOBO_WORKERS,
    SCHEMA_VERSION,
    STATUS,
    build_run_identity,
    derive_inventory,
    format_compact_table,
    load_pinned_a0_reference,
    run_true_lobo,
)
from stylo.eval.dispatch import frozen_run_contract  # noqa: E402
from stylo.eval.paired_audit.run_plan import verify_installed_environment  # noqa: E402
from stylo.eval.provenance import verify_dataset_against_disk  # noqa: E402
from stylo.eval.run_attestation import LiveRunAttestor  # noqa: E402
from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY  # noqa: E402
from stylo.features.reps import make_rep_cache  # noqa: E402
from stylo.dataset import resolve_dataset  # noqa: E402
from stylo.pipeline.split import resolve_fragment_snapshot  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT = (
    ROOT / "docs" / "exploratory" / "work_balanced" /
    "stylo_lobo_work_weighting_a0_a1_a4_v2.json"
)
OUTPUT_ROOT = ROOT / "docs" / "exploratory" / "work_balanced"
REFERENCE_PATH = ROOT / "docs" / "lobo_books.txt"
LEGACY_DATASET_DIGEST = "b4886a7cd723c04515b43f042467bc372af0aeaf28c47f517f0b2aa9d46b8c92"
_REQUIRED_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
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
    # The validation implementation may consist entirely of untracked files.  Include those paths
    # so the repository state cannot be reported as clean; the exact runtime bytes are also bound
    # independently by ``code_hashes``.
    dirty = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    return commit, bool(dirty)


def _code_hashes() -> dict[str, str]:
    paths = [pathlib.Path(__file__).resolve()]
    paths.extend(sorted((ROOT / "src" / "stylo").rglob("*.py")))
    paths.extend(
        (
            ROOT / "pyproject.toml",
            ROOT / "requirements.lock",
            ROOT / ".python-version",
        )
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"true-LOBO code-hash input missing: {missing}")
    return {path.relative_to(ROOT).as_posix(): _sha256_file(path) for path in paths}


def _pool_fingerprint() -> list[dict[str, Any]]:
    import threadpoolctl

    pools = []
    for pool in threadpoolctl.threadpool_info():
        pools.append({
            key: pool.get(key)
            for key in (
                "user_api", "internal_api", "prefix", "filepath", "version", "num_threads",
                "threading_layer", "architecture",
            )
        })
    pools.sort(key=lambda item: (
        str(item.get("internal_api")), str(item.get("prefix")), str(item.get("filepath"))))
    return pools


def _runtime_fingerprint(
    cfg,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import joblib
    import numpy as np
    import scipy
    import scipy.linalg
    import sklearn
    import spacy
    import threadpoolctl
    from threadpoolctl import threadpool_limits

    np.dot(np.ones((2, 2)), np.ones((2, 2)))
    scipy.linalg.svd(np.ones((2, 2)))
    model_name = cfg.get_path("language.spacy_model", "ru_core_news_lg")
    model_info = spacy.info(model_name)
    libc_name, libc_version = platform.libc_ver()
    observed_before = _pool_fingerprint()
    with threadpool_limits(limits=1):
        observed_limited = _pool_fingerprint()
    runtime = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libc": {"name": libc_name, "version": libc_version},
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "spacy": spacy.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "spacy_model": {
            key: model_info.get(key)
            for key in ("name", "lang", "version", "spacy_version", "spacy_git_version")
        },
    }
    thread = {
        "environment": {key: os.environ.get(key) for key in _THREAD_ENV},
        "observed_before_fold_limit": observed_before,
        "observed_under_fold_limit": observed_limited,
        "worker_policy": "threadpoolctl.threadpool_limits(limits=1) around every outer fold",
    }
    if any(
        pool.get("num_threads") not in (None, 1)
        for pool in observed_limited
        if pool.get("user_api") in {"blas", "openmp"}
    ):
        raise RuntimeError("numerical threadpool did not honor the one-thread fold limit")
    return runtime, thread


def _require_environment() -> None:
    # The path-free contract verifies exact installed core distribution pins
    # before this bound run may construct an identity or touch checkpoints.
    verify_installed_environment(ROOT)
    wrong = {
        key: os.environ.get(key) for key, expected in _REQUIRED_ENV.items()
        if os.environ.get(key) != expected
    }
    if wrong:
        raise RuntimeError(
            f"true-LOBO requires exact deterministic/thread environment {_REQUIRED_ENV}, got {wrong}")
    if sys.flags.hash_randomization != 0:
        raise RuntimeError(
            "PYTHONHASHSEED=0 was not active at interpreter startup "
            f"(sys.flags.hash_randomization={sys.flags.hash_randomization})")


def _require_ignored_output(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
        resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"true-LOBO output must stay under {OUTPUT_ROOT}: {resolved}") from exc
    if resolved.suffix != ".json":
        raise ValueError(f"true-LOBO output must be a .json file: {resolved}")
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", "--", relative.as_posix()],
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(f"true-LOBO output is not ignored by git: {relative}")
    if resolved.exists():
        existing = load_strict(resolved)
        if (
            not isinstance(existing, dict)
            or existing.get("status") != STATUS
            or existing.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError(f"existing output is not a {SCHEMA_VERSION} artifact: {resolved}")
    return resolved


def _checkpoint_root(output: pathlib.Path) -> pathlib.Path:
    root = output.with_suffix(".checkpoints")
    try:
        relative = root.resolve().relative_to(ROOT.resolve())
        root.resolve().relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"checkpoint root escapes exploratory output: {root}") from exc
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", "--", relative.as_posix()],
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(f"checkpoint root is not ignored by git: {relative}")
    return root


def _parse_cells(value: str) -> list[str]:
    cells = value.split(",")
    if not cells or any(not cell or cell.strip() != cell for cell in cells):
        raise ValueError("--cells must be a strict comma-separated list")
    if len(set(cells)) != len(cells) or any(cell not in CELL_ORDER for cell in cells):
        raise ValueError(f"--cells must contain unique values from {CELL_ORDER}")
    return cells


def _progress(event: dict[str, Any]) -> None:
    print(
        "progress " + dumps_strict(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _system_resources() -> dict[str, Any]:
    memory: dict[str, int] = {}
    meminfo = pathlib.Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[f"{key}_kib"] = int(value.strip().split()[0])
    return {"logical_cpus": os.cpu_count(), **memory}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run/resume the 47-class per-book LOBO stylo A0/A4/A1 validation.")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cells", default="A0,A1,A4")
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="compute/save only the first missing A0 fold for resource measurement",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_environment()
    if (
        type(args.n_jobs) is not int
        or args.n_jobs <= 0
        or args.n_jobs > MAX_TRUE_LOBO_WORKERS
    ):
        raise ValueError(
            f"--n-jobs must be a positive integer <= {MAX_TRUE_LOBO_WORKERS}"
        )
    cells = _parse_cells(args.cells)
    if args.smoke_only and cells != ["A0"]:
        raise ValueError("--smoke-only requires --cells A0")
    if not args.smoke_only and set(cells) != set(CELL_ORDER):
        raise ValueError("the scientific run requires exactly --cells A0,A1,A4")

    config_path = _repo_path(args.config)
    output_path = _require_ignored_output(_repo_path(args.output))
    checkpoint_root = _checkpoint_root(output_path)
    cfg = load_config(config_path)
    runtime_fingerprint, thread_fingerprint = _runtime_fingerprint(cfg)

    data_root = resolve_fragment_snapshot(
        _repo_path(cfg.get_path("paths.data", "data"))
    ).train_root
    dataset = resolve_dataset(
        cfg,
        CHUNK_WEIGHTED_LEGACY,
        data_root,
        exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []),
        unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"),
    )
    verified = verify_dataset_against_disk(
        cfg, dataset, CHUNK_WEIGHTED_LEGACY, frozen_run_contract(cfg, data_root))
    if verified != CHUNK_WEIGHTED_LEGACY:
        raise RuntimeError(f"unexpected verified corpus arm {verified!r}")
    # The lower-level resumable runner repeats this fail-closed check at its
    # own mutation boundary. This early copy is intentional: the executable
    # entrypoint must reject an ineligible snapshot before warming the
    # representation cache.
    from stylo.domain.corpus_identity import (  # noqa: E402
        assert_cross_work_content_isolation,
    )
    assert_cross_work_content_isolation(dataset.texts, dataset.groups)
    if dataset.provenance.rows_digest != LEGACY_DATASET_DIGEST:
        raise RuntimeError(
            f"legacy corpus digest drift: {dataset.provenance.rows_digest} != {LEGACY_DATASET_DIGEST}")

    inventory = derive_inventory(dataset, enforce_target=True)
    reference = load_pinned_a0_reference(REFERENCE_PATH, inventory)
    rep_cache = make_rep_cache(cfg)
    warmed = rep_cache.warm(
        list(dataset.texts), n_process=cfg.get_path("language.parse_n_process", 4))
    print(f"representation_cache_warmed={warmed}", flush=True)
    if not rep_cache.path.is_file():
        raise RuntimeError(f"warmed representation cache is missing: {rep_cache.path}")
    representation_cache = {
        "path": str(rep_cache.path.resolve()),
        "size_bytes": int(rep_cache.path.stat().st_size),
        "sha256": _sha256_file(rep_cache.path),
        "rep_version": str(rep_cache.rep_ver),
    }
    print(
        "representation_cache "
        + dumps_strict(representation_cache, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    print("resources " + dumps_strict(_system_resources(), sort_keys=True), flush=True)

    git_commit, git_dirty = _git_metadata()
    config = {
        "path": str(config_path),
        "sha256": _sha256_file(config_path),
        "resolved_sha256": canonical_hash(cfg.to_dict()),
    }
    code_hashes = _code_hashes()
    identity = build_run_identity(
        dataset=dataset,
        inventory=inventory,
        config=config,
        code_hashes=code_hashes,
        git_commit=git_commit,
        git_dirty=git_dirty,
        runtime_fingerprint=runtime_fingerprint,
        thread_fingerprint=thread_fingerprint,
        representation_cache=representation_cache,
        reference_sha256=reference["sha256"],
        seed=42,
        bootstrap_iters=10_000,
        ci_level=0.95,
        noninferiority_margin=0.02,
    )
    attestor = LiveRunAttestor.build(
        repository_root=ROOT,
        code_hashes=code_hashes,
        config_path=config_path,
        config_sha256=config["sha256"],
        cache_path=representation_cache["path"],
        cache_sha256=representation_cache["sha256"],
        cache_size_bytes=representation_cache["size_bytes"],
    )
    result = run_true_lobo(
        cfg,
        dataset,
        identity,
        reference,
        output_path=output_path,
        checkpoint_root=checkpoint_root,
        n_jobs=args.n_jobs,
        cells=cells,
        smoke_only=args.smoke_only,
        progress=_progress,
        attestor=attestor,
    )
    if args.smoke_only:
        print("smoke " + dumps_strict(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    print(format_compact_table(result))
    a0 = next(cell for cell in result["cells"] if cell["cell"] == "A0")
    a1 = result["comparisons"]["A1_minus_A0"]
    gate = result["primary_A4_noninferiority_gate"]
    print(f"A0 exact parity: {a0['metrics']['correct'] == 221 and result['a0_frozen_parity']['status'] == 'passed'}")
    print(f"A4 signed gate: {gate['decision']}")
    print(
        "A1 directional delta: "
        f"{a1['delta_accuracy']:+.6f} "
        f"[{a1['author_clustered_percentile_ci']['lo']:+.6f},"
        f"{a1['author_clustered_percentile_ci']['hi']:+.6f}] "
        f"gains/losses={a1['gains_count']}/{a1['losses_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
