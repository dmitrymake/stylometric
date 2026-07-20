"""Canonical RunPlan → run_id and the runtime/env/BLAS fingerprints (§4.2).

One canonical ``RunPlan`` whose sha256 **is** the ``run_id``. It binds every identity input of the
confirmatory audit — both dataset digests, both fold-manifest digests, both class-order pairs, the
applicability-matrix digest, config id, RunContract/selection digests, the pinned A0 reference SHAs,
run kind, tolerances, seeds and stat settings, audit version, git commit, execution-source hash,
environment-lock hash, the BLAS/thread fingerprint, the installed numerical-runtime fingerprint, the
corpus-chain digests, and the golden-fixture inventory SHA — so any code/config/env/runtime change
yields a different ``run_id`` and no mixing is representable.

The runtime identity deliberately **omits OS/kernel release and platform strings** while binding
libc and the numerical stack (Python / NumPy / SciPy / scikit-learn / spaCy / BLAS), as required.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import platform
import re
import subprocess
from typing import Optional

from ...jsonio import dumps_strict

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

AUDIT_VERSION = "work_balanced_paired_audit_v1"
_RUN_PLAN_VERSION = "paired_audit.run_plan.v1"

# Stat settings are frozen by the protocol (§3.3/§3.5): author-cluster bootstrap B=10000, seed=42,
# two-sided 95% CI, noninferiority margin δ=0.02, family α=0.05.
FROZEN_STATS = {
    "bootstrap_B": 10000,
    "bootstrap_iters": 10000,
    "seed": 42,
    "ci_level": 0.95,
    "quantiles": [2.5, 97.5],
    "noninferiority_margin": 0.02,
    "family_alpha": 0.05,
}


class RunPlanError(ValueError):
    """Fail-closed: the RunPlan is missing a required binding or carries a forbidden identity field."""


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _src_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]     # src/stylo


# ── fingerprints ─────────────────────────────────────────────────────────────
def runtime_fingerprint() -> dict:
    """Installed numerical-runtime identity — Python, libc, and the numerical stack versions.

    Deliberately omits ``platform.system``/``release``/``version``/``platform`` and the machine arch
    so a kernel/OS upgrade never re-keys the scientific identity; libc and the numerical stack ARE
    bound.
    """
    import numpy
    import scipy
    import sklearn
    libc = "/".join(x for x in platform.libc_ver() if x) or "unknown"
    fp = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "libc": libc,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import spacy
        fp["spacy"] = spacy.__version__
    except Exception:
        fp["spacy"] = None
    try:
        import joblib
        fp["joblib"] = joblib.__version__
    except Exception:
        fp["joblib"] = None
    try:
        import threadpoolctl
        fp["threadpoolctl"] = threadpoolctl.__version__
    except Exception:
        fp["threadpoolctl"] = None
    return fp


def blas_thread_fingerprint() -> dict:
    """BLAS/thread identity — threadpool internal APIs + thread counts and the pinned thread env,
    without leaking filesystem paths (prefix/filepath are dropped)."""
    pools = []
    try:
        import threadpoolctl
        for p in threadpoolctl.threadpool_info():
            pools.append({
                "internal_api": p.get("internal_api"),
                "version": p.get("version"),
                "num_threads": p.get("num_threads"),
                "threading_layer": p.get("threading_layer"),
                "architecture": p.get("architecture"),
            })
    except Exception:
        pools = []
    env = {k: os.environ.get(k, "") for k in
           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")}
    return {"threadpools": pools, "thread_env": env}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def env_lock_sha256(repo_root: Optional[pathlib.Path] = None) -> str:
    """sha256 over the environment lock files (requirements.lock and/or uv.lock). At least one must
    exist — an unlockable environment fails closed."""
    repo_root = pathlib.Path(repo_root) if repo_root is not None else _repo_root()
    h = hashlib.sha256()
    found = False
    for name in ("requirements.lock", "uv.lock"):
        p = repo_root / name
        if p.is_symlink():
            raise RunPlanError(f"environment lock file is a symlink (rejected): {name}")
        if p.is_file():
            found = True
            h.update(len(name).to_bytes(8, "big") + name.encode("utf-8"))
            h.update(p.read_bytes())
    if not found:
        raise RunPlanError("no environment lock file found (requirements.lock / uv.lock)")
    return h.hexdigest()


def execution_source_sha256(src_root: Optional[pathlib.Path] = None) -> str:
    """Content hash of the executing code tree (src/stylo/**/*.py) — binds actual bytes+relpath+mode,
    including untracked files; a symlinked .py fails closed."""
    src_root = pathlib.Path(src_root) if src_root is not None else _src_root()
    files = sorted(p for p in src_root.rglob("*.py") if p.is_file())
    if not files:
        raise RunPlanError(f"no .py source files under {src_root} — refusing an empty-tree attestation")
    if any(p.is_symlink() for p in files):
        raise RunPlanError("symlinked .py in the code tree — refusing to attest")
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(src_root).as_posix()
        h.update(f"{rel}\x00{oct(p.stat().st_mode & 0o777)}\x00".encode("utf-8"))
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def git_commit_info(repo_root: Optional[pathlib.Path] = None) -> dict:
    """The executed commit and its dirtiness (never bound to kernel/OS strings)."""
    repo_root = pathlib.Path(repo_root) if repo_root is not None else _repo_root()

    def _git(*args):
        return subprocess.check_output(["git", *args], cwd=repo_root,
                                       stderr=subprocess.DEVNULL, text=True)
    try:
        commit = _git("rev-parse", "HEAD").strip() or None
        dirty = bool(_git("status", "--porcelain").strip())
    except Exception:
        commit, dirty = None, True
    return {"git_commit": commit, "git_dirty": dirty}


def config_id(cfg) -> str:
    """sha256 of the fully-resolved config dict (incl. overrides)."""
    return hashlib.sha256(dumps_strict(cfg.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


# ── the canonical RunPlan ────────────────────────────────────────────────────
_REQUIRED_DATASET_KEYS = ("dataset_digest", "fold_manifest_digest",
                          "probability_class_order", "metric_label_order", "run_contract_digest")
# kernel/OS/hardware identity strings that must NEVER appear anywhere in the RunPlan. platform.system()
# (the bare OS name "Linux"/"Darwin") is deliberately excluded — it is a short, collision-prone token
# that would false-abort a legitimate field, and release/version/platform already prevent kernel drift.
_FORBIDDEN_IDENTITY = {platform.release(), platform.version(), platform.platform(),
                       platform.machine(), platform.node(), platform.processor()}


def _require(mapping: dict, keys, where: str) -> None:
    missing = [k for k in keys if k not in mapping or mapping[k] in (None, "")]
    if missing:
        raise RunPlanError(f"{where} missing required keys: {missing}")


def build_run_plan(*, run_kind: str, git_commit: str, git_dirty: bool,
                   execution_source_sha256: str, env_lock_sha256: str, config_id: str,
                   runtime_fingerprint: dict, blas_thread_fingerprint: dict,
                   applicability_matrix_digest: str, a0_reference_shas: dict,
                   tolerances: dict, corpus_chain: dict, golden_fixture_inventory_sha: str,
                   lobo: dict, ruaa: dict, stats: Optional[dict] = None,
                   audit_version: str = AUDIT_VERSION) -> dict:
    """Assemble the canonical RunPlan binding every §4.2 identity input. Missing bindings fail
    closed; OS/kernel strings are rejected. The plan is strict-JSON canonical (sort_keys)."""
    stats = dict(FROZEN_STATS) if stats is None else dict(stats)
    if set(stats) != set(FROZEN_STATS):
        raise RunPlanError("stats must carry exactly the frozen stat keys (§3.3/§3.5)")
    if run_kind == "confirmatory" and stats != FROZEN_STATS:
        raise RunPlanError("a confirmatory run requires the frozen stat values (seed/B/δ/α/quantiles)")

    # top-level §4.2 scalar identity inputs must be present and non-empty (fail-closed)
    _require({"run_kind": run_kind, "audit_version": audit_version, "git_commit": git_commit,
              "execution_source_sha256": execution_source_sha256, "env_lock_sha256": env_lock_sha256,
              "config_id": config_id, "applicability_matrix_digest": applicability_matrix_digest,
              "golden_fixture_inventory_sha": golden_fixture_inventory_sha},
             ("run_kind", "audit_version", "git_commit", "execution_source_sha256", "env_lock_sha256",
              "config_id", "applicability_matrix_digest", "golden_fixture_inventory_sha"), "run_plan")
    for k, v in (("execution_source_sha256", execution_source_sha256),
                 ("env_lock_sha256", env_lock_sha256), ("config_id", config_id),
                 ("applicability_matrix_digest", applicability_matrix_digest),
                 ("golden_fixture_inventory_sha", golden_fixture_inventory_sha)):
        if not (isinstance(v, str) and _HEX64.match(v)):
            raise RunPlanError(f"{k} must be a sha256 hex digest")
    if not (isinstance(runtime_fingerprint, dict)
            and all(runtime_fingerprint.get(k) for k in ("python", "libc", "numpy", "scipy", "sklearn"))):
        raise RunPlanError("runtime_fingerprint must bind python/libc/numpy/scipy/sklearn")
    if not (isinstance(blas_thread_fingerprint, dict)
            and {"threadpools", "thread_env"} <= set(blas_thread_fingerprint)):
        raise RunPlanError("blas_thread_fingerprint must carry threadpools/thread_env")

    _require(lobo, _REQUIRED_DATASET_KEYS, "lobo")
    _require(ruaa, _REQUIRED_DATASET_KEYS + ("selection_digest",), "ruaa")
    _require(a0_reference_shas, ("lobo_books_txt", "ruaa_reference_submission"), "a0_reference_shas")
    _require(corpus_chain, ("legacy_anchor", "semantic_parity_digest"), "corpus_chain")
    if not isinstance(git_dirty, bool):
        raise RunPlanError("git_dirty must be a bool")

    plan = {
        "schema": _RUN_PLAN_VERSION,
        "audit_version": audit_version,
        "run_kind": run_kind,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "execution_source_sha256": execution_source_sha256,
        "env_lock_sha256": env_lock_sha256,
        "config_id": config_id,
        "runtime_fingerprint": runtime_fingerprint,
        "blas_thread_fingerprint": blas_thread_fingerprint,
        "applicability_matrix_digest": applicability_matrix_digest,
        "a0_reference_shas": a0_reference_shas,
        "tolerances": tolerances,
        "stats": stats,
        "corpus_chain": corpus_chain,
        "golden_fixture_inventory_sha": golden_fixture_inventory_sha,
        "lobo": lobo,
        "ruaa": ruaa,
    }
    assert_run_plan_omits_kernel_strings(plan)
    return plan


def _iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def assert_run_plan_omits_kernel_strings(plan: dict) -> None:
    """Fail-closed if any OS/kernel/platform identity string leaked into the RunPlan (the scientific
    identity must not depend on the kernel release)."""
    forbidden = {s for s in _FORBIDDEN_IDENTITY if s}
    for s in _iter_strings(plan):
        if s in forbidden:
            raise RunPlanError(f"RunPlan must not bind an OS/kernel/platform string: {s!r}")


def run_id(run_plan: dict) -> str:
    """The canonical sha256 of the RunPlan — the immutable run identity."""
    return hashlib.sha256(dumps_strict(run_plan, sort_keys=True).encode("utf-8")).hexdigest()
