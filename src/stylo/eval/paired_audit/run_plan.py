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
import inspect
import math
import os
import pathlib
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from ...jsonio import dumps_strict

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# the runtime identity is a STRUCTURAL allowlist of numerical-stack fields — the kernel/OS is never
# collected and never scanned for (kernel strings do not participate in the scientific identity, not
# even as an anti-field fingerprint).
RUNTIME_ALLOWED_FIELDS = frozenset({"python", "python_implementation", "libc", "numpy", "scipy",
                                    "sklearn", "spacy", "joblib", "threadpoolctl"})
_BLAS_POOL_ALLOWED = frozenset({"internal_api", "version", "num_threads", "threading_layer",
                                "architecture"})
REGISTERED_RUN_KINDS = frozenset({"confirmatory", "smoke", "dry_preflight"})
REGISTERED_DTYPES = frozenset({"float64", "float32", "int64", "int32"})
# every continuous quantity the audit compares needs a registered numerical tolerance — a confirmatory
# run must bind ALL of them (an under-specified tolerance contract is fatal).
REGISTERED_TOLERANCE_QUANTITIES = frozenset({"probability", "accuracy", "delta_accuracy",
                                             "cluster_pvalue", "ci_endpoint"})
# the FROZEN confirmatory tolerances — tight (the recompute is deterministic, so these guard only float
# noise). Pinned like FROZEN_STATS so an oversized tolerance can never neuter the independent auditor.
FROZEN_TOLERANCES = {
    "probability": {"atol": 1e-9, "rtol": 0.0, "dtype": "float64"},
    "accuracy": {"atol": 1e-9, "rtol": 0.0, "dtype": "float64"},
    "delta_accuracy": {"atol": 1e-9, "rtol": 0.0, "dtype": "float64"},
    "cluster_pvalue": {"atol": 1e-9, "rtol": 0.0, "dtype": "float64"},
    "ci_endpoint": {"atol": 1e-9, "rtol": 0.0, "dtype": "float64"},
}


def class_order_digest(order) -> str:
    """The single canonical producer for a class-order digest (shared by the RunPlan, the fold
    manifest, and the checkpoint bindings, so no module invents its own scheme).

    Self-contained on the committed ``dumps_strict`` (no dependency on any rework-only helper), so a
    clean committed-snapshot checkout reproduces it bit-for-bit.
    """
    return hashlib.sha256(dumps_strict(list(order), sort_keys=True).encode("utf-8")).hexdigest()

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


# ── production evaluator identity (§4.2) ─────────────────────────────────────
# A confirmatory run may only execute a REGISTERED evaluator (by name); its source bytes, import
# identity, estimator config and mechanism passport are all folded into the run_id. A smoke/dummy
# evaluator is never in this allowlist, so it can never be run confirmatorily.
REGISTERED_CONFIRMATORY_EVALUATORS = frozenset({"work_balanced_ablation_factory"})
_EVALUATOR_IDENTITY_KEYS = ("name", "import_module", "import_qualname", "source_digest",
                            "estimator_config_digest", "mechanism_passport_digest")


@dataclass(frozen=True)
class EvaluatorSpec:
    """A registered per-fold estimator: its stable ``name``, the callable ``fn`` whose source bytes are
    hashed, its ``estimator_config`` and its ``mechanism_passport`` (the axis-mechanism description).
    All four bind into the run_id via :func:`evaluator_identity`."""
    name: str
    fn: Callable
    estimator_config: dict
    mechanism_passport: dict


def evaluator_identity(spec, *, confirmatory: bool) -> dict:
    """Resolve the run_id-binding identity of an :class:`EvaluatorSpec`: recompute the source-byte
    digest (so a changed factory re-keys the run), bind the import module+qualname, and hash the
    estimator config + mechanism passport. A confirmatory evaluator MUST be registered by name; a bare
    callable (no spec) is rejected outright."""
    if not isinstance(spec, EvaluatorSpec):
        raise RunPlanError("evaluator must be a registered EvaluatorSpec, not a bare callable")
    fn = spec.fn
    if not callable(fn):
        raise RunPlanError("EvaluatorSpec.fn is not callable")
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname:
        raise RunPlanError("EvaluatorSpec.fn lacks a stable import identity (module/qualname)")
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise RunPlanError(f"cannot read evaluator source for {qualname}: {exc}") from exc
    if not (isinstance(spec.name, str) and spec.name):
        raise RunPlanError("EvaluatorSpec.name must be a non-empty string")
    if not isinstance(spec.estimator_config, dict) or not spec.estimator_config:
        raise RunPlanError("EvaluatorSpec.estimator_config must be a non-empty dict")
    if not isinstance(spec.mechanism_passport, dict) or not spec.mechanism_passport:
        raise RunPlanError("EvaluatorSpec.mechanism_passport must be a non-empty dict")
    if confirmatory and spec.name not in REGISTERED_CONFIRMATORY_EVALUATORS:
        raise RunPlanError(
            f"evaluator {spec.name!r} is not a registered confirmatory evaluator "
            f"{sorted(REGISTERED_CONFIRMATORY_EVALUATORS)}")
    return {
        "name": spec.name,
        "import_module": module,
        "import_qualname": qualname,
        "source_digest": _sha256_bytes(source.encode("utf-8")),
        "estimator_config_digest": hashlib.sha256(
            dumps_strict(spec.estimator_config, sort_keys=True).encode("utf-8")).hexdigest(),
        "mechanism_passport_digest": hashlib.sha256(
            dumps_strict(spec.mechanism_passport, sort_keys=True).encode("utf-8")).hexdigest(),
    }


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


def _require(mapping: dict, keys, where: str) -> None:
    if not isinstance(mapping, dict):
        raise RunPlanError(f"{where} must be a dict")
    missing = [k for k in keys if k not in mapping or mapping[k] in (None, "")]
    if missing:
        raise RunPlanError(f"{where} missing required keys: {missing}")


def _validate_tolerance(quantity: str, tol) -> None:
    if not isinstance(tol, dict):
        raise RunPlanError(f"tolerances[{quantity!r}] must be a dict")
    for k in ("atol", "rtol"):
        v = tol.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise RunPlanError(f"tolerances[{quantity!r}].{k} must be a finite, non-negative number")
    if tol.get("dtype") not in REGISTERED_DTYPES:
        raise RunPlanError(f"tolerances[{quantity!r}].dtype must be one of {sorted(REGISTERED_DTYPES)}")


def _validate_runtime_fields(runtime_fingerprint, blas_thread_fingerprint, *, confirmatory: bool) -> None:
    """Structural allowlist for the runtime + BLAS fingerprints — the kernel/OS is never present."""
    if not isinstance(runtime_fingerprint, dict) or set(runtime_fingerprint) - RUNTIME_ALLOWED_FIELDS:
        raise RunPlanError(f"runtime_fingerprint may only carry {sorted(RUNTIME_ALLOWED_FIELDS)}")
    if any(not runtime_fingerprint.get(k) for k in ("python", "libc", "numpy", "scipy", "sklearn")):
        raise RunPlanError("runtime_fingerprint must bind python/libc/numpy/scipy/sklearn")
    if confirmatory and not runtime_fingerprint.get("spacy"):
        raise RunPlanError("a confirmatory runtime fingerprint must include spaCy (§4.2)")
    if not (isinstance(blas_thread_fingerprint, dict)
            and set(blas_thread_fingerprint) == {"threadpools", "thread_env"}):
        raise RunPlanError("blas_thread_fingerprint must carry exactly threadpools/thread_env")
    if not isinstance(blas_thread_fingerprint["threadpools"], list):
        raise RunPlanError("blas threadpools must be a list")
    for pool in blas_thread_fingerprint["threadpools"]:
        if not isinstance(pool, dict) or set(pool) - _BLAS_POOL_ALLOWED:
            raise RunPlanError(f"a blas threadpool may only carry {sorted(_BLAS_POOL_ALLOWED)}")


def build_run_plan(*, run_kind: str, git_commit: str, git_dirty: bool,
                   execution_source_sha256: str, env_lock_sha256: str, config_id: str,
                   runtime_fingerprint: dict, blas_thread_fingerprint: dict,
                   applicability_matrix_digest: str, a0_reference_shas: dict,
                   tolerances: dict, corpus_chain: dict, golden_fixture_inventory_sha: str,
                   evaluator_identity: dict, lobo: dict, ruaa: dict, stats: Optional[dict] = None,
                   audit_version: str = AUDIT_VERSION) -> dict:
    """Assemble the canonical RunPlan binding every §4.2 identity input. Missing bindings fail
    closed; OS/kernel strings are rejected. The plan is strict-JSON canonical (sort_keys)."""
    if run_kind not in REGISTERED_RUN_KINDS:
        raise RunPlanError(f"unknown run_kind {run_kind!r}; allowed {sorted(REGISTERED_RUN_KINDS)}")
    confirmatory = run_kind == "confirmatory"
    if stats is None:
        stats = dict(FROZEN_STATS)
    elif not isinstance(stats, dict):
        raise RunPlanError("stats must be a dict")
    else:
        stats = dict(stats)
    if set(stats) != set(FROZEN_STATS):
        raise RunPlanError("stats must carry exactly the frozen stat keys (§3.3/§3.5)")
    if confirmatory and stats != FROZEN_STATS:
        raise RunPlanError("a confirmatory run requires the frozen stat values (seed/B/δ/α/quantiles)")
    if confirmatory and git_dirty:
        raise RunPlanError("a confirmatory run requires a clean tree (git_dirty must be False)")
    # tolerances: reject NaN/Inf/negative and unregistered dtype for ANY run_kind (a NaN would
    # serialize identically and collide run_ids); require non-empty for a confirmatory run.
    if not isinstance(tolerances, dict):
        raise RunPlanError("tolerances must be a dict")
    for quantity, tol in tolerances.items():
        _validate_tolerance(quantity, tol)
    if confirmatory and tolerances != FROZEN_TOLERANCES:
        raise RunPlanError(
            "a confirmatory run must bind EXACTLY the frozen tolerances (no oversized value can neuter "
            "the independent auditor)")
    _validate_runtime_fields(runtime_fingerprint, blas_thread_fingerprint, confirmatory=confirmatory)

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

    _require(lobo, _REQUIRED_DATASET_KEYS, "lobo")
    _require(ruaa, _REQUIRED_DATASET_KEYS + ("selection_digest",), "ruaa")
    _require(a0_reference_shas, ("lobo_books_txt", "ruaa_reference_submission"), "a0_reference_shas")
    _require(corpus_chain, ("legacy_anchor", "semantic_parity_digest"), "corpus_chain")
    _require(evaluator_identity, _EVALUATOR_IDENTITY_KEYS, "evaluator_identity")
    if confirmatory and evaluator_identity["name"] not in REGISTERED_CONFIRMATORY_EVALUATORS:
        raise RunPlanError(f"confirmatory evaluator {evaluator_identity['name']!r} is not registered")
    for k in ("source_digest", "estimator_config_digest", "mechanism_passport_digest"):
        if not (isinstance(evaluator_identity.get(k), str) and _HEX64.match(evaluator_identity[k])):
            raise RunPlanError(f"evaluator_identity.{k} must be a sha256 hex digest")
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
        "evaluator_identity": evaluator_identity,
        "tolerances": tolerances,
        "stats": stats,
        "corpus_chain": corpus_chain,
        "golden_fixture_inventory_sha": golden_fixture_inventory_sha,
        "lobo": lobo,
        "ruaa": ruaa,
    }
    return plan


def run_id(run_plan: dict) -> str:
    """The canonical sha256 of the RunPlan — the immutable run identity."""
    return hashlib.sha256(dumps_strict(run_plan, sort_keys=True).encode("utf-8")).hexdigest()


# the exact build_run_plan parameters, so an embedded plan can be re-derived from its own fields
_PLAN_BUILD_KEYS = ("run_kind", "git_commit", "git_dirty", "execution_source_sha256", "env_lock_sha256",
                    "config_id", "runtime_fingerprint", "blas_thread_fingerprint",
                    "applicability_matrix_digest", "a0_reference_shas", "evaluator_identity",
                    "tolerances", "corpus_chain", "golden_fixture_inventory_sha", "stats", "lobo",
                    "ruaa", "audit_version")


def assert_wellformed_run_plan(plan: dict) -> None:
    """Re-derive the plan from its OWN fields via :func:`build_run_plan` and require bit-equality.

    Recomputing the run_id from an embedded plan only proves the plan hashes to its id — it does NOT
    re-apply the build invariants (a confirmatory plan's FROZEN stats + tolerances, clean tree, spaCy,
    a registered evaluator, hex digests). A trust boundary that receives a plan from an untrusted
    summary (the publisher, the loader) must re-run those invariants, or a forged confirmatory plan
    (oversized tolerance neutering the auditor, a weakened bootstrap, an inflated noninferiority margin)
    would still be self-consistent with its own id. Rebuilding via ``build_run_plan`` re-raises every
    invariant; a plan that does not rebuild to itself is rejected.
    """
    if not isinstance(plan, dict):
        raise RunPlanError("run_plan must be a dict")
    missing = [k for k in _PLAN_BUILD_KEYS if k not in plan]
    if missing:
        raise RunPlanError(f"run_plan missing build fields: {missing}")
    rebuilt = build_run_plan(**{k: plan[k] for k in _PLAN_BUILD_KEYS})
    if rebuilt != plan:
        raise RunPlanError("run_plan does not rebuild to itself (forged or non-canonical)")
