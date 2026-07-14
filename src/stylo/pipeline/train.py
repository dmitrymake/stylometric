"""Обучение продакшен-модели (config-driven, без тяжёлых артефактов).

Сохраняет:
  data/model.pkl    — Pipeline(StyloVectorizer -> Scaler -> LR)
  data/delta.pkl    — BurrowsDelta (настоящая, на MFW) для ансамбля
  data/authors.json — список авторов (индекс == метка)

НЕ сохраняет train_vectors.pkl: диагностики берут векторы из модели.
"""
from __future__ import annotations

import logging
import pathlib

import joblib

from ..config import load_config
from ..eval.dispatch import fit_estimator
from ..eval.lobo import make_factory
from ..eval.provenance import verify_dataset_against_disk
from ..eval.work_weighting import (CHUNK_WEIGHTED_LEGACY, WORK_BALANCED,
                                   require_weighting)
from ..features.reps import make_rep_cache
from ..jsonio import dump_strict
from ..models.delta import BurrowsDelta
from ..workdoc import chunker_config_hash, resolve_dataset
from .bundle import publish_bundle

log = logging.getLogger("stylo.pipeline.train")


def _code_tree_sha256() -> str | None:
    """Hash the CONTENT of the executing code tree (src/stylo/**/*.py), tracked or not — a bare
    ``git diff`` misses untracked B1/B2 files entirely, so this binds actual bytes+relpath+mode."""
    import hashlib
    src = pathlib.Path(__file__).resolve().parents[1]        # src/stylo
    try:
        files = sorted(p for p in src.rglob("*.py") if p.is_file())
        if any(p.is_symlink() for p in files):               # a symlinked .py must not be silently skipped
            raise RuntimeError("symlinked .py in the code tree — refusing to attest")
        h = hashlib.sha256()
        for p in files:
            rel = p.relative_to(src).as_posix()
            h.update(f"{rel}\x00{oct(p.stat().st_mode & 0o777)}\x00".encode("utf-8"))
            h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
            h.update(b"\n")
        return h.hexdigest()
    except Exception:
        return None


def _attestation(cfg) -> dict:
    """Bind the artifact to the executed code + resolved config.

    ``git_commit``/``git_dirty`` record the commit + dirtiness; ``code_tree_sha256`` hashes the
    actual on-disk code content (incl. untracked files); ``config_id`` hashes the fully-resolved
    config (incl. --set overrides). All four are required non-null by the bundle schema, so an
    environment that cannot attest fails publish closed.
    """
    import hashlib
    import subprocess

    from ..jsonio import dumps_strict
    root = pathlib.Path(__file__).resolve().parents[2]

    def _git(*args):
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True)
    try:
        commit = _git("rev-parse", "HEAD").strip() or None
        dirty = bool(_git("status", "--porcelain").strip())
    except Exception:
        commit, dirty = None, True
    config_id = hashlib.sha256(dumps_strict(cfg.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    return {"git_commit": commit, "git_dirty": dirty,
            "code_tree_sha256": _code_tree_sha256(), "config_id": config_id}


def run(cfg=None, warm: bool = True, *, weighting: str) -> None:
    cfg = cfg or load_config()
    weighting = require_weighting(weighting)   # strict; resolved once upstream, not re-read from cfg
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    data.mkdir(parents=True, exist_ok=True)

    exclude = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    unknown = cfg.get_path("corpus_policy.unknown_dir_name", "unknown")
    frags = data / "frags_train"
    ds = resolve_dataset(cfg, weighting, frags, exclude_authors=exclude, unknown_name=unknown)
    from ..eval.dispatch import frozen_run_contract
    verify_dataset_against_disk(cfg, ds, weighting, frozen_run_contract(cfg, frags))   # disk-anchored gate
    log.info("Train[%s]: %d чанков, %d авторов", weighting, len(ds), ds.n_authors)

    # snapshot the code/config attestation BEFORE the (multi-hour) fit; re-verify before publish so
    # a mid-run code edit cannot be signed as the trained tree.
    attest = _attestation(cfg)

    if warm:
        make_rep_cache(cfg).warm(list(ds.texts),
                                 n_process=cfg.get_path("language.parse_n_process", 4))

    # stylo pipeline built by the SAME factory LOBO uses (estimand baked in by weighting)
    pipe = make_factory("stylo", cfg, weighting=weighting)()
    log.info("Обучаю основной пайплайн…")
    fit_estimator(pipe, list(ds.texts), ds.y, ds.groups)   # groups routed iff needs_groups (WB); legacy: fit(X,y)

    mfw = cfg.get_path("delta.mfw_sizes", [300])
    delta = BurrowsDelta(mfw_count=mfw[len(mfw) // 2] if mfw else 300,
                         metric=cfg.get_path("delta.metric", "manhattan"),
                         training_weighting=weighting)
    fit_estimator(delta, list(ds.texts), ds.y, ds.groups)

    if weighting == CHUNK_WEIGHTED_LEGACY:
        # legacy production bundle — byte-identical layout, no new fields
        joblib.dump(pipe, data / "model.pkl")
        joblib.dump(delta, data / "delta.pkl")
        dump_strict(ds.authors, data / "authors.json", trailing_newline=False)
        log.info("Сохранено (legacy): %s", data / "model.pkl")
    else:
        # work_balanced: separate root + strict sidecar hashing all three artifacts + attestation
        from ..eval.provenance import safe_exploratory_dir
        bundle_dir = safe_exploratory_dir(data, "exploratory", "work_balanced")   # symlink-safe root
        if _code_tree_sha256() != attest["code_tree_sha256"]:   # code must not have changed mid-run
            raise RuntimeError("code tree changed during training — refusing to publish the bundle")
        meta = {
            "training_weighting": weighting,
            "dataset_contract": ds.provenance.loader_kind,
            "rows_digest": ds.provenance.rows_digest,
            "chunker_config_hash": chunker_config_hash(cfg),
            **attest,                                           # snapshot taken BEFORE fit
        }
        publish_bundle(bundle_dir, {
            "model.pkl": lambda p: joblib.dump(pipe, p),
            "delta.pkl": lambda p: joblib.dump(delta, p),
            "authors.json": lambda p: dump_strict(ds.authors, p, trailing_newline=False),
        }, meta)
        log.info("Сохранено (work_balanced bundle): %s", bundle_dir)
    log.info("Обучение завершено.")
