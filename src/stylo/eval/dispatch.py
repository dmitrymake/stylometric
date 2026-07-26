"""Single-fit model routing for LOBO, GKF and train.

A spec's estimand lives *inside* the estimator that ``make_factory`` builds from the resolved
``training_weighting`` enum; this helper is pure, uniform group-routing so the same spec cannot
fit with different estimands across evaluation engines (GKF previously fit ``needs_groups``
models without groups). See research/work_balanced/model_routing.md §10.
"""
from __future__ import annotations

import pathlib


def frozen_run_contract(cfg, frags_root=None):
    """The frozen run-config corpus identity (root + policy) — the anchor the disk gate loads."""
    from .provenance import RunContract
    if frags_root is None:
        from ..dataset import resolve_fragment_roots

        frags_root = resolve_fragment_roots(cfg).train_root
    return RunContract.build(
        frags_root,
        cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [],
        cfg.get_path("corpus_policy.unknown_dir_name", "unknown"),
    )


def _canonical_wb_digest(cfg, frags_root, exclude, unknown) -> str:
    # NO cross-run cache (a warm digest could go stale vs a changed/cleared input_clean): recompute
    # from disk each call. run_final computes the contract ONCE and threads it to every spec, so a
    # full run still loads the corpus once — but never trusts a stale alias.
    from ..workdoc import load_work_balanced_dataset
    ds = load_work_balanced_dataset(frags_root, cfg=cfg, exclude_authors=exclude, unknown_name=unknown)
    return ds.provenance.rows_digest


def expected_data_contract(cfg, weighting, frags_root=None):
    """The current run's expected DATA contract (root, corpus policy, chunker/normalization hash).

    Derived from cfg — NOT the full model-config, so ablation overrides stay valid. For the
    work_balanced arm it also carries the canonical rows-digest recomputed independently from the
    on-disk manifests, so the guard can reject a fabricated/non-manifest Dataset. Used by each
    engine's ``require_dataset_for_weighting`` guard.
    """
    from .provenance import CorpusPolicyProvenance, DataContract
    from .work_weighting import WORK_BALANCED, resolve_training_weighting

    w = resolve_training_weighting(weighting)
    if frags_root is None:
        from ..dataset import resolve_fragment_roots

        frags_root = resolve_fragment_roots(cfg).train_root
    exclude = cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []
    unknown = cfg.get_path("corpus_policy.unknown_dir_name", "unknown")
    policy = CorpusPolicyProvenance.build(exclude, unknown)
    chash = None
    canonical = None
    if w == WORK_BALANCED:
        from ..workdoc import chunker_config_hash
        chash = chunker_config_hash(cfg)
        canonical = _canonical_wb_digest(cfg, frags_root, exclude, unknown)
    return DataContract(str(pathlib.Path(frags_root).resolve()), policy, chash, canonical)


def fit_estimator(est, texts, y, groups):
    """Fit ``est`` routing per-chunk ``groups`` iff it declares ``needs_groups``; fail-closed."""
    if getattr(est, "needs_groups", False):
        if groups is None:
            raise ValueError(f"{type(est).__name__} needs per-chunk groups but none were given")
        est.fit(texts, y, groups=groups)
    else:
        est.fit(texts, y)
    return est
