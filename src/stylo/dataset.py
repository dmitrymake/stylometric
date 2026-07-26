"""Application-level corpus routing over one coherently resolved fragment snapshot.

Neutral corpus/work identity remains in :mod:`stylo.domain`; this module owns
the outward choice between the legacy recursive and work-manifest loaders.
"""
from __future__ import annotations

import pathlib
from collections.abc import Iterable

from .corpus import load_dataset
from .pipeline.split import FragmentSnapshot, resolve_fragment_snapshot
from .workdoc import load_work_balanced_dataset


def resolve_fragment_roots(cfg) -> FragmentSnapshot:
    """Resolve the configured data root to one train/unknown/map generation."""

    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    return resolve_fragment_snapshot(data)


def resolve_dataset(
    cfg,
    training_weighting: str,
    frags_root: str | pathlib.Path | None = None,
    *,
    exclude_authors: Iterable[str] = (),
    unknown_name: str = "unknown",
):
    """Dispatch the configured weighting to its canonical corpus loader."""

    from .domain.work_weighting import WORK_BALANCED, resolve_training_weighting

    root = (
        pathlib.Path(frags_root)
        if frags_root is not None
        else resolve_fragment_roots(cfg).train_root
    )
    if resolve_training_weighting(training_weighting) == WORK_BALANCED:
        return load_work_balanced_dataset(
            root,
            cfg=cfg,
            exclude_authors=exclude_authors,
            unknown_name=unknown_name,
        )
    return load_dataset(
        root,
        exclude_authors=exclude_authors,
        unknown_name=unknown_name,
    )


__all__ = ["resolve_dataset", "resolve_fragment_roots"]
