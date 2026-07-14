#!/usr/bin/env python3
"""Freeze the GKF screening-panel fold assignment -> docs/screening_panel_v1.json.

Deterministic: the panel (authors with >= 2 works) and the per-author rotating round-robin fold
assignment are pure functions of the disk-anchored corpus. Loads the corpus exactly as the sweep
does (``resolve_dataset`` under the legacy weighting) so the manifest's parent digest matches what
``eval/groupkfold.py`` verifies at run time. Run from the repo root.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylo.config import load_config  # noqa: E402
from stylo.eval.screening_panel import build_manifest, verify_manifest, MANIFEST_PATH  # noqa: E402
from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY  # noqa: E402
from stylo.jsonio import dumps_strict  # noqa: E402
from stylo.workdoc import resolve_dataset  # noqa: E402


def main() -> int:
    cfg = load_config()
    ds = resolve_dataset(
        cfg, CHUNK_WEIGHTED_LEGACY,
        ROOT / "data" / "frags_train",
        exclude_authors=set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []),
        unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"))
    manifest = build_manifest(ds)
    verify_manifest(manifest)
    out = ROOT / MANIFEST_PATH
    out.write_text(dumps_strict(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}: {manifest['n_authors']} authors / {manifest['n_works']} works, "
          f"folds={manifest['fold_sizes']}, self_hash={manifest['self_hash'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
