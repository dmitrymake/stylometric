# Ignored local duplicate pruning

## Metadata
- Status: active
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2 (flags: irreversible)
- Primary subsystem/domain: local research storage / Allowed cross-domain: none
- Standard version: 1.3

## Frozen behavior
- Observable behavior to preserve: every tracked path, every sha-bound artifact, every gate result.
- Explicit non-goals: touching `data/`, `input*`, `_staging_corpora/`, `.venv`, `site/node_modules`,
  tracked `log/*.py`, `log/autographs/`, any `.tar.zst` archive, `research/local/wip-checkpoint-*`,
  `research/local/*insurance*.tar`.
- Rejected/excluded artifacts: `paired-audit-control-plane.bundle` (`d4d6eed4…`, sole carrier of
  `repair_commit 07a8df82`), `research/local/ruaa-r1-v5-exploratory-result-bundle-v1.tar.zst`
  (`e37dfeaf…`, ledger-bound).

## Scope
- In-scope paths (all git-ignored, none tracked):
  - `research/local/ruaa-r1-v5-verify-a-3c17766c/` and `…-verify-b-3c17766c/` — full repository
    clones at commit `3c17766c`, which is an ancestor of `main`;
  - `research/local/ruaa-r1-replay-bundle-v1/` — unpacked form of the retained `.tar.zst`
    (2 748 members; `MANIFEST.json` sha `784904874e…` identical in both);
  - `research/local/ruaa-r1-v5-exploratory-result-bundle-v1/` — unpacked form of the ledger-bound archive;
  - `research/local/lobo-final-914034e8-20260731T104531Z/bundle/` — third copy of the same replay bundle;
  - `research/local/lobo-final-914034e8-20260731T104531Z/tools/python/` — vendored CPython, absent
    from that directory's `MANIFEST.json` (59 entries, zero under `tools/python`);
  - `paired-audit-control-plane-c24b79b3.bundle` (`ec891cb6…`, referenced nowhere);
  - empty `scripts/meta/`, `scripts/statistic/`;
  - regenerable caches and corpus copies under `log/experiments/`: `_crossgenre_norm_cache/`,
    `broken_backup/`, `*.npy`, `*.pkl`, `*.txt`;
  - `__pycache__` trees outside `.venv`.

## Reduction target
- Production / test / docs LOC delta: 0 (no tracked file is touched)
- Disk target: ≥ 1 000 MB
- New files/framework/state/dependency/public entry point: forbidden
- Maximum mutation passes: 2

## Pass 1 — deletion
- Recorded below in Result.

## Adversarial review
- Blockers limited to: a deleted path proving tracked, a keep-list hash changing, a gate regressing.

## Independent deletion review
- Confirms keep-list hashes and `git status --porcelain` equality before/after.

## Verification

```
git status --porcelain                      # identical before and after
sha256sum paired-audit-control-plane.bundle research/local/ruaa-r1-v5-exploratory-result-bundle-v1.tar.zst
.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD
.venv/bin/python scripts/check_executable_source_inventory.py
```

## Result
- Status: done. Working tree 6.3 GB → 5.2 GB; `research/local/` 1.1 GB → 58 MB.
- Removed: both `ruaa-r1-v5-verify-*` repository clones (649 MB), `ruaa-r1-replay-bundle-v1/` and
  `lobo-final-*/bundle/` (266 MB of identical unpacked bundle), `lobo-final-*/tools/python/`
  (103 MB vendored CPython, absent from that manifest's 59 entries),
  `ruaa-r1-v5-exploratory-result-bundle-v1/` (14 MB unpacked form),
  `paired-audit-control-plane-c24b79b3.bundle` (`ec891cb6…`, 2 MB), empty `scripts/meta/` and
  `scripts/statistic/`, regenerable `log/experiments/` caches and corpus copies (≈39 MB),
  and all `__pycache__` trees outside `.venv`.
- Keep-list re-verified after deletion: `paired-audit-control-plane.bundle` `d4d6eed4…`,
  `ruaa-r1-v5-exploratory-result-bundle-v1.tar.zst` `e37dfeaf…`, both unchanged.
- `git status --porcelain` line count identical before and after; zero tracked paths touched.
- Production/test/docs delta: 0. Residual risk: the two repository clones are not recoverable; both
  sat at commit `3c17766c`, which is an ancestor of `main`, and their verification outputs remain in
  `research/local/ruaa-r1-v5-verification-3c17766c-20260802/`.

## DoD references
- [ ] Applicable DOD-01..DOD-12, DOD-13
