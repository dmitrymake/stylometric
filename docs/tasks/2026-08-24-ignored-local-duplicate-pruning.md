# Ignored local duplicate pruning

## Metadata
- Status: done
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2 (flags: irreversible)
- Primary subsystem/domain: local research storage / Allowed cross-domain: none
- Standard version: 1.3

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
