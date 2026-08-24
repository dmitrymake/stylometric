# Exploratory packet cache pruning

## Metadata
- Status: active
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2 (flags: irreversible)
- Primary subsystem/domain: exploratory evidence storage / Allowed cross-domain: none
- Standard version: 1.3

## Frozen behavior
- Observable behavior to preserve: ledger milestone identities, site provenance, every gate result.
- Explicit non-goals: `docs/exploratory/work_balanced/` (holds sha-bound
  `stylo_equal_channels_repair_smoke_v1.json` = `0d4a7832…`), `real_corpus/`, `corpora/`,
  the final-validation patch, anything tracked.

## Scope
- In: `docs/exploratory/lobo_vnext/packets/` (274 MB) and
  `docs/exploratory/lobo_vnext/pristine-final-archive-mMneey.source.tar` (9 MB).
- Both are git-ignored. `runner_catalog.json:134` declares the packet namespace
  create-if-absent and content-addressed, regenerable by
  `scripts/evaluation/prepare_stylo_lobo_vnext_packet.py`.

## Falsifiers executed before mutation
- `packet_generation_id d08f8cb7…` is recorded outside `packets/` — in
  `research/local/ruaa-r1-v5-verification-3c17766c-20260802/logs/verification-summary.json`,
  `research/local/ruaa-r1-v5-exploratory-3c17766c-20260802-evidence/logs/{post-run,pre-fresh}-gate.json`
  and in `status_ledger.json` itself, so the identity survives the byte deletion.
- `site/src/generated/manifest.json` cites zero sources under `docs/exploratory/`.

## Reduction target
- Production / test / docs LOC delta: 0
- Disk target: ≥ 280 MB
- Maximum mutation passes: 2

## Verification

```
git status --porcelain                      # identical before and after
.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD
.venv/bin/python scripts/check_executable_source_inventory.py
node scripts/check-provenance.mjs
```

## Result
- Status: done. `docs/exploratory/` 309 MB → 26 MB; working tree 5.2 GB → 4.9 GB.
- Removed `docs/exploratory/lobo_vnext/packets/` (274 MB) and
  `pristine-final-archive-mMneey.source.tar` (9 MB). Both git-ignored.
- `stylo_equal_channels_repair_smoke_v1.json` re-hashed after deletion: `0d4a7832…`, unchanged.
- `check-provenance.mjs` verified 93 source digests and 1 output digest after the deletion.
- Production/test/docs delta: 0.

## DoD references
- [ ] Applicable DOD-01..DOD-12, DOD-13
