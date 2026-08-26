# Exploratory packet cache pruning

## Metadata
- Status: done
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2 (flags: irreversible)
- Primary subsystem/domain: exploratory evidence storage / Allowed cross-domain: none
- Standard version: 1.3

## Scope
- In: `docs/exploratory/lobo_vnext/packets/` (274 MB) and
  `docs/exploratory/lobo_vnext/pristine-final-archive-mMneey.source.tar` (9 MB).
- Both are git-ignored. `runner_catalog.json:134` declares the packet namespace
  create-if-absent and content-addressed, regenerable by
  `scripts/evaluation/prepare_stylo_lobo_vnext_packet.py`.

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
