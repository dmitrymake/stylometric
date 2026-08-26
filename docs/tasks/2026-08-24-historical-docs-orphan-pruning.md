# Historical docs orphan pruning

## Metadata
- Status: done
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2
- Primary subsystem/domain: reporting artifacts / Allowed cross-domain: none
- Standard version: 1.3

## Scope

In — 21 tracked artifacts of the closed June-2026 dossier phase. Each one satisfies all four
conditions, verified mechanically before mutation:

1. absent from `site/src/generated/manifest.json` sources;
2. present in `docs/p0_baseline_snapshot.json` `artifacts.sha256` and **byte-identical** to the
   recorded digest — the standing retirement protocol under `topology.historical_evidence`;
3. no reference in `tests/`, `scripts/gen-site-data.mjs`, `scripts/gen-paper.mjs`,
   `scripts/check-paper-numbers.mjs`, `site/src/`, `research/governance/`;
4. stem not cited in `PAPER.md`, `README.md`, `docs/honest_protocol_paper.md`,
   `docs/publication_readiness_audit.md`.

Clusters: `tomsk_*` (6), `sholokhov_*` (8), `audit_*` (3), plus `alexander3_check.json`,
`emperor_sources.json`, `instrument_validation.json`, `sweep_table.csv`.

Withdrawn after falsification — cited in prose, therefore retained: `docs/feature_audit.json`,
`docs/sholokhov_rigor.json`.

## Verification

```
node scripts/gen-site-data.mjs && node scripts/check-provenance.mjs
git diff --exit-code -- site/src/generated
.venv/bin/python scripts/check_executable_source_inventory.py
PYTHONPATH=src .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

## Result
- Status: done. 21 tracked artifacts removed (50 993 bytes); `docs/feature_audit.json` and
  `docs/sholokhov_rigor.json` withdrawn because prose cites them.
- All 21 were byte-identical to their `docs/p0_baseline_snapshot.json` digests at deletion time, so
  the retirement anchor stays valid and the snapshot itself was not edited.
- Site regeneration reproduced `site-data.json` and `manifest.json` byte-for-byte
  (`git diff --exit-code -- site/src/generated` clean); provenance verified 93 sources.
- Full pytest green.
