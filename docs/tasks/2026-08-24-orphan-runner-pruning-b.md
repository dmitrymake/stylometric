# Orphan runner pruning — wave B

## Metadata
- Status: active
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2
- Primary subsystem/domain: scripts surface / Allowed cross-domain: release inventory
- Standard version: 1.3

## Frozen behavior
- Observable behavior to preserve: `docs/cases/chekhonte_15_micro.json` — the only published
  Chekhonte source — keeps its generator `scripts/run_chekhonte_15_micro.py`;
  `tests/test_alternative_case_centroids.py` keeps `scripts/run_chekhonte_dubia_oskolki.py`, which
  it loads by file path.
- Explicit non-goals: deleting case artifacts under `docs/cases/`, or the two retained scripts above.

## Scope

In — eight `scripts/*.py` of the closed Chekhonte/M2 campaigns, zero live references, no published
output:

| File | LOC |
|---|---:|
| `run_chekhonte_brother_confound.py` | 652 |
| `run_chekhonte_dubia.py` | 553 |
| `fetch_chekhonte_dubia.py` | 285 |
| `run_chekhonte_dubia_alloskolki.py` | 240 |
| `build_chekhonte_neighbor_controls.py` | 233 |
| `build_chekhonte_brother_panel.py` | 195 |
| `build_chekhonte_budilnik_controls.py` | 154 |
| `run_m2_delta_baseline.py` | 150 |

`run_m2_delta_baseline.py` mentions `docs/cases/nekrasov_panaeva.json` and
`docs/cases/sovremennik.json` only in comments; those published artifacts are owned by
`scripts/run_nekrasov_panaeva_gate.py` and `scripts/run_sovremennik_gate.py`, both retained.

## Accepted consequence

`research/CHEKHONTE_BUDILNIK_20_EVIDENCE.md` remains a description of a closed case without a live
regeneration path. This matches the existing `topology.historical_evidence` contract: pre-retirement
bytes stay recoverable from Git history.

## Reduction target
- Production LOC delta: −2 462; test/docs delta: 0
- Inventory: `release_python_file_count` 295 → 287 with recomputed digest
- Maximum mutation passes: 2

## Verification

Same block as wave A.

## Result
- Status: done. Eight files removed; `run_luar_proza.py` withdrawn from the candidate list because
  `scripts/gen-paper.mjs` consumes its output `docs/luar_proza.json`.
- Combined waves A+B: 16 files, 3 731 lines of production Python removed; inventory 303 → 287.
- `tests/test_alternative_case_centroids.py` still loads the retained
  `scripts/run_chekhonte_dubia_oskolki.py`; `docs/cases/chekhonte_15_micro.json` still has its
  generator. Full pytest green after removal.

## DoD references
- [ ] Applicable DOD-01..DOD-12, DOD-13
