# Orphan runner pruning — wave B

## Metadata
- Status: done
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2
- Primary subsystem/domain: scripts surface / Allowed cross-domain: release inventory
- Standard version: 1.3

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

## Verification

Same block as wave A.

## Result
- Status: done. Eight files removed; `run_luar_proza.py` withdrawn from the candidate list because
  `scripts/gen-paper.mjs` consumes its output `docs/luar_proza.json`.
- Combined waves A+B: 16 files, 3 731 lines of production Python removed; inventory 303 → 287.
- `tests/test_alternative_case_centroids.py` still loads the retained
  `scripts/run_chekhonte_dubia_oskolki.py`; `docs/cases/chekhonte_15_micro.json` still has its
  generator. Full pytest green after removal.
