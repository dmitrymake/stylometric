# Orphan runner pruning — wave A

## Metadata
- Status: done
- Owner / Baseline commit: Dmitry Purtov / `ac617a5f`
- Type: pruning
- Size / Risk: M / R2
- Primary subsystem/domain: scripts surface / Allowed cross-domain: release inventory
- Standard version: 1.3

## Scope

In — eight `scripts/*.py` with zero references in `tests/`, `src/`, `site/`, `.github/`, `run.sh`,
`configs/`, `research/governance/`, `release/`, and no published output:

| File | LOC | Output |
|---|---:|---|
| `run_sociolit_tf.py` | 297 | none tracked as published |
| `fetch_sociolit_fulltext.py` | 268 | corpus fetch |
| `run_finetune_rubert_proza.py` | 169 | `docs/neuro_finetune_proza.json` (not published) |
| `run_vertex_embedding_proza.py` | 144 | `docs/vertex_embedding_proza.json` (not published) |
| `fetch_kolokol_ogaryov_ws.py` | 118 | corpus fetch |
| `run_petersburg_chronicle_NN.py` | 116 | co-owned by retained gate script |
| `fetch_kolokol_herzen_ws.py` | 106 | corpus fetch |
| `dl_case.py` | 51 | none |

Withdrawn from the original candidate list after falsification — these write **published** sources
and are retained: `run_consistency.py` (`docs/consistency.json`), `run_proza_compare.py`
(`docs/proza_compare.json`), `run_luar_proza.py` (`docs/luar_proza.json`, consumed by
`scripts/gen-paper.mjs`).

## Verification

```
.venv/bin/python scripts/check_executable_source_inventory.py
.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD
node scripts/gen-site-data.mjs && node scripts/check-provenance.mjs
git diff --exit-code -- site/src/generated
PYTHONPATH=src .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

## Result
- Status: done, executed together with wave B in one inventory cascade.
- Eight files removed; combined with wave B the tracked Python surface drops 303 → 287 and
  `release_python_paths_sha256` becomes `e1f0f1464472e2e140afbf3fb9d17784767fb00b087a723c5afe2cfbef644cb6`.
- Falsification before mutation found three candidates from the original list that write published
  sources; they were withdrawn and remain in the tree.
- Gates after the cascade: inventory OK, release hygiene OK, provenance 93/1 verified,
  `git diff --exit-code -- site/src/generated` clean, full pytest green.
