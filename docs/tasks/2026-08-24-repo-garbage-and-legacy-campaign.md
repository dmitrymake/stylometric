# Repository garbage and legacy reduction campaign

## Metadata and immutable baseline

- Status: active
- Owner: Dmitry Purtov
- Created: 2026-08-24
- Baseline commit: `ac617a5f`; worktree clean at capture
- Type / Size / Risk: pruning campaign / L / R1 for the campaign file itself
- Standard version: 1.3
- Approval: owner turn 2026-08-24 — «сильно почистить проект от мусора, удалить весь легаси»;
  «автоматически прими предложенный план и начни исполнять»

## Ordering decision

The topic-validity fixed-8 run and this campaign cannot run concurrently.
`execution_source_sha256` (`src/stylo/eval/paired_audit/run_plan.py:398`) hashes every
`src/stylo/**/*.py` including untracked files, and fork workers read modules from disk for the full
26-hour window. Mutating the tree during the run either raises `ImportError` inside a worker —
which terminates the pool before any output — or produces an aggregate bound to a tree that no
longer exists. Cleanup therefore completes first; the run executes afterwards on final identities.

## Excluded artifacts and keep-list

These are not design or evidence inputs for this campaign and must not be reinstated without an
explicit owner decision: branches `rework/stylo-v4`, `archive/local-main-private-20260726`,
`release-backup-20260707`, `refs/stash`, `research/local/wip-checkpoint-*`.

Byte-verified keep-list — deletion forbidden:

| Path | sha256 | Held by |
|---|---|---|
| `paired-audit-control-plane.bundle` | `d4d6eed4…` | `research/evidence/stack_class_coverage_repair_smoke_v1/source_manifest.json`; sole carrier of `repair_commit 07a8df82` |
| `research/local/ruaa-r1-v5-exploratory-result-bundle-v1.tar.zst` | `e37dfeaf…` | `status_ledger.json` milestone `ruaa_r1_v5` |
| `docs/exploratory/work_balanced/stylo_equal_channels_repair_smoke_v1.json` | `0d4a7832…` | same source manifest |
| `scripts/experimental/compare_multilingual_authorship_embeddings.py` | — | keeps `scripts/experimental/` non-empty for `test_medium_governance.py:699` under `git archive` |
| `scripts/build_screening_panel.py` | — | `src/stylo/eval/groupkfold.py:155` names it as the only regeneration path |
| `scripts/gen-paper.mjs` | — | only generator of tracked `PAPER.md` |
| 22 tracked files under `log/` | — | generators of 58 published site sources |
| `refs/heads/release` | — | CI trigger branch |

## Bounded tasks

`PRUNE-11` forbids a single repo-wide mutation. Each task below is separate, starts from the shared
baseline, and carries its own acceptance.

| ID | Task file | Effect | Risk |
|---|---|---|---|
| T1 | `2026-08-24-merged-branch-ref-hygiene.md` | 8 merged branch refs removed, 0 objects | R1 |
| T2 | `2026-08-24-ignored-local-duplicate-pruning.md` | ≈1 086 MB ignored duplicates | R2 |
| T3 | `2026-08-24-exploratory-packet-cache-pruning.md` | ≈283 MB regenerable packet cache | R2 |
| T4 | `2026-08-24-orphan-runner-pruning-a.md` | 10 files / 1 489 LOC | R2 |
| T5 | `2026-08-24-orphan-runner-pruning-b.md` | 9 files / 2 565 LOC | R2 |
| T6 | `2026-08-24-historical-docs-orphan-pruning.md` | 23 tracked docs artifacts | R2 |
| T7 | `2026-08-24-unrelated-history-retirement.md` | ≈422 MB git history, copyright exposure | R3a |
| T8 | `2026-08-24-dead-eval-module-assessment.md` | assessment only, expected decision «keep» | research |

## Registry cascade for any tracked `.py` removal

1. remove the file(s);
2. recompute `release_python_file_count` and `release_python_paths_sha256` from
   `scripts/check_executable_source_inventory.py --show-paths`;
3. refresh `sha256_bindings` for every touched governance JSON;
4. `runner_catalog.json` when the path is under `scripts/evaluation/`;
5. `topology.json` when the path is a canonical path;
6. regenerate site data and require `git diff --exit-code -- site/src/generated`;
7. verification block V.

## Verification block V

```
.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD
.venv/bin/python scripts/check_executable_source_inventory.py
node scripts/gen-site-data.mjs && node scripts/check-provenance.mjs
git diff --exit-code -- site/src/generated
PYTHONPATH=src .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

## Stop conditions

Any governance gate that fails and cannot be fixed inside the current task's frozen scope; any
deletion touching the keep-list; any private-corpus path appearing in the index; owner refusal at
the T7 approval gate.

## Result

- Pending.
