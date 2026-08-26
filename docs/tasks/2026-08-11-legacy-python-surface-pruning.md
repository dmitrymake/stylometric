# Legacy Python surface pruning D1+D2

## Metadata

- Task ID: `2026-08-11-legacy-python-surface-pruning`
- Status: done
- Owner: repository owner
- Created: 2026-08-11
- Baseline commit: `12b5ce74f3fce519040e30586e6bb3ef8ec833fa`
- Target branch/worktree: `main`; one worktree per selected fix from the committed task baseline
- Type: pruning
- Size: M
- Risk: R2
- Risk flags: mutation
- Primary domain: feature-extraction
- Allowed cross-domain: reporting/publication historical writer safety; research-governance release metadata only
- Standard version: 1.3

## 1. Goal and decision

- Desired observable result: remove the owner-selected D1 legacy statistic island and D2
  superseded macro-F1 correction entrypoint while preserving current feature, erratum, scientific,
  release and publication behavior.
- Consumer: maintainers and the Git-free source archive.
- Decision unlocked: fewer false entrypoints and fewer obsolete implementations in code search.
- Success is not merely: moving files, hiding them from inventory, or deleting historical evidence.

## 3. Scope

## 8. Verification plan

- Targeted: frozen-writer, medium governance, executable inventory and release integrity tests.
- Regression: full `pytest tests`; `py_compile`; provenance; checkout and Git-free archive hygiene.
- Negative: no selected paths/imports; no generated legacy statistic outputs; active-path hashes
  unchanged; historical snapshot/artifacts still tracked.
- Cannot-run results are recorded separately and never inferred as pass.
