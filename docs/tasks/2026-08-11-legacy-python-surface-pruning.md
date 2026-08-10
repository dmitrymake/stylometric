# Legacy Python surface pruning D1+D2

## Metadata

- Task ID: `2026-08-11-legacy-python-surface-pruning`
- Status: active
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

## 2. Confirmed starting facts

| ID | Fact | Evidence | Confidence |
|---|---|---|---|
| F-01 | D1 is an internally closed 10-Python-file + one-JSON island; no tracked consumer outside the island or p0 snapshot exists. | exact-path/import grep; package entrypoints | high |
| F-02 | D1's five report outputs are absent from tracked `docs/`; active Stylo feature/NLP code lives under `src/stylo/`. | `git ls-files`; source tree | high |
| F-03 | D2 always fails closed before a write and names `apply_ci_sign_erratum.py` as its replacement. | `log/correct_macrof1_convention.py:88-96` | high |
| F-04 | D2 has no runtime consumer; one test docstring mentions the historical path. | tracked grep | high |
| F-05 | The executable inventory covers D1 through the `scripts` root but does not cover D2 under tracked `log/`. | inventory check and archive listing | high |

## 3. Scope

### In scope

- D1: delete `scripts/meta/`, `scripts/statistic/`, `scripts/nlp.py` and `scripts/utils.py`.
- D2: delete `log/correct_macrof1_convention.py` and remove its stale test-docstring mention.
- Refresh executable-source inventory after D1; record task result and concise handoff.

### Out of scope

- Active `src/stylo` feature/NLP/model code or behavior.
- Historical p0 snapshot, scientific result/evidence artifacts, withdrawn Fano artifacts, or other
  tracked/ignored `log/` files.
- Release-wide `log/` classification, D3-D6, corpus/data, paired audit, site or publication.
- Dependencies, framework/state/abstraction, public CLI, external writes, push or next cleanup wave.

### Stop / reclassification conditions

- A supported consumer or tracked output contract for D1/D2 is found.
- A historical artifact must be removed or changed to make tests pass.
- Any active feature/scientific math, public output, dependency or publication behavior changes.
- Cumulative production Python deletion is less than 1,098 LOC or positive LOC is required.

## 4. Frozen acceptance and tripwires

- Max touched production subsystems: two legacy entrypoint islands.
- Allowed production paths: deletion only of the D1 paths and D2 file listed above.
- Allowed support paths: `tests/test_frozen_writer_protection.py`,
  `release/executable_sources.json`, this task and `docs/handoff/CURRENT.md`.
- New production files / production LOC added: 0 / 0.
- Production Python LOC net target: at most `-1,098` from the original baseline.
- Test LOC added/net budget: 1 / non-positive.
- Task/handoff docs added budget: 230 lines; metadata does not count as simplification progress.
- New framework/layer/state/dependency/generic abstraction/public entrypoint: forbidden.
- Context target: 11-file legacy statistic island -> 0; erratum packet 5 -> 4 files;
  executable inventory 308 -> 298 Python paths.
- Review budget: one deletion review and at most one bounded correction pass.
- Tripwire crossing requires stop/reclassification, never silent expansion.

## 5. Domain and context contract

- D1 producer/consumer: obsolete report scripts consume their private legacy metadata/NLP helpers;
  there is no downstream tracked artifact. Delete semantics are whole-island removal; Git/p0 retain
  history. Active `src/stylo` interfaces are separate and unchanged.
- D2 producer/consumer: the superseded writer consumes historical inputs but fails before output;
  `scripts/apply_ci_sign_erratum.py` plus `src/stylo/eval/ci_erratum.py` own the active safe path.
- Grain: one operational Python path or one private helper/resource.
- Temporal semantics: historical sources remain reconstructable from Git and p0; no late-arrival,
  NULL, data-row or corpus semantics are involved.
- Contract checks: tracked imports/references, package entrypoints, frozen-writer regression,
  inventory/release/archive checks and full pytest.

Constant measurement: complete selected files, physical bytes/lines, token estimate `bytes / 4`.

| Representative scenario | Baseline packet | Domain transitions | Target |
|---|---:|---:|---|
| Legacy statistic report maintenance | 11 files; 41,365 B; 998 Python LOC; 5 report entrypoints | 1 | remove packet and all 5 entrypoints |
| Feature/NLP source discovery | 28 files; 182,908 B; 17 active + 11 legacy files | 1 | 17 files; 141,543 B; no legacy source |
| Macro-F1 erratum orientation | 5 files; 25,035 B; 521 lines; 2 operational paths | 2 | 4 files; 19,539 B; 421 lines; 1 path |
| v3.2 evaluator maintenance | 10 files; 221,369 B; 4,470 lines | 2 | unchanged |

- Central-config-only reads: executable inventory only.
- Duplicate/conflicting sources: old `scripts/meta`/`scripts/nlp.py` beside active package code;
  hard-disabled correction path beside its named replacement.
- Rejected/excluded artifacts: p0 hashes, other `log/`, scientific evidence, local caches/backups,
  D3-D6 and all previous rejected cleanup branches/diffs.

## 6. Invariants and acceptance

- [ ] AC-01 All 11 D1 files and D2 are absent; no new replacement/wrapper exists.
- [ ] AC-02 No supported import, CLI, tracked-output producer contract or package entrypoint is lost.
- [ ] AC-03 Active feature/NLP behavior and active CI-sign erratum behavior are byte-unchanged.
- [ ] AC-04 Historical p0/scientific artifacts and frozen-writer protections remain unchanged and pass.
- [ ] AC-05 Inventory, focused tests, full regression, compile, provenance and release/archive hygiene pass.
- [ ] AC-06 Production Python delta is at most -1,098 LOC; inventory is exactly 298 paths.
- [ ] AC-07 No corpus, evaluator, registry, freeze, authorization, execution or publication state changes.

## 7. Phases, evidence and review

- Read-only audit and owner selection D1+D2 completed on 2026-08-11.
- Commit this contract as the common worktree baseline; implement D1/D2 separately.
- Falsifiers: repeat exact-path/import/output grep; verify active-path hashes; reject a deletion if a
  supported consumer or active output is found.
- Integrate accepted commits only, run a clean-context deletion review against this frozen contract,
  then rerun frozen verification. No second architecture review.
- Allowed review blockers: violated AC/invariant, reproducible regression, supported consumer,
  safety gate or tripwire. Style, hardening and future `log/` architecture are follow-up only.
- Rollback: do not integrate a failed fix; integrated commits remain individually revertible.

## 8. Verification plan

- Targeted: frozen-writer, medium governance, executable inventory and release integrity tests.
- Regression: full `pytest tests`; `py_compile`; provenance; checkout and Git-free archive hygiene.
- Negative: no selected paths/imports; no generated legacy statistic outputs; active-path hashes
  unchanged; historical snapshot/artifacts still tracked.
- Cannot-run results are recorded separately and never inferred as pass.

## 9. Documentation impact and result

- ADR/domain/runbook: none expected; capabilities and semantics are only reduced.
- Handoff: update at close only.
- Final commits, cumulative metrics, review, checks, residual risks and campaign stop: pending.

## 10. DoD references

- [ ] Applicable `DOD-01` through `DOD-15`; `DOD-06` is N/A unless risk changes.
