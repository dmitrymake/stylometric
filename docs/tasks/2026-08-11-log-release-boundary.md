# Tracked log release-boundary D6

## Metadata

- Task ID: `2026-08-11-log-release-boundary`
- Status: done
- Owner: repository owner
- Created: 2026-08-11
- Baseline commit: `29cbc966b9f8eade52d36bf00822cfc5c18a7cb3`
- Target branch/worktree: `main`; selected fix in one worktree from the committed task baseline
- Type: pruning
- Size: M
- Risk: R2
- Risk flags: mutation, external-contract
- Primary domain: reporting/publication
- Allowed cross-domain: research-governance release metadata and tests only
- Standard version: 1.3

## 1. Goal and decision

- Desired observable result: tracked historical/local `log/` runners remain recoverable in Git but
  are absent from the Git-free public source archive, whose executable surface stays explicit.
- Consumer: maintainers, CI source-archive job and downstream archive readers.
- Owner-selected decision: D6 export exclusion; no move, deletion, inventory framework or runner redesign.
- Success is not merely: hiding a failing release dependency or deleting scientific provenance.

## 2. Confirmed starting facts

| ID | Fact | Evidence | Confidence |
|---|---|---|---|
| F-01 | Git tracks 22 `log/` files: 21 Python runners plus one historical shell recompute script. | `git ls-files log` | high |
| F-02 | Those files total 215,791 bytes and all enter `git archive` except the already excluded shell script. | `wc -c`; archive listing | high |
| F-03 | The reviewed executable inventory contains 298 Python paths under `scripts`, `src/stylo` and `tests`; `log/` is not a release root. | `release/executable_sources.json`; inventory check | high |
| F-04 | The archive contains 327 Python files: the 298 inventoried paths, 21 `log/` runners and eight immutable evidence-source snapshots. | archive/inventory listings | high |
| F-05 | Adding physical `log/` as an inventory root would also discover 129 ignored local Python files and leak local path classification into the contract. | `rg --files -uu`; ignored-path census | high |
| F-06 | Scientific artifacts cite several tracked `log/` producers, so Git deletion or history rewriting is unsupported. | bounded tracked grep | high |

## 3. Scope

### In scope

- Replace the one-file `log/experiments/requote_recompute.sh` archive exclusion with one scoped
  recursive export exclusion for all `log/**` content.
- Update the existing release-integrity assertion and `.gitattributes` SHA binding.
- Prove both checkout and Git-free archive behavior; record task result and concise handoff.

### Out of scope

- Deleting, moving, editing, executing or classifying any tracked/ignored `log/` content.
- D3-D5, remaining `scripts/` census, large-module refactor or inventory schema redesign.
- Scientific artifacts/claims, corpus/data, evaluator, site, deploy, publication or external writes.
- Formatter/linter/dependency/framework/state/public-entrypoint changes or another cleanup wave.

### Stop / reclassification conditions

- Any archive test or supported Git-free consumer requires a `log/` member.
- Export exclusion removes a non-`log/` path or changes tracked `log/` bytes/history.
- A framework/schema/new checker or positive production code LOC becomes necessary.
- Scientific/publication data, provenance output or execution state would change.

## 4. Frozen acceptance and tripwires

- Max touched production subsystems: one source-archive boundary.
- Allowed paths: `.gitattributes`, `tests/test_release_integrity.py`,
  `release/executable_sources.json`, this task and `docs/handoff/CURRENT.md`.
- New production files / production code LOC: 0 / 0; tracked runners deleted/moved: 0.
- Test LOC added/net budget: 2 / non-positive.
- Task/handoff docs added budget: 190 lines; metadata is not simplification progress.
- New dependency/framework/layer/state/schema/generic abstraction/public entrypoint: forbidden.
- Context target: archive `log/` members 22 -> 0; archive Python files 327 -> 306; archive bytes
  at least 215,791 lower; checkout/Git paths and 298-path inventory unchanged.
- Review budget: one clean-context deletion review and at most one bounded correction.
- Tripwire crossing requires stop/reclassification.

## 5. Domain and context contract

- Producer/consumer: Git history owns historical runner recoverability; `.gitattributes` owns archive
  membership; CI materializes the archive; inventory and provenance gates consume its reviewed roots.
- Grain/key: one repository-relative archive member with exact POSIX path; one tracked path maps to
  zero or one archive member.
- Temporal/delete semantics: current and historical Git commits keep runner bytes; only future
  archive materializations omit them. No artifact tombstone, data row, late arrival or NULL semantics.
- Cross-domain rule: scientific artifact producer references remain valid for a Git checkout and
  must not be rewritten into claims about Git-free execution.
- Contract verification: exact Git tree hashes, archive listing, archive full pytest, inventory,
  provenance and hygiene checks.

Constant measurement: tracked file bytes and `git archive --format=tar HEAD` member listing.

| Representative scenario | Baseline | Target |
|---|---:|---|
| Public archive `log/` inspection | 22 tracked files; 215,791 B | 0 members / 0 B |
| Archive Python exposure | 327 Python members | 306; 298 inventory + 8 evidence snapshots |
| Git scientific reproduction | 22 tracked `log/` paths; exact tree blobs | unchanged |
| Checkout active source inventory | 298 reviewed Python paths | unchanged |

- Domain transitions: Git tree -> archive -> release gates remains two.
- Central-config-only reads: `.gitattributes` and executable inventory SHA binding.
- Blurred boundary removed: ignored/local research workspace and public source archive share `log/`.
- Rejected/excluded artifacts: all runner contents, ignored caches/backups, D3-D5, previous cleanup
  branches/diffs and any inventory-framework alternative.

## 6. Invariants and acceptance

- [x] AC-01 `git archive HEAD` contains no path under `log/` and no unrelated member disappears.
- [x] AC-02 All 22 baseline `log/` files remain tracked and byte-identical in the checkout/Git tree.
- [x] AC-03 Executable inventory remains exactly 298 paths; evidence snapshots remain present.
- [x] AC-04 Existing checkout and Git-free archive tests, provenance and hygiene gates pass.
- [x] AC-05 No production/scientific/site/publication bytes or behavior changes.
- [x] AC-06 Diff stays inside allowed paths with zero production/test LOC growth.
- [x] AC-07 No deploy, registry/freeze/preflight/authorization/execution, external write or push occurs.

## 7. Phases, evidence and review

- Fresh read-only snapshot and owner selection D6 completed on 2026-08-11.
- Commit this contract, implement in one isolated worktree, and integrate only an accepted commit.
- Falsifier: materialize a clean archive and run its complete suite; any required `log/` dependency
  blocks the fix rather than causing a compensating copy/wrapper.
- Clean-context deletion review is limited to AC/invariants, unsupported archive consumer,
  regression, safety or tripwire. Future packaging architecture is non-blocking follow-up.
- After review rerun frozen verification; at most one correction pass; no next wave.
- Rollback: do not integrate a failed commit; after integration the single commit is revertible.

## 8. Verification plan

- Targeted: release integrity, inventory and governance tests.
- Runtime: compare baseline/result Git-tree blob IDs and archive member sets; run full pytest from
  checkout and materialized Git-free archive.
- Gates: physical-file `py_compile`, provenance, checkout/archive hygiene and inventory.
- Negative: no `log/` archive members; no non-`log/` archive member loss; all tracked `log/` hashes equal.
- Cannot-run results are recorded separately and never inferred as pass.

## 9. Result and DoD

- Commits: task baseline `a3b59cca`; implementation `73f31e3a`; this task/handoff update is a
  metadata-only closeout.
- `.gitattributes` now excludes the directory-level `/log` from `git archive`; the prior redundant
  one-file exception is gone. The release-integrity assertion and `.gitattributes` SHA binding were
  updated with no code/test LOC growth.
- Archive delta: `log/` tar members 23 -> 0, Python members 327 -> 306 and content at least 215,791
  bytes smaller. The non-`log/` archive member set is identical; the 298-path executable inventory
  and eight evidence-source snapshots remain.
- All 22 baseline `log/` files remain tracked and byte-identical; no scientific producer, artifact,
  claim or Git history was removed or rewritten.
- Clean detached deletion review at candidate `6c280481` (patch-equivalent to integrated
  `73f31e3a`): PASS; no unsupported archive consumer, regression or tripwire; zero correction passes.
- PASS: focused release/inventory/governance tests; exact Git-free archive full pytest and gates;
  checkout full pytest; physical-file `py_compile`; provenance (93 source + one output digest);
  checkout/archive hygiene and inventory. Checkout retained one expected real-bundle skip and two
  pre-existing invalid-escape warnings; archive-only skips reflected absent Git/local data.
- Invalid environment/procedure attempts were not counted as passes: `/log/**` left an empty tar
  directory before candidate commit; one archive gate ran from the checkout; one checkout full run
  exhausted `/tmp`; another used a missing basetemp parent. Each falsifier was resolved within scope,
  and the exact final commands passed.
- Separately authorized local disk hygiene removed rebuildable caches/temp environments after D6
  evidence capture; it changed no tracked file and is not counted as D6 simplification progress.
- New production files/code/concepts/framework/state/dependencies/public entrypoints: none.
- No deploy, publication, corpus/scientific execution, registry/freeze/preflight/authorization,
  external write or push occurred.
- ADR/domain/runbook: none; archive membership was reduced without new architecture.
- Campaign stop: selected D6 accepted; D3-D5 remain unselected and no next wave starts.
- [x] Applicable `DOD-01` through `DOD-15`; `DOD-06` is N/A because no R3 action occurred.
