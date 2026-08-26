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

## 3. Scope

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
