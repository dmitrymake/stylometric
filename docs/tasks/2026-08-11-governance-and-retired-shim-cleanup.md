# Governance state and retired-shim cleanup

## Metadata

- Task ID: `2026-08-11-governance-and-retired-shim-cleanup`
- Status: active
- Owner: repository owner
- Created: 2026-08-11
- Baseline commit: `35da4301172d220c1dd7940227b460e0634f1734`
- Target branch/worktree: `main`; fixes in separate worktrees from the task baseline
- Type: pruning
- Size: M
- Risk: R2
- Risk flags: mutation, external-contract
- Primary domain: research-governance
- Allowed cross-domain: evaluation/paired-audit metadata and retired operational entrypoints;
  reporting/publication release inventory only
- Standard version: 1.3

## 1. Goal and decision

- Desired observable result: current v3.2 status has one authority and cannot be confused with the
  immutable v3.1 15-member contract; twelve hard-disabled legacy script shims and their dead tests
  are removed while historical source hashes remain recoverable.
- Consumer: repository owner and the next separately authorized v3.2 RunPlan task.
- Decision/process unlocked: a smaller, contradiction-free context packet for supported work.
- Success is not merely: moving prose, renaming files, weakening safety gates, or deleting evidence.

## 2. Confirmed starting facts

| ID | Fact | Evidence locator | Confidence |
|---|---|---|---|
| F-01 | Ledger records implemented corrected v3.2 candidate with no second-review PASS. | `research/governance/status_ledger.json` | high |
| F-02 | Protocol still says pending implementation while defining 16/11. | `paired_audit_protocol.md:9,101` | high |
| F-03 | Requirements presents historical v3.1 Holm-15 without a versioned ID and lacks a v3.2 applicability binding. | `requirements.json:147` | high |
| F-04 | Runner catalog contains mutable `pending_review` preparation status. | `runner_catalog.json:95` | high |
| F-05 | Twelve tracked scripts are `retired_hard_disabled_shim`; historical pre-retirement hashes are preserved in the p0 snapshot. | `topology.json`; `docs/p0_baseline_snapshot.json` | high |
| F-06 | Current runtime identity explicitly excludes kernel/OS. | `run_plan.py:35-39` | high |

## 3. Unknowns and assumptions

| ID | Unknown/assumption | Why it matters | Resolution | Blocking? |
|---|---|---|---|---|
| U-01 | A downstream user may invoke a deleted shim for its explicit error. | Deletion changes unsupported invocation from a tailored error to file-not-found. | Accepted by owner selection A+B; canonical replacements remain. | no |
| U-02 | Historical source evidence could be mistaken for live source. | Deletion must not erase audit history. | Keep p0 hashes and Git history unchanged. | no |

## 4. Scope

### In scope

- A: remove mutable milestone prose outside the ledger; version the v3.1 Holm requirement and bind
  the exact v3.2 applicability contract to code/tests.
- B: delete the twelve named retired shims; remove only tests and topology entries whose contract
  disappears; refresh executable-source inventory.
- Task result and concise handoff update.

### Out of scope

- Evaluator, applicability, evidence receipt, model math, bundle/fold/corpus bytes.
- v3.2 RunPlan, registry activation, freeze, preflight, authorization, execution, publication.
- Historical p0/evidence deletion; `log/` census/deletion; pnpm cleanup; POSIX verifier redesign.
- Legacy v3.1 control-plane pruning beyond the twelve selected shims.

### Stop / reclassification conditions

- Any supported consumer imports a selected shim rather than invoking it only as a retired path.
- Any change to scientific estimand/matrix, production/publication state, evidence bytes, or public CLI.
- New dependency/framework/state, positive production LOC, or more than the selected two fixes.

## 5. Frozen acceptance and complexity tripwires

- Max touched production subsystems: one retired compatibility-script surface.
- Allowed production paths/files: deletion only of the twelve `retired_hard_disabled_shim` paths
  recorded at baseline in `research/governance/topology.json`.
- New production files / production LOC added: 0 / 0.
- Production LOC net target/limit: at most `-193` lines; no canonical implementation deletion.
- Test LOC added budget: 20; cumulative test LOC must be negative.
- Docs/governance LOC added budget: 260 for the required task contract and correctness mapping;
  this positive correctness overhead is not counted as simplification progress.
- New framework/layer/persistent state/dependency/generic abstraction/public entrypoint: forbidden.
- Representative context target: status authority 4 sources to 1; operational entrypoints minus 12;
  no regression in domain transitions or kernel/runtime packet.
- Review correction budget: one adversarial/deletion review and one bounded correction pass.
- Stop/reclassification trigger: any tripwire crossing or unsupported deletion counterexample.

## 6. Domain contract

- Primary domain: research-governance.
- Producers / consumers: ledger owns mutable status; protocol owns v3.2 scientific design;
  requirements maps versioned invariants to code/tests; topology maps live entrypoints.
- Allowed external domains: evaluation/paired-audit metadata; release inventory as consumer.
- Forbidden domains/areas: corpus/data content, feature math, site/publication claims, execution gates.
- Grain: one status item, requirement, runner, or operational path.
- Keys/cardinality: unique requirement ID, runner path, topology path; one mutable status authority.
- Temporal semantics: v3.1 contracts are historical; v3.2 design is current; ledger alone is mutable.
- NULL/delete/late-arrival: deleted shims have no tombstone source file; history remains in Git/p0.
- Contract verification: governance nodeid collection, topology uniqueness/discovery, inventory/release.

## 6.1. Context locality

Constant method: full required files for each representative scenario; physical UTF-8 bytes and
lines; token estimate `bytes / 4`.

| Scenario | Entry point | Baseline packet | Domain transitions | Sources/entrypoints | Target |
|---|---|---:|---:|---:|---|
| Next v3.2 control-plane orientation | ledger/task/protocol | 22 files; 464,642 B; 8,821 lines | 2 | 4 competing status reads | ledger-only mutable status |
| Runtime/kernel identity question | `run_plan.py` allowlist | 4 files; 7,061 B; 135 relevant lines | 1 | 1 runtime entrypoint | unchanged; kernel remains excluded |
| Retired entrypoint discovery | topology + shims/tests | 19 files; 238,711 B; 5,475 lines | 2 | 12 retired entrypoints | remove all 12 |
| Public claim change | ledger -> generator -> site | 8 files; 201,575 B; 2,812 lines | 2 | protected publication chain | unchanged |

- Central-config-only reads: requirements, runner catalog, topology, executable inventory.
- Blurred boundaries: mutable v3.2 milestone repeated in protocol/catalog; historical Holm-15 has an
  unversioned requirement ID beside current v3.2 protocol.
- Duplicate/conflicting sources: ledger, protocol status paragraph, runner status, requirements ID.
- Rejected/excluded artifacts: historical kernel snapshot, p0 hashes, `log/`, pnpm, prepared bundle.

## 7. Invariants

- INV-01: exact current applicability remains 25 statuses / 16 applied / 11 Holm members.
- INV-02: historical v3.1 Holm-15 remains executable historical regression, never current v3.2.
- INV-03: ledger remains sole mutable authorization/status authority.
- INV-04: p0 source hashes and Git history remain unchanged and release-recoverable.
- INV-05: all canonical CLI/module paths, production registry, hard-disable gates and publication
  state remain unchanged.

## 8. Phases and gates

- Phase A research completed read-only; four packets measured; owner selected A+B on 2026-08-11.
- Phase B: create one branch/worktree per fix from this committed contract baseline; one commit each.
- Phase C: integrate only accepted commits, run combined deletion review in a clean context, then
  frozen verification; no automatic next wave.
- Allowed blockers: AC/invariant regression, supported consumer counterexample, safety gate, tripwire.
- Deletion review required: yes.

## 9. Evidence plan

| Claim | Source/reproduction | Negative check | Confidence |
|---|---|---|---|
| Mutable state consolidated | bounded grep across four governance sources | no `pending implementation/review` for v3.2 | high |
| Historical/current Holm separated | requirement bindings and collected nodeids | both v3.1 m15 and v3.2 16/11 tests collect | high |
| Shims have no supported consumer | tracked exact-path grep and topology | canonical entrypoints still resolve | high |
| Release set is closed | inventory/release/archive checks | Git-free archive check | high |

## 10. Acceptance criteria

- [ ] AC-01 Ledger is the only mutable v3.2 status authority; protocol/catalog contain stable scope.
- [ ] AC-02 Requirements explicitly distinguish historical v3.1 Holm-15 and current v3.2 16/11.
- [ ] AC-03 All twelve selected shim files and their live topology entries are absent.
- [ ] AC-04 Historical p0 hashes remain unchanged and validated.
- [ ] AC-05 Canonical entrypoints and relevant governance/release regressions pass.
- [ ] AC-06 Production LOC is at least 193 lines lower with no new production concepts/files.
- [ ] AC-07 Evaluator/freeze/preflight/authorization/execution/publication state is unchanged.

## 11. Verification plan

- Targeted: governance, retired-path, inventory, release-hygiene tests.
- Regression: full `pytest tests`; `py_compile`; provenance; release and Git-free archive hygiene.
- Negative: grep forbidden mutable status; invoke/check canonical replacements; verify deleted paths
  absent and historical hashes present.
- Cannot-run: record separately; no pass inference from static checks.

## 12. Documentation impact

- ADR/domain/runbook: none; no durable architecture/scientific semantics change.
- Handoff: update only for active/closed cleanup routing.

## 13. Result

- Pending implementation, review, verification and cumulative metrics.

## 14. DoD references

- [ ] DOD-01 through DOD-05
- [ ] DOD-07, DOD-08, DOD-10 through DOD-15
