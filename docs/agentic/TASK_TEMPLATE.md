# Task Templates

Проект: **Stylo**. Каноническое размещение задач: `docs/tasks/YYYY-MM-DD-<slug>.md`.

Шаблоны производны от `STANDARD.md`. Definition of Done задаётся только `DOD-*` в стандарте.

## A. Short contract — только S/R1

```markdown
# <Task title>

- Status: proposed | active | blocked | review | done | cancelled
- Owner: <human owner>
- Created: YYYY-MM-DD
- Baseline commit: <sha>
- Type: context-only | research | implementation | pruning | review | incident
- Size / Risk: S / R1
- Primary domain: <domain>
- Allowed cross-domain: none | <explicit list>

## Goal
<Observable result and consumer.>

## Scope
- In: ...
- Out: ...

## Frozen acceptance and tripwires
- Acceptance freezes before mutation: yes/no
- Max touched production subsystems:
- New production files: forbidden | allowed: <explicit>
- Production LOC added budget:
- Desired/required production net delta:
- Test/docs LOC added budget:
- New public entry point: forbidden | <explicit>
- New framework/state/dependency/generic abstraction: forbidden | <explicit>
- Representative context packet target: N/A | <files/sections/tokens/domain hops/entry points/sources>
- Review correction budget: 1 by default
- Stop/reclassification trigger:

## Acceptance
- [ ] <observable criterion>

## Verification
- Command/check:
- Negative scenario:

## Result
- Changes/evidence/residual risk/Git state:

## DoD
- [ ] Applicable `DOD-01..DOD-15` satisfied.
```

Short contract не применяется к R2/R3.

## B. Full contract — M/L или R2/R3

```markdown
# <Task title>

## Metadata
- Task ID: YYYY-MM-DD-<slug>
- Status: proposed | active | blocked | review | done | cancelled
- Owner:
- Created: YYYY-MM-DD
- Baseline commit:
- Target branch/worktree:
- Type: context-only | research | implementation | pruning | review | incident
- Size: S | M | L
- Risk: R1 | R2 | R3b | R3a
- Risk flags: mutation | production | money | security | pii | external-contract | load | irreversible
- Primary domain:
- Allowed cross-domain:
- Standard version: 1.3

## 1. Goal and decision
- Desired observable result:
- Consumer:
- Decision/process unlocked:
- Success is not merely:

## 2. Confirmed starting facts
| ID | Fact | Evidence locator | Confidence |
|---|---|---|---|
| F-01 | | | |

## 3. Unknowns and assumptions
| ID | Unknown/assumption | Why it matters | Resolution | Blocking? |
|---|---|---|---|---|

## 4. Scope
### In scope
- ...
### Out of scope
- ...
### Stop / reclassification conditions
- New production/publication mutation requirement.
- New PII/security/external-contract exposure.
- Unknown user WIP overlaps scope.
- Source/scientific contract contradicts premise.

## 5. Frozen acceptance and complexity tripwires
- Max touched production subsystems:
- Allowed production paths/files:
- New production files: forbidden | allowed: <explicit>
- Production LOC added budget:
- Production LOC net target/limit:
- Test LOC added budget:
- Docs LOC added budget:
- New framework/layer:
- New persistent/runtime state:
- New dependency:
- New generic abstraction:
- New public/operational entry point:
- Representative context packet target: N/A | <metric targets>
- Review correction budget: 1 by default
- Stop/reclassification trigger:

## 6. Domain contract
- Primary domain:
- Producers / consumers:
- Allowed external domains:
- Forbidden domains/areas:
- Grain:
- Keys/cardinality:
- Temporal semantics:
- NULL/delete/late-arrival semantics:
- Shared/conformed interfaces:
- Contract/reconciliation tests:

## 6.1. Context locality — for maintainability/pruning/simplification claims
| Representative scenario | Authoritative entry point | Required files/sections | Approx tokens/bytes | Domain transitions | Entry points | Authoritative sources | Target |
|---|---|---|---|---|---|---|---|
- Central-config-only reads:
- Blurred capability boundaries:
- Duplicate/conflicting sources:
- Rejected/excluded artifacts:
- Constant measurement method:

## 7. Invariants
- INV-01:

## 8. Phases and gates
### Phase A — context/research
- Questions/sources/coverage/falsifiers:
- For simplification audit: max findings/candidates and false positives:
- Human selection before mutation: yes/no
- Gate:
### Phase B — implementation
- Bounded changes:
- Migration/backfill:
- Rollback/abort:
### Phase C — bounded review/close
- Adversarial review brief:
- Allowed blockers: AC/invariant | regression | supported unsafe counterexample | safety gate
- Correction budget:
- Deletion review required: yes/no; mandatory for pruning
- Evidence audit sample:
- Close artifacts:
- After deletion review: rerun frozen verification; no new architecture loop.

## 9. Evidence plan
| Claim | Required source | Reproduction | Negative check | Minimum confidence |
|---|---|---|---|---|

## 10. Acceptance criteria
- [ ] AC-01 ...

## 11. Verification plan
### Targeted
- ...
### Regression
- ...
### Runtime/data
- ...
### Negative scenarios
- ...
### Cannot-run handling
- ...

## 12. R3 safety and approval
- Access path/role:
- Bounded query/load plan:
- Data minimization:
- Mutation plan:
- Rollback/abort:
- Human approval immediately before:
- Approval record:

## 13. Documentation impact
- ADR:
- Domain docs:
- Runbook:
- Monitoring:
- Handoff/state packet:

## 14. Review contract
- Frozen criteria, non-goals and invariants:
- Evidence and negative scenarios:
- Regression vs original baseline:
- Domain leakage and scope/risk/tripwires:
- Code/docs/runtime alignment:
- Production/publication and rollback gates:
- Context packet/source regression when declared:

## 15. Result
- Final commit(s):
- Changes/evidence/verification:
- Deviations/waivers/residual risks:
- Production status: not-tested | blocked | ready-for-approval | activated
- Final Git status:
- Touched production subsystems/files:
- Production LOC added/deleted/net vs baseline:
- Test/docs LOC added/deleted/net:
- New concepts/framework/state/dependencies:
- Review passes used/budget:
- Context packet before/after:
- Domain transitions/entry points/sources before/after:
- Campaign stop reason:

## 16. Metrics — L/pruning/simplification
- Unreproducible significant claims:
- Late requirements/reclassifications:
- Rules moved to automated gates:
- Defects found by independent review:
- Correction passes:
- Production LOC and new files/concepts:
- Deletion-review removals:
- Context packet metrics before/after:
- Domain transitions/entry points/sources before/after:
- Candidate/selected/accepted/blocked fixes and stop reason:

## 17. DoD references
- [ ] DOD-01
- [ ] DOD-02
- [ ] DOD-03
- [ ] DOD-04
- [ ] DOD-05
- [ ] DOD-06, if applicable
- [ ] DOD-07
- [ ] DOD-08
- [ ] DOD-09, if applicable
- [ ] DOD-10
- [ ] DOD-11
- [ ] DOD-12, implementation/pruning
- [ ] DOD-13, pruning
- [ ] DOD-14, maintainability claim
- [ ] DOD-15, campaign
```

## C. Pruning contract — bounded codebase reduction

Это task artifact, не новый permanent process file. Нормативные правила — `PRUNE-*`, `CTX-*` и `REV-*` стандарта.

```markdown
# <Pruning title>

## Metadata
- Status: proposed | active | blocked | review | done | partially-done | cancelled
- Owner / Baseline commit:
- Type: pruning
- Size / Risk:
- Primary subsystem/domain / Allowed cross-domain:
- Standard version: 1.3

## Frozen behavior
- Observable behavior to preserve:
- Frozen acceptance IDs:
- Explicit non-goals:
- Supported runtime paths:
- Rejected/excluded artifacts:

## Scope
- In-scope paths:
- Explicitly out-of-scope:
- Pre-existing bugs become follow-up unless frozen behavior requires them.

## Representative context packets
| Scenario | Entry point | Required files/sections | Tokens/bytes | Domain transitions | Entry points | Sources | Target |
|---|---|---|---|---|---|---|---|
- Measurement method:
- Central-config-only reads / blurred boundaries:

## Reduction target
- Production / test / docs LOC at baseline:
- Production net target:
- Test/docs delta target:
- Context packet reduction target:
- Domain transitions / entry points / sources target:
- Files/concepts/duplication to remove:
- Maximum production subsystems:
- New production files/framework/state/dependency/public entry point: forbidden by default
- Generic abstraction: forbidden unless `CTX-05` and reclassification allow it
- Context growth: forbidden
- Maximum mutation passes: 2

## Candidate inventory
| Candidate | Redundancy evidence | Removal risk | Expected context/LOC effect | Decision |
|---|---|---|---|---|

## Pass 1 — deletion/simplification
- Changes and cumulative deltas from original baseline:
- Targeted/regression/negative verification:

## Optional Pass 2 — bounded correction
- Introduced regression or missed in-scope simplification:
- Correction / cumulative delta / verification:

## Adversarial review
- Map blockers only to frozen AC/invariant, introduced regression, unsupported deletion, safety gate or pruning tripwire.
- Hardening/alternative architecture/pre-existing defects are follow-up.

## Independent deletion review
- What task-added or newly redundant code can be removed?
- Can branches/helpers/files/concepts collapse without changing behavior?
- Was a concept/framework/state introduced?
- Do tests prove behavior rather than implementation structure?
- Applied deletions receive frozen verification only.

## Result
- Final status: done | partially-done | blocked
- Production/test/docs deltas from original baseline:
- Context packet and boundary metrics before/after:
- Files/concepts removed / remaining candidates:
- Follow-up findings not auto-started:
- Residual risks / final Git status:

## DoD references
- [ ] Applicable DOD-01..DOD-12
- [ ] DOD-13
- [ ] DOD-14 when claimed
```

## D. Context simplification campaign — audit → selection → fixes → integration

Это campaign task artifact. Нормативные правила — `CTX-*`, `SIMPL-*`, `REV-*`, `PRUNE-*`.

```markdown
# <Context simplification campaign>

## Metadata and immutable baseline
- Status: proposed | audit | awaiting-selection | active | integration | done | partially-done | blocked | cancelled
- Owner / Baseline commit:
- Main worktree clean: yes/no; evidence:
- Primary domains/capabilities:
- Standard version: 1.3
- Maximum findings: 15 by default
- Maximum candidate/selected fixes: 6 by default
- Consecutive blocked-fix stop: 2 by default

## Past failure constraints and excluded artifacts
- Bounded lesson:
- Rejected branches/stashes/diffs/plans/reviews:
- Why not design/evidence sources:
- Reinstatement requires:

## Representative scenarios and baseline context
| ID | Scenario | Entry point | Required files/sections | Tokens/bytes | Domain transitions | Entry points | Sources | Central reads |
|---|---|---|---|---|---|---|---|---|

## Stack/domain-specific audit matrix
| Area/capability | Checks and semantic hazards | Authoritative sources | Explicit non-goals |
|---|---|---|---|

## Baseline totals
- Production/test/docs LOC and files:
- Public/operational entry points:
- Duplicate sources / blurred boundaries:
- Relevant runtime surfaces:

## Phase 1 — read-only audit
No file, Git-state or external-system mutation.

Output: at most five lessons from prior failure; baseline metrics/packets; boundary map; bounded findings (`location — problem — context cost — minimal deletion`); false positives; bounded atomic candidates; final line `STOP: awaiting human selection; no changes were made.`

## Candidate fix contract
- Fix ID / one problem / one measurable effect:
- Frozen acceptance (3–7):
- Allowed / forbidden paths:
- Production/test/docs budget and expected delta:
- Expected context packet change:
- Smoke/runtime path / rollback / tripwires:
- Reduced `CTX-02` metric:
- Correctness-only positive delta justification:

## Human selection gate
- Selected / rejected / deferred IDs:
- Frozen cumulative defaults: production `<= 0`; test `<= 0`; docs `< 0`; new production files/public entry points/frameworks/generic checkers/runtime states `0`.
- Deviations and operator justification:
- Approval record:

## Per-fix execution
- Independent worktree/branch from same baseline.
- Frozen acceptance, minimal diff, one adversarial review, one correction pass, deletion review.
- Outcome: accepted commit | blocked | rejected.

## Integration
- Integration worktree / accepted commits / excluded commits:
- Cumulative deltas and same-method packets:
- Cross-fix conflicts / full flow / combined deletion review:
- New findings moved to backlog:

## Stop conditions
- Max fixes, exhausted budget, no next measurable reduction, two blocked fixes, cosmetic remainder or operator stop.
- Do not start another wave automatically.

## Result
- Final status / accepted commits / deltas:
- Context and boundary metrics before/after:
- Entry points/sources/concepts removed:
- Regressions/unknowns/risks/backlog:
- Stop reason / final Git state:

## DoD references
- [ ] Applicable DOD-01..DOD-14
- [ ] DOD-15
```

## E. Adaptive section rules

| Classification | Required template |
|---|---|
| S/R1 | Short contract |
| M/R1-R2 | Full; R3 fields may be N/A with reason |
| L | Full with coverage, independent review and metrics |
| R3b/R3a | Full + R3 safety; R3a includes immediate human approval and rollback |
| Pruning | Section C; deletion review; max two mutation passes |
| Context simplification | Section D; audit, selection, same-baseline fixes, bounded integration |
| Research | Phase B optional; decision gate and falsifiers |
| Review | No implementation; baseline/result/evidence and independence |
