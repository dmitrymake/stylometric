# Paired-audit v3.2 receipt semantic closure

## Metadata

- Task ID: `2026-08-23-v3-2-receipt-semantic-closure`
- Status: complete
- Owner: Dmitry Purtov (repository owner)
- Created: 2026-08-23
- Baseline commit: `d95a5cb8e13bcdb89eaf8e7f2a5015d535618826`
- Target: attached `main` worktree; local changes only
- Type / Size / Risk: implementation / M / R2
- Risk flags: mutation, scientific-contract
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: research-governance (requirement binding only)

## 1. Goal and decision

- Observable result: a coherently rehashed v3.2 fold receipt cannot lie about its prediction,
  probability evidence, route, class alignment, fold/context bindings, axes, or train/test identities
  and still pass the owning validator.
- Consumer: the later, separately authorized v3.2 RunPlan/evaluator-registration/preflight task.
- Decision unlocked: accept the receipt/evaluator candidate as semantically closed enough to be an
  input to that later control-plane task, or record a bounded blocker.
- Success is not extra hashing, schema presence, production registration, freeze, real fit/predict,
  execution, result generation, headline change, publication, push, or deployment.

The operator instruction of 2026-08-23 explicitly requests an autonomous iterative accuracy loop.
The independently reproduced supported-path counterexample below selects this bounded first
implementation iteration; it does not authorize any excluded later stage.

## 2. Confirmed starting facts

| ID | Fact | Evidence | Confidence |
|---|---|---|---|
| F-01 | The evaluator task freezes truthful bindings/evidence and coherent-mutation negatives. | `docs/tasks/2026-08-10-paired-audit-v3-2-evaluator.md:85-112` | high |
| F-02 | `validate_receipt_v3_2` currently checks schema, numeric contract, outer self-hash, numeric vote coherence, and fitted-state digest, but not the rest of the receipt semantics. | `src/stylo/eval/paired_audit/evidence_v3_2.py:295-318` at baseline | high |
| F-03 | After recomputing `self_hash`, the baseline validator accepts five synthetic mutations: wrong `vote.pred_author`, wrong probability digest, wrong model, wrong fold binding, and empty class alignment. | bounded Python falsifier recorded in session/task result | high |
| F-04 | Focused paired-audit baseline tests pass with one environment-gated real-context skip; production registry is empty, freeze unapproved, execution hard-disabled. | exact commands in task result; ledger and executable constants | high |

## 3. Unknowns and assumptions

| ID | Item | Why it matters | Resolution | Blocking? |
|---|---|---|---|---|
| U-01 | Smallest non-tautological expectation surface for external bindings. | A caller-derived copy of the receipt would not authenticate context/fold truth. | Expectations must be derived from the verified context, manifest, registry, and evaluated fold inputs, not from receipt fields. | yes |
| U-02 | Whether every existing applied route exposes enough common evidence for one validator. | A model-specific exception could create another false-green path. | Exercise all 16 applied synthetic routes plus model-specific coherent mutations. | yes |
| U-03 | Availability of the ignored real v3.2 bundle. | Real corpus access is unnecessary and outside this task. | Keep the existing environment-gated test as a separately reported skip. | no |

## 4. Scope

### In scope

- `src/stylo/eval/paired_audit/evidence_v3_2.py`
- minimal owning integration in `src/stylo/eval/paired_audit/evaluator_v3_2.py`
- `tests/test_paired_audit_evaluator_v3_2.py`
- one exact requirement/nodeid binding in `research/governance/requirements.json`
- this task result and, only if unfinished state remains, `docs/handoff/CURRENT.md`

### Out of scope

- Applicability semantics, estimator math, feature math, corpus/fold/bundle bytes, config, protocol,
  dependency or runtime changes.
- `run_plan.py`, `runner.py`, checkpoints, inference, publisher, production evaluator registry,
  freeze pins, preflight, authorization, real fit/predict, result/headline/publication/site changes.
- Raw/private corpus inspection, external services/writes, security hardening, concurrency, generic
  evidence framework, legacy v3.1 control-plane repair, and unrelated findings.

### Stop / reclassification

- Any need for an excluded runtime/control-plane/publication path, real corpus operation, dependency,
  framework/state/public entry point, or new R3 factor.
- More than two production files, tripwire crossing, inability to construct expectations independently
  of receipt fields, or an unresolved blocker after the single correction pass.

## 5. Frozen acceptance and complexity tripwires

- Max production subsystems: one (`evaluation/paired-audit`).
- Allowed production files: the two existing v3.2 evaluator/evidence modules above.
- New production files/public or operational entry points: 0.
- Production LOC added ceiling: 220; production net delta ceiling: +220.
- Test LOC added ceiling: 260; task/governance/handoff LOC added ceiling: 360.
- New framework/layer, persistent/runtime state, dependency, generic abstraction: forbidden.
- Existing receipt schema/name may remain candidate-versioned; no migration/persistence is in scope.
- Representative context packet: N/A; no maintainability claim.
- Review budget: one independent adversarial review and at most one bounded correction pass.
- Any crossing is a stop/reclassification, never an automatic budget expansion.

## 6. Domain and scientific contract

- Producer: verified `V32EvaluationContext`, exact v3.2 applicability row, fitted estimator, and one
  held-out whole-work evaluation.
- Consumer: future v3.2 checkpoint/control-plane adapter, not implemented here.
- Grain: one receipt per `(context_identity, dataset, model, cell, fold_index, full work_id)`.
- Keys/cardinality: one tested fold row maps 1:1 to one full work ID and one receipt; class alignment is
  a bijection between estimator columns, dataset labels/authors, and the complete probability order.
- Time/history: immutable accepted candidate identities; no current pointer or late arrival.
- NULL/delete semantics: `ruaa_work_selection` and dataset row-selection may be null only where the
  verified context says so; metadata-only/equivalent/withdrawn cells produce no receipt.
- Shared interface: stable-top1/worst-tie-rank prediction contract and strict canonical numeric/hash
  helpers; no scientific semantics move into a shared layer.
- Verification: independent expectation construction, exact field/key/type/cardinality checks,
  semantic recomputation, all-route synthetic oracle, and coherent-rehash negatives.

## 7. Invariants

- INV-01: outer `self_hash` proves byte integrity only; acceptance also re-derives semantic truth.
- INV-02: public validation requires independently supplied authoritative expectations for every
  external context/fold/order binding; a receipt cannot authenticate itself.
- INV-03: literal probabilities determine their digest and vote; class alignment determines the
  probability-order authors and must agree with estimator/fitted-state classes.
- INV-04: model/cell registry semantics determine requested/effective axes; fitted-state model and
  estimator class agree with the receipt route.
- INV-05: validation is fail-closed before any future persistence/aggregation boundary.
- INV-06: registry stays empty, freeze stays absent, and no real fit/predict or publication occurs.

## 8. Phases and gates

### Phase A — context/research

- Sources: evaluator task, current evidence/evaluator modules, direct consumers, governance mapping.
- Falsifier: coherently mutate and rehash the five fields in F-03.
- Gate: complete. Five of five false receipts were accepted on the clean baseline; the operator's
  autonomous accuracy instruction selects this bounded fix.

### Phase B — implementation

- Make the owning validator strict and expectation-bound; integrate it at receipt construction.
- Add synthetic oracle and coherent-rehash negatives without changing estimator mathematics.
- Add one governance requirement binding exact executable tests.
- Rollback/abort: revert only this bounded working-tree diff; no external or durable runtime state.

### Phase C — bounded review/close

- One clean-context adversarial review against this frozen contract and original baseline.
- Blocking only for violated AC/invariant, reproducible regression, supported false receipt, safety
  gate, or tripwire breach; unrelated hardening is follow-up.
- At most one correction pass, followed by frozen verification; no architecture-review loop.

## 9. Evidence plan

| Claim | Required source | Reproduction | Negative check | Minimum confidence |
|---|---|---|---|---|
| Baseline validator is semantically open | baseline source + synthetic receipt | coherent mutation, recompute self-hash, call validator | at least one mutation unexpectedly rejected would narrow the claim | high |
| New validator closes internal semantics | code + focused tests | recompute probability/order/state/route relations | coherent digest/vote/alignment/axis mutations | high |
| External bindings are non-tautological | verified-context integration + tests | mutate expected fold/context/order independently | expectations sourced from receipt are forbidden | high |
| No later stage is enabled | registry/freeze/ledger checks | exact imports/status assertions | any non-empty registry/freeze or real execution is a blocker | high |

## 10. Frozen acceptance criteria

- [ ] AC-01 All five baseline coherent-rehash false receipts are rejected for their semantic cause,
  not merely because their outer self-hash was left stale.
- [ ] AC-02 Probability evidence is recomputed; vote, `pred_author`, true author/label, class-order
  evidence, alignment bijection, estimator classes, and fitted-state model/class agree exactly.
- [ ] AC-03 Context, bindings, dataset/model/cell/fold/work/content identities, orders, axes, and
  train/test identities are checked against expectations derived independently from receipt fields.
- [ ] AC-04 Missing/extra/wrong-type fields and coherent axis/order/count/digest mutations fail closed.
- [ ] AC-05 All 16 applied routes produce deterministic synthetic receipts accepted by the same strict
  public validator; metadata-only/equivalent/withdrawn routes still fail before factory/fit.
- [ ] AC-06 Governance maps the semantic-closure requirement to owning symbols and exact collecting
  negative/oracle nodeids.
- [ ] AC-07 Focused/relevant/full regression and hygiene checks pass; warnings/skips are separate.
- [ ] AC-08 Production registry remains empty; freeze/preflight/authorization/execution/publication
  status and all real scientific bytes remain unchanged.

## 11. Verification plan

### Targeted

- `.venv/bin/python -m pytest tests/test_paired_audit_evaluator_v3_2.py -q -p no:cacheprovider`
- `.venv/bin/python -m pytest tests/test_medium_governance.py -q -p no:cacheprovider`

### Relevant regression

- paired-audit evaluator/corrected/control-plane/inference/runner suites.
- Full `.venv/bin/python -m pytest tests -q -p no:cacheprovider` before close.

### Runtime/data

- Synthetic fixtures only. The existing real-context test remains environment-gated and any skip is
  reported; no raw/private corpus is opened.

### Negative scenarios

- The five baseline falsifiers plus wrong axes, orders, counts/types, estimator/fitted-state relation,
  and expected context/fold/train/test identities with a freshly recomputed outer self-hash.

### Hygiene

- physical-file py_compile, requirements-nodeid collection, executable inventory, release hygiene,
  Git diff/status and protected-surface audit.

## 12. R3 safety and approval

- N/A: no production/sensitive-data access, external write, publication, irreversible action, or real
  corpus fit/predict. Encountering one is a stop/reclassification.

## 13. Documentation impact

- ADR/domain/runbook: none; this enforces the already frozen receipt truth contract and changes no
  durable architecture, domain semantics, or operation.
- Governance: one requirement mapping; ledger status wording changes only if review accepts the
  corrected candidate and must remain explicitly unregistered/non-authorizing.
- Handoff: update only if needed to route unfinished state or the next separately authorized task.

## 14. Review contract

- Review the original-baseline diff, frozen AC/invariants, exact falsifiers, regression evidence,
  domain leakage, tripwires, and continued hard-disable state.
- A blocking finding must map to an AC/invariant/regression/counterexample/safety gate.
- No second review of the historical evaluator task is claimed; this review covers only receipt
  semantic closure introduced by this task.

## 15. Result

- Final candidate commit: `5eeae5542c12aa6a6c788c772f5a3b55a11c304a` on attached `main`.
- Baseline falsifier: five of five coherently rehashed false receipts were accepted. The corrected
  validator requires a separately built expectation and independently reconciles context/bindings,
  route/axes, literal orders, split identities, class alignment, probabilities/vote, axis evidence,
  and fitted state. The final negative oracle rejects the original five plus probability/state,
  axis/order/count, missing/extra field, and exact JSON scalar-type mutations.
- Review: the single independent review returned **FAIL** on AC-04/INV-05. It reproduced acceptance
  of equal-valued type aliases (`False` for fold `0`, integers for booleans/floats, floats for integer
  counts) and found the task's mistyped full baseline SHA. The one allowed correction pass added
  recursive exact-type equality, seven alias negatives, and corrected the baseline. Frozen
  verification passed afterward; no second-review PASS is claimed or required by the budget.
- Verification after correction: evaluator + governance `49 passed, 1` real-bundle skip; relevant
  paired-audit set `171 passed, 1` real-bundle skip; full suite exit `0` with 1,649 collected tests,
  three executed environment/data skips plus the separately collection-skipped live-golden module,
  and two pre-existing invalid-escape deprecation warnings. Physical-file `py_compile`, diff check,
  release hygiene, 299-path executable inventory (`747a07c5...`), governance nodeid collection, and
  hard-disable assertions passed.
- Tripwires from the original baseline: two existing production files, +220/-45 production LOC
  (net +175); +112/-14 test LOC (net +98); +348/-97 task/governance/handoff/release metadata LOC; no
  production file, public entry point, dependency, framework, generic abstraction,
  persistent state, registry entry, freeze, preflight, authorization, execution, or publication.
- Bounded scope clarification: the existing byte-bound governance gate required two source-hash
  updates in `status_ledger.json`, and the declared executable-inventory check required two matching
  hash updates in `release/executable_sources.json`. Status semantics, inventory membership, and
  publication bytes did not change; this is the only late requirement/deviation.
- Residual scientific risks: real-bundle context remains intentionally unrun; the active `stylo`
  MFW channel has a separately reproduced topical-content confound and must not be called
  factology-ready before a bounded topic-invariance study. The public Sholokhov circularity wording
  mismatch is a separate reporting/publication task. Neither finding expands this task.
- Final scientific state: evaluator remains unregistered, freeze `None`, preflight/authorization
  absent, execution hard-disabled, and headline/publication not authorized.

## 16. DoD references

- [x] DOD-01 through DOD-05
- [x] DOD-07 through DOD-12
- DOD-06/13/14/15: N/A unless scope is reclassified.
