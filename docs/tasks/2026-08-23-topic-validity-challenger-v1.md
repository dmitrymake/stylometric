# Topic-validity challenger adapter and aggregate v1

## Metadata

- Task ID: `2026-08-23-topic-validity-challenger-v1`
- Status: active
- Owner: Dmitry Purtov (repository owner)
- Created: 2026-08-23
- Baseline commit: `2db3be9c1604c90e360d3f662ecf7b1f1c73e0f0`
- Target: attached `main`; local commits only
- Type / Size / Risk: implementation / M / R2
- Risk flags: mutation, scientific-contract
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: feature-extraction (read-only canonical topic-strict semantics);
  research-governance (status/test mapping only)
- Standard: `docs/agentic/STANDARD.md` v1.3

## 1. Goal and decision

- Observable result: one versioned research-only adapter binds exact A0/A4 current/topic-strict
  factories, and one aggregate-only schema can be semantically reconstructed from transient paired
  fold probabilities without persisting protected row-level evidence.
- Consumer: a later separately approved corrected-LOBO R3b execution task.
- Decision unlocked: accept the implementation/schema as a safe unexecuted input to that task or
  record a bounded blocker.
- Success is not real fit/predict, a model choice, official v3.2 receipt/result/run identity,
  registration, freeze, preflight, authorization, headline, publication, CLI or runner.

The operator's ongoing accuracy loop authorizes this reversible implementation predecessor. The
protected-corpus run remains outside this task and needs a new approval/reclassification.

## 2. Confirmed starting facts

| ID | Fact | Evidence | Confidence |
|---|---|---|---|
| F-01 | Current v3.2 evaluator is sealed to one official factory/receipt route and cannot represent a challenger. | `evaluator_v3_2.py:311-459`; `evidence_v3_2.py:370-479` | high |
| F-02 | Canonical topic-strict semantics are `StyloVectorizer.from_config(..., topic_strict=True)`, not a config override. | completed topic task; `registry.py:28-65`; `vectorizer.py:28-38` | high |
| F-03 | Only A0/A4 are decision-relevant for the bounded real comparison; both require legacy corner coupling `relative_fw=None`. | topic task/handoff; `function_words.py:1-16,122-164` | high |
| F-04 | No existing aggregate contract provides the required paired transitions/digests without row-level persistence. | bounded active-code search | high |
| F-05 | Registry is empty, freeze `None`, fit/predict authorization absent and execution hard-disabled. | executable gates + ledger | high |

## 3. Unknowns and assumptions

| ID | Unknown | Why it matters | Resolution | Blocking? |
|---|---|---|---|---|
| U-01 | Real A0/A4 predictions and transition counts. | They determine model decision. | Separate R3b run; no substitutes. | blocks model decision, not implementation |
| U-02 | Exact real execution/runtime closure identity. | Required for a durable aggregate. | Schema reserves independently supplied implementation/environment digests; R3b preflight derives them. | yes for execution |
| U-03 | Whether topic-strict should become canonical. | Changes scientific semantics. | Owner decision after aggregate; likely ADR then. | yes for replacement |

## 4. Scope

### In scope

- New `src/stylo/eval/paired_audit/topic_validity_v1.py` with no package export.
- New `tests/test_paired_audit_topic_validity_v1.py` using generated synthetic contexts/records only.
- Versioned sealed study context/binding; A0/A4 current/topic-strict fresh factories.
- Strict aggregate builder/validator over transient probabilities with per-author transition counts and
  prediction-vector digests.
- Necessary requirement/status/roadmap/inventory metadata, task result and handoff.

### Out of scope

- Edits to every existing production Python/config/resource/protocol/applicability/model/evaluator/
  receipt/run-plan/runner/checkpoint/result module or registry.
- Script/CLI/runner/catalog/runbook, persistent execution state, real context construction, protected
  corpus read, fit/predict, checkpoint, receipt/result artifact, publication/site/external write.
- Model selection/readiness claim, CI/p-value/Holm/headline/verdict, work-level or probability output.

### Stop / reclassification

- Any need to refactor official evaluator, add an operational entry point, access real/private data,
  change a frozen path/status, persist forbidden detail, or cross a tripwire.
- Challenger differs from current beyond canonical topic-strict vectorizer semantics.
- Unresolved blocker after the one allowed correction pass.

## 5. Frozen acceptance and complexity tripwires

- Production subsystems: one (`evaluation/paired-audit`).
- Production files: exactly one new module; existing production edits: 0.
- Production LOC added/net ceiling: 500/500.
- Tests: exactly one new file; test LOC added ceiling: 550.
- Task/governance/roadmap/inventory/handoff LOC added ceiling: 450 after task bootstrap accounting.
- New dependency/framework/layer/generic abstraction/persistent state/CLI/public export/registry/
  official receipt/result schema: 0.
- Real data/fit/predict and production/publication state changes: 0.
- Review: one independent scientific review and at most one bounded correction pass.

## 6. Domain and data contract

- Producer: sealed `V32EvaluationContext`, authoritative A0/A4 factory, canonical topic-strict
  vectorizer and transient per-fold probability vectors.
- Consumer: aggregate-only decision-support study; never official paired-audit evidence.
- Grain: transient record = one `(cell, arm, held-out whole-work fold)` probability vector; aggregate
  row = one cell or one author within a cell.
- Keys/cardinality: cells exactly A0/A4; arms exactly current/topic_strict; identical contiguous fold
  order across four routes; one fold identity and one true label per tested work; author aggregates in
  metric-label order.
- Time/status: one immutable verified context and adapter source identity per future study.
- NULL/delete/late arrival: none; missing/extra/duplicate/misaligned cell/arm/fold/author is fatal.
- Allowed persisted detail: bound content hashes, counts/rational accuracy/delta, author IDs with
  transition counts and prediction-vector digests.
- Forbidden persisted detail: text/chunks/tokens/vocabulary/features, work IDs/slugs, fold rows or
  identities, probabilities/scores, prediction/correctness vectors, checkpoints/fitted state/model
  bytes, paths/host/timestamps, official receipt/result/run_id, inferential/headline/readiness fields.

## 7. Challenger binding and factory invariants

- INV-01: validate exact sealed context/cfg before any factory construction.
- INV-02: selectors are exact strings and only `stylo`, A0/A4, current/topic_strict exist.
- INV-03: both arms originate from a fresh authoritative `make_factory_for_ablation` call.
- INV-04: strict replaces only the vectorizer with `topic_strict=True, relative_fw=None`; pipeline,
  scaler, classifier, seed/loss/F/R/W cell and params remain exact.
- INV-05: every call returns fresh estimator/vectorizer objects; binding rotates with context/config/
  adapter source and contains no forbidden payload.
- INV-06: no official evaluator/receipt/registry/config identity is reused as challenger authority.

## 8. Aggregate v1 contract

- Schema: `stylo.topic_validity.aggregate.v1`; status/authorization flags are research-only/false.
- Study context is sealed and built independently of aggregate/transient records from verified
  context, manifests, class orders, adapter binding and supplied implementation/environment digests.
- Transient record exact fields: `fold_index`, domain-separated `fold_identity`, and one finite
  exact-float probability vector. Records are never returned, serialized or logged.
- Prediction uses existing stable-top1-lowest-index; truth/fold/author come from sealed context.
- Aggregate persists exact bindings/design, A0/A4 rational correct/total counts, strict-minus-current
  delta numerator/denominator, four prediction-vector digests, and per-author five-way correctness/
  changed-prediction transition counts.
- Validator requires exact keys/types/order and independently recomputes predictions, metrics,
  transitions, digests, study identity and self-hash from sealed context + transient records.
- Transition categories partition every fold: both correct; current only; strict only; both wrong
  same prediction; both wrong changed prediction.

## 9. Phases and gates

### Phase A — implementation

- Add the isolated module, exact source/context binding, factories, sealed aggregate context, builder
  and validator.
- Add synthetic positive/negative tests; no operation on repository/private corpora.
- Gate: operator's active loop plus this frozen contract.

### Phase B — independent review

- One clean-context review of exact baseline..candidate against AC/invariants and forbidden payloads.
- Blocking only for wrong route/delta/schema/recomputation, false-green mutation, regression,
  scope/tripwire or safety/authorization violation.
- At most one correction, then frozen verification without a second review loop.

### Phase C — real study

- Explicitly out of scope. New R3b task must assert exact 47/43/248 official context, derive runtime/
  source closure, supply/load/output/abort plan and obtain owner approval before access or fit.

## 10. Frozen acceptance criteria

- [ ] AC-01 Binding is deterministic, source/context/config/class-order/fold bound, research-only,
  self-hashed and free of forbidden details; coherent mutations reject against sealed context.
- [ ] AC-02 Factories accept only A0/A4 and current/topic_strict, validate context before factory,
  preserve exact non-vectorizer routes/params and freshness, and expose canonical MFW/fixed-list
  semantics with legacy corner coupling.
- [ ] AC-03 Aggregate accepts exactly four aligned transient routes, computes stable top-1 from
  probabilities, and emits only the whitelisted aggregate schema.
- [ ] AC-04 Counts, rational accuracy/delta, per-author transitions, prediction digests, study identity
  and self-hash independently reconcile from sealed context + transient records.
- [ ] AC-05 Negatives reject wrong types/keys/bindings, A1/RuAA/non-stylo/extra arm, route drift,
  247/249/duplicate/shuffled folds, class/order mismatch, nonfinite/integer probability, coherent
  metric/digest/transition/self-hash forgery and every forbidden persistent field.
- [ ] AC-06 Existing synthetic topic oracle still passes; official registry/freeze/authorization/
  execution/publication state remains unchanged and no real data operation occurs.
- [ ] AC-07 Requirement/nodeids, targeted/relevant/full tests, py-compile, inventory/release hygiene and
  diff audit pass; skips/warnings/cannot-run are separate.

## 11. Verification plan

- Targeted new module tests plus existing topic/receipt tests.
- Relevant evaluator/factory/prediction/json/governance suites; full Python regression.
- Exact module/test path inventory, physical py-compile, diff check and registry/freeze assertions.
- Mutation matrix includes semantic outer-rehash attacks, bool/int/float aliases, strict-JSON
  duplicate/nonfinite inputs, reordered dict determinism and forbidden-field injections.
- Confirm no scripts/CLI/catalog entry and no protected corpus/site/generated/evidence change.

## 12. R3 and documentation boundary

- This task is R2 because it executes only generated synthetic data. Encountering protected data is a
  stop and new R3b task/approval, not an implementation shortcut.
- ADR/domain/runbook: none now. Canonical model selection later requires a scientific ADR; real
  operation may require a runbook. Protocol is unchanged because challenger is nonconfirmatory.
- Governance: implemented/reviewed-but-unexecuted limitation only; no readiness promotion.

## 13. Result

- Pending.

## 14. DoD references

- [ ] DOD-01 through DOD-05
- [ ] DOD-07 through DOD-12
- DOD-06/13/14/15: N/A unless reclassified.
