# Topic-validity challenger adapter and aggregate v1

## Metadata

- Task ID: `2026-08-23-topic-validity-challenger-v1`
- Status: complete
- Owner: Dmitry Purtov (repository owner)
- Created: 2026-08-23
- Baseline commit: `2db3be9c1604c90e360d3f662ecf7b1f1c73e0f0`
- Target: attached `main`; local commits only
- Type / Size / Risk: implementation / M / R2
- Risk flags: mutation, scientific-contract
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: feature-extraction (read-only canonical topic-strict semantics);
  research-governance (status/test mapping only)

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

## 4. Scope

## 11. Verification plan

- Targeted new module tests plus existing topic/receipt tests.
- Relevant evaluator/factory/prediction/json/governance suites; full Python regression.
- Exact module/test path inventory, physical py-compile, diff check and registry/freeze assertions.
- Mutation matrix includes semantic outer-rehash attacks, bool/int/float aliases, strict-JSON
  duplicate/nonfinite inputs, reordered dict determinism and forbidden-field injections.
- Confirm no scripts/CLI/catalog entry and no protected corpus/site/generated/evidence change.

## 13. Result

- Commits: implementation/review candidate `82b57c8a`; single correction `c0a55a1f`.
- Adapter: a sealed research-only study context binds verified v3.2 context/config/corpus/fold/class
  identities, adapter/source bytes, exact A0/A4 cells and current/topic-strict arms. Factories are
  fresh authoritative `stylo` routes; strict replaces only the canonical vectorizer with
  `topic_strict=True, relative_fw=None`, preserving pipeline/scaler/classifier and corner semantics.
- Aggregate: `stylo.topic_validity.aggregate.v1` derives stable top-1 from transient exact-float
  probabilities and persists only bound hashes, rational correct/total and strict-minus-current
  delta counts, four prediction-vector digests, and metric-order per-author five-way transitions.
  Validation reconstructs the full artifact from sealed context + transient records; work IDs,
  probabilities, row predictions, paths, official receipt/result/run IDs and inferential/readiness
  claims are absent and extra fields fail closed.
- Independent review: **FAIL** on AC-01/INV-05 because the first self-hashed binding used six
  `src/...` paths as source-identity keys. The one allowed correction retains identical source-byte
  closure as semantic role→SHA mappings (`adapter`, `official_evaluator`, etc.) and adds explicit
  path/value negatives. Frozen verification passed; no second-review PASS is claimed.
- Negative evidence: wrong selectors/context/binding, nonfresh/route drift, drop/extra/duplicate/
  shuffled fold records, wrong identities/width/types/nonfinite probabilities, extra cells/arms,
  coherent metric/binding/transition/self-hash mutation, forbidden aggregate fields and duplicate/
  nonfinite JSON all reject. Reordered strict JSON round-trips deterministically.
- Verification: targeted challenger/governance `43 passed`; relevant topic/evaluator/prediction suite
  `100 passed, 1` intentionally unavailable real-bundle skip; full regression exited zero with 1,672
  collected tests, three executed environment/data skips plus the separately collection-skipped
  live-golden module, and two pre-existing invalid-escape warnings. Py-compile, diff/protected-path
  audit, release hygiene and 302-path inventory (`ada8410c...`) passed.
- Tripwires: one new production module +399 LOC; one new test +356 plus six governance
  test lines; +282/-23 task/governance/roadmap/inventory/handoff LOC; no existing production edit, dependency,
  framework/layer/state/generic abstraction, CLI/export/registry/official schema, protected data,
  real fit/predict, site/publication or external write. Review/correction used `1/1`, `1/1`.
- Final state: challenger status is `implemented_review_blocker_corrected_unexecuted`; official
  registry `{}`, freeze `None`, execution authorization absent and confirmatory execution hard-disabled.
  The real corrected-LOBO study remains a separate R3b task/approval, not implied by this result.
