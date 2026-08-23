# Corrected-LOBO topic-validity run

## Metadata

- Status: blocked
- Owner: Dmitry Purtov
- Created: 2026-08-23
- Baseline: `42d7e8a482a1cd1f26490a897831fcc15f7cb050`
- Type / Size / Risk: research / M / R3b
- Flags: load, protected-corpus-read, scientific-result
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: corpus/data read-only; research-governance aggregate status
- Approval: repository owner, current turn — “разрешаю! Мы делаем чище! Без излишней формализации!”

## Goal

Run the already reviewed A0/A4 current-versus-topic_strict study on the exact corrected LOBO-248
folds and retain one aggregate-only artifact sufficient for a model-semantics decision.

Success is not confirmatory execution, evaluator registration, freeze, publication, headline, or an
automatic model replacement.

## Scope and boundaries

In scope:

- Reproduce one exact temporary v3.2 bundle from the frozen historical parent.
- Extend the research-only topic module with one transient whole-work fold evaluator and runtime/thread
  bindings; add one thin explicit runner and focused tests.
- Execute exactly `248 folds × 2 cells × 2 arms = 992` fresh fits, sequentially unless a reviewed
  bounded worker count is proven identical.
- Persist only `research/evidence/topic_validity_lobo_v1/aggregate.json` and task/governance evidence.

Out of scope:

- RuAA, A1–A3, other models, probabilities/work IDs/text/vocabulary/checkpoints/fold receipts in the
  retained artifact, official result/receipt/run IDs, model selection by accuracy alone, any publish,
  deploy, push, registry/freeze/preflight/headline change.
- Raw text or row-level prediction output in logs/task/transcript/Git.

Stop on wrong bundle/context identity, counts other than 47/43/248, dirty/unsupported environment,
nonfinite output, aggregate schema violation, protected detail in output, unreviewed route change,
or load estimate that cannot be bounded in the current environment.

## Frozen scientific contract

- Dataset/model/cells/arms: LOBO / `stylo` / `[A0,A4]` / `[current,topic_strict]`.
- Identical ordered folds, train/test rows, class orders, seeds, scaler/classifier/loss/F/R/W route.
- Only challenger delta: canonical `topic_strict=True, relative_fw=None` vectorizer.
- Whole-work probability = aligned mean of held-out chunk probabilities rounded to the existing
  12-decimal contract; top-1 uses stable lowest-index tie rule.
- Aggregate direction: `topic_strict_minus_current`; per-author five-way transitions and predicted-
  label digests are reconstructed from transient probabilities.
- Runtime binds executable source, requirements lock, installed numerical stack and thread contract.

## Output minimization

Allowed artifact fields are exactly those accepted by `stylo.topic_validity.aggregate.v1`: identities,
design counts, rational accuracy/delta, prediction digests and per-author aggregate transitions.
Transient probabilities/fold identities live only in process memory. Existing identity-scoped,
ignored NLP/RepCache may be warmed as regenerable runtime acceleration; it is not evidence. No study
checkpoint/result sidecar is written; the temporary bundle stays outside Git.

## Implementation and load tripwires

- Existing research module: at most +140 LOC; one script at most 220 LOC; tests at most +220 LOC.
- No new dependency/framework/state/official schema/registry/public entry point.
- One implementation review and at most one correction before access.
- One-fold timing probe after reviewed no-fit preflight; extrapolate 992 fits and record abort threshold.
- Sequential execution default; no worker expansion without deterministic equivalence evidence.

## Acceptance

- [ ] Exact bundle/context identities and 47/43/248 universe reproduce without fit.
- [ ] Runner cannot select other cells/arms/dataset/model or output forbidden detail.
- [ ] Implementation/runtime/thread identities bind the aggregate.
- [ ] Synthetic evaluator/aggregate tests and implementation review pass within budget.
- [ ] One-fold timing probe yields a bounded full-run estimate accepted by the recorded approval.
- [ ] All 992 fits complete once; aggregate validates independently against transient reconstruction
  during execution and contains no forbidden keys/values.
- [ ] Independent clean-context aggregate audit verifies hashes/counts/math and reports current versus
  strict accuracy, delta and author transitions without claiming canonical model selection.
- [ ] Relevant/full tests, inventory/release hygiene and protected-data scan pass.
- [ ] Registry remains empty, freeze `None`, confirmatory execution hard-disabled, publication absent.

## Verification and abort

- No-fit preflight: prepare/verify bundle, build context/study binding, exact counts and environment.
- Targeted: topic module/runner tests, existing topic oracle, evaluator/context tests.
- Result audit: strict JSON load, aggregate reconstruction receipt produced in memory, forbidden-field
  scan, identity/hash/count/transition reconciliation.
- Abort before output on any failure. Existing output is create-once/no-clobber; no partial artifact.
- No rollback beyond removing local temporary bundle/output candidate; no external state changes.

## Result

- Implementation commits: runner `7cdebea0`; bounded cache-warm correction `5e581f85`.
- Independent pre-execution review: PASS, zero blockers. No second implementation review loop.
- Exact no-fit preflight passed on commit `5e581f85`: candidate `ff620b05...`, context `2805aff9...`,
  study binding `a2c80327...`, universe `248/43/47`, runtime `3301ed9a...`, threads `869ca93d...`.
- Existing identity-scoped cache warm completed for 22,908 rows with 8 spaCy workers in 3,093.8 s;
  it is ignored/regenerable and not evidence.
- One approved A0/current whole-work timing fit completed in 357.051 s. Sequential 992-fit estimate is
  98.4 h plus warm, exceeding this task's bounded load horizon.
- No aggregate/output/checkpoint/official artifact was created; registry/freeze/publication state did
  not change. Execution stopped at the declared load tripwire.
- Next task: fixed 8-worker fork execution with synthetic serial/parallel byte-equivalence proof. At
  measured warm-fold cost its conservative horizon is about 12.3 h plus cache load.
