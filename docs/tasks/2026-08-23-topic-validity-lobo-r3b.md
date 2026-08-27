# Corrected-LOBO topic-validity run

## Metadata

- Status: superseded (superseded — execution continues under the measured-fixed8 task)
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
