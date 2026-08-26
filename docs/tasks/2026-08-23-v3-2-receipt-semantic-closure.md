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

## 4. Scope

## 11. Verification plan

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
