# Paired-audit v3.2 evaluator candidate

## Metadata
- Task ID: `2026-08-10-paired-audit-v3-2-evaluator`
- Status: completed
- Owner: Dmitry Purtov (repository owner)
- Created: 2026-08-10
- Baseline commit: `df663d0bff48944dcc5b4f939f10fcdaf4809c0b`
- Target: attached `main` worktree; local commits only
- Type / Size / Risk: implementation / M / R2
- Risk flags: mutation
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: corpus/data (read-only verified context); research-governance (metadata/status only)

## 1. Goal and decision
- Observable result: a versioned v3.2 applicability registry, verified evaluation context,
  per-fold evaluator, and truthful self-hashed in-memory evidence receipt are implemented and pass
  one bounded independent scientific review.
- Consumer: a later, separately authorized freeze/RunPlan task.
- Unlocked decision: accept the evaluator as an unregistered candidate or record a frozen blocker.
- Success is not production registration, freeze, preflight, authorization, real fit/prediction,
  result generation, headline decision, publication, push, or deployment.

## 4. Scope
## 9. Verification
- Focused v3.2 tests; work-balanced routing/golden structural tests. Historical fixtures are only
  code-regression for stylo/bow/delta/char; majority is hand-derived.
- Two real verified context loads with an explicit no-fit/no-predict assertion.
- Full pytest; py_compile; inventory; provenance; release and Git-free archive hygiene; diff check.
- Warnings/skips/cannot-run are separate and never become PASS.

## 11. Result
- Commits: implementation `f2f9eb95816febfe7a82d6050f5e5fc1bdb35331`; bounded correction
  `4c74df5e` after the single review. The metadata-only closeout commit contains this result.
- Matrix/context evidence: 25 statuses, 16 applied cells, 11 Holm members, alpha 0.05, frozen
  applicability digest `92c2d1cc…`; two real read-only loads produced context identity
  `2805aff9…` with LOBO 47/252 and RuAA 22/134. No real fit/predict ran.
- Review: exactly one independent review at commit `f2f9eb95` returned FAIL on AC-07 because a
  partial estimator class universe was accepted. The one allowed correction rejects that case before
  prediction while preserving reordered complete-universe alignment. No second review/PASS is claimed.
- Verification after correction: 41 evaluator/context tests and 167 routing/golden tests passed;
  full pytest passed 1695 collected tests with 1 expected env-gated skip, 0 failures/errors, plus
  py_compile, executable inventory (321 paths, digest `eee8fc5f…`), provenance, release, archive,
  and diff hygiene.
- Cumulative tripwires from baseline: three production modules, 942/1600 production LOC added,
  425/1500 test LOC added, and 263/350 task/governance doc lines; no dependency, framework,
  persistent state, CLI, legacy control-plane, production registry, or prepared-byte edit.
- Accepted result: corrected local evaluator/evidence candidate, unregistered and non-authorizing.
  Freeze remains unapproved; preflight/authorization absent; execution hard-disabled; publication
  not authorized. Residual risk is the deliberately absent second independent review of the corrected
  commit; the next task must independently review its own freeze/control-plane acceptance.
- Review/correction used: 1/1 and 1/1. Final Git state is recorded in the closeout handoff/report.
