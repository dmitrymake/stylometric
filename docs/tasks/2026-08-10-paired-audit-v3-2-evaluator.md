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
- Standard: `docs/agentic/STANDARD.md` v1.3

## 1. Goal and decision
- Observable result: a versioned v3.2 applicability registry, verified evaluation context,
  per-fold evaluator, and truthful self-hashed in-memory evidence receipt are implemented and pass
  one bounded independent scientific review.
- Consumer: a later, separately authorized freeze/RunPlan task.
- Unlocked decision: accept the evaluator as an unregistered candidate or record a frozen blocker.
- Success is not production registration, freeze, preflight, authorization, real fit/prediction,
  result generation, headline decision, publication, push, or deployment.

## 2. Confirmed starting facts
| ID | Fact | Evidence | Confidence |
|---|---|---|---|
| F-01 | v3.2 bundle/corpus/manifest/LOBO/RuAA identities are `ff620b…`, `1a9a07…`, `a2dc0c…`, `117b8e…`, `d84282…` | preparation record; `verify_v3_2_candidate` | high |
| F-02 | Corrected universes are LOBO 47/252 train, 43/248 tested, and RuAA 22/134 | protocol plus corrected-v3.2 tests | high |
| F-03 | Matrix is 25 statuses, 16 applied, 11 Holm, alpha 0.05, digest `92c2d1cca7de759a88167ad0a0c0395ab797a6b932f3f816bcc8c37e5ed2018b` | operator-frozen contract; corrected registry reconciliation | high |
| F-04 | Preparation is owner-accepted for evaluator implementation; independent security audit is not claimed and its review was terminated by owner | operator disposition, 2026-08-10 | high |
| F-05 | Production registry is empty; freeze unapproved; preflight/authorization absent; execution hard-disabled; publication not authorized | ledger and `CONFIRMATORY_EVALUATOR_REGISTRY` | high |

## 3. Unknowns and assumptions
| ID | Item | Resolution | Blocking? |
|---|---|---|---|
| U-01 | Estimator observation surface for truthful fitted-state evidence | inspect direct estimators; add only a narrow read-only observation hook if required | if actual state cannot be observed within tripwires |
| U-02 | Real local bundle availability | two read-only verified context loads; no fit/predict | yes if acceptance cannot run |
| U-03 | Historical schemas could look structurally similar | exact v3.2 schemas/identities and negatives | yes if rejection is unproved |

## 4. Scope
### In scope
- `src/stylo/eval/paired_audit/applicability_v3_2.py`
- `src/stylo/eval/paired_audit/evaluator_v3_2.py`
- optional `src/stylo/eval/paired_audit/evidence_v3_2.py`
- optional minimal `src/stylo/eval/paired_audit/__init__.py` export
- focused tests; necessary governance metadata/tests/inventory; task/handoff

### Out of scope
- Legacy `applicability.py`, `manifest.py`, `references.py`, `runner.py`, `run_plan.py`,
  `checkpoints.py`, `inference.py`, `result_audit.py`, `publisher.py`, historical fixtures/references.
- `paired_audit_protocol.md`, `models.registry.CONFIRMATORY_MODEL_SPECS`, production evaluator
  registry, and prepared v3.2 bundle/fold bytes.
- RunPlan/checkpoints/inference/auditor/publisher/preflight/freeze/authorization/real execution and
  publication surfaces.

### Stop/reclassification
- Any production/publication/external write, dependency/framework/state/public-entry-point need.
- Any frozen path or LOC/file-count tripwire crossing.
- Real corpus fit/predict, changed bundle bytes, unknown overlapping WIP, or new R3 factor.
- An unresolved second blocker after the single correction pass.

## 5. Frozen complexity tripwires
- Max production subsystem: one (`evaluation/paired-audit`).
- New production modules: at most three, exactly the named applicability/evaluator/optional evidence files.
- Production LOC added ceiling: 1600; test LOC added ceiling: 1500.
- Task/governance docs added ceiling beyond bootstrap: 350.
- New dependency/framework/generic abstraction/persistent runtime state/public CLI: forbidden.
- Legacy v3.1 control-plane edits and production registry edits: forbidden.
- Review budget: exactly one independent scientific review; at most one bounded correction pass.
- Any crossing requires STOP/reclassification, never silent budget expansion.

## 6. Domain and scientific contract
- Producer: corpus/data v3.2 preparation; consumer: evaluation/paired-audit candidate.
- Grain: context = one verified dataset manifest; evaluation = one held-out whole work; receipt = one
  dataset/model/cell/fold work vote.
- Keys/cardinality: full NFC work ID; fold maps 1:1 to one tested work and 1:N test chunks; train
  contains every eligible row except all rows of that work.
- Time: immutable prepared candidate identities; no current pointer or late arrival.
- NULL/delete: non-applicable/equivalent/withdrawn cells are metadata-only; exclusions are fixed
  upstream tombstones. RuAA work selection differs from DatasetProvenance row selection.
- Verification: independent row/work identities, exact counts/orders, leakage negatives, and two
  real read-only context loads with identical identity.

## 7. Frozen acceptance
- AC-01: five-model 25-status registry has 16 applied and 11 Holm cells, alpha 0.05 and frozen
  digest; it equals `corrected_v3_2.applicability_matrix()`.
- AC-02: `stylo_stack` is withdrawn before factory/fit; non-applied/equivalent cells produce no metrics.
- AC-03: `V32EvaluationContext` exists only after candidate verification and binds exact bundle,
  corpus, folds, applicability, config/protocol/content/work/row/order identities.
- AC-04: LOBO verifies 47/252, 43/248, widths 47/43 and four train-only singleton authors; RuAA
  verifies 22/134, widths 22/22 and held-out-author presence in train.
- AC-05: context loads only bundle-local corrected corpus/frags and `input_clean`; real context
  construction is read-only and performs no fit/predict.
- AC-06: evaluator rejects invalid context/dataset/model/cell/fold/work/content/order and v1/v3.1
  inputs before factory/fit.
- AC-07: each call creates a fresh exact `make_factory_for_ablation` estimator, excludes the whole
  held-out work, fits train only, predicts held-out chunks only, aligns frozen probability order,
  means chunk probabilities, and emits shared stable-top1/worst-tie-rank work vote.
- AC-08: LOBO train-only classes stay in the 47-wide vector but outside the 43-label macro-F1 universe.
- AC-09: canonical in-memory receipt binds requested identities, row/work counts/digests,
  requested/effective axes, factory/estimator/alignment, probability, vote, actual fitted state, and self-hash.
- AC-10: axis evidence comes from train inputs plus actual fitted state; expectations are not called
  observations; no pickle/repr; numeric encoding fixes dtype, shape, C order, and one rounding contract.
- AC-11: toy tests exercise all 16 routes; repeat is identical; fresh estimator/leakage, pre-fit
  rejections, ties/rank/alignment, and receipt/state/probability mutation are proved.
- AC-12: no real receipts/results persist and no evaluator reachability enters legacy runtime paths;
  production registry remains empty.

## 8. Review and evidence
- Falsifiers: wrong identities/orders/fold/work, train leakage, equivalent/withdrawn cells, coherent
  evidence mutation, historical schemas, and altered estimator class order.
- Review exact commit C in a separate clean worktree/session; reviewer makes no edits.
- Blocking only: wrong matrix; fold/work identity; leakage; unit; factory/axis route; class alignment;
  unobserved evidence; v3.1 reachability; deterministic regression; tripwire violation.
- Security/mount/TOCTOU hardening, concurrency, alternate framework, style/refactor, future control
  plane, and unrelated debt are non-blocking/out of scope.
- FAIL needs deterministic AC-mapped reproduction; at most one correction commit and no second review loop.

## 9. Verification
- Focused v3.2 tests; work-balanced routing/golden structural tests. Historical fixtures are only
  code-regression for stylo/bow/delta/char; majority is hand-derived.
- Two real verified context loads with an explicit no-fit/no-predict assertion.
- Full pytest; py_compile; inventory; provenance; release and Git-free archive hygiene; diff check.
- Warnings/skips/cannot-run are separate and never become PASS.

## 10. Documentation and safety
- ADR/domain/runbook: none; task/protocol already freeze the boundary and no durable architecture or
  operation is introduced. Handoff/ledger/status tests receive metadata-only updates.
- R3 approval: N/A; no production/sensitive mutation or external/public action.
- Rollback: local commits only; no existing scientific bytes mutate.

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

## 12. DoD
- [x] DOD-01 through DOD-05
- [x] DOD-07 through DOD-12
- DOD-06/13/14/15: N/A (no R3, pruning, maintainability claim, or campaign)
