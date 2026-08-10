# Current Handoff

- State: **v3.2 evaluator accepted/unregistered; bounded cleanup A+B, C1+C2 and D1+D2 complete**
- Updated: 2026-08-11
- Verified baseline commit: `bf89f381` (integrated D1+D2 legacy Python surface pruning)
- Branch/worktree at capture: `main`; attached; seventeen commits ahead of `origin/main` before this
  metadata-only closeout
- Active task: none; cleanup contract is complete
- Active standard: `docs/agentic/STANDARD.md` v1.3

This file routes the next session. Scientific and authorization status remains owned by the task,
governance ledger, protocol, and executable gates, not by this handoff.

## 1. Verified completed state

| Item | Verified status | Evidence owner |
|---|---|---|
| Cleanup A+B+C1+C2+D1+D2 | Prior cleanup plus legacy statistic island and superseded macro-F1 writer removed; reviews PASS | cleanup tasks; `bf89f381` |
| Preparation | Owner-accepted for evaluator implementation | `status_ledger.json`; preparation review |
| Independent security audit | **Not claimed**; review terminated by owner | `status_ledger.json` |
| Evaluator candidate | Implemented at `f2f9eb95`; single review FAIL on AC-07 corrected at `4c74df5e`; no second review/PASS claimed | evaluator task Result |
| Scientific acceptance | Corrected candidate accepted locally after affected/full regression; exactly 25/16/11 | evaluator task; v3.2 tests |
| Production evaluator | Registry empty; candidate unregistered | `CONFIRMATORY_EVALUATOR_REGISTRY` |
| Freeze | Unapproved | governance ledger |
| Preflight/authorization | Absent | governance ledger |
| Execution | Hard-disabled; no real v3.2 fit/predict occurred | governance ledger; task evidence |
| Headline/publication | Not authorized | governance ledger |

## 2. Material changes in the cleanup wave

- Protocol/catalog no longer repeat stale `pending` milestone state; requirements distinguish
  historical v3.1 m15 from current v3.2 25/16/11.
- Twelve hard-disabled legacy script shims and their dead contract tests were deleted. Historical p0
  hashes and Git evidence remain unchanged.
- The unbounded codebase dumper was deleted and npm is now the sole site package-manager contract;
  dependency versions, npm lock bytes and generated/public site bytes are unchanged.
- The internally closed legacy statistic/helper island and hard-disabled macro-F1 correction writer
  were deleted: 11 Python paths plus one private JSON resource, net -1,098 production Python LOC.
  Active feature/NLP and erratum implementations, scientific artifacts and claims are unchanged.
- No corpus bytes, results, receipts, production runtime, publication surface, external system,
  freeze, preflight, authorization, or execution changed.

## 3. Next separately authorized task

Define the v3.2 RunPlan/evaluator-registration/freeze/preflight boundary under a new task contract.
Do not begin checkpoints, inference, result auditing, publishing, freeze pinning, authorization, or
the real 6112-fold execution from this handoff.

Primary domain remains `evaluation/paired-audit`; corpus/data access is read-only verified context,
and research-governance changes are metadata/status only. Legacy v3.1 control-plane modules and
prepared v3.2 bundle/fold bytes remain frozen.

Cleanup candidates D3-D6 remain unselected; do not start another simplification wave without a new
state snapshot and explicit owner selection.

## 4. Revalidation conditions

Revalidate before mutation if HEAD history does not contain baseline `3b8cb83b` plus this closeout,
branch is not attached `main`, task/governance bindings differ, the production registry is nonempty,
freeze/preflight/authorization/execution status changes, or a new R3/sensitive-data factor appears.

## 5. Available and unavailable evidence

- Available: tracked task/governance/protocol/code/tests, local Git history, and the verified local
  bundle for bounded read-only context construction.
- Unavailable/not inspected: external services, GitHub Pages runtime, secrets, production publication
  state, and raw/private corpus content as reportable evidence.

## 6. References

- Task: `docs/tasks/2026-08-10-paired-audit-v3-2-evaluator.md`
- Cleanup: `docs/tasks/2026-08-11-governance-and-retired-shim-cleanup.md`
- Active surface: `docs/tasks/2026-08-11-active-surface-cleanup.md`
- Legacy Python surface: `docs/tasks/2026-08-11-legacy-python-surface-pruning.md`
- Scientific status: `research/governance/status_ledger.json`, `research/ROADMAP.md`
- Preparation/protocol: `research/work_balanced/paired_audit_v3_2_preparation_review.md`,
  `research/work_balanced/paired_audit_protocol.md`
- Process: `docs/agentic/STANDARD.md`, `docs/handoff/README.md`
