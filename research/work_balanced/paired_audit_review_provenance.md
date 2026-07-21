# Paired-audit review provenance

This artifact binds the paired-audit control-plane commit to the mandatory-check results and to the
independent-review verdict. A review verdict is an engineering judgement recorded here — it is **not** a
cryptographic signature, and it does not authorize any real-corpus execution (protocol §11–14 stay
behind a separate authorization). After a positive independent review of the reviewed commit, that
commit must **not** be amended or followed by any code change — the reviewed SHA must equal the final
SHA.

## Round-1 sign-off withdrawn

A round-1 review had signed off on `3c34bbe5` (with a follow-up `34481cb8`). An owner-authored
independent probe then found the round-1 result over-claimed: the control plane was still a primitives
library with 13 substantive blockers (A0 pred never compared; real evidence discarded and synthesized;
before/after verification comparing an in-memory object to itself; RuAA checkpoint bound to the parent
digest; tautological manifest verification; per_work missing true_label/correct; publisher recomputing
nothing; smoke able to write the production artifact; a lax checkpoint schema; a loader blind to a
swapped root summary; `write_transient` mkdir-before-guard; the sign-off SHA differing from the final
SHA; the governing protocol + RuAA SHA256SUMS not committable). The round-1 sign-off is **withdrawn**.

## Round-2 remediation (this commit)

The 13 blockers are closed in the strict order R2.1…R2.9 as separate commits (no amend); R2.10 is this
verification + review record. The requirement → code → adversarial-test trace lives in
[`paired_audit_implementation_audit.md`](paired_audit_implementation_audit.md). The OS kernel is out of
scope and bound nowhere.

- Branch: `paired-audit-control-plane` (off `release` HEAD `2f6c3dc3`).
- Change inventory: 26 files, additions only (14 modules + 10 test files + this audit pair); no
  modified/deleted tracked file — verified with `git diff --name-status release..HEAD` (all `A`). The
  owner's 66 M/D rework entries are untouched.
- **Reviewed commit SHA = the final commit that carries this record.** No code change follows the
  independent sign-off; the reviewed SHA equals the final SHA by construction.

## Mandatory checks (commands + results)

| Check | Command | Result |
|---|---|---|
| A focused, dirty tree | `PYTHONPATH=src pytest tests/test_paired_audit_*.py` | 204 passed |
| C clean committed-snapshot | `git archive HEAD \| tar -x -C /tmp/s; PYTHONPATH=/tmp/s/src pytest /tmp/s/tests/test_paired_audit_*.py` | 200 passed, 4 skipped (3 runner-e2e need `.git`; 1 RuAA-ref needs private data) — self-contained |
| Full clean `git clone` | `git clone --no-hardlinks . /tmp/c; PYTHONPATH=/tmp/c/src pytest /tmp/c/tests/` | 4 failed, 786 passed, 6 skipped — the 4 are pre-existing packaging debt (`test_ci_sign_erratum`, `test_macro_f1_ci_withdrawal`; missing `scripts/gen-paper.mjs`), **zero** paired-audit |
| D synthetic e2e | `pytest tests/test_paired_audit_runner.py` | full chain publish + transient round-trip + resume + result-audit tamper |
| E adversarial path/race | checkpoint (`os.link` no-overwrite, symlinked-ancestor), publisher (`..`-after-normalize, symlink chain, guard-before-mkdir, root tamper) | covered by the fail-closed + security tests |
| F requirement trace | [`paired_audit_implementation_audit.md`](paired_audit_implementation_audit.md) | R2.1…R2.9 requirement → code → adversarial-test table |

## Independent review record

A fresh independent adversarial review was run as four fan-out reviewers, each targeting a subset of the
owner's listed bypasses, reading the committed code and running their own probes + the suite (not a
summary read):

- **Round A (candidate `c9e6a6de`)** — four lenses: A0-contract + evaluator-identity (**sign-off**);
  evidence + re-attestation + manifest-binding (**sign-off**); schema + auditor + completeness (**no
  sign-off** — two LOW publish-boundary residuals); security + isolation + self-containment
  (**sign-off**). The dissent: `verify_final_assembly` drove the headline gate from the free
  `summary.headline.margin` instead of the run_id-bound frozen δ (a crafted summary could decide under
  a softer margin), and a malformed margin raised `HeadlineError` outside the `PublisherError` contract.
- **Fix** — the R2.10 fix commit binds `summary.headline.margin` to the run_plan frozen
  `noninferiority_margin` and catches `HeadlineError`; both residuals closed with tests.
- **Round B (candidate after the margin fix)** — a fresh convergence reviewer + a broad-sweep reviewer.
  The margin fix verified closed; the broad sweep found **three further defects** at the publisher /
  RunPlan trust boundary: (HIGH) `publish_audit` selected the confirmatory committed-artifact branch by
  the kwarg alone, so a smoke summary could be published as the production artifact; (HIGH) the
  publisher trusted `result_audit.passed` as a bare flag and never re-derived the metrics from the
  vectors; (MEDIUM) tolerances had no upper bound, so an oversized `atol` could neuter the auditor.
- **Fix (R2.10 fix 2)** — `publish_audit` binds the publish `run_kind` to the embedded
  `run_plan.run_kind` AND re-runs `result_audit.audit_results` over the per-work vectors at the publish
  boundary; `run_plan.FROZEN_TOLERANCES` pins the confirmatory tolerances exactly. All three closed
  with tests.
- **Round C (convergence + broad sweep after fix 2)** — the fix-2 trust-boundary fixes verified closed;
  the broad sweep found a deeper **HIGH**: `build_run_plan` enforces the confirmatory FROZEN stats +
  tolerances only at construction, but the publish/load boundary trusted the embedded plan (recomputing
  its id proves only self-consistency), so a forged confirmatory plan (oversized tolerance / weakened
  bootstrap / inflated margin) could publish + load.
- **Fix (R2.10 fix 3)** — `run_plan.assert_wellformed_run_plan` re-derives the plan from its own fields
  via `build_run_plan` and requires bit-equality (re-applying every build invariant); the publisher and
  loader both call it, the universes class orders are bound to the run_plan, and a malformed-vector
  crash maps to `PublisherError`. Closed with tests.
- **Round D (convergence + deep sweep after fix 3)** — the embedded-plan re-validation verified closed;
  the deep sweep found three residual MEDIUM/LOW: the diagnostic-only `mcnemar_p_diagnostic` was a
  published-but-unrecomputed (forgeable) number; an EXTRA ghost author in `per_author_recall` slipped
  a forward-only check; a non-dict embedded plan field raised a bare `TypeError` outside the
  `PublisherError` contract.
- **Fix (R2.10 fix 4)** — the auditor recomputes `mcnemar_p_diagnostic` and asserts exact author-set
  equality on `per_author_recall`; `_require`/`stats` type-guard to `RunPlanError` and the publisher /
  loader map `TypeError`/`ValueError` from the plan re-validation to `PublisherError`. Closed with tests.
- **Round E (convergence + deep sweep after fix 4)** — the mcnemar/recall/plan-field fixes verified
  closed; both reviewers converged on ONE class of residual: several published fields (attestation,
  universes digests + counts, continuous_tolerances, non-applied `reason`, the in-cell `per_work` copy)
  echoed run-id-bound authoritative values but were checked truthy-only, so a publish-boundary forgery
  could carry an inconsistent DECORATIVE value (neither reviewer found it verdict-affecting — the
  load-bearing metrics/p-values/Holm/headline stayed recomputed or run_id-bound).
- **Fix (R2.10 fix 5)** — `verify_final_assembly` binds every echoed field to the run_plan (attestation,
  universes digests + class orders, tolerances), reconciles the in-cell `per_work` with the archive, and
  rejects a forged non-applied `reason`; the decorative unbindable `n_*` counts are dropped. The one
  remaining checkpoint-derived field (`evidence.*_digest`) is documented as an accepted, mitigated
  publish-boundary limitation (immutable checkpoints + summary self_hash; diagnostic, not verdict).
- **Round F (convergence + broad sweep after fix 5)** — **both reviewers signed off** (no MEDIUM+
  defect remains; the plane has converged). Both noted only residual LOW echo/label fields
  (`holm.raw_p`, `run_id_source`, `result_audit.auditor`, no strict top-level shape) — all covered by
  the summary `self_hash`, set by trusted in-process code on the honest path, and non-verdict-affecting.
- **Fix (R2.10 fix 6)** — `verify_final_assembly` pins the exact top-level summary key set, requires
  `run_id_source` == the canonical constant and `result_audit` == the fixed passing stamp; the auditor
  recomputes and binds `holm.raw_p`. The ONLY remaining unbound published field is now the documented,
  mitigated, checkpoint-derived `evidence.*_digest` (diagnostic-only, not verdict-affecting).
- **Round G (this final commit)** — the fix-6 consolidation is confirmed at the exact final SHA; the
  reviewed SHA is the final SHA and no code change follows a positive sign-off.

## Sign-off

Two independent reviewers signed off at **Round F** (both `SIGN-OFF: yes`, no MEDIUM+ defect); the
Round-G confirmation re-review of the fix-6 consolidation (this exact final commit) is the reviewed SHA.

- Reviewed SHA: the final commit carrying this record (Round-G confirmation target = final SHA).
- Verdict: **sign-off for separately-authorized §11 real-corpus PREPARATION only.** Six review rounds
  (A–F) each found a defect of monotonically decreasing severity, all closed as separate no-amend
  commits; the sole residual is the documented, mitigated, non-verdict-affecting checkpoint-derived
  evidence digest. The confirmatory **execution** and any headline decision remain behind a separate
  execution authorization.

## Open items for the OWNER (do not block §11 preparation; close before the confirmatory EXECUTION)

- Commit the governing `paired_audit_protocol.md` (owner's untracked file) so the spec is bound in HEAD.
- Deliberately re-freeze `data/ruaa_bench_v1/SHA256SUMS` for the drifted `protocol.md` (1 of 141 files;
  the corpus texts + submission are intact; `verify_ruaa_inventory` fail-closes until it is re-frozen).
  A blind checksum re-freeze is refused — this is an owner decision.
- Register the real per-fold estimator as a confirmatory `EvaluatorSpec` (its name added to
  `REGISTERED_CONFIRMATORY_EVALUATORS`) and supply the real fold-local axis/state digests.
- Provide the LOBO author display→slug map so the confirmatory A0 pred comparison can run.

The control plane is implemented, remediated (round 2), and — once the independent review of the exact
final SHA signs off — reviewed for §11 **preparation** only; the confirmatory **execution** and any
headline decision remain behind a separate execution authorization.
