# Paired-audit control-plane implementation audit

Status: control plane + confirmatory runner implemented, Gate-10 remediation applied, and committed on
branch `paired-audit-control-plane` as additions only. No real audit corpus, no fold manifest, and no
confirmatory cell has been prepared or run on real data. Execution stays gated behind an independent
code review of the exact final commit and a separate execution authorization (protocol §11–14).

Normative source: [`paired_audit_protocol.md`](paired_audit_protocol.md) (v3.1). On any conflict the
protocol governs. This document is a declarative implementation record, not a sign-off; the sign-off
lives in [`paired_audit_review_provenance.md`](paired_audit_review_provenance.md).

## Package inventory (13 modules, 10 test files — all committed additions)

`src/stylo/eval/paired_audit/`: `semantic_parity`, `corpus`, `work_subset`, `applicability`,
`run_plan`, `inference`, `headline`, `checkpoints`, `publisher`, `manifest`, `references`, `runner`,
`__init__`. `tests/test_paired_audit_{corpus, control_plane, inference, headline, checkpoints,
publisher, manifest, fail_closed_sweep, references, runner}.py`.

## Requirement → module → test trace

| Protocol | Requirement | Module | Tests |
|---|---|---|---|
| §1.2 | Loader-agnostic semantic digest; frozen legacy anchor; exact row equality | `semantic_parity` | `corpus::TestSemanticParity`, `TestSemanticRowDigest` |
| §1.3 | Whole immutable audit-corpus root; atomic publish; reverify-before-pointer; per-chunk byte/filename compare; live corpus never mutated | `corpus` | `corpus::TestBuilder`, `TestBuilderFailClosed` |
| §1.4 | Audit dataset is work_balanced_manifest for EVERY cell incl. A0 (legacy loader only for anchor/parity) | `corpus` | `corpus::test_audit_dataset_is_work_balanced_for_every_cell`, `test_audit_only_verifier_rejects_legacy_dataset` |
| §1.5 | `derive_work_subset` whole-work exact-set RuAA panel; three-digest binding | `work_subset` | `corpus::TestWorkSubset` |
| §1.6/§12 | Fold manifests; counts recomputed from works; runner rebuilds & requires exact equality; frozen 47/255-137/22 universes; 4 named singletons | `manifest` | `manifest` (incl. lying-`n_works` regression) |
| §2.4/§3.4 | Exactly 21 applied / 15 comparisons; per-model decomposition pinned; requested+effective axes; typed signals carry no metrics | `applicability` | `control_plane::TestApplicability` |
| §3.2 | Pinned A0 reference SHA before parse; lobo 221/251 + per-work; RuAA parse + full SHA256SUMS inventory; fail-closed preflight | `references` | `references` |
| §3.3 | Two-sided null-centered author-cluster bootstrap p (B=10000, seed=42, +1, degenerate order); McNemar diagnostic-only; stat input validation | `inference` | `inference::TestClusterPValue`, `TestMcNemarDiagnostic` |
| §3.4 | Holm on unrounded p over the fixed 15-member family; m never reduced; strict significance | `inference` | `inference::TestHolm`, `TestHolmRegisteredFamily` |
| §3.5 | Single stylo A4−A0 headline; author-clustered percentile CI; δ=0.02; unrounded/boundary-equality decision; macro-F1 CI withdrawn | `headline` | `headline` |
| §4.1 | Artifact schema literally §4.1 (status applied/not_applicable/equivalent_to; requested+effective axes; validated evidence) | `applicability` | `control_plane::test_cell_record_validation`, `test_applied_evidence_field_validation` |
| §4.2 | Canonical RunPlan → run_id; structural runtime allowlist (no kernel); finite tolerances; registered kinds; class-order digest producer | `run_plan` | `control_plane::TestRunPlan` |
| §4.3/§7 | Per-fold immutable checkpoints; atomic os.link no-overwrite; guards; registry; resume; run-COMPLETE + inventory | `checkpoints` | `checkpoints` |
| §4.4/§8 | Verified publisher: full-assembly verification, content-addressed archive + SHA256SUMS, atomic pointer/COMPLETE, path guard | `publisher` | `publisher` |
| §5 (runner) | One chain refs→dataset→manifests→matrix→RunPlan→cells→checkpoints→COMPLETE→metrics→cluster-p→Holm→headline→publisher; synthetic-only | `runner` | `runner` (synthetic end-to-end + resume) |
| §9 | Full fail-closed catalog | all above | `fail_closed_sweep` (coverage manifest) |

## Gate-10 remediation (independent-review blockers closed, separate commits, no amend)

#1 real runner (synthetic-only); #2 A0 on the WB-manifest dataset; #3 self-contained committed snapshot
(no rework-only `canonical_hash`; unit suite independent of ignored RuAA data); #4 manifest counts
recomputed from works + strict validation; #5 complete §3.2 A0 reference parse/verify; #6 §4.1 status
values + effective_axes; #7 verified publisher (full-assembly verification); #8 checkpoint atomic
os.link / guards / registry / run-COMPLETE; #9 RunPlan structural runtime allowlist + finite tolerances;
#10 statistics input validation.

## Verification results

- Diff vs `release` HEAD `2f6c3dc3`: 25 files, additions only; no modified/deleted tracked file — the
  control plane depends only on committed HEAD APIs and does not import the uncommitted working-tree
  rework.
- Clean committed-snapshot replay via `git archive` (mandatory check C): self-contained; the runner
  end-to-end tests skip without a git repo (they need a live commit binding), the RuAA-reference test
  skips without the gitignored private data.
- Full working-tree suite (check B) and the synthetic end-to-end runner (check D) pass; adversarial
  path/symlink/race guards (check E) are covered by the checkpoint (os.link, symlinked-ancestor) and
  publisher (traversal-after-normalization, symlink chain) fail-closed tests.
- `python -m py_compile` clean on every module and test. `ruff` is absent from the environment and CI.

## Provisioning finding for §11

`data/ruaa_bench_v1/protocol.md` has drifted from its frozen `SHA256SUMS` entry (1 of 141 files; the
corpus texts + submission + manifest are intact). The runner's real preflight (`verify_ruaa_inventory`)
fail-closes on it until the inventory is re-frozen. The frozen private data is not edited here.

## Claim boundary

`claim_status = exploratory_internal`. RuAA is a nested secondary sensitivity panel, never a headline
selector. This is not external replication and not a proof of any disputed authorship. SOCIOLIT-lite is
excluded. A noninferiority result is not superiority; a gate p-value is not a probability of authorship.
The historical `0.8805`, the frozen baseline snapshot, and the README/site/PAPER headline artifacts are
untouched.

## Remaining gated work

Real audit-corpus preparation, fold-manifest freezing, clean-tree preflight, the confirmatory run with
the real injected estimator, the independent result audit, and the separately preregistered headline
decision (protocol §11–14) remain. None begins before an independent review of the exact final commit
signs off and a separate execution authorization is given.
