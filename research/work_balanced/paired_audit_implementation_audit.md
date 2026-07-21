# Paired-audit control-plane implementation audit

Status: control plane implemented, per-gate independently reviewed, and committed on branch
`paired-audit-control-plane`. No real audit corpus, no fold manifest, and no confirmatory cell has
been prepared or run. Execution stays gated behind an independent code audit and a separate execution
authorization.

Normative source: [`paired_audit_protocol.md`](paired_audit_protocol.md) (v3.1). On any conflict the
protocol governs.

## Requirement → module → test trace

| Protocol | Requirement | Module | Tests |
|---|---|---|---|
| §1.2 | Loader-agnostic semantic-row digest over `(texts,y,groups,authors)` only; frozen legacy anchor `b4886a7c`; exact row equality | `paired_audit/semantic_parity.py` | `test_paired_audit_corpus::TestSemanticParity`, `TestSemanticRowDigest` |
| §1.3 | Whole immutable audit-corpus root `data/audit_corpus/<digest>/`; atomic publication; reverify-before-pointer; per-chunk byte/filename equality; conflict/partial fatal; live `frags_train` never mutated | `paired_audit/corpus.py` | `test_paired_audit_corpus::TestBuilder`, `TestBuilderFailClosed` |
| §1.4 | Audit-only dataset verifier (`dataset_contract=work_balanced_manifest` ⊥ estimator axes) with self-consistency recompute | `paired_audit/corpus.verify_audit_dataset` | `test_paired_audit_corpus::test_audit_only_verifier_*` |
| §1.5 | `derive_work_subset` whole-work exact-set RuAA panel with three-digest binding | `paired_audit/work_subset.py` | `test_paired_audit_corpus::TestWorkSubset` |
| §1.6/§12 | LOBO/RuAA fold-manifest fields + self-hash; runner rebuilds and requires exact equality (never self-signs); frozen 47/255/43/251 and 137/22 universes; RuAA selection-digest bound at build | `paired_audit/manifest.py` | `test_paired_audit_manifest` |
| §3.2 | Pinned A0 reference SHA256 before any parse (`lobo_books.txt`, RuAA reference submission + frozen SHA256SUMS) | `paired_audit/references.py` | `test_paired_audit_references` |
| §2.4/§3.4 | Exactly 21 applied cells / 15 A0 comparisons; per-model decomposition pinned; typed signals carry no metrics; applicability digest | `paired_audit/applicability.py` | `test_paired_audit_control_plane::TestApplicability` |
| §3.3 | Two-sided null-centered author-cluster bootstrap p, B=10000, seed=42, +1 correction, degenerate order; McNemar diagnostic-only | `paired_audit/inference.py` | `test_paired_audit_inference::TestClusterPValue`, `TestMcNemarDiagnostic` |
| §3.4 | Holm–Bonferroni on unrounded p over the fixed 15-member family; `m` never reduced; `significant := holm_p < 0.05` | `paired_audit/inference.py` | `test_paired_audit_inference::TestHolm`, `TestHolmRegisteredFamily` |
| §3.5 | Single headline endpoint stylo A4−A0; author-clustered percentile CI (10000/seed42/[2.5,97.5]); δ=0.02; relabel/keep_legacy/inconclusive on unrounded bounds; boundary equality → inconclusive; macro-F1 CI withdrawn | `paired_audit/headline.py` | `test_paired_audit_headline` |
| §4.2 | Canonical RunPlan → run_id binding every identity input; runtime/env/BLAS fingerprints omit OS/kernel/platform strings; fail-closed bindings | `paired_audit/run_plan.py` | `test_paired_audit_control_plane::TestRunPlan` |
| §4.3/§7 | Per-fold immutable checkpoints keyed `(dataset,model,cell,fold)`; atomic no-overwrite; self-hash + binding verify; resume skip/pending/fatal; COMPLETE only when all folds present | `paired_audit/checkpoints.py` | `test_paired_audit_checkpoints` |
| §4.4/§8 | Path-guarded transient store; content-addressed archive + SHA256SUMS; immutable version dir + atomic current/COMPLETE; summary self-hash; never writes headline paths | `paired_audit/publisher.py` | `test_paired_audit_publisher` |
| §9 | Full fail-closed catalog | all above + gap-fills | `test_paired_audit_fail_closed_sweep` (coverage manifest) |

## Verification results

- Diff vs `release` HEAD: 19 files, additions only (11 modules + 8 test files); no modified or deleted
  tracked file — the control plane does not import the uncommitted working-tree rework and each gate
  commits only its own new files.
- Focused, full working-tree test suite, and the release-integrity + provenance + work-balanced infra
  battery: all green. A0/A4 golden replay passes via `tests/test_work_balanced_ablation_goldens.py`;
  the heavy capture-env live replay is opt-in (`WORK_BALANCED_LIVE_GOLDEN_REPLAY=1`) and skips by
  default; production-default invariants pass via `tests/test_work_balanced_ablation_config.py`.
- `python -m py_compile` clean on every module and test. `ruff` is absent from the environment and CI
  (no `[tool.ruff]`); lint relies on compilation and the test suite.

## Per-gate independent review

Each gate was reviewed by an independent adversarial agent (fresh context, code + negative probes,
not a summary read). Findings were addressed with fixes and regression tests: corpus 9.5; applicability
+ run_plan 7/7 hardened; inference 9.5 (bit-exact reference); headline 9.5; checkpoints 9 hardened;
publisher 8 hardened (path normalization); manifest 8.5 hardened (class-order contents).

A final four-lens whole-package code audit (integration/security/statistics/completeness) then ran.
Integration/security/statistics signed off (8/9/9); the completeness lens (7, withheld) surfaced real
primitive gaps, all now closed: the §3.2 pinned A0-reference verifier (`references.py`), the §4.1
applied-cell evidence schema in `assert_cell_record`, a single shared `class_order_digest` producer
wired into the checkpoint bindings, the RuAA selection-digest bound at manifest build, confirmatory
`continuous_tolerances` + clean-tree (`git_dirty`) enforcement, the extended headline denylist plus an
`assert_archive_committable` durability guard, 0/1 correctness guards on both bootstraps, and orphan
audit-root cleanup on a failed re-verify. The `.gitignore` whitelist for the committed per-work archive
subtree is a repository-config action the runner performs at real publication (it is not edited here
because `.gitignore` is part of the separate working-tree rework); `assert_archive_committable` fails
closed until it exists.

## Claim boundary

`claim_status = exploratory_internal`. RuAA is a nested secondary sensitivity panel, never a headline
selector. This is not external replication and not a proof of any disputed authorship. SOCIOLIT-lite is
excluded. A noninferiority result is not superiority; a gate p-value is not a probability of authorship.
The historical `0.8805`, the frozen baseline snapshot, and the README/site/PAPER headline artifacts are
untouched.

## Remaining gated work

Real audit-corpus preparation, fold-manifest freezing, clean-tree preflight, the confirmatory run, the
independent result audit, and the separately preregistered headline decision (protocol §11–14) remain.
None begins before this independent code audit is signed off and a separate execution authorization is
given.
