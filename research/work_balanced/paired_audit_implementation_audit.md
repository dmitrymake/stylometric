# Paired-audit control-plane implementation audit

Status: control plane + confirmatory runner + independent result-auditor implemented, Gate-10
remediation **round 2** applied (R2.1–R2.9), and committed on branch `paired-audit-control-plane` as
additions only. No real audit corpus, no fold manifest, and no confirmatory cell has been prepared or
run on real data. Execution stays gated behind an independent code review of the exact final commit and
a separate execution authorization (protocol §11–14).

Normative source: [`paired_audit_protocol.md`](paired_audit_protocol.md) (v3.1). On any conflict the
protocol governs. This document is a declarative implementation record, not a sign-off; the sign-off
lives in [`paired_audit_review_provenance.md`](paired_audit_review_provenance.md).

## Package inventory (14 modules, 10 test files — all committed additions)

`src/stylo/eval/paired_audit/`: `semantic_parity`, `corpus`, `work_subset`, `applicability`,
`run_plan`, `inference`, `headline`, `checkpoints`, `publisher`, `manifest`, `references`,
`result_audit`, `runner`, `__init__`. `tests/test_paired_audit_{corpus, control_plane, inference,
headline, checkpoints, publisher, manifest, fail_closed_sweep, references, runner}.py`.

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

## Gate-10 remediation round 2 (requirement → code → adversarial test)

An owner-authored independent probe found the round-1 remediation was still a primitive library with
13 substantive blockers. Round 2 closes each as a separate commit (no amend). The OS kernel is entirely
out of scope and is not bound anywhere.

| # | Requirement | Code | Adversarial test |
|---|---|---|---|
| R2.1 | Exact one-to-one A0 contract (work_id + true author + **pred** + correct + rank; no basename; no duplicate/missing/extra); RuAA A0 vs `reference_submission_stylo.csv` | `references.{index_from_records,build_lobo_reference_index,build_ruaa_reference_index,assert_a0_matches_index}`, `runner._a0_index/_assert_a0_matches_reference` | all-preds-wrong-but-correct/rank-match; duplicate basename kept distinct; genuine duplicate work id; missing/extra row; row permutation; RuAA pred mismatch |
| R2.2 | Confirmatory evaluator identity in run_id (registered name, source digest, import identity, estimator config, mechanism passport); no bare callable | `run_plan.{EvaluatorSpec,evaluator_identity,REGISTERED_CONFIRMATORY_EVALUATORS}` + `build_run_plan` binding | source recompute + config re-keying; bare callable rejected; unregistered name fatal only confirmatory; empty config/passport; identity folds into run_id |
| R2.3 | Real fold-local evidence persisted + aggregated (no synthesis); missing required axis/passport digest fatal | `applicability.required_evidence_digests`, `runner._assert_fold_evidence/_aggregate_evidence` | fold-evidence required+hex+stack passport; aggregate propagation (changed fold digest → changed artifact, missing fatal); e2e artifact == aggregate of real checkpoints |
| R2.4 | Live DISK re-attestation before/after each fold+cell (code/config/env/corpus/manifest), not in-memory self-compare | `corpus.verify_corpus_manifest_light`, `runner._reattest` | config / corpus-chain / manifest drift and a physical on-disk manifest tamper each fail closed |
| R2.5 | Non-tautological manifest binding (registered algorithm/seed + actual disk digests, not copied); child+parent+selection bound; work_id↔dataset labels | `manifest.{REGISTERED_ALGORITHM,assert_manifest_consistent_with_dataset}` + child `dataset_digest`, `checkpoints.dataset_bindings(parent)`, `runner._rebuild_manifest` | forged algorithm/parent caught by rebuild; tampered manifest author label / disagreeing dataset labels |
| R2.6 | Strict checkpoint/result schema (pred/true range, correct/rank/argmax coherence, normalized proba, non-empty evidence); richer per_work | `checkpoints._validate_result`, `runner` authoritative true_label + `_per_work`, `applicability._validate_applied_record` | negative/out-of-range rank, out-of-range labels, non-normalized / out-of-[0,1] proba, pred≠argmax, correct/label + correct/rank inconsistency, empty evidence |
| R2.7 | Artifact completeness (embedded run_plan, both class orders, universes/digests, tolerances, full attestation) + golden inventory FROM DISK + all registered tolerance quantities | `run_plan.REGISTERED_TOLERANCE_QUANTITIES`, `references.golden_fixture_inventory`, `runner` run_plan/universes, `publisher` run_id recompute | golden inventory deterministic/re-keying/missing-fatal; tolerance exact-set; run_id must recompute; missing completeness sections; round-trip recomputes run_id |
| R2.8 | Independent result-auditor recomputes accuracy/F1/top2/recall/Δ/CI/cluster-p/Holm/headline from vectors; publisher accepts only PASS; headline decision a separate stage; smoke/dry never write the committed artifact | `result_audit.audit_results`, `runner` (candidate→audit→decide→publish), `publisher.{verify_final_assembly,publish_audit,load_published_audit}` | auditor rejects a tampered accuracy / cluster p; Holm↔cell + headline↔CI consistency; smoke writes no committed artifact + transient round-trip; root↔version equality |
| R2.9 | Path guard BEFORE any mkdir/write + re-check; loader detects a swapped committed root summary | `publisher.write_transient` (guard→mkdir→re-check), `publisher.load_published_audit` root↔version | swapped committed root summary detected; write_transient rejects a symlinked run namespace before mkdir |

## Gate-10 remediation round 3 (owner-reproduced bypasses → code → adversarial test)

A second owner-authored independent audit against the round-2 sign-off SHA `1a56e57d` REPRODUCED
several critical bypasses that the round-2 reviewers missed. Round 3 closes each (the OS kernel stays
out of scope; §11 stays hard-stopped).

| # | Reproduced bypass | Code | Adversarial test |
|---|---|---|---|
| R3.1 | Publisher accepted `proba=[-5,6]` (the auditor dropped probabilities); resume trusted a re-self-hashed invalid checkpoint (`pred=-9,true=99,rank=-5,proba=[-7,8]`, empty evidence) | `result_audit._validate_fold_coherence` (revalidates every per-work proba/rank/label from scratch), `checkpoints._load_path` (runs `_validate_result` + evidence + proba_digest on LOAD) | out-of-range / non-normalized proba rejected at publish; a re-self-hashed invalid checkpoint rejected on resume |
| R3.2 | A 2-author/4-work fixture accepted as production; a full `true_label` permutation contradicting the work_id author passed | `result_audit._assert_frozen_universe` (47/43/251, 22/22/137) + `_validate_fold_coherence` asserts `true_label == prob_order.index(author)` | a toy universe rejected for confirmatory; a permuted true_label rejected |
| R3.3 | Any hex64 accepted as `proba_digest`; the Delta/stack passports did not match the literal §4.1 schema | `checkpoints.proba_digest` (authoritative, runner-computed + enforced at save/load), `result_audit._cell_proba_digest` (recomputes from the real vectors); `applicability` `delta_mean_std_centroid_digest` + `calibration_passport` full structure | a fake proba_digest rejected; the calibration_passport must be a full structure not a digest |
| R3.4 | `golden_fixture_inventory_sha` hashed lobo_books/RuAA submission, not the external A0/A4 goldens; no live replay | `references.verify_b4_goldens` (pinned SHA of `b4_goldens_v1.json` + structural panel replay), `runner` binds it | the b4 fixture is pinned + panels replayed; a tampered/missing fixture fatal |
| R3.5 | Fold re-attestation checked only `corpus_manifest.json`, not the data tree; manifests compared in-memory; dataset arrays never re-checked | `runner._reattest` (FULL `verify_published_corpus` tree re-hash + `verify_audit_dataset` array re-verify + manifest re-derived from the disk dataset, every fold and cell) | fails closed on a dataset-digest drift + a manifest-digest drift + a physical tamper |
| R3.6 | Assembler + auditor used the SAME headline/inference/metrics; one call auto-ran execution→audit→headline→publish | `result_audit` re-implements every verdict quantity independently (no shared import); `runner.{run_execution,run_result_audit,decide_headline_stage,publish_stage}` durable stages, confirmatory refuses the all-in-one driver | the independent impl agrees with the shared one; execution hard-stops with a durable candidate; the headline needs its own authorization token |

## Verification results (mandatory checks)

- **Diff vs `release` HEAD `2f6c3dc3`:** additions only (paired_audit modules + tests + this audit pair);
  zero modified/deleted tracked file — the control plane depends only on committed HEAD APIs and never
  imports the uncommitted working-tree rework. The owner's uncommitted rework is untouched.
- **A — focused, dirty working tree** (`pytest tests/test_paired_audit_*.py`): 214 passed.
- **C — clean committed-snapshot** (`git archive HEAD | pytest`): 209 passed, 5 skipped (4 runner-e2e
  need a live `.git` for the commit binding; 1 RuAA-reference needs the gitignored private data) —
  self-contained, no rework dependency.
- **Full clean `git clone` suite** (`pytest tests/`): 4 failed, 786 passed, 6 skipped. All 4 failures
  are pre-existing packaging debt (`test_ci_sign_erratum.py`, `test_macro_f1_ci_withdrawal.py` — both
  present on `release`) caused by `scripts/gen-paper.mjs` not being tracked in git; **zero** are
  paired-audit tests. No paired-audit regression.
- `python -m py_compile` clean on every module and test. `ruff` is absent from the environment and CI.

## Publish-boundary evidence note (accepted, mitigated)

The per-cell `evidence.*_digest` values are the runner's deterministic aggregate of the immutable
per-fold checkpoints (`_aggregate_evidence`). The publisher validates them as hex64 with the exact
required axis/passport keys but cannot RECOMPUTE them at the publish boundary (the per-fold checkpoints
are not present there). They are mitigated by being atomically-immutable, create-without-overwrite
per-fold checkpoints that are re-verified at COMPLETE, aggregated deterministically, and bound into the
published summary `self_hash`. This is the one published field that is checkpoint-derived rather than
recomputed/run_id-bound; it is diagnostic evidence, not a metric/p-value/Holm/headline number, so it
cannot alter any verdict. A publish-time recompute would require carrying the immutable checkpoints
into the publisher, which is a §11-execution wiring decision, not a control-plane one.

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
