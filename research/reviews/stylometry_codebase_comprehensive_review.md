# Comprehensive stylometry codebase review

**Audit date:** 2026-07-26 (Europe/Moscow)
**Reviewed HEAD:** `80da05df5fe909f55d94aed07943f394cfaca1b6` on `paired-audit-control-plane`
**Primary review object:** current dirty working tree, explicitly separated from committed HEAD and repair bundle.
**Inventory:** `stylometry_codebase_inventory.json`; current source/test/config/doc inventory SHA-256 `9d7292516a671ad007ad54bd69c8cddd22aa0c253d33957a55d4f8571aefe1c4`.

> Review-first constraint observed: no production source, tests, manifests, data, frozen artifacts, configuration, documentation, checkpoints or confirmatory state were changed. Only this report and its JSON inventory are audit outputs.

## 1. Executive verdict

**NO-GO** for merge, full exploratory LOBO, confirmatory execution, and journal/repository release in the reviewed state. The codebase contains valuable fail-closed work and unusually strong paired-audit validation primitives, but the scientific system boundary is not sealed.

The catalog contains **6 CRITICAL, 24 HIGH, 27 MEDIUM and 4 LOW** findings. The decisive blockers are:

- actual near-complete cross-work content leakage in the frozen LOBO/RuAA corpus (`AUD-001`);
- silent raw→clean corpus substitution and destructive/partial chunk publication (`AUD-003`, `AUD-020`, `AUD-021`);
- executable deserialization (`AUD-002`);
- direct leakage/escrow gaps in the public blind benchmark (`AUD-005`, `AUD-054`);
- current executable source is not clean-reconstructible (`AUD-009`);
- confirmatory stack cells are infeasible and rank/evaluator contracts disagree (`AUD-006`–`AUD-008`);
- historical case confidence can be circular, iid-chunk inflated, or applied out of domain (`AUD-004`, `AUD-025`, `AUD-026`).

The repair itself has important positive properties: repaired `stylo_stack` fails closed with `StackClassCoverageError`; no sentinel remains in that stack path; `stylo_equal_channels_v1` is explicit-only, absent from defaults and the confirmatory matrix, and its passport matches the actual six-channel identity-softmax/equal-mean mathematics for the current single prediction call. The 5/5 smoke is correctly scoped as a five-fold diagnostic, not full accuracy.

A **bounded refactor is required before full exploratory LOBO**: seal/migrate corpus identity, close the executable source set, make validate/clean/split fail-closed and atomic, and add live source/checkpoint integrity. Broad package reorganization and the estimator lifecycle rewrite should wait until after a corrected exploratory result because they carry numerical-parity risk.

## 2. Exact reviewed scope

| State | How inspected | What conclusions apply |
|---|---|---|
| Committed HEAD | `80da05df5fe909f55d94aed07943f394cfaca1b6`; git objects and per-file SHA inventory | Committed architecture/source only; not credited with untracked repair/equal code. |
| Current working tree | Tracked modifications/deletions plus untracked source and relevant ignored package/artifacts | Primary merge/rework assessment and current imports/tests. |
| Repair snapshot | `07a8df82…` from verified complete bundle, temporary bare repository, no checkout | Static repair math/source comparison only; smoke runner is absent from bundle. |
| Release/base | `2f6c3dc3…` git object | Historical structural/scientific persistence comparison. |
| Ignored runtime/evidence | Count+path-list digest; relevant source/docs/artifacts hashed and selectively validated | Never treated as committed/reconstructible source; evidence applies only to exact bound bytes. |

| Inventory class | Count | LF path-list SHA-256 |
|---|---|---|
| tracked | 643 | 7de775680b83cafd12588059003c9c9874df6b245cb0337e03215750bf4d674d |
| modified (includes deleted) | 70 | 6ec7644706f5a4139bcf3edb4ce596bac7d046c6604e24e3ae99ff865510537e |
| deleted | 30 | a1b976b01c4987aea15aab15fe9b2cacad91c510483f3d18c87a9fc0e72ce752 |
| untracked files | 48 | 84f66b21406f45bc8e72f47eda28349975bde3c05a7bf938e11b497881a8808c |
| ignored paths | 166972 | 3c06eeb1c26448e49204f7e67c83a34de4840150d7354ca08899da70288a638f |

- Branch: `paired-audit-control-plane`; remote: `origin	git@github.com-dmitrymake:dmitrymake/stylometric.git (fetch)`.
- Base→HEAD: 26 files changed, 7265 insertions(+). HEAD→repair: 61 files changed, 9797 insertions(+), 248 deletions(-).
- No clean repair checkout exists; only the dirty worktree is registered. Repair was never materialized or executed.
- Current considered source/test/config/doc set: 1423 files, 19,668,557 bytes.
- The full ignored list is deliberately not expanded into 166k JSON strings; count+defined LF-list digest freezes it, while all relevant ignored source/doc/scientific paths are enumerated.
- Current-vs-repair conclusions are never inferred from smoke metrics: artifact source hashes match repair bytes and differ from current dirty sources.

## 3. Current architecture map

```text
stylo CLI (pyproject: stylo = stylo.cli:main)
├── pipeline/                  mutable application I/O
│   ├── clean.py              raw → cleaned text
│   ├── split.py              cleaned text → chunks/manifests
│   ├── train.py              deployment fitting/publication
│   └── predict.py            deployment scoring/report
├── corpus.py + workdoc.py    dataset/work identity and loaders
│          ↕
│   eval/provenance.py        row identity, verification and output I/O
├── features/                 channel representations, NLP/Rep caches
├── models/                   Delta, LR, stack, work-balanced/equal fusion
├── eval/
│   ├── lobo/groupkfold       generic exploratory evaluation
│   ├── final/sweep           comparison orchestration
│   ├── stylo_lobo_validation.py*   untracked true-LOBO runner core
│   ├── work_balanced_ablation_screen.py*  untracked screening core
│   └── paired_audit/         confirmatory control-plane contracts/I/O
├── benchmarks/               public blind benchmark schema/scoring
├── cases/                    historical case framework and CLI
├── corpus_tools/             acquisition and report-only validation
└── report/                   legacy HTML assembly

scripts/
├── evaluation/*              exploratory/evidence entrypoints (* = untracked)
├── artifacts/*               golden capture
├── experimental/*            untracked replacement of deleted experemental/
└── train.py/predict.py/...    competing legacy application stack

tests/                        59 modules / 872 collected tests
research/                     protocols, current design, local history
docs/                         public/generated/frozen/ignored evidence mixed together
site/                         generated public presentation
```

Public entrypoints are the installed `stylo` CLI, `run.sh`, importable `stylo.benchmarks`/case APIs, and many direct scripts. The intended dependency direction is application/CLI → evaluation/models/features/domain → artifact I/O. Actual violations are the `workdoc ↔ corpus ↔ eval.provenance` SCC, model/evaluation private imports, two competing train/predict stacks, and scientific source embedded in scripts/ignored package files.

| Responsibility | Current owner | Assessment |
|---|---|---|
| Domain/work identity | corpus.py, workdoc.py, eval/provenance.py | Split ownership and import cycle; cross-work content identity absent. |
| Features/channels | features/*, nlp.py, features/reps.py | Generally cohesive numerical blocks; runtime/cache trust identity weak. |
| Estimators | models/* plus eval/calibration.py | Routing scattered; stack/equal share private lazy mechanics. |
| Exploratory orchestration | eval/lobo.py, final.py, sweep.py, two untracked 600–1476 LOC cores | Duplicate identity/checkpoint/report responsibilities. |
| Confirmatory control | eval/paired_audit/* | Strong validators/publisher/inference; evaluator pin/rank/matrix/corpus blockers remain. |
| Mutable application I/O | pipeline/* and root scripts | Competing artifact layouts; clean/split not transactional. |
| Blind benchmark | benchmarks/* | Well-structured package, but split/escrow/result bindings fail open. |
| Historical cases | cases/framework.py | God module; role/uncertainty/applicability contracts invalid. |

AST import graph: 95 local modules, 584 local edges, SCC `[['stylo.corpus', 'stylo.eval.provenance', 'stylo.workdoc']]`. Current filesystem has no unresolved local target only because untracked/ignored source is present; that is not clean source closure.

## 4. Lens scores (0–10)

| Lens | Score | Reason |
|---|---|---|
| Architecture | 4.0 | Useful layers exist, but cycles, duplicated control planes and competing application stacks obscure ownership. |
| Structure and naming | 4.5 | Many responsibility names are good; final/segment/legacy/experemental and mixed normative/history trees are misleading. |
| File size and complexity | 5.0 | No production class >400 LOC, but several >900 LOC modules and >120 LOC functions own unrelated seams. |
| Python correctness | 4.5 | Strong strict JSON/paired validators coexist with silent fallbacks, unsafe loads, weak general contracts and races. |
| Scientific validity | 2.5 | Actual outer-boundary content leakage and invalid legacy scientific claims block publication despite sound train-only vectorizer mechanics. |
| Tests | 5.5 | 872 tests and strong adversarial paired coverage; suite is red and major corpus/benchmark/case boundaries lack tests. |
| Documentation | 4.0 | Extensive evidence and caveats, but normative status and code references contradict. |
| Repository hygiene/packaging | 3.0 | Current source is untracked/ignored/history-dependent; wheel and clean clone are not self-contained. |
| Performance/operations | 4.0 | True-LOBO telemetry/checkpoints are promising; generic resume, live attestation, memory/concurrency/durability remain weak. |
| Security/integrity | 3.0 | Paired symlink/path publication is comparatively strong, but executable serialization and corpus/escrow substitution are critical. |

Overall publication-tool readiness is approximately **3.8/10**. This is not an average-accuracy judgment; it is a systems/scientific assurance judgment.

## 5. Findings, ordered by severity

### CRITICAL

#### AUD-001 — Frozen corpus contains nested works across the LOBO boundary

- **Severity / confidence:** CRITICAL / 0.999.
- **State:** current_dirty_working_tree, ignored_frozen_corpus_and_manifests.
- **Exact locations:** `src/stylo/eval/lobo.py:270-284; src/stylo/eval/stylo_lobo_validation.py:372-385; data/audit_corpus/15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029/frags/turgenev/хорь_и_калиныч/manifest.json:20; data/audit_corpus/15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029/frags/turgenev/записки_охотника/manifest.json:430; data/paired_audit_preparation/f3566c0e7308fd84f58e5f72b5651dd584908a80fc6ca95ac027bc6881f0140a/lobo_fold_manifest_v1.json:1466-1529`.
- **Evidence:** Бирюк, Певцы and Хорь и Калиныч have word-5gram containment 95.72%, 97.77%, and 97.24% in Записки охотника; one chunk is byte-identical (SHA-256 12549537…). All four works are tested separately in frozen LOBO and RuAA manifests.
- **Reproduction/static trace:** The fold mask removes only groups == held-out work_id. Therefore each standalone fold trains on almost the same text inside the collection; the collection fold trains on the standalone stories. Independent word- and char-shingle scans reproduced the containment.
- **Consequence:** The independence assumption is false and corpus-derived accuracy/headline values can be inflated. The historical 0.8805 value is not a clean external-accuracy estimate until a corrected corpus protocol is rerun.
- **Scope:** Frozen audit corpus, legacy full benchmark, planned exploratory and confirmatory LOBO/RuAA; not the five-fold repair smoke.
- **Minimum correction:** Block new accuracy/headline claims and mark the affected manifests ineligible for new runs.
- **Preferred architectural correction:** Create a registered corpus/fold migration with a cross-work exact+asymmetric-containment graph; adjudicate collection versus constituents or treat their component as one exclusion unit. Preserve old artifacts as historical evidence.
- **Regression test:** Reject exact cross-work chunks and short-in-long containment; assert every train/test fold has disjoint content components; allow repetitions only within one registered work.
- **Cost / change risk:** L/XL; High scientific and compatibility risk; requires owner/editorial decision and reruns.

#### AUD-002 — Untrusted pickle/joblib/NumPy object deserialization can execute code

- **Severity / confidence:** CRITICAL / 0.995.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/features/reps.py:114-125; src/stylo/pipeline/predict.py:43-49; src/stylo/eval/certificates.py:337; scripts/run_benchmark.py:76`.
- **Evidence:** Configured cache and deployment paths are loaded with pickle.load/joblib.load/allow_pickle=True before any external digest or regular-file trust check. Catching exceptions occurs after pickle opcodes have executed.
- **Reproduction/static trace:** Place a crafted pickle at the predictable configured reps/model path; invoking cache load or legacy predict executes its reducer before schema validation. The review did not execute a payload.
- **Consequence:** Arbitrary code execution under the runner account; a valid-looking substituted cache can also silently change features and scientific results.
- **Scope:** Production representation cache, legacy model deployment, ignored certificate and legacy benchmark utilities.
- **Minimum correction:** Document and enforce trusted-local-only inputs; reject symlinks and verify a pre-registered digest before loader invocation.
- **Preferred architectural correction:** Use a non-executable, schema-validated representation format and one content-addressed model bundle with externally trusted digest/signature policy.
- **Regression test:** Monkeypatch deserializers to prove trust validation precedes load; reject symlink/substitution/hash mismatch using inert fixtures.
- **Cost / change risk:** L; High migration risk for existing caches/models; cache rebuild required.

#### AUD-003 — Cleaning can retain stale corpus bytes and silently substitute the modeled source

- **Severity / confidence:** CRITICAL / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/pipeline/clean.py:102-117; src/stylo/pipeline/clean.py:120-142`.
- **Evidence:** Per-file failures return 0, the command never stages or clears input_clean, and success count is only logged. Deleted raw inputs and failed replacements leave old cleaned files in place.
- **Reproduction/static trace:** Clean a work, remove or change its raw file, then rerun; the prior input_clean work survives. Inject normalize failure and the command still returns success with old bytes.
- **Consequence:** The modeled corpus can differ silently from the current raw source while appearing successfully rebuilt, enabling corpus substitution and false reproducibility/headlines.
- **Scope:** Raw-to-clean corpus build used by run.sh and subsequent split/train/evaluation.
- **Minimum correction:** Make any expected-file failure fatal and verify exact input/output bijection before success.
- **Preferred architectural correction:** Build an immutable, content-addressed cleaned snapshot in staging with raw-source manifest/digests and atomically switch one pointer only after complete validation.
- **Regression test:** Deletion removes stale output; injected failure leaves the prior complete snapshot current; exact source/output inventory and digests match.
- **Cost / change risk:** L; High corpus migration/operational risk; do not clean in place.

#### AUD-004 — Case framework binds roles by path, not content, permitting target-to-train leakage

- **Severity / confidence:** CRITICAL / 0.97.
- **State:** release_base, committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/cases/framework.py:592-631; src/stylo/cases/framework.py:690-710; src/stylo/cases/framework.py:257-302; src/stylo/cases/framework.py:1326-1333`.
- **Evidence:** Containment checks compare resolved paths only. Inputs use errors='ignore'; passports store mutable paths/user provenance without computed byte inventory. Current 53 case specs had no exact target/candidate duplicate, but the API is fail-open.
- **Reproduction/static trace:** Copy target bytes to a differently named candidate file with a legal path. Path validation passes, target features enter closed-set training, and the framework can issue strong/high.
- **Consequence:** A circular historical attribution can be presented as strong; mutations and UTF-8 data loss are not detectable from the passport.
- **Scope:** Case-study framework and existing/future passports; no claim that current passports contain this exact duplication.
- **Minimum correction:** Strict UTF-8; SHA-256 every input; reject target/candidate exact and registered normalized duplicates; bind an inventory digest.
- **Preferred architectural correction:** Introduce a versioned CaseInputManifest/CorpusIdentity with immutable role, byte, normalized-content, and source-provenance bindings.
- **Regression test:** Different-path copy, symlink/hardlink, normalized containment, invalid UTF-8 and post-run mutation all fail closed.
- **Cost / change risk:** M/L; High passport schema and historical-compatibility risk.

#### AUD-005 — Blind benchmark schema does not enforce train/test content and identity isolation

- **Severity / confidence:** CRITICAL / 0.97.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/benchmarks/schema.py:141-216; src/stylo/benchmarks/validator.py:94-114; src/stylo/benchmarks/validator.py:239-245; src/stylo/benchmarks/artifacts.py:112-165; research/protocol_v1.yaml:24-31`.
- **Evidence:** Blind rows may retain source.sha256, text_path, work, edition/period/topic/register. Validation forbids labels/spans only and never rejects duplicate source bytes/path/revision or a work crossing train to held-out/blind.
- **Reproduction/static trace:** Give an opaque blind doc_id the train row's source.sha256, text_path and work while omitting labels. Manifest and artifact validation pass and the same bytes can be read on both sides.
- **Consequence:** Direct test leakage and identity disclosure can produce a false benchmark headline.
- **Scope:** SPOOF-RU/IDIOSHIFT-RU public benchmark boundary; unrelated to repair smoke/LOBO.
- **Minimum correction:** Reject duplicate content/path/revision and work/group crossing incompatible split roles; forbid identity-bearing blind metadata.
- **Preferred architectural correction:** Separate public-redacted and custodian full-provenance manifests, bind them by a versioned mapping/digests, and enforce split-isolation centrally.
- **Regression test:** Exact-byte, same-work, same-path and same-revision train↔blind pairs fail; legal development-only multi-edition grouping passes.
- **Cost / change risk:** M; Medium-high public schema compatibility risk; requires versioned migration.

#### AUD-054 — Blind scorer does not bind truth bytes to a pre-scoring escrow commitment

- **Severity / confidence:** CRITICAL / 0.97.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `research/protocol_v1.yaml:15-25; research/README.md:28-32; src/stylo/benchmarks/scoring.py:55-64; src/stylo/benchmarks/scoring.py:273-320; src/stylo/benchmarks/scoring.py:480-500`.
- **Evidence:** Truth carries only manifest_sha256. Loader/scorer never require an independently published SHA-256/signature of exact truth bytes, despite protocol escrow language.
- **Reproduction/static trace:** After submissions are fixed, change labels, spans or evidence in truth.json while retaining identity/blind IDs/manifest_sha256; scorer accepts it and emits different metrics.
- **Consequence:** Custodian error or malicious substitution can change a blind external-test headline after submissions.
- **Scope:** Blind benchmark escrow/scoring; independent of train/blind split isolation and dual manifest source findings.
- **Minimum correction:** Require expected independently published truth SHA-256 before parse/scoring and reject missing commitment in scientific blind mode.
- **Preferred architectural correction:** Versioned signed escrow record binding redacted/full manifest, truth, protocol, commit and timestamp; audit output carries all bindings.
- **Regression test:** Any committed truth byte mutation fails; exact truth passes; only clearly named synthetic integration bypass may omit escrow.
- **Cost / change risk:** M; Medium scoring/escrow schema compatibility; version rather than mutate frozen identifiers.

### HIGH

#### AUD-006 — Fail-closed repaired stack makes the frozen confirmatory matrix infeasible

- **Severity / confidence:** HIGH / 0.995.
- **State:** current_dirty_working_tree, repair_bundle, ignored_frozen_protocol.
- **Exact locations:** `src/stylo/models/stacked_clf.py:319-398; src/stylo/eval/stylo_lobo_validation.py:63-67; research/work_balanced/paired_audit_protocol.md:110-121; src/stylo/eval/paired_audit/applicability.py:69-127`.
- **Evidence:** Four train-only singleton classes make class-complete inner group CV impossible: when a singleton work is validation, its class is absent from inner training. Stack cells remain mandatory in A0-A4/Holm.
- **Reproduction/static trace:** The class-coverage preflight theorem applies to every outer stack fold; the repair smoke demonstrates StackClassCoverageError rather than a sentinel score.
- **Consequence:** The safe stack correctly fails, but the registered confirmatory family cannot finish.
- **Scope:** Legacy stylo_stack cells in frozen paired-audit matrix; equal channels cannot be silently substituted.
- **Minimum correction:** Run a manifest-level feasibility preflight and block scheduling before model fitting.
- **Preferred architectural correction:** Owner-approved versioned protocol/matrix amendment or explicit withdrawal of infeasible stack cells, preserving old registration.
- **Regression test:** Frozen manifest preflight proves infeasibility and no checkpoint is created; feasible synthetic matrix passes.
- **Cost / change risk:** M/L; High protocol compatibility risk.

#### AUD-007 — Prediction and rank tie semantics disagree across evaluation/control paths

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree, repair_bundle.
- **Exact locations:** `src/stylo/eval/lobo.py:287-290; src/stylo/eval/groupkfold.py:94-96; src/stylo/eval/paired_audit/checkpoints.py:107-115; src/stylo/eval/paired_audit/result_audit.py:89-102`.
- **Evidence:** Core paths use stable smallest-index argmax and worst-tie rank sum(p>=p_true); paired control uses any tied argmax and best rank 1+sum(p>p_true).
- **Reproduction/static trace:** Uniform [0.5,0.5] with true/pred class 1 passes paired as correct rank 1, but stable argmax is class 0 and conservative rank is 2. Majority one-hot nonmajority rank is 47 in core and 2 in paired.
- **Consequence:** Checkpoints/auditor can accept false correctness, or repair evaluator/checkpoint validation can deterministically disagree.
- **Scope:** Generic LOBO, group-fold, paired checkpoints/result audit, majority baseline.
- **Minimum correction:** Choose one explicit versioned tie/rank contract and use it in all validators/assemblers.
- **Preferred architectural correction:** Create shared PredictionContract with class order, stable top1, rank definition and compatibility reader for historical artifacts.
- **Regression test:** All-zero, tied-top, majority and three-class ties yield identical results through every path.
- **Cost / change risk:** S/M; High artifact/schema migration risk despite small code diff.

#### AUD-008 — Confirmatory evaluator allowlist authenticates a label, not canonical code

- **Severity / confidence:** HIGH / 0.97.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/paired_audit/run_plan.py:84-141; src/stylo/eval/paired_audit/runner.py:156-191; tests/test_paired_audit_control_plane.py:350-370`.
- **Evidence:** Any callable named work_balanced_ablation_factory passes. Source/config/passport hashes are self-declared; test deliberately blesses _fake_factory. No canonical production callable with that name was found.
- **Reproduction/static trace:** Supply a different callable with the allowed name and internally consistent evidence; runner executes it and shape/hash checks do not establish estimator identity.
- **Consequence:** Once the execution pin is opened, noncanonical mathematics could enter a confirmatory result.
- **Scope:** Paired-audit evaluator registration/control plane; current hard pin prevents execution today.
- **Minimum correction:** Pin module, qualname, canonical source/config/mechanism digest and reject substitutions.
- **Preferred architectural correction:** Use an immutable evaluator registry whose evidence is independently derived from the instantiated estimator.
- **Regression test:** A same-name fake callable is rejected before a fold/checkpoint.
- **Cost / change risk:** M; Medium registry migration; latent criticality after execution approval.

#### AUD-009 — Current executable source set is not closed or clean-reconstructible

- **Severity / confidence:** HIGH / 0.999.
- **State:** current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/lobo.py:221-225; src/stylo/models/equal_channel_ensemble.py:1; scripts/syntax_features.py:12-13; .gitignore:98-108; src/stylo/eval/certificates.py:1`.
- **Evidence:** Modified tracked lobo imports untracked equal-channel source; syntax_features imports untracked experimental replacements after tracked misspelled modules are deleted. An ignored 535-line .py lives inside the discovered production package and changes wheels/code hashes.
- **Reproduction/static trace:** Build/archive only tracked HEAD: current imports/features/evidence differ or disappear; a nominally git-clean tree can still package ignored certificates.py.
- **Consequence:** Merge/archive/wheel cannot reproduce the reviewed current behavior.
- **Scope:** Current user rework only; committed HEAD remains a separate closed snapshot but lacks the repair/equal implementation.
- **Minimum correction:** Track/review intended executable sources or move historical/local code outside discovered package; do not rely on ignored .py.
- **Preferred architectural correction:** Enforce an executable-source manifest and CI rule that all imports/discovered package files are tracked and archive-reconstructible.
- **Regression test:** git archive/wheel imports every public route and exactly matches the source manifest; ignored/untracked dependency fails.
- **Cost / change risk:** S/M; Low functional but high evidence/history compatibility risk.

#### AUD-010 — CI is not an ordinary/scientific release gate and current suite is red

- **Severity / confidence:** HIGH / 0.995.
- **State:** current_dirty_working_tree, committed_HEAD.
- **Exact locations:** `.github/workflows/ci.yml:27-35; tests/test_release_integrity.py:190; scripts/evaluation/run_stack_class_coverage_repair_smoke.py:89-115; tests/test_macro_f1_ci_withdrawal.py:62-76`.
- **Evidence:** CI runs only 61 release-integrity tests out of 872 and installs floating dependencies. Current full suite has one failure (raw json.dumps in smoke runner); clean clone tests depend on ignored PAPER/gen-paper files.
- **Reproduction/static trace:** Ordinary suite result: 871 pass, one fail, two warnings. A clean archive lacks ignored generators and cannot satisfy history/local-file-dependent tests.
- **Consequence:** Scientific, package and clean-clone regressions can merge green; current branch has no valid green baseline.
- **Scope:** CI, ordinary tests, docs/claim guards, clean archive.
- **Minimum correction:** Run focused repair/paired tests and the ordinary clean-archive suite; make all referenced test/source paths tracked.
- **Preferred architectural correction:** Exact environment, wheel/sdist isolated install, CLI smoke, scientific gates and generated-doc freshness jobs.
- **Regression test:** Workflow contract asserts required nodeids/jobs; git archive suite and wheel install must pass.
- **Cost / change risk:** M; Low-medium CI duration/environment risk.

#### AUD-011 — Installed package is not self-contained

- **Severity / confidence:** HIGH / 0.96.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `pyproject.toml:18-29; pyproject.toml:47-54; src/stylo/config.py:22-27; src/stylo/lang.py:132-150; src/stylo/corpus_tools/fetch_classics.py:19; src/stylo/pipeline/train.py:52-75`.
- **Evidence:** Wheel discovery includes Python packages only while defaults/metadata/config are repo-relative. fetch_classics imports undeclared requests; training attestation assumes .git.
- **Reproduction/static trace:** Build/install outside repository: default config/author metadata are absent or silently degraded, fetch-classics may fail import, and git provenance is unavailable.
- **Consequence:** A normal clean wheel cannot reproduce documented CLI/research behavior.
- **Scope:** Packaging and public stylo CLI.
- **Minimum correction:** Declare missing dependency/extra and fail explicitly for workspace-only commands/resources.
- **Preferred architectural correction:** Package runtime resources via importlib.resources; separate workspace runners and define honest non-git provenance.
- **Regression test:** Build wheel in git archive; isolated install runs help/config/meta/import smoke.
- **Cost / change risk:** M/L; Medium API/resource migration.

#### AUD-012 — Ignored uv.lock changes confirmatory run identity

- **Severity / confidence:** HIGH / 0.98.
- **State:** current_dirty_working_tree, committed_HEAD.
- **Exact locations:** `.gitignore:13-15; src/stylo/eval/paired_audit/run_plan.py:215-231; tests/test_paired_audit_control_plane.py:233-236`.
- **Evidence:** run_plan hashes requirements.lock and uv.lock when present, although uv.lock is ignored/noncanonical. Measured digest changes from requirements-only 289987… to 9359aa… with the ignored file.
- **Reproduction/static trace:** Copy the same commit/corpus/config to clean checkout without uv.lock; run_id/checkpoint namespace changes.
- **Consequence:** Same scientific inputs do not have a stable reproducible identity.
- **Scope:** Paired-audit run plan/checkpoint identity.
- **Minimum correction:** Hash exactly one tracked canonical environment spec and reject extra binding inputs.
- **Preferred architectural correction:** Commit a reviewed lock/constraints family or bind an installed-distribution fingerprint under a versioned identity schema.
- **Regression test:** Presence/absence of ignored uv.lock cannot change identity; canonical lock change must.
- **Cost / change risk:** S; Run IDs/checkpoints become incompatible and need documented migration.

#### AUD-013 — Frozen golden migration depends on git HEAD history

- **Severity / confidence:** HIGH / 0.99.
- **State:** current_dirty_working_tree.
- **Exact locations:** `tests/test_paired_audit_references.py:202-208; tests/test_paired_audit_runner.py:36-46; src/stylo/eval/paired_audit/references.py:195-220; scripts/artifacts/capture_work_balanced_ablation_goldens.py:15-18`.
- **Evidence:** Current deletes b4_goldens files and adds renamed identical bytes; tests recover deleted fixture with git show HEAD:. After committing deletion or in an archive, fallback fails. Capture docs still name deleted runner/path.
- **Reproduction/static trace:** Commit current deletion or run tests in git archive; HEAD no longer contains the fallback object in the new commit.
- **Consequence:** Clean release cannot validate the frozen reference; compatibility identifier drift is hidden by local history.
- **Scope:** Frozen work-balanced golden fixtures/references/tests.
- **Minimum correction:** Track the replacement bytes/checksum and add explicit old→new compatibility mapping/env aliases.
- **Preferred architectural correction:** Version registered golden identifiers and migrate readers without changing frozen math/evidence.
- **Regression test:** Post-commit git archive tests with no .git validate both compatibility IDs.
- **Cost / change risk:** M; Medium-high frozen identifier compatibility risk.

#### AUD-014 — Repair smoke runner is not reconstructible from repair snapshot

- **Severity / confidence:** HIGH / 0.999.
- **State:** repair_bundle, current_dirty_working_tree, ignored_smoke_artifact.
- **Exact locations:** `scripts/evaluation/run_stack_class_coverage_repair_smoke.py:1; docs/exploratory/work_balanced/stylo_equal_channels_repair_smoke_v1.json:1`.
- **Evidence:** Artifact driver SHA matches current untracked runner (5e06df…), and source hashes match repair 07a8…, but the verified repair bundle contains no runner and HEAD lacks it.
- **Reproduction/static trace:** An independent party can authenticate a local runner but cannot recover its bytes from repair commit/bundle alone.
- **Consequence:** The diagnostic evidence is not independently rerunnable from the claimed committed snapshot.
- **Scope:** Five-fold repair smoke only; its claim exclusions are correct and it is not full accuracy.
- **Minimum correction:** Preserve the exact runner bytes as tracked evidence source or an immutable evidence bundle.
- **Preferred architectural correction:** Create an evidence-source manifest tying runner, source snapshot, corpus subset and output hashes.
- **Regression test:** Clean bundle/archive recovers exact runner SHA and focused identity/output-isolation tests pass.
- **Cost / change risk:** M; Low numeric but medium evidence-history risk; do not rewrite bound runner.

#### AUD-015 — Two incompatible trainers publish mixed, non-atomic artifacts to the same names

- **Severity / confidence:** HIGH / 0.995.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/pipeline/train.py:111-116; scripts/train.py:26-29; scripts/train.py:164-228; src/stylo/pipeline/predict.py:47-49`.
- **Evidence:** Canonical trainer sequentially writes model.pkl/delta.pkl/authors.json; root legacy trainer overwrites model.pkl and different siblings. Both default to data. Interrupted canonical writes also mix generations.
- **Reproduction/static trace:** Run canonical training, then legacy scripts/train defaults or inject failure after one write; predictor reads incompatible new+stale siblings without a generation manifest.
- **Consequence:** Crash, wrong scoring if dimensions align, or expensive rerun loss.
- **Scope:** Legacy deployment training/prediction.
- **Minimum correction:** Force distinct responsibility-named roots and reject incomplete/mixed sibling sets.
- **Preferred architectural correction:** Use one versioned content-addressed bundle with manifest/digests and atomic current pointer; deprecate legacy writer after reference proof.
- **Regression test:** Cross-producer rejection, failure injection preserves old generation, concurrent publishers never mix.
- **Cost / change risk:** L; High existing-model compatibility risk.

#### AUD-016 — Legacy corpus loader silently drops scientific rows and can retain zero-row authors

- **Severity / confidence:** HIGH / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/corpus.py:37-48; src/stylo/corpus.py:74-107; src/stylo/pipeline/train.py:111-115`.
- **Evidence:** Recursive/fallback layout and broad read exception silently skip unreadable/empty files. Authors are enumerated first, so an unreadable-only author remains in class universe with no rows; disk provenance reload repeats the same omission.
- **Reproduction/static trace:** Make one author's only chunks unreadable/empty. Loader continues, provenance agrees on truncated rows, and legacy train can publish a model that makes that listed author impossible.
- **Consequence:** Corpus rows/classes and work weights change silently; result can be self-consistently but incorrectly attested.
- **Scope:** Legacy load/train/deploy path; strict paired loader is stronger.
- **Minimum correction:** Malformed, missing, unreadable and empty expected chunks must be fatal; every discovered author needs rows/work.
- **Preferred architectural correction:** Use the versioned CorpusIdentity inventory as the only loader contract.
- **Regression test:** Unreadable/non-UTF8/empty/renamed/missing chunk and zero-row author all fail before fitting.
- **Cost / change risk:** M; High legacy corpus compatibility.

#### AUD-017 — Sentinel class scoring remains outside repaired stack

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/ensemble.py:17-19; src/stylo/eval/ensemble.py:37-48; scripts/run_benchmark.py:132-184`.
- **Evidence:** Absent classes receive -1e9 and are converted to finite -30 before softmax/metrics. Current inspected full/PD outer folds happen to be class-complete, so no current row is proven wrong by this path.
- **Reproduction/static trace:** Use an inventory whose training fold lacks a class; benchmark emits a plausible normalized vector instead of failing.
- **Consequence:** Future/alternate benchmark silently fabricates probability mass and metrics.
- **Scope:** Obsolete/legacy benchmark fusion; repaired stylo_stack itself is fail-closed.
- **Minimum correction:** Shared class-coverage preflight and remove sentinel completion.
- **Preferred architectural correction:** One ProbabilityContract/ClassUniverse validator across all evaluation paths; version historical reader only.
- **Regression test:** Missing class, NaN/inf, wrong width/order must raise before softmax/publication.
- **Cost / change risk:** S/M; Medium historical benchmark compatibility.

#### AUD-018 — NLP/representation cache identity does not bind actual runtime pipeline

- **Severity / confidence:** HIGH / 0.97.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/nlp.py:41-63; src/stylo/nlp.py:100-168; src/stylo/features/reps.py:92-128`.
- **Evidence:** Fallback NLP is cached under requested primary name; keys omit actual resolved model/fallback/max_length. DocBin/Rep payload text/model metadata are not verified. Smoke adds independent SHA/runtime receipts, but general runs do not.
- **Reproduction/static trace:** Primary unavailable → fallback bytes cached under primary key; later primary or different fallback is available and stale fallback results are accepted as primary.
- **Consequence:** Same nominal configuration can use different/stale linguistic representations and scientific output.
- **Scope:** General LOBO/train/deployment caches; repair smoke specifically mitigates with pinned cache hash.
- **Minimum correction:** Bind resolved actual package/version/digest and validate doc.text/Rep payload against key.
- **Preferred architectural correction:** Versioned safe cache schema with resolved runtime identity and content-addressed entries.
- **Regression test:** Fallback→primary, fallback A→B, swapped doc/text/Rep and corrupt payload reject/rebuild explicitly.
- **Cost / change risk:** M/L; Cache rebuild and possible numerical drift.

#### AUD-019 — Fano module publishes invalid information and Bayes-error bounds

- **Severity / confidence:** HIGH / 0.995.
- **State:** release_base, committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/fano.py:20-31; src/stylo/eval/fano.py:71-82; src/stylo/eval/fano.py:126-190; docs/fano_frontier.json:4-23`.
- **Evidence:** Mean entropy of arbitrary model posteriors is called H(A|F); H(A)-that is called a valid MI lower bound; h^-1 is called unavoidable error. Data processing does not justify either without true conditional calibration.
- **Reproduction/static trace:** Balanced binary labels with constant q=[.999,.001] have true I=0 and Bayes error .5, but code reports ~.9886 bits and floor .001. Balanced three-class constant q similarly reports positive MI.
- **Consequence:** False quantitative information/impossibility claims in tracked scientific artifact.
- **Scope:** Legacy Fano scientific module/docs, outside confirmatory matrix.
- **Minimum correction:** Withdraw valid-bound/Bayes-floor labels and mark values model-dependent descriptive.
- **Preferred architectural correction:** Implement a separately validated proven bound with explicit assumptions, versioned schema and counterexamples; archive old JSON.
- **Regression test:** Constant overconfident output never yields positive MI lower bound or sub-.5 unavoidable binary error.
- **Cost / change risk:** M; High semantic/artifact schema migration.

#### AUD-020 — validate-corpus reports fatal errors but CLI exits success

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/corpus_tools/validate_corpus.py:78-116; src/stylo/corpus_tools/validate_corpus.py:182-200; src/stylo/cli.py:119-121; run.sh:28-35`.
- **Evidence:** Validator records severity=error for unreadable/empty/exact duplicates, but run returns report; CLI discards it and returns 0. run.sh explicitly relies on set -e to stop.
- **Reproduction/static trace:** Two authors with identical books produce exact_dup error; stylo validate-corpus exits 0 and split/train/evaluate continue.
- **Consequence:** Known invalid/leaky corpus can pass the advertised fatal pipeline gate.
- **Scope:** Canonical all workflow and corpus report.
- **Minimum correction:** Nonzero/typed exception on any error; explicit --report-only for advisory use.
- **Preferred architectural correction:** Central validation policy/gate object consumed by every corpus builder/orchestrator.
- **Regression test:** Subprocess exact duplicate/unreadable/empty cases exit nonzero and downstream stage marker is absent.
- **Cost / change risk:** S/M; Low compatibility; advisory automation must opt in.

#### AUD-021 — split deletes complete corpus before replacement and silently publishes partial output

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/pipeline/split.py:38-45; src/stylo/pipeline/split.py:53-95`.
- **Evidence:** train/unknown roots are rmtree'd before sentencizer/load/chunking; read/empty/no-sentence inputs continue and writes are sequential without staging/bijection.
- **Reproduction/static trace:** Inject sentencizer exception after deletion: old corpus is lost. Inject one-book read error: command succeeds with a smaller corpus.
- **Consequence:** Data loss, partial corpus and silent estimand change before long evaluation.
- **Scope:** Raw cleaned-to-chunk corpus build.
- **Minimum correction:** Fail on any expected work and do not delete current output until full replacement validates.
- **Preferred architectural correction:** Versioned staging root bound to clean manifest; atomically switch pointer and retain prior complete version.
- **Regression test:** Failures before/inside chunking, missing/extra works, crash and concurrent publisher leave old snapshot usable.
- **Cost / change risk:** L; High corpus layout/migration risk.

#### AUD-022 — Exploratory runners attest source once, allowing mixed-code long runs

- **Severity / confidence:** HIGH / 0.96.
- **State:** current_dirty_working_tree.
- **Exact locations:** `scripts/evaluation/run_stylo_lobo_validation.py:92-99; scripts/evaluation/run_stylo_lobo_validation.py:331-358; src/stylo/eval/stylo_lobo_validation.py:1266-1407; scripts/evaluation/run_work_balanced_ablation_screen.py:81-90; src/stylo/eval/work_balanced_ablation_screen.py:527-633`.
- **Evidence:** Code hashes are captured once before multi-cell/process execution; no rehash occurs before later cells, checkpoints or final publication despite comments claiming interrupted artifacts cannot mix states.
- **Reproduction/static trace:** Change one hashed source after A0/first cell. Later workers/cells can import changed bytes, but final artifact retains pre-run attestation.
- **Consequence:** A multi-hour full exploratory result can be signed as code it did not exclusively execute.
- **Scope:** Both untracked true-LOBO and ablation-screen runners; no claim an existing run drifted.
- **Minimum correction:** Re-attest code/config/cache before/after each cell and immediately before final publication; abort on drift.
- **Preferred architectural correction:** Reusable live RunAttestor passed into orchestration/workers with immutable source snapshot option.
- **Regression test:** Injected mid-run source change blocks new checkpoint/final publication and preserves prior complete state.
- **Cost / change risk:** M; Low schema if hashes unchanged; high operational/numerical rerun cost.

#### AUD-023 — HTML report consumes stale legacy evidence and labels default GKF as LOBO

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/report/build.py:16-24; src/stylo/report/build.py:48-50; src/stylo/cli.py:188-213; run.sh:33`.
- **Evidence:** Report hardcodes ignored docs/sweep_table.txt while current sweep writes v2 artifacts. Default run.sh sweep is GKF, but report says LOBO with bootstrap CI; missing evidence becomes '(нет данных)' with a fresh timestamp.
- **Reproduction/static trace:** Run default sweep then report: v2 GKF is ignored; old legacy file or missing placeholder is rendered under a current LOBO claim.
- **Consequence:** Scientifically misleading/stale HTML that appears newly generated.
- **Scope:** Legacy report command/site input, not confirmatory publisher.
- **Minimum correction:** Read exact v2 artifact and derive wording from verified strategy/provenance; fail on missing/mismatch.
- **Preferred architectural correction:** Artifact registry keyed by run_id/weighting/strategy with section-level hashes and typed claims.
- **Regression test:** GKF is never labeled LOBO; missing/stale/tampered provenance fails.
- **Cost / change risk:** M; Medium report compatibility.

#### AUD-024 — Legacy prediction lacks authors/class/probability/distance contracts

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/pipeline/predict.py:47-105`.
- **Evidence:** authors JSON is unchecked; classes_ are mapped into zero arrays without exact universe/order/type/range validation; probability/distance shapes, finiteness, nonnegativity and normalization are unchecked.
- **Reproduction/static trace:** Model classes_=[0,2] with three authors silently zeros class 1; duplicate/negative indices overwrite/alias; NaN can reach argsort/winner.
- **Consequence:** Mixed/incomplete bundle can produce a wrong attribution rather than fail closed.
- **Scope:** Legacy deployment predict; mixed-generation producer makes it reachable.
- **Minimum correction:** Shared strict authors/classes/probability/distance validator before fusion.
- **Preferred architectural correction:** Retire loose triple for schema/hash-bound bundle and public ProbabilityContract.
- **Regression test:** Incomplete/duplicate/bool/negative/out-of-range classes, duplicate authors, NaN/inf/wrong rows/unnormalized probabilities reject.
- **Cost / change risk:** M; Medium-high existing artifact migration.

#### AUD-025 — Case strong verdict uses forbidden iid chunk pseudo-bootstrap

- **Severity / confidence:** HIGH / 0.995.
- **State:** release_base, committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/cases/framework.py:679-687; src/stylo/cases/framework.py:1224-1281; src/stylo/cases/framework.py:1326-1333; research/protocol_v1.yaml:103-110`.
- **Evidence:** Target chunks are flattened without work/document IDs; rows are resampled iid for margin_ci95; CI lower bound upgrades strong/high. Protocol explicitly forbids iid chunk bootstrap.
- **Reproduction/static trace:** Duplicate/autocorrelate chunks of one disputed work; pseudo-sample size rises and interval narrows without new independent evidence.
- **Consequence:** Exploratory case attribution can be upgraded to strong confidence by dependent chunks.
- **Scope:** Case passports/verdicts; ten tracked passports say strong.
- **Minimum correction:** Set CI unavailable for one independent work and forbid CI-driven strong verdict; preserve work IDs.
- **Preferred architectural correction:** Registered hierarchical/block/work uncertainty with estimand and minimum independent units.
- **Regression test:** Duplicated chunks cannot narrow CI/promote status; work boundaries survive load.
- **Cost / change risk:** M; High historical passport semantic migration.

#### AUD-026 — Case required applicability/open-set gates are loaded but not enforced

- **Severity / confidence:** HIGH / 0.99.
- **State:** release_base, committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/cases/framework.py:52-76; src/stylo/cases/framework.py:198-262; src/stylo/cases/framework.py:1224-1265; src/stylo/cases/framework.py:1326-1339; research/protocol_v1.yaml:122-127`.
- **Evidence:** required_gates is echoed but not validated; gate_pass uses only panel feasibility. Closed-set nearest centroid always wins and strong/moderate needs no absolute fit/OOD/negative-control threshold.
- **Reproduction/static trace:** Give an outsider uniformly poor but consistently nearest to one candidate; relative winner/margin can be strong.
- **Consequence:** Model applicability failure can be reported as historical attribution despite registered abstention policy.
- **Scope:** Case target decisions/passports.
- **Minimum correction:** Fail unknown required gates and prohibit strong/moderate absent registered target applicability gate.
- **Preferred architectural correction:** Calibrated development-only open-set rejection bound into versioned passport and independent control evidence.
- **Regression test:** Outsider controls abstain; low absolute fit cannot become strong; required gate names are enforced.
- **Cost / change risk:** M/L; High existing verdict semantic migration.

#### AUD-055 — Benchmark score artifact is not bound to inputs or scoring configuration

- **Severity / confidence:** HIGH / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/benchmarks/scoring.py:78-96; src/stylo/benchmarks/scoring.py:416-500; src/stylo/cli.py:318-336`.
- **Evidence:** BenchmarkScore stores dataset identity and metric objects only. Manifest/truth/submission digests, tolerance, IoU threshold, bootstrap iterations/seed and verification status are discarded before CLI serialization.
- **Reproduction/static trace:** Score different truth/submission bytes or change seed/threshold while preserving dataset identity; result JSON has no binding capable of distinguishing/replaying the run.
- **Consequence:** An archived blind score cannot prove what was scored or be independently bit-reproduced.
- **Scope:** Public benchmark scoring/result artifact.
- **Minimum correction:** Add exact input digests and all scoring parameters/seed to a versioned result envelope.
- **Preferred architectural correction:** Self-hashed ScoreEnvelope binding protocol/code/environment, manifest/truth/submission, verifier receipt and deterministic replay.
- **Regression test:** Any input/parameter change changes binding; exact replay reproduces bytes.
- **Cost / change risk:** M; Medium output-schema compatibility.

#### AUD-056 — Literal abstention sentinel collision lets submissions inflate coverage

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/benchmarks/scoring.py:153-160; src/stylo/benchmarks/scoring.py:229-236; src/stylo/benchmarks/scoring.py:391-412; research/protocol_v1.yaml:89-94`.
- **Evidence:** Every non-None string counts as predicted; true None is converted to ordinary '__abstain__'. Labels accept any nonempty string, so submitting that literal yields the same comparisons but coverage 1.0 instead of zero. A truth class with that literal also makes a missing prediction compare as correct.
- **Reproduction/static trace:** Replace every optional author prediction with '__abstain__': n_predicted=N and coverage=1. Or set a truth label to the same legal literal and a None prediction becomes correct after conversion.
- **Consequence:** Selective coverage endpoint can be directly gamed; arbitrary unknown labels also count as coverage.
- **Scope:** Public benchmark classification/selective-risk scoring.
- **Minimum correction:** Use typed internal abstention, reserve/reject sentinel strings and validate labels against allowed universe.
- **Preferred architectural correction:** Versioned prediction schema with explicit abstain/confidence and preregistered selective-risk contract.
- **Regression test:** None, reserved string and unknown label cannot increase coverage; valid labels behave normally.
- **Cost / change risk:** S/M; Small collision fix; medium-high selective endpoint schema migration.

#### AUD-058 — Public benchmark scorer does not require artifact/text verification

- **Severity / confidence:** HIGH / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/cli.py:88-98; src/stylo/cli.py:318-330; src/stylo/benchmarks/scoring.py:416-433; src/stylo/benchmarks/scoring.py:480-500`.
- **Evidence:** CLI --root is optional. Without it, score_files skips artifact hash verification/document lengths and score_submission skips truth offset bounds.
- **Reproduction/static trace:** Manifest can bind a 10-token file while truth and matching submission cover [0,1000); score without --root accepts IDs/hash, derives length from truth and returns perfect segmentation.
- **Consequence:** Official entrypoint can emit a false blind score for missing/tampered corpus or fictitious offsets.
- **Scope:** Public benchmark scientific scoring path.
- **Minimum correction:** Require artifact root or a digest-bound preverified ArtifactReport; always validate bytes/hash/length.
- **Preferred architectural correction:** ScoreEnvelope consumes one mandatory artifact-verification receipt; unbound scoring is private/test-only.
- **Regression test:** CLI without root fails; out-of-range truth fails; result binds artifact-report digest.
- **Cost / change risk:** S/M; Low-medium CLI compatibility.

### MEDIUM

#### AUD-027 — Core import cycle and cross-layer private imports blur dependency direction

- **Severity / confidence:** MEDIUM / 0.97.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/corpus.py:101-112; src/stylo/workdoc.py; src/stylo/eval/provenance.py`.
- **Evidence:** AST graph has SCC stylo.workdoc ↔ stylo.corpus ↔ stylo.eval.provenance. Domain loading imports evaluation provenance at runtime; CLI/models reach private helpers across layers.
- **Reproduction/static trace:** Static module graph/SCC and inbound-private census.
- **Consequence:** Changes to identity/I/O/evaluation propagate bidirectionally and are hard to test locally.
- **Scope:** Production package architecture.
- **Minimum correction:** Move neutral row/work identity contracts out of eval and stop new private cross-layer imports.
- **Preferred architectural correction:** domain.corpus_identity owns types; application/evaluation depend inward; artifact I/O depends on contracts, never vice versa.
- **Regression test:** Import-boundary test and acyclic graph assertion.
- **Cost / change risk:** M/L; Medium public/private API migration.

#### AUD-028 — Model registry and exploratory/confirmatory routing are scattered

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/lobo.py:190-230; src/stylo/eval/final.py:31-33; src/stylo/eval/paired_audit/applicability.py:28-127; src/stylo/cli.py:38-365`.
- **Evidence:** Factories/default sets/applicability/CLI help maintain overlapping model knowledge. final.py is exploratory comparison, while confirmatory control lives elsewhere.
- **Reproduction/static trace:** A new estimator must be edited/tested in multiple independent registries; equal isolation currently relies on several assertions.
- **Consequence:** Default/confirmatory leakage regression and inconsistent public discoverability are likely.
- **Scope:** Model routing/public CLI.
- **Minimum correction:** Document one authoritative registry and generate default/applicability views.
- **Preferred architectural correction:** Typed ModelRegistry with explicit exploratory/default/confirmatory capabilities and immutable protocol IDs.
- **Regression test:** Registry coverage test ensures every estimator has one owner and equal stays opt-in/out of confirmatory.
- **Cost / change risk:** M; Medium compatibility for public labels.

#### AUD-029 — Several modules/functions combine too many independent responsibilities

- **Severity / confidence:** MEDIUM / 0.995.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/cases/framework.py:1-1378; src/stylo/cli.py:38-365; src/stylo/eval/stylo_lobo_validation.py:1-1476; src/stylo/eval/provenance.py:1-617; src/stylo/eval/segmentation.py:1-1199`.
- **Evidence:** Cases mixes schema/path loading/features/gates/bootstrap/verdict/passport; CLI main is 328 lines/depth 17; true-LOBO duplicates checkpoint/identity/evaluation/reporting; provenance mixes identity, verification and publication.
- **Reproduction/static trace:** AST sizes/responsibility census; length alone was not used—each candidate has distinct state/I/O/statistics seams.
- **Consequence:** Local reasoning and focused testing are expensive; copy-paste checkpoint/validation variants diverge.
- **Scope:** Primary orchestration/god modules.
- **Minimum correction:** Extract only stable seams before full LOBO: validators/contracts/checkpoint store, leave numeric kernels intact.
- **Preferred architectural correction:** Purpose-named modules with one-way APIs described in target tree; split cases and CLI after scientific blockers.
- **Regression test:** Characterization tests before split; import/API and byte-output parity after.
- **Cost / change risk:** L/XL; High numerical/refactor risk; most structural split should follow exploratory LOBO.

#### AUD-030 — Stack/equal fit does not finish fitting; predict refits all channels

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree, repair_bundle.
- **Exact locations:** `src/stylo/models/stacked_clf.py:494-522; src/stylo/models/equal_channel_ensemble.py:50-134; tests/test_work_balanced_weights_only.py:550-576`.
- **Evidence:** fit stores raw training arrays; predict_proba creates/fits vectorizers and SVCs. Equal also inherits private stack mechanics/fake arguments. Single LOBO prediction remains train-only and passport math is accurate.
- **Reproduction/static trace:** Call predict twice or serialize after fit: learning repeats and raw corpus is retained; mutable factory/config changes can affect later prediction.
- **Consequence:** Estimator contract, privacy, repeat inference and performance are misleading.
- **Scope:** Legacy stack and exploratory equal estimator; not proof of current outer leakage.
- **Minimum correction:** Document evaluation-only lazy final fit and forbid deployment/serialization.
- **Preferred architectural correction:** Train six channel pipelines/SVCs in fit; transform/score only in predict while preserving exact math.
- **Regression test:** Fit counters, repeated/batch/serialization parity and exact six-channel order.
- **Cost / change risk:** L; High numerical parity risk; schedule after full exploratory LOBO unless deployment required.

#### AUD-031 — General macro-F1 label universe depends on predictions

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/metrics.py:121-127; src/stylo/eval/stylo_lobo_validation.py:872-900`.
- **Evidence:** General metric uses unique(y_true ∪ y_pred). A predicted train-only singleton expands denominator while a model predicting none uses only 43 labels; strict paired path freezes labels.
- **Reproduction/static trace:** Predict one singleton outside tested labels and compare to a model that does not; macro-F1 denominators differ.
- **Consequence:** Cross-model macro-F1 is not the same estimand.
- **Scope:** Generic evaluation summaries; strict true/paired paths mitigate.
- **Minimum correction:** Require explicit frozen metric label order.
- **Preferred architectural correction:** Central MetricContract bound to run identity.
- **Regression test:** Singleton prediction cannot alter label denominator; unknown label rejects or follows registered policy.
- **Cost / change risk:** S; Metric values change for affected historical runs.

#### AUD-032 — General metrics/significance accept malformed or broadcasting inputs

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/metrics.py:23-127; src/stylo/eval/significance.py:32-116`.
- **Evidence:** No equal 1-D shape, finite/range, iteration/level/bin or exact-bool validation. y_true (n,1) and y_pred (n,) broadcasts to n×n; arbitrary values coerce to bool.
- **Reproduction/static trace:** Pass mismatched shapes or NaN/invalid parameters; functions can return plausible values.
- **Consequence:** Silent wrong statistical summaries through public helpers.
- **Scope:** General metrics/significance; paired audit has stricter validators.
- **Minimum correction:** Central input validators and exact parameter types/ranges.
- **Preferred architectural correction:** Typed MetricInputs/PairedOutcomes contract reused everywhere.
- **Regression test:** Adversarial shape, NaN/inf, bool/int, zero iters, invalid level/bin tests.
- **Cost / change risk:** S/M; Low compatibility except callers relying on coercion.

#### AUD-033 — Stack internal model selection reuses calibration labels

- **Severity / confidence:** MEDIUM / 0.87.
- **State:** current_dirty_working_tree, repair_bundle.
- **Exact locations:** `src/stylo/models/stacked_clf.py:435-493; src/stylo/eval/calibration.py:175-210`.
- **Evidence:** Each calibrator is selected/refit on all OOF labels, then meta CV/model choice evaluates globally calibrated Moof; validation row labels influenced their calibration mapping. W-off split is chunk-random rather than group-disjoint.
- **Reproduction/static trace:** Static dependency trace label→global calibrator→meta-validation features for the same row.
- **Consequence:** Internal selection score/passport is optimistic, although outer held-out LOBO work stays untouched.
- **Scope:** Legacy learned stack internals.
- **Minimum correction:** Label the limitation and do not use internal score as unbiased evidence.
- **Preferred architectural correction:** Nested/cross-fitted group-aware calibration or fixed registered fusion.
- **Regression test:** Calibration for a meta-validation row never fits/selects on its label/group.
- **Cost / change risk:** L; High estimand/numerical change; may be superseded by protocol withdrawal.

#### AUD-034 — BurrowsDelta public branding contradicts frozen legacy denominator

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/models/delta.py:1-12; src/stylo/models/delta.py:157-166; research/work_balanced/paired_audit_protocol.md:99-108; README.md:146-184`.
- **Evidence:** A0 relative frequency normalizes by selected-MFW mass, not all tokens; protocol acknowledges this while public name/docs call it true/classic Burrows Delta.
- **Reproduction/static trace:** Inspect _rel_freq denominator against registered A3/A4 all-token implementation.
- **Consequence:** Readers can attribute results to a standard algorithm that was not implemented.
- **Scope:** Frozen A0/public display naming.
- **Minimum correction:** Document A0 as frozen legacy selected-mass Delta without changing math.
- **Preferred architectural correction:** Compatibility display alias and versioned registry name; preserve frozen identifier mapping.
- **Regression test:** Known denominator example distinguishes legacy and canonical variants; docs/registry match.
- **Cost / change risk:** S/M; Medium public/frozen naming cost.

#### AUD-035 — Ablation resume accepts self-hashed but semantically forged records

- **Severity / confidence:** MEDIUM / 0.97.
- **State:** current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/work_balanced_ablation_screen.py:190-310; src/stylo/eval/work_balanced_ablation_screen.py:394-452`.
- **Evidence:** Resume checks probability width/self-hash but not finite/range/sum or pred/rank/metric consistency; triage trusts stored metrics. Newly applied records are stricter.
- **Reproduction/static trace:** Edit probabilities/metrics coherently enough to recompute self-hash; resumed triage consumes them.
- **Consequence:** Exploratory triage and model-selection evidence can be altered without detection.
- **Scope:** Untracked ablation screen/resume only.
- **Minimum correction:** Reuse full applied-record semantic validation and recompute metrics on load.
- **Preferred architectural correction:** One RecordContract for live and resumed paths bound to data/source identity.
- **Regression test:** Forged self-hashed NaN, wrong pred/rank and inconsistent metrics fail.
- **Cost / change risk:** S/M; Low-medium artifact compatibility.

#### AUD-036 — Documented exact environment is neither canonical nor portable

- **Severity / confidence:** MEDIUM / 0.96.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `README.md:193-207; pyproject.toml:14-29; requirements.lock:35-49; .github/workflows/ci.yml:29-31`.
- **Evidence:** README/CI ignore requirements.lock; raw freeze has no hashes/platform/Python metadata and includes CUDA/heavy model packages, while pyproject ranges float. Local env drifted from lock in numpy/scipy/pandas/spacy/joblib.
- **Reproduction/static trace:** Resolve/install on a fresh supported platform or compare current distribution versions to lock.
- **Consequence:** Researchers/CI can execute different numerical stacks; lock may be un-installable off origin platform.
- **Scope:** Packaging, CI and run identity.
- **Minimum correction:** Document one exact supported install path and fail on fingerprint mismatch for bound runs.
- **Preferred architectural correction:** Platform-aware hashed constraints/lock family with CI resolution/install tests.
- **Regression test:** Fresh supported environment matches recorded distribution fingerprint.
- **Cost / change risk:** M/L; Medium environment migration/cost.

#### AUD-037 — Research/documentation has contradictory normative status and stale code references

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** current_dirty_working_tree, committed_HEAD.
- **Exact locations:** `research/ROADMAP.md:16-34; research/work_balanced/model_routing.md:3-12; research/work_balanced/paired_audit_protocol.md:272-294; research/work_balanced/paired_audit_review_provenance.md:10-148; research/work_balanced/estimand.md:88-94`.
- **Evidence:** Roadmap says control plane must be implemented while implementation audit says complete; protocol says both complete and no runner/builder committed; provenance alternates in-progress/sign-off. Line references point to unrelated current code. Actual Markdown target scan found zero broken link targets.
- **Reproduction/static trace:** Follow each status/line reference against current inventory; mutually exclusive states coexist with no normative ledger.
- **Consequence:** Independent researcher cannot determine authoritative implementation/preparation/approval/execution state.
- **Scope:** README/research protocol/design/audit navigation.
- **Minimum correction:** Add one status ledger and label historical sections; replace unstable line refs with symbol+SHA.
- **Preferred architectural correction:** Generate status/inventory assertions; frozen protocol edits require version/migration, not in-place rename.
- **Regression test:** Doc status contract matches tracked inventory/registrations; symbol links resolve at pinned SHA.
- **Cost / change risk:** M; Low code, medium frozen-document governance risk.

#### AUD-038 — Pages generation/provenance can stay stale

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `.github/workflows/deploy-pages.yml:6-10; .github/workflows/deploy-pages.yml:32; site/scripts/check-provenance.mjs:11-29`.
- **Evidence:** Deploy triggers only site/workflow changes although generator consumes docs; npm install is nondeterministic; generated manifest lacks source/output digests and provenance check verifies tracking only.
- **Reproduction/static trace:** Change a consumed docs artifact without touching site: no deployment/freshness gate runs.
- **Consequence:** Public site can lag or differ from reviewed evidence.
- **Scope:** GitHub Pages and generated site data.
- **Minimum correction:** Trigger on consumed sources/locks, use npm ci and fail on generated diff.
- **Preferred architectural correction:** Typed source registry with source/output digests and reproducible build provenance.
- **Regression test:** Fixture source change changes output; clean CI generation produces zero diff and bound hashes.
- **Cost / change risk:** S/M; Low deployment risk.

#### AUD-039 — Requirement-to-test map and critical runner coverage are not executable

- **Severity / confidence:** MEDIUM / 0.96.
- **State:** current_dirty_working_tree.
- **Exact locations:** `tests/test_paired_audit_fail_closed_sweep.py:1-93; scripts/evaluation/run_stack_class_coverage_repair_smoke.py:1-1222; scripts/evaluation/prepare_paired_audit_inputs.py:1`.
- **Evidence:** Fail-closed map is comments with one executable test; nodeids can disappear while map stays green. Smoke runner has zero test references; preparation runner has little contract coverage and no runner catalog.
- **Reproduction/static trace:** Rename/delete a referenced test node: comment remains and suite does not validate the requirement map.
- **Consequence:** Governance coverage and evidence runners drift silently.
- **Scope:** Tests, exploratory/evidence scripts and docs.
- **Minimum correction:** Machine-readable required nodeids/markers and collect-only validation; document runner contracts.
- **Preferred architectural correction:** Generate requirement→code→test registry and runner identity/output/claim tests.
- **Regression test:** Every required node collects; missing/renamed node and undocumented runner fail CI.
- **Cost / change risk:** M; Low compatibility.

#### AUD-040 — Doc/Rep cache writers race and are not crash durable

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/features/reps.py:87-90; src/stylo/features/reps.py:153-159; src/stylo/nlp.py:110-168`.
- **Evidence:** Fixed *.pkl.tmp and <key>.spacy.tmp names, no lock/merge-under-lock/fsync; global caches unsynchronized. Concurrent warms can truncate/replace each other's temp/canonical state.
- **Reproduction/static trace:** Two processes synchronize before writing the same temp; one replaces/removes while the other writes. Inject crash before rename.
- **Consequence:** Regenerable cache corrupts or derails expensive long runs; stale plausible payload can be consumed.
- **Scope:** General representation and spaCy Doc caches.
- **Minimum correction:** Unique same-dir temp, fsync and explicit corrupt-cache handling.
- **Preferred architectural correction:** Per-key/file lock with merge-under-lock and content-addressed cache entries.
- **Regression test:** Two-process barrier and injected crash preserve prior canonical readable cache.
- **Cost / change risk:** M; Cache rebuild/performance risk.

#### AUD-041 — safe_write_batch promises a transaction but can leave mixed generations

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/provenance.py:561-607; src/stylo/cli.py:211; src/stylo/cli.py:267`.
- **Evidence:** All temps are written, then independent os.replace calls execute in a loop without rollback/commit pointer. Failure after first replacement leaves new+old siblings despite all-or-nothing docstring.
- **Reproduction/static trace:** Monkeypatch second os.replace to raise; first final file is already replaced.
- **Consequence:** Exploratory evidence/provenance siblings can describe different generations.
- **Scope:** Sweep/final exploratory multi-file publication.
- **Minimum correction:** Rename contract to per-file atomic and verify a batch manifest on consumption.
- **Preferred architectural correction:** Versioned staging directory + self-hashed manifest + single atomic pointer/directory publication.
- **Regression test:** Second-replace failure leaves no resolvable new generation; crash/restart recovery test.
- **Cost / change risk:** M; Low-medium output layout migration.

#### AUD-042 — Exploratory run identity/checkpoint/concurrency and full-run operations are weaker than paired control

- **Severity / confidence:** MEDIUM / 0.97.
- **State:** current_dirty_working_tree, committed_HEAD.
- **Exact locations:** `src/stylo/eval/stylo_lobo_validation.py:518-576; src/stylo/eval/stylo_lobo_validation.py:714-782; src/stylo/eval/lobo.py:327-353; scripts/evaluation/run_stylo_lobo_validation.py:312-340`.
- **Evidence:** Run IDs bind checkout-local absolute config/cache paths; checkpoint save is exists-then-replace so concurrent timing-different writers can overwrite; ancestry/fsync durability is weaker. Generic LOBO uses n_jobs=-1 and has no per-fold resume; current Rep cache is ~690 MiB.
- **Reproduction/static trace:** Relocate identical bytes: run_id changes. Barrier two same-run writers after exists(): last replace wins instead of conflict. Interrupt generic LOBO: completed folds are lost.
- **Consequence:** Relocation hurts reproducibility; concurrent/crash behavior and memory amplification make full exploratory operation harder to trust/diagnose.
- **Scope:** Untracked true-LOBO/checkpoint store and generic LOBO; paired store is stronger and telemetry in true-LOBO is positive.
- **Minimum correction:** Make absolute paths display-only, use create-without-overwrite conflict semantics, bound n_jobs and document memory/resume.
- **Preferred architectural correction:** Extract one checkpoint/attestation service reused by exploratory/paired; content identities only, fsync directory/pointer, explicit resource budget.
- **Regression test:** Relocation equivalence, concurrent conflicting writers, crash durability, bounded worker/RSS and resume tests.
- **Cost / change risk:** M/L; Run ID/checkpoint migration and operational cost.

#### AUD-043 — Legacy/duplicate topology and ambiguous names obscure ownership

- **Severity / confidence:** MEDIUM / 0.94.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/final.py:1; src/stylo/eval/segment.py:1; src/stylo/eval/segmentation.py:1; scripts/train.py:1; scripts/predict.py:1; scripts/experiments.py:1`.
- **Evidence:** Package and root scripts compete for train/predict/evaluation; segment vs segmentation are distinct but ambiguous; final.py is exploratory; deleted experemental and untracked experimental form an incomplete rename. AST found exact script clones/private duplicate helpers.
- **Reproduction/static trace:** Import/reachability/name census and artifact collision trace AUD-015.
- **Consequence:** Maintainers cannot infer canonical entrypoint or safely delete/archive old algorithms.
- **Scope:** Scripts/eval naming and dead/duplicate candidates.
- **Minimum correction:** Document canonical/legacy status and prevent legacy outputs colliding now.
- **Preferred architectural correction:** After reference proof: move to scripts/legacy with responsibility names; use eval/exploratory/model_comparison.py and rolling_attribution.py with compatibility shims.
- **Regression test:** No-reference proof, CLI/API compatibility and historical artifact replay before deletion/archive.
- **Cost / change risk:** M/L; Medium external script/doc compatibility.

#### AUD-044 — Known-invalid certificate remains importable production code

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/certificates.py:1-31; src/stylo/eval/certificates.py:372-457; docs/breakthrough_leads.md:56-69`.
- **Evidence:** Ignored certificate tensorizes event affinity by chunk count and takes min per-channel floor as joint bound. Project docs record 546/903 claimed floors above achieved error and close it as invalid.
- **Reproduction/static trace:** Run ignored log/certify_pairs.py; module can still emit CERTIFY_INDISTINGUISHABLE despite documented falsification.
- **Consequence:** A future caller can resurrect a known-invalid impossibility claim.
- **Scope:** Ignored current-only historical method, absent default/confirmatory paths.
- **Minimum correction:** Hard-disable inferential verdict with WITHDRAWN_INVALID_UNIT.
- **Preferred architectural correction:** Move exact implementation/runner to historical reproduction or replace with descriptive divergence API; preserve hashes.
- **Regression test:** Known counterexamples never emit a lower-bound/certificate; package audit excludes withdrawn method.
- **Cost / change risk:** S/M; Low functional, medium historical evidence handling.

#### AUD-045 — Heterogeneity API defaults to incomparable in-sample spaces and a fake significance label

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** release_base, committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/heterogeneity.py:73-159; log/ilfpetrov_heterogeneity.py:24-60`.
- **Evidence:** Default basis=None separately fits representation/SVD/scalers for target and each control despite doc saying this is impermissible. A four-control population-z >=2 is labeled significant without a null/sampling distribution.
- **Reproduction/static trace:** Call compare_to_controls defaults: silhouettes from different optimized coordinate systems are compared and categorical verdict emitted.
- **Consequence:** Can manufacture an inferential two-hands/no-two-hands conclusion from incomparable descriptive scores.
- **Scope:** Dormant tracked API and ignored caller; valid docs/mixture_detection uses outsider basis separately.
- **Minimum correction:** Require one shared outsider-fitted basis and remove significance wording.
- **Preferred architectural correction:** Registered work-level control/permutation design with shared representation and typed descriptive/inferential outputs.
- **Regression test:** Calls without shared basis fail; null/control order cannot create an inferential label.
- **Cost / change risk:** M; Legacy caller compatibility break.

#### AUD-046 — RuAA SHA inventory validation is not bijective/count-exact

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** current_dirty_working_tree, ignored_prep_artifacts.
- **Exact locations:** `src/stylo/eval/paired_audit/references.py:255-276; scripts/evaluation/prepare_paired_audit_inputs.py:115-139; scripts/evaluation/prepare_paired_audit_inputs.py:253-297`.
- **Evidence:** Duplicate path lines and extra/unlisted files are not rejected; n>0 suffices. Preparation records expected_files=141 beside verified count without enforcing exact registered set.
- **Reproduction/static trace:** Duplicate one valid line and omit another while preserving line count/known mismatch; verification can pass.
- **Consequence:** A freeze candidate can overstate complete RuAA evidence inventory.
- **Scope:** Unapproved preparation candidate; becomes more severe if promoted unchanged.
- **Minimum correction:** Parse unique path→digest map and reject duplicates/omissions/extras; require exact registered set/count.
- **Preferred architectural correction:** Signed/canonical inventory schema with root allowlist and exact set binding.
- **Regression test:** Duplicate, omitted, extra and 140-file success-path all reject.
- **Cost / change risk:** S/M; Low prep compatibility.

#### AUD-047 — Invariance evaluator scores impossible/confounded splits

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** release_base, committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/invariance.py:247-385; src/stylo/eval/invariance.py:562-608; src/stylo/eval/invariance.py:746-802; src/stylo/cli.py:338-362`.
- **Evidence:** Split builder marks impossible/confounded, but prediction alignment requires and evaluator scores all splits; worst_group includes them.
- **Reproduction/static trace:** Two authors each exclusive to one source with y_pred=y gives possible coverage 0 yet worst/overall accuracy 1.0.
- **Consequence:** CLI/API can report perfect invariance where closed-set attribution is unidentifiable.
- **Scope:** Public invariance API/CLI; registered purged pilots mitigate and are not impugned.
- **Minimum correction:** Reject/omit predictions and estimates for impossible splits; exclude confounded cells from headline.
- **Preferred architectural correction:** Typed feasible/unconfounded evaluation set bound to exact plan; diagnostic-only confounded slices.
- **Regression test:** Two-source counterexample emits no invariance point/worst-group; plan mismatch rejects.
- **Cost / change risk:** M; Report schema compatibility.

#### AUD-048 — Site publication gate silently accepts malformed/nonfinite model metrics

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `scripts/gen-site-data.mjs:24-53; scripts/gen-site-data.mjs:839-854`.
- **Evidence:** Generator rewrites raw NaN/Infinity tokens to null before JSON parse (including inside strings); unchecked Number() handles CSV; substring allowlist exempts entire models subtree from null/nonfinite gate.
- **Reproduction/static trace:** NaN, bogus or blank required model accuracy/F1/top2 becomes null/NaN, h.includes('models') bypasses gate, JSON.stringify publishes null with exit 0.
- **Consequence:** Comparative scientific metrics can disappear/change while site build succeeds.
- **Scope:** Public site generator.
- **Minimum correction:** Strict JSON, checked numeric parsing, exact nullable field paths only.
- **Preferred architectural correction:** Typed source schemas with source/output hashes and atomic generated output.
- **Regression test:** NaN/Infinity/bogus/blank required metrics fail; legitimate exact null passes; prose token remains unchanged.
- **Cost / change risk:** S/M; May expose latent malformed artifacts.

#### AUD-049 — Benchmark scorer hashes one manifest path but can score another object

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/benchmarks/scoring.py:313-348; src/stylo/benchmarks/scoring.py:440-499; src/stylo/benchmarks/__init__.py:37-101`.
- **Evidence:** score_files accepts manifest object and manifest_path independently, hashes only path, but uses object for tasks/work/bootstrap semantics. CLI passes same source; exported public API permits divergence.
- **Reproduction/static trace:** Pass validated manifest A and path B with matching identity/doc IDs but different work/task metadata; truth hash B passes while grouping/estimand comes from A.
- **Consequence:** Result can claim digest B while scoring semantics came from A.
- **Scope:** Public benchmark Python API; normal CLI wiring currently safe.
- **Minimum correction:** Load manifest internally from the one hashed path or exact-compare supplied object to strict load.
- **Preferred architectural correction:** One immutable ManifestBinding object produced by loader and consumed by scorer.
- **Regression test:** A/B same IDs but different work/task metadata fails before scoring.
- **Cost / change risk:** S; Low-medium API compatibility.

#### AUD-057 — Manual benchmark validator makes an allowed unknown span unrepresentable

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/benchmarks/schema.py:152-171; src/stylo/benchmarks/validator.py:46; src/stylo/benchmarks/validator.py:267-292; src/stylo/benchmarks/artifacts.py:84-89`.
- **Evidence:** JSON schema conditionally forbids evidence when ground_truth_known=false, but manual validator includes evidence in its unconditional required key set and then rejects it when false. Omitted and present evidence both fail.
- **Reproduction/static trace:** Submit a schema-valid unknown span with start/end/label/ground_truth_known=false and no evidence: manual loader reports missing required evidence; adding it reports must be omitted.
- **Consequence:** Honest uncertain/unlabelled benchmark regions cannot be represented through the public runtime despite declared schema.
- **Scope:** Public mixed-span benchmark interoperability.
- **Minimum correction:** Require only start/end/label/ground_truth_known; allow evidence then enforce it conditionally.
- **Preferred architectural correction:** Generate/manual validation from one schema contract to prevent drift.
- **Regression test:** False-without-evidence passes both schema/runtime; false+evidence fails; true evidence requirements remain.
- **Cost / change risk:** S; Low migration risk: accepts previously impossible valid input.

#### AUD-059 — Invariance public API accepts an incomplete metric label universe

- **Severity / confidence:** MEDIUM / 0.98.
- **State:** release_base, committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/invariance.py:615-630; src/stylo/eval/invariance.py:711-756; src/stylo/eval/invariance.py:806-845`.
- **Evidence:** Caller-supplied labels are averaged directly without requiring observed y_true subset, uniqueness or exact type. CLI derives labels and is currently safe.
- **Reproduction/static trace:** y_true=[A,B], y_pred=[A,A], labels=[A] reports macro-F1 2/3 instead of fixed-universe 1/3.
- **Consequence:** Public API callers can omit a difficult truth class and inflate invariance macro-F1/n_labels.
- **Scope:** Latent invariance API risk, not evidence current CLI artifacts are wrong.
- **Minimum correction:** Validate nonempty ordered-unique universe and require observed truth ⊆ labels; allow registered absent classes.
- **Preferred architectural correction:** Reuse MetricContract/frozen class universe across evaluation APIs.
- **Regression test:** Missing observed class and duplicate labels reject; registered absent class remains and scores zero.
- **Cost / change risk:** S; Low compatibility except invalid callers.

#### AUD-060 — Character-offset segmentation is published under token metric names

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/benchmarks/schema.py:20; src/stylo/benchmarks/schema.py:120-139; src/stylo/benchmarks/artifacts.py:138-146; src/stylo/benchmarks/scoring.py:443-477; src/stylo/eval/segmentation.py:56-153`.
- **Evidence:** Schema accepts offset_unit=character and verifier uses len(text), but scorer passes ranges to token-named SegmentationDocument/TokenMetrics and emits n_tokens/token_accuracy/token_macro_f1 without offset unit.
- **Reproduction/static trace:** Score a valid character-offset mixed document; numerical character-position metrics are serialized as token metrics.
- **Consequence:** Metric estimand/unit is mislabeled in public benchmark results.
- **Scope:** Mixed-authorship benchmark character-offset mode.
- **Minimum correction:** Forbid character offsets in v1 scientific scoring or make every result field/unit explicit.
- **Preferred architectural correction:** Unit-parameterized segmentation contract and ScoreEnvelope binding offset_unit.
- **Regression test:** Character mode either rejects or emits character_* names/unit; token fixtures remain unchanged.
- **Cost / change risk:** S/M; Medium result-schema compatibility.

#### AUD-061 — Benchmark endpoints are truth-field-driven rather than task-registration-driven

- **Severity / confidence:** MEDIUM / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/benchmarks/validator.py:224-229; src/stylo/benchmarks/schema.py:60-77; src/stylo/benchmarks/scoring.py:321-349; src/stylo/benchmarks/scoring.py:382-469; src/stylo/eval/segmentation.py:992-1017`.
- **Evidence:** Manifest validation only checks document tasks are a subset of top-level tasks. Truth validation requires spans for mixed_authorship but does not forbid them otherwise or enforce exact task→truth field membership/nonzero endpoint counts. Work is optional; all-missing work silently selects document rather than work bootstrap.
- **Reproduction/static trace:** A spoof-only document with truth/predicted spans emits segmentation metrics; a declared top-level task can have zero observations and silently produce no metric. Omit work on all documents and bootstrap auto falls back to document groups.
- **Consequence:** Later truth can add/omit metric families and denominators outside registration; missing work can pseudoreplicate multiple documents/editions and change the registered uncertainty unit.
- **Scope:** Public benchmark endpoint/control-plane contract.
- **Minimum correction:** Define and enforce task→allowed/required truth fields and nonzero registered endpoint coverage.
- **Preferred architectural correction:** Versioned endpoint matrix in manifest/escrow and ScoreEnvelope with exact denominators.
- **Regression test:** Spoof-only spans reject; unused declared task rejects; valid multi-task docs score only registered endpoints.
- **Cost / change risk:** S/M; Medium schema compatibility.

### LOW

#### AUD-050 — Configuration invariants use asserts that disappear under python -O

- **Severity / confidence:** LOW / 0.99.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/features/function_words.py:43; src/stylo/models/delta.py:43`.
- **Evidence:** User/config invariants are assert statements.
- **Reproduction/static trace:** Run Python with -O; asserts are removed.
- **Consequence:** Invalid configuration may proceed to later opaque failure.
- **Scope:** Feature/Delta configuration.
- **Minimum correction:** Replace with typed ValueError/domain exception.
- **Preferred architectural correction:** Central constructor validation.
- **Regression test:** Optimized interpreter rejects same invalid configs.
- **Cost / change risk:** S; Low.

#### AUD-051 — StyloVectorizer consumes generators twice

- **Severity / confidence:** LOW / 0.97.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/vectorizer.py:55-67`.
- **Evidence:** Representation preprocessing consumes iterable, then original exhausted iterator is passed to blocks.
- **Reproduction/static trace:** Pass a generator rather than list; downstream blocks see no texts.
- **Consequence:** Public-ish API behaves differently by iterable type.
- **Scope:** Vectorizer input normalization.
- **Minimum correction:** Materialize once and reuse.
- **Preferred architectural correction:** Explicit Sequence contract or normalized immutable batch.
- **Regression test:** List and generator produce identical matrices.
- **Cost / change risk:** S; Low.

#### AUD-052 — Headline gate does not explicitly reject nonfinite CI bounds

- **Severity / confidence:** LOW / 0.94.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/eval/paired_audit/headline.py:95-106`.
- **Evidence:** NaN comparison becomes inconclusive rather than a typed malformed-result failure; internal computed CIs are otherwise validated/frequently finite.
- **Reproduction/static trace:** Pass NaN bound into gate object.
- **Consequence:** Malformed external input can be mislabeled inconclusive rather than corrupt.
- **Scope:** Headline helper boundary, not evidence of current headline corruption.
- **Minimum correction:** Validate finite ordered bounds.
- **Preferred architectural correction:** Reuse statistical interval contract.
- **Regression test:** NaN/inf/reversed bounds reject.
- **Cost / change risk:** S; Low.

#### AUD-053 — Dormant LR factory would use non-grouped chunk calibration if reactivated

- **Severity / confidence:** LOW / 0.93.
- **State:** committed_HEAD, current_dirty_working_tree.
- **Exact locations:** `src/stylo/models/lr.py:37-47`.
- **Evidence:** make_classifier builds CalibratedClassifierCV(cv=3) without groups. No callers were found in current src/scripts/tests; active full pipeline does not use it.
- **Reproduction/static trace:** Static reachability scan; if called on chunks from works, ordinary CV can mix a work across calibration folds.
- **Consequence:** Dormant regression risk, not current scientific result.
- **Scope:** Obsolete candidate factory.
- **Minimum correction:** Mark deprecated/private and fail if group-aware use is required.
- **Preferred architectural correction:** Delete after proof of no external references or replace with explicit group-aware calibration API.
- **Regression test:** Reachability/API test and group-overlap adversarial fixture if retained.
- **Cost / change risk:** S; Low external import uncertainty.

## 6. File, function and class size/complexity inventory

### 30 longest Python source/script/test files

| # | Path | State | Physical | Logical nonblank/noncomment |
|---|---|---|---|---|
| 1 | src/stylo/eval/stylo_lobo_validation.py | untracked | 1476 | 707 |
| 2 | src/stylo/cases/framework.py | tracked_clean | 1378 | 878 |
| 3 | scripts/evaluation/run_stack_class_coverage_repair_smoke.py | untracked | 1222 | 432 |
| 4 | src/stylo/eval/segmentation.py | tracked_clean | 1199 | 587 |
| 5 | src/stylo/eval/invariance.py | tracked_clean | 895 | 505 |
| 6 | tests/test_work_balanced_model_routing.py | untracked | 802 | 511 |
| 7 | tests/test_work_balanced_function_word_axes.py | untracked | 694 | 480 |
| 8 | src/stylo/eval/work_balanced_ablation_screen.py | untracked | 686 | 317 |
| 9 | tests/test_release_integrity.py | untracked | 674 | 489 |
| 10 | tests/test_work_balanced_weights_only.py | untracked | 674 | 463 |
| 11 | scripts/run_chekhonte_brother_confound.py | tracked_clean | 652 | 340 |
| 12 | tests/test_stack_class_coverage.py | untracked | 632 | 334 |
| 13 | src/stylo/eval/provenance.py | tracked_modified | 617 | 426 |
| 14 | src/stylo/eval/paired_audit/runner.py | tracked_modified | 610 | 367 |
| 15 | tests/test_paired_audit_publisher.py | tracked_clean | 609 | 390 |
| 16 | scripts/run_taras_delta_replication.py | tracked_clean | 558 | 314 |
| 17 | scripts/run_chekhonte_dubia.py | tracked_clean | 553 | 194 |
| 18 | tests/test_stylo_lobo_validation.py | untracked | 550 | 277 |
| 19 | src/stylo/workdoc.py | tracked_modified | 546 | 322 |
| 20 | src/stylo/eval/certificates.py | ignored | 535 | 351 |
| 21 | src/stylo/models/stacked_clf.py | tracked_modified | 528 | 288 |
| 22 | src/stylo/eval/paired_audit/corpus.py | tracked_modified | 522 | 354 |
| 23 | src/stylo/benchmarks/scoring.py | tracked_clean | 519 | 305 |
| 24 | src/stylo/eval/paired_audit/publisher.py | tracked_clean | 506 | 382 |
| 25 | src/stylo/models/sequence_segmenter.py | tracked_clean | 494 | 317 |
| 26 | tests/test_cases_framework.py | tracked_clean | 478 | 263 |
| 27 | tests/test_paired_audit_corpus.py | tracked_modified | 469 | 326 |
| 28 | src/stylo/eval/paired_audit/run_plan.py | tracked_clean | 444 | 276 |
| 29 | tests/test_paired_audit_control_plane.py | tracked_clean | 437 | 251 |
| 30 | src/stylo/benchmarks/validator.py | tracked_clean | 422 | 231 |

### 20 longest functions

| # | Function | Path:lines | LOC | Args | Branches | Max nesting |
|---|---|---|---|---|---|---|
| 1 | main | src/stylo/cli.py:38-365 | 328 | 1 | 43 | 17 |
| 2 | main | scripts/predict.py:90-321 | 232 | 0 | 25 | 2 |
| 3 | build_package | scripts/build_breakthrough_synthetic.py:80-293 | 214 | 2 | 13 | 4 |
| 4 | main | scripts/run_petersburg_chronicle_gate.py:48-244 | 197 | 0 | 58 | 3 |
| 5 | _prepare_inputs | scripts/evaluation/run_stack_class_coverage_repair_smoke.py:357-536 | 180 | 2 | 24 | 2 |
| 6 | main | scripts/run_ccat50.py:37-207 | 171 | 0 | 43 | 3 |
| 7 | main | scripts/train.py:62-231 | 170 | 0 | 17 | 3 |
| 8 | main | scripts/evaluation/run_stack_class_coverage_repair_smoke.py:1053-1218 | 166 | 1 | 19 | 2 |
| 9 | run_ablation_screen | src/stylo/eval/work_balanced_ablation_screen.py:479-644 | 166 | 19 | 20 | 3 |
| 10 | build_panel | scripts/build_wikisource_edition_panel.py:131-295 | 165 | 3 | 18 | 3 |
| 11 | main | scripts/run_chekhonte_15_micro.py:57-221 | 165 | 0 | 18 | 1 |
| 12 | tune_decoder_on_controls | src/stylo/models/sequence_segmenter.py:329-484 | 156 | 8 | 26 | 3 |
| 13 | _evaluate_equal_fold | scripts/evaluation/run_stack_class_coverage_repair_smoke.py:671-817 | 147 | 5 | 9 | 1 |
| 14 | build_inner_split_preflight_report | src/stylo/models/stacked_clf.py:52-194 | 143 | 5 | 38 | 2 |
| 15 | build_audit_corpus | src/stylo/eval/paired_audit/corpus.py:296-437 | 142 | 9 | 27 | 3 |
| 16 | _aggregate_prepared | src/stylo/eval/segmentation.py:817-958 | 142 | 5 | 14 | 3 |
| 17 | build_leave_one_factor_level_out | src/stylo/eval/invariance.py:247-385 | 139 | 5 | 32 | 3 |
| 18 | main | scripts/run_benchmark.py:95-232 | 138 | 0 | 37 | 3 |
| 19 | main | scripts/run_chekhonte_brother_confound.py:466-603 | 138 | 0 | 19 | 0 |
| 20 | main | scripts/umap_vis.py:64-200 | 137 | 0 | 16 | 3 |

### 20 longest classes

| # | Class | Path:lines | LOC | Methods |
|---|---|---|---|---|
| 1 | BurrowsDelta | src/stylo/models/delta.py:33-365 | 333 | 17 |
| 2 | TestReleaseHygiene | tests/test_release_integrity.py:333-657 | 325 | 21 |
| 3 | StackedChannelClassifier | src/stylo/models/stacked_clf.py:213-528 | 316 | 13 |
| 4 | CheckpointStore | src/stylo/eval/paired_audit/checkpoints.py:137-365 | 229 | 14 |
| 5 | PairedEditionResidualizer | src/stylo/models/invariant.py:50-219 | 170 | 7 |
| 6 | TestBundle | tests/test_work_balanced_model_routing.py:349-492 | 144 | 17 |
| 7 | TestFailClosed | tests/test_paired_audit_checkpoints.py:97-238 | 142 | 16 |
| 8 | TestPublishGate | tests/test_paired_audit_publisher.py:426-559 | 134 | 15 |
| 9 | FunctionWordBlock | src/stylo/features/function_words.py:37-167 | 131 | 9 |
| 10 | CheckpointStore | src/stylo/eval/stylo_lobo_validation.py:714-842 | 129 | 10 |
| 11 | TestBuilder | tests/test_paired_audit_corpus.py:150-276 | 127 | 11 |
| 12 | EqualChannelEnsembleClassifier | src/stylo/models/equal_channel_ensemble.py:19-144 | 126 | 3 |
| 13 | TestApplicability | tests/test_paired_audit_control_plane.py:49-174 | 126 | 10 |
| 14 | TestRunPlan | tests/test_paired_audit_control_plane.py:223-346 | 124 | 16 |
| 15 | DocCache | src/stylo/nlp.py:110-229 | 120 | 8 |
| 16 | TestBuilderFailClosed | tests/test_paired_audit_corpus.py:280-399 | 120 | 15 |
| 17 | WorkLevelVectorizer | src/stylo/features/work_vectorizer.py:158-267 | 110 | 6 |
| 18 | TestStrictJson | tests/test_release_integrity.py:28-130 | 103 | 19 |
| 19 | TestMorphDepWorkDF | tests/test_block_work_balancing.py:189-288 | 100 | 13 |
| 20 | TestDispatchAndLifecycle | tests/test_work_balanced_model_routing.py:253-345 | 93 | 12 |

### Highest top-level responsibility counts

| Path | Top-level definitions | Examples |
|---|---|---|
| src/stylo/cases/framework.py | 62 | CorpusSource, CaseSpec, GateResult, AttributionResult, CasePassport, Work, FwContext, FwPermutationCache |
| src/stylo/eval/segmentation.py | 45 | LabeledSpan, PRF1, TokenMetrics, BoundaryMatch, BoundaryReport, SegmentMatch, SegmentIoUReport, DocumentSegmentationReport |
| tests/test_work_balanced_weights_only.py | 43 | _has_spacy, _panel, _cfg, _warm, _sha, _nrows, _recorder, capture_stack_supervised |
| tests/test_work_balanced_function_word_axes.py | 39 | _has_spacy, _fw, test_fw_grid_vocab_transform_and_denominator, _sel_counts, test_fw_transform_invariant_to_post_fit_relative_fw_mutation, test_fw_R1_all_event_denominator_includes_oov_and_zero_row, test_fw_fixed_list_R_axis_only, _fw_fixed |
| src/stylo/eval/stylo_lobo_validation.py | 34 | TrueLoboError, A0ParityError, CheckpointError, _sha256_bytes, _axes, _checked_label, derive_inventory, load_pinned_a0_reference |
| tests/test_stack_class_coverage.py | 31 | _three_class_panel, _score_matrix, _encoded_channel, _EchoDecisionClassifier, _ClassCompleteSplitter, _ClassIncompleteSplitter, _NonComplementSplitter, test_absent_class_sentinel_can_make_converged_platt_invert_valid_scores |
| src/stylo/eval/provenance.py | 30 | VariantRole, UnsupportedVariantError, ProvenanceError, CorpusPolicyProvenance, RowIdentity, DataContract, DatasetProvenance, _lp |
| src/stylo/eval/invariance.py | 28 | PlanDiagnostics, SplitDiagnostics, FactorSplit, FactorSplitPlan, PurgedFactorSplit, PurgedPlanDiagnostics, PurgedFactorPlan, MetricEstimate |
| src/stylo/eval/paired_audit/runner.py | 26 | RunnerError, _author_of, _run_contract_digest, _expected_folds, _bindings_for, _rebuild_manifest, assert_confirmatory_freeze_approved, run_execution |
| scripts/evaluation/run_stack_class_coverage_repair_smoke.py | 25 | SmokeError, _sha256_file, _canonical_json, _json_hash, _self_hash, _registered_self_hash, _utc_now, _atomic_write |
| src/stylo/benchmarks/scoring.py | 25 | ScoringSpan, TruthRecord, PredictionRecord, BenchmarkTruth, BenchmarkSubmission, ClassificationScore, BenchmarkScore, ScoringFormatError |
| src/stylo/workdoc.py | 24 | ManifestError, sha256_text, canonical_chunk_text, source_provenance_sha256, ChunkerConfig, _req_str_cfg, frozen_chunker_config, chunker_config_hash |
| tests/test_work_balanced_delta_axes.py | 24 | _sel_counts, _all_events, _ref_group_freqs, _ref_z_state, _fit, _fit_corner_weighting, test_delta_vocab_is_the_F_axis, test_delta_R0_is_mean_of_selected_ratios |
| src/stylo/release/hygiene.py | 23 | HygieneError, _run, _text, _lines, _decode_z, is_private_path, _is_shallow, _resolve_commit |
| tests/test_stylo_lobo_validation.py | 23 | _Cfg, tiny_target, _SpyEstimator, _runtime_binding, _identity, test_kernel_release_is_absent_while_libc_and_runtime_inventory_are_binding, test_runtime_ledger_tamper_fails_closed, _probability_for |

Manual threshold conclusions:

- `cases/framework.py`, `stylo_lobo_validation.py`, `work_balanced_ablation_screen.py`, `cli.main` and `provenance.py` merit split because of independent responsibilities—not length alone.
- `segmentation.py` (1,199 LOC), paired `runner.py`, `delta.BurrowsDelta` (333 LOC) and the repair smoke runner are presently cohesive/evidence-bound enough to keep; do not split the SHA-bound smoke in place.
- No production class exceeds 400 LOC; no test module exceeds 1,000 LOC; no reviewed Markdown/YAML/shell document exceeds 800 LOC. Test fixture duplication is local/LOW, not a structural blocker.
- `run_ablation_screen` has 19 arguments; `cli.main` has the largest approximate branch/nesting burden. AST metrics are structural approximations because no new complexity tool was installed.

### Evidence-based split candidates

| Current owner and responsibilities | Proposed modules | API and import direction | Migration risk and regression boundary |
|---|---|---|---|
| `cases/framework.py`: spec parsing, path/content loading, feature extraction, panel gates, target attribution, bootstrap, verdict and passport | `cases/spec_schema.py`, `input_manifest.py`, `panel_gates.py`, `target_attribution.py`, `uncertainty.py`, `passport.py`, `service.py` | `service → {spec_schema,input_manifest,panel_gates,target_attribution,uncertainty,passport}`; computational modules receive immutable typed inputs and never read files | High passport/result risk. First freeze current bytes/status; then copied-target, outsider, work-cluster and compatibility tests. |
| `stylo_lobo_validation.py`: run identity, checkpoints, fold evaluation, statistics, assembly, formatting | `evaluation/exploratory/true_lobo/{contracts,identity,checkpoint_store,fold_evaluator,statistics,artifact,service}.py` | `service → identity/checkpoint_store/fold_evaluator → statistics/artifact`; artifact/checkpoint code may import domain contracts, never model internals | High numerical/checkpoint risk. Use a new checkpoint schema/root and prove old artifact reconstruction plus per-cell numerical parity. |
| `work_balanced_ablation_screen.py`: schemas, resume validation, execution, triage and persistence | `evaluation/exploratory/ablation_screen/{records,executor,triage,artifact,service}.py` | `service → records/executor/triage/artifact`; triage consumes validated records only | Medium artifact risk. Characterize current golden triage, then forged-resume and source-drift tests. |
| `cli.main`: parser construction and all command orchestration | `cli/parser.py`, `cli/commands/{corpus,evaluation,benchmarks,cases,release}.py`, `cli/main.py` | `main → parser/commands → application/evaluation`; production/domain code never imports CLI | Medium public CLI risk. Snapshot `--help`, exit codes and argument routing; retain `stylo.cli:main` shim. |
| `eval/provenance.py`: row types, dataset verification, paths, single/batch publication and headline guards | `domain/corpus_identity.py`, `artifacts/{dataset_verifier,atomic_publication,claim_guard}.py` | Domain types are leaf dependencies; verifiers depend on domain; publication depends on verified artifact values | Medium/high evidence risk. Preserve canonical hashes and add partial-replace/crash tests before moving callers. |

`segmentation.py`, paired `runner.py`, `BurrowsDelta` and the exact smoke runner are deliberately absent from this split table: their size does not currently prove a higher-cohesion decomposition worth the migration risk.

## 7. Naming and structure audit

| Current name | Proposed responsibility name | Why / compatibility |
|---|---|---|
| src/stylo/eval/final.py | src/stylo/evaluation/exploratory/model_comparison.py | It is exploratory comparison, not final/confirmatory. Keep import shim and public CLI token. |
| src/stylo/eval/segment.py | src/stylo/evaluation/exploratory/rolling_attribution.py | Distinguish old rolling attribution from modern segmentation.py; verify external imports first. |
| scripts/experemental/* → scripts/experimental/* | complete one `scripts/experimental/` migration | Misspelling/partial deleted→untracked rename; preserve old path shim or archived mapping until refs proven. |
| scripts/train.py | scripts/legacy/train_chunk_weighted_artifacts.py | Makes legacy schema/output ownership explicit; namespace fix precedes move. |
| scripts/predict.py | scripts/legacy/predict_chunk_weighted_artifacts.py | Distinguishes incompatible legacy deployment consumer. |
| scripts/experiments.py | scripts/legacy/run_chunked_lobo_experiment.py | Names actual algorithm/workflow. |
| scripts/report.py | scripts/legacy/build_html_report.py | Names concrete output and legacy status. |
| scripts/split.py | scripts/legacy/split_sentence_chunks.py | Names transformation. |
| scripts/nlp.py | scripts/legacy/spacy_model_cache.py | Names cache/model responsibility. |
| scripts/run_petersburg_chronicle_NN.py | scripts/run_petersburg_chronicle_nn_attribution.py | Consistent snake_case/acronym casing; compatibility wrapper if public. |
| src/stylo/eval/ensemble.py | src/stylo/evaluation/legacy_score_fusion.py | Exposes sentinel/legacy scope; archive after no-reference proof. |
| src/stylo/eval/certificates.py | research/history/withdrawn_pair_indistinguishability_certificate.py | Known-invalid historical method must not remain importable production; preserve exact SHA. |

Do **not** rename registered schemas, protocol cells A0–A4, estimator tokens, frozen artifacts or their SHA files in place. Every such change needs a new schema/protocol version, compatibility mapping/reader, old→new provenance and a declared retirement date.

## 8. Dead, duplicate and obsolete-code candidates

| Candidate | Classification | Evidence / required proof |
|---|---|---|
| eval/{ensemble,fano,heterogeneity,segment}.py | OWNER DECISION / ARCHIVE | No inbound src edge; some ignored/docs callers and scientific history. Findings AUD-017/AUD-019/AUD-045 apply. |
| models/{invariant,sequence_segmenter}.py | OWNER DECISION | No inbound src edge does not prove no script/public consumer. |
| models/lr.make_classifier | DELETE AFTER PROOF | No current callers; dormant ungrouped calibration AUD-053. |
| scripts/{train,predict,lobo_cv,experiments,utils,nlp,split,report}.py | REFACTOR/ARCHIVE THEN DELETE | Competing legacy stack; resolve artifact namespace and docs/external refs first. |
| deleted scripts/experemental/* | ARCHIVE/MIGRATION EVIDENCE | Partial path migration; do not restore/delete user rework automatically. |
| src/stylo/eval/certificates.py | ARCHIVE HISTORICAL EVIDENCE | Ignored, SHA-bound in P0 snapshot, scientifically withdrawn; not garbage. |
| repair smoke runner | KEEP EXACT / TRACK AS EVIDENCE | SHA-bound and unreconstructible from repair bundle; never cosmetic-split. |
| exact fetch/gate helper clones | CONSOLIDATE AFTER CHARACTERIZATION | Mechanical duplicates, but scripts encode distinct source/protocol assumptions. |

## 9. Documentation contradictions

| Side A | Side B | Conflict |
|---|---|---|
| research/ROADMAP.md:16-34 | research/work_balanced/paired_audit_implementation_audit.md:3-18 | Control plane/runner 'must be implemented' versus implemented. |
| research/work_balanced/model_routing.md:3-12 | paired_audit implementation audit:3-18 | Paired audit unimplemented versus implemented. |
| paired_audit_protocol.md:272-281 | paired_audit_protocol.md:289-294 | Control/preparation complete versus no builder/runner/fold manifest committed. |
| paired_audit_review_provenance.md:10-23 | same:124-148 | Remediation in progress, sign-off, then conditional again. |
| estimand.md:88-94 | pipeline/train.py:45 and :103 | Cited fit-dispatch line now code hashing; actual fit is later. |
| model_routing.md:84 | current cli.py referenced lines | Line numbers no longer identify claimed routing. |
| paired protocol:100-101 | models/delta.py:157-166 | Reference says line 63; actual legacy denominator moved. |
| run.sh:28-35 | validate_corpus.py:182-200 + cli.py:119-121 | Claims fatal validator stops pipeline; CLI always exits zero. |
| README 'honest leakage-free' accuracy sections | AUD-001 corpus containment evidence | Work-ID split is exact but content independence is false. |
| report/build.py:18,48-50 | run.sh default GKF + cli v2 artifact paths | Fresh report labels stale/missing GKF evidence as LOBO/CI. |

There were **zero broken concrete Markdown link targets** in the automated target scan. The documentation defect is semantic status/numeric/source-reference contradiction, not missing Markdown files alone. `0.8805` and 7/60 occurrences were context-bound; nevertheless AUD-001 invalidates the independence interpretation of corpus-derived accuracy. No reviewed path promoted repair-smoke 5/5 to full accuracy.

## 10. Test and CI matrix

| Requirement | Code | Tests | Assessment |
|---|---|---|---|
| legacy stack inner class completeness | src/stylo/models/stacked_clf.py:52, src/stylo/models/stacked_clf.py:319 | tests/test_stack_class_coverage.py | strong focused coverage; frozen-manifest feasibility preflight missing |
| equal estimator fixed six-channel fusion and routing isolation | src/stylo/models/equal_channel_ensemble.py:19, src/stylo/eval/lobo.py:221 | tests/test_stack_class_coverage.py:441 | factory/default isolation covered; exact real-six lifecycle/repeat/serialization missing |
| LOBO exact work complement and train-only feature fitting | src/stylo/eval/lobo.py:270, src/stylo/eval/stylo_lobo_validation.py:372 | tests/test_stylo_lobo_validation.py, tests/test_work_balanced_model_routing.py | work-id complement covered; content-equivalent/nested work boundary absent |
| probability class order, finiteness, normalization, stable top1/rank | src/stylo/eval/lobo.py:287, src/stylo/eval/paired_audit/result_audit.py:89 | tests/test_paired_audit_control_plane.py, tests/test_stylo_lobo_validation.py | strict paths well covered; cross-path tie contract and legacy predictor absent |
| checkpoint identity, corruption, resume, collision and atomic publication | src/stylo/eval/paired_audit/checkpoints.py:137, src/stylo/eval/paired_audit/publisher.py:330 | tests/test_paired_audit_checkpoints.py, tests/test_paired_audit_publisher.py | paired control plane strong; exploratory duplicate store concurrency/durability weaker |
| author-clustered inference, fixed 15-test Holm family, headline gate | src/stylo/eval/paired_audit/inference.py, src/stylo/eval/paired_audit/headline.py | tests/test_paired_audit_control_plane.py, tests/test_paired_audit_publisher.py | mostly strong; content leakage and rank/evaluator bindings supersede readiness |
| corpus identity and cross-work independence | src/stylo/workdoc.py:331, src/stylo/eval/paired_audit/corpus.py:338 | tests/test_workdoc.py, tests/test_paired_audit_corpus.py | manifest integrity covered; exact/containment graph across works missing |
| exploratory/confirmatory/default estimator isolation | src/stylo/eval/final.py:31, src/stylo/eval/paired_audit/applicability.py:28 | tests/test_stack_class_coverage.py:535, tests/test_work_balanced_model_routing.py | equal estimator isolation covered; registry remains scattered |
| clean clone, wheel/sdist, CLI, docs and claim guards | .github/workflows/ci.yml:27, pyproject.toml:47 | tests/test_release_integrity.py, tests/test_release_hygiene.py | current ordinary suite has one failure; no clean archive/wheel installation job |

Executed ordinary suite: **872 collected; 871 passed, 1 failed**, with two warnings. The failure is `test_no_raw_json_dump_in_production_code`, identifying raw JSON serialization in the untracked smoke runner. Focused repair/routing suite: **145 passed**. AST compile of all 278 filesystem Python files and `git diff --check` passed. No confirmatory or model-heavy full LOBO was run.

Tests are strongest around paired checkpoint/result/publication corruption, symlink/path traversal, model routing and frozen writer protection. Missing high-value seams are nested-content corpus isolation; validate/clean/split failure publication; blind benchmark split/escrow/score bindings; case target-role/cluster/open-set behavior; cache/concurrent writer durability; clean archive/wheel; and cross-path ties.

## 11. Scientific-validity assessment

### End-to-end stylometry path

1. Raw corpus enters `pipeline.clean`; this boundary is not bijective/atomic (`AUD-003`).
2. `pipeline.split` builds work manifests but deletes the prior corpus first and skips failures (`AUD-021`).
3. Work-balanced/paired loaders perform strong per-manifest/hash checks, but no corpus-sealing cross-work content graph exists (`AUD-001`).
4. Outer LOBO excludes the exact work ID and fits feature vocabularies/scalers on training rows only. This train-only implementation is a positive result, but exact ID complement is insufficient when one registered work contains another.
5. Repaired stack class-coverage preflight is genuinely fail-closed. Its frozen confirmatory cells are therefore infeasible, not repaired-to-runnable.
6. Equal channels use six full-fold train-only channels, class-complete LinearSVC, fixed identity-softmax and arithmetic mean; no learned calibration, OOF or meta classifier is invoked. It remains opt-in exploratory.
7. Probability alignment is mostly strict in modern paths, but rank ties disagree and legacy sentinel/predict paths fail open.
8. Paired author-clustered bootstrap, literal 15-comparison Holm family, independent result audit and publisher are generally strong. They cannot cure contaminated corpus identity, infeasible cells or unpinned evaluator semantics.

### Status of key claims

| Claim/evidence | Assessment |
|---|---|
| Repair smoke 5/5 vs stack 0/5 | Valid diagnostic for the exact repair-bound bytes/five folds; not accuracy, CI, significance or headline. |
| Legacy stack repair | Fail-closed behavior verified; sentinel remains only in separate legacy paths. |
| 0.8805 legacy headline | Arithmetic/historical context preserved, but corpus content independence is disproved; corrected migration/rerun required for clean accuracy interpretation. |
| 7/60 broken stack | Correctly historical/diagnostic in reviewed references; not generalized. |
| Equal estimator passport | Matches actual one-call fusion mathematics; lifecycle/API/performance still misleading. |
| Paired inference/Holm | Implementation largely sound conditional on valid folds/results; upstream blockers dominate. |
| Fano/certificate/heterogeneity claims | Invalid or withdrawn; must be descriptive/historical, never publication bounds/verdicts. |

Scientific limitations that code alone cannot resolve: editorial identity of a collection versus its constituent works; singleton/low-work authors; topic/genre/edition/time confounding; external generalization beyond these corpora; and uncertainty from a limited author/work sample. These require protocol/editorial decisions and new evidence, not clever caching or refactoring.

## 12. Reproducibility assessment

- HEAD is exact and inventoried, but does not contain the current equal/repair runner rework.
- Repair bundle is complete at `07a8df82…`, but lacks the exact smoke runner bound by the ignored artifact.
- Current tree is not archive/wheel reconstructible due untracked import dependencies, ignored production Python and history-dependent fixtures.
- Environment identity is unstable: floating CI/pyproject ranges, ignored uv binding and a nonportable raw freeze.
- Current ordinary suite is red; no isolated wheel/clean archive job exists.
- Long exploratory source attestation is capture-once, and generic LOBO lacks robust resume/resource contracts.
- Positive controls: strict JSON, many self-hashes/digests, paired immutable version directories, path/symlink adversarial tests, and smoke cache/source receipts.

### Operational and performance profile

- The registered true-LOBO inventory has 251 tested works. `stylo_equal_channels_v1` entails six full-train channel/vectorizer/SVC fits per fold: at least **1,506 channel fits** for one 251-fold pass, before model variants or retries. No empirical wall-time projection was fabricated because the full run was prohibited.
- The current representation pickle is about **690 MiB**; `data/frags_train` contains 26,422 files, the audit-corpus tree 23,738 files, and the full ignored `data` inventory contributes 122,303 paths. Process workers can amplify resident/cache state; actual copy-on-write behavior must be measured rather than assuming either 690 MiB total or a simple `n_jobs` multiple.
- Generic LOBO defaults to configuration `n_jobs=-1`, pre-dispatches `2*n_jobs`, warms the shared representation cache, and has no per-fold checkpoint/resume. The untracked true-LOBO path improves this with one-fold checkpoints, bounded positive `n_jobs`, progress events and wall/CPU/peak-RSS telemetry.
- Equal/stack learning currently happens in `predict_proba`, so repeated prediction repeats the expensive six-channel fit and keeps raw training texts in estimator state.
- Current runner identity hashes package/config/cache once, but does not re-attest live source between cells. Rehashing every corpus file inside every fold would be wasteful; the safe optimization is one immutable content-addressed corpus/source snapshot plus cheap per-cell identity checks—not weaker scientific binding.
- Power-loss durability is uneven: paired immutable directories and atomic pointer patterns are strong, but several pointers/checkpoints lack directory `fsync`, cache writers use fixed temp names, and `safe_write_batch` is not a transaction.

### Security and filesystem posture

- The primary trust-boundary failure is executable deserialization (`AUD-002`); model/cache paths must be treated as code, not passive data.
- Paired-audit corpus/checkpoint/publisher modules have comparatively strong exact inventories, path containment, symlink and conflict tests. The duplicate exploratory checkpoint store and legacy cache/model paths do not reach the same standard.
- No archive extraction path or `shell=True` invocation was found in the reviewed production/scripts search; subprocesses generally pass argv lists. No committed private-key pattern was found. These negative searches do not compensate for pickle/joblib execution or evaluator substitution.
- Predictable/fixed temp names, missing live attestation, non-bijective inventories and optional benchmark artifact verification provide substitution/TOCTOU surfaces even without shell injection.
- Destructive corpus operations are concentrated in `clean`/`split`; safe remediation must stage under an owned version root and atomically select a completed generation, never recursively clean an unresolved broad path.

Reproducibility verdict: **not clean-reproducible today**. Repair evidence is cryptographically inspectable but not self-contained; the current rework is executable locally but not a committed scientific snapshot.

## 13. Current tree → proposed target tree

```text
src/stylo/
├── domain/
│   ├── corpus_identity.py
│   ├── work_identity.py
│   ├── class_universe.py
│   ├── prediction_contract.py
│   └── metric_contract.py
├── features/                         numerical channel implementations
├── estimators/
│   ├── channel_margin.py
│   ├── equal_channel_ensemble.py
│   ├── legacy_stacked_channels.py
│   └── legacy_selected_mass_delta.py
├── evaluation/
│   ├── common/{folds,statistics}.py
│   ├── exploratory/
│   │   ├── model_comparison.py
│   │   ├── true_lobo/{identity,checkpoint_store,fold_evaluator,artifact,service}.py
│   │   └── ablation_screen/{records,triage,service}.py
│   └── confirmatory/paired_audit/
│       ├── registry.py
│       ├── evaluator.py
│       └── existing control-plane modules
├── artifacts/
│   ├── strict_json.py
│   ├── content_addressed_bundle.py
│   ├── checkpoint_store.py
│   └── live_attestation.py
├── application/{corpus_build,train,predict}.py
└── cli/{parser,commands/...}.py

scripts/
├── corpus/
├── evaluation/
├── evidence/
└── legacy/                            only after reference/compatibility proof

tests/{unit,integration,contracts,e2e,fixtures}/
research/{protocols,history,reviews}/
docs/{public,generated,historical_evidence}/
```

The target is a dependency-boundary proposal, not authorization for a wholesale move now. Numerical kernels stay stable while identity, validation, artifact and orchestration contracts move first. Public/frozen names get shims/versioned readers.

## 14. Refactor dependency graph

```text
owner decision: nested works ──> consolidate_corpus_identity ──> corrected exploratory corpus
                                      │
close_executable_source_inventory ────┼──> build_clean_release_and_package_gate
                                      ├──> extract_checkpoint_validation_and_live_attestation
                                      └──> pin_confirmatory_evaluator_registry

unify_rank_and_prediction_semantics ──> remove_sentinel_score_completion
                                      └──> confirmatory result compatibility

owner decision: stack cells ─────────> resolve_infeasible_stack_protocol_cells

seal_blind_benchmark_inputs_and_truth ─> bind_benchmark_score_envelopes

replace_unsafe_serialization_boundaries ─> atomic deployment generation

corrected exploratory LOBO complete ──> split_case_cli_and_validation_orchestrators
                                     └──> remove_obsolete_experiment_scripts
```

Critical path to a full exploratory LOBO is corpus identity → source closure → live attestation/checkpoint safety → clean focused tests. Confirmatory adds rank/evaluator/stack protocol decisions. Benchmark and case remediation are parallel claim-specific tracks.

## 15. Ordered remediation backlog

### 1. `consolidate_corpus_identity`

- **Files:** pipeline/{clean,split}.py, corpus.py, workdoc.py, corpus_tools/validate_corpus.py, paired_audit/corpus.py, new domain/corpus_identity.py.
- **Move/rename/split/delete/change:** Extract exact raw→clean→chunk inventory and cross-work content graph; publish immutable snapshots. Do not mutate frozen corpus/manifests in place.
- **Dependencies:** Owner decision for collection/constituent identity; AUD-001/AUD-003/AUD-020/AUD-021.
- **Compatibility impact:** New registered corpus/protocol version; old digests remain historical.
- **Tests before:** Characterize current 255 works/23,226 chunks and known overlap; failure-injection baselines.
- **Tests after:** Bijection, exact/containment, fail-closed validator, staging/crash/concurrency tests.
- **Rollback boundary:** Atomic pointer keeps previous complete corpus current.
- **Approximate diff:** 1,200–2,000 LOC.
- **Independent?:** Yes from model refactors; mandatory before full exploratory LOBO.

### 2. `close_executable_source_inventory`

- **Files:** lobo.py, equal_channel_ensemble.py, syntax_features.py, scripts/experimental/*, certificates.py, .gitignore, pyproject.toml.
- **Move/rename/split/delete/change:** Track/review intended source; move withdrawn/local Python outside discovered package; record executable-source manifest.
- **Dependencies:** AUD-009; owner classification of ignored/untracked files.
- **Compatibility impact:** Preserve exact smoke runner/hash as evidence; do not rewrite it.
- **Tests before:** Current/HEAD/repair SHA inventories and clean-import characterization.
- **Tests after:** git archive and wheel source-closure/import tests.
- **Rollback boundary:** One source-manifest commit; no data/artifact movement.
- **Approximate diff:** 200–600 LOC plus tracked moves.
- **Independent?:** Yes; mandatory before merge/full LOBO.

### 3. `seal_blind_benchmark_inputs_and_truth`

- **Files:** benchmarks/{schema,validator,artifacts,scoring}.py, research/protocol_v1.yaml (new version), benchmark tests.
- **Move/rename/split/delete/change:** Enforce train/blind content/work isolation, redacted manifest, precommitted truth digest and one manifest source.
- **Dependencies:** AUD-005/AUD-049/AUD-054; protocol/custodian decision.
- **Compatibility impact:** Version schema/escrow; do not rename existing registered IDs without mapping.
- **Tests before:** Current valid manifest scoring and adversarial duplicate/substitution fixtures.
- **Tests after:** Split-isolation, truth-byte commitment, A/B manifest mismatch tests.
- **Rollback boundary:** Old scorer remains historical/integration-only, new blind mode opt-in until migration.
- **Approximate diff:** 500–900 LOC.
- **Independent?:** Yes from LOBO; mandatory before benchmark claims.

### 4. `bind_benchmark_score_envelopes`

- **Files:** benchmarks/scoring.py, cli.py, score schemas/tests.
- **Move/rename/split/delete/change:** Add typed abstention, label universe and self-hashed score envelope binding all inputs/parameters/code/protocol.
- **Dependencies:** seal_blind_benchmark_inputs_and_truth; AUD-055/AUD-056.
- **Compatibility impact:** Version output schema and selective endpoint.
- **Tests before:** Normal classification/segmentation score fixtures.
- **Tests after:** Reserved/unknown-label coverage, parameter/input binding and exact replay.
- **Rollback boundary:** Dual reader for old descriptive score files.
- **Approximate diff:** 350–650 LOC.
- **Independent?:** After escrow substrate; independent of stylometry model.

### 5. `replace_unsafe_serialization_boundaries`

- **Files:** features/reps.py, nlp.py, pipeline/{bundle,train,predict}.py, scripts/train.py.
- **Move/rename/split/delete/change:** Eliminate executable Rep cache; verify model trust before load; consolidate mixed model triples into content-addressed bundle.
- **Dependencies:** AUD-002/AUD-015/AUD-018/AUD-024/AUD-040.
- **Compatibility impact:** Explicit one-time legacy model/cache migration; old bytes trusted only under opt-in offline converter.
- **Tests before:** Golden predictions/cache representation and malicious-loader call-order tests.
- **Tests after:** Hash/symlink/schema/mixed-generation/concurrent publication tests.
- **Rollback boundary:** Keep previous complete bundle pointer; converter separate from runtime.
- **Approximate diff:** 900–1,500 LOC.
- **Independent?:** Mostly; source closure first is preferable.

### 6. `unify_rank_and_prediction_semantics`

- **Files:** lobo.py, groupkfold.py, paired_audit/{checkpoints,result_audit}.py, pipeline/predict.py, new domain/prediction_contract.py.
- **Move/rename/split/delete/change:** One versioned class/probability/top1/rank contract with stable tie rule.
- **Dependencies:** AUD-007/AUD-024/AUD-031/AUD-032.
- **Compatibility impact:** Historical artifacts need explicit reader semantics; no silent recomputation.
- **Tests before:** Capture current non-tie outputs and both tie behaviors.
- **Tests after:** Uniform/tied/majority/malformed vectors agree through all paths.
- **Rollback boundary:** Compatibility adapters selected by artifact schema version.
- **Approximate diff:** 350–650 LOC.
- **Independent?:** Yes; mandatory before confirmatory work.

### 7. `remove_sentinel_score_completion`

- **Files:** eval/ensemble.py, scripts/run_benchmark.py, stacked_clf.py.
- **Move/rename/split/delete/change:** Replace -1e9/-30 completion with shared class-coverage preflight and typed error.
- **Dependencies:** unify_rank_and_prediction_semantics; AUD-017.
- **Compatibility impact:** Archive legacy results; alternate incomplete inventories will now fail.
- **Tests before:** Current class-complete output characterization.
- **Tests after:** Missing/nonfinite/wrong-order classes fail before softmax.
- **Rollback boundary:** Historical reader only; no runtime sentinel fallback.
- **Approximate diff:** 100–250 LOC.
- **Independent?:** Yes after shared contract.

### 8. `resolve_infeasible_stack_protocol_cells`

- **Files:** stacked_clf.py, paired_audit/applicability.py, paired_audit protocol/matrix (new version), tests.
- **Move/rename/split/delete/change:** Add theorem-like feasibility gate; obtain owner decision to amend/withdraw impossible stack cells.
- **Dependencies:** AUD-006; protocol owner approval.
- **Compatibility impact:** Equal estimator cannot substitute; preserve A0–A4 identifiers and old matrix.
- **Tests before:** Frozen manifest/singleton feasibility proof and repair failure tests.
- **Tests after:** Whole matrix preflight completes without fitting or produces approved versioned amendment.
- **Rollback boundary:** No execution; old registration remains immutable.
- **Approximate diff:** 150–400 LOC plus protocol.
- **Independent?:** Yes; mandatory before confirmatory readiness.

### 9. `pin_confirmatory_evaluator_registry`

- **Files:** paired_audit/{run_plan,runner,applicability}.py, new registry.py.
- **Move/rename/split/delete/change:** Map registered name to canonical callable/module/qualname/config/mechanism digests; derive evidence independently.
- **Dependencies:** Source closure and AUD-008.
- **Compatibility impact:** Run IDs/checkpoints version; hard execution pin remains until migration.
- **Tests before:** Fake same-name callable currently accepted.
- **Tests after:** Callable/config/source substitution rejected before fold.
- **Rollback boundary:** Registry version switch; no checkpoint reuse across versions.
- **Approximate diff:** 300–550 LOC.
- **Independent?:** Yes; mandatory before confirmatory execution.

### 10. `extract_checkpoint_validation_and_live_attestation`

- **Files:** stylo_lobo_validation.py, work_balanced_ablation_screen.py, evaluation runners, paired checkpoint primitives.
- **Move/rename/split/delete/change:** Reusable content-only run identity, create-without-overwrite checkpoints, live source/cache re-attestation, fsync/resource budget.
- **Dependencies:** Source closure; AUD-022/AUD-035/AUD-041/AUD-042.
- **Compatibility impact:** New run/checkpoint schema; absolute paths become nonbinding display metadata.
- **Tests before:** Existing checkpoint/resume/output bytes and relocation mismatch.
- **Tests after:** Relocation, concurrent writer, mid-run drift, crash/resume and bounded RSS tests.
- **Rollback boundary:** New schema uses separate root; old checkpoints remain read-only.
- **Approximate diff:** 800–1,300 LOC.
- **Independent?:** Yes; mandatory before full exploratory LOBO.

### 11. `separate_exploratory_and_confirmatory_routing`

- **Files:** eval/final.py, lobo.py, paired_audit/applicability.py, cli.py, new model registry.
- **Move/rename/split/delete/change:** Generate defaults/capability views from one registry; rename final.py through a shim to exploratory/model_comparison.py.
- **Dependencies:** Source closure; evaluator registry design.
- **Compatibility impact:** Keep CLI/model tokens and frozen cells; deprecate import path with shim.
- **Tests before:** Routing/default/claim guards including equal non-default.
- **Tests after:** Registry completeness and namespace-isolation tests.
- **Rollback boundary:** Compatibility shim routes old imports.
- **Approximate diff:** 450–800 LOC.
- **Independent?:** After immediate blockers; not required for first corrected exploratory run if registry assertions stay.

### 12. `harden_case_input_and_applicability_contracts`

- **Files:** cases/framework.py, research/protocol_v1.yaml (version), case passport schemas/tests.
- **Move/rename/split/delete/change:** Content-bound case manifest, strict UTF-8, work-level uncertainty, enforced open-set/required gates.
- **Dependencies:** AUD-004/AUD-025/AUD-026; case protocol owner.
- **Compatibility impact:** Reclassify old passports historical; do not silently upgrade/downgrade in place.
- **Tests before:** Current passport byte/status characterization.
- **Tests after:** Copied target, dependent chunks, outsider and unknown gate adversarial tests.
- **Rollback boundary:** New schema alongside historical reader.
- **Approximate diff:** 700–1,200 LOC.
- **Independent?:** Yes from core LOBO; mandatory before case claims.

### 13. `withdraw_invalid_information_bound_apis`

- **Files:** eval/{fano,certificates,heterogeneity}.py, docs/fano_frontier.json, historical docs/callers.
- **Move/rename/split/delete/change:** Remove inferential labels/defaults; archive exact historical implementations and expose descriptive diagnostics only.
- **Dependencies:** AUD-019/AUD-044/AUD-045; publication owner.
- **Compatibility impact:** Version artifacts and preserve historical hashes.
- **Tests before:** Constant-posterior, achieved-error and heterogeneity counterexamples.
- **Tests after:** No invalid bound/significance/certificate emitted; shared-basis requirement enforced.
- **Rollback boundary:** Historical reproduction path remains read-only.
- **Approximate diff:** 300–700 LOC plus docs.
- **Independent?:** Yes; do before journal/repository release, not needed for corrected LOBO mechanics.

### 14. `build_clean_release_and_package_gate`

- **Files:** .github/workflows/ci.yml, pyproject.toml, requirements/lock strategy, package resources, site workflows.
- **Move/rename/split/delete/change:** Clean archive ordinary suite, focused scientific tests, isolated wheel install, exact environment and generated-doc/site freshness.
- **Dependencies:** Source closure, package resources, test fixture migration.
- **Compatibility impact:** May expose hidden failures; pin supported platform/Python.
- **Tests before:** Record current 871-pass/1-fail and clean-clone failures.
- **Tests after:** Green archive/wheel/CLI/docs/claim/scientific jobs with exact dependency fingerprint.
- **Rollback boundary:** Add jobs nonblocking once, then make required after stabilization.
- **Approximate diff:** 300–650 LOC.
- **Independent?:** Implementation can start now; required before merge.

### 15. `split_case_cli_and_validation_orchestrators`

- **Files:** cases/framework.py, cli.py, stylo_lobo_validation.py, provenance.py.
- **Move/rename/split/delete/change:** Split by target-tree responsibilities without changing numerical kernels.
- **Dependencies:** Immediate contract fixes and a completed corrected exploratory run for parity.
- **Compatibility impact:** Public shims and byte-for-byte artifact parity.
- **Tests before:** Characterization/import/output tests.
- **Tests after:** Module boundary, API compatibility and artifact parity tests.
- **Rollback boundary:** One module at a time behind shims.
- **Approximate diff:** 2,000–3,500 LOC moved/edited.
- **Independent?:** No—schedule after full exploratory LOBO to reduce numerical risk.

### 16. `remove_obsolete_experiment_scripts`

- **Files:** scripts/{train,predict,lobo_cv,experiments,utils,nlp,split,report}.py, orphan eval/model modules.
- **Move/rename/split/delete/change:** Prove references, archive evidence-bearing methods, move retained legacy runners to responsibility-named paths, then delete true duplicates.
- **Dependencies:** Artifact namespace, docs/source closure and owner decision.
- **Compatibility impact:** CLI/script shims or release note for external users; frozen evidence remains.
- **Tests before:** Static+dynamic import, docs, shell, artifact producer and external-owner reference inventory.
- **Tests after:** No-reference proof and clean replay of historical evidence that is retained.
- **Rollback boundary:** Archive/move commit separate from deletion commit.
- **Approximate diff:** 1,500–4,000 LOC mostly moves/deletes.
- **Independent?:** After full exploratory LOBO and owner proof.

### Disposition table

| Disposition | Items |
|---|---|
| KEEP AS IS | Strict JSON core; paired immutable publisher/path guards; author-clustered/Holm primitives conditional on valid input; modern segmentation core; registered cells/schemas/frozen artifacts; exact smoke artifact. |
| REFACTOR NOW | Corpus identity/validate/clean/split; source closure; benchmark escrow; unsafe serialization/model bundle; rank/class contracts; live attestation/checkpoints; CI/package gate. |
| REFACTOR AFTER FULL EXPLORATORY LOBO | Estimator fit/predict lifecycle; broad CLI/cases/true-LOBO/provenance module splits; directory-wide package reorganization. |
| DELETE AFTER PROOF OF NO REFERENCES | Dormant LR factory; true duplicate helpers; obsolete root legacy scripts after artifact/docs/external consumer proof. |
| ARCHIVE AS HISTORICAL EVIDENCE | Invalid certificates/Fano outputs, old experemental paths, superseded protocol/design docs, exact repair smoke runner/source receipt. |
| OWNER DECISION REQUIRED | Collection-versus-constituent corpus identity; frozen stack cell amendment; public benchmark schema/escrow version; case passport reclassification; legacy deployment support; orphan public modules. |

## 16. Go / no-go

| Decision | Verdict | Blocking IDs / condition |
|---|---|---|
| Merge readiness | NO-GO | AUD-009/AUD-010 plus unresolved CRITICAL/HIGH source and corpus boundaries. |
| Full exploratory LOBO | NO-GO | AUD-001/AUD-003/AUD-020–AUD-022/AUD-042; run only after corrected registered corpus and executable snapshot. |
| Confirmatory readiness | NO-GO | All exploratory blockers plus AUD-006–AUD-008; current hard pin remains appropriate. |
| Journal/repository readiness | NO-GO | Corpus leakage, invalid scientific helpers/cases/benchmark escrow, clean reproduction/package/docs failures. |
| Start remediation | GO WITH CONTROLS | Use a separate remediation prompt/branch keyed to finding IDs; no confirmatory execution/refreeze; owner decisions/versioned migrations first. |

It is safe to begin remediation **as a separate, reviewable task**, starting with characterization tests and immutable/versioned boundaries. It is not safe to begin a broad aesthetic refactor or to run the full experiment against the current corpus.

## 17. Accepted residual risks

After the required fixes, the following may remain explicitly accepted rather than 'solved':

- finite author/work sample and singleton authors;
- genre/topic/period/edition confounding not identifiable from code alone;
- historical artifacts retain old semantics and known limitations under immutable versioned readers;
- exact numerical parity may constrain estimator lifecycle cleanup until after corrected exploratory execution;
- local trusted conversion of legacy pickle may be needed once, but never in routine runtime;
- performance bounds depend on available memory/CPU and must be recorded, not inferred from this read-only audit;
- the diagnostic smoke remains only five folds and cannot support an accuracy superiority claim.

## 18. Independent reviewer provenance and convergence

| Reviewer | Primary lens | Cross-review |
|---|---|---|
| Aristotle | Architecture, structure, naming, modularity, AST/import/size | Checked corpus overlap/mixed artifacts and latest benchmark boundaries. |
| Nash | Python/scientific/statistical correctness | Checked source closure/equal semantics and benchmark/C findings. |
| Mendel | Tests, CI, packaging, docs, reproducibility | Independently reproduced corpus containment and security/checkpoint findings; audited benchmark end-to-end. |
| Root | Security, filesystem, concurrency, performance; evidence reconciliation | Reproduced severe traces, executed ordinary/focused tests, kept HEAD/current/repair states separate. |

The adversarial convergence sequence deliberately reset whenever a new MEDIUM+ appeared. It uncovered additional Fano/certificate/heterogeneity, corpus build, live-source, case, invariance, site and benchmark defects after the initial lens reviews. The final two complete catalog passes reported no new MEDIUM+; their exact records are stored in the JSON inventory.

Sign-off criterion outcome: inventory complete; state separation complete; MEDIUM+ evidence recorded; scientific path traced end-to-end; architecture/naming/sizes/docs/CI included; smoke/accuracy boundary explicit; only the two audit files remain as repository changes. This is an **audit sign-off**, not a product/research readiness sign-off.
