# Stylometry codebase remediation log

Started: 2026-07-26 (Europe/Moscow)

Source review:

- `research/reviews/stylometry_codebase_comprehensive_review.md`
- `research/reviews/stylometry_codebase_inventory.json`

Working rules:

1. Preserve the pre-existing dirty working tree and avoid unrelated rewrites.
2. Process findings strictly by severity: CRITICAL, then HIGH, then MEDIUM.
3. Prefer fail-closed corrections where a scientific/protocol migration cannot be
   completed safely in place.
4. Record implementation and verification evidence for every finding.
5. Do not mutate frozen corpora, registrations, or historical evidence in place.

Initial state:

- Reviewed baseline commit: `80da05df5fe909f55d94aed07943f394cfaca1b6`.
- The working tree already contained tracked modifications/deletions and untracked
  repair sources before this remediation started.
- Review catalog: 6 CRITICAL, 24 HIGH, 27 MEDIUM, 4 LOW.
- LOW findings were outside the original remediation scope and are handled in
  the continuation recorded below.

## Progress

| Finding | Severity | Status | Resolution / verification |
|---|---|---|---|
| AUD-001 | CRITICAL | fixed / migration gated | Added exact cross-work chunk and exact asymmetric word-5-gram containment preflight to generic, true-LOBO, ablation, and paired-audit paths. The known Turgenev collision is rejected; affected corpus/headline is explicitly registered ineligible and README claims are withdrawn. A new content-component corpus version/rerun remains required before accuracy claims. |
| AUD-002 | CRITICAL | fixed | Removed executable Rep pickle loading (SQLite + strict JSON); deployment deserializes only externally token-pinned bytes read with `O_NOFOLLOW` and re-hashed before `BytesIO`; certificate NumPy loads forbid object arrays; legacy benchmark cache is strict JSON. Every remaining loose-artifact `joblib`/object-NumPy diagnostic entrypoint is hard-disabled, and an AST allowlist permits only authenticated in-memory bundle loads. |
| AUD-003 | CRITICAL | fixed | Clean now uses strict UTF-8, fatal per-file errors, exact raw/output SHA inventory, staging, fsync and atomic directory exchange. Raw inventory fails on nested/unexpected payloads or empty authors, partial rebuild mode is disabled so preprocessing generations cannot mix, stale deleted raw outputs disappear, and failures preserve the current snapshot. One supported side-effect-free masking helper keeps the Taras preparation consumer working without reactivating the retired builder. |
| AUD-004 | CRITICAL | fixed | Case inputs are strict UTF-8, role/content/inode bound, exact+normalized+containment checked, and recorded in a hashed manifest. Feature computation consumes the same immutable in-memory text snapshot; final disk recheck detects mutation. |
| AUD-005 | CRITICAL | fixed / v2 migration gated | Cross-role exact bytes, canonical path, revision and work crossings reject; blind identity metadata rejects. Because schema 1.0 cannot both redact blind identity and verify full isolation, scientific blind scoring now hard-stops; only the explicitly non-scientific synthetic integration path remains until the documented dual-manifest v2 migration. |
| AUD-054 | CRITICAL | fixed | Truth SHA-256 commitment is mandatory and checked on exact bytes before parsing; those same bytes are parsed. CLI requires the commitment. In-memory scoring is integration-only, preventing construction of a fake committed truth object from reaching scientific scoring. |
| AUD-006 | HIGH | fixed / amendment gated | Added a full manifest × outer-work stack-feasibility preflight using the estimator's actual inner-split contract. It runs before run-plan/checkpoint creation and proves the frozen stack cells infeasible; execution remains fail-closed pending an owner-approved amendment/withdrawal. |
| AUD-007 | HIGH | fixed | Introduced one versioned prediction contract with exact class/probability validation, stable lowest-index top-1 and conservative worst-tie rank. LOBO, GKF, runner, checkpoints and independent audit all use it; incompatible v1 artifacts are rejected. |
| AUD-008 | HIGH | fixed / evaluator-registration gated | Confirmatory evaluator authority is now an immutable canonical object registry binding module, qualname, source, config and mechanism digests. The registry is intentionally empty until independent review, so confirmatory execution and same-name substitutes fail before folds/checkpoints. |
| AUD-009 | HIGH | fixed / commit gate | Added a fail-closed executable-Python inventory binding the exact path set, Git trackedness and SHA-bound governance/evidence files. Ignored/local-only executable sources and unregistered additions fail; the final 280-path set/hash matches, while intended new files must be included in the final commit. |
| AUD-010 | HIGH | fixed | CI now runs the complete and focused suites, reconstructs a Git-free source archive, and builds/tests a clean wheel in the pinned constraint environment. Archive, wheel and release-integrity tests pass. |
| AUD-011 | HIGH | fixed | Runtime defaults, author registry and classics catalog are packaged and loaded with `importlib.resources`; `requests` is declared. Wheel-only runtime access is tested, and workspace-only training fails with an explicit diagnostic instead of a missing-file accident. |
| AUD-012 | HIGH | fixed | Confirmatory environment identity now hashes exactly the canonical tracked `requirements.lock`; missing/symlinked lock fails, while ignored `uv.lock` presence/content cannot re-key a run. |
| AUD-013 | HIGH | fixed | Removed the `git show HEAD:` fallback. Legacy and current golden aliases now resolve to one local fixture whose exact bytes are checked against the pinned SHA-256 before use; replay tests pass without Git metadata. |
| AUD-014 | HIGH | fixed | Preserved the exact repair smoke runner and bound its source snapshot and evidence manifest by SHA-256. A reconstruction test executes the recorded source independently and verifies the expected class-coverage failure mode. |
| AUD-015 | HIGH | fixed | Retired both unauthenticated loose-artifact scripts. The sole trainer publishes an immutable content-addressed bundle; publishers are interprocess-locked, pointer creation is safe/atomic/durable, and concurrent generations cannot mix. |
| AUD-016 | HIGH | fixed | Legacy corpus loading now requires a strict symlink-free `author/work/chunk` inventory, strict UTF-8, nonempty chunks/works/authors and, when present, exact manifest filename/hash bijection. No expected row/class can be silently dropped. |
| AUD-017 | HIGH | fixed | Removed sentinel-to-finite completion from legacy ensemble/benchmark paths. The shared class/score contract rejects incomplete order, sentinel-like missing classes, wrong shapes and nonfinite values before softmax or metrics. |
| AUD-018 | HIGH | fixed | Added `ResolvedNLPIdentity` binding the requested and actual model/fallback, package RECORD digest/version, spaCy version, pipes and max length. Doc/Rep keys and metadata use it; cached Doc text/count and Rep text are verified before reuse. |
| AUD-019 | HIGH | fixed / invalid claims withdrawn | Disabled the invalid H(A\|F), MI-lower-bound, Fano and Bayes-floor APIs; v2 reports only explicitly descriptive posterior diagnostics. Historical artifacts are marked withdrawn and constant-overconfident counterexamples pin the distinction. |
| AUD-020 | HIGH | fixed | Corpus validation uses strict UTF-8/symlink checks and raises typed `CorpusValidationError` on any error by default, making `run.sh` stop. Advisory automation must explicitly select `--report-only`. |
| AUD-021 | HIGH | fixed | Split builds a complete staged source/work bijection with source receipts and strict per-work failure, then publishes train, unknown and chunk-map inside one immutable generation selected by one atomic `CURRENT.json`. Exact hashes/inventory are verified on resolution, pointer failure preserves the entire prior generation, `leave_out` accepts exact `author/work` IDs, and the reproducible benchmark resolves this current snapshot instead of a stale legacy sibling. |
| AUD-022 | HIGH | fixed | Added a reusable live code/config/cache attestor. True-LOBO verifies before/after every fold and before checkpoints/gates/final output (including workers); ablation verifies every cell/checkpoint/final return. Drift aborts without publishing a new checkpoint. |
| AUD-023 | HIGH | fixed | HTML reporting consumes only verified sweep, corpus-validation and prediction evidence envelopes. The latter bind exact rendered bytes to current code/config/input snapshots; prediction additionally binds the authenticated bundle, fragment generation/root and unknown inventory, so switching `CURRENT` invalidates old evidence. The legacy report path is an exact canonical shim, and GKF is explicitly a screening proxy rather than LOBO. |
| AUD-024 | HIGH | fixed | Deployment validates unique authors, exact complete integer class order, probability normalization/finiteness/shape and nonnegative finite distance matrices before fusion; unknown input reads are strict rather than skipped. |
| AUD-025 | HIGH | fixed | Case target work identities are preserved and uncertainty is resampled only across independent works; fewer than two works yields no CI instead of an iid chunk pseudo-bootstrap. |
| AUD-026 | HIGH | fixed / open-set calibration gated | Gate names come from an exact immutable registry; unknown/duplicate/missing gates fail. Scientific target verdicts require the currently unavailable calibrated open-set gate and therefore abstain/inconclusive; the closed-set winner is diagnostic-only. |
| AUD-055 | HIGH | fixed | Added a self-hashed v2 score envelope binding exact manifest/truth/submission digests, artifact-verification receipt, every scoring parameter/seed, protocol, all parser/loader/scorer/segmentation-contract code paths and runtime versions including SciPy. Exact replay is byte-stable under the same bindings. |
| AUD-056 | HIGH | fixed | Abstention is a typed internal object and external abstention is explicit JSON `null` only. A missing classification field is rejected, while present `null` counts as abstention; the reserved sentinel and labels outside the registered truth universe cannot inflate coverage. |
| AUD-058 | HIGH | fixed / v2 migration gated | Public CLI/file scoring requires an artifact root, re-parses the exact manifest bytes, verifies no-follow text bytes/hashes/offset lengths and binds the artifact-report digest. Scientific v1 blind scoring remains intentionally blocked by the dual-manifest migration gate. |
| AUD-027 | MEDIUM | fixed | Moved neutral corpus identity, weighting and span contracts inward and calibration into the model layer. Corpus, workdoc and every model module are free of outward `stylo.eval` imports; old weighting/calibration/provenance paths remain exact-object compatibility aliases. A release-wide models/workdoc/corpus AST boundary test enforces the direction. |
| AUD-028 | MEDIUM | fixed | Added one immutable typed model registry that generates exploratory defaults, calibration views, confirmatory applicability and CLI help. Dynamic Delta specs are parsed exactly, forged/unknown specs fail, and equal-channel remains explicit-only outside confirmatory defaults. |
| AUD-029 | MEDIUM | fixed / staged refactor | Extracted the stable identity and metric-validation seams from the large orchestrators with byte-parity characterization and compatibility tests. The high-risk numerical/god-module decomposition remains deliberately staged until a valid exploratory LOBO rerun rather than being mixed into scientific remediation. |
| AUD-030 | MEDIUM | fixed / fit-time redesign gated | Stack/equal now explicitly declare their lazy-final-fit evaluation-only contract; deployment and serialization hard-fail so raw training rows cannot enter bundles. Repeated/batch parity, exact six-channel order and refit counters are pinned; moving channel fits into `fit` remains a versioned numerical-parity migration. |
| AUD-031 | MEDIUM | fixed | Generic macro-F1 consumes an explicit frozen registered label order and never derives its denominator from predictions. Both true-LOBO assemblers now use the 43 tested-author `metric_label_order`; predictions into the wider 47-class probability universe count only as errors and never enter the macro denominator. |
| AUD-032 | MEDIUM | fixed | Added a shared strict metric contract covering exact one-dimensional aligned inputs, bool/integer discipline, finite normalized probabilities/statistics, frozen label/rank ranges, positive iterations/bins, confidence levels, seeds and group vectors. Broadcasting, NaN/Inf and coercion regressions fail. |
| AUD-033 | MEDIUM | fixed / nested-calibration gated | The globally calibrated internal stack selection score is marked withdrawn and ineligible as unbiased evidence. Stack confirmatory routing hard-fails until a versioned nested/cross-fitted group-aware calibration design exists; historical math remains exploratory only. |
| AUD-034 | MEDIUM | fixed | Preserved the frozen `delta:N` identifier/math but renamed its class/registry/docs display to frozen legacy selected-mass Delta and exposed the denominator contract. A counterexample distinguishes selected-mass frequency 1.0 from canonical all-token 1/3. |
| AUD-035 | MEDIUM | fixed | Live and resumed ablation cells share one semantic record contract. Resume independently revalidates probabilities, prediction/rank/correct, fold/work inventory and recomputes accuracy, macro-F1, top-2, paired deltas/CIs and triage, so a coherent self-rehash cannot forge selection evidence. |
| AUD-036 | MEDIUM | fixed | Added a path-free canonical environment contract derived from `.python-version` and exact portable core pins in `requirements.lock`, plus installed Python/distribution drift verification in CI and bound runners. The supported constraint-based install path is documented and wheel/archive jobs verify it. |
| AUD-037 | MEDIUM | fixed | Added one normative typed status ledger with symbol+SHA-256 bindings for implementation, preparation, approval, execution and headline state. Contradictory narrative rounds/line references are explicitly historical; the affected research documents now defer to the ledger. |
| AUD-038 | MEDIUM | fixed | Pages triggers on every consumed source, uses the committed npm lock with `npm ci`, regenerates and fails on diff. The typed provenance registry binds the generator, 91 actually consumed source files and the generated output by SHA-256; its checker rejects source/output drift. |
| AUD-039 | MEDIUM | fixed | Added a machine-readable 13-requirement code→exact-nodeid registry and an exact catalog of all four evaluation runners with identity/output/claim contracts. The governance gate uses pytest collect-only and rejects missing nodeids or undocumented runners. |
| AUD-040 | MEDIUM | fixed | Doc/Rep cache publication now uses unique same-directory temps, fsync, interprocess locking and atomic publication. Rep SQLite uses `BEGIN IMMEDIATE` immutable merge/conflict detection; concurrent writers preserve both entries and injected crashes preserve the prior readable cache. |
| AUD-041 | MEDIUM | fixed | Replaced misleading sibling replacement with content-addressed immutable batch generations, a self-hashed exact manifest and one atomic `CURRENT.json` pointer. Consumers resolve and validate one generation; pointer failure leaves the prior generation wholly resolvable. |
| AUD-042 | MEDIUM | fixed | Exploratory run IDs bind content rather than absolute checkout/cache paths; paths are display-only. Checkpoints/gates use durable create-if-absent conflict semantics, runtime ledgers lock, generic/true LOBO cap workers at eight, and canonical serialization makes completed resumes byte-identical. |
| AUD-043 | MEDIUM | fixed | Added an exact-discovery topology/ownership registry covering canonical pipeline/report/evaluation surfaces and every conflicting legacy basename release-wide. Config-relative clean/bundle/evidence/exploratory namespaces match their actual writers, frozen historical outputs have no live owner, and all legacy builders, loose-artifact diagnostics, unbounded LOBO and biased ablation entrypoints hard-stop before output. |
| AUD-044 | MEDIUM | fixed / historical reproduction preserved | All invalid inferential certificate entrypoints hard-stop with `WITHDRAWN_INVALID_UNIT`; live code cannot emit the falsified certificate verdict. Exact historical source, output and falsification evidence are SHA-bound under a withdrawn-evidence package; only descriptive divergence helpers remain callable. |
| AUD-045 | MEDIUM | fixed / inferential design gated | Heterogeneity scoring now requires one outsider-fitted shared basis for target and controls. The four-control standardized contrast is explicitly descriptive, and the invalid significance/two-hands verdict is no longer emitted. |
| AUD-046 | MEDIUM | fixed | RuAA verification parses a unique canonical path→digest map, rejects duplicate/omitted/extra paths and all symlinks, checks every digest, and pins the confirmatory inventory to exactly 141 payload files. Preparation records the same constant. |
| AUD-047 | MEDIUM | fixed | Invariance alignment rejects outputs for impossible splits. Factor estimates and worst-group headlines include only feasible, unconfounded registered cells; impossible/confounded cells retain diagnostics with no point estimate, and supplied plans are re-derived and exact-compared. |
| AUD-048 | MEDIUM | fixed | Site generation uses strict JSON without token rewriting, validates required/optional CSV numbers and intervals as finite, and permits null only at exact semantic field paths. A Node self-test covers NaN/Infinity, prose preservation, bogus and blank numbers; the real generator passes. |
| AUD-049 | MEDIUM | fixed | The public file scorer re-parses the exact manifest bytes it hashes and requires equality with any supplied immutable object before reading truth or scoring. An A/B manifest-semantics regression rejects divergence. |
| AUD-057 | MEDIUM | fixed | Manual span validation now requires only start/end/label/ground-truth-known and applies evidence conditionally, matching the JSON Schema: unknown-without-evidence passes, unknown-with-evidence and known-without-evidence fail. |
| AUD-059 | MEDIUM | fixed | Invariance APIs validate a nonempty ordered, exact-type, duplicate-free metric universe containing every observed truth label. One frozen universe is reused for overall and slice macro-F1; registered absent classes remain in the denominator. |
| AUD-060 | MEDIUM | fixed / v1 character-segmentation gated | Schema-v1 character-offset artifacts remain representable, but segmentation scoring fails closed instead of publishing character positions under token metric names. Token scoring records the offset unit and explicit bootstrap contract. |
| AUD-061 | MEDIUM | fixed / v1 work-identity migration gated | Added a central task→allowed/required endpoint matrix, exact declared-task coverage, nonzero blind endpoint checks, and endpoint counts in the score envelope. Segmentation requires an explicit bootstrap unit; scientific scoring requires registered work IDs, while document bootstrap is synthetic-integration-only pending the already-required blind v2 identity migration. |
| AUD-050 | LOW | fixed | Replaced the FunctionWord mode and Delta metric configuration asserts with explicit exact-string `ValueError` contracts. A subprocess regression proves both invalid configurations still fail under `python -O`. |
| AUD-051 | LOW | fixed | `StyloVectorizer.transform` and `fit_transform` now materialize each public input iterable before representation and block passes. Real-block regressions prove list/generator parity for both direct transform and combined fit-transform. |
| AUD-052 | LOW | fixed | The headline gate now validates both CI bounds as non-bool finite real scalars before ordering or decision comparisons and raises the existing typed `HeadlineError`. NaN and both infinities reject in either bound; equal finite bounds retain registered semantics. |
| AUD-053 | LOW | fixed / deletion owner-gated | Release-wide search found no internal caller, but direct-module wheel imports and two SHA-bound config references prevent proof of no external use. The compatibility factory still returns the exact uncalibrated scaler+LR pipeline, while every explicit or config-driven calibration request now raises typed `UngroupedCalibrationError`; the ordinary `cv=3` wrapper is removed. The frozen-config comment and physical API cleanup require a future versioned owner decision rather than evidence refreezing. |
| POST-REVIEW-PUB-001 | HIGH | fixed / historical claims withdrawn | The registered corpus snapshot was already ineligible because of cross-work content leakage, but the site and README still rendered its accuracy, macro-F1 intervals and McNemar result as active claims. Every public headline surface now displays a local withdrawal notice and treats the full and PD-only numbers as historical arithmetic only. Both generators consume the ineligibility registry fail-closed; CI checks byte-identical generated surfaces and provenance, Pages rebuilds when the registry changes, and an SSR/banned-phrase regression protects the rendered contract. No corpus, frozen evidence, governance status, approval or confirmatory gate was changed. |

Open scientific and API-owner decisions are classified in
[`stylometry_owner_decision_memo.md`](stylometry_owner_decision_memo.md).
That memo has no approval authority, leaves every disposition `UNSET`, and does
not change the normative ledger or frozen evidence.

## Chronology

- 2026-07-26: Read the review, captured the pre-existing dirty-tree inventory,
  created this log, and started a baseline test run.
- 2026-07-26: Baseline suite reproduced the review result: 871 passed / 1
  failed (`test_no_raw_json_dump_in_production_code`).
- 2026-07-26: Completed the CRITICAL block. Added focused adversarial tests in
  `tests/test_critical_remediation.py`; the critical/case/benchmark/bundle
  focused set passed. A subsequent full-suite run reached three failures:
  two compatibility fixtures were corrected, while the pre-existing raw-JSON
  smoke-runner guard remains for the HIGH CI/source-closure block.
- 2026-07-26: Implemented HIGH control-plane fixes AUD-006/007/008/012. The
  combined prediction/checkpoint/control-plane suites and all 14 paired-audit
  runner tests passed; infeasible frozen stack cells and the absent reviewed
  confirmatory evaluator remain explicit scientific gates, not silent rewrites.
- 2026-07-26: Implemented HIGH Fano/case fixes AUD-019/025/026. The 57-test
  focused suite, Fano counterexample self-test, strict artifact JSON and syntax
  checks passed.
- 2026-07-26: Implemented HIGH data/deployment/scoring fixes
  AUD-015–018, AUD-020–024 and AUD-055/056/058. Added
  `tests/test_high_remediation.py`; its bundle concurrency, strict corpus,
  class/sentinel, resolved-NLP, validation, split-failure, live-attestation,
  report-provenance and score-envelope controls pass. A broader combined
  critical/HIGH focused run passed 201 tests.
- 2026-07-26: Completed HIGH release-closure fixes AUD-009/010/011/013/014.
  The executable-source inventory binds 267 paths (path-set SHA prefix
  `7432565c`), CI covers full/focused, Git-free archive and clean-wheel
  installs, packaged resources load outside the repository, golden replay is
  Git-independent, and the exact repair-smoke evidence is reconstructible. The
  105-test release/CI/resource/reference cluster passed. Newly created sources
  remain intentionally fail-closed until they are tracked in the final commit.
- 2026-07-26: Ran the complete repository suite after closing the HIGH block;
  every collected test passed (two pre-existing invalid-escape deprecation
  warnings only). This is the frozen pre-MEDIUM regression baseline.
- 2026-07-26: Implemented MEDIUM scientific/publication fixes
  AUD-045–049, AUD-057 and AUD-059–061. Focused benchmark, manifest,
  invariance, RuAA, heterogeneity and site-generator regressions pass; the real
  site generator also completes under strict inputs. Character-offset v1
  segmentation and scientific work-bootstrap without the dual-manifest v2
  identity remain explicit migration gates rather than mislabeled scores.
- 2026-07-26: Implemented MEDIUM governance/topology fixes
  AUD-037/038/039/043/044. The 40-test focused governance cluster, exact
  collect-only requirement/runner catalog, syntax/JSON checks, deterministic
  site generation, 91-source provenance verification and production site build
  pass. A clean npm reinstall was not repeated in the network-restricted
  sandbox; CI uses the committed lock via `npm ci`.
- 2026-07-26: Implemented MEDIUM architecture/model/statistics fixes
  AUD-027–034. The 19-test adversarial cluster, a 341-test affected suite and
  all 13 true-LOBO tests pass, with syntax and scoped diff checks green.
  High-risk stack fit-time/nested-calibration redesigns remain explicit
  deployment/confirmatory gates; compatibility facades preserve valid imports
  and frozen identifiers without preserving misleading claims.
- 2026-07-26: Implemented MEDIUM resume/environment/durability fixes
  AUD-035/036/040/041/042. Semantic-forgery, environment-relocation/drift,
  two-process cache, injected-crash, atomic batch-pointer, relocated run-ID,
  concurrent checkpoint and bounded-worker regressions pass. The affected
  focused set passed 156 tests before the shared metric-contract correction;
  the final combined true-LOBO/A/B set then passed all 36 tests.
- 2026-07-26: Completed the post-MEDIUM integration pass. The first full run
  exposed two stale integration expectations: the synthetic control declared
  `idio_shift` without a blind no-shift observation, and an old axis test still
  expected pickle support from the now evaluation-only lazy stack. Registered
  the existing single-author blind control for `idio_shift`, asserted exact
  endpoint counts, changed the stack regression to require fail-closed
  serialization, and synchronized the two normative routing documents.
  Independent read-only checks confirmed both corrections. The repeated full
  suite passed all 1005 collected tests (only two pre-existing invalid-escape
  deprecation warnings). All Python sources compile, `git diff --check` passes,
  and the release inventory binds 275 paths with path-set SHA256
  `3fe3331c88010b8781ad0a2d9d1dde67c13ba811ed28f4d7bd7d5a9ce8140fb5`.
  Strict site self-test, two identical generations, all 91 source/one output
  provenance digests, and the production Vite build pass.
- 2026-07-26: Launched independent science-contract and
  architecture/governance reviewers after the original 57-item remediation.
  Their confirmed residuals were handled in severity order: all executable
  loose-artifact deserializers and alternative corpus/report builders were
  retired or exact-shimmed; clean became full-generation-only and fail-closed
  over the complete raw inventory; split now uses one immutable
  train/unknown/map generation and one atomic pointer; corpus/prediction report
  sections gained current-input evidence envelopes; benchmark abstention and
  score bindings were tightened; the tested-author macro-F1 contract was
  aligned in both assemblers; topology, import boundaries and the normative
  environment-lock text were made exact. Focused adversarial reruns passed.
- 2026-07-26: Repeated the complete integration pass after the independent
  findings. One stale paired-audit fixture still addressed the retired direct
  `data/frags_train` path; it was migrated to the verified fragment-snapshot
  resolver. The next full run passed all 1033 collected tests (the same two
  pre-existing invalid-escape warnings only). Every Python source compiles,
  `git diff --check` passes, the release path inventory binds 280 files with
  SHA256 `1491901246e81783254dfbe2812c4d4049ad5816559889262ddaae0082a6f8f6`,
  the strict site self-test and 91-source/one-output provenance check pass, and
  the production Vite build succeeds. The Git-aware release check now reports
  only the expected repository-state condition: 80 intended new release files
  must be included in the final commit; code/path/hash closure itself matches.
- 2026-07-26: The requested reviewer re-check then found two final HIGH and
  three MEDIUM integration seams outside the original catalog: the reproducible
  benchmark still addressed a legacy fragment sibling; prediction evidence did
  not compare its generation with current `CURRENT`; the span contract was
  absent from benchmark code binding; an active Taras masker imported the
  retired cleaner; and several topology namespaces described historical rather
  than actual writers. Fixed them in severity order, added regressions, and
  repeated the affected and complete suites.
- 2026-07-26: Final independent science and architecture/governance re-reviews
  report no residual CRITICAL/HIGH/MEDIUM findings. The final complete run
  passed all 1035 collected tests (two unchanged invalid-escape deprecation
  warnings), all Python compiled, diff/provenance/path/hash checks passed, and
  the previously verified production site build remains reproducible.
- 2026-07-26: Integrated the audited tree into the sanitized public `main` as
  merge `fd034e00`; the unrelated legacy local `main` was quarantined at
  `archive/local-main-private-20260726` because its history contains private
  corpus paths. GitHub Actions run `30203153366` then exposed two clean-runner
  boundary defects hidden by the developer checkout: five construction-only
  routing tests eagerly resolved the locally installed Russian spaCy model,
  and the same five plus one provenance test failed in the Git-free archive
  because checkout trackedness was attempted without Git metadata. Made the
  representation-cache path lazy while retaining resolved-NLP identity in its
  namespace/metadata/keying, and added an explicit fail-closed provenance
  `--archive` mode that cannot be used when `.git` is present. Added
  model-free factory and archive-mode regressions and updated the checker's
  release SHA binding. The exact former failures pass in both checkout and a
  materialized Git-free archive; archive inventory and 91-source provenance
  pass; the complete suite now passes all 1038 collected tests with only the
  same two invalid-escape deprecation warnings.
- 2026-07-26: Continued with LOW finding AUD-050. Replaced optimization-sensitive
  configuration asserts in FunctionWord and Delta constructors with typed
  fail-closed validation; the focused suite and an actual optimized-interpreter
  subprocess regression pass.
- 2026-07-26: Completed LOW finding AUD-051 by normalizing single-use text
  iterables at both vectorizer entrypoints. List/generator parity now holds for
  direct transform and fit-transform in the focused block/vectorizer suite.
- 2026-07-26: Completed LOW finding AUD-052. The headline helper now rejects
  nonnumeric, boolean and nonfinite CI bounds before applying the registered
  ordering and noninferiority comparisons; paired-audit headline and publisher
  regressions pass.
- 2026-07-26: Completed the safe portion of LOW finding AUD-053. With no
  repository caller but unresolved external direct-module compatibility, the
  uncalibrated factory remains available and structurally compatible while every
  learned-calibration request fails closed. Deletion is deferred to the owner
  decision memo.
- 2026-07-26: Classified the remaining owner decisions and confirmatory gates
  in a non-authorizing draft memo. Every disposition remains `UNSET`; the
  normative status ledger, frozen protocol/matrix, corpus/evidence bytes,
  freeze-root pin and evaluator registry are unchanged.
- 2026-07-26: The first post-LOW full-suite pass exposed one provenance
  regression: editing the dormant calibration comment changed the authoritative
  default-config digest bound by the frozen ablation golden. Restored both
  config copies byte-for-byte instead of refreezing evidence; the runtime
  calibration hard stop and its tests remain in force.
- 2026-07-26: Reproduced the user-reported production-site crash caused by the
  stale free identifier `MF1_CI` in `Method.jsx`. Replaced the withdrawn
  macro-F1 interval claim with the still-authoritative point estimate and an
  explicit withdrawal explanation, and corrected the same stale claim in the
  corpus findings copy. Added a full-app Vite SSR/render smoke to the ordinary
  Pages build and bound that executable gate in the release inventory. The
  focused site/withdrawal/inventory tests, production build and 91-source
  provenance verification pass; frozen scientific evidence is unchanged.
- 2026-07-26: The subsequent Git-free Pages rehearsal exposed an npm-version
  compatibility gap in the existing lockfile: npm 12 rejected `npm ci` because
  the esbuild and Rollup optional-dependency declarations lacked their
  cross-platform package records. Mechanically completed the lockfile without
  changing resolved runtime or tool versions, added a closure regression, and
  rebound its release digest. A clean archive install and the production
  build/render/provenance chain then passed.
- 2026-07-26: Independent post-LOW science/site review found a release-blocking
  public-contract gap beyond the `MF1_CI` runtime fix: the registered ineligible
  corpus snapshot still appeared as a current accuracy/significance claim on
  the site and in the generated README, including the PD-only slice. Withdrew
  those interpretations locally on every affected surface while retaining the
  exact historical arithmetic, made both generators consume the ineligibility
  registry fail-closed, added visible status and banned-phrase regressions, and
  extended checkout/archive CI provenance gates. The focused contract tests,
  production Vite build, full-app SSR and 92-source/one-output provenance check
  pass; frozen scientific and governance evidence remains byte-identical.
