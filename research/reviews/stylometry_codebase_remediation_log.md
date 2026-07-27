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
| AUD-038 | MEDIUM | fixed | Pages triggers on every consumed source, uses the committed npm lock with `npm ci`, regenerates and fails on diff. Provenance schema v2 binds the generator, all 93 actually consumed source files and the generated output by SHA-256; its 32 unique field mappings exact-cover the output roots and reject source/output/mapping drift. |
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
| POST-REVIEW-LOCK-001 | LOW | fixed | The npm 12-compatible lock already contained the complete optional-platform path closure, but 66 ordinary/platform package records omitted registry `resolved`/`integrity` metadata. Added the exact registry metadata without changing any of the 117 package paths or resolved versions, and strengthened the lock regression to require both fields on every installed package record. |
| POST-REVIEW-PROTOCOL-001 | LOW | fixed | The withdrawn Method/Problem/README surfaces still described exact work-id or book-id exclusion as “no peeking” or leakage-free, although the same content remained in train under another work-id. Recast them as requirements for the next content-safe protocol: the entire content component must be absent from fit and appear only at prediction. Static regressions ban the contradictory legacy wording. |
| POST-REVIEW-COMMENT-001 | LOW | fixed | Normalized stale generator/data comments that called the ineligible full/PD snapshot canonical, production, publishable or leakage-free. They now describe historical arithmetic only; no generated metric value or frozen evidence changed. |
| POST-REVIEW-META-001 | HIGH | fixed | Static SEO/Open Graph/Twitter metadata bypassed the React withdrawal notice and still advertised “leakage-free LOBO” and confidence intervals as active claims. Replaced all three descriptions with an explicit withdrawal, added a static metadata regression, and corrected the canonical/Open Graph domain typo from `russkykod.com` to the deployed `russkiykod.com`. |
| POST-REVIEW-RENDER-001 | MEDIUM | fixed | The SSR smoke rendered only the default framework tab, so an undefined identifier in any of the four click-only chapters could pass CI and fail in production. `App` now accepts a validated test-only initial chapter while production defaults remain unchanged; the smoke renders all five registered chapters with chapter-specific markers and exact registry coverage. A Babel scope gate scans every JS/JSX source, including callbacks/effects that SSR cannot execute, and proves sensitivity with an undefined-identifier negative fixture. Its exact direct parser/traverse dependencies are synchronized in both tracked package-manager locks. |
| POST-REVIEW-REPRO-001 | LOW | fixed | Repro incorrectly said `run.sh all` fetched classics on a clean machine and reproduced the historical metric. Acquisition is now documented as the separate `fetch-classics` command; `all` requires an already assembled registered content-safe corpus, while the ineligible historical snapshot must stop at the isolation gate. Historical arithmetic is described only as stored artifact/provenance replay. |
| POST-REVIEW-EXEC-001 | HIGH | fixed | Independent executable review proved that `run_final`, both sweep strategies, public GKF and the legacy channel benchmark could call raw CV workers without the registered cross-work content-isolation gate. Scientific LOBO/GKF kernels now require an exact sealed, read-only, dataset-bound context produced only after disk provenance and content isolation; the context remains sealed across joblib serialization, suite orchestrators reuse it, and raw `Dataset` calls fail before cache/factory/fit. `run.sh all` performs an explicit read-only corpus gate after split and before warm/train. The channel benchmark selects provenance-preserving uncapped and capped children directly from the disk parent, gates both before cache/fit, emits no replacement for the withdrawn malformed macro-F1 interval, and atomically writes only an `exploratory_internal` candidate generation. The SHA-bound historical `docs/validation{,_pd}.json` have no live writer, are registered in topology, and both public generators verify their P0 hashes before consuming them. Frozen evidence, the ineligibility registry, status ledger, approvals and confirmatory gates were not changed. |
| POST-REVIEW-SEAL-001 | HIGH | fixed | Adversarial construction showed that a frozen dataclass seal could be copied onto changed rows with `dataclasses.replace`, mutated through `object.__setattr__`, self-minted through internal restore/freeze helpers, or transported as claimed disk authority. Scientific contexts now have no public dataclass initializer or value equality. A disk capability is minted only after an actual frozen-contract disk comparison or a fully validated ordered subsequence of an already disk-verified parent, and is bound to a full value snapshot of every provenance field plus the exact scientific payload. Every use rechecks immutable arrays, semantics, receipts, provenance and the registered fingerprint. Serialization deliberately drops disk authority; a production worker must repeat the disk comparison from its resolved config before evaluation. Clone, mutation, helper-mint, forged-child, pickle-downgrade/reverification and serialized-cross-work-duplicate regressions fail closed. |
| POST-REVIEW-TRUELOBO-001 | HIGH | fixed | The true-LOBO compatibility route accepted a bare mutable `Dataset`, computed its run identity, and could reach checkpoint creation after the caller changed rows; a later adversarial pass also showed that injected evaluators and synthetic checkpoint trees could masquerade as production work. Production LOBO, GKF, panel, sweep, ablation and true-LOBO routes now require disk-verified context authority; synthetic contexts are confined to explicit test-only seams. True-LOBO run identity v4 binds the exact context rows digest, `disk_verified` versus `synthetic_test` mode and canonical versus injected evaluator; checkpoint, A0-gate and final-artifact schemas repeat that authority. Production rejects injected evaluators/clocks, legacy identities without authority are non-resumable, and context/run bindings are revalidated before checkpoint mutation and around evaluator calls. The ablation artifact has the equivalent authority binding and exact panel-parent check. Mutation, synthetic resume laundering and evaluator injection regressions are blocked. |
| POST-REVIEW-ATTEST-001 | MEDIUM | fixed | The exploratory benchmark attestation previously omitted its own runner, dependency lock, package contract, exact installed environment, loaded spaCy model and cache inputs, and tolerated a dirty worktree. Its before/after receipt now requires the exact base attestation shape, a clean Git state, SHA-256 bindings for the runner, `requirements.lock` and `pyproject.toml`, the verified canonical installed-environment contract, numerical runtime versions and the live no-fallback `ru_core_news_lg` package/pipeline identity. Production starts with empty process-local NLP/Doc/representation caches, routes every disk cache to one unique empty temporary workspace, uses an identity-scoped in-memory DSP cache, removes that workspace before publication, and resnapshots the live model and attestation. Persistent cache poisoning, model/environment drift, runner/lock drift or dirty state aborts candidate publication. |
| POST-REVIEW-TOPOLOGY-001 | LOW | fixed | The raw-kernel caller inventory inspected direct call syntax only, so aliases, imported renames or dynamic string lookup could hide a new caller. A release-wide AST reference inventory now covers names, attributes, import aliases and string-based lookup for every raw kernel and exact-compares their permitted source files. The runtime context checks remain the primary boundary; this inventory protects architectural drift. |
| POST-REVIEW-ARCHIVE-001 | MEDIUM | fixed for release archive / published-history decision gated | The exact Git-free archive exposed absolute developer-workspace and private corpus-layout strings from 32 frozen passports plus an internal review inventory and obsolete recompute shell script. Those tracked historical bytes were not rewritten: exact `export-ignore` rules omit the 34 internal records from public source archives, and a Git-free content gate rejects protected paths, absolute home-directory layouts, unsafe absolute symlinks and unreadable/special members. Its CLI disables bytecode before importing the scanner so the validation run cannot add a self-matching `__pycache__` member to the tree it is inspecting. Existing published Git history cannot be erased by an ordinary commit; the optional rewrite/sanitization decision is `ODM-009`, disposition `UNSET`. |
| POST-REVIEW-PAGES-001 | MEDIUM | fixed | Pages previously deployed after its own site-only build even when the independent release-integrity workflow was red. Deployment now triggers only from a completed `CI (release integrity)` run whose conclusion is success, original event is a push, branch is `main`, and certified SHA is still the default-branch tip. Checkout pins that exact certified SHA; direct push and manual deployment bypasses are removed. |
| POST-REVIEW-PACKAGE-001 | MEDIUM | fixed | Active package metadata and the README opening promised an “honest leakage-free evaluation” despite the immediately documented withdrawal. The wheel summary now describes a reproducible framework with fail-closed scientific gates, and the generated README opening explicitly declines to call the current registered snapshot leakage-free pending content-safe migration and full rerun. Historical arithmetic and frozen evidence are unchanged. |
| POST-REVIEW-PROVENANCE-001 | LOW | fixed | Provenance `entries` previously accepted any array and even contained a duplicate `models` record plus human pseudo-globs. Schema v2 gives every unique dotted site-data key a nonempty note and a sorted nonempty array of canonical exact sources. The generator and checker both require every key to resolve, every top-level output to be covered, every entry source to belong to the digest-verified registry, and every consumed source to be cited; unsafe, unverified, duplicate and nonexistent field mappings fail adversarial tests. |
| POST-REVIEW-P0-001 | LOW | external owner decision gated | The in-repository P0 hash table detects accidental one-sided mutation but cannot independently prevent a coordinated change to both expected and actual bytes. No local file can truthfully manufacture that external authority, so frozen validation inputs and gates remain byte-identical and `ODM-010` records protected signed tag/attestation or separately controlled registry options with disposition `UNSET`. |
| SCI-FINAL-001 | MEDIUM | fixed | Final exact-commit review showed that the benchmark reconstructed its snapshot with `resolved=requested`, so a real `ru_core_news_md` fallback could be published as a no-fallback `ru_core_news_lg` run. The snapshot now consumes only the identity registered by `load_nlp`, rejects any fallback/request mismatch, verifies every hashed wheel-`RECORD` member against its current bytes and size, and binds an exact serialization digest of the live pipeline into a self-hashed benchmark identity used by the DSP cache. Before/after snapshots detect live component-state drift; regressions distinguish same-pipe-name pipelines with different segmentation and reproduce fallback rejection. |
| SITE-FINAL-001 | MEDIUM | fixed | Final public-surface review found a stale hard-coded Sholokhov paragraph in the generated README and therefore in wheel/sdist metadata: it said 4/4 and mixed a 0.455 open-set control with the registered noncircular LOBO gradient. The generator now fails closed on and renders directly from `docs/sholokhov_lobo.json`: 3/4 and 0.595→0.035. The separate 0.455 block-permutation control is explicitly labeled as a distinct test; active site fields no longer consume the older frozen `sholokhov_rigor12.json` verdict; and a source-bound regression prevents README/package prose from drifting from the registered JSON. No evidence bytes or scientific approval state changed. |
| SCI-FINAL-002 | MEDIUM | fixed | Rereview of the wheel boundary showed that an arbitrary `RECORD` row with an empty digest was skipped, so an unhashed vector/vocab/config payload could change without changing the package identity. Empty digests are now accepted only for the wheel's own `*.dist-info/RECORD` row and installer-generated `__pycache__/*.pyc`; every other member must carry a verified SHA-256 and exact size. A forged `vectors.bin,,` regression fails closed, while the real `ru_core_news_lg` wheel's two legitimate unhashed installer rows still verify. |
| SCI-FINAL-003 | HIGH | fixed | Exact rereview showed that the remaining path-shaped `__pycache__/*.pyc,,` exception could execute bytecode unrelated to its RECORD-hashed source, and verification still occurred only after `spacy.load`. The verifier now binds the sole empty `RECORD` row to the distribution's actual metadata file, discovers canonical runtime caches even when RECORD omits them, requires every such pyc to map to an already SHA-256/size-verified source, and exact-compares its tag, header, optimization path and marshalled body with bytecode compiled from that source. Both primary and fallback packages are verified before and after model loading for full and NER routes; an integrity failure cannot trigger fallback. Adversarial listed/unlisted bytecode, ignored-code-field, load-order and real `ru_core_news_{lg,md}` regressions pass without changing frozen evidence, owner dispositions or confirmatory gates. |
| SCI-FINAL-004 | HIGH | fixed / strengthened by SCI-FINAL-005 | Final exact-commit review showed that wheel verification and Python import resolution were still separate authorities: a same-named package earlier on `sys.path`, or an unrecorded native `__init__` extension beside the verified source, could be imported while the genuine distribution's RECORD identity was recorded. The standard source-package spec, canonical `__init__.py` origin and single package search root are bound to a SHA-256/size-verified member of the same distribution; package-root inventory rejects every unrecorded file or symlink except bytecode proven derivable from a verified source. Primary/fallback shadow and native-extension regressions fail before load. SCI-FINAL-005 subsequently removed the mutable loaded module from the trust boundary altogether. Frozen evidence, scientific approvals, owner dispositions and confirmatory gates remain unchanged. |
| SCI-FINAL-005 | HIGH | fixed | Exact rereview of `37063bc8` showed that a genuine model package imported earlier could retain its verified spec/path while its mutable in-memory `load` callable returned an unrelated pipeline; the benchmark then self-hashed that pipeline under the genuine wheel identity. Scientific loading no longer executes the model package namespace: it binds the canonical RECORD-hashed `name/__init__.py` through the standard path finder and calls spaCy's trained-pipeline loader directly by that exact verified path. Preloaded primary/fallback modules, their globals and `name.*` state are therefore non-authoritative; full and NER routes reverify package bytes and path binding after both successful and failed loads, so an integrity-changing `OSError` cannot be laundered into fallback. Synthetic order/route regressions and real poisoned lg/md packages prove the substituted callables are never invoked. Frozen evidence, scientific approvals, owner dispositions and confirmatory gates remain unchanged. |

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
- 2026-07-26: Closed the remaining lockfile-integrity LOW without dependency
  drift. Reconstructed missing npm registry metadata in a clean temporary
  directory, preserved the exact 117-package path/version graph, and added 66
  `resolved`/`integrity` pairs. The regression now rejects any incomplete
  installed package record. A clean `npm ci` followed by the production
  build/full-app SSR passes with the enriched lock.
- 2026-07-26: Final independent science/site review found two LOW wording
  contradictions after the public withdrawal: Method still described work-id
  exclusion as “no peeking”, and internal generator/data comments still called
  the ineligible full/PD snapshot canonical, production or leakage-free.
  Reframed the visible block as a requirement for the next content-component
  protocol, normalized the comments to historical-only status, and extended the
  static public-contract regression. No metric, corpus or frozen evidence value
  was changed.
- 2026-07-26: The same end-stage public-surface audit found a release-blocking
  metadata gap outside React rendering: search/social descriptions still
  advertised the withdrawn LOBO as leakage-free and the canonical/OG URL
  misspelled the production domain. Replaced every static description with the
  withdrawal status, corrected the URL to `stylometry.russkiykod.com`, and
  added an exact metadata gate so SSR-only checks cannot miss this class again.
- 2026-07-26: Strengthened the runtime gate after an adversarial check proved
  that the original full-app SSR exercised only the default framework tab: an
  undefined identifier inserted into a click-only chapter escaped detection.
  Added validated explicit chapter selection for tests, exact five-chapter
  registry coverage and chapter-specific render markers. Added a direct,
  version-pinned Babel scope scan over every JS/JSX source so event/effect-only
  undefined identifiers fail too; its negative fixture proves the gate detects
  the original failure class. Production `<App />` initialization is unchanged.
- 2026-07-26: Corrected the final Repro command contract: `run.sh all` never
  performs acquisition and must not regenerate a metric from the registered
  ineligible snapshot. Documented `fetch-classics` as a separate step, limited
  fresh `all`/benchmark execution to a registered content-safe corpus, and
  distinguished byte/provenance replay of stored historical artifacts from a
  new scientific run.
- 2026-07-26: A final independent executable probe then found that the prose
  contract was stronger than the code: `run_final`, GKF, both sweep strategies
  and `run_benchmark.py` could still reach fit workers without content
  isolation, while the benchmark could overwrite the historical site inputs.
  Introduced one sealed scientific-evaluation context, required it at every raw
  LOBO/GKF/panel kernel, moved the `all` eligibility check ahead of cache/train,
  and converted the legacy channel runner to a twice-gated, provenance-bound,
  atomic exploratory candidate writer. Added behavioral, caller-inventory,
  serialization, pre-CAP, frozen-writer and generator-hash regressions. The
  registered ineligible corpus now fails before any scientific cache, fit,
  model-bundle or candidate-output side effect.
- 2026-07-26: The exact-commit follow-up then broke the first context design
  adversarially: dataclass replacement and low-level attribute mutation could
  transplant its value seal, a serialized payload could claim synthetic
  authority, and the true-LOBO compatibility route could identify mutable bare
  rows before checkpointing. Replaced value authority with a revalidated
  identity registry, made deserialization repeat content isolation, proved
  derived rows are an ordered parent subsequence, split every synthetic test
  seam from disk-verified production routing, and revalidated true-LOBO
  identity around checkpoints/evaluator calls. The benchmark receipt now binds
  its runner, lock, package contract, runtime and clean Git state; the
  raw-kernel topology test also catches aliases and dynamic lookup. The focused
  adversarial/ablation/true-LOBO set passes all 61 tests without changing any
  frozen evidence or scientific approval.
- 2026-07-26: The independent site/release pass confirmed that the repaired
  application itself builds and contains no `MF1_CI`, but found five adjacent
  release-contract gaps. Public source archives now exclude 34 internal
  path-bearing historical records and scan exported content; Pages is chained
  to the exact successful release-CI main tip; package/README metadata no longer
  advertises the withdrawn snapshot as leakage-free; and provenance schema v2
  exact-binds every output field to verified sources. The self-mutable P0
  anchor and any published-history rewrite were recorded only as `UNSET` owner
  decisions. No frozen evidence, governance status or approval was changed, and
  no push, tag, release or deployment was performed.
- 2026-07-26: A second adversarial science pass demonstrated that internal
  restore/freeze helpers could still self-mint `disk_verified`, provenance
  metadata mutation escaped an identity-only snapshot, production true-LOBO and
  ablation accepted injected evaluators, synthetic resume trees lacked an
  authority mode, and the benchmark trusted mutable persistent caches without
  binding the loaded model or installed environment. Replaced boolean authority
  with disk-comparison capabilities bound to the complete provenance value,
  dropped authority on serialization and repeated disk verification in workers,
  versioned run/checkpoint/gate/artifact identities around production versus
  synthetic evaluators, and isolated all benchmark caches in an empty ephemeral
  workspace under exact environment/model attestation. The integrated
  adversarial/true-LOBO/ablation set passes all 69 tests; frozen evidence,
  governance state and owner dispositions remain unchanged.
- 2026-07-26: The first materialized exact-HEAD archive rehearsal exposed a
  checker-side false positive: importing the hygiene module generated a new
  `__pycache__` file whose compiled marker constants matched the scanner. The
  archive CLI now disables bytecode before its first local import, with an
  ordering regression. This changes no exported evidence and is rechecked from
  a newly materialized archive rather than deleting or mutating the failed
  rehearsal.
- 2026-07-26: Final independent science review of exact commit `8cefa51e`
  reproduced a fallback-model attestation bypass. The benchmark now rejects the
  actual resolved fallback, verifies installed model payloads against wheel
  RECORD, and binds stable live tokenizer/config/component state without
  treating normal vocabulary growth as drift. Focused adversarial regressions
  pass; frozen evidence, owner dispositions and confirmatory gates are
  unchanged.
- 2026-07-26: Final independent site/release review of exact commit `8cefa51e`
  found a source-drifted Sholokhov claim that propagated into package long
  descriptions. README generation now takes the 3/4 verdict and 0.595→0.035
  gradient from `docs/sholokhov_lobo.json`, labels the separate 0.455 control,
  and keeps the contradictory older P0-bound `sholokhov_rigor12.json` bytes
  frozen rather than silently rewriting evidence. The site and README active
  fields share the newer registered source, protected by a source-bound test.
- 2026-07-26: Science rereview of `5aa48f18` confirmed the fallback/live-state
  fix, then adversarially placed an unhashed `vectors.bin` beside one benign
  hashed member. Restricted the wheel-RECORD empty-digest exception to its
  exact installer-generated rows; unhashed model payloads now fail before an
  identity is issued. The canonical model's RECORD remains valid.
- 2026-07-26: Architecture rereview of `62ee922b` demonstrated that the
  remaining path-only pyc exception could still execute unrelated bytecode
  before post-load verification. Listed and unlisted runtime bytecode is now
  accepted only when its exact marshalled body derives from a verified source,
  and both primary and fallback model wheels are checked before import and
  resnapshotted after load. The canonical lg/md wheels and focused adversarial
  import-order tests pass; scientific approvals and frozen evidence remain
  untouched.
- 2026-07-26: Independent exact review of `08ace924` then separated the
  installed distribution from Python's import resolver with a benign
  same-named shadow package, and found the equivalent unrecorded native-module
  path inside a package root. Model loading now exact-binds its pre/post import
  spec, origin, search root and loaded module to the verified distribution,
  while package inventory rejects every other unrecorded payload. Primary and
  fallback shadow/native regressions fail before import; frozen evidence and
  all owner-controlled gates remain unchanged.
- 2026-07-26: Exact science/architecture review of `37063bc8` then preloaded a
  genuine model package and replaced only its mutable in-memory `load`
  callable. The unchanged wheel/spec/path was accepted while a blank pipeline
  was attributed to the genuine model. Scientific loading now bypasses package
  executable state and loads trained-pipeline data directly from the canonical
  RECORD-hashed init path, with identical pre/post binding on successful and
  failed primary/fallback loads. Full/NER route-order and real lg/md poisoned
  namespace regressions prove that no substituted package callable executes;
  frozen evidence and every owner-controlled gate remain unchanged.
- 2026-07-27: Reworked the public presentation as one reader-facing
  popular-science narrative. The first experiment is now explained once in
  plain Russian, raw governance statuses no longer reach rendered pages, and
  the «Тарас Бульба» chapter again tells the literary investigation before its
  methodological correction. README and metadata use the same framing; SSR
  rejects internal audit jargon. Generated scientific data, frozen evidence,
  registrations and owner-controlled gates were not changed.
- 2026-07-27: The first release-CI run after the public rewrite correctly
  stopped because its reviewed support-file inventory still named the removed
  withdrawal banner and pinned the previous public bytes. Rebound only those
  public support paths and digests to `ResearchUpdate` and the generated
  science-pop surface; the 280-file Python path-set identity and all frozen
  scientific evidence remain unchanged.
