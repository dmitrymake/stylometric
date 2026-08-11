# Paired-audit v3.2 frozen-input reconciliation

## Metadata

- Task ID: `2026-08-11-v3-2-frozen-input-reconciliation`
- Status: active
- Owner: repository owner
- Created: 2026-08-11
- Baseline commit: `c65ca28ab2c387a9e4deedb071f778cbe0ebeb23`
- Target: attached `main`; local commits only
- Type / Size / Risk: implementation / M / R2
- Risk flags: mutation, scientific-contract, dependency-version
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: feature-extraction runtime identity; research-governance metadata; release
  dependency contract
- Standard: `docs/agentic/STANDARD.md` v1.3

## 1. Goal and observed facts

Make the already accepted v3.2 candidate constructible and verifiable from the supported locked
environment without substituting mutable status prose or patching runtime metadata in memory.

- Design-freeze commit `adc395638e186b290b83caac9bf71eb42f4ed89e` contains protocol bytes with
  SHA-256 `02341845749431ba99fde0cac4335dcce86f9d0a3389c6c0382f6bcf077b6334`; the accepted
  candidate binds those bytes.
- Later commit `c6b131b9` changed only protocol status prose, producing current document SHA-256
  `4efcc7b752eedb39a478380d884f302b714a4e58b26807e4cc7dffb551958147`.
- The preparation CLI currently hashes the mutable current document and therefore reconstructs a
  different candidate.
- Real corpus manifests bind chunker identity
  `23361b5f07514f15b681e575a685d1119f38a9982facb102a8b692f8180c1963`, exactly derived by
  the existing chunker contract with spaCy 3.8.14. `requirements.lock` and the rebuilt environment
  currently use spaCy 3.8.11 and derive `492d7ea7...`.
- PyPI publishes spaCy 3.8.14 for CPython 3.11. No new package is required.

Observable result: one versioned frozen protocol identity is owned by the v3.2 preparation
capability; the CLI and evaluator context use it without hashing current status prose;
`requirements.lock` installs spaCy 3.8.14; two ordinary real context loads reproduce the accepted
identities without monkeypatching and without fit/predict.

## 2. Scope and non-goals

### In scope

- `src/stylo/eval/paired_audit/corrected_v3_2.py`
- `src/stylo/eval/paired_audit/evaluator_v3_2.py`
- `scripts/evaluation/prepare_corrected_paired_audit_v3_2.py`
- `requirements.lock`
- directly owning v3.2, environment and governance tests
- necessary preparation/governance bindings, executable inventory, task and handoff
- local `.venv` reconciliation to the tracked lock for verification

### Out of scope

- Editing current protocol semantics or the prepared bundle/corpus/fold bytes.
- Changing config, chunker algorithm, model packages, feature math, evaluator/receipt math, matrix,
  registry, RunPlan, freeze, preflight, authorization, execution, result or publication.
- New dependency, frozen-protocol document copy, generic registry/framework, runtime state, public
  CLI or external write other than downloading the exact approved package version into local `.venv`.

## 3. Domain contract

- Producer: design-freeze protocol Git blob plus frozen corpus work manifests. Consumer: v3.2
  preparation and evaluator context.
- Grain/cardinality: one protocol byte identity per v3.2 candidate; one chunker runtime identity per
  frozen work manifest; exact 1:1 identity reconciliation, no fallback.
- Time/history: protocol status prose may evolve, but accepted candidate provenance remains the
  immutable design-freeze blob. Historical manifests are not rewritten.
- NULL/delete/late arrival: none; missing or differing identities fail closed.
- Verification: exact constant/source-commit hash, CLI/context routing, locked runtime and chunker
  hash, real candidate/context identities, negative current-protocol substitution test.

## 4. Frozen acceptance

- [ ] AC-01 One capability-owned constant equals `023418...` and names design-freeze commit
  `adc39563`; CLI and evaluator consume it without a duplicate literal or current-document hash.
- [ ] AC-02 Preparation with supported inputs reproduces candidate `ff620b05...`, manifest
  `a2dc0c4a...`, LOBO `117b8ec9...` and RuAA `d8428290...`; passing current status-prose SHA does
  not become accepted provenance.
- [ ] AC-03 `requirements.lock` pins exactly spaCy 3.8.14, local runtime reports 3.8.14, dependency
  validation passes, and `chunker_config_hash(default cfg)` equals `23361b5f...`.
- [ ] AC-04 Two real read-only context loads without metadata patching both produce `2805aff9...`,
  exact 47/252 and 22/134 universes, and never reach fit/predict.
- [ ] AC-05 Candidate/bundle/fold/applicability identities and bytes remain unchanged; production
  registry stays empty, freeze unapproved, preflight/authorization absent, execution hard-disabled,
  publication not authorized.
- [ ] AC-06 Focused/full tests, py_compile, dependency validation, inventory, provenance,
  checkout/archive release hygiene and diff check pass.

## 5. Complexity tripwires and review

- Existing production subsystems: at most two (`evaluation/paired-audit`, dependency lock); existing
  production files: at most three named Python paths; new production files/public entrypoints: 0.
- Production LOC added ceiling: 45; tests added ceiling: 90; task/governance/handoff added ceiling:
  260. Net production LOC target: non-positive excluding the required identity constant/comment.
- Dependency count change: 0; allowed version change: only `spacy 3.8.11 -> 3.8.14`. Any required
  transitive-lock, model-package, Python or config change is a STOP/reclassification.
- No framework, abstraction layer, generic registry, state, schema or prepared-byte change.
- One bounded adversarial review and at most one correction pass.
- STOP on candidate/context identity drift, real fit/predict, model incompatibility, unsupported
  consumer, non-status protocol semantic divergence, registry/freeze/preflight/execution change,
  private-data write into Git, or new R3 factor.

## 6. Verification and result

- Targeted: v3.2 corrected/evaluator tests, environment/lock tests, governance and inventory.
- Runtime: exact installed versions, `pip check`, spaCy validate, chunker hash, temporary real bundle,
  two context loads with forbidden fit/predict.
- Regression: full pytest; physical-file py_compile; provenance; checkout/Git-free archive gates.
- Review only AC/invariants/regression/tripwires. Existing absence of freeze/control-plane is not a
  blocker and must not be implemented here.

Result: pending.
