# Paired-audit v3.2 frozen-input reconciliation

## Metadata

- Task ID: `2026-08-11-v3-2-frozen-input-reconciliation`
- Status: complete
- Owner: repository owner
- Created: 2026-08-11
- Baseline commit: `c65ca28ab2c387a9e4deedb071f778cbe0ebeb23`
- Target: attached `main`; local commits only
- Type / Size / Risk: implementation / M / R2
- Risk flags: mutation, scientific-contract, frozen-runtime-identity
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: feature-extraction runtime identity; research-governance metadata; release
  dependency contract

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
- A dry-run resolver showed that upgrading to spaCy 3.8.14 would change fourteen transitive
  packages. The owner rejected that expansion: constructing a read-only context does not execute
  the chunker, and a future scientific run must independently bind its actual locked environment.

Observable result: one versioned frozen protocol identity is owned by the v3.2 preparation
capability; the CLI and evaluator context use it without hashing current status prose;
the evaluator supplies the exact frozen manifest identity to the canonical read-only loader while
`requirements.lock` remains unchanged at spaCy 3.8.11; two ordinary real context loads reproduce
the accepted identities without monkeypatching and without fit/predict.

## 2. Scope and non-goals

### In scope

- `src/stylo/eval/paired_audit/corrected_v3_2.py`
- `src/stylo/eval/paired_audit/evaluator_v3_2.py`
- `src/stylo/workdoc.py` (optional exact expected manifest identity for read-only loading only)
- `scripts/evaluation/prepare_corrected_paired_audit_v3_2.py`
- directly owning v3.2, environment and governance tests
- necessary preparation/governance bindings, executable inventory, task and handoff
- local locked-environment validation

### Out of scope

- Editing current protocol semantics or the prepared bundle/corpus/fold bytes.
- Changing config, chunker algorithm, model packages, feature math, evaluator/receipt math, matrix,
  registry, RunPlan, freeze, preflight, authorization, execution, result or publication.
- Any dependency/environment change, frozen-protocol document copy, generic registry/framework,
  runtime state, public CLI or external write.

## 3. Domain contract

- Producer: design-freeze protocol Git blob plus frozen corpus work manifests. Consumer: v3.2
  preparation and evaluator context. The loader validates existing chunk bytes; it does not invoke
  spaCy or reconstruct the historical chunking process.
- Grain/cardinality: one protocol byte identity per v3.2 candidate; one chunker runtime identity per
  frozen work manifest; exact 1:1 identity reconciliation, no fallback.
- Time/history: protocol status prose may evolve, but accepted candidate provenance remains the
  immutable design-freeze blob. Historical manifests are not rewritten.
- NULL/delete/late arrival: none; missing or differing identities fail closed.
- Verification: exact constants/source identities, CLI/context routing, unchanged locked runtime,
  exact manifest validation, real candidate/context identities, negative protocol/manifest tests.

## 4. Frozen acceptance

- [x] AC-01 One capability-owned constant equals `023418...` and names design-freeze commit
  `adc39563`; CLI and evaluator consume it without a duplicate literal or current-document hash.
- [x] AC-02 Preparation with supported inputs reproduces candidate `ff620b05...`, manifest
  `a2dc0c4a...`, LOBO `117b8ec9...` and RuAA `d8428290...`; passing current status-prose SHA does
  not become accepted provenance.
- [x] AC-03 `requirements.lock` and local runtime remain at spaCy 3.8.11; dependency validation
  passes; the v3.2 context supplies exact frozen chunker identity `23361b5f...` to the canonical
  loader, which accepts only matching manifests and never represents it as current runtime state.
- [x] AC-04 Two real read-only context loads without metadata patching both produce `2805aff9...`,
  exact 47/252 and 22/134 universes, and never reach fit/predict.
- [x] AC-05 Candidate/bundle/fold/applicability identities and bytes remain unchanged; production
  registry stays empty, freeze unapproved, preflight/authorization absent, execution hard-disabled,
  publication not authorized.
- [x] AC-06 Focused/full tests, py_compile, dependency validation, inventory, provenance,
  checkout/archive release hygiene and diff check pass.

## 5. Complexity tripwires and review

- Existing production subsystems: at most two (`evaluation/paired-audit`, canonical corpus loader);
  existing production files: at most four named Python paths; new production files/public
  entrypoints: 0.
- Production LOC added ceiling: 45; tests added ceiling: 90; task/governance/handoff added ceiling:
  260. Net production LOC target: non-positive excluding the required identity constant/comment.
- Dependency count/version change: 0. Any dependency, model-package, Python, config or environment
  mutation is a new STOP/reclassification.
- No framework, abstraction layer, generic registry, state, schema or prepared-byte change.
- One bounded adversarial review and at most one correction pass.
- STOP on candidate/context identity drift, real fit/predict, model incompatibility, unsupported
  consumer, non-status protocol semantic divergence, registry/freeze/preflight/execution change,
  private-data write into Git, or new R3 factor.

## 6. Verification and result

- Targeted: v3.2 corrected/evaluator tests, environment/lock tests, governance and inventory.
- Runtime: exact installed/locked spaCy 3.8.11, `pip check`, spaCy validate, frozen manifest identity,
  temporary real bundle, two context loads with forbidden fit/predict.
- Regression: full pytest; physical-file py_compile; provenance; checkout/Git-free archive gates.
- Review only AC/invariants/regression/tripwires. Existing absence of freeze/control-plane is not a
  blocker and must not be implemented here.

Result: PASS at review-candidate commit `a4908d08` with zero correction passes.

- Frozen identities are capability-owned and single-routed: protocol `02341845749431ba99fde0cac4335dcce86f9d0a3389c6c0382f6bcf077b6334`
  and corpus chunker manifest `23361b5f07514f15b681e575a685d1119f38a9982facb102a8b692f8180c1963`.
  The mutable current protocol document remains `4efcc7...` and is not candidate provenance.
- Preparation reproduced candidate `ff620b05...`, corpus manifest `a2dc0c4a...`, LOBO `117b8ec9...`
  and RuAA `d8428290...`. Two ordinary real read-only context loads produced `2805aff9...`, exact
  47/252 and 22/134 universes, and never reached the forbidden fit/predict hook.
- A dry-run spaCy 3.8.14 resolver expansion was rejected before environment mutation. The tracked
  lock and runtime remain 3.8.11; `uv pip check` verified 61 compatible packages and local loading
  verified `ru_core_news_lg` 3.8.0 (`>=3.8.0,<3.9.0`). The network-dependent spaCy compatibility
  table was unavailable, which is not substituted for local evidence.
- Focused tests, full pytest, physical-file py_compile, provenance (93 sources + one output),
  299-path executable inventory, checkout and Git-free archive hygiene/inventory/provenance, and
  diff check passed. Full pytest retained two pre-existing invalid-escape warnings.
- The single clean detached review of exact `a4908d08` reproduced the frozen hashes, context,
  universes, empty production registry and no-fit/predict invariant: PASS, no blocker. This was the
  bounded reconciliation review, not a second review of the original evaluator acceptance.
- Tripwires held: four existing production paths, +23/-12 production LOC, +57 test LOC, no new
  dependency, module, entrypoint, framework, state or prepared-byte change. Freeze, preflight,
  authorization, execution, headline and publication remain unchanged and non-authorizing.
