# Stylo topic-validity gate

## Metadata

- Task ID: `2026-08-23-stylo-topic-validity-gate`
- Status: complete
- Owner: Dmitry Purtov (repository owner)
- Created: 2026-08-23
- Baseline commit: `8d75313fee2c8099de3ec04a7d6bbb52c8ce2deb`
- Target: attached `main`; local commits only
- Type / Size / Risk: research / M / R2
- Risk flags: mutation, scientific-contract
- Primary domain: evaluation/paired-audit
- Allowed cross-domain: feature-extraction (read-only behavior); research-governance (gate metadata)

## 1. Goal and decision

- Observable result: an executable synthetic oracle determines whether the active `stylo` MFW
  channel can learn label-correlated topical nouns, and whether the already defined `topic_strict`
  vectorizer removes that exact dependency without being laundered into an official v3.2 identity.
- Consumer: the owner decision that must precede any evaluator registration or real corrected-corpus
  topic-validity comparison.
- Decision unlocked: either authorize a separately versioned aggregate-only challenger study, or keep
  evaluator registration blocked as not factology-ready.
- Success is not real-corpus accuracy evidence, a claim that `topic_strict` is pure idiolect, model or
  protocol replacement, production registration, freeze, execution, headline, or publication.

The operator's 2026-08-23 instruction selects model/fact accuracy over speculative hardening and
authorizes this bounded synthetic research iteration. It does not explicitly authorize raw/private
corpus access or real v3.2 fit/predict; those remain separate gates.

## 2. Confirmed starting facts

| ID | Fact | Evidence | Confidence |
|---|---|---|---|
| F-01 | Active/package default and R1 primary configure `function_words.mode="mfw"`. | `configs/default.yaml:49-53`; packaged-resource parity test; `lobo_vnext_models.py:68-113` | high |
| F-02 | MFW uses unrestricted `CountVectorizer` / work-level `max_features`, so content words are eligible. | `function_words.py:104-119` | high |
| F-03 | `topic_strict=True` canonically changes function words to `fixed_list` and disables syntax `pos_ratios`/`lexical_richness`. | `registry.py:28-65`; `vectorizer.py:28-38` | high |
| F-04 | No v3.2 A0-A4 `stylo` factory passes `topic_strict`; existing YAML `topic_control` is not an executable selector. | `lobo.py:445-461,525-533`; bounded active-code search | high |
| F-05 | On an exact work-balanced two-author synthetic panel, nouns `космос/компас` enter MFW and produce clean/swapped predictions `[0,1]`/`[1,0]` in every A0-A4 route; a test-only topic-strict vectorizer is matrix/prediction invariant. | bounded Python reproduction, no corpus data | high |
| F-06 | Existing strict aggregate reports genre AUC 0.835 and explicitly disclaim pure idiolect; this is not a current content-word counterfactual or v3.2 classifier result. | `docs/audit_genre_crossauthor.json`; `heterogeneity.py:73-85` | medium |
| F-07 | Governance records no v3.2 fit/predict authorization; registry is empty and freeze is `None`. | ledger + executable constants | high |

## 3. Unknowns and assumptions

| ID | Unknown | Why it matters | Resolution | Blocking? |
|---|---|---|---|---|
| U-01 | Magnitude/direction of the MFW effect on corrected LOBO-248. | Synthetic causality cannot estimate real accuracy or author flips. | Separate explicitly authorized aggregate-only run; not this task. | blocks factology-ready/registration, not Phase A |
| U-02 | Whether full `topic_strict` improves validity without unacceptable accuracy loss. | It changes MFW plus two syntax subblocks and residual genre signal remains possible. | Future paired current/challenger decision; accuracy alone cannot select semantics. | yes for model replacement |
| U-03 | Stable challenger execution/receipt identity. | Current v3.2 receipt hardcodes one factory route and config identity. | Future versioned challenger task; never reuse official receipt/registry here. | yes for real run |

## 4. Scope

### In scope

- One new focused test file containing synthetic fixtures and test-only challenger construction.
- Exact current A0-A4 `stylo` factory routes with the same classifier/loss/F/R wrappers.
- The existing `StyloVectorizer.from_config(..., topic_strict=True)` as an evaluation-only challenger.
- One research-governance requirement/nodeid, a truthful roadmap/ledger limitation sentence, necessary
  byte inventory bindings, this task result, and handoff.

### Out of scope

- Any production Python/config/resource/factory/evaluator/receipt/protocol/applicability/model registry
  change or new runtime/CLI/runner.
- Raw/private corpus reads, real v3.2 fit/predict/checkpoints/results, RuAA, LOBO-248, external data or
  services, and persisted vocabularies/text/work IDs/row-level predictions.
- Model selection/replacement, production registration, freeze/preflight/authorization, headline,
  site/public claim, push, deploy, or publication.
- Withdrawn/historical artifacts as design/evidence sources and the separate Sholokhov wording task.

### Stop / reclassification

- Any need for production code, official identity/schema, raw/private data access, real fit/predict,
  external write, R3 factor, new dependency/state/public entry point, or tripwire crossing.
- Challenger differs from current routes beyond `topic_strict` vectorizer semantics, or the synthetic
  counterfactual cannot isolate content tokens.
- Unresolved blocker after the one allowed correction pass.

## 5. Frozen acceptance and complexity tripwires

- Production subsystems/files/LOC changed: 0 / 0 / 0.
- New test files: at most one; test LOC added ceiling: 220.
- Task/governance/roadmap/handoff/inventory LOC added ceiling: 380.
- New dependency/framework/layer/persistent state/generic abstraction/public or operational entrypoint: 0.
- Frozen configs/resources/protocol/model/evaluator/receipt/corpus/fold/result/site bytes: unchanged.
- Review: one independent scientific review; at most one bounded correction pass.
- Any crossing requires stop/reclassification; budgets do not expand automatically.

## 6. Domain and scientific contract

- Producer: active config + function-word feature contract + A0-A4 `stylo` factories.
- Consumer: evaluator-registration/model-semantics decision.
- Grain: one synthetic `(noun_pair, repetition_strength, cell, arm, clean_or_swapped)` observation.
- Keys/cardinality: exactly two authors, at least two independent train works per author, one held-out
  synthetic work per author; identical ordered samples and class order for current/challenger.
- Temporal semantics: current baseline behavior only; no reuse as future model identity.
- NULL/delete/late arrival: none; a missing cell/arm/negative invalidates the oracle.
- Challenger delta: `topic_strict=True` only. Pipeline class, scaler, classifier, seed, loss wrapper,
  enabled feature set, F/R cell, train/test rows and labels remain equal.
- Persistence: test assertions and aggregate task result only; no texts, vocabulary, work IDs,
  probabilities, predictions, cache/checkpoint/result artifact are committed.

## 7. Invariants

- INV-01: content-token swaps preserve text length, token positions/counts, punctuation, POS/morph
  role, labels, work grouping, train/test split, and every non-topic input controlled by the fixture.
- INV-02: current MFW and strict arms use exact matching estimator/loss/F/R routes per cell.
- INV-03: `topic_strict` is an explicitly labelled challenger, never official v3.2 evidence.
- INV-04: fixed-list invariance is proved alongside a genuine-function-word sensitivity control so a
  zero/unused block cannot masquerade as success.
- INV-05: mechanism confidence and corrected-corpus effect confidence remain separate.
- INV-06: registry/freeze/authorization/execution/publication state remains unchanged.

## 8. Phases and gates

### Phase A — synthetic mechanism research

- Portable block oracle: at least two non-function noun pairs and two repetition strengths.
- Full supported-route oracle: A0-A4 current versus test-only topic-strict on one controlled Russian
  pair; exact matrix invariance and prediction flips/repeat determinism.
- Gate: implementation allowed by the operator instruction after the read-only reproduction in F-05.

### Phase B — real aggregate comparison

- Not authorized in this task. A future exact LOBO-248 A0/A4-only study requires a new versioned
  challenger identity, bounded aggregate schema, independent review, and explicit owner approval or
  R3b reclassification before any fit/predict.

### Phase C — review/close

- One clean-context review against frozen AC/invariants and original baseline; no edits.
- Blocking only for invalid isolation, route drift, unsupported scientific claim, regression,
  authorization/safety violation, or tripwire crossing.
- At most one correction, then frozen verification without a second review loop.

## 9. Evidence and falsifiers

| Claim | Reproduction | Falsifier | Confidence |
|---|---|---|---|
| MFW admits arbitrary topical nouns | inspect fitted feature names and matrices | nouns absent in all current routes | high |
| MFW drives the controlled decision | clean vs noun-swapped predictions | unchanged predictions or non-unit flip/stress effect | high |
| strict removes that exact dependency | exact clean/swapped feature equality and prediction stability | any matrix/prediction change | high |
| strict block is live | replace genuine function word at fixed content | unchanged strict vector | high |
| real effect remains unknown | absence of authorized real result | any valid authorized paired aggregate result | high |

## 10. Frozen acceptance criteria

- [ ] AC-01 Exact active/package config and every current A0-A4 `stylo` route expose MFW.
- [ ] AC-02 For controlled non-function nouns, current MFW vocabulary/matrices carry the noun signal;
  clean accuracy is 1.0, swapped accuracy 0.0, and prediction flip rate 1.0.
- [ ] AC-03 In every A0-A4 route the test-only strict arm has identical clean/swapped matrices and
  unchanged predictions; `topic_stress_delta = 1.0` for the frozen primary fixture.
- [ ] AC-04 At least two noun pairs/two strengths reproduce the block mechanism; a genuine function
  word change remains observable in fixed-list vectors.
- [ ] AC-05 Current/challenger estimator classes and non-vectorizer parameters match exactly; repeat
  outputs are deterministic; restoring MFW to challenger breaks invariance.
- [ ] AC-06 Governance maps exact tests and states: causal mechanism confirmed; real corrected-corpus
  magnitude and model decision unmeasured; evaluator not factology-ready for registration.
- [ ] AC-07 Focused/relevant/full tests and hygiene pass; warnings/skips/cannot-run are separate.
- [ ] AC-08 Production code/config/identities and registry/freeze/authorization/execution/publication
  state remain unchanged; no real/private corpus operation occurs.

## 11. Verification plan

- Targeted: new topic-validity test plus function-word/factory routing tests.
- Relevant: v3.2 evaluator, packaged config/resources, governance and no-leakage suites.
- Full: `.venv/bin/python -m pytest tests -q -p no:cacheprovider`.
- Hygiene: py-compile for the test, exact nodeid collection, executable inventory, release hygiene,
  diff/status and protected-surface check.
- Negative: MFW-restored challenger, genuine function-word substitution, wrong cell route, changed
  non-vectorizer estimator parameter, and repeat digest drift.

## 12. R3 safety and approval

- Phase A uses generated synthetic strings only and is R2.
- Any protected corrected-corpus fit/predict is a stop. Before it, a new task must record exact access,
  load/output minimization, aggregate schema, abort/cleanup, and owner approval/reclassification.

## 13. Documentation and review impact

- ADR/domain/runbook: none; no model/architecture/operation changes. A future model selection may need
  a scientific-semantic ADR.
- Governance/roadmap: limitation/gate only, no readiness/registration status promotion.
- Handoff: route the separately authorized aggregate study and retain the reporting wording backlog.

## 14. Result

- Candidate commit: `60c61101639bc954c489dca2d54f2efd5b4b0142` on attached `main`.
- Synthetic mechanism: for two verified non-function noun pairs at two repetition strengths, the
  work-balanced MFW block admitted both nouns, produced clean/swapped predictions `[0,1]`/`[1,0]`,
  and the fixed-list block excluded them with exact swapped-vector/prediction invariance. A genuine
  function-word substitution changed the fixed-list vector, ruling out a dead control.
- Supported routes: every current `stylo` A0-A4 route exposed `mode="mfw"`. With identical factory
  wrapper, scaler, classifier, seed, loss and F/R cell, the test-only `topic_strict` vectorizer yielded
  exact clean/swapped matrix equality and unchanged predictions. Current clean/swapped accuracy was
  `1.0/0.0`, current/strict flip rate `1.0/0.0`, and `topic_stress_delta=1.0` in every cell; two
  independent cache-root runs had identical aggregate digests.
- Independent review: **PASS**, zero blockers, no correction pass. The reviewer separately verified
  both noun pairs as matching `NOUN`, inanimate/nominative/masculine/singular tokens with matching
  dependency roles and confirmed the route equality and stress math. A stronger future exact-vector
  digest was non-blocking because current exact matrix/prediction assertions already satisfy AC-05.
- Verification: targeted topic/governance `25 passed`; relevant `209 passed, 1` real-bundle skip;
  full regression exited zero with 1,651 collected tests, three executed environment/data skips plus
  the separately collection-skipped live-golden module, and two pre-existing invalid-escape
  deprecation warnings. Test py-compile, exact nodeid collection, diff check, release hygiene, and the
  300-path executable inventory (`10ebce7b...`) passed.
- Tripwires: production files/LOC `0/0`; one new test file, +162/-1 test LOC; +282/-46
  task/governance/roadmap/handoff/inventory LOC; no dependency, framework, state, generic abstraction, public
  entrypoint, config/resource/protocol/model/evaluator/receipt/corpus/fold/result/site byte change.
- Evidence conclusion: causal MFW content-token sensitivity is high confidence. Its magnitude and
  direction on corrected LOBO-248 are **unknown**; no protected corpus was read and no real fit,
  prediction or artifact was produced. Therefore the current evaluator is not factology-ready for
  registration. The next bounded A0/A4 aggregate-only challenger study requires a separately
  versioned identity and explicit owner authorization or R3b reclassification.
- Final control state: evaluator registry `{}`, freeze `None`, preflight/authorization absent,
  execution hard-disabled, headline/publication not authorized. Review/correction used: `1/1`, `0/1`.

## 15. DoD references

- [x] DOD-01 through DOD-05
- [x] DOD-07 through DOD-12
- DOD-06/13/14/15: N/A unless reclassified.
