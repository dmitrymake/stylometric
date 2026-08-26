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

## 4. Scope

## 11. Verification plan

- Targeted: new topic-validity test plus function-word/factory routing tests.
- Relevant: v3.2 evaluator, packaged config/resources, governance and no-leakage suites.
- Full: `.venv/bin/python -m pytest tests -q -p no:cacheprovider`.
- Hygiene: py-compile for the test, exact nodeid collection, executable inventory, release hygiene,
  diff/status and protected-surface check.
- Negative: MFW-restored challenger, genuine function-word substitution, wrong cell route, changed
  non-vectorizer estimator parameter, and repeat digest drift.

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
