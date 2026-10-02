# Current Handoff

- Updated: **2026-10-02**.
- State: review remediation implemented and validated; full topic-validity execution prepared.
- Review baseline: `f312efb8f5fa291a961c337e6efa3b67cc247be4`.
- Implementation: the local commit containing this handoff and the remediation task; verify HEAD
  and worktree before acting. No push or publication is authorized.
- Active task: `docs/tasks/2026-10-02-review-remediation.md`.
- Case corpus input: `docs/tasks/2026-10-02-case-corpus-census.md`.

## Verified implementation

R1–R8 are fixed: explicit deployment panels and abstention, full-text shingle coverage, point-only
macro-F1 uncertainty, whole-work controls, clean/split identity, embedding cache, bundle-token
configuration and chunk tails. A diagnostic LZMA baseline is implemented but is not registered in
the frozen evaluator and has no measured case accuracy. No calibrated open-set gate exists yet.

Full regression: 1333 passed, 4 skipped, 2 warnings. The opt-in real v3.2 context test then passed
separately. Package build, executable inventory and release/site provenance passed. The separately
enabled live-golden module had 15 passed and two environment-fingerprint failures: its capture
versions differ from requirements.lock. No full capture-environment parity PASS is claimed.

## Topic-validity execution

The owner authorized execution after the review. The exact 248-fold A0/A4 current/topic_strict
study remains exploratory. New chunker v2 does not require re-chunking the frozen bundle: its
recorded old identity is explicitly validated by the evaluator.

The old checkpoint contains **70 A0/current fits**, unlike the historical ledger narrative of
260 fits. It is preserved and cannot resume against changed source. The new run uses:

- `research/local/topic_validity_lobo_v1.20261002.repaired.process.json` for launch PID/commit/source;
- `research/local/topic_validity_lobo_v1.20261002.repaired.log` for progress;
- `research/local/topic_validity_lobo_v1.20261002.repaired.checkpoint.json` for resumable fits;
- `research/evidence/topic_validity_lobo_v1/aggregate.json` only after all 992 fits finish.

Check these ignored-local records and process state to determine whether launch occurred and
whether execution is still running. Do not infer completion from this handoff. The full command
and environment are in the active task. Preserve the old checkpoint; never use `--fresh` on it.
While fits run, do not mutate `src/stylo`, the runner, runtime or HEAD. Stop/resume through the
runner's signal/checkpoint mechanism when needed; do not mix records from different identities.

After completion, independently audit the aggregate before any model-semantics decision.
Production registration, confirmatory execution and publication remain separate uncompleted work.
The normative scientific ledger is still `research/governance/status_ledger.json`; historical
evidence and public claims were not rewritten by the implementation task.

## Case research

Targets: Bulgakov / «Двенадцать стульев», Sholokhov / «Тихий Дон». The corrected benchmark
bundle contains neither the Ilf–Petrov nor Sholokhov label and is not a ready case panel. The census
lists exact available work identities and source/edition gaps; new independent reference panels
and a calibrated verification benchmark still need preparation. R9 public wording was not changed.
