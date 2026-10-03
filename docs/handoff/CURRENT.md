# Current Handoff

- Updated: **2026-10-03**.
- State: cloud study complete (992/992), results retrieved; temporary VM and boot disk deleted.
- Review baseline: `f312efb8f5fa291a961c337e6efa3b67cc247be4`.
- Implementation: the local commit containing this handoff and the remediation task; verify HEAD
  and worktree before acting. No push or publication is authorized.
- Active task: `docs/tasks/2026-10-03-gcp-validation.md`.
- Implementation task: `docs/tasks/2026-10-02-review-remediation.md`.
- Case corpus input: `docs/tasks/2026-10-02-case-corpus-census.md`.

## Parallel product cleanup draft

The owner requested a simpler project and a restrained technical rewrite of the article on
2026-10-03. Uncommitted changes are isolated in branch `product-cleanup`, worktree
`research/local/worktrees/product-cleanup/`; inspect that worktree before continuing this work.
The completed cloud run used `bb3760f`; main HEAD still matches that scientific source.
Cleanup removes unused code and prose-matching tests, updates README/ROADMAP, and prepares a
local site draft. These changes are not merged or published. Remaining product work includes
portable snapshot publication, ordinary training without mandatory Git, and the two case runs.
The readable static preview starts at `research/local/product-preview-20261003/index.html`.
Validation and remaining warnings are appended to the remediation task in that worktree.
The owner accepted the five navigation labels/order: «Как это работает», «Тихий Дон»,
«12 стульев», «Дневник Николая II», «Тарас Бульба». Keep this structure in the rewrite.

The temporary `n2-standard-128` VM (64 workers) completed the study on 2026-10-03.
The collector verified and retrieved results, then deleted the VM and its boot disk; independent
GCP exact-name lists confirmed both absent. Receipts and all 992 checkpoint records are retained in
`research/local/cloud_20261003/retrieval-lrbk7f_m/results/`. Collector completion is recorded in
`research/local/cloud_20261003/collector-state.json`. No further cloud polling or restart is needed.

## Verified implementation

R1–R8 are fixed: explicit deployment panels and abstention, full-text shingle coverage, point-only
macro-F1 uncertainty, whole-work controls, clean/split identity, embedding cache, bundle-token
configuration and chunk tails. A diagnostic LZMA baseline is implemented but is not registered in
the frozen evaluator and has no measured case accuracy. No calibrated open-set gate exists yet.

Cloud checks on bb3760f: **1343 passed, 1 skipped** (the separate live-golden module).
Git-free archive: **1335 passed, 9 expected skips**. Package build, fresh wheel smoke,
executable inventory, release/site provenance and site build passed. Separate historical capture
replay: **14 passed, 3 failed** (thread-pool identity and two numerical goldens). This is not an
all-tests-green result; fixtures/tolerances were not changed.

## Completed topic-validity study

Canonical aggregate: `research/evidence/topic_validity_lobo_v1/aggregate.json`.
Self-hash: `06b01f7a0fdf4e8e0440bd25ac5b499b943d77336813f78be2497f75d584290e`.
Execution took 12447 seconds; all 992 fits completed. Each comparison evaluates 248 held-out
works, with 43 metric labels and 47 probability classes:

| Training weighting | Current features | Topic-restricted features |
|---|---|---|
| A0: equal fragment weights | 216/248 (87.10%) | 218/248 (87.90%) |
| A4: equal work weights | 225/248 (90.73%) | 225/248 (90.73%) |

Transition counts are consistent with these totals; the canonical aggregate matches the retrieved
bytes. The runner provides accuracy and per-author transitions, not confidence intervals. This is
an exploratory comparison on the corrected corpus, not an authorship result for either novel.
The aggregate is local/uncommitted, and public site data/governance were not rewritten. Inspect
actual evidence before a model-selection or publication decision. Historical checkpoints remain
separate and must not be relabelled or resumed against this source.

## Case research

Targets: Bulgakov / «Двенадцать стульев», Sholokhov / «Тихий Дон». The corrected benchmark
bundle contains neither the Ilf–Petrov nor Sholokhov label and is not a ready case panel. The census
lists exact available work identities and source/edition gaps; new independent reference panels
and a calibrated verification benchmark still need preparation. R9 public wording was not changed.
