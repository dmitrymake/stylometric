# Current Handoff

- Updated: **2026-10-03**.
- State: cloud study complete (992/992), results retrieved; temporary VM and boot disk deleted.
- Review baseline: `f312efb8f5fa291a961c337e6efa3b67cc247be4`.
- Implementation: the local commit containing this handoff and the remediation task; verify HEAD
  and worktree before acting. No push or publication is authorized.
- Active task: `docs/tasks/2026-10-03-product-hardening.md`. GCP task is complete.
- Implementation task: `docs/tasks/2026-10-02-review-remediation.md`.
- Case corpus input: `docs/tasks/2026-10-02-case-corpus-census.md`.

## Active product hardening candidate

Owner authorisation: full product/methodology/editorial hardening toward 9/10 on 2026-10-03,
followed by independent cold Astra review. Work is local; no push/publication is authorised.
Candidate branch `product-cleanup` is in `research/local/worktrees/product-cleanup/`, based on
`5582d23`. Initial candidate `70be5c5` completed integration checks (1362 passed, 8 skipped),
real wheel workflows outside Git, site build/SSR and desktop/mobile Chromium checks.

The first cold Astra review reproduced three defects: target/reference near-duplicates could
survive under the same author; a second model overwrote the first result for one target; report
location/top-k were bound unnecessarily to model identity. Corrections are complete in the candidate worktree; final regression is running.
Do not call the initial candidate accepted or rated 9/10. The exact review is
`research/local/reviews/astra-product-hardening-20261003.md` in the main checkout.
Installed-wheel regression covers the three real reproductions, including historical result reading
without its original inputs/model. Final validation and independent closure follow the corrections. Details and evidence:
`docs/tasks/2026-10-03-product-hardening.md`.

The owner accepted the five navigation labels/order: «Как это работает», «Тихий Дон»,
«12 стульев», «Дневник Николая II», «Тарас Бульба». Keep this structure. The local preview is
`research/local/product-preview-20261003/index.html`; refresh it from the accepted candidate.
Historical case panels still require separate corpus/edition work and fresh measurements.

The temporary `n2-standard-128` VM (64 workers) completed all 992 fits on 2026-10-03.
Its scientific source is `bb3760f9b9237a9fbadfaf847dca56bba8024161`, not the product candidate.
Results were verified/retrieved and the VM plus boot disk deleted; independent exact-name GCP
lists confirmed both absent. Retained receipts/checkpoint:
`research/local/cloud_20261003/retrieval-lrbk7f_m/results/`; collector state:
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
| A0: fragment-fitted features, balanced author classes | 216/248 (87.10%) | 218/248 (87.90%) |
| A4: work-fitted features, work weights within author | 225/248 (90.73%) | 225/248 (90.73%) |

Transition counts are consistent with these totals; the canonical aggregate matches the retrieved
bytes. The runner provides accuracy and per-author transitions, not confidence intervals. This is
an exploratory comparison on the corrected corpus, not an authorship result for either novel.
The canonical aggregate is committed in `5582d23`; local site/governance updates are in the product
candidate. Nothing has been published. Inspect actual evidence before a model-selection decision. Historical checkpoints remain
separate and must not be relabelled or resumed against this source.

## Case research

Targets: Bulgakov / «Двенадцать стульев», Sholokhov / «Тихий Дон». The corrected benchmark
bundle contains neither the Ilf–Petrov nor Sholokhov label and is not a ready case panel. The census
lists exact available work identities and source/edition gaps; new independent reference panels
and a calibrated verification benchmark still need preparation. The product candidate revises the local article; it does not replace the historical case measurements.
