# Current Handoff

- Updated: **2026-10-03**.
- State: cloud study complete (992/992), results retrieved; temporary VM and boot disk deleted.
- Product baseline: `5582d23`; final source/site review: `786b3df3311c06a7ff30030128956b3b3650416b`.
- Implementation: the local commit containing this handoff and the remediation task; verify HEAD
  and worktree before acting. No push or publication is authorized.
- Completed task: `docs/tasks/2026-10-03-product-hardening.md`. GCP task is also complete.
- Implementation task: `docs/tasks/2026-10-02-review-remediation.md`.
- Case corpus input: `docs/tasks/2026-10-02-case-corpus-census.md`.

## Product hardening complete locally

Owner authorised the product/methodology/editorial hardening on 2026-10-03 and requested a final
cold Astra review. Implementation commits: `70be5c5`, `c96c6f7`; final editorial corrections:
`786b3df`. These commits have been fast-forwarded into local main; verify HEAD and tree.
There is no push/publication authorisation.

The first cold review found target/reference leakage, overwritten result versions and unnecessary
binding of output settings to a model. All three were fixed and independently reproduced as closed.
The final review also corrected excessive article claims about topic independence and author counts.
Astra's final scores: correctness 9, methodology 8.5, simplicity 8, usability 8.5, article 8.5.
No blocker for ordinary local use was found; a uniform 9/10 is not claimed.
Both exact reviews are retained in the main checkout under
`research/local/reviews/astra-product-hardening-20261003.md` and
`research/local/reviews/astra-final-product-20261003.md`.

Final source regression: **1390 passed, 8 skipped**, 84.64 s. Wheel/sdist build and ten real
installed-wheel scenarios outside Git passed. These include two models for one target, unchanged
old results, output/top-k changes without fitting, near-copy rejection before fitting, and reading
an old result without its original texts/model/YAML. Final site build/no-undef/SSR5, provenance94/2,
deterministic generation and repository gates passed. Native Windows/macOS are not runtime-tested;
the separate historical replay failures below remain recorded. Full details are in the completed task.

The accepted menu is «Как это работает», «Тихий Дон», «12 стульев», Дневник Николая II,
«Тарас Бульба». Read the current local preview at
`research/local/product-preview-20261003/index.html`. Historical case panels still require separate
corpus/edition work and fresh measurements; this product work has not produced new novel attributions.

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
