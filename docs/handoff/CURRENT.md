# Current Handoff

- Updated: **2026-10-03**.
- State: cloud study complete (992/992), results retrieved; temporary VM and boot disk deleted.
- Product baseline: `5582d23`; final source/site review: `786b3df3311c06a7ff30030128956b3b3650416b`.
- Latest owner instruction (2026-10-03): push the current Stylo commits to GitHub
  and allow the existing automatic Pages deployment. Task:
  `docs/tasks/2026-10-03-publish-stylo.md`; actual delivery receipt is retained in
  `research/local/publish-20261003/receipt.json`. Earlier no-push statements below
  describe their historical task scopes. No journal submission was requested.
- Implementation: verify HEAD and worktree before acting.
- Completed task: `docs/tasks/2026-10-03-product-hardening.md`. GCP task is also complete.
- Implementation task: `docs/tasks/2026-10-02-review-remediation.md`.
- Case corpus input: `docs/tasks/2026-10-02-case-corpus-census.md`.

## Latest UI remediation complete locally

Owner authorised all fixes from the rk/article review and added the unstyled chapter
loading/error state. Task: `docs/tasks/2026-10-03-rk-remediation.md`, baseline
`36887a0b089cc27760bb908e9e399061a9d078c1`. The local commit containing this update
is the implementation; verify HEAD/tree. No push or publication occurred.

All original RK-01–RK-07, EX-01–EX-02 and the extra summary-focus CR-01 are closed
in the independently examined slice. Focus/Combobox/Button/defaults/themes/Panel,
responsive demo navigation and GraphControls are fixed. DataTable now bounds its
first virtual render and keeps a visible keyboard target after scroll; mobile
overshoot retains a nonempty final window. Article timeline has keyboard/touch
window selection; loading/error uses the article grid and a real route-preserving
retry through the kit Button. Scientific source/data are unchanged.

Final kit regression **698 passed / 93 files**, typecheck/lint/build/size/pack PASS.
Original kit and demo builds PASS. Site build/no-undef23/SSR5, fresh offline npm ci,
14 generator tests, provenance97/8, deterministic generation and repository gates
PASS. Axe preload timeout and esbuild side-effect import warnings remain recorded
separately. Cold report: `research/local/reviews/rk-remediation-cold-20261003.md`.
API/cleanliness and reader clarity scored 9, laconicity 8.5; no uniform 9/10 claim.
Performance has limited large-table/cursor traces, not an all-kit benchmark.

38 reviewed file changes were applied to rk/rk-example after old/new hash checks;
457 baseline files, logos and brand colors were checked. User WIP is preserved and
not committed. Receipt: `/tmp/stylo-rk-fixes-l29b81q_/integration-receipt.json`.
Final tarball is installed in Stylo with matching lock integrity; platform metadata
preserved. Local article http://127.0.0.1:4174/ (session4980); original demo
http://127.0.0.1:5180/. Production browser checked five chapters at1440/390 and
download/recovery/demo scenarios. These are local previews, not published sites.

## Previous UI review and reader fixes

Owner requested fixes from five screenshots and a comprehensive independent rk/article review.
Task: `docs/tasks/2026-10-03-ui-review-followup.md` (baseline47c9027).
The article fixes include clean disclosures, rewritten control-panel layout, generated chapter
JSON downloads, responsive meter tracks, nested disclosure deep links, route-preserving skip link,
heading hierarchy and removal of unsupported likelihood badges. Build/SSR5,14targeted tests and
provenance97/8 pass. No model/data fit was rerun. Local article remains http://127.0.0.1:4174/.

Independent reports: `research/local/reviews/rk-comprehensive-20261003.md` and
`research/local/reviews/article-comprehensive-20261003.md`. Their then-open kit/demo
findings and timeline P3 are superseded by the completed remediation above. The kit
and demo were read-only during that earlier review; later edits are recorded separately.

## Editorial/design task complete locally

Owner requested an Astra redesign through rk-example, declarative reader prose, unchanged logos,
and journal-quality research preparation. User feedback after example iteration1 was: make it more
compact and enlarge text. Iteration2 implements body22/20px; the same kit drives the five Stylo chapters.
Task: `docs/tasks/2026-10-03-editorial-rk.md`. No journal has been selected; external publication,
push and actual manuscript submission were not performed.

Source candidate `9ce9676`, reviewed text corrections `38ce624`/`87df3e9`, runner registry `c2d384a`.
Full regression **1396 passed,8 skipped**; final site build/no-undef22/SSR5 and gates pass.
A separate cold Astra reproduced paired_summary and verified layouts; its two findings are closed.
Review: `research/local/reviews/astra-editorial-science-20261003.md`.

Scientific additions: `research/evidence/topic_validity_lobo_v1/paired_summary.json`, reusable reducer,
`docs/runbooks/publication-methods.md`, `docs/runbooks/manuscript.md`. Original benchmark source
remains bb3760f and its aggregate unchanged. New numbers reduce saved predictions; no new fit.
They do not establish topic independence or novel authorship. Journal submission still needs verified
bibliographic corpus links, external controls, author details and journal-specific preparation.
Working publication figure: `research/local/journal-figures-20261003/` (SVG/PDF and generator).

rk and rk-example had user WIP. Eight reviewed files were applied after checking original hashes;
other baseline source files and logos are unchanged. Originals build successfully. Baseline/receipt:
`/tmp/stylo-editorial-rk-jy6jx_7d/{baseline,integration-receipt}.json`. Their uncommitted user work is preserved.
Local live article: http://127.0.0.1:4174/ (session52478). Original rk-example:
http://127.0.0.1:5180/ (session44269). These supersede the earlier static preview for design review.

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
