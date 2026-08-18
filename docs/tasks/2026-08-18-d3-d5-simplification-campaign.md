# D3 + D4 + D5 bounded simplification campaign

## Metadata

- Task ID: `2026-08-18-d3-d5-simplification-campaign`
- Status: active
- Owner: repository owner
- Created: 2026-08-18
- Baseline commit: `0f2897af9bcff40119dfda9b0963f467e81abe98`
- Target branch/worktree: `main`; each selected fix uses a separate worktree/branch from the committed task baseline
- Type: simplification / pruning
- Size: L
- Risk: R2
- Risk flags: mutation, scientific-contract, publication
- Primary domain: research-governance
- Allowed cross-domain: reporting/publication metadata and release metadata only, within candidate paths
- Standard version: 1.3

## Goal and decision

Reduce the bounded orientation and release-metadata context packets without changing scientific
semantics, status authority, evidence, publication bytes, runtime behavior, or execution authority.
The consumer is a maintainer or reviewer locating executable-source, work-balanced, or case-status
orientation. Observable result: D3 removes stale inventory rows, D4 makes the existing work-balanced
README directly locate the accepted preparation record, and D5 makes the existing cases README the
single compact current/historical orientation surface. The owner selects D3, D4, and D5, ordered
D3 → D4 → D5; each fix is independently droppable if its falsifier or packet target fails.

## Confirmed starting facts

| ID | Fact | Evidence | Confidence |
|---|---|---|---|
| F1 | Inventory contains 26 nonexistent `scripts/_fetch_tmp/*.py` entries; two Petersburg builders remain local-only pending fresh proof. | Plan-hard bounded audit; `release/executable_sources.json` | high |
| F2 | Work-balanced README omits the v3.2 preparation review and requires directory/document reconciliation. | `research/work_balanced/README.md`; plan-hard audit | high |
| F3 | `status_ledger.json` is the sole current-status authority; freeze, preflight, authorization, and execution remain absent/disabled. | `docs/handoff/CURRENT.md`; governance ledger | high |
| F4 | Cases README distributes current, withdrawn, and historical classifications across several fragments and the handoff. | `docs/cases/README.md`; plan-hard audit | high |
| F5 | The campaign is documentation/release-metadata only: no source, test, data, generated, or publication bytes are selected. | Frozen plan scope and repository instructions | high |

## Scope

In scope: the three candidate paths and this task file; handoff changes only when the shared closeout
rule requires them. Out of scope: `data/`, `scripts/`, `src/`, `tests/`, `research/` except the named
D4 README, `docs/cases/` except the named D5 README, exploratory artifacts, logs, site, `.github/`,
`.gitattributes`, configs, release inventory except D3's named JSON, generated reports, registries,
freeze/preflight/authorization/execution, dependencies, and external writes.

Stop and reclassify on baseline/worktree mismatch, unknown WIP, owner/task-baseline absence, any
scientific/status dispute, protected-path mutation, new authority or entry point, positive production
or test LOC, positive D4/D5 README lines or bytes, failed packet reduction, generated/publication drift,
need for code/test/schema/checker changes, a second blocker, or any new R3/sensitive-data factor.

## Frozen campaign acceptance criteria

- [ ] Exactly three independently revertible candidates are implemented only in their allowed paths.
- [ ] Cumulative production/test LOC added is 0/0; no new production/test/docs files, dependencies,
  framework, state, registry, schema, checker semantics, or public/operational entrypoint exists.
- [ ] D3 removes exactly 26 stale inventory rows while preserving the verified 299-path count and
  `747a07c57648d91467d477cab6c68fc8f3d17e0db4a7ddba8250ac8abcd7811f` path-set digest.
- [ ] D4 and D5 each have non-growing whole-file lines and bytes and reduce their required orientation
  packet without creating authority, claim, chronology, or publication implication.
- [ ] Existing links, status/withdrawal/abstention boundaries, inventory/archive semantics, and all
  protected artifacts remain intact; targeted and full verification pass where runnable.
- [ ] Accepted commits are integrated only after cumulative metrics, cross-fix conflict checks, and
  one bounded deletion/adversarial review; no automatic follow-on wave begins.

## Fix D3 — reconcile stale local-only executable-source entries

**Problem.** `release/executable_sources.json` lists 26 `scripts/_fetch_tmp/*.py` paths deleted in E1.

**Scope.** Edit only `release/executable_sources.json` and shared task/closeout metadata. Freshly prove
the two Petersburg builders are ordinary Python, Git-untracked, Git-ignored, and absent from the
Git-free archive; any contrary result blocks D3.

**ACs.** (1) Remove exactly the 26 rows, without replacement or recreation. (2) Preserve the two
Petersburg local-only rows only after the stated proof. (3) Preserve count `299` and the exact path-set
SHA-256 above. (4) Checkout/archive inventory and release hygiene pass without membership drift.
(5) No source, test, schema, scientific, or publication bytes change.

**Allowed/forbidden paths.** Allowed: the named JSON, task file, and handoff only at closeout. Forbidden:
all `scripts/**`, `src/**`, `tests/**`, `research/**`, `docs/cases/**`, `docs/exploratory/**`, `data/**`,
`log/**`, `.github/**`, `site/**`, `.gitattributes`, dependencies/configuration, and other release files.

**LOC budget.** Production/test `0/0`; inventory metadata exactly `-26` rows; shared task/handoff ≤240 lines.

**Packet effect.** `local_only_python_files` 28→2; false entries 26→0; array span lines 10–39→a
three-line envelope plus two entries; bytes ≈1.17 KiB→0; transitions 1 unchanged; authorities and
entrypoints unchanged.

**Smoke verification.** Capture tracked/ignored/archive status for both builders; run targeted inventory,
checkout/archive, release-hygiene, full pytest, and `git diff --check` at integration.

**Rollback.** Revert the single commit or restore the baseline JSON blob. Abort rather than changing
checker semantics or reclassifying the Petersburg files.

**Tripwires.** Baseline/count/digest drift; unexpected builder status; release membership change; edits
outside allowed paths; positive metadata rows/bytes; protected artifact/state change.

## Fix D4 — compress work-balanced research navigation in place

**Problem.** `research/work_balanced/README.md` omits the existing v3.2 preparation review.

**Scope.** Rewrite only its navigation list to link `paired_audit_v3_2_preparation_review.md` and
group current contract/design, current preparation, and immutable historical evidence. Linked document
content, ledger, protocol, estimand, review, roadmap, evidence, and `research/README.md` are excluded.

**ACs.** (1) Directly identify the review only as owner-accepted preparation/remediation. (2) Grant no
freeze, registration, preflight, authorization, execution, security-PASS, or publication implication.
(3) State `status_ledger.json` remains sole current-status authority. (4) Preserve distinct protocol,
estimand, preparation, historical audit/provenance, and LOBO roles. (5) Preserve all valid links and
avoid new authority/claim/chronology. (6) Packet is README+directory search+target→README+target.
(7) README lines and bytes do not grow.

**Allowed/forbidden paths.** Allowed: named README, task file, handoff at closeout. Forbidden: all other
`research/work_balanced/**`, `research/governance/**`, `research/evidence/**`, `research/ROADMAP.md`,
`research/README.md`, source/tests/scripts, cases/exploratory, data, site, and release inventory.

**LOC budget.** Production/test `0/0`; docs added ≤6 lines and net ≤0; whole-file bytes ≤baseline; no new file.

**Packet effect.** Surfaces 3→2; orientation sections 2+→1; README file count unchanged; transitions 1,
entrypoints 0, and current-status authorities 1 unchanged.

**Smoke verification.** Check every relative link, focused governance contracts, negative status-language
search, exact line/byte deltas, full pytest, and `git diff --check` at integration.

**Rollback.** Revert the one documentation commit or restore the baseline README.

**Tripwires.** Status ambiguity; request to edit ledger/protocol/review; governance mutation; broken link;
new authority; README line/byte growth; need for test/source changes.

## Fix D5 — collapse case historical/current orientation into one README table

**Problem.** Cases README spreads current, withdrawn, and historical classification across introduction,
withdrawal, handoff pointer, and family bullets.

**Scope.** Replace `docs/cases/README.md:1-23` with one compact table covering current v2 abstaining gate,
Taras hardened historical family, and Petersburg hardened historical family. Keep the detailed handoff
link; do not edit case JSON/YAML/passports/manifests, handoff content, index, reports, site, or CLI behavior.

**ACs.** (1) Table states v2 is abstaining/inconclusive without a registered calibrated open-set gate,
Taras positive target claim is withdrawn, and Petersburg is historical closed-set diagnostics only.
(2) No verdict is revived, softened, ranked, or publishable. (3) Preserve `diagnostic_closed_set_top`,
work-level feasibility semantics, and v2 rejection of old passports. (4) Replace prose, do not duplicate.
(5) Keep detailed history in the existing handoff. (6) Packet README+HANDOFF→README table alone.
(7) Whole README lines/bytes do not grow and generated output does not drift.

**Allowed/forbidden paths.** Allowed: named README, task file, handoff at closeout. Forbidden: case data
files, `docs/cases/HANDOFF.md`, `docs/index.html`, exploratory/site/generated files, source/tests/research,
data/log, and release inventory.

**LOC budget.** Production/test `0/0`; docs added ≤10 lines and net ≤0; whole-file bytes ≤baseline; no new file.

**Packet effect.** Classification files 2→1; orientation fragments 4→1 table; replacement stays within
lines 1–23; whole README ≈95 lines with no growth; authorities and entrypoints unchanged.

**Smoke verification.** Compare statements with baseline semantics; verify links; search contract terms
including `inconclusive`, `abstained`, and `diagnostic_closed_set_top`; run focused case/CLI tests, full
pytest, provenance/release hygiene, generated-tree diff checks, and `git diff --check` at integration.

**Rollback.** Revert the one documentation commit or restore the baseline README.

**Tripwires.** Scientific/status dispute; provenance repair request; generated/site diff; README growth;
new authority; code/test/schema change; HANDOFF still mandatory for basic classification.

## Dependencies and integration plan

Sequence: fresh state snapshot + owner selection → committed common task baseline → isolated D3, D4,
and D5 branches → accepted commits only → combined packet/LOC reconciliation → integrated deletion/
adversarial review → frozen verification and closeout. D3 is recommended first for release metadata;
D4 and D5 may proceed in parallel after the common baseline. Cumulative ceilings: 3 fixes; production
files/LOC `0/0`; test files/LOC `0/0`; new files `0`; dependencies/frameworks/state/registries/
entrypoints `0`; scientific/publication/generated artifacts `0`; shared task/handoff ≤240 lines; one
review and at most one bounded correction per fix.

## Tripwire catalog

- **Baseline/process:** HEAD, branch, or worktree mismatch; unknown WIP; no owner/common baseline; >3
  fixes; automatic wave; >1 correction; non-revertible fix; inconsistent measurements; two blocked candidates.
- **Architecture/code:** any source/test edit, new file/dependency/config/framework/layer/helper/registry/
  checker/schema/runtime state/entrypoint, positive production/test LOC, or positive D4/D5 docs bytes.
- **Scientific/data/governance/publication:** data/raw/private corpus, exploratory packet, evidence,
  receipts/manifests, tracked logs, registry/freeze/preflight/authorization/execution, claim/security-
  PASS/publication/deploy/external write, case data, generated/site/index, governance authority, or
  historical attestation changes.
- **Candidate-specific:** D3 count/digest/builder/release drift; D4 authority/status/link/README growth;
  D5 HANDOFF dependence, withdrawal/abstention reversal, generated impact, or README growth.

## Verification plan

- Targeted: executable inventory; README link integrity; governance and case-classification contracts.
- Regression: full pytest; physical-file `py_compile`; provenance; checkout/Git-free archive inventory
  and hygiene; release hygiene; `git diff --check`.
- Negative: search for authorization, security-PASS, freeze, publication, registry, claim, or scientific
  mutation language; assert no positive docs LOC/bytes and preserved 299-path inventory.
- Cannot-run: record separately with exact command/environment; never infer pass from static checks.

## DoD references

- DOD-01: contract and deviations.
- DOD-02: scope/domain boundaries and no creep.
- DOD-03: reproducible claims/evidence audit.
- DOD-04: targeted/regression checks and separated unknowns.
- DOD-05: negative scenarios/falsifiers.
- DOD-06: safety and rollback/abort (R3 approvals N/A).
- DOD-07: result, evidence, residual risk, and Git state.
- DOD-08: ADR/domain/runbook/handoff only by canonical need (ADR/domain/runbook N/A unless triggered).
- DOD-09: independent review for L task.
- DOD-10: reproducible result without transcript.
- DOD-11: blockers, budgets, and no hardening/debt creep.
- DOD-12: tripwires and cumulative metrics.
- DOD-13: pruning/deletion review and reduction or partial/block.
- DOD-14: identical before/after context-packet method.
- DOD-15: audit, selection, bounded fixes, integration, metrics, and finite stop without auto-wave.
