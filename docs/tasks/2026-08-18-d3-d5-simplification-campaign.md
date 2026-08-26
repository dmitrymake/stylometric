# D3 + D4 + D5 bounded simplification campaign

## Metadata

- Task ID: `2026-08-18-d3-d5-simplification-campaign`
- Status: completed
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

## Verification plan

- Targeted: executable inventory; README link integrity; governance and case-classification contracts.
- Regression: full pytest; physical-file `py_compile`; provenance; checkout/Git-free archive inventory
  and hygiene; release hygiene; `git diff --check`.
- Negative: search for authorization, security-PASS, freeze, publication, registry, claim, or scientific
  mutation language; assert no positive docs LOC/bytes and preserved 299-path inventory.
- Cannot-run: record separately with exact command/environment; never infer pass from static checks.

## Result

### Verification PASS

- `git diff --check 493325c6..HEAD` empty
- `python3 -c "..."` inventory check: 299 paths, sha256 unchanged
- `.venv/bin/python scripts/check_executable_source_inventory.py` → OK
- `.venv/bin/python scripts/check_release_hygiene.py` → no private corpus paths
- `pytest tests/test_release_integrity.py`: 100%
- `pytest tests/test_executable_source_inventory.py`: 100%
- `pytest tests/test_medium_governance.py`: 23/23
- `pytest tests/test_release_hygiene.py`: 5/5
- `pytest tests/test_cases_framework.py`: 23/23
- `pytest tests/test_ci_release_matrix.py`: 3/3
- Git-free archive: 299 paths, sha256 unchanged, no `log/` subtree
