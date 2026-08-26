# Merged branch ref hygiene

- Status: done
- Owner: Dmitry Purtov
- Created: 2026-08-24
- Baseline commit: `ac617a5f`
- Type: pruning
- Size / Risk: S / R1
- Primary domain: repository hygiene
- Allowed cross-domain: none

## Goal

Remove local branch refs that carry zero commits outside `main`/`origin/main`, so that no stale
campaign ref can be pushed past the release gates by accident.

## Scope

- In: `fix-d3-inventory`, `fix-d4-work-balanced-nav`, `fix-d5-cases-orientation`, `review-d3`,
  `review-d4`, `review-d5`, `integrate/remediation-main-20260726`, `paired-audit-control-plane`.
- Out: `release` (CI trigger branch), the three unrelated-history branches, stash, any object deletion.

## Verification

- Command: pre-check counts, `git branch -D`, `git count-objects -vH`,
  `.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD`
- Negative scenario: a branch with unique commits must abort the deletion for that branch.

## Result

- Status: done. Deleted `fix-d3-inventory` (`1935a99d`), `fix-d4-work-balanced-nav` (`1a606467`),
  `fix-d5-cases-orientation` (`dc3b09c9`), `review-d3`, `review-d4`, `review-d5` (same three SHAs),
  `integrate/remediation-main-20260726` (`fd034e00`), `paired-audit-control-plane` (`3d5961da`).
- All eight reported zero unique commits against `main`/`origin/main` immediately before deletion.
- `size-pack` unchanged at 427.92 MiB, confirming a ref-only operation.
- Remaining local branches: `main`, `release`, and the three unrelated-history branches handled by
  the separate T7 task.
- Production/test/docs delta: 0. Residual risk: none.
