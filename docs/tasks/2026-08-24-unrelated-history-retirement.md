# Unrelated-history retirement

## Metadata
- Status: done
- Owner / Baseline commit: Dmitry Purtov / `fc335823`
- Type: pruning
- Size / Risk: L / R3a (flags: mutation, irreversible, security)
- Primary subsystem/domain: repository history / Allowed cross-domain: none
- Standard version: 1.3
- SAFE-02 approval: owner selected «Всё: ветки + stash + gc» in the 2026-08-24 operator turn,
  immediately before the action, after the backup bundle was created and restore-verified.

## Frozen behavior
- Observable behavior to preserve: `main`, `origin/main`, `release` and every object reachable from
  them; all tracked bytes; every gate result.
- Explicit non-goals: rewriting `main`, pushing anything, touching `origin`, deleting the backup.

## Scope

Removed local refs, none of which shares a merge base with `main`:

| Ref | Content |
|---|---|
| `rework/stylo-v4` (`8ca0b826`, 143 commits) | private corpus `input/`, `input_clean/`, `data/train_vectors.pkl` |
| `archive/local-main-private-20260726` (`6a3dd614`) | fully contained in `rework/stylo-v4` |
| `release-backup-20260707` (`0c38351a`, 10 commits) | unrelated history, no private paths |
| `stash@{0}` (`c806adf0`) / `stash@{1}` (`09c6622a`) | 3 619 and 2 416 private paths |
| unreachable reflog entries | abandoned commits |

## Reversibility

`~/backup/stylo-history-20260824/retired-history.bundle`, 424 MB,
sha256 `6859b2de5e1f29135039a0f152810953c7c74f782617c27e7a657505ab1bb94f`, stored outside the
repository. `git bundle verify` reports a complete history for all three refs. A mirror clone of the
bundle was made before deletion and returned 143 commits on `rework/stylo-v4` and a readable
`c806adf0` stash commit. `refs-before.txt`, `stash-before.txt` and `count-before.txt` sit beside it.

## Acceptance

- [x] Backup bundle created, verified and restore-checked before any deletion
- [x] Three branches deleted, both stashes cleared, unreachable reflog expired, `gc --prune=now` run
- [x] `git count-objects -vH` size-pack drops from 427.92 MiB to single-digit MiB
- [x] `check_release_hygiene.py --audit-local-refs` reports no private objects in any ref or stash
- [x] `main` unchanged; release gates still pass

## Verification

```
git count-objects -vH
git rev-parse HEAD                                  # must equal fc335823
.venv/bin/python scripts/check_release_hygiene.py --publish-ref HEAD --audit-local-refs
.venv/bin/python scripts/check_executable_source_inventory.py
node scripts/check-provenance.mjs
```

## Result
- Status: done. `size-pack` 427.92 MiB → 3.73 MiB; `.git` 431 MB → 4.1 MB; in-pack objects
  43 894 → 3 531; working tree 6.3 GB → 4.5 GB across the whole campaign.
- `HEAD` is `fc335823` before and after; local branches are now exactly `main` and `release`;
  `git stash list` is empty; `git fsck` reports nothing.
- `check_release_hygiene.py --publish-ref HEAD --audit-local-refs` now reports
  «no private objects in other refs/stash» — previously it counted 16 793 + 16 772 + 3 619 + 2 416
  private paths across the retired refs and stashes.
- Inventory OK at 287 paths; site provenance verified 93 sources and 1 output.
- Residual risk: the retired history exists only in the external bundle. Restoring it re-introduces
  the copyrighted corpus, so it must never be pushed.

## DoD references
- [ ] Applicable DOD-01..DOD-12, DOD-13
