# Governance state and retired-shim cleanup

## Metadata

- Task ID: `2026-08-11-governance-and-retired-shim-cleanup`
- Status: done
- Owner: repository owner
- Created: 2026-08-11
- Baseline commit: `35da4301172d220c1dd7940227b460e0634f1734`
- Target branch/worktree: `main`; fixes in separate worktrees from the task baseline
- Type: pruning
- Size: M
- Risk: R2
- Risk flags: mutation, external-contract
- Primary domain: research-governance
- Allowed cross-domain: evaluation/paired-audit metadata and retired operational entrypoints;
  reporting/publication release inventory only
- Standard version: 1.3

## 1. Goal and decision

- Desired observable result: current v3.2 status has one authority and cannot be confused with the
  immutable v3.1 15-member contract; twelve hard-disabled legacy script shims and their dead tests
  are removed while historical source hashes remain recoverable.
- Consumer: repository owner and the next separately authorized v3.2 RunPlan task.
- Decision/process unlocked: a smaller, contradiction-free context packet for supported work.
- Success is not merely: moving prose, renaming files, weakening safety gates, or deleting evidence.

## 4. Scope

## 11. Verification plan

- Targeted: governance, retired-path, inventory, release-hygiene tests.
- Regression: full `pytest tests`; `py_compile`; provenance; release and Git-free archive hygiene.
- Negative: grep forbidden mutable status; invoke/check canonical replacements; verify deleted paths
  absent and historical hashes present.
- Cannot-run: record separately; no pass inference from static checks.

## 13. Result

- Commits: task baseline `8cdda5cd`; A `c6b131b9`; B `bfc44f11`; inventory reconciliation
  `d2819f4d`; this task/handoff update is metadata-only closeout.
- A: the ledger is the only mutable v3.2 state authority; protocol and runner catalog now state
  stable scope. Requirements bind current 25/16/11 independently from historical v3.1 m15.
- B: all twelve selected shim files and one shim-only test module are deleted. The p0 digest map and
  Git history are unchanged; canonical `scripts/report.py` and `stylo` surfaces remain.
- Review: combined deletion review in clean detached worktree at `d2819f4d` — PASS; no unsupported
  deletion, invariant violation, or tripwire breach; zero correction passes used.
- Verification: focused governance/inventory/release checks PASS; full pytest PASS after rerun with
  basetemp in `/dev/shm`; py_compile, provenance (93 source + 1 output digest), checkout release
  hygiene, executable inventory and Git-free archive hygiene PASS.
- The first full pytest attempt was infrastructure-invalid after `/tmp` reached 100% inode use;
  hundreds of `tmp_path` setups failed. The unchanged suite passed on `/dev/shm`. One intentional
  skip remained because `STYLO_V32_BUNDLE_ROOT` was unset; two pre-existing invalid-escape
  deprecation warnings remained.
- Production delta from original baseline: +0 / -193 / net -193 LOC; 12 operational entrypoints
  removed. Tests: +12 / -108 / net -96 LOC. Before closeout metadata, all changed files: +210 / -399.
- Context: retired-entrypoint packet 19 -> 6 files, 238,711 -> 227,370 bytes, 5,475 -> 5,102
  lines; mutable v3.2 status authorities 4 -> 1. Domain transitions unchanged.
- New production files/concepts/framework/state/dependencies/public entrypoints: none.
- Production status: not activated; registry empty, freeze unapproved, preflight/authorization
  absent, execution hard-disabled, publication not authorized.
- Campaign stop: both owner-selected fixes accepted; no automatic next wave.
