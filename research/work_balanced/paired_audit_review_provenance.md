# Paired-audit review provenance

This artifact binds the paired-audit control-plane commit to the mandatory-check results and to the
independent-review verdict. A review verdict is an engineering judgement recorded here — it is **not** a
cryptographic signature, and it does not authorize any real-corpus execution (protocol §11–14 stay
behind a separate authorization). After a positive independent review of the reviewed commit, that
commit must **not** be amended.

## Reviewed commit

- Branch: `paired-audit-control-plane` (off `release` HEAD `2f6c3dc3`).
- Reviewed commit SHA: _stamped by the review-record commit that follows the independent review of the
  exact final SHA._
- Change inventory: 25 files, additions only (13 modules + 10 test files + this audit pair), no
  modified/deleted tracked file — verified with `git diff --name-status <base>..HEAD`.

## Mandatory checks (commands + results at implementation time)

| Check | Command | Result |
|---|---|---|
| A focused | `PYTHONPATH=src pytest tests/test_paired_audit_*.py` | 151 passed (working tree, git present) |
| B full repo | `PYTHONPATH=src pytest tests/` | exit 0 (all pass; 2 pre-existing DeprecationWarnings in a rework test) |
| C clean snapshot | `git archive HEAD \| tar -x -C /tmp/pa_snap; PYTHONPATH=/tmp/pa_snap/src pytest tests/test_paired_audit_*.py` | 148 passed, 3 skipped (2 runner-e2e need git; 1 RuAA-ref needs private data) |
| D synthetic e2e | `pytest tests/test_paired_audit_runner.py` | 2 passed (full chain publish+round-trip; idempotent resume) |
| E adversarial path/race | checkpoint (`os.link` no-overwrite, symlinked-ancestor), publisher (`..`-after-normalize, symlink chain) | covered by the fail-closed tests |
| F requirement trace | [`paired_audit_implementation_audit.md`](paired_audit_implementation_audit.md) | requirement → module → test table |

Each remediation commit additionally re-ran the clean committed-snapshot replay (check C) before the
next commit, so self-containment holds at every step, not only at HEAD.

## Independent review record

Two rounds of fresh independent review were run against the exact commits (fan-out subagents that read
the code and ran their own probes + the test suite, not a summary read):

- **Round 1 — full four-lens review of `e8d9b8cd`** (blockers / runner / security / isolation). Verdict:
  blockers 9 (sign-off), security 8.5 (sign-off), isolation 9.4 (sign-off), runner 6.5 (**no** sign-off)
  — the runner lens found a CRITICAL (confirmatory `a0_references` TypeError) and a HIGH (stylo LOBO A0
  results never compared to the pinned 221/251 reference). Both, plus the medium/low/nit, were fixed.
- **Round 2 — convergence re-review of the fixed `3c34bbe5`** (fixes-verification + security/regression).
  Verdict: **fixes 9.4 (sign-off), security 9.5 (sign-off)** — all six prior findings verified resolved
  by probe, no regression, 152 working-tree tests, clean-snapshot 149 passed / 3 skipped, TOCTOU
  hardness confirmed, additions-only diff. Two non-blocking residuals were flagged (headline
  margin/quantiles binding; a split-guard exception type).

- Signed-off reviewed SHA: `3c34bbe5` (round-2 sign-off by both lenses).
- Final SHA: `34481cb8` — applies ONLY the two non-blocking residual fixes the round-2 reviewers
  themselves requested (headline decision + CIs bound fully to `plan['stats']`; `_assert_a0_matches_reference`
  split guard). The signed-off commit `3c34bbe5` is **not amended**; the final commit is a follow-up.
- Change-inventory SHA (`git diff --name-only 2f6c3dc3..34481cb8 | sort | per-file blob-sha | sha256`):
  `d2b118b5dea9ffe4b4d5399388d67a733696f46e15197e9a03d494aaf33e1feb`.
- Test-result summary: focused 152 passed (working tree); clean committed-snapshot 149 passed / 3 skipped.
- Verdict: **sign-off for separately-authorized §11 real-corpus PREPARATION.** All 11 Gate-10 blockers
  and the round-1 CRITICAL + HIGH are resolved.
- Open items (do not block §11 preparation; close before the confirmatory EXECUTION under a separate
  authorization): commit the governing `paired_audit_protocol.md` (owner's untracked file) to bind the
  spec; re-freeze `data/ruaa_bench_v1/SHA256SUMS` for the drifted `protocol.md`; supply the real
  per-fold estimator + real fold-local axis digests via the injected evaluator.

The control plane is implemented, remediated, and independently signed off for §11 **preparation** only;
the confirmatory **execution** and any headline decision remain behind a separate execution authorization.
