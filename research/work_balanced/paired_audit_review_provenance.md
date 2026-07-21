# Paired-audit review provenance

This artifact binds the paired-audit control-plane commit to the mandatory-check results and to the
independent-review verdict. A review verdict is an engineering judgement recorded here — it is **not** a
cryptographic signature, and it does not authorize any real-corpus execution (protocol §11–14 stay
behind a separate authorization). After a positive independent review of the reviewed commit, that
commit must **not** be amended or followed by any code change — the reviewed SHA must equal the final
SHA.

## Round-1 sign-off withdrawn

A round-1 review had signed off on `3c34bbe5` (with a follow-up `34481cb8`). An owner-authored
independent probe then found the round-1 result over-claimed: the control plane was still a primitives
library with 13 substantive blockers (A0 pred never compared; real evidence discarded and synthesized;
before/after verification comparing an in-memory object to itself; RuAA checkpoint bound to the parent
digest; tautological manifest verification; per_work missing true_label/correct; publisher recomputing
nothing; smoke able to write the production artifact; a lax checkpoint schema; a loader blind to a
swapped root summary; `write_transient` mkdir-before-guard; the sign-off SHA differing from the final
SHA; the governing protocol + RuAA SHA256SUMS not committable). The round-1 sign-off is **withdrawn**.

## Round-2 remediation (this commit)

The 13 blockers are closed in the strict order R2.1…R2.9 as separate commits (no amend); R2.10 is this
verification + review record. The requirement → code → adversarial-test trace lives in
[`paired_audit_implementation_audit.md`](paired_audit_implementation_audit.md). The OS kernel is out of
scope and bound nowhere.

- Branch: `paired-audit-control-plane` (off `release` HEAD `2f6c3dc3`).
- Change inventory: 26 files, additions only (14 modules + 10 test files + this audit pair); no
  modified/deleted tracked file — verified with `git diff --name-status release..HEAD` (all `A`). The
  owner's 66 M/D rework entries are untouched.
- **Reviewed commit SHA = the final commit that carries this record.** No code change follows the
  independent sign-off; the reviewed SHA equals the final SHA by construction.

## Mandatory checks (commands + results)

| Check | Command | Result |
|---|---|---|
| A focused, dirty tree | `PYTHONPATH=src pytest tests/test_paired_audit_*.py` | 185 passed |
| C clean committed-snapshot | `git archive HEAD \| tar -x -C /tmp/s; PYTHONPATH=/tmp/s/src pytest /tmp/s/tests/test_paired_audit_*.py` | 181 passed, 4 skipped (3 runner-e2e need `.git`; 1 RuAA-ref needs private data) — self-contained |
| Full clean `git clone` | `git clone --no-hardlinks . /tmp/c; PYTHONPATH=/tmp/c/src pytest /tmp/c/tests/` | 4 failed, 786 passed, 6 skipped — the 4 are pre-existing packaging debt (`test_ci_sign_erratum`, `test_macro_f1_ci_withdrawal`; missing `scripts/gen-paper.mjs`), **zero** paired-audit |
| D synthetic e2e | `pytest tests/test_paired_audit_runner.py` | full chain publish + transient round-trip + resume + result-audit tamper |
| E adversarial path/race | checkpoint (`os.link` no-overwrite, symlinked-ancestor), publisher (`..`-after-normalize, symlink chain, guard-before-mkdir, root tamper) | covered by the fail-closed + security tests |
| F requirement trace | [`paired_audit_implementation_audit.md`](paired_audit_implementation_audit.md) | R2.1…R2.9 requirement → code → adversarial-test table |

## Independent review record

_A fresh independent adversarial review of the EXACT final SHA is recorded here (fan-out reviewers that
read the code and run their own probes + the suite, not a summary read). The review targets every bypass
the owner listed. The reviewed SHA is the final SHA; no code change follows a positive sign-off._

- Reviewed SHA: _the final commit carrying this record_.
- Verdict: _stamped by the independent review of that exact SHA_.

## Open items for the OWNER (do not block §11 preparation; close before the confirmatory EXECUTION)

- Commit the governing `paired_audit_protocol.md` (owner's untracked file) so the spec is bound in HEAD.
- Deliberately re-freeze `data/ruaa_bench_v1/SHA256SUMS` for the drifted `protocol.md` (1 of 141 files;
  the corpus texts + submission are intact; `verify_ruaa_inventory` fail-closes until it is re-frozen).
  A blind checksum re-freeze is refused — this is an owner decision.
- Register the real per-fold estimator as a confirmatory `EvaluatorSpec` (its name added to
  `REGISTERED_CONFIRMATORY_EVALUATORS`) and supply the real fold-local axis/state digests.
- Provide the LOBO author display→slug map so the confirmatory A0 pred comparison can run.

The control plane is implemented, remediated (round 2), and — once the independent review of the exact
final SHA signs off — reviewed for §11 **preparation** only; the confirmatory **execution** and any
headline decision remain behind a separate execution authorization.
