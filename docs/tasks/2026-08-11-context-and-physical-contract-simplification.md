# Context and physical-contract simplification E1+E2+E3

## Metadata

- Task ID: `2026-08-11-context-and-physical-contract-simplification`
- Status: active
- Owner: repository owner
- Created: 2026-08-11
- Baseline commit: `e4dad4c419a85adee78651f6a2dc544ba7314f82`
- Target branch/worktree: `main`; selected tracked fixes use separate worktrees from the committed task baseline
- Type: simplification / pruning
- Size: L
- Risk: R2
- Risk flags: mutation, scientific-contract
- Primary domains: corpus/data and evaluation/paired-audit
- Allowed cross-domain: research-governance contract metadata; local ignored developer artifacts
- Standard version: 1.3

## 1. Goal and owner decisions

Reduce the cost of maintaining the canonical R1 and v3.2 paths without changing scientific
identities, model math, applicability, corpus/fold bytes, publication state or execution authority.
The owner selected three fixes from the read-only audit:

- E1: delete the ignored local-only `scripts/_fetch_tmp/` island.
- E2: replace repeated technical strict-field primitives in the R1 family with one internal neutral
  implementation while keeping domain schemas, errors and validation call sites local.
- E3: replace the hostile/concurrent Linux-filesystem preparation claim with a cooperative local
  filesystem contract. Exact paths, bytes, hashes, artifact inventory and scientific derivation
  remain fail-closed; adversarial inode/mount/hardlink/TOCTOU guarantees are no longer claimed.

Observable result: fewer physical source files and repeated production LOC, a smaller R1 context
packet, and a substantially smaller v3.2 preparation module with the real verified v3.2 bundle and
evaluator context identities unchanged.

## 2. Baseline evidence and representative packets

Constant method: complete selected files, physical UTF-8 bytes/lines, tokens estimated as bytes/4.

| Scenario | Files | Bytes | Lines | Approx tokens | Domain transitions |
|---|---:|---:|---:|---:|---:|
| CLI registration/help | 8 | 67,783 | 1,511 | 16,946 | 2 |
| v3.2 evaluator maintenance | 10 | 226,691 | 4,556 | 56,673 | 2 |
| pinned Wikisource acquisition | 9 | 363,313 | 10,763 | 90,829 | 2 |
| real R1 control plane | 16 | 695,027 | 19,606 | 173,757 | 3 |
| public claim/site generation | 8 | 203,934 | 4,529 | 50,984 | 2 |

Additional observed facts:

- `scripts/_fetch_tmp/` contains 26 ignored/local-only Python files, 1,453 LOC and no tracked
  consumer; the executable inventory classifies every member as local-only.
- 28 production/script files repeat 87 `_exact_*`, `_sha256`, symlink and lock helpers totalling
  759 LOC. E2 is limited to the R1 family and may share only type/shape primitives.
- `corrected_v3_2.py` is 1,672 lines and includes Linux-specific descriptor traversal,
  `fcntl.flock` and `renameat2`; governance currently claims hostile concurrent-writer resistance.
- The verified v3.2 candidate, corpus, folds and applicability identities are scientific content
  contracts. They are not re-created or edited by E3.

## 3. Scope and non-goals

### In scope

- E1 deletion of the exact ignored `scripts/_fetch_tmp/` directory.
- E2 internal strict primitive used only by selected R1 modules; delete duplicated implementations.
- E3 simplify `corrected_v3_2.py` physical reads/publication and reconcile its tests and canonical
  governance descriptions with the cooperative filesystem model.
- Required inventory bindings, task result and concise handoff updates.

### Out of scope

- Scientific schemas, exclusion set, content grouping, corpus/fold identities, 25/16/11 matrix,
  estimator/evidence math, model registry or real bundle bytes.
- R1/v3.2 RunPlan, freeze, preflight, authorization, fit/predict, checkpoints, inference, result
  audit, publication, deploy, external write or push.
- Site generator redesign, experimental-runner retirement, CLI redesign or further cleanup wave.
- Broad shared framework, schema DSL, generic validator registry or error-message normalization.

## 4. Frozen physical contract E3

The supported environment is a trusted, local, cooperative single-writer filesystem. Inputs and
bundles are verified at the time they are read; there is no claim against a concurrent adversarial
filesystem mutator.

- Required: ordinary Python 3.11 filesystem APIs; existing directories; regular files; no symlinks
  or special files inside verified trees; exact relative-path inventory; exact canonical bytes and
  SHA-256/self-hashes; exact corpus/fold reconstruction; create-if-absent staging and final bundle.
- Existing mode fields remain artifact-layout bytes and are checked where encoded; they are not a
  security boundary. Existing v1 bundle bytes remain valid.
- Allowed: same-content hardlinks and ordinary inode/device changes; these do not change scientific
  content identity. Concurrent writers are unsupported and must be externally serialized.
- Removed claims: stable descriptor pinning, mount/inode continuity, ancestor race resistance,
  hostile hardlink protection, lock-based writer convergence and Linux
  `renameat2(RENAME_NOREPLACE)` availability.
- Publication remains local candidate preparation only. A pre-existing final destination is only
  revalidated; it is never overwritten or repaired.

## 5. Domain and context boundaries

- E2 may consolidate only neutral checks for exact mapping keys, list/string/integer/sha256 types.
  Work IDs, schemas, status, corpus eligibility and error ownership remain in their current modules.
- Producer/consumer: R1 domain/corpus modules own schemas; the internal helper consumes only an
  expected type/key contract and raises the caller-provided domain error.
- Grain: one parsed field or one artifact file/tree member. Keys/cardinality and scientific time
  semantics are unchanged.
- E3 producer/consumer: corrected v3.2 preparation produces the existing content-addressed bundle;
  evaluator context consumes the same verified bytes and identities.
- Delete/history: E1 is ignored and unrecoverable except from external backups; E2/E3 are recoverable
  from Git. No raw/private corpus content is read into task evidence.

## 6. Frozen acceptance criteria

- [ ] AC-01 E1 removes exactly 26 local-only files/1,453 LOC; no tracked path or consumer changes.
- [ ] AC-02 E2 preserves all R1 schemas, self-hashes, exception classes/messages and tests while
  deleting at least 300 cumulative production LOC from repeated strict primitives.
- [ ] AC-03 E2 adds at most one internal non-public module, no dependency/state/registry/framework,
  and reduces the measured 16-file R1 packet bytes or required sections.
- [ ] AC-04 E3 has no `ctypes`, `fcntl`, `renameat2`, descriptor-walk or storage-capability gate and
  makes no hostile-concurrency/security claim.
- [ ] AC-05 E3 still rejects symlinks, special files, extra/missing/tampered members, wrong canonical
  bytes/hashes and wrong scientific identities; an existing destination is never overwritten.
- [ ] AC-06 Two real verified context loads retain the existing candidate/corpus/fold/applicability
  and context identities; no real-corpus fit/predict occurs.
- [ ] AC-07 Governance describes the cooperative physical contract honestly; it does not claim an
  independent security review or alter evaluator/freeze/execution/publication status.
- [ ] AC-08 Full regression, compile, provenance, inventory, release/archive hygiene and diff check
  pass; net production LOC is negative and packet metrics improve.

## 7. Complexity tripwires

- Maximum selected fixes: 3; no automatic next wave.
- New production files: at most 1 internal E2 helper; new public entrypoints: 0.
- E2 allowed production paths: R1 `src/stylo/domain/lobo_vnext*.py`,
  `src/stylo/corpus_tools/*vnext.py`, `ruaa_r1_*.py`, `src/stylo/eval/lobo_vnext*.py`, plus one
  internal helper under `src/stylo/`.
- E3 allowed production path: `src/stylo/eval/paired_audit/corrected_v3_2.py` only.
- Tests may change only in directly owning R1/v3.2 tests and governance tests; cumulative test LOC
  must be non-positive.
- Task/governance/handoff docs net addition ceiling: 320 lines.
- Cumulative production target: at least -500 LOC; no positive-LOC fix counts as simplification.
- New dependency, runtime state, framework, schema DSL, generic registry or publication surface: 0.
- At most one adversarial/deletion review and one bounded correction pass.
- Stop/reclassify on scientific identity drift, real-bundle byte mutation, unsupported consumer,
  public behavior change, positive production delta, new sensitive-data factor or second blocker.

## 8. Verification and falsifiers

- Before/after exact files/bytes/lines for all five packets and production/test/docs deltas.
- E1: physical inventory, tracked/reference negative search, executable inventory.
- E2: focused R1 domain/acquisition/packet/control-plane tests and hash/schema fixtures.
- E3: corrected-v3.2 synthetic negative tests; real context load twice, identity comparison only.
- Full pytest; physical-file py_compile; provenance; executable inventory; release and Git-free
  archive hygiene; `git diff --check`; final clean status.
- Falsifiers: differing error/schema/self-hash; a symlink/special/tampered tree accepted; existing
  destination overwritten; candidate/context identity drift; hidden fit/predict; tripwire crossing.

## 9. Review, rollback and artifacts

- Implement E2 and E3 as separate commits/worktrees from the committed task baseline. E1 is a
  local-only deletion recorded in Result but has no Git commit.
- Review only frozen AC/invariants, regression, context reduction and unsupported deletion. Do not
  reintroduce adversarial filesystem/security architecture as a blocker after the owner decision.
- One correction pass maximum. Commits remain individually revertible; E1 is not recoverable from
  Git and is explicitly owner-authorized.
- ADR: not required; E3 narrows an implementation-specific physical guarantee and is recorded in
  the owning preparation/governance contract. Domain docs/runbook: no new durable domain/operation.
- Update this Result and `docs/handoff/CURRENT.md` at closeout.

## 10. Result

Pending.
