# Context and physical-contract simplification E1+E2+E3

## Metadata

- Task ID: `2026-08-11-context-and-physical-contract-simplification`
- Status: completed
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

## 3. Scope and non-goals

## 8. Verification and falsifiers

- Before/after exact files/bytes/lines for all five packets and production/test/docs deltas.
- E1: physical inventory, tracked/reference negative search, executable inventory.
- E2: focused R1 domain/acquisition/packet/control-plane tests and hash/schema fixtures.
- E3: corrected-v3.2 synthetic negative tests; real context load twice, identity comparison only.
- Full pytest; physical-file py_compile; provenance; executable inventory; release and Git-free
  archive hygiene; `git diff --check`; final clean status.
- Falsifiers: differing error/schema/self-hash; a symlink/special/tampered tree accepted; existing
  destination overwritten; candidate/context identity drift; hidden fit/predict; tripwire crossing.

## 10. Result

- Commits: activation `d9a4eeb7`; exact direct-consumer amendment `293f59b8`; E2
  `3d8b4df1`; E3 `9ad57506`; this task/handoff closeout is metadata-only. E1 was the separately
  authorized deletion of the ignored `scripts/_fetch_tmp/`: exactly 26 files / 1,453 Python LOC,
  no tracked diff and no remaining tracked consumer.
- E2 introduced only internal `stylo._strict_fields.ExactFieldReader`. Representative baseline and
  candidate exception type/message probes were byte-equal; 289 owning tests and the full regression
  passed. Its tracked production delta is +202/-549 (`-347` LOC). The broader R1 packet changed
  16 -> 17 files, 695,027 -> 692,898 bytes and 19,606 -> 19,516 lines; the directly owning source
  packet changed 11 -> 12 files, 631,098 -> 621,097 bytes and 18,161 -> 17,814 lines.
- E3 replaced the Linux hostile-writer implementation with an ordinary cooperative local
  single-writer contract. Exact tree shape, modes, bytes, hashes, canonical metadata, scientific
  reconstruction, symlink/special-file rejection and no-overwrite/revalidation behavior remain;
  hardlink/inode/mount/concurrent-writer guarantees are explicitly not claimed. The preparation
  module changed 82,800 -> 70,686 bytes and 1,672 -> 1,425 lines; its owning test changed 684 ->
  484 lines. E3 tracked production delta is `-250` LOC and test delta is `-200` LOC.
- A real temporary bundle built from the frozen preparation protocol identity
  `02341845749431ba99fde0cac4335dcce86f9d0a3389c6c0382f6bcf077b6334` reproduced candidate
  `ff620b05...`, corpus manifest `a2dc0c4a...`, LOBO `117b8ec9...` and RuAA `d8428290...`.
  With the manifest's frozen chunker runtime identity, two independent read-only context loads both
  produced `2805aff9...`; a forbidden fit/predict hook was never reached.
- Review: one integrated deletion/adversarial review against the frozen criteria, baseline and
  result diff: **PASS**. No supported consumer, schema/error drift, scientific identity drift,
  overwrite path, positive production delta or tripwire violation was found; zero correction passes.
- Verification PASS: focused E2 and E3 suites; full pytest with a workspace basetemp; physical-file
  `py_compile`; provenance (93 sources + one output); 299-path executable inventory; checkout and
  Git-free archive hygiene/inventory/provenance; `git diff --check`. The first full run was invalid
  because old unrelated `/tmp` artifacts exhausted tmpfs inodes and is not counted as a pass.
- Non-blocking prerequisite for the separately authorized freeze/control-plane task: the accepted
  bundle binds frozen protocol bytes `023418...`, while later status-only edits made the current
  tracked document hash `4efcc7...`; the immutable input must be made explicit rather than silently
  substituting current prose. Also, corpus manifests bind chunker runtime identity spaCy 3.8.14,
  while `requirements.lock` currently installs 3.8.11. Read-only E3 identity equivalence is proven,
  but a normal locked-environment context load remains unavailable until that pre-existing runtime
  contract is reconciled. No dependency or evaluator scope was expanded here.
- Cumulative tracked delta is production +310/-907 (`-597` LOC), tests +19/-219 (`-200` LOC), and
  task/governance/handoff +116/-85 (`+31` LOC). New dependency/state/framework/registry/public
  entrypoint: 0.
  Production evaluator remains unregistered; freeze unapproved; preflight/authorization absent;
  execution hard-disabled; headline/publication not authorized. No fit, prediction, push, deploy or
  external write occurred.
