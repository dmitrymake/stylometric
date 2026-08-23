# Fixed-8 topic-validity execution

- Status: active
- Owner: Dmitry Purtov
- Baseline: `5e581f850b166c3abef3fdcb2e511a298ddbda5a`
- Type / Risk: implementation + research / R3b
- Approval: inherited exact owner authorization for LOBO-248 A0/A4 current/topic_strict aggregate-only study.

## Goal

Replace only the sequential orchestration with a fixed eight-process fork pool after representation
warm-up, prove byte-identical synthetic aggregate versus serial execution, then complete the approved
992 fits and retain the same single aggregate schema.

## Frozen scope

- One existing runner edit and focused tests; aggregate module/schema/factories/fold math unchanged.
- Exact task order remains A0/current, A0/topic_strict, A4/current, A4/topic_strict and fold 0..247.
- Parent warms all representations before fork. Workers inherit the sealed study and in-memory cache;
  each task still constructs a fresh estimator and returns one transient fold record.
- `executor.map` preserves submitted order; worker count is exact integer `8`, not a CLI selector.
- Output remains create-once aggregate-only. No worker writes artifact/cache/checkpoint/log detail.
- No reduced folds, A1–A3, RuAA, other model, official receipt/result, publication or registration.

## Acceptance

- [ ] Synthetic serial and fixed-8 transient records/aggregate are exact-tree and byte identical.
- [ ] Worker exception aborts before output; no partial aggregate or worker-side persisted detail.
- [ ] Source/runtime/thread/process-start identities bind the execution.
- [ ] One independent review; at most one correction; focused/full tests and hygiene pass.
- [ ] Warm cache hit path is bounded; full 992 fits complete within 16 h or stop without output.
- [ ] Aggregate independently validates and retains no forbidden detail.

## Tripwires

- Runner +100 LOC, tests +160 LOC; no module/schema/dependency/framework/state change.
- Fixed Linux `fork` only; unavailable fork is a hard stop, never silent sequential fallback.

## Result

- Pending.
