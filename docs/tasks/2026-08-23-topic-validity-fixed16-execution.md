# Fixed-16 topic-validity execution

- Status: active
- Owner: Dmitry Purtov
- Baseline: `b67372e0a26e0a96eb46c91be32372440973684f`
- Type / Risk: implementation + research / R3b
- Approval: inherited exact aggregate-only LOBO-248 authorization.

## Contract

- Change only `FIXED_WORKERS = 8` to exact `16` and matching governance/test expectations.
- Preserve Linux fork, ordered `Pool.imap(chunksize=1)`, 16-hour terminate-before-output deadline,
  warmed inherited cache, exact 992 task order, schema and output path.
- Synthetic serial/parallel aggregate bytes must remain identical; worker/deadline tests remain green.
- Pre-execution review verifies one-line semantic delta and measured memory/load basis.
- Execute once; stop with no output on >16 h, worker failure or memory pressure.
- No further worker-count change inside this task.

## Result

- Pending.
