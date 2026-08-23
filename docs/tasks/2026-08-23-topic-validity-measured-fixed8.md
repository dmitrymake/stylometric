# Measured fixed-8 final topic run

- Status: active
- Owner: Dmitry Purtov
- Baseline: `122e1bd567817c7a1ed4a0bac2383aeb29c4a254`
- Type / Risk: implementation + research / R3b
- Approval: original exact LOBO-248 aggregate-only authorization; owner explicitly requested less formalization.

## Contract

- Set `FIXED_WORKERS=8` and `MAX_EXECUTION_SECONDS=30*60*60`; no other runner/scientific change.
- Basis: sequential 98.4 h, fixed-8 26.4 h with safe memory, fixed-16 32.0 h from contention.
- Keep reviewed fork/order/warm/failure termination/schema/output and serial/parallel equivalence.
- One review of the two-constant diff, full verification, then one uninterrupted run.
- No more orchestration tuning in this task. Complete within 30 h or stop without output.

## Result

- Pending.
