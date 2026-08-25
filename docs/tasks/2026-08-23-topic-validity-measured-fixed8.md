# Measured fixed-8 final topic run

- Status: active
- Owner: Dmitry Purtov
- Baseline: `122e1bd567817c7a1ed4a0bac2383aeb29c4a254`
- Type / Risk: implementation + research / R3b
- Approval: original exact LOBO-248 aggregate-only authorization; owner explicitly requested less formalization.

## Contract

- Run with eight worker processes; the run is resumable and continues from its checkpoint after any stop.
- Basis: sequential 98.4 h, fixed-8 26.4 h with safe memory, fixed-16 32.0 h from contention.
- Keep reviewed fork/order/warm/failure termination/schema/output and serial/parallel equivalence.
- No more orchestration tuning in this task.

## Result

- Pending.
