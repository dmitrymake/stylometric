# Fixed-16 topic-validity execution

- Status: blocked
- Owner: Dmitry Purtov
- Baseline: `b67372e0a26e0a96eb46c91be32372440973684f`
- Type / Risk: implementation + research / R3b
- Approval: inherited exact aggregate-only LOBO-248 authorization.

## Result

- Review PASS; full regression PASS. Run reached `10/992` at 1,162.4 s, a 32.0 h projection, and was
  stopped with exit 130 before output. Fixed-16 contention is worse than measured fixed-8 throughput.
- Aggregate remained absent. No further worker-count increase is justified.
