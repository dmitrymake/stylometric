# Stylo per-book LOBO validation of work-balanced training

Status: **completed and independently verified**, 2026-07-20. This protocol tested the selected
work-weighting signal on the real target LOBO universe. It is narrower than the confirmatory
[`paired_audit_protocol.md`](paired_audit_protocol.md) and is not an external replication.

## Scientific question

Does work-balanced training preserve or improve the existing stylo result when every tested book is
held completely outside training and the four singleton-author books remain train-only?

- Full corpus: 47 authors, 255 works.
- Tested universe: 43 authors, 251 held-out works.
- Outer split: exactly one complete work held out per fold.
- Training fold: all other 254 works and all 47 classes.
- Probability order: fixed 47-author order.
- Point metrics: 251 tested works, with macro-F1 over the fixed 43 tested-author order.

Run the same ordered folds for exactly three stylo cells:

- `A0`: legacy chunk-weighted training and the frozen parity baseline;
- `A4`: complete work-balanced training, the pre-specified primary comparison;
- `A1`: weights-only secondary mechanism check selected by the exploratory screen.

No A2/A3 cell, stack, other model family, or result-dependent tuning may enter this run.

## Frozen baseline gate

A0 must pass before results from A4 or A1 are interpreted:

- `docs/lobo_books.txt` SHA-256 is
  `26db64475e77657eaec6db895c55bad8bcd513344584ef5a64e9a580cf9f648d`;
- exact accuracy is `221/251 = 0.8804780876494024`;
- prediction, correctness, and rank match the reference for every work;
- tested work order, truth, fold identity, and class order match exactly;
- every probability vector is finite, normalized, and 47-wide.

Any failure stops the comparison; the reference is never repaired to fit a new result.

## Resumable execution contract

One fresh estimator is fit for each `(cell, held-out work)`. Every chunk of the held-out work is
excluded and no other work is removed. Representation data may be cached once, while learned model
and feature state remains fold-local. Parallelism is only across outer folds, with one numerical
thread per worker.

Each completed fold is atomically stored next to the output and is reused only when all scientific
identity fields match: code, configuration, corpus, representation cache, work and class order,
cell, fold, and numerical runtime contract. Kernel and operating-system release strings are neither
collected nor used as identity or resume criteria.

Execution order is `A0 → A4 → A1`. The durable checkpoint root is
`docs/exploratory/work_balanced/b4_true_lobo_a0_a1_a4_v1.checkpoints/`; the final ignored artifact is
`docs/exploratory/work_balanced/b4_true_lobo_a0_a1_a4_v1.json`. A missing checkpoint means pending;
an inconsistent checkpoint fails closed.

The maintained implementation now lives at `src/stylo/eval/stylo_lobo_validation.py` with CLI
`scripts/evaluation/run_stylo_lobo_validation.py`. Exact executed v1 bytes remain preserved under
[`../evidence/stylo_lobo_validation_v1/`](../evidence/stylo_lobo_validation_v1/) for provenance.

## Output and inference

The strict-JSON artifact records provenance, fixed class/work orders, per-work truth/prediction/rank,
47-wide probabilities, accuracy, macro-F1, top-2, per-author recall, paired gains/losses, timing,
checkpoint inventory, leave-one-author-out sensitivity, and a canonical self-hash.

Paired accuracy differences use an author-clustered percentile bootstrap with 10,000 iterations,
seed 42, and a 95% interval.

The primary A4 noninferiority decision uses the unrounded A4−A0 interval and margin `δ = 0.02`:

- `noninferior` if the lower bound is strictly greater than `-0.02`;
- `inferior` if the upper bound is strictly less than `-0.02`;
- `inconclusive` otherwise.

A1 is a secondary mechanistic validation. Its interval is descriptive, not a new confirmatory
endpoint, and no post-hoc multiplicity story may be added.

## Completion gate

Completion requires all 251 folds for all three cells, exact A0 parity, deterministic final
assembly, self-hash verification, focused checkpoint/gate tests, live A0/A4 golden replay, the full
test suite, an independent result review, and an explicit commit decision.

This run must not mutate the corpus, frozen references, configuration, lockfiles, public headline,
paper, README, or site artifacts. It does not build the audit corpus, use RuAA, or replace the full
paired audit and later external replication.

## Verified result

All `753/753` folds completed and independently reassembled from checkpoints. A0 reproduced exactly
at `221/251`. A4 reached `227/251` (`+2.3904` percentage points, author-clustered 95% CI
`[-0.3571, +5.5085]`) and passed the signed `noninferior` gate; its interval crosses zero, so this is
not a superiority claim. Secondary A1 reached `228/251` (`+2.7888` points, interval
`[+0.3817, +5.8333]`). Full integrity and interpretation details are recorded in
[`../evidence/stylo_lobo_validation_v1/independent_audit.md`](../evidence/stylo_lobo_validation_v1/independent_audit.md).
