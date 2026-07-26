# Independent audit of the completed stylo LOBO validation

Status: **passed**, 2026-07-20. The audit was read-only and independently recomputed the artifact
from the durable checkpoint set.

## Integrity

- 758 strict-JSON files inspected.
- 22,998 explicit conditions checked.
- All 753 fold checkpoints have valid canonical self-hashes.
- Reassembly from checkpoints is byte-for-value identical to the final artifact.
- Run id: `5598ed784c4e42027daa80297f7722a6990783a8fe2cad2ea6fa21994854fc04`.
- Artifact self-hash: `0e44573c354042eb58af94de034343d3f8378cb2f3560a8cf6fa3cd8b4d05bf6`.
- Artifact file SHA-256: `13e11f8eb825be3e7f8b1342a5f12aa5afef5e836c02bbbf204f8505edca28b9`.

The corpus was reread from disk: 23,226 chunks, 47 classes, 255 works, and 251 tested works. The
dataset digest, ordered inventories, class bindings, fold splits, and four train-only singleton
authors match the saved run identity exactly.

All 753 probability vectors are finite, non-negative, normalized to within `1e-15`, 47-wide, free
of top-1 ties, and consistent with their stored prediction, author, rank, and correctness fields.

## Recomputed results

| Cell | Correct | Accuracy | Macro-F1 | Top-2 |
|---|---:|---:|---:|---:|
| A0 | 221/251 | 0.880478 | 0.839785 | 0.904382 |
| A4 | 227/251 | 0.904382 | 0.873651 | 0.932271 |
| A1 | 228/251 | 0.908367 | 0.879711 | 0.928287 |

- A4−A0: `+0.023904`, author-clustered 95% CI `[-0.003571, +0.055085]`, 8 gains / 2 losses.
- A1−A0: `+0.027888`, author-clustered 95% CI `[+0.003817, +0.058333]`, 8 gains / 1 loss.
- A1−A4: `+0.003984`, author-clustered 95% CI `[0, +0.013158]`, 1 gain / 0 losses.

A0 matches all 251 frozen reference rows exactly. The signed A4 decision is `noninferior` because
its lower interval bound is greater than `-0.02`. Its interval crosses zero, so this is not evidence
of superiority. A1 remains a secondary mechanistic result, not a confirmatory endpoint.

## Interrupted-run provenance

The recovery audit is internally valid and binds the final file SHA. Its start counts were
`251/126/0`; finish counts are `251/251/251`. Only the explicitly non-binding OS platform and release
observations changed. All numerical runtime, thread, code, configuration, corpus, cache, class, and
work-order bindings matched exactly.
