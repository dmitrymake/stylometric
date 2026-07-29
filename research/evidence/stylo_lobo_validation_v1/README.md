# Stylo LOBO validation v1 evidence

This directory preserves the exact executable sources and independent verification record for the
completed historical A0/A4/A1 per-book LOBO artifact. It is historical evidence, not the current
implementation tree.

## Historical scientific record

The completed validation covered a universe of 47 authors and 255 works and tested 43 authors
across 251 held-out works. All `753/753` folds completed. A0 reproduced all 251 frozen reference
rows exactly at `221/251`. Primary A4 reached `227/251` (`+2.3904` percentage points;
author-clustered 95% CI `[-0.3571, +5.5085]`) and passed its signed noninferiority gate with
`δ = 0.02`; because the interval crosses zero, this is not a superiority result. Secondary A1
reached `228/251` (`+2.7888` points; interval `[+0.3817, +5.8333]`) and remains a secondary
mechanistic result.

This run was neither confirmatory nor an external replication. Its later narrative protocol was a
historical summary, not a preregistration. The corpus was subsequently registered as
`ineligible_for_new_scientific_runs` because of cross-work content overlap. The run is therefore
historical evidence only and cannot support a new scientific run or claim. See the
[`independent_audit.md`](independent_audit.md), [`SHA256SUMS`](SHA256SUMS), and
[ineligibility registry](../ineligible_corpus_registrations_v1.json).

The original basenames under `source/` intentionally reproduce the paths attested by the v1
`RUN.json` and recovery audit. Canonical maintained code uses purpose-based names elsewhere.

| Executed role | Original attested path | Preserved source | SHA-256 |
|---|---|---|---|
| validation runner | `scripts/run_b4_true_lobo.py` | `source/scripts/run_b4_true_lobo.py` | `01ddcd6d1d1c659ff80fd789abd286d608ecb9008595d8eaf04377bbd7ffa054` |
| fold/checkpoint evaluator | `src/stylo/eval/b4_true_lobo.py` | `source/src/stylo/eval/b4_true_lobo.py` | `f0165e6966d54fec77b751c8e2a5e1a8ed64866d84881c7846526e6fe264cc3a` |
| interrupted-run recovery bridge | `scripts/run_b4_true_lobo_kernel_compat.py` | `source/scripts/run_b4_true_lobo_kernel_compat.py` | `f4cb6863e17ce65166754d0c51e51cce254236586acfd3665037a30d3d674187` |

The exploratory helper `src/stylo/eval/b4_pilot.py` is recoverable byte-for-byte from attested Git
commit `2f6c3dc3`; its SHA-256 is
`0d978894babf28d1e03d543f36888df4216b4e77afcd996df3a86904db4b39a7`. The other 80 files in the
83-file code fingerprint are likewise bound by the run identity and recoverable from that commit.

Generated evidence remains ignored under `docs/exploratory/work_balanced/` and is not duplicated
here. [`SHA256SUMS`](SHA256SUMS) binds those generated files and the preserved source snapshot.
See [`independent_audit.md`](independent_audit.md) for the recomputation results.
