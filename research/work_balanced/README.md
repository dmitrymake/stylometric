# Work-balanced authorship evaluation

This directory owns the complete work-balancing research thread. File names state the scientific
or runtime responsibility; implementation chronology and commit ids live inside the files.

- [`estimand.md`](estimand.md) — defines equal-author/equal-work training mass and core invariants.
- [`model_routing.md`](model_routing.md) — maps the estimand into each model family and provenance
  gate.
- [`stack_routing.md`](stack_routing.md) — applies work grouping to stack features and fitted losses.
- [`group_aware_calibration.md`](group_aware_calibration.md) — selects and fits calibration without
  splitting a work across folds.
- [`exploratory_ablation_screen.md`](exploratory_ablation_screen.md) — records the completed bounded
  screen that selected mechanisms for full validation.
- [`stylo_lobo_validation_protocol.md`](stylo_lobo_validation_protocol.md) — records the completed
  and independently verified per-book LOBO comparison for the stylo cells.
- [`paired_audit_protocol.md`](paired_audit_protocol.md) — specifies the still-incomplete
  confirmatory legacy-versus-work-balanced audit across the registered grid.

Stable cell ids remain inside artifacts because they denote exact scientific variants:

- `A0`: legacy chunk-weighted training;
- `A1`: weights-only work balancing;
- `A4`: complete work-balanced stylo path.

They are not directory or file names because they do not explain responsibility on their own.
