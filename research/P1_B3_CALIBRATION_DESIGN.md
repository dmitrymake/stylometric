# P1 · B3 — group-aware calibration + work_balanced stack unblock

Status: **DRAFT for Codex audit.** Implements the signed `research/P1_WORK_BALANCED_DESIGN.md` §5
calibration clause ("group-aware калибровка (StratifiedGroupKFold); n_splits_eff =
min(n_splits, min_works_per_class); все классы в каждом split, иначе калибровка ОТКЛ целиком — нет
chunk-CV fallback") and lifts the B2-core preflight block on `work_balanced + stylo_stack`.

## 1. Group-aware calibrator selection (`eval/calibration.py`)

`choose_calibrator(oof, y, …, groups=None, n_splits=3)`:

- **`groups=None` — legacy, unchanged.** The single random held-out third of chunks, deterministic
  seed, identical passport. Byte-identical for the legacy stack (verified).
- **`groups` given — work_balanced.** The calibrator method is chosen on a **StratifiedGroupKFold by
  work** so a work's chunks are never split across fit/held-out (a chunk-level split leaks book
  identity into the *selection*, not just the fit). The score is a **pooled equal-work NLL**: the
  mean per-chunk NLL of each held-out work is **summed across all works** (every work is held out in
  exactly one fold) and divided by the **total number of works** — so each work carries equal weight
  regardless of its fold or chunk count (a fold-averaged mean would weight works unequally). The
  winner is refit on all OOF rows.
  - `min_works_per_class` = the smallest number of distinct works any class contributes.
  - **Fail-closed, no chunk-CV fallback:** if `min_works_per_class < 2` (cannot hold a work out for
    some class), or a class is absent from **either side** (train or validation) of any split,
    calibration is **disabled entirely** — the identity calibrator is returned with
    `calibration_disabled: True` and a reason. This is the signed "1-work author → calibration OFF"
    rule; the stack then falls back to **identity calibrators + an equal-weight ensemble (no
    meta-CV / meta-LR selection, `mode_` cannot become "stacked")** for that fit.
  - `n_splits_eff = min(n_splits, min_works_per_class)`.
  - **Fail-closed input contract (before any fit):** finite 2-D `oof`; 1-D non-bool integral `y` in
    `[0, n_classes)`; equal non-empty lengths; `validate_work_ids(groups, len(oof))`; exactly one
    label per work. `methods` is materialised to an ordered-unique tuple of known methods; `n_splits`
    must be a non-bool integer ≥ 2. All splits are built and both sides class-checked before any
    method is fit.

The four calibration methods (identity/temperature/Platt/isotonic) and their fits are unchanged.

## 2. Stack passes groups (`models/stacked_clf.py`)

Under `work_balanced` the stack calls `choose_calibrator(oof[name], y_local, groups=groups)`; under
legacy it passes `groups=None` (chunk-level, unchanged). This is the only stack change in B3 — the
B2a feature/loss wiring is untouched.

## 3. Unblock (`eval/lobo.py`, `eval/final.py`)

The B2-core preflight that raised `UnsupportedVariantError` for `work_balanced + stylo_stack` is
removed in `make_factory` and `run_final`; the factory now builds
`StackedChannelClassifier(training_weighting=work_balanced)`. `_variant_role` returns `primary` for
the stack (it is fully implemented), no longer `blocked_not_implemented`. The five-model
work_balanced suite is therefore runnable end-to-end (headline recompute/audit is B4).

## 4. Out of scope

Headline recompute/relabel and the paired legacy→work_balanced audit are **B4**. The deployment
`predict.run` work_balanced block is a separate deployment concern, unchanged here.

## 5. Acceptance tests

1. **Legacy parity:** `choose_calibrator(oof, y)` == `choose_calibrator(oof, y, groups=None)` (same
   method, same calibrated output); the legacy stack passport carries no `group_aware`.
2. **Disabled fail-closed:** a class with a single work → identity calibrator, `calibration_disabled`.
3. **`n_splits_eff`:** with ≥`n_splits` works per class the passport reports `n_splits` folds and
   is not disabled; `n_splits_eff = min(n_splits, min_works_per_class)`.
4. **Group-aware selection:** held-out NLL is computed over whole works (no chunk of a held-out work
   in the fit fold).
5. **Unblocked:** `make_factory("stylo_stack", weighting=work_balanced)` builds; the work_balanced
   stack fits + predicts; its calibration passport is `group_aware`; `_variant_role` == `primary`.
6. **Legacy stack unchanged:** predict/numeric parity with the pre-B3 legacy stack.
