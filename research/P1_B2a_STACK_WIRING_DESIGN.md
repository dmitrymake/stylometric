# P1 · B2a — stylo_stack work-balanced feature/loss wiring

Status: **DRAFT for Codex audit.** Implements the signed `research/P1_WORK_BALANCED_DESIGN.md` §4
stack clause ("Stacked: веса в inner/full SVC + meta-CV + meta-LR; ChannelFn получают groups") and
the six-point patch clause 4 ("class_weight=None + sample_weight во ВСЕХ weighted; legacy=balanced
без весов"). Scope is **feature + loss only**; the stack stays **blocked** under `work_balanced`
until B3 wires group-aware calibration (`research/P1_B2_MODEL_WIRING_DESIGN.md` §6a).

## 1. Feature side — `ChannelFn` gains work groups (`models/channels.py`)

`ChannelFn` becomes `(train_texts, test_texts, train_groups=None) -> (Xtr, Xte)`. `train_groups=None`
reproduces the legacy chunk-level fit **byte-for-byte** (all existing callers — `run_benchmark.py`,
`log/experiments/*` — pass two args and are unaffected).

- **Hashing channels** (`ch_char`, `ch_word`): the `HashingVectorizer` is stateless; only the
  `TfidfTransformer` learns. Legacy fits IDF on the per-chunk count matrix. Work-balanced fits IDF on
  the **work-summed** count matrix `G @ counts` (`G` = one row per train work), so document frequency
  is counted **per work, not per chunk**. With `TfidfTransformer`'s default `smooth_idf=True` and
  `n_docs = W` (work rows) this is `idf = ln((W + 1) / (df_work + 1)) + 1` — the same smoothed form as
  the B1 `WorkLevelVectorizer`. The per-chunk rows are then transformed with that frozen work-IDF —
  long books no longer inflate DF. `train_groups` passes the single B1 `validate_work_ids` contract
  first (fail-closed on a bare str / mapping / set / generator / non-string id / length mismatch), so
  `G`, `W` and the work-DF can never be built from a malformed groups argument.
- **Block channels** (`block_channel`): `StyloVectorizer.fit_transform(train, groups=train_groups)`
  routes the B1 work-level fitting (equal-work rel-TF MFW, work-DF prune, frozen work-IDF); the
  `MaxAbsScaler` is fit on the resulting work-balanced `Xtr`.

## 2. Loss side — `StackedChannelClassifier(training_weighting=…)` (`models/stacked_clf.py`)

New required-by-default keyword `training_weighting` (default `chunk_weighted_legacy`). Under
`work_balanced`, **every** weighted estimator drops `class_weight="balanced"` for
`class_weight=None` and receives an explicit `sample_weight`; the weights are the **fold-local**
`work_sample_weights(y_sub, groups_sub)` (recomputed on each fit's own rows so the mass sums to the
fit's `W_train`, matching `WorkBalancedStyloPipeline`), never a subset of a global vector:

| estimator | rows fit on | legacy | work_balanced |
|---|---|---|---|
| inner-fold channel SVC | `tr_i` chunks | `class_weight="balanced"` | `class_weight=None`, `sample_weight=w(y[tr_i], g[tr_i])` |
| full channel SVC (predict) | all train chunks | `class_weight="balanced"` | `class_weight=None`, `sample_weight=w(y, g)` |
| meta-CV LR | `tr_i` OOF rows | `class_weight="balanced"` | `class_weight=None`, `sample_weight=w(y[tr_i], g[tr_i])` |
| full meta LR | all OOF rows | `class_weight="balanced"` | `class_weight=None`, `sample_weight=w(y, g)` |

Under `work_balanced` the channel functions are called with the fit's train groups (`fn(tr, te,
g[tr_i])` inner, `fn(tr, te, g)` at predict); under legacy they are called with the **strict two
positional arguments** `fn(tr, te)` — so any pre-B2a channel with a 2-arg signature still works
(no spurious `None` third argument). The inner/CV `StratifiedGroupKFold` **fold structure is
unchanged** (splitting is by book either way; work-balancing re-weights the loss, not the folds).

## 3. Calibration is UNTOUCHED (B3)

`choose_calibrator` still runs on the chunk-level OOF scores with no groups. Because chunk-level
calibration must not run under work-balancing, the preflight in `eval/lobo.py:make_factory` and
`eval/final.py:run_final` **still raises `UnsupportedVariantError`** for `work_balanced + stylo_stack`.
The B2a feature/loss path is therefore **dormant** in the run engines and is exercised only by direct
construction in tests, exactly as sequenced. B3 threads `groups` into `choose_calibrator` and lifts
the block.

## 4. Legacy prediction/numeric parity (normative)

With `training_weighting=chunk_weighted_legacy` (the default) the stack reproduces the pre-B2a
estimator's **predictions**: `class_weight="balanced"`, no `sample_weight`, channel fns called with
two positional args, identical folds/calibrators/mode selection. The contract is
**prediction / numeric parity**, not byte-for-byte pickle parity — the new `training_weighting` and
`_train_groups` attributes necessarily change the pickled `__dict__`. Verified by comparing
`predict_proba` against the HEAD estimator (identical to the recorded digest).

## 5. Acceptance tests

1. **Legacy parity:** default `StackedChannelClassifier` and one built with
   `training_weighting=chunk_weighted_legacy` produce identical `predict_proba` on a fixed toy; the
   inner estimators carry `class_weight="balanced"` and no sample weight.
2. **WB loss:** under `work_balanced` the inner SVC / meta LR carry `class_weight is None`, and the
   fold-local `sample_weight` sums to that fit's work count `W`.
3. **WB feature:** a hashing channel fit with `train_groups` on sharply unequal chunk counts differs
   from the legacy chunk-IDF fit (work-DF ≠ chunk-DF); a block channel routes `groups` to the
   StyloVectorizer (work vocab differs from the pooled vocab).
4. **Still blocked:** `make_factory("stylo_stack", weighting=work_balanced)` and the `run_final`
   preflight both raise `UnsupportedVariantError`; calibration code is unchanged.
