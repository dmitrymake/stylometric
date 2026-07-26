# Stack feature and loss routing for work-balanced training

Status: **stack feature/loss routing is implemented and signed** in commit `c6335586`; group-aware
calibration is implemented and signed in `f1b8e165`. This implements the stack clause in
[`estimand.md`](estimand.md) §4 and composes with
[`group_aware_calibration.md`](group_aware_calibration.md). Stack feature/loss routing and
group-aware calibration are production-routable; the confirmatory paired audit remains incomplete
under [`paired_audit_protocol.md`](paired_audit_protocol.md).

## 1. Stack feature routing through `ChannelFn` (`models/channels.py`)

`ChannelFn` becomes `(train_texts, test_texts, train_groups=None) -> (Xtr, Xte)`. `train_groups=None`
reproduces the legacy chunk-level fit **byte-for-byte** (all existing callers — `run_benchmark.py`,
`log/experiments/*` — pass two args and are unaffected).

- **Hashing channels** (`ch_char`, `ch_word`): the `HashingVectorizer` is stateless; only the
  `TfidfTransformer` learns. Legacy fits IDF on the per-chunk count matrix. Work-balanced fits IDF on
  the **work-summed** count matrix `G @ counts` (`G` = one row per train work), so document frequency
  is counted **per work, not per chunk**. With `TfidfTransformer`'s default `smooth_idf=True` and
  `n_docs = W` (work rows) this is `idf = ln((W + 1) / (df_work + 1)) + 1` — the same smoothed form as
  the shared `WorkLevelVectorizer`. The per-chunk rows are then transformed with that frozen work-IDF —
  long books no longer inflate DF. `train_groups` passes the shared fail-closed
  `validate_work_ids` row-identity contract
  first (fail-closed on a bare str / mapping / set / generator / non-string id / length mismatch), so
  `G`, `W` and the work-DF can never be built from a malformed groups argument.
- **Block channels** (`block_channel`): `StyloVectorizer.fit_transform(train, groups=train_groups)`
  routes work-level vectorizer fitting (equal-work rel-TF MFW, work-DF prune, frozen work-IDF); the
  `MaxAbsScaler` is fit on the resulting work-balanced `Xtr`.

## 2. Stack loss routing in `StackedChannelClassifier` (`models/stacked_clf.py`)

The `training_weighting` keyword defaults to `chunk_weighted_legacy`. Under
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
positional arguments** `fn(tr, te)` — so any legacy channel with a two-argument signature still works
(no spurious `None` third argument). The inner/CV `StratifiedGroupKFold` **fold structure is
unchanged** (splitting is by book either way; work-balancing re-weights the loss, not the folds).

## 3. Group-aware calibration routing

Under `work_balanced`, the stack passes the fit's groups to
`choose_calibrator(oof[name], y_local, groups=groups)`. Calibrator selection uses grouped work folds
and equal-work held-out NLL; no work can appear on both sides of a calibration split. Under legacy,
the stack passes `groups=None`, preserving chunk-level selection and its numeric contract.

If any class contributes fewer than two training works, calibration is disabled for the whole fit:
the stack uses identity calibrators plus an equal-weight ensemble, with no chunk-level fallback and
no learned meta-selection. The work-balanced stack is therefore enabled in `eval/lobo.py` and
`eval/final.py` as a `primary` variant. The exact eligibility, validation, and passport contract is
defined in [`group_aware_calibration.md`](group_aware_calibration.md).

## 4. Legacy prediction/numeric parity (normative)

With `training_weighting=chunk_weighted_legacy` (the default) the stack reproduces the predecessor
estimator's **predictions**: `class_weight="balanced"`, no `sample_weight`, channel fns called with
two positional args, identical folds/calibrators/mode selection. The contract is
**prediction / numeric parity** for evaluation calls. Serialization is outside the contract and
fails closed: the current stack retains raw training rows and performs a lazy final fit during
`predict_proba`, so `pickle`/deployment is forbidden until a fit-time-finalized estimator version
replaces it. Numeric parity is verified by comparing `predict_proba` against the frozen predecessor
implementation.

## 5. Acceptance tests

1. **Legacy parity:** default `StackedChannelClassifier` and one built with
   `training_weighting=chunk_weighted_legacy` produce identical `predict_proba` on a fixed toy; the
   inner estimators carry `class_weight="balanced"` and no sample weight.
2. **WB loss:** under `work_balanced` the inner SVC / meta LR carry `class_weight is None`, and the
   fold-local `sample_weight` sums to that fit's work count `W`.
3. **WB feature:** a hashing channel fit with `train_groups` on sharply unequal chunk counts differs
   from the legacy chunk-IDF fit (work-DF ≠ chunk-DF); a block channel routes `groups` to the
   StyloVectorizer (work vocab differs from the pooled vocab).
4. **Group-aware calibration:** `make_factory("stylo_stack", weighting=work_balanced)` builds a
   `primary` variant; eligible fits use grouped calibrator selection, while a single-work class
   activates identity calibrators plus the equal-weight ensemble without chunk-level fallback.
5. **Serialization gate:** both unfitted and fitted stack/equal estimators raise
   `EvaluationOnlyEstimatorError` when serialization is attempted.
