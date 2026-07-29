# Work-balanced training implementation contract

Normative current status lives only in [`../governance/status_ledger.json`](../governance/status_ledger.json).
This document defines executable behavior; it is not a status, history, or evidence log.
The `chunk_weighted_legacy` and `work_balanced` arms remain isolated implementations of one evaluation interface.

## Target mass and W/F/R axes

For one training fold, let `A` be the number of authors, `W_a` the number of works by author `a`,
`C_w` the chunks in work `w`, and `W = Σ_a W_a` the number of works. For chunk `i` from `w`:

`s_i = W / (A · W_a · C_w)`.

Thus each author has mass `W/A`, each of that author's works has mass `W/(A·W_a)`, and total mass is
`W`. Authors are equal; works are equal within author; chunk duplication or resegmentation changes
per-chunk weights but not work or author mass. Each work has one author; malformed inputs fail closed.

The audit axes are independent implementation mechanisms:

- **W** — equal-author/equal-work loss mass; for the stack it also selects group-aware calibration.
- **F** — vocabulary, document frequency, IDF, and other learned feature state fitted by work, including function words.
- **R** — function-word/MFW relative frequency with all analyzer events as denominator.

Legacy is `(W0,F0,R0)` and full work-balanced training is `(W1,F1,R1)`. One-axis combinations are audit-only;
model-specific applicability and registered cells belong to the paired-audit protocol.

## Dataset identity, provenance, and atomic subsets

`work_id` is a logical corpus and split identity, not a content hash. Equal text at different positions is
retained. Only a repeated full work-balanced row identity is a defect:

`(work_id, provenance_sha256, chunker_config_hash, span_ordinal, text_sha256)`.

Source, chunker, span, and text validation are manifest duties. Cross-work content isolation is a separate
fail-closed scientific-evaluation gate; neither `work_id` nor `text_sha256` substitutes for it.

Frozen `DatasetProvenance` binds loader kind, ordered row identities/authors, row count,
manifest/config/chunker identity, corpus policy, root, and a versioned boundary-unambiguous `rows_digest`
over current `texts`, `y`, `groups`, row identities, and authors.

`require_dataset_for_weighting` is both-sided: work-balanced accepts only a manifest dataset; legacy only
a recursive dataset. It recomputes digest/count and validates label range, unique authors, and
`authors[y[i]] == author(groups[i])`; mutation and hand-built provenance are rejected.
`prepare_scientific_evaluation` also rebinds to disk, verifies content isolation, and freezes arrays.

`derive_dataset(parent, indices)` is the only provenance-preserving subset builder. It rejects duplicate
or out-of-range indices, slices `texts/y/groups` together, recomputes authors, labels, row identities,
selection binding, and provenance, and returns one atomic `Dataset`.
`prepare_derived_scientific_evaluation` authorizes only a child bound to its verified parent.

## Work-level feature fit and transform

`WorkLevelVectorizer` fits on outer-train data only:

`chunk counts → work sums → work DF/prune → equal-work relative-TF rank → deterministic cap → work IDF`.

The smoothed state is `idf(f) = ln((W + 1)/(df_w(f) + 1)) + 1`; vocabulary/ties are deterministic
and accumulation uses `float64`.

Every relative denominator counts all analyzer events before DF pruning/cap; selected, pruned, and OOV
events remain. Grouped transform is `Σ work counts / Σ work analyzer events`, never a mean of chunk
ratios. A zero-event work remains one `W` vote and yields zero without division.

Frozen train state transforms test chunks: char/POS/punctuation use sublinear TF × work IDF then L2;
BoW uses counts; function words use relative all-word frequency. Delta uses `transform_grouped` for
one train row/work; prediction stays per chunk then work soft vote. DeltaCos normalizes work-z before centroid.

The historical `delta:N` route is a selected-mass Delta compatibility estimator, not canonical Burrows's
Delta: legacy `_rel_freq` divides by selected-MFW mass, not all events. R-on changes that denominator
explicitly; it does not relabel or silently alter legacy.

## Model-family routing

| family | legacy arm | work-balanced arm |
|---|---|---|
| `stylo` | plain pipeline, balanced class loss | `WorkBalancedStyloPipeline`; work-fitted blocks and fold-local weights |
| `char_cos` | chunk TF-IDF, work centroid | `WorkLevelVectorizer(mode=tfidf)`; work vocabulary/DF/IDF, same centroid |
| `delta` / `delta_cos` | selected-mass features, work centroid | work-ranked state and grouped all-event frequencies; same z/centroid math |
| `bow_lr` | pooled count pipeline, balanced class loss | cloneable `WorkLevelCountTransformer` in `WorkBalancedBowPipeline`; fold-local weights |
| `stylo_stack` | legacy channels/loss/calibration | work-fitted channels, fold-local loss weights, and grouped calibration; evaluation-only |

W-on weighted LR/SVC estimators use `class_weight=None`; legacy retains `class_weight="balanced"` without
work weights. `majority` has no weighting hook; the legacy BoW reference is WB-exploratory only.

## Single fit dispatch

`fit_estimator` is the only estimator-fit dispatch used by LOBO, GKF, and training:

```python
def fit_estimator(est, texts, y, groups):
    if getattr(est, "needs_groups", False):
        if groups is None:
            raise ValueError(f"{type(est).__name__} needs groups")
        est.fit(texts, y, groups=groups)
    else:
        est.fit(texts, y)
    return est
```

`training_weighting` is resolved once and passed explicitly. Estimators with `needs_groups=True` own
feature/weight routing; lower APIs neither reread configuration nor dispatch by class name.

## Group-aware calibration

With W on, `StratifiedGroupKFold` keeps works intact; each work has one label and every accepted split
has every class on both sides. Score is pooled equal-work held-out NLL: mean chunk NLL per work,
summed over works and divided by their count, not a mean of fold means.

`n_splits_eff = min(configured_n_splits, min_works_per_class)`. A singleton class or invalid grouped
splits disable calibration for the whole fit: identity calibrators plus equal ensemble, without
chunk-CV fallback, calibrated/raw mixing, or learned meta-selection. Legacy `groups=None` is unchanged.

## Stack feature and loss routing

`ChannelFn(train, test, train_groups=None)` receives groups only for work-balanced; legacy calls strict
two-argument form. Hash channels sum counts by train work before IDF and apply frozen IDF to chunks.
Block channels call `StyloVectorizer.fit_transform(..., groups=train_groups)` before scaling.

Every inner/full channel SVC and meta-CV/final meta-LR recomputes weights on its own fit rows, never
slicing a global vector. W-on uses `class_weight=None` with mass equal to that fit's work count;
legacy uses balanced loss/no weights. Splits stay grouped and calibration gets the validated groups.

## Output and artifact isolation

Legacy headline documents/author mapping gain no fields and stay byte-stable. Deployable legacy
model/Delta pickles are schema-versioned with prediction parity, not byte parity. Other results are versioned.

Work-balanced evaluation writes only below `docs/exploratory/work_balanced/`; training only below
`data/exploratory/work_balanced/`. The atomic bundle is exactly `model.pkl`, `delta.pkl`, and mandatory
`authors.json`; `assert_headline_write_allowed` blocks work-balanced headline writes.

Work-balanced provenance stays in a strict namespaced sidecar binding weighting, dataset digest,
config/code identity, and all three hashes; it never changes legacy artifacts or reuses their authors.

## Evaluation-only and paired-audit boundary

`stylo_stack` and `stylo_equal_channels_v1` retain rows/lazy final fit, are evaluation-only, and fail serialization closed with `EvaluationOnlyEstimatorError`.

Corpus freeze, applicability, authorization, checkpoints, inference, review, and any headline decision are governed
only by [`paired_audit_protocol.md`](paired_audit_protocol.md); this contract neither authorizes confirmatory execution nor replaces that protocol.
