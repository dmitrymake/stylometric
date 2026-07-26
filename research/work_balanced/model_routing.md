# Runtime model routing for the work-balanced estimand

Normative current status is
[`../governance/status_ledger.json`](../governance/status_ledger.json); historical commit labels,
future-tense prose, and line references in this design note are non-normative.

Status: **core model routing is implemented and signed** in commit `2c75c591`; stack feature/loss
routing and group-aware calibration are also implemented. The synthetic paired-audit control plane
and preparation command are implemented; confirmatory execution has not occurred and is
hard-disabled pending the approved freeze and evaluator gates. See
[`paired_audit_protocol.md`](paired_audit_protocol.md). This
document elaborates [`estimand.md`](estimand.md) §4. The contract covers single dispatch,
full-vocabulary BoW, the work-balanced namespace, the RuAA pin, and fail-closed routing. Empirical
facts below were verified against sklearn 1.7.2.

Contract map: **row identity → work-level vectorizer → core model routing → stack feature/loss
routing → group-aware calibration → paired audit**. Routing through calibration is implemented;
confirmatory execution and its independently audited result remain the outstanding deliverable.

## 1. Work-level vectorization for Delta / DeltaCos

`WorkLevelVectorizer.transform(mode=relative)` returns per-chunk `count/chunk_tokens`; averaging
those (`delta.py` `_group_means`) yields `mean(chunk_count/chunk_tokens)` — **wrong**. The signed
estimand (§3 L123) is `sum(work_counts)/sum(all_work_analyzer_events)`: sum across the work's
chunks **before** dividing.

**Grouped transform contract:**
`WorkLevelVectorizer.transform_grouped(docs, groups) -> (work_ids, rows)`:
per work `w`, `rows[w, f] = (Σ_chunks∈w counts[f]) / (Σ_chunks∈w all_analyzer_events)`, using the
**frozen** train vocabulary; denominator counts **all** analyzer tokens (OOV/pruned included),
not just selected features. Sparse aggregation `G @ counts` / `G @ events` (no densify).

**Delta work-balanced flow** (legacy Delta unchanged):
1. fit vocabulary/IDF on **outer-train only** via `WorkLevelVectorizer` (MFW by equal-work
   rel-TF ranking, §3);
2. `transform_grouped(train_docs, train_groups)` → **one row per train-work**;
3. assert **one label per work** (reuse `_group_means`' single-label check);
4. z-mean/std and author centroids computed on those **work rows** (unchanged math);
5. **DeltaCos** L2-normalizes each work-z **before** the centroid mean (unchanged math);
6. **prediction stays per-chunk** (per-chunk z vs centroid), then the existing book-level
   soft-vote of chunk probabilities.

Mandatory tests: sharply unequal chunk lengths (mean-of-ratios ≠ ratio-of-sums), OOV/pruned
tokens present in the denominator, a zero-event work (row = 0, still one of W), and a fixed
`vocabulary=` list.

## 2. Row identity and structured dataset provenance

`_manifest_paths` truthiness is spoofable and goes stale if `texts/groups` mutate. Replace with a
frozen, hashable `DatasetProvenance` whose digest is **unambiguous and complete**:

```
DatasetProvenance(frozen):
  digest_version: str                               # exact DIGEST_VERSION from eval/provenance.py
  loader_kind: "work_balanced_manifest" | "legacy_recursive"
  row_ids: tuple[RowIdentity]                        # one per row, ordered
  authors: tuple[str]                               # ordered; index == label
  n_rows: int
  manifest_hash, config_id, chunker_config_hash: str
  corpus_policy: CorpusPolicyProvenance             # immutable frozen object, NOT a dict
  frags_root: str
  rows_digest: str                                  # see canonical stream below
```

- **Canonical digest** = sha256 over a **versioned length-prefixed byte stream** (never bare
  concatenation): `digest_version`, then for every row in order the utf-8 length-prefixed
  `text`, the fixed-width `y`, the length-prefixed `group`, and the row's full identity tuple;
  then the length-prefixed ordered `authors`. Length-prefixing removes all field-boundary and
  encoding ambiguity; `authors` is inside the digest because permuting authors reassigns every
  `y`. (Equivalently: one strict-canonical-JSON document hashed once.)
- **`RowIdentity`.** WB rows use the **full canonical identity** (`workdoc.py:7`):
  `(work_id, provenance_sha256, chunker_config_hash, span_ordinal, text_sha256)`. Legacy rows
  use `(group, within_group_ordinal, text_sha256)` (no manifest hashes exist in the recursive
  loader).
- **`CorpusPolicyProvenance`** is a frozen dataclass `(exclude_from_benchmark: tuple[str] sorted,
  unknown_dir_name: str)` — hashable, not a dict.

`require_dataset_for_weighting(dataset, weighting)` at the entry of `run_final`, `lobo_evaluate`,
`gkf_evaluate`, `train.run` — **both-sided**, fail-closed:
- `work_balanced` ⇒ `loader_kind == "work_balanced_manifest"`; `chunk_weighted_legacy` ⇒
  `loader_kind == "legacy_recursive"`;
- a **recomputed** `rows_digest` + `n_rows` over the current `texts/y/groups/authors` matches the
  stored one (catches stale mutation / hand-built Datasets);
- **semantic checks:** every `y[i] ∈ [0, len(authors))`; `authors` are unique; and
  `authors[y[i]]` equals the author parsed from `groups[i]` (`"author/work".split("/")[0]`).

Subsetting is provided **only** by an atomic `derive_dataset(parent, indices) -> Dataset`
(see §2.1); hand-built `Dataset` objects are rejected on any provenance-bearing path. `frags_root`,
`exclude_from_benchmark`, and `unknown_dir_name` are threaded **identically** through every
CLI / train / evaluate / sweep path (`cli.py:129,143,157,174`).

### 2.1. Atomic subset derivation

`derive_dataset(parent_dataset, indices) -> Dataset` is the **only** way to build a subset on a
provenance-bearing path (replacing manual construction in `run_ruaa_baselines.py:76`
`bench_subset`). It:
- requires `parent` to carry a valid `DatasetProvenance` (else reject);
- validates `indices` — reject duplicates and out-of-range;
- slices `texts/y/groups` from the parent and **recomputes** `authors` (sorted unique present)
  with a consistent relabelling of `y`;
- recomputes the full `DatasetProvenance` (row identities drawn only from the validated parent,
  fresh digest, same `loader_kind`);
- returns the `Dataset` **and** its provenance as one object.

## 3. Cloneable work-level BoW transformer

Verified: `sklearn.clone(WorkLevelVectorizer)` → `TypeError: Cannot clone` (no `get_params`);
`Pipeline.fit(bow__groups=…)` → `TypeError: fit() got multiple values for argument 'groups'`
(its `fit(docs, groups)` collides with Pipeline's positional `y`).

New **`WorkLevelCountTransformer(BaseEstimator, TransformerMixin)`** (thin, delegates to the
signed `WorkLevelVectorizer(mode=count)` math — invents nothing):

```
fit(self, X, y=None, *, groups)          # keyword-only groups; absorbs Pipeline's positional y
fit_transform(self, X, y=None, *, groups)
transform(self, X)
get_feature_names_out(self, input_features=None)
```

Constructor params (`analyzer_params, max_features, min_df_works, sublinear_tf, vocabulary`) are
plain attributes → `BaseEstimator` gives `get_params`/`set_params`/clone for free.
`WorkBalancedBowPipeline(Pipeline)` = `[("bow", WorkLevelCountTransformer), ("scaler",
MaxAbsScaler), ("lr", LR(class_weight=None))]`; its `fit` routes `bow__groups` and
`lr__sample_weight` (§4).

## 4. Core model fit routing and ambient metadata

`WorkBalancedStyloPipeline(Pipeline)` and `WorkBalancedBowPipeline(Pipeline)` subclass Pipeline
(keep `named_steps`, `get_params`, clone). `needs_groups = True`. No incompatible `__init__`
(inherit the `steps` signature). `fit`:

```
def fit(self, X, y=None, *, groups, **fit_params):
    if y is None: raise ValueError("work_balanced fit needs y")
    groups = validate_work_ids(groups, len(X))                    # row-identity contract
    if {"<vec>__groups", "<clf>__sample_weight", "sample_weight"} & set(fit_params):
        raise ValueError("groups/weights are computed internally; do not pass them")
    w = work_sample_weights(y, groups)                            # fold-local, sum = W_train
    import sklearn
    with sklearn.config_context(enable_metadata_routing=False):   # stable step-scoped routing
        return super().fit(X, y, **{"<vec>__groups": groups, "<clf>__sample_weight": w}, **fit_params)
```

Verified: with `enable_metadata_routing=True`, sklearn 1.7.2 rejects `Pipeline.fit(<step>__param)`
(`Pipeline.fit got unexpected argument(s) …, which are not routed`). Wrapping `super().fit` in
`config_context(enable_metadata_routing=False)` forces the legacy step-scoped routing so **both
ambient modes yield the identical estimand** (asserted by a test that fits under both).

## 5. Runtime weighting selection and per-row result provenance

- The `training_weighting` enum is resolved **once** in the CLI
  (`resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))`).
- `run_final`, `lobo_evaluate`, `gkf_evaluate`, `train.run`, `make_factory` take a **required
  keyword-only** `weighting` enum; **lower APIs never re-read cfg**. An AST guard (test) asserts
  every production call site passes the explicit arg. No boolean builder anywhere.
- One global label is insufficient for reference/blocked rows. Each leaderboard row carries five
  **distinct** fields:
  - `suite_weighting` — the arm (`chunk_weighted_legacy` | `work_balanced`);
  - `dataset_contract` — `legacy_recursive` | `work_balanced_manifest`;
  - `estimator_training_weighting` — what **this** estimator actually did;
  - `variant_role` — enum `primary | reference | not_applicable | blocked_not_implemented`;
  - `claim_status` — the **existing 5-value `ClaimStatus`, unchanged/not extended**.

## 6. Core model routing by model

| spec | legacy arm (prediction-parity) | work_balanced arm | variant_role |
|---|---|---|---|
| **stylo** | plain `make_full_pipeline`, `class_weight="balanced"` | `WorkBalancedStyloPipeline` (§4), StyloVectorizer fitted through the work-level vectorizer contract | primary |
| **delta / delta_cos** | chunk MFW + z-mean centroids | §1 work-balanced feature state; centroid/z unchanged | primary |
| **char_cos** | chunk `TfidfVectorizer` + work centroid | vocab/DF/IDF via `WorkLevelVectorizer(mode=tfidf)`; centroid unchanged | primary |
| **bow_lr** | frozen `build_bow_lr` | `WorkBalancedBowPipeline`, full work-balanced route (§3) | primary |
| **bow_lr_ref_legacy** | — (absent) | frozen historical BoW, **WB-namespace only** | reference |
| **majority** | chunk-count prior | same estimator (no weighting hook) | not_applicable |
| **stylo_stack** | legacy stack route | `StackedChannelClassifier(training_weighting=work_balanced)` with work-routed channels, fold-local loss weights, and group-aware calibration | primary |

`bow_lr_ref_legacy` appears **only** in the WB exploratory namespace — never in the legacy
`final_comparison.csv` (there it would duplicate `bow_lr` and break byte-parity).

### 6.1. Stack feature/loss and calibration routing

The stack route is implemented end to end. Channel feature state is learned per work, every
weighted inner/full estimator receives fold-local work weights, and calibrator selection receives
work groups. If any class has fewer than two training works, calibration fails closed to identity
calibrators plus the equal-weight ensemble; it never falls back to chunk-level calibration. See
[`stack_routing.md`](stack_routing.md) and
[`group_aware_calibration.md`](group_aware_calibration.md) for the exact contracts.

## 7. Output and artifact isolation

- **Legacy headline DOCS files stay byte-identical, with NO new fields**: `docs/final_comparison.*`,
  `docs/lobo_books.txt`. The deployment pickles `data/model.pkl`/`data/delta.pkl` are **not**
  byte-identical (routing adds `_ctor_state`/`_wv`/`training_weighting` attrs) — the honest contract
  for them is **prediction parity** (identical predictions), with the schema versioned via the
  deployable `stylo`/Delta estimators' `__setstate__` migration. This deployment contract explicitly
  excludes the evaluation-only `stylo_stack` and `stylo_equal_channels_v1`, whose serialization
  fails closed while they retain training rows and use lazy final fit. `docs/sweep_table.*` numbers
  change (GKF groups fix) and are versioned as `sweep_table.v2.*` (§7 output isolation).
- **Work_balanced writes only under a separate root**: eval → `docs/exploratory/work_balanced/…`;
  train → `data/exploratory/work_balanced/{model.pkl, delta.pkl, authors.json}`. The WB train
  bundle **must** include `authors.json` (legacy train writes it at `train.py:56`) so the
  experimental model cannot silently reuse the legacy class mapping. A
  `assert_headline_write_allowed(weighting)` guard fail-closes any headline-path write when
  `weighting != chunk_weighted_legacy`.
- WB provenance lives in a **strict sidecar run-manifest inside the WB namespace** — weighting,
  `DatasetProvenance` digest, `config_id`, git/code hash, and a hash of **all three** bundle
  files (`model.pkl`, `delta.pkl`, `authors.json`) — never smeared onto legacy artifacts.
- **RuAA pin (normative):** `scripts/run_ruaa_baselines.py` loads via the legacy
  recursive loader and passes `weighting=CHUNK_WEIGHTED_LEGACY` explicitly to `run_final`; its
  hardcoded `"training_weighting": "chunk_weighted_training_legacy"` label (`:118`) thus matches
  what actually ran and cannot drift.
- **GKF proxy fix (answer 1):** adding `groups` to `needs_groups` models in GKF corrects a
  cross-engine estimand bug; legacy GKF proxy numbers may change and are therefore **versioned**
  (new artifact, e.g. `sweep_table.v2.*`) — the old sweep is **not silently overwritten**.
  Byte-parity is guaranteed for **LOBO / headline**, not for the corrected GKF proxy.

## 8. Legacy byte and prediction compatibility

Default (`chunk_weighted_legacy`) arm: `resolve_dataset → load_dataset`; `make_factory` returns
the plain pipelines/baselines; `fit_estimator` reproduces the production `needs_groups` routing for
LOBO (the only legacy-path difference is groups→GKF, a versioned proxy fix, §7). No config
default change; `to_claim_label` unchanged; no legacy artifact gains fields.

## 9. Paired-audit boundary

Core model routing, stack feature/loss routing, and group-aware calibration are implemented. The
confirmatory legacy-versus-work-balanced paired audit, its RuAA sensitivity panel, and any resulting
headline recompute/relabel remain governed by
[`paired_audit_protocol.md`](paired_audit_protocol.md) and have not yet been completed. SOCIOLIT-lite
remains outside that confirmatory audit.

## 10. Single fit dispatch: `fit_estimator`

`eval/dispatch.py`: the one fit entrypoint for LOBO / GKF / train.
```
def fit_estimator(est, texts, y, groups):
    if getattr(est, "needs_groups", False):
        if groups is None: raise ValueError(f"{type(est).__name__} needs groups")
        est.fit(texts, y, groups=groups)
    else:
        est.fit(texts, y)
    return est
```

## 11. Core model routing acceptance matrix

**Adapters (`WorkBalancedStyloPipeline`, `WorkBalancedBowPipeline`, `WorkLevelCountTransformer`)**
1. `sklearn.clone` returns an unfitted, independent estimator; `get_params`/`set_params`
   round-trip; `named_steps` present; `joblib.dump`/`load` round-trip; `classes_` exposed after fit.
2. `fit` produces the **identical** fitted state under `enable_metadata_routing` **True and
   False** (same estimand either ambient mode).
3. Reserved params rejected: passing `vectorizer__groups` / `classifier__sample_weight` /
   bare `sample_weight` (BoW: `bow__groups` / `lr__sample_weight`) raises.
4. Exact weights: `classifier__sample_weight == work_sample_weights(y, groups)`, `sum == W_train`;
   the inner LR has `class_weight is None`; legacy pipeline has `class_weight == "balanced"` and
   receives no weights/groups.
5. `WorkLevelCountTransformer.fit` and `.fit_transform` agree; `get_feature_names_out` stable;
   keyword-only `groups` (positional `groups` is a `TypeError`).
6. **No stale cache across groups:** same docs + different `groups` ⇒ different fitted vocab/state
   (any memoization keys on `groups`; reps stay fold-independent/leak-free).

**Delta / CharCos work-balanced feature state (§1)**
7. Sharply unequal chunk lengths: `transform_grouped` gives `Σcounts/Σevents`, **not**
   `mean(count/tokens)` (the two differ); zero-event work → zero row, still one of W; OOV/pruned
   tokens are in the denominator; fixed `vocabulary=` path honored.

**Provenance & subset (§§2, 2.1)**
8. Canonical digest is order- and boundary-unambiguous and **changes** when any of texts, `y`,
   `groups`, or `authors` changes (incl. an authors permutation).
9. `require_dataset_for_weighting` is both-sided: WB rejects a legacy Dataset and a hand-built /
   mutated Dataset (recomputed digest mismatch); legacy rejects a WB Dataset; semantic checks
   (y-range, unique authors, `authors[y[i]]==author(groups[i])`) all fail-closed.
10. `derive_dataset` rejects duplicate/out-of-range indices, recomputes authors/y/digest, and the
    result passes the guard; a manually built subset does not.

**Dispatch, isolation, and stack routing (§§5-7, 6.1)**
11. AST/import guard: every production call to `run_final`/`lobo_evaluate`/`gkf_evaluate`/
    `train.run`/`make_factory` passes an explicit keyword `weighting`.
12. `assert_headline_write_allowed` blocks any headline-path write under work_balanced; WB output
    lands only under the WB namespace and carries the five per-row fields + sidecar (hashing
    `model.pkl`+`delta.pkl`+`authors.json`).
13. `work_balanced + stylo_stack` builds as a `primary` variant, routes groups through feature,
    loss, and calibration stages, and uses the fail-closed equal-ensemble fallback when grouped
    calibration is ineligible.

**Legacy deployable serialization contract (normative; excludes stack/equal)**
14. In the legacy arm:
    - `docs/final_comparison.*`, `docs/lobo_books.txt` and `data/authors.json` are **byte-identical**
      to the frozen legacy reference and gain **no** new fields; `to_claim_label` unchanged.
    - `data/model.pkl` / `data/delta.pkl` are **schema-versioned** (`ARTIFACT_SCHEMA_VERSION = 2`,
      a versionless pickle = v1) and MAY differ byte-for-byte (routing adds `_ctor_state`/`_wv`/
      `training_weighting`/`_schema_version`). Their contract is **prediction parity**, verified by:
      (a) a **golden predecessor** artifact loads → predicts → refits; (b) a fresh legacy retrain
      preserves `classes_` order and feature names, and predictions/argmax; probabilities/distances
      within a fixed tolerance.
