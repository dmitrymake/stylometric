# Work-balanced training estimand and implementation contract

Normative current status is
[`../governance/status_ledger.json`](../governance/status_ledger.json); historical commit labels,
future-tense prose, and line references in this contract are non-normative.

Status: **implemented through work-level data, model routing, stack routing, and group-aware
calibration.** The synthetic paired-audit control plane and preparation command are implemented,
while the freeze remains unapproved and confirmatory execution is hard-disabled; see
[`paired_audit_protocol.md`](paired_audit_protocol.md) and the repository
[`ROADMAP.md`](../ROADMAP.md). Implementation evidence is recorded by commits
`91a96293`, `6dc6d153`, `2c75c591`, `c6335586`, and `f1b8e165`.

This document includes the six contract refinements needed to make the estimand executable. The
legacy default remains `chunk_weighted_legacy`, so the baseline stays reproducible until the paired
audit and its separate headline decision are complete.

## 0. Training weights and public claim labels

`src/stylo/domain/work_weighting.py` (with
`src/stylo/eval/work_weighting.py` retained as an exact-object compatibility alias):

- `sw_i = W / (A·W_a·C_w)`, `sum = W`, strict length checks + `zip(strict=True)`,
  multi-author work rejected.
- **Weighting validation:** `resolve_training_weighting` validates the chosen value *including the
  fallback default* (an invalid default raises). Negative test added.
- **Claim-label mapping:** runtime label ≠ public claim string is made explicit with
  `to_claim_label()`: `chunk_weighted_legacy` → `chunk_weighted_training_legacy` (the
  legacy headline claim, unchanged), `work_balanced` → `work_balanced`. The committed legacy
  headline artifact is not touched.
- Tests: `sum == W`, equal author/work mass, **unequal W_a = {3,1}**, chunk-dup mass
  invariance, both-direction length mismatch, invalid default, `to_claim_label`, config
  default + `--set` override. 15 tests.

## 1. Canonical work/chunk identity — NOT content hash

Content-hash dedup is rejected: two legitimately identical fragments at different places in
one work would be deleted. Canonical chunk identity:

```
identity = (work_id, provenance_sha256, chunker_config_hash, span_ordinal, text_sha256)
```

- `provenance_sha256` — sha256 of the work's normalized source bytes.
- `chunker_config_hash` — hash of the frozen chunker config (chunk_words, sentencizer id,
  name-masking) so a re-segmentation is a different identity, not a silent collision.
- `span_ordinal` — the chunk's (start,end) token span / ordinal within the work.
- `text_sha256` — sha256 of the chunk's normalized bytes.

**Policy:** a repeat of one full identity (same span AND same text in the same work under
the same chunker) is a defect and **always fails closed (raises)** — there is no
logged-no-op variant. Identical `text_sha256` at a *different* `span_ordinal` is KEPT
(legitimate repetition). Canonicalization runs **before** weights, feature fitting and
evaluation; the `chunk_weighted_legacy` path receives the raw per-chunk arrays with no
dedup.

**Required persistence:** spans/provenance are not persisted in the live corpus today —
`chunking.make_sent_chunks` (`chunking.py:18`) and `pipeline/split.py:72` write chunk text
only. The audit-only corpus preparation must add a **chunk manifest** at
`<audit-corpus-root>/<author>/<book>/manifest.json` (strict JSON), recording per-chunk
`span_ordinal`, `text_sha256`, and the work's `provenance_sha256` + `chunker_config_hash`.
The work-balanced corpus loader reads it to build the canonical `WorkDocument` view over the
existing `texts/reps/groups` arrays (no new heavy format). The live corpus remains unchanged.

**Manifest validator (required, blocks `work_balanced`):**

- bijection between manifest entries and the chunk `.txt` files (no orphan either side);
- recompute and match every `text_sha256` from the on-disk bytes;
- identity `(work_id, provenance_sha256, chunker_config_hash, span_ordinal, text_sha256)`
  is unique within the work;
- `span_ordinal`s are ordered and non-overlapping;
- reject a missing / extra / stale manifest (provenance or chunker hash mismatch);
- `chunker_config_hash` is canonical over: `chunk_size`, `min_words`, `overlap`,
  sentencizer/model id, normalization + name-masking pipeline, and the chunker algorithm
  version.

Any discrepancy blocks the `work_balanced` path (fail closed); `chunk_weighted_legacy`
reads the raw arrays and does not require the manifest. This manifest is the concrete
`WorkDocument` the plan requires and is the gate that makes strict duplication-invariance
testable.

## 2. Full Pipeline groups contract

`groups` must reach every learned block; the Pipeline calls `fit_transform`, so BOTH entry
points take it:

```
Pipeline.fit
  → StyloVectorizer.fit_transform(X, y=None, groups=None)   # vectorizer.py:55
      → StyloVectorizer.fit(X, y=None, groups=None)          # vectorizer.py:35
          → FeatureBlock.fit(texts, reps, groups=None)       # base.py:26 (+ fit_transform)
```

`FeatureBlock.fit_transform` and `StyloVectorizer.fit_transform` gain `groups=None`
(additive). Non-fitted blocks (syntax, length_dist, embeddings) accept and ignore it.
Routing to the work_balanced stylo model is the verified `WorkBalancedStyloPipeline(Pipeline)`
subclass (`needs_groups=True`, routes `vectorizer__groups` + `classifier__sample_weight`;
preserves `named_steps`/`clone`/joblib/`classes_`). A single `fit_estimator(est,X,y,groups)`
dispatch (needs_groups) is used by `eval/lobo.py`, `eval/groupkfold.py` AND
`pipeline/train.py:45`, avoiding the unsupported direct `Pipeline.fit(groups=)` call.

## 3. Work-level sparse vectorizer — fit AND transform

One shared helper `features/work_vectorizer.py` (fit + transform fully specified):

Fit contract: chunk counts → work×feature sum → work-DF (prune `df_w < 2`) →
equal-work relative-TF ranking → deterministic `max_features` cap → work-level IDF.

**Relative-TF denominator = ALL analyzer events of the work** (every token the analyzer
emits), computed BEFORE `min_df`/cap — not the sum over surviving/selected features. This
applies to MFW selection (function_words), char/POS/punct ranking, AND Delta's z-input
relative frequencies (Delta divides by all analyzer tokens, not just selected MFW).

Transform (frozen train state, applied to CHUNK rows at test):

- char / POS / punct: `sublinear_tf` term frequency, multiply by the frozen work-level IDF,
  then row-wise L2 normalization — same shape as the current `TfidfVectorizer(sublinear_tf,
  use_idf, norm='l2')`, only the vocabulary/IDF are work-fitted.
- bow: count-only mode (no IDF, no sublinear), matching `build_bow_lr`.
- function_words: relative frequencies in both modes; denominator = all word tokens.

Determinism: sorted vocabulary, string tie-break on equal rank, float64 accumulation.

**Numerical edges (frozen):**

- `idf(f) = ln((W + 1) / (df_w(f) + 1)) + 1` (smoothed, W train works).
- A work with **zero analyzer events** still counts as one of the W votes (denominator
  stays W), contributes 0 to every feature's equal-work relative-TF mean, and its chunks
  transform to the **zero vector** — no division by zero anywhere.
- Equal-work relative-TF and Delta's z-input use `sum(counts_over_work) /
  sum(all_analyzer_tokens_over_work)`, i.e. counts and token totals are summed across the
  work's chunks BEFORE dividing — **not** the mean of per-chunk relative frequencies. A
  Delta test with sharply unequal chunk lengths must confirm this (the two differ whenever
  chunk lengths vary).

## 4. Five-model wiring spec

Same estimand, one `training_weighting` switch, explicit per-model changes:

- **stylo** — `WorkBalancedStyloPipeline` (§2); StyloVectorizer work-fitted (§3);
  `sample_weight` (sum W); `class_weight=None`. Legacy = plain Pipeline, `class_weight
  ='balanced'`, no weights.
- **char_cos** (`baselines.py:53`) — char vectorizer → work-level sparse (§3); centroid
  step unchanged (already work-mean). `needs_groups` already True.
- **delta** (`delta.py`) — MFW vocab selection + relative-freq z-input → work-level (§3);
  z-mean/std and centroids unchanged (already work-mean). `needs_groups` already True.
- **bow_lr** (`baselines.py:86`) — becomes a `WorkBalancedBowPipeline(Pipeline)` mirror:
  work-level count vocab + `classifier__sample_weight`, `class_weight=None`,
  `needs_groups=True`.
- **stylo_stack** (`stacked_clf.py`) — feature fitting per channel work-level. **No double
  weighting:** every weighted branch (inner-CV SVC, full SVC, meta-CV, final meta-LR) sets
  `class_weight=None` and `sample_weight=work_weights`; the legacy branch keeps
  `class_weight="balanced"` with no work weights. `ChannelFn` gains train `groups`;
  `choose_calibrator` gets `groups`; calibration is disabled when any class has <2 train
  works and the stack falls back to identity calibrators + equal ensemble (no learned
  meta-selection).

Constructors/factories: `make_factory` (`lobo.py:40`) and `make_full_pipeline`/
`build_bow_lr` gain the `training_weighting` argument; `groupkfold` and `train` use the
shared dispatch.

## 5. Group-aware calibration eligibility

The cloneable group-calibration wrapper builds explicit StratifiedGroupKFold (train,val) work
splits inside the outer-train fold.

**Effective fold count:** `n_splits_eff = min(configured_n_splits, min_works_per_class)`,
and every (train, validation) split must contain all classes. If that is not achievable —
i.e. **any class has < 2 train works** — calibration is disabled for the whole outer fit
(there is NO chunk-CV fallback), and calibrated and raw classes are never mixed in one
model. For Stacked this means identity calibrators + equal ensemble weights with no learned
meta-selection in that fold. Headline stylo in LOBO is uncalibrated, so this is for
uniformity/other uses.

## 6. Normalized paired-audit artifact

```
main_work_balanced_audit/summary.json  (strict JSON, claim_status: exploratory_internal)
  frozen: { git_commit, config_id_sha256, data_manifest_sha256, env_lock_sha256,
            seeds, clean_tree_assertion: true,
            tolerances: {                     # continuous quantities, NOT discrete metrics
              sample_weight, idf_feature_state, probability } }
  class_order: [author_id, ...]               # explicit alignment for every proba vector
  datasets:
    <dataset_id>:                             # ruaa_core, full_corpus_lobo
      fold_manifest_sha256; fold_manifest: [ordered held-out work_ids]
      models:
        <model_id>:                           # stylo, char_cos, delta, bow_lr, stylo_stack
          resolved_model_id; resolved_config_id
          variants:                           # the five-row ablation ladder, §7
            legacy | weights_only | feature_state_only | relative_fw_only | full:
              status: applied | not_applicable # e.g. relative_fw_only for a char-only model
              per_work: [ {work_id, author, pred, rank,
                           proba: [vector aligned to class_order]} ]
              aggregate: {acc, macro_f1, author_clustered_ci, mcnemar_p_vs_legacy}
          failures; exclusions
  artifact_sha256                             # canonical JSON with this field removed (or sidecar .sha256)
```

- **class_order** is explicit and every `proba` is a full vector aligned to it (not just the
  top class).
- **Tolerances** are set on the continuous intermediates — `sample_weight`,
  feature-state/IDF, probability vectors — not on discrete accuracy/macro-F1 (which are
  step functions and unstable near ties). Discrete metrics are reported but the pass/fail
  reproduction check is on the continuous quantities.
- **Self-hash** (`artifact_sha256`) is computed over the canonical JSON with the hash field
  itself removed (or stored as a sidecar `summary.json.sha256`), so the artifact can carry
  its own verifiable hash.
- A variant that does not apply to a model is marked `status: not_applicable` rather than
  omitted.

Full **class-aligned probability vectors** per work. **Separate fold manifest per dataset.**
Author-clustered bootstrap CI on per-work paired predictions. `SOCIOLIT-lite` is excluded
from the strict paired audit (precomputed global features + plain StratifiedKFold cannot
carry the work-balanced estimand); it stays a descriptive lexical layer.

## 7. Registered ablation ladder

1. **legacy**;
2. **weights_only** — work `sample_weight`, chunk features;
3. **feature_state_only** — work-level vocab/IDF/DF, uniform weights;
4. **relative_fw_only** — the relative-frequency FW transform from §3, else legacy;
5. **full = weights + feature_state + relative_fw** (2+3+4) — the target estimand.

## 8. Frozen design decisions

- **Dedup:** by canonical identity/span (§1), NOT content hash; fail-closed on identity
  repeat.
- **max_features:** keep current per-block caps for the confirmatory audit; any retuning is
  a separate pre-registered grouped-CV stage.
- **Single-work calibration:** disable calibration for the whole outer fit if any class has
  <2 train works; no mixing calibrated/raw; Stacked → identity calibrators + equal ensemble.
- **Helper consolidation:** now consolidate only grouping / validation / sparse-index
  primitives; do NOT merge Delta/Char/case centroids until after the paired audit (their
  L2/normalization semantics differ).

## 9. Remaining paired-audit dependencies

The work-level vectorizer (§3), Pipeline group contract (§2), five-model routing (§4), shared
`fit_estimator` dispatch, and group-aware calibration (§5) are implemented. They remain behind the
`training_weighting` switch with `chunk_weighted_legacy` as the production default.

The remaining work is dependency-ordered:

1. Implement, test, and independently review the audit-only corpus verifier and immutable
   audit-corpus builder. It must emit the chunk manifests and canonical `WorkDocument` identities
   from §1 without mutating the live corpus.
2. Use that reviewed preparation path to prove the legacy corpus anchor and semantic parity, bind
   the exact RuAA subset, then freeze and independently verify the LOBO and RuAA fold manifests.
3. Implement and test the paired-audit orchestration around the existing model mechanisms: the full
   applicability grid, one bound run plan, immutable per-fold checkpoints, verified publication,
   the normalized artifact from §6, and the inference and headline-gate calculations.
4. Pass the clean-tree preflight, execute both registered datasets under separate authorization,
   assemble every applicable cell, and independently audit the result and durable evidence.
5. Make the headline decision as a separate action only after the complete paired-audit artifact
   passes review. Until then, the legacy headline and its committed artifacts stay untouched.
