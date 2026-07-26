# Paired audit protocol: legacy versus work-balanced training (v3.1)

Normative current status is
[`../governance/status_ledger.json`](../governance/status_ledger.json). This document is the
normative scientific protocol, but its chronology, review-round labels, future-tense prose, commit
labels, and line references do not override the current-state ledger.

Status: **approved design, canonical v3.1**. The earlier `a8285c81` record was explicitly a DRAFT;
this canonical protocol is first committed with the audited §11 preparation implementation rather
than being attributed retroactively to that commit. Estimator-axis routing, the narrow stylo LOBO
validation, and the synthetic confirmatory control plane are implemented; execution remains gated.
The exploratory routing and narrow stylo LOBO validation do not substitute for this protocol.
Version 3.1 consolidates six contract corrections over v3. The headline (`0.8805`),
README, site, and headline artifacts remain untouched; RuAA is a **nested secondary sensitivity**
panel, not replication or a best-result picker; SOCIOLIT-lite is excluded from the confirmatory
audit.

Required responsibility sequence: **protocol design → independent design review and commit →
audit-corpus builder, runner, and synthetic-test implementation → independent code audit and commit
→ audit-corpus preparation and equality proof → independent review and commit of frozen manifests →
clean-tree preflight and separate execution authorization → confirmatory execution → independent
result audit → separately preregistered headline decision.** Audit-corpus preparation never
precedes the audited, committed audit-corpus builder.

---

## 1. Universes, corpus chain and manifests

### 1.1 LOBO universe (unchanged from v2)
Train **47 authors / 255 works**; tested **43 / 251** (single-work authors `goncharov, grigorovich,
reshetnikov, voloshin` train-only); probability vectors **47-wide**; A0 stylo = **221/251**.

### 1.2 Corpus chain anchored to the committed frozen ancestor
Old↔WB parity alone could hide coordinated loader drift, so the chain is anchored to a **committed**
digest before any staging:
1. **Legacy anchor:** `load_dataset(frags_train)` must reproduce the committed
   `parent_dataset_digest = b4886a7cd723c04515b43f042467bc372af0aeaf28c47f517f0b2aa9d46b8c92`
   (`docs/screening_panel_v1.json`) — proves the corpus has not drifted from the signed ancestor.
2. **Semantic parity:** a loader-agnostic **semantic-row digest** over `(texts, y, groups, authors)`
   only (no loader_kind / no per-row identity, so it is comparable across loaders) computed over the
   **legacy** load and the **WB** load must be **equal**.
Both must hold. (1) pins legacy to the frozen ancestor; (2) pins WB to legacy; transitively WB is
anchored to `b4886a7c…`. The WB provenance `rows_digest` still differs (it binds
`loader_kind=work_balanced_manifest` + per-work manifest identities) and is *not* expected to equal
`b4886a7c…`.

### 1.3 Audit-corpus preparation — a whole immutable audit-corpus root
(After the prep-tool is implemented, tested on synthetic data, audited and committed.) 255 per-work
`manifest.json` **cannot be laid down atomically** inside the live `data/frags_train`, so preparation
instead builds a **complete, immutable audit-corpus root** — a separate directory
`data/audit_corpus/<digest>/` holding the 255 works + their manifests — published **whole and
atomically** (immutable-dir + pointer) only after the §1.2 legacy anchor + semantic parity pass
(per-chunk exact compare of filename, text bytes, group, label is part of the equality proof). The
audit **RunContract binds this immutable root** (not `frags_train`);
`load_work_balanced_dataset` reads it. The live `data/frags_train` is never mutated, and
`load_work_balanced_dataset` cannot run until this root is published.

### 1.4 Audit-only dataset↔estimand contract (unchanged from v2)
A0 (legacy estimand) runs on the WB-manifest dataset via a separate audit-only verifier
(`dataset_contract=work_balanced_manifest` ⊥ estimator axes); the production both-sided provenance
guard is not weakened.

### 1.5 RuAA nested panel — three-digest binding + whole-work selection
`derive_work_subset(parent_wb, work_ids)` binds all three: the **full WB parent digest**, the **exact
137-work selection-manifest digest**, and the **derived child digest**. It requires **whole works**
(every chunk of each selected work) and the **exact committed 137-work set** (not merely an ordered
subsequence — a missing or extra work is a hard fail). RuAA is a **nested secondary sensitivity** panel
of the same corpus: it may confirm or weaken a LOBO finding but never selects the headline result and
is never a blind leaderboard.

### 1.6 Manifest freeze and independent review before clean-tree preflight and authorization
`lobo_fold_manifest_v1.json`, `ruaa_fold_manifest_v1.json`: work/author IDs, per-work fold,
`probability_class_order` (47 / 22), `metric_label_order` (43 / 22), parent-dataset digest,
selection digest (RuAA), algorithm/seed/config hash, self-hash. The confirmatory runner **rebuilds the
expected manifest from disk and requires exact equality** with the committed one (never self-signs).

---

## 2. Ablation axes — exact per-block/per-model decomposition

### 2.1 What is already ON in legacy
Legacy Delta (`delta.py:88 _group_means`) and char_cos (`baselines.py:93 _sparse_group_means`) already
use **equal-work centroids** → their **W axis is already-in-legacy**. LR-family (stylo, bow_lr, stack)
legacy is chunk-weighted → W is a real axis.

### 2.2 Function-word block/channel is a full (F,R) grid
`function_words.py` legacy = `CountVectorizer(max_features).fit` (**pooled vocab**) + **raw counts**;
WB = `WorkLevelVectorizer(mode=relative, min_df_works=2)` which changes **both** the vocabulary
(work-ranked, work-DF prune → **F**) **and** the transform (relative all-event frequency → **R**). So
FW (and the stack's FW channel) has its own four corners — F is *not* "all blocks except FW":

| FW corner | vocabulary | transform |
|---|---|---|
| **F0R0** (A0) | pooled `CountVectorizer(max_features).fit` | raw counts |
| **F1R0** (A2) | `WorkLevelVectorizer(mode=count, min_df_works=2)` work-ranked/work-DF | raw counts |
| **F0R1** (A3) | pooled vocab | per-chunk relative (count / all-analyzer-events) |
| **F1R1** (A4) | work-ranked/work-DF vocab | relative all-event |

So for **stylo/stack**: **F** = work-level vocabulary/DF for **every** block/channel **including** the
FW vocabulary; **R** = the FW block/channel relative transform; the FW block must expose the four
corners (COUNT vs RELATIVE × pooled vs work vocab). **W** = LR/SVC/meta `sample_weight` (+ stack
group-aware calibration, §2.5).

### 2.3 Frozen legacy selected-mass Delta as an exact (F,R) grid
The immutable A0 identifier `delta:N` names a **legacy selected-mass Delta
compatibility estimator**, not canonical Burrows's Delta. Its `_rel_freq`
divides by **Σ selected-MFW counts in the chunk**, *not* by all tokens. With W
already-legacy, the four compatibility corners are:

| cell | vocabulary selection | train frequency | predict frequency | work aggregation |
|---|---|---|---|---|
| **A0=F0R0** | pooled `CountVectorizer(max_features)` | per-chunk `selected_count / Σselected_counts` | same, per chunk | `_group_means` (equal-work) → z |
| **A2=F1R0** | `WorkLevelVectorizer` work-rank + work-DF prune | per-chunk `selected_count / Σselected_counts` | same, per chunk | equal-work |
| **A3=F0R1** | pooled `CountVectorizer(max_features)` | per-work `Σselected_counts / Σall_analyzer_events` | per-chunk `count / all-analyzer-events` → soft vote | equal-work |
| **A4=F1R1** | `WorkLevelVectorizer` work-rank + work-DF prune | per-work `Σselected_counts / Σall_analyzer_events` | per-chunk `count / all-analyzer-events` → soft vote | equal-work |

### 2.4 Honest applicability matrix — 21 cells / 15 comparisons (unchanged from v2)
stylo 5, stylo_stack 5, bow_lr 4 (R n/a), delta_cos:500 4 (A1 already-legacy), char_cos 2 (A2≡A4,
A1/A3 n/a), majority 1. Δaccuracy-vs-A0 comparisons = 4+4+3+3+1+0 = **15**.

### 2.5 Stack W is a bundled protocol — and a pre-registered LOBO consequence
Stack **W** = sample-weights **and** group-aware calibration (calibration group-aware iff W on).
**Pre-registered consequence of the 47-class universe:** every LOBO train fold contains the four
**single-work** authors (1 work each), so `min_works_per_class = 1` in **all W-on stack folds** ⇒ the
calibration layer is disabled (identity), `mode_ = "equal"`, `meta_ = null`. This is expected and
recorded, not an error; **singleton authors are never removed** (they are part of the 47-class train
universe). The stack's W effect on this corpus is therefore *sample-weights + equal-ensemble
fallback*, documented as such (not a pure marginal).

### 2.6 Non-tautological golden fixtures
Parity is proven against **external fixtures captured from the committed `f1b8e165`** (checked out in a
separate `git worktree`), **before** any estimator refactor — not by comparing two post-refactor paths.
For A0 and A4, per model, the fixtures freeze: fitted **vocabulary/IDF**, **sample weights**, **Delta
z-state (mean/std/centroids)**, **stack calibration passport**, and **class-aligned probability
vectors** on a fixed panel. The refactored A0/A4 must reproduce these committed fixtures (numeric
tolerances, §4.1).

---

## 3. Metrics, class orders, cluster-valid inference, gate

### 3.1 Per-dataset class orders (unchanged from v2)
Inside each dataset manifest: `probability_class_order` (LOBO **47**, RuAA **22**) for the emitted
vectors, and a frozen `metric_label_order` (LOBO **43** tested, RuAA **22**) that macro-F1 averages
over. macro-F1 never uses `unique(y_true∪y_pred)`; its author-clustered CI stays **withdrawn**.

### 3.2 Exact A0 verification against pinned reference files
The A0 reference files are verified by **pinned SHA256 before any parse** (not merely folded into
`run_id`): `docs/lobo_books.txt` == `26db64475e77657eaec6db895c55bad8bcd513344584ef5a64e9a580cf9f648d`;
`data/ruaa_bench_v1/reference_submission_stylo.csv` ==
`05e334f65d81aaff7ef240e7f4b5c1c9e422e050b906906482bfaa063da90db0`; RuAA is **additionally** checked
against the frozen `data/ruaa_bench_v1/SHA256SUMS`. Only then: stylo LOBO **221/251**; exact per-work
`pred/correct/rank` == `lobo_books.txt`; RuAA A0 == the reference submission; semantic-row parity
(§1.2). Any mismatch aborts the confirmatory preflight **before** parsing.

### 3.3 Cluster-valid raw p — a real, pre-registered test
`paired_bootstrap_diff_clustered` returns a **CI, not a p-value** (`significance.py:84`); v3 therefore
pre-registers a proper cluster p-value (new implementation function, e.g.
`paired_cluster_pvalue`), **not** a plain bootstrap p:
- **Test:** two-sided **null-centered cluster (author) bootstrap** p for Δaccuracy `spec − A0` on the
  identical fold set. Resample **authors** with replacement `B = 10000` times (`seed = 42`); for each
  draw compute `Δ*`; center the bootstrap distribution at its mean (impose H0: E[Δ]=0);
  `p = (1 + #{ |Δ*_centered| ≥ |Δ_observed| }) / (1 + B)` (**+1 correction**, so p is never 0).
- **Degenerate cases (checked in this exact order):** (1) `n_unique_authors < 2` → `p = 1` (a
  single author cluster cannot be resampled); (2) all paired book-correctness differences are 0 →
  `p = 1`; (3) otherwise the general bootstrap formula above, **including a non-zero constant effect**
  (not special-cased). Without rule (1) a single author with a non-zero Δ would falsely yield
  `p = 1/10001`. An empty comparison invalidates its family (§3.4), not p.
- **McNemar** (book-level) is stored **diagnostic-only**; it never enters a claim (anti-conservative;
  Holm cannot repair an invalid raw p).

### 3.4 Literal frozen Holm family
The confirmatory family per dataset is exactly these **15** Δaccuracy-vs-A0 cluster p-values, listed
literally: **stylo {A1,A2,A3,A4}; stylo_stack {A1,A2,A3,A4}; bow_lr {A1,A2,A4}; delta_cos:500
{A2,A3,A4}; char_cos {A4}** (char A4 ≡ its A2). **`family_alpha = 0.05`; Holm–Bonferroni runs on the
UNROUNDED cluster p-values; `significant := holm_p < 0.05`.** A **missing/failed cell
invalidates the whole family** (m is not reduced). RuAA's family is evaluated only as secondary
sensitivity (§1.5), never to pick a result.

### 3.5 Separately preregistered formal noninferiority headline decision
Single pre-registered cell: **stylo LOBO Δaccuracy A4 − A0**, author-clustered bootstrap.
- **Margin δ = 0.02**, justified **normatively**: the work-balanced estimand (one work, one
  vote) is the scientifically correct target; a **≤2 pp accuracy change is a pre-committed acceptable
  cost** for removing the length/chunk bias of the legacy chunk-weighted training. δ is **not** derived
  from the width of any prior CI.
- **Frozen settings:** two-sided 95% CI by the **percentile method**, cluster bootstrap resampling
  **authors**, `iters = 10000`, `seed = 42`, quantiles `[2.5, 97.5]`. The **absolute** author-clustered
  accuracy CI of A4 uses the **identical** resampling settings and percentile method.
- **Symmetric decision on UNROUNDED bounds:** **relabel** iff CI **lower > −δ**; **keep_legacy**
  iff CI **upper < −δ**; **inconclusive** if the CI straddles or equals −δ.
- **A1–A3 are descriptive**, out of the headline gate (mechanistic marginals, not the estimand claim).
- The artifact stores the **absolute** author-clustered accuracy CI of A4 (and every cell), not only
  the difference CI.
- **Relabel** publishes *new versioned* headline artifacts; `0.8805` and the frozen baseline snapshot
  are immutable regardless of outcome; macro-F1 CI stays withdrawn.

---

## 4. Artifact schema, run/resume, atomicity, committed location

### 4.1 Schema expresses honest applicability + evidence
Per `(dataset, model, cell)`:
```
{ "status": "applied" | "not_applicable" | "equivalent_to",
  "equivalent_to": "A4" | null,                       # e.g. char A2 -> "A4"
  "requested_axes": { "W":bool, "F":bool, "R":bool },
  "effective_axes": { "W":"applied|not_applicable|already_in_legacy", "F":..., "R":... },
  "per_work": [ {work_id,true_label,pred_label,rank,"proba":[|prob_order|]} ... ],   # applied only
  "point": { accuracy, macro_f1(frozen metric_label_order), top2, per_author_recall },
  "abs_accuracy_authorclustered_ci": [lo,hi],
  "vs_A0": { dacc, dacc_authorclustered_ci, cluster_p, holm_p, mcnemar_p_diagnostic, significant },
  "evidence": {                                        # executable, fold-local
     "per_author_mass": {expected, actual}, "per_work_mass": {expected, actual},   # or ordered_weight_digest
     "ordered_weight_digest": <sha256>,
     "vocab_digest": <sha256>, "idf_digest": <sha256>,
     "r_denominator_trace_digest": <sha256>,           # executable fold-local denominator trace, not just a kind string
     "delta_mean_std_centroid_digest": <sha256>,       # Delta z-state
     "calibration_passport": {...},                    # full stack passport (stack: calibration_disabled/mode/meta)
     "proba_digest": <sha256> },
  "claim_status": "exploratory_internal" }
```
Top-level: per-dataset `probability_class_order`/`metric_label_order`, `continuous_tolerances`
(explicit `atol`/`rtol`/`dtype` per continuous quantity — sample_weight, IDF, proba, Delta z-state)
and `attestation` (§4.2). `not_applicable`/`equivalent_to` cells carry **no metrics** and are
never a silent A0 copy. The three axes are stored explicitly — **never** collapsed to a corner enum
`legacy`/`work_balanced`.

### 4.2 Canonical RunPlan → run_id
One canonical `RunPlan` whose sha256 **is** the `run_id`, binding: both dataset digests, both
fold-manifest digests, both `(probability_class_order, metric_label_order)` pairs, the exact
**applicability-matrix digest**, `config_id`, both `RunContract`/selection digests, the **A0 reference
SHAs** (`lobo_books.txt`, RuAA reference submission), `run_kind`, all numeric **tolerances**, all seeds
and stat settings (`iters, quantiles, δ, α, B`), `audit_version`, `git_commit`,
`execution_source_sha256`, `env_lock_sha256` (SHA-256 of the tracked `requirements.lock` only;
the ignored local `uv.lock` is explicitly outside the run identity), the
BLAS/thread fingerprint, the **installed runtime fingerprint** (Python / NumPy / SciPy / scikit-learn /
spaCy / BLAS versions), the **corpus-chain digests** (§1.2 legacy anchor + semantic-parity digest), and
the **inventory SHA of the external golden fixtures** (§2.6). Different code/config/env/runtime ⇒
different `run_id`; no mixing is representable.

### 4.3 Per-fold immutable checkpoints
A stack cell is 251 folds, too costly to redo — so checkpoints are **per-fold**. Each checkpoint is
published **atomically without overwrite** and binds `run_id`, the RunPlan hash and the
manifest/class-order digests, with a **self-hash**, exact `(dataset, model, cell, fold)` identity, and
**fold-local evidence** (§4.4). Resume semantics:
- a **valid** existing checkpoint (self-hash + identity + bound digests all match) is **skipped**;
- a **missing** fold *during a run* is **pending** — it is computed (missing is **not** fatal mid-run);
- **corrupt / conflicting / extra** checkpoints are **always fatal**.
`COMPLETE` is declared only when **every** applicable cell/fold is present — so a **missing fold is
fatal only at COMPLETE assembly**, never mid-run. Before and after each fold/cell the runner
re-verifies code/config/env/data/manifest digests.

### 4.4 Crash-atomic publication + committed, unignored location
The run's transient output (per-fold checkpoints, full per-work vectors) lives under the
**gitignored** `docs/exploratory/work_balanced/audit/runs/<run_id>/` (path-aware guard, no headline
path). Publication uses the **immutable version-directory + single atomic `current.json`/`COMPLETE`
pointer** pattern (`pipeline/bundle.publish_bundle`). Because `docs/exploratory/…` is gitignored
(`.gitignore` `docs/*`), the **final verified summary artifact** (metrics, CIs, cluster/Holm p,
gate inputs, evidence digests, self-hash) is promoted to the unignored, versioned committed
`docs/work_balanced_paired_audit_v1.json` (matched by `!docs/*.json`). The **full per-work probability
vectors are also durably committed** — resolving the v3 §4.1↔§4.4 contradiction — as a
**content-addressed archive** `docs/work_balanced_paired_audit_v1/` (each per-cell per-work file named
by its content hash) with a committed `SHA256SUMS` inventory that the summary references by hash; a
`.gitignore` whitelist for that directory is added during audit-pipeline publication. The per-work
data is **not** left to `run_id` reproduction alone.

---

## 5. Implemented estimator mechanisms and remaining control plane

Already implemented and covered by focused tests:

1. Frozen `AblationConfig(weights, feature_fit, relative_fw)` with the production default unchanged.
2. The **FW (F,R) four-corner grid** (§2.2), corrected Delta grid (§2.3), and W routing through
   estimator loss plus stack calibration.
3. `make_factory(..., ablation)` routing with typed applicability signals.
4. External A0/A4 golden fixtures captured from commit `f1b8e165`, plus fail-closed live replay of
   vocabulary/IDF, weights, Delta state, stack calibration, and probability contracts.

The confirmatory workflow is split into completed synthetic control-plane work and remaining gates:

1. Implemented and synthetically tested: audit-only dataset contract, immutable corpus builder,
   whole-work RuAA derivation, fold-manifest construction, RunPlan/checkpoints, clustered inference,
   Holm, independent result audit, publisher, and headline gate.
2. Current gate: real corpus preparation and an **unapproved** exact manifest freeze candidate,
   followed by independent equality/manifest review.
3. Still required before execution: a reviewed approved-freeze root pinned by digest, the canonical
   production fold evaluator/evidence adapter, mandatory live external-golden replay, immutable stage
   receipts, clean preflight, and separate execution authorization.

## 6. Compute cost
21 LOBO + 21 RuAA ≈ 42 cells, dominated by stylo/stack LOBO rows; envelope ≈ tens of CPU-hours. The
synthetic smoke test times one representative `stylo_stack` fold (wall/CPU/peak RSS) to convert this
into a measured per-cell number before confirmatory execution is authorised. BLAS pinned; `n_jobs`
per the hardware note.

## 7. Protocol-design scope boundary (hard constraints)

The first sentence of this section was a historical implementation gate and is now satisfied for
the synthetic builder/runner and preparation-only candidate flow. The candidate remains unapproved,
the production evaluator remains unregistered, and confirmatory execution remains hard-disabled.
No benchmark or change to
`0.8805`/README/site/headline is allowed; production provenance and headline-write guards stay
unchanged. RuAA remains nested secondary sensitivity (not blind, not a result picker), and
SOCIOLIT-lite remains excluded.
