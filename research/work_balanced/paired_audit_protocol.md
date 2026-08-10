# Paired-audit protocol: legacy versus work-balanced training (v3.2)

Normative current status is
[`../governance/status_ledger.json`](../governance/status_ledger.json). This document is the
normative scientific design. The ledger controls authorization; neither this protocol nor an
implementation artifact grants permission to prepare, preflight, execute, publish, or change a
headline.

Version v3.2 supersedes v3.1 for every new scientific run. This document fixes the scientific
design; current implementation, review, freeze, authorization, execution, and publication status is
recorded only in the governance ledger.

## 0. Supersession and preserved historical evidence

The v3.1 corpus snapshot and its LOBO/RuAA fold manifests are registered
[`ineligible_for_new_scientific_runs`](../evidence/ineligible_corpus_registrations_v1.json): distinct
registered work IDs include overlapping content from `turgenev/записки_охотника` and its constituent
works. Therefore the v3.1 snapshot, all v3.1 freeze candidates, its exact A0 predictions, the
`221/251` LOBO parity result, the RuAA-137 parity result, golden fixtures, and corpus bytes are
immutable **historical evidence only**. They are not an acceptance target, source of a new freeze,
or authorization for a new scientific run.

The historical RuAA source dispositions independently require excluding
`serafimovich/у_нас_и_у_них` (authorship mismatch) and
`sevsky/дон_на_костылях` (source quality); `turgenev/записки_охотника` is a collection umbrella.
No historical artifact, R1 v5 milestone, sealed evidence, README/site headline, or public claim is
altered by v3.2.

The required responsibility sequence is:

**design correction → audited v3.2 corpus/fold/evaluator implementation → corrected-corpus
preparation and equality/independence proof → independent manifest review and digest pin → clean-tree
preflight and separate execution authorization → one complete run plus exact resume → separate clean
session result audit → separately authorized headline decision.**

## 1. Corrected universe and corpus contract

### 1.1 Fixed exclusions and mechanically checked counts

The v3.2 corpus contract excludes exactly these work IDs before any fold is derived:

| work ID | disposition |
|---|---|
| `turgenev/записки_охотника` | collection umbrella; its constituent works remain separately represented |
| `serafimovich/у_нас_и_у_них` | authorship mismatch |
| `sevsky/дон_на_костылях` | source-quality exclusion |

The contract is mechanically accounted from the historical 255-work parent inventory, the
251-work LOBO panel, the 137-work RuAA manifest, and the three source-disposition IDs. All three
excluded IDs occur once in each tested panel. The resulting v3.2 targets are:

| panel | historical | exclusions | v3.2 contract |
|---|---:|---:|---:|
| LOBO train universe | 47 authors / 255 works | 3 works | **47 authors / 252 works** |
| LOBO tested | 43 authors / 251 works | 3 works | **43 authors / 248 works** |
| RuAA nested sensitivity | 22 authors / 137 works | 3 works | **22 authors / 134 works** |

LOBO retains the four train-only singleton authors (`goncharov`, `grigorovich`, `reshetnikov`,
`voloshin`); no author is removed by these three exclusions. Probability-vector and metric label
orders must be rebuilt from the corrected corpus, with expected widths 47/43 for LOBO and 22/22 for
RuAA. A count, identity, author, or exclusion mismatch is a hard stop; it must not be repaired by
editing a manifest or by reducing a test family.

### 1.2 New immutable root, not a mutation or a reused freeze

The future builder shall create a new, complete immutable root
`data/audit_corpus/<v3_2_digest>/`, never mutate `data/frags_train`, and bind an explicit corrected
source-inventory digest, the literal ordered exclusion set above, per-work manifests, and a new
content-independence proof. The old v3.1 parent digest, semantic-parity digest, and freeze-root
digest are historical identifiers only and cannot be copied into a v3.2 RunPlan.

The eventual LOBO and RuAA manifests must be new versioned artifacts (for example
`lobo_fold_manifest_v3_2.json` and `ruaa_fold_manifest_v3_2.json`), self-hashed, rebuilt from disk,
and independently reviewed before a non-empty pin can be committed. RuAA remains a whole-work,
exact-selection, nested secondary sensitivity panel; it is not a replication, blind benchmark, or
headline selector.

## 2. Confirmatory estimands and applicability

### 2.1 New corrected A0; no historical-parity gate

Corrected A0 is the frozen legacy algorithm/spec applied to the corrected v3.2 corpus and new folds.
Its precise source/config/environment digests, reference predictions, and any fixtures must be
created only after the v3.2 implementation has been audited and the corrected corpus has been
frozen. A future A0 reference may be SHA-pinned before parsing, but it must be v3.2-native.

The historical `lobo_books.txt`, historical RuAA submission, `221/251`, and RuAA-137 predictions are
not compared for equality and are not a preflight, release, or acceptance gate. Historical bytes
remain immutable evidence; their preservation does not imply v3.2 parity.

### 2.2 Frozen axes

The estimand axes retain their definitions: W is equal-work training mass where applicable; F is
work-level vocabulary/DF selection; R is the function-word relative transform. The function-word
block still exposes F0R0, F1R0, F0R1, and F1R1. Delta and char-cos retain their already-in-legacy W
semantics. The immutable `delta:N` identifier remains the **legacy selected-mass Delta**
compatibility estimator, not canonical Burrows's Delta: its compatibility denominator is the sum of
selected-MFW counts. Non-applied/equivalent cells remain metadata-only with no copied metrics.

### 2.3 Confirmatory matrix — 16 applied cells / 11 comparisons

`stylo_stack` is withdrawn from the v3.2 confirmatory applicability matrix. It contributes no cells,
fixtures, calibration passport, compute estimate, or Holm comparison. This release does **not**
replace it with a nested or cross-fitted stack.

The literal v3.2 matrix has **16 applied cells**:

| model | applied cells | Δaccuracy versus A0 comparisons |
|---|---|---:|
| `stylo` | A0, A1, A2, A3, A4 | A1, A2, A3, A4 (4) |
| `bow_lr` | A0, A1, A2, A4 | A1, A2, A4 (3) |
| `delta_cos:500` | A0, A2, A3, A4 | A2, A3, A4 (3) |
| `char_cos` | A0, A4 (A2 equivalent to A4) | A4 (1) |
| `majority` | A0 | none |

The Holm family is the literal 11-member set: **stylo {A1,A2,A3,A4}; bow_lr {A1,A2,A4};
delta_cos:500 {A2,A3,A4}; char_cos {A4}**. `family_alpha = 0.05`; Holm operates on unrounded
cluster p-values; a missing or failed member invalidates the entire 11-member family.

## 3. Metrics, inference, and headline boundary

Every applied cell is evaluated on the identical corrected fold set for its dataset. Accuracy,
macro-F1 with the manifest's frozen metric-label order, top-2 accuracy, and per-author recall are
recorded. Author-clustered macro-F1 intervals remain withdrawn.

For each non-A0 comparison, use the pre-registered two-sided null-centered author-cluster bootstrap
for Δaccuracy: `B = 10000`, seed `42`, +1 correction, and the existing documented degenerate-case
order. McNemar is diagnostic-only. The single possible headline decision remains `stylo` LOBO
A4−A0 with a two-sided 95% percentile author-clustered CI, `iters = 10000`, seed `42`, and
noninferiority margin `δ = 0.02`; relabel only if the unrounded lower bound is strictly greater than
`−δ`, retain legacy only if the upper bound is strictly less than `−δ`, otherwise report
inconclusive.

This is only a decision rule for a future correctly authorized run. The historical `0.8805` and all
baseline/headline artifacts are immutable, and no headline or publication is authorized now.

## 4. Future execution contract

The future v3.2 RunPlan shall bind the two new dataset digests, new fold-manifest digests and class
orders, corrected source-inventory/exclusion/independence-proof digests, exact v3.2
applicability-matrix digest, A0 reference SHAs if created, evaluator identity, configuration and
source hashes, environment/runtime fingerprints, numerical tolerances, seeds, and this protocol
version. The environment lock is the **SHA-256 of the tracked `requirements.lock` only**; the ignored local `uv.lock` is explicitly outside the run identity. A v3.1 digest is never a substitute for any of these bindings.

Checkpoints shall be per fold, immutable, self-hashed, and re-attested before and after every work;
valid checkpoints are skipped on resume, missing checkpoints are computed during a run, and missing,
corrupt, conflicting, or extra checkpoints are fatal at completion as specified by the future
implementation. One authorized full run and its exact resume are both required. A separate clean
session then reopens the durable candidate and independently recomputes the metrics, intervals,
p-values, Holm family, and headline inputs. No stage may auto-authorize a later stage.

## 5. Present hard gates and implementation boundary

The v3.1 implementation is retained as historical, synthetically tested control-plane evidence; it
is not an implementation of v3.2. The v3.2 freeze pin is empty and unapproved, the production
evaluator is unregistered, and confirmatory execution is hard-disabled. No corpus build, fit,
prediction, preflight, authorization, publication, push, or deployment occurs in this design release.

The next implementation change is limited to: (1) corrected corpus and fold builders that enforce
the exact exclusions and 252/248/134 contract; (2) the 16-cell/11-comparison applicability registry
without `stylo_stack`; (3) a v3.2 evaluator and preflight that reject historical A0 parity bindings;
and (4) new receipts and independent review gates. It must not alter R1 v5, sealed evidence,
historical corpus bytes, or public headline artifacts.
