# Stylometry owner-decision memo

Status: **DRAFT_FOR_OWNER_REVIEW**

Authority: **NONE**. This memo approves nothing — no corpus migration, protocol amendment, freeze
candidate, evaluator, execution, headline, publication, API removal, or mutation of evidence.

The normative current-state authority is [`research/governance/status_ledger.json`](../governance/status_ledger.json).
Every owner disposition below is intentionally `UNSET`.

## 1. Decision order

corpus → protocol → freeze → evaluator → preflight → execution authorization → independent audit → publication decision

The other decisions are parallel claim-specific tracks and do not substitute for that order.

## 2. Open owner decisions

### ODM-001 — Collection versus constituent corpus identity

- **State:** the known Turgenev content component is ineligible for a new LOBO/RuAA accuracy or headline claim; historical bytes and manifests remain immutable evidence.
- **Decision:** choose the registered exclusion and reporting unit for collections holding near-complete constituent works: exclude the constituents, exclude the collection, or register the overlap as one fold-exclusion unit. Every route produces a new versioned corpus.
- **Evidence:** exact and asymmetric containment graph, affected-work inventory, fold-disjointness tests.
- **Disposition:** `UNSET`.

### ODM-002 — Frozen stack cells and calibration estimand

- **State:** manifest-wide feasibility preflight rejects the registered `stylo_stack` cells; confirmatory routing also rejects their withdrawn internal selection evidence.
- **Decision:** withdraw the stack cells by versioned amendment and register the resulting comparison family, fund a new nested/cross-fitted group-aware design, or keep the matrix blocked.
- **Evidence:** outer-fold feasibility, class/work coverage, calibration selection/fit separation.
- **Disposition:** `UNSET`.

### ODM-003 — Public blind benchmark v2 and escrow

- **State:** schema 1.0 blind scoring is historical and integration-only; scientific blind scoring is blocked.
- **Decision:** authorize a dual-manifest v2 benchmark with truth escrow under a named external custodian, or retain v1 as synthetic integration and history only.
- **Evidence:** redacted public manifest, custodian provenance manifest, independent isolation audit.
- **Disposition:** `UNSET`.

### ODM-004 — Historical-case open-set applicability

- **State:** target passports abstain because the open-set applicability gate is deliberately unavailable; the closed-set winner is diagnostic.
- **Decision:** retain permanent abstention, or preregister a new calibrated gate and passport version.
- **Evidence:** outsider controls, absolute-fit thresholds, independent validation.
- **Disposition:** `UNSET`.

### ODM-005 — Heterogeneity inference

- **State:** the shared-basis standardized contrast is descriptive; the former two-hands significance verdict is withdrawn.
- **Decision:** keep the API permanently descriptive, or preregister a work-level inferential design.
- **Evidence:** independent works and controls, one registered sampling unit, a valid null scheme.
- **Disposition:** `UNSET`.

### ODM-006 — Lazy-final-fit estimator deployment

- **State:** stack and equal-channel estimators are evaluation-only and cannot be serialized into bundles.
- **Decision:** retain evaluation-only status, or schedule a versioned fit-time redesign after a corrected exploratory LOBO.
- **Evidence:** fitted-state schema, batch/repeat parity, serialization trust boundary.
- **Disposition:** `UNSET`.

### ODM-007 — Compatibility API retirement

- **State:** the dormant `make_classifier` keeps its uncalibrated import-compatible route while learned-calibration requests fail closed; legacy builders and loose-artifact entrypoints are hard-disabled.
- **Decision:** identify which direct-module imports need a compatibility window and which may be removed later — only after consumer and deprecation review.
- **Evidence:** reference inventory, external-consumer assessment, replacement API, deprecation window.
- **Disposition:** `UNSET`.

### ODM-008 — Author-clustered macro-F1 interval

- **State:** the interval remains `withdrawn_pending_preregistered_recompute`; the point macro-F1 uses a fixed 43-label denominator and does not substitute for it.
- **Decision:** keep the interval withdrawn, or preregister a recomputation with an exact author/work resampling estimand.
- **Evidence:** frozen label universe, resampling unit, review before results are seen.
- **Disposition:** `UNSET`.

### ODM-009 — Published-history path disclosure

- **State:** the Git-free public archive excludes frozen passports and internal audit records whose bytes disclose an absolute developer workspace and private corpus layout; those bytes are not edited. Copies reachable from published Git history remain.
- **Decision:** retain the historical disclosure, publish versioned sanitized replacements while keeping history, or authorize a coordinated rewrite of the exact affected paths.
- **Evidence:** exact object/path inventory, proof that no corpus text or credential is present.
- **Disposition:** `UNSET`.

### ODM-010 — External anchor for frozen historical inputs

- **State:** the digest table rejects accidental one-sided drift of the frozen validation inputs and historical bytes are unchanged, but the expected hashes live in the same mutable repository.
- **Decision:** name an external authority to anchor the frozen-input digests — a protected signed tag, a separately controlled transparency record, or another reviewed registry — or explicitly retain the present accidental-drift gate without claiming independent immutability.
- **Evidence:** named key/ref custodian, exact digests, verification independent of the mutable checkout.
- **Disposition:** `UNSET`.

## 3. Non-options

- `stylo_equal_channels_v1` must not silently replace `stylo_stack` — it is a different estimator.
- The blind benchmark v1 must not be upgraded in place; v2 is a new versioned artifact.
- The open-set gate must not be enabled by flipping a boolean.
- Frozen evidence and published Git history must not be rewritten silently.
- The local digest table must not be described as independent external immutability.

## 4. Confirmatory gates

| Gate | Current state | Evidence/action required | Authority |
|---|---|---|---|
| CG-001 corrected corpus | blocked by ODM-001 | New content-safe corpus and exact fold manifests | owner + editorial review |
| CG-002 feasible protocol matrix | blocked by ODM-002 | Versioned amendment or reviewed nested design | protocol/statistical owner |
| CG-003 manifest freeze | `candidate_unapproved`; pin is `None` | Independent review of the candidate, then a separate reviewed digest-pin change | freeze owner + independent reviewer |
| CG-004 production evaluator | `blocked_unregistered`; canonical registry is empty | Independently reviewed callable, config, mechanism passport and source digests | evaluator owner + independent reviewer |
| CG-005 execution preflight | not satisfied for confirmatory use | Display-to-slug mapping, pinned golden replay, clean tree, durable receipts | release/science owner |
| CG-006 execution authorization | not granted | Separate authorization for the exact freeze/evaluator/run identity | execution owner |
| CG-007 independent result audit | no confirmatory result exists | Audit the candidate and per-work vectors without reusing producer trust | independent reviewer |
| CG-008 headline/publication | `not_authorized` | Separate decision only after CG-007; publication remains a distinct action | headline/publication owner |

Runner authorization strings and hard-stop constants are mechanisms, not approval evidence.

## 5. Rules for a future disposition

A real owner decision is a separate versioned record naming owner, scope, evidence, option and date.
Implementation and independent review are separate steps performed after it.

This memo must not be used to:

- alter frozen corpus, protocol, matrix, truth, manifests, or historical evidence;
- populate approval pins such as `APPROVED_FREEZE_ROOT_SHA256`;
- populate the canonical evaluator registry;
- authorize execution, a headline, publication, a tag, a push, or a release.

Memo disposition: `UNSET`.
