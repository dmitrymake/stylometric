# Stylometry owner-decision memo

Status: **DRAFT_FOR_OWNER_REVIEW**

Prepared: 2026-07-26 (Europe/Moscow)

Inspected parent baseline:
`68eef2517fd2e3440b31c5a97383cdcf6c97a038`

Authority: **NONE**. This memo does not approve a corpus migration, protocol
amendment, freeze candidate, evaluator, confirmatory execution, headline,
publication, API removal, or mutation of evidence.

The normative current-state authority remains
[`research/governance/status_ledger.json`](../governance/status_ledger.json).
Every owner disposition in this memo is intentionally `UNSET`.

## 1. Decision boundary

The current freeze candidate must not be approved as-is. Two scientific
decisions precede any freeze review:

1. resolve the ineligible collection-versus-constituent corpus identity;
2. amend, withdraw, or redesign the infeasible registered stack cells.

Only a new versioned corpus/protocol result may then enter independent freeze
review. The confirmatory chain remains:

```text
corpus identity decision
  -> corrected versioned corpus and fold manifests
  -> stack matrix/calibration decision
  -> versioned feasible confirmatory protocol
  -> independent freeze-candidate review and digest pin
  -> canonical evaluator registration
  -> clean preflight, golden replay, and durable stage receipts
  -> separately authorized confirmatory execution
  -> independent result audit
  -> separately authorized headline/publication decision
```

The benchmark, historical-case, heterogeneity, deployment, and compatibility
API decisions below are claim-specific parallel tracks. None can substitute for
the confirmatory chain.

## 2. Open owner decisions

### ODM-001 — Collection versus constituent corpus identity

- **Linked findings:** AUD-001 and the migration-gated parts of AUD-003,
  AUD-020, and AUD-021.
- **Current fail-closed state:** the known Turgenev content component is
  ineligible for a new LOBO/RuAA accuracy or headline claim. Historical corpus
  bytes and manifests remain immutable evidence.
- **Question for the owner:** what is the registered exclusion and reporting
  unit when a collection contains near-complete constituent works?
- **Admissible dispositions:** register the collection and exclude its
  constituents; register constituents and exclude the containing collection;
  or register the entire overlap component as one fold-exclusion unit while
  retaining separately described editorial records.
- **Evidence required before disposition:** the exact and asymmetric
  containment graph, bibliographic/editorial rationale, full affected-work
  inventory, proposed versioned identity rules, fold-disjointness tests, and a
  complete rerun plan.
- **Required implementation after a decision:** a new corpus/fold version and
  new results. Never rewrite the existing snapshot or reinterpret `0.8805`.
- **Owner disposition:** `UNSET`.

### ODM-002 — Frozen stack cells and calibration estimand

- **Linked findings:** AUD-006 and AUD-033.
- **Current fail-closed state:** manifest-wide feasibility preflight rejects the
  registered `stylo_stack` cells; confirmatory routing also rejects their
  withdrawn internal selection evidence.
- **Question for the owner:** should the next protocol withdraw the stack cells,
  or fund a new nested/cross-fitted group-aware stack design?
- **Admissible dispositions:** publish a versioned amendment that withdraws the
  stack cells and registers the resulting comparison/multiplicity family;
  register a new nested/cross-fitted group-aware design after feasibility and
  bias review; or keep the current matrix blocked.
- **Non-option:** `stylo_equal_channels_v1` cannot silently replace
  `stylo_stack`; it is a different exploratory estimator.
- **Evidence required before disposition:** full outer-fold feasibility,
  class/work coverage, exact calibration selection/fit separation, revised Holm
  family, numerical characterization, and independent statistical review.
- **Owner disposition:** `UNSET`.

### ODM-003 — Public blind benchmark v2 and escrow

- **Linked findings:** AUD-005, AUD-054, AUD-055, AUD-058, AUD-060, and AUD-061.
- **Current fail-closed state:** schema 1.0 blind scoring is
  historical/integration-only; scientific blind scoring is blocked.
- **Question for the owner/custodian:** whether and under whose custody to
  register a dual-manifest v2 benchmark and truth escrow.
- **Required v2 contract:** a public redacted manifest; a custodian
  full-provenance manifest; immutable mapping/digests; exact truth, protocol,
  code and timestamp/signature bindings; work identities; and explicit metric
  units/bootstrap units.
- **Admissible dispositions:** authorize design/review of v2 under a named
  custodian, or retain v1 as synthetic integration/history only.
- **Required implementation after a decision:** new schemas, fixtures, scorer
  route, escrow procedure, independent isolation audit, and versioned outputs.
  No in-place v1 upgrade is allowed.
- **Owner disposition:** `UNSET`.

### ODM-004 — Historical-case open-set applicability

- **Linked finding:** AUD-026.
- **Current fail-closed state:** target passports abstain because
  `target_open_set_applicability_gate_v1` is deliberately unavailable; the
  closed-set winner is diagnostic only.
- **Question for the scientific owner:** retain permanent abstention, or design
  a calibrated development-only open-set/negative-control gate?
- **Admissible dispositions:** keep target attribution descriptive and
  abstaining; or preregister a new gate/passport version with outsider controls,
  absolute-fit thresholds, calibration partitions, and independent validation.
- **Non-option:** changing `implemented=False` to true without a versioned design
  and evidence is not approval.
- **Owner disposition:** `UNSET`.

### ODM-005 — Heterogeneity inference

- **Linked finding:** AUD-045.
- **Current fail-closed state:** the shared-basis standardized contrast is
  descriptive; the former significance/two-hands verdict is withdrawn.
- **Question for the scientific owner:** keep the API permanently descriptive,
  or preregister a work-level inferential design?
- **Evidence required for an inferential route:** independent works/controls,
  one registered sampling unit, a valid null/permutation scheme, multiplicity
  family, minimum sample feasibility, and independent statistical review.
- **Owner disposition:** `UNSET`.

### ODM-006 — Lazy-final-fit estimator deployment

- **Linked finding:** AUD-030.
- **Current fail-closed state:** stack/equal are evaluation-only and cannot be
  serialized into deployment bundles.
- **Question for the model owner:** retain evaluation-only status, or schedule a
  versioned fit-time redesign after a corrected exploratory LOBO?
- **Evidence required for deployment:** fitted-state schema, batch/repeat
  parity, no retained raw training rows, serialization trust boundary,
  numerical parity/declared migration, and bundle compatibility tests.
- **Owner disposition:** `UNSET`.

### ODM-007 — Compatibility API retirement

- **Linked finding:** AUD-053 and the original orphan/legacy-module disposition.
- **Current safe state:** the dormant `make_classifier` keeps its uncalibrated
  import-compatible route, while every learned-calibration request fails
  closed. Legacy builders and loose-artifact entrypoints are hard-disabled or
  exact compatibility shims. `segment.py` is a canonical diagnostic;
  `invariant.py` and `sequence_segmenter.py` have active consumers and are not
  proven orphans.
- **Question for the API owner:** which direct-module imports require a
  compatibility window, and which may be removed in a future major version?
- **Evidence required before deletion:** documented public support policy,
  repository/docs/script reference inventory, external-consumer assessment,
  replacement API, deprecation/release window, and retained historical replay.
- **Owner disposition:** `UNSET`.

### ODM-008 — Author-clustered macro-F1 interval

- **Current fail-closed state:** the interval remains
  `withdrawn_pending_preregistered_recompute`; point macro-F1 uses the fixed
  43-label denominator and is not a substitute for that interval.
- **Question for the statistical owner:** keep the interval withdrawn, or
  preregister a recomputation with an exact author/work resampling estimand?
- **Evidence required for reinstatement:** frozen label universe, resampling
  unit, missing/singleton-author behavior, interval method, simulation or
  coverage justification, and independent review before results are inspected.
- **Owner disposition:** `UNSET`.

## 3. Confirmatory gates

| Gate | Current state | Evidence/action required | Authority |
|---|---|---|---|
| CG-001 corrected corpus | blocked by ODM-001 | New content-safe corpus and exact fold manifests | owner + editorial review |
| CG-002 feasible protocol matrix | blocked by ODM-002 | Versioned amendment or reviewed nested design | protocol/statistical owner |
| CG-003 manifest freeze | `candidate_unapproved`; pin is `None` | Independent review of the post-CG-001/002 candidate, then a separate reviewed digest-pin change | freeze owner + independent reviewer |
| CG-004 production evaluator | `blocked_unregistered`; canonical registry is empty | Implement and independently review the exact callable, config, mechanism passport, evidence adapter, and source digests | evaluator owner + independent reviewer |
| CG-005 execution preflight | not satisfied for confirmatory use | Exact LOBO display-to-slug mapping, pinned-environment live golden replay, clean tree, immutable inputs and durable stage receipts | release/science owner |
| CG-006 execution authorization | not granted | Separate explicit authorization for the exact freeze/evaluator/run identity | execution owner |
| CG-007 independent result audit | no confirmatory result exists | Audit the completed candidate and per-work vectors without reusing producer trust | independent reviewer |
| CG-008 headline/publication | `not_authorized` | Separate decision only after CG-007; publication remains a distinct action | headline/publication owner |

The runner authorization strings and hard-stop constants are mechanisms, not
evidence that any gate has been approved.

## 4. Resolved or retired items not awaiting owner approval

- Executable-source classification and the 280-path release inventory are
  implemented and committed; new path changes remain ordinary release gates.
- The RuAA exact 141-file verification contract is implemented.
- Historical case passports are withdrawn/reclassified; only a future open-set
  design remains open.
- Invalid Fano/certificate inferential claims are withdrawn and exact historical
  evidence is archived.
- Legacy corpus builders, unauthenticated trainers/predictors, and loose-artifact
  diagnostics are hard-disabled or compatibility shims.
- The old corpus-derived `0.8805` claim is ineligible pending a new corpus and
  rerun; it is not a candidate for narrative reapproval.
- The old request to re-freeze a blind checksum and stale uncommitted-protocol
  notes are historical, not current approval tasks.

## 5. Rules for a future disposition

A real owner decision must be a separate, versioned record that identifies the
owner, scope, reviewed evidence, chosen option, date, and affected protocol or
schema version. Implementation then requires its own code/artifact change,
tests, and independent review before the normative ledger can change.

This memo must not be used to:

- alter frozen corpus, protocol, matrix, truth, manifests, or historical
  evidence;
- populate `APPROVED_FREEZE_ROOT_SHA256`;
- populate the canonical evaluator registry;
- enable case open-set or confirmatory execution gates;
- emit a headline, publish evidence, tag, push, or create a release.

Memo disposition: `UNSET`.
