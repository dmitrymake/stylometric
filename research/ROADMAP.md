# Research and release roadmap

Normative current status is recorded in
[`governance/status_ledger.json`](governance/status_ledger.json). If chronology, review-round
labels, future-tense prose, commit labels, or line references in this document disagree with that
ledger, the ledger wins.

This is the single current roadmap for the repository. It replaces root-level TODO files, agent
prompts, numbered waves, increments, and review rounds. Scientific cell identifiers such as
`A0/A1/A4` remain stable where they define an estimand; execution chronology belongs in status
metadata, not filenames or directories.

Last reconciled: 2026-07-26.

## Active scientific deliverable

### Complete the chunk-weighted versus work-balanced paired audit

Normative design: [`work_balanced/paired_audit_protocol.md`](work_balanced/paired_audit_protocol.md).

The synthetic control plane and preparation command now exist. The preparation flow can produce a
local, explicitly unapproved freeze candidate; this is not permission to execute. Current blockers
are:

- independent review and approval of the exact freeze candidate, followed by pinning its digest;
- registration of the canonical production evaluator and evidence adapter;
- mandatory live golden replay and immutable stage receipts;
- clean-tree preflight and separate confirmatory execution authorization;
- confirmatory execution, independent result audit, and a separately authorized headline decision.

The paired audit is complete only when both the LOBO and RuAA sides are assembled and independently
reviewed. Exploratory screens cannot substitute for it.

The audit-only corpus verifier, immutable builder, and synthetic fail-closed tests satisfy the
implementation prerequisite. Confirmatory execution remains hard-disabled until all current ledger
gates are satisfied.

## External evidence

### Run an independent Russian authorship replication

The current corpus and RuAA-derived benchmark are internal/reproducible evidence, not a never-seen
external confirmation. A publication-grade replication needs:

- a frozen public corpus or independently held test set;
- a preregistered split, metrics, exclusions, seeds, and multiplicity family;
- classical baselines on exactly the same folds (char-SVM, BoW, Delta and PAN-style features);
- a domain-specific contrastive authorship baseline;
- author/work-clustered intervals and effect sizes;
- no model or threshold selection after test labels are visible.

### Freeze a reproducible public corpus slice

Keep the full local research corpus separate from a redistributable public-domain snapshot. Publish
the manifest, hashes, licences, acquisition commands, environment lock, and one reconstruction
command.

## Release and publication

### Produce the release artifact

- scrub private/copyright corpus material from public Git history in a separately authorized,
  destructive release operation;
- provide a clean-clone reproduction command and release tag;
- generate a claim table mapping every claim to its dataset, protocol, test, evidence tier, and
  limitation;
- remove remaining hand-copied site literals only after an artifact registry owns those values;
- keep disputed historical cases as stress tests, not the primary methodological claim.

### Prepare the paper decision

Dialogue/NTI-style submission becomes actionable after the paired audit, external baseline table,
claim table, and reproducible artifact are complete. A stronger venue additionally requires the
domain-specific neural baseline and genuinely external replication.

## Completed foundations

- work-level document and corpus contracts;
- work-balanced feature and loss routing;
- group-aware calibration and stacking support;
- frozen legacy goldens;
- weights-only, feature-state, and relative-frequency ablation routing;
- frozen-panel exploratory signal screen;
- resumable true-LOBO implementation and exact legacy parity gate;
- completed `753/753` stylo A0/A4/A1 LOBO validation with independent artifact reassembly and audit;
- post-validation runtime identity that omits OS/kernel release strings while binding libc and the
  numerical stack;
- purpose-based research, runner, evaluator, fixture, and test paths, with historical executed
  sources isolated under [`evidence/stylo_lobo_validation_v1/`](evidence/stylo_lobo_validation_v1/);
- focused and full Python tests, live frozen-golden replay, provenance verification, and site build
  passed on 2026-07-20.

The verified working-tree rework is intentionally not auto-committed; commit only after reviewing
the complete rename/delete diff. This is the explicit commit decision for the current session.

Historical implementation handoffs are local-only under `research/local/`; they are not normative
inputs and must not be linked as the current plan.

## Deferred work

These are useful but do not block the active work-weighting decision:

- broad site redesign and narrative polish;
- further disputed-authorship case intake;
- mixed-authorship benchmark expansion;
- large package-cycle refactors unrelated to checkpoint/provenance contracts;
- bulk movement of `docs/*.json` before an artifact registry replaces hardcoded paths.

## Naming contract

- Name files as `<domain>_<operation>[_<variant>]` or use a domain directory plus a short concrete
  noun such as `work_balanced/estimand.md`.
- Do not use agent names, `wave`, `increment`, `round`, or a bare phase identifier in canonical
  paths.
- Keep a version suffix only when it identifies a real schema, dataset, frozen protocol, or public
  artifact contract.
- Put status, dates, commit hashes, and execution order inside the document or artifact metadata.
- One canonical roadmap; archived plans never compete with it.
