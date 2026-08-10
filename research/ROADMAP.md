# Research and release roadmap

Normative current status is recorded in
[`governance/status_ledger.json`](governance/status_ledger.json). If chronology, review-round labels,
future-tense prose, commit labels, or line references in this document disagree with that ledger, the
ledger wins.

This is the single current roadmap for the repository. Scientific cell identifiers such as
`A0/A1/A4` remain stable where they define an estimand; execution chronology belongs in status
metadata, not filenames or directories.

Last reconciled: 2026-08-10.

## Active scientific deliverable

### Complete paired-audit v3.2 on a corrected internal corpus

Normative design: [`work_balanced/paired_audit_protocol.md`](work_balanced/paired_audit_protocol.md).

The former paired-audit v3.1 snapshot and LOBO/RuAA folds are registered as
`ineligible_for_new_scientific_runs`; neither its corpus, freeze candidates, nor historical A0
predictions can be reused for a new scientific run. v3.2 corrected-corpus/fold preparation is
owner-accepted for evaluator implementation; the security review was terminated by the owner and no
independent security-audit PASS is claimed. The evaluator candidate is now implemented. Its single
bounded scientific review found one incomplete-class-universe blocker, which the one allowed
correction pass fixed; no second independent-review PASS is claimed. The corrected candidate is
accepted only as an unregistered input to a later task. This is not a reviewed freeze or an execution
grant.

The remediated preparation boundary derives one atomic local bundle with the exact three exclusions,
full `author_id/work_slug` identities, diagnostic-only expected basename collisions, and the
252/248/134 universe. The former `a70d82f2` candidate was unapproved and is superseded by the new
storage contract. The next gates require separate authorization, in order:

1. Scope the v3.2 RunPlan/evaluator-registration/freeze/preflight boundary as a new task; do not
   infer authorization from the accepted evaluator candidate.
2. Independently review and pin the exact new freeze, then obtain a separate execution authorization.
3. Execute one full run, verify exact resume, and have a separate clean session independently audit
   the durable result before any separately authorized headline decision.

Until all of these gates are satisfied, the freeze is unapproved, the production evaluator is
unregistered, execution is hard-disabled, and headline/publication are not authorized. R1 v5,
sealed evidence, scientific artifacts, historical bytes, and the `0.8805` headline remain unchanged.

## External evidence after the paired audit

### Conduct «внешняя репликация на публичном benchmark без независимого ослепления»

After the corrected paired audit is complete, run the frozen procedure once on a third-party public
corpus. This is an external replication without independent blinding, not a blind benchmark or a
publication decision by itself.

First qualify Russian Stylometric Dataset (RSD) v1.0 with a **metadata-only census**: enumerate its
25 subcorpora, document/document-part and same-novel relationships, available work IDs, author panel,
fixed-split status, licensing/DOI metadata, and potential overlap with R1. Do not fit, predict, or
construct an external split before that census demonstrates a clean panel.

If RSD cannot supply 40–60 authors with several independent whole prose works each, stable work IDs,
and no R1 overlap, do not force it into a prose replication. Instead conduct a public RusDraCor run
as a cross-genre drama stress test and a separate Russian Poetry 2026 out-of-domain stress test.
NCRL «Русская классика» remains a source to qualify only after its offline-export terms and mixed-genre
inventory are separately resolved.

There is presently no known ready corpus that has already demonstrated all of: 40–60 prose authors,
multiple independent whole works per author, stable work IDs, fixed split, and no R1 overlap.

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

Dialogue/NTI-style submission becomes actionable after the v3.2 paired audit, the external benchmark
table, claim table, and reproducible artifact are complete. A stronger venue additionally requires a
domain-specific neural baseline and a qualified external prose replication.

## Completed foundations and historical records

- work-level document and corpus contracts;
- work-balanced feature and loss routing;
- frozen legacy goldens and historical resumable true-LOBO evidence;
- historical completed `753/753` stylo A0/A4/A1 LOBO validation, later made evidence-only by the
  ineligible-corpus registration;
- v3.1 synthetic paired-audit control-plane components, retained as historical implementation evidence
  but not an implementation or authorization of v3.2;
- an independently audited, sealed RuAA R1 v5 bounded exploratory LOBO run, complete as a local,
  not-published milestone; it neither completes the paired audit nor counts as external replication;
- runtime identity binding that omits OS/kernel release strings while binding libc and the numerical
  stack;
- focused and full Python tests, live frozen-golden replay, provenance verification, and site build
  passed on 2026-07-20.

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
