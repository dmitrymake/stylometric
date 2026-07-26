# Authorship robustness research

## Repository map

- [`ROADMAP.md`](ROADMAP.md) — the only current execution and publication roadmap.
- [`work_balanced/`](work_balanced/) — estimand, runtime routing, calibration, exploratory evidence,
  active LOBO validation, and the confirmatory paired-audit protocol in one domain-owned section.
- `research/local/` — ignored session handoffs and historical plans; never normative.

Canonical paths describe purpose, not chronology or the agent that authored them. Names such as
`wave_1`, `increment_2`, `round_3`, and bare phase codes are prohibited for first-class files.
Version suffixes remain only for actual dataset, schema, or frozen-protocol versions.

This directory is the public, machine-checkable research plan for three linked
tracks:

1. **SPOOF-RU / IDIOSHIFT-RU** — a blind benchmark of natural impersonation,
   editorial intervention, co-authorship, and diachronic style shift;
2. **source/edition invariance** — attribution that survives held-out topic,
   genre, period, source, and edition rather than exploiting digitisation
   traces;
3. **mixed-authorship segmentation** — token-offset evaluation of author
   regions and their boundaries, including single-author negative controls.

The three tracks share one document manifest.  A document receives an opaque
`doc_id`; provenance and nuisance variables are public, while labels for blind
test documents live in a separate truth file held by an external custodian.
For a genuinely blind endpoint, however, an opaque id is not sufficient:
title, source URL, and case-specific provenance can reveal the answer.  The
submission-time manifest therefore redacts case identity and source, while the
custodian escrows full provenance and truth under a public hash and releases
them after scoring.

## Scientific claim hierarchy

The programme deliberately distinguishes four levels of claim:

- **engineering validation**: schemas, hashes, split checks, metrics, and a
  synthetic end-to-end run work;
- **internal evidence**: an improvement is reproduced on development data;
- **external evidence**: the frozen method succeeds on a never-inspected blind
  test set;
- **breakthrough claim**: the external result beats strong classical and
  representation-learning baselines under the full nuisance-factor protocol,
  survives multiplicity correction, and is independently replicated.

No historical attribution is called confirmatory merely because it passes an
internal gate.  A gate p-value measures whether the reference panel is usable;
it is not a p-value for the disputed target's authorship.

## Freeze procedure

[`protocol_v1.yaml`](protocol_v1.yaml) is currently a **draft**, not a
preregistration.  Before any blind target is opened, freeze it with all of:

1. a public immutable timestamp (OSF/Zenodo or an equivalent registry);
2. the exact Git commit and environment lock hash;
3. a manifest hash and opaque test IDs;
4. an independent truth custodian;
5. declared primary endpoints, baselines, exclusions, and multiplicity family.

Changing a frozen primary decision creates a new protocol version and leaves
the old version available.  Exploratory analyses remain useful but are labelled
as such.

## Required benchmark evidence

Every source document must have a full SHA-256 digest, a stable source locator,
source revision or archival shelfmark, licence/public-domain status, work and
edition identifiers, and the nuisance-factor fields needed by the selected
track.  Mixed-authorship ground truth uses half-open token offsets
`[start, end)` and records the non-stylometric evidence supporting each
boundary.

Synthetic documents are only integration controls.  They can never contribute
to a historical or state-of-the-art claim.

## Primary evaluation principles

- Fit vocabularies, scalers, calibration, and model parameters on the training
  side of every outer split.
- When a model uses author centroids, form one unit profile per work and then
  average works equally; work-level test scoring does not by itself remove
  train-side chunk pseudoreplication.
- Keep all editions of a held-out work together unless edition transfer is the
  explicitly registered task.
- Report macro-F1, worst-group accuracy, selective risk/coverage, and
  author/work-clustered uncertainty; do not rely on chunk-weighted accuracy.
- For segmentation, report token macro-F1, boundary F1 at registered
  tolerances, segment IoU, and the false-positive rate on single-author works.
- Resample independent works/documents and contiguous target blocks, never
  neighbouring chunks as IID observations.
- Apply multiplicity correction across the complete registered family of
  models, feature sets, targets, tolerances, and candidate panels.

## Breakthrough bar

A method qualifies for a breakthrough submission only if a frozen external
test shows all of the following:

- a practically meaningful gain over the best registered baseline on the
  primary author-attribution endpoint;
- no material collapse in any registered held-out nuisance factor;
- lower source/edition recoverability at matched author-attribution quality;
- calibrated abstention for unknown authors;
- for segmentation, improvement in both boundary and region metrics without an
  increased single-author false-positive rate;
- an independent rerun from a clean checkout and a second research group or
  textual scholar signing off on provenance and ground truth.

The exact numerical margins are frozen in the dataset-specific protocol before
the blind labels are released; they are not selected after seeing test scores.

## Implemented research stack

- Strict manifest and artifact validation: `stylo benchmark validate`.
- Opaque blind ids, a separately held truth file bound to the public manifest
  hash, and scoring: `stylo benchmark score`.
- Leave-factor-out diagnostics plus purged factor×work splits that share neither
  nuisance level nor work between train and test.
- A paired-edition residualizer that learns nuisance directions only from
  multiple realisations of the same training work.
- Mixed-author region, boundary, IoU, explicitly anonymous-cluster, and
  work-bootstrap metrics, with separate document/work single-author false-positive
  rates. Named attribution cannot use truth-fitted label permutations.
- A Viterbi sequence decoder whose windows must cover every assigned token, and
  development-only tuning with joint single-author specificity, mixed-boundary
  sensitivity, and small-contribution gates. External work-fold validation is
  still required for a scientific claim.

The deterministic integration run is:

```bash
uv run python scripts/build_breakthrough_synthetic.py --out /tmp/stylo-breakthrough
uv run stylo benchmark validate /tmp/stylo-breakthrough/manifest.json \
  --root /tmp/stylo-breakthrough
uv run stylo benchmark score /tmp/stylo-breakthrough/manifest.json \
  /tmp/stylo-breakthrough/truth.synthetic-public.json \
  /tmp/stylo-breakthrough/submission.reference.json \
  --root /tmp/stylo-breakthrough
uv run python scripts/run_breakthrough_pilot.py /tmp/stylo-breakthrough \
  --out /tmp/stylo-breakthrough/pilot-report.json
```

The synthetic pilot is intentionally easy: both char-LR and the paired-edition
model reach 1.0 under purged source/work and edition/work splits.  Its useful
result is plumbing: 100% split coverage, exact blind scoring, and a roughly
two-order-of-magnitude reduction in the constructed within-work nuisance
variance.  It supplies no scientific or historical claim.

The real-data identifiability gap and intake priorities are recorded in
[`DATA_AUDIT.md`](DATA_AUDIT.md) and [`case_registry_v1.yaml`](case_registry_v1.yaml).
The first content-matched real-text feasibility result, including exact hashes
and its deliberately negative claim boundary, is in
[`REAL_EDITION_PILOT_V1.md`](REAL_EDITION_PILOT_V1.md).
The first primary-source SPOOF-RU public development intake, including exact
scan hashes and extraction anchors for Yastrzhembsky's fake Gogol fragments, is in
[`yastrzhembsky_spoof_v1.yaml`](yastrzhembsky_spoof_v1.yaml).
It can be acquired or reverified with
`uv run python scripts/fetch_yastrzhembsky_spoof.py --verify-only` (omit
`--verify-only` for the first download and page extraction).
Transcription renders and their page-level hash manifest are built or checked
with `uv run python scripts/stage_yastrzhembsky_transcription.py` and
`uv run python scripts/stage_yastrzhembsky_transcription.py --verify-only`.
The PDFs have no usable embedded text layer; the OCR/readiness findings and
the explicit scorer block are in
[`YASTRZHEMBSKY_OCR_AUDIT.md`](YASTRZHEMBSKY_OCR_AUDIT.md).
The required independent two-key transcription procedure is in
[`YASTRZHEMBSKY_TRANSCRIPTION_PROTOCOL.md`](YASTRZHEMBSKY_TRANSCRIPTION_PROTOCOL.md).
Because this team has already inspected and published its identity and truth,
Yastrzhembsky is gold development material, not an external blind endpoint.
