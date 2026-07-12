# Data identifiability audit — 2026-07-10

## Source and edition invariance

The current corpus is not yet capable of identifying a source/edition-invariant
authorship representation.

- The main manifest contains 290 author/work rows.  Provenance is populated for
  156 rows; 134 are `local/неизвестно`.
- Exactly eight authors occur in both broad source domains (local and
  Wikisource): Andreev, Bunin, Chekhov, Dostoevsky, Gogol, Ilf–Petrov, Tolstoy,
  and Turgenev.
- There are **zero duplicated `(author, work)` rows**, so those crossings use
  different works.  They cannot isolate a digitisation or edition effect from
  content.
- The current 137-document RuAA candidate is entirely sourced from Wikisource,
  so it contains no source contrast.
- The 1835 and 1842 *Taras Bulba* texts are genuine edition variants, but they
  are a large authorial rewrite (20,559 versus 36,601 words).  They are useful
  for IDIOSHIFT-RU, not as a clean same-content source pair.

Consequently, the new paired-edition residualizer must not be presented as
validated on the historical corpus yet.  Its present evidence is a deterministic
synthetic control plus the small exploratory feasibility panel described below.

### Exploratory acquisition update

On 2026-07-10, a reproducible two-author feasibility panel was built from six
works, with three content-matched realisations per work.  It is sufficient to
exercise genuinely purged edition×work evaluation, but it is not sufficient to
validate the scientific claim: only Chekhov and Gogol are represented, both
Wikisource variants come from one platform, and the local files have unknown
upstream provenance.  Author is also perfectly confounded with period (Chekhov
= 1890s; Gogol = 1830s), so its labelled author endpoint is not identifiable as
authorship rather than period/orthography.  The neutral result and exact
reconstruction hashes are reported in
[`REAL_EDITION_PILOT_V1.md`](REAL_EDITION_PILOT_V1.md).

### Acquisition requirement

The first real source/edition pilot should obtain, per included author:

- at least three independent works;
- at least three source/edition realisations per work where legally possible;
- exact revision/shelfmark, raw-byte SHA-256, and a documented normalisation
  chain;
- both same-edition/different-source and different-edition/same-source contrasts;
- author×period crossing sufficient to prevent period from identifying author;
- at least one work held out together with one nuisance level in every primary
  outer split.

The development acquisition target is 12 authors × 3 works × 3 realisations.
This is a planning target, not a frozen confirmatory sample-size claim.

## Natural ground-truth candidates

The existing `input_cases/` collection is valuable, but most entries are
disputed targets rather than gold labels.  The strongest intake candidates are:

| Candidate | Usable task | Current ground truth | Blocker before benchmark inclusion |
|---|---|---|---|
| Yastrzhembsky's fake *Dead Souls* fragments | natural derivative spoof | strong `non-Gogol`; high-confidence responsible author from signed acknowledgement and journal manuscript/letter check | diplomatic transcription and derivative-passage audit before any sentence-level or pure-author claim |
| Fake Vyrubova diary | natural spoof; single-author negative control from the genuine memoir | strong that the diary is a fabrication; creators usually identified jointly as Shchegolev/Tolstoy | primary archival citation and no claim about individual contribution |
| Cherubina de Gabriak | persona spoof; pseudonym/idiolect shift | identity with Elizaveta Dmitrieva is established; Voloshin helped create the persona | disentangle Dmitrieva's verse from Voloshin's editorial/co-creative influence |
| Kozma Prutkov | collective pseudonym; mixed/cluster attribution | collective authorship is established | work-level allocations require a critical edition or manuscript evidence |
| Nekrasov–Panaeva novels | co-authorship and segmentation stress test | co-authorship is established | chapter/region allocations remain textologically disputed and are not gold boundaries |
| Chekhov/Chekhonte | pseudonym and diachronic/register positive control | identity is certain | same-edition, source-matched anchors; Dubia texts remain blind targets |
| *Taras Bulba* 1835→1842 | authorial edition shift | both editions are attributable to Gogol | content change is too large for a source-only nuisance pair |

Veles Book, *Konek-Gorbunok*, the Lenin Testament, and anonymous periodical
targets remain evaluation targets, not ground truth.  They cannot enter the
labelled training or calibration portion merely because a stylometric method
prefers one hypothesis.

## Mixed-authorship minimum

Every gold mixed document needs a complete, gap-free partition.  Unknown regions
are labelled explicitly and remain `ground_truth_known: false`; they are not
silently dropped.  Boundaries require non-stylometric evidence.  The initial
development target is:

- 10 or more natural mixed/co-edited documents with defensible boundaries;
- at least 30 matched single-author controls;
- multiple boundary lengths and contribution ratios;
- at least one unknown/unrepresented author condition;
- a truth custodian who did not build the evaluated model.

Until that intake is complete, segmentation scores are engineering validation,
not historical evidence.

## SPOOF-RU acquisition update

The first primary-source acquisition is now in progress for Yastrzhembsky's
1872 derivative imitation of Gogol.  Both public-domain journal scans have been
downloaded and hashed.  The target occupies source-PDF pages 95–123 (printed
pages 89–117); the 1873 acknowledgement and editorial investigation occupy
source-PDF pages 251–259 (printed pages 244–252).  Exact hashes, anchors, claim
limits, and release requirements are machine-readable in
[`yastrzhembsky_spoof_v1.yaml`](yastrzhembsky_spoof_v1.yaml).  After double-key
transcription it may enter a public development/gold `spoof_non_gogol` task; it
may not be used as an external blind endpoint, because its title, source,
responsible author, and truth have already been inspected and published by the
model-building team.  It also may not be converted into invented mixed-author
boundaries.

The image-only/OCR audit is now executable: 38 full-page 300-dpi staging
renders have page-level hashes in
[`yastrzhembsky_transcription_staging_manifest_v1.json`](yastrzhembsky_transcription_staging_manifest_v1.json).
Neither local PDF contains an embedded text layer, and no acceptable pinned
Cyrillic OCR draft was available.  The scorer remains blocked until independent
double-key reconciliation and source-matched genuine-Gogol controls; see
[`YASTRZHEMBSKY_OCR_AUDIT.md`](YASTRZHEMBSKY_OCR_AUDIT.md).
