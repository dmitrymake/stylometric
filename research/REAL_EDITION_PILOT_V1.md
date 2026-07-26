# Exploratory real-text edition pilot v1

**Run date:** 2026-07-10  
**Status:** internal exploratory evidence; not preregistered; no confirmatory or
breakthrough claim allowed.

## Panel and design

The first real-text feasibility panel contains 18 aligned documents: six works
by Chekhov and Gogol, each represented by a pinned Wikisource page, a pinned
Wikisource `Version 2` page, and the project's local cleaned file.  The exact 12
Wikisource revision ids live in
[`wikisource_edition_pilot_v1.yaml`](wikisource_edition_pilot_v1.yaml).

For every work, the builder retains only long normalised-word blocks present in
all three realisations, while passing the original spelling and punctuation to
the stylometric model.  It found 23,888 common words.  The resulting artifacts
contain 89,519 benchmark-tokenizer tokens; these counts differ because the
benchmark tokenizer also emits punctuation.

The final public manifest SHA-256 is
`f75687712740503a77e35d269c68bd007eccd01cc510449db1eade0cb13270a2`.
A fresh rebuild from the pinned revisions produced a byte-identical text tree;
the final metadata-only repack reverified every artifact hash and records the
source-manifest lineage.  The deterministic pilot report SHA-256 is
`88b6d8a9a082b6b7b0285e5b1b75c9283aa1c1d4219b62f67ad34f7af7eee228`.

This is not yet a release dataset.  The upstream provenance of the local
cleaned files is unknown, and the two Wikisource variants are two editions from
one platform rather than independent digitisation sources.  In addition,
author and period are perfectly confounded in this panel: all three Chekhov
works are labelled `1890s`, while all three Gogol works are labelled `1830s`.
The reported "author" endpoint therefore measures an inseparable
author-or-period contrast and cannot establish preserved authorship signal.

## Leakage-resistant evaluation

Two exploratory checks were run across the three acquisition/version
pipelines:

1. a leave-one-work-out representation probe, with author and edition probes
   trained independently inside every fold;
2. 18 purged edition×work cells, where a test document shares neither its work
   nor its edition level with training.

Every vocabulary, SVD projection, nuisance basis, and classifier was fitted on
the outer training side only.  Bootstrap intervals resample the six works, not
text chunks.  They are descriptive for these six selected works; with only two
authors they are not population-level uncertainty intervals.  This feasibility
panel has no untouched validation split.

## Result

| Evaluation | Char n-gram baseline | Same SVD, no projection | Paired residualizer |
|---|---:|---:|---:|
| Work-holdout author accuracy | 0.667 | 0.667 | 0.667 |
| Work-holdout author macro-F1 | 0.625 | 0.625 | 0.625 |
| Work-holdout edition-probe accuracy | 0.556 | 0.556 | 0.500 |
| Work-holdout edition-probe macro-F1 | 0.519 | 0.519 | 0.522 |
| Purged edition×work author accuracy | 0.667 | 0.667 | 0.667 |
| Purged edition×work author macro-F1 | 0.625 | 0.625 | 0.625 |

The work-cluster bootstrap intervals are necessarily wide: author accuracy
0.333–1.000 and macro-F1 0.250–1.000 for both models.  The residualizer retained
8.2% of measured within-work edition variance in the work-holdout probe and
2.9% in the purged fits.  That variance reduction is partly the fitted
objective, so it is a mechanism diagnostic rather than independent evidence of
better authorship attribution.

## Interpretation

The honest result is **neutral engineering evidence with a mechanism
diagnostic**.  The labelled author-or-period score did not improve or degrade;
edition recoverability fell by 0.056 in accuracy but not in macro-F1.  Because
author is perfectly confounded with period, even retention of that score cannot
be interpreted as retention of author identity.  With two authors and six
works, the panel also cannot distinguish a useful invariant representation from
sampling noise.  It does show that exact content matching, work-purged
evaluation, and deterministic real-text reconstruction run end to end without
same-work leakage.

The estimand is deliberately narrow: the aligner uses every version to retain
only exact normalised-word runs of at least 80 words.  This measures robustness
of surface style on content that survived across all three files.  It does not
measure robustness to rewritten passages, and dataset construction has seen the
unlabelled test text.  A confirmatory general-edition claim therefore needs
externally fixed passage boundaries or an alignment procedure frozen and run
by the benchmark custodian before model development.

The next evidential gate remains a crossed panel of at least 12 authors × three
works × three traceable realisations, including independent-source copies and
an author×period design in which period is not a deterministic proxy for
author.  Its protocol, primary metric, minimum effect, exclusions, and
residualizer rank must be frozen before its test labels are inspected.

## Reproduction

```bash
uv run python scripts/build_wikisource_edition_panel.py \
  --spec research/wikisource_edition_pilot_v1.yaml \
  --out /tmp/wikisource-edition-pilot-v1
uv run stylo benchmark validate \
  /tmp/wikisource-edition-pilot-v1/manifest.json \
  --root /tmp/wikisource-edition-pilot-v1
uv run python scripts/run_edition_invariance_pilot.py \
  /tmp/wikisource-edition-pilot-v1 \
  --out /tmp/wikisource-edition-pilot-v1/pilot_report.json
```
