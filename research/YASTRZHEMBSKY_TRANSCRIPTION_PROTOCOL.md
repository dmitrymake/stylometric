# Yastrzhembsky spoof: transcription protocol

Status: `transcription_staging_ready_double_key_pending`.  The primary scans
and page extracts are verified; no stylometric model has seen a transcription.
The PDFs are image-only: `pdftotext` finds zero non-whitespace characters in
both the 29-page target extract and the 9-page evidence extract.

The target is the text beginning after the editorial parenthetical under
`ГЛАВА I` in printed pages 89 onward and ending immediately before
`Сообщ. М. М. Богоявленский` on printed page 117.  The exact physical PDF ranges
are recorded in [`yastrzhembsky_spoof_v1.yaml`](yastrzhembsky_spoof_v1.yaml).

Reproducible full-page 300-dpi grayscale renders and their page-by-page hashes
are created with:

```bash
.venv/bin/python scripts/stage_yastrzhembsky_transcription.py
.venv/bin/python scripts/stage_yastrzhembsky_transcription.py --verify-only
```

The generated hash inventory is
[`yastrzhembsky_transcription_staging_manifest_v1.json`](yastrzhembsky_transcription_staging_manifest_v1.json).
It is staging metadata, not a released benchmark text.

## Two-key procedure

1. Transcriber A keys the diplomatic text directly from the derived scan,
   preserving pre-reform spelling, punctuation, paragraphing, italics markers,
   and uncertain glyphs.  Page and line anchors are retained outside the text.
2. Transcriber B independently keys the same pages without seeing A's output or
   any stylometric result.
3. A reconciliation pass classifies every disagreement as glyph, punctuation,
   word boundary, paragraph, or illegible scan.  Illegible material remains an
   explicit unknown span; it is never silently guessed.
4. A deterministic normalisation script creates a separate modernised copy.
   The diplomatic copy remains the provenance-bearing source of truth.
5. A third reviewer checks the target start/end anchors, page joins, footnote
   exclusion, and all unresolved disagreements against the PDF.

Each key uses one UTF-8 file per source-PDF page (`page-095.txt` through
`page-123.txt`).  Page identifiers remain metadata outside the scored text.
Editorial parentheticals are first transcribed and tagged; they are excluded
only after human source adjudication, never by an inferred author boundary.

## Work estimate

The scored region covers 29 printed pages.  A modern control transcription of
the corresponding narrative interval contains about 13,500 Unicode word
tokens; this is only a planning proxy, not source text and not an estimate of
exact agreement with the 1872 spoof.  Plan for roughly 13,000–14,500 words per
independent key: 10–14 hours for each careful diplomatic key, 4–8 hours for
reconciliation, and 2–4 hours for final source review, or approximately 26–40
person-hours total.  Fully keying the separate nine-page evidence article
would add roughly 8–14 person-hours; for the document-level truth it is enough
to key and review the decisive acknowledgement passages with exact page/line
locators.

## Release gates

The document may enter the public development/gold portion of SPOOF-RU only
after:

- both keyed copies and their SHA-256 hashes are archived;
- reconciliation has zero unresolved word-boundary disagreements in the scored
  region, or those regions are explicitly masked;
- the normalisation transform is versioned and independently rerunnable;
- the truth file labels the document as `spoof_non_gogol`, without asserting
  sentence-level pure authorship by Yastrzhembsky;
- the released manifest states that the case is development-only and was
  already known to the model builders.

It cannot serve as an external blind endpoint: the title, source URL,
responsible author, and document-level truth were all inspected and published
before model evaluation.  A genuinely blind endpoint must instead expose a
redacted submission manifest while a separate custodian escrows provenance and
truth under a public hash until scoring is complete.

OCR or a modern scholarly edition may be used for navigation and discrepancy
triage only.  It cannot replace either independent key from the public-domain
scan, and it must not be copied wholesale into the released corpus without a
separate rights review.

The current scorer must not be run on OCR snippets, the modern control text,
or an empty placeholder.  Until the reconciled diplomatic and deterministic
normalised copies exist, the machine-readable status is
`blocked_pending_reconciled_double_key_text`.  Any later run is public
development/gold only and must include source-matched genuine Gogol controls;
it is not blind evaluation.
