# Yastrzhembsky intake: OCR and execution audit

Audit date: 2026-07-11.  Role: public development/gold only, not blind.

## What is locally complete

Both raw journal volumes and both exact page extracts pass the pinned byte-size
and SHA-256 checks in `yastrzhembsky_spoof_v1.yaml`:

- 1872 target: source-PDF pages 95–123, printed pages 89–117, 29 pages;
- 1873 evidence: source-PDF pages 251–259, printed pages 244–252, 9 pages.

The reproducible staging command renders 38 full pages at 300 dpi without
cropping and records every render hash, physical-to-printed page mapping,
source hash, tool version, and anchor in
[`yastrzhembsky_transcription_staging_manifest_v1.json`](yastrzhembsky_transcription_staging_manifest_v1.json).
The manifest SHA-256 is
`ec2407ce82ff7fd567d5d9d6ab2599cc25c171b1a6fb5d254dc8dede55606278`.
The ignored local render directory occupies about 11 MiB.

## Text-layer and OCR findings

The local PDFs have no embedded text.  `pdftotext -layout` returns exactly 29
form-feed bytes for the target and 9 for the evidence article, with zero
non-whitespace characters.

The alternative 1,032-page [Wikimedia Commons DjVu](https://commons.wikimedia.org/wiki/File:%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B0%D1%8F_%D0%A1%D1%82%D0%B0%D1%80%D0%B8%D0%BD%D0%B0_%D0%A2%D0%BE%D0%BC_05_%D0%92%D1%8B%D0%BF%D1%83%D1%81%D0%BA%D0%B8_1-6_1872.djvu)
also exposes an empty text-metadata entry for every page through the Wikimedia
API.  It is a higher-resolution image container, not a ready transcription.

[Google Books volume 5](https://books.google.com/books?id=SZ9UAAAAcAAJ)
can locate the opening phrase on printed page 89, but its search endpoint
returns a damaged short snippet rather than a page transcription.  Full EPUB
or accessibility text was not available without an interactive download gate.
It was therefore not harvested or used as corpus text.

No pinned Cyrillic OCR engine was available locally: the installed EasyOCR
weights are English-only, and neither Tesseract nor OCRmyPDF is installed.
An attempted fetch of the public EasyOCR Cyrillic weights failed at the network
layer and produced no model or OCR artifact.  Consequently there is no OCR
draft whose quality can honestly be scored.  This is preferable to treating a
modern edition or search snippets as if they were scan transcription.

Visual inspection at 300 dpi shows a legible, single-column scan suitable for
manual keying.  The main error risks are pre-reform glyphs, bold worn type,
line-end hyphenation, first/last-page editorial matter, and internal editorial
parentheticals.  OCR, if later pinned and run, remains navigation and
discrepancy-triage material only.

## Volume and next executable step

A modern local control copy of the corresponding narrative interval contains
13,482 Unicode word tokens.  Because the 1872 item is a derivative imitation
with variants, this number is only a planning proxy.  The defensible working
range is 13,000–14,500 words across the 29 scored pages.

The next executable step is therefore two independent diplomatic keys:

1. Key A and Key B independently transcribe source pages 95–123 into separate
   page files, preserving spelling, punctuation, paragraphs, hyphenation
   decisions, and explicit uncertainty marks.
2. Both keys use the printed first-text and stop-before anchors.  Editorial
   material is transcribed and tagged before any exclusion; no author boundary
   is inferred.
3. Reconciliation classifies every difference and masks unresolved word
   boundaries rather than guessing.
4. A third reviewer checks page joins, anchor cuts, editorial tags, and all
   unresolved readings against the hashed renders.
5. Only then is a deterministic modernised derivative produced and hashed.

Estimated effort is 10–14 hours per target key, 4–8 hours reconciliation, and
2–4 hours final review: roughly 26–40 person-hours.  Fully double-keying the
separate nine-page evidence article would add about 8–14 person-hours; exact
keying of the decisive acknowledgement passages plus page/line locators is
sufficient for document-level truth evidence.

## Scorer decision

The current scorer was deliberately not run.  There is no reconciled target
text, no valid text artifact hash, and no source-matched genuine-Gogol control
package.  Scoring an empty file, a modern edition, or OCR snippets would test
plumbing while falsely presenting it as this historical case.  The manifest
therefore records `blocked_pending_reconciled_double_key_text`.

When those gates close, the case may be added as known
`spoof_non_gogol` public development/gold.  It cannot become an external blind
endpoint, and the document-level truth must not be expanded into invented
sentence-level or mixed-author boundaries.
