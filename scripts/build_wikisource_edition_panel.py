#!/usr/bin/env python3
"""Build a content-matched real-text edition panel from pinned Wikisource pages."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from urllib.parse import quote

import requests
import yaml

from stylo.benchmarks import (
    align_editions,
    extract_multi_block_texts,
    intersect_reference_alignments,
    tokenize_with_offsets,
)
from stylo.corpus_tools.fetch_classics import API, HEADERS, wikitext_to_plain
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402


def _opaque(prefix: str, value: str, length: int = 20) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def _canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value independently of whitespace/key order."""
    payload = dumps_strict(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fetch_exact_page(
    title: str,
    revid: int,
    timeout: int = 30,
    *,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
) -> dict:
    """Fetch one explicitly pinned MediaWiki revision.

    Fetching by title and merely recording the then-current revision would make
    a rebuild silently drift.  The acquisition spec therefore owns the
    revision id, and the response is rejected unless MediaWiki returns exactly
    that revision.
    """
    if isinstance(revid, bool) or not isinstance(revid, int) or revid < 1:
        raise ValueError("revid must be a positive integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "ids|timestamp|sha1|content",
        "rvslots": "main",
        "revids": str(revid),
        "format": "json",
        "formatversion": "2",
    }
    retryable_statuses = {429, 500, 502, 503, 504}
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                API,
                params=params,
                headers=HEADERS,
                timeout=timeout,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == max_retries:
                raise
            time.sleep(retry_backoff * (2**attempt))
            continue
        if response.status_code not in retryable_statuses or attempt == max_retries:
            break
        retry_after = response.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after is not None else 0.0
        except ValueError:
            wait = 0.0
        time.sleep(max(wait, retry_backoff * (2**attempt)))
    if response is None:  # pragma: no cover - defensive after the bounded loop.
        raise RuntimeError("Wikisource request produced no response")
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise RuntimeError(f"Wikisource revision not found: {title} @ {revid}")
    revision = pages[0]["revisions"][0]
    content = revision["slots"]["main"]["content"]
    returned_revid = int(revision["revid"])
    if returned_revid != revid:
        raise RuntimeError(
            f"Wikisource returned revision {returned_revid}, expected {revid}"
        )
    returned_title = str(pages[0]["title"])
    if returned_title.replace("_", " ").strip() != title.replace("_", " ").strip():
        raise RuntimeError(
            f"Wikisource revision {revid} belongs to {returned_title!r}, "
            f"not requested title {title!r}"
        )
    return {
        "requested_title": title,
        "title": returned_title,
        "revid": returned_revid,
        "parentid": int(revision.get("parentid", 0)),
        "timestamp": revision["timestamp"],
        "wiki_sha1": revision["sha1"],
        "url": f"https://ru.wikisource.org/w/index.php?title={quote(returned_title.replace(' ', '_'))}&oldid={returned_revid}",
        "plain": wikitext_to_plain(content),
    }


def _write_text(root: pathlib.Path, doc_id: str, text: str) -> tuple[str, str, int]:
    relative = f"texts/{doc_id}.txt"
    payload = (text.strip() + "\n").encode("utf-8")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return relative, hashlib.sha256(payload).hexdigest(), len(tokenize_with_offsets(text.strip() + "\n"))


def build_panel(spec_path: pathlib.Path, out: pathlib.Path, delay: float = 0.25) -> dict:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    project_root = spec_path.resolve().parents[1]
    min_block = int(spec.get("min_block_words", 80))
    min_total = int(spec.get("min_total_matched_words", 1000))
    documents = []
    audits = []

    for entry in spec["works"]:
        if len(entry.get("versions", [])) != 2:
            raise ValueError(f"{entry['work_key']}: exactly two Wikisource versions required")
        fetched = []
        for version in entry["versions"]:
            if not isinstance(version, dict) or set(version) != {"title", "revid"}:
                raise ValueError(
                    f"{entry['work_key']}: each version needs exactly title and revid"
                )
            fetched.append(
                _fetch_exact_page(str(version["title"]), version["revid"])
            )
            time.sleep(delay)
        local_path = project_root / entry["local_path"]
        local_text = local_path.read_text(encoding="utf-8")
        texts = [fetched[0]["plain"], fetched[1]["plain"], local_text]
        pair_reports = [
            align_editions(texts[0], texts[1], min_block_words=min_block),
            align_editions(texts[0], texts[2], min_block_words=min_block),
        ]
        shared = intersect_reference_alignments(pair_reports, min_block_words=min_block)
        matched = sum(block.n_words for block in shared)
        if matched < min_total:
            raise RuntimeError(
                f"{entry['work_key']}: only {matched} common words, need {min_total}"
            )
        rows = extract_multi_block_texts(texts, shared)
        aligned_surfaces = ["\n\n".join(row[i] for row in rows) for i in range(3)]
        work_id = _opaque("work", f"{entry['author']}:{entry['work_key']}")
        variants = [
            {
                "edition": "wikisource_main",
                "source_id": "wikisource",
                "provenance": fetched[0]["url"],
                "revision": (
                    f"revid:{fetched[0]['revid']};timestamp:{fetched[0]['timestamp']};"
                    f"sha1:{fetched[0]['wiki_sha1']}"
                ),
                "evidence": fetched[0]["url"],
            },
            {
                "edition": "wikisource_version2",
                "source_id": "wikisource",
                "provenance": fetched[1]["url"],
                "revision": (
                    f"revid:{fetched[1]['revid']};timestamp:{fetched[1]['timestamp']};"
                    f"sha1:{fetched[1]['wiki_sha1']}"
                ),
                "evidence": fetched[1]["url"],
            },
            {
                "edition": "local_clean_unknown",
                "source_id": "local_unknown",
                "provenance": str(entry["local_path"]),
                "revision": "local-clean:file-provenance-unknown",
                "evidence": f"known canonical author; local file {entry['local_path']}",
            },
        ]
        for variant_index, (variant, surface) in enumerate(zip(variants, aligned_surfaces)):
            semantic = f"{entry['author']}:{entry['work_key']}:{variant['edition']}"
            doc_id = _opaque("doc", semantic)
            text_path, digest, words = _write_text(out, doc_id, surface)
            documents.append(
                {
                    "doc_id": doc_id,
                    "source": {
                        "source_id": variant["source_id"],
                        "provenance": variant["provenance"],
                        "revision": variant["revision"],
                        "sha256": digest,
                    },
                    # This feasibility panel is evaluated only by explicit
                    # work-purged CV; it has no untouched validation set.
                    "split": "train",
                    "task_types": ["idio_shift"],
                    "text_path": text_path,
                    "author_label": entry["author"],
                    "work": work_id,
                    "edition": variant["edition"],
                    "period": str(entry["period"]),
                    "genre": "prose",
                    "topic": _opaque("topic", entry["work_key"]),
                    "register": "literary",
                    "spans": [
                        {
                            "start": 0,
                            "end": words,
                            "label": entry["author"],
                            "ground_truth_known": True,
                            "evidence": variant["evidence"],
                        }
                    ],
                }
            )
        audits.append(
            {
                "author": entry["author"],
                "work_key": entry["work_key"],
                "work_id": work_id,
                "raw_words": [len(text.split()) for text in texts],
                "pairwise": [report.to_dict() for report in pair_reports],
                "shared_blocks": len(shared),
                "shared_words": matched,
                "shared_reference_coverage": matched / pair_reports[0].n_words_a,
                "wikisource_revisions": [
                    {key: value for key, value in item.items() if key != "plain"}
                    for item in fetched
                ],
            }
        )

    manifest = {
        "schema_version": "1.0",
        "dataset": {
            "name": "Wikisource edition alignment pilot",
            "version": "0.1.0",
            "license": "internal-exploratory-mixed-provenance",
            "language": "ru",
            "description": (
                "Content-matched real-text pilot; local_clean upstream provenance is unknown"
            ),
            "offset_unit": "token",
            "tokenizer": "stylo_unicode_word_punct_v1",
        },
        "task_types": ["idio_shift"],
        "documents": documents,
    }
    manifest_payload = (
        dumps_strict(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (out / "manifest.json").write_text(manifest_payload, encoding="utf-8")
    alignment_artifact = {
        "artifact_schema_version": "1.0",
        "acquisition_mode": "spec_pinned_revision",
        # The canonical hash binds the embedded parsed spec, while the manifest
        # byte hash binds this audit to the exact downstream document panel.
        "spec_sha256": _canonical_json_sha256(spec),
        "manifest_sha256": hashlib.sha256(
            manifest_payload.encode("utf-8")
        ).hexdigest(),
        "spec": spec,
        "works": audits,
    }
    (out / "alignment_report.json").write_text(
        dumps_strict(alignment_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "out": str(out.resolve()),
        "n_documents": len(documents),
        "n_works": len(audits),
        "n_authors": len({item["author"] for item in audits}),
        "shared_words": sum(item["shared_words"] for item in audits),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", default="research/wikisource_edition_pilot_v1.yaml"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()
    result = build_panel(pathlib.Path(args.spec), pathlib.Path(args.out), args.delay)
    print(dumps_strict(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
