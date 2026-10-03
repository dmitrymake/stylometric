#!/usr/bin/env python3
"""Reconstruct the attested digital witnesses; never print literary content."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LibraryPage(HTMLParser):
    """Read the provider's explicit bibliography and prose body containers."""

    def __init__(self, markup: str):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str | None]]] = []
        self.metadata: list[str] = []
        self.body: list[str] = []
        self.links: list[str] = []
        self.description = ""
        self.body_containers = 0
        self.feed(markup)

    def inside(self, key: str, value: str) -> bool:
        return any(attrs.get(key) == value for _, attrs in self.stack)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and str(attrs.get("name", "")).lower() == "description":
            self.description = str(attrs.get("content", ""))
        if tag == "a" and attrs.get("href"):
            self.links.append(str(attrs["href"]))
        if tag == "div" and attrs.get("id") == "tbd":
            self.body_containers += 1
        if self.narrative_paragraph() and tag == "br":
            self.body.append("\n")
        if tag == "z" and self.stack and self.stack[-1][1].get("id") == "text":
            self.body.append("\n")
        if tag not in {"meta", "link", "img", "br", "hr", "input", "wbr"}:
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        if self.narrative_paragraph() and tag == "z":
            self.body.append("\n")
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.inside("class", "tabout"):
            self.metadata.append(data)
        if self.narrative_paragraph():
            # Provider prose paragraphs are direct-child <z> elements inside
            # #text. Headers, chapter headings, navigation,
            # dates and separate notes therefore cannot enter the witness.
            if any(tag in {"script", "style", "sup", "note", "fn"} for tag, _ in self.stack):
                return
            self.body.append(data)

    def narrative_paragraph(self) -> bool:
        return any(tag == "z" and index > 0 and self.stack[index - 1][1].get("id") == "text"
                   for index, (tag, _) in enumerate(self.stack))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_body(markup: str) -> str:
    page = LibraryPage(markup)
    if page.body_containers != 1:
        raise ValueError("Expected exactly one prose body container")
    paragraphs = [re.sub(r"\s+", " ", part).strip()
                  for part in "".join(page.body).split("\n")]
    paragraphs = [part for part in paragraphs if part]
    if not paragraphs:
        raise ValueError("Empty prose body")
    return unicodedata.normalize("NFC", "\n\n".join(paragraphs)) + "\n"


def word_count(text: str) -> int:
    """Unicode letter runs, with internal hyphens/apostrophes kept as one word."""
    return len(re.findall(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", text, re.UNICODE))


def bibliography(markup: str) -> dict[str, str]:
    page = LibraryPage(markup)
    metadata = " ".join(" ".join(page.metadata).split())
    date = re.search(r"Даты написания:\s*(.*?)\s*Источник:", metadata)
    source = re.search(r"Источник:\s*(.*?)\s*Добавлено в библиотеку:", metadata)
    if not date or not source:
        raise ValueError("Missing explicit provider bibliography")
    return {"date_label": date.group(1), "edition_label": source.group(1)}


def chapter_urls(markup: str, source_url: str) -> list[str]:
    number = re.search(r"/text/(\d+)/index\.html$", source_url)
    if not number:
        raise ValueError("Invalid provider work URL")
    pattern = rf"/text/{number.group(1)}/p\.(\d+)/index\.html"
    urls = {urljoin(source_url, href) for href in LibraryPage(markup).links
            if re.fullmatch(pattern, href)}
    return sorted(urls, key=lambda url: int(re.search(r"/p\.(\d+)/", url).group(1)))


def fetch(url: str) -> bytes:
    if urlparse(url).scheme != "https" or urlparse(url).hostname != "ilibrary.ru":
        raise ValueError("Only the declared HTTPS provider is supported")
    request = Request(url, headers={"User-Agent": "Stylo-research-reconstruction/1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=15) as response:
                if urlparse(response.url).hostname != "ilibrary.ru":
                    raise ValueError("Unexpected provider redirect")
                return response.read()
        except HTTPError as error:
            raise ValueError(f"Provider HTTP {error.code}: {url}") from None
        except (URLError, TimeoutError):
            if attempt == 2:
                raise ValueError(f"Provider connection failed after three attempts: {url}") from None
    raise AssertionError("Unreachable")


def reconstruct(work: dict, cache: Path, *, fresh: bool = False) -> tuple[bytes, list[dict]]:
    bodies, receipt = [], []
    metadata_path = cache / "metadata.html"
    metadata_raw = metadata_path.read_bytes() if metadata_path.exists() and not fresh else fetch(work["source_url"])
    declared = bibliography(metadata_raw.decode("windows-1251"))
    if declared != {"date_label": work["date_label"], "edition_label": work["edition_label"]}:
        raise ValueError(f"Provider bibliography mismatch for {work['work_id']}")
    if chapter_urls(metadata_raw.decode("windows-1251"), work["source_url"]) != [page["url"] for page in work["pages"]]:
        raise ValueError(f"Provider chapter inventory mismatch for {work['work_id']}")
    if not metadata_path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        metadata_path.write_bytes(metadata_raw)
    receipt.append({"kind": "metadata", "url": work["source_url"], "html_sha256": digest(metadata_raw),
                    "acquired_html_sha256": work["metadata_html_sha256"],
                    "rendering_matches_acquired": digest(metadata_raw) == work["metadata_html_sha256"]})
    for number, source in enumerate(work["pages"], 1):
        target = cache / f"{number:03d}.html"
        raw = target.read_bytes() if target.exists() and not fresh else fetch(source["url"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        bodies.append(extract_body(raw.decode("windows-1251")).rstrip("\n"))
        receipt.append({"kind": "prose", "url": source["url"], "html_sha256": digest(raw),
                        "acquired_html_sha256": source["html_sha256"],
                        "rendering_matches_acquired": digest(raw) == source["html_sha256"]})
    text = "\n\n".join(bodies) + "\n"
    encoded = text.encode("utf-8")
    if digest(encoded) != work["text_sha256"]:
        raise ValueError(f"Text checksum mismatch for {work['work_id']}")
    if word_count(text) != work["word_count"]:
        raise ValueError(f"Word count mismatch for {work['work_id']}")
    return encoded, receipt


def audit_content(manifest: dict, text_root: Path, historical_manifest: Path) -> dict:
    """Exact word-5 set containment, reusing the repository overlap tokenizer.

    Only IDs, hashes, counts and coverage leave this function. The historical
    source/edition quality is not inferred from a low containment value.
    """
    import numpy as np
    from stylo.domain import corpus_identity as ci

    threshold = 0.90
    works = manifest["works"]
    arrays = {}
    for work in works:
        raw = (text_root / f"{work['work_id']}.txt").read_bytes()
        if digest(raw) != work["text_sha256"]:
            raise ValueError(f"Audit text checksum mismatch for {work['work_id']}")
        arrays[work["work_id"]] = ci._word_shingles([raw.decode("utf-8")], 5)

    def coverage(a, b):
        if min(len(a), len(b)) == 0:
            raise ValueError("Empty shingle set in overlap audit")
        return int(np.intersect1d(a, b, assume_unique=True).size) / min(len(a), len(b))

    panel_max, boundary_max, panel_flags = 0.0, 0.0, []
    panel_pairs, boundary_pairs = 0, 0
    for index, left in enumerate(works):
        for right in works[index + 1:]:
            value = coverage(arrays[left["work_id"]], arrays[right["work_id"]])
            panel_pairs += 1
            panel_max = max(panel_max, value)
            if left["role"] != right["role"]:
                boundary_pairs += 1
                boundary_max = max(boundary_max, value)
            if value >= threshold:
                panel_flags.append({"left_work": left["work_id"], "right_work": right["work_id"],
                                    "containment": value})
    historical_raw = historical_manifest.read_bytes()
    historical = json.loads(historical_raw)
    expected = manifest["historical_exposure_check"]["corrected_manifest_sha256"]
    if digest(historical_raw) != expected:
        raise ValueError("Historical manifest checksum mismatch")
    inputs = [row for row in historical["corrected_content_inventory"]
              if row["path"].startswith("input_clean/") and row["path"].endswith(".txt")]
    if len(inputs) != manifest["historical_exposure_check"]["historical_n_works"]:
        raise ValueError("Historical work count mismatch")
    locked = [work for work in works if work["role"] == "locked_test"]
    best = {work["work_id"]: {"work_id": work["work_id"], "historical_work_id": None,
                              "max_containment": 0.0} for work in locked}
    historical_flags = []
    below_minimum = []
    for row in inputs:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe historical input path")
        raw = (historical_manifest.parent / relative).read_bytes()
        if digest(raw) != row["sha256"]:
            raise ValueError("Historical input checksum mismatch")
        historical_id = str(relative.with_suffix(""))[len("input_clean/"):]
        historical_shingles = ci._word_shingles([raw.decode("utf-8")], 5)
        if len(historical_shingles) < 20:
            below_minimum.append({"historical_work_id": historical_id,
                                  "n_unique_word5_shingles": len(historical_shingles)})
        for work in locked:
            wid = work["work_id"]
            value = coverage(arrays[wid], historical_shingles)
            if value > best[wid]["max_containment"]:
                best[wid] = {"work_id": wid, "historical_work_id": historical_id,
                             "max_containment": value}
            if value >= threshold:
                historical_flags.append({"work_id": wid, "historical_work_id": historical_id,
                                         "containment": value})
    return {"method": "stylo.domain.corpus_identity._word_shingles; exact word-5 set intersection / smaller set",
            "method_source_sha256": digest(Path(ci.__file__).read_bytes()),
            "threshold": threshold, "minimum_shingles": 20,
            "interpretation": "No >=0.90 containment means no near-copy found by this lexical check; it does not prove thematic, stylistic, editorial or archival independence.",
            "fresh_text_identities": [{"work_id": w["work_id"], "text_sha256": w["text_sha256"]} for w in works],
            "within_panel": {"n_pairs": panel_pairs, "max_containment": panel_max,
                             "development_locked_pairs": boundary_pairs,
                             "development_locked_max_containment": boundary_max, "flags": panel_flags},
            "locked_vs_historical": {"historical_manifest_sha256": digest(historical_raw),
                                     "n_historical_inputs_verified": len(inputs),
                                     "n_historical_inputs_eligible": len(inputs) - len(below_minimum),
                                     "n_locked_works": len(locked), "n_pairs": len(inputs) * len(locked),
                                     "n_eligible_pairs": (len(inputs) - len(below_minimum)) * len(locked),
                                     "historical_inputs_below_minimum": below_minimum,
                                     "n_pairs_below_minimum": len(below_minimum) * len(locked),
                                     "below_minimum_interpretation": "Exact intersections are computed, but inputs below 20 unique shingles cannot support the ordinary near-copy absence claim for a literary work.",
                                     "per_locked_max": list(best.values()), "flags": historical_flags}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("research/corpora/publication_prose_v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/local/publication_prose_v1"))
    parser.add_argument("--work-id", action="append")
    parser.add_argument("--fresh", action="store_true", help="Fetch every page again even when a cache exists")
    parser.add_argument("--audit-historical-manifest", type=Path,
                        help="After reconstruction, audit all panel texts and locked/historical containment")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    works = manifest["works"]
    if args.work_id:
        requested = set(args.work_id)
        works = [work for work in works if work["work_id"] in requested]
        if requested != {work["work_id"] for work in works}:
            parser.error("Unknown work ID")
    receipts = []
    for work in works:
        work_id = work["work_id"]
        if not re.fullmatch(r"[a-z_]+/[a-z0-9_]+", work_id):
            raise ValueError("Unsafe work ID")
        encoded, pages = reconstruct(work, args.output / "html" / work_id, fresh=args.fresh)
        target = args.output / "texts" / f"{work_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != encoded:
            raise ValueError(f"Refusing to overwrite different text for {work_id}")
        target.write_bytes(encoded)
        receipts.append({"work_id": work_id, "text_sha256": digest(encoded),
                         "word_count": work["word_count"], "pages": pages})
        print(f"{work_id}: {work['word_count']} words; SHA256 {digest(encoded)}")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "reconstruction_receipt.json"
    receipt_path.write_text(json.dumps({"manifest_sha256": digest(args.manifest.read_bytes()),
                                        "works": receipts}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"Verified {len(works)} digital witnesses")
    rendering_changes = sum(not page["rendering_matches_acquired"] for row in receipts for page in row["pages"])
    print(f"HTML rendering differs on {rendering_changes} pages; all processed hashes and bibliographic labels verified")
    if args.audit_historical_manifest:
        audit = audit_content(manifest, args.output / "texts", args.audit_historical_manifest)
        (args.output / "content_overlap_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Content audit: {audit['within_panel']['n_pairs']} panel pairs, "
              f"{audit['locked_vs_historical']['n_pairs']} historical pairs; "
              f"{len(audit['within_panel']['flags']) + len(audit['locked_vs_historical']['flags'])} flags")


if __name__ == "__main__":
    main()
