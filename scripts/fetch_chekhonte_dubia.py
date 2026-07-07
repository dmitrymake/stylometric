"""Fetch Chekhov PSS vol. 18 Dubia from FEB.

The raw texts are research inputs, so they are written under ignored
`input_cases/chekhonte_dubia/`. The tracked part is this reproducible fetcher.
"""
from __future__ import annotations

import argparse
import html
import hashlib
import pathlib
import re
import time
from dataclasses import dataclass

import requests
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "input_cases" / "chekhonte_dubia"
BASE = "https://feb-web.ru"
TREE_PATH = "/feb/chekhov/texts/sp0/spj/chj-035-.htm1"
TREE_URL = f"{BASE}/feb/common/tree.asp?{TREE_PATH}"
HEADERS = {"User-Agent": "stylo-research/0.4 (authorship attribution; academic use)"}

TAG = re.compile(r"<[^>]+>")
SCRIPT = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
HIDDEN = re.compile(r"(?is)<div[^>]+style=[\"']?display\s*:\s*none[^>]*>.*")
PAGE_SPAN = re.compile(r"(?is)<span\s+class=page[^>]*>.*?</span>")
H4 = re.compile(r"(?is)<h4\b[^>]*>.*?</h4>")
AZ_TAG = re.compile(r"<[^>]+>")
AZ_JUNK = re.compile(
    r"az\.lib\.ru|lib\.ru|https?://|оставить\s+комментар|copyright|©|"
    r"\bocr\b|оригинал\s+этого|аннотаци|оценка:|перепечат",
    re.I,
)

AZ_CANDIDATES = {
    "lazarev_gruzinsky": {
        "base": "http://az.lib.ru/l/lazarewgruzinskij_a_s/",
        "out": "cand_lazarev_gruzinsky",
        "include_titles": {"Рассказы"},
    },
    "alexander_chekhov": {
        "base": "http://az.lib.ru/c/chehow_aleksandr_pawlowich/",
        "out": "cand_alexander_chekhov",
        "include_titles": {"Рассказы", "Ночной трезвон"},
    },
}


@dataclass
class Work:
    title: str
    href: str


def get(url: str, retries: int = 4, delay: float = 1.5) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = "windows-1251"
            return resp.text
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url}") from last_error


def clean_inline(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slug(text: str) -> str:
    text = clean_inline(text).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "untitled"


def list_dubia() -> list[Work]:
    tree = get(TREE_URL)
    start = tree.find(">Dubia</a>")
    end = tree.find(">Коллективное</a>", start)
    if start < 0 or end < 0:
        raise RuntimeError("could not isolate Dubia section in FEB tree")
    section = tree[start:end]
    works: list[Work] = []
    seen: set[str] = set()
    for href, raw_title in re.findall(
        r'<a href="(/feb/chekhov/texts/sp0/spj/[^"]+?\.htm)\?cmd=1"[^>]*>(.*?)</a>',
        section,
        flags=re.I | re.S,
    ):
        title = clean_inline(TAG.sub("", raw_title))
        if not title or href in seen:
            continue
        seen.add(href)
        works.append(Work(title=title, href=href))
    if not works:
        raise RuntimeError("no Dubia works found in FEB tree")
    return works


def az_list_works(index_html: str) -> list[tuple[str, str]]:
    works: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"<a\s+href=[\"']?(text_\d+\.shtml)[\"']?\s*>(.*?)</a>",
        index_html,
        flags=re.I | re.S,
    ):
        work_id = match.group(1)
        title = clean_inline(AZ_TAG.sub("", match.group(2)))
        if work_id not in seen:
            seen.add(work_id)
            works.append((work_id, title))
    return works


def az_extract_text(page: str) -> str:
    page = SCRIPT.sub(" ", page)
    idx = page.lower().find("<div align=justify")
    body = page[idx:] if idx >= 0 else page
    body = re.sub(r"(?i)<dd>|<p>|<br\s*/?>|</p>", "\n", body)
    body = AZ_TAG.sub(" ", body)
    body = html.unescape(body).replace("\xa0", " ")
    lines: list[str] = []
    for line in body.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or AZ_JUNK.search(line):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def fetch_az_candidates(delay: float) -> list[dict]:
    rows: list[dict] = []
    for name, spec in AZ_CANDIDATES.items():
        base = spec["base"]
        outdir = OUT / spec["out"]
        outdir.mkdir(parents=True, exist_ok=True)
        index = get(base)
        hashes: set[str] = set()
        for work_id, title in az_list_works(index):
            if title not in spec["include_titles"]:
                continue
            page = get(base + work_id)
            text = az_extract_text(page)
            words = len(re.findall(r"[А-Яа-яЁёA-Za-z]+", text))
            if words < 200:
                continue
            digest = hashlib.md5(text[:2000].encode("utf-8", "ignore")).hexdigest()
            if digest in hashes:
                continue
            hashes.add(digest)
            rel = pathlib.Path(spec["out"]) / f"{slug(title)}__{work_id.replace('.shtml', '')}.txt"
            (OUT / rel).write_text(text + "\n", encoding="utf-8")
            rows.append(
                {
                    "candidate": name,
                    "title": title,
                    "words": words,
                    "source_url": base + work_id,
                    "path": str(rel),
                }
            )
            print(f"candidate {name}: {title} {words}w -> {rel}", flush=True)
            time.sleep(delay)
    return rows


def extract_text(page: str) -> tuple[str, str, str]:
    title_match = re.search(r"<meta name=title content='([^']+)'", page, flags=re.I)
    title = clean_inline(title_match.group(1)) if title_match else ""
    kind_match = re.search(r"<div class=text id=([a-z]+)>", page, flags=re.I)
    kind = kind_match.group(1).lower() if kind_match else "unknown"

    body = page
    marker = re.search(r'id=Текст></h4>', body)
    if marker:
        body = body[marker.end():]
    elif kind_match:
        body = body[kind_match.end():]

    body = HIDDEN.sub("", body)
    body = SCRIPT.sub("", body)
    body = body.replace("<!-- BR-->", " ")
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    body = re.sub(r"(?i)</p>|<p\b[^>]*>", "\n", body)
    body = PAGE_SPAN.sub("", body)
    body = H4.sub("", body)
    body = TAG.sub(" ", body)
    body = html.unescape(body).replace("\xa0", " ")
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" *\n *", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return title, kind, body


def write_manifest(rows: list[dict], candidate_rows: list[dict]) -> None:
    manifest = {
        "case": "chekhonte_dubia",
        "source": "FEB: Chekhov PSS vol. 18, Dubia",
        "tree_url": TREE_URL,
        "raw_policy": "Downloaded texts live under ignored input_cases/ and are not committed.",
        "items": rows,
        "candidate_sources": candidate_rows,
    }
    (OUT / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--save-html", action="store_true")
    args = ap.parse_args()

    texts_dir = OUT / "texts"
    prose_dir = OUT / "mystery_prose"
    all_dir = OUT / "mystery_all"
    html_dir = OUT / "html"
    for path in (texts_dir, prose_dir, all_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_html:
        html_dir.mkdir(parents=True, exist_ok=True)
        comments_dir = OUT / "html_comments"
        comments_dir.mkdir(parents=True, exist_ok=True)
        comments_page = get(f"{BASE}/feb/chekhov/texts/sp0/spj/chj-195-.htm?cmd=2")
        (comments_dir / "chj-195-.html").write_text(comments_page, encoding="utf-8")

    rows: list[dict] = []
    prose_parts: list[str] = []
    all_parts: list[str] = []
    for idx, work in enumerate(list_dubia(), start=1):
        url = f"{BASE}{work.href}?cmd=2"
        page = get(url)
        title, kind, text = extract_text(page)
        title = title or work.title
        words = len(re.findall(r"[А-Яа-яЁёA-Za-z]+", text))
        name = f"{idx:02d}_{slug(title)}.txt"
        if args.save_html:
            (html_dir / name.replace(".txt", ".html")).write_text(page, encoding="utf-8")
        rel = pathlib.Path("texts") / name
        (OUT / rel).write_text(text + "\n", encoding="utf-8")
        (all_dir / name).write_text(text + "\n", encoding="utf-8")
        all_parts.append(f"{title}\n\n{text}")
        if kind == "prose":
            (prose_dir / name).write_text(text + "\n", encoding="utf-8")
            prose_parts.append(f"{title}\n\n{text}")
        rows.append(
            {
                "id": f"dubia_{idx:02d}",
                "title": title,
                "tree_title": work.title,
                "kind": kind,
                "words": words,
                "source_url": url,
                "path": str(rel),
                "status": "dubia",
            }
        )
        print(f"{idx:02d}. {title} [{kind}] {words}w -> {rel}", flush=True)
        time.sleep(args.delay)

    (OUT / "mystery_all.txt").write_text("\n\n\n".join(all_parts).strip() + "\n", encoding="utf-8")
    (OUT / "mystery_prose.txt").write_text("\n\n\n".join(prose_parts).strip() + "\n", encoding="utf-8")
    candidate_rows = fetch_az_candidates(args.delay)
    write_manifest(rows, candidate_rows)
    print(
        f"saved={len(rows)} prose={sum(r['kind'] == 'prose' for r in rows)} "
        f"words={sum(r['words'] for r in rows)} candidates={len(candidate_rows)} out={OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
