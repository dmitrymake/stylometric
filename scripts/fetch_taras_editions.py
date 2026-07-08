"""Fetch and clean both Taras Bulba editions from ФЭБ (feb-web.ru).

ФЭБ serves the academic АН СССР text as a frameset; the full document body
lives at `?cmd=2` (cp1251). Page numbers are injected mid-sentence as
`<span class=page>` and footnote references as `<sup><a ...>` — both are
stripped. Output: plain text, one paragraph per line, chapter headings as
`[ГЛАВА <roman>]`, cut to the exact novel body (opening line .. canonical
final sentence).

Writes input_cases/taras_bulba/gogol1835_mirgorod.txt and gogol1842_full.txt;
previous copies are kept under input_cases/taras_bulba/_superseded/.
"""
from __future__ import annotations

import html
import pathlib
import re
import shutil
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "input_cases" / "taras_bulba"
BACKUP_DIR = CASE_DIR / "_superseded"

EDITIONS = {
    "gogol1842_full.txt": {
        "url": "https://feb-web.ru/feb/gogol/texts/gtb/gtb-005-.htm?cmd=2",
        "start": "А поворотись-ка, сын!",
        "end": "и говорили про своего атамана.",
    },
    "gogol1835_mirgorod.txt": {
        "url": "https://feb-web.ru/feb/gogol/texts/gtb/gtb-097-.htm?cmd=2",
        "start": "А поворотись, сынку!",
        "end": "и говорили про своего атамана.",
    },
}

HEADING_RE = re.compile(r"^([IVXLC]+|\d{1,2})$")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("cp1251", "ignore")


def clean(raw: str, start: str, end: str) -> str:
    # Drop inline apparatus before tag stripping.
    raw = re.sub(r"<sup>.*?</sup>", "", raw, flags=re.S)
    # Page break = optional bare <p>, the page-number span, then an optional
    # <p class=text0> continuation of the interrupted sentence: join with a space.
    raw = re.sub(
        r"(?:<p>\s*)?<span class=page[^>]*>[^<]*</span>(?:\s*<p class=text0\b[^>]*>)?",
        " ", raw)
    raw = re.sub(r"<p class=page-note>[^<]*", "", raw)
    raw = re.sub(r"<(script|style).*?</\1>", "", raw, flags=re.S | re.I)
    # A stray continuation paragraph without a page span keeps sentence flow too.
    raw = re.sub(r"<p class=text0\b[^>]*>", " ", raw)
    # Source-HTML line wrapping is not a paragraph boundary: collapse it first,
    # then let only <p>/<h*> tags introduce newlines.
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"<(p|h\d)\b[^>]*>", "\n", raw, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = text.replace("*", " ")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if HEADING_RE.match(line):
            lines.append(f"[ГЛАВА {line}]")
        else:
            lines.append(line)
    body = "\n".join(lines)
    i = body.find(start)
    if i < 0:
        raise RuntimeError(f"start marker not found: {start}")
    # Include the chapter heading immediately preceding the opening line.
    head = body.rfind("[ГЛАВА", 0, i)
    i = head if head >= 0 else i
    j = body.find(end, i)
    if j < 0:
        raise RuntimeError(f"end marker not found: {end}")
    return body[i:j + len(end)].strip() + "\n"


def main() -> int:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in EDITIONS.items():
        out = CASE_DIR / name
        if out.exists():
            shutil.copy2(out, BACKUP_DIR / name)
        body = clean(fetch(spec["url"]), spec["start"], spec["end"])
        out.write_text(body, encoding="utf-8")
        words = len(re.findall(r"[\w\-]+", body, re.U))
        chapters = body.count("[ГЛАВА")
        print(f"wrote {out.relative_to(ROOT)}: {words} words, {chapters} chapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
