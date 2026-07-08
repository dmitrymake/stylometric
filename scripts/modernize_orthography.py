"""Modernize pre-reform Russian orthography for stylometric processing.

Deterministic character-level rules only:
  ѣ→е, і→и, ѳ→ф, ѵ→и (with case preserved), word-final ъ removed.
Morphological endings («-аго/-яго», «-ыя/-ія») are NOT rewritten: changing
them would edit inflection, not spelling, and fw_fixed does not depend on
them. The fact of modernization must be recorded in the case manifest.

Usage:
  .venv/bin/python scripts/modernize_orthography.py FILE_OR_DIR [...]
    --suffix ""      rewrite in place (default: write <name>.modern.txt)
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

CHAR_MAP = str.maketrans({
    "ѣ": "е", "Ѣ": "Е",
    "і": "и", "І": "И",
    "ѳ": "ф", "Ѳ": "Ф",
    "ѵ": "и", "Ѵ": "И",
})
# ъ only at the end of a word (hard sign inside a word is a real separator).
FINAL_HARD_RE = re.compile(r"(?<=[б-джзк-нп-тф-щ])[ъЪ](?![а-яёА-ЯЁ])")
OLD_CHARS_RE = re.compile(r"[ѣѢіІѳѲѵѴ]")


def modernize(text: str) -> str:
    # Сначала замена букв (ѣ→е и т.д.), затем срез финального ъ: иначе
    # разделительный ъ перед ещё-не-заменённой ѣ (съѣздъ) считался бы финальным.
    text = text.translate(CHAR_MAP)
    return FINAL_HARD_RE.sub("", text)


def is_oldorfo(text: str, probe: int = 20000) -> bool:
    sample = text[:probe]
    return bool(sample) and len(OLD_CHARS_RE.findall(sample)) / len(sample) > 0.002


def process(path: pathlib.Path, suffix: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if not is_oldorfo(text):
        return False
    out = path if not suffix else path.with_name(path.stem + suffix + path.suffix)
    out.write_text(modernize(text), encoding="utf-8")
    print(f"modernized {path} -> {out.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--suffix", default=".modern",
                    help="суффикс имени результата; пустая строка = перезапись на месте")
    args = ap.parse_args()
    n = 0
    for p in args.paths:
        path = pathlib.Path(p)
        files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
        for f in files:
            n += process(f, args.suffix)
    print(f"modernized files: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
