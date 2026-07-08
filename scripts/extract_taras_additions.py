"""Extract the 1842 additions to Taras Bulba by sentence-level alignment.

Method (recorded in docs/cases/taras_bulba.json): the 1842 edition is split
into sentences; a sentence counts as an addition when the share of its word
4-grams found anywhere in the 1835 edition is below the threshold —
strict < 0.10, loose < 0.20. Sentences shorter than MIN_WORDS are skipped.
Output: one fragment per line, in text order.

Inputs are the complete ФЭБ editions fetched by scripts/fetch_taras_editions.py.
Previous target files are kept under input_cases/taras_bulba/_superseded/.
"""
from __future__ import annotations

import pathlib
import re
import shutil
from typing import List, Set

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "input_cases" / "taras_bulba"
BACKUP_DIR = CASE_DIR / "_superseded"

EDITION_1835 = CASE_DIR / "gogol1835_mirgorod.txt"
EDITION_1842 = CASE_DIR / "gogol1842_full.txt"
OUT = {
    "strict": (CASE_DIR / "dobavleniya1842_strict.txt", 0.10),
    "loose": (CASE_DIR / "dobavleniya1842_loose.txt", 0.20),
}

WORD_RE = re.compile(r"[\w\-]+", re.U)
BRACKET_RE = re.compile(r"\[[^\]]*\]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
NGRAM = 4
MIN_WORDS = 5


def norm_tokens(text: str) -> List[str]:
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def ngrams(toks: List[str], n: int = NGRAM) -> Set[str]:
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def sentences(path: pathlib.Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = BRACKET_RE.sub(" ", line).strip()
        if not line:
            continue
        out.extend(s.strip() for s in SENT_SPLIT_RE.split(line) if s.strip())
    return out


def overlap(sent_toks: List[str], base: Set[str]) -> float:
    grams = ngrams(sent_toks)
    if not grams:
        # Short sentence without a full 4-gram: fall back to token trigrams.
        grams = ngrams(sent_toks, min(3, len(sent_toks)))
        if not grams:
            return 1.0
    return sum(1 for g in grams if g in base) / len(grams)


def main() -> int:
    base = ngrams(norm_tokens(
        BRACKET_RE.sub(" ", EDITION_1835.read_text(encoding="utf-8"))))
    sents = sentences(EDITION_1842)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name, (out_path, thr) in OUT.items():
        if out_path.exists():
            shutil.copy2(out_path, BACKUP_DIR / out_path.name)
        kept: List[str] = []
        total_words = 0
        for sent in sents:
            toks = norm_tokens(sent)
            if len(toks) < MIN_WORDS:
                continue
            if overlap(toks, base) < thr:
                kept.append(sent)
                total_words += len(toks)
        out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"wrote {out_path.relative_to(ROOT)}: {len(kept)} fragments, "
              f"{total_words} words (threshold {thr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
