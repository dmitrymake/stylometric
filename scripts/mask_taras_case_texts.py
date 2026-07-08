"""Symmetric NER masking for the Taras Bulba case texts.

The benchmark anchors under input_clean/ are NER-masked (PERSON -> "@"), while
case targets and the az.lib panel are raw. Character-level channels (char3)
compare texts literally, so both sides must be masked the same way. This
script applies scripts.clean_text.mask_names to every case text and mirrors
the layout under input_cases/taras_bulba/masked/.

Run from the repo root:
  .venv/bin/python scripts/mask_taras_case_texts.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.clean_text import mask_names  # noqa: E402

CASE_DIR = ROOT / "input_cases" / "taras_bulba"
OUT_DIR = CASE_DIR / "masked"

TARGET_FILES = [
    "dobavleniya1842_strict.txt",
    "dobavleniya1842_loose.txt",
    "tovarishchestvo_speech.txt",
    "gogol1835_mirgorod.txt",
    "prokopovich_letters_1843.txt",
]
CAND_DIRS = [
    "cand_annenkov_1840s",
    "cand_somov",
    "cand_narezhny",
    "cand_grebenka",
]


def mask_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(mask_names(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"masked {src.relative_to(CASE_DIR)}", flush=True)


def main() -> int:
    for name in TARGET_FILES:
        mask_file(CASE_DIR / name, OUT_DIR / name)
    for cand in CAND_DIRS:
        for src in sorted((CASE_DIR / cand).glob("*.txt")):
            mask_file(src, OUT_DIR / cand / src.name)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
