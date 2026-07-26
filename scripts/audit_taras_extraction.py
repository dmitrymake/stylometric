"""Extraction audit for the Taras Bulba 1842 additions targets.

Checks, with 8-word shingles over normalized tokens:
- strict/loose additions are absent from the 1835 edition and present in the
  1842 edition;
- the tovarishchestvo speech is contained in the loose additions;
- the local 1842 edition file and the benchmark anchor copy of Taras Bulba are
  the same work in different normalizations, and the anchor copy is excluded
  from the Gogol anchor in every hardened spec.

Writes docs/cases/taras_hardened/reports/extraction_audit.json.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Dict, List, Set

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
CASE_DIR = ROOT / "input_cases" / "taras_bulba"
DOC_DIR = ROOT / "docs" / "cases" / "taras_hardened"
OUT_PATH = DOC_DIR / "reports" / "extraction_audit.json"
ANCHOR_TB = ROOT / "input_clean" / "gogol" / "тарас_бульба.txt"
SPEC_DIR = DOC_DIR / "specs"

WORD_RE = re.compile(r"[\w\-]+", re.U)
BRACKET_RE = re.compile(r"\[[^\]]*\]")
SHINGLE_N = 8

THRESHOLDS = {
    "additions_in_1835_max": 0.02,
    "loose_in_1835_max": 0.05,
    "additions_in_1842_min": 0.98,
    "speech_in_loose_min": 0.98,
}

FILES = {
    "edition_1835": CASE_DIR / "gogol1835_mirgorod.txt",
    "edition_1842": CASE_DIR / "gogol1842_full.txt",
    "strict": CASE_DIR / "dobavleniya1842_strict.txt",
    "loose": CASE_DIR / "dobavleniya1842_loose.txt",
    "speech": CASE_DIR / "tovarishchestvo_speech.txt",
}

SPEECH_MARKER = "Хочется мне вам сказать, панове, что такое есть наше товарищество."


def _norm(text: str) -> str:
    return BRACKET_RE.sub(" ", text).lower().replace("ё", "е")


def tokens(path: pathlib.Path) -> List[str]:
    return WORD_RE.findall(_norm(path.read_text(encoding="utf-8")))


def fragments(path: pathlib.Path) -> List[List[str]]:
    """One extracted fragment per non-empty line."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        toks = WORD_RE.findall(_norm(line))
        if toks:
            out.append(toks)
    return out


def shingles(toks: List[str], n: int = SHINGLE_N) -> Set[str]:
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def containment(inner: Set[str], outer: Set[str]) -> float:
    if not inner:
        return 0.0
    return len(inner & outer) / len(inner)


def fragment_containment(frags: List[List[str]], source_toks: List[str]) -> float:
    """Word-weighted share of fragments found in the source.

    Fragments are extracted line-wise, so shingles never cross fragment
    boundaries. Fragments shorter than the shingle length are matched as an
    exact token subsequence of the source stream.
    """
    source_sh = shingles(source_toks)
    source_stream = " " + " ".join(source_toks) + " "
    found_words = 0
    total_words = 0
    for toks in frags:
        total_words += len(toks)
        if len(toks) >= SHINGLE_N:
            frag_sh = shingles(toks)
            share = containment(frag_sh, source_sh)
            found_words += share * len(toks)
        else:
            if " " + " ".join(toks) + " " in source_stream:
                found_words += len(toks)
    return found_words / total_words if total_words else 0.0


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    toks = {name: tokens(path) for name, path in FILES.items()}
    sh = {name: shingles(t) for name, t in toks.items()}
    frags = {name: fragments(FILES[name]) for name in ("strict", "loose")}

    cont = {
        "strict_in_1835": fragment_containment(frags["strict"], toks["edition_1835"]),
        "strict_in_1842": fragment_containment(frags["strict"], toks["edition_1842"]),
        "loose_in_1835": fragment_containment(frags["loose"], toks["edition_1835"]),
        "loose_in_1842": fragment_containment(frags["loose"], toks["edition_1842"]),
        "speech_in_loose": containment(sh["speech"], sh["loose"]),
    }

    checks = {
        "strict_absent_from_1835": cont["strict_in_1835"] <= THRESHOLDS["additions_in_1835_max"],
        "loose_absent_from_1835": cont["loose_in_1835"] <= THRESHOLDS["loose_in_1835_max"],
        "strict_present_in_1842": cont["strict_in_1842"] >= THRESHOLDS["additions_in_1842_min"],
        "loose_present_in_1842": cont["loose_in_1842"] >= THRESHOLDS["additions_in_1842_min"],
        "speech_inside_loose": cont["speech_in_loose"] >= THRESHOLDS["speech_in_loose_min"],
        "speech_marker_only_in_1842": (
            " ".join(WORD_RE.findall(_norm(SPEECH_MARKER)))
            in " ".join(toks["edition_1842"])
        ) and (
            " ".join(WORD_RE.findall(_norm(SPEECH_MARKER)))
            not in " ".join(toks["edition_1835"])
        ),
    }

    # Leak check: the additions must not appear in the remaining Gogol anchor
    # works (the anchor is NER-masked, which only lowers verbatim overlap, so a
    # near-zero bound still detects contamination).
    anchor_works = [
        p for p in sorted((ROOT / "input_clean" / "gogol").glob("*.txt"))
        if p.name != "тарас_бульба.txt"
    ]
    anchor_rest_toks: List[str] = []
    for p in anchor_works:
        anchor_rest_toks.extend(tokens(p))
    cont["strict_in_gogol_anchor_rest"] = fragment_containment(
        frags["strict"], anchor_rest_toks)
    cont["loose_in_gogol_anchor_rest"] = fragment_containment(
        frags["loose"], anchor_rest_toks)
    checks["strict_absent_from_anchor_rest"] = (
        cont["strict_in_gogol_anchor_rest"] <= THRESHOLDS["additions_in_1835_max"])
    checks["loose_absent_from_anchor_rest"] = (
        cont["loose_in_gogol_anchor_rest"] <= THRESHOLDS["loose_in_1835_max"])

    # Anchor copy: same work, different file/normalization, excluded everywhere.
    anchor_toks = tokens(ANCHOR_TB)
    anchor = {
        "anchor_path": str(ANCHOR_TB.relative_to(ROOT)),
        "anchor_sha256": sha256(ANCHOR_TB),
        "edition_1842_sha256": sha256(FILES["edition_1842"]),
        "distinct_files": sha256(ANCHOR_TB) != sha256(FILES["edition_1842"]),
        "anchor_words": len(anchor_toks),
        "anchor_mask_chars": ANCHOR_TB.read_text(encoding="utf-8").count("@"),
        "anchor_mask_note": "input_clean держит NER-маскированные тексты, поэтому дословный shingle-overlap с немаскированной редакцией ФЭБ снижен; величина ниже — описательная.",
        "anchor_shingle_overlap_with_1842": containment(
            shingles(anchor_toks), sh["edition_1842"]
        ),
        "anchor_excluded_from_gogol_works": len(anchor_works),
    }
    excluded_in_specs = {}
    for spec in sorted(SPEC_DIR.glob("*.yaml")):
        text = spec.read_text(encoding="utf-8")
        excluded_in_specs[spec.name] = "тарас_бульба.txt" in text and "exclude" in text
    anchor["anchor_excluded_in_every_spec"] = all(excluded_in_specs.values())
    anchor["per_spec_exclusion"] = excluded_in_specs

    report = {
        "case_id": "taras_bulba_1842_additions",
        "shingle_words": SHINGLE_N,
        "normalization": "lowercase, ё→е, bracketed chapter markers removed, word tokens",
        "containment_unit": "per-line fragment (word-weighted); fragments shorter than the shingle length matched as exact token subsequence",
        "fragment_counts": {name: len(f) for name, f in frags.items()},
        "word_counts": {name: len(t) for name, t in toks.items()},
        "shingle_counts": {name: len(s) for name, s in sh.items()},
        "containment": {k: round(v, 6) for k, v in cont.items()},
        "thresholds": THRESHOLDS,
        "checks": checks,
        "all_checks_pass": all(checks.values()) and anchor["distinct_files"]
        and anchor["anchor_excluded_in_every_spec"],
        "anchor": anchor,
        "sources": {
            "edition_1835": "https://feb-web.ru/feb/gogol/texts/gtb/gtb-097-.htm (ФЭБ, изд. АН СССР: редакция «Миргорода» 1835 г., 9 глав)",
            "edition_1842": "https://feb-web.ru/feb/gogol/texts/gtb/gtb-005-.htm (ФЭБ, изд. АН СССР: редакция 1842 г., 12 глав)",
            "extraction": "Выравнивание редакций по 4-граммам предложений; strict = куски 1842 с overlap<0.10 к 1835, loose = overlap<0.20.",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(dumps_strict(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    for k, v in cont.items():
        print(f"  {k}: {v:.4f}")
    print(f"  all_checks_pass: {report['all_checks_pass']}")
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
