"""Build local Taras Bulba case targets and a public provenance manifest.

Raw case texts live under ignored input_cases/. The tracked output is the
manifest with hashes/counts plus specs/passports generated from those local
texts.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Dict


ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
CASE_DIR = ROOT / "input_cases" / "taras_bulba"
DOC_DIR = ROOT / "docs" / "cases" / "taras_hardened"
SPEECH_PATH = CASE_DIR / "tovarishchestvo_speech.txt"
MANIFEST_PATH = DOC_DIR / "target_manifest.json"
WORD_RE = re.compile(r"[\w\-]+", re.U)

SOURCE_FILES = {
    "gogol1835_mirgorod": CASE_DIR / "gogol1835_mirgorod.txt",
    "gogol1842_full": CASE_DIR / "gogol1842_full.txt",
    "additions1842_strict": CASE_DIR / "dobavleniya1842_strict.txt",
    "additions1842_loose": CASE_DIR / "dobavleniya1842_loose.txt",
}

SPEECH_START = "Хочется мне вам сказать, панове, что такое есть наше товарищество."
SPEECH_END = "Пусть же знают они все, что такое значит в Русской земле товарищество."


def _read(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _write_speech() -> None:
    loose = _read(SOURCE_FILES["additions1842_loose"])
    start = loose.find(SPEECH_START)
    if start < 0:
        raise RuntimeError(f"speech start marker not found: {SPEECH_START}")
    end = loose.find(SPEECH_END, start)
    if end < 0:
        raise RuntimeError(f"speech end marker not found: {SPEECH_END}")
    end += len(SPEECH_END)
    speech = loose[start:end].strip() + "\n"
    SPEECH_PATH.write_text(speech, encoding="utf-8")


def _file_meta(path: pathlib.Path) -> Dict[str, object]:
    text = _read(path)
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "words": len(WORD_RE.findall(text)),
    }


def main() -> int:
    _write_speech()
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    files = {name: _file_meta(path) for name, path in SOURCE_FILES.items()}
    files["tovarishchestvo_speech"] = _file_meta(SPEECH_PATH)
    manifest = {
        "case_id": "taras_bulba_1842_additions",
        "raw_policy": "Raw texts live under ignored input_cases/ and are not committed.",
        "targets": files,
        "editions": {
            "gogol1835_mirgorod": {
                "source_edition": "Н. В. Гоголь. ПСС АН СССР, т. 2: «Тарас Бульба», редакция «Миргорода» 1835 г., 9 глав",
                "source_url": "https://feb-web.ru/feb/gogol/texts/gtb/gtb-097-.htm",
                "fetch_script": "scripts/fetch_taras_editions.py",
            },
            "gogol1842_full": {
                "source_edition": "Н. В. Гоголь. ПСС АН СССР, т. 2: «Тарас Бульба», редакция 1842 г., 12 глав",
                "source_url": "https://feb-web.ru/feb/gogol/texts/gtb/gtb-005-.htm",
                "fetch_script": "scripts/fetch_taras_editions.py",
            },
        },
        "extraction": {
            "additions": {
                "script": "scripts/extract_taras_additions.py",
                "method": "Редакция 1842 разбита на предложения; предложение считается добавлением, если доля его словесных 4-грамм, найденных в редакции 1835, ниже порога: strict < 0.10, loose < 0.20. Предложения короче 5 слов не участвуют.",
                "audit": "docs/cases/taras_hardened/reports/extraction_audit.json",
            },
            "tovarishchestvo_speech": {
                "source": str(SOURCE_FILES["additions1842_loose"].relative_to(ROOT)),
                "start_marker": SPEECH_START,
                "end_marker": SPEECH_END,
                "scope": "Focused diagnostic only: one short passage, not the headline claim.",
            },
        },
    }
    MANIFEST_PATH.write_text(dumps_strict(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"wrote {SPEECH_PATH.relative_to(ROOT)}")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
