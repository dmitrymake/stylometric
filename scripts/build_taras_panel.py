"""Build the hardened Taras Bulba candidate panel from staged az.lib corpora.

Curated whitelists only: the historical suspect (Annenkov, 1840s prose), the
cossack/Ukrainian topic controls (Somov, Narezhny, Grebenka), plus the
Prokopovich 1843 letters as a diagnostic target (he has too little prose to be
a candidate). Old-orthography files are modernized on copy
(scripts/modernize_orthography.py) and flagged in the manifest.

Outputs:
  input_cases/taras_bulba/cand_<author>/<file>.txt   (ignored research inputs)
  input_cases/taras_bulba/prokopovich_letters_1843.txt
  docs/cases/taras_hardened/panel_manifest.json      (tracked provenance)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
STAGING = ROOT / "_staging_corpora"
CASE_DIR = ROOT / "input_cases" / "taras_bulba"
MANIFEST_PATH = ROOT / "docs" / "cases" / "taras_hardened" / "panel_manifest.json"
WORD_RE = re.compile(r"[\w\-]+", re.U)


def _load(module_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ortho = _load(ROOT / "scripts" / "modernize_orthography.py")
_azlib = _load(ROOT / "log" / "fetch_azlib.py")

# candidate id -> (staging dir, az.lib author path, [staged filenames])
PANEL: Dict[str, tuple] = {
    "annenkov_1840s": ("annenkov", "a/annenkow_p_w", [
        # Проза 1840-х — современница добавлений 1842; поздние критика и
        # мемуары исключены.
        "письма_из_за_границы__0030.txt",
        "парижские_письма__0040.txt",
        "путевые_записки__0060.txt",
        "февраль_и_март_в_париже_1848_года__0050.txt",
        "записки_о_французской_революции_1848_года__0070.txt",
    ]),
    "somov": ("somov", "s/somow_o_m", [
        # Повести и былички (умер в 1833; казачья/украинская тематика).
        "гайдамак__0010.txt",
        "юродивый__0020.txt",
        "оборотень__0030.txt",
        "кикимора__0040.txt",
        "матушка_и_сынок__0050.txt",
        "роман_в_двух_письмах__0060.txt",
        "сватовство__0070.txt",
        "бродящий_огонь__0080.txt",
        "в_поле_съезжаются_родом_не_считаются__0090.txt",
        "киевские_ведьмы__0100.txt",
        "русалка__0110.txt",
        "сказание_о_храбром_витязе_укроме_табунщике__0120.txt",
        "сказка_о_никите_вдовиниче__0140.txt",
        "сказки_о_кладах__0150.txt",
        "вывеска__0160.txt",
        "почтовый_дом_в_шато_тьерри__0170.txt",
        "приказ_с_того_света__0180.txt",
        "странный_поединок__0190.txt",
        "купалов_вечер__0280.txt",
        "недобрый_глаз__0290.txt",
    ]),
    "narezhny": ("narezhny", "n/narezhnyj_w", [
        # Украинские романы и повести; «Российский Жилблаз» (1814, не
        # тематический) не включён.
        "бурсак__0020.txt",
        "гаркуша_малороссийский_разбойник__0030.txt",
        "запорожец__0070.txt",
        "два_ивана_или_страсть_к_тяжбам__0040.txt",
    ]),
    "grebenka": ("grebenka", "g/grebenka_e_p", [
        # Проза 1835-1848; стихи и поэма «Богдан» исключены.
        "чайковский__1843_chaykovskiy.txt",
        "нежинский_полковник_золотаренко__1842_polkovnik_zolotarenko_oldorfo.txt",
        "кулик__0050.txt",
        "злой_человек__1844_zloy_chelovek.txt",
        "страшный_зверь__1835_zver.txt",
        "странная_перепелка__1844_perepelka.txt",
        "хвастун__1846_havastun.txt",
        "сеня__1841_senya.txt",
        "заборов__1848_zaborov.txt",
        "рассказ__1846_rasskaz_oldorfo.txt",
    ]),
}

PROKOPOVICH = {
    "url": "http://az.lib.ru/p/prokopowich_n_j/text_1843_letters_to_shevyrev.shtml",
    "out": CASE_DIR / "prokopovich_letters_1843.txt",
    "note": "Письма Н. Я. Прокоповича к С. П. Шевырёву (1843). Прозы для "
            "кандидатской панели у Прокоповича нет (documented-but-unmodelled); "
            "письма идут отдельным диагностическим target.",
}


def source_url(author_path: str, staged_name: str) -> str:
    wid = staged_name.rsplit("__", 1)[-1].removesuffix(".txt")
    return f"http://az.lib.ru/{author_path}/text_{wid}.shtml"


def file_entry(path: pathlib.Path, url: str, modernized: bool) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "words": len(WORD_RE.findall(text)),
        "source_url": url,
        "modernized_orthography": modernized,
    }


def build_candidate(cand_id: str, staging_dir: str, author_path: str,
                    names: List[str]) -> List[Dict[str, object]]:
    out_dir = CASE_DIR / f"cand_{cand_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in names:
        src = STAGING / staging_dir / name
        text = src.read_text(encoding="utf-8")
        modernized = _ortho.is_oldorfo(text)
        if modernized:
            text = _ortho.modernize(text)
        clean_name = name.replace("_oldorfo", "")
        dst = out_dir / clean_name
        dst.write_text(text, encoding="utf-8")
        entries.append(file_entry(dst, source_url(author_path, name), modernized))
    return entries


def fetch_prokopovich() -> Dict[str, object]:
    out = PROKOPOVICH["out"]
    if not out.exists():
        html = _azlib.get(PROKOPOVICH["url"])
        if not html:
            raise RuntimeError(f"unreachable: {PROKOPOVICH['url']}")
        out.write_text(_azlib.extract_text(html), encoding="utf-8")
    entry = file_entry(out, PROKOPOVICH["url"], False)
    entry["note"] = PROKOPOVICH["note"]
    return entry


def main() -> int:
    manifest = {
        "case_id": "taras_bulba_1842_additions",
        "panel_policy": "Кураторские whitelist-ы; тексты об авторах, критика "
                        "поздних периодов, стихи и переводы исключены.",
        "candidates": {},
        "diagnostic_targets": {},
    }
    for cand_id, (staging_dir, author_path, names) in PANEL.items():
        entries = build_candidate(cand_id, staging_dir, author_path, names)
        manifest["candidates"][cand_id] = {
            "files": entries,
            "works": len(entries),
            "words": sum(e["words"] for e in entries),
        }
        print(f"cand_{cand_id}: {len(entries)} работ, "
              f"{manifest['candidates'][cand_id]['words']} слов")
    manifest["diagnostic_targets"]["prokopovich_letters_1843"] = fetch_prokopovich()
    MANIFEST_PATH.write_text(
        dumps_strict(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
