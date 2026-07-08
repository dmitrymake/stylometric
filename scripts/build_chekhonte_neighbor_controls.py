"""Build cleaner neighbour-author controls for the Chekhonte Dubia case.

The existing AZ candidate files for Bilibin and Leikin are anthology dumps with
tables of contents and editorial commentary. This script extracts individual
work bodies into ignored `input_cases/` directories for diagnostic panels.
"""
from __future__ import annotations

import pathlib
import re
import shutil

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "input_cases" / "chekhonte"
CASE_DIR = ROOT / "input_cases" / "chekhonte_dubia"
WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+")


BILIBIN_SOURCES = [
    {
        "path": SOURCE / "cand_bilibin" / "рассказы__0030.txt",
        "titles": [
            "ИЗ МОЛОДЫХ, НО РАННИЙ",
            "СНОВИДЕНИЯ",
            "ИЗ ЗАПИСОК ИНОСТРАНЦА О РОССИИ",
            "ПОД НОВЫЙ ГОД",
            "Я И ОКОЛОТОЧНЫЙ НАДЗИРАТЕЛЬ",
            'ИССЛЕДОВАНИЕ СТРАНЫ, "КУДА МАКАР ТЕЛЯТ НЕ ГОНЯЛ"',
            "ЯЗЫК ПОЭТОВ",
            "ДЕКАДЕНТСКАЯ ПРОЗА",
            "СОКРАЩЕННЫЕ ЛИБРЕТТО",
        ],
    },
    {
        "path": SOURCE / "cand_bilibin" / "рассказы__0040.txt",
        "titles": [
            "ПО ГОРЯЧИМ СЛЕДАМ",
            "ГРАММАТИКА ВЛЮБЛЕННЫХ",
            "КАРТОЧНАЯ РЕФОРМА",
            "ГРЕХИ И ГРЕШКИ",
            "ЛИТЕРАТУРНАЯ ЭНЦИКЛОПЕДИЯ",
            "ВЕСЕЛЫЕ КАРТИНКИ",
            "ЕСЛИ БЫ",
            "ЗАПИСКИ СУМАСШЕДШЕГО ПИСАТЕЛЯ",
            "ДНЕВНИК ПРИКЛЮЧЕНИЙ",
            "У ДОКТОРА",
            "МАРЬЯ ИВАНОВНА",
            "НЕМНОЖКО ФИЛОСОФИИ",
        ],
    },
]

LEIKIN_SOURCES = [
    {
        "path": SOURCE / "cand_lejkin" / "рассказы__0050.txt",
        "titles": [
            "ПТИЦА",
            "ПОСЛЕ СВЕТЛОЙ ЗАУТРЕНИ",
            "САМОГЛОТ-ЗАГРЕБАЕВЫ",
            "КУСТОДИЕВСКИЙ",
            "ИМЕНИНЫ СТАРШЕГО ДВОРНИКА",
            "ПРАЗДНИЧНЫЙ",
            "АЙВАЗОВСКИЙ",
            "В ГОСТЯХ У ХОЗЯИНА",
        ],
    },
    {
        "path": SOURCE / "cand_lejkin" / "рассказы__0090.txt",
        "titles": [
            "ПОЛУЧЕНИЕ МЕДАЛИ",
            "ЧЕРНОЕ МОРЕ",
            "В БАНЯХ",
            "ЕГО СТЕПЕНСТВО",
        ],
    },
    {
        "path": SOURCE / "cand_lejkin" / "сцены__0080.txt",
        "titles": [
            "НОВЫЙ ГОД",
            "НА ПОХОРОНАХ",
            "ДОМОВЛАДЕЛЕЦ",
            "17 СЕНТЯБРЯ",
            "ВО ВРЕМЯ ТАНЦЕВ",
            "ТОРПЕДА",
            "В АПТЕКЕ",
            "НАЛИМ",
            "ЗАТРАВКИН",
            "В РЫБНОЙ ЛАВКЕ",
        ],
    },
]

LEIKIN_SINGLE_WORKS = [
    SOURCE / "cand_lejkin" / "в_рождество__0070.txt",
    SOURCE / "cand_lejkin" / "осенняя_охота__0040.txt",
    SOURCE / "cand_lejkin" / "свет_яблочкова__0020.txt",
]

STOP_HEADINGS = {
    "КОММЕНТАРИИ",
    "СТАТЬИ И КОММЕНТАРИИ",
    "БИЛИБИН ВИКТОР ВИКТОРОВИЧ",
    "ЛЕЙКИН НИКОЛАЙ АЛЕКСАНДРОВИЧ",
}


def norm(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"\s+", " ", text)
    return text.strip().upper()


def slug(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80]


def stop_index(lines: list[str], start: int) -> int:
    for idx in range(start + 1, len(lines)):
        if norm(lines[idx]) in STOP_HEADINGS:
            return idx
    return len(lines)


def find_heading(lines: list[str], title: str, search_from: int) -> int:
    target = norm(title)
    for idx in range(search_from, len(lines)):
        if norm(lines[idx]) == target:
            return idx
    raise RuntimeError(f"heading not found after line {search_from}: {title}")


def clean_body(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and re.fullmatch(r"\([^)]+\)", lines[0].strip()):
        lines.pop(0)
    text = "\n".join(line.rstrip() for line in lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_ordered(source: pathlib.Path, titles: list[str]) -> list[dict]:
    lines = source.read_text("utf-8", "ignore").splitlines()
    starts: list[tuple[str, int]] = []
    search_from = 8
    for title in titles:
        idx = find_heading(lines, title, search_from)
        starts.append((title, idx))
        search_from = idx + 1

    rows = []
    for pos, (title, start) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else stop_index(lines, start)
        body = clean_body(lines[start + 1 : end])
        words = len(WORD.findall(body))
        if words < 80:
            raise RuntimeError(f"extracted too little text for {title}: {words} words")
        rows.append({"title": title, "source": source, "text": body, "words": words})
    return rows


def clean_single(source: pathlib.Path) -> str:
    text = source.read_text("utf-8", "ignore")
    for marker in ("Лейкин Николай Александрович\n", "\n*1\n", "\n*2\n"):
        pos = text.find(marker)
        if pos > 0:
            text = text[:pos]
    lines = text.splitlines()
    if lines and norm(lines[0]).startswith("Н. А. ЛЕЙКИН"):
        lines = lines[1:]
    return clean_body(lines)


def write_controls(name: str, rows: list[dict]) -> None:
    out = CASE_DIR / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for idx, row in enumerate(rows, start=1):
        path = out / f"{idx:02d}_{slug(row['title'])}.txt"
        path.write_text(row["text"] + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "title": row["title"],
                "words": row["words"],
                "source": str(row["source"].relative_to(ROOT)),
                "path": str(path.relative_to(out.parent)),
            }
        )
    manifest = {
        "case": "chekhonte_dubia",
        "control_set": name,
        "raw_policy": "Derived controls live under ignored input_cases/ and are not committed.",
        "items": manifest_rows,
        "total_words": sum(row["words"] for row in manifest_rows),
    }
    (out / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} controls, {manifest['total_words']} words -> {out.relative_to(ROOT)}")


def main() -> None:
    bilibin = []
    for spec in BILIBIN_SOURCES:
        bilibin.extend(extract_ordered(spec["path"], spec["titles"]))
    write_controls("cand_bilibin_clean", bilibin)

    leikin = []
    for spec in LEIKIN_SOURCES:
        leikin.extend(extract_ordered(spec["path"], spec["titles"]))
    for source in LEIKIN_SINGLE_WORKS:
        text = clean_single(source)
        leikin.append(
            {
                "title": source.stem.rsplit("__", 1)[0].replace("_", " ").upper(),
                "source": source,
                "text": text,
                "words": len(WORD.findall(text)),
            }
        )
    write_controls("cand_lejkin_clean", leikin)


if __name__ == "__main__":
    main()
