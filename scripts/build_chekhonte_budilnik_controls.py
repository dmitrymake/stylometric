"""Build same-register Chekhov controls for the Chekhonte Dubia case.

The source text is the local AZ/PSS dump already used by the Chekhonte
validation case. This script extracts only work bodies, not editorial notes, and
writes them under ignored `input_cases/`.
"""
from __future__ import annotations

import pathlib
import re

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "input_cases" / "chekhonte" / "mystery_heldout_chehov_1884_85.txt"
CASE_DIR = ROOT / "input_cases" / "chekhonte_dubia"
OUT_1885 = CASE_DIR / "cand_chehov_budilnik_1885"
OUT_1884_1885 = CASE_DIR / "cand_chehov_budilnik_1884_1885"
WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+")


CONTROLS_1884 = [
    {
        "title": "ИДЕАЛЬНЫЙ ЭКЗАМЕН",
        "first_publication": "Будильник, 1884, no. 23, signature А. Чехонте",
    },
    {
        "title": "КАВАРДАК В РИМЕ",
        "heading": '"КАВАРДАК В РИМЕ"',
        "first_publication": "Будильник, 1884, no. 38, signature Брат моего брата",
    },
    {
        "title": "УСТРИЦЫ",
        "first_publication": "Будильник, 1884, no. 48, signature А. Чехонте",
    },
]

CONTROLS_1885 = [
    {
        "title": "МАСЛЕНИЧНЫЕ ПРАВИЛА ДИСЦИПЛИНЫ",
        "first_publication": "Будильник, 1885, no. 4, signature Брат моего брата",
    },
    {
        "title": "ТОСТ ПРОЗАИКОВ",
        "first_publication": "Будильник, 1885, no. 12, jubilee issue",
    },
    {
        "title": "ЖЕНСКИЙ ТОСТ",
        "first_publication": "Будильник, 1885, no. 12, jubilee issue",
    },
    {
        "title": "ПРАВИЛА ДЛЯ НАЧИНАЮЩИХ АВТОРОВ",
        "first_publication": "Будильник, 1885, no. 12, unsigned",
    },
    {
        "title": "БЕЗНАДЕЖНЫЙ",
        "first_publication": "Будильник, 1885, no. 15, signature А. Чехонте",
    },
    {
        "title": "НА ГУЛЯНЬЕ В СОКОЛЬНИКАХ",
        "first_publication": "Будильник, 1885, no. 17, signature Брат моего брата",
    },
    {
        "title": "ЖЕНЩИНА С ТОЧКИ ЗРЕНИЯ ПЬЯНИЦЫ",
        "first_publication": "Будильник, 1885, no. 17, signature Брат моего брата",
    },
    {
        "title": "КОЕ-ЧТО ОБ А. С. ДАРГОМЫЖСКОМ",
        "first_publication": "Будильник, 1885, no. 20, signature А. Ч.",
    },
    {
        "title": "БУМАЖНИК",
        "first_publication": "Будильник, 1885, no. 20, signature Брат моего брата",
    },
]


def slug(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def body_region(source: str) -> str:
    end = source.find("\nПРИМЕЧАНИЯ\nУСЛОВНЫЕ СОКРАЩЕНИЯ")
    if end < 0:
        raise RuntimeError("could not find notes boundary")
    return source[:end]


def extract_work(body: str, title: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(title)}\s*$")
    match = pattern.search(body)
    if not match:
        raise RuntimeError(f"work title not found: {title}")
    start = body.find("\n", match.end()) + 1
    end = body.find("\nПримечания", start)
    if end < 0:
        raise RuntimeError(f"work end not found: {title}")
    text = body[start:end].strip()
    lines = text.splitlines()
    while lines and re.fullmatch(r"\([^)]+\)", lines[0].strip()):
        lines.pop(0)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def write_set(body: str, out: pathlib.Path, name: str, controls: list[dict]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, item in enumerate(controls, start=1):
        title = item["title"]
        text = extract_work(body, item.get("heading", title))
        words = len(WORD.findall(text))
        path = out / f"{idx:02d}_{slug(title)}.txt"
        path.write_text(text + "\n", encoding="utf-8")
        rows.append(
            {
                "title": title,
                "first_publication": item["first_publication"],
                "words": words,
                "path": str(path.relative_to(out.parent)),
            }
        )
    manifest = {
        "case": "chekhonte_dubia",
        "control_set": name,
        "source": str(SOURCE.relative_to(ROOT)),
        "raw_policy": "Derived controls live under ignored input_cases/ and are not committed.",
        "items": rows,
        "total_words": sum(row["words"] for row in rows),
    }
    (out / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} controls, {manifest['total_words']} words -> {out.relative_to(ROOT)}")
    return manifest


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = body_region(source)
    write_set(body, OUT_1885, "chehov_budilnik_1885", CONTROLS_1885)
    write_set(
        body,
        OUT_1884_1885,
        "chehov_budilnik_1884_1885",
        CONTROLS_1884 + CONTROLS_1885,
    )


if __name__ == "__main__":
    main()
