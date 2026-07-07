"""Нарезать корпуса братьев Чеховых на отдельные чистые рассказы для brother-confound панели.

Антон: два склеенных тома ПСС (1880-1882, 1883-1884) с ALL-CAPS заголовками рассказов
после секционного маркера "РАССКАЗЫ...". Преамбула, оглавление и редакционные предисловия
идут ДО маркера и отбрасываются.

Александр (А. Седой): три файла-сборника из антологии "Писатели чеховской поры". Нарезаются
по ALL-CAPS заголовкам; СОДЕРЖАНИЕ, КОММЕНТАРИИ, УСЛОВНЫЕ СОКРАЩЕНИЯ и повторы заголовков в
комментарном блоке вырезаются как контаминация.

Результат — игнорируемые work-body .txt по одному рассказу:
  input_cases/chekhonte_dubia/brother_panel/anton/*.txt
  input_cases/chekhonte_dubia/brother_panel/alexander/*.txt
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANTON_SRC = [
    ROOT / "input_cases" / "chekhonte" / "cand_chehov" / "1880_1882.txt",
    ROOT / "input_cases" / "chekhonte" / "cand_chehov" / "1883_1884.txt",
]
ALEX_SRC = ROOT / "input_cases" / "chekhonte_dubia" / "cand_alexander_chekhov"
OUT = ROOT / "input_cases" / "chekhonte_dubia" / "brother_panel"

# ALL-CAPS строка-заголовок: кириллица в верхнем регистре, без строчных букв, 3..70 знаков.
TITLE = re.compile(r"^[А-ЯЁ][А-ЯЁ0-9 ,.\-?!():«»\"'’/]*$")
LOWER = re.compile(r"[а-яё]")
# Маркер начала тела тома: секционный заголовок "РАССКАЗЫ, ПОВЕСТИ, ЮМОРЕСКИ" / "РАССКАЗЫ, ЮМОРЕСКИ".
ANTON_BODY_MARKER = re.compile(r"^РАССКАЗ")
# Служебные ALL-CAPS заголовки, которые не являются рассказами.
NON_STORY = {
    "СОДЕРЖАНИЕ",
    "КОММЕНТАРИИ",
    "КОММЕНТАРИЙ",
    "ПРИМЕЧАНИЯ",
    "УСЛОВНЫЕ СОКРАЩЕНИЯ",
    "ОТ РЕДАКЦИИ",
    "СОЧИНЕНИЯ",
}
# В александровских сборниках всё, начиная с этих заголовков, — комментарный аппарат.
ALEX_STOP = {"КОММЕНТАРИИ", "КОММЕНТАРИЙ", "ПРИМЕЧАНИЯ", "УСЛОВНЫЕ СОКРАЩЕНИЯ"}
# Трейлер lib.ru / сайта «Михаил Чехов» в конце исходника: рейтинг-виджет, имя автора, год,
# e-mail программиста. Эти строки несут утечку метки (имя автора + год) — режем от первого
# маркера до конца файла. Все маркеры стоят в самом хвосте источника.
FOOTER = re.compile(
    r"^\s*(Комментарии:\s*\d|Связаться с программистом|Оценка:|Обновлено:\s*\d"
    r"|Год:\s*\d{4}\s*$|Статистика\.|Сборник\b.*:|Иллюстрации/приложения:"
    r"|Чехов (Антон|Александр|Михаил) Павлович\s*$|шедевр\s*$)"
)


def strip_site_footer(lines: list[str]) -> list[str]:
    for i, ln in enumerate(lines):
        if FOOTER.match(ln):
            return lines[:i]
    return lines


def is_title(line: str) -> bool:
    s = line.strip()
    if not (3 <= len(s) <= 70):
        return False
    if LOWER.search(s):
        return False
    return bool(TITLE.match(s)) and any("А" <= ch <= "я" or ch == "Ё" or ch == "ё" for ch in s)


def slug(title: str, idx: int) -> str:
    base = re.sub(r"[^а-яёa-z0-9]+", "_", title.lower().strip()).strip("_")[:48] or "story"
    return f"{idx:03d}_{base}"


def words(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁёA-Za-z]+", text))


def split_titled(lines: list[str], start: int, stop_titles: set[str]) -> list[tuple[str, str]]:
    """Разбить строки от индекса start на (заголовок, тело) по ALL-CAPS заголовкам.

    Останавливается на первом заголовке из stop_titles. Невошедшие служебные заголовки из
    NON_STORY пропускаются вместе со своим телом.
    """
    stories: list[tuple[str, str]] = []
    cur_title: str | None = None
    buf: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        if cur_title is None:
            return
        body = "\n".join(buf).strip()
        key = cur_title.strip().upper()
        if key in NON_STORY or key in seen:
            return
        if body:
            stories.append((cur_title.strip(), body))
            seen.add(key)

    for line in lines[start:]:
        if is_title(line):
            key = line.strip().upper()
            if key in stop_titles:
                flush()
                cur_title = None
                break
            flush()
            cur_title = line
            buf = []
        else:
            buf.append(line)
    else:
        flush()
    return stories


def segment_anton() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for src in ANTON_SRC:
        lines = strip_site_footer(src.read_text("utf-8", "ignore").splitlines())
        body_start = next(
            (i + 1 for i, ln in enumerate(lines) if ANTON_BODY_MARKER.match(ln.strip())),
            0,
        )
        out.extend(split_titled(lines, body_start, stop_titles=set()))
    return out


BYLINE = re.compile(r"^(рассказ|очерк)\b", re.IGNORECASE)


def segment_alex() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for src in sorted(ALEX_SRC.glob("*.txt")):
        lines = strip_site_footer(src.read_text("utf-8", "ignore").splitlines())
        # Первый ALL-CAPS заголовок-рассказ = начало тела (до него — антологическая преамбула/оглавление).
        first = next(
            (
                i
                for i, ln in enumerate(lines)
                if is_title(ln) and ln.strip().upper() not in NON_STORY
            ),
            None,
        )
        if first is not None:
            out.extend(split_titled(lines, first, stop_titles=ALEX_STOP))
            continue
        # Файл без ALL-CAPS заголовков = один рассказ. Заголовок — первая непустая строка,
        # за ней возможен байлайн ("Рассказ Александра Павловича Чехова (А. Седой)") — отрезаем.
        body_lines = list(lines)
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        title = body_lines[0].strip() if body_lines else src.stem
        body_lines = body_lines[1:]
        if body_lines and BYLINE.match(body_lines[0].strip()):
            body_lines = body_lines[1:]
        body = "\n".join(body_lines).strip()
        if body:
            out.append((title, body))
    return out


def write(stories: list[tuple[str, str]], sub: str) -> list[dict]:
    target = OUT / sub
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.txt"):
        old.unlink()
    rows = []
    for idx, (title, body) in enumerate(stories, 1):
        name = slug(title, idx)
        (target / f"{name}.txt").write_text(body + "\n", encoding="utf-8")
        rows.append({"file": f"{sub}/{name}.txt", "title": title, "words": words(body)})
    return rows


def main() -> None:
    anton = write(segment_anton(), "anton")
    alex = write(segment_alex(), "alexander")
    for label, rows in (("anton", anton), ("alexander", alex)):
        total = sum(r["words"] for r in rows)
        ge300 = [r for r in rows if r["words"] >= 300]
        print(
            f"{label:10s}: {len(rows):3d} stories, {total:6d} words; "
            f">=300w: {len(ge300):3d} stories, {sum(r['words'] for r in ge300):6d} words"
        )
    if len(alex) <= 15:
        print("\nalexander stories:")
        for r in alex:
            print(f"  {r['words']:5d}w  {r['title']}")


if __name__ == "__main__":
    main()
