"""Добрать подписанную публицистику Огарёва с Викитеки (чистый текст, action=parse).

Гейт kolokol_herzen_ogaryov заблокирован тем, что у Огарёва на az.lib чистым текстом по сути одна
крупная работа («Моя исповедь»). Викитека держит десятки его подписанных публицистических статей
1857-1867 («Колокол», «Общее вече», «Голоса из России»), чистым текстом через action=parse (он
разворачивает транслюзию из Page:-пространства). Несколько НЕЗАВИСИМЫХ работ снимают блокировку
leave-one-WORK-out (центроид Огарёва перестаёт быть вырожденным).

Статьи подписанные → как якоря не циркулярны с анонимной целью (передовыми «Колокола»).

Сырьё пишется в gitignored input_cases/kolokol_herzen_ogaryov/ogaryov_wikisource/.
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import subprocess
import time
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "input_cases" / "kolokol_herzen_ogaryov" / "ogaryov_wikisource"
API = "https://ru.wikisource.org/w/api.php"
UA = "Mozilla/5.0"

# Публицистика Огарёва (без стихов). Каждый — отдельная независимая работа.
TITLES = [
    "Что надо делать народу",
    "Современное положение России",
    "Политические письма к старообрядцам",
    "Старовоздвиженская философия",
    "Надгробное слово",
    "Письма к соотечественнику",
    "Письма деревенского жителя",
    "Письма к «Одному из многих»",
    "Письмо из провинции",
    "Письмо к издателю",
    "Замечание на замечание г. Чихачева",
    "Ответ на письмо малороссийского помещика",
    "По поводу чтения Сеченова и Саши",
    "Знание и революция",
    "Идеалы",
    "Предисловие к \"Колоколу\"",
    "Настоящее и думы",
    "Предисловие к неизданному и недоконченному",
    "Письмо к С. Г. Волконскому",
    "Ответы на статью Герцена «Между старичками»",
]

# шапка издания-источника и редакторские пометы, которые надо снять с начала тела
SRC_HEAD = re.compile(r"^(Н\.\s*П\.\s*Огарев|Огарев|Источник|Печатается|Том\b|Государственное издательство|"
                      r"Избранные)", re.I)


def parse_text(title: str) -> str | None:
    for suffix in (" (Огарев)", " (Огарёв)", ""):
        page = title + suffix
        url = f"{API}?action=parse&prop=text&format=json&page={urllib.parse.quote(page)}"
        r = subprocess.run(["curl", "-s", "--max-time", "60", "-A", UA, url], capture_output=True)
        if r.returncode != 0 or not r.stdout:
            continue
        try:
            d = json.loads(r.stdout.decode("utf-8", "ignore"))
        except json.JSONDecodeError:
            continue
        if "error" in d:
            continue
        h = d.get("parse", {}).get("text", {}).get("*", "")
        if h:
            return _clean(h)
    return None


def _clean(h: str) -> str:
    # выкинуть таблицы (навигация/инфо), сноски-надстрочники, стили/служебное
    h = re.sub(r"(?is)<table.*?</table>|<style.*?</style>|<sup[^>]*>.*?</sup>", " ", h)
    h = re.sub(r"(?is)<div class=\"references.*?</div>", " ", h)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", h))
    txt = txt.replace("[ править ]", " ").replace("[править]", " ")
    txt = re.sub(r"\[\d+\]", " ", txt)                 # маркеры сносок
    lines = [ln.strip() for ln in re.split(r"\n|\s{3,}", txt)]
    # отрезать ведущую шапку-источник (издание 1952/1956)
    out, started = [], False
    for ln in lines:
        if not started:
            if not ln or SRC_HEAD.match(ln) or len(re.findall(r"[А-Яа-яЁё]", ln)) < 30:
                continue
            started = True
        out.append(ln)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()
    total = 0
    for i, title in enumerate(TITLES):
        text = parse_text(title)
        if not text:
            print(f"  MISS {title}", flush=True)
            continue
        w = len(re.findall(r"[А-Яа-яЁё]+", text))
        if w < 250:
            print(f"  SKIP {title}: {w} слов (мало)", flush=True)
            continue
        slug = re.sub(r"[^a-z0-9а-яё]+", "_", title.lower())[:50].strip("_")
        (OUT / f"{slug}.txt").write_text(text + "\n", encoding="utf-8")
        total += w
        print(f"  {slug}: {w} слов", flush=True)
        time.sleep(1.0)
    print(f"\n== Огарёв Викитека: {total} слов в {len(list(OUT.glob('*.txt')))} работах ==")


if __name__ == "__main__":
    main()
