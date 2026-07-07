"""Укрепить соло-корпус Некрасова прозой с Викитеки (action=parse) для кейса nekrasov_panaeva.

В кейсе nekrasov_panaeva есть значимый, тематически-нейтральный сигнал руки, но macro тянет вниз
тонкий соло-корпус Некрасова (2 работы с az.lib). Добираем его раннюю прозу 1840-х → центроид
стабильнее. Пишется в тот же gitignored nekrasov_solo/ (рядом с az.lib-текстами).
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
OUT = ROOT / "input_cases" / "nekrasov_panaeva" / "nekrasov_solo"
API = "https://ru.wikisource.org/w/api.php"
UA = "Mozilla/5.0"

TITLES = [
    "Петербургские углы",
    "Жизнь и похождения Тихона Тростникова",
    "Опытная женщина",
    "Без вести пропавший пиита",
    "Макар Осипович Случайный",
    "Необыкновенный завтрак",
    "Карета",
    "Сургучов",
    "Капитан Кук",
    "Психологическая задача",
    "Очерки литературной жизни",
    "Двадцать пять рублей",
    "Петербургский ростовщик",
]

SRC_HEAD = re.compile(r"^(Н\.\s*А\.\s*Некрасов|Некрасов|Источник|Печатается|Том\b|Собрание сочинений|"
                      r"Полное собрание|Государственное издательство)", re.I)


def parse_text(title: str) -> str | None:
    for p in (f"{title} (Некрасов)", title):
        url = f"{API}?action=parse&prop=text&format=json&page={urllib.parse.quote(p)}"
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
    h = re.sub(r"(?is)<table.*?</table>|<style.*?</style>|<sup[^>]*>.*?</sup>", " ", h)
    h = re.sub(r"(?is)<div class=\"references.*?</div>", " ", h)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", h))
    txt = txt.replace("[ править ]", " ").replace("[править]", " ")
    txt = re.sub(r"\[\d+\]", " ", txt)
    # отрезать академический аппарат ПСС (комментарии, варианты, примечания) — это near-duplicate
    # черновики и редакторский текст, дают within-work утечку и асимметрично раздувают char3.
    cut = re.search(r"(Другие редакции и варианты|КОММЕНТАРИИ|ПРИМЕЧАНИЯ|Варианты\b|Печатается по|"
                    r"Впервые опубликован|Автограф\b|Список условных сокращений)", txt)
    if cut and cut.start() > 2000:  # резать только после начала тела
        txt = txt[:cut.start()]
    lines = [ln.strip() for ln in re.split(r"\n|\s{3,}", txt)]
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
    total = 0
    for title in TITLES:
        text = parse_text(title)
        if not text:
            print(f"  MISS {title}", flush=True)
            continue
        w = len(re.findall(r"[А-Яа-яЁё]+", text))
        if w < 250:
            print(f"  SKIP {title}: {w} слов", flush=True)
            continue
        slug = "ws_" + re.sub(r"[^a-z0-9а-яё]+", "_", title.lower())[:46].strip("_")
        (OUT / f"{slug}.txt").write_text(text + "\n", encoding="utf-8")
        total += w
        print(f"  {slug}: {w} слов", flush=True)
        time.sleep(1.0)
    print(f"\n== добрано прозы Некрасова: {total} слов ==")


if __name__ == "__main__":
    main()
