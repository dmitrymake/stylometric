"""Добрать register/era-matched публицистику Герцена 1857-1868 с Викитеки (чистый текст).

Эталон Герцена с az.lib на 92% — «Письма из Франции и Италии» (травелог, 1846-52), не совпадает по
эпохе и регистру с колокольной политпублицистикой Огарёва 1857-67. Для честного позитив-контроля
нужна колокольная/полярнозвёздная публицистика самого Герцена тех же лет. Викитека итемизирует её
немного, но достаточно для register/era-matched набора.

Пишется в gitignored input_cases/kolokol_herzen_ogaryov/herzen_kolokol/.
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
OUT = ROOT / "input_cases" / "kolokol_herzen_ogaryov" / "herzen_kolokol"
API = "https://ru.wikisource.org/w/api.php"
UA = "Mozilla/5.0"

TITLES = [
    "Письмо к Александру II",
    "Русские немцы и немецкие русские",
    "Ещё раз Базаров",
    "Кончина Добролюбова",
    "Концы и начала",
    "Very dangerous!!!",
    "10 апреля 1861 и убийства в Варшаве",
    "Resurrexit",
    "Лишние люди и желчевики",
    "Mortuos plango",
    "По поводу одного письма",
    "Très dangereux!!!",
    "Россия и Польша",
    "Ученая Москва",
    "Репетиция",
]

SRC_HEAD = re.compile(r"^(А\.\s*И\.\s*Герцен|Герцен|Источник|Печатается|Том\b|Собрание сочинений|"
                      r"Государственное издательство|Полное собрание)", re.I)


def parse_text(title: str) -> str | None:
    page = f"{title} (Герцен)"
    for p in (page, title):
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
    for old in OUT.glob("*.txt"):
        old.unlink()
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
        slug = re.sub(r"[^a-z0-9а-яё]+", "_", title.lower())[:50].strip("_")
        (OUT / f"{slug}.txt").write_text(text + "\n", encoding="utf-8")
        total += w
        print(f"  {slug}: {w} слов", flush=True)
        time.sleep(1.0)
    print(f"\n== Герцен Колокол Викитека: {total} слов в {len(list(OUT.glob('*.txt')))} работах ==")


if __name__ == "__main__":
    main()
