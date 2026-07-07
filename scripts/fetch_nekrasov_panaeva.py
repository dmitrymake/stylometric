"""Скачать корпуса кейса nekrasov_panaeva с az.lib.ru (чистый текст).

Калибровочный кейс: соавторские романы Некрасова и А. Я. Панаевой (псевдоним «Н. Станицкий») —
«Три страны света» (1848-49) и «Мёртвое озеро» (1851). Раздел труда задокументирован мемуарами
Панаевой. Это КАЛИБРОВКА, не открытие: автор≡тема (у Панаевой соло — семейно-женская линия, у
Некрасова — социально-сатирическая), поэтому разделение может ловить тему. Честный тест — на
тематически-нейтральных служебных словах.

Позитив-контроль (make-or-break): делится ли соло-проза Некрасова и Панаевой по служебным словам.
Соавторские романы качаем целиком; нарезка по главам — в скрипте атрибуции.

Сырьё — gitignored input_cases/nekrasov_panaeva/.
"""
from __future__ import annotations

import html
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "input_cases" / "nekrasov_panaeva"
UA = "Mozilla/5.0"

# (поддиректория, az-путь, [(text_id, имя, год)])
JOBS = {
    "nekrasov_solo": ("n/nekrasow_n_a", [
        ("text_1841", "rostovshik", 1841),          # «Ростовщик»
        ("text_0610", "tonkij_chelovek", 1856),     # «В тот же день...» (= «Тонкий человек», соло)
    ]),
    "panaeva_solo": ("p/panaewa_a_j", [
        ("text_0030", "semejstvo_talnikovyh", 1848),  # дебютный соло-роман
        ("text_0040", "stepnaya_baryshnya", 1855),    # соло-повесть
    ]),
    "coauthored": ("n/nekrasow_n_a", [
        ("text_0200", "tri_strany_sveta", 1849),       # Некрасов + Панаева
        ("text_0510", "mertvoe_ozero_ch1", 1851),
        ("text_0520", "mertvoe_ozero_ch2", 1851),
    ]),
}

FOOTER = re.compile(r"^\s*(Оценка:|Ваша оценка|шедевр\b|Связаться с программистом|Обновлено:|"
                    r"Комментарии:|Год:\s*\d|Статистика)", re.I)


def _cyr(s: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]", s))


def fetch(author_path: str, text_id: str) -> str | None:
    url = f"http://az.lib.ru/{author_path}/{text_id}.shtml"
    r = subprocess.run(["curl", "-s", "--max-time", "120", "-A", UA, url], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    return r.stdout.decode("cp1251", "ignore")


def clean(raw: str) -> str:
    body = re.sub(r"(?is)<head.*?</head>|<script.*?</script>", "", raw)
    body = re.sub(r"(?i)<(br|/?p|/?dd|/?center|h[1-6]|/h[1-6])[^>]*>", "\n", body)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", body))
    txt = re.sub(r"[ \t]+", " ", txt)
    # отрезать академический аппарат ПСС (секция КОММЕНТАРИИ/ПРИМЕЧАНИЯ в конце) — near-duplicate
    # черновики и редакторский текст = within-work утечка. Режем по однозначному заголовку секции,
    # не по инлайн-ссылкам «(см.: Другие редакции...)».
    cut = re.search(r"\bКОММЕНТАРИИ\b|\bПРИМЕЧАНИЯ\b", txt)
    if cut and cut.start() > 2000:
        txt = txt[:cut.start()]
    lines = [ln.rstrip() for ln in txt.split("\n")]
    he = next((i for i, ln in enumerate(lines) if ln.strip().lower() == "не читать"), None)
    rest = lines[he + 1:] if he is not None else lines
    out = []
    for ln in rest:
        if FOOTER.match(ln) or ln.strip().lower() == "не читать":
            break
        out.append(ln)
    while out and _cyr(out[0]) < 40:
        out.pop(0)
    return re.sub(r"\n\s*\n+", "\n\n", "\n".join(out)).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for sub, (author_path, texts) in JOBS.items():
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.txt"):
            old.unlink()
        for text_id, fname, year in texts:
            raw = fetch(author_path, text_id)
            if raw is None:
                print(f"  FAIL {sub}/{fname} ({text_id})", flush=True)
                continue
            text = clean(raw)
            (d / f"{fname}.txt").write_text(text + "\n", encoding="utf-8")
            w = len(re.findall(r"[А-Яа-яЁё]+", text))
            summary.append((sub, fname, year, w))
            print(f"  {sub}/{fname} [{year}]: {w} слов", flush=True)
            time.sleep(1.5)
    print("\n== объёмы ==")
    for sub in JOBS:
        tot = sum(w for s, _, _, w in summary if s == sub)
        print(f"  {sub}: {tot} слов")


if __name__ == "__main__":
    main()
