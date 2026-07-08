"""Скачать корпуса кейса dostoevsky_petersburg_chronicle с az.lib.ru (чистый текст, не сканы).

Цель: «Петербургская летопись» Достоевского (1847, 4 фельетона Ф.Д. — позитив-контроль/эталон
фельетонного регистра) и спорный фельетон Н.Н. (13.04.1847). Кандидаты в близком регистре 1840-х:
Достоевский (ранняя проза 1846-1849), Плещеев («Житейские сцены»), Соллогуб (светская проза 1840-х
+ петербургский фельетон «Букеты»). Губер/Корф на az.lib отдельной прозой не представлены —
documented-but-unmodelled.

Сырьё — research input, пишется в gitignored input_cases/dostoevsky_petersburg_chronicle/.
"""
from __future__ import annotations

import html
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "input_cases" / "dostoevsky_petersburg_chronicle"
UA = "Mozilla/5.0"

# (поддиректория, az-путь автора, [(text_id, имя_файла, год)])
JOBS = {
    "petersburg_chronicle": ("d/dostoewskij_f_m", [("text_0350", "peterburgskaya_letopis_FD", 1847)]),
    "cand_dostoevsky": ("d/dostoewskij_f_m", [
        ("text_0010", "bednye_lyudi", 1846), ("text_0140", "dvojnik", 1846),
        ("text_0160", "gospodin_proharchin", 1847), ("text_0170", "hozyajka", 1847),
        ("text_0190", "slaboe_serdce", 1848), ("text_0230", "belye_nochi", 1848),
        ("text_0240", "netochka_nezvanova", 1849)]),
    "cand_dostoevsky_publicistic": ("d/dostoewskij_f_m", [
        ("text_0480", "dnevnik_1876", 1876), ("text_0490", "dnevnik_1877a", 1877),
        ("text_0500", "dnevnik_1877b", 1877), ("text_0520", "dnevnik_1880", 1880)]),
    "cand_pleshcheev": ("p/plesheew_a_n", [("text_0130", "zhitejskie_sceny", 1856)]),
    "cand_sollogub": ("s/sollogub_w_a", [
        ("text_0020", "istoriya_dvuh_kalosh", 1839), ("text_0030", "bolshoj_svet", 1840),
        ("text_0050", "aptekarsha", 1841), ("text_0070", "sobachka", 1845),
        ("text_0080", "vospitannica", 1846), ("text_0090", "metel", 1849),
        ("text_0200", "bukety_peterburgskoe_cvetobesie", 1845)]),
}

# Подвальный виджет lib.ru (рейтинг/e-mail/связь). Эти же слова есть и в ШАПКЕ, поэтому
# отсекаем их ТОЛЬКО после начала тела (первого длинного абзаца).
FOOTER = re.compile(r"^\s*(Оценка:|Ваша оценка|шедевр\b|Связаться с программистом|Обновлено:|"
                    r"Комментарии:|Год:\s*\d|Статистика)", re.I)


def _cyr(s: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]", s))


def fetch(author_path: str, text_id: str) -> str | None:
    url = f"http://az.lib.ru/{author_path}/{text_id}.shtml"
    r = subprocess.run(["curl", "-s", "--max-time", "60", "-A", UA, url], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    return r.stdout.decode("cp1251", "ignore")


def clean(raw: str) -> str:
    body = re.sub(r"(?is)<head.*?</head>|<script.*?</script>", "", raw)
    body = re.sub(r"(?i)<(br|/?p|/?dd|/?center|h[1-6]|/h[1-6])[^>]*>", "\n", body)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", body))
    txt = re.sub(r"[ \t]+", " ", txt)
    lines = [ln.rstrip() for ln in txt.split("\n")]
    # Шапка lib.ru заканчивается рейтинг-виджетом «...очень плохо / не читать». Тело — после него.
    he = next((i for i, ln in enumerate(lines) if ln.strip().lower() == "не читать"), None)
    rest = lines[he + 1:] if he is not None else lines
    out = []
    for ln in rest:
        if FOOTER.match(ln) or ln.strip().lower() == "не читать":
            break
        out.append(ln)
    # отбросить ведущие короткие строки (байлайн «Автор. Заглавие», подзаголовок, эпиграф)
    while out and _cyr(out[0]) < 40:
        out.pop(0)
    return re.sub(r"\n\s*\n+", "\n\n", "\n".join(out)).strip()


def fetch_nn() -> int:
    """Спорный фельетон Н.Н. (13.04.1847) с rvb.ru. rvb отдаёт gzip — нужен --compressed, иначе
    загрузка усекается. Тело — от зачина «Говорят, что в Петербурге весна» до библиоссылки."""
    url = "https://rvb.ru/dostoevski/01text/vol2/17.htm"
    r = subprocess.run(["curl", "-s", "--max-time", "60", "--compressed", "-A", UA, url],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        print("  FAIL target_NN (rvb)", flush=True)
        return 0
    raw = r.stdout.decode("cp1251", "ignore")
    body = re.sub(r"(?is)<head.*?</head>|<script.*?</script>|<style.*?</style>", "", raw)
    body = re.sub(r"(?i)<(br|/?p|/?div|h[1-6]|/h[1-6]|/?td|/?tr)[^>]*>", "\n", body)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", body))
    txt = re.sub(r"[ \t]+", " ", txt)
    i = txt.find("Говорят, что в Петербурге весна")
    if i < 0:
        print("  FAIL target_NN: зачин не найден", flush=True)
        return 0
    tail = txt[i:]
    for mk in ("Петербургская летопись (Коллективное) //", "// Достоевский Ф",
               "Собрание сочинений в 15", "Печатается по"):
        j = tail.find(mk)
        if j > 500:
            tail = tail[:j]
            break
    text = re.sub(r"\n\s*\n+", "\n\n", tail).strip()
    d = OUT / "target_NN"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.txt"):
        old.unlink()
    (d / "nn_13aprelya_1847.txt").write_text(text + "\n", encoding="utf-8")
    w = len(re.findall(r"[А-Яа-яЁё]+", text))
    print(f"  target_NN/nn_13aprelya_1847 [1847, rvb]: {w} слов", flush=True)
    return w


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
    fetch_nn()
    print("\n== объёмы по кандидатам ==")
    for sub in JOBS:
        tot = sum(w for s, _, _, w in summary if s == sub)
        print(f"  {sub}: {tot} слов")


if __name__ == "__main__":
    main()
