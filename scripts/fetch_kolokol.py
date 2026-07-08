"""Скачать подписанные корпуса кейса kolokol_herzen_ogaryov с az.lib.ru (чистый текст, не сканы).

Цель кейса: неподписанные передовые «Колокола» (1857-1867) — Герцен или Огарёв. ПЕРВЫЙ шаг —
позитив-контроль: разделяет ли панель самих авторов в публицистическом регистре. Герцен и Огарёв
десятилетиями правили тексты друг друга, поэтому это make-or-break: если их подписанная публицистика
не разделяется, кейс закрывается на пороге.

Связывающее ограничение — Огарёв: его подписанная публицистика чистым текстом тонкая («Моя исповедь»
1862 + предисловия); колокольная политпублицистика Огарёва («Политические письма» 1864) лежит на
az.lib только в сканах (/img/, OCR на потом). Якоря берём ВНЕ «Колокола», чтобы исключить
циркулярность с целью.

Сырьё — research input, пишется в gitignored input_cases/kolokol_herzen_ogaryov/.
"""
from __future__ import annotations

import html
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "input_cases" / "kolokol_herzen_ogaryov"
UA = "Mozilla/5.0"

# (поддиректория, az-путь автора, [(text_id, имя_файла, год, жанр)])
# Якоря ВНЕ «Колокола»: публицистическая/очерковая проза обоих. Французские тексты Герцена
# (La Russie, Lettre a Mazzini) исключены. Художественная беллетристика («Кто виноват?», «Сорока-
# воровка») исключена — другой регистр. «Very Dangerous!!!» (1859) исключён: это сам «Колокол».
JOBS = {
    "herzen_publicistic": ("g/gercen_a_i", [
        ("text_0420", "pisma_iz_francii_i_italii", 1852),   # журналистская публицистика, крупный
        ("text_0390", "sostav_russkogo_obshestva", 1846),
        ("text_0470", "sharlotta_korde", 1850),
        ("text_0380", "mihail_bakunin", 1852),              # биографическая публицистика
        ("text_0440", "vmesto_predisloviya", 1849),
    ]),
    "ogaryov_publicistic": ("o/ogarew_n_p", [
        ("text_0230", "moya_ispoved", 1862),                # Мемуары, Публицистика — основной якорь
        ("text_0110", "predislovie_k_dumam_ryleeva", 1859), # Критика/публицистика
    ]),
}

# Подвальный виджет lib.ru (рейтинг/e-mail/связь). Те же слова есть и в ШАПКЕ, поэтому
# отсекаем их ТОЛЬКО после начала тела (первого длинного абзаца).
FOOTER = re.compile(r"^\s*(Оценка:|Ваша оценка|шедевр\b|Связаться с программистом|Обновлено:|"
                    r"Комментарии:|Год:\s*\d|Статистика)", re.I)


def _cyr(s: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]", s))


def fetch(author_path: str, text_id: str) -> str | None:
    url = f"http://az.lib.ru/{author_path}/{text_id}.shtml"
    r = subprocess.run(["curl", "-s", "--max-time", "90", "-A", UA, url], capture_output=True)
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
    print("\n== объёмы по авторам ==")
    for sub in JOBS:
        tot = sum(w for s, _, _, w in summary if s == sub)
        print(f"  {sub}: {tot} слов")


if __name__ == "__main__":
    main()
