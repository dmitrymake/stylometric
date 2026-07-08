"""Скачать подписанную литкритику кейса sovremennik (5 критиков) с az.lib.ru (чистый текст).

Кейс: безымянная критика «Современника» 1854-1862. Две оси:
- разделимая (валидация инструмента): радикалы (Чернышевский, Добролюбов) ↔ эстетики (Дружинин,
  Анненков, Боткин) — разные школы, должны делиться;
- неразделимая (честный негатив): Чернышевский ↔ Добролюбов (учитель↔ученик), ожидаемо не делятся.

Якоря — подписанная ЛИТЕРАТУРНАЯ критика тех же лет (исключены философия, травелоги, рецензии чужих
книг под тем же названием). Сырьё — gitignored input_cases/sovremennik/.
"""
from __future__ import annotations

import html
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "input_cases" / "sovremennik"
UA = "Mozilla/5.0"

# (поддиректория, школа, az-путь, [(text_id, имя, год)])
JOBS = {
    "chernyshevsky": ("radical", "c/chernyshewskij_n_g", [
        ("text_0190", "pushkin", 1855), ("text_0130", "gogol", 1857),
        ("text_0250", "gubernskie_ocherki", 1857), ("text_0420", "vozvyshennoe_komicheskoe", 1855),
        ("text_0270", "ne_nachalo_li_peremeny", 1861)]),
    "dobrolyubov": ("radical", "d/dobroljubow_n_a", [
        ("text_0180", "temnoe_carstvo", 1859), ("text_0040", "luch_sveta", 1860),
        ("text_0150", "literaturnye_melochi", 1859), ("text_0470", "sobesednik", 1856),
        ("text_0750", "russkaya_satira", 1859)]),
    "druzhinin": ("aesthete", "d/druzhinin_a_w", [
        ("text_0120", "kritika_gogolevskogo_perioda", 1856), ("text_0080", "ostrovskij", 1859),
        ("text_0070", "oblomov", 1859), ("text_0090", "ocherk_istorii_poezii", 1858),
        ("text_0060", "sochineniya_belinskogo", 1860)]),
    "annenkov": ("aesthete", "a/annenkow_p_w", [
        ("text_0144", "romany_iz_prostonarodnogo_byta", 1854), ("text_0169", "dvoryanskoe_gnezdo", 1859),
        ("text_0150", "o_mysli", 1855), ("text_0180", "groza", 1860),
        ("text_0167", "delovoj_roman", 1859)]),
    "botkin": ("aesthete", "b/botkin_w_p", [
        ("text_0150", "stihotvoreniya_feta", 1857), ("text_0290", "pisma_ob_ispanii", 1857)]),
}

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
    for sub, (school, author_path, texts) in JOBS.items():
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
            time.sleep(1.2)
    print("\n== объёмы по авторам ==")
    for sub, (school, _, _) in JOBS.items():
        tot = sum(w for s, _, _, w in summary if s == sub)
        print(f"  {sub} ({school}): {tot} слов")


if __name__ == "__main__":
    main()
