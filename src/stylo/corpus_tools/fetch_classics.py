"""Докачка public-domain классиков (расширение корпуса).

Источник — русская Викитека (ru.wikisource.org) через Action API (wikitext).
СТРОГИЙ whitelist: качаем только авторов из PUBLIC_DOMAIN (умерли достаточно давно).
Современные авторы под копирайтом (Сорокин, Акунин, Пелевин, Довлатов…) — НЕ качаем.

Манифест: configs/classics.yaml — список {author, title, source} (source = заголовок
страницы Викитеки). Скачанное кладётся в input/<author>/<slug>.txt (как сырой текст),
далее проходит обычный clean. Качество проверяйте `stylo validate-corpus`.
"""
from __future__ import annotations

import importlib.resources
import logging
import pathlib
import re
import time
from typing import List, Optional

import requests
import yaml

log = logging.getLogger("stylo.corpus_tools.fetch")

API = "https://ru.wikisource.org/w/api.php"
HEADERS = {"User-Agent": "stylo-research/0.4 (authorship attribution; academic use)"}

# Авторы, умершие > 70 лет назад. ТОЛЬКО их можно качать/хранить — и только для
# ЛОКАЛЬНОЙ валидации бенчмарка: скачанные тексты не редистрибутируются.
# ВНИМАНИЕ: критерий «смерть+70» не покрывает два продления охраны по ГК РФ:
#   * ст. 1281 п. 5 — для реабилитированных срок считается от даты реабилитации
#     (gumilev — охрана в РФ до ~2057; pilnyak — до 01.01.2027);
#   * ст. 1337-1340 — «право публикатора» на современные первоиздания.
PUBLIC_DOMAIN = {
    "dostoevsky", "gogol", "chehov", "rozanov", "tolstoy", "turgenev",
    "kuprin", "andreev", "leskov", "saltykov", "garshin", "korolenko",
    "bunin",  # умер 1953 — в РФ public domain
    # разнообразие классов:
    "pushkin", "lermontov", "goncharov", "pisemsky", "grigorovich",
    # спорное/коллективное/мистификации авторства (PD):
    "prutkov",        # Козьма Прутков — коллективный псевдоним (А.К. Толстой + Жемчужниковы)
    "ilf-petrov",     # «12 стульев» — спор Ильф-Петров vs Булгаков (Ильф †1937, Петров †1942 — PD)
    # донские кандидаты в «литературные негры» «Тихого Дона» (все PD):
    "serafimovich",   # Александр Серафимович †1949 — донской писатель, редактор/покровитель Шолохова
    "sevsky",         # Виктор Севский (Вен. Краснушкин) †1920 — донской журналист/писатель
    "kumov",          # Роман Кумов †1919 — донской писатель, круг Крюкова
    "rodionov",       # Иван Родионов †1940 — донской писатель
    # жанровые корпуса для cross-author genre-теста (DSP топик-инвариантность)
    # СЕЛЬСКАЯ/народническая проза (разные авторы — чтобы «жанр» отделить от «автора»):
    "uspensky",       # Глеб Успенский †1902 — народническая деревенская проза
    "reshetnikov",    # Фёдор Решетников †1871 — «Подлиповцы», крестьянская проза
    "zlatovratsky",   # Николай Златовратский †1911 — деревенская проза
    "zasodimsky",     # Павел Засодимский †1912 — народническая проза
    "naumov",         # Николай Наумов †1901 — сибирская крестьянская проза
    # ВОЕННАЯ проза (разные авторы):
    "veresaev",       # Викентий Вересаев †1945 — «На войне» (русско-японская)
    "novikov_priboy", # Алексей Новиков-Прибой †1944 — «Цусима», морская война
    "furmanov",       # Дмитрий Фурманов †1926 — «Чапаев», гражданская война
    # казусы авторства: советские/русские мистификации и внутриавторская вариативность
    "dmitrieva",      # Елизавета Дмитриева (Васильева), «Черубина де Габриак» †1928 — within-author маска
    "voloshin",       # Максимилиан Волошин †1932 — режиссёр мистификации Черубины
    "tolstoy_an",     # Алексей Н. Толстой †1945 (life+74=2019 PD) — НЕ путать с tolstoy=Лев Толстой
    "bryusov",        # Валерий Брюсов †1924 — «Стихи Нелли» (мужской поэт в женской маске)
    "pilnyak",        # Борис Пильняк †1938 (расстрелян, реабилит. 1956 → охрана РФ до 01.01.2027!) — контроль
    "grin",           # Александр Грин (Гриневский) †1932 — псевдоним, «европеизированный» стиль (контроль)
    "gumilev",        # Николай Гумилёв †1921 (реабилит. 1991 → охрана РФ до ~2057!) — маргинальный кандидат ТД
    # Фадеев †1956, Лавренёв †1959, Шолохов †1984, Пантелеев †1987 НЕ PD — вне корпуса.
}

# Реабилитационное продление охраны (ст. 1281 п. 5 ГК РФ): срок считается от
# даты реабилитации, а не смерти. Эти авторы остаются в исследовательском
# корпусе, но НЕ входят в публикуемый/редистрибутируемый срез.
REHAB_RESTRICTED = {
    "gumilev",   # реабилитирован 1991 → охрана в РФ до ~2057
    "pilnyak",   # реабилитирован 1956 → охрана в РФ до 01.01.2027
}

# Юридически чистый публикуемый срез: публичные артефакты (validation_pd.json,
# RuAA-Bench) строятся ТОЛЬКО из него.
PUBLIC_DOMAIN_CLEAR = PUBLIC_DOMAIN - REHAB_RESTRICTED

# очистка вики-разметки
_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I)
_RE_REF_SELF = re.compile(r"<ref[^>]*/>", re.I)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_RE_HEADER = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_RE_BOLDIT = re.compile(r"'{2,5}")


def _strip_templates(text: str) -> str:
    """Удалить {{...}} с учётом вложенности."""
    out = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i:i + 2] == "{{":
            depth += 1
            i += 2
        elif text[i:i + 2] == "}}":
            if depth > 0:
                depth -= 1
            i += 2
        elif depth == 0:
            out.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(out)


def _strip_links(text: str) -> str:
    # [[a|b]] -> b ; [[a]] -> a ; убираем File/Image
    text = re.sub(r"\[\[(?:File|Image|Файл|Изображение):[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    return text


_RE_MAGIC = re.compile(r"__[A-ZА-Я]+__")
_RE_CATLINK = re.compile(r"\[\[(?:Категория|Category|Файл|File|Image|Изображение|[a-z]{2,3}):[^\]]*\]\]", re.I)


def wikitext_to_plain(wt: str) -> str:
    wt = _RE_COMMENT.sub("", wt)
    wt = _RE_REF.sub("", wt)
    wt = _RE_REF_SELF.sub("", wt)
    wt = _RE_TABLE.sub("", wt)
    wt = _RE_MAGIC.sub("", wt)        # __NOEDITSECTION__/__NOTOC__ и пр.
    wt = _RE_CATLINK.sub("", wt)      # [[Категория:…]] и интервики — целиком, до _strip_links
    wt = _strip_templates(wt)
    wt = _strip_links(wt)
    wt = _RE_TAG.sub("", wt)
    wt = _RE_HEADER.sub("", wt)
    wt = _RE_BOLDIT.sub("", wt)
    wt = re.sub(r"^[*#:;]+\s*", "", wt, flags=re.M)  # списки/отступы
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def fetch_wikitext(title: str, timeout: int = 30, retries: int = 4) -> Optional[str]:
    params = {
        "action": "query", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "titles": title, "format": "json", "formatversion": "2",
        "maxlag": "5",
    }
    backoff = 3.0
    for attempt in range(retries):
        r = requests.get(API, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 429 or (r.status_code == 200 and "maxlag" in r.text[:200]):
            wait = backoff * (2 ** attempt)
            log.warning("rate-limit (%s), пауза %.0fs…", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        try:
            return pages[0]["revisions"][0]["slots"]["main"]["content"]
        except (KeyError, IndexError):
            return None
    log.error("не удалось скачать после %d попыток: %s", retries, title)
    return None


def fetch_subpages(title: str, timeout: int = 30, retries: int = 4) -> List[str]:
    """Собрать wikitext всех подстраниц `title` (главы романа и т.п.).

    Крупные произведения на Викитеке разбиты на подстраницы вида
    "Заголовок/Часть первая/Глава I" — главная страница это лишь оглавление.
    Через generator=allpages с префиксом получаем содержимое всех подстраниц.
    """
    contents: List[str] = []
    cont: dict = {}
    backoff = 3.0
    while True:
        params = {
            "action": "query", "generator": "allpages",
            "gapprefix": title + "/", "gapnamespace": "0", "gaplimit": "50",
            "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "format": "json", "formatversion": "2", "maxlag": "5",
        }
        params.update(cont)
        ok = False
        for attempt in range(retries):
            r = requests.get(API, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff * (2 ** attempt))
                continue
            r.raise_for_status()
            data = r.json()
            ok = True
            break
        if not ok:
            break
        for p in data.get("query", {}).get("pages", []):
            try:
                contents.append(p["revisions"][0]["slots"]["main"]["content"])
            except (KeyError, IndexError):
                pass
        if "continue" in data:
            cont = data["continue"]
            time.sleep(1.0)
        else:
            break
    return contents


def _dedup_doubled(text: str) -> str:
    """Срезать целостное задвоение: если начало текста дословно повторяется ближе к
    середине (некоторые страницы Викитеки содержат текст дважды — старая/новая
    орфография, дубль-трансклюзия), оставляем первую копию. Работаем по списку слов
    (без учёта пробелов/переносов)."""
    w = text.split()
    n = len(w)
    if n < 200:
        return text
    key = w[:40]
    for i in range(max(40, n // 4), (3 * n) // 4):   # ищем повтор в окне 25–75%
        if w[i:i + 40] == key:
            return " ".join(w[:i])
    return text


def fetch_work(title: str) -> Optional[str]:
    """Полный текст произведения.

    Берём МАКСИМУМ из (главная страница) и (склейка подстраниц-глав), а не сумму —
    иначе у произведений, где полный текст лежит и на главной, и на подстраницах,
    получается дублирование внутри одной книги. Плюс срезаем целостное задвоение.
    """
    main = fetch_wikitext(title)
    main_plain = wikitext_to_plain(main) if main else ""
    subs = fetch_subpages(title)
    subs_plain = "\n\n".join(p for p in (wikitext_to_plain(s) for s in subs) if p)

    chosen = subs_plain if len(subs_plain.split()) > len(main_plain.split()) else main_plain
    chosen = _dedup_doubled(chosen) if chosen else chosen
    return chosen or None


def _slug(title: str) -> str:
    s = re.sub(r"[^\w]+", "_", title.lower(), flags=re.U).strip("_")
    return s[:60] or "work"


def load_classics_manifest(manifest: str | pathlib.Path | None = None) -> list[dict]:
    """Load the packaged default manifest or an explicit filesystem override."""
    if manifest is None:
        source = importlib.resources.files("stylo.resources").joinpath("classics.yaml")
        with source.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or []
    path = pathlib.Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"classics manifest not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def run(cfg=None, manifest: str | None = None, min_words: int = 1500,
        delay: float = 1.0) -> dict:
    from ..config import load_config
    cfg = cfg or load_config()
    entries = load_classics_manifest(manifest)
    input_root = pathlib.Path(cfg.get_path("paths.input_raw", "input"))
    stats = {"downloaded": 0, "skipped": 0, "refused": 0, "failed": 0}

    for e in entries:
        author = e["author"]
        title = e["title"]
        source = e.get("source", title)
        if author not in PUBLIC_DOMAIN:
            log.warning("ОТКАЗ (копирайт/не в whitelist): %s — %s", author, title)
            stats["refused"] += 1
            continue
        out = input_root / author / f"{_slug(title)}.txt"
        if out.exists() and len(out.read_text('utf-8', errors='ignore').split()) >= min_words:
            log.info("уже есть: %s/%s", author, title)
            stats["skipped"] += 1
            continue
        try:
            plain = fetch_work(source)
            if not plain:
                log.warning("не найдено в Викитеке: %s", source)
                stats["failed"] += 1
                continue
            wc = len(plain.split())
            if wc < min_words:
                log.warning("слишком коротко (%d слов): %s — %s", wc, author, title)
                stats["failed"] += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(plain, encoding="utf-8")
            log.info("OK %s/%s: %d слов", author, title, wc)
            stats["downloaded"] += 1
        except Exception as exc:
            log.error("ошибка %s — %s: %s", author, title, exc)
            stats["failed"] += 1
        time.sleep(delay)

    log.info("Докачка завершена: %s", stats)
    return stats
