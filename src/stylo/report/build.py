"""Собрать docs/index.html из текстовых артефактов пайплайна.

Намеренно простой и честный отчёт: показывает то, что реально измерено
(sweep-таблица с CI, валидация корпуса, атрибуция), без приукрашивания.
"""
from __future__ import annotations

import datetime
import html
import logging
import pathlib
from typing import List, Tuple

log = logging.getLogger("stylo.report")

_SECTIONS: List[Tuple[str, str]] = [
    ("Состав и качество корпуса", "corpus_validation.txt"),
    ("Что работает: ablation-sweep (LOBO, с доверительными интервалами)", "sweep_table.txt"),
    ("Атрибуция спорного текста", "prediction.txt"),
]


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else "(нет данных)"


def run(cfg=None) -> None:
    from ..config import load_config
    cfg = cfg or load_config()
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    docs.mkdir(parents=True, exist_ok=True)

    model = cfg.get_path("language.spacy_model", "?")
    mver = cfg.get_path("language.spacy_model_version", "?")
    parts: List[str] = []
    parts.append(f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Стилометрия авторства — отчёт</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
 h2 {{ margin-top: 2rem; color: #234; }}
 pre {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
        padding: 1rem; overflow-x: auto; font-size: 13px; line-height: 1.4; }}
 .meta {{ color: #666; font-size: 13px; }}
</style></head><body>""")
    parts.append("<h1>Стилометрия авторства русской прозы</h1>")
    parts.append(f'<p class="meta">Сгенерировано: {datetime.datetime.now():%d.%m.%Y %H:%M} · '
                 f'модель spaCy: {html.escape(str(model))} v{html.escape(str(mver))} · '
                 f'leakage-free LOBO, метрики с 95% bootstrap-CI по книгам</p>')
    for title, fname in _SECTIONS:
        body = html.escape(_read(docs / fname))
        parts.append(f"<h2>{html.escape(title)}</h2><pre>{body}</pre>")
    imgs = sorted(docs.glob("*.png"))
    if imgs:
        parts.append("<h2>Графики</h2>")
        for im in imgs:
            parts.append(f'<p><img src="{im.name}" style="max-width:100%"></p>')
    parts.append("</body></html>")

    (docs / "index.html").write_text("\n".join(parts), encoding="utf-8")
    log.info("Отчёт: %s", docs / "index.html")
    print(f"Отчёт собран: {docs / 'index.html'}")
