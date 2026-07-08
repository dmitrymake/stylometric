"""Атрибуция спорного текста (каталог unknown) ансамблем LR + настоящая Delta.

Выдаёт топ-K кандидатов с калиброванными/усреднёнными оценками, уверенность и margin.
Честно сообщает неопределённость: при близких оценках margin мал → вывод осторожный.
"""
from __future__ import annotations

import datetime
import json
import logging
import pathlib
from typing import List

import joblib
import numpy as np
from scipy.stats import trim_mean

from ..config import load_config
from ..lang import display_name

log = logging.getLogger("stylo.pipeline.predict")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


def run(cfg=None, unknown_dir: str | None = None) -> dict:
    cfg = cfg or load_config()
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    docs_dir = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    docs_dir.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(data / "model.pkl")
    delta = joblib.load(data / "delta.pkl")
    authors: List[str] = json.loads((data / "authors.json").read_text(encoding="utf-8"))

    unk_root = pathlib.Path(unknown_dir) if unknown_dir else (data / "frags_unknown")
    texts = []
    for fp in sorted(unk_root.rglob("*.txt")):
        try:
            t = fp.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if t:
            texts.append(t)
    if not texts:
        raise RuntimeError(f"Нет фрагментов unknown в {unk_root}")
    log.info("Unknown фрагментов: %d", len(texts))

    lr_probs = pipe.predict_proba(texts)
    lr_mean = lr_probs.mean(axis=0)
    # выровнять на полный набор авторов (pipe.classes_ обычно == все)
    lr_full = np.zeros(len(authors))
    for j, c in enumerate(pipe.classes_):
        lr_full[int(c)] = lr_mean[j]

    d_dist = delta.distances(texts)            # (n, n_classes)
    d_mean = trim_mean(d_dist, 0.1, axis=0)
    delta_full = np.zeros(len(authors))
    delta_score = _softmax(-d_mean)
    for j, c in enumerate(delta.classes_):
        delta_full[int(c)] = delta_score[j]

    ens = 0.6 * lr_full + 0.4 * delta_full
    order = np.argsort(ens)[::-1]
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)
    margin = float(ens[order[0]] - ens[order[1]]) if len(order) > 1 else 0.0

    lines: List[str] = []
    lines.append("=== Авторская атрибуция (ансамбль LR + Burrows Delta) ===")
    lines.append(f"Дата: {datetime.datetime.now():%d.%m.%Y %H:%M}")
    lines.append(f"Фрагментов: {len(texts)}")
    lines.append("")
    lines.append(f"Топ-{top_k} кандидатов (оценка ансамбля):")
    for i in order[:top_k]:
        lines.append(f"  {display_name(authors[int(i)]):24} ens={ens[int(i)]:.4f} "
                     f"(LR={lr_full[int(i)]:.4f}, Delta={delta_full[int(i)]:.4f})")
    lines.append("")
    lines.append(f"Победитель: {display_name(authors[int(order[0])])}")
    lines.append(f"Margin над 2-м местом: {margin:.4f}")
    if margin < 0.05:
        lines.append("⚠ Низкий margin — вывод НЕнадёжен (кандидаты близки).")
    lines.append("")
    lines.append("Отдельные методы:")
    lines.append(f"  LR    → {display_name(authors[int(np.argmax(lr_full))])}")
    lines.append(f"  Delta → {display_name(authors[int(np.argmax(delta_full))])}")

    report = "\n".join(lines)
    (docs_dir / "prediction.txt").write_text(report, encoding="utf-8")
    print(report)
    return {"winner": authors[int(order[0])], "margin": margin,
            "ensemble": {authors[i]: float(ens[i]) for i in range(len(authors))}}
