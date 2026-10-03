"""Create original synthetic Russian texts for exercising the Stylo workflow.

This demo checks that the pipeline runs. Its invented styles and targets do not
provide an accuracy benchmark or evidence about real authorship.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random


SUBJECTS = (
    "наблюдатель", "путник", "сосед", "мастер", "хозяин", "читатель",
    "прохожий", "слушатель", "смотритель", "товарищ", "помощник", "гость",
)
VERBS = (
    "замечает", "проверяет", "разглядывает", "описывает", "обсуждает", "вспоминает",
    "находит", "сравнивает", "оценивает", "объясняет", "рассматривает", "представляет",
)
ADVERBS = (
    "медленно", "внимательно", "неожиданно", "спокойно", "снова", "осторожно",
    "уверенно", "неторопливо", "молча", "сразу", "изредка", "подробно",
    "наконец", "вскользь", "упорно", "задумчиво",
)
# Nominative and accusative forms keep both full clauses and fragments readable.
THINGS = (
    ("ровная линия", "ровную линию"), ("слабый звук", "слабый звук"),
    ("светлое окно", "светлое окно"), ("узкая полоса", "узкую полосу"),
    ("чёткий след", "чёткий след"), ("старая дверь", "старую дверь"),
    ("маленький круг", "маленький круг"), ("простая деталь", "простую деталь"),
    ("лёгкая тень", "лёгкую тень"), ("короткий шаг", "короткий шаг"),
    ("ровный ритм", "ровный ритм"), ("тихая пауза", "тихую паузу"),
    ("новая мысль", "новую мысль"), ("давний вопрос", "давний вопрос"),
    ("ясный ответ", "ясный ответ"), ("случайный поворот", "случайный поворот"),
    ("чужой жест", "чужой жест"), ("общий порядок", "общий порядок"),
    ("верный путь", "верный путь"), ("незаметный знак", "незаметный знак"),
)
LOCATIONS = (
    "у окна", "возле стены", "на дороге", "за дверью", "рядом с крыльцом",
    "внутри комнаты", "между деревьями", "над столом", "около ограды",
    "перед поворотом", "в начале пути", "возле скамьи",
)
PANEL = ("demo_periodic", "demo_fragmented")
TARGETS = ("target_periodic", "target_fragmented")
NOTICE = "synthetic workflow demo, not an accuracy benchmark"


def _clause(rng: random.Random) -> str:
    return (f"{rng.choice(SUBJECTS)} {rng.choice(ADVERBS)} "
            f"{rng.choice(VERBS)} {rng.choice(THINGS)[1]}")


def _periodic(rng: random.Random) -> str:
    return (f"{rng.choice(LOCATIONS).capitalize()} {_clause(rng)}; "
            f"хотя {_clause(rng)}, {_clause(rng)}, "
            f"потому что {_clause(rng)}. "
            f"Если {_clause(rng)}, то {_clause(rng)}, "
            f"а когда {_clause(rng)}, {_clause(rng)}.")


def _fragmented(rng: random.Random) -> str:
    return (f"{_clause(rng).capitalize()}. "
            f"{rng.choice(LOCATIONS).capitalize()} — {rng.choice(THINGS)[0]}. "
            f"{_clause(rng).capitalize()}? "
            f"{rng.choice(SUBJECTS).capitalize()}: {rng.choice(VERBS)}, "
            f"{rng.choice(VERBS)}, {rng.choice(VERBS)}; {rng.choice(THINGS)[0]}! "
            f"{_clause(rng).capitalize()} (не сразу). "
            f"{rng.choice(THINGS)[0].capitalize()} — и {_clause(rng)}.")


def _text(style: str, seed: int, minimum_words: int = 1400) -> str:
    rng = random.Random(seed)
    paragraph = _periodic if style == "periodic" else _fragmented
    paragraphs = []
    words = 0
    while words < minimum_words:
        value = paragraph(rng)
        paragraphs.append(value)
        words += len(value.split())
    return "\n\n".join(paragraphs) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="New directory for the synthetic corpus and case.yaml")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output already exists; nothing overwritten: {output}")
    output = output.resolve()
    input_root = output / "input"
    for style_index, style in enumerate(("periodic", "fragmented")):
        author_root = input_root / PANEL[style_index]
        author_root.mkdir(parents=True)
        for work_index in range(1, 4):
            (author_root / f"work_{work_index}.txt").write_text(
                _text(style, 1000 + style_index * 100 + work_index),
                encoding="utf-8", newline="\n",
            )
    unknown = input_root / "unknown"
    unknown.mkdir()
    for style_index, style in enumerate(("periodic", "fragmented")):
        (unknown / f"{TARGETS[style_index]}.txt").write_text(
            _text(style, 2000 + style_index), encoding="utf-8", newline="\n",
        )
    config = {
        "deployment": {"candidate_authors": list(PANEL)},
        "paths": {
            "input_raw": str(input_root),
            "input_clean": str(output / "input_clean"),
            "data": str(output / "data"),
            "docs": str(output / "results"),
            "doc_cache": str(output / "data" / "doc_cache"),
        },
        "language": {
            "spacy_model": "ru_core_news_lg", "spacy_model_version": "3.8.0",
            "spacy_fallback": None, "parse_n_process": 1,
        },
        "evaluation": {"training_weighting": "work_balanced", "n_jobs": 1},
    }
    config_path = output / "case.yaml"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"config": str(config_path),
                      "targets": [f"unknown/{name}" for name in TARGETS],
                      "reference_works": 6, "target_works": 2, "notice": NOTICE},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
