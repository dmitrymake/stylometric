"""
Анализ маркеров стиля: цвета и характерные прилагательные.
Использует Craig's Zeta (segment-based).
Это делает анализ устойчивым к длине текстов и "взрывным" словам.
"""

import os
import spacy
import math
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scripts.nlp import get_stylometry_nlp
from meta.meta import BASE_LANG_MODEL, ERR_LANG_MODEL

# Цвета
COLORS = [
    "белый", "черный", "чёрный", "красный", "синий", "зеленый", "зелёный", "желтый", "жёлтый",
    "золотой", "серебряный", "серебристый", "серый", "голубой", "розовый", "фиолетовый",
    "лиловый", "багровый", "алый", "рубиновый", "янтарный", "бирюзовый", "рыжий",
    "кремовый", "шоколадный", "сиреневый", "лазурный", "оранжевый", "бледный"
]

CHUNK_SIZE = 1000  # Размер сегмента для Zeta-анализа


def make_segments(text, size=CHUNK_SIZE):
    """Разбивает текст на сегменты слов для подсчета DF."""
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


def load_and_process_zeta(root_dir: str, nlp):
    """
    Сканирует папку и создает сегменты для каждого автора.
    Возвращает: {author: {'segments': [seg1, seg2...], 'df_counts': Counter}}
    """
    data = {}
    root = Path(root_dir)
    print(f"Сканирование {root} для Zeta-анализа...")

    for author_dir in sorted(root.iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name

        full_text_parts = []
        for f in author_dir.rglob("*.txt"):
            if f.stat().st_size > 0:
                full_text_parts.append(f.read_text(encoding="utf-8"))

        combined_text = " ".join(full_text_parts)

        segments = make_segments(combined_text)

        df_counter = Counter()

        # Ограничение длины для nlp (5M) применяется глобально в get_global_nlp()
        docs = nlp.pipe(segments, batch_size=20)

        for doc in docs:
            adjs_in_seg = set(t.lemma_.lower() for t in doc if t.pos_ == "ADJ")
            df_counter.update(adjs_in_seg)

        data[author] = {
            'num_segments': len(segments),
            'df_counts': df_counter
        }
        print(f"  -> {author:<15}: {len(segments)} сегментов")

    return data


def calculate_zeta(target_data, comparison_data_list):
    scores = []

    target_N = target_data['num_segments']
    if target_N == 0:
        return []

    comp_df = Counter()
    comp_N = 0
    for d in comparison_data_list:
        comp_df.update(d['df_counts'])
        comp_N += d['num_segments']

    if comp_N == 0:
        return []

    for word, target_count in target_data['df_counts'].items():
        p_target = target_count / target_N

        # Сглаживание Лапласа для p_comp, чтобы не делить на 0:
        # если слова нет у конкурентов, считаем, что оно встретилось бы 0.5 раза (штраф)
        comp_count = comp_df[word]
        p_comp = (comp_count + 0.5) / (comp_N + 1)

        # Фильтр стабильности: слово должно встретиться хотя бы в нескольких
        # сегментах, чтобы не ловить опечатки и имена
        if target_count < 2:
            continue

        zeta = p_target - p_comp

        ratio = p_target / p_comp

        # Отбираем слова, которые либо очень популярны у автора и средни у других
        # (сильный маркер по Zeta), либо умеренно популярны у автора и почти
        # отсутствуют у других (эксклюзивный маркер по Ratio)
        is_strong_marker = (zeta > 0.1)
        is_exclusive_marker = (zeta > 0.02 and ratio > 4.0)

        if is_strong_marker or is_exclusive_marker:
            scores.append({
                'word': word,
                'score': zeta,
                'p_target': p_target,
                'p_comp': p_comp,
                'ratio': ratio,
                'count_target': target_count
            })

    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores


def main():
    print("=== АНАЛИЗ ЛЕКСИЧЕСКИХ МАРКЕРОВ (ZETA - SEGMENT BASED) ===")

    nlp_pipe = get_stylometry_nlp()

    data = load_and_process_zeta(
        "input_clean", nlp_pipe)

    if "unknown" not in data:
        print("Ошибка: нет папки unknown")
        return

    unknown_data = data.pop("unknown")
    authors = sorted(data.keys())

    report_lines = []
    author_scores = []

    report_lines.append("\n=== ZETA SCORE ANALYSIS ===")
    report_lines.append(
        "Метод ищет слова, которые стабильно встречаются в сегментах автора,")
    report_lines.append("но отсутствуют в сегментах конкурентов.\n")

    for cand in authors:
        others = [data[a] for a in authors if a != cand]
        markers = calculate_zeta(data[cand], others)

        # Zeta Intersection: сумма Zeta-score тех маркеров, что есть в unknown
        intersection_score = 0
        found_markers = []

        for m in markers:
            word = m['word']
            if unknown_data['df_counts'][word] > 0:
                # Вес маркера = его сила (Zeta) * плотность в неизвестном тексте,
                # нормированная на число сегментов unknown, чтобы длина книги не влияла
                density_in_unk = unknown_data['df_counts'][word] / \
                    unknown_data['num_segments']
                weight = m['score'] * (1.0 + density_in_unk)

                intersection_score += weight
                found_markers.append({
                    **m,
                    'density_unk': density_in_unk
                })

        author_scores.append((cand, intersection_score, found_markers))

    author_scores.sort(key=lambda x: x[1], reverse=True)

    for auth, score, markers in author_scores:
        report_lines.append(f"\nАВТОР: {auth.upper()}")
        report_lines.append(f"Zeta Intersection Score: {score:.4f}")
        report_lines.append("-" * 75)

        if not markers:
            report_lines.append("  (Маркеров не найдено)")
        else:
            report_lines.append(
                f"  {'МАРКЕР':<15} | {'ZETA':<6} | {'P(Auth)':<7} | {'P(Rest)':<7} | {'Density(Unk)'}")
            top_m = sorted(
                markers, key=lambda x: x['score'], reverse=True)[:15]
            for m in top_m:
                report_lines.append(
                    f"  {m['word']:<15} | {m['score']:<6.2f} | {m['p_target']:<7.2f} | {m['p_comp']:<7.2f} | {m['density_unk']:.2%}")

    report_lines.append("\n=== ИТОГ ПО МАРКЕРАМ ===")
    for auth, score, _ in author_scores:
        bar = "█" * int(score * 2)
        report_lines.append(f"{auth:<15} : {score:<6.2f} {bar}")

    os.makedirs("docs", exist_ok=True)
    with open("docs/markers_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print("Отчет сохранен: docs/markers_report.txt")
