"""Позитив-контроль кейса sovremennik — две оси.

(1) РАЗДЕЛИМАЯ ось (валидация инструмента): радикалы (Чернышевский, Добролюбов) ↔ эстетики (Дружинин,
    Анненков, Боткин). Разные школы — должны делиться. Если делятся на служебных словах — инструмент
    различает руки в критике одного журнала/эпохи.
(2) НЕРАЗДЕЛИМАЯ ось (честный негатив): Чернышевский ↔ Добролюбов (учитель↔ученик). Ожидаемо не
    делятся — предел метода на сросшихся руках.

Метрика — leave-one-WORK-out (устойчиво к корреляции кусков) + перестановка ярлыков работ +
уравнивание объёма. Признаки: служебные слова (leak-free) и char-3gram (сравнение).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
from stylo.lang import function_words  # noqa: E402
from _gate_metrics import (  # noqa: E402
    both_metrics,
    leave_one_work_out,
    work_balanced_centroid,
    work_permutation_p,
)

CASE = ROOT / "input_cases" / "sovremennik"
HISTORICAL_OUT = ROOT / "docs" / "cases" / "sovremennik.json"
DEFAULT_OUT = (
    ROOT / "docs" / "cases" / "work_balanced_audit" / "custom"
    / "sovremennik.work_balanced.json"
)
SCHOOL = {"chernyshevsky": "radical", "dobrolyubov": "radical",
          "druzhinin": "aesthete", "annenkov": "aesthete", "botkin": "aesthete"}
WORD = r"[а-яёА-ЯЁ]+"
WIN = 600


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def load():
    """[(author, work, chunk_text)] по всем авторам."""
    out = []
    for author in SCHOOL:
        for f in sorted((CASE / author).glob("*.txt")):
            w = f.read_text("utf-8", "ignore").split()
            for i in range(0, len(w), WIN):
                piece = " ".join(w[i:i + WIN])
                if len(piece.split()) >= WIN // 2:
                    out.append((author, f.stem, piece))
    return out


def build_vec(all_texts, use_char3):
    fw = sorted(function_words("ru"))
    fwi = {w: i for i, w in enumerate(fw)}
    top3, t3i = [], {}
    if use_char3:
        grams = {}
        for text in all_texts:
            flat = re.sub(r"\s+", " ", text.lower())
            for i in range(max(len(flat) - 2, 0)):
                grams[flat[i:i + 3]] = grams.get(flat[i:i + 3], 0) + 1
        top3 = [g for g, _ in sorted(grams.items(), key=lambda kv: kv[1], reverse=True)[:800]]
        t3i = {g: i for i, g in enumerate(top3)}

    def vec(text):
        toks = re.findall(WORD, text.lower())
        fwv = np.zeros(len(fw))
        for t in toks:
            j = fwi.get(t)
            if j is not None:
                fwv[j] += 1
        fwv /= len(toks) or 1
        fwv = _unit(fwv)
        if not use_char3:
            return fwv
        flat = re.sub(r"\s+", " ", text.lower())
        c3 = np.zeros(len(top3))
        for i in range(max(len(flat) - 2, 0)):
            j = t3i.get(flat[i:i + 3])
            if j is not None:
                c3[j] += 1
        c3 /= max(len(flat) - 2, 1)
        return np.concatenate([fwv, _unit(c3)])

    return vec


def _centroid(vs):
    return _unit(np.mean(vs, axis=0))


def analyze(data, label_of, rng):
    """work-level и chunk-weighted раздельно (общий модуль) + within-work уравнивание + перестановка работ.

    data: [(author, work, vec)]. Класс куска = label_of(author) (школа для школьной оси, сам автор для
    пары). Для метрик данные переразмечаются в классы, ключ работы = (author, work) — один текст один голос.
    """
    labels = sorted({label_of(a) for a, _, _ in data})
    cdata = [(label_of(a), (a, w), v) for a, w, v in data]  # первый элемент = КЛАСС, работа уникальна

    # ОБЕ метрики раздельно: work_macro_recall (один текст = один голос) и chunk_weighted (диагностика).
    wcp, confusion, _w = leave_one_work_out(cdata, labels)
    M = both_metrics(wcp, labels)
    macro_recall = M["work_macro_recall"]            # порог 0.80 применяется К НЕЙ
    chunk_weighted = M["chunk_weighted_recall"]      # НЕ work-level: коррелированные куски = много голосов
    per_recall = M["work_recall"]                    # work-level recall по классам

    # уравнивание объёма: равное число кусков на метку (within-work диагностика, в решении не участвует)
    bylabel = {l: [v for a, _, v in data if label_of(a) == l] for l in labels}
    nmin = min(len(bylabel[l]) for l in labels)
    accs = []
    for _ in range(200):
        sub = {l: [bylabel[l][i] for i in rng.choice(len(bylabel[l]), nmin, replace=False)] for l in labels}
        ok = tot = 0
        for tl in labels:
            for i in range(nmin):
                cents = {m: (_centroid([x for j, x in enumerate(sub[m]) if j != i]) if m == tl
                            else _centroid(sub[m])) for m in labels}
                pred = max(labels, key=lambda m: float(np.dot(_unit(sub[tl][i]), cents[m])))
                ok += pred == tl
                tot += 1
        accs.append(ok / tot)
    balanced = round(float(np.mean(accs)), 4)

    # перестановка ярлыков работ на work-level метрике; exact-перечисление при малом числе работ
    perm_p, perm_method, perm_floor = work_permutation_p(cdata, lambda l: l, labels)

    cents_full = {
        label: work_balanced_centroid(
            (work, vector) for row_label, work, vector in cdata if row_label == label
        )
        for label in labels
    }
    cos = (round(float(np.dot(cents_full[labels[0]], cents_full[labels[1]])), 4)
           if len(labels) == 2 else None)
    return {"per_recall": per_recall,                          # work-level (один текст = один голос)
            "macro_recall": macro_recall,                      # = work_macro_recall (порог 0.80 к ней)
            "chunk_weighted_recall": chunk_weighted,           # диагностика: куски как голоса (не work-level)
            "chunk_recall": M["chunk_recall"],
            "within_work_chunk_accuracy": balanced,
            "perm_p": perm_p,
            "permutation_method": perm_method,                 # exact_N или random_N
            "permutation_exact_floor": perm_floor,             # минимально достижимое точное p = 1/C(W,n1)
            "centroid_cos": cos,
            "train_centroid_weighting": "equal_work_direction_after_within_work_chunk_mean_l2",
            "confusion": confusion}


def run_axis(label_of, subset_authors, use_char3):
    rng = np.random.default_rng(20260630)
    raw = [(a, w, t) for a, w, t in load() if a in subset_authors]
    vec = build_vec([t for _, _, t in raw], use_char3)
    data = [(a, w, vec(t)) for a, w, t in raw]
    return analyze(data, label_of, rng)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--overwrite-historical",
        action="store_true",
        help="allow --out to replace the preserved legacy report",
    )
    args = parser.parse_args(argv)
    if (
        args.out.expanduser().resolve() == HISTORICAL_OUT.resolve()
        and not args.overwrite_historical
    ):
        parser.error(
            "refusing to overwrite the historical report without "
            "--overwrite-historical"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out.expanduser()
    try:
        shown = out.resolve().relative_to(ROOT)
    except ValueError:
        shown = out.resolve()
    radicals = {"chernyshevsky", "dobrolyubov"}
    aesthetes = {"druzhinin", "annenkov", "botkin"}
    alla = radicals | aesthetes
    school_of = lambda a: SCHOOL[a]
    author_of = lambda a: a

    axis_school = {f: run_axis(school_of, alla, f == "c3") for f in ("fw", "c3")}
    axis_pair = {f: run_axis(author_of, radicals, f == "c3") for f in ("fw", "c3")}
    # 5-way per-автор recall (Боткин 1 работа — в work-out не тестируется, попадёт с total<кусков)
    five_way = {f: run_axis(author_of, alla, f == "c3") for f in ("fw", "c3")}

    # «Разделяет» = work-level macro >=0.80 (один текст = один голос) И значимость перестановки ярлыков
    # работ (perm_p<=0.05). На служебных словах (leak-free). Школьную ось и пару судим отдельно; char3 —
    # для сравнения «рука vs содержание». within_work_chunk_accuracy (within-work диагностика, утечка)
    # в решении не участвует; chunk_weighted_recall ниже (длинные работы весят больше) и тоже только диагностика.
    def sep(r):
        return r["macro_recall"] >= 0.80 and r["perm_p"] <= 0.05
    school_separates = bool(sep(axis_school["fw"]))
    pair_separates = bool(sep(axis_pair["fw"]))
    pair_c3_separates = bool(sep(axis_pair["c3"]))

    report = {
        "case": "sovremennik",
        "stage": "feasibility_gate",
        "title": "Безымянная критика «Современника» (1854-1862): две оси — валидация и честный негатив",
        "candidates": {"radical": ["Чернышевский", "Добролюбов"],
                       "aesthete": ["Дружинин", "Анненков", "Боткин"]},
        "feature": ("формальный признак — служебные слова (предлоги, союзы, частицы; тему не выдают, словарь "
                    "не учится = leak-free). char-3gram — зависимая от данных диагностика (словарь учится на "
                    "текстах, ловит тему), в векторе идёт вместе со служебными словами; не формальное "
                    "доказательство. within_work_chunk_accuracy — within-work диагностика (утечка), в решении "
                    "не используется; значимость — по перестановке ярлыков работ."),
        "note": ("Две оси: (1) валидация — радикалы↔эстетики (разные школы, должны делиться); (2) проверка "
                 "предела — Чернышевский↔Добролюбов (учитель↔ученик). Две РАЗНЫЕ метрики: work_macro_recall "
                 "(один удержанный текст = один голос, большинство его кусков) — к ней применяется порог 0.80; "
                 "chunk_weighted_recall (каждый кусок — голос) НЕ work-level (коррелированные куски = много "
                 "голосов) и приводится как диагностика. Обучение — leave-one-WORK-out; перестановка ярлыков на "
                 "уровне работ (точное перечисление при малом числе работ, иначе случайно) + уравнивание объёма."),
        "axis_school_radical_vs_aesthete": axis_school,
        "axis_pair_chernyshevsky_vs_dobrolyubov": axis_pair,
        "five_way_author_recall": {f: five_way[f]["per_recall"] for f in ("fw", "c3")},
        "school_axis_separates_on_function_words": school_separates,
        "pair_separates_on_function_words": pair_separates,
        "verdict": _verdict(school_separates, pair_separates, pair_c3_separates, axis_school, axis_pair),
        "caveat": ("Якоря — подписанная литкритика 1854-1862 (исключены философия Чернышевского, травелоги, "
                   "рецензии чужих книг). Боткин тонкий (1 работа, ~14k) — в work-out не самотестируется, но "
                   "входит в эстетиков. Большинство атрибуций «Современника» Боград закрыл по гонорарным "
                   "ведомостям, не по стилю — значимая доля прогона будет калибровкой."),
        "next": ["Если школьная ось делится, а пара Чернышевский↔Добролюбов — нет: это и есть два честных "
                 "результата (валидация инструмента + предел на учителе↔ученике).",
                 "Применять метод к открытым dubia «Современника» ТОЛЬКО на разделимой оси; цели на оси "
                 "учитель↔ученик помечать как неатрибутируемые.",
                 "Короткие рецензии (<1500 слов) поодиночке не атрибутировать."],
        "data_status": ("Все тексты — общественное достояние (Чернышевский †1889, Добролюбов †1861, Дружинин "
                        "†1864, Анненков †1887, Боткин †1869). Подписанная литературная критика 1854-1862 — "
                        "чистый текст с az.lib.ru. Сырьё пишется в gitignored input_cases/sovremennik/; в git "
                        "— только скрипт добычи (scripts/fetch_sovremennik.py) и этот JSON."),
        "sources": [
            {"cite": "Н. Г. Чернышевский, литкритика 1855-1861 («Пушкин», «Гоголь» и др.) — radical",
             "url": "http://az.lib.ru/c/chernyshewskij_n_g/"},
            {"cite": "Н. А. Добролюбов, литкритика 1856-1860 («Тёмное царство», «Луч света» и др.) — radical",
             "url": "http://az.lib.ru/d/dobroljubow_n_a/"},
            {"cite": "А. В. Дружинин, литкритика 1856-1860 («Обломов», «Островский» и др.) — aesthete",
             "url": "http://az.lib.ru/d/druzhinin_a_w/"},
            {"cite": "П. В. Анненков, литкритика 1854-1860 («Дворянское гнездо», «Гроза» и др.) — aesthete",
             "url": "http://az.lib.ru/a/annenkow_p_w/"},
            {"cite": "В. П. Боткин, литкритика 1857 («Стихотворения Фета», «Письма об Испании») — aesthete",
             "url": "http://az.lib.ru/b/botkin_w_p/"},
            {"cite": "В. Э. Боград. «Журнал „Современник“ 1847-1866. Указатель содержания» (атрибуции по "
                     "гонорарным ведомостям) — опорные точки шкалы атрибуций, значимая часть прогона — "
                     "калибровка", "url": "https://archive.org/"},
        ],
        "status": "exploratory_adversarial_rerun",
        "lineage": {
            "historical_report": str(HISTORICAL_OUT.relative_to(ROOT)),
            "historical_report_status": "superseded_for_scientific_interpretation",
        },
        "analysis_command": (
            "PYTHONPATH=src python3 scripts/run_sovremennik_gate.py "
            f"--out {shown}"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_strict(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {shown}")
    print("── ОСЬ ШКОЛ (валидация) ──")
    for f in ("fw", "c3"):
        r = axis_school[f]
        print(f"  [{f}] {r['per_recall']} macro {r['macro_recall']} balanced {r['within_work_chunk_accuracy']} "
              f"perm_p {r['perm_p']} cos {r['centroid_cos']}")
    print("── ПАРА Чернышевский↔Добролюбов (честный негатив) ──")
    for f in ("fw", "c3"):
        r = axis_pair[f]
        print(f"  [{f}] {r['per_recall']} macro {r['macro_recall']} balanced {r['within_work_chunk_accuracy']} "
              f"perm_p {r['perm_p']} cos {r['centroid_cos']}")
    print("── 5 авторов (fw recall) ──", five_way["fw"]["per_recall"])
    print(f"school_separates_fw: {school_separates} | pair_separates_fw: {pair_separates}")
    print("VERDICT:", report["verdict"])


def _verdict(school_sep, pair_sep, pair_c3_sep, school, pair):
    sf, sc = school["fw"], school["c3"]
    pf, pc = pair["fw"], pair["c3"]
    head = (
        f"Ось школ (валидация): радикалы (Чернышевский, Добролюбов) против эстетиков (Дружинин, Анненков, "
        f"Боткин). На служебных словах work-level «один удержанный текст = один голос» средняя доля верных "
        f"опознаний работ {sf['macro_recall']} (по классам {sf['per_recall']}); chunk-weighted доля (каждый "
        f"кусок — голос: длинные работы весят больше, в этом кейсе ниже; диагностика) {sf['chunk_weighted_recall']}. Перестановка "
        f"ярлыков работ p={sf['perm_p']} ({sf['permutation_method']}, точный пол p {sf['permutation_exact_floor']}). "
        f"На содержательных char-3gram work-level {sc['macro_recall']} (перестановка {sc['perm_p']}). Косинус "
        f"между усреднёнными профилями {sf['centroid_cos']}. ")
    head += ("Инструмент РАЗДЕЛЯЕТ этих конкретных критиков двух школ выше порога 0.80 и значимо (это "
             "валидация на конкретных руках; перенос на уровень школы как класса не показан). " if school_sep
             else "Ось школ порога 0.80 и/или значимости перестановки не достигает. ")
    tail = (
        f"Пара учитель↔ученик (Чернышевский↔Добролюбов): на служебных словах work-level {pf['macro_recall']} "
        f"(по классам {pf['per_recall']}), chunk-weighted {pf['chunk_weighted_recall']}; перестановка работ "
        f"p={pf['perm_p']} ({pf['permutation_method']}, точный пол p {pf['permutation_exact_floor']}), косинус "
        f"{pf['centroid_cos']}. На содержательных char-3gram work-level {pc['macro_recall']}, перестановка "
        f"{pc['perm_p']}" + ("" if pair_c3_sep else " (не значимо)") + ". ")
    tail += ("Пара делится выше порога 0.80 и значимо. " if pair_sep else
             f"work-level доля по работам достигает {pf['macro_recall']}, но перестановка ярлыков работ "
             f"p={pf['perm_p']} значимости не даёт (>0.05): на этом числе работ перевес не отделим от "
             f"случайной расстановки ярлыков — разделение пары НЕ подтверждено. ")
    caveat = (
        "Оговорки: Боткин — 1 работа (~14k слов), отдельным классом в work-out не самотестируется, входит "
        "только в эстетиков; пара учитель↔ученик — у самой границы различимости (перестановка перешагивает "
        "0.05); служебные слова делят слабее содержательного char-3gram; значительная доля атрибуций "
        "«Современника» закрыта Боградом по гонорарным ведомостям, не по стилю, поэтому часть прогона — "
        "калибровка опорных точек шкалы.")
    return head + tail + caveat


if __name__ == "__main__":
    main()
