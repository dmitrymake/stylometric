# -*- coding: utf-8 -*-
"""Поправка Холма на множественные сравнения для первичных permutation-p
исследовательских кейсов (docs/cases/*.json + панель династии).

Реестр семей задан ЯВНО: файл -> список (метка, точный json-путь к p).
В семью входят только первичные (confirmatory) p-значения — те, на которых
стоит вердикт кейса. Вторичные, диагностические и контрольные p (fw_char3
с обучаемым словарём, пер-панельные/пер-секционные разбивки, бутстрап-доли,
конфаунд-контроли) в семьи не входят; ключевые контроли перечислены
в controls_excluded для прозрачности.

Холм: p сортируются по возрастанию, p_holm_i = max по j<=i из (m-j+1)*p_j,
клип на 1. Запуск: .venv/bin/python log/experiments/holm_correction.py
Выход: docs/holm_correction.json
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(ROOT, "docs", "holm_correction.json")
ALPHA = 0.05


def get_by_path(obj, dotted_path):
    """Достаёт значение по точечному json-пути ('a.b.c')."""
    cur = obj
    for key in dotted_path.split("."):
        cur = cur[key]
    return cur


def holm(p_values):
    """Поправка Холма (step-down). Вход — список сырых p, выход — список
    поправленных p в том же порядке."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):  # rank 0-базный; j = rank + 1
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


# ---------------------------------------------------------------------------
# РЕЕСТР: кейс -> файл, семья первичных p (метка, json-путь), контроли вне
# семьи (для прозрачности) и заметка о влиянии поправки на вердикт.
# ---------------------------------------------------------------------------

REGISTRY = {
    "chekhonte_bilibin_hardening": {
        "file": "docs/cases/chekhonte_bilibin_hardening.json",
        "family": [
            ("pooled фельетон, панель alloskolki, fw_only (leak-free): p_cos",
             "step2_work_label_permutation.target_pooled_feuilleton.alloskolki_fw_only.p_cos"),
            ("pooled фельетон, панель alloskolki, fw_only (leak-free): p_margin",
             "step2_work_label_permutation.target_pooled_feuilleton.alloskolki_fw_only.p_margin"),
            ("pooled фельетон, панель alloskolki, fw_only (leak-free): p_distinctiveness",
             "step2_work_label_permutation.target_pooled_feuilleton.alloskolki_fw_only.p_distinctiveness"),
            ("pooled фельетон, панель oskolki, fw_only (leak-free): p_cos",
             "step2_work_label_permutation.target_pooled_feuilleton.oskolki_fw_only.p_cos"),
            ("pooled фельетон, панель oskolki, fw_only (leak-free): p_margin",
             "step2_work_label_permutation.target_pooled_feuilleton.oskolki_fw_only.p_margin"),
            ("pooled фельетон, панель oskolki, fw_only (leak-free): p_distinctiveness",
             "step2_work_label_permutation.target_pooled_feuilleton.oskolki_fw_only.p_distinctiveness"),
        ],
        "controls_excluded": [
            ("секция 24 мая, alloskolki fw_only: p_cos — пер-секционная разбивка",
             "step2_work_label_permutation.target_section_24may_536w.alloskolki_fw_only.p_cos"),
            ("pooled фельетон, alloskolki fw_char3: p_cos — словарь учится на всех данных, диагностика",
             "step2_work_label_permutation.target_pooled_feuilleton.alloskolki_fw_char3.p_cos"),
        ],
        "verdict_note": (
            "Влияет частично, вердикт кейса не меняется. Семья — шесть перестановочных p "
            "на leak-free признаке (служебные слова) для цельного фельетона на двух панелях; "
            "пер-секционные прогоны дат и признак fw_char3 (словарь учится на всех данных, "
            "включая цель) — диагностика, в семью не входят. Порог 0.05 после поправки "
            "выдерживают 2 из 6 значений: p_cos на панели «Осколков» (0.0036) и p_margin на "
            "панели alloskolki (0.0155); остальные четыре получают 0.0524 — чуть выше порога. "
            "На каждой панели хотя бы одна статистика остаётся значимой, победитель всюду "
            "Билибин, поэтому вердикт «частично» (Билибин — самый похожий, текст составной, "
            "не доказанная атрибуция) стоит; слабый позитив-контроль по длине и так держит "
            "вывод осторожным."
        ),
    },
    "chekhonte_brother_confound": {
        "file": "docs/cases/chekhonte_brother_confound.json",
        "family": [
            ("различимость братьев: перестановочный тест полного LOO (balanced accuracy)",
             "positive_control.permutation_null.p_value"),
            ("различимость братьев при выровненной длине (окна 500 слов): перестановочный тест",
             "positive_control.length_matched_retest.brothers.permutation_p"),
        ],
        "controls_excluded": [
            ("length-only baseline (только длина текста) — конфаунд-контроль, не тест гипотезы",
             "positive_control.length_only_baseline.permutation_p"),
            ("within-author регистровый контроль (длинный/короткий Антон) — конфаунд-контроль",
             "positive_control.length_matched_retest.within_author_register_control.permutation_p"),
        ],
        "verdict_note": (
            "Не влияет. Первичные — два совместных теста одной гипотезы «братья различимы»: "
            "полный LOO (0.0225) и length-matched ре-тест (0.0107); оба после поправки остаются "
            "ниже 0.05 (0.0225 и 0.0214). Контрольные p — length-only baseline (0.0145) и "
            "within-author регистровый контроль (0.0753) — проверяют конфаунды длины и регистра, "
            "а не гипотезу авторства, и в семью не входят. Вердикт «непроверяемо/неубедительно» "
            "стоит не на значимости, а на регистровом контроле, который воспроизводит большую "
            "часть сигнала внутри одного автора — поправка этого не касается."
        ),
    },
    "sovremennik": {
        "file": "docs/cases/sovremennik.json",
        "family": [
            ("ось школ (радикалы против эстетиков), служебные слова: перестановка ярлыков работ",
             "axis_school_radical_vs_aesthete.fw.perm_p"),
            ("ось школ (радикалы против эстетиков), char-3gram: перестановка ярлыков работ",
             "axis_school_radical_vs_aesthete.c3.perm_p"),
            ("пара Чернышевский-Добролюбов, служебные слова: перестановка ярлыков работ",
             "axis_pair_chernyshevsky_vs_dobrolyubov.fw.perm_p"),
            ("пара Чернышевский-Добролюбов, char-3gram: перестановка ярлыков работ",
             "axis_pair_chernyshevsky_vs_dobrolyubov.c3.perm_p"),
        ],
        "verdict_note": (
            "Не влияет. Семья — четыре перестановочных p уровня работ (две оси на двух "
            "признаках), вердикт цитирует все четыре. После поправки ось школ значима на обоих "
            "признаках (0.002 и 0.002), пара учитель-ученик незначима (0.1428 и 0.1667) — тот "
            "же расклад, что и без поправки. Валидация на оси школ стоит, предел различимости "
            "на паре Чернышевский-Добролюбов остаётся."
        ),
    },
    "kolokol_herzen_ogaryov": {
        "file": "docs/cases/kolokol_herzen_ogaryov.json",
        "family": [
            ("Герцен против Огарёва, служебные слова: перестановка ярлыков работ",
             "fw_only.work_level_permutation_p"),
        ],
        "single_test": True,
        "controls_excluded": [
            ("fw_char3 перестановка — словарь учится на всех данных, диагностика, в решении не участвует",
             "fw_char3.work_level_permutation_p"),
        ],
        "verdict_note": (
            "Не влияет: первичный p один (0.0015, leak-free служебные слова), поправка не нужна. "
            "Перестановка на fw_char3 (0.0035) — диагностика с обучаемым словарём, вердикт на "
            "ней не стоит. Вердикт «панель разделяет» без изменений."
        ),
    },
    "dostoevsky_petersburg_chronicle": {
        "file": "docs/cases/dostoevsky_petersburg_chronicle.json",
        "family": [
            ("позитив-контроль панели: перестановка ярлыков работ",
             "positive_control.work_level_permutation_p"),
        ],
        "single_test": True,
        "verdict_note": (
            "Не влияет: первичный p один (0.0005), поправка не нужна. Вердикт (позитив-контроль "
            "выполним, фельетоны Ф.Д. уходят к публицистике Достоевского) без изменений."
        ),
    },
    "nekrasov_panaeva": {
        "file": "docs/cases/nekrasov_panaeva.json",
        "family": [
            ("Некрасов против Панаевой, служебные слова: перестановка ярлыков работ",
             "fw_only.work_level_permutation_p"),
            ("Некрасов против Панаевой, char-3gram: перестановка ярлыков работ",
             "fw_char3.work_level_permutation_p"),
        ],
        "verdict_note": (
            "Влияет на второстепенное утверждение, главный вывод стоит. Вердикт двухчастный, и "
            "семья — оба перестановочных p: служебные слова руки НЕ делят (0.1364), "
            "содержательные char-3gram делят (0.0303). После поправки fw остаётся незначимым "
            "(0.1364), а char-3gram порог 0.05 не выдерживает (0.0606). Главный вывод — "
            "тема-нейтральный идиолект соавторов не отделим — не меняется. Утверждение "
            "«содержательный признак делит значимо» после поправки формальной значимости не "
            "имеет и читается как направление по доле опознаний работ (0.95), согласное с "
            "тематическим разделом труда."
        ),
    },
    "nekrasov_panaeva_chapters": {
        "file": "docs/cases/nekrasov_panaeva_chapters.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: кейс — поглавная иллюстрация на согласии двух разметок "
            "(Cohen's kappa), перестановочных тестов не содержит. Поправка не применима."
        ),
    },
    "konek_gorbunok": {
        "file": "docs/cases/konek_gorbunok.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: вердикт inconclusive стоит на косинусных запасах и "
            "негативном контроле (ложное притяжение «Сузге» к Пушкину), без перестановочных "
            "тестов. Поправка не применима."
        ),
    },
    "cherubina": {
        "file": "docs/cases/cherubina.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: вердикт стоит на LOO-точности позитив-контроля (0.88) и "
            "косинусном запасе к Дмитриевой, без перестановочных тестов. Поправка не применима."
        ),
    },
    "prutkov_hands": {
        "file": "docs/cases/prutkov_hands.json",
        "family": [],
        "single_test": True,
        "no_primary_p": True,
        "verdict_note": (
            "Единственный перестановочный p (0.0055, запас Минаев минус Толстой) записан только "
            "в тексте key_findings, отдельного json-поля нет; поправка к одному тесту не нужна. "
            "Вердикт inconclusive стоит на бутстрап-интервале запаса, накрывающем ноль "
            "([-0.0087, +0.0341]), и z=1.58 над полем — они перевешивают одиночный значимый p, "
            "и поправка этого не меняет. binomtest позитив-контроля (1.3e-76) — контроль "
            "различимости панели, не тест атрибуции."
        ),
    },
    "lenin_testament": {
        "file": "docs/cases/lenin_testament.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: вердикт стоит на LOO-точности (0.89) и косинусных "
            "рангах кандидатов, без перестановочных тестов. Поправка не применима."
        ),
    },
    "taras_bulba": {
        "file": "docs/cases/taras_bulba.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: вердикт refuted стоит на LOO-точности (0.76-0.79) и "
            "косинусных запасах к Гоголю, без перестановочных тестов. Поправка не применима."
        ),
    },
    "vyrubova": {
        "file": "docs/cases/vyrubova.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: вердикт refuted стоит на LOO-точности (0.92) и рангах "
            "кандидатов по чанкам, без перестановочных тестов. Поправка не применима."
        ),
    },
    "veles_book": {
        "file": "docs/cases/veles_book.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: стилометрическая часть — insufficient_data (текст вне "
            "признакового пространства современного русского), вердикт «подделка» держится на "
            "лингвистике и текстологии. Поправка не применима."
        ),
    },
    "m2_delta_baseline": {
        "file": "docs/cases/m2_delta_baseline.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Собственных первичных p-значений нет: Delta-baseline даёт только macro-recall без "
            "перестановок; поля fw_cosine_gate.perm (0.1364 и 0.0714) — копии значений из "
            "кейсов nekrasov_panaeva и sovremennik, учтены в их семьях. Поправка не применима."
        ),
    },
    "chekhonte": {
        "file": "docs/cases/chekhonte.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: кейс — валидация метода (LOO 0.92, held-out Чехов "
            "возвращается к себе), без перестановочных тестов. Поправка не применима."
        ),
    },
    "chekhonte_15_micro": {
        "file": "docs/cases/chekhonte_15_micro.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: доли побед при пересборках (bootstrap_winner_share, "
            "p_bilibin/p_chehov) — устойчивость результата при случайных пересчётах, а не "
            "вероятности и не тесты гипотез. Поправка не применима."
        ),
    },
    "chekhonte_dubia": {
        "file": "docs/cases/chekhonte_dubia.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: сводный кейс Dubia держится на косинусных рангах и "
            "бутстрап-долях побед, перестановочных тестов не содержит. Поправка не применима."
        ),
    },
    "chekhonte_dubia_alloskolki": {
        "file": "docs/cases/chekhonte_dubia_alloskolki.json",
        "family": [
            ("гейт панели: перестановочный тест LOO-точности против случайной (fw_only)",
             "robustness_fw_only.gate_permutation.p_value"),
            ("все 27 Dubia: биномиальный тест «не меньше 4 из 5 длинных к Чехову» против базовой доли",
             "robustness_fw_only.all_27_attribution.binom_p_ge4of5_chehov"),
        ],
        "verdict_note": (
            "Не влияет. Семья — два теста, на которых стоит чтение кейса: гейт панели (LOO выше "
            "случайного, 0.002) и биномиальный тест перевеса Чехова среди длинных Dubia "
            "(0.0662). После поправки гейт значим (0.004), перевес Чехова незначим (0.0662) — "
            "тот же расклад, что и без поправки. Вердикт «спор набор не решает» без изменений."
        ),
    },
    "chekhonte_dubia_oskolki": {
        "file": "docs/cases/chekhonte_dubia_oskolki.json",
        "family": [],
        "no_primary_p": True,
        "verdict_note": (
            "Первичных p-значений нет: кейс держится на LOO-точностях и косинусных запасах "
            "(same-edition сдвиг к Билибину), без перестановочных тестов. Поправка не применима."
        ),
    },
    "calibration_reference": {
        "file": "docs/cases/calibration_reference.json",
        "family": [
            ("лёгкая пара Достоевский-Чернышевский: точная перестановка ярлыков работ",
             "pairs.easy_diff_author_register.work_level_permutation_p"),
            ("средняя пара Достоевский-Соллогуб: точная перестановка ярлыков работ",
             "pairs.medium_diff_author_same_era.work_level_permutation_p"),
        ],
        "verdict_note": (
            "Не влияет. Семья — два точных перестановочных p опорных пар шкалы (0.0063 и "
            "0.0047); после поправки оба 0.0094, ниже 0.05. Обе опорные точки «уверенно "
            "различимого» стоят."
        ),
    },
    "nikolai_dynasty": {
        "file": "docs/royal_register.json",
        "family": [
            ("панель династии: перестановочный тест кластера престольной линии (силуэт, n=2)",
             "perm_p"),
        ],
        "single_test": True,
        "verdict_note": (
            "Не влияет: панель дневников династии публикует один перестановочный тест "
            "(p=0.0278, взаимная близость двух дневников престольной линии при силуэте 0.54), "
            "семья из одного значения — поправка не нужна. Это же значение показано на сайте "
            "(dynastyPanel.permP). docs/nikolai_crossreg.json содержит size-matched z-оценки "
            "без p-значений. 0.0278 — пол перестановочного теста для двух точек, и вердикт "
            "уже читает его как предел n=2, не как установление «царского регистра» классом."
        ),
    },
}


def main():
    result = {}
    for case, spec in REGISTRY.items():
        path = os.path.join(ROOT, spec["file"])
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)

        entry = {"source_file": spec["file"]}

        labels = [lbl for lbl, _ in spec["family"]]
        json_paths = [jp for _, jp in spec["family"]]
        p_raw = [float(get_by_path(doc, jp)) for jp in json_paths]
        p_adj = holm(p_raw) if p_raw else []

        entry["family"] = [
            {
                "label": lbl,
                "path": jp,
                "p_raw": round(pr, 4),
                "p_holm": round(pa, 4),
                "survives_005": pa < ALPHA,
            }
            for lbl, jp, pr, pa in zip(labels, json_paths, p_raw, p_adj)
        ]

        if spec.get("single_test"):
            entry["single_test"] = True
        if spec.get("no_primary_p"):
            entry["no_primary_p"] = True

        if spec.get("controls_excluded"):
            controls = []
            for lbl, jp in spec["controls_excluded"]:
                controls.append({
                    "label": lbl,
                    "path": jp,
                    "p_raw": round(float(get_by_path(doc, jp)), 4),
                })
            entry["controls_excluded"] = controls

        entry["verdict_note"] = spec["verdict_note"]
        result[case] = entry

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    # Короткая сводка в stdout.
    for case, entry in result.items():
        fam = entry["family"]
        if not fam:
            print(f"{case}: первичных p нет")
            continue
        flips = [f for f in fam if f["p_raw"] < ALPHA and not f["survives_005"]]
        surv = sum(1 for f in fam if f["survives_005"])
        print(f"{case}: m={len(fam)}, выдерживают 0.05 после Холма {surv}/{len(fam)}"
              + (f", флипов {len(flips)}: " + "; ".join(f['label'] for f in flips) if flips else ""))
    print(f"\nЗаписано: {OUT_PATH}")


if __name__ == "__main__":
    main()
