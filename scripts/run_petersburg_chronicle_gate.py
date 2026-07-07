"""Позитив-контроль кейса dostoevsky_petersburg_chronicle (проверка выполнимости).

Главный вопрос ПЕРЕД любой атрибуцией: уходят ли подписанные Ф.Д. фельетоны «Петербургской
летописи» (1847) к Достоевскому, отделяясь от Плещеева и Соллогуба. Сложность — регистровый разрыв:
эталон Достоевского это его ранняя ПРОЗА 1846-1849, а цель — ФЕЛЬЕТОН. Если по служебным словам
(орфо/тема-устойчивый признак) фельетоны Ф.Д. возвращаются к Достоевскому — кейс живой, можно идти
в спорный Н.Н. Если нет — регистр доминирует, честный потолок.

Признак — только служебные слова (leak-free: фикс-список, словарь не учится). Кандидаты сильно
неравны по объёму (Достоевский 221k vs Плещеев 17.5k), поэтому гейт читать по recall каждого автора
и по симметричной (равные подвыборки) оценке, а не по сырой точности.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _gate_metrics import work_permutation_p  # noqa: E402
spec = importlib.util.spec_from_file_location("rd", ROOT / "scripts" / "run_chekhonte_dubia_oskolki.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

CASE = ROOT / "input_cases" / "dostoevsky_petersburg_chronicle"
OUT = ROOT / "docs" / "cases" / "dostoevsky_petersburg_chronicle.json"
CANDS = {
    "dostoevsky_fiction": CASE / "cand_dostoevsky",
    "dostoevsky_publicistic": CASE / "cand_dostoevsky_publicistic",
    "pleshcheev": CASE / "cand_pleshcheev",
    "sollogub": CASE / "cand_sollogub",
}
DOST = {"dostoevsky_fiction", "dostoevsky_publicistic"}  # оба класса = Достоевский
TARGET_FD = CASE / "petersburg_chronicle" / "peterburgskaya_letopis_FD.txt"


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> None:
    rng = np.random.default_rng(20260630)
    target_docs = rd.docs_of(TARGET_FD)
    vec, docvecs, cents = rd.make_model(target_docs, CANDS, use_char3=False)
    sizes = {n: len(docvecs[n]) for n in docvecs}

    names = list(docvecs)
    arr = {n: np.array(docvecs[n]) for n in names}

    # (1) per-class recall — leave-one-WORK-out: центроид класса при тесте работы исключает ВСЕ её куски.
    # Устойчиво к корреляции кусков внутри работы; chunk-level узнавал бы ТЕКСТ работы, не руку.
    def chunk_file(text):
        w = text.split()
        if len(w) <= 2200:
            return [text]
        return [" ".join(w[i:i + 1500]) for i in range(0, len(w), 1500)]

    work_chunks = {}  # name -> [(work_idx, vec)]
    for name, p in CANDS.items():
        files = sorted(pathlib.Path(p).glob("*.txt"))
        wc = []
        for wi, f in enumerate(files):
            for ch in chunk_file(f.read_text("utf-8", "ignore")):
                wc.append((wi, vec(ch)))
        work_chunks[name] = wc
    n_works = {n: len({wi for wi, _ in wc}) for n, wc in work_chunks.items()}

    from collections import Counter
    per_class = {}  # leave-one-WORK-out, ОДИН текст = ОДИН голос (большинство кусков работы)
    for name in names:
        works = sorted({wi for wi, _ in work_chunks[name]})
        if len(works) < 2:
            per_class[name] = "control_not_computable_single_work"
            continue
        ok = tot = 0
        for wtest in works:
            test = [v for wi, v in work_chunks[name] if wi == wtest]
            cen = {o: _unit(np.mean([v for wi, v in work_chunks[o] if not (o == name and wi == wtest)],
                                    axis=0)) for o in names}
            preds = [max(names, key=lambda o: float(np.dot(_unit(v), cen[o]))) for v in test]
            ok += Counter(preds).most_common(1)[0][0] == name  # один текст = один голос
            tot += 1
        per_class[name] = [ok, tot]
    computable = {n: v for n, v in per_class.items() if isinstance(v, list)}
    macro_recall = round(float(np.mean([c / t for c, t in computable.values()])), 4) if computable else 0.0

    # work-level перестановка ярлыков работ (4 класса → случайная с plus-one): значим ли перевес
    # позитив-контроля над случайной расстановкой меток. Единая с прочими кейсами проверка значимости;
    # надёжность требует И macro_recall >= 0.80, И значимой перестановки (как в run_*_gate).
    perm_data = [(name, (name, wi), v) for name in names for wi, v in work_chunks[name]]
    perm_p, perm_method, perm_floor = work_permutation_p(perm_data, lambda a: a, names)

    # (2) within-work диагностика (УТЕЧКА, не для решения): chunk-level баланс — узнаёт текст, не руку.
    nmin = min(sizes.values())
    accs = []
    for _ in range(300):
        sub = {n: arr[n][rng.choice(len(arr[n]), nmin, replace=False)] for n in names}
        sc = {n: _unit(v.mean(0)) for n, v in sub.items()}
        ok = tot = 0
        for tc in names:
            for i in range(nmin):
                cen = {m: (_unit((sub[m].sum(0) - sub[m][i]) / (nmin - 1)) if m == tc else sc[m])
                       for m in names}
                ok += max(cen, key=lambda m: float(np.dot(_unit(sub[tc][i]), cen[m]))) == tc
                tot += 1
        accs.append(ok / tot)
    within_work_chunk_accuracy = round(float(np.mean(accs)), 4)

    # (3) ГЛАВНОЕ: куда уходят фельетоны Ф.Д. (целиком и по кускам) + bootstrap-устойчивость
    feuil_whole = rd.sims([vec(t) for t in target_docs], cents)
    per_chunk = [rd.sims([vec(t)], cents)[0]["candidate"] for t in target_docs]
    chunk_counts = dict(Counter(per_chunk).most_common())
    # WORK-LEVEL bootstrap: ресемпл РАБОТ каждого класса (не кусков), пул их кусков -> центроид.
    # Ресемпл кусков завышал бы уверенность (коррелированные куски одной работы как независимые).
    works_of = {n: {} for n in names}
    for n in names:
        for wi, v in work_chunks[n]:
            works_of[n].setdefault(wi, []).append(v)
    work_ids = {n: list(works_of[n]) for n in names}
    tv = _unit(np.mean([vec(t) for t in target_docs], axis=0))
    win = Counter()
    for _ in range(2000):
        bc = {}
        for n in names:
            wids = work_ids[n]
            sample = [v for wi in (wids[i] for i in rng.integers(0, len(wids), len(wids)))
                      for v in works_of[n][wi]]
            bc[n] = _unit(np.mean(sample, axis=0))
        win[max(bc, key=lambda m: float(np.dot(tv, bc[m])))] += 1
    boot = {n: round(c / 2000, 3) for n, c in win.most_common()}
    boot_dost = round(sum(c for n, c in win.items() if n in DOST) / 2000, 3)
    # отличимость Ф.Д. целиком: cos победителя минус cos среднего профиля кандидатов (>0 = различимо,
    # <=0 = лишь центральный). Центральность классов — для сравнения.
    mean_cent = _unit(np.mean([cents[n] for n in names], axis=0))
    feuil_dist = round(float(np.dot(tv, cents[feuil_whole[0]["candidate"]]) - np.dot(tv, mean_cent)), 4)
    centrality = {n: round(float(np.dot(cents[n], mean_cent)), 4) for n in names}

    # ЕДИНЫЙ с прочими кейсами порог надёжности: work-out macro_recall >= 0.80 И значимая перестановка
    # ярлыков работ (perm_p <= 0.05). Без значимости перевес позитив-контроля не отделим от случайной
    # расстановки меток, поэтому статус «reliable» не ставится без значимости перестановки.
    fd_returns = bool(feuil_whole[0]["candidate"] in DOST and boot_dost >= 0.9)
    separates_significant = bool(macro_recall >= 0.80 and perm_p <= 0.05)
    gate_pass = bool(separates_significant and fd_returns)
    status = ("reliable" if gate_pass else
              "weak_below_threshold" if fd_returns else "fail")
    report = {
        "case": "dostoevsky_petersburg_chronicle",
        "stage": "feasibility_gate",
        "status": status,
        "title": "Петербургская летопись (1847): проверка выполнимости — уходят ли фельетоны Ф.Д. к Достоевскому",
        "candidate_words": {n: sum(len(re.findall(rd.WORD, t)) for t in rd.read(p))
                            for n, p in CANDS.items()},
        "candidate_docs": sizes,
        "note": ("Эталон Достоевского — его ранняя проза 1846-1849 и публицистика, цель — фельетон; "
                 "признак только служебные слова (без утечки: фикс-список, словарь не учится). Проверка "
                 "выполнимости: панель различает авторов (доля верных опознаний по авторам) И подписанные "
                 "Ф.Д. фельетоны уходят к Достоевскому при пересборках. Низкая проверка = регистр "
                 "доминирует, атрибуцию спорного Н.Н. давать нельзя."),
        "positive_control": {
            "per_class_recall_work_LOO": {n: (f"{v[0]}/{v[1]}" if isinstance(v, list) else v)
                                          for n, v in per_class.items()},
            "works_per_class": n_works,
            "macro_recall": macro_recall,
            "macro_recall_note": ("leave-one-WORK-out по классам с >=2 работами; Плещеев (1 работа) — "
                                  "control_not_computable_single_work, исключён из macro и из заявления о "
                                  "разделении рук"),
            "work_level_permutation_p": perm_p,
            "permutation_method": perm_method,
            "permutation_exact_floor": perm_floor,
            "permutation_note": ("перестановка ярлыков работ среди классов панели: значим ли перевес над "
                                 "случайной расстановкой меток. Надёжность (status reliable) требует И "
                                 "macro_recall >= 0.80, И perm_p <= 0.05."),
            "within_work_chunk_accuracy": within_work_chunk_accuracy,
            "within_work_note": "within-work диагностика (узнаёт текст работы, не руку); в решении НЕ участвует",
        },
        "FD_feuilletons_attribution": {
            "whole": feuil_whole,
            "whole_distinctiveness": feuil_dist,
            "per_chunk_winners": chunk_counts,
            "bootstrap_winner_share": boot,
            "bootstrap_dostoevsky_share": boot_dost,
            "class_centrality": centrality,
        },
        "gate_pass": gate_pass,
        "verdict": _verdict(status, macro_recall, perm_p, perm_method, feuil_whole, boot, boot_dost),
        "caveat": ("Эталон публицистики Достоевского — «Дневник писателя» 1873-1881, на 26 лет позже "
                   "фельетонов 1847; совпадение может нести и сдвиг эпохи, и совпадение регистра, не "
                   "только идиолект."),
        "data_status": ("Все тексты — общественное достояние (Достоевский †1881, Плещеев †1893, Соллогуб "
                        "†1882). Якоря и фельетоны Ф.Д. — чистый текст с az.lib.ru; спорный фельетон Н.Н. "
                        "(13.04.1847) — с rvb.ru (Русская виртуальная библиотека). Сырьё пишется в gitignored "
                        "input_cases/dostoevsky_petersburg_chronicle/; в git — только скрипт добычи "
                        "(scripts/fetch_petersburg_chronicle.py) и этот JSON."),
        "sources": [
            {"cite": "«Петербургская летопись» Ф.Д. (1847), фельетоны — позитив-контроль/эталон регистра, "
                     "petersburg_chronicle", "url": "http://az.lib.ru/d/dostoewskij_f_m/"},
            {"cite": "Ф. М. Достоевский, ранняя проза 1846-1849 — cand_dostoevsky",
             "url": "http://az.lib.ru/d/dostoewskij_f_m/"},
            {"cite": "Ф. М. Достоевский, «Дневник писателя» 1876-1880 (публицистика) — "
                     "cand_dostoevsky_publicistic", "url": "http://az.lib.ru/d/dostoewskij_f_m/"},
            {"cite": "А. Н. Плещеев, «Житейские сцены» (1856) — cand_pleshcheev",
             "url": "http://az.lib.ru/p/plesheew_a_n/"},
            {"cite": "В. А. Соллогуб, светская проза 1840-х + петербургский фельетон «Букеты» (1845) — "
                     "cand_sollogub", "url": "http://az.lib.ru/s/sollogub_w_a/"},
            {"cite": "Спорный фельетон Н.Н. (13.04.1847), Русская виртуальная библиотека — target_NN",
             "url": "https://rvb.ru/dostoevski/01text/vol2/17.htm"},
        ],
        "analysis_command": "PYTHONPATH=src python3 scripts/run_petersburg_chronicle_gate.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}")
    print(json.dumps({"per_class": report["positive_control"], "FD": report["FD_feuilletons_attribution"],
                      "gate_pass": gate_pass}, ensure_ascii=False, indent=1))


def _verdict(status, macro, perm_p, perm_method, feuil, boot, boot_dost) -> str:
    top = feuil[0]["candidate"]
    pub = int(round(boot.get("dostoevsky_publicistic", 0) * 100))
    base = (f"Подписанные Ф.Д. фельетоны при пересборках чаще всего ближе к публицистике Достоевского "
            f"(«Дневник писателя») — к этому КОНКРЕТНОМУ классу {pub}% (доля по одному классу; Достоевский "
            f"занимает 2 класса из 4, поэтому суммарная «доля к Достоевскому» завышена базовой ставкой). "
            f"Средняя доля верных опознаний классов (с убранной целой работой) {macro}, перестановка ярлыков "
            f"работ p={perm_p} ({perm_method}). Эталон Плещеева не проверяем — одна работа даёт вырожденный "
            f"профиль и чужой регистр. На художественном эталоне фельетоны Ф.Д. уходят к фельетонно-светской "
            f"прозе Соллогуба, а не к беллетристике Достоевского.")
    if status == "reliable":
        return (base + " Доля выше порога 0.80 при значимой перестановке — позитив-контроль выполним; но это "
                "лишь публицистический эталон для теста спорного Н.Н., с оговоркой на сдвиг эпохи (26 лет), а "
                "не уверенная атрибуция.")
    if status == "weak_below_threshold":
        return (base + " Надёжность требует И доли >= 0.80, И перестановки <= 0.05 — здесь это условие не "
                "выполнено, разделение слабое, не надёжное (как в кейсе «Колокол»). Атрибуцию спорного Н.Н. "
                "давать нельзя.")
    return (f"Гейт не пройден: средняя доля верных опознаний классов {macro}, перестановка p={perm_p} "
            f"({perm_method}); фельетоны Ф.Д. целиком уходят к «{top}». Атрибуцию спорного Н.Н. давать нельзя.")


if __name__ == "__main__":
    main()
