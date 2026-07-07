"""Калибровочная шкала для каталога «честный протокол».

Косинусы между центроидами авторов во всех кейсах высоки (0.92-0.98), и без эталона «вот так
выглядит уверенно различимое» доля опознаний и косинус повисают без шкалы. Этот скрипт даёт опорные
точки той же связкой метрик, что и кейсы: leave-one-WORK-out + перестановка ярлыков работ (точное
перечисление при малом числе работ) + косинус центроидов, признак — служебные слова. РАЗДЕЛЁННЫЕ
метрики: work_macro_recall (один удержанный текст = один голос, порог 0.80 к ней) и
chunk_weighted_recall (каждый кусок — голос; длинные работы весят больше, в этих кейсах ниже; диагностика).

- ЛЁГКАЯ пара (разные эпоха и регистр): Достоевский, проза 1840-х ↔ Чернышевский, критика 1850-х.
  Верхний (положительный) полюс шкалы.
- СРЕДНЯЯ пара (один регистр/эпоха, разные авторы): Достоевский, проза 1840-х ↔ Соллогуб, светская
  проза 1840-х.

Обе пары разделяются на work-level со значимой перестановкой — это уверенное разделение (калибровка),
а НЕ предел кейса. Числа читают рядом с кейсами: где доля и перестановка близки к опорным — руки
различимы; где доля около 0.6-0.7 при косинусе 0.97-0.98 — руки близки к неразличимым.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stylo.lang import function_words  # noqa: E402
from _gate_metrics import leave_one_work_out, both_metrics, work_permutation_p  # noqa: E402

OUT = ROOT / "docs" / "cases" / "calibration_reference.json"
WORD = r"[а-яёА-ЯЁ]+"
WIN = 600

PAIRS = {
    "easy_diff_author_register": {
        "label": "Достоевский (проза 1840-х) ↔ Чернышевский (критика 1850-х) — обе толстые, разные",
        "a": ("dostoevsky", ROOT / "input_cases/dostoevsky_petersburg_chronicle/cand_dostoevsky"),
        "b": ("chernyshevsky", ROOT / "input_cases/sovremennik/chernyshevsky"),
    },
    "medium_diff_author_same_era": {
        "label": "Достоевский (проза 1840-х) ↔ Соллогуб (светская проза 1840-х) — один регистр/эпоха",
        "a": ("dostoevsky", ROOT / "input_cases/dostoevsky_petersburg_chronicle/cand_dostoevsky"),
        "b": ("sollogub", ROOT / "input_cases/dostoevsky_petersburg_chronicle/cand_sollogub"),
    },
}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def chunks(path):
    out = []
    for f in sorted(pathlib.Path(path).glob("*.txt")):
        w = f.read_text("utf-8", "ignore").split()
        for i in range(0, len(w), WIN):
            piece = " ".join(w[i:i + WIN])
            if len(piece.split()) >= WIN // 2:
                out.append((f.stem, piece))
    return out


def fw_vec(fw, fwi):
    def vec(text):
        toks = re.findall(WORD, text.lower())
        v = np.zeros(len(fw))
        for t in toks:
            j = fwi.get(t)
            if j is not None:
                v[j] += 1
        v /= len(toks) or 1
        return _unit(v)
    return vec


def run(a_name, a_dir, b_name, b_dir):
    fw = sorted(function_words("ru"))
    vec = fw_vec(fw, {w: i for i, w in enumerate(fw)})
    data = []  # (author, work, vec)
    for name, d in ((a_name, a_dir), (b_name, b_dir)):
        for work, text in chunks(d):
            data.append((name, work, vec(text)))
    authors = [a_name, b_name]
    works = {a: sorted({w for au, w, _ in data if au == a}) for a in authors}
    cen = lambda vs: _unit(np.mean(vs, axis=0))

    # ОБЕ метрики раздельно: work_macro_recall (один текст = один голос) и chunk_weighted (диагностика).
    wcp, confusion, _w = leave_one_work_out(data, authors)
    M = both_metrics(wcp, authors)
    macro_recall = M["work_macro_recall"]            # порог 0.80 применяется К НЕЙ
    chunk_weighted = M["chunk_weighted_recall"]      # НЕ work-level: коррелированные куски = много голосов
    per_author_recall = M["work_recall"]

    # перестановка ярлыков работ на work-level метрике; exact-перечисление при малом числе работ.
    perm_p, perm_method, perm_floor = work_permutation_p(data, lambda a: a, authors)

    full = {a: cen([v for au, _, v in data if au == a]) for a in authors}
    cos = round(float(np.dot(full[authors[0]], full[authors[1]])), 4)
    return {
        "per_author_recall": per_author_recall,            # work-level (один текст = один голос)
        "macro_recall": macro_recall,                      # = work_macro_recall (порог 0.80 к ней)
        "chunk_weighted_recall": chunk_weighted,           # диагностика: куски как голоса (не work-level)
        "chunk_recall": M["chunk_recall"],
        "work_level_permutation_p": perm_p,
        "permutation_method": perm_method,                 # exact_N или random_N
        "permutation_exact_floor": perm_floor,             # минимально достижимое точное p = 1/C(W,n1)
        "cross_author_centroid_cos": cos,
        "confusion": confusion,
        "works": {a: len(works[a]) for a in authors},
    }


def _status(r) -> str:
    """Опорная точка «разделяет», когда work-level выше порога 0.80 И перестановка значима."""
    return ("separates" if (r["macro_recall"] >= 0.80 and r["work_level_permutation_p"] <= 0.05)
            else "weak")


def _verdict(easy, medium) -> str:
    def line(tag, r):
        return (f"{tag}: верные опознания работ {r['per_author_recall']}, в среднем {r['macro_recall']} "
                f"(work-level, один текст = один голос); chunk-weighted (каждый кусок — голос: длинные работы "
                f"весят больше, в этом кейсе ниже; диагностика) {r['chunk_weighted_recall']}; перестановка ярлыков работ "
                f"p={r['work_level_permutation_p']} ({r['permutation_method']}, точный пол "
                f"{r['permutation_exact_floor']}); косинус центроидов {r['cross_author_centroid_cos']}. ")
    sep = ("Обе опорные пары разделяются на work-level выше порога 0.80 со значимой точной перестановкой"
           if (_status(easy) == "separates" and _status(medium) == "separates")
           else "Опорные пары на work-level разделяются не полностью")
    return ("Калибровка — опорные точки шкалы (уверенное разделение), а НЕ предел кейса. " +
            line("ЛЁГКАЯ пара (разные эпоха и регистр)", easy) +
            line("СРЕДНЯЯ пара (разные авторы, один регистр/эпоха)", medium) +
            sep + ". Это верхний полюс различимости: косинусы служебных слов русской прозы вообще высоки, "
            "поэтому кейсы каталога читаются по доле опознаний и перестановке относительно этих точек, а "
            "косинус — относительно этой шкалы.")


def main():
    results = {}
    for key, spec in PAIRS.items():
        an, ad = spec["a"]
        bn, bd = spec["b"]
        r = run(an, ad, bn, bd)
        r["label"] = spec["label"]
        r["status"] = _status(r)
        results[key] = r
        print(f"  [{key}] {spec['label']}")
        print(f"     recall {r['per_author_recall']} macro {r['macro_recall']} | chunk_weighted "
              f"{r['chunk_weighted_recall']} | perm_p {r['work_level_permutation_p']} "
              f"({r['permutation_method']}, floor {r['permutation_exact_floor']}) | "
              f"cos {r['cross_author_centroid_cos']} | works {r['works']} | {r['status']}")
    easy = results["easy_diff_author_register"]
    medium = results["medium_diff_author_same_era"]
    report = {
        "case": "calibration_reference",
        "title": "Калибровочная шкала: как выглядит уверенно различимое",
        "note": ("Опорные точки для чтения кейсов каталога. Две РАЗНЫЕ метрики: work_macro_recall (один "
                 "удержанный текст = один голос, большинство его кусков) — к ней применяется порог 0.80; "
                 "chunk_weighted_recall (каждый кусок — голос) НЕ work-level (коррелированные куски = много "
                 "голосов) и приводится как диагностика. Обучение — leave-one-WORK-out; перестановка ярлыков "
                 "работ точным перечислением (малое число работ). Лёгкая пара (разные эпоха и регистр) — "
                 "верхний полюс различимости; same-register пара — насколько различимы даже разные авторы "
                 "одного жанра."),
        "pairs": results,
        "scale_reading": (f"Уверенно различимые авторы. Лёгкая пара (разные эпоха/регистр): доля верных "
                          f"опознаний работ {easy['macro_recall']} (work-level), перестановка p="
                          f"{easy['work_level_permutation_p']} ({easy['permutation_method']}), косинус "
                          f"центроидов {easy['cross_author_centroid_cos']}. Средняя пара (один регистр/эпоха, "
                          f"разные авторы): доля {medium['macro_recall']}, перестановка p="
                          f"{medium['work_level_permutation_p']} ({medium['permutation_method']}), косинус "
                          f"{medium['cross_author_centroid_cos']}. Косинусы служебных слов русской прозы вообще "
                          f"высоки, поэтому судить надо по доле опознаний работ и перестановке, а косинус читать "
                          f"относительно этой шкалы."),
        "verdict": _verdict(easy, medium),
        "data_status": ("Все тексты — общественное достояние (Достоевский †1881, Чернышевский †1889, Соллогуб "
                        "†1882). Чистый текст с az.lib.ru; корпуса переиспользованы из кейсов "
                        "dostoevsky_petersburg_chronicle и sovremennik. Сырьё в gitignored input_cases/; в git "
                        "— только скрипты добычи (scripts/fetch_petersburg_chronicle.py, "
                        "scripts/fetch_sovremennik.py) и этот JSON."),
        "sources": [
            {"cite": "Ф. М. Достоевский, ранняя проза 1846-1849 — "
                     "input_cases/dostoevsky_petersburg_chronicle/cand_dostoevsky",
             "url": "http://az.lib.ru/d/dostoewskij_f_m/"},
            {"cite": "Н. Г. Чернышевский, литкритика 1850-х — input_cases/sovremennik/chernyshevsky",
             "url": "http://az.lib.ru/c/chernyshewskij_n_g/"},
            {"cite": "В. А. Соллогуб, светская проза 1840-х — "
                     "input_cases/dostoevsky_petersburg_chronicle/cand_sollogub",
             "url": "http://az.lib.ru/s/sollogub_w_a/"},
        ],
        "analysis_command": "PYTHONPATH=src python3 scripts/run_calibration_pair.py",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}")
    print("SCALE:", report["scale_reading"])
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
