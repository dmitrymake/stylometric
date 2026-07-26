"""Позитив-контроль калибровочного кейса nekrasov_panaeva.

Make-or-break: делится ли СОЛО-проза Некрасова и Панаевой по тематически-нейтральным служебным словам.
У Панаевой соло — семейно-женская линия, у Некрасова — социально-сатирическая, поэтому разделение по
содержательным признакам спутано с темой. Если руки делятся на СЛУЖЕБНЫХ словах (предлоги/союзы/
частицы, тему не выдают) — это сигнал руки, и калибровку «увидеть две руки» в соавторских романах
можно вести честно. Если делятся только на char-3gram (содержательнее), но не на служебных словах —
сигнал темовой.

Метрика — leave-one-WORK-out (устойчиво к корреляции кусков) + уравнивание объёма + перестановка
ярлыков работ. Признаки: служебные слова (leak-free) и char-3gram (сравнение).
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

CASE = ROOT / "input_cases" / "nekrasov_panaeva"
HISTORICAL_OUT = ROOT / "docs" / "cases" / "nekrasov_panaeva.json"
DEFAULT_OUT = (
    ROOT / "docs" / "cases" / "work_balanced_audit" / "custom"
    / "nekrasov_panaeva.work_balanced.json"
)
DIRS = {"nekrasov": [CASE / "nekrasov_solo"], "panaeva": [CASE / "panaeva_solo"]}
WORD = r"[а-яёА-ЯЁ]+"
WIN = 600


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def chunks_by_work(paths):
    out = []
    for path in paths:
        for f in sorted(path.glob("*.txt")):
            w = f.read_text("utf-8", "ignore").split()
            for i in range(0, len(w), WIN):
                piece = " ".join(w[i:i + WIN])
                if len(piece.split()) >= WIN // 2:
                    out.append((f.stem, piece))
    return out


def build_vectorizer(all_texts, use_char3):
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


def evaluate(use_char3):
    raw = {a: chunks_by_work(p) for a, p in DIRS.items()}
    all_texts = [t for a in raw for _, t in raw[a]]
    vec = build_vectorizer(all_texts, use_char3)
    data = [(a, work, vec(text)) for a in raw for work, text in raw[a]]
    authors = list(DIRS)
    works = {a: sorted({w for au, w, _ in data if au == a}) for a in authors}

    def centroid(vs):
        return _unit(np.mean(vs, axis=0))

    # ОБЕ метрики раздельно: work_macro_recall (один текст = один голос) и chunk_weighted (диагностика).
    wcp, confusion, _w = leave_one_work_out(data, authors)
    M = both_metrics(wcp, authors)
    macro_recall = M["work_macro_recall"]            # порог 0.80 применяется К НЕЙ
    chunk_weighted = M["chunk_weighted_recall"]      # НЕ work-level: коррелированные куски = много голосов
    per_author_recall = M["work_recall"]

    rng = np.random.default_rng(20260630)
    byauthor = {a: [v for au, _, v in data if au == a] for a in authors}
    nmin = min(len(byauthor[a]) for a in authors)
    accs = []
    for _ in range(300):
        sub = {a: [byauthor[a][i] for i in rng.choice(len(byauthor[a]), nmin, replace=False)] for a in authors}
        ok = tot = 0
        for ta in authors:
            for i in range(nmin):
                cents = {b: (centroid([x for j, x in enumerate(sub[b]) if j != i]) if b == ta
                            else centroid(sub[b])) for b in authors}
                pred = max(authors, key=lambda b: float(np.dot(_unit(sub[ta][i]), cents[b])))
                ok += pred == ta
                tot += 1
        accs.append(ok / tot)
    balanced = round(float(np.mean(accs)), 4)

    # перестановка ярлыков работ на work-level метрике; exact-перечисление при малом числе работ
    # (10 Некрасов + 2 Панаева -> C(12,2)=66, пол p=1/66≈0.015).
    perm_p, perm_method, perm_floor = work_permutation_p(data, lambda a: a, authors)

    full_cent = {
        a: work_balanced_centroid((work, v) for au, work, v in data if au == a)
        for a in authors
    }
    cross_cos = round(float(np.dot(full_cent["nekrasov"], full_cent["panaeva"])), 4)

    return {
        "per_author_recall": per_author_recall,            # work-level (один текст = один голос)
        "macro_recall": macro_recall,                      # = work_macro_recall (порог 0.80 к ней)
        "chunk_weighted_recall": chunk_weighted,           # диагностика: куски как голоса (не work-level)
        "chunk_recall": M["chunk_recall"],
        "within_work_chunk_accuracy": balanced,
        "work_level_permutation_p": perm_p,
        "permutation_method": perm_method,                 # exact_N или random_N
        "permutation_exact_floor": perm_floor,             # минимально достижимое точное p = 1/C(W,n1)
        "cross_author_centroid_cos": cross_cos,
        "train_centroid_weighting": "equal_work_direction_after_within_work_chunk_mean_l2",
        "confusion": confusion,
        "chunks": {a: len(byauthor[a]) for a in authors},
        "works": {a: works[a] for a in authors},
    }


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
    fw_only = evaluate(use_char3=False)
    fw_char3 = evaluate(use_char3=True)
    # «Разделяет» = НАДЁЖНО (work_macro_recall>=0.80) И ЗНАЧИМО (перестановка ярлыков работ <=0.05).
    # within_work_chunk_accuracy в решении НЕ участвует: это within-work диагностика (центроид при
    # тесте куска сохраняет другие куски ТОЙ ЖЕ работы — узнаёт текст, не руку, завышает).
    def separates(r):
        return bool(r["macro_recall"] >= 0.80 and r["work_level_permutation_p"] <= 0.05)
    sep_fw = bool(separates(fw_only))
    sep_c3 = bool(separates(fw_char3))
    rr = {a: int(v.split("/")[0]) / max(int(v.split("/")[1]), 1)
          for a, v in fw_only["per_author_recall"].items()}
    weak_side = min(rr, key=rr.get)
    if sep_fw:
        fw_status = "separates_on_function_words"       # рука: служебные слова делят надёжно и значимо
    elif fw_only["work_level_permutation_p"] <= 0.05:
        fw_status = "significant_but_weak_thin_corpus"  # значимо на служебных, но work-recall ниже 0.80
    elif sep_c3:
        fw_status = "topic_signal_only"                 # делит только содержание (char3), не служебные слова
    else:
        fw_status = "no_separation"
    report = {
        "case": "nekrasov_panaeva",
        "stage": "feasibility_gate",
        "title": "Некрасов и Панаева («Н. Станицкий»): проверка калибровки — делятся ли руки на служебных словах",
        "type": "calibration",
        "candidates": ["Н. А. Некрасов", "А. Я. Панаева"],
        "feature": ("формальный признак — служебные слова (предлоги, союзы, частицы; тему не выдают, словарь "
                    "не учится = leak-free). char-3gram — ЗАВИСИМАЯ ОТ ДАННЫХ диагностика (словарь top-800 "
                    "учится на всех текстах, ловит тему) и в векторе идёт вместе со служебными словами; не "
                    "формальное доказательство. within_work_chunk_accuracy — within-work диагностика (утечка), "
                    "в решении не используется; значимость — по перестановке ярлыков работ."),
        "note": ("Калибровочный кейс: соавторские «Три страны света» (1849) и «Мёртвое озеро» (1851), раздел "
                 "труда задокументирован мемуарами Панаевой. Автор≡тема (Панаева — семейно-женская линия, "
                 "Некрасов — социально-сатирическая), поэтому деление по содержанию спутано с темой. Честный "
                 "тест — служебные слова: если соло-руки делятся на них (тему не выдают), сигнал руки; если "
                 "только на char-3gram — сигнал темовой. Две РАЗНЫЕ метрики: work_macro_recall (один удержанный "
                 "текст = один голос, большинство его кусков) — к ней применяется порог 0.80; "
                 "chunk_weighted_recall (каждый кусок — голос) НЕ work-level (коррелированные куски = много "
                 "голосов), приводится как диагностика. Перестановка ярлыков работ — точное перечисление всех "
                 "расстановок (10 Некрасов + 2 Панаева -> C(12,2)=66, пол p=1/66≈0.015)."),
        "fw_only": fw_only,
        "fw_char3": fw_char3,
        "separates_on_function_words": bool(sep_fw),
        "separates_on_char3": bool(sep_c3),
        "fw_status": fw_status,
        "weak_side": weak_side,
        "verdict": _verdict(fw_status, weak_side, fw_only, fw_char3),
        "caveat": ("Соло-корпус Некрасова ~136k в 10 работах (az.lib + Викитека; академический аппарат ПСС — "
                   "комментарии и «другие редакции/варианты» — вырезан, иначе near-duplicate черновики дают "
                   "within-work утечку и асимметрично раздувают char-3gram). Панаева ~57k в ДВУХ работах "
                   "(«Семейство Тальниковых» 1848, «Степная барышня» 1855) — нижняя граница: на её фолде "
                   "центроид вырождается до одной работы. Раздел труда в романах документирован, но местами "
                   "спорен."),
        "next": ["Если делятся на служебных словах — нарезать «Три страны света»/«Мёртвое озеро» по главам, "
                 "классифицировать по соло-центроидам, сверить с задокументированным разделом труда (калибровка).",
                 "Сравнить fw и char3 поглавно: совпадение = сигнал руки; расхождение = тема тянет char3.",
                 "Пограничные/спорные главы и вопрос третьей руки (И. Панаев) — отметить как неразрешимые, там "
                 "тематический сигнал исчезает."],
        "data_status": ("Все тексты — общественное достояние (Некрасов †1877, Панаева †1893). Соло-проза — "
                        "чистый текст с az.lib.ru плюс ранняя проза Некрасова 1840-х с Викитеки через "
                        "action=parse; академический аппарат ПСС (комментарии, «другие редакции/варианты») "
                        "вырезан, иначе near-duplicate черновики дают within-work утечку. Сырьё пишется в "
                        "gitignored input_cases/nekrasov_panaeva/; в git — только скрипты добычи "
                        "(scripts/fetch_nekrasov_panaeva.py, scripts/fetch_nekrasov_prose_ws.py) и этот JSON."),
        "sources": [
            {"cite": "Н. А. Некрасов, соло-проза («Ростовщик» 1841, «Тонкий человек» 1856) — якорь "
                     "nekrasov_solo", "url": "http://az.lib.ru/n/nekrasow_n_a/"},
            {"cite": "Н. А. Некрасов, ранняя проза 1840-х («Петербургские углы», «Жизнь и похождения Тихона "
                     "Тростникова» и др.) — укрепление nekrasov_solo, Викитека action=parse",
             "url": "https://ru.wikisource.org/w/api.php"},
            {"cite": "А. Я. Панаева, соло («Семейство Тальниковых» 1848, «Степная барышня» 1855) — якорь "
                     "panaeva_solo", "url": "http://az.lib.ru/p/panaewa_a_j/"},
            {"cite": "Соавторские «Три страны света» (1849), «Мёртвое озеро» (1851) под псевд. «Н. Станицкий» "
                     "— материал поглавной калибровки (раздел труда документирован мемуарами Панаевой)",
             "url": "http://az.lib.ru/n/nekrasow_n_a/"},
        ],
        "status": "exploratory_adversarial_rerun",
        "lineage": {
            "historical_report": str(HISTORICAL_OUT.relative_to(ROOT)),
            "historical_report_status": "superseded_for_scientific_interpretation",
        },
        "analysis_command": (
            "PYTHONPATH=src python3 scripts/run_nekrasov_panaeva_gate.py "
            f"--out {shown}"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_strict(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {shown}")
    for tag, r in (("fw_only", fw_only), ("fw_char3", fw_char3)):
        print(f"  [{tag}] work-recall {r['per_author_recall']} macro {r['macro_recall']} | chunk_weighted "
              f"{r['chunk_weighted_recall']} | balanced {r['within_work_chunk_accuracy']} | perm_p "
              f"{r['work_level_permutation_p']} ({r['permutation_method']}, floor {r['permutation_exact_floor']}) | "
              f"cos(N,P) {r['cross_author_centroid_cos']} | chunks {r['chunks']}")
    print(f"fw_status: {fw_status} | weak_side: {weak_side} | separates_fw: {sep_fw} char3: {sep_c3}")
    print("VERDICT:", report["verdict"])


def _verdict(fw_status, weak_side, fw, c3):
    rf, pf, cf = fw["macro_recall"], fw["work_level_permutation_p"], fw["cross_author_centroid_cos"]
    cwf, methf, floorf = fw["chunk_weighted_recall"], fw["permutation_method"], fw["permutation_exact_floor"]
    rc, pc = c3["macro_recall"], c3["work_level_permutation_p"]
    cwc, methc = c3["chunk_weighted_recall"], c3["permutation_method"]
    side_ru = {"nekrasov": "Некрасова", "panaeva": "Панаевой"}[weak_side]
    caveat = ("Оговорки: соло Панаевой — всего 2 работы, поэтому work-метрика над 2 точками груба (на её фолде "
              "центроид вырождается до одной оставшейся работы). Автор≡тема (Панаева — семейно-женская линия, "
              "Некрасов — социально-сатирическая): служебные слова темы не выдают, char-3gram её несёт. "
              "Пограничные главы и третья рука (И. Панаев) остаются неразрешимыми.")
    if fw_status == "separates_on_function_words":
        return (f"На корректной метрике «один текст = один голос» (work-level) соло-руки делятся на "
                f"тематически-нейтральных служебных словах: верные опознания работ {fw['per_author_recall']}, в "
                f"среднем {rf} (>=0.80); chunk-weighted доля (каждый кусок — голос) {cwf} (длинные работы весят "
                f"больше по числу кусков, в этом кейсе ниже) — диагностика. Перестановка ярлыков работ p={pf} ({methf}, пол {floorf}). "
                f"Косинус между усреднёнными профилями {cf}. Это сигнал руки, а не темы — поглавную калибровку "
                f"«увидеть две руки» можно вести. " + caveat)
    if fw_status == "significant_but_weak_thin_corpus":
        return (f"На служебных словах перестановка ярлыков работ значима (p={pf}, {methf}, пол {floorf}) и держится "
                f"именно на них: на содержательных char-3gram перестановка p={pc}. Но work-level средняя доля {rf} "
                f"ниже планки 0.80 — её тянет вниз сторона {side_ru} ({fw['per_author_recall'][weak_side]}); "
                f"chunk-weighted доля {cwf} (диагностика). Косинус {cf}. Сигнал руки тематически-нейтрален, но "
                f"соло-корпус {side_ru} тонкий — до его укрепления поглавную атрибуцию романов давать рано. " + caveat)
    if fw_status == "topic_signal_only":
        return (f"На корректной метрике «один текст = один голос» (work-level) служебные слова руки НЕ делят: "
                f"средняя доля {rf} (ниже 0.80), перестановка ярлыков работ p={pf} ({methf}, пол {floorf}); "
                f"chunk-weighted доля {cwf} (диагностика; в этом кейсе ниже). Содержательные char-3gram делят: "
                f"work-level средняя {rc}, перестановка p={pc} ({methc}), chunk-weighted {cwc}. Косинус между "
                f"усреднёнными профилями {cf}. Значит видимое разделение несёт ТЕМУ (у Панаевой соло — "
                f"семейно-женская линия, у Некрасова — социально-сатирическая), а не тема-нейтральный идиолект. "
                f"Поглавная классификация романов воспроизведёт задокументированный раздел труда по содержанию, "
                f"но идиолектный сигнал двух соавторов на служебных словах метод не различает. " + caveat)
    return (f"work-level: служебные слова — средняя доля {rf}, перестановка ярлыков работ p={pf} ({methf}, пол "
            f"{floorf}); char-3gram — средняя доля {rc}, перестановка p={pc}. Косинус между усреднёнными "
            f"профилями {cf}. Надёжного разделения соло-рук на доступном корпусе нет. " + caveat)


if __name__ == "__main__":
    main()
