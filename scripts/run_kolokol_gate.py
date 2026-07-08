"""Позитив-контроль кейса kolokol_herzen_ogaryov (проверка выполнимости).

Главный вопрос ПЕРЕД любой атрибуцией передовых «Колокола»: разделяет ли панель самих авторов —
подписанную публицистику Герцена и Огарёва. Герцен и Огарёв десятилетиями правили тексты друг друга,
поэтому это make-or-break: если их подписанная проза не разделяется, кейс закрывается на пороге.

Признак — служебные слова (leak-free: фикс-список, словарь не учится); char-3gram как вторая,
зависимая от данных проверка. Объёмы резко неравны (Герцен ~159k, Огарёв ~14k), поэтому главная
метрика — leave-one-WORK-out: куски каждой работы классифицируются по центроидам из ОСТАЛЬНЫХ работ
(центроид автора при тесте его работы её НЕ включает). Это устойчиво к корреляции кусков внутри одной
работы и к перекосу объёма (центроид — среднее, не зависит от числа кусков). Плюс уравнивание объёма
и проверка центральности.

Оговорка: у Огарёва чистым текстом по сути одна крупная работа («Моя исповедь»), поэтому позитивный
исход — НЕОБХОДИМОЕ, но не достаточное условие (может мерить работу, а не руку); отрицательный —
закрывает кейс.
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

CASE = ROOT / "input_cases" / "kolokol_herzen_ogaryov"
OUT = ROOT / "docs" / "cases" / "kolokol_herzen_ogaryov.json"
DIRS = {
    "herzen": [CASE / "herzen_publicistic", CASE / "herzen_kolokol"],
    "ogaryov": [CASE / "ogaryov_publicistic", CASE / "ogaryov_wikisource"],
}
# «Письмо из провинции» (Колокол 1860, подпись «Русский человек») в части традиции приписывается
# Добролюбову/Чернышевскому — исключаем из якорей Огарёва, чтобы не загрязнять контроль.
EXCLUDE = {"письмо_из_провинции"}  # stem кириллический (Викитека-слаг), иначе исключение молча не сработает
WORD = r"[а-яёА-ЯЁ]+"
WIN = 600  # окно в словах


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def chunks_by_work(paths: list[pathlib.Path]) -> list[tuple[str, str]]:
    """[(work_name, chunk_text)] — каждая работа нарезана на окна WIN слов; читает несколько папок."""
    out = []
    for path in paths:
        for f in sorted(path.glob("*.txt")):
            if f.stem in EXCLUDE:
                continue
            w = f.read_text("utf-8", "ignore").split()
            for i in range(0, len(w), WIN):
                piece = " ".join(w[i:i + WIN])
                if len(piece.split()) >= WIN // 2:  # последний огрызок < половины окна отбросить
                    out.append((f.stem, piece))
    return out


def build_vectorizer(all_texts: list[str], use_char3: bool):
    fw = sorted(function_words("ru"))
    fwi = {w: i for i, w in enumerate(fw)}
    top3, t3i = [], {}
    if use_char3:
        grams: dict[str, int] = {}
        for text in all_texts:
            flat = re.sub(r"\s+", " ", text.lower())
            for i in range(max(len(flat) - 2, 0)):
                grams[flat[i:i + 3]] = grams.get(flat[i:i + 3], 0) + 1
        top3 = [g for g, _ in sorted(grams.items(), key=lambda kv: kv[1], reverse=True)[:800]]
        t3i = {g: i for i, g in enumerate(top3)}

    def vec(text: str) -> np.ndarray:
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


def evaluate(use_char3: bool) -> dict:
    # все куски с метками (author, work)
    data = []  # (author, work, vec)
    raw = {a: chunks_by_work(p) for a, p in DIRS.items()}
    all_texts = [t for a in raw for _, t in raw[a]]
    vec = build_vectorizer(all_texts, use_char3)
    for a in raw:
        for work, text in raw[a]:
            data.append((a, work, vec(text)))
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

    # уравнивание объёма: равное число кусков на автора, leave-one-chunk-out внутри подвыборки, 300 раз
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

    # перестановка ярлыков работ на work-level метрике; exact-enumeration при малом числе работ.
    perm_p, perm_method, perm_floor = work_permutation_p(data, lambda a: a, authors)

    # центральность/различимость: cos между центроидами авторов (высокий -> руки сближены)
    full_cent = {a: centroid([v for au, _, v in data if au == a]) for a in authors}
    cross_cos = round(float(np.dot(full_cent["herzen"], full_cent["ogaryov"])), 4)

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
        "confusion": confusion,
        "chunks": {a: len(byauthor[a]) for a in authors},
        "works": {a: works[a] for a in authors},
    }


def main() -> None:
    fw_only = evaluate(use_char3=False)
    fw_char3 = evaluate(use_char3=True)
    # gate_pass требует НАДЁЖНОЙ (>=0.80) И значимой разделимости на устойчивой к корреляции метрике.
    sep_significant = fw_only["work_level_permutation_p"] <= 0.05
    strong = bool(fw_only["macro_recall"] >= 0.80)  # надёжность — по work-out macro; within_work не в решении (утечка)
    gate_pass = bool(strong and sep_significant)
    # асимметрия (одна сторона тянет всё) = вырожденный центроид/центральность при тонком корпусе.
    rh = fw_only["per_author_recall"]["herzen"]
    ro = fw_only["per_author_recall"]["ogaryov"]
    h_rec = int(rh.split("/")[0]) / max(int(rh.split("/")[1]), 1)
    o_rec = int(ro.split("/")[0]) / max(int(ro.split("/")[1]), 1)
    asymmetric = bool(abs(h_rec - o_rec) >= 0.4)
    if gate_pass:
        status = "passed"
    elif asymmetric and not sep_significant:
        status = "blocked_insufficient_independent_corpus"
    elif sep_significant:
        status = "weak_separation_below_threshold"  # значимо, но ниже планки надёжности
    else:
        status = "inseparable_hands"
    report = {
        "case": "kolokol_herzen_ogaryov",
        "stage": "feasibility_gate",
        "title": "«Колокол» (1857-1867): проверка выполнимости — разделяет ли панель Герцена и Огарёва",
        "status": status,
        "candidates": ["А. И. Герцен", "Н. П. Огарёв"],
        "feature": ("формальный признак — служебные слова (предлоги, союзы, частицы; тему не выдают, словарь "
                    "не учится = leak-free). char-3gram — ЗАВИСИМАЯ ОТ ДАННЫХ диагностика (словарь top-800 "
                    "учится на всех текстах, ловит тему), не формальное доказательство; и в векторе он идёт "
                    "ВМЕСТЕ со служебными словами, то есть не независим от них"),
        "note": ("Две РАЗНЫЕ метрики: work_macro_recall (один удержанный текст = один голос, большинство его "
                 "кусков) — к ней применяется порог 0.80; chunk_weighted_recall (каждый кусок — голос) НЕ "
                 "work-level (коррелированные куски = много голосов) и приводится как диагностика. Обучение — "
                 "leave-one-WORK-out (центроид автора при тесте его работы её не включает). Перестановка "
                 "ярлыков на уровне работ; при малом числе работ — точное перечисление, иначе случайно."),
        "fw_only": fw_only,
        "fw_char3": fw_char3,
        "gate_pass": gate_pass,
        "verdict": _verdict(status, fw_only, fw_char3),
        "caveat": ("Корпус: Герцен ~177k слов (на 82% — «Письма из Франции и Италии» 1846-52, центроид "
                   "доминирован одной работой и смещён по эпохе/регистру относительно цели 1857-67), Огарёв "
                   "~59k в 15 работах (Викитека + «Моя исповедь»). «Письмо из провинции» (1860, традицией "
                   "приписываемое Добролюбову/Чернышевскому) исключено из якорей Огарёва. Косинус профилей "
                   "высок, но work-голос их различает; часть сигнала может нести эпоху/регистр Герцена."),
        "next": ["Слабый сигнал на границе значимости: укрепить эталон Герцена колокольной публицистикой "
                 "1857-1867 того же регистра (распознать его статьи «Колокола» из 30-томника АН СССР на "
                 "imwerden/feb-web) — отделить остаточный идиолект от смещения эпохи/регистра.",
                 "Если после уравнивания эпохи и регистра перевес устоит — применять только к ОДНОАВТОРСКИМ "
                 "передовым как тай-брейк к филологии, исключив соавторские по комментарию 30-томника, никогда "
                 "как разрешение атрибуции.",
                 "Если перевес исчезнет при выровненной эпохе — закрыть кейс как честный предел метода на "
                 "сросшихся руках."],
        "data_status": ("Все тексты — общественное достояние (Герцен †1870, Огарёв †1877). Якоря Герцена и "
                        "часть Огарёва — чистый текст с az.lib.ru; register/era-matched колокольная "
                        "публицистика 1857-1868 — Викитека через action=parse (разворачивает транслюзию "
                        "Page-пространства). Сырьё пишется в gitignored input_cases/kolokol_herzen_ogaryov/; "
                        "в git — только скрипты добычи (scripts/fetch_kolokol*.py) и этот JSON."),
        "sources": [
            {"cite": "А. И. Герцен, публицистика 1846-1852 («Письма из Франции и Италии» и др.) — якорь "
                     "herzen_publicistic", "url": "http://az.lib.ru/g/gercen_a_i/"},
            {"cite": "А. И. Герцен, колокольная/полярнозвёздная публицистика 1857-1868 («Концы и начала», "
                     "«Письмо к Александру II» и др.) — register/era-matched якорь herzen_kolokol, Викитека "
                     "action=parse", "url": "https://ru.wikisource.org/w/api.php"},
            {"cite": "Н. П. Огарёв, «Моя исповедь» (1862) и предисловия — якорь ogaryov_publicistic",
             "url": "http://az.lib.ru/o/ogarew_n_p/"},
            {"cite": "Н. П. Огарёв, подписанная публицистика 1857-1867 («Колокол», «Общее вече», «Голоса из "
                     "России») — якорь ogaryov_wikisource, Викитека action=parse",
             "url": "https://ru.wikisource.org/w/api.php"},
            {"cite": "Опорная PD-публикация для сверки колокольных атрибуций (Лемке vs академический "
                     "30-томник): А. И. Герцен. Собрание сочинений в 30 томах. М.: АН СССР, 1954-1966 — "
                     "imwerden.de, feb-web.ru (планируемое укрепление эталона Герцена)",
             "url": "https://imwerden.de/"},
        ],
        "analysis_command": "PYTHONPATH=src python3 scripts/run_kolokol_gate.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}")
    for tag, r in (("fw_only", fw_only), ("fw_char3", fw_char3)):
        print(f"  [{tag}] recall {r['per_author_recall']} macro {r['macro_recall']} | "
              f"balanced {r['within_work_chunk_accuracy']} | perm_p {r['work_level_permutation_p']} | "
              f"cos(H,O) {r['cross_author_centroid_cos']} | chunks {r['chunks']}")
    print("GATE_PASS:", gate_pass)
    print("VERDICT:", report["verdict"])


def _verdict(status, fw, c3) -> str:
    wr, wm = fw["per_author_recall"], fw["macro_recall"]        # work-level (один текст = один голос)
    cw = fw["chunk_weighted_recall"]                            # диагностика (куски как голоса)
    p, meth = fw["work_level_permutation_p"], fw["permutation_method"]
    cos = fw["cross_author_centroid_cos"]
    metrics = (f"На корректной метрике «один текст = один голос» (work-level) верные опознания работ {wr}, "
               f"в среднем {wm}; chunk-weighted доля (каждый кусок — голос) {cw} (длинные работы весят больше "
               f"по числу кусков, в этом кейсе ниже) приводится как диагностика. Перестановка ярлыков работ p={p} ({meth}). Косинус "
               f"между усреднёнными профилями {cos} (профили близки). ")
    cav = ("Оговорки: эталон Герцена смещён по эпохе и регистру (основной объём — «Письма из Франции и "
           "Италии» 1846-52, не колокольная публицистика 1857-67), поэтому часть сигнала может нести "
           "эпоху/регистр; передовые «Колокола» частью соавторские — там единственного автора нет. "
           "Атрибуцию вести только по одноавторским передовым, как тай-брейк к филологии.")
    if status == "passed":
        return ("Под work-level панель РАЗДЕЛЯЕТ подписанных Герцена и Огарёва выше порога надёжной "
                "атрибуции 0.80. " + metrics + cav)
    return ("Под work-level разделение ниже порога/незначимо. " + metrics + cav)


if __name__ == "__main__":
    main()
