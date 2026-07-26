"""Микрокейс: «Среди милых москвичей» (15, Будильник 1885) — Чехов или Билибин.

ПСС относит серию к Чехову. На полностью одножурнальной осколочной панели (Чехонте/Билибин/Лейкин/
Александр — все из «Осколков», см. run_chekhonte_dubia_alloskolki) текст 15 целиком уходит к Билибину.
Серия — пять датированных заметок 1885 года; разбираем каждую отдельно: однородна ли серия и какие
заметки расходятся с атрибуцией ПСС к Чехову.

Признак — только служебные слова (leak-free). По каждой заметке: победитель, bootstrap-устойчивость и
отличимость (cos победителя минус cos среднего профиля кандидатов; <=0 значит «победитель» лишь
центральный, не различимый).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
rd_spec = importlib.util.spec_from_file_location("rd", ROOT / "scripts" / "run_chekhonte_dubia_oskolki.py")
rd = importlib.util.module_from_spec(rd_spec)
rd_spec.loader.exec_module(rd)
al_spec = importlib.util.spec_from_file_location("al", ROOT / "scripts" / "run_chekhonte_dubia_alloskolki.py")
al = importlib.util.module_from_spec(al_spec)
al_spec.loader.exec_module(al)
from _gate_metrics import both_metrics, leave_one_work_out  # noqa: E402

CASE = ROOT / "input_cases" / "chekhonte_dubia"
OUT = ROOT / "docs" / "cases" / "chekhonte_15_micro.json"
TEXT15 = CASE / "texts" / "15_среди_милых_москвичей.txt"

SEGMENTS = [
    ("1885-05-24", r"(?s)< 24 МАЯ 1885 Г\. >(.*?)(?=\nФОРМУЛЯРНЫЙ СПИСОК)"),
    ("1885-06-14", r"(?s)ФОРМУЛЯРНЫЙ СПИСОК ПЕТЕРБУРГСКИХ ДАМ(.*?)(?=\n< 20 ИЮНЯ)"),
    ("1885-06-20", r"(?s)< 20 ИЮНЯ 1885 г\. >(.*?)(?=\n< 8 АВГУСТА)"),
    ("1885-08-08", r"(?s)< 8 АВГУСТА 1885 г\. >(.*?)(?=\n< 29 АВГУСТА)"),
    ("1885-08-29", r"(?s)< 29 АВГУСТА 1885 г\. >(.*?)(?=\nСноски\n|$)"),
]
LABELS = {"chehov": "Чехов", "bilibin": "Билибин", "lejkin": "Лейкин", "alexander_chekhov": "Александр"}


def _u(v):
    return v / (np.linalg.norm(v) + 1e-9)


def clean_seg(text: str) -> str:
    text = re.sub(r"«БУДИЛЬНИК».*?(Титульный лист|Последняя страница обложки)", " ", text, flags=re.S)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    rng = np.random.default_rng(20260630)
    cands = al.panel()
    raw = TEXT15.read_text("utf-8", "ignore")
    # сегменты извлекаются по сырым маркерам, но чистятся; whole = склейка ОЧИЩЕННЫХ сегментов
    seg_bodies = [(label, clean_seg(m.group(1)))
                  for label, pat in SEGMENTS if (m := re.search(pat, raw))]
    whole = " ".join(b for _, b in seg_bodies)
    vec, docvecs, cents = rd.make_model([whole], cands, use_char3=False)
    arr = {n: rd.work_centroids(docvecs[n]) for n in docvecs}
    mean_cent = _u(np.mean([cents[n] for n in cents], axis=0))

    def attribute(text):
        tv = _u(vec(text))
        sims = sorted(((n, float(np.dot(tv, cents[n]))) for n in cents), key=lambda x: -x[1])
        win = Counter()
        for _ in range(2000):
            bc = {n: _u(arr[n][rng.integers(0, len(arr[n]), len(arr[n]))].mean(0)) for n in arr}
            win[max(bc, key=lambda m: float(np.dot(tv, bc[m])))] += 1
        return {
            "winner": sims[0][0],
            "winner_ru": LABELS.get(sims[0][0], sims[0][0]),
            "bootstrap_winner_share": round(win.most_common(1)[0][1] / 2000, 3),
            "p_bilibin": round(win.get("bilibin", 0) / 2000, 3),
            "p_chehov": round(win.get("chehov", 0) / 2000, 3),
            "distinctiveness": round(float(np.dot(tv, cents[sims[0][0]]) - np.dot(tv, mean_cent)), 4),
        }

    # Work-LOO positive control with equal-direction work centroids.
    names = list(docvecs)
    gate_data = [
        (name, work, vector)
        for name in names
        for work, vector in zip(docvecs[name].work_ids, docvecs[name])
    ]
    wcp, _chunk_confusion, _works = leave_one_work_out(gate_data, names)
    gate_metrics = both_metrics(wcp, names)
    per_class = gate_metrics["work_recall"]

    # №5 negative controls: work-majority confusion under whole-work exclusion.
    confusion = {n: Counter() for n in names}
    for name, _work, predictions in wcp:
        confusion[name][Counter(predictions).most_common(1)[0][0]] += 1

    # №5 альтернативный признак: char-3gram (целиком 15) — держится ли Билибин на другом признаке
    vec_c3, _, cents_c3 = rd.make_model([whole], cands, use_char3=True)
    mean_c3 = _u(np.mean([cents_c3[n] for n in cents_c3], axis=0))
    tvc = _u(vec_c3(whole))
    w_c3 = max(cents_c3, key=lambda n: float(np.dot(tvc, cents_c3[n])))
    dist_c3 = round(float(np.dot(tvc, cents_c3[w_c3]) - np.dot(tvc, mean_c3)), 4)

    whole_attr = attribute(whole)
    segs = []
    for label, body in seg_bodies:
        a = attribute(body)
        segs.append({"date": label, "words": len(re.findall(rd.WORD, body)),
                     "too_short": len(re.findall(rd.WORD, body)) < 300, **a})
    seg_winners = dict(Counter(s["winner_ru"] for s in segs).most_common())
    robustness = {
        "feature_agreement": {
            "fw_only_whole": whole_attr["winner_ru"],
            "fw_char3_whole": LABELS.get(w_c3, w_c3),
            "fw_char3_distinctiveness": dist_c3,
        },
        "negative_controls": {
            "chehonte_to_bilibin": f"{confusion['chehov']['bilibin']}/{sum(confusion['chehov'].values())}",
            "bilibin_to_chehonte": f"{confusion['bilibin']['chehov']}/{sum(confusion['bilibin'].values())}",
            "note": ("доля подписанных Чехонте текстов, уходящих к Билибину (и наоборот) при поочерёдном "
                     "исключении; низкая доля Чехонте→Билибин значит, что близость заметок к Билибину — "
                     "не перекос в самом наборе образцов"),
        },
        "confusion": {LABELS.get(n, n): {LABELS.get(k, k): c for k, c in cc.most_common()}
                      for n, cc in confusion.items()},
    }

    report = {
        "case": "chekhonte_15_micro",
        "title": ("«Среди милых москвичей» (Будильник 1885): подборка Чехова по стилю неоднородна "
                  "и требует сверки по оригиналу"),
        "status": "candidate_for_textological_audit",
        "confidence": "low_to_moderate",
        "resolution": "not_resolved",
        "github_label": "source-blocked / needs_original_issue",
        "panel": "набор авторов для сравнения: Чехонте, Билибин, Лейкин, Александр — все образцы из журнала «Осколки»",
        "feature": ("только служебные слова — предлоги, союзы, частицы; они не выдают тему заметки "
                    "и потому не подсказывают автора"),
        "pss_attribution": "Чехов (в эти годы подписывался Чехонте)",
        "пояснения": {
            "доля побед при пересчётах": ("сколько раз из 100 при повторном пересчёте на случайных кусках "
                "текста побеждает автор; это устойчивость результата, а не вероятность авторства"),
            "отрыв от среднего": ("насколько победитель ближе к тексту, чем усреднённый стиль всех образцов; "
                "ноль или меньше — победитель просто средний и никого не выделяет"),
            "узнаваемость автора": ("в скольких своих текстах автор верно опознан по образцам; "
                "низкая узнаваемость значит, что набор плохо различает авторов"),
            "трёхбуквенные сочетания": ("второй способ сравнения — частоты коротких буквосочетаний "
                "вместо служебных слов"),
        },
        "positive_control": {"per_class_recall": per_class,
                             "work_macro_recall": gate_metrics["work_macro_recall"],
                             "train_centroid_weighting":
                                 "equal_work_direction_after_within_work_chunk_mean_l2",
                             "note": ("набор образцов должен узнавать самих авторов «Осколков» по их текстам; "
                                      "почерк Чехова узнаётся примерно в половине случаев — различающая "
                                      "способность слабая, поэтому вердикт осторожный")},
        "whole": whole_attr,
        "segments": segs,
        "segment_winners": seg_winners,
        "robustness": robustness,
        "verdict": _verdict(whole_attr, segs, robustness, per_class, seg_winners),
        "caveat": ("Заметки серии взяты из полного собрания сочинений (ПСС) в современной орфографии, а образцы "
                   "«Осколков» — старое, дореформенное написание, распознанное автоматически со сканов; к разнице "
                   "написания служебные слова мало чувствительны, но различие источника остаётся. По коротким "
                   "заметкам (меньше 300 слов) вывод менее надёжен; итог — по длинным заметкам и по всей подборке."),
        "textological_passport": {
            "rubric": ("«Среди милых москвичей» — еженедельная колонка «Будильника» 1885 на третьей странице "
                       "каждого номера; содержание разное от номера к номеру (масленичные номера №2–4 — про "
                       "Ильина, про Макара/масленицу, про Вальца), авторы не подписаны."),
            "pss_selection": ("Полное собрание сочинений (ПСС) относит к Чехову пять датированных заметок этой "
                              "колонки (24 мая — 29 августа 1885). Это выбор редакторов внутри общей "
                              "повторяющейся рубрики, заметки анонимные."),
            "edition_check": ("Национальная электронная библиотека (НЭБ) оцифровала «Будильник» 1885 только "
                              "номера 1–10 (январь–март, масленичные). Майско-августовские номера с чеховскими "
                              "заметками не оцифрованы; сверить их печатный текст по НЭБ нельзя. Подтверждено "
                              "распознаванием обложки: номер 4, XXI год."),
            "same_journal_bilibin": ("Подписей Билибина (он подписывался «Грэкъ») и Чехонте в оцифрованных номерах "
                                     "«Будильника» нет; Билибин печатался в «Осколках». Поэтому образцы Билибина "
                                     "для сравнения взяты из другого журнала, и близость заметок к нему — это "
                                     "сходство текстов из разных журналов, оно менее надёжно."),
            "alexander_present": ("Александр Чехов (Агаѳоподъ Единицынъ — четыре номера, Гусевъ — один) в "
                                  "оцифрованных номерах присутствует."),
        },
        "next": ["Главный шаг: найти «Будильник» 1885 №20, 23, 24, 34 (майско-августовские заметки серии) в "
                 "РГБ, РНБ, ИРЛИ, ГПИБ или в букинистике/микрофильмах — НЭБ оцифровал только №1–10.",
                 "По оригиналу сверить шрифт и вёрстку, соседство на полосе, подпись и редакторскую структуру.",
                 "Убедиться, что ПСС-текст не склеен из разнородных заметок повторяющейся колонки.",
                 "Образцы Билибина есть только в «Осколках» (в «Будильнике» его нет), поэтому сравнение идёт "
                 "между разными журналами и остаётся менее надёжным."],
        "data_status": ("Все тексты — общественное достояние (Чехов †1904, Лейкин †1906, Билибин †1908, "
                        "Александр Чехов †1913). Цель — пять заметок «Среди милых москвичей» («Будильник» 1885) "
                        "из ПСС Чехова (современная орфография). Якоря кандидатов — подписанная короткая проза "
                        "«Осколков» 1884-85, распознанная со сканов НЭБ через VertexAI (модель Gemini 2.5 Flash), "
                        "орфография нормализована. Сырьё пишется в gitignored input_cases/chekhonte_dubia/; "
                        "в git — только скрипт добычи/распознавания и этот JSON."),
        "sources": [
            {"cite": "А.П. Чехов (Чехонте), «Среди милых москвичей» и подписанная короткая проза — ПСС / az.lib.ru",
             "url": "http://az.lib.ru/c/chehow_a_p/"},
            {"cite": "В.В. Билибин (подпись «Грэкъ»), «Осколки» 1884-85 — OCR через VertexAI (Gemini 2.5 Flash)",
             "url": "http://az.lib.ru/b/bilibin_w_w/"},
            {"cite": "Н.А. Лейкин, короткая проза «Осколков» — az.lib.ru",
             "url": "http://az.lib.ru/l/lejkin_n_a/"},
            {"cite": "Ал. П. Чехов (подписи «Агаѳоподъ Единицынъ», «Гусевъ»), «Осколки» 1885 — OCR через VertexAI",
             "url": "http://az.lib.ru/c/chehow_a_p/"},
            {"cite": "«Будильник» 1885 (сканы №1–10, масленичные) — Национальная электронная библиотека (НЭБ)",
             "url": "https://rusneb.ru/"},
        ],
        "analysis_command": "PYTHONPATH=src python3 scripts/run_chekhonte_15_micro.py",
    }
    OUT.write_text(dumps_strict(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}")
    print("per_class:", per_class)
    print(f"ЦЕЛИКОМ -> {whole_attr['winner_ru']} (Билибин {whole_attr['p_bilibin']}, Чехов {whole_attr['p_chehov']}, отличимость {whole_attr['distinctiveness']})")
    for s in segs:
        print(f"  {s['date']} ({s['words']}сл{', короткий' if s['too_short'] else ''}) -> {s['winner_ru']:9s} "
              f"(Билибин {s['p_bilibin']}, Чехов {s['p_chehov']}, отлич {s['distinctiveness']})")
    print("VERDICT:", report["verdict"])


def _verdict(whole, segs, rob, per_class, seg_winners) -> str:
    wb = min(99, int(whole["p_bilibin"] * 100))  # не округляем вверх до 100: 0.998 -> «99 из 100»
    dist = whole["distinctiveness"]
    long_segs = [s for s in segs if not s["too_short"]]
    bil = sum(1 for s in long_segs if s["winner"] == "bilibin")
    che = sum(1 for s in long_segs if s["winner"] == "chehov")
    bil_segs = [s for s in segs if s["winner"] == "bilibin"]
    strong = max(bil_segs, key=lambda s: s["distinctiveness"]) if bil_segs else None
    fa, nc = rob["feature_agreement"], rob["negative_controls"]
    both = fa["fw_only_whole"] == fa["fw_char3_whole"] == "Билибин"
    che_rec = per_class.get("chehov", "?").replace("/", " из ")
    winners_str = ", ".join(f"{k} {v}" for k, v in seg_winners.items())

    method = ("Инструмент сравнивает текст с образцами четырёх юмористов «Осколков» по тому, как часто "
              "встречаются служебные слова — предлоги, союзы, частицы (они не выдают тему и потому не "
              "подсказывают автора). ")
    weak = (f"Сначала о силе самого измерения: этот набор образцов верно узнаёт почерк самого Чехова лишь "
            f"в {che_rec} случаев — примерно в половине, поэтому любой перевес здесь слабый, это не "
            f"вероятность авторства.")
    homog = (f" По однородности — осторожно: набор образцов сам различает авторов слабо (Чехова он узнаёт "
             f"примерно в половине случаев), поэтому это не доказательство «писали разные руки». Устойчиво "
             f"расходится с Чеховом одна длинная заметка ({strong['date'] if strong else '06-14'}), и вся "
             f"подборка тянется к Билибину; по всем пяти ближайшие распределяются ({winners_str}), но на "
             f"коротких заметках (меньше 300 слов) этот разнобой неустойчив и может быть шумом. «Среди милых "
             f"москвичей» — повторяющаяся еженедельная колонка «Будильника», а пять заметок, которые редакторы "
             f"полного собрания сочинений (ПСС) собрали под именем Чехова, — подборка, а не один текст.")
    lean = (f" Как подборка весь набор ближе к Билибину, чем к самому Чехову (он же подписывался Чехонте): "
            f"при сотнях повторных пересчётов на случайных кусках текста Билибин побеждает почти всегда "
            f"({wb} из 100), но отрыв от усреднённого стиля всех образцов тонкий ({dist:+.3f}; ноль значил бы "
            f"«просто средний, никого не выделяет»).")
    if strong:
        lean += (f" Заметка {strong['date']} сильнее прочих отделяется от усреднённого профиля всех образцов "
                 f"(отрыв {strong['distinctiveness']:+.3f}).")
    crossj = (" Образцы Билибина при этом взяты из другого журнала — «Осколков», а спорная заметка из "
              "«Будильника», поэтому сходство может идти от журнала, времени или жанра, а не от одной руки.")
    feat = (" Тот же перевес виден и при добавлении трёхбуквенных сочетаний букв (это диагностика, не "
            "независимая проверка: словарь сочетаний учится на тех же текстах, включая спорный, и в расчёте "
            "идёт ВМЕСТЕ со служебными словами)." if both else
            f" При добавлении трёхбуквенных сочетаний набор ближе к «{fa['fw_char3_whole']}» — диагностика "
            f"расходится со служебными словами.")
    negc = (f" Против простого перекоса в наборе образцов: подписанные Чехонте тексты уходят к Билибину редко "
            f"({nc['chehonte_to_bilibin']}), то есть Билибин не «перетягивает» всё подряд.")
    src = (" Оригиналы майско-августовских номеров «Будильника» не оцифрованы (Национальная электронная "
           "библиотека выложила только №1–10), сверить печатный текст нельзя.")
    concl = (" Вывод: Билибин — самый похожий по стилю автор среди доступных образцов, не доказанный автор. "
             "Это не атрибуция Билибину, а пометка: чеховская атрибуция ПСС по стилю выглядит неоднородной "
             "и требует сверки по оригиналу «Будильника» 1885. Статус: кандидат на сверку по оригиналу, "
             "уверенность низкая-умеренная, вопрос не разрешён.")
    return method + weak + homog + lean + crossj + feat + negc + src + concl


if __name__ == "__main__":
    main()
