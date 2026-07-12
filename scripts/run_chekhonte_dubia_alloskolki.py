"""Финальный честный тест Dubia на ПОЛНОСТЬЮ same-edition панели «Осколков» 1884-85.

Все главные кандидаты добыты OCR «Осколков» через VertexAI (log/oskolki_pipeline.py),
дереформированы в современную орфографию — никакого register/edition home-field advantage
у одного кандидата (в отличие от chekhonte_dubia_oskolki, где same-edition был только Билибин):
  chehov   <- cand_chehonte_oskolki  (А. Чехонте)
  lejkin   <- cand_leikin_oskolki    (Н. Лейкинъ)
  bilibin  <- cand_bilibin_oskolki   (И. Грэкъ)
  alexander<- cand_alexander_oskolki  (Агаѳоподъ Единицынъ, 2 пьесы — тонкий)

ГЕЙТ: позитив-контроль = LOO по подписанным пьесам кандидатов. Если панель НЕ различает самих
авторов «Осколков» между собой — вердикту по Dubia верить нельзя. Признаки: fw+char3 и fw-only
(цели — ПСС-современная орфография, кандидаты — дереформенный OCR; fw-only устойчивее к этому).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
spec = importlib.util.spec_from_file_location("rd", ROOT / "scripts" / "run_chekhonte_dubia_oskolki.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
from _gate_metrics import (  # noqa: E402
    both_metrics,
    leave_one_work_out,
    work_permutation_p,
)

CASE = ROOT / "input_cases" / "chekhonte_dubia"
BROTHER = CASE / "brother_panel"
OUT = ROOT / "docs" / "cases" / "chekhonte_dubia_alloskolki.json"
WORD = rd.WORD


def build_alexander_oskolki() -> pathlib.Path:
    """Дереформенные копии 2 Осколки-пьес Александра (на диске они в дореформенной орфографии)."""
    src = BROTHER / "alexander_oskolki1885"
    dst = CASE / "cand_alexander_oskolki"
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.txt"):
        old.unlink()
    for f in sorted(src.glob("*.txt")):
        (dst / f.name).write_text(_dereform_text(f.read_text("utf-8", "ignore")), encoding="utf-8")
    return dst


def _dereform_text(text: str) -> str:
    for old, new in (("ѣ", "е"), ("Ѣ", "Е"), ("і", "и"), ("І", "И"),
                     ("ѳ", "ф"), ("Ѳ", "Ф"), ("ѵ", "и"), ("Ѵ", "И")):
        text = text.replace(old, new)
    return re.sub(r"[ъЪ]\b", "", text)


def panel() -> dict:
    return {
        "chehov": CASE / "cand_chehonte_oskolki",
        "lejkin": CASE / "cand_leikin_oskolki",
        "bilibin": CASE / "cand_bilibin_oskolki",
        "alexander_chekhov": build_alexander_oskolki(),
    }


def words_of(p) -> int:
    return sum(len(re.findall(WORD, t)) for t in rd.read(p))


def run(use_char3: bool, candidates: dict) -> dict:
    mystery = rd.docs_of(CASE / "mystery_prose.txt")
    vec, docvecs, cents = rd.make_model(mystery, candidates, use_char3)
    agg = rd.sims([vec(t) for t in mystery], cents)
    files = sorted((CASE / "texts").glob("*.txt"))
    rows = []
    for f in files:
        txt = f.read_text("utf-8", "ignore")
        if len(re.findall(r"[А-Яа-яЁёA-Za-z]+", txt)) >= 300:
            s = rd.sims([vec(txt)], cents)
            rows.append({"text": f.stem, "top": s[0]["candidate"], "second": s[1]["candidate"],
                         "margin": round(s[0]["cos"] - s[1]["cos"], 4)})
    return {
        "pooled_prose": agg,
        "train_centroid_weighting": "equal_work_direction_after_within_work_chunk_mean_l2",
        "positive_control_loo": rd.loo(docvecs),
        "eligible_per_text": rows,
    }


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def robustness(candidates: dict, use_char3: bool, rng, iters: int = 2000) -> dict:
    """Честная устойчивость по-текстово: (1) bootstrap центроидов кандидатов -> доля прогонов, где цель
    осталась за тем же автором; (2) permutation-тест гейта LOO; (3) distinctiveness = cos(winner)-
    cos(средний центроид кандидатов): <=0 значит «победитель» лишь самый ЦЕНТРАЛЬНЫЙ, а не различимый."""
    mystery = rd.docs_of(CASE / "mystery_prose.txt")
    vec, docvecs, cents = rd.make_model(mystery, candidates, use_char3)
    names = list(docvecs)
    targets = {f.stem: vec(f.read_text("utf-8", "ignore"))
               for f in sorted((CASE / "texts").glob("*.txt"))
               if len(re.findall(r"[А-Яа-яЁёA-Za-z]+", f.read_text("utf-8", "ignore"))) >= 300}
    # (1) bootstrap-стабильность
    from collections import Counter
    stab = {t: Counter() for t in targets}
    arr = {n: rd.work_centroids(docvecs[n]) for n in names}
    for _ in range(iters):
        bc = {n: _unit(arr[n][rng.integers(0, len(arr[n]), len(arr[n]))].mean(0)) for n in names}
        for t, tv in targets.items():
            stab[t][max(bc, key=lambda n: float(np.dot(_unit(tv), bc[n])))] += 1
    boot = {t: {"winner": c.most_common(1)[0][0],
                "p_winner": round(c.most_common(1)[0][1] / iters, 3),
                "p_chehov": round(c.get("chehov", 0) / iters, 3)} for t, c in stab.items()}
    # (3) distinctiveness
    mean_cent = _unit(np.mean([cents[n] for n in names], axis=0))
    dist = {}
    for t, tv in targets.items():
        tvn = _unit(tv)
        win = max(cents, key=lambda n: float(np.dot(tvn, cents[n])))
        dist[t] = round(float(np.dot(tvn, cents[win]) - np.dot(tvn, mean_cent)), 4)
    # (2) Work-LOO gate and (3) work-label permutation use the same
    # equal-direction train estimator as the maintained case framework.
    gate_data = [
        (name, work, vector)
        for name in names
        for work, vector in zip(docvecs[name].work_ids, docvecs[name])
    ]
    wcp, _confusion, _works = leave_one_work_out(gate_data, names)
    gate_metrics = both_metrics(wcp, names)
    per_class = gate_metrics["work_recall"]
    macro_recall = gate_metrics["work_macro_recall"]
    p, perm_method, perm_floor = work_permutation_p(
        gate_data, lambda author: author, names, n_random=iters
    )
    # (4) атрибуция ВСЕХ Dubia (не только 5 длинных) + базовый уровень Чехова
    from math import comb
    from collections import Counter as _C
    all_win = _C()
    for f in sorted((CASE / "texts").glob("*.txt")):
        tvn = _unit(vec(f.read_text("utf-8", "ignore")))
        all_win[max(cents, key=lambda n: float(np.dot(tvn, cents[n])))] += 1
    n_all = sum(all_win.values())
    chehov_share = all_win.get("chehov", 0) / n_all
    # биномиальный тест: >=4 из 5 длинных к Чехову при базовом уровне chehov_share
    base_p = round(sum(comb(5, i) * chehov_share ** i * (1 - chehov_share) ** (5 - i)
                       for i in range(4, 6)), 4)
    # (5) частота «не» у кандидатов «Чехов vs Билибин»
    def _ne(path):
        ne = tot = 0
        for txt in rd.read(path):
            ws = re.findall(r"[а-яё]+", txt.lower())
            tot += len(ws)
            ne += ws.count("не")
        return round(ne / tot, 4) if tot else None
    return {
        "feature": "fw_only" if not use_char3 else "fw_char3",
        "leak_free": not use_char3,
        "_leak_note": ("fw_only — фикс-список служебных слов, рефита нет, leave-one-out leak-free. "
                       "char-3gram словарь строится на всех данных, включая held-out, поэтому fw_char3 "
                       "LOO/permutation — диагностические, не формальные."),
        "per_class_recall": per_class,
        "macro_recall": macro_recall,
        "train_centroid_weighting": "equal_work_direction_after_within_work_chunk_mean_l2",
        "bootstrap_stability": boot,
        "distinctiveness_vs_central": dist,
        "gate_permutation": {
            "observed_work_macro_recall": macro_recall,
            "method": perm_method,
            "exact_floor": perm_floor,
            "p_value": p,
        },
        "all_27_attribution": {"winners": dict(all_win.most_common()), "n": n_all,
                               "chehov_share": round(chehov_share, 4),
                               "binom_p_ge4of5_chehov": base_p},
        "ne_frequency": {"chehov": _ne(candidates["chehov"]), "bilibin": _ne(candidates["bilibin"])},
    }


def main() -> None:
    cands = panel()
    sizes = {n: words_of(p) for n, p in cands.items()}
    report = {
        "case": "chekhonte_dubia_alloskolki",
        "title": "Dubia на полностью same-edition панели «Осколков» 1884-85 (все кандидаты получены через VertexAI-OCR)",
        "candidate_words": sizes,
        "note": (
            "Все кандидаты — проза журнала «Осколки» в режиме same-edition (ни у одного кандидата нет "
            "register-преимущества). Цели — в ПСС-современной орфографии. Формальный (leak-free) признак — "
            "fw_only: фикс-список служебных слов, без рефита словаря. fw_char3 строит словарь char-3gram "
            "на всех данных, включая held-out, поэтому его LOO/permutation — диагностические, не "
            "формальные. Гейт читать по macro-recall (raw accuracy прячет неузнанные классы)."
        ),
        "panels": {},
    }
    for feat, uc3 in (("fw_char3", True), ("fw_only", False)):
        report["panels"][feat] = run(uc3, cands)
    rng = np.random.default_rng(20260630)
    report["robustness_fw_only"] = robustness(cands, False, rng)
    r = report["robustness_fw_only"]
    a27, ne = r["all_27_attribution"], r["ne_frequency"]
    loo_pct = int(round(r["gate_permutation"]["observed_work_macro_recall"] * 100))
    perm_p = r["gate_permutation"]["p_value"]
    report["verdict"] = (
        f"Все авторы взяты из одного журнала «Осколки» 1883-1885 — ни у кого нет преимущества «своего» "
        f"издания (точные числа в robustness_fw_only). Метод плохо различает самих авторов «Осколков»: "
        f"при проверке с поочерёдным исключением текстов он угадывает автора лишь в {loo_pct}% случаев "
        f"(work-label permutation p={perm_p}). Александра он не узнаёт совсем "
        f"({r['per_class_recall']['alexander_chekhov']} текста) — выводы про Александра не делаем. Из пяти "
        f"длинных спорных текстов к Чехову устойчиво (при любых случайных пересборках выборки) тянется "
        f"только «Мачеха». «Корреспонденции» и «Ревнивый муж» формально тоже ближе к Чехову, но лишь "
        f"потому, что профиль Чехова — самый «усреднённый»: к нему стягивается всё подряд; "
        f"собственного чеховского сходства у них нет. «Моя семья» — на грани (чуть качнёшь выборку — "
        f"меняется). «Среди милых москвичей» устойчиво ближе к Билибину, и это расходится с академическим "
        f"собранием (ПСС относит текст к Чехову). Если взять все {a27['n']} спорных текстов, большинство "
        f"тянется к Билибину ({a27['winners'].get('bilibin')} против {a27['winners'].get('chehov')} к "
        f"Чехову). Главное, чем Чехов отличается от Билибина, — частота слова «не» (у Чехова "
        f"{ne['chehov']}, у Билибина {ne['bilibin']}). Итог: этот спор набор текстов не решает. Уверенно "
        f"опознан один текст к Чехову («Мачеха») и один к Билибину («Среди милых москвичей», вопреки "
        f"собранию). Это показывает предел метода на близких авторах одного журнала, а не разгадку."
    )
    report["confidence"] = "низкая"
    OUT.write_text(dumps_strict(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}")
    print("слов у кандидатов:", sizes)
    for feat in report["panels"]:
        p = report["panels"][feat]
        print(f"\n== {feat} | позитив-контроль LOO = {p['positive_control_loo']['accuracy']} ==")
        print("  пул:", ", ".join(f"{r['candidate']}={r['cos']}" for r in p["pooled_prose"][:4]))
        for r in p["eligible_per_text"]:
            print(f"    {r['text'][:32]:32s} -> {r['top']:12s} (2-й {r['second']}, зазор {r['margin']:+.4f})")


if __name__ == "__main__":
    main()
