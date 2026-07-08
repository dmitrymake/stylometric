"""Основная панель chekhonte_dubia на same-edition Билибине из «Осколков».

Ограничение: clean Билибин 7.6k слов не держит атрибуцию — панель требует same-form Билибина
>=20k. Корпус Билибина (cand_bilibin_oskolki) добыт OCR «Осколков» 1884-85 через
VertexAI (log/oskolki_pipeline.py), дереформирован в современную орфографию.

ВАЖНАЯ ОГОВОРКА (edition-asymmetry): Билибин из «Осколков» — OCR дореформенных «Осколков», а chehov/
lejkin/цели — современная орфография ПСС. Поэтому любой выигрыш Билибина может быть OCR/орфо-
артефактом. Контроль: гоняем ДВА признака — FW+char3 и FW-only (служебные слова устойчивы к OCR/
орфографии). Если вывод держится FW-only — он не орфографический.

Сравниваем три панели на pooled-прозе Dubia и подписанных текстах:
  base   — clean Билибин (cand_bilibin),
  oskolki— same-edition Билибин (cand_bilibin_oskolki),
  augment— clean + «Осколки» вместе.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from stylo.lang import function_words  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE = ROOT / "input_cases" / "chekhonte_dubia"
CHEK = ROOT / "input_cases" / "chekhonte"
OUT = ROOT / "docs" / "cases" / "chekhonte_dubia_oskolki.json"
WORD = r"[а-яёА-ЯЁ]+"

BASE = {
    "chehov": CHEK / "cand_chehov",
    "lejkin": CHEK / "cand_lejkin",
    "bilibin": CHEK / "cand_bilibin",
    "lazarev_gruzinsky": CASE / "cand_lazarev_gruzinsky",
    "alexander_chekhov": CASE / "cand_alexander_chekhov",
}
OSK = {**BASE, "bilibin": CASE / "cand_bilibin_oskolki"}
AUG = {**BASE, "bilibin": [CHEK / "cand_bilibin", CASE / "cand_bilibin_oskolki"]}


def read(path) -> list[str]:
    paths = path if isinstance(path, list) else [path]
    out = []
    for p in paths:
        files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
        out += [f.read_text("utf-8", "ignore") for f in files if f.exists()]
    return out


def docs_of(path) -> list[str]:
    out = []
    for text in read(path):
        w = text.split()
        if len(w) <= 2200:
            out.append(text)
        else:
            for i in range(0, len(w), 1500):
                out.append(" ".join(w[i:i + 1500]))
    return out


def make_model(mystery_docs: list[str], candidates: dict, use_char3: bool):
    fw = sorted(function_words("ru"))
    fwi = {w: i for i, w in enumerate(fw)}
    corpus = {n: docs_of(p) for n, p in candidates.items()}
    everything = [t for docs in corpus.values() for t in docs] + mystery_docs
    top3, t3i = [], {}
    if use_char3:
        grams: dict[str, int] = {}
        for text in everything:
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
        fwv /= np.linalg.norm(fwv) + 1e-9
        if not use_char3:
            return fwv
        flat = re.sub(r"\s+", " ", text.lower())
        c3 = np.zeros(len(top3))
        for i in range(max(len(flat) - 2, 0)):
            j = t3i.get(flat[i:i + 3])
            if j is not None:
                c3[j] += 1
        c3 /= max(len(flat) - 2, 1)
        c3 /= np.linalg.norm(c3) + 1e-9
        return np.concatenate([fwv, c3])

    docvecs = {n: [vec(t) for t in docs] for n, docs in corpus.items() if docs}
    cents = {}
    for n, v in docvecs.items():
        c = np.mean(v, axis=0)
        cents[n] = c / (np.linalg.norm(c) + 1e-9)
    return vec, docvecs, cents


def sims(vecs, cents) -> list[dict]:
    m = np.mean(vecs, axis=0)
    m /= np.linalg.norm(m) + 1e-9
    rows = [{"candidate": n, "cos": round(float(np.dot(m, c)), 6)} for n, c in cents.items()]
    return sorted(rows, key=lambda r: r["cos"], reverse=True)


def loo(docvecs) -> dict:
    names = list(docvecs)
    correct = total = 0
    for name in names:
        if len(docvecs[name]) < 2:
            continue
        for idx, v in enumerate(docvecs[name]):
            nv = v / (np.linalg.norm(v) + 1e-9)
            row = {}
            for o in names:
                tr = [x for j, x in enumerate(docvecs[o]) if not (o == name and j == idx)]
                if tr:
                    c = np.mean(tr, axis=0)
                    row[o] = float(np.dot(nv, c / (np.linalg.norm(c) + 1e-9)))
            if max(row, key=row.get) == name:
                correct += 1
            total += 1
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 4) if total else None}


def words_of(path) -> int:
    return sum(len(re.findall(WORD, t)) for t in read(path))


def run_panel(candidates: dict, use_char3: bool) -> dict:
    mystery = docs_of(CASE / "mystery_prose.txt")
    vec, docvecs, cents = make_model(mystery, candidates, use_char3)
    agg = sims([vec(t) for t in mystery], cents)
    # по каждому тексту (подходящие, >=300 слов)
    files = sorted((CASE / "texts").glob("*.txt"))
    counts = Counter()
    rows = []
    for f in files:
        txt = f.read_text("utf-8", "ignore")
        if len(re.findall(r"[А-Яа-яЁёA-Za-z]+", txt)) >= 300:
            s = sims([vec(txt)], cents)
            counts[s[0]["candidate"]] += 1
            rows.append({"text": f.stem, "top": s[0]["candidate"],
                         "second": s[1]["candidate"], "margin": round(s[0]["cos"] - s[1]["cos"], 4)})
    return {
        "pooled_prose": agg,
        "loo": loo(docvecs),
        "eligible_per_text_top_counts": dict(counts.most_common()),
        "eligible_per_text": rows,
    }


def main() -> None:
    bilibin_words = {
        "base_cand_bilibin": words_of(BASE["bilibin"]),
        "oskolki_cand_bilibin_oskolki": words_of(OSK["bilibin"]),
        "augmented": words_of(BASE["bilibin"]) + words_of(OSK["bilibin"]),
    }
    report = {
        "case": "chekhonte_dubia_oskolki",
        "title": "Основная панель Dubia перестроена на same-edition Билибине из «Осколков» (OCR через VertexAI)",
        "bilibin_corpus_words": bilibin_words,
        "panels": {},
        "caveat": (
            "Билибин из «Осколков» — это OCR через VertexAI дореформенного журнального текста, "
            "дереформированного в современную орфографию, тогда как chehov/lejkin/цели — современные "
            "переиздания ПСС. Из-за этой edition-асимметрии выигрыш Билибина может оказаться OCR/орфо-"
            "артефактом; контролем служит признак fw_only (служебные слова, устойчивые к OCR/орфографии) "
            "— доверяй выводу только в том случае, если он держится на нём."
        ),
    }
    for feat, use_c3 in (("fw_char3", True), ("fw_only", False)):
        report["panels"][feat] = {
            "base": run_panel(BASE, use_c3),
            "oskolki": run_panel(OSK, use_c3),
            "augment": run_panel(AUG, use_c3),
        }
    report["verdict"] = (
        "Same-edition Билибин из «Осколков» (7.8k слов, регистр журнальной юморески совпадает с Dubia) "
        "сдвигает POOLED-прозу Dubia в сторону Билибина при обоих наборах признаков (базовый near-tie "
        "alexander~bilibin 0.956 -> augment bilibin 0.976 / fw-only 0.980). НО это не чистая "
        "атрибуция Билибину: (1) лишь ОДИН текст Dubia, 15_среди_милых_москвичей, уходит к Билибину при "
        "ОБОИХ признаках — и char-3gram, И устойчивом к OCR/орфографии признаке служебных слов (он всегда "
        "был кандидатом-Билибиным) — это единственный устойчивый выигрыш; (2) 08_корреспонденции и "
        "12_моя_семья на char-3-граммах идут к Билибину, на служебных словах — к Чехову, "
        "так что их тяготение к Билибину — артефакт char-поверхности/издания, а не "
        "авторства; (3) по служебным словам per-text Чехов у 4 из 5. Отрывы крошечные "
        "(+0.001..+0.034) на насыщенных косинусах ~0.85-0.95. ОГОВОРКА О СПРАВЕДЛИВОСТИ: только "
        "Билибин совпадает с Dubia по изданию/регистру (same-edition); Чехов (ПСС) и Александр "
        "(антология 1904) — нет, поэтому у Билибина регистровое преимущество своего поля. Для чистого "
        "теста нужно добрать и ДРУГИХ кандидатов (Чехонте, Лейкина) тоже same-edition из «Осколков». "
        "Устойчива одна атрибуция 15->Билибин; в остальном атрибуция Dubia "
        "атрибуция Dubia определяется регистром/изданием, а не решена."
    )
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}")
    for feat in report["panels"]:
        print(f"\n== {feat} ==")
        for pname, pdata in report["panels"][feat].items():
            top = pdata["pooled_prose"][:3]
            print(f"  {pname:8s} LOO={pdata['loo']['accuracy']} pooled: " +
                  ", ".join(f"{r['candidate']}={r['cos']}" for r in top) +
                  f" | per-text {pdata['eligible_per_text_top_counts']}")


if __name__ == "__main__":
    main()
