"""Атрибуция спорного фельетона «Н.Н.» (Петербургская летопись, 13.04.1847) на панели, прошедшей
позитив-контроль (см. run_petersburg_chronicle_gate). Классы: художественная проза и публицистика
Достоевского, Плещеев, Соллогуб.

Целиком и по сегментам (для версии о соавторстве Достоевского с Плещеевым) + bootstrap-устойчивость.
Признак — только служебные слова (без утечки). Оговорка: эталон Плещеева — рассказы «Житейские сцены»
(не фельетон), поэтому Плещеева надёжно отличить нельзя; вывод по нему — «не исключён», не «автор».
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g", ROOT / "scripts" / "run_petersburg_chronicle_gate.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
rd, CANDS, DOST, CASE = g.rd, g.CANDS, g.DOST, g.CASE
OUT = ROOT / "docs" / "cases" / "dostoevsky_petersburg_chronicle.json"
NN = CASE / "target_NN" / "nn_13aprelya_1847.txt"


def _u(v):
    return v / (np.linalg.norm(v) + 1e-9)


def segments(text: str, n: int = 4) -> list[str]:
    w = re.findall(r"\S+", text)
    step = max(len(w) // n, 1)
    return [" ".join(w[i:i + step]) for i in range(0, len(w), step)][:n]


def main() -> None:
    rng = np.random.default_rng(20260630)
    nn_text = NN.read_text("utf-8", "ignore")
    vec, docvecs, cents = rd.make_model([nn_text], CANDS, use_char3=False)
    arr = {n: np.array(docvecs[n]) for n in docvecs}
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
            "cos": {n: round(c, 4) for n, c in sims},
            # отличимость: cos победителя минус cos среднего профиля кандидатов; <=0 значит
            # «победитель» лишь самый центральный (крупнейший корпус), а не различимый.
            "distinctiveness": round(float(np.dot(tv, cents[sims[0][0]]) - np.dot(tv, mean_cent)), 4),
            "bootstrap": {n: round(c / 2000, 3) for n, c in win.most_common()},
            "bootstrap_dostoevsky_share": round(sum(c for n, c in win.items() if n in DOST) / 2000, 3),
            "bootstrap_pleshcheev_share": round(win.get("pleshcheev", 0) / 2000, 3),
        }

    whole = attribute(nn_text)
    segs = [{"segment": i + 1, "words": len(re.findall(rd.WORD, s)), **attribute(s)}
            for i, s in enumerate(segments(nn_text, 4))]
    seg_winners = dict(Counter(s["winner"] for s in segs).most_common())

    # дописать в JSON кейса (gate уже записан run_petersburg_chronicle_gate)
    report = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    fd_dist = report.get("FD_feuilletons_attribution", {}).get("whole_distinctiveness")
    report["NN_attribution"] = {
        "target": "Петербургская летопись <13 апреля> (Н.Н.), 1847, спорный",
        "words": len(re.findall(rd.WORD, nn_text)),
        "candidates_note": ("Достоевский представлен двумя классами (проза + публицистика); эталон "
                            "Плещеева — рассказы «Житейские сцены», не фельетон, поэтому различение "
                            "Плещеева ненадёжно (вывод по нему — «не исключён», не «автор»)."),
        "whole": whole,
        "segments": segs,
        "segment_winners": seg_winners,
        "verdict": _verdict(whole, segs, seg_winners, fd_dist),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано {OUT.relative_to(ROOT)}")
    print(json.dumps(report["NN_attribution"], ensure_ascii=False, indent=1))


def _verdict(whole, segs, seg_winners, fd_dist=None) -> str:
    w = whole["winner"]
    dost = whole["bootstrap_dostoevsky_share"]
    ple = whole["bootstrap_pleshcheev_share"]
    dist = whole["distinctiveness"]
    fd = f" (у контрольных Ф.Д. отличимость {fd_dist:+.4f})" if fd_dist is not None else ""
    seg_other = sorted((n for n in seg_winners if n not in DOST), key=lambda n: -seg_winners[n])
    other = seg_other[0] if seg_other else None
    seg_tail = (f" По сегментам часть тянет к «{other}» (регистр фельетона), но не к Плещееву — "
                f"положительных следов руки Плещеева нет." if other else "")
    if w in DOST and dist <= 0:
        return (f"Целиком Н.Н. в пересборках уходит к Достоевскому ({int(dost*100)}%, к Плещееву "
                f"{int(ple*100)}%), но отличимость {dist:+.4f} (≤0): Н.Н. сидит у центра пространства "
                f"кандидатов, к Достоевскому ближе неразличимо — в отличие от подписанных Ф.Д., которые к "
                f"Достоевскому различимо ближе{fd}.{seg_tail} Вывод: слабый, неразличимый довод в пользу "
                f"Достоевского, не уверенная атрибуция. Рука Плещеева не проверяема (эталона в фельетонном "
                f"регистре нет). Сильнее этот корпус не даёт.")
    if w in DOST:
        return (f"Целиком Н.Н. уходит к Достоевскому ({int(dost*100)}% пересборок, отличимость "
                f"{dist:+.4f}), к Плещееву {int(ple*100)}%.{seg_tail} Согласуется с авторством "
                f"Достоевского; рука Плещеева не проверяема (эталона в фельетонном регистре нет).")
    return (f"Целиком Н.Н. уходит к «{w}» (к Достоевскому {int(dost*100)}%, к Плещееву {int(ple*100)}%, "
            f"отличимость {dist:+.4f}). Регистр доминирует; авторство Н.Н. неразрешено.")


if __name__ == "__main__":
    main()
