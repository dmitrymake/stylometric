"""Топик-контроль для unmasking/imposters: развести ДОНСКУЮ ТЕМУ и РУКУ в атрибуции ТД.

Топик-чувствительные методы подвержены конфаунду «донская тема vs рука»: Крюков делит с ТД казачью тему.
Эталон Шолохова — ТОЛЬКО его РАННИЕ ДОНСКИЕ рассказы (та же казачья тема/эпоха, что ТД), плюс
опционально FW_ONLY (только служебные слова — топик-робастно). Якоря: SAME = ранне-донской Шолохов сам
с собой; DIFF = ранне-донской Шолохов vs Крюков (оба донские → 'разные руки при общей теме').
  ТД vs ранне-донской Шолохов ≈ SAME → Шолохов; ≈ DIFF → не Шолохов (Крюков-подобно).
Запуск дважды: обычный (частые слова) и FW_ONLY=1 (служебные).
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json, pathlib, sys, re, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "log"))
warnings.filterwarnings("ignore")
import numpy as np
import unmasking as UM
import imposters as IMP

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODE = "FW_ONLY (служебные слова, топик-робастно)" if os.environ.get("FW_ONLY") else "частые слова (топик-чувствительно)"
TD = ["tihiy_don_1", "tihiy_don_2", "tihiy_don_3", "tihiy_don_4"]
DON_EARLY = ["rodinka", "zherebenok", "pastuh", "lazorevaya_step", "aleshkino_serdce", "chuzhaya_krov", "batraki"]


def text_of(author, book=None):
    d = ROOT / "input_clean" / author
    if book:
        return (d / f"{book}.txt").read_text("utf-8", "ignore")
    return " ".join(f.read_text("utf-8", "ignore") for f in sorted(d.glob("*.txt")))


def chunks_w(text, size=UM.CHUNK):
    w = re.findall(r"[а-яёА-ЯЁa-zA-Zà-ÿ]+", text.lower())
    return [w[i:i+size] for i in range(0, len(w) - size + 1, size)]


def main():
    print(f"=== ТОПИК-КОНТРОЛЬ; режим признаков: {MODE} ===")
    sh_don = " ".join(text_of("sholohov", b) for b in DON_EARLY)
    don_ch = chunks_w(sh_don); kr_ch = chunks_w(text_of("krukov"))
    print(f"ранне-донской Шолохов: {len(don_ch)} чанков; Крюков: {len(kr_ch)} чанков")

    half = len(don_ch) // 2
    same = UM.score(UM.unmask(don_ch[:half], don_ch[half:]))
    diff = UM.score(UM.unmask(don_ch, kr_ch))
    thr = (same + diff) / 2 if (same is not None and diff is not None) else None
    print(f"\n  ЯКОРЯ: SAME (ранне-донской Шолохов↔себя)={same} | DIFF (↔Крюков, общая тема)={diff} | порог≈{thr}")
    calib_ok = (same is not None and diff is not None and diff > same + 0.03)
    print(f"  калибровка {'РАЗДЕЛЯЕТ ✓' if calib_ok else 'слабая ✗'}")

    print(f"\n  UNMASKING: ТД ↔ ранне-донской Шолохов vs ↔ Крюков (оба донская тема)")
    um_rows = []
    for v in TD:
        td_ch = chunks_w(text_of("sholohov", v))
        s_sh = UM.score(UM.unmask(td_ch, don_ch))
        s_kr = UM.score(UM.unmask(td_ch, kr_ch))
        closer = "Шолохов" if (s_sh is not None and s_kr is not None and s_sh < s_kr) else "Крюков"
        um_rows.append({"vol": v, "vs_sholokhov_don": s_sh, "vs_krukov": s_kr, "closer": closer})
        print(f"  {v:14} vs Шолохов-Дон={s_sh} | vs Крюков={s_kr} → ближе: {closer}")
    um_sh = sum(1 for r in um_rows if r["closer"] == "Шолохов")

    print(f"\n  IMPOSTERS: P(ТД ближе к ранне-донскому Шолохову, чем к донским импосторам)")
    imps = [text_of(a) for a in ["krukov", "serafimovich", "sevsky", "kumov"]]
    imp_rows = []
    for v in TD:
        sc = IMP.imposters(text_of("sholohov", v), sh_don, imps)
        imp_rows.append({"vol": v, "score": round(sc, 3)})
        print(f"  {v:14} score={sc:.3f} → {'Шолохов' if sc>0.5 else 'не верифицирован'}")
    imp_sh = sum(1 for r in imp_rows if r["score"] > 0.5)

    interp = (f"genre-matched ({MODE}): unmasking {um_sh}/4 + imposters {imp_sh}/4 в пользу Шолохова. "
              + ("→ при контроле общей донской темы ТД ближе к РАННЕ-ДОНСКОМУ Шолохову: крюковский сигнал был во многом ТОПИКОМ"
                 if um_sh + imp_sh >= 5 else
                 "→ ДАЖЕ при genre-matched/топик-робастном контроле ТД склоняется к Крюкову: сигнал устойчив, требует осторожной интерпретации"))
    print(f"\n  ИТОГ: {interp}")

    tag = "fw" if os.environ.get("FW_ONLY") else "freq"
    out = {"mode": MODE, "anchors": {"same": same, "diff_vs_krukov": diff, "calibration_separates": bool(calib_ok)},
           "unmasking_td": um_rows, "unmasking_to_sholokhov": f"{um_sh}/4",
           "imposters_td": imp_rows, "imposters_to_sholokhov": f"{imp_sh}/4", "interpretation": interp}
    (ROOT / "docs" / f"sholokhov_verify2_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved docs/sholokhov_verify2_{tag}.json")


if __name__ == "__main__":
    main()
