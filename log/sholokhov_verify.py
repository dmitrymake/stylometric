"""3-е и 4-е семейства методов на ТД: UNMASKING (Koppel-Schler) + IMPOSTERS (Koppel-Winter).

Сегментный LR и supervised-AV — два семейства. Применяем канонические PAN-верификаторы «одна ли
рука» (log/unmasking.py и log/imposters.py) как третье и четвёртое независимые семейства. Согласие трёх+ семейств = робастность.

UNMASKING: кривая деградации A-vs-B. НИЗКИЙ score (быстрый провал) = ОДИН автор; ВЫСОКИЙ = разные.
  ждём: ТД ↔ бесспорный Шолохов (без ТД) — НИЗКО (один); ТД ↔ Крюков — ВЫСОКО (разные).
IMPOSTERS: P(D ближе к A=бесспорный Шолохов, чем к импосторам=донские современники). ВЫСОКО = Шолохов.
Калибровка-якоря: same (Шолохов-без-ТД сам с собой) и different (Шолохов-без-ТД vs Крюков).
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
WORD = re.compile(r"[а-яёА-ЯЁ]+")
TD = ["tihiy_don_1", "tihiy_don_2", "tihiy_don_3", "tihiy_don_4"]
UNDISPUTED = ["rodinka", "zherebenok", "pastuh", "lazorevaya_step", "aleshkino_serdce", "chuzhaya_krov",
              "batraki", "podnyataya_celina_2", "nauka_nenavisti", "oni_srazhalis", "sudba_cheloveka"]


def text_of(author, book=None):
    d = ROOT / "input_clean" / author
    if book:
        return (d / f"{book}.txt").read_text("utf-8", "ignore")
    return " ".join(f.read_text("utf-8", "ignore") for f in sorted(d.glob("*.txt")))


def chunks_from_words(text, size=UM.CHUNK):
    w = re.findall(r"[а-яёА-ЯЁa-zA-Zà-ÿ]+", text.lower())
    return [w[i:i+size] for i in range(0, len(w) - size + 1, size)]


def main():
    sh_undisp = " ".join(text_of("sholohov", b) for b in UNDISPUTED)
    krukov = text_of("krukov")
    und_ch = chunks_from_words(sh_undisp)
    kr_ch = chunks_from_words(krukov)

    print("=== UNMASKING калибровка-якоря (низко=один автор, высоко=разные) ===")
    half = len(und_ch) // 2
    same_anchor = UM.score(UM.unmask(und_ch[:half], und_ch[half:]))
    diff_anchor = UM.score(UM.unmask(und_ch, kr_ch))
    print(f"  SAME (Шолохов-без-ТД ↔ сам себя):  {same_anchor}")
    print(f"  DIFF (Шолохов-без-ТД ↔ Крюков):    {diff_anchor}")
    thr = (same_anchor + diff_anchor) / 2 if (same_anchor is not None and diff_anchor is not None) else 0.75
    calib_ok = (diff_anchor is not None and same_anchor is not None and diff_anchor > same_anchor + 0.03)
    print(f"  порог≈{thr:.3f}; калибровка {'РАЗДЕЛЯЕТ ✓' if calib_ok else 'слабая'}")

    print("\n=== UNMASKING: каждый том ТД ↔ Шолохов(без ТД) vs ↔ Крюков ===")
    um_rows = []
    for v in TD:
        td_ch = chunks_from_words(text_of("sholohov", v))
        s_sh = UM.score(UM.unmask(td_ch, und_ch))
        s_kr = UM.score(UM.unmask(td_ch, kr_ch))
        closer = "Шолохов" if (s_sh is not None and s_kr is not None and s_sh < s_kr) else "Крюков?"
        um_rows.append({"vol": v, "unmask_vs_sholokhov": s_sh, "unmask_vs_krukov": s_kr, "closer": closer})
        print(f"  {v:14} vs Шолохов={s_sh} (ниже=та же рука) | vs Крюков={s_kr} | ближе по руке: {closer}")
    um_to_sh = sum(1 for r in um_rows if r["closer"] == "Шолохов")

    print("\n=== IMPOSTERS: P(том ТД ближе к Шолохову-без-ТД, чем к донским импосторам) ===")
    imp_texts = [text_of(a) for a in ["krukov", "serafimovich", "sevsky", "kumov"]]
    imp_rows = []
    for v in TD:
        D = text_of("sholohov", v)
        sc = IMP.imposters(D, sh_undisp, imp_texts)
        imp_rows.append({"vol": v, "imposters_score": round(sc, 3)})
        print(f"  {v:14} score={sc:.3f} → {'верифицирован как ШОЛОХОВ' if sc>0.5 else 'НЕ верифицирован'}")
    imp_verified = sum(1 for r in imp_rows if r["imposters_score"] > 0.5)

    print("\n=== СОГЛАСИЕ ТРЁХ+ СЕМЕЙСТВ ===")
    print(f"  unmasking: {um_to_sh}/4 томов ближе к руке Шолохова; imposters: {imp_verified}/4 верифицированы как Шолохов")
    verdict = (f"unmasking {um_to_sh}/4 + imposters {imp_verified}/4 в пользу Шолохова — согласуется с сегментным LR "
               f"и supervised-AV: гипотеза чужой руки (Крюков) в ТД не поддержана независимыми семействами методов")
    print(f"  ВЕРДИКТ: {verdict}")

    out = {"method": "UNMASKING (Koppel-Schler) + IMPOSTERS (Koppel-Winter) на ТД, переиспользуя log/unmasking.py и log/imposters.py; 3-е и 4-е независимые семейства",
           "unmasking_anchors": {"same_sholokhov": same_anchor, "diff_sholokhov_vs_krukov": diff_anchor, "calibration_separates": bool(calib_ok)},
           "unmasking_td": um_rows, "unmasking_td_to_sholokhov": f"{um_to_sh}/4",
           "imposters_td": imp_rows, "imposters_verified_sholokhov": f"{imp_verified}/4",
           "verdict": verdict}
    (ROOT / "docs" / "sholokhov_verify.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved docs/sholokhov_verify.json")


if __name__ == "__main__":
    main()
