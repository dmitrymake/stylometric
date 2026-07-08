"""Решающий контроль валидности unmasking/imposters: узнают ли они ЗРЕЛОГО Шолохова?

Контроль смещения метода: при узком/раннем эталоне, взрослении автора или дисбалансе корпуса
unmasking/imposters могут смещаться. Скрипт проверяет, правильно ли метод атрибутирует
БЕССПОРНЫЕ ЗРЕЛЫЕ работы Шолохова (Поднятая целина, Они сражались, Судьба человека, Наука ненависти)
ЕГО ЖЕ руке. Leave-one-out: каждую такую работу C сравниваем с (бесспорный Шолохов БЕЗ C) и с Крюковым.
  • если все известные работы → Шолохову: метод ВАЛИДЕН для этого режима, 'ТД→Крюков' — реальная аномалия;
  • если известные работы тоже → Крюкову: метод СМЕЩЁН (малый эталон/взросление/дисбаланс), 'ТД→Крюков' артефакт.
Тот же leave-one-out прогон даёт честный потолок самого метода (его FPR на known-Sholokhov).
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
MODE = "FW_ONLY" if os.environ.get("FW_ONLY") else "freq"
TD = ["tihiy_don_1", "tihiy_don_2", "tihiy_don_3", "tihiy_don_4"]
DON_EARLY = ["rodinka", "zherebenok", "pastuh", "lazorevaya_step", "aleshkino_serdce", "chuzhaya_krov", "batraki"]
MATURE_UNDISPUTED = ["podnyataya_celina_2", "oni_srazhalis", "sudba_cheloveka", "nauka_nenavisti"]
ALL_UNDISPUTED = DON_EARLY + MATURE_UNDISPUTED


def text_of(author, book=None):
    d = ROOT / "input_clean" / author
    if book: return (d / f"{book}.txt").read_text("utf-8", "ignore")
    return " ".join(f.read_text("utf-8", "ignore") for f in sorted(d.glob("*.txt")))


def chunks_w(text, size=UM.CHUNK):
    w = re.findall(r"[а-яёА-ЯЁa-zA-Zà-ÿ]+", text.lower())
    return [w[i:i+size] for i in range(0, len(w) - size + 1, size)]


def attribute(name, C_books, ref_books, kr_ch, imps_texts, sh_ref_text):
    C_text = " ".join(text_of("sholohov", b) for b in C_books)
    C_ch = chunks_w(C_text)
    ref_ch = chunks_w(" ".join(text_of("sholohov", b) for b in ref_books))
    s_sh = UM.score(UM.unmask(C_ch, ref_ch))
    s_kr = UM.score(UM.unmask(C_ch, kr_ch))
    um_closer = "Шолохов" if (s_sh is not None and s_kr is not None and s_sh < s_kr) else "Крюков"
    imp = IMP.imposters(C_text, sh_ref_text, imps_texts)
    return {"name": name, "unmask_vs_sholokhov": s_sh, "unmask_vs_krukov": s_kr,
            "unmask_closer": um_closer, "imposters_score": round(imp, 3),
            "imposters_verdict": "Шолохов" if imp > 0.5 else "не верифиц."}


def main():
    print(f"=== КОНТРОЛЬ ВАЛИДНОСТИ unmasking/imposters; режим: {MODE} ===")
    kr_ch = chunks_w(text_of("krukov"))
    imps_texts = [text_of(a) for a in ["krukov", "serafimovich", "sevsky", "kumov"]]

    print("\n[A] БЕССПОРНЫЕ ЗРЕЛЫЕ работы Шолохова (leave-one-out): должны → ШОЛОХОВУ, если метод валиден")
    ctrl = []
    for C in MATURE_UNDISPUTED:
        ref = [b for b in ALL_UNDISPUTED if b != C]
        sh_ref_text = " ".join(text_of("sholohov", b) for b in ref)
        r = attribute(C, [C], ref, kr_ch, imps_texts, sh_ref_text)
        ctrl.append(r)
        print(f"  {C:20} unmask: Шолохов={r['unmask_vs_sholokhov']} Крюков={r['unmask_vs_krukov']} → {r['unmask_closer']}"
              f" | imposters={r['imposters_score']} → {r['imposters_verdict']}")
    ctrl_um_sh = sum(1 for r in ctrl if r["unmask_closer"] == "Шолохов")
    ctrl_imp_sh = sum(1 for r in ctrl if r["imposters_score"] > 0.5)
    print(f"  ИТОГ контроля: unmasking {ctrl_um_sh}/{len(ctrl)} → Шолохову; imposters {ctrl_imp_sh}/{len(ctrl)} → Шолохову")

    print("\n[B] ТД (та же процедура, эталон = весь бесспорный Шолохов без ТД)")
    sh_ref_text = " ".join(text_of("sholohov", b) for b in ALL_UNDISPUTED)
    td = []
    for v in TD:
        r = attribute(v, [v], ALL_UNDISPUTED, kr_ch, imps_texts, sh_ref_text)
        td.append(r)
        print(f"  {v:20} unmask: Шолохов={r['unmask_vs_sholokhov']} Крюков={r['unmask_vs_krukov']} → {r['unmask_closer']}"
              f" | imposters={r['imposters_score']} → {r['imposters_verdict']}")
    td_um_sh = sum(1 for r in td if r["unmask_closer"] == "Шолохов")

    # ВЕРДИКТ контроля
    valid = ctrl_um_sh >= 3   # большинство известных работ должны атрибутироваться Шолохову
    if valid:
        concl = (f"метод ВАЛИДЕН (известные зрелые работы Шолохова {ctrl_um_sh}/{len(ctrl)} unmask → Шолохову), "
                 f"значит ТД→Крюков ({4-td_um_sh}/4) — РЕАЛЬНАЯ аномалия парных верификаторов, требует осторожного доклада")
    else:
        concl = (f"метод СМЕЩЁН: даже известные зрелые работы Шолохова идут к Крюкову (только {ctrl_um_sh}/{len(ctrl)} к Шолохову) — "
                 f"причина: малый/узкий ранне-донской эталон + взросление автора + дисбаланс; 'ТД→Крюков' — АРТЕФАКТ метода, "
                 f"не свидетельство; топик-робастные калиброванные методы (LR-сегмент, AV) надёжнее")
    print(f"\n  ВЕРДИКТ КОНТРОЛЯ ВАЛИДНОСТИ: {concl}")

    out = {"mode": MODE, "validity_control_mature_undisputed": ctrl,
           "control_unmask_to_sholokhov": f"{ctrl_um_sh}/{len(ctrl)}",
           "control_imposters_to_sholokhov": f"{ctrl_imp_sh}/{len(ctrl)}",
           "td": td, "td_unmask_to_sholokhov": f"{td_um_sh}/4",
           "method_valid": bool(valid), "conclusion": concl}
    (ROOT / "docs" / f"sholokhov_verify3_{MODE}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved docs/sholokhov_verify3_{MODE}.json")


if __name__ == "__main__":
    main()
