"""Authorship Verification через UNMASKING (Koppel-Schler) — улучшенный детектор «одна ли рука».

Зачем: косинус по char/служебным словам НЕ ловит умелую стилизацию (провал на фальшивке Вырубовой)
и тонет в жанре. Unmasking меряет ГЛУБИНУ различий: тренируем классификатор A-vs-B, итеративно
убираем самые различающие признаки. У ОДНОГО автора точность падает быстро (отличия поверхностны/
жанровы — после их снятия тексты неразличимы); у РАЗНЫХ авторов держится (различия устойчивы, глубоки).

Вердикт: «degradation score» = средняя точность последних итераций. НИЗКО → один автор; ВЫСОКО → разные.
Калибруется на эталоне (подлинные same-пары vs разные-пары) — печатается в batch-режиме.

Признаки: топ-N частых слов (доминируют служебные) — классический unmasking-набор.
"""
from __future__ import annotations
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from stylo.lang import function_words
FW_ONLY = bool(os.environ.get("FW_ONLY"))
FWSET = function_words("ru")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK = 500
K_ITERS = 14          # итераций unmasking
M_REMOVE = 20         # признаков удаляем за итерацию (по 3 с каждого полюса)
TOPN = 250           # размер словаря частых слов


def chunks_of(src):
    p = ROOT / src
    files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    text = " ".join(f.read_text("utf-8", "ignore") for f in files if f.exists())
    w = re.findall(r"[а-яёА-ЯЁa-zA-Zà-ÿ]+", text.lower())
    return [w[i:i + CHUNK] for i in range(0, len(w) - CHUNK + 1, CHUNK)]


def unmask(ca, cb):
    """Кривая деградации точности A-vs-B по мере удаления различающих слов."""
    if len(ca) < 5 or len(cb) < 5:
        return None
    freq = Counter(t for ch in ca + cb for t in ch)
    if FW_ONLY:
        vocab = [w for w, _ in freq.most_common() if w in FWSET][:TOPN]
    else:
        vocab = [w for w, _ in freq.most_common(TOPN)]
    vi = {w: i for i, w in enumerate(vocab)}
    def vecs(cl):
        M = np.zeros((len(cl), len(vocab)))
        for r, ch in enumerate(cl):
            c = Counter(ch)
            for w, n in c.items():
                if w in vi: M[r, vi[w]] = n / len(ch)
        return M
    Xa, Xb = vecs(ca), vecs(cb)
    X = np.vstack([Xa, Xb]); y = np.array([0] * len(Xa) + [1] * len(Xb))
    alive = np.ones(len(vocab), bool)
    curve = []
    cvn = min(5, len(ca), len(cb))
    for _ in range(K_ITERS):
        cols = np.where(alive)[0]
        if len(cols) < M_REMOVE + 2: break
        Xc = X[:, cols]
        acc = float(np.mean(cross_val_score(LinearSVC(C=1, max_iter=5000), Xc, y, cv=cvn)))
        curve.append(round(acc, 3))
        clf = LinearSVC(C=1, max_iter=5000).fit(Xc, y)
        w = clf.coef_[0]
        order = np.argsort(w)
        drop = list(order[:M_REMOVE // 2]) + list(order[-M_REMOVE // 2:])
        for j in drop: alive[cols[j]] = False
    return curve


def score(curve):
    """degradation score = средняя точность последних 3 итераций (ниже = один автор)."""
    return round(float(np.mean(curve)), 3) if curve else None


def run(label, a, b):
    cv = unmask(chunks_of(a), chunks_of(b))
    if cv is None:
        print(f"  {label:42} — мало данных"); return None
    s = score(cv)
    print(f"  {label:42} площадь={s:.3f} финал={cv[-1]:.3f}  кривая={cv}")
    return s


def main():
    if len(sys.argv) >= 3:
        run(f"{sys.argv[1]} vs {sys.argv[2]}", sys.argv[1], sys.argv[2]); return
    print("=== КАЛИБРОВКА: подлинные ОДНОАВТОРСКИЕ кросс-регистровые пары (ждём НИЗКИЙ score) ===")
    same = [
        ("Толстой: дневник↔письма", "input_personal/tolstoy_diary", "input_personal/tolstoy_letters"),
        ("Пушкин: дневник↔письма", "input_personal/pushkin_diary", "input_personal/pushkin_letters"),
        ("Чехов: дневник↔письма", "input_personal/chehov_diary", "input_personal/chehov_letters"),
        ("Александр III: дневник↔письма", "input_personal/alexander3_diary", "input_personal/alexander3_letters"),
    ]
    ss = [run(l, a, b) for l, a, b in same]
    print("\n=== РАЗНЫЕ авторы (ждём ВЫСОКИЙ score) ===")
    diff = [
        ("Толстой-дневник vs Чехов-письма", "input_personal/tolstoy_diary", "input_personal/chehov_letters"),
        ("Пушкин-дневник vs Достоевский-письма", "input_personal/pushkin_diary", "input_personal/dostoevsky_letters"),
    ]
    ds = [run(l, a, b) for l, a, b in diff]
    print("\n=== ФАЛЬШИВКА: поддельный дневник Вырубовой vs её НАСТОЯЩИЕ мемуары (ждём ВЫСОКИЙ=разные=поймали) ===")
    vy = run("Вырубова: фальшивка vs реальные мемуары",
             "input_cases/vyrubova/vyrubova_diary_fake.txt", "input_cases/vyrubova/vyrubova_memoir_real.txt")
    print("\n=== ЦЕЛЬ: Николай — дневник vs его письма ===")
    nik = run("Николай: дневник↔письма", "input_personal/nikolas2_diary", "input_personal/nikolas2_letters")

    print("\n" + "=" * 60)
    # МЕТРИКА = провал кривой. Koppel: ОДИН автор → БОЛЬШОЙ провал; РАЗНЫЕ → малый провал.
    sg = [x for x in ss if x is not None]; dg = [x for x in ds if x is not None]
    if sg and dg:
        thr = (np.mean(sg) + np.mean(dg)) / 2
        print(f"Провал: подлинные-ОДИН-автор {np.mean(sg):+.3f} vs РАЗНЫЕ {np.mean(dg):+.3f}; порог ≈ {thr:+.3f}")
        ok = np.mean(sg) > np.mean(dg) + 0.05
        print(f"Калибровка {'РАЗДЕЛЯЕТ same/different ✓' if ok else 'НЕ разделяет — метод не работает на этих данных ✗'}")
        if vy is not None:
            print(f"Вырубова-фальшивка провал={vy:+.3f}: {'малый → РАЗНЫЕ → подделку поймали ✓' if vy<thr else 'большой → выглядит как ОДИН автор → НЕ поймали ✗'}")
        if nik is not None:
            print(f"Николай провал={nik:+.3f}: {'малый → как РАЗНЫЕ (подозрительно)' if nik<thr else 'большой → как ОДИН автор (подлинно)'}")


if __name__ == "__main__":
    main()
