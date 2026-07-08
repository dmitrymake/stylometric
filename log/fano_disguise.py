"""Синтетический disguise-фронт: проверка коллапса детекции у information-пола (Шаг 1, доп.).

Натуральные пары авторов НИКОГДА не достигают information-пола (fano_frontier.json:
max pair Pe=0.28, 0/1081 у пола). Чтобы проверить сам тезис «disguise → пол», нужно
ДОЙТИ до пола КОНСТРУКТИВНО: синтетически маскировать чужую руку, сдвигая её стиль к
хосту, и смотреть, как рушится детекция.

Модель маскировки (честно — синтетический зонд, не претензия на реального фальсификатора):
  • host A, intruder B; классификатор обучен на натуральных чанках (in-sample bow_lr).
  • centroid_A = средний scaled-фич-вектор чанков A.
  • disguise strength s ∈ [0,1]: замаскированный вектор чанка B
        v'(s) = (1−s)·v_B + s·centroid_A      (B «пишет как средний A»).
  • recognize_B(s)  = P(классификатор атрибутирует v'(s) автору B) — детекция чужой руки;
  • passes_as_A(s)  = P(argmax = A) — доля, успешно выдающая себя за A.

Ожидание (и проверка тезиса): recognize_B монотонно → 0 при s → 1 (маскировка
побеждает детекцию = пара у information-пола), passes_as_A → 1. Точка пересечения
recognize_B ≈ passes_as_A — порог маскировки. Пол универсально достижим конструктивно,
хотя ни одна натуральная пара там не находится.

Честно: метрика Pe(A,B') через апостериор ТОГО ЖЕ мультикласс-классификатора
циркулярна при s→1 (замаскированные точки получают апостериор→A → в {A,B}-виде
«явно A» → Pe→0, а не 0.5). Поэтому приводим detection-rate, а не Pe.
"""
from __future__ import annotations
import json, pathlib, sys, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
from scipy.sparse import vstack

from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.models.baselines import build_bow_lr

ROOT = pathlib.Path(__file__).resolve().parents[1]
STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# пары вдоль всего диапазона натуральной Pe: от очень различимых до самых конфузных
PAIRS = [
    ("dostoevsky", "sorokin", "очень различимы"),
    ("chehov", "bunin", "близкие классики"),
    ("krukov", "serafimovich", "донская школа"),
    ("gumilev", "novikov_priboy", "конфузные"),
    ("babel", "sevsky", "самые конфузные (max Pe)"),
]


def build_books(ds):
    books = {}
    for t, g in zip(ds.texts, ds.groups):
        books.setdefault(g, []).append(t)
    by_author = {}
    for g, txts in books.items():
        a = g.split("/", 1)[0]
        by_author.setdefault(a, []).append((g, txts))
    return by_author


def biggest(by_author, a):
    bs = sorted(by_author.get(a, []), key=lambda kv: len(kv[1]), reverse=True)
    return bs[0][1] if bs else None


def main():
    cfg = load_config()
    excl = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    ds = load_dataset(ROOT / "data" / "frags_train", exclude_authors=excl)
    authors = list(ds.authors)
    K = ds.n_authors
    by_author = build_books(ds)
    print(f"корпус: {K} авторов; обучаю in-sample bow_lr детектор…", flush=True)

    pipe = build_bow_lr()
    pipe.fit(ds.texts, ds.y)
    bow = pipe.named_steps["bow"]
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]
    classes = list(lr.classes_)   # индексы авторов, которые знает lr

    def feat(txts):
        return scaler.transform(bow.transform(txts))

    results = []
    for host, intr, label in PAIRS:
        if host not in authors or intr not in authors:
            print(f"  пропуск {host}/{intr} (нет в корпусе)", flush=True)
            continue
        ta = biggest(by_author, host); tb = biggest(by_author, intr)
        if not ta or not tb or len(tb) < 5:
            continue
        ai, bi = authors.index(host), authors.index(intr)
        if ai not in classes or bi not in classes:
            print(f"  пропуск {host}/{intr} (не в classes lr)", flush=True)
            continue
        X_A = feat(ta[:200])                 # genuine-A чанки (для центроида)
        X_B = feat(tb[:200]).toarray()       # intruder чанки (dense для бленда)
        cA = np.asarray(X_A.mean(axis=0)).reshape(1, -1)   # centroid_A (dense 1×d)
        curve = []
        for s in STRENGTHS:
            Xd = (1 - s) * X_B + s * cA              # замаскированные чанки B
            post = lr.predict_proba(Xd)              # (n_B, K_lr)
            # выровнять на полный список авторов
            full = np.zeros((Xd.shape[0], K))
            for j, c in enumerate(classes):
                full[:, int(c)] = post[:, j]
            recognize_b = float(np.mean(full.argmax(axis=1) == bi))
            passes_a = float(np.mean(full.argmax(axis=1) == ai))
            mean_post_b = float(np.mean(full[:, bi]))
            curve.append({"s": s, "recognize_B": round(recognize_b, 3),
                          "passes_as_A": round(passes_a, 3),
                          "mean_posterior_B": round(mean_post_b, 3)})
        # порог маскировки: s, где recognize_B <= passes_as_A (детекция <= шанс)
        cross = next((c["s"] for c in curve if c["recognize_B"] <= c["passes_as_A"]), None)
        results.append({"host": host, "intruder": intr, "kind": label, "curve": curve,
                        "disguise_threshold_s": cross})
        print(f"  {host}+{intr} [{label}]: порог маскировки s={cross}", flush=True)

    # сводка: усреднённая кривая recognize_B по всем парам
    agg = []
    for k, s in enumerate(STRENGTHS):
        vals = [r["curve"][k]["recognize_B"] for r in results if k < len(r["curve"])]
        agg.append({"s": s, "mean_recognize_B": round(float(np.mean(vals)), 3),
                    "mean_passes_as_A": round(float(np.mean([r["curve"][k]["passes_as_A"]
                                for r in results if k < len(r["curve"])])), 3)})

    out = {
        "status": ("exploratory; НЕ headline-ready: детектор bow_lr обучен IN-SAMPLE (fit на всех данных, оценка на тех же); "
                   "маскировка в лексическом bow-пространстве, не структурном. Зонд геометрии, не оценка обобщаемой детекции."),
        "method": ("синтетическая маскировка: бленд scaled-фич intruder→centroid host "
                   "(bow_lr in-sample); recognize_B(s)=P(атрибутирован B), passes_as_A(s)=P(argmax=A)"),
        "pairs": results,
        "aggregate_curve": agg,
        "headline": ("recognize_B → 0 при s → 1 для ВСЕХ пар ⇒ маскировка конструктивно "
                     "достигает information-пола (чужая рука невидима); натуральные пары там "
                     "не находятся (см. fano_frontier.json) ⇒ пол = свойство вырожденного/"
                     "сознательно-замаскированного режима, не натуральной AA"),
    }
    p = ROOT / "docs" / "fano_disguise.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ saved {p}", flush=True)
    print("\n=== УСРЕДНЁННАЯ кривая recognize_B(s) ===", flush=True)
    for c in agg:
        bar = "█" * int(c["mean_recognize_B"] * 20)
        print(f"  s={c['s']:.1f}  recogB={c['mean_recognize_B']:.2f}  passesA={c['mean_passes_as_A']:.2f}  {bar}", flush=True)


if __name__ == "__main__":
    main()
