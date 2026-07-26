"""Общие метрики gate-кейсов: РАЗДЕЛЁННЫЕ work-level и chunk-weighted + exact-permutation.

Критическое различие work-level и chunk-weighted метрик:
- work_macro_recall — ОДИН удержанный текст = ОДИН голос (большинство его кусков). Это та метрика,
  к которой применим порог надёжной атрибуции; единица сравнения — произведение.
- chunk_weighted_recall — обучение leave-one-WORK-out, но качество усредняется по КУСКАМ удержанной
  работы: длинные работы весят больше (по числу кусков), а коррелированные куски считаются независимыми
  голосами — это меняет вес работ и эффективный размер выборки; в этих кейсах даёт более низкую долю. НЕ
  work-level; приводится только как диагностика.

Во всех train-side центроидах сначала усредняются куски КАЖДОЙ работы, затем профили работ усредняются
с равным весом. Поэтому `chunk_weighted_recall` означает только способ подсчёта test-side ошибок; длинная
работа не получает больший вес при обучении центроида.

Перестановка ярлыков на уровне работ: при малом числе работ берётся ТОЧНОЕ перечисление всех
расстановок, сохраняющих размеры классов (для 2 классов — C(W, n1)); иначе случайная выборка с
plus-one. Точный пол p при n1/n2 работах = 1 / C(W, n1) (например, 1/66 ≈ 0.015 при 10/2).
"""
from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from math import comb

import numpy as np


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _cent(vs):
    return _unit(np.mean(vs, axis=0))


def work_balanced_centroid(work_vectors):
    """Единичный центроид, в котором каждая работа имеет равный train-side вес.

    ``work_vectors`` — iterable пар ``(work_id, vector)``. Куски сначала
    усредняются внутри работы, каждое среднее L2-нормируется, затем направления
    работ усредняются между собой и итог снова L2-нормируется.
    """
    by_work = {}
    for work, vector in work_vectors:
        by_work.setdefault(work, []).append(np.asarray(vector, dtype=float))
    if not by_work:
        raise ValueError("work-balanced centroid requires at least one work")
    work_means = [_unit(np.mean(vectors, axis=0)) for vectors in by_work.values()]
    return _cent(work_means)


def leave_one_work_out(data, authors):
    """data: [(author, work, vec)]. Для каждой работы — предсказания её кусков; центроид класса при
    тесте работы исключает ВСЕ её куски. Возвращает [(author, work, [pred,...])] + confusion + works."""
    works = {a: sorted({w for au, w, _ in data if au == a}) for a in authors}
    wcp, conf = [], {a: {b: 0 for b in authors} for a in authors}
    for a in authors:
        for w in works[a]:
            test = [v for au, wk, v in data if au == a and wk == w]
            cents, ok = {}, True
            for b in authors:
                pool = [
                    (wk, v)
                    for au, wk, v in data
                    if au == b and not (b == a and wk == w)
                ]
                if not pool:
                    ok = False
                    break
                cents[b] = work_balanced_centroid(pool)
            if not ok:
                continue
            preds = [max(authors, key=lambda b: float(np.dot(_unit(v), cents[b]))) for v in test]
            wcp.append((a, w, preds))
            for p in preds:
                conf[a][p] += 1
    return wcp, conf, works


def both_metrics(wcp, authors):
    """Из предсказаний кусков по работам — обе метрики. work — большинство кусков работы."""
    cw_c = {a: 0 for a in authors}
    cw_t = {a: 0 for a in authors}
    wl_c = {a: 0 for a in authors}
    wl_t = {a: 0 for a in authors}
    for a, _w, preds in wcp:
        for p in preds:
            cw_c[a] += p == a
            cw_t[a] += 1
        wl_c[a] += Counter(preds).most_common(1)[0][0] == a
        wl_t[a] += 1
    cw = float(np.mean([cw_c[a] / cw_t[a] for a in authors if cw_t[a]])) if any(cw_t.values()) else 0.0
    wl = float(np.mean([wl_c[a] / wl_t[a] for a in authors if wl_t[a]])) if any(wl_t.values()) else 0.0
    return {
        "work_macro_recall": round(wl, 4),
        "chunk_weighted_recall": round(cw, 4),
        "work_recall": {a: f"{wl_c[a]}/{wl_t[a]}" for a in authors},
        "chunk_recall": {a: f"{cw_c[a]}/{cw_t[a]}" for a in authors},
    }


def _precompute_work_means(by_work):
    """Фолд-НЕЗАВИСИМые средние работ и юнит-нормированные test-куски.

    Считаются ОДИН раз на кейс; дальше центроид класса под любой расстановкой ярлыков получается
    вычитанием среднего удержанной работы из суммы средних работ класса — без пересборки пула на каждой
    из тысяч перестановок (главный ускоритель work_permutation_p)."""
    work_ids = list(by_work.keys())
    wmean = {
        w: _unit(np.mean(np.asarray(by_work[w], dtype=float), axis=0))
        for w in work_ids
    }
    uchunks = {w: [_unit(np.asarray(v, dtype=float)) for v in by_work[w]] for w in work_ids}
    return work_ids, wmean, uchunks


def _workmacro_from_work_means(work_ids, wmean, uchunks, label_of_work, authors):
    """work-level macro под расстановкой ярлыков label_of_work, из предподсчитанных сумм.

    Центроид класса b при тесте работы w = unit(Σ средних работ класса b − среднее w, если w в b).
    Деление на число работ не меняет направление после unit-нормировки. Это математически совпадает с
    ``work_balanced_centroid`` без пересборки пула — O(работ) на перестановку вместо O(всех кусков)."""
    tot_sum = {b: None for b in authors}
    tot_count = {b: 0 for b in authors}
    for w in work_ids:
        b = label_of_work[w]
        tot_sum[b] = wmean[w].copy() if tot_sum[b] is None else tot_sum[b] + wmean[w]
        tot_count[b] += 1
    wl_c = {a: 0 for a in authors}
    wl_t = {a: 0 for a in authors}
    for w in work_ids:
        truth = label_of_work[w]
        cents, ok = {}, True
        for b in authors:
            same = label_of_work[w] == b
            pc = tot_count[b] - (1 if same else 0)
            if pc <= 0:                       # пул класса b (без w) пуст — как ok=False в исходнике
                ok = False
                break
            ps = (tot_sum[b] - wmean[w]) if same else tot_sum[b]
            cents[b] = _unit(ps / pc)
        if not ok:
            continue
        preds = [max(authors, key=lambda b: float(np.dot(uc, cents[b]))) for uc in uchunks[w]]
        wl_c[truth] += Counter(preds).most_common(1)[0][0] == truth
        wl_t[truth] += 1
    return float(np.mean([wl_c[a] / wl_t[a] for a in authors if wl_t[a]])) if any(wl_t.values()) else 0.0


def _workmacro_under(by_work, label_of_work, authors):
    """Совместимость: work-macro под расстановкой ярлыков (делегирует в work-mean estimator с
    предподсчётом сумм). На горячем пути work_permutation_p предподсчёт вынесен наружу."""
    return _workmacro_from_work_means(
        *_precompute_work_means(by_work), label_of_work, authors
    )


def work_permutation_p(data, label_of, authors,
                       max_exact=4000, n_random=2000, seed=20260630, n_jobs=-1):
    """Перестановка ярлыков (label_of(author)) на уровне работ; метрика — work_macro_recall.
    Точное перечисление для 2 классов при малом C(W, n1); иначе случайно с plus-one. Возвращает
    (p, method, exact_floor). exact_floor = 1/C(W,n1) — минимально достижимое точное p.

    observed считается ВНУТРИ той же функцией `_workmacro_under`, что и перестановки, СЫРЫМ
    (без округления). Округлённое значение отчёта здесь НЕ используется: round(wl,4) вверх делал бы
    истинную метку < observed, и даже она не проходила бы `>=` — отсюда невозможное p=0.0 при
    exact-floor > 0."""
    # Уникальный ключ работы включает исходного автора: одинаковые basename у
    # разных авторов не должны сливаться в один permutation unit.
    seen, works, truth = set(), [], []
    for a, w, _v in data:
        work_key = (a, w)
        if work_key not in seen:
            seen.add(work_key)
            works.append(work_key)
            truth.append(label_of(a))
    by_work = {}
    for a, w, v in data:
        by_work.setdefault((a, w), []).append(v)

    # предподсчёт по-работных сумм ОДИН раз: центроиды под каждой перестановкой считаются вычитанием
    work_ids, wmean, uchunks = _precompute_work_means(by_work)

    # СЫРОЙ observed той же функцией, что и null-перестановки (см. docstring) — гарантирует p >= floor
    true_lab = {works[i]: truth[i] for i in range(len(works))}
    observed = _workmacro_from_work_means(
        work_ids, wmean, uchunks, true_lab, authors
    )

    labels = sorted(set(truth))
    counts = Counter(truth)
    W = len(works)
    floor = None

    # ниже этого числа расстановок последовательный путь быстрее: накладные расходы пула
    # потоков превышают выгоду (напр. exact_66/exact_252). Параллелим только большие циклы
    # (random_2000 и крупные exact). Порог не влияет на РЕЗУЛЬТАТ — только на скорость.
    _PAR_MIN = 256

    def _ge_count(labs):
        """Число расстановок ярлыков с work-macro >= observed. Для больших циклов — на потоках
        stdlib (by_work общий read-only, numpy отпускает GIL; сумма порядко-независима ->
        результат детерминирован и совпадает с последовательным). Потоки, не процессы: без
        зависимостей и без пиклинга by_work; работает и в системном python3."""
        cores = os.cpu_count() or 4
        workers = max(1, cores - 2) if n_jobs in (-1, None) else max(1, int(n_jobs))
        workers = min(workers, 16, len(labs))
        _wm = lambda lab: _workmacro_from_work_means(
            work_ids, wmean, uchunks, lab, authors
        )
        if workers <= 1 or len(labs) < _PAR_MIN:
            return int(sum(1 for lab in labs if _wm(lab) >= observed - 1e-9))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            wms = list(ex.map(_wm, labs))
        return int(sum(1 for wm in wms if wm >= observed - 1e-9))

    if len(labels) == 2:
        n1 = counts[labels[0]]
        n_assign = comb(W, n1)
        floor = round(1.0 / n_assign, 5)
        if n_assign <= max_exact:
            idx = list(range(W))
            labs = [{works[i]: (labels[0] if i in set(combo) else labels[1]) for i in idx}
                    for combo in combinations(idx, n1)]
            ge = _ge_count(labs)
            return round(ge / len(labs), 4), f"exact_{len(labs)}", floor
    # случайная перестановка с plus-one (большие N или >2 классов).
    # Перестановки строятся ПОСЛЕДОВАТЕЛЬНО из seeded-rng,
    # затем оцениваются параллельно -> детерминизм сохранён.
    rng = np.random.default_rng(seed)
    labs = [{works[i]: perm[i] for i in range(W)}
            for perm in (list(rng.permutation(truth)) for _ in range(n_random))]
    ge = _ge_count(labs)
    return round((ge + 1) / (n_random + 1), 4), f"random_{n_random}", floor
