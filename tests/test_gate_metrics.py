"""Тесты gate-метрик (scripts/_gate_metrics.py) — фундамент вердиктов каталога казусов:
work-LOO leak-free, work_macro_recall, точная перестановка ярлыков работ и её пол 1/C(W,n1)."""
import importlib.util
from math import comb
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_gate_metrics", _ROOT / "scripts" / "_gate_metrics.py")
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)


def _sep_data(n_works=3):
    """Разделимые данные: автор A у оси X, автор B у оси Y; по 2 куска на работу."""
    data = []
    for w in range(n_works):
        base = np.array([1.0, 0.05 * w])
        data += [("A", f"A{w}", base.copy()), ("A", f"A{w}", base + np.array([0.0, 0.03]))]
        baseb = np.array([0.05 * w, 1.0])
        data += [("B", f"B{w}", baseb.copy()), ("B", f"B{w}", baseb + np.array([0.03, 0.0]))]
    return data, ["A", "B"]


def _unequal_work_data():
    """A-long has nine chunks but must have the same train weight as A-short."""
    data = [("A", "A-long", np.array([1.0, 0.0])) for _ in range(9)]
    data += [
        ("A", "A-short", np.array([0.0, 1.0])),
        ("A", "A-test", np.array([0.6, 0.8])),
        ("B", "B0", np.array([0.0, 1.0])),
        ("B", "B1", np.array([0.0, 1.0])),
    ]
    return data, ["A", "B"]


def test_train_centroid_gives_each_work_equal_weight():
    rows = [("long", np.array([1.0, 0.0])) for _ in range(9)]
    rows.append(("short", np.array([0.0, 1.0])))

    centroid = gm.work_balanced_centroid(rows)

    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert np.allclose(centroid, expected)
    assert not np.allclose(centroid, gm._cent([v for _work, v in rows]))


def test_train_centroid_gives_each_work_direction_equal_weight():
    rows = [
        ("heterogeneous", np.array([1.0, 0.0])),
        ("heterogeneous", np.array([-0.8, 0.0])),
        ("stable", np.array([0.0, 1.0])),
    ]

    centroid = gm.work_balanced_centroid(rows)

    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    raw_work_mean = gm._unit(np.mean([[0.1, 0.0], [0.0, 1.0]], axis=0))
    assert np.allclose(centroid, expected)
    assert not np.allclose(centroid, raw_work_mean)


def test_optimized_permutation_precompute_uses_unit_work_directions():
    by_work = {
        "heterogeneous": [np.array([1.0, 0.0]), np.array([-0.8, 0.0])],
        "stable": [np.array([0.0, 1.0])],
    }

    work_ids, work_means, _unit_chunks = gm._precompute_work_means(by_work)

    assert work_ids == ["heterogeneous", "stable"]
    assert np.allclose(work_means["heterogeneous"], [1.0, 0.0])
    assert np.allclose(work_means["stable"], [0.0, 1.0])


def test_work_loo_uses_work_balanced_train_centroids():
    data, authors = _unequal_work_data()

    wcp, _conf, _works = gm.leave_one_work_out(data, authors)

    predictions = {work: preds for _author, work, preds in wcp}
    # Equal work weights put A at [0.5, 0.5], so A-test is closer to A.
    # A flat chunk mean would put A near [0.9, 0.1] and predict B instead.
    assert predictions["A-test"] == ["A"]


def test_optimized_permutation_estimator_matches_public_work_balanced_loo():
    data, authors = _unequal_work_data()
    by_work = {}
    label_of_work = {}
    for author, work, vector in data:
        by_work.setdefault(work, []).append(vector)
        label_of_work[work] = author

    optimized = gm._workmacro_under(by_work, label_of_work, authors)
    wcp, _conf, _works = gm.leave_one_work_out(data, authors)
    public = gm.both_metrics(wcp, authors)["work_macro_recall"]

    assert round(optimized, 4) == public


def test_work_macro_recall_separable():
    data, authors = _sep_data(3)
    wcp, _conf, _works = gm.leave_one_work_out(data, authors)
    m = gm.both_metrics(wcp, authors)
    assert m["work_macro_recall"] == 1.0
    # work_macro и chunk_weighted — РАЗНЫЕ метрики по определению; обе в [0,1]
    assert 0.0 <= m["chunk_weighted_recall"] <= 1.0


def test_work_loo_is_leak_free():
    """Удержанная работа НЕ входит в центроид своего класса: работа-выброс, похожая на
    чужой класс, под leak-free LOO уходит в чужой класс (сама себе не помогает)."""
    data = [
        ("A", "A0", np.array([1.0, 0.0])), ("A", "A0", np.array([1.0, 0.0])),
        ("A", "A1", np.array([1.0, 0.0])), ("A", "A1", np.array([1.0, 0.0])),
        ("A", "Aout", np.array([0.0, 1.0])), ("A", "Aout", np.array([0.0, 1.0])),
        ("B", "B0", np.array([0.0, 1.0])), ("B", "B0", np.array([0.0, 1.0])),
        ("B", "B1", np.array([0.0, 1.0])), ("B", "B1", np.array([0.0, 1.0])),
    ]
    authors = ["A", "B"]
    wcp, _conf, _works = gm.leave_one_work_out(data, authors)
    preds = {w: p for _a, w, p in wcp}
    # Aout под leak-free LOO классифицируется как B — её собственные куски исключены из центроида A
    assert all(p == "B" for p in preds["Aout"])


def test_permutation_p_geq_exact_floor():
    data, authors = _sep_data(3)  # 3 работы A + 3 работы B -> W=6, n1=3, C(6,3)=20
    p, method, floor = gm.work_permutation_p(data, lambda a: a, authors)
    assert method.startswith("exact_")
    assert floor == round(1.0 / comb(6, 3), 5)     # 1/20 = 0.05
    assert p >= floor - 1e-9                        # инвариант отчёта: p не ниже точного пола
    assert 0.0 < p <= 1.0


def test_permutation_exact_floor_formula():
    """Точный пол = 1/C(W, n1) для 2 классов при разных размерах."""
    data, authors = _sep_data(4)  # 4+4 работы -> C(8,4)=70
    _p, method, floor = gm.work_permutation_p(data, lambda a: a, authors, max_exact=1000)
    assert method == "exact_70"
    assert floor == round(1.0 / comb(8, 4), 5)


def test_permutation_treats_same_work_name_from_different_authors_as_distinct():
    data, authors = _sep_data(2)
    renamed = [(author, "shared" if work.endswith("0") else work, vector)
               for author, work, vector in data]

    _p, method, floor = gm.work_permutation_p(renamed, lambda a: a, authors)

    # Four works remain distinct: C(4, 2), not C(3, 1).
    assert method == "exact_6"
    assert floor == round(1.0 / comb(4, 2), 5)
