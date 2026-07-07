"""Information-theoretic «пол» для авторской атрибуции (Fano / Bayes-error).

«Потолок» метода — не эмпирическая неточность, а НИЖНЯЯ граница ошибки, заданная
тем, сколько информации о метке автора НЕСЁТ вектор признаков. Меньше информации
о авторе в признаках → выше неразрешимая ошибка → выше «пол», который не пробить
никаким классификатором. Это и есть формальный язык для «почему спорный кейс
неразрешим» (Harrison & Yener, ISIT 2025, только на блогерском корпусе).

Формулы (биты, log2):
  H(A)       = -Σ_a p_a·log2 p_a              — априорная неопределённость автора
  H(A|F)     = ⟨ -Σ_a p̂(a|f)·log2 p̂(a|f) ⟩   — остаточная неопределённость,
                                                  оцениваемая по OOF-постериорам
                                                  модели (soft-vote на уровне книги)
  I(A;F)     = H(A) − H(A|F)                   — сколько бит о авторе несут признаки
  Fano (пол): P_e ≥ (H(A|F) − 1) / log2(K−1)   — нижняя граница ошибки атрибуции
  пара (A,B): P_e(A,B) = h⁻¹(⟨H₂(p̂_AB)⟩)      — двоичная Bayes-ошибка «это A или B?»
                                                (h — двоичная энтропия; h⁻¹ берёт
                                                 меньший корень = честную ошибку)

ЧЕСТНОСТЬ (критично):
  * fano_floor из H(A|F) — НЕ граница ошибки, а МОДЕЛЬ-ЗАВИСИМАЯ описательная
    величина. Энтропия постериоров модели ВЕРХНЕ оценивает истинную H(A|F)
    (модель хуже байесовской), а для НИЖНЕЙ границы ошибки Фано нужна НИЖНЯЯ
    оценка H(A|F). Поэтому plug-in fano_floor может ПРЕВЫШАТЬ эмпирику
    (gap_empirical_minus_floor < 0 в docs/fano_frontier.json) — это признак, что
    величина не работает как нижняя граница. Читать как индикатор остаточной
    неопределённости данной модели, не как «пол» атрибуции.
  * I(A;F) = H(A) − H(A|F) — ВАЛИДНАЯ нижняя граница информации, которую признаки
    несут о авторе: истинная взаимная информация не меньше оценённой по данной
    модели (обработка данных не добавляет информации). Эту величину цитировать
    можно как «признаки несут не менее I(A;F) бит».
  * Модель-НЕЗАВИСИМОЙ нетривиальной нижней границы ошибки для книжных объёмов
    здесь нет: корректная двухточечная граница (Ле Кам / Бхаттачарья по пуловым
    частотам, при токенной единице счёта с поправкой на пачкообразность слов)
    для реальных пар авторов на книжной длине падает практически к нулю —
    «сертификат неразрешимости» пары из неё не получается. Поэтому проект не
    публикует никакой нижней границы ошибки как границы.
  * H(A|F) оценивается по OOF-постериорам; при плохой калибровке (ECE≈0.30 —
    см. Boenninghoff 2021 и нашу metrics.expected_calibration_error) оценка
    СМЕЩЕНА. Мы ОТЧЁТНО сообщаем ECE рядом. НЕ сравнивайте «пол» и «эмпирику»
    вслепую.
  * K=2 (пара) нельзя через (H−1)/log(K−1) (деление на 0) — для пар берём
    h⁻¹ бинарной энтропии.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

_LOG2 = np.log(2.0)
_EPS = 1e-12  # клиппинг постериоров для устойчивости логарифма


def _row_entropy(probs: np.ndarray) -> np.ndarray:
    """Энтропия Шеннона (биты) построчно: (n,) для (n, k) постериоров."""
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log2(p)).sum(axis=1)


def prior_entropy(y_true: np.ndarray, n_authors: int) -> float:
    """H(A) по эмпирическому распределению меток авторов в выборке (биты)."""
    y = np.asarray(y_true)
    counts = np.bincount(y, minlength=n_authors).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(prob_matrix: np.ndarray) -> float:
    """H(A|F) = средняя по книгам энтропия OOF-постериора (биты).

    Это оценка остаточной неопределённости Bayes-оптимального классификатора при
    данных признаках. Смещена при плохой калибровке постериоров (см. ECE).
    """
    return float(np.mean(_row_entropy(prob_matrix)))


def mutual_information(H_A: float, H_A_given_F: float) -> float:
    """I(A;F) = H(A) − H(A|F) (биты). Сколько информации о авторе несут признаки."""
    return float(max(0.0, H_A - H_A_given_F))


def fano_floor(H_A_given_F: float, n_authors: int) -> float:
    """Плагин-величина Фано (H(A|F) − 1)/log2(K−1) от энтропии постериоров МОДЕЛИ.

    ВНИМАНИЕ: это НЕ валидная нижняя граница ошибки. Энтропия постериоров модели
    верхне оценивает истинную H(A|F), а Фано для нижней границы требует нижнюю
    оценку H(A|F); поэтому величина может превышать эмпирику (см. секцию ЧЕСТНОСТЬ).
    Нетривиальной модель-независимой замены для книжных объёмов нет (см. ЧЕСТНОСТЬ). Для K≤1 — 0.
    Для пар (K=2) используйте binary_bayes_floor (здесь log2(1)=0).
    """
    if n_authors <= 2:
        return float("nan")  # парный случай — отдельной функцией
    denom = np.log2(n_authors - 1)
    pe = (H_A_given_F - 1.0) / denom
    return float(min(1.0, max(0.0, pe)))


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """h(p) = -p·log2 p - (1-p)·log2(1-p), для p в [0,1] (поэлементно)."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _hinv_scalar(h_target: float) -> float:
    """Меньший корень h(p) = h_target (бинарная Bayes-ошибка по энтропии).

    h(p)=h(1-p), симметрична; меньший p в [0, 0.5] — честная ошибка. Бисекция.
    """
    if h_target <= 0:
        return 0.0
    if h_target >= 1.0:
        return 0.5
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _binary_entropy(np.array([mid]))[0] < h_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binary_bayes_floor(prob_AB: np.ndarray) -> float:
    """Бинарный Bayes-пол для пары (A,B): h⁻¹(⟨H₂ постериора⟩) по книгам пары.

    prob_AB: (n_books_in_pair, 2) — постериоры, ограниченные на {A,B} и
    перенормированные. Возвращает P_e в [0, 0.5] — вероятность перепутать A и B
    даже идеальным классификатором. ~0.5 = пара неразличима («пол»).
    """
    p = np.clip(prob_AB[:, 1], _EPS, 1 - _EPS)  # вер-ть «второго» автора
    h2 = float(np.mean(_binary_entropy(p)))
    return _hinv_scalar(h2)


def fano_book_level(prob_matrix: np.ndarray, y_true: np.ndarray,
                    n_authors: int, ece: float | None = None) -> Dict[str, float]:
    """Пол + эмпирика на уровне книг по OOF-матрице постериоров.

    Возвращает словарь с H(A), H(A|F), I(A;F), fano_floor (нижняя граница ошибки),
    empirical_error (1 − accuracy), gap (= empirical − floor), и ECE (если задан).
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    y_true = np.asarray(y_true)
    H_A = prior_entropy(y_true, n_authors)
    H_AF = conditional_entropy(prob_matrix)
    I_AF = mutual_information(H_A, H_AF)
    floor = fano_floor(H_AF, n_authors)
    pred = prob_matrix.argmax(axis=1)
    empirical_err = float(1.0 - np.mean(pred == y_true))
    out = {
        "n_books": int(len(y_true)),
        "n_authors": int(n_authors),
        "H_A_bits": round(H_A, 4),
        "H_A_max_bits": round(float(np.log2(n_authors)), 4),  # если бы классы были равны
        "H_A_given_F_bits": round(H_AF, 4),
        "I_AF_bits": round(I_AF, 4),
        "fano_floor_Pe": None if np.isnan(floor) else round(floor, 4),
        "empirical_error": round(empirical_err, 4),
        "empirical_accuracy": round(1.0 - empirical_err, 4),
        "gap_empirical_minus_floor": None if np.isnan(floor) else round(empirical_err - floor, 4),
    }
    if ece is not None:
        out["ECE"] = round(float(ece), 4)
    return out


def pairwise_floor(prob_matrix: np.ndarray, y_true: np.ndarray,
                   n_authors: int, pairs: Sequence[Tuple[int, int]]) -> List[Dict]:
    """Пер-pair Bayes-пол неразличимости. prob_matrix — полный (n_books, K).

    Для каждой пары (i,j): берём книги этих авторов, ограничиваем постериор на {i,j},
    перенормируем, считаем бинарный пол. Чем ближе к 0.5 — тем неразличимее пара
    (тем труднее заметить «чужую руку» между ними — это и есть floor детекции).
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    y_true = np.asarray(y_true)
    rows = []
    for i, j in pairs:
        mask = (y_true == i) | (y_true == j)
        if mask.sum() < 2:
            continue
        sub = prob_matrix[mask][:, [i, j]]
        s = sub.sum(axis=1, keepdims=True)
        sub = sub / np.where(s > 0, s, 1.0)
        pe = binary_bayes_floor(sub)
        rows.append({"a": int(i), "b": int(j), "n_books_pair": int(mask.sum()),
                     "bayes_floor_Pe": round(pe, 4), "indistinguishable": pe >= 0.45})
    return rows


# --- open-set / outsider: p(M_out | data) ---

def typicality_scores(prob_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-book «тичичность»: насколько постериор похож на уверенную in-set атрибуцию.

    Возвращает несколько сигналов (выше = «типичнее/увереннее» = скорее in-set):
      max_prob   — максимум постериора (главный сигнал);
      top2_mass  — суммарная масса топ-2 (устойчив к расплывчатости);
      margin     — разрыв топ1−топ2;
      neg_entropy — минус энтропия (бит).
    Outsider/замаскированный текст → низкая типичность (диффузный постериор).
    """
    p = np.clip(np.asarray(prob_matrix, dtype=np.float64), _EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    order = np.sort(p, axis=1)
    ent = _row_entropy(p)
    return {
        "max_prob": order[:, -1],
        "top2_mass": order[:, -1] + order[:, -2],
        "margin": order[:, -1] - order[:, -2],
        "neg_entropy": float(np.log2(p.shape[1])) - ent,   # 0 = равномерный, max = уверен
    }


def outsider_probability(score_in: np.ndarray, score_out: np.ndarray,
                         scores_query: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """P(M_out | score) через эмпирическое отношение плотностей (гистограммы + Лаплас).

    score: скаляр «выше = типичнее/более in-set» (напр. max_prob). По двум выборкам
    (in-set книг, outsider-книг) строим P(outsider|score) для query-книг. Возвращает
    массив p_out ∈ [0,1]. Честно: зависят от бина и от того, что считается «outsider»
    (здесь — hold-out автор, не замаскированный in-set — см. disguise в fano_disguise).
    """
    si = np.asarray(score_in, dtype=np.float64)
    so = np.asarray(score_out, dtype=np.float64)
    sq = np.asarray(scores_query, dtype=np.float64)
    lo = float(min(si.min(), so.min(), sq.min()))
    hi = float(max(si.max(), so.max(), sq.max()))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)

    def hist(s):
        h, _ = np.histogram(s, bins=edges)
        return h.astype(np.float64) + 1.0  # сглаживание Лапласа

    hin, hout = hist(si), hist(so)
    pin, pout = hin / hin.sum(), hout / hout.sum()
    prior_out = len(so) / (len(si) + len(so))
    prior_in = 1.0 - prior_out
    idx = np.clip(np.digitize(sq, edges) - 1, 0, n_bins - 1)
    num = prior_out * pout[idx]
    den = prior_in * pin[idx] + prior_out * pout[idx]
    return num / den


# ---------------------------------------------------------------------------
# self-test: синтетика с известным ответом (запуск: python -m stylo.eval.fano)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    K = 10
    rng = np.random.default_rng(0)
    y = rng.integers(0, K, size=400)

    # 1) почти-уверенные постериоры → пол ≈ 0, эмпирика ≈ 0
    conf = np.full((400, K), 0.01)
    conf[np.arange(400), y] = 0.91
    conf /= conf.sum(axis=1, keepdims=True)
    r1 = fano_book_level(conf, y, K)
    assert r1["fano_floor_Pe"] is not None and r1["fano_floor_Pe"] < 0.05, r1
    assert r1["empirical_error"] == 0.0, r1

    # 2) равномерные постериоры → H(A|F)=log2 K=I=0 → пол = (log2K − 1)/log2(K−1)
    uni = np.full((400, K), 1.0 / K)
    r2 = fano_book_level(uni, y, K)
    # значения в словаре округлены до 4 знаков — толеранс 1e-3
    assert abs(r2["H_A_given_F_bits"] - np.log2(K)) < 1e-3, r2
    assert r2["I_AF_bits"] < 1e-3, r2
    expected_floor = (np.log2(K) - 1.0) / np.log2(K - 1)
    assert abs(r2["fano_floor_Pe"] - expected_floor) < 1e-3, (r2, expected_floor)

    # 3) пара: идентичные распределения A и B → пол ≈ 0.5 (неразличимы)
    pair = np.full((20, 2), 0.5)
    assert abs(binary_bayes_floor(pair) - 0.5) < 1e-6, binary_bayes_floor(pair)
    # пара: идеально разделённые → пол ≈ 0
    pair2 = np.zeros((20, 2)); pair2[:, 0] = 0.999; pair2[:, 1] = 0.001
    assert binary_bayes_floor(pair2) < 0.02, binary_bayes_floor(pair2)

    print("fano self-test OK:", r1, r2)


if __name__ == "__main__":
    _self_test()
