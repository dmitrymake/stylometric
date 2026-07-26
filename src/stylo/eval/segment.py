"""Сегментная (rolling) атрибуция: поиск «чужих» отрезков ВНУТРИ книги и их авторство.

Идея (rolling stylometry, Eder + почанковая LR):
  - книга режется на чанки в ПОРЯДКЕ следования;
  - для каждого чанка — вероятности по набору КАНДИДАТОВ (renormalized);
  - последовательность сглаживается скользящим окном;
  - находятся контигуальные СЕГМЕНТЫ, где доминирует НЕ основной автор (host) —
    это кандидаты в «чужие» отрезки.

ЧЕСТНОСТЬ (критично): метод склонен выдумывать «смешанное авторство» из-за шума и
тематических сдвигов. Поэтому ОБЯЗАТЕЛЕН нуль-контроль: тот же расчёт на ЗАВЕДОМО
одноавторских книгах host'а даёт фоновую долю «чужих» чанков (baseline). «Чужой»
отрезок считается сигналом, только если доля/уверенность СУЩЕСТВЕННО выше фона.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TOPOLOGY_ROLE = "rolling_attribution_diagnostic"


def chunk_probs(texts: Sequence[str], pipe, authors: List[str]) -> np.ndarray:
    """(n_chunks, n_authors) вероятности, выровненные на полный список authors."""
    p = np.asarray(pipe.predict_proba(list(texts)))
    full = np.zeros((len(texts), len(authors)), dtype=np.float64)
    for j, c in enumerate(pipe.classes_):
        full[:, int(c)] = p[:, j]
    return full


def restrict_renorm(probs: np.ndarray, authors: List[str], candidates: List[str]) -> Tuple[np.ndarray, List[int]]:
    """Оставить только кандидатов и перенормировать по строкам."""
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"duplicate candidates: {candidates}")
    idx = [authors.index(c) for c in candidates]
    sub = probs[:, idx].astype(np.float64)
    s = sub.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return sub / s, idx


def rolling_mean(probs: np.ndarray, win: int) -> np.ndarray:
    """Скользящее среднее по оси чанков (центрированное окно win)."""
    if win <= 1 or probs.shape[0] <= 1:
        return probs
    k = min(win, probs.shape[0])
    if k % 2 == 0:          # чётное окно смещает центр на полшага — приводим к нечётному
        k -= 1
    if k <= 1:
        return probs
    pad = k // 2
    padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    out = np.empty_like(probs)
    for j in range(probs.shape[1]):
        out[:, j] = np.convolve(padded[:, j], ker, mode="valid")[: probs.shape[0]]
    return out


@dataclasses.dataclass
class Segment:
    start: int          # индекс чанка (включительно)
    end: int            # включительно
    author: str
    mean_conf: float

    def length(self) -> int:
        return self.end - self.start + 1


def detect_segments(sub: np.ndarray, candidates: List[str], host: str,
                    conf: float = 0.6, min_run: int = 3) -> List[Segment]:
    """Контигуальные сегменты, где argmax == НЕ host, со средней уверенностью >= conf
    и длиной >= min_run чанков."""
    if host not in candidates:
        raise ValueError(f"host {host!r} not in candidates {candidates}")
    pred = np.argmax(sub, axis=1)
    names = [candidates[i] for i in pred]
    host_idx = candidates.index(host)
    segs: List[Segment] = []
    i = 0
    n = len(names)
    while i < n:
        if pred[i] != host_idx:
            j = i
            while j + 1 < n and pred[j + 1] == pred[i]:
                j += 1
            run_conf = float(np.mean(sub[i : j + 1, pred[i]]))
            if (j - i + 1) >= min_run and run_conf >= conf:
                segs.append(Segment(i, j, candidates[pred[i]], run_conf))
            i = j + 1
        else:
            i += 1
    return segs


def foreign_fraction(sub: np.ndarray, candidates: List[str], host: str) -> float:
    """Доля чанков, чей argmax != host."""
    if host not in candidates:
        raise ValueError(f"host {host!r} not in candidates {candidates}")
    pred = np.argmax(sub, axis=1)
    host_idx = candidates.index(host)
    return float(np.mean(pred != host_idx))


@dataclasses.dataclass
class SegmentReport:
    host: str
    candidates: List[str]
    n_chunks: int
    timeline: List[Tuple[str, float]]       # на чанк: (доминирующий автор, уверенность)
    segments: List[Segment]
    foreign_fraction: float
    null_mean: Optional[float]
    null_std: Optional[float]
    null_z: Optional[float]                 # (foreign - null_mean)/null_std (None если нуль вырожден)
    verdict: str
    null_p_emp: Optional[float] = None      # эмпирический p по одноавторским контролям (псевдосчёт)
    null_degenerate: bool = False           # нуль вырожден (std≈0) → z неинформативен


def analyze(sub: np.ndarray, candidates: List[str], host: str,
            null_mean: Optional[float] = None, null_std: Optional[float] = None,
            null_fracs: Optional[Sequence[float]] = None,
            conf: float = 0.6, min_run: int = 3) -> SegmentReport:
    """Вердикт по сегментам с КОРРЕКТНОЙ обработкой вырожденного нуля.

    null_fracs — пофайловые foreign_fraction одноавторских контролей; по ним считается
    эмпирический p (псевдосчёт). При вырожденном нуле (std≈0) z НЕ считается (был бы делёж
    на ноль → ложное «нет свидетельств»); вместо этого — эмпирический p или явный ABORT.
    """
    pred = np.argmax(sub, axis=1)
    timeline = [(candidates[pred[i]], float(sub[i, pred[i]])) for i in range(sub.shape[0])]
    segs = detect_segments(sub, candidates, host, conf=conf, min_run=min_run)
    ff = foreign_fraction(sub, candidates, host)

    z = None
    p_emp = None
    degenerate = False
    verdict = "—"
    if null_fracs is not None and len(null_fracs) > 0:
        arr = np.asarray(list(null_fracs), dtype=np.float64)
        p_emp = float((1 + np.sum(arr >= ff)) / (1 + len(arr)))   # эмпирический p с псевдосчётом

    if null_mean is not None and null_std is not None and null_std > 1e-9:
        z = (ff - null_mean) / null_std
        if z >= 3 and segs:
            verdict = "сигнал «чужих» отрезков ВЫШЕ фона (z≥3) — стоит присмотреться"
        elif z >= 2:
            verdict = "слабый сигнал (z≈2) — на грани шума"
        else:
            verdict = "в пределах фона одноавторского текста — НЕТ свидетельств смешанного авторства"
    elif null_mean is not None:
        # вырожденный нуль (std≈0): z неинформативен — НЕ делаем вид, что «нет свидетельств»
        degenerate = True
        if p_emp is not None and p_emp <= 0.1 and segs:
            verdict = f"эмпирический p={p_emp:.2f}: доля «чужих» выше одноавторского фона — присмотреться"
        elif p_emp is not None:
            verdict = f"эмпирический p={p_emp:.2f}; нуль вырожден (std≈0) — формального z нет"
        else:
            verdict = "НУЛЬ ВЫРОЖДЕН (std≈0): тест неинформативен — вывод о смешанном авторстве НЕ делается"
    elif p_emp is not None:
        # переданы только пофайловые null_fracs — вердикт ведёт эмпирический p, без z
        if p_emp <= 0.1 and segs:
            verdict = f"эмпирический p={p_emp:.2f}: доля «чужих» выше одноавторского фона — присмотреться"
        else:
            verdict = f"эмпирический p={p_emp:.2f} — в пределах одноавторского фона"
    return SegmentReport(host, candidates, sub.shape[0], timeline, segs, ff,
                         null_mean, null_std, z, verdict, null_p_emp=p_emp,
                         null_degenerate=degenerate)


def delta_chunk_probs(texts: Sequence[str], delta, authors: List[str]) -> np.ndarray:
    """Почанковые «вероятности» от настоящей Burrows Delta (softmax(-distance)),
    выровненные на authors. Второй НЕЗАВИСИМЫЙ движок для перекрёстной проверки с LR."""
    d = np.asarray(delta.distances(list(texts)))      # (n, n_classes)
    x = -d
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
    full = np.zeros((len(texts), len(authors)), dtype=np.float64)
    for j, c in enumerate(delta.classes_):
        full[:, int(c)] = probs[:, j]
    return full


def agreement_fraction(sub_lr: np.ndarray, sub_delta: np.ndarray) -> float:
    """Доля чанков, где оба движка дают одинаковый argmax (согласие LR и Delta)."""
    return float(np.mean(np.argmax(sub_lr, axis=1) == np.argmax(sub_delta, axis=1)))


def synthetic_splice_test(texts_a: List[str], texts_b: List[str], pipe, delta,
                          authors: List[str], candidates: List[str], host: str,
                          win: int = 5, conf: float = 0.6, min_run: int = 3) -> Dict:
    """Нуль-модель 2 (recall детектора): склеить два РАЗНЫХ автора и проверить,
    что метод видит известную точку склейки. Возвращает позицию склейки,
    найденные сегменты «чужого» и попал ли детектор около склейки."""
    texts = list(texts_a) + list(texts_b)
    splice = len(texts_a)
    probs = chunk_probs(texts, pipe, authors)
    sub, _ = restrict_renorm(probs, authors, candidates)
    sub = rolling_mean(sub, win)
    segs = detect_segments(sub, candidates, host, conf=conf, min_run=min_run)
    # «попадание»: есть ли сегмент-«чужой», накрывающий вторую половину (после splice)
    hit = any(s.start >= splice - win and s.author != host for s in segs)
    return {
        "splice_at_chunk": splice,
        "n_chunks": len(texts),
        "segments": [(s.start, s.end, s.author, round(s.mean_conf, 3)) for s in segs],
        "detected_switch_near_splice": bool(hit),
    }


def null_baseline(host_books: Dict[str, List[str]], pipe, authors: List[str],
                  candidates: List[str], host: str, win: int = 5,
                  conf: float = 0.6, min_run: int = 3) -> Tuple[float, float, List[float]]:
    """Фоновая доля «чужих» чанков на ЗАВЕДОМО одноавторских книгах host'а.

    host_books: {book_id: [chunk_texts в порядке]}. Возвращает (mean, std, per_book).
    """
    fracs: List[float] = []
    for book, texts in host_books.items():
        if len(texts) < min_run:
            continue
        probs = chunk_probs(texts, pipe, authors)
        sub, _ = restrict_renorm(probs, authors, candidates)
        sub = rolling_mean(sub, win)
        fracs.append(foreign_fraction(sub, candidates, host))
    if not fracs:
        return 0.0, 0.0, []
    std = float(np.std(fracs, ddof=1)) if len(fracs) > 1 else 0.0   # выборочная std (ddof=1); n<2 → вырожден
    return float(np.mean(fracs)), std, fracs
