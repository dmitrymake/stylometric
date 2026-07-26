"""Descriptive distribution diagnostics; invalid certificate APIs are withdrawn.

The former implementation tensorized pooled feature affinity by chunk count and
treated the minimum per-channel result as a joint lower bound. That observation
unit is invalid for dependent chunks/channels, and repository evidence records
achieved errors below the claimed floors. Public lower-bound and verdict entry
points therefore fail with ``WITHDRAWN_INVALID_UNIT``.

Exact pre-withdrawal source, output, and falsification bytes are preserved under
``research/evidence/withdrawn_certificates_v1``. Bhattacharyya/Hellinger and
bootstrap-distance helpers remain available only as descriptive diagnostics.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Dict, List, NoReturn, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12
_FW_TOKEN = r"(?u)\b\w+\b"
_POS_TOKEN = r"(?u)[A-Z]+"
WITHDRAWN_INVALID_UNIT = "WITHDRAWN_INVALID_UNIT"
HISTORICAL_METHOD_SHA256 = (
    "596cbdb9977b56e12c7b4ca8819100d91102ff20a3a01ede0c49005815d781ea"
)


class WithdrawnCertificateError(RuntimeError):
    """A caller attempted to use the known-invalid inferential certificate."""


def _raise_withdrawn(operation: str) -> NoReturn:
    raise WithdrawnCertificateError(
        f"{WITHDRAWN_INVALID_UNIT}: {operation} used dependent chunks/channels as "
        "independent tensor-product observations; use descriptive divergence only. "
        f"Historical method sha256={HISTORICAL_METHOD_SHA256}"
    )


# ---------------------------------------------------------------------------
# базовые дивергенции
# ---------------------------------------------------------------------------
def bhattacharyya_affinity(freq_a: np.ndarray, freq_b: np.ndarray) -> float:
    """ρ = Σ_j √(p_A[j]·p_B[j]) — аффинность Бхаттачарья двух частотных векторов."""
    fa = np.asarray(freq_a, dtype=np.float64)
    fb = np.asarray(freq_b, dtype=np.float64)
    rho = float(np.sqrt(fa * fb).sum())
    return min(1.0, max(0.0, rho))


def hellinger2(freq_a: np.ndarray, freq_b: np.ndarray) -> float:
    """D = 1 − ρ — квадрат расстояния Хеллингера (аффинностная форма)."""
    return 1.0 - bhattacharyya_affinity(freq_a, freq_b)


def _affinity_rows(fa_rows: np.ndarray, fb_rows: np.ndarray) -> np.ndarray:
    """ρ построчно для двух (m, d) наборов частот → (m,)."""
    return np.clip(np.sqrt(fa_rows * fb_rows).sum(axis=1), 0.0, 1.0)


def _normalize_rows(counts: np.ndarray) -> np.ndarray:
    """Счётчики → частоты построчно; пустые строки → равномерное."""
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum(axis=1, keepdims=True)
    d = counts.shape[1]
    out = np.where(s > 0, counts / np.where(s > 0, s, 1.0), 1.0 / d)
    return out


def author_pooled_freq(book_counts: np.ndarray, book_author: np.ndarray,
                       a: int, book_subset: Optional[Sequence[int]] = None) -> np.ndarray:
    """Пуловая частота признаков автора a: сумма счётчиков по его книгам, нормированная.

    book_counts: (n_books, d) — суммарные счётчики признаков по каждой книге.
    book_author: (n_books,) — индекс автора каждой книги.
    book_subset: если задан, пул только по этим индексам книг (внутри автора a).
    """
    book_counts = np.asarray(book_counts, dtype=np.float64)
    if book_subset is None:
        idx = np.where(np.asarray(book_author) == a)[0]
    else:
        idx = np.asarray(book_subset, dtype=int)
    pooled = book_counts[idx].sum(axis=0)
    s = pooled.sum()
    if s <= 0:
        return np.full(book_counts.shape[1], 1.0 / book_counts.shape[1])
    return pooled / s


# ---------------------------------------------------------------------------
# кривая пола, горизонт, эффективное n
# ---------------------------------------------------------------------------
def floor_curve(D: float, n) -> np.ndarray | float:
    """Withdrawn lower-bound compatibility entry point; always fails closed."""
    _raise_withdrawn("floor_curve")


def separability_horizon(D: float, tau: float = 0.05) -> float:
    """Withdrawn floor-derived compatibility entry point; always fails closed."""
    _raise_withdrawn("separability_horizon")


def effective_n(n: float, r: float) -> float:
    """n_eff = n·(1 − r)/(1 + r), r — лаг-1 автокорреляция; r клиппится к [0, 0.999]."""
    r = float(min(0.999, max(0.0, r)))
    return float(max(1.0, n * (1.0 - r) / (1.0 + r)))


# ---------------------------------------------------------------------------
# кластерный бутстрап по книгам
# ---------------------------------------------------------------------------
def author_boot_freqs(book_counts_a: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """B бутстрап-реплик пуловых частот автора: ресэмпл книг с возвращением.

    book_counts_a: (n_books_a, d). Возврат: (B, d) — нормированные пуловые частоты.
    Векторизовано: индексная выборка книг + суммирование строк.
    """
    book_counts_a = np.asarray(book_counts_a, dtype=np.float64)
    n_a = book_counts_a.shape[0]
    idx = rng.integers(0, n_a, size=(B, n_a))          # (B, n_a)
    pooled = book_counts_a[idx].sum(axis=1)            # (B, d)
    return _normalize_rows(pooled).astype(np.float32)


def clustered_bootstrap_D(boot_freqs_a: np.ndarray, boot_freqs_b: np.ndarray,
                          freq_a_full: np.ndarray, freq_b_full: np.ndarray,
                          level: float = 0.95) -> Dict[str, float]:
    """D-сертификат пары из предвычисленных бутстрап-реплик частот A и B.

    boot_freqs_*: (B, d) реплики (author_boot_freqs). freq_*_full — пуловые частоты
    по всем книгам (точечная оценка D). Возврат: точка, нижний/верхний 95%-предел, ширина.
    """
    fa = np.asarray(boot_freqs_a, dtype=np.float64)
    fb = np.asarray(boot_freqs_b, dtype=np.float64)
    rho_boot = _affinity_rows(fa, fb)
    D_boot = 1.0 - rho_boot
    lo = 100.0 * (1.0 - level)          # 5-й перцентиль
    hi = 100.0 * level                  # 95-й перцентиль
    D_lcb = float(np.percentile(D_boot, lo))
    D_ucb = float(np.percentile(D_boot, hi))
    D_point = hellinger2(freq_a_full, freq_b_full)
    return {"D_point": D_point, "D_lcb": D_lcb, "D_ucb": D_ucb,
            "ci_width": float(D_ucb - D_lcb)}


def hellinger2_self(book_counts: np.ndarray, book_author: np.ndarray, a: int,
                    B: int, rng: np.random.Generator, level: float = 0.95) -> Dict[str, float]:
    """Split-half дивергенция автора a: делим книги на две половины, D между пулами.

    Шумовой пол (тот же автор). Возврат: точка (медиана), верхний 95%-предел.
    Требует ≥2 книги; иначе None.
    """
    idx = np.where(np.asarray(book_author) == a)[0]
    if len(idx) < 2:
        return {"D_self_point": None, "D_self_ucb": None, "n_books": int(len(idx))}
    bc = np.asarray(book_counts, dtype=np.float64)
    half = len(idx) // 2
    Ds = np.empty(B)
    for i in range(B):
        perm = rng.permutation(idx)
        left, right = perm[:half], perm[half:2 * half] if 2 * half <= len(idx) else perm[half:]
        fa = author_pooled_freq(bc, book_author, a, book_subset=left)
        fb = author_pooled_freq(bc, book_author, a, book_subset=right)
        Ds[i] = hellinger2(fa, fb)
    return {"D_self_point": float(np.median(Ds)),
            "D_self_ucb": float(np.percentile(Ds, 100.0 * level)),
            "n_books": int(len(idx))}


# ---------------------------------------------------------------------------
# лаг-1 автокорреляция лог-отношения частот внутри книг
# ---------------------------------------------------------------------------
def _lag1_autocorr(x: np.ndarray) -> Optional[float]:
    """Лаг-1 автокорреляция ряда (общая нормировка на дисперсию). None при len<3."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return None
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= _EPS:
        return None
    num = float((xc[:-1] * xc[1:]).sum())
    return num / denom


def pair_autocorr_r(chunk_csr, chunk_rowsum: np.ndarray, book_spans: Dict[int, Tuple[int, int]],
                    books_a: Sequence[int], books_b: Sequence[int],
                    freq_a: np.ndarray, freq_b: np.ndarray) -> float:
    """Взвешенная средняя лаг-1 автокорреляция лог-LR чанков внутри книг пары.

    Дискриминант w = log(p_A/p_B) (частоты со сглаживанием только для проекции);
    score чанка = его нормированный вектор частот · w. Внутри каждой книги ≥3 чанков
    считаем лаг-1 автокорреляцию, усредняем с весом (число лаг-пар).
    """
    d = len(freq_a)
    sa = (np.asarray(freq_a) * 1.0)
    sb = (np.asarray(freq_b) * 1.0)
    # аддитивное сглаживание только для устойчивости логарифма дискриминанта
    pa = (sa + 1.0 / d) / (sa.sum() + 1.0)
    pb = (sb + 1.0 / d) / (sb.sum() + 1.0)
    w = np.log(pa) - np.log(pb)
    num_w = 0.0
    den_w = 0.0
    for bk in list(books_a) + list(books_b):
        start, length = book_spans[bk]
        if length < 3:
            continue
        sub = chunk_csr[start:start + length]
        s = sub.dot(w)
        rs = chunk_rowsum[start:start + length]
        valid = rs > 0
        if valid.sum() < 3:
            continue
        score = s[valid] / rs[valid]
        r = _lag1_autocorr(score)
        if r is None:
            continue
        weight = valid.sum() - 1
        num_w += r * weight
        den_w += weight
    if den_w <= 0:
        return 0.0
    return float(num_w / den_w)


# ---------------------------------------------------------------------------
# структура канала
# ---------------------------------------------------------------------------
def build_channel_chunk_matrix(channel: str, texts: Sequence[str], reps: Sequence,
                               groups: Sequence[str], authors: List[str],
                               cache_path: Optional[pathlib.Path] = None) -> "Channel":
    """Построить канал (счётчики признаков на уровне чанков + агрегация по книгам).

    channel='fw'  — функциональные слова (fixed_list ~399, общий носитель на всём корпусе);
    channel='pos' — POS-биграммы строго (2,2) (размерность ограничена алфавитом POS-тегов).

    Чанки берутся В ПОРЯДКЕ следования (внутри книги — порядок файлов = порядок текста),
    что нужно для лаг-1 автокорреляции. Кэш — npz (cache_path).
    """
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer
    from ..lang import function_words

    if cache_path is not None and pathlib.Path(cache_path).exists():
        return _load_channel(channel, pathlib.Path(cache_path), authors)

    if channel == "fw":
        vocab = sorted(function_words("ru"))
        vec = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern=_FW_TOKEN)
        X = vec.fit_transform(list(texts)).tocsr()
        stamp = "function_words:fixed_list"
    elif channel == "pos":
        vec = CountVectorizer(analyzer="word", ngram_range=(2, 2),
                              token_pattern=_POS_TOKEN, lowercase=False)
        X = vec.fit_transform([r.pos_str for r in reps]).tocsr()
        stamp = "pos_bigrams:(2,2)"
    else:
        raise ValueError(f"неизвестный канал: {channel}")
    feat_names = list(vec.get_feature_names_out())
    d = X.shape[1]

    # порядок книг = порядок первого появления в потоке чанков (книги контигуальны)
    auth2idx = {a: k for k, a in enumerate(authors)}
    book_order: List[str] = []
    book_of_chunk = np.empty(len(groups), dtype=int)
    spans: Dict[int, Tuple[int, int]] = {}
    cur = -1
    for ci, g in enumerate(groups):
        if not book_order or book_order[-1] != g:
            book_order.append(g)
            cur += 1
            spans[cur] = (ci, 0)
        book_of_chunk[ci] = cur
        st, ln = spans[cur]
        spans[cur] = (st, ln + 1)
    n_books = len(book_order)
    book_author = np.array([auth2idx[g.split("/", 1)[0]] for g in book_order], dtype=int)
    chunks_per_book = np.array([spans[b][1] for b in range(n_books)], dtype=int)

    # book_counts: суммирование строк чанков по книгам через матрицу инцидентности
    incid = csr_matrix((np.ones(len(groups)), (book_of_chunk, np.arange(len(groups)))),
                       shape=(n_books, len(groups)))
    book_counts = np.asarray((incid @ X).todense(), dtype=np.float64)
    chunk_rowsum = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)

    ch = Channel(name=channel, stamp=stamp, chunk_csr=X, chunk_rowsum=chunk_rowsum,
                 book_counts=book_counts, book_author=book_author, book_ids=book_order,
                 book_spans=spans, chunks_per_book=chunks_per_book, d=d, authors=list(authors))
    if cache_path is not None:
        _save_channel(ch, pathlib.Path(cache_path), feat_names)
    return ch


def _save_channel(ch: "Channel", path: pathlib.Path, feat_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=ch.name, stamp=ch.stamp,
        data=ch.chunk_csr.data, indices=ch.chunk_csr.indices, indptr=ch.chunk_csr.indptr,
        shape=np.array(ch.chunk_csr.shape),
        chunk_rowsum=ch.chunk_rowsum, book_counts=ch.book_counts, book_author=ch.book_author,
        book_ids=np.array(ch.book_ids), chunks_per_book=ch.chunks_per_book,
        span_start=np.array([ch.book_spans[b][0] for b in range(len(ch.book_ids))]),
        span_len=np.array([ch.book_spans[b][1] for b in range(len(ch.book_ids))]),
        feat_names=np.array(feat_names),
    )


def _load_channel(channel: str, path: pathlib.Path, authors: List[str]) -> "Channel":
    from scipy.sparse import csr_matrix
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"channel cache must be a regular non-symlink file: {path}")
    # Object arrays invoke pickle.  The writer above stores only numeric and
    # Unicode arrays, so executable deserialisation is neither needed nor
    # permitted at this regenerable cache boundary.
    z = np.load(path, allow_pickle=False)
    required = {
        "name", "stamp", "data", "indices", "indptr", "shape",
        "chunk_rowsum", "book_counts", "book_author", "book_ids",
        "chunks_per_book", "span_start", "span_len", "feat_names",
    }
    if set(z.files) != required:
        raise ValueError(
            f"channel cache schema mismatch: expected {sorted(required)}, got {sorted(z.files)}"
        )
    if any(z[name].dtype.kind == "O" for name in z.files):
        raise ValueError("channel cache contains forbidden object arrays")
    X = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    spans = {b: (int(z["span_start"][b]), int(z["span_len"][b])) for b in range(len(z["book_ids"]))}
    return Channel(name=str(z["name"]), stamp=str(z["stamp"]), chunk_csr=X,
                   chunk_rowsum=z["chunk_rowsum"], book_counts=z["book_counts"],
                   book_author=z["book_author"], book_ids=list(z["book_ids"]),
                   book_spans=spans, chunks_per_book=z["chunks_per_book"],
                   d=int(z["shape"][1]), authors=list(authors))


@dataclasses.dataclass
class Channel:
    name: str
    stamp: str                       # штамп Φ (человекочитаемый)
    chunk_csr: object                # (n_chunks, d) csr счётчики
    chunk_rowsum: np.ndarray         # (n_chunks,)
    book_counts: np.ndarray          # (n_books, d)
    book_author: np.ndarray          # (n_books,)
    book_ids: List[str]
    book_spans: Dict[int, Tuple[int, int]]   # book_idx -> (start_chunk, len)
    chunks_per_book: np.ndarray      # (n_books,)
    d: int
    authors: List[str]


# ---------------------------------------------------------------------------
# вердикт по паре
# ---------------------------------------------------------------------------
def certify_pair(i: int, j: int, channels: List[Channel],
                 boot: Dict[str, Dict[int, np.ndarray]],
                 full_freq: Dict[str, Dict[int, np.ndarray]],
                 self_div: Dict[str, Dict[int, Dict]],
                 params: Dict) -> Dict:
    """Withdrawn inferential verdict entry point; always fails closed."""
    _raise_withdrawn("certify_pair")


def certify_all_pairs(channels: List[Channel], params: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
    """Withdrawn all-pairs certificate entry point; always fails closed."""
    _raise_withdrawn("certify_all_pairs")


# ---------------------------------------------------------------------------
# self-test: descriptive primitives remain; inferential APIs fail closed
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # аффинность: одинаковые частоты → ρ=1, D=0
    f = np.array([0.2, 0.3, 0.5])
    assert abs(bhattacharyya_affinity(f, f) - 1.0) < 1e-9
    assert hellinger2(f, f) < 1e-9
    # непересекающиеся → ρ=0
    fa = np.array([1.0, 0.0]); fb = np.array([0.0, 1.0])
    assert bhattacharyya_affinity(fa, fb) < 1e-9
    # effective_n: r=0 → n; r>0 → меньше n
    assert abs(effective_n(50, 0.0) - 50) < 1e-9
    assert effective_n(50, 0.5) < 50
    # бутстрап D: две одинаковые книги-матрицы → D≈0
    rng = np.random.default_rng(0)
    bc = np.array([[10.0, 5, 1], [8, 6, 2], [9, 4, 3]])
    bf = author_boot_freqs(bc, 200, rng)
    res = clustered_bootstrap_D(bf, bf, _normalize_rows(bc.sum(0, keepdims=True))[0],
                                _normalize_rows(bc.sum(0, keepdims=True))[0])
    assert res["D_point"] < 1e-9, res
    for operation in (
        lambda: floor_curve(0.1, 5),
        lambda: separability_horizon(0.1),
        lambda: certify_pair(0, 1, [], {}, {}, {}, {}),
        lambda: certify_all_pairs([]),
    ):
        try:
            operation()
        except WithdrawnCertificateError as exc:
            assert WITHDRAWN_INVALID_UNIT in str(exc)
        else:
            raise AssertionError("withdrawn certificate operation did not fail closed")
    print("descriptive certificate diagnostics self-test OK; inferential APIs withdrawn")


if __name__ == "__main__":
    _self_test()
