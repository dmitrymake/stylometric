"""Безнадзорный поиск двух «рук» ВНУТРИ корпуса автора (идея: Ильф vs Петров).

Проблема: у соавторов нет сольных образцов → обучить раздельные классы не на чем.
Обход: НЕ обучать, а проверить, не распадается ли их собственный корпус на два
устойчивых стилевых кластера ПО ТЕМАТИЧЕСКИ-ИНВАРИАНТНЫМ признакам (функц. слова,
POS/синтаксис/морфология — НЕ лексика/тема). Сравниваем «двугорбость» с одноавторским
контролем: если у соавторов разделение значимо сильнее, чем у заведомо одного автора —
это след двух рук (а не темы/шума).

Честная оговорка: POS/синтаксис тоже частично коррелируют с жанром; вывод — только в
сравнении с контролями, и только как ЭКСПЕРИМЕНТ, не доказательство.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

# тематически-инвариантный набор блоков (без char-n-грамм и без сырой лексики)
TOPIC_INVARIANT = {
    "char_ngrams": False,
    "function_words": True,
    "syntax": True,
    "pos_ngrams": True,
    "punctuation_ngrams": True,
    "dependency": True,
    "morphology": True,
    "length_dist": True,
    "embeddings": False,
}


@dataclass
class StyleBasis:
    """Leak-free базис стиль-пространства: фитится на REFERENCE (без цели), применяется к цели.

    Решает in-sample SVD-leak: silhouette/кластеризация на пространстве, фититом ВНУТРИ
    целевого корпуса, ЗАВЫШАЕТ разделение (см. заметку об IN-SAMPLE в style_embedding). Здесь векторизатор,
    MaxAbsScaler, TruncatedSVD, StandardScaler фитятся ТОЛЬКО на reference-текстах; целевые
    тексты только transform'ятся — никакой информации о цели не утекает в геометрию.
    """
    vec: object
    maxabs: MaxAbsScaler
    svd: TruncatedSVD
    std: StandardScaler

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vec.transform(list(texts))
        X = self.maxabs.transform(X)
        Z = self.svd.transform(X)
        return self.std.transform(Z)


def fit_style_basis(reference_texts: Sequence[str], cfg, n_components: int = 40,
                    topic_strict: bool = True) -> StyleBasis:
    """Обучить StyleBasis на REFERENCE-текстах (НЕ на цели). leak-free для силуэта/кластеризации."""
    from ..vectorizer import StyloVectorizer
    vec = StyloVectorizer.from_config(cfg, enabled_override=TOPIC_INVARIANT, topic_strict=topic_strict)
    X = vec.fit_transform(list(reference_texts))
    maxabs = MaxAbsScaler().fit(X)
    Xs = maxabs.transform(X)
    nc = max(2, min(n_components, Xs.shape[1] - 1, Xs.shape[0] - 1))
    svd = TruncatedSVD(n_components=nc, random_state=42).fit(Xs)
    std = StandardScaler().fit(svd.transform(Xs))
    return StyleBasis(vec=vec, maxabs=maxabs, svd=svd, std=std)


def style_embedding(texts: Sequence[str], cfg, n_components: int = 40,
                    topic_strict: bool = True, basis: "StyleBasis | None" = None) -> np.ndarray:
    """Тематически-инвариантные стиль-признаки чанков → плотное низкоразмерное пространство.

    topic_strict=True (умолчание): строгий topic-control — function_words=fixed_list +
    глушение pos_ratios/lexical_richness (mfw недопустим как 'тематически-инвариантный':
    частотные слова несут тему/жанр). cross-author genre-AUC≈0.84
    (docs/audit_genre_crossauthor.json) — представление НЕ чисто-идиолектное, несёт жанр.

    LEAK-FREE: передайте `basis` (fit_style_basis на отдельном reference) — тогда векторизатор/
    SVD/scaler фитились БЕЗ цели, силуэт/кластеризация не завышены. Если basis=None —
    фитится IN-SAMPLE (на тех же текстах): для метрик разделимости недопустимо (завышает
    разделение); допустимо только для дисперсии/центроидов или быстрых просмотров.
    """
    if basis is not None:
        return basis.transform(texts)
    from ..vectorizer import StyloVectorizer
    vec = StyloVectorizer.from_config(cfg, enabled_override=TOPIC_INVARIANT, topic_strict=topic_strict)
    X = vec.fit_transform(list(texts))
    X = MaxAbsScaler().fit_transform(X)
    nc = max(2, min(n_components, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(n_components=nc, random_state=42).fit_transform(X)
    return StandardScaler().fit_transform(Z)


def bimodality(Z: np.ndarray, k: int = 2, seed: int = 42) -> Tuple[float, np.ndarray]:
    """Силуэт разбиения на k кластеров (выше → отчётливее два стиля) + метки."""
    if Z.shape[0] < k + 1:
        return 0.0, np.zeros(Z.shape[0], dtype=int)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    try:
        sil = float(silhouette_score(Z, km.labels_))
    except Exception:
        sil = 0.0
    return sil, km.labels_


def heterogeneity_score(texts: Sequence[str], cfg, k: int = 2,
                        basis: "StyleBasis | None" = None) -> dict:
    """Оценка «двугорбости» корпуса: силуэт k=2 + размеры кластеров.

    basis: если задан (fit_style_basis на отдельном reference) — leak-free; иначе in-sample
    (для силуэта недопустимо, завышает разделение).
    """
    Z = style_embedding(texts, cfg, basis=basis)
    sil, labels = bimodality(Z, k=k)
    sizes = [int(np.sum(labels == c)) for c in range(k)]
    return {"silhouette_k2": round(sil, 4), "n_chunks": int(Z.shape[0]),
            "cluster_sizes": sizes, "labels": labels.tolist()}


def compare_to_controls(target_texts: Sequence[str], target_name: str,
                        control_corpora: dict, cfg,
                        reference_texts: Sequence[str] | None = None) -> dict:
    """Сравнить «двугорбость» цели (Ильф-Петров) с одноавторскими контролями.

    control_corpora: {author_name: [chunk_texts]}. Возвращает силуэты и вывод:
    выделяется ли цель сильнее контролей.

    LEAK-FREE: передайте reference_texts (напр. объединение контролев или сторонний корпус) —
    basis фитится на нём и применяется к цели+контролям, силуэт не завышен in-sample SVD.
    Без reference_texts — in-sample (для метрик разделимости недопустимо, завышает разделение).
    """
    basis = None
    if reference_texts is not None:
        basis = fit_style_basis(reference_texts, cfg)
    tgt = heterogeneity_score(target_texts, cfg, basis=basis)
    controls = {name: heterogeneity_score(txts, cfg, basis=basis)["silhouette_k2"]
                for name, txts in control_corpora.items()}
    cvals = list(controls.values())
    cmean = float(np.mean(cvals)) if cvals else 0.0
    cstd = float(np.std(cvals)) if cvals else 0.0
    z = (tgt["silhouette_k2"] - cmean) / cstd if cstd > 1e-9 else 0.0
    return {
        "target": target_name,
        "target_silhouette": tgt["silhouette_k2"],
        "target_cluster_sizes": tgt["cluster_sizes"],
        "control_silhouettes": controls,
        "control_mean": round(cmean, 4),
        "control_std": round(cstd, 4),
        "z_vs_controls": round(z, 2),
        "verdict": (
            "корпус соавторов значимо «двугорбее» одноавторских (z≥2) — возможен след двух рук"
            if z >= 2 else
            "разделение не сильнее, чем у одного автора — НЕТ статистического следа двух рук"
        ),
        "labels": tgt["labels"],
    }
