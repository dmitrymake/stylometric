"""Канальные фабрики признаков — единый источник для GKF-бенчмарка и LOBO-стекинга.

Каждый канал — функция (train_texts, test_texts) -> (Xtr, Xte); всё, что
обучается (tf-idf статистики, скейлеры, словари блоков), учится ТОЛЬКО на train.
Определения подняты из scripts/run_benchmark.py без изменения поведения; DSP-канал
остаётся локальным в бенчмарке (тяжёлый spaCy-lg кэш, в стек не входит).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler

from ..features.work_vectorizer import validate_work_ids
from ..vectorizer import StyloVectorizer

# (train_texts, test_texts, train_groups=None) -> (Xtr, Xte). ``train_groups=None`` is the legacy
# chunk-level fit (byte-identical); a work-id per train chunk routes work-balanced feature fitting.
ChannelFn = Callable[[List[str], List[str], Optional[Sequence]], Tuple[object, object]]

_ALL_BLOCKS = ["char_ngrams", "function_words", "syntax", "pos_ngrams",
               "punctuation_ngrams", "dependency", "morphology", "length_dist",
               "embeddings"]


def _work_sum_matrix(groups: Sequence) -> csr_matrix:
    """0/1 aggregation matrix G (n_works x n_chunks), one row per train work (first-seen order)."""
    index: Dict = {}
    rows = []
    for g in groups:
        if g not in index:
            index[g] = len(index)
        rows.append(index[g])
    cols = np.arange(len(rows))
    data = np.ones(len(rows))
    return csr_matrix((data, (np.asarray(rows), cols)), shape=(len(index), len(rows)))


def _hashing_channel(hv: HashingVectorizer):
    def f(tr, te, tr_groups=None):
        tr = list(tr)
        Xtr = hv.transform(tr)                             # stateless chunk counts
        tf = TfidfTransformer(sublinear_tf=True)
        if tr_groups is None:
            Xtr_out = tf.fit_transform(Xtr)                # legacy: chunk-level document frequency
        else:
            tr_groups = validate_work_ids(tr_groups, len(tr))   # B1 contract, fail-closed (no bare str/dict/int)
            tf.fit(_work_sum_matrix(tr_groups) @ Xtr)      # work-balanced: IDF from work-level DF
            Xtr_out = tf.transform(Xtr)                    # per-chunk rows, frozen work-IDF
        return Xtr_out, tf.transform(hv.transform(list(te)))
    return f


def ch_char(tr: List[str], te: List[str], tr_groups=None):
    return _hashing_channel(HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), n_features=2**18,
        alternate_sign=False, norm=None))(tr, te, tr_groups)


def ch_word(tr: List[str], te: List[str], tr_groups=None):
    return _hashing_channel(HashingVectorizer(
        analyzer="word", ngram_range=(1, 2), n_features=2**19,
        alternate_sign=False, norm=None))(tr, te, tr_groups)


def block_channel(cfg, blocks: List[str]) -> ChannelFn:
    def f(tr, te, tr_groups=None):
        ov = {k: False for k in _ALL_BLOCKS}
        for b in blocks:
            ov[b] = True
        vec = StyloVectorizer.from_config(cfg, enabled_override=ov)
        # groups=None -> legacy pooled-chunk fit; groups -> B1 work-level feature fitting
        Xtr = vec.fit_transform(list(tr)) if tr_groups is None \
            else vec.fit_transform(list(tr), groups=tr_groups)
        Xte = vec.transform(list(te))
        mas = MaxAbsScaler().fit(Xtr)
        return mas.transform(Xtr), mas.transform(Xte)
    return f


def make_channels(cfg) -> Dict[str, ChannelFn]:
    """Канальный набор бенчмарка (без DSP): идентичен scripts/run_benchmark.py."""
    return {
        "char (2-5)": ch_char,
        "word (1-2)": ch_word,
        "syntax (dep+pos+syn)": block_channel(cfg, ["dependency", "pos_ngrams", "syntax"]),
        "dependency": block_channel(cfg, ["dependency"]),
        "function_words": block_channel(cfg, ["function_words"]),
        "morphology": block_channel(cfg, ["morphology"]),
    }
