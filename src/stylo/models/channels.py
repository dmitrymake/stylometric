"""Канальные фабрики признаков — единый источник для GKF-бенчмарка и LOBO-стекинга.

Каждый канал — функция (train_texts, test_texts) -> (Xtr, Xte); всё, что
обучается (tf-idf статистики, скейлеры, словари блоков), учится ТОЛЬКО на train.
Определения подняты из scripts/run_benchmark.py без изменения поведения; DSP-канал
остаётся локальным в бенчмарке (тяжёлый spaCy-lg кэш, в стек не входит).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler

from ..vectorizer import StyloVectorizer

ChannelFn = Callable[[List[str], List[str]], Tuple[object, object]]

_ALL_BLOCKS = ["char_ngrams", "function_words", "syntax", "pos_ngrams",
               "punctuation_ngrams", "dependency", "morphology", "length_dist",
               "embeddings"]


def ch_char(tr: List[str], te: List[str]):
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 5), n_features=2**18,
                           alternate_sign=False, norm=None)
    tf = TfidfTransformer(sublinear_tf=True)
    return tf.fit_transform(hv.transform(tr)), tf.transform(hv.transform(te))


def ch_word(tr: List[str], te: List[str]):
    hv = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2**19,
                           alternate_sign=False, norm=None)
    tf = TfidfTransformer(sublinear_tf=True)
    return tf.fit_transform(hv.transform(tr)), tf.transform(hv.transform(te))


def block_channel(cfg, blocks: List[str]) -> ChannelFn:
    def f(tr, te):
        ov = {k: False for k in _ALL_BLOCKS}
        for b in blocks:
            ov[b] = True
        vec = StyloVectorizer.from_config(cfg, enabled_override=ov)
        Xtr = vec.fit_transform(list(tr))
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
