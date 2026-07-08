from __future__ import annotations

import logging
from collections import Counter
from typing import List

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from experemental.approved.semantic_alignment import calculate_alignment_metrics
from experemental.approved.vowel_rhythm import calculate_vowel_rhythm_entropy
from meta.meta import FUNCTION_WORDS, POS_REPLACEMENTS as CFG_POS_REPLACEMENTS
from scripts.nlp import get_stylometry_nlp

_GLOBAL_NLP = None


def get_global_nlp():
    """Возвращает глобальный инстанс NLP для скрипта."""
    global _GLOBAL_NLP
    if _GLOBAL_NLP is None:
        _GLOBAL_NLP = get_stylometry_nlp()
    return _GLOBAL_NLP


# В корпусе после clean_text.py имена PERSON маскируются в символ "@"
# (см. normalize(): "<PER>" -> "@"). Если в bleaching заменять NOUN тоже на "@",
# в char-ngrams сольются "персоны" и "существительные" -> паразитный сигнал.
# Поэтому для bleaching гарантируем, что NOUN != "@" (даже если в конфиге иначе).
BLEACH_POS_REPLACEMENTS = dict(CFG_POS_REPLACEMENTS)

if BLEACH_POS_REPLACEMENTS.get("NOUN") == "@":
    logging.warning(
        "Bleaching: NOUN marker совпадает с PER marker '@'. "
        "Принудительно заменяю NOUN marker на '¤' для избежания конфликта."
    )
    BLEACH_POS_REPLACEMENTS["NOUN"] = "¤"


def bleach_text(text: str) -> str:
    """
    Topic Bleaching (маскировка темы) для char-ngrams.
    Должна быть на уровне модуля, чтобы joblib/pickle могли сериализовать.
    """
    nlp = get_global_nlp()
    doc = nlp(text)

    tokens: list[str] = []
    for t in doc:
        # Числа маскируем и по POS, и по like_num (на случай несовпадений)
        if t.pos_ == "NUM" or t.like_num:
            marker = BLEACH_POS_REPLACEMENTS.get("NUM", "%")
            tokens.append(marker)
            continue

        marker = BLEACH_POS_REPLACEMENTS.get(t.pos_)
        if marker is not None:
            tokens.append(marker)
        elif t.is_alpha:
            tokens.append(t.text.lower())
        elif t.is_punct:
            tokens.append(t.text)

    return " ".join(tokens)


class SyntaxFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Извлечение синтаксических/стилометрических метрик:
    - длины предложений/слов
    - POS-распределения
    - пунктуация
    - лексическое богатство (TTR/Hapax/YuleK)
    - простая эвристика "прямой речи"
    - экспериментальные признаки: Vowel Rhythm Entropy + SSA Alignment
    """

    def fit(self, X, y=None):
        return self

    def transform(self, texts: List[str]):
        nlp = get_global_nlp()
        out: list[list[float]] = []

        for doc in nlp.pipe(texts, batch_size=20):
            sents = list(doc.sents)

            sent_lens = [sum(1 for t in s if t.is_alpha) for s in sents]
            avg_sent = float(np.mean(sent_lens)) if sent_lens else 0.0

            words = [t for t in doc if t.is_alpha]
            total_w = len(words)
            avg_word = float(np.mean([len(t.text) for t in words])) if words else 0.0

            pos_counts = Counter(t.pos_ for t in words)
            noun = pos_counts["NOUN"] / total_w if total_w else 0.0
            verb = pos_counts["VERB"] / total_w if total_w else 0.0
            adj = pos_counts["ADJ"] / total_w if total_w else 0.0
            adv = pos_counts["ADV"] / total_w if total_w else 0.0

            punct_count = sum(1 for t in doc if t.is_punct)
            punct_ratio = punct_count / len(doc) if len(doc) else 0.0
            dash_per_100 = (
                100.0 * (doc.text.count("—") + doc.text.count("-")) / (total_w or 1)
            )
            semi_per_100 = 100.0 * doc.text.count(";") / (total_w or 1)

            if words:
                lemmas = [t.lemma_.lower() for t in words]
                counts = Counter(lemmas)
                V = len(counts)
                N = total_w
                ttr = V / N if N else 0.0
                hapax = (sum(1 for c in counts.values() if c == 1) / V) if V else 0.0
                M1 = sum(c * c for c in counts.values())
                yuleK = 10_000.0 * (M1 - N) / (N**2) if N > 1 else 0.0
            else:
                ttr = 0.0
                hapax = 0.0
                yuleK = 0.0

            if sents:
                ds_count = sum(
                    1
                    for s in sents
                    if (st := s.text.strip())
                    and (st.startswith("—") or st.startswith("-"))
                )
                ds_ratio = ds_count / len(sents)
            else:
                ds_ratio = 0.0

            vowel_ent = float(calculate_vowel_rhythm_entropy(doc))
            slope, intercept, r2 = calculate_alignment_metrics(doc)

            out.append(
                [
                    avg_sent,
                    avg_word,
                    noun,
                    verb,
                    adj,
                    adv,
                    punct_ratio,
                    ttr,
                    hapax,
                    yuleK,
                    ds_ratio,
                    dash_per_100,
                    semi_per_100,
                    vowel_ent,
                    float(slope),
                    float(intercept),
                    float(r2),
                ]
            )

        return np.asarray(out, dtype=np.float32)


class StyloVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        char_ngram_range=(3, 5),
        max_char_features=5000,
        char_min_df=3,
        use_char=True,
        use_func=True,
        use_mfw=True,
        mfw_count=300,
        use_syntax=True,
        auto_bleach=True,
    ):
        self.char_ngram_range = char_ngram_range
        self.max_char_features = max_char_features
        self.char_min_df = char_min_df
        self.use_char = use_char
        self.use_func = use_func
        self.use_mfw = use_mfw
        self.mfw_count = mfw_count
        self.use_syntax = use_syntax
        self.auto_bleach = auto_bleach

    def fit(self, X: List[str], y=None):
        if self.use_char:
            preprocessor = bleach_text if self.auto_bleach else None
            if self.auto_bleach:
                logging.info("Topic Bleaching включен для char-grams (замена POS).")

            self.char_vec = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.char_ngram_range,
                lowercase=True,
                min_df=self.char_min_df,
                max_features=self.max_char_features,
                sublinear_tf=True,
                use_idf=True,
                preprocessor=preprocessor,
            )
            self.char_vec.fit(X)

        # Функциональные слова и синтаксис — на сыром тексте, без bleaching.
        if self.use_func:
            if self.use_mfw:
                self.func_vec = CountVectorizer(
                    max_features=self.mfw_count,
                    lowercase=True,
                    token_pattern=r"(?u)\b\w+\b",
                )
            else:
                self.func_vec = CountVectorizer(
                    vocabulary=sorted(FUNCTION_WORDS),
                    lowercase=True,
                    token_pattern=r"(?u)\b\w+\b",
                )
            self.func_vec.fit(X)

        if self.use_syntax:
            self.syntax_vec = SyntaxFeatureExtractor()
            self.syntax_vec.fit(X)

        return self

    def transform(self, X: List[str]):
        parts = []

        if self.use_char:
            parts.append(self.char_vec.transform(X))
        if self.use_func:
            parts.append(self.func_vec.transform(X))
        if self.use_syntax:
            parts.append(csr_matrix(self.syntax_vec.transform(X)))

        if not parts:
            raise RuntimeError("Все блоки отключены")

        return hstack(parts, format="csr")
