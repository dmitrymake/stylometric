"""Символьные n-граммы с опциональным topic-bleaching.

Bleaching маскирует POS (PROPN/NOUN/NUM/ADJ) в спец-символы, чтобы char-n-граммы
ловили стиль, а не тему. NOUN намеренно != '@' (clean_text маскирует PER -> '@').

Bleached-строки строятся из ГОТОВЫХ spaCy Doc (без повторного разбора).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc

from .base import FeatureBlock


def bleach_doc(doc: Doc, replacements: Dict[str, str]) -> str:
    """Заменить токены по POS на маркеры; буквы -> lower, пунктуация как есть."""
    num_marker = replacements.get("NUM", "%")
    out: List[str] = []
    for t in doc:
        if t.pos_ == "NUM" or t.like_num:
            out.append(num_marker)
            continue
        marker = replacements.get(t.pos_)
        if marker is not None:
            out.append(marker)
        elif t.is_alpha:
            out.append(t.text.lower())
        elif t.is_punct:
            out.append(t.text)
    return " ".join(out)


class CharNgramBlock(FeatureBlock):
    group = "char_ngrams"

    def __init__(
        self,
        ngram_range=(3, 5),
        max_features: int = 5000,
        min_df: int = 3,
        sublinear_tf: bool = True,
        bleach: bool = True,
        pos_replacements: Dict[str, str] | None = None,
        name: str | None = None,
    ):
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.bleach = bleach
        self.pos_replacements = pos_replacements or {
            "PROPN": "^", "NOUN": "¤", "NUM": "%", "ADJ": "&"
        }
        self.name = name or ("char_ngrams" if bleach else "char_ngrams_raw")
        self._vec: TfidfVectorizer | None = None

    def _strings(self, texts, reps) -> List[str]:
        if self.bleach:
            return [r.bleach for r in reps]
        return [t.lower() for t in texts]

    def fit(self, texts, reps):
        self._vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            lowercase=False,           # bleach уже lower; raw приводим в _strings
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            use_idf=True,
        )
        self._vec.fit(self._strings(texts, reps))
        return self

    def transform(self, texts, reps) -> csr_matrix:
        assert self._vec is not None, "fit перед transform"
        return self._vec.transform(self._strings(texts, reps))

    def feature_names(self) -> List[str]:
        assert self._vec is not None
        return [f"char::{f}" for f in self._vec.get_feature_names_out()]
