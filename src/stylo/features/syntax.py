"""Синтаксические/стилометрические метрики, разбитые на ИМЕНОВАННЫЕ субблоки.

Субблоки (включаются по отдельности через config.features.syntax.subblocks):
  sentence         : avg_sent
  word_len         : avg_word
  pos_ratios       : noun, verb, adj, adv
  punctuation      : punct_ratio, dash_per_100, semi_per_100
  lexical_richness : ttr, hapax, yuleK
  speech           : ds_ratio (доля предложений-реплик по тире)
  vre              : vowel rhythm entropy
  ssa              : slope, intercept, r2 (вычисляется через _textstats)

Это позволяет sweep честно ответить, какой именно субблок работает.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from .base import FeatureBlock
from ._textstats import ssa_metrics, vowel_rhythm_entropy

# Какие признаки даёт каждый субблок (порядок фиксирован).
SUBBLOCK_FEATURES: Dict[str, List[str]] = {
    "sentence": ["avg_sent"],
    "word_len": ["avg_word"],
    "pos_ratios": ["noun", "verb", "adj", "adv"],
    "punctuation": ["punct_ratio", "dash_per_100", "semi_per_100"],
    "lexical_richness": ["ttr", "hapax", "yuleK"],
    "speech": ["ds_ratio"],
    "vre": ["vowel_ent"],
    "ssa": ["ssa_slope", "ssa_intercept", "ssa_r2"],
}
SUBBLOCK_ORDER = list(SUBBLOCK_FEATURES.keys())


class SyntaxBlock(FeatureBlock):
    group = "syntax"
    name = "syntax"

    def __init__(self, subblocks: Dict[str, bool] | None = None,
                 vowels_hard: str = "аоуэы", vowels_soft: str = "иеяёю"):
        # по умолчанию включены все, кроме ssa
        default = {k: (k != "ssa") for k in SUBBLOCK_ORDER}
        self.subblocks = {**default, **(subblocks or {})}
        self.vowels_hard = vowels_hard
        self.vowels_soft = vowels_soft
        self._active = [s for s in SUBBLOCK_ORDER if self.subblocks.get(s)]

    def fit(self, texts, reps, groups=None):
        return self

    def _compute_all(self, doc: Doc) -> Dict[str, List[float]]:
        """Все субблоки за один проход (фолд-независимо) — мемоизируется по doc."""
        words = [t for t in doc if t.is_alpha]
        total_w = len(words)
        sents = list(doc.sents)
        out: Dict[str, List[float]] = {}

        sent_lens = [sum(1 for t in s if t.is_alpha) for s in sents]
        out["sentence"] = [float(np.mean(sent_lens)) if sent_lens else 0.0]
        out["word_len"] = [float(np.mean([len(t.text) for t in words])) if words else 0.0]

        pc = Counter(t.pos_ for t in words)
        out["pos_ratios"] = [pc[tag] / total_w if total_w else 0.0
                             for tag in ("NOUN", "VERB", "ADJ", "ADV")]

        punct = sum(1 for t in doc if t.is_punct)
        txt = doc.text
        out["punctuation"] = [
            punct / len(doc) if len(doc) else 0.0,
            100.0 * (txt.count("—") + txt.count("-")) / (total_w or 1),
            100.0 * txt.count(";") / (total_w or 1),
        ]

        if words:
            counts = Counter(t.lemma_.lower() for t in words)
            V, N = len(counts), total_w
            M1 = sum(c * c for c in counts.values())
            out["lexical_richness"] = [
                V / N if N else 0.0,
                sum(1 for c in counts.values() if c == 1) / V if V else 0.0,
                10_000.0 * (M1 - N) / (N * N) if N > 1 else 0.0,
            ]
        else:
            out["lexical_richness"] = [0.0, 0.0, 0.0]

        if sents:
            ds = sum(1 for s in sents
                     if (st := s.text.strip()) and (st.startswith("—") or st.startswith("-")))
            out["speech"] = [ds / len(sents)]
        else:
            out["speech"] = [0.0]

        out["vre"] = [vowel_rhythm_entropy(doc, self.vowels_hard, self.vowels_soft)]
        out["ssa"] = list(ssa_metrics(doc))
        return out

    def _row_from_rep(self, rep) -> List[float]:
        allv = rep.syntax_all
        vals: List[float] = []
        for sub in self._active:
            vals.extend(allv[sub])
        return vals

    def transform(self, texts, reps) -> csr_matrix:
        rows = [self._row_from_rep(r) for r in reps]
        arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, self.n_features()), dtype=np.float32)
        return csr_matrix(arr)

    def feature_names(self) -> List[str]:
        out: List[str] = []
        for sub in self._active:
            out.extend(f"syn::{sub}::{f}" for f in SUBBLOCK_FEATURES[sub])
        return out
