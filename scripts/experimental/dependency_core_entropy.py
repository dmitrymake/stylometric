"""Experimental subject-verb-object dependency-pattern entropy features."""

import numpy as np
import math
from collections import Counter
from typing import List, Tuple
import spacy


def extract_core_triplets(doc: spacy.tokens.Doc) -> Tuple[float, float]:
    """
    Извлекает ядро предложения (триплеты Субъект-Глагол-Объект) и 
    рассчитывает Энтропию их распределения (CDT Entropy).
    """

    # 1. Сбор Core Triplet Patterns
    # Фиксированный порядок: (Субъект, Глагол, Объект)
    core_triplets = []

    CORE_POS = {"NOUN", "PRON"}

    SUBJ_DEPS = {"nsubj", "nsubjpass"}
    OBJ_DEPS = {"dobj", "iobj", "obj"}  # Прямое, косвенное, общее дополнение

    for sent in doc.sents:
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break

        if root and root.pos_ == "VERB":
            verb_lemma = root.lemma_.lower()

            subject_lemma = "NO_SUBJ"
            object_lemma = "NO_OBJ"

            for child in root.children:
                if child.dep_ in SUBJ_DEPS and child.pos_ in CORE_POS:
                    subject_lemma = child.lemma_.lower()
                elif child.dep_ in OBJ_DEPS and child.pos_ in CORE_POS:
                    object_lemma = child.lemma_.lower()

            if subject_lemma != "NO_SUBJ":
                triplet = (subject_lemma, verb_lemma, object_lemma)
                core_triplets.append(triplet)

    n_triplets = len(core_triplets)
    n_sentences = len(list(doc.sents))

    if n_triplets == 0:
        return 0.0, 0.0

    # 2. Расчет CDT Entropy
    freq = Counter(core_triplets)
    h = 0.0
    for count in freq.values():
        p = count / n_triplets
        h -= p * math.log2(p)

    # 3. Расчет CDT Proportion (плотность паттернов)
    # Насколько много предложений в чанке содержат ядро Subj-Verb-Obj
    cdt_proportion = n_triplets / n_sentences if n_sentences > 0 else 0.0

    return h, cdt_proportion
