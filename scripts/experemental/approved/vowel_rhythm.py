import numpy as np
import math
from collections import Counter
import spacy
from meta.meta import HARD_VOWELS, SOFT_VOWELS

ALL_VOWELS = HARD_VOWELS + SOFT_VOWELS

def get_vowel_type(char: str) -> str:
    char = char.lower()
    if char in HARD_VOWELS:
        return 'H'  # Hard/Open
    if char in SOFT_VOWELS:
        return 'S'  # Soft/Closed
    return ''

def calculate_vowel_rhythm_entropy(doc: spacy.tokens.Doc) -> float:
    """
    Calculates the Shannon Entropy of Vowel Rhythm Bigrams (H, S).
    Language-agnostic (via config).
    """
    vowel_sequence = []

    for token in doc:
        if token.is_alpha:
            for char in token.text:
                v_type = get_vowel_type(char)
                if v_type:
                    vowel_sequence.append(v_type)

    if len(vowel_sequence) < 20:
        return 0.0

    rhythm_bigrams = []
    for i in range(len(vowel_sequence) - 1):
        bigram = vowel_sequence[i] + vowel_sequence[i+1]
        rhythm_bigrams.append(bigram)

    if not rhythm_bigrams:
        return 0.0

    total = len(rhythm_bigrams)
    freq = Counter(rhythm_bigrams)
    h = 0.0
    for count in freq.values():
        p = count / total
        h -= p * math.log2(p)

    return h
