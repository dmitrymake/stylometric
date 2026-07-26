"""Content-matched alignment of multiple source/edition realisations.

Authorship invariance cannot be identified by comparing unrelated works from
different sources.  This module extracts exact *normalised-word* matching
blocks from two versions of the same work while retaining their original text
surfaces.  Front matter, commentary, and substantive rewrites remain outside
the matched blocks.
"""
from __future__ import annotations

import dataclasses
import difflib
import re
from typing import Sequence


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_FINAL_HARD_SIGN_RE = re.compile(r"(?<=[а-яёѣіѳѵ])ъ$", flags=re.IGNORECASE)
_HISTORICAL_TRANSLATION = str.maketrans(
    {
        "ё": "е",
        "Ё": "е",
        "ѣ": "е",
        "Ѣ": "е",
        "і": "и",
        "І": "и",
        "ѳ": "ф",
        "Ѳ": "ф",
        "ѵ": "и",
        "Ѵ": "и",
    }
)


@dataclasses.dataclass(frozen=True)
class WordToken:
    surface: str
    normalised: str
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class AlignedBlock:
    a_start_token: int
    a_end_token: int
    b_start_token: int
    b_end_token: int
    a_start_char: int
    a_end_char: int
    b_start_char: int
    b_end_char: int
    n_words: int


@dataclasses.dataclass(frozen=True)
class AlignmentReport:
    n_words_a: int
    n_words_b: int
    matched_words: int
    coverage_a: float
    coverage_b: float
    symmetric_coverage: float
    blocks: tuple[AlignedBlock, ...]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MultiAlignedBlock:
    """One exact normalised-word interval shared by every supplied version."""

    token_ranges: tuple[tuple[int, int], ...]
    n_words: int


def normalise_historical_word(word: str) -> str:
    value = word.translate(_HISTORICAL_TRANSLATION).casefold()
    value = _FINAL_HARD_SIGN_RE.sub("", value)
    return value


def word_tokens(text: str) -> tuple[WordToken, ...]:
    if type(text) is not str:
        raise TypeError("text must be str")
    return tuple(
        WordToken(
            surface=match.group(0),
            normalised=normalise_historical_word(match.group(0)),
            start=match.start(),
            end=match.end(),
        )
        for match in _WORD_RE.finditer(text)
    )


def align_editions(
    text_a: str,
    text_b: str,
    *,
    min_block_words: int = 80,
    autojunk: bool = True,
) -> AlignmentReport:
    """Align long exact runs after conservative historical normalisation.

    ``SequenceMatcher`` is used only to discover common content.  It receives
    no author labels and does not modify the surface text used by downstream
    stylometry.  Each returned block is one-to-one and non-overlapping.
    """
    if isinstance(min_block_words, bool) or not isinstance(min_block_words, int):
        raise TypeError("min_block_words must be an integer")
    if min_block_words < 1:
        raise ValueError("min_block_words must be positive")
    tokens_a = word_tokens(text_a)
    tokens_b = word_tokens(text_b)
    normalised_a = [token.normalised for token in tokens_a]
    normalised_b = [token.normalised for token in tokens_b]
    matcher = difflib.SequenceMatcher(
        None, normalised_a, normalised_b, autojunk=autojunk
    )
    blocks = []
    for match in matcher.get_matching_blocks():
        if match.size < min_block_words:
            continue
        a_end = match.a + match.size
        b_end = match.b + match.size
        blocks.append(
            AlignedBlock(
                a_start_token=match.a,
                a_end_token=a_end,
                b_start_token=match.b,
                b_end_token=b_end,
                a_start_char=tokens_a[match.a].start,
                a_end_char=tokens_a[a_end - 1].end,
                b_start_char=tokens_b[match.b].start,
                b_end_char=tokens_b[b_end - 1].end,
                n_words=match.size,
            )
        )
    matched = sum(block.n_words for block in blocks)
    coverage_a = matched / len(tokens_a) if tokens_a else 0.0
    coverage_b = matched / len(tokens_b) if tokens_b else 0.0
    denominator = min(len(tokens_a), len(tokens_b))
    symmetric = matched / denominator if denominator else 0.0
    return AlignmentReport(
        n_words_a=len(tokens_a),
        n_words_b=len(tokens_b),
        matched_words=matched,
        coverage_a=float(coverage_a),
        coverage_b=float(coverage_b),
        symmetric_coverage=float(symmetric),
        blocks=tuple(blocks),
    )


def extract_block_texts(
    text_a: str,
    text_b: str,
    blocks: Sequence[AlignedBlock],
) -> tuple[tuple[str, str], ...]:
    """Return original-surface text pairs for already validated blocks."""
    pairs = []
    previous_a = previous_b = 0
    for index, block in enumerate(blocks):
        if block.n_words <= 0:
            raise ValueError(f"block {index} has no words")
        if block.a_start_char < previous_a or block.b_start_char < previous_b:
            raise ValueError("blocks must be ordered and non-overlapping")
        if block.a_end_char > len(text_a) or block.b_end_char > len(text_b):
            raise ValueError("block lies outside source text")
        pairs.append(
            (
                text_a[block.a_start_char:block.a_end_char],
                text_b[block.b_start_char:block.b_end_char],
            )
        )
        previous_a = block.a_end_char
        previous_b = block.b_end_char
    return tuple(pairs)


def intersect_reference_alignments(
    reports: Sequence[AlignmentReport],
    *,
    min_block_words: int = 80,
) -> tuple[MultiAlignedBlock, ...]:
    """Intersect pairwise reference→variant alignments across all variants.

    Every report must use the same text as side A.  The resulting token ranges
    are ordered ``(reference, variant_1, ..., variant_n)`` and represent exactly
    the same normalised word sequence in every version.
    """
    if not reports:
        raise ValueError("at least one pairwise alignment report is required")
    if min_block_words < 1:
        raise ValueError("min_block_words must be positive")
    reference_size = reports[0].n_words_a
    if any(report.n_words_a != reference_size for report in reports):
        raise ValueError("all reports must share the same reference tokenisation")

    # State: reference interval plus its mapped interval in each processed variant.
    state: list[tuple[int, int, list[tuple[int, int]]]] = [
        (
            block.a_start_token,
            block.a_end_token,
            [(block.b_start_token, block.b_end_token)],
        )
        for block in reports[0].blocks
    ]
    for report in reports[1:]:
        updated: list[tuple[int, int, list[tuple[int, int]]]] = []
        for ref_start, ref_end, mapped in state:
            for block in report.blocks:
                start = max(ref_start, block.a_start_token)
                end = min(ref_end, block.a_end_token)
                if end <= start:
                    continue
                left_trim = start - ref_start
                right_trim = ref_end - end
                adjusted_existing = [
                    (variant_start + left_trim, variant_end - right_trim)
                    for variant_start, variant_end in mapped
                ]
                new_start = block.b_start_token + (start - block.a_start_token)
                new_end = new_start + (end - start)
                updated.append(
                    (start, end, adjusted_existing + [(new_start, new_end)])
                )
        state = updated

    result = []
    for ref_start, ref_end, mapped in sorted(state):
        size = ref_end - ref_start
        if size < min_block_words:
            continue
        result.append(
            MultiAlignedBlock(
                token_ranges=((ref_start, ref_end), *mapped),
                n_words=size,
            )
        )
    return tuple(result)


def extract_multi_block_texts(
    texts: Sequence[str],
    blocks: Sequence[MultiAlignedBlock],
) -> tuple[tuple[str, ...], ...]:
    """Extract original-surface strings for shared multi-version blocks."""
    if len(texts) < 2:
        raise ValueError("at least two texts are required")
    tokenised = [word_tokens(text) for text in texts]
    rows = []
    for block_index, block in enumerate(blocks):
        if len(block.token_ranges) != len(texts):
            raise ValueError(
                f"block {block_index} has {len(block.token_ranges)} ranges for {len(texts)} texts"
            )
        surfaces = []
        for text_index, ((start, end), tokens, text) in enumerate(
            zip(block.token_ranges, tokenised, texts)
        ):
            if start < 0 or end <= start or end > len(tokens):
                raise ValueError(
                    f"block {block_index} has invalid range for text {text_index}"
                )
            surfaces.append(text[tokens[start].start:tokens[end - 1].end])
        rows.append(tuple(surfaces))
    return tuple(rows)


__all__ = [
    "AlignedBlock",
    "AlignmentReport",
    "MultiAlignedBlock",
    "WordToken",
    "align_editions",
    "extract_block_texts",
    "extract_multi_block_texts",
    "intersect_reference_alignments",
    "normalise_historical_word",
    "word_tokens",
]
