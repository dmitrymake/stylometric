"""Умная нарезка текста по предложениям (не рвёт предложения)."""
from __future__ import annotations

from typing import List, Sequence


class CombinedDoc:
    """Обёртка со свойством .sents — позволяет склеить предложения из нескольких
    обработанных кусков большого текста и передать в make_sent_chunks."""

    def __init__(self, sentences: Sequence):
        self.sents = list(sentences)

    def __len__(self) -> int:
        return sum(len(s) for s in self.sents)


def make_sent_chunks(doc, size: int, min_size: int, overlap: float = 0.0) -> List[str]:
    """Нарезать doc на чанки ~size токенов, не разрывая предложения.

    Гигантское предложение (> size) берётся целиком (иначе потеря текста/зацикливание).
    overlap (0..0.9) — доля перекрытия окна.
    """
    sentences = list(doc.sents)
    if not sentences:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(sentences):
        cur: List[str] = []
        cur_len = 0
        idx = start
        while idx < len(sentences):
            sent = sentences[idx]
            slen = len(sent)
            if cur_len + slen > size:
                if cur_len == 0:           # одно гигантское предложение
                    cur.append(sent.text)
                    cur_len += slen
                    idx += 1
                break
            cur.append(sent.text)
            cur_len += slen
            idx += 1

        if cur_len >= min_size or (len(cur) == 1 and cur_len > 0):
            chunks.append(" ".join(cur))

        # Сдвиг окна. consumed — сколько предложений реально вошло в чанк.
        # При overlap=0 следующее окно начинается ровно там, где кончился чанк
        # (idx), без перешагивания/потери предложений. При overlap>0 — откат назад.
        consumed = max(1, idx - start)
        advance = max(1, round(consumed * (1.0 - overlap)))
        start += advance

    return chunks


def split_text_safe(text: str, limit: int = 1_000_000) -> List[str]:
    """Разбить огромный текст на куски <= limit символов по границам слов (для spaCy)."""
    parts: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + limit, n)
        if end < n:
            sp = text.rfind(" ", start, end)
            if sp != -1:
                end = sp
        piece = text[start:end]
        if piece.strip():
            parts.append(piece)
        start = end + 1
    return parts


def sentences_for_text(text: str, nlp) -> list:
    """Список предложений (Span) для текста любого размера."""
    if len(text) < 1_000_000:
        return list(nlp(text).sents)
    out = []
    for d in nlp.pipe(split_text_safe(text)):
        out.extend(list(d.sents))
    return out
