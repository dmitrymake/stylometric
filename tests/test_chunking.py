"""Инварианты умной нарезки: не рвём предложения, не теряем/не дублируем текст."""
import pytest

from stylo.chunking import CombinedDoc, make_sent_chunks, split_text_safe


class FakeSpan:
    def __init__(self, text):
        self.text = text
        self._n = len(text.split())

    def __len__(self):
        return self._n


def _doc(sentences):
    return CombinedDoc([FakeSpan(s) for s in sentences])


def test_no_sentence_split():
    sents = [f"Это предложение номер {i} из нескольких слов подряд." for i in range(20)]
    chunks = make_sent_chunks(_doc(sents), size=20, min_size=5, overlap=0.0)
    # каждый чанк состоит из целых исходных предложений
    joined = " ".join(chunks)
    for s in sents:
        assert s in joined  # ни одно предложение не разорвано


def test_giant_sentence_kept():
    giant = "слово " * 100
    chunks = make_sent_chunks(_doc([giant.strip()]), size=20, min_size=5)
    assert len(chunks) == 1
    assert chunks[0].split() == giant.split()


def test_empty():
    assert make_sent_chunks(_doc([]), 20, 5) == []


@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_short_single_sentence_tail_respects_min_size(overlap):
    full = " ".join(["слово"] * 500)
    chunks = make_sent_chunks(_doc([full, "Конец."]), 500, 200, overlap)
    assert chunks == [full]


def test_single_short_work_is_not_a_complete_window():
    assert make_sent_chunks(_doc(["Очень коротко."]), 500, 200) == []


def test_giant_sentence_survives_even_if_minimum_exceeds_size():
    giant = " ".join(["слово"] * 21)
    assert make_sent_chunks(_doc([giant]), 20, 30) == [giant]


@pytest.mark.parametrize("text", ["абвгдежзийклмноп", "один два   три четыре", " " * 27,
                                  " длинноесловобезпробелов конец ", ""])
@pytest.mark.parametrize("limit", [1, 5, 10])
def test_safe_character_chunks_preserve_every_character(text, limit):
    chunks = split_text_safe(text, limit)
    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= limit for chunk in chunks)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_safe_character_chunks_reject_invalid_limits(limit):
    with pytest.raises(ValueError, match="positive integer"):
        split_text_safe("текст", limit)
