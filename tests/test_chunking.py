"""Инварианты умной нарезки: не рвём предложения, не теряем/не дублируем текст."""
from stylo.chunking import CombinedDoc, make_sent_chunks


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
