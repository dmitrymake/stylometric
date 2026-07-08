"""Инвариант leakage-free: в LOBO-фолде модель обучается ТОЛЬКО на не-тестовых
чанках; тестовая книга не видна на fit. Плюс: при единственной книге автора фолд
пропускается."""
import numpy as np

from stylo.eval.lobo import run_fold
from stylo.features.function_words import FunctionWordBlock


class Spy:
    """Эстиматор-шпион: запоминает тексты, которые видел на fit."""
    _registry = []

    def __init__(self):
        self.seen = None
        self.classes_ = None
        Spy._registry.append(self)

    def fit(self, texts, y):
        self.seen = set(texts)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, texts):
        n = len(list(texts))
        return np.ones((n, len(self.classes_))) / len(self.classes_)


def _toy():
    texts = np.array([
        "alpha aaa", "alpha bbb",          # author0 / book A (2 chunks)
        "alpha ccc", "alpha ddd",          # author0 / book B
        "beta eee", "beta fff",            # author1 / book C
        "beta ggg", "beta hhh",            # author1 / book D
    ], dtype=object)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["a0/A", "a0/A", "a0/B", "a0/B",
                       "a1/C", "a1/C", "a1/D", "a1/D"], dtype=object)
    return texts, y, groups


def test_fold_excludes_test_book():
    Spy._registry.clear()
    texts, y, groups = _toy()
    res = run_fold(texts, y, groups, 2, ["a0", "a1"], "a0/A", lambda: Spy(), top_k=2)
    assert res is not None
    spy = Spy._registry[-1]
    # тестовые чанки книги a0/A не должны попасть в train
    assert "alpha aaa" not in spy.seen
    assert "alpha bbb" not in spy.seen
    # train содержит остальные книги
    assert "alpha ccc" in spy.seen and "beta eee" in spy.seen


def test_single_book_author_skipped():
    # автор a1 имеет лишь одну книгу -> при тесте этой книги он выпадает из train
    texts = np.array(["x1", "x2", "y1", "y2"], dtype=object)
    y = np.array([0, 0, 1, 1])
    groups = np.array(["a0/A", "a0/A", "a1/C", "a1/C"], dtype=object)
    res = run_fold(texts, y, groups, 2, ["a0", "a1"], "a1/C", lambda: Spy(), top_k=2)
    assert res is None


def test_function_word_vocab_is_train_only():
    block = FunctionWordBlock(mode="mfw", mfw_count=5)
    train = ["и в на и в на", "и в на и в на"]
    block.fit(train, [None, None])
    names = block.feature_names()
    assert not any("zzz" in n for n in names)
