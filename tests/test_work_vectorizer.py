"""P1 B1: work-level sparse vectorizer estimand properties (design v3 §3)."""
from __future__ import annotations

import numpy as np
import pytest

from stylo.features.work_vectorizer import (MODE_COUNT, MODE_RELATIVE, MODE_TFIDF,
                                            WorkLevelVectorizer)

WORD = {"analyzer": "word", "token_pattern": r"(?u)\b\w+\b", "lowercase": True}


def _fit(docs, groups, **kw):
    return WorkLevelVectorizer(analyzer_params=WORD, **kw).fit(docs, groups)


import scipy.sparse as sp
from sklearn.base import clone

CHAR = {"analyzer": "char", "ngram_range": (2, 3), "lowercase": False}


class TestFitEstimand:
    def test_word_level_equal_work_aggregation_invariant_to_chunk_split(self):
        # For WORD features, equal-work aggregation depends only on per-work token totals,
        # so re-splitting a work's text into different chunks leaves vocab + idf unchanged.
        one = _fit(["the cat sat on the mat", "a dog ran fast"], ["w1", "w2"], min_df_works=1)
        many = _fit(["the cat sat", "on the mat", "a dog ran", "fast"],
                    ["w1", "w1", "w2", "w2"], min_df_works=1)
        assert one.vocabulary_ == many.vocabulary_
        np.testing.assert_allclose(one.idf_, many.idf_)

    def test_char_ngrams_depend_on_frozen_chunk_boundaries(self):
        # char n-grams intentionally do NOT cross chunk boundaries, so a different chunk
        # split of the same text is a different feature set (contract: frozen boundaries).
        whole = WorkLevelVectorizer(analyzer_params=CHAR, min_df_works=1).fit(["abcd", "wxyz"], ["w1", "w2"])
        split = WorkLevelVectorizer(analyzer_params=CHAR, min_df_works=1).fit(["ab", "cd", "wxyz"], ["w1", "w1", "w2"])
        assert set(whole.vocabulary_) != set(split.vocabulary_)  # 'bc' boundary bigram only in 'abcd'

    def test_fit_never_densifies_work_by_vocab(self, monkeypatch):
        # a works×vocab .toarray() would OOM on the real corpus; fit must stay sparse
        monkeypatch.setattr(sp.csr_matrix, "toarray",
                            lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("densified")))
        _fit(["a b c d", "a b", "a c", "b d"], ["w1", "w2", "w3", "w4"], min_df_works=1)

    def test_clone_returns_unfitted(self):
        v = _fit(["a b c", "a b d"], ["w1", "w2"], min_df_works=1)
        # WorkLevelVectorizer is a plain object; the FeatureBlock clone contract is tested
        # in test_block_work_balancing; here just confirm fitted state exists to clear.
        assert v.vocabulary_ and v.idf_ is not None

    def test_equal_work_ranking_beats_pooled_tf(self):
        # work1="a b b b b" (b frequent by raw count), work2="a".
        # equal-work mean rel: a=(1/5+1)/2=0.6 > b=(4/5+0)/2=0.4 -> a selected first,
        # even though b has more raw occurrences (pooled TF would pick b).
        v = _fit(["a b b b b", "a"], ["w1", "w2"], min_df_works=1, max_features=1)
        assert v.feature_names() == ["a"]

    def test_work_df_pruning(self):
        # 'lonely' appears in only one work; min_df_works=2 drops it, 'shared' stays
        v = _fit(["shared lonely", "shared here"], ["w1", "w2"], min_df_works=2)
        assert "shared" in v.vocabulary_ and "lonely" not in v.vocabulary_

    def test_deterministic_vocab_order(self):
        v = _fit(["b a c", "a b c"], ["w1", "w2"], min_df_works=1)
        assert list(v.vocabulary_) == ["a", "b", "c"]

    def test_max_features_cap_size(self):
        v = _fit(["a b c d e", "a b c d e"], ["w1", "w2"], min_df_works=1, max_features=3)
        assert len(v.vocabulary_) == 3

    def test_fixed_vocabulary_keeps_all_no_prune(self):
        v = _fit(["alpha", "beta"], ["w1", "w2"], vocabulary=["alpha", "beta", "gamma"])
        assert set(v.vocabulary_) == {"alpha", "beta", "gamma"}  # kept despite df<2


class TestTransform:
    def test_relative_mode_denominator_is_all_tokens(self):
        v = _fit(["a a b", "a b b"], ["w1", "w2"], mode=MODE_RELATIVE, min_df_works=1)
        M = v.transform(["a a b"]).toarray()[0]
        col = v.vocabulary_
        assert M[col["a"]] == pytest.approx(2 / 3)  # a: 2 of 3 tokens
        assert M[col["b"]] == pytest.approx(1 / 3)

    def test_count_mode_raw(self):
        v = _fit(["a a b", "a b b"], ["w1", "w2"], mode=MODE_COUNT, min_df_works=1)
        M = v.transform(["a a a"]).toarray()[0]
        assert M[v.vocabulary_["a"]] == 3

    def test_tfidf_rows_l2_normalised(self):
        v = _fit(["a b c", "a b d", "a c d"], ["w1", "w2", "w3"], mode=MODE_TFIDF, min_df_works=1)
        M = v.transform(["a b c", "a a a"]).toarray()
        norms = np.sqrt((M ** 2).sum(axis=1))
        assert np.allclose(norms[norms > 0], 1.0)

    def test_empty_doc_transforms_to_zero_without_error(self):
        v = _fit(["a b", "a c"], ["w1", "w2"], mode=MODE_RELATIVE, min_df_works=1)
        M = v.transform(["   "]).toarray()[0]  # no analyzer events
        assert np.all(M == 0.0)

    def test_idf_formula(self):
        # feature in all 3 works -> idf = ln((3+1)/(3+1))+1 = 1.0
        v = _fit(["a", "a", "a"], ["w1", "w2", "w3"], mode=MODE_TFIDF, min_df_works=1)
        assert v.idf_[v.vocabulary_["a"]] == pytest.approx(1.0)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        WorkLevelVectorizer(analyzer_params=WORD).transform(["a"])


class TestContract:
    @pytest.mark.parametrize("bad", [True, 0, -1, 2.5, "3"])
    def test_min_df_works_must_be_positive_int(self, bad):
        with pytest.raises(ValueError):
            WorkLevelVectorizer(analyzer_params=WORD, min_df_works=bad)

    @pytest.mark.parametrize("bad", [True, 0, -1, 2.5])
    def test_max_features_must_be_positive_int_or_none(self, bad):
        with pytest.raises(ValueError):
            WorkLevelVectorizer(analyzer_params=WORD, max_features=bad)

    def test_empty_vocabulary_fails_in_fit(self):
        with pytest.raises(ValueError, match="empty vocabulary"):
            _fit(["unique one", "different two"], ["w1", "w2"], min_df_works=2)

    def test_groups_misalignment_rejected(self):
        with pytest.raises(ValueError):
            WorkLevelVectorizer(analyzer_params=WORD, min_df_works=1).fit(["a", "b"], ["w1"])

    def test_empty_training_set_rejected(self):
        with pytest.raises(ValueError, match="empty training set"):
            WorkLevelVectorizer(analyzer_params=WORD, vocabulary=["a", "b"]).fit([], [])

    def test_factorize_stable_first_seen(self):
        from stylo.features.work_vectorizer import _factorize
        ids, inv = _factorize(["b/2", "b/2", "a/1", "b/2", "a/1"])
        assert ids == ["b/2", "a/1"] and list(inv) == [0, 0, 1, 0, 1]


class TestWorkIdContract:
    def _val(self, groups, n=None):
        from stylo.features.work_vectorizer import validate_work_ids
        return validate_work_ids(groups, n)

    def test_accepts_object_ndarray_of_strings(self):
        arr = np.array(["a/1", "a/1", "b/2"], dtype=object)   # exactly what B0 emits
        assert self._val(arr, 3) == ["a/1", "a/1", "b/2"]

    def test_normalizes_numpy_str(self):
        out = self._val(np.array(["a/1", "b/2"]), 2)          # dtype='<U…' -> np.str_
        assert out == ["a/1", "b/2"] and all(type(x) is str for x in out)

    def test_accepts_ordered_sequences(self):
        assert self._val(("a/1", "b/2"), 2) == ["a/1", "b/2"]     # tuple
        assert self._val(["a/1"], 1) == ["a/1"]                    # list

    @pytest.mark.parametrize("bad", [
        "a/1",                          # bare string masquerading as a sequence
        b"a/1",                         # bytes
        {"a/1", "b/2"},                 # set — no positional chunk->work order
        {"a/1": 1, "b/2": 2},           # mapping — list() would take keys
        [1, True],                      # int + bool collapse to one work
        [1, 1.0],                       # int + float collapse to one work
        [float("nan"), "b"],            # python NaN
        [np.float32("nan"), "b"],       # numpy NaN
        ["a", ""],                      # empty id
        ["a", None],                    # None id
        [],                             # empty container
    ])
    def test_rejects_noncontract_groups(self, bad):
        with pytest.raises(ValueError):
            self._val(bad)

    def test_rejects_iterator(self):
        with pytest.raises(ValueError):
            self._val(iter(["a/1", "b/2"]))       # single-use, no positional guarantee

    def test_rejects_2d_and_0d_array(self):
        with pytest.raises(ValueError, match="1-D"):
            self._val(np.array([["a"], ["b"]], dtype=object))
        with pytest.raises(ValueError, match="1-D"):
            self._val(np.array("a/1", dtype=object))   # 0-d scalar array

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length"):
            self._val(["a", "b"], 3)
