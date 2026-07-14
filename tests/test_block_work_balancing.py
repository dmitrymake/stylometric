"""P1 B1-c: learned blocks switch to work-level fitting when groups is given (else legacy)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.base import clone

from stylo.features.char_ngrams import CharNgramBlock
from stylo.features.dependency import DependencyBlock
from stylo.features.embeddings import EmbeddingBlock
from stylo.features.function_words import FunctionWordBlock
from stylo.features.length_dist import LengthDistBlock
from stylo.features.morphology import MorphologyBlock
from stylo.features.pos_ngrams import PosNgramBlock
from stylo.features.punctuation import PunctNgramBlock
from stylo.features.syntax import SyntaxBlock

TEXTS = ["the cat sat on the mat", "the dog ran in the park", "a bird flew over the sea", "cats and dogs play"]
GROUPS = ["a/w1", "a/w1", "b/w2", "b/w3"]
REPS = [None] * len(TEXTS)


class TestFunctionWords:
    def test_legacy_when_groups_none_is_raw_counts(self):
        blk = FunctionWordBlock(mode="mfw", mfw_count=5).fit(TEXTS, REPS, groups=None)
        assert blk._vec is not None and blk._wv is None
        M = blk.transform(TEXTS, REPS).toarray()
        assert M.dtype.kind in "iu" or np.allclose(M, M.astype(int))  # integer raw counts

    def test_work_balanced_when_groups_given_is_relative(self):
        blk = FunctionWordBlock(mode="mfw", mfw_count=5).fit(TEXTS, REPS, groups=GROUPS)
        assert blk._wv is not None and blk._vec is None
        M = blk.transform(TEXTS, REPS).toarray()
        # relative frequencies: each selected-feature row-value <= 1 and non-integer scale
        assert M.max() <= 1.0 + 1e-9
        assert np.any((M > 0) & (M < 1))

    def test_fixed_list_both_modes(self):
        legacy = FunctionWordBlock(mode="fixed_list").fit(TEXTS, REPS, groups=None)
        wb = FunctionWordBlock(mode="fixed_list").fit(TEXTS, REPS, groups=GROUPS)
        assert legacy._vec is not None and wb._wv is not None
        assert len(legacy.feature_names()) == len(wb.feature_names())  # same fixed vocab size


class TestCharNgrams:
    def test_legacy_when_groups_none(self):
        blk = CharNgramBlock(ngram_range=(2, 3), min_df=1, max_features=50, bleach=False).fit(TEXTS, REPS, groups=None)
        assert blk._vec is not None and blk._wv is None
        assert blk.transform(TEXTS, REPS).shape[0] == len(TEXTS)

    def test_work_balanced_tfidf_rows_l2_normalised(self):
        blk = CharNgramBlock(ngram_range=(2, 3), min_df=1, max_features=50, bleach=False).fit(TEXTS, REPS, groups=GROUPS)
        assert blk._wv is not None and blk._vec is None
        M = blk.transform(TEXTS, REPS).toarray()
        norms = np.sqrt((M ** 2).sum(axis=1))
        assert np.allclose(norms[norms > 0], 1.0)

    def test_feature_names_prefixed_both_ways(self):
        for groups in (None, GROUPS):
            blk = CharNgramBlock(ngram_range=(2, 3), min_df=1, bleach=False).fit(TEXTS, REPS, groups=groups)
            assert all(n.startswith("char::") for n in blk.feature_names())


class TestExactLegacyParity:
    def test_function_words_legacy_matches_raw_countvectorizer(self):
        from sklearn.feature_extraction.text import CountVectorizer
        blk = FunctionWordBlock(mode="mfw", mfw_count=6).fit(TEXTS, REPS, groups=None)
        ref = CountVectorizer(max_features=6, lowercase=True, token_pattern=r"(?u)\b\w+\b").fit(TEXTS)
        np.testing.assert_array_equal(
            blk.transform(TEXTS, REPS).toarray(), ref.transform(TEXTS).toarray())

    def test_char_legacy_matches_raw_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        blk = CharNgramBlock(ngram_range=(2, 3), min_df=1, max_features=40, bleach=False,
                             sublinear_tf=True).fit(TEXTS, REPS, groups=None)
        ref = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), lowercase=False, min_df=1,
                              max_features=40, sublinear_tf=True, use_idf=True).fit([t.lower() for t in TEXTS])
        np.testing.assert_allclose(
            blk.transform(TEXTS, REPS).toarray(), ref.transform([t.lower() for t in TEXTS]).toarray())


class TestCloneUnfitted:
    def test_clone_resets_fitted_state(self):
        for blk in (FunctionWordBlock(mode="mfw", mfw_count=5).fit(TEXTS, REPS, groups=GROUPS),
                    CharNgramBlock(ngram_range=(2, 3), min_df=1, bleach=False).fit(TEXTS, REPS, groups=None)):
            fresh = clone(blk)
            assert fresh._vec is None and fresh._wv is None      # unfitted copy
            assert (blk._vec is not None) or (blk._wv is not None)  # original still fitted


# every FeatureBlock subclass, constructed with defaults (HIGH-2 mutation isolation)
ALL_BLOCKS = {
    "char_ngrams": lambda: CharNgramBlock(),
    "function_words": lambda: FunctionWordBlock(),
    "pos_ngrams": lambda: PosNgramBlock(),
    "punctuation": lambda: PunctNgramBlock(),
    "morphology": lambda: MorphologyBlock(),
    "dependency": lambda: DependencyBlock(),
    "syntax": lambda: SyntaxBlock(),
    "length_dist": lambda: LengthDistBlock(),
    "embeddings": lambda: EmbeddingBlock(),
}


class TestCloneMutationIsolation:
    @pytest.mark.parametrize("key", list(ALL_BLOCKS))
    def test_clone_shares_no_mutable_state(self, key):
        blk = ALL_BLOCKS[key]()
        cl = clone(blk)
        assert cl is not blk
        # every mutable container captured at construction must be a DISTINCT object,
        # and mutating the clone in place must never reach back into the original.
        for attr, val in blk._ctor_state.items():
            if not isinstance(val, (list, dict, set)):
                continue
            assert getattr(cl, attr) is not getattr(blk, attr), f"{key}.{attr} shared object"
            before = copy.deepcopy(getattr(blk, attr))
            cont = getattr(cl, attr)
            if isinstance(cont, dict):
                cont["__probe__"] = "x"
            elif isinstance(cont, set):
                cont.add("__probe__")
            else:
                cont.append("__probe__")
            assert getattr(blk, attr) == before, f"{key}.{attr} leaked clone mutation"

    def test_signature_preserved_after_wrapping(self):
        import inspect
        # __init_subclass__ wraps __init__ but functools.wraps must keep the real signature
        params = inspect.signature(CharNgramBlock.__init__).parameters
        assert "ngram_range" in params and "bleach" in params and "self" in params


class _Rep:
    """Minimal Rep stand-in for the string/dict-driven blocks (B1-c)."""
    def __init__(self, pos_str="", punct_str="", morph=None, dep_counts=None, dep_agg=None):
        self.pos_str = pos_str
        self.punct_str = punct_str
        self.morph = dict(morph or {})
        self.dep_counts = dict(dep_counts or {})
        self.dep_n = sum(v for v in self.dep_counts.values() if v > 0)
        self.dep_agg = list(dep_agg) if dep_agg is not None else [0.0] * 6


# 4 chunks over 3 works; "NOUN"/"VERB"/"NOUN VERB" each span >=2 works (survive min_df_works=2)
POS = ["NOUN VERB", "NOUN ADP", "NOUN VERB", "NOUN ADJ"]
PUNCT = ["— ,", "— .", "— ,", ". ."]
WG = ["a/w1", "a/w1", "b/w2", "c/w3"]


class TestPosPunctWorkBalanced:
    def test_pos_routing_and_l2(self):
        reps = [_Rep(pos_str=s) for s in POS]
        wb = PosNgramBlock(ngram_range=(1, 2), max_features=50, min_df=1).fit(None, reps, groups=WG)
        assert wb._wv is not None and wb._vec is None
        M = wb.transform(None, reps).toarray()
        norms = np.sqrt((M ** 2).sum(axis=1))
        assert np.allclose(norms[norms > 0], 1.0)

    def test_pos_legacy_exact_parity(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        reps = [_Rep(pos_str=s) for s in POS]
        blk = PosNgramBlock(ngram_range=(1, 2), max_features=50, min_df=1).fit(None, reps, groups=None)
        assert blk._vec is not None and blk._wv is None
        ref = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50, min_df=1,
                              sublinear_tf=True, token_pattern=r"(?u)[A-Z]+", lowercase=False).fit(POS)
        np.testing.assert_allclose(blk.transform(None, reps).toarray(), ref.transform(POS).toarray())

    def test_punct_routing_and_l2(self):
        reps = [_Rep(punct_str=s) for s in PUNCT]
        wb = PunctNgramBlock(ngram_range=(1, 2), max_features=50).fit(None, reps, groups=WG)
        assert wb._wv is not None and wb._vec is None
        M = wb.transform(None, reps).toarray()
        norms = np.sqrt((M ** 2).sum(axis=1))
        assert np.allclose(norms[norms > 0], 1.0)

    def test_punct_legacy_exact_parity(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        reps = [_Rep(punct_str=s) for s in PUNCT]
        blk = PunctNgramBlock(ngram_range=(1, 2), max_features=50).fit(None, reps, groups=None)
        assert blk._vec is not None and blk._wv is None
        ref = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50, min_df=2,
                              sublinear_tf=True, token_pattern=r"\S+", lowercase=False).fit(PUNCT)
        np.testing.assert_allclose(blk.transform(None, reps).toarray(), ref.transform(PUNCT).toarray())


class TestMorphDepWorkDF:
    def _morph_reps(self):
        # Case=Nom in w1(x2)+w2 -> 2 works; Tense=Past only in w1 -> 1 work
        return [
            _Rep(morph={"Case=Nom": 3, "Tense=Past": 1}),   # w1
            _Rep(morph={"Case=Nom": 2, "Tense=Past": 1}),   # w1
            _Rep(morph={"Case=Nom": 1}),                    # w2
        ]

    def test_morphology_work_df_prunes_single_work_feature(self):
        reps = self._morph_reps()
        wb = MorphologyBlock(min_df=1).fit(None, reps, groups=["a/w1", "a/w1", "b/w2"])
        assert wb.vocab == ["Case=Nom"]                     # Tense=Past dropped: only 1 work

    def test_morphology_legacy_chunk_df_unchanged(self):
        reps = self._morph_reps()
        legacy = MorphologyBlock(min_df=1).fit(None, reps, groups=None)
        assert legacy.vocab == ["Case=Nom", "Tense=Past"]   # both survive chunk-DF>=1

    def test_dependency_work_df_prunes_single_work_feature(self):
        reps = [
            _Rep(dep_counts={"nsubj": 2, "obl": 1}),        # w1
            _Rep(dep_counts={"nsubj": 1, "obl": 1}),        # w1
            _Rep(dep_counts={"nsubj": 1}),                  # w2
        ]
        wb = DependencyBlock(min_df=1).fit(None, reps, groups=["a/w1", "a/w1", "b/w2"])
        assert wb.dep_vocab == ["nsubj"]                    # obl only in w1 -> dropped

    def test_morph_dep_reject_bad_groups(self):
        reps = self._morph_reps()
        with pytest.raises(ValueError):
            MorphologyBlock().fit(None, reps, groups={"a/w1", "b/w2", "c/w3"})  # set

    # ── Codex B1-c fail-closed regressions ────────────────────────────────────
    def test_no_public_min_df_works_knob(self):
        # HIGH: threshold is a fixed constant, not a corruptible public param
        for cls in (MorphologyBlock, DependencyBlock):
            with pytest.raises(TypeError):
                cls(min_df_works=3)

    def test_zero_count_is_absence(self):
        # ghost=0 in w1, real in w2 -> feature present in ONLY 1 work -> dropped at >=2
        reps = [
            _Rep(morph={"Case=Nom": 1, "ghost": 0}),        # w1: ghost count 0 = absent
            _Rep(morph={"Case=Nom": 1, "ghost": 0}),        # w2: ghost count 0 = absent
            _Rep(morph={"ghost": 1}),                       # w3: ghost present
        ]
        wb = MorphologyBlock().fit(None, reps, groups=["a/w1", "b/w2", "c/w3"])
        assert wb.vocab == ["Case=Nom"] and "ghost" not in wb.vocab

    @pytest.mark.parametrize("bad", [
        -1, float("nan"), float("inf"), 2.5,          # python int-negative / float / non-integral
        np.float32("nan"), np.float32("inf"),         # numpy float32 — NOT a python float subclass
        np.longdouble("nan"), np.longdouble("inf"),   # numpy longdouble — type-hole Codex flagged
        np.float64("nan"),
    ])
    def test_non_integer_or_negative_count_rejected(self, bad):
        reps = [_Rep(morph={"Case=Nom": 1}), _Rep(morph={"Case=Nom": bad})]
        with pytest.raises(ValueError):
            MorphologyBlock().fit(None, reps, groups=["a/w1", "b/w2"])

    def test_numpy_integer_count_accepted(self):
        # a legitimate numpy integer count still marks presence (Integral, not bool)
        reps = [_Rep(morph={"Case=Nom": np.int64(2)}), _Rep(morph={"Case=Nom": np.int32(1)})]
        wb = MorphologyBlock().fit(None, reps, groups=["a/w1", "b/w2"])
        assert wb.vocab == ["Case=Nom"]

    def test_morphology_empty_work_vocabulary_raises(self):
        # every feature confined to a single distinct work -> nothing survives -> raise
        reps = [_Rep(morph={"A=1": 1}), _Rep(morph={"B=1": 1})]
        with pytest.raises(ValueError, match="empty vocabulary"):
            MorphologyBlock().fit(None, reps, groups=["a/w1", "b/w2"])

    def test_dependency_empty_vocab_ok_aggregates_remain(self):
        # dep types each confined to one work -> empty dep_vocab is fine; 6 aggregates stay
        reps = [_Rep(dep_counts={"nsubj": 1}, dep_agg=[1, 2, 3, 4, 5, 6]),
                _Rep(dep_counts={"obl": 1}, dep_agg=[6, 5, 4, 3, 2, 1])]
        wb = DependencyBlock().fit(None, reps, groups=["a/w1", "b/w2"])
        assert wb.dep_vocab == []
        assert wb.transform(None, reps).shape == (2, 6)     # only the aggregates

    def test_morph_legacy_transform_exact(self):
        reps = self._morph_reps()
        blk = MorphologyBlock(min_df=1).fit(None, reps, groups=None)
        M = blk.transform(None, reps).toarray()
        cols = {n.split("::")[1]: i for i, n in enumerate(blk.feature_names())}
        # chunk0: Case=Nom 3, Tense=Past 1, total 4 -> 0.75 / 0.25
        assert M[0, cols["Case=Nom"]] == pytest.approx(0.75)
        assert M[0, cols["Tense=Past"]] == pytest.approx(0.25)
        assert M[2, cols["Case=Nom"]] == pytest.approx(1.0)  # chunk2: only Case=Nom

    def test_dep_aggregates_invariant_across_paths(self):
        reps = [_Rep(dep_counts={"nsubj": 2, "obl": 1}, dep_agg=[1, 2, 3, 4, 5, 6]),
                _Rep(dep_counts={"nsubj": 1, "obl": 1}, dep_agg=[7, 8, 9, 10, 11, 12]),
                _Rep(dep_counts={"nsubj": 1, "obl": 1}, dep_agg=[0, 0, 0, 0, 0, 0])]
        g = ["a/w1", "a/w1", "b/w2"]
        legacy = DependencyBlock(min_df=1).fit(None, reps, groups=None).transform(None, reps).toarray()
        work = DependencyBlock(min_df=1).fit(None, reps, groups=g).transform(None, reps).toarray()
        # last 6 columns are the tree aggregates in both paths
        np.testing.assert_allclose(legacy[:, -6:], work[:, -6:])


class TestBFourStateSwitch:
    @pytest.mark.parametrize("make,gk", [
        (lambda: PosNgramBlock(ngram_range=(1, 2), min_df=1), "pos"),
        (lambda: PunctNgramBlock(ngram_range=(1, 2)), "punct"),
        (lambda: MorphologyBlock(min_df=1), "morph"),
        (lambda: DependencyBlock(min_df=1), "dep"),
    ])
    def test_fitted_clone_is_unfitted_and_roundtrips(self, make, gk):
        reps_map = {
            "pos": [_Rep(pos_str=s) for s in POS],
            "punct": [_Rep(punct_str=s) for s in PUNCT],
            "morph": [_Rep(morph={"Case=Nom": 2, "Tense=Past": 1}) for _ in range(4)],
            "dep": [_Rep(dep_counts={"nsubj": 2, "obl": 1}) for _ in range(4)],
        }
        reps = reps_map[gk]
        blk = make()
        # legacy -> work -> legacy round-trip stays consistent
        n_leg1 = len(blk.fit(None, reps, groups=None).feature_names())
        n_work = len(blk.fit(None, reps, groups=WG).feature_names())
        n_leg2 = len(blk.fit(None, reps, groups=None).feature_names())
        assert n_leg1 == n_leg2 and n_work >= 0
        # clone of a fitted block is unfitted
        fitted = make().fit(None, reps, groups=WG)
        fresh = clone(fitted)
        if hasattr(fresh, "_wv"):
            assert fresh._wv is None and fresh._vec is None
        else:  # morph/dep: fitted vocab must reset to the empty constructor list
            vocab_attr = "vocab" if gk == "morph" else "dep_vocab"
            assert getattr(fresh, vocab_attr) == []


class TestGroupsFailClosed:
    def test_vectorizer_rejects_misaligned_groups(self):
        from stylo.vectorizer import StyloVectorizer

        class _NoRepCache:
            def get_reps(self, X):  # never reached: the length check fails first
                raise AssertionError("should not be called")

        sv = StyloVectorizer(blocks=[], rep_cache=_NoRepCache())
        with pytest.raises(ValueError):
            sv.fit(["a", "b", "c"], groups=["w1", "w2"])

    def test_empty_work_balanced_train_rejected_before_reps(self):
        from stylo.vectorizer import StyloVectorizer

        class _NoRepCache:
            def get_reps(self, X):
                raise AssertionError("should not be called")

        # empty groups must be rejected by the central validator, even with no/stateless blocks
        sv = StyloVectorizer(blocks=[], rep_cache=_NoRepCache())
        with pytest.raises(ValueError, match="empty groups"):
            sv.fit([], groups=[])
