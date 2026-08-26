"""Work-balanced feature and loss routing for ``stylo_stack``.

These focused tests exercise the stack directly. Channel tests are spaCy-free (hashing channels);
the loss tests monkeypatch ``make_channels`` to the same two hashing channels so the stack fits
without spaCy. Group-aware calibration and end-to-end dispatch have their own integration tests.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from stylo.config import load_config
from stylo.domain.work_weighting import CHUNK_WEIGHTED_LEGACY, WORK_BALANCED
from stylo.models import stacked_clf as sc
from stylo.models.channels import ch_char

CFG = load_config()


def _toy():
    # 2 authors x 6 books x 2 chunks; distinct per-author vocab so the hashing channels separate.
    texts, y, g = [], [], []
    for a, w in ((0, "alpha kappa lambda"), (1, "beta gamma delta")):
        for b in range(6):
            for c in range(2):
                texts.append(f"{w} book{b} chunk{c} {'z' * (a + 1)}")
                y.append(a)
                g.append(f"auth{a}/book{b}")
    return texts, np.array(y), np.array(g, dtype=object)


# ── feature side: ChannelFn work groups ────────────────────────────────────────
class TestChannelsWorkBalanced:
    def test_hashing_channel_groups_none_is_legacy_byte_identical(self):
        tr, te = ["a b c", "b c d", "c d e"], ["a c e"]
        x2, _ = ch_char(tr, te)             # legacy 2-arg call
        x3, _ = ch_char(tr, te, None)       # explicit groups=None
        assert np.allclose(x2.toarray(), x3.toarray())

    def test_hashing_channel_work_idf_differs_from_chunk_idf(self):
        # work A = 3 identical chunks, work B = 1 -> chunk-DF (3/4) != work-DF (1/2) -> IDF differs
        tr = ["alpha beta", "alpha beta", "alpha beta", "gamma delta"]
        g = ["A", "A", "A", "B"]
        xc, _ = ch_char(tr, ["alpha"], None)     # chunk-level IDF
        xw, _ = ch_char(tr, ["alpha"], g)        # work-level IDF
        assert not np.allclose(xc.toarray(), xw.toarray())


# ── loss side: sample weights + class_weight ───────────────────────────────────
@pytest.fixture
def _hashing_only(monkeypatch):
    from stylo.models.channels import ch_char as cc, ch_word as cw
    monkeypatch.setattr(sc, "make_channels", lambda cfg: {"char": cc, "word": cw})


class TestStackLossWiring:
    def test_legacy_default_matches_explicit_legacy(self, _hashing_only):
        t, y, g = _toy()
        a = sc.StackedChannelClassifier(CFG, inner_folds=3, seed=1).fit(t, y, groups=g)
        b = sc.StackedChannelClassifier(CFG, inner_folds=3, seed=1,
                                        training_weighting=CHUNK_WEIGHTED_LEGACY).fit(t, y, groups=g)
        np.testing.assert_allclose(a.predict_proba(t), b.predict_proba(t))
        assert a._class_weight() == "balanced" and a._fold_weights(y, g) is None

    def test_work_balanced_fits_predicts_and_weights_sum_to_W(self, _hashing_only):
        t, y, g = _toy()
        m = sc.StackedChannelClassifier(CFG, inner_folds=3, seed=1,
                                        training_weighting=WORK_BALANCED).fit(t, y, groups=g)
        assert m._class_weight() is None                      # WB drops class_weight="balanced"
        p = m.predict_proba(t)
        assert p.shape == (len(t), 2) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        w = m._fold_weights(y, g)
        assert w is not None and w.sum() == pytest.approx(len(set(g.tolist())))   # sum == W works

    def test_legacy_calls_strict_two_arg_channel(self, monkeypatch):
        # A strict legacy channel takes EXACTLY two positional args; the legacy stack must not pass
        # a spurious third None (which would raise TypeError). fit + predict must both succeed.
        t, y, g = _toy()

        def strict2(tr, te):
            return ch_char(tr, te)

        monkeypatch.setattr(sc, "make_channels", lambda cfg: {"s": strict2})
        m = sc.StackedChannelClassifier(CFG, inner_folds=3, seed=1).fit(t, y, groups=g)
        assert m.predict_proba(t).shape == (len(t), 2)

    def test_work_balanced_loss_downweights_a_long_book(self):
        # the loss the stack passes to every weighted estimator: a book split into more chunks gets
        # LESS per-chunk weight, and each work still carries equal total mass (deterministic).
        m = sc.StackedChannelClassifier(CFG, training_weighting=WORK_BALANCED)
        y = np.array([0, 0, 0, 0, 0])
        g = np.array(["a/long", "a/long", "a/long", "a/long", "a/short"], dtype=object)
        w = m._fold_weights(y, g)
        assert not np.allclose(w, w[0])                     # non-uniform (unlike legacy None)
        assert w[:4].sum() == pytest.approx(w[4])           # long book == short book in total mass
        assert w.sum() == pytest.approx(2)                  # sum == W works


# -- hashing-channel work-id validation
class TestHashingGroupsValidation:
    def test_two_arg_and_groups_none_stay_legacy(self):
        tr, te = ["x y", "y z"], ["x"]
        assert np.allclose(ch_char(tr, te).__getitem__(0).toarray(),
                           ch_char(tr, te, None).__getitem__(0).toarray())

    @pytest.mark.parametrize("bad", [
        "auth0/book0",                       # bare string (scalar posing as a per-chunk sequence)
        {"auth0/book0": 0, "auth0/book1": 1},  # mapping (no positional order)
        {"auth0/book0", "auth0/book1"},      # set (unstable order)
        [1, 1],                              # non-string ids (would collapse works, change W/IDF)
        [None, "auth0/book1"],               # None id
        ["auth0/book0"],                     # length mismatch (2 train chunks)
    ])
    def test_hashing_channel_rejects_malformed_groups(self, bad):
        with pytest.raises(ValueError):
            ch_char(["x y", "y z"], ["x"], bad)

    def test_hashing_channel_rejects_generator_groups(self):
        gen = (w for w in ("auth0/book0", "auth0/book1"))
        with pytest.raises(ValueError):
            ch_char(["x y", "y z"], ["x"], gen)


# -- complete work-balanced stack routing
class TestWorkBalancedStackWired:
    def test_factory_builds_work_balanced_stack(self):
        from stylo.eval.lobo import make_factory
        assert make_factory("stylo_stack", CFG, weighting=WORK_BALANCED)() is not None
        assert make_factory("stylo_stack", CFG, weighting=CHUNK_WEIGHTED_LEGACY)() is not None

    def test_calibration_is_group_aware(self):
        from stylo.models.calibration import choose_calibrator
        assert "groups" in inspect.signature(choose_calibrator).parameters
