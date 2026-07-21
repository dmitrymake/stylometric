"""Synthetic tests for the pre-registered noninferiority headline gate (§3.5)."""
from __future__ import annotations

import numpy as np
import pytest

from stylo.eval.paired_audit import headline as hl


class TestHeadlineGate:
    def test_relabel_when_lower_above_neg_margin(self):
        assert hl.headline_gate(-0.01, 0.05, margin=0.02) == "relabel"

    def test_keep_legacy_when_upper_below_neg_margin(self):
        assert hl.headline_gate(-0.06, -0.03, margin=0.02) == "keep_legacy"

    def test_inconclusive_when_straddling(self):
        assert hl.headline_gate(-0.05, 0.05, margin=0.02) == "inconclusive"

    def test_boundary_equality_is_inconclusive(self):
        # lower bound exactly at -margin: not strictly > -margin -> inconclusive
        assert hl.headline_gate(-0.02, 0.5, margin=0.02) == "inconclusive"
        # upper bound exactly at -margin: not strictly < -margin -> inconclusive
        assert hl.headline_gate(-0.5, -0.02, margin=0.02) == "inconclusive"

    def test_inverted_ci_rejected(self):
        with pytest.raises(hl.HeadlineError):
            hl.headline_gate(0.1, -0.1, margin=0.02)

    def test_stat_input_validation(self):
        with pytest.raises(hl.HeadlineError):
            hl.headline_gate(-0.01, 0.05, margin=0)                             # margin must be > 0
        with pytest.raises(hl.HeadlineError):
            hl.author_clustered_accuracy_ci([1, 0], ["x", "y"], iters=0)        # iters positive
        with pytest.raises(hl.HeadlineError):
            hl.author_clustered_accuracy_ci([1, 0], ["x", "y"], iters=10, quantiles=(50, 50))
        with pytest.raises(hl.HeadlineError):
            hl.author_clustered_accuracy_ci([1, 0], ["x", ""], iters=10)        # empty author id


class TestClusterCI:
    def test_accuracy_ci_contains_point_and_is_deterministic(self):
        rng = np.random.default_rng(0)
        correct = (rng.random(60) < 0.85).astype(int)
        authors = [f"a{i % 6}" for i in range(60)]
        ci1 = hl.author_clustered_accuracy_ci(correct, authors, iters=500, seed=42)
        ci2 = hl.author_clustered_accuracy_ci(correct, authors, iters=500, seed=42)
        assert ci1 == ci2
        assert ci1["lo"] <= ci1["point"] <= ci1["hi"]
        assert ci1["point"] == pytest.approx(correct.mean())

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(hl.HeadlineError):
            hl.author_clustered_accuracy_ci([1, 0], ["a"], iters=10)
        with pytest.raises(hl.HeadlineError):
            hl.author_clustered_accuracy_ci([], [], iters=10)

    def test_non_binary_correctness_rejected(self):
        with pytest.raises(hl.HeadlineError):
            hl.author_clustered_accuracy_ci([0.5, 1], ["x", "y"], iters=10)
        with pytest.raises(hl.HeadlineError):
            hl.paired_accuracy_diff_ci([0.5, 1], [0, 1], ["x", "y"], iters=10)


class TestEvaluateHeadline:
    def test_strong_gain_relabels_and_reports_both_cis(self):
        a4 = [1] * 40
        a0 = [0] * 40
        authors = [f"a{i % 8}" for i in range(40)]
        out = hl.evaluate_headline(a4, a0, authors, iters=500)
        assert out["endpoint"] == "stylo_lobo_a4_minus_a0_accuracy"
        assert out["decision"] == "relabel"                       # diff CI ~ [1,1], lo > -0.02
        assert out["diff_ci"]["point"] == pytest.approx(1.0)
        assert out["a4_abs_accuracy_ci"]["point"] == pytest.approx(1.0)
        assert out["macro_f1_ci"] == "withdrawn"
        assert out["seed"] == 42 and out["quantiles"] == [2.5, 97.5]

    def test_strong_loss_keeps_legacy(self):
        a4 = [0] * 40
        a0 = [1] * 40
        authors = [f"a{i % 8}" for i in range(40)]
        out = hl.evaluate_headline(a4, a0, authors, iters=500)
        assert out["decision"] == "keep_legacy"                   # diff CI ~ [-1,-1], hi < -0.02

    def test_near_parity_is_inconclusive(self):
        rng = np.random.default_rng(1)
        authors = [f"a{i % 10}" for i in range(80)]
        base = (rng.random(80) < 0.8).astype(int)
        a4 = base.copy()
        a0 = base.copy()
        a4[0] = 1 - a4[0]                                         # a one-book difference near zero
        out = hl.evaluate_headline(a4, a0, authors, iters=1000)
        assert out["decision"] == "inconclusive"
