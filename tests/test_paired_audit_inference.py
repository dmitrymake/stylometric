"""Synthetic tests for cluster-valid paired inference: cluster p, McNemar diagnostic, Holm (§3.3/§3.4)."""
from __future__ import annotations

import pytest

from stylo.eval.paired_audit import applicability as ap
from stylo.eval.paired_audit import inference as inf


class TestClusterPValue:
    def test_rule1_single_author_returns_one_even_with_effect(self):
        # a single author cluster cannot be resampled: p=1, NOT the 1/(B+1) trap
        a = [1, 1, 1, 1]
        b = [0, 0, 0, 0]
        authors = ["x", "x", "x", "x"]
        assert inf.paired_cluster_pvalue(a, b, authors, B=100) == 1.0

    def test_rule2_all_zero_differences_returns_one(self):
        a = [1, 0, 1, 0]
        b = [1, 0, 1, 0]
        authors = ["x", "x", "y", "y"]
        assert inf.paired_cluster_pvalue(a, b, authors, B=100) == 1.0

    def test_rule3_constant_nonzero_effect_not_special_cased(self):
        # every book: a correct, b wrong; >=2 authors -> general algorithm -> minimal p 1/(B+1)
        a = [1, 1, 1, 1]
        b = [0, 0, 0, 0]
        authors = ["x", "x", "y", "y"]
        assert inf.paired_cluster_pvalue(a, b, authors, B=200) == pytest.approx(1 / 201)

    def test_two_sided_symmetry_and_determinism(self):
        a = [1, 1, 0, 1, 0, 1]
        b = [0, 1, 1, 0, 0, 1]
        authors = ["x", "x", "y", "y", "z", "z"]
        p1 = inf.paired_cluster_pvalue(a, b, authors, B=500, seed=42)
        p2 = inf.paired_cluster_pvalue(a, b, authors, B=500, seed=42)
        p_swap = inf.paired_cluster_pvalue(b, a, authors, B=500, seed=42)
        assert p1 == p2                                  # deterministic
        assert p1 == p_swap                              # two-sided: |obs| and |centered| symmetric
        assert 0.0 < p1 <= 1.0

    def test_misaligned_or_empty_inputs_rejected(self):
        with pytest.raises(inf.PairedInferenceError):
            inf.paired_cluster_pvalue([1, 0], [1], ["x", "y"], B=10)
        with pytest.raises(inf.PairedInferenceError):
            inf.paired_cluster_pvalue([], [], [], B=10)

    def test_non_binary_correctness_rejected(self):
        with pytest.raises(inf.PairedInferenceError):
            inf.paired_cluster_pvalue([0.5, 1], [0, 1], ["x", "y"], B=10)


class TestMcNemarDiagnostic:
    def test_diagnostic_only_shape(self):
        d = inf.mcnemar_diagnostic([1, 1, 0, 1], [0, 1, 1, 0])
        assert d["role"] == "diagnostic_only"
        assert set(d) == {"b", "c", "mcnemar_p_diagnostic", "role"}
        assert 0.0 <= d["mcnemar_p_diagnostic"] <= 1.0


class TestHolm:
    def test_known_step_down_values(self):
        out = inf.holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.04}, m=3, alpha=0.05)
        assert out["a"]["holm_p"] == pytest.approx(0.03)
        assert out["b"]["holm_p"] == pytest.approx(0.04)
        assert out["c"]["holm_p"] == pytest.approx(0.04)
        assert all(out[k]["significant"] for k in out)

    def test_monotone_non_decreasing_by_rank(self):
        out = inf.holm_bonferroni({"a": 0.04, "b": 0.01}, m=2)
        assert out["b"]["holm_p"] == pytest.approx(0.02)     # smallest raw p, rank 0
        assert out["a"]["holm_p"] == pytest.approx(0.04)

    def test_m_is_not_reduced(self):
        small_family = inf.holm_bonferroni({"a": 0.01}, m=15)
        assert small_family["a"]["holm_p"] == pytest.approx(0.15)   # 15*0.01
        assert not small_family["a"]["significant"]                 # 0.15 !< 0.05

    def test_significant_is_strict(self):
        out = inf.holm_bonferroni({"a": 0.05}, m=1, alpha=0.05)     # adj == 0.05
        assert out["a"]["holm_p"] == pytest.approx(0.05)
        assert not out["a"]["significant"]                          # 0.05 < 0.05 is False

    def test_invalid_pvalues_and_m_rejected(self):
        with pytest.raises(inf.PairedInferenceError):
            inf.holm_bonferroni({"a": 1.2}, m=1)
        with pytest.raises(inf.PairedInferenceError):
            inf.holm_bonferroni({"a": True}, m=1)
        with pytest.raises(inf.PairedInferenceError):
            inf.holm_bonferroni({"a": 0.1, "b": 0.2}, m=1)         # m < number of hypotheses
        with pytest.raises(inf.PairedInferenceError):
            inf.holm_bonferroni({}, m=1)


class TestHolmRegisteredFamily:
    def test_full_family_runs_with_m15(self):
        raw = {member: 0.0001 for member in ap.holm_family()}
        out = inf.holm_over_registered_family(raw)
        assert len(out) == 15
        assert all(out[m]["significant"] for m in out)             # 15*0.0001 = 0.0015 < 0.05

    def test_missing_or_extra_member_invalidates_family(self):
        full = {member: 0.001 for member in ap.holm_family()}
        missing = dict(list(full.items())[:-1])
        with pytest.raises(ap.ApplicabilityError):
            inf.holm_over_registered_family(missing)
        extra = dict(full)
        extra[("majority", "A0")] = 0.001
        with pytest.raises(ap.ApplicabilityError):
            inf.holm_over_registered_family(extra)
