"""Work-balanced training-weight estimand invariants and dispatch contracts."""
from __future__ import annotations

import numpy as np
import pytest

from stylo.domain import work_weighting as ww


def _author_mass(weights, y):
    y = np.asarray(y)
    return {a: float(weights[y == a].sum()) for a in np.unique(y)}


def _work_mass(weights, groups):
    groups = np.asarray(groups, dtype=object)
    return {g: float(weights[groups == g].sum()) for g in dict.fromkeys(groups)}


class TestWorkSampleWeights:
    def test_total_mass_is_W_and_authors_equal(self):
        # W=4 works, A=2 authors: total = W, each author = W/A = 2.
        y = [0, 0, 0, 1, 1]
        groups = ["a/w1", "a/w1", "a/w2", "b/w3", "b/w4"]
        w = ww.work_sample_weights(y, groups)
        assert w.sum() == pytest.approx(4.0)
        assert _author_mass(w, y) == pytest.approx({0: 2.0, 1: 2.0})

    def test_each_work_within_author_is_equal(self):
        # author 0, two works (w1: 3 chunks, w2: 1). W=2, A=1 -> each work = W/(A*W_a) = 1
        y = [0, 0, 0, 0]
        groups = ["a/w1", "a/w1", "a/w1", "a/w2"]
        w = ww.work_sample_weights(y, groups)
        assert _work_mass(w, groups) == pytest.approx({"a/w1": 1.0, "a/w2": 1.0})

    def test_long_and_short_work_get_equal_total_train_weight(self):
        y = [0] * 11
        groups = ["a/long"] * 10 + ["a/short"]
        w = ww.work_sample_weights(y, groups)
        m = _work_mass(w, groups)
        assert m["a/long"] == pytest.approx(m["a/short"])

    def test_duplicating_a_chunk_within_a_work_keeps_author_total(self):
        y = [0, 0, 1]
        groups = ["a/w1", "a/w2", "b/w3"]
        base = _author_mass(ww.work_sample_weights(y, groups), y)
        # duplicate a chunk of a/w1
        y2 = [0, 0, 0, 1]
        groups2 = ["a/w1", "a/w1", "a/w2", "b/w3"]
        dup = _author_mass(ww.work_sample_weights(y2, groups2), y2)
        assert dup == pytest.approx(base)

    def test_work_spanning_two_authors_is_rejected(self):
        with pytest.raises(ValueError):
            ww.work_sample_weights([0, 1], ["shared/w", "shared/w"])

    def test_length_mismatch_rejected_both_directions(self):
        with pytest.raises(ValueError):
            ww.work_sample_weights([0, 0], ["a/w1"])          # more labels than groups
        with pytest.raises(ValueError):
            ww.work_sample_weights([0], ["a/w1", "a/w1"])     # more groups than labels
        with pytest.raises(ValueError):
            ww.aggregate_by_work(["c1"], ["w1", "w2"])
        with pytest.raises(ValueError):
            ww.aggregate_by_work(["c1", "c2"], ["w1"])

    def test_effective_sample_size_equals_num_works(self):
        # sum(weights) == W for a ragged corpus: work-level effective N.
        y = [0, 0, 0, 0, 1, 1]
        groups = ["a/w1", "a/w1", "a/w1", "a/w2", "b/w3", "b/w4"]
        assert ww.work_sample_weights(y, groups).sum() == pytest.approx(4.0)

    def test_legacy_returns_none(self):
        assert ww.training_sample_weights([0, 1], ["a", "b"], ww.CHUNK_WEIGHTED_LEGACY) is None

    def test_work_balanced_dispatch_matches_direct(self):
        y = [0, 0, 1]
        groups = ["a/w1", "a/w2", "b/w3"]
        np.testing.assert_allclose(
            ww.training_sample_weights(y, groups, ww.WORK_BALANCED),
            ww.work_sample_weights(y, groups),
        )


    def test_unequal_works_per_author(self):
        # author 0 has 3 works, author 1 has 1 work (W_a = {3, 1}); W=4, A=2
        y = [0, 0, 0, 0, 1]
        groups = ["a/w1", "a/w1", "a/w2", "a/w3", "b/w4"]
        w = ww.work_sample_weights(y, groups)
        assert w.sum() == pytest.approx(4.0)
        assert _author_mass(w, y) == pytest.approx({0: 2.0, 1: 2.0})          # each W/A
        assert _work_mass(w, groups)["a/w1"] == pytest.approx(2.0 / 3)        # 3 works share 2
        assert _work_mass(w, groups)["b/w4"] == pytest.approx(2.0)            # 1 work carries 2


class TestResolveAndAggregate:
    def test_resolve_defaults_and_validates(self):
        assert ww.resolve_training_weighting(None, default=ww.WORK_BALANCED) == ww.WORK_BALANCED
        assert ww.resolve_training_weighting(ww.CHUNK_WEIGHTED_LEGACY) == ww.CHUNK_WEIGHTED_LEGACY
        with pytest.raises(ValueError):
            ww.resolve_training_weighting("headline")

    def test_resolve_validates_the_fallback_default_too(self):
        # an invalid default must be rejected, not silently returned
        with pytest.raises(ValueError):
            ww.resolve_training_weighting(None, default="headline")

    def test_to_claim_label_maps_runtime_to_public(self):
        assert ww.to_claim_label(ww.CHUNK_WEIGHTED_LEGACY) == "chunk_weighted_training_legacy"
        assert ww.to_claim_label(ww.WORK_BALANCED) == "work_balanced"
        assert ww.to_claim_label(None) == "chunk_weighted_training_legacy"
        with pytest.raises(ValueError):
            ww.to_claim_label("headline")

    def test_config_default_and_override(self):
        from stylo.config import load_config, parse_set_overrides
        cfg = load_config()
        default = cfg.get_path("evaluation.training_weighting")
        assert ww.resolve_training_weighting(default) == ww.CHUNK_WEIGHTED_LEGACY
        over = load_config(overrides=parse_set_overrides(["evaluation.training_weighting=work_balanced"]))
        assert ww.resolve_training_weighting(over.get_path("evaluation.training_weighting")) == ww.WORK_BALANCED

    def test_aggregate_by_work_preserves_first_seen_order(self):
        items = ["c1", "c2", "c3", "c4"]
        groups = ["w2", "w1", "w2", "w1"]
        assert ww.aggregate_by_work(items, groups) == [
            ("w2", ["c1", "c3"]),
            ("w1", ["c2", "c4"]),
        ]
