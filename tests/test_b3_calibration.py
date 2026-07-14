"""B3: group-aware calibrator selection + work_balanced stack unblock.

Calibration unit tests are spaCy-free; the end-to-end stack test monkeypatches ``make_channels`` to
the two hashing channels so the work_balanced stack fits without spaCy.
"""
from __future__ import annotations

import numpy as np
import pytest

from stylo.config import load_config
from stylo.eval.calibration import choose_calibrator
from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY, WORK_BALANCED
from stylo.models import stacked_clf as sc
from stylo.models.channels import ch_char, ch_word

CFG = load_config()


def _separable(works_per_class):
    """OOF scores + y + work groups: class-separable, ``works_per_class`` works (2 chunks) per class."""
    oof, y, g = [], [], []
    for c in range(2):
        for w in range(works_per_class[c]):
            for _ in range(2):
                row = [0.0, 0.0]; row[c] = 2.0
                oof.append(row); y.append(c); g.append(f"c{c}/w{w}")
    return np.array(oof, float), np.array(y), np.array(g, dtype=object)


class TestGroupAwareCalibration:
    def test_legacy_path_is_unchanged(self):
        oof, y, _ = _separable([4, 4])
        a_cal, a = choose_calibrator(oof, y, seed=1)
        b_cal, b = choose_calibrator(oof, y, seed=1, groups=None)
        assert a["method"] == b["method"] and "group_aware" not in a
        assert np.allclose(a_cal(oof), b_cal(oof))

    def test_single_work_class_disables_calibration(self):
        oof, y, g = _separable([3, 1])          # class 1 has a single work -> cannot hold one out
        cal, p = choose_calibrator(oof, y, groups=g)
        assert p["method"] == "identity" and p["calibration_disabled"] is True
        assert "min_works_per_class=1" in p["reason"]
        assert np.allclose(cal(oof).sum(axis=1), 1.0)

    def test_n_splits_eff_is_min_of_nsplits_and_min_works(self):
        oof, y, g = _separable([3, 3])
        assert choose_calibrator(oof, y, groups=g, n_splits=3)[1]["n_splits"] == 3
        assert choose_calibrator(oof, y, groups=g, n_splits=2)[1]["n_splits"] == 2   # capped by n_splits
        oof2, y2, g2 = _separable([2, 3])
        assert choose_calibrator(oof2, y2, groups=g2, n_splits=5)[1]["n_splits"] == 2  # capped by min_works

    def test_group_aware_passport_reports_work_held_out(self):
        oof, y, g = _separable([4, 4])
        _, p = choose_calibrator(oof, y, groups=g)
        assert p["group_aware"] is True and "heldout_work_nll" in p
        assert not p.get("calibration_disabled")

    def test_pooled_equal_work_nll_matches_independent_reference(self):
        # HIGH-1: the reported score is the POOLED equal-work NLL (sum of per-work loss / total works),
        # not a fold-averaged mean — reproduced independently with the same folds.
        from sklearn.model_selection import StratifiedGroupKFold
        from stylo.eval.calibration import _fit_method, _work_loss_sum
        oof, y, g = _separable([5, 5])                      # 10 works, uneven fold sizes at k=3
        _, p = choose_calibrator(oof, y, groups=g, n_splits=3, seed=7)
        sgkf = StratifiedGroupKFold(3, shuffle=True, random_state=7)
        total, works = 0.0, 0
        for fit_i, val_i in sgkf.split(oof, y, g):
            cal, _ = _fit_method("identity", oof[fit_i], y[fit_i], 7)
            total += _work_loss_sum(cal(oof[val_i]), y[val_i], g[val_i])
            works += len({str(w) for w in g[val_i]})
        assert works == 10                                 # every work held out exactly once == W
        assert p["heldout_work_nll"]["identity"] == round(total / works, 4)

    def test_class_absent_from_validation_side_disables(self, monkeypatch):
        # HIGH-3: a class missing from a fold's VALIDATION side (not only train) must disable.
        import stylo.eval.calibration as C
        oof, y, g = _separable([2, 2])

        class _FakeSGKF:
            def __init__(self, *a, **k):
                pass

            def split(self, X, yy, groups):
                # fit has both classes; this val fold holds out only class-0 works -> class 1 absent
                return [(np.array([2, 3, 4, 5, 6, 7]), np.array([0, 1]))]

        monkeypatch.setattr(C, "StratifiedGroupKFold", _FakeSGKF)
        _, p = choose_calibrator(oof, y, groups=g, n_splits=2)
        assert p.get("calibration_disabled") is True and "absent" in p["reason"]


class TestGroupedInputContract:
    # HIGH-4: fail-closed inputs for the group-aware path
    def _ok(self):
        return _separable([2, 2])

    def test_mixed_label_work_rejected(self):
        with pytest.raises(ValueError):
            choose_calibrator(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]),
                              groups=np.array(["w", "w"], dtype=object))

    def test_nonfinite_oof_rejected(self):
        oof, y, g = self._ok()
        oof = oof.copy(); oof[0, 0] = np.nan
        with pytest.raises(ValueError):
            choose_calibrator(oof, y, groups=g)

    def test_bad_labels_rejected(self):
        oof, y, g = self._ok()
        for bad in (y.astype(float), np.where(y == 0, -1, y), np.full_like(y, 5)):
            with pytest.raises(ValueError):
                choose_calibrator(oof, bad, groups=g)

    def test_length_mismatch_and_short_groups_rejected(self):
        oof, y, g = self._ok()
        with pytest.raises(ValueError):
            choose_calibrator(oof, y[:-1], groups=g)              # y too short
        with pytest.raises(ValueError):
            choose_calibrator(oof, y, groups=g[:-1])              # groups too short (not a disabled pass)

    def test_bare_string_and_mapping_groups_rejected(self):
        oof, y, g = self._ok()
        for bad in ("c0/w0", {w: 0 for w in g}):
            with pytest.raises(ValueError):
                choose_calibrator(oof, y, groups=bad)

    def test_oof_dtype_checked_before_float_cast(self):
        # MEDIUM: bool / complex / numeric-string oof must be rejected, not silently coerced to float
        _, y, g = self._ok()
        bool_oof = np.zeros((8, 2), dtype=bool)
        complex_oof = self._ok()[0].astype(complex)
        str_oof = np.array([["1", "0"]] * 8)
        for bad in (bool_oof, complex_oof, str_oof):
            with pytest.raises(ValueError):
                choose_calibrator(bad, y, groups=g)


class TestMethodsAndNSplitsValidation:
    # MEDIUM: materialise methods (ordered-unique), strict n_splits
    def test_generator_and_duplicate_methods_are_materialised(self):
        oof, y, g = _separable([3, 3])
        gen = (m for m in ("identity", "identity", "temperature"))   # duplicate + single-use generator
        _, p = choose_calibrator(oof, y, groups=g, methods=gen)
        assert set(p["heldout_work_nll"]) == {"identity", "temperature"}   # deduped, not phantom zeros

    def test_unknown_method_rejected(self):
        oof, y, g = _separable([3, 3])
        with pytest.raises(ValueError):
            choose_calibrator(oof, y, groups=g, methods=("identity", "bogus"))

    def test_set_or_mapping_methods_rejected(self):
        # MEDIUM: an unordered Set/Mapping is hash-seed dependent -> reject (non-deterministic choice)
        oof, y, g = _separable([3, 3])
        for bad in ({"identity", "temperature"}, {"identity": 1, "temperature": 2}):
            with pytest.raises(ValueError):
                choose_calibrator(oof, y, groups=g, methods=bad)

    def test_n_splits_must_be_integer_ge_2(self):
        oof, y, g = _separable([3, 3])
        for bad in (2.9, 1, True, "3"):
            with pytest.raises(ValueError):
                choose_calibrator(oof, y, groups=g, n_splits=bad)


class TestPooledDenominatorContract:
    # LOW: assert (not assume) the splitter holds out every row/work in validation exactly once
    def _bad_splitter(self, folds):
        class _Fake:
            def __init__(self, *a, **k):
                pass

            def split(self, X, yy, gg):
                return [(np.array(f), np.array(v)) for f, v in folds]
        return _Fake

    def test_row_not_held_out_exactly_once_rejected(self, monkeypatch):
        import stylo.eval.calibration as C
        oof, y, g = _separable([2, 2])                       # rows 0-3 class0, 4-7 class1
        # both folds class-complete on both sides, but rows 6,7 never validated and 4,5 twice
        monkeypatch.setattr(C, "StratifiedGroupKFold",
                            self._bad_splitter([([2, 3, 6, 7], [0, 1, 4, 5]),
                                                ([0, 1, 6, 7], [2, 3, 4, 5])]))
        with pytest.raises(ValueError):
            choose_calibrator(oof, y, groups=g, n_splits=2)

    def test_train_validation_overlap_rejected(self, monkeypatch):
        import stylo.eval.calibration as C
        oof, y, g = _separable([2, 2])
        # exact-once validation coverage, but fold 1 has row 0 in BOTH train and validation
        monkeypatch.setattr(C, "StratifiedGroupKFold",
                            self._bad_splitter([([0, 2, 3, 6, 7], [0, 1, 4, 5]),
                                                ([0, 1, 4, 5], [2, 3, 6, 7])]))
        with pytest.raises(ValueError):
            choose_calibrator(oof, y, groups=g, n_splits=2)

    def test_train_not_complement_of_validation_rejected(self, monkeypatch):
        # defense-in-depth: train must be EXACTLY complement(validation) — fold 1 drops row 2 entirely
        import stylo.eval.calibration as C
        oof, y, g = _separable([2, 2])
        monkeypatch.setattr(C, "StratifiedGroupKFold",
                            self._bad_splitter([([3, 6, 7], [0, 1, 4, 5]),
                                                ([0, 1, 4, 5], [2, 3, 6, 7])]))
        with pytest.raises(ValueError):
            choose_calibrator(oof, y, groups=g, n_splits=2)


# ── work_balanced stack unblocked end-to-end ───────────────────────────────────
def _toy():
    texts, y, g = [], [], []
    for a, w in ((0, "alpha kappa lambda"), (1, "beta gamma delta")):
        for b in range(6):
            for c in range(2):
                texts.append(f"{w} book{b} chunk{c} {'z' * (a + 1)}")
                y.append(a)
                g.append(f"auth{a}/book{b}")
    return texts, np.array(y), np.array(g, dtype=object)


@pytest.fixture
def _hashing_only(monkeypatch):
    monkeypatch.setattr(sc, "make_channels", lambda cfg: {"char": ch_char, "word": ch_word})


class TestWorkBalancedStackUnblocked:
    def test_factory_builds_and_variant_role_primary(self):
        from stylo.eval.final import _variant_role
        from stylo.eval.lobo import make_factory
        assert make_factory("stylo_stack", CFG, weighting=WORK_BALANCED)() is not None
        assert _variant_role("stylo_stack", WORK_BALANCED) == "primary"

    def test_run_final_preflight_no_longer_blocks_stack(self):
        import inspect
        from stylo.eval import final
        src = inspect.getsource(final.run_final)
        assert "stylo_stack under work_balanced is blocked" not in src

    def test_wb_stack_fits_predicts_with_group_aware_calibration(self, _hashing_only):
        t, y, g = _toy()
        m = sc.StackedChannelClassifier(CFG, inner_folds=3, seed=1,
                                        training_weighting=WORK_BALANCED).fit(t, y, groups=g)
        p = m.predict_proba(t)
        assert p.shape == (len(t), 2) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        cal = m.passport_["calibration"]
        assert all(v.get("group_aware") is True for v in cal.values())   # groups reached the calibrator

    def test_legacy_stack_calibration_is_not_group_aware(self, _hashing_only):
        t, y, g = _toy()
        m = sc.StackedChannelClassifier(CFG, inner_folds=3, seed=1,
                                        training_weighting=CHUNK_WEIGHTED_LEGACY).fit(t, y, groups=g)
        assert all("group_aware" not in v for v in m.passport_["calibration"].values())

    def test_disabled_calibration_forces_equal_and_no_meta(self, _hashing_only, monkeypatch):
        # HIGH-2: when calibration is disabled the stack must fall back to identity + equal ensemble;
        # no meta-CV/meta-LR selection may run and mode_ cannot become "stacked".
        import stylo.models.stacked_clf as S
        from stylo.eval.calibration import _fit_method

        def fake_choose(oof, y, seed=42, groups=None, **kw):
            cal, _ = _fit_method("identity", oof, y, seed)
            return cal, {"method": "identity", "group_aware": True,
                         "calibration_disabled": True, "reason": "min_works_per_class=1 < 2"}

        monkeypatch.setattr(S, "choose_calibrator", fake_choose)
        t, y, g = _toy()
        m = sc.StackedChannelClassifier(CFG, inner_folds=3, seed=1,
                                        training_weighting=WORK_BALANCED).fit(t, y, groups=g)
        assert m.mode_ == "equal" and m.meta_ is None
        assert m.passport_["calibration_disabled"] is True
        assert m.predict_proba(t).shape == (len(t), 2)      # equal ensemble still predicts
