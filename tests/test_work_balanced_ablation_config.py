"""AblationConfig and model-factory routing without changing estimator math.

The legacy and fully work-balanced corners, plus the three single-axis configurations, are runnable;
the remaining two-axis combinations fail closed until their routing is implemented. The A0/A4
corners reached through the ablation path must reproduce the frozen pre-routing goldens
quantized-exact at the fixture's declared round_decimals (proved here for the spaCy-free models).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from stylo import jsonio
from stylo.config import load_config
from stylo.eval.dispatch import fit_estimator
from stylo.eval.lobo import make_factory, make_factory_for_ablation
from stylo.eval.work_weighting import (AblationConfig, AblationNotImplementedError,
                                       CHUNK_WEIGHTED_LEGACY, FULL_WB_ABLATION,
                                       LEGACY_ABLATION, WEIGHTS_ONLY_ABLATION, WORK_BALANCED)

CFG = load_config()
FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "work_balanced_ablation_goldens_v1.json"


def test_corners_map_to_the_two_enums():
    assert LEGACY_ABLATION == AblationConfig(False, False, False)
    assert FULL_WB_ABLATION == AblationConfig(True, True, True)
    assert LEGACY_ABLATION.to_weighting() == CHUNK_WEIGHTED_LEGACY
    assert FULL_WB_ABLATION.to_weighting() == WORK_BALANCED
    assert AblationConfig.from_weighting(CHUNK_WEIGHTED_LEGACY) == LEGACY_ABLATION
    assert AblationConfig.from_weighting(WORK_BALANCED) == FULL_WB_ABLATION
    assert LEGACY_ABLATION.is_legacy_corner and FULL_WB_ABLATION.is_full_wb_corner


def test_intermediate_axes_are_not_yet_runnable():
    # Only the three two-axis combinations (WFR 110/101/011) remain unrouted; the three single-axis
    # configurations A1/A2/A3 (100/010/001) are already supported.
    for cfg in (AblationConfig(True, True, False), AblationConfig(True, False, True),
                AblationConfig(False, True, True)):
        assert not (cfg.is_legacy_corner or cfg.is_full_wb_corner or cfg.is_weights_only_corner
                    or cfg.is_feature_state_only_corner or cfg.is_relative_fw_only_corner)
        with pytest.raises(AblationNotImplementedError):
            cfg.to_weighting()
        with pytest.raises(AblationNotImplementedError):        # and through the factory routing
            make_factory_for_ablation("stylo", CFG, ablation=cfg)


def test_weights_only_A1_is_corner_only_in_to_weighting_but_routes_via_factory():
    # A1 has NO production weighting enum (to_weighting stays corner-only) but IS runnable for the
    # LR-family through the audit-only factory path — it must never collapse to WORK_BALANCED.
    assert WEIGHTS_ONLY_ABLATION == AblationConfig(True, False, False)
    assert WEIGHTS_ONLY_ABLATION.is_weights_only_corner
    assert not (WEIGHTS_ONLY_ABLATION.is_legacy_corner or WEIGHTS_ONLY_ABLATION.is_full_wb_corner)
    with pytest.raises(AblationNotImplementedError):
        WEIGHTS_ONLY_ABLATION.to_weighting()
    for spec in ("stylo", "bow_lr", "stylo_stack"):
        est = make_factory_for_ablation(spec, CFG, ablation=WEIGHTS_ONLY_ABLATION)()
        assert getattr(est, "training_weighting", CHUNK_WEIGHTED_LEGACY) != WORK_BALANCED


def test_non_bool_axes_rejected():
    for bad in ((1, False, False), (False, "yes", False), (False, False, None)):
        with pytest.raises(TypeError):
            AblationConfig(*bad)


def test_from_weighting_is_fail_closed():
    import numpy as np
    with pytest.raises(TypeError):
        AblationConfig.from_weighting(None)                     # None must NOT become legacy
    with pytest.raises(TypeError):
        AblationConfig.from_weighting(np.str_(CHUNK_WEIGHTED_LEGACY))   # np.str_ / subclass rejected
    with pytest.raises(ValueError):
        AblationConfig.from_weighting("bogus")


def test_routing_rejects_non_exact_ablation():
    # a duck-typed / subclass object whose to_weighting could route A4 axes to a legacy estimator
    class _Sub(AblationConfig):
        def to_weighting(self):
            return CHUNK_WEIGHTED_LEGACY
    class _Fake:
        weights = feature_fit = relative_fw = True
        def to_weighting(self):
            return CHUNK_WEIGHTED_LEGACY
    for bad in (_Sub(True, True, True), _Fake(), CHUNK_WEIGHTED_LEGACY, None):
        with pytest.raises(TypeError):
            make_factory_for_ablation("stylo", CFG, ablation=bad)
    # a tampered (frozen) instance with a non-bool axis is rejected at the boundary
    corrupt = AblationConfig(False, False, False)
    object.__setattr__(corrupt, "weights", 1)
    with pytest.raises(TypeError):
        make_factory_for_ablation("stylo", CFG, ablation=corrupt)


def test_routing_ignores_a_shadowed_to_weighting():
    # a valid A4 instance whose to_weighting is shadowed to return legacy must STILL build the full-WB
    # estimator: the weighting is computed from re-checked axes via the class method, not the instance's.
    # stylo_stack carries an observable training_weighting so full-WB is distinguishable from legacy.
    shadowed = AblationConfig(True, True, True)
    object.__setattr__(shadowed, "to_weighting", lambda: CHUNK_WEIGHTED_LEGACY)
    routed = make_factory_for_ablation("stylo_stack", CFG, ablation=shadowed)()
    assert routed.training_weighting == WORK_BALANCED != CHUNK_WEIGHTED_LEGACY   # shadow ignored, not legacy


def test_ablation_routing_equals_the_matching_weighting():
    # ablation-routed and weighting-built estimators are the SAME construction (math unchanged)
    for spec in ("stylo", "delta_cos:500", "char_cos", "bow_lr", "majority", "stylo_stack"):
        for corner, w in ((LEGACY_ABLATION, CHUNK_WEIGHTED_LEGACY), (FULL_WB_ABLATION, WORK_BALANCED)):
            a = make_factory_for_ablation(spec, CFG, ablation=corner)()
            b = make_factory(spec, CFG, weighting=w)()
            assert type(a) is type(b)
            assert getattr(a, "training_weighting", None) == getattr(b, "training_weighting", None)


def test_make_factory_weighting_contract_unchanged():
    # Ablation routing must not weaken the production contract: weighting stays a required keyword.
    import inspect
    p = inspect.signature(make_factory).parameters
    assert p["weighting"].default is inspect._empty and p["weighting"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "ablation" not in p                                  # routing lives in make_factory_for_ablation
    a = inspect.signature(make_factory_for_ablation).parameters["ablation"]
    assert a.kind == inspect.Parameter.KEYWORD_ONLY             # ablation is keyword-only


# ── the frozen goldens are preserved through the ablation path (spaCy-free models) ──
def _panel():
    d = jsonio.load_strict(FIXTURE)
    p = d["panels"]["P1"]
    return (np.array(p["texts"], dtype=object), np.array(p["y"]),
            np.array(p["groups"], dtype=object), d["models"], d["numeric_contract"]["round_decimals"])


@pytest.mark.parametrize("spec", ["delta_cos:500", "delta_cos:12", "char_cos", "bow_lr"])
def test_goldens_reproduced_via_ablation_path(spec):
    # quantized-exact at the fixture's declared round_decimals (the contract, not a hard-coded 12)
    texts, y, groups, models, rd = _panel()
    for corner_name, ablation in (("A0", LEGACY_ABLATION), ("A4", FULL_WB_ABLATION)):
        est = make_factory_for_ablation(spec, CFG, ablation=ablation)()
        fit_estimator(est, texts, y, groups)
        got = np.round(np.asarray(est.predict_proba(texts), dtype=np.float64), rd)
        want = np.round(np.asarray(models[spec][corner_name]["proba"], dtype=np.float64), rd)
        np.testing.assert_array_equal(got, want)               # routing must not move the goldens
