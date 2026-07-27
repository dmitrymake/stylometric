from __future__ import annotations

import copy

import numpy as np
import pytest

from stylo.config import load_config, with_overrides
from stylo.domain.lobo_vnext import FoldSpec, ModelSpec, canonical_sha256
from stylo.eval import lobo_vnext_models as vm
from stylo.models.baselines import CharCosineBaseline
from stylo.models.work_balanced import WorkBalancedStyloPipeline


def _cfg():
    return load_config()


def _fold() -> FoldSpec:
    return FoldSpec.build(
        fold_id="fold-a3",
        test_work_id="a/a3",
        content_component_id="component-a3",
        train_work_ids=("a/a1", "a/a2", "b/b1", "b/b2"),
        purged_work_ids=(),
        probability_class_order=("a", "b"),
        metric_label_order=("a", "b"),
    )


def _train_rows():
    texts = np.asarray(
        [
            "общий стиль альфа альфа",
            "общий стиль альфа один",
            "общий стиль бета бета",
            "общий стиль бета два",
        ],
        dtype=object,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = np.asarray(["a/a1", "a/a2", "b/b1", "b/b2"], dtype=object)
    return texts, labels, groups


def _rehash_model(raw: dict) -> ModelSpec:
    raw["self_hash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    return ModelSpec.from_dict(raw)


def test_r1_specs_are_exact_owner_selected_no_inner_cv_and_source_bound():
    cfg = _cfg()
    primary = vm.build_r1_model_spec(role="primary", cfg=cfg)
    baseline = vm.build_r1_model_spec(role="baseline", cfg=cfg)

    assert primary.model_id == primary.family == "stylo"
    assert baseline.model_id == baseline.family == "char_cos"
    assert primary.weighting == baseline.weighting == "work_balanced"
    assert primary.features == (
        "char_ngrams",
        "function_words",
        "syntax",
        "pos_ngrams",
        "punctuation_ngrams",
        "dependency",
        "morphology",
        "length_dist",
    )
    assert baseline.features == ("char_ngrams",)
    for spec in (primary, baseline):
        assert spec.approved_for_exploratory is True
        assert spec.owner_selected is True
        assert spec.requires_inner_cv is False
        assert spec.inner_cv_splits is None
        assert spec.supports_component_aware_inner_cv is False
        assert dict(spec.seeds) == {"model": 42}
        hyper = dict(spec.hyperparameters)
        assert hyper["adapter_source_sha256"] == vm.r1_adapter_source_sha256()
        assert (
            hyper["scientific_config_sha256"]
            == vm.r1_scientific_config_sha256(cfg)
        )
        assert vm.validate_r1_model_spec(spec, cfg=cfg) in {
            "primary",
            "baseline",
        }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("seed", True),
        ("seed", 42.0),
        ("model.classifier.C", 2.0),
        ("model.classifier.max_iter", 2000.0),
        ("model.calibration.enabled", True),
        ("features.embeddings.enabled", True),
        ("features.char_ngrams.ngram_range", [3, 4]),
        ("language.spacy_model_version", "different"),
    ],
)
def test_r1_profile_rejects_config_drift(path, value):
    cfg = with_overrides(_cfg(), {path: value})
    with pytest.raises(vm.R1ModelAdapterError, match="owner-selected R1 profile"):
        vm.build_r1_model_spec(role="primary", cfg=cfg)


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("family", "other"),
        ("features", ["char_ngrams", "extra"]),
        ("weighting", "chunk_weighted_legacy"),
        ("seeds", {"model": 43}),
        ("requires_inner_cv", True),
        ("owner_selected", False),
    ],
)
def test_rehashed_nearby_model_specs_are_rejected(field, mutation):
    cfg = _cfg()
    raw = vm.build_r1_model_spec(role="baseline", cfg=cfg).to_dict()
    raw[field] = mutation
    if field == "requires_inner_cv":
        raw["inner_cv_splits"] = 2
        raw["supports_component_aware_inner_cv"] = True
    mutated = _rehash_model(raw)
    with pytest.raises(vm.R1ModelAdapterError, match="exact live R1 contract"):
        vm.validate_r1_model_spec(mutated, cfg=cfg)


def test_rehashed_extra_hyperparameter_and_stale_adapter_sha_are_rejected():
    cfg = _cfg()
    for change in ("extra", "source"):
        raw = vm.build_r1_model_spec(role="primary", cfg=cfg).to_dict()
        if change == "extra":
            raw["hyperparameters"]["extra"] = "not-r1"
        else:
            raw["hyperparameters"]["adapter_source_sha256"] = "0" * 64
        mutated = _rehash_model(raw)
        with pytest.raises(vm.R1ModelAdapterError, match="exact live R1 contract"):
            vm.validate_r1_model_spec(mutated, cfg=cfg)


def test_adapter_receipt_is_exact_self_hashed_and_model_bound():
    cfg = _cfg()
    primary = vm.build_r1_model_spec(role="primary", cfg=cfg)
    receipt = vm.build_r1_model_adapter_receipt(primary, cfg=cfg)
    payload = {key: value for key, value in receipt.items() if key != "self_hash"}

    assert receipt["role"] == "primary"
    assert receipt["model_spec_sha256"] == primary.self_hash
    assert receipt["self_hash"] == canonical_sha256(payload)
    assert vm.validate_r1_model_adapter_receipt(
        receipt, model_spec=primary, cfg=cfg
    ) == receipt

    bad = {**receipt, "extra": 1}
    with pytest.raises(vm.R1ModelAdapterError, match="wrong keys"):
        vm.validate_r1_model_adapter_receipt(
            bad, model_spec=primary, cfg=cfg
        )
    bool_as_int = {**receipt, "outer_train_only": 1}
    bool_as_int["self_hash"] = canonical_sha256(
        {
            key: value
            for key, value in bool_as_int.items()
            if key != "self_hash"
        }
    )
    with pytest.raises(vm.R1ModelAdapterError, match="does not match"):
        vm.validate_r1_model_adapter_receipt(
            bool_as_int, model_spec=primary, cfg=cfg
        )
    baseline = vm.build_r1_model_spec(role="baseline", cfg=cfg)
    with pytest.raises(vm.R1ModelAdapterError, match="does not match"):
        vm.validate_r1_model_adapter_receipt(
            receipt, model_spec=baseline, cfg=cfg
        )


def test_primary_and_baseline_construct_only_exact_generic_components():
    cfg = _cfg()
    fold = _fold()
    primary = vm.build_r1_model_spec(role="primary", cfg=cfg)
    baseline = vm.build_r1_model_spec(role="baseline", cfg=cfg)

    primary_adapter = vm.make_r1_model_factory(
        cfg=cfg, model_spec=primary
    )(primary, fold)
    primary_builder = vm.make_factory(
        "stylo", cfg, weighting="work_balanced"
    )
    primary_estimator = primary_builder()
    vm._verify_constructed_estimator(primary_estimator, role="primary")
    assert type(primary_estimator) is WorkBalancedStyloPipeline
    assert primary_adapter.classes_ is None

    baseline_builder = vm.make_factory(
        "char_cos", cfg, weighting="work_balanced"
    )
    baseline_estimator = baseline_builder()
    vm._verify_constructed_estimator(baseline_estimator, role="baseline")
    assert type(baseline_estimator) is CharCosineBaseline


def test_outer_train_receipt_is_checked_before_generic_factory(monkeypatch):
    cfg = _cfg()
    spec = vm.build_r1_model_spec(role="baseline", cfg=cfg)
    adapter = vm.make_r1_model_factory(cfg=cfg, model_spec=spec)(spec, _fold())
    texts, labels, groups = _train_rows()
    calls = []

    def forbidden_factory(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("factory must not be reached")

    monkeypatch.setattr(vm, "make_factory", forbidden_factory)
    bad_groups = groups.copy()
    bad_groups[-1] = "a/a3"
    with pytest.raises(vm.R1ModelAdapterError, match="train_work_ids"):
        adapter.fit(
            texts,
            labels,
            groups=bad_groups,
            inner_splits=(),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("labels", "groups", "inner_splits", "message"),
    [
        (
            np.asarray([0, 0, 0, 0], dtype=np.int64),
            np.asarray(["a/a1", "a/a2", "b/b1", "b/b2"], dtype=object),
            (),
            "retain every class",
        ),
        (
            np.asarray([0, 1, 1, 1], dtype=np.int64),
            np.asarray(["a/a1", "a/a1", "b/b1", "b/b2"], dtype=object),
            (),
            "train_work_ids",
        ),
        (
            np.asarray([0, 0, 1, 1], dtype=np.int64),
            np.asarray(["a/a1", "a/a2", "b/b1", "b/b2"], dtype=object),
            ("silent-inner-cv",),
            "inner_splits",
        ),
    ],
)
def test_outer_train_rejects_class_work_and_inner_cv_drift(
    labels, groups, inner_splits, message
):
    cfg = _cfg()
    spec = vm.build_r1_model_spec(role="baseline", cfg=cfg)
    adapter = vm.make_r1_model_factory(cfg=cfg, model_spec=spec)(spec, _fold())
    texts, _, _ = _train_rows()
    with pytest.raises(vm.R1ModelAdapterError, match=message):
        adapter.fit(
            texts,
            labels,
            groups=groups,
            inner_splits=inner_splits,
        )


def test_baseline_synthetic_fit_has_exact_p_order_and_finite_probabilities():
    cfg = _cfg()
    spec = vm.build_r1_model_spec(role="baseline", cfg=cfg)
    adapter = vm.make_r1_model_factory(cfg=cfg, model_spec=spec)(spec, _fold())
    texts, labels, groups = _train_rows()

    adapter.fit(texts, labels, groups=groups, inner_splits=())
    probabilities = adapter.predict_proba(
        np.asarray(["общий стиль альфа проверка"], dtype=object)
    )

    assert np.array_equal(adapter.classes_, np.asarray([0, 1]))
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    with pytest.raises(vm.R1ModelAdapterError, match="immutable"):
        adapter.fit(texts, labels, groups=groups, inner_splits=())


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (np.asarray([[1.0, 0.0, 0.0]]), "width"),
        (np.asarray([[np.nan, np.nan]]), "NaN"),
        (np.asarray([["0.5", "0.5"]], dtype=object), "numeric scalars"),
        (np.asarray([[True, False]], dtype=object), "numeric scalars"),
    ],
)
def test_prediction_rejects_wrong_width_nonfinite_and_coercible_values(
    raw, message
):
    cfg = _cfg()
    spec = vm.build_r1_model_spec(role="baseline", cfg=cfg)
    adapter = vm.make_r1_model_factory(cfg=cfg, model_spec=spec)(spec, _fold())
    texts, labels, groups = _train_rows()
    adapter.fit(texts, labels, groups=groups, inner_splits=())
    adapter._estimator.predict_proba = lambda _texts: raw

    with pytest.raises(vm.R1ModelAdapterError, match=message):
        adapter.predict_proba(np.asarray(["test"], dtype=object))


def test_prediction_rejects_class_order_mutation_after_fit():
    cfg = _cfg()
    spec = vm.build_r1_model_spec(role="baseline", cfg=cfg)
    adapter = vm.make_r1_model_factory(cfg=cfg, model_spec=spec)(spec, _fold())
    texts, labels, groups = _train_rows()
    adapter.fit(texts, labels, groups=groups, inner_splits=())
    adapter._estimator.classes_ = np.asarray([1, 0], dtype=np.int64)

    with pytest.raises(vm.R1ModelAdapterError, match="P-order"):
        adapter.predict_proba(np.asarray(["test"], dtype=object))


def test_factory_is_bound_to_exact_model_spec_digest():
    cfg = _cfg()
    primary = vm.build_r1_model_spec(role="primary", cfg=cfg)
    baseline = vm.build_r1_model_spec(role="baseline", cfg=cfg)
    factory = vm.make_r1_model_factory(cfg=cfg, model_spec=primary)

    with pytest.raises(vm.R1ModelAdapterError, match="factory-bound"):
        factory(baseline, _fold())


def test_receipt_copy_cannot_hide_nested_mutation():
    cfg = _cfg()
    spec = vm.build_r1_model_spec(role="primary", cfg=cfg)
    receipt = vm.build_r1_model_adapter_receipt(spec, cfg=cfg)
    bad = copy.deepcopy(receipt)
    bad["outer_train_only"] = False
    bad["self_hash"] = canonical_sha256(
        {key: value for key, value in bad.items() if key != "self_hash"}
    )
    with pytest.raises(vm.R1ModelAdapterError, match="does not match"):
        vm.validate_r1_model_adapter_receipt(
            bad, model_spec=spec, cfg=cfg
        )
