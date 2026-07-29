"""Adversarial regression coverage for review findings AUD-027..AUD-034."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import pathlib
import pickle

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from stylo.eval.metric_contract import MetricContractError


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _module_imports(module_name: str, path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module_name.split(".")[:-1]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                prefix = package[:keep]
                target = ".".join(
                    [*prefix, *(node.module or "").split(".")]
                ).rstrip(".")
            else:
                target = node.module or ""
            if target:
                imports.add(target)
    return imports


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise AssertionError(f"import cycle through {node}")
        if node in visited:
            return
        active.add(node)
        for child in graph[node]:
            visit(child)
        active.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def test_aud027_identity_contract_is_inward_and_import_graph_is_acyclic():
    from stylo.domain import corpus_identity
    from stylo.domain import work_weighting as domain_weighting
    from stylo.domain.segmentation import LabeledSpan as DomainSpan
    from stylo.eval import provenance
    from stylo.eval import work_weighting as compatibility_weighting
    from stylo.eval.segmentation import LabeledSpan as EvalSpan
    import stylo.eval.calibration as compatibility_calibration
    import stylo.models.calibration as model_calibration

    assert provenance.RowIdentity is corpus_identity.RowIdentity
    assert provenance.DatasetProvenance is corpus_identity.DatasetProvenance
    assert provenance.build_provenance is corpus_identity.build_provenance
    assert compatibility_weighting is domain_weighting
    assert compatibility_calibration is model_calibration
    assert EvalSpan is DomainSpan

    paths = {
        "stylo.corpus": ROOT / "src/stylo/corpus.py",
        "stylo.workdoc": ROOT / "src/stylo/workdoc.py",
        "stylo.eval.provenance": ROOT / "src/stylo/eval/provenance.py",
        "stylo.domain.corpus_identity": (
            ROOT / "src/stylo/domain/corpus_identity.py"
        ),
    }
    raw = {
        module: _module_imports(module, path)
        for module, path in paths.items()
    }
    assert "stylo.eval.provenance" not in raw["stylo.corpus"]
    assert "stylo.eval.provenance" not in raw["stylo.workdoc"]
    inward_paths = {
        "stylo.corpus": ROOT / "src/stylo/corpus.py",
        "stylo.workdoc": ROOT / "src/stylo/workdoc.py",
        **{
            "stylo.models." + path.relative_to(
                ROOT / "src/stylo/models"
            ).with_suffix("").as_posix().replace("/", "."): path
            for path in (ROOT / "src/stylo/models").rglob("*.py")
        },
    }
    for module, path in inward_paths.items():
        forbidden = {
            target
            for target in _module_imports(module, path)
            if target == "stylo.eval" or target.startswith("stylo.eval.")
        }
        assert not forbidden, f"{module} imports outward evaluation modules: {forbidden}"
    graph = {
        module: {target for target in imports if target in paths}
        for module, imports in raw.items()
    }
    _assert_acyclic(graph)


def test_aud029_extracted_contract_facade_preserves_identity_bytes():
    from stylo.domain.corpus_identity import (
        LEGACY_RECURSIVE,
        CorpusPolicyProvenance,
        RowIdentity,
        build_provenance,
    )
    from stylo.eval import provenance as compatibility
    from stylo.jsonio import dumps_strict

    texts = ["alpha", "beta"]
    groups = ["a/w1", "b/w1"]
    rows = [
        RowIdentity(
            group=group,
            ordinal=0,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
        for text, group in zip(texts, groups, strict=True)
    ]
    kwargs = {
        "loader_kind": LEGACY_RECURSIVE,
        "texts": texts,
        "y": [0, 1],
        "groups": groups,
        "authors": ["a", "b"],
        "row_ids": rows,
        "frags_root": "/frozen",
        "corpus_policy": CorpusPolicyProvenance.build([], "unknown"),
    }
    inward = build_provenance(**kwargs)
    facade = compatibility.build_provenance(**kwargs)
    assert type(inward) is type(facade)
    assert dumps_strict(dataclasses.asdict(inward), sort_keys=True) == dumps_strict(
        dataclasses.asdict(facade),
        sort_keys=True,
    )
    assert inward.rows_digest == facade.rows_digest


def test_aud028_one_registry_generates_all_routing_views():
    from stylo.eval.final import DEFAULT_SPECS, ECE_SPECS
    from stylo.eval.paired_audit.applicability import MODELS
    from stylo.models.registry import (
        CALIBRATION_MODEL_SPECS,
        CONFIRMATORY_MODEL_SPECS,
        DEFAULT_EXPLORATORY_SPECS,
        MODEL_REGISTRY,
        ModelRegistryError,
        public_model_help,
        resolve_model_spec,
    )

    assert tuple(DEFAULT_SPECS) == DEFAULT_EXPLORATORY_SPECS
    assert tuple(MODELS) == CONFIRMATORY_MODEL_SPECS
    assert frozenset(ECE_SPECS) == CALIBRATION_MODEL_SPECS
    all_specs = {
        *DEFAULT_EXPLORATORY_SPECS,
        *CONFIRMATORY_MODEL_SPECS,
        "stylo_equal_channels_v1",
        "bow_lr_ref_legacy",
    }
    assert all(resolve_model_spec(spec).key in MODEL_REGISTRY for spec in all_specs)
    assert "stylo_equal_channels_v1" not in DEFAULT_EXPLORATORY_SPECS
    assert "stylo_equal_channels_v1" not in CONFIRMATORY_MODEL_SPECS
    assert "stylo_equal_channels_v1" in public_model_help()
    with pytest.raises(TypeError):
        MODEL_REGISTRY["forged"] = MODEL_REGISTRY["stylo"]
    for malformed in ("delta:0", "delta:-1", "delta:1.0", "delta:", True):
        with pytest.raises(ModelRegistryError):
            resolve_model_spec(malformed)


def test_aud030_lazy_estimators_block_deployment_and_serialization():
    from stylo.models.equal_channel_ensemble import (
        EqualChannelEnsembleClassifier,
    )
    from stylo.models.registry import ModelRegistryError, assert_model_route
    from stylo.models.stacked_clf import (
        EvaluationOnlyEstimatorError,
        StackedChannelClassifier,
    )

    for spec, estimator in (
        ("stylo_stack", StackedChannelClassifier(object())),
        (
            "stylo_equal_channels_v1",
            EqualChannelEnsembleClassifier(object()),
        ),
    ):
        assert estimator.evaluation_only is True
        assert estimator.lazy_final_fit is True
        assert estimator.deployment_supported is False
        with pytest.raises(EvaluationOnlyEstimatorError):
            pickle.dumps(estimator)
        with pytest.raises(ModelRegistryError):
            assert_model_route(
                spec,
                weighting="chunk_weighted_legacy",
                deployment=True,
            )
        with pytest.raises(ModelRegistryError):
            assert_model_route(
                spec,
                weighting="chunk_weighted_legacy",
                serialization=True,
            )


def test_aud030_equal_lazy_contract_has_repeat_batch_parity(monkeypatch):
    import stylo.models.stacked_clf as stack_module
    from stylo.models.equal_channel_ensemble import (
        EqualChannelEnsembleClassifier,
    )

    channel_names = [f"channel_{index}" for index in range(6)]
    channel_calls: list[str] = []
    fit_calls: list[tuple[int, ...]] = []

    def make_channel(name: str, offset: int):
        def channel(train_texts, test_texts):
            channel_calls.append(name)
            train = np.asarray(
                [[len(text) + offset] for text in train_texts],
                dtype=float,
            )
            test = np.asarray(
                [[len(text) + offset] for text in test_texts],
                dtype=float,
            )
            return train, test

        return channel

    channels = {
        name: make_channel(name, index)
        for index, name in enumerate(channel_names)
    }
    monkeypatch.setattr(
        stack_module,
        "make_channels",
        lambda _cfg: channels,
    )

    class DummySVC:
        def fit(self, _features, labels, sample_weight=None):
            assert sample_weight is None
            self.classes_ = np.unique(labels)
            fit_calls.append(tuple(int(label) for label in labels))
            return self

        def decision_function(self, features):
            return np.asarray(features)[:, 0] / 10.0

    estimator = EqualChannelEnsembleClassifier(object())
    monkeypatch.setattr(estimator, "_svc", lambda: DummySVC())
    estimator.fit(
        ["aa", "bbb", "cccc", "ddddd"],
        np.asarray([0, 0, 1, 1]),
        np.asarray(["a/w1", "a/w2", "b/w1", "b/w2"]),
    )
    batch = estimator.predict_proba(["xx", "yyyy"])
    repeated = estimator.predict_proba(["xx", "yyyy"])
    singles = np.vstack(
        [
            estimator.predict_proba(["xx"]),
            estimator.predict_proba(["yyyy"]),
        ]
    )
    np.testing.assert_allclose(batch, repeated, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch, singles, rtol=0.0, atol=0.0)
    assert estimator.fusion_passport_["channels"] == channel_names
    assert channel_calls == channel_names * 4
    assert len(fit_calls) == 6 * 4


def test_aud031_summary_uses_frozen_author_order_not_prediction_union():
    from stylo.eval.metrics import (
        AuthorClusteredInferenceSpec,
        summarize_book_results,
    )

    inference_spec = AuthorClusteredInferenceSpec.build(
        iterations=20,
        confidence_level=0.95,
        seed=42,
    )

    summary = summarize_book_results(
        np.asarray([0, 0]),
        np.asarray([0, 0]),
        np.asarray([1, 1]),
        probability_class_order=["a", "b"],
        metric_label_order=[0, 1],
        book_authors=["a", "a"],
        inference_spec=inference_spec,
    )
    assert summary["macro_f1"].point == pytest.approx(0.5)
    with pytest.raises(MetricContractError):
        summarize_book_results(
            np.asarray([0, 1]),
            np.asarray([0, 2]),
            np.asarray([1, 2]),
            probability_class_order=["a", "b"],
            metric_label_order=[0, 1],
            book_authors=["a", "b"],
            inference_spec=inference_spec,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: __import__(
            "stylo.eval.metrics",
            fromlist=["accuracy"],
        ).accuracy(np.asarray([[0], [1]]), np.asarray([0, 1])),
        lambda: __import__(
            "stylo.eval.metrics",
            fromlist=["accuracy"],
        ).accuracy(np.asarray([0]), np.asarray([0, 1])),
        lambda: __import__(
            "stylo.eval.metrics",
            fromlist=["topk_accuracy"],
        ).topk_accuracy(np.asarray([1, 2]), True),
        lambda: __import__(
            "stylo.eval.metrics",
            fromlist=["bootstrap_ci"],
        ).bootstrap_ci(lambda index: 1.0, 2, iters=0),
        lambda: __import__(
            "stylo.eval.metrics",
            fromlist=["bootstrap_ci"],
        ).bootstrap_ci(lambda index: 1.0, 2, level=float("nan")),
        lambda: __import__(
            "stylo.eval.metrics",
            fromlist=["expected_calibration_error"],
        ).expected_calibration_error(
            np.asarray([[float("nan"), 0.0]]),
            np.asarray([0]),
        ),
        lambda: __import__(
            "stylo.eval.metrics",
            fromlist=["expected_calibration_error"],
        ).expected_calibration_error(
            np.asarray([[0.5, 0.5]]),
            np.asarray([0]),
            n_bins=False,
        ),
        lambda: __import__(
            "stylo.eval.significance",
            fromlist=["mcnemar"],
        ).mcnemar(np.asarray([1, 0]), np.asarray([True, False])),
        lambda: __import__(
            "stylo.eval.significance",
            fromlist=["mcnemar"],
        ).mcnemar(
            np.asarray([[True], [False]]),
            np.asarray([True, False]),
        ),
        lambda: __import__(
            "stylo.eval.significance",
            fromlist=["paired_bootstrap_diff"],
        ).paired_bootstrap_diff(
            lambda index: float("inf"),
            lambda index: 0.0,
            2,
            iters=2,
        ),
        lambda: __import__(
            "stylo.eval.significance",
            fromlist=["paired_bootstrap_diff_clustered"],
        ).paired_bootstrap_diff_clustered(
            lambda index: 1.0,
            lambda index: 0.0,
            np.asarray([["a"], ["b"]]),
            iters=2,
        ),
    ],
)
def test_aud032_metrics_and_significance_reject_malformed_inputs(call):
    with pytest.raises((MetricContractError, TypeError)):
        call()


def test_aud033_biased_stack_selection_is_withdrawn_and_blocked():
    from stylo.models.registry import (
        MODEL_REGISTRY,
        ModelRegistryError,
        assert_model_route,
    )
    from stylo.models.stacked_clf import (
        STACK_SELECTION_EVIDENCE_STATUS,
        withdrawn_internal_selection_diagnostic,
    )

    registration = MODEL_REGISTRY["stylo_stack"]
    assert registration.internal_selection_evidence is False
    assert registration.confirmatory_execution_eligible is False
    assert registration.scientific_status == (
        "withdrawn_pending_nested_group_calibration"
    )
    diagnostic = withdrawn_internal_selection_diagnostic(0.8, 0.7)
    assert diagnostic["status"] == STACK_SELECTION_EVIDENCE_STATUS
    assert diagnostic["eligible_as_unbiased_evidence"] is False
    assert set(diagnostic) == {
        "status",
        "eligible_as_unbiased_evidence",
        "descriptive_only",
    }
    with pytest.raises(ModelRegistryError):
        assert_model_route(
            "stylo_stack",
            weighting="work_balanced",
            confirmatory=True,
        )
    with pytest.raises(ValueError):
        withdrawn_internal_selection_diagnostic(float("nan"), 0.7)


def test_aud034_delta_identifier_is_explicitly_selected_mass():
    from stylo.models.delta import BurrowsDelta
    from stylo.models.registry import MODEL_REGISTRY

    estimator = BurrowsDelta(vocabulary=["a"])
    estimator._vec = CountVectorizer(
        vocabulary=["a"],
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    ).fit(["a b b"])
    selected_mass = estimator._rel_freq(["a b b"])[0, 0]
    canonical_all_token_frequency = 1.0 / 3.0
    assert selected_mass == pytest.approx(1.0)
    assert selected_mass != pytest.approx(canonical_all_token_frequency)
    assert estimator.FREQUENCY_DENOMINATOR == "sum_selected_mfw_counts"
    assert estimator.PUBLIC_DISPLAY_NAME == "Frozen legacy selected-mass Delta"
    assert "selected-mass" in MODEL_REGISTRY["delta"].description
    # the documented surface is the protocol, not the entry page: README no longer
    # publishes the baseline method comparison the identifier belonged to
    protocol = (
        ROOT / "research" / "work_balanced" / "paired_audit_protocol.md"
    ).read_text(encoding="utf-8")
    assert "legacy selected-mass Delta" in protocol
    assert "not canonical Burrows's Delta" in protocol
