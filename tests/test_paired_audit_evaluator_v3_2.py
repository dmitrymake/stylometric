from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib

import numpy as np
import pytest

from stylo.config import ConfigNode, load_config
from stylo.corpus import Dataset
from stylo.domain.corpus_identity import (
    WORK_BALANCED_MANIFEST,
    CorpusPolicyProvenance,
    RowIdentity,
    build_provenance,
)
from stylo.domain.prediction_contract import stable_top1_and_worst_tie_rank
from stylo.eval.paired_audit import evaluator_v3_2 as ev
from stylo.eval.paired_audit.applicability_v3_2 import (
    APPLICABILITY_V3_2_DIGEST,
    APPLIED_CELLS,
    HOLM_FAMILY,
    REGISTRY_V3_2,
    V32ApplicabilityError,
    resolve_cell_v3_2,
)
from stylo.eval.paired_audit.corrected_v3_2 import (
    FROZEN_CORPUS_CHUNKER_CONFIG_SHA256,
    applicability_matrix,
)
from stylo.eval.paired_audit.evidence_v3_2 import (
    V32EvidenceError,
    validate_receipt_v3_2,
)
from stylo.models.baselines import MajorityBaseline

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _toy_cfg() -> ConfigNode:
    raw = load_config(ROOT / "configs/default.yaml").to_dict()
    for name in raw["features"]:
        raw["features"][name]["enabled"] = name == "function_words"
    raw["features"]["function_words"]["mfw_count"] = 12
    raw["model"]["classifier"]["max_iter"] = 300
    return ConfigNode(raw)


def _toy_dataset(cfg) -> Dataset:
    authors = ["a", "b", "c"]
    texts, labels, groups, row_ids = [], [], [], []
    chunk_hash = "c" * 64
    for label, author in enumerate(authors):
        for work_index in range(2):
            work = f"{author}/w{work_index}"
            provenance = _sha(f"source:{work}")
            for ordinal in range(2):
                # Every token occurs often enough for pooled min_df=3 and work min_df=2.
                text = (
                    f"common shared words and function tokens author{author} marker{author} "
                    f"work{work_index} chunk{ordinal} common shared words and tokens"
                )
                texts.append(text)
                labels.append(label)
                groups.append(work)
                row_ids.append(RowIdentity(
                    group=work, ordinal=ordinal, text_sha256=_sha(text), work_id=work,
                    provenance_sha256=provenance, chunker_config_hash=chunk_hash,
                ))
    provenance = build_provenance(
        loader_kind=WORK_BALANCED_MANIFEST,
        texts=texts, y=labels, groups=groups, authors=authors, row_ids=row_ids,
        frags_root="/synthetic/v32/frags",
        corpus_policy=CorpusPolicyProvenance.build((), "unknown"),
        chunker_config_hash=chunk_hash,
        manifest_hash=_sha("toy-manifest"),
    )
    return Dataset(
        texts=np.asarray(texts, dtype=object), y=np.asarray(labels, dtype=int),
        groups=np.asarray(groups, dtype=object), authors=authors, provenance=provenance,
    )


def _toy_manifest(dataset, kind: str) -> dict:
    works = []
    for fold_index, work in enumerate(sorted(set(map(str, dataset.groups)))):
        author = work.split("/", 1)[0]
        works.append({
            "work_id": work, "author_id": author,
            "work_content_identity": _sha(f"work:{work}"),
            "content_component_identity": _sha(f"component:{work}"),
            "tested": True, "fold_index": fold_index,
        })
    return {
        "schema": f"synthetic_{kind}_v3_2",
        "self_hash": _sha(f"fold:{kind}"),
        "selection_digest": _sha(f"selection:{kind}"),
        "probability_class_order": list(dataset.authors),
        "metric_label_order": list(dataset.authors),
        "works": works,
    }


def _toy_context():
    cfg = _toy_cfg()
    dataset = _toy_dataset(cfg)
    lobo = _toy_manifest(dataset, "lobo")
    ruaa = _toy_manifest(dataset, "ruaa")
    identity = ev._dataset_identity(dataset)
    values = dict(
        cfg=cfg, bundle_root=pathlib.Path("/synthetic/v32"),
        candidate_identity=ev.CANDIDATE_IDENTITY,
        corrected_corpus_identity=ev.CORRECTED_CORPUS_IDENTITY,
        corpus_manifest_identity=ev.CORPUS_MANIFEST_IDENTITY,
        config_identity=ev._config_identity(cfg), protocol_identity=_sha("protocol"),
        applicability_identity=APPLICABILITY_V3_2_DIGEST,
        content_isolation_identity=_sha("isolation"),
        work_identity_catalog_identity=_sha("catalog"),
        lobo_manifest=lobo, ruaa_manifest=ruaa,
        lobo_dataset=dataset, ruaa_dataset=dataset,
        lobo_dataset_identity=identity, ruaa_dataset_identity=identity,
        ruaa_work_selection_identity=ruaa["selection_digest"],
        context_identity="", _seal=ev._CONTEXT_SEAL,
    )
    provisional = ev.V32EvaluationContext(**values)
    values["context_identity"] = ev.canonical_hash(ev._context_material(provisional))
    return cfg, ev.V32EvaluationContext(**values)


def _evaluate_majority(cfg, context, **overrides):
    row = context.lobo_manifest["works"][0]
    kwargs = dict(
        cfg=cfg, context=context, dataset_kind="lobo", model="majority", cell="A0",
        fold_index=0, work_id=row["work_id"],
        work_content_identity=row["work_content_identity"],
        content_component_identity=row["content_component_identity"],
        probability_class_order=context.lobo_manifest["probability_class_order"],
        metric_label_order=context.lobo_manifest["metric_label_order"],
    )
    kwargs.update(overrides)
    return ev.evaluate_fold_v3_2(**kwargs)


def test_registry_is_exact_25_16_11_and_matches_corrected_preparation():
    assert len(REGISTRY_V3_2) == 25
    assert len(APPLIED_CELLS) == 16
    assert len(HOLM_FAMILY) == 11
    assert applicability_matrix()["digest"] == APPLICABILITY_V3_2_DIGEST
    assert tuple(
        (row["model"], row["cell"])
        for row in applicability_matrix()["applied_cells"]
    ) == APPLIED_CELLS


@pytest.mark.parametrize(
    ("model", "cell", "status"),
    [
        ("bow_lr", "A3", "not_applicable"),
        ("delta_cos:500", "A1", "already_in_legacy"),
        ("char_cos", "A1", "not_applicable"),
        ("char_cos", "A2", "equivalent_to"),
        ("char_cos", "A3", "not_applicable"),
        ("majority", "A4", "not_applicable"),
    ],
)
def test_metadata_only_cells_are_not_evaluable(model, cell, status):
    assert resolve_cell_v3_2(model, cell, require_applied=False).status == status
    with pytest.raises(V32ApplicabilityError, match="metadata-only"):
        resolve_cell_v3_2(model, cell)


def test_stylo_stack_withdrawn_before_any_factory(monkeypatch):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("factory reached")

    monkeypatch.setattr(ev, "make_factory_for_ablation", forbidden)
    cfg, context = _toy_context()
    with pytest.raises(V32ApplicabilityError, match="withdrawn"):
        _evaluate_majority(cfg, context, model="stylo_stack")
    assert called is False


@pytest.mark.parametrize(("model", "cell"), APPLIED_CELLS)
def test_all_16_applied_factory_routes_fit_and_predict_toy_data(monkeypatch, model, cell):
    class DummyRepCache:
        def get_reps(self, texts):
            return [None] * len(texts)

    monkeypatch.setattr("stylo.vectorizer.make_rep_cache", lambda cfg: DummyRepCache())
    cfg, context = _toy_context()
    receipt = _evaluate_majority(cfg, context, model=model, cell=cell)
    assert receipt["model"] == model and receipt["cell"] == cell
    assert receipt["test"]["n_rows"] == 2
    assert len(receipt["whole_work_probabilities"]) == 3
    validate_receipt_v3_2(receipt)


def test_repeat_evaluation_is_identical_and_factory_is_fresh(monkeypatch):
    cfg, context = _toy_context()
    created = []
    original = ev.make_factory_for_ablation

    def tracked(*args, **kwargs):
        factory = original(*args, **kwargs)
        def make():
            estimator = factory()
            created.append(estimator)
            return estimator
        return make

    monkeypatch.setattr(ev, "make_factory_for_ablation", tracked)
    first = _evaluate_majority(cfg, context)
    second = _evaluate_majority(cfg, context)
    assert first == second
    assert first["self_hash"] == second["self_hash"]
    assert len(created) == 2 and created[0] is not created[1]


def test_whole_held_out_work_never_reaches_fit(monkeypatch):
    cfg, context = _toy_context()
    observed = []
    original = ev.fit_estimator

    def inspect_fit(estimator, texts, y, groups):
        observed.extend(map(str, groups))
        return original(estimator, texts, y, groups)

    monkeypatch.setattr(ev, "fit_estimator", inspect_fit)
    receipt = _evaluate_majority(cfg, context)
    assert receipt["work_id"] not in observed
    assert receipt["test"]["n_rows"] == 2
    assert receipt["train"]["n_works"] == 5


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"cell": "A2"}, "metadata-only"),
        ({"fold_index": 1}, "fold index and full work ID"),
        ({"work_id": "a/nope"}, "fold index and full work ID"),
        ({"work_content_identity": "0" * 64}, "work/content identity"),
        ({"content_component_identity": "0" * 64}, "work/content identity"),
        ({"probability_class_order": ["c", "b", "a"]}, "probability class order"),
        ({"metric_label_order": ["c", "b", "a"]}, "metric class order"),
        ({"dataset_kind": "v3_1"}, "unknown v3.2 dataset"),
    ],
)
def test_wrong_cell_fold_identity_and_orders_reject_before_factory(monkeypatch, overrides, match):
    cfg, context = _toy_context()
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("factory reached")

    monkeypatch.setattr(ev, "make_factory_for_ablation", forbidden)
    with pytest.raises((ev.V32EvaluationError, V32ApplicabilityError), match=match):
        _evaluate_majority(cfg, context, **overrides)
    assert called is False


def test_historical_or_mutated_context_rejected_before_factory(monkeypatch):
    cfg, context = _toy_context()
    context.lobo_manifest["self_hash"] = _sha("v3.1-fold")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(ev, "make_factory_for_ablation", forbidden)
    with pytest.raises(ev.V32EvaluationError, match="context identity"):
        _evaluate_majority(cfg, context)
    assert called is False


def test_class_alignment_ties_and_worst_rank_are_coherent(monkeypatch):
    cfg, context = _toy_context()

    class ReversedTieMajority(MajorityBaseline):
        def fit(self, texts, y):
            self.classes_ = np.array([2, 1, 0])
            self._maj = 2
            return self

        def predict_proba(self, texts):
            return np.full((len(texts), 3), 1 / 3)

    monkeypatch.setattr(ev, "make_factory_for_ablation", lambda *a, **k: ReversedTieMajority)
    receipt = _evaluate_majority(cfg, context)
    assert [row["probability_column"] for row in receipt["class_alignment"]] == [2, 1, 0]
    assert receipt["whole_work_probabilities"] == [round(1 / 3, 12)] * 3
    assert receipt["vote"] == {
        "true_label": 0, "pred_label": 0, "pred_author": "a", "correct": True, "rank": 3,
    }
    assert stable_top1_and_worst_tie_rank(
        receipt["whole_work_probabilities"], true_label=0
    ).true_rank == 3


def test_incomplete_estimator_class_universe_rejects_before_predict(monkeypatch):
    cfg, context = _toy_context()
    predicted = False

    class PartialUniverseEstimator:
        def fit(self, texts, y):
            assert set(map(int, y)) == {0, 1, 2}
            self.classes_ = np.array([0, 1], dtype=np.int64)
            return self

        def predict_proba(self, texts):
            nonlocal predicted
            predicted = True
            raise AssertionError("incomplete class universe reached predict")

    monkeypatch.setattr(
        ev, "make_factory_for_ablation", lambda *a, **k: PartialUniverseEstimator
    )
    with pytest.raises(ev.V32EvaluationError, match="complete frozen probability class universe"):
        _evaluate_majority(cfg, context)
    assert predicted is False


def test_probability_and_evidence_mutation_rotate_or_reject():
    cfg, context = _toy_context()
    receipt = _evaluate_majority(cfg, context)
    changed = copy.deepcopy(receipt)
    changed["whole_work_probabilities"][0] = 0.5
    with pytest.raises(V32EvidenceError, match="self-hash"):
        validate_receipt_v3_2(changed)
    coherent_outer = copy.deepcopy(receipt)
    coherent_outer["actual_fitted_state"]["majority_class"] = 2
    coherent_outer["self_hash"] = ev.canonical_hash(
        {key: value for key, value in coherent_outer.items() if key != "self_hash"}
    )
    with pytest.raises(V32EvidenceError, match="fitted-state"):
        validate_receipt_v3_2(coherent_outer)


def test_receipt_binds_train_derived_and_actual_fitted_evidence_without_repr_pickle():
    cfg, context = _toy_context()
    receipt = _evaluate_majority(cfg, context)
    assert receipt["axis_evidence"]["W"]["source"] == "train_inputs_canonical_derivation"
    assert receipt["axis_evidence"]["majority"]["source"] == "train_inputs_canonical_derivation"
    assert receipt["actual_fitted_state"]["schema"] == "paired_audit.actual_fitted_state.v3_2"
    encoded = json.dumps(receipt, sort_keys=True)
    assert "pickle" not in encoded and "repr" not in encoded
    validate_receipt_v3_2(receipt)


def _real_selection() -> list[str]:
    raw = json.loads((ROOT / "data/ruaa_bench_v1/manifest.json").read_text())
    return sorted(
        f"{author}/{book['book']}"
        for author, record in raw["authors"].items()
        for book in record["books"]
    )


def test_two_real_context_loads_are_identical_and_never_fit_predict(monkeypatch):
    bundle_raw = os.environ.get("STYLO_V32_BUNDLE_ROOT")
    if not bundle_raw:
        pytest.skip("set STYLO_V32_BUNDLE_ROOT to the exact verified local v3.2 bundle")
    bundle = pathlib.Path(bundle_raw)
    parent = ROOT / "data/audit_corpus/15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029"

    def forbidden(*args, **kwargs):
        raise AssertionError("real context construction reached fit/predict")

    monkeypatch.setattr(ev, "fit_estimator", forbidden)
    cfg1 = load_config(ROOT / "configs/default.yaml")
    cfg2 = load_config(ROOT / "configs/default.yaml")
    first = ev.build_evaluation_context_v3_2(
        cfg=cfg1, bundle_root=bundle, historical_parent_root=parent,
        ruaa_parent_selection=_real_selection(),
    )
    second = ev.build_evaluation_context_v3_2(
        cfg=cfg2, bundle_root=bundle, historical_parent_root=parent,
        ruaa_parent_selection=_real_selection(),
    )
    assert first.context_identity == second.context_identity
    assert first.context_identity == "2805aff91988173561be783c92c048d0276b9e0caeeef42318fe354d05680b81"
    assert (len(first.lobo_dataset.authors), len(set(first.lobo_dataset.groups))) == (47, 252)
    assert (len(first.ruaa_dataset.authors), len(set(first.ruaa_dataset.groups))) == (22, 134)
    assert first.lobo_dataset.provenance.chunker_config_hash == FROZEN_CORPUS_CHUNKER_CONFIG_SHA256
    assert first.ruaa_work_selection_identity != first.ruaa_dataset_identity["row_selection_identity"]


def test_candidate_has_no_legacy_runtime_reachability_and_registry_stays_empty():
    source = (ROOT / "src/stylo/eval/paired_audit/evaluator_v3_2.py").read_text()
    for forbidden in (
        "paired_audit.runner", "paired_audit.run_plan", "paired_audit.checkpoints",
        "paired_audit.inference", "paired_audit.publisher", "paired_audit.references",
    ):
        assert forbidden not in source
    from stylo.eval.paired_audit.run_plan import CONFIRMATORY_EVALUATOR_REGISTRY
    assert dict(CONFIRMATORY_EVALUATOR_REGISTRY) == {}
