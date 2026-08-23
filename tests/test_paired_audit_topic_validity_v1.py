from __future__ import annotations

import copy
import hashlib
import json
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
from stylo.eval.lobo import make_factory_for_ablation
from stylo.eval.paired_audit import evaluator_v3_2 as ev
from stylo.eval.paired_audit.applicability_v3_2 import resolve_cell_v3_2
from stylo.eval.paired_audit.runner import APPROVED_FREEZE_ROOT_SHA256
from stylo.eval.paired_audit import run_plan
from stylo.eval.paired_audit.topic_validity_v1 import (
    TOPIC_ARMS_V1,
    TOPIC_CELLS_V1,
    TopicValidityV1Error,
    build_topic_aggregate_v1,
    build_topic_study_context_v1,
    make_topic_challenger_factory_v1,
    validate_topic_aggregate_json_v1,
    validate_topic_aggregate_v1,
)
from stylo.jsonio import artifact_self_hash, dumps_strict


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _cfg(tmp_path) -> ConfigNode:
    raw = load_config().to_dict()
    raw["language"]["parse_n_process"] = 1
    raw["paths"]["data"] = str(tmp_path / "data")
    raw["paths"]["doc_cache"] = str(tmp_path / "doc_cache")
    raw["model"]["classifier"]["max_iter"] = 300
    return ConfigNode(raw)


def _dataset(cfg) -> Dataset:
    authors = ["a", "b", "c"]
    scaffold = "И в, на — но; потому что этот тот был очень! Затем снова? "
    texts, labels, groups, row_ids = [], [], [], []
    for label, author in enumerate(authors):
        noun = ("космос", "компас", "ландыш")[label]
        for work_index in range(2):
            work = f"{author}/w{work_index}"
            provenance = _sha(f"source:{work}")
            for ordinal in range(2):
                text = (scaffold + ((noun + ", ") * 12) + scaffold) * 3
                texts.append(text)
                labels.append(label)
                groups.append(work)
                row_ids.append(RowIdentity(
                    group=work, ordinal=ordinal, text_sha256=_sha(text), work_id=work,
                    provenance_sha256=provenance, chunker_config_hash="c" * 64,
                ))
    provenance = build_provenance(
        loader_kind=WORK_BALANCED_MANIFEST,
        texts=texts, y=labels, groups=groups, authors=authors, row_ids=row_ids,
        frags_root="/synthetic/topic-v1/frags",
        corpus_policy=CorpusPolicyProvenance.build((), "unknown"),
        chunker_config_hash="c" * 64,
        manifest_hash=_sha("topic-v1-manifest"),
    )
    return Dataset(
        texts=np.asarray(texts, dtype=object), y=np.asarray(labels, dtype=int),
        groups=np.asarray(groups, dtype=object), authors=authors, provenance=provenance,
    )


def _manifest(dataset, kind: str) -> dict:
    works = []
    for fold_index, work in enumerate(sorted(set(map(str, dataset.groups)))):
        works.append({
            "work_id": work, "author_id": work.split("/", 1)[0],
            "work_content_identity": _sha(f"work:{work}"),
            "content_component_identity": _sha(f"component:{work}"),
            "tested": True, "fold_index": fold_index,
        })
    return {
        "schema": f"synthetic_{kind}_topic_v1", "self_hash": _sha(f"fold:{kind}"),
        "selection_digest": _sha(f"selection:{kind}"),
        "probability_class_order": list(dataset.authors),
        "metric_label_order": list(dataset.authors), "works": works,
    }


def _context(tmp_path, *, tag="one"):
    cfg = _cfg(tmp_path)
    dataset = _dataset(cfg)
    lobo, ruaa = _manifest(dataset, "lobo"), _manifest(dataset, "ruaa")
    identity = ev._dataset_identity(dataset)
    values = dict(
        cfg=cfg, bundle_root=pathlib.Path("/synthetic/topic-v1"),
        candidate_identity=ev.CANDIDATE_IDENTITY,
        corrected_corpus_identity=ev.CORRECTED_CORPUS_IDENTITY,
        corpus_manifest_identity=ev.CORPUS_MANIFEST_IDENTITY,
        config_identity=ev._config_identity(cfg), protocol_identity=_sha(f"protocol:{tag}"),
        applicability_identity=ev.APPLICABILITY_V3_2_DIGEST,
        content_isolation_identity=_sha("isolation"), work_identity_catalog_identity=_sha("catalog"),
        lobo_manifest=lobo, ruaa_manifest=ruaa,
        lobo_dataset=dataset, ruaa_dataset=dataset,
        lobo_dataset_identity=identity, ruaa_dataset_identity=identity,
        ruaa_work_selection_identity=ruaa["selection_digest"],
        context_identity="", _seal=ev._CONTEXT_SEAL,
    )
    provisional = ev.V32EvaluationContext(**values)
    values["context_identity"] = ev.canonical_hash(ev._context_material(provisional))
    return cfg, ev.V32EvaluationContext(**values)


def _one_hot(width: int, label: int) -> list[float]:
    values = [0.0] * width
    values[label] = 1.0
    return values


def _records(study):
    predictions = {cell: {arm: [] for arm in TOPIC_ARMS_V1} for cell in TOPIC_CELLS_V1}
    for cell in TOPIC_CELLS_V1:
        for fold in study.folds:
            wrong = (fold.true_label + 1) % len(study.probability_order)
            other_wrong = (fold.true_label + 2) % len(study.probability_order)
            mode = fold.fold_index % 5
            current = fold.true_label if mode in (0, 1) else wrong
            strict = fold.true_label if mode in (0, 2) else (wrong if mode == 3 else other_wrong)
            for arm, prediction in (("current", current), ("topic_strict", strict)):
                predictions[cell][arm].append({
                    "fold_index": fold.fold_index,
                    "fold_identity": fold.fold_identity,
                    "whole_work_probabilities": _one_hot(len(study.probability_order), prediction),
                })
    return predictions


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def test_binding_and_fresh_factories_preserve_only_topic_strict_delta(tmp_path):
    pytest.importorskip("ru_core_news_lg")
    cfg, context = _context(tmp_path)
    study = build_topic_study_context_v1(cfg=cfg, context=context)
    again = build_topic_study_context_v1(cfg=cfg, context=context)
    assert study.binding == again.binding
    cfg_other, context_other = _context(tmp_path / "other", tag="other")
    other = build_topic_study_context_v1(cfg=cfg_other, context=context_other)
    assert study.binding["self_hash"] != other.binding["self_hash"]
    assert study.binding["status"] == "research_only_unexecuted"
    assert study.binding["confirmatory_authorized"] is False
    assert not ({"text", "work_id", "probabilities", "prediction", "receipt", "result", "run_id"}
                & set(_walk_keys(study.binding)))

    for cell in TOPIC_CELLS_V1:
        row = resolve_cell_v3_2("stylo", cell)
        direct = make_factory_for_ablation("stylo", cfg, ablation=row.ablation)()
        current_factory = make_topic_challenger_factory_v1(study=study, cell=cell, arm="current")
        strict_factory = make_topic_challenger_factory_v1(
            study=study, cell=cell, arm="topic_strict"
        )
        current, current_again = current_factory(), current_factory()
        strict, strict_again = strict_factory(), strict_factory()
        assert type(current) is type(direct) is type(strict)
        assert current is not current_again and strict is not strict_again
        assert current.named_steps["vectorizer"] is not current_again.named_steps["vectorizer"]
        assert strict.named_steps["vectorizer"] is not strict_again.named_steps["vectorizer"]
        for step in ("scaler", "classifier"):
            assert type(current.named_steps[step]) is type(strict.named_steps[step])
            assert current.named_steps[step].get_params() == strict.named_steps[step].get_params()
        current_fw = next(b for b in current.named_steps["vectorizer"].blocks if b.name == "function_words")
        strict_fw = next(b for b in strict.named_steps["vectorizer"].blocks if b.name == "function_words")
        strict_syntax = next(b for b in strict.named_steps["vectorizer"].blocks if b.name == "syntax")
        assert current_fw.mode == "mfw" and strict_fw.mode == "fixed_list"
        assert current_fw.relative_fw is strict_fw.relative_fw is None
        assert {"pos_ratios", "lexical_richness"}.isdisjoint(strict_syntax._active)

    for cell, arm in (("A1", "current"), ("A3", "topic_strict"), ("A0", "other")):
        with pytest.raises(TopicValidityV1Error):
            make_topic_challenger_factory_v1(study=study, cell=cell, arm=arm)
    class Text(str):
        pass
    with pytest.raises(TopicValidityV1Error):
        make_topic_challenger_factory_v1(study=study, cell=Text("A0"), arm="current")


def test_context_drift_rejects_before_factory(tmp_path, monkeypatch):
    cfg, context = _context(tmp_path)
    study = build_topic_study_context_v1(cfg=cfg, context=context)
    context.lobo_dataset.texts[0] = "mutated after seal"
    calls = []
    monkeypatch.setattr(
        "stylo.eval.paired_audit.topic_validity_v1.make_factory_for_ablation",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(TopicValidityV1Error, match="context"):
        make_topic_challenger_factory_v1(study=study, cell="A0", arm="current")
    assert calls == []

    cfg_fresh, context_fresh = _context(tmp_path / "binding")
    binding_study = build_topic_study_context_v1(cfg=cfg_fresh, context=context_fresh)
    binding_study.binding["status"] = "forged"
    with pytest.raises(TopicValidityV1Error, match="binding"):
        make_topic_challenger_factory_v1(study=binding_study, cell="A0", arm="current")


def test_aggregate_is_recomputed_and_contains_only_whitelisted_detail(tmp_path):
    cfg, context = _context(tmp_path)
    study = build_topic_study_context_v1(cfg=cfg, context=context)
    records = _records(study)
    kwargs = dict(
        study=study, records=records,
        implementation_source_identity="1" * 64, environment_lock_identity="2" * 64,
    )
    artifact = build_topic_aggregate_v1(**kwargs)
    validate_topic_aggregate_v1(artifact, **kwargs)
    loaded = validate_topic_aggregate_json_v1(dumps_strict(artifact, sort_keys=True), **kwargs)
    assert loaded == artifact
    assert artifact["status"] == "bounded_research_aggregate_only"
    assert artifact["confirmatory_authorized"] is False
    assert artifact["publication_authorized"] is False
    assert [row["cell"] for row in artifact["cells"]] == ["A0", "A4"]
    forbidden = {
        "text", "texts", "work_id", "work_ids", "probabilities", "whole_work_probabilities",
        "predictions", "correctness", "checkpoint", "receipt", "result", "run_id", "verdict",
        "headline", "p_value", "holm",
    }
    assert not (forbidden & set(_walk_keys(artifact)))
    for cell in artifact["cells"]:
        assert cell["delta_accuracy"]["direction"] == "topic_strict_minus_current"
        assert sum(row["n_folds"] for row in cell["per_author_transitions"]) == len(study.folds)
        for row in cell["per_author_transitions"]:
            assert row["n_folds"] == sum(
                row[key] for key in (
                    "both_correct", "current_only_correct", "topic_strict_only_correct",
                    "both_wrong_same_prediction", "both_wrong_changed_prediction",
                )
            )

    changed_records = copy.deepcopy(records)
    changed_records["A0"]["current"][0]["whole_work_probabilities"] = [0.0, 1.0, 0.0]
    changed_artifact = build_topic_aggregate_v1(
        study=study, records=changed_records,
        implementation_source_identity="1" * 64, environment_lock_identity="2" * 64,
    )
    with pytest.raises(TopicValidityV1Error, match="reconstructed"):
        validate_topic_aggregate_v1(changed_artifact, **kwargs)


@pytest.mark.parametrize("mutation", [
    "metric", "binding", "transition", "extra", "bool_alias", "cell_order", "self_hash",
])
def test_coherently_rehashed_aggregate_falsehoods_reject(tmp_path, mutation):
    cfg, context = _context(tmp_path)
    study = build_topic_study_context_v1(cfg=cfg, context=context)
    records = _records(study)
    kwargs = dict(
        study=study, records=records,
        implementation_source_identity="1" * 64, environment_lock_identity="2" * 64,
    )
    forged = copy.deepcopy(build_topic_aggregate_v1(**kwargs))
    if mutation == "metric":
        forged["cells"][0]["accuracy"]["current"]["correct"] += 1
    elif mutation == "binding":
        forged["bindings"]["config_identity"] = "0" * 64
        forged["study_identity"] = "0" * 64
    elif mutation == "transition":
        row = forged["cells"][0]["per_author_transitions"][0]
        row["both_correct"] += 1
        row["both_wrong_changed_prediction"] -= 1
    elif mutation == "extra":
        forged["work_ids"] = []
    elif mutation == "bool_alias":
        forged["design"]["fold_count"] = bool(forged["design"]["fold_count"])
    elif mutation == "cell_order":
        forged["cells"].reverse()
    elif mutation == "self_hash":
        forged["self_hash"] = "0" * 64
    if mutation != "self_hash":
        forged["self_hash"] = artifact_self_hash(forged)
    with pytest.raises(TopicValidityV1Error, match="reconstructed"):
        validate_topic_aggregate_v1(forged, **kwargs)


@pytest.mark.parametrize("mutation", [
    "drop", "extra_row", "duplicate", "shuffle", "fold_identity", "wrong_width",
    "int_probability", "nan", "extra_arm",
])
def test_transient_record_contract_rejects_misalignment_and_bad_numbers(tmp_path, mutation):
    cfg, context = _context(tmp_path)
    study = build_topic_study_context_v1(cfg=cfg, context=context)
    records = _records(study)
    rows = records["A0"]["current"]
    if mutation == "drop":
        rows.pop()
    elif mutation == "extra_row":
        rows.append(copy.deepcopy(rows[-1]))
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "shuffle":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "fold_identity":
        rows[0]["fold_identity"] = "0" * 64
    elif mutation == "wrong_width":
        rows[0]["whole_work_probabilities"].append(0.0)
    elif mutation == "int_probability":
        rows[0]["whole_work_probabilities"][0] = 0
    elif mutation == "nan":
        rows[0]["whole_work_probabilities"][0] = float("nan")
    elif mutation == "extra_arm":
        records["A0"]["extra"] = []
    with pytest.raises(TopicValidityV1Error):
        build_topic_aggregate_v1(
            study=study, records=records,
            implementation_source_identity="1" * 64,
            environment_lock_identity="2" * 64,
        )


def test_strict_json_and_official_gates_remain_closed(tmp_path):
    cfg, context = _context(tmp_path)
    study = build_topic_study_context_v1(cfg=cfg, context=context)
    kwargs = dict(
        study=study, records=_records(study),
        implementation_source_identity="1" * 64, environment_lock_identity="2" * 64,
    )
    for malformed in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        with pytest.raises(TopicValidityV1Error, match="strict JSON"):
            validate_topic_aggregate_json_v1(malformed, **kwargs)
    assert dict(run_plan.CONFIRMATORY_EVALUATOR_REGISTRY) == {}
    assert APPROVED_FREEZE_ROOT_SHA256 is None
