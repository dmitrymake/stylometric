"""Focused anti-false-green gates for the exploratory W/F/R ablation screen.

The synthetic panel is deliberately tiny (two authors, two works per author, two chunks per work),
but it exercises the real frozen-panel evaluator: every learned estimator is fold-local, chunk
probabilities are aligned and averaged exactly once per held-out work, and every applied screen cell
must preserve the same ordered work/fold inventory. The screen remains an exploratory
artifact, never a confirmatory result or an authorship proof.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stylo.corpus import Dataset
from stylo.eval import groupkfold
from stylo.eval import screening_panel as sp
from stylo.jsonio import artifact_self_hash, canonical_hash, dump_strict, load_strict


class _Cfg:
    """Only the config path needed by the frozen-panel evaluator."""

    def __init__(self, marker="cfg-v1"):
        self.marker = marker

    def get_path(self, path, default=None):
        return {"evaluation.top_k_candidates": 2}.get(path, default)


_CHUNK_PROBS = {
    "aa/w1:0": (0.9, 0.1),
    "aa/w1:1": (0.5, 0.5),
    "aa/w2:0": (0.2, 0.8),
    "aa/w2:1": (0.4, 0.6),
    "bb/w1:0": (0.6, 0.4),
    "bb/w1:1": (0.2, 0.8),
    "bb/w2:0": (0.8, 0.2),
    "bb/w2:1": (0.4, 0.6),
}


@pytest.fixture
def tiny_panel():
    # Deliberately non-work-sorted rows: result order must still follow sorted work ids.
    texts = np.array([
        "bb/w2:1", "aa/w1:0", "bb/w1:1", "aa/w2:0",
        "aa/w1:1", "bb/w2:0", "aa/w2:1", "bb/w1:0",
    ], dtype=object)
    groups = np.array([str(t).split(":", 1)[0] for t in texts], dtype=object)
    y = np.array([0 if str(g).startswith("aa/") else 1 for g in groups], dtype=int)
    dataset = Dataset(
        texts=texts,
        y=y,
        groups=groups,
        authors=["aa", "bb"],
        # build_manifest needs only this disk-anchor field; the evaluator itself never trusts or
        # re-verifies provenance because its caller has already bound the frozen panel.
        provenance=SimpleNamespace(rows_digest="1" * 64),
    )
    manifest = sp.build_manifest(dataset, k=2, seed=42)
    sp.verify_manifest(manifest)
    return _Cfg(), dataset, manifest


class _SpyEstimator:
    needs_groups = True
    classes_ = np.arange(2, dtype=int)

    def __init__(self, instances):
        self.instance_id = len(instances)
        self.fit_texts = None
        self.fit_groups = None
        self.predict_texts = None
        instances.append(self)

    def fit(self, texts, y, *, groups):
        self.fit_texts = tuple(str(t) for t in texts)
        self.fit_groups = tuple(str(g) for g in groups)
        self.fit_labels = tuple(int(v) for v in y)
        return self

    def predict_proba(self, texts):
        assert self.fit_groups is not None, "predict must use this fold's fitted estimator"
        self.predict_texts = tuple(str(t) for t in texts)
        predicted_works = {t.split(":", 1)[0] for t in self.predict_texts}
        assert predicted_works.isdisjoint(self.fit_groups), "held-out work leaked into fitted state"
        return np.asarray([_CHUNK_PROBS[t] for t in self.predict_texts], dtype=np.float64)


def _spy_factory():
    instances = []

    def factory():
        return _SpyEstimator(instances)

    return factory, instances


class _TickClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        out = self.value
        self.value += 1.0
        return out


def _work_ids(df):
    return [f"{r.test_author}/{r.test_book}" for r in df.itertuples()]


def test_frozen_factory_is_fold_local_and_averages_chunks_exactly_once(tiny_panel):
    cfg, dataset, manifest = tiny_panel
    factory, instances = _spy_factory()
    df, probs, y_true, timing = groupkfold.evaluate_frozen_panel_factory(
        cfg, dataset, factory, manifest, spec="spy", clock=_TickClock())

    ordered_works = ["aa/w1", "aa/w2", "bb/w1", "bb/w2"]
    assert _work_ids(df) == ordered_works
    assert y_true.tolist() == [0, 0, 1, 1]
    expected = np.array([[0.7, 0.3], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]])
    np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-15)
    assert df["pred_label"].tolist() == [0, 1, 1, 0]
    assert df["correct"].tolist() == [True, False, True, False]
    assert df["rank"].tolist() == [1, 2, 1, 2]

    # One genuinely fresh estimator per fold; its fitted state is exactly the complementary works.
    assert len(instances) == manifest["k_folds"] == 2
    assert len({id(est) for est in instances}) == 2
    work_fold = {w["work_id"]: w["fold"] for w in manifest["works"]}
    for est in instances:
        test_works = {t.split(":", 1)[0] for t in est.predict_texts}
        fit_works = set(est.fit_groups)
        assert test_works.isdisjoint(fit_works)
        fold = {work_fold[w] for w in test_works}
        assert len(fold) == 1
        f = fold.pop()
        assert fit_works == {w for w in ordered_works if work_fold[w] != f}
        assert test_works == {w for w in ordered_works if work_fold[w] == f}

    assert [r["fold"] for r in timing["folds"]] == [0, 1]
    assert [(r["n_train_chunks"], r["n_test_chunks"], r["n_test_works"])
            for r in timing["folds"]] == [(4, 4, 2), (4, 4, 2)]
    assert [(r["fit_seconds"], r["predict_seconds"]) for r in timing["folds"]] == [(1.0, 1.0)] * 2
    assert timing["fit_seconds"] == 2.0 and timing["predict_seconds"] == 2.0
    assert timing["total_seconds"] == 9.0


def test_frozen_gkf_uses_stable_top1_and_worst_tie_rank(tiny_panel):
    cfg, dataset, manifest = tiny_panel

    class UniformEstimator:
        needs_groups = True
        classes_ = np.arange(2, dtype=int)

        def fit(self, texts, y, *, groups):
            return self

        def predict_proba(self, texts):
            return np.full((len(texts), 2), 0.5, dtype=np.float64)

    df, probabilities, _y_true, _timing = (
        groupkfold.evaluate_frozen_panel_factory(
            cfg, dataset, UniformEstimator, manifest, spec="uniform",
            clock=_TickClock(),
        )
    )
    np.testing.assert_allclose(probabilities, 0.5)
    assert df["pred_label"].tolist() == [0, 0, 0, 0]
    assert df["correct"].tolist() == [True, True, False, False]
    assert df["rank"].tolist() == [2, 2, 2, 2]


def test_injected_A0_helper_exactly_matches_existing_panel_worker(tiny_panel, monkeypatch):
    cfg, dataset, manifest = tiny_panel
    direct_factory, _ = _spy_factory()
    direct = groupkfold.evaluate_frozen_panel_factory(
        cfg, dataset, direct_factory, manifest, spec="majority", clock=_TickClock())[:3]

    old_factory, _ = _spy_factory()
    monkeypatch.setattr(groupkfold, "make_factory", lambda *args, **kwargs: old_factory)
    wrapped = groupkfold._gkf_run_panel(
        cfg, dataset, "majority", None, "chunk_weighted_legacy", manifest)

    pd.testing.assert_frame_equal(wrapped[0], direct[0], check_exact=True)
    np.testing.assert_array_equal(wrapped[1], direct[1])
    np.testing.assert_array_equal(wrapped[2], direct[2])


def _controlled_evaluator(calls):
    """A deterministic evaluator that records only genuinely applied cells.

    ``run_ablation_screen`` still has to construct the requested factory first, so typed applicability is
    raised on the real routing path.  For applied rows we substitute the cheap spy estimator while
    retaining the real frozen-fold/aggregation implementation.
    """
    def evaluate(cfg, dataset, factory, manifest, *, spec="injected", clock=None):
        calls.append(spec)
        spy_factory, _ = _spy_factory()
        return groupkfold.evaluate_frozen_panel_factory(
            cfg, dataset, spy_factory, manifest, spec=spec, clock=_TickClock())
    return evaluate


def _screen_kwargs(config_sha256="c" * 64):
    return {
        "config_path": "synthetic-config.yaml",
        "config_sha256": config_sha256,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "code_hashes": {"src/stylo/eval/work_balanced_ablation_screen.py": "b" * 64},
        "runtime_fingerprint": {
            "python": "test-python",
            "numpy": "test-numpy",
        },
        "seed": 42,
        "bootstrap_iters": 20,
        "ci_level": 0.95,
        "clock": lambda: 0.0,
        "continue_on_error": False,
    }


def _cell(artifact, model, cell):
    matches = [r for r in artifact["cells"] if r["model"] == model and r["cell"] == cell]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    "model,cell,status,field,value",
    [
        ("bow_lr", "A3", "not_applicable", "reason", "not_applicable"),
        ("char_cos", "A2", "equivalent", "equivalent_to", "A4"),
    ],
)
def test_runner_records_typed_applicability_without_fake_metrics(
        tiny_panel, tmp_path, model, cell, status, field, value):
    from stylo.eval import work_balanced_ablation_screen as ablation_screen

    cfg, dataset, manifest = tiny_panel
    calls = []
    artifact = ablation_screen.run_ablation_screen(
        cfg, dataset, manifest, tmp_path / f"{model}-{cell}.json",
        models=[model], cells=[cell], evaluator=_controlled_evaluator(calls), **_screen_kwargs())

    # Every requested comparison has an A0 dependency; only A0 reaches evaluation.  The typed row is
    # resolved before evaluator invocation and must not copy A0 metrics, works, probabilities or time.
    assert len(calls) == 1
    assert _cell(artifact, model, "A0")["status"] == "applied"
    typed = _cell(artifact, model, cell)
    assert typed["status"] == status and typed[field] == value
    for forbidden in ("metrics", "works", "probabilities", "folds", "timing"):
        assert forbidden not in typed


def test_resume_is_byte_stable_and_rejects_changed_inputs_before_eval(tiny_panel, tmp_path):
    from stylo.eval import work_balanced_ablation_screen as ablation_screen

    cfg, dataset, manifest = tiny_panel
    output = tmp_path / "resume.json"
    calls = []
    kwargs = _screen_kwargs()
    first = ablation_screen.run_ablation_screen(
        cfg, dataset, manifest, output, models=["majority", "stylo"], cells=["A0"],
        evaluator=_controlled_evaluator(calls), **kwargs)
    first_bytes = output.read_bytes()
    assert len(calls) == 2

    # Both completed cells resume without constructing/evaluating them again; canonical bytes and
    # self-hash remain unchanged.
    second = ablation_screen.run_ablation_screen(
        cfg, dataset, manifest, output, models=["majority", "stylo"], cells=["A0"],
        evaluator=_controlled_evaluator(calls), **kwargs)
    assert len(calls) == 2
    assert second == first and output.read_bytes() == first_bytes
    assert second["self_hash"] == artifact_self_hash(second)

    # A resume belongs to one exact config and one exact frozen panel.  Both mismatches fail before
    # even the injected evaluator sees a cell.
    before = list(calls)
    with pytest.raises(ValueError):
        ablation_screen.run_ablation_screen(
            cfg, dataset, manifest, output, models=["majority", "stylo"], cells=["A0"],
            evaluator=_controlled_evaluator(calls), **_screen_kwargs(config_sha256="d" * 64))
    assert calls == before
    other_manifest = copy.deepcopy(manifest)
    other_manifest["parent_dataset_digest"] = "2" * 64
    other_manifest["self_hash"] = sp._self_hash(other_manifest)
    sp.verify_manifest(other_manifest)
    with pytest.raises(ValueError):
        ablation_screen.run_ablation_screen(
            cfg, dataset, other_manifest, output, models=["majority", "stylo"], cells=["A0"],
            evaluator=_controlled_evaluator(calls), **kwargs)
    assert calls == before


def test_resume_rejects_rehashed_semantic_forgery_before_triage_or_eval(
        tiny_panel, tmp_path):
    """A valid self-hash cannot authorize incoherent scientific fields."""
    from stylo.eval import work_balanced_ablation_screen as ablation_screen

    cfg, dataset, manifest = tiny_panel
    original_path = tmp_path / "original.json"
    original = ablation_screen.run_ablation_screen(
        cfg,
        dataset,
        manifest,
        original_path,
        models=["majority"],
        cells=["A0"],
        evaluator=_controlled_evaluator([]),
        **_screen_kwargs(),
    )

    mutations = {}

    forged = copy.deepcopy(original)
    forged["cells"][0]["works"][0]["probabilities"] = [-0.1, 1.1]
    mutations["range"] = forged

    forged = copy.deepcopy(original)
    row = forged["cells"][0]["works"][0]
    row["pred_label"] = 1 - row["pred_label"]
    row["pred_author"] = forged["authors"][row["pred_label"]]
    row["correct"] = row["pred_label"] == row["true_label"]
    mutations["prediction"] = forged

    forged = copy.deepcopy(original)
    row = forged["cells"][0]["works"][0]
    row["rank"] = 1 if row["rank"] != 1 else 2
    mutations["rank"] = forged

    forged = copy.deepcopy(original)
    forged["cells"][0]["metrics"]["accuracy"] += 0.25
    mutations["metric"] = forged

    forged = copy.deepcopy(original)
    forged["triage"]["label"] = "promising_directional_signal"
    mutations["triage"] = forged

    for name, artifact in mutations.items():
        artifact["self_hash"] = artifact_self_hash(artifact)
        path = tmp_path / f"forged-{name}.json"
        dump_strict(artifact, path)
        calls = []
        with pytest.raises(ablation_screen.AblationScreenArtifactError):
            ablation_screen.run_ablation_screen(
                cfg,
                dataset,
                manifest,
                path,
                models=["majority"],
                cells=["A0"],
                evaluator=_controlled_evaluator(calls),
                **_screen_kwargs(),
            )
        assert calls == []


def test_artifact_hash_load_and_format_fail_closed_on_tamper(tiny_panel, tmp_path):
    from stylo.eval import work_balanced_ablation_screen as ablation_screen

    cfg, dataset, manifest = tiny_panel
    output = tmp_path / "artifact.json"
    artifact = ablation_screen.run_ablation_screen(
        cfg, dataset, manifest, output, models=["majority"], cells=["A0"],
        evaluator=_controlled_evaluator([]), **_screen_kwargs())

    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert artifact["self_hash"] == artifact_self_hash(artifact)
    assert load_strict(output) == artifact
    table = ablation_screen.format_compact_table(artifact)
    assert "model" in table and "cell" in table and "majority" in table and "A0" in table

    tampered = copy.deepcopy(artifact)
    _cell(tampered, "majority", "A0")["status"] = "error"
    assert artifact_self_hash(tampered) != tampered["self_hash"]
    tampered_path = tmp_path / "tampered.json"
    dump_strict(tampered, tampered_path)
    calls = []
    with pytest.raises(ValueError):
        ablation_screen.run_ablation_screen(
            cfg, dataset, manifest, tampered_path, models=["majority"], cells=["A0"],
            evaluator=_controlled_evaluator(calls), **_screen_kwargs())
    assert calls == []


def test_controlled_fresh_runs_are_deterministic_and_share_work_inventory(tiny_panel, tmp_path):
    from stylo.eval import work_balanced_ablation_screen as ablation_screen

    cfg, dataset, manifest = tiny_panel
    artifacts = []
    payloads = []
    for name in ("one.json", "two.json"):
        path = tmp_path / name
        artifacts.append(ablation_screen.run_ablation_screen(
            cfg, dataset, manifest, path, models=["majority", "stylo"], cells=["A0"],
            evaluator=_controlled_evaluator([]), **_screen_kwargs()))
        payloads.append(path.read_bytes())
    assert artifacts[0] == artifacts[1] and payloads[0] == payloads[1]

    applied = [r for r in artifacts[0]["cells"] if r["status"] == "applied"]
    assert [(r["model"], r["cell"]) for r in applied] == [("stylo", "A0"), ("majority", "A0")]
    inventories = [[(w["work_id"], w["fold"]) for w in r["works"]] for r in applied]
    assert inventories[0] == inventories[1]
    assert [work_id for work_id, _ in inventories[0]] == ["aa/w1", "aa/w2", "bb/w1", "bb/w2"]


def test_complete_matrix_uses_one_exact_ordered_fold_inventory(tiny_panel, tmp_path):
    from stylo.eval import work_balanced_ablation_screen as ablation_screen

    cfg, dataset, manifest = tiny_panel
    calls = []
    artifact = ablation_screen.run_ablation_screen(
        cfg, dataset, manifest, tmp_path / "complete-matrix.json",
        evaluator=_controlled_evaluator(calls), **_screen_kwargs())

    assert len(artifact["cells"]) == 21
    assert len(calls) == 16
    assert len([r for r in artifact["cells"] if r["status"] == "applied"]) == 16
    expected = [(w["work_id"], w["fold"]) for w in manifest["works"]]
    for record in artifact["cells"]:
        if record["status"] == "applied":
            assert [(w["work_id"], w["fold"]) for w in record["works"]] == expected
        else:
            assert "works" not in record and "metrics" not in record


def test_interrupted_run_keeps_prior_cell_and_resumes_without_recompute(tiny_panel, tmp_path):
    from stylo.eval import work_balanced_ablation_screen as ablation_screen

    cfg, dataset, manifest = tiny_panel
    output = tmp_path / "interrupted.json"
    completed = _controlled_evaluator([])
    attempts = []

    def interrupt_second(*args, **kwargs):
        attempts.append(kwargs["spec"])
        if len(attempts) == 2:
            raise KeyboardInterrupt("synthetic interruption")
        return completed(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        ablation_screen.run_ablation_screen(
            cfg, dataset, manifest, output, models=["stylo", "majority"], cells=["A0"],
            evaluator=interrupt_second, **_screen_kwargs())

    partial = load_strict(output)
    assert partial["self_hash"] == artifact_self_hash(partial)
    assert [(r["model"], r["cell"]) for r in partial["cells"]] == [("stylo", "A0")]

    resumed_calls = []
    final = ablation_screen.run_ablation_screen(
        cfg, dataset, manifest, output, models=["stylo", "majority"], cells=["A0"],
        evaluator=_controlled_evaluator(resumed_calls), **_screen_kwargs())
    assert resumed_calls == ["majority/A0"]
    assert [(r["model"], r["cell"]) for r in final["cells"]] == [
        ("stylo", "A0"), ("majority", "A0")]
