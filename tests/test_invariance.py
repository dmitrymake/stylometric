import dataclasses

import numpy as np
import pytest

from stylo.eval.invariance import (
    align_purged_predictions,
    align_split_predictions,
    build_purged_factor_group_splits,
    build_leave_one_factor_level_out,
    evaluate_predictions,
    factor_slice_indices,
)


def _crossed_metadata():
    # Every author has an independent work in both sources/editions, so holding
    # out a factor level remains a closed-set authorship task.
    return {
        "author": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "work": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
        "topic": ["city", "city", "rural", "rural"] * 2,
        "genre": ["story"] * 8,
        "period": ["early", "early", "late", "late"] * 2,
        "source": ["s1", "s1", "s2", "s2"] * 2,
        "edition": ["e1", "e1", "e2", "e2"] * 2,
    }


def test_leave_one_source_out_indices_are_disjoint_and_feasible():
    meta = _crossed_metadata()
    y = np.array(meta["author"], dtype=object)

    plan = build_leave_one_factor_level_out(meta, "source", y)

    assert plan.diagnostics.n_levels == 2
    assert plan.diagnostics.author_factor_confounded is False
    assert plan.diagnostics.author_level_overlap == 1.0
    assert {split.level for split in plan.splits} == {"s1", "s2"}
    covered = []
    for split in plan.splits:
        assert set(split.train_idx).isdisjoint(split.test_idx)
        assert len(split.train_idx) == len(split.test_idx) == 4
        assert split.diagnostics.possible is True
        assert split.diagnostics.confounded is False
        assert split.diagnostics.test_author_overlap == 1.0
        covered.extend(split.test_idx.tolist())
    assert sorted(covered) == list(range(8))


def test_evaluator_reports_slices_worst_group_coverage_and_cluster_ci():
    meta = _crossed_metadata()
    y = np.array(meta["author"], dtype=object)
    # s1 is perfect.  On s2 one prediction is wrong and one is unavailable.
    pred = np.array(["a", "a", "a", None, "b", "b", "a", "b"], dtype=object)

    report = evaluate_predictions(
        y,
        pred,
        meta,
        factors=("source", "edition"),
        bootstrap_iters=200,
        seed=7,
    )

    assert report.overall.n_total == 8
    assert report.overall.n_evaluated == 7
    assert report.overall.coverage == pytest.approx(7 / 8)
    assert report.overall.n_clusters == 4  # composite (author, work)
    assert report.overall.accuracy.point == pytest.approx(6 / 7)
    assert report.overall.accuracy.lo <= report.overall.accuracy.point
    assert report.overall.accuracy.hi >= report.overall.accuracy.point

    source = report.factors["source"]
    assert source.overall.accuracy == report.overall.accuracy
    assert source.overall.macro_f1 == report.overall.macro_f1
    by_level = {row.level: row for row in source.slices}
    assert by_level["s1"].metrics.accuracy.point == 1.0
    assert by_level["s2"].metrics.coverage == 0.75
    assert by_level["s2"].metrics.accuracy.point == pytest.approx(2 / 3)
    assert source.worst_group_level == "s2"
    assert source.worst_group_accuracy == pytest.approx(2 / 3)
    assert source.possible_split_coverage == 1.0
    assert source.unconfounded_split_coverage == 1.0


def test_author_source_confounding_is_detected_as_impossible_closed_set_split():
    # Author a exists only in source s1 and b only in s2.  A source holdout is
    # therefore an author holdout; source invariance cannot be identified.
    meta = {
        "author": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "work": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
        "topic": ["same"] * 8,
        "genre": ["story"] * 8,
        "period": ["same"] * 8,
        "source": ["s1"] * 4 + ["s2"] * 4,
        "edition": ["e1"] * 4 + ["e2"] * 4,
    }
    y = np.array(meta["author"], dtype=object)

    plan = build_leave_one_factor_level_out(meta, "source", y)

    assert plan.diagnostics.author_factor_confounded is True
    assert "author_fully_confounded_with_source" in plan.diagnostics.messages
    for split in plan.splits:
        assert split.diagnostics.possible is False
        assert split.diagnostics.confounded is True
        assert split.diagnostics.test_author_overlap == 0.0
        assert "no_test_author_seen_in_train" in split.diagnostics.reasons
        assert "test_labels_absent_from_train" in split.diagnostics.reasons

    report = evaluate_predictions(
        y, y.copy(), meta, factors=("source",), bootstrap_iters=0
    )
    source = report.factors["source"]
    assert source.possible_split_coverage == 0.0
    assert source.unconfounded_split_coverage == 0.0
    assert source.overall.accuracy.point is None
    assert source.worst_group_accuracy is None
    assert all(row.metrics.accuracy.point is None for row in source.slices)

    with pytest.raises(ValueError, match="forbidden for impossible"):
        align_split_predictions(
            plan,
            {
                split.level: y[split.test_idx]
                for split in plan.splits
            },
        )


def test_test_only_predictions_can_be_aligned_without_train_predictions():
    meta = _crossed_metadata()
    y = np.array(meta["author"], dtype=object)
    plan = build_leave_one_factor_level_out(meta, "source", y)
    by_level = {}
    for split in plan.splits:
        # In real use each value is emitted by a model fitted only on train_idx.
        by_level[split.level] = y[split.test_idx]

    aligned = align_split_predictions(plan, by_level)

    assert np.array_equal(aligned, y)
    with pytest.raises(ValueError, match="expected"):
        align_split_predictions(plan, {"s1": ["a"], "s2": y[:4]})
    with pytest.raises(KeyError, match="missing predictions"):
        align_split_predictions(plan, {"s1": y[plan.by_level()["s1"].test_idx]})


def test_factor_slices_include_missing_metadata_and_single_level_is_diagnosed():
    meta = {
        "author": ["a", "a", "b", "b"],
        "work": ["a1", "a2", "b1", "b2"],
        "source": ["s1", None, "s1", None],
        "edition": ["only"] * 4,
    }
    slices = factor_slice_indices(meta, "source")
    assert set(slices) == {"s1", "<MISSING>"}

    plan = build_leave_one_factor_level_out(meta, "edition", meta["author"])
    assert "factor_has_fewer_than_two_levels" in plan.diagnostics.messages
    assert len(plan.splits) == 1
    assert plan.splits[0].diagnostics.possible is False
    assert "empty_train" in plan.splits[0].diagnostics.reasons


def test_purged_factor_work_splits_share_neither_factor_nor_work_with_train():
    metadata = {
        "author": [],
        "work": [],
        "source": [],
    }
    for author in ["a", "b"]:
        for work_no in range(3):
            for source in ["s1", "s2", "s3"]:
                metadata["author"].append(author)
                metadata["work"].append(f"{author}_w{work_no}")
                metadata["source"].append(source)
    y = np.asarray(metadata["author"], dtype=object)

    plan = build_purged_factor_group_splits(metadata, "source", y)

    assert plan.diagnostics.test_coverage == 1.0
    assert plan.diagnostics.possible_split_coverage == 1.0
    predictions = {}
    for split in plan.splits:
        assert set(np.asarray(metadata["source"], dtype=object)[split.train_idx]) == (
            {"s1", "s2", "s3"} - {split.level}
        )
        assert split.group not in set(np.asarray(metadata["work"], dtype=object)[split.train_idx])
        assert set(split.train_idx).isdisjoint(split.test_idx)
        assert set(split.purged_idx).isdisjoint(split.test_idx)
        predictions[(split.level, split.group)] = y[split.test_idx]
    assert np.array_equal(align_purged_predictions(plan, predictions), y)


def test_metric_label_universe_is_complete_unique_and_frozen():
    meta = _crossed_metadata()
    y = np.asarray(meta["author"], dtype=object)
    pred = y.copy()

    with pytest.raises(ValueError, match="omits observed truth"):
        evaluate_predictions(
            y, pred, meta, factors=("source",), labels=["a"], bootstrap_iters=0
        )
    with pytest.raises(ValueError, match="duplicate-free"):
        evaluate_predictions(
            y,
            pred,
            meta,
            factors=("source",),
            labels=["a", "b", "b"],
            bootstrap_iters=0,
        )

    report = evaluate_predictions(
        y,
        pred,
        meta,
        factors=("source",),
        labels=["a", "b", "registered_absent"],
        bootstrap_iters=0,
    )
    assert report.overall.n_labels == 3
    assert report.overall.macro_f1.point == pytest.approx(2 / 3)
    assert all(row.metrics.n_labels == 3 for row in report.factors["source"].slices)


def test_supplied_plan_must_match_current_metadata_and_truth():
    meta = _crossed_metadata()
    y = np.asarray(meta["author"], dtype=object)
    plan = build_leave_one_factor_level_out(meta, "source", y)
    first = plan.splits[0]
    forged = dataclasses.replace(
        plan,
        splits=(
            dataclasses.replace(first, test_idx=first.test_idx[::-1]),
            *plan.splits[1:],
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_predictions(
            y,
            y.copy(),
            meta,
            factors=("source",),
            plans={"source": forged},
            bootstrap_iters=0,
        )
