import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "taras_delta_replication", _ROOT / "scripts" / "run_taras_delta_replication.py"
)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _rows():
    # Three works per class keep both author pools non-empty in every LOO fold.
    # Every chunk is unique so a fit-text spy can detect held-out leakage.
    return [
        ("A", "a0", "и и и в на alpha_a0_0"),
        ("A", "a0", "и и в в на alpha_a0_1"),
        ("A", "a1", "и и и на на alpha_a1_0"),
        ("A", "a1", "и и в на на alpha_a1_1"),
        ("A", "a2", "и и и и на alpha_a2_0"),
        ("A", "a2", "и и и в в alpha_a2_1"),
        ("B", "b0", "не не не но а beta_b0_0"),
        ("B", "b0", "не не но но а beta_b0_1"),
        ("B", "b1", "не не не а а beta_b1_0"),
        ("B", "b1", "не не но а а beta_b1_1"),
        ("B", "b2", "не не не не а beta_b2_0"),
        ("B", "b2", "не не не но но beta_b2_1"),
    ]


@pytest.mark.parametrize("mode", ["delta_fw", "delta_mfw"])
def test_cached_permutation_scores_match_literal_full_refits(mode):
    rows = _rows()
    cache = runner.build_permutation_cache(rows, mode)
    original = list(cache.observed_labels)
    swapped = original[:]
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    rotated = original[2:] + original[:2]
    assignments = [original, swapped, rotated]

    optimized = runner.work_macro_scores(cache, assignments)
    literal = np.asarray(
        [runner._work_loo_for_labels(rows, mode, labels)[0] for labels in assignments]
    )

    assert np.allclose(optimized, literal)


def test_permutation_is_deterministic_for_seed():
    rows = _rows()
    cache = runner.build_permutation_cache(rows, "delta_fw")

    first = runner.permutation_p(rows, "delta_fw", n=41, seed=1729, cache=cache)
    second = runner.permutation_p(rows, "delta_fw", n=41, seed=1729, cache=cache)

    assert first == second
    assert 1 / 42 <= first <= 1.0


@pytest.mark.parametrize("mode", ["delta_fw", "delta_mfw"])
def test_held_out_text_cannot_change_its_fold_training_transform(mode):
    rows = _rows()
    before = runner.build_permutation_cache(rows, mode)
    heldout = before.works[0]
    changed = [
        (author, work, "heldoutonly heldoutonly radicallychanged")
        if (author, work) == heldout else (author, work, chunk)
        for author, work, chunk in rows
    ]
    after = runner.build_permutation_cache(changed, mode)

    before_fold = before.folds[0]
    after_fold = after.folds[0]
    assert np.array_equal(before_fold.train_indices, after_fold.train_indices)
    assert np.allclose(before_fold.train_work_z, after_fold.train_work_z)


def test_mfw_cutoff_ties_match_count_vectorizer_literal_fit():
    tied = " ".join(f"token{i:03d}" for i in range(350))
    rows = [
        (author, f"{author}{work}", f"{tied} unique_{author}_{work}")
        for author in ("A", "B")
        for work in range(3)
    ]
    cache = runner.build_permutation_cache(rows, "delta_mfw")
    heldout = cache.works[0]
    train = [(a, w, text) for a, w, text in rows if (a, w) != heldout]
    literal = runner.make_model("delta_mfw").fit(
        [text for _a, _w, text in train],
        [a for a, _w, _text in train],
        groups=[(a, w) for a, w, _text in train],
    )

    assert cache.folds[0].feature_names == tuple(literal.feature_names())


def test_gate_requires_recall_and_significant_refit_null():
    assert runner.gate_passes(0.80, 0.05)
    assert not runner.gate_passes(0.79, 0.01)
    assert not runner.gate_passes(0.95, 0.051)


def test_out_cli_is_independent_from_historical_report(tmp_path):
    out = tmp_path / "exploratory.json"

    args = runner.parse_args(
        ["--out", str(out), "--n-perm", "17", "--seed", "99"]
    )

    assert args.out == out
    assert args.n_perm == 17
    assert args.seed == 99
    assert args.out != runner.HISTORICAL_OUT
    assert runner.parse_args([]).out == runner.DEFAULT_OUT


def test_historical_report_requires_explicit_overwrite_flag():
    with pytest.raises(SystemExit):
        runner.parse_args(["--out", str(runner.HISTORICAL_OUT)])

    args = runner.parse_args(
        [
            "--out", str(runner.HISTORICAL_OUT),
            "--overwrite-historical",
        ]
    )
    assert args.out == runner.HISTORICAL_OUT
