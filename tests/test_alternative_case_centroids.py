import importlib.util
from pathlib import Path

import numpy as np


_PATH = Path(__file__).parents[1] / "scripts" / "run_chekhonte_dubia_oskolki.py"
_SPEC = importlib.util.spec_from_file_location("chekhonte_case_runner", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(runner)


def test_alternative_runner_retains_equal_work_directions():
    vectors = runner.WorkVectors(
        [
            np.array([1.0, 0.0]),
            np.array([-0.8, 0.0]),
            np.array([0.0, 1.0]),
        ],
        ["heterogeneous", "heterogeneous", "stable"],
    )

    work_rows = runner.work_centroids(vectors)
    author_centroid = runner.work_balanced_centroid(
        zip(vectors.work_ids, vectors)
    )

    assert np.allclose(work_rows, [[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(author_centroid, np.array([1.0, 1.0]) / np.sqrt(2.0))


def test_alternative_runner_loo_holds_out_whole_works():
    # Each A work points away from the other A work and loses to B when held out.
    # Leaking the other chunks of A/outlier would make that work classify as A.
    for repetitions in (1, 10):
        docvecs = {
            "A": runner.WorkVectors([np.array([0., 1.])] * repetitions + [np.array([1., 0.])],
                                     ["A/outlier"] * repetitions + ["A/other"]),
            "B": runner.WorkVectors([np.array([0.4, 0.9]), np.array([0.4, 0.9])], ["B/one", "B/two"]),
        }
        assert runner.loo(docvecs) == {"correct": 2, "total": 4, "accuracy": 0.5}
