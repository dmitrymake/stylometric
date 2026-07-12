import importlib.util
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


_PATH = Path(__file__).parents[1] / "scripts" / "lobo_cv.py"
_SPEC = importlib.util.spec_from_file_location("legacy_lobo_cv", _PATH)
legacy_lobo = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(legacy_lobo)


def test_legacy_lobo_delta_centroid_weights_books_not_chunks():
    X = csr_matrix(
        [[1.0, 0.0]] * 9
        + [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    y = np.array([0] * 10 + [1, 1])
    groups = np.array(["a/long"] * 9 + ["a/short", "b/0", "b/1"])

    centroids, labels = legacy_lobo.compute_centroids_sparse_mean(X, y, groups)

    a_idx = int(np.flatnonzero(labels == 0)[0])
    assert np.allclose(centroids[a_idx], [0.5, 0.5])
    assert not np.allclose(centroids[a_idx], [0.9, 0.1])
