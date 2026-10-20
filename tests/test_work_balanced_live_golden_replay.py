"""Opt-in prediction replay in the frozen numerical environment.

Run with WORK_BALANCED_LIVE_GOLDEN_REPLAY=1, PYTHONHASHSEED=0 and
OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2 before Python starts.
Historical fixture bytes stay immutable. Ordered fit calls, private state and
learner class names are archived observations, not runtime behavior contracts.
"""
from __future__ import annotations

# ── env gate FIRST, before numpy is imported (mirrors scripts/artifacts/capture_work_balanced_ablation_goldens.py) ──
import hashlib
import os
import pathlib

import pytest

_REQUIRED_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2",
                 "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}

if os.environ.get("WORK_BALANCED_LIVE_GOLDEN_REPLAY") != "1":
    pytest.skip(
        "capture-env parity gate — run with:\n"
        "  WORK_BALANCED_LIVE_GOLDEN_REPLAY=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2 uv run pytest tests/test_work_balanced_live_golden_replay.py -q",
        allow_module_level=True)

# flag is set -> from here on NO silent skip: a wrong environment is a hard failure.
_bad_env = {k: (v, os.environ.get(k)) for k, v in _REQUIRED_ENV.items() if os.environ.get(k) != v}
if _bad_env:
    raise RuntimeError(
        f"WORK_BALANCED_LIVE_GOLDEN_REPLAY=1 requires the pinned deterministic env set BEFORE numpy import; "
        f"wrong/missing {{var: (expected, got)}} = {_bad_env}")

import numpy as np  # noqa: E402  (must follow the env assertion above)

from stylo import jsonio  # noqa: E402
from stylo.config import load_config, with_overrides  # noqa: E402
from stylo.eval.dispatch import fit_estimator  # noqa: E402
from stylo.eval.lobo import make_factory_for_ablation  # noqa: E402
from stylo.domain.work_weighting import FULL_WB_ABLATION, LEGACY_ABLATION  # noqa: E402
from stylo.features.reps import make_rep_cache  # noqa: E402
from stylo.models.stacked_clf import (  # noqa: E402
    validate_withdrawn_internal_selection_diagnostic,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX_DIR = ROOT / "tests" / "fixtures"
FIXTURE = FIX_DIR / "work_balanced_ablation_goldens_v1.json"
INVENTORY = FIX_DIR / "work_balanced_ablation_goldens_v1.SHA256SUMS"
FIXTURE_SHA256 = "c66e63e7e8af36b03b6aaa28f3d8cb23b71ef8e7a433ddd0c85322463b993a25"
_FIX = jsonio.load_strict(FIXTURE)
RD = _FIX["numeric_contract"]["round_decimals"]

# (fixture model key, spec, panel key). A0/A4 both replayed for each.
_REPLAY = [
    ("delta_cos:500", "delta_cos:500", "P1"),
    ("delta_cos:12", "delta_cos:12", "P1"),
    ("char_cos", "char_cos", "P1"),
    ("bow_lr", "bow_lr", "P1"),
    ("stylo", "stylo", "P1"),
    ("stylo_stack", "stylo_stack", "P1"),
    ("stylo_stack__faildisabled", "stylo_stack", "P2"),
]


def _proba(est, texts):
    return np.round(np.asarray(est.predict_proba(texts), dtype=np.float64), RD).tolist()


def _panel(pk):
    p = _FIX["panels"][pk]
    # capture semantics: the raw LIST of texts (fit wraps it in np.array(dtype=object); extract/predict
    # receive the list), y and groups as arrays.
    return list(p["texts"]), np.array(p["y"]), np.array(p["groups"], dtype=object)


def _warm_all_thread_pools():
    """Load EVERY BLAS/OpenMP pool the capture had live (numpy + scipy OpenBLAS, sklearn + spaCy/blis
    OpenMP) so the fingerprint multiset is complete regardless of import/test order."""
    import scipy.linalg
    import spacy
    from sklearn.svm import LinearSVC
    np.dot(np.ones((8, 8)), np.ones((8, 8)))
    scipy.linalg.svd(np.ones((6, 6)))
    X = np.vstack([np.zeros((10, 2)), np.full((10, 2), 10.0)])
    LinearSVC(max_iter=1000).fit(X, [0] * 10 + [1] * 10)      # separable -> loads sklearn OpenMP, no warning
    nlp = spacy.load(load_config().get_path("language.spacy_model", "ru_core_news_lg"))
    list(nlp.pipe(["Кот сидел на окне."]))                    # loads the spaCy/blis OpenMP pool


def _threadpool_multiset(pools):
    return sorted((p.get("internal_api"), p.get("version"), p.get("num_threads")) for p in pools)


# ── fail-closed environment / provenance checks (run before any replay) ─────────
def test_fixture_sha_and_inventory_are_pinned():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    lines = INVENTORY.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split()
    assert name == FIXTURE.name and digest == FIXTURE_SHA256


def test_runtime_fingerprint_matches_fixture():
    import sys

    import scipy
    import sklearn
    import spacy
    e = _FIX["environment"]
    live = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "spacy": spacy.__version__}
    mismatch = {k: (e[k], v) for k, v in live.items() if e[k] != v}
    assert not mismatch, f"runtime fingerprint drift {{lib: (fixture, live)}} = {mismatch}"
    # spaCy model metadata by every field the fixture froze
    nlp = spacy.load(load_config().get_path("language.spacy_model", "ru_core_news_lg"))
    for field in ("name", "lang", "version", "spacy_git_version"):
        assert e["spacy_model"][field] == nlp.meta.get(field), field


def test_thread_pool_fingerprint_matches_fixture():
    """The ACTUAL runtime pools must reproduce the fixture's recorded BLAS/OpenMP fingerprint AND be
    pinned to 2 threads — the goldens are non-deterministic otherwise. Compared as an order-independent
    multiset of (internal_api, version, num_threads)."""
    import threadpoolctl
    _warm_all_thread_pools()
    pools = threadpoolctl.threadpool_info()
    assert pools, "no BLAS/OpenMP thread pools detected"
    bad = [(p.get("internal_api"), p.get("num_threads")) for p in pools if p.get("num_threads") != 2]
    assert not bad, f"thread pools not pinned to 2 (non-deterministic capture env): {bad}"
    live = _threadpool_multiset(pools)
    want = _threadpool_multiset(_FIX["environment"]["threadpool"])
    assert live == want, f"BLAS/OpenMP fingerprint drift: live={live} fixture={want}"


# ── the whole frozen contract is reproduced live ───────────────────────────────
@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    cache = tmp_path_factory.mktemp("work_balanced_live_replay")
    return with_overrides(load_config(), {"paths.data": str(cache),
                                          "paths.doc_cache": str(cache / "doc_cache")})


@pytest.mark.parametrize("model_key,spec,panel_key", _REPLAY, ids=[m for m, _, _ in _REPLAY])
@pytest.mark.parametrize("corner,ablation", [("A0", LEGACY_ABLATION), ("A4", FULL_WB_ABLATION)])
def test_frozen_contract_is_reproduced(cfg, model_key, spec, panel_key, corner, ablation):
    texts, y, groups = _panel(panel_key)
    make_rep_cache(cfg).warm(texts, n_process=1)
    estimator = make_factory_for_ablation(spec, cfg, ablation=ablation)()
    fit_estimator(estimator, np.array(texts, dtype=object), y, groups)
    expected = _FIX["models"][model_key][corner]
    assert list(estimator.classes_) == expected["classes_"]
    np.testing.assert_array_equal(_proba(estimator, texts), expected["proba"])
    # Repeated prediction must survive an implementation that caches fitted channels.
    np.testing.assert_array_equal(_proba(estimator, texts), expected["proba"])
    if spec == "stylo_stack":
        passport = estimator.passport_
        safety = passport["inner_oof_book_top1"]
        validate_withdrawn_internal_selection_diagnostic(safety)
        assert safety["eligible_as_unbiased_evidence"] is False
        assert safety["descriptive_only"] == expected["passport"]["inner_oof_book_top1"]
        assert passport.get("calibration_disabled", False) == expected["passport"].get("calibration_disabled", False)
