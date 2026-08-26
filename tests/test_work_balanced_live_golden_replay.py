"""Heavy capture-environment replay gate for the frozen A0/A4 external goldens.

A plain ``uv run pytest`` is NOT proof of live external parity for ``stylo``/``stylo_stack``: it is not
guaranteed a spaCy model, and — decisively — not the pinned deterministic thread environment the
goldens were captured under. This module is that proof. It is SKIPPED unless
``WORK_BALANCED_LIVE_GOLDEN_REPLAY=1``; once the flag is set it is fail-closed — it verifies the environment,
the pinned fixture SHA + inventory, the installed runtime fingerprint and the ACTUAL BLAS/OpenMP
thread pools, then replays the WHOLE frozen contract.

    WORK_BALANCED_LIVE_GOLDEN_REPLAY=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
    OPENBLAS_NUM_THREADS=2 uv run pytest tests/test_work_balanced_live_golden_replay.py -q

For A0 (legacy) and A4 (full work-balanced), reached through the refactored
``make_factory_for_ablation`` audit path, this re-runs the SAME instrumentation the capture tool used
(a separate replay helper — the canonical ``scripts/artifacts/capture_work_balanced_ablation_goldens.py`` is never edited) and
requires the reconstructed per-model corner — ``requested_axes``, the full ordered ``fit_trace``
(vocab/IDF/weights/coef digests), the Delta z-state, the stack calibration passport and the
class-aligned ``proba`` — to equal the committed fixture EXACTLY. The fixture and capture tool are
read-only here.
"""
from __future__ import annotations

# ── env gate FIRST, before numpy is imported (mirrors scripts/artifacts/capture_work_balanced_ablation_goldens.py) ──
import contextlib
import hashlib
import importlib
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
    STACK_PASSPORT_SCHEMA_V1,
    STACK_PASSPORT_SCHEMA_V2,
    STACK_SELECTION_EVIDENCE_STATUS,
    project_stack_passport_compatibility,
    validate_withdrawn_internal_selection_diagnostic,
)
from stylo.vectorizer import StyloVectorizer  # noqa: E402

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


# ── instrumentation mirroring scripts/artifacts/capture_work_balanced_ablation_goldens.py (never edits that frozen tool) ──
def _sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else str(b).encode("utf-8")).hexdigest()


def _adig(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "dtype": "float64", "digest": _sha(np.round(a, RD).tobytes())}


def _proba(est, texts):
    return np.round(np.asarray(est.predict_proba(texts), dtype=np.float64), RD).tolist()


def _nrows(X):
    if X is None:
        return None
    if hasattr(X, "shape"):
        return int(X.shape[0])
    try:
        return len(X)
    except Exception:
        return None


def _safe_fn(block):
    try:
        return list(block.feature_names())
    except Exception:
        return []


@contextlib.contextmanager
def record_fits():
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from stylo.features import base as _base
    from stylo.features.work_vectorizer import WorkLevelVectorizer
    trace, patched = [], []

    def patch(cls, attr, kind, extract):
        orig = getattr(cls, attr)

        def wrapper(self, *a, **kw):
            sw = kw.get("sample_weight", a[2] if len(a) >= 3 else None)
            r = orig(self, *a, **kw)
            rec = {"kind": kind, "n_rows": _nrows(a[0] if a else kw.get("X"))}
            if sw is not None:
                rec["sample_weight"] = _adig(sw)
            rec.update(extract(self))
            trace.append(rec)
            return r
        setattr(cls, attr, wrapper)
        patched.append((cls, attr, orig))

    def _cw(s):
        return s.class_weight if not isinstance(s.class_weight, dict) else "dict"
    patch(LogisticRegression, "fit", "lr", lambda s: {"class_weight": _cw(s), "coef_": _adig(s.coef_)})
    patch(LinearSVC, "fit", "svc", lambda s: {"class_weight": _cw(s), "coef_": _adig(s.coef_)})
    patch(TfidfTransformer, "fit", "tfidf", lambda s: {"idf_": _adig(s.idf_)})
    patch(CountVectorizer, "fit", "count_vec",
          lambda s: {"vocab_digest": _sha("|".join(map(str, s.get_feature_names_out())))})
    patch(CountVectorizer, "fit_transform", "count_vec",
          lambda s: {"vocab_digest": _sha("|".join(map(str, s.get_feature_names_out())))})
    patch(WorkLevelVectorizer, "fit", "work_vec",
          lambda s: {"vocab_digest": _sha("|".join(map(str, s.feature_names()))),
                     **({"idf_": _adig(s.idf_)} if getattr(s, "idf_", None) is not None else {})})
    for mod in ("char_ngrams", "function_words", "pos_ngrams", "punctuation", "dependency",
                "morphology", "length_dist", "syntax"):
        m = importlib.import_module(f"stylo.features.{mod}")
        for obj in vars(m).values():
            if (isinstance(obj, type) and issubclass(obj, _base.FeatureBlock)
                    and obj is not _base.FeatureBlock and "fit" in obj.__dict__):
                patch(obj, "fit", f"block:{obj.__name__}",
                      lambda s: {"vocab_digest": _sha("|".join(map(str, _safe_fn(s))))})
    try:
        yield trace
    finally:
        for cls, attr, orig in patched:
            setattr(cls, attr, orig)


def _stylo_blocks(est):
    vec = getattr(est, "named_steps", {}).get("vectorizer")
    if not isinstance(vec, StyloVectorizer):
        return {}
    return {"vectorizer_feature_names_digest": _sha("|".join(map(str, vec.feature_names()))),
            "block_feature_digests": {type(b).__name__: _sha("|".join(map(str, _safe_fn(b)))) for b in vec.blocks}}


def _extract(spec, est, texts):
    st = {"classes_": [int(c) for c in est.classes_], "proba": _proba(est, texts)}
    if spec.startswith("delta"):
        st.update({"group_weighting_": est.group_weighting_, "feature_names": list(map(str, est.feature_names())),
                   "mean_": _adig(est.mean_), "std_": _adig(est.std_), "centroids_": _adig(est.centroids_)})
    elif spec == "char_cos":
        fn = est._wv.feature_names() if getattr(est, "_wv", None) is not None else est._vec.get_feature_names_out()
        st.update({"group_weighting_": est.group_weighting_, "feature_names": list(map(str, fn)),
                   "centroids": _adig(est._centroids)})
    elif spec == "stylo_stack":
        st.update({"passport": est.passport_, "mode_": est.mode_, "meta_is_none": est.meta_ is None})
    else:
        st.update(_stylo_blocks(est))
    return st


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
    texts, y, groups = _panel(panel_key)                      # texts is the raw list
    make_rep_cache(cfg).warm(texts, n_process=1)
    est = make_factory_for_ablation(spec, cfg, ablation=ablation)()
    on = corner == "A4"
    with record_fits() as trace:                              # kept OPEN through fit AND predict_proba
        # literal capture semantics: fit receives np.array(dtype=object); extract/predict the list
        fit_estimator(est, np.array(texts, dtype=object), y, groups)
        state = _extract(spec, est, texts)
    if spec == "stylo_stack":
        # The current safety wrapper is independently authoritative: a historical
        # comparison is allowed only after proving the live scores remain explicitly
        # withdrawn and ineligible as unbiased evidence.
        safety = state["passport"]["inner_oof_book_top1"]
        validate_withdrawn_internal_selection_diagnostic(safety)
        assert safety["status"] == STACK_SELECTION_EVIDENCE_STATUS
        assert safety["eligible_as_unbiased_evidence"] is False
        state["passport"] = project_stack_passport_compatibility(
            state["passport"],
            source_schema_version=STACK_PASSPORT_SCHEMA_V2,
            target_schema_version=STACK_PASSPORT_SCHEMA_V1,
        )
        # Only the wrapper is projected. Its exact live numerical values remain
        # exposed to the unchanged v1 fixture comparison below.
        assert state["passport"]["inner_oof_book_top1"] == safety["descriptive_only"]
    reconstructed = {"requested_axes": {"W": on, "F": on, "R": on}, "fit_trace": trace, **state}
    want = _FIX["models"][model_key][corner]
    # exact whole-contract equality: vocab/IDF/weights/coef digests, Delta z-state, calibration
    # passport, class-aligned proba, and the full ordered fit-trace (a broken axis moves at least one).
    assert reconstructed == want
