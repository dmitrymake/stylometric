#!/usr/bin/env python3
"""B4-B1: freeze EXTERNAL A0/A4 goldens for the audited models from the committed B3 state.

Captured from a clean pinned `f1b8e165` worktree BEFORE any B4-B estimator-axis refactor (design
§2.6). All learned state comes from the REAL fit calls: the learners (LogisticRegression, LinearSVC,
TfidfTransformer, CountVectorizer.{fit,fit_transform}, WorkLevelVectorizer) AND every FeatureBlock are
instrumented, and the instrumentation stays open through predict_proba, so the trace records the
inner-fold AND full-train fits (e.g. the stack's 18 SVC calls) with their actual weights/vocab/IDF —
a broken routing cannot pass by recomputing an "expected" value. STRICT JSON (no pickle); numeric
state is ``round(12)`` digests under a quantized_exact contract. Fail-closed provenance: env
(PYTHONHASHSEED, OMP) is required BEFORE numpy/sklearn import; full pinned commit + clean tree; the
config/lock hashes are read from the pinned SOURCE worktree; a fresh isolated cache is used; the
canonical fixture is never overwritten (``--output`` is required and must not exist).

    git worktree add --detach /tmp/b4_gold f1b8e165
    PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTHONPATH=/tmp/b4_gold/src .venv/bin/python \
        scripts/capture_b4_goldens.py --output tests/fixtures/b4_goldens_v1.json
    git worktree remove /tmp/b4_gold --force
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


REQUIRED_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2"}


def _require_env() -> None:
    for k, v in REQUIRED_ENV.items():
        if os.environ.get(k) != v:
            raise SystemExit(f"deterministic capture requires {k}={v!r} BEFORE numpy import (got {os.environ.get(k)!r})")


if __name__ == "__main__":
    _require_env()                          # fail-closed before numpy/sklearn are imported

import numpy as np

SOURCE_COMMIT = "f1b8e165fad4a1ae7ce30c0a613ffcf0deaa7b3d"
TOOL = pathlib.Path(__file__).resolve()
RD = 12
LEGACY, WB = "chunk_weighted_legacy", "work_balanced"


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


@contextlib.contextmanager
def record_fits():
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from stylo.features import base as _base
    from stylo.features.registry import BLOCK_ORDER, build_blocks   # noqa: F401 (ensures blocks importable)
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
    # every FeatureBlock (incl. the custom dependency/morphology channels that use no sklearn vocab)
    import importlib
    for mod in ("char_ngrams", "function_words", "pos_ngrams", "punctuation", "dependency",
                "morphology", "length_dist", "syntax"):
        m = importlib.import_module(f"stylo.features.{mod}")
        for obj in vars(m).values():
            if isinstance(obj, type) and issubclass(obj, _base.FeatureBlock) and obj is not _base.FeatureBlock and "fit" in obj.__dict__:
                patch(obj, "fit", f"block:{obj.__name__}",
                      lambda s: {"vocab_digest": _sha("|".join(map(str, _safe_fn(s))))})
    try:
        yield trace
    finally:
        for cls, attr, orig in patched:
            setattr(cls, attr, orig)


def _safe_fn(block):
    try:
        return list(block.feature_names())
    except Exception:
        return []


def _panel(layout, nouns):
    VERBS = ["сидел", "смотрел", "думал", "стоял", "молчал", "ждал"]
    texts, y, groups = [], [], []
    for lab, wnum, n in layout:
        wid, u = f"{'ab'[lab]}/w{wnum}", nouns[f"{'ab'[lab]}/w{wnum}"]
        for c in range(n):
            v1, v2 = VERBS[c % 6], VERBS[(c + 3) % 6]
            texts.append(f"{u.capitalize()} {v1} на {u}, и {v2} в {u}. {u.capitalize()} {v1} у {u} — а {u} {v2}!")
            y.append(lab); groups.append(wid)
    return texts, np.array(y), np.array(groups, dtype=object)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=pathlib.Path,
                    help="fixture path; must NOT already exist (canonical is never overwritten)")
    args = ap.parse_args()
    out = args.output
    sha_out = out.parent / (out.stem + ".SHA256SUMS")
    if os.path.lexists(out) or os.path.lexists(sha_out):     # lexists: also refuse a symlink/broken link
        raise SystemExit(f"{out} or {sha_out} already exists (incl. symlink) — the canonical golden is never overwritten")

    import stylo
    repo = pathlib.Path(stylo.__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    if head != SOURCE_COMMIT:
        raise SystemExit(f"stylo import HEAD {head} != pinned {SOURCE_COMMIT}")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]).strip():
        raise SystemExit("source worktree is dirty")

    import spacy
    import threadpoolctl
    from stylo.config import load_config, with_overrides
    from stylo.eval.dispatch import fit_estimator
    from stylo.eval.lobo import make_factory
    from stylo.features.reps import make_rep_cache
    from stylo.vectorizer import StyloVectorizer
    from stylo.jsonio import dump_strict

    np.dot(np.ones(2), np.ones(2))                       # load BLAS, then verify the ACTUAL pools
    for pool in threadpoolctl.threadpool_info():
        if pool.get("num_threads") != 2:
            raise SystemExit(f"{pool.get('internal_api')} pool num_threads={pool.get('num_threads')} != 2 — non-deterministic")

    cache = pathlib.Path(tempfile.mkdtemp(prefix="b4_gold_cache_"))
    cfg = with_overrides(load_config(), {"paths.data": str(cache), "paths.doc_cache": str(cache / "doc_cache")})

    def _stylo_blocks(est):
        vec = getattr(est, "named_steps", {}).get("vectorizer")
        if not isinstance(vec, StyloVectorizer):
            return {}
        return {"vectorizer_feature_names_digest": _sha("|".join(map(str, vec.feature_names()))),
                "block_feature_digests": {type(b).__name__: _sha("|".join(map(str, _safe_fn(b)))) for b in vec.blocks}}

    def extract(spec, est, texts):
        st = {"classes_": [int(c) for c in est.classes_], "proba": _proba(est, texts)}
        if spec.startswith("delta"):
            st.update({"group_weighting_": est.group_weighting_, "feature_names": list(map(str, est.feature_names())),
                       "mean_": _adig(est.mean_), "std_": _adig(est.std_), "centroids_": _adig(est.centroids_)})
        elif spec == "char_cos":
            fn = est._wv.feature_names() if getattr(est, "_wv", None) is not None else est._vec.get_feature_names_out()
            st.update({"group_weighting_": est.group_weighting_, "feature_names": list(map(str, fn)), "centroids": _adig(est._centroids)})
        elif spec == "stylo_stack":
            st.update({"passport": est.passport_, "mode_": est.mode_, "meta_is_none": est.meta_ is None})
        else:
            st.update(_stylo_blocks(est))
        return st

    def capture(spec, texts, y, groups):
        make_rep_cache(cfg).warm(list(texts), n_process=1)
        out_c = {}
        for corner, w in (("A0", LEGACY), ("A4", WB)):
            est = make_factory(spec, cfg, weighting=w)()
            with record_fits() as trace:                # kept OPEN through fit AND predict_proba
                fit_estimator(est, np.array(texts, dtype=object), y, groups)
                st = extract(spec, est, texts)
            out_c[corner] = {"requested_axes": {"W": w == WB, "F": w == WB, "R": w == WB}, "fit_trace": trace, **st}
        return out_c

    P1 = [(0, i + 1, n) for i, n in enumerate([6, 3, 2, 2, 2, 2])] + [(1, i + 7, n) for i, n in enumerate([6, 3, 2, 2, 2, 2])]
    p1 = _panel(P1, {f"{'ab'[l]}/w{wn}": f"{'ab'[l]}предмет{wn}" for l, wn, _ in P1})
    P2 = [(0, 1, 3), (0, 2, 2), (0, 3, 2), (1, 4, 3), (1, 5, 2), (1, 6, 2)]
    p2 = _panel(P2, {f"{'ab'[l]}/w{wn}": f"{'ab'[l]}слово{wn}" for l, wn, _ in P2})

    def pmeta(p):
        t, y, g = p
        return {"n_chunks": len(t), "texts": list(t), "y": [int(v) for v in y], "groups": [str(x) for x in g],
                "authors": sorted({str(x).split("/", 1)[0] for x in g}),
                "panel_sha256": _sha("␟".join(t) + "␞" + ",".join(map(str, y.tolist())) + "␞" + ",".join(map(str, g.tolist())))}

    models = {}
    for spec in ("delta_cos:500", "delta_cos:12", "char_cos", "bow_lr", "stylo", "stylo_stack"):
        print(f"capturing {spec} (P1) …", flush=True)
        models[spec] = {"panel": "P1", **capture(spec, *p1)}
    print("capturing stylo_stack (P2 fail-closed) …", flush=True)
    models["stylo_stack__faildisabled"] = {"panel": "P2", **capture("stylo_stack", *p2)}

    np.dot(np.ones(4), np.ones(4))
    tp = [{"internal_api": i.get("internal_api"), "version": i.get("version"), "num_threads": i.get("num_threads")}
          for i in threadpoolctl.threadpool_info()]
    nlp = spacy.load(cfg.get_path("language.spacy_model", "ru_core_news_lg"))
    fixture = {
        "fixture_version": "b4.goldens.v1", "source_commit": SOURCE_COMMIT, "clean_tree": True,
        "numeric_contract": {"kind": "quantized_exact", "round_decimals": RD},
        "panels": {"P1": pmeta(p1), "P2": pmeta(p2)},
        "environment": {
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__, "spacy": spacy.__version__,
            "spacy_model": {"name": nlp.meta.get("name"), "lang": nlp.meta.get("lang"),
                            "version": nlp.meta.get("version"), "spacy_git_version": nlp.meta.get("spacy_git_version")},
            "threadpool": tp, "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
            # hashes read from the PINNED SOURCE worktree, not the CWD
            "config_sha256": _sha((repo / "configs" / "default.yaml").read_bytes()),
            "lock_sha256": _sha((repo / "requirements.lock").read_bytes()),
            "capture_tool_sha256": _sha(TOOL.read_bytes())},
        "models": models,
    }
    shutil.rmtree(cache, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_strict(fixture, out, indent=2)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sha_out.write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"wrote {out} ({digest[:16]}) + SHA256SUMS from {head[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
