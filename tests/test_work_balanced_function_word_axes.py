"""Function-word F×R grid and the A2/A3 feature-state estimators.

Two independent axes on the UNCHANGED A0 loss:
  F (feature state) — pooled ``CountVectorizer`` (F0) vs work-level ``WorkLevelVectorizer`` (F1);
      routed by ``groups`` at fit; affects EVERY learned block.
  R (relative FW)   — raw selected counts (R0) vs ``selected / all analyzer events`` (R1); a
      FunctionWord-only constructor policy, independent of F.

    A0 F0R0 pooled·raw    A2 F1R0 work·raw
    A3 F0R1 pooled·rel    A4 F1R1 work·rel

The FunctionWord block and BoW are spaCy-free; the full ``stylo``/``stylo_stack`` estimators need the
ru spaCy model and skip without it. A2/A3 are NOT self-goldens — each is checked against an
independent recomputation and the derived invariants (non-FW A2==A4, A3==A0; FW raw/relative).
"""
from __future__ import annotations

import contextlib
import io
import pickle

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer

from stylo.config import load_config, with_overrides
from stylo.eval.dispatch import fit_estimator
from stylo.eval.lobo import make_factory, make_factory_for_ablation
from stylo.domain.work_weighting import (AblationConfig, FEATURE_STATE_ONLY_ABLATION, FULL_WB_ABLATION,
                                       LEGACY_ABLATION, RELATIVE_FW_ONLY_ABLATION, WORK_BALANCED,
                                       CHUNK_WEIGHTED_LEGACY)
from stylo.features.function_words import _ANALYZER, FunctionWordBlock
from stylo.features.work_vectorizer import analyzer_event_counts

CFG = load_config()
_TP = r"(?u)\b\w+\b"


def _has_spacy() -> bool:
    try:
        import spacy
        spacy.load(CFG.get_path("language.spacy_model", "ru_core_news_lg"))
        return True
    except Exception:
        return False


requires_spacy = pytest.mark.skipif(not _has_spacy(), reason="spaCy ru model not installed")


# ═══════════════════════ 1. FunctionWord four-corner grid (spaCy-free) ═══════════
# pooled-TF top-3 != work-DF top-3: 'x' has highest raw TF but one work only (pruned); 'c' in 2 works.
FW_TEXTS = ["x x x x x a", "x x x x x", "a a b", "a b", "a a b b c", "b", "a b b c", "c"]
FW_GROUPS = ["w1", "w1", "w2", "w2", "w3", "w3", "w4", "w4"]


def _fw(relative_fw, groups):
    b = FunctionWordBlock(mode="mfw", mfw_count=3, relative_fw=relative_fw)
    b.fit(FW_TEXTS, None, groups=(np.array(groups, dtype=object) if groups else None))
    X = b.transform(FW_TEXTS, None)
    vocab = [n.replace("fw::", "") for n in b.feature_names()]
    return b, X, vocab


def test_fw_grid_vocab_transform_and_denominator():
    baseline_block, baseline_X, baseline_vocab = _fw(None, None)       # A0 F0R0
    feature_fit_block, feature_fit_X, feature_fit_vocab = _fw(False, FW_GROUPS)  # A2 F1R0
    relative_block, relative_X, relative_vocab = _fw(True, None)       # A3 F0R1
    full_block, full_X, full_vocab = _fw(None, FW_GROUPS)              # A4 F1R1
    # F axis: A0==A3 pooled, A2==A4 work, pooled != work; independent pooled reference
    assert baseline_vocab == relative_vocab
    assert feature_fit_vocab == full_vocab
    assert set(baseline_vocab) != set(feature_fit_vocab)
    assert sorted(baseline_vocab) == sorted(
        CountVectorizer(max_features=3, lowercase=True, token_pattern=_TP)
        .fit(FW_TEXTS)
        .get_feature_names_out()
    )
    assert "x" in set(baseline_vocab) and "x" not in set(feature_fit_vocab)
    assert "c" in set(feature_fit_vocab) and "c" not in set(baseline_vocab)
    # R axis: A0/A2 raw integers; A3/A4 fractions == independent selected/all-events; sparse (no densify)
    for matrix in (baseline_X, feature_fit_X, relative_X, full_X):
        assert issparse(matrix)
    assert baseline_X.dtype == np.int64                                  # A0 raw CSR stays int64
    baseline_dense, feature_fit_dense, relative_dense, full_dense = (
        matrix.toarray() for matrix in (baseline_X, feature_fit_X, relative_X, full_X)
    )
    assert np.array_equal(baseline_dense, np.round(baseline_dense))
    assert np.array_equal(feature_fit_dense, np.round(feature_fit_dense))  # raw counts
    ev = analyzer_event_counts(_ANALYZER, FW_TEXTS)
    relative_ref = _sel_counts(FW_TEXTS, relative_vocab) / np.where(ev == 0, 1.0, ev)[:, None]
    full_ref = _sel_counts(FW_TEXTS, full_vocab) / np.where(ev == 0, 1.0, ev)[:, None]
    np.testing.assert_allclose(relative_dense, relative_ref, atol=1e-12)
    np.testing.assert_allclose(full_dense, full_ref, atol=1e-12)
    assert not np.array_equal(relative_dense, np.round(relative_dense))  # genuinely fractional
    assert (baseline_block.feature_fit_, baseline_block.relative_fw_) == (False, False)
    assert (feature_fit_block.feature_fit_, feature_fit_block.relative_fw_) == (True, False)
    assert (relative_block.feature_fit_, relative_block.relative_fw_) == (False, True)
    assert (full_block.feature_fit_, full_block.relative_fw_) == (True, True)


def _sel_counts(texts, vocab):
    return CountVectorizer(vocabulary=list(vocab), lowercase=True, token_pattern=_TP) \
        .transform(list(texts)).toarray().astype(np.float64)


def test_fw_transform_invariant_to_post_fit_relative_fw_mutation():
    # transform must read the FITTED R state, never the mutable constructor param -> mutating
    # relative_fw after fit cannot flip the branch or double-divide (fail-open regression).
    probe = ["kept oov oov oov"]
    two = ["w1", "w2"]                                          # 'kept' in both works -> survives work-DF
    for relfw, groups in [(None, None), (False, two), (True, None), (None, two)]:
        b = FunctionWordBlock("mfw", 1, relative_fw=relfw)
        b.fit(["kept kept", "kept"], None,
              groups=(np.array(groups, dtype=object) if groups else None))
        before = b.transform(probe, None).toarray().copy()
        for tamper in (None, True, False):
            b.relative_fw = tamper                          # hostile post-fit mutation
            np.testing.assert_array_equal(b.transform(probe, None).toarray(), before)
    # an OLD fw pickle (no relative_fw / effective-axes fields) predicts raw A0 or relative A4 by state
    old_a0 = FunctionWordBlock("mfw", 1)
    old_a0.fit(["kept kept", "kept"], None, groups=None)
    st = old_a0.__dict__.copy()
    for k in ("relative_fw", "relative_fw_", "feature_fit_"):
        st.pop(k, None)
    revived = FunctionWordBlock.__new__(FunctionWordBlock)
    revived.__setstate__(st)
    assert np.array_equal(revived.transform(probe, None).toarray(), np.array([[1.0]]))  # raw A0


def test_fw_R1_all_event_denominator_includes_oov_and_zero_row():
    b = FunctionWordBlock(mode="mfw", mfw_count=1, relative_fw=True)
    texts = ["kept oov oov oov", ""]                    # 'kept' selected; 3 OOV; then an empty chunk
    b.fit(["kept kept", "kept"], None, groups=None)      # vocab -> {'kept'}
    X = b.transform(texts, None).toarray()
    assert X[0, 0] == pytest.approx(0.25)                # kept / all-events(4), NOT kept/kept(1/1)
    assert X[1, 0] == 0.0 and np.isfinite(X).all()       # zero-event chunk -> zero row, no NaN


# a Russian-function-word panel (with content OOV) so the fixed_list vocabulary actually matches
FW_RU_TEXTS = ["и и в не на солнце", "но а и дом", "в в не окно поле", "и а но лес",
               "не не в сад", "а и стол"]
FW_RU_GROUPS = ["w1", "w1", "w2", "w2", "w3", "w3"]


def test_fw_fixed_list_R_axis_only():
    # a fixed vocabulary is inert to F (no work-DF pruning); only R0/R1 differ across the corners
    v0 = _fw_fixed(None, None)
    v0g = _fw_fixed(None, FW_RU_GROUPS)
    raw = _fw_fixed(False, FW_RU_GROUPS)
    rel = _fw_fixed(True, None)
    assert v0[2] == v0g[2] == raw[2] == rel[2]            # identical fixed vocabulary in all corners
    assert np.array_equal(raw[1], np.round(raw[1]))       # R0 raw counts
    assert not np.array_equal(rel[1], np.round(rel[1]))   # R1 fractional (function words matched)


def _fw_fixed(relative_fw, groups):
    b = FunctionWordBlock(mode="fixed_list", lang="ru", relative_fw=relative_fw)
    b.fit(FW_RU_TEXTS, None, groups=(np.array(groups, dtype=object) if groups else None))
    return b, b.transform(FW_RU_TEXTS, None).toarray(), [n.replace("fw::", "") for n in b.feature_names()]


# ═══════════════════════ 2. BoW A2 (spaCy-free) ═════════════════════════════════
def _bow_panel():
    texts = ["альфа бета альфа", "гамма альфа", "бета бета гамма", "дельта дельта альфа",
             "гамма гамма", "альфа бета"]
    y = np.array([0, 0, 0, 1, 1, 1])
    g = np.array(["a/1", "a/1", "a/2", "b/3", "b/3", "b/4"], dtype=object)
    return texts, y, g


def test_bow_A2_vocab_equals_A4_loss_is_A0():
    texts, y, g = _bow_panel()
    a0 = make_factory("bow_lr", CFG, weighting=CHUNK_WEIGHTED_LEGACY)()
    a2 = make_factory_for_ablation("bow_lr", CFG, ablation=FEATURE_STATE_ONLY_ABLATION)()
    a4 = make_factory("bow_lr", CFG, weighting=WORK_BALANCED)()
    for est in (a0, a2, a4):
        fit_estimator(est, texts, y, g)
    # A2 vocabulary/counts == A4 (work-level F); != A0 pooled
    assert tuple(a2.named_steps["bow"].get_feature_names_out()) == tuple(a4.named_steps["bow"].get_feature_names_out())
    assert np.array_equal(a2.named_steps["bow"].transform(texts).toarray(),
                          a4.named_steps["bow"].transform(texts).toarray())
    assert tuple(a2.named_steps["bow"].get_feature_names_out()) != tuple(a0.named_steps["bow"].get_feature_names_out())
    # loss == A0: class_weight 'balanced' (NOT None), no work weights (proba != A4 which uses None+weights)
    assert a2.named_steps["lr"].class_weight == "balanced" == a0.named_steps["lr"].class_weight
    assert a4.named_steps["lr"].class_weight is None
    assert not np.allclose(a2.predict_proba(texts), a4.predict_proba(texts))     # A0 loss != A4 loss


def test_bow_A2_rejects_reserved_and_A3_is_not_applicable():
    from stylo.domain.work_weighting import AblationNotApplicableError
    texts, y, g = _bow_panel()
    est = make_factory_for_ablation("bow_lr", CFG, ablation=FEATURE_STATE_ONLY_ABLATION)()
    for bad in ({"sample_weight": np.ones(len(y))}, {"lr__sample_weight": np.ones(len(y))},
                {"bow__groups": g}):
        e = make_factory_for_ablation("bow_lr", CFG, ablation=FEATURE_STATE_ONLY_ABLATION)()
        with pytest.raises(ValueError):
            e.fit(texts, y, groups=g, **bad)
    with pytest.raises(AblationNotApplicableError):
        make_factory_for_ablation("bow_lr", CFG, ablation=RELATIVE_FW_ONLY_ABLATION)


def test_A2_A3_applicability_matrix_and_typed_signals():
    import pickle
    from stylo.domain.work_weighting import AblationEquivalentError, AblationNotApplicableError
    # char_cos A2 -> equivalent to A4 (typed, pickle-safe, carries requested); A3 -> not applicable
    with pytest.raises(AblationEquivalentError) as ei:
        make_factory_for_ablation("char_cos", CFG, ablation=FEATURE_STATE_ONLY_ABLATION)
    assert ei.value.equivalent_to == "A4" and ei.value.requested == FEATURE_STATE_ONLY_ABLATION
    rt = pickle.loads(pickle.dumps(ei.value))
    assert rt.spec == "char_cos" and rt.equivalent_to == "A4" and rt.requested == FEATURE_STATE_ONLY_ABLATION
    with pytest.raises(AblationNotApplicableError):
        make_factory_for_ablation("char_cos", CFG, ablation=RELATIVE_FW_ONLY_ABLATION)
    # majority / bow_lr_ref_legacy: n/a for both A2 and A3, requested carried + pickle-safe
    for spec in ("majority", "bow_lr_ref_legacy"):
        for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
            with pytest.raises(AblationNotApplicableError) as e:
                make_factory_for_ablation(spec, CFG, ablation=ab)
            assert e.value.reason == "not_applicable" and e.value.requested == ab
            assert pickle.loads(pickle.dumps(e.value)).requested == ab
    # char/bow n/a carry exact spec + reason fields
    e = None
    try:
        make_factory_for_ablation("char_cos", CFG, ablation=RELATIVE_FW_ONLY_ABLATION)
    except AblationNotApplicableError as exc:
        e = exc
    assert e.spec == "char_cos" and e.reason == "not_applicable" and e.requested == RELATIVE_FW_ONLY_ABLATION
    # malformed / unknown Delta and unknown spec -> plain ValueError (not laundered into applicability)
    for spec in ("delta:bogus", "delta_cos:", "delta:0", "no_such_model"):
        for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
            with pytest.raises(ValueError) as ve:
                make_factory_for_ablation(spec, CFG, ablation=ab)
            assert not isinstance(ve.value, AblationNotApplicableError)
    # A2/A3 have NO production weighting enum (to_weighting stays corner-only)
    from stylo.domain.work_weighting import AblationNotImplementedError
    for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
        with pytest.raises(AblationNotImplementedError):
            ab.to_weighting()


def test_bow_A2_rejects_class_weight_tampering_all_paths():
    from sklearn.base import clone
    texts, y, g = _bow_panel()

    def a2():
        return make_factory_for_ablation("bow_lr", CFG, ablation=FEATURE_STATE_ONLY_ABLATION)()
    # the canonical A0 token is NOT a settable estimator param (set_params cannot reach it)
    with pytest.raises(ValueError):
        a2().set_params(expected_class_weight=None)
    # direct set_params tamper is rejected at fit
    d = a2(); d.set_params(lr__class_weight=None)
    with pytest.raises(ValueError):
        d.fit(texts, y, groups=g)
    # laundered through clone AND pickle (canonical token preserved, not re-derived from tampered steps)
    for launder in (lambda e: clone(e), lambda e: pickle.loads(pickle.dumps(e))):
        t = a2(); t.set_params(lr__class_weight=None)
        with pytest.raises(ValueError):
            launder(t).fit(texts, y, groups=g)
    # a forged numeric-subclass weight (whose __eq__ always returns True) cannot masquerade as A0
    class _FakeNum(float):
        def __eq__(self, o):
            return True

        def __hash__(self):
            return 0
    f = a2(); f.named_steps["lr"].class_weight = {0: _FakeNum(100.0), 1: _FakeNum(100.0)}
    with pytest.raises(ValueError):
        f.fit(texts, y, groups=g)
    # untampered clone / pickle still fit fine
    clone(a2()).fit(texts, y, groups=g)
    pickle.loads(pickle.dumps(a2())).fit(texts, y, groups=g)


def test_fw_all_corners_clone_joblib_and_old_pickle_roundtrip():
    import io

    import joblib
    from sklearn.base import clone
    probe = ["kept oov oov oov"]
    two = ["w1", "w2"]
    for relfw, groups in [(None, None), (False, two), (True, None), (None, two)]:
        b = FunctionWordBlock("mfw", 1, relative_fw=relfw)
        b.fit(["kept kept", "kept"], None, groups=(np.array(groups, dtype=object) if groups else None))
        want = b.transform(probe, None).toarray()
        # pickle + joblib round-trip preserve the fitted transform exactly
        assert np.array_equal(pickle.loads(pickle.dumps(b)).transform(probe, None).toarray(), want)
        buf = io.BytesIO(); joblib.dump(b, buf); buf.seek(0)
        assert np.array_equal(joblib.load(buf).transform(probe, None).toarray(), want)
        # clone is UNFITTED but preserves the constructor R policy
        cl = clone(b)
        assert cl.relative_fw == b.relative_fw and getattr(cl, "_vec", None) is None and getattr(cl, "_wv", None) is None
def _strip(block, keys, ctor=True):
    """Remove ``keys`` from the top-level state and (optionally) the _ctor_state snapshot."""
    st = {k: v for k, v in block.__dict__.items() if k not in keys}
    cs = dict(block.__dict__.get("_ctor_state") or {})
    if ctor:
        for k in keys:
            cs.pop(k, None)
    st["_ctor_state"] = cs
    return st


def test_fw_legacy_pickle_without_axis_fields_is_honest_and_stripped_current_artifact_is_rejected():
    fields = ("relative_fw", "relative_fw_", "feature_fit_")
    # A genuine compatibility-era A4 pickle lacks the explicit axis fields in BOTH the top-level
    # state AND _ctor_state; it must still predict relative (MODE_RELATIVE _wv). The corresponding
    # compatibility-era A0 pickle predicts raw.
    a4 = FunctionWordBlock("mfw", 1)
    a4.fit(["kept kept", "kept"], None, groups=np.array(["w1", "w2"], dtype=object))
    old4 = FunctionWordBlock.__new__(FunctionWordBlock); old4.__setstate__(_strip(a4, fields, ctor=True))
    g4 = old4.transform(["kept oov oov oov"], None).toarray()
    assert not np.array_equal(g4, np.round(g4))               # genuine old A4 -> relative
    a0 = FunctionWordBlock("mfw", 3); a0.fit(["и а но", "а и"], None, groups=None)
    old0 = FunctionWordBlock.__new__(FunctionWordBlock); old0.__setstate__(_strip(a0, fields, ctor=True))
    assert old0.transform(["и а"], None).dtype == np.int64    # genuine old A0 -> raw int
    # A current artifact whose fields are stripped ONLY at the top level (still in _ctor_state) is
    # corrupt, not a compatibility artifact: it must be rejected, not downgraded to legacy.
    a3 = FunctionWordBlock("mfw", 1, relative_fw=True)
    a3.fit(["kept kept", "kept"], None, groups=None)
    with pytest.raises(ValueError):
        FunctionWordBlock.__new__(FunctionWordBlock).__setstate__(_strip(a3, fields, ctor=False))


# ═══════════════════════ 3. stylo A2/A3 (spaCy) ═════════════════════════════════
def _cfg(tmp_path):
    return with_overrides(load_config(), {"paths.data": str(tmp_path / "c"),
                                          "paths.doc_cache": str(tmp_path / "c" / "dc")})


def _warm(cfg, texts):
    from stylo.features.reps import make_rep_cache
    make_rep_cache(cfg).warm(list(texts), n_process=1)


_VERBS = ["сидел", "смотрел", "думал", "стоял", "молчал", "ждал"]


def _make_panel(layout, tag):
    """capture-style texts: shared function words (на/и/в/у/а) in EVERY chunk survive work-DF pruning,
    author-specific nouns give a separable signal. ``layout`` = list of (label, work#, n_chunks)."""
    texts, y, groups = [], [], []
    for lab, wnum, n in layout:
        wid = f"{'ab'[lab]}/w{wnum}"
        u = f"{'ab'[lab]}{tag}{wnum}"
        for c in range(n):
            v1, v2 = _VERBS[c % 6], _VERBS[(c + 3) % 6]
            texts.append(f"{u.capitalize()} {v1} на {u}, и {v2} в {u}. {u.capitalize()} {v1} у {u} — а {u} {v2}!")
            y.append(lab)
            groups.append(wid)
    return np.array(texts, dtype=object), np.array(y), np.array(groups, dtype=object)


def _panel():
    return _make_panel([(0, 1, 3), (0, 2, 2), (1, 3, 3), (1, 4, 2)], "стил")


@contextlib.contextmanager
def _capture_block_routing():
    import stylo.models.channels as ch
    from stylo.features.work_vectorizer import WorkLevelVectorizer
    from stylo.vectorizer import StyloVectorizer
    rec = {"groups": [], "work_vec": 0, "work_sum": 0, "hashing_groups": []}
    o_sv, o_wv, o_ws = StyloVectorizer.fit, WorkLevelVectorizer.fit, ch._work_sum_matrix

    def sv(self, X, y=None, groups=None):                       # block channels: exact ORDERED groups
        rec["groups"].append(None if groups is None else tuple(np.asarray(groups).tolist()))
        return o_sv(self, X, y, groups)

    def wv(self, docs, groups):
        rec["work_vec"] += 1
        return o_wv(self, docs, groups)

    def ws(groups):                                             # hashing channels: exact ORDERED groups
        rec["work_sum"] += 1
        rec["hashing_groups"].append(tuple(np.asarray(groups).tolist()))
        return o_ws(groups)

    StyloVectorizer.fit, WorkLevelVectorizer.fit, ch._work_sum_matrix = sv, wv, ws
    try:
        yield rec
    finally:
        StyloVectorizer.fit, WorkLevelVectorizer.fit, ch._work_sum_matrix = o_sv, o_wv, o_ws


def _block_digests(vec):
    return {type(b).__name__: tuple(b.feature_names()) for b in vec.blocks}


def _block_matrices(vec, tx):
    X = vec.transform(tx).toarray()                              # populates block_slices() in block order
    return {type(b).__name__: X[:, lo:hi] for b, (nm, lo, hi) in zip(vec.blocks, vec.block_slices())}


@requires_spacy
def test_stylo_A2_A3_feature_projection_and_A0_loss(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel()
    _warm(cfg, tx)
    ests = {}
    recs = {}
    for name, ab in (("A0", LEGACY_ABLATION), ("A2", FEATURE_STATE_ONLY_ABLATION),
                     ("A3", RELATIVE_FW_ONLY_ABLATION), ("A4", FULL_WB_ABLATION)):
        e = make_factory_for_ablation("stylo", cfg, ablation=ab)()
        with _capture_block_routing() as rec:
            fit_estimator(e, tx, y, g)
        ests[name], recs[name] = e, rec
    vecs = {k: e.named_steps["vectorizer"] for k, e in ests.items()}
    dig = {k: _block_digests(v) for k, v in vecs.items()}
    mats = {k: _block_matrices(v, tx) for k, v in vecs.items()}       # per-block transform MATRICES
    # non-FW blocks: A3 == A0 (F0) and A2 == A4 (F1) — BIT-EXACT matrices (catches an R0<->R1 regression)
    for blk in dig["A0"]:
        if blk == "FunctionWordBlock":
            continue
        assert dig["A3"][blk] == dig["A0"][blk] and np.array_equal(mats["A3"][blk], mats["A0"][blk])
        assert dig["A2"][blk] == dig["A4"][blk] and np.array_equal(mats["A2"][blk], mats["A4"][blk])
    # FW block: A3 vocab == A0 (pooled), A2 vocab == A4 (work); transforms follow the R axis
    assert dig["A3"]["FunctionWordBlock"] == dig["A0"]["FunctionWordBlock"]
    assert dig["A2"]["FunctionWordBlock"] == dig["A4"]["FunctionWordBlock"]
    fw = "FunctionWordBlock"
    assert np.array_equal(mats["A0"][fw], np.round(mats["A0"][fw]))   # A0 raw
    assert np.array_equal(mats["A2"][fw], np.round(mats["A2"][fw]))   # A2 raw (work vocab)
    assert not np.array_equal(mats["A3"][fw], np.round(mats["A3"][fw]))  # A3 relative (pooled vocab)
    assert not np.array_equal(mats["A4"][fw], np.round(mats["A4"][fw]))  # A4 relative
    # routing: A0/A3 no groups to blocks + no work_vec; A2/A4 groups + work_vec
    for name in ("A0", "A3"):
        assert all(gr is None for gr in recs[name]["groups"]) and recs[name]["work_vec"] == 0
    for name in ("A2", "A4"):
        assert any(gr is not None for gr in recs[name]["groups"]) and recs[name]["work_vec"] > 0
    # loss == A0: LR class_weight is the A0 config value, and NOT the A4 None
    a0_cw = ests["A0"].named_steps["classifier"].class_weight
    assert ests["A2"].named_steps["classifier"].class_weight == a0_cw
    assert ests["A3"].named_steps["classifier"].class_weight == a0_cw
    assert ests["A4"].named_steps["classifier"].class_weight is None


@requires_spacy
def test_stylo_A2_A3_real_lr_fit_has_A0_loss(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel()
    _warm(cfg, tx)
    a0_cw = make_factory("stylo", CFG, weighting=CHUNK_WEIGHTED_LEGACY)().named_steps["classifier"].class_weight
    for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
        est = make_factory_for_ablation("stylo", cfg, ablation=ab)()
        clf = est.named_steps["classifier"]
        cap = {}
        orig = clf.fit

        def fit(X, y=None, sample_weight=None, _c=clf, _cap=cap, _o=orig, **kw):
            _cap["cw"], _cap["sw"] = _c.class_weight, sample_weight
            return _o(X, y, sample_weight=sample_weight, **kw)
        clf.fit = fit
        fit_estimator(est, tx, y, g)
        assert cap["cw"] == a0_cw and cap["sw"] is None          # A0 loss: legacy class_weight, no weights


@requires_spacy
def test_stylo_A3_disabled_function_words_fails_closed(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel()
    _warm(cfg, tx)
    # A3 with the FunctionWord consumer removed is a silent A0 no-op -> must fail closed
    with pytest.raises(ValueError):
        est = make_factory_for_ablation("stylo", cfg, ablation=RELATIVE_FW_ONLY_ABLATION,
                                        enabled_override={"function_words": False})()
        fit_estimator(est, tx, y, g)


@requires_spacy
def test_stylo_A2_sklearn_compat_and_routing_invariance(tmp_path):
    import sklearn
    from sklearn.base import clone
    cfg = _cfg(tmp_path)
    tx, y, g = _panel()
    _warm(cfg, tx)

    def proba(flag):
        with sklearn.config_context(enable_metadata_routing=flag):
            e = make_factory_for_ablation("stylo", cfg, ablation=FEATURE_STATE_ONLY_ABLATION)()
            fit_estimator(e, tx, y, g)
            return e, e.predict_proba(tx)
    e_off, p_off = proba(False)
    e_on, p_on = proba(True)
    assert list(e_off.classes_) == list(e_on.classes_)
    assert np.array_equal(p_off, p_on)                           # ambient routing cannot change estimand
    assert type(clone(e_off)) is type(e_off)
    buf = io.BytesIO()
    __import__("joblib").dump(e_off, buf)
    buf.seek(0)
    assert np.array_equal(__import__("joblib").load(buf).predict_proba(tx), p_off)


@requires_spacy
def test_stylo_A2_A3_reject_class_weight_tampering(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel()
    _warm(cfg, tx)
    for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
        est = make_factory_for_ablation("stylo", cfg, ablation=ab)()
        est.set_params(classifier__class_weight=None)          # tamper with the A0 loss
        with pytest.raises(ValueError):
            fit_estimator(est, tx, y, g)


# ═══════════════════════ 4. stack A2/A3 (spaCy) ═════════════════════════════════
@contextlib.contextmanager
def _capture_fw_relative_fw():
    """Record the ``relative_fw`` policy every FunctionWordBlock is constructed with (via the name used
    inside the registry) — proves the R axis is threaded to the sole FW channel at fit AND predict."""
    import stylo.features.registry as reg
    orig = reg.FunctionWordBlock
    seen = []

    class _Rec(orig):
        def __init__(self, *a, relative_fw=None, **kw):
            seen.append(relative_fw)
            super().__init__(*a, relative_fw=relative_fw, **kw)
    reg.FunctionWordBlock = _Rec
    try:
        yield seen
    finally:
        reg.FunctionWordBlock = orig
def _recorder(base, kind, sink):
    class _Rec(base):
        def fit(self, X, y=None, sample_weight=None, **kw):
            sink.append({"kind": kind, "class_weight": self.class_weight,
                         "sample_weight": None if sample_weight is None else np.asarray(sample_weight)})
            return super().fit(X, y, sample_weight=sample_weight, **kw)
    return _Rec


@contextlib.contextmanager
def _capture_stack():
    import stylo.models.stacked_clf as sc
    sink = []
    o_lr, o_svc = sc.LogisticRegression, sc.LinearSVC
    sc.LogisticRegression = _recorder(o_lr, "meta_lr", sink)
    sc.LinearSVC = _recorder(o_svc, "svc", sink)
    try:
        yield sink
    finally:
        sc.LogisticRegression, sc.LinearSVC = o_lr, o_svc


@contextlib.contextmanager
def _capture_calibrator_groups():
    # directly record the ``groups`` kwarg the stack passes to choose_calibrator (None iff W off)
    import stylo.models.stacked_clf as sc
    seen = []
    orig = sc.choose_calibrator

    def rec(oof, y, *a, groups=None, **kw):
        seen.append(groups)
        return orig(oof, y, *a, groups=groups, **kw)
    sc.choose_calibrator = rec
    try:
        yield seen
    finally:
        sc.choose_calibrator = orig


@contextlib.contextmanager
def _capture_fw_channel_matrices():
    # record every FunctionWord CHANNEL transform (raw R0 vs fractional R1), proving real matrices
    from stylo.features.function_words import FunctionWordBlock
    seen = []
    orig = FunctionWordBlock.transform

    def t(self, texts, reps):
        out = orig(self, texts, reps)
        seen.append(out.toarray())
        return out
    FunctionWordBlock.transform = t
    try:
        yield seen
    finally:
        FunctionWordBlock.transform = orig


def _inner_splits(cfg, y, g):
    from sklearn.model_selection import StratifiedGroupKFold
    k = (cfg.get_path("evaluation.stacking", {}) or {}).get("inner_folds", 3) or 3
    seed = cfg.get_path("seed", 42)
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), np.asarray(y), np.asarray(g, dtype=object)))


def _stack_panel():
    # 3 works/author (>=3 for the inner 3-fold CV), shared function words survive work-DF
    return _make_panel([(0, 1, 3), (0, 2, 2), (0, 3, 2), (1, 4, 3), (1, 5, 2), (1, 6, 2)], "стек")


@requires_spacy
def test_stack_A2_exact_group_and_R_routing_A3_strict_two_arg(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _stack_panel()
    _warm(cfg, tx)
    splits = _inner_splits(cfg, y, g)
    g_arr = np.asarray(g, dtype=object)
    fold_tuples = {tuple(g_arr[tr].tolist()) for tr, _ in splits}       # EXACT ordered fold-train arrays
    full_tuple = tuple(g_arr.tolist())                                  # EXACT ordered full-train array
    # -- A2 (F on): block AND hashing channels get EXACT ORDERED groups; fit sees fold-train arrays,
    #    predict the full-train array; FW channel emits raw (R0) matrices; R policy on fit AND predict.
    est2 = make_factory_for_ablation("stylo_stack", cfg, ablation=FEATURE_STATE_ONLY_ABLATION)()
    assert est2._feature_on and not est2._relative_fw_on and est2.ablation_ == FEATURE_STATE_ONLY_ABLATION
    with _capture_block_routing() as fit2, _capture_fw_channel_matrices() as fwm2fit, \
            _capture_fw_relative_fw() as fwr2:
        fit_estimator(est2, tx, y, g)                                   # FIT trace only
    with _capture_block_routing() as pred2, _capture_fw_channel_matrices() as fwm2pred:
        est2.predict_proba(tx)                                         # PREDICT trace only
    for cap, expect in ((fit2, fold_tuples), (pred2, {full_tuple})):
        block = [gr for gr in cap["groups"] if gr is not None]
        assert block and all(gr in expect for gr in block)             # exact ORDERED arrays (order+repeats)
        assert cap["hashing_groups"] and all(gr in expect for gr in cap["hashing_groups"])  # hashing too
    assert set(g for g in fit2["groups"] if g) == fold_tuples          # every fold-train array appears
    assert set(pred2["groups"] if pred2["groups"] else []) == {full_tuple}
    assert (fwm2fit + fwm2pred) and all(np.array_equal(m, np.round(m)) for m in fwm2fit + fwm2pred)  # R0
    assert fwr2 and all(v is False for v in fwr2)
    # -- A3 (F off): genuine strict 2-arg, no groups/work_vec/work_sum; FW channel emits relative (R1)
    est3 = make_factory_for_ablation("stylo_stack", cfg, ablation=RELATIVE_FW_ONLY_ABLATION)()
    assert not est3._feature_on and est3._relative_fw_on
    with _capture_block_routing() as rec3, _capture_fw_channel_matrices() as fw3, \
            _capture_fw_relative_fw() as fwr3:
        fit_estimator(est3, tx, y, g)
        est3.predict_proba(tx)
    assert all(gr is None for gr in rec3["groups"]) and rec3["work_vec"] == 0 and rec3["work_sum"] == 0
    assert fw3 and any(not np.array_equal(m, np.round(m)) for m in fw3)            # FW channel relative (R1)
    assert fwr3 and all(v is True for v in fwr3)


@requires_spacy
def test_stack_A3_uses_a_genuine_strict_two_arg_channel(tmp_path, monkeypatch):
    # a REAL strict-signature channel (exactly two positional params): A3 must call channels with two
    # args on fit AND predict, keeping old two-arg ChannelFns working (a 3rd arg would TypeError).
    import inspect

    import stylo.models.stacked_clf as sc
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=48, alternate_sign=False, norm=None)
    calls = []

    def strict_channel(train_texts, test_texts):              # EXACTLY two positional params
        calls.append((len(list(train_texts)), len(list(test_texts))))
        return hv.transform(list(train_texts)), hv.transform(list(test_texts))
    assert len(inspect.signature(strict_channel).parameters) == 2
    monkeypatch.setattr(sc, "make_channels", lambda cfg, relative_fw=None: {"strict": strict_channel})
    cfg = _cfg(tmp_path)
    tx, y, g = _stack_panel()
    est = make_factory_for_ablation("stylo_stack", cfg, ablation=RELATIVE_FW_ONLY_ABLATION)()
    fit_estimator(est, tx, y, g)                              # OOF inner-fold 2-arg calls
    proba = est.predict_proba(tx)                             # full-train 2-arg calls
    n = len(y)
    assert any(tr < n for tr, _ in calls) and any(tr == n for tr, _ in calls)
    assert proba.shape == (n, len(est.classes_)) and np.allclose(proba.sum(1), 1.0)


@requires_spacy
def test_stack_A2_A3_supervised_fits_have_A0_loss_and_calibrator_groups_none(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _stack_panel()
    _warm(cfg, tx)
    for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
        from stylo.models.channels import make_channels
        n_ch = len(make_channels(cfg))
        n_folds = len(_inner_splits(cfg, y, g))
        est = make_factory_for_ablation("stylo_stack", cfg, ablation=ab)()
        with _capture_stack() as sink, _capture_calibrator_groups() as cal_g:
            fit_estimator(est, tx, y, g)
            est.predict_proba(tx)
        assert sink
        for rec in sink:                                        # A0 loss on EVERY supervised fit
            assert rec["class_weight"] == "balanced"            # legacy, NOT None
            assert rec["sample_weight"] is None                 # W off: no work weights
        # EXACT ordering/multiplicity: OOF svc = n_ch*n_folds, full-train svc = n_ch (predict);
        # meta-CV = 3, plus a final meta-LR iff mode_ == "stacked". A disappeared meta-CV fails here.
        n_svc = sum(r["kind"] == "svc" for r in sink)
        n_meta = sum(r["kind"] == "meta_lr" for r in sink)
        assert n_svc == n_ch * n_folds + n_ch
        assert n_meta == (4 if est.mode_ == "stacked" else 3)
        # calibrator groups=None captured DIRECTLY for every channel (W off -> chunk-level selection)
        assert cal_g and all(gr is None for gr in cal_g)


@requires_spacy
def test_stack_A2_A3_provenance_single_source_and_serialization_gate(tmp_path):
    from stylo.models.stacked_clf import EvaluationOnlyEstimatorError

    cfg = _cfg(tmp_path)
    for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
        est = make_factory_for_ablation("stylo_stack", cfg, ablation=ab)()
        assert est.ablation_ == ab and est.ablation_.weights == est._weights_on
        with pytest.raises(AttributeError):
            est.ablation_ = FULL_WB_ABLATION
        with pytest.raises(EvaluationOnlyEstimatorError, match="evaluation-only"):
            pickle.dumps(est)
