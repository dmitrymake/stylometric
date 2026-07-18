"""B4-B increment 3: the Delta / DeltaCos F×R four-corner grid, proven against INDEPENDENT references.

W is already-in-legacy for Delta (equal-work centroids), so the only axes are F (pooled vs
work-rank/DF vocabulary) and R (train z-input = mean of per-chunk ``selected/Σselected`` vs per-work
``Σselected/Σall-events``; predict = per-chunk ``selected/Σselected`` vs ``selected/all-events``):

    A0 F0R0 pooled  · mean-of-ratios      A2 F1R0 work · mean-of-ratios
    A3 F0R1 pooled  · Σsel/Σall           A4 F1R1 work · Σsel/Σall

Everything is spaCy-free (Delta consumes raw text). The new A2/A3 corners are NOT self-goldens: each
is checked against an independent numpy recomputation of its own denominators, and against the derived
invariant ``A2 = A4-F + A0-R``, ``A3 = A0-F + independently-computed-R``.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from stylo.config import load_config
from stylo.eval.lobo import make_factory, make_factory_for_ablation
from stylo.eval.work_weighting import (AblationConfig, FEATURE_STATE_ONLY_ABLATION,
                                       RELATIVE_FW_ONLY_ABLATION, WEIGHTS_ONLY_ABLATION)
from stylo.models.delta import BurrowsDelta

CFG = load_config()
_TP = r"(?u)\b\w+\b"


def _sel_counts(texts, vocab):
    cv = CountVectorizer(vocabulary=list(vocab), lowercase=True, token_pattern=_TP)
    return cv.transform(list(texts)).toarray().astype(np.float64)


def _all_events(texts):
    an = CountVectorizer(lowercase=True, token_pattern=_TP).build_analyzer()
    return np.array([len(an(t)) for t in texts], dtype=np.float64)


def _ref_group_freqs(texts, groups, vocab, relative):
    """Independent per-work z-input row: R0 = mean over chunks of selected/Σselected; R1 =
    Σselected/Σall-events. Column order follows ``vocab`` (pass the model's feature_names)."""
    counts = _sel_counts(texts, vocab)
    order = list(dict.fromkeys(groups))
    rows = []
    for w in order:
        idx = [i for i, g in enumerate(groups) if g == w]
        if relative:
            ev = _all_events([texts[i] for i in idx]).sum()
            rows.append(counts[idx].sum(0) / (ev if ev > 0 else 1.0))
        else:
            chunk = counts[idx]
            tot = chunk.sum(1, keepdims=True)
            tot[tot == 0] = 1.0
            rows.append((chunk / tot).mean(0))
    return np.vstack(rows), order


def _ref_z_state(texts, y, groups, vocab, relative, metric):
    gf, order = _ref_group_freqs(texts, groups, vocab, relative)
    mean = gf.mean(0)
    std = gf.std(0)
    std[std == 0] = 1e-9
    z = (gf - mean) / std
    if metric == "cosine":
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    lbl = {g: int(y[i]) for i, g in enumerate(groups)}
    gy = np.array([lbl[w] for w in order])
    classes = np.unique(list(lbl.values()))
    cent = np.vstack([z[gy == c].mean(0) for c in classes])
    return mean, std, cent


def _fit(spec, ablation, texts, y, groups):
    est = make_factory_for_ablation(spec, CFG, ablation=ablation)()
    est.fit(list(texts), np.asarray(y), np.asarray(groups, dtype=object))
    return est


def _fit_corner_weighting(spec, weighting, texts, y, groups):
    est = make_factory(spec, CFG, weighting=weighting)()
    est.fit(list(texts), np.asarray(y), np.asarray(groups, dtype=object))
    return est


# ragged panel where pooled-TF top-3 differs from work-DF-ranked top-3: token 'x' has the highest raw
# corpus TF but occurs in ONE work only, so work-DF pruning (min_df_works=2) drops it, while 'c' (in
# two works) survives the work vocabulary but not the pooled top-3. -> pooled={a,b,x}, work={a,b,c}.
TEXTS = [
    "x x x x x a", "x x x x x",     # w1 (author 0): x=10, a=1  (x only here)
    "a a b", "a b",                 # w2 (author 0): a=3, b=2
    "a a b b c", "b",               # w3 (author 1): a=2, b=3, c=1
    "a b b c", "c",                 # w4 (author 1): a=1, b=2, c=2
]
Y = [0, 0, 0, 0, 1, 1, 1, 1]
GROUPS = ["w1", "w1", "w2", "w2", "w3", "w3", "w4", "w4"]


# ═══════════════════════ F axis: vocab(A0)==A3, vocab(A2)==A4 ══════════════════
@pytest.mark.parametrize("spec", ["delta:3", "delta_cos:3"])
def test_delta_vocab_is_the_F_axis(spec):
    a0 = _fit_corner_weighting(spec, "chunk_weighted_legacy", TEXTS, Y, GROUPS)
    a2 = _fit(spec, FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a3 = _fit(spec, RELATIVE_FW_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a4 = _fit_corner_weighting(spec, "work_balanced", TEXTS, Y, GROUPS)
    v0, v2, v3, v4 = (list(e.feature_names()) for e in (a0, a2, a3, a4))
    assert v0 == v3, "A0/A3 share the pooled vocabulary (F0)"
    assert v2 == v4, "A2/A4 share the work-rank/DF vocabulary (F1)"
    assert set(v0) != set(v2), "pooled vs work vocabulary must actually differ on this panel"
    # independent references for both vocabularies
    pooled = sorted(CountVectorizer(max_features=3, lowercase=True, token_pattern=_TP)
                    .fit(TEXTS).get_feature_names_out())
    assert sorted(v0) == pooled


# ═══════════════════════ R axis: exact train z-state ═══════════════════════════
@pytest.mark.parametrize("metric,spec", [("manhattan", "delta:4"), ("cosine", "delta_cos:4")])
def test_delta_R0_is_mean_of_selected_ratios(metric, spec):
    for ablation, weighting in [(None, "chunk_weighted_legacy"), (FEATURE_STATE_ONLY_ABLATION, None)]:
        est = (_fit_corner_weighting(spec, weighting, TEXTS, Y, GROUPS) if ablation is None
               else _fit(spec, ablation, TEXTS, Y, GROUPS))
        mean, std, cent = _ref_z_state(TEXTS, Y, GROUPS, est.feature_names(), relative=False, metric=metric)
        np.testing.assert_allclose(est.mean_, mean, atol=1e-12)
        np.testing.assert_allclose(est.std_, std, atol=1e-12)
        np.testing.assert_allclose(est.centroids_, cent, atol=1e-12)


@pytest.mark.parametrize("metric,spec", [("manhattan", "delta:4"), ("cosine", "delta_cos:4")])
def test_delta_R1_is_sum_selected_over_sum_all_events(metric, spec):
    for ablation, weighting in [(RELATIVE_FW_ONLY_ABLATION, None), (None, "work_balanced")]:
        est = (_fit_corner_weighting(spec, weighting, TEXTS, Y, GROUPS) if ablation is None
               else _fit(spec, ablation, TEXTS, Y, GROUPS))
        mean, std, cent = _ref_z_state(TEXTS, Y, GROUPS, est.feature_names(), relative=True, metric=metric)
        np.testing.assert_allclose(est.mean_, mean, atol=1e-12)
        np.testing.assert_allclose(est.std_, std, atol=1e-12)
        np.testing.assert_allclose(est.centroids_, cent, atol=1e-12)


def test_delta_A2_is_A4_F_plus_A0_R_and_A3_is_A0_F_plus_R1():
    # derived invariants (not self-goldens): A2 = A4 vocabulary + A0 mean-of-ratios; A3 = A0 vocabulary
    # + independently computed Σ/Σ. Proven by the reference using each corner's own feature_names.
    a2 = _fit("delta:4", FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a4 = _fit_corner_weighting("delta:4", "work_balanced", TEXTS, Y, GROUPS)
    assert list(a2.feature_names()) == list(a4.feature_names())          # A2 F == A4 F
    m2, s2, c2 = _ref_z_state(TEXTS, Y, GROUPS, a2.feature_names(), relative=False, metric="manhattan")
    np.testing.assert_allclose(a2.mean_, m2, atol=1e-12)                 # A2 R == A0 mean-of-ratios


# ═══════════════════════ denominators, zero rows, invariance ═══════════════════
def test_delta_oov_moves_R1_not_R0():
    # fixed vocab so F is inert; adding OOV tokens to a work changes only the R1 (all-event) denominator
    base = ["a a b", "b b a"]
    with_oov = ["a a b zzz zzz", "b b a"]
    g = ["w1", "w1"]
    y = [0, 0]

    def gf(texts, relative):
        est = BurrowsDelta(4, "manhattan",
                           vocabulary=["a", "b"], ablation=RELATIVE_FW_ONLY_ABLATION if relative
                           else FEATURE_STATE_ONLY_ABLATION)
        est.fit(list(texts), np.array(y), np.array(g, dtype=object))
        return est.mean_          # single work -> mean_ is that work's z-input row
    np.testing.assert_allclose(gf(base, False), gf(with_oov, False), atol=1e-12)      # R0 unchanged
    assert not np.allclose(gf(base, True), gf(with_oov, True))                        # R1 changed


def test_delta_R1_invariant_to_resplit_but_R0_changes():
    # same work token multiset, different chunk split: R1 (Σ/Σ) is invariant; R0 (mean-of-ratios) is not
    y = [0, 0]
    split_a = (["a a b", "a"], ["w", "w"])                # w = {a:3, b:1}
    split_b = (["a a", "a b"], ["w", "w"])                # same multiset {a:3, b:1}, re-split

    def z(texts_groups, relative):
        t, g = texts_groups
        est = BurrowsDelta(4, "manhattan", vocabulary=["a", "b"],
                           ablation=RELATIVE_FW_ONLY_ABLATION if relative else FEATURE_STATE_ONLY_ABLATION)
        est.fit(list(t), np.array(y), np.array(g, dtype=object))
        return est.mean_
    np.testing.assert_allclose(z(split_a, True), z(split_b, True), atol=1e-12)        # R1 invariant
    assert not np.allclose(z(split_a, False), z(split_b, False))                      # R0 segmentation-sensitive


def test_delta_zero_row_stays_a_work_vote():
    # a work with zero selected counts (R0) still contributes a (zero) z-input row -> centroid over works
    texts = ["a a", "b b", "zzz zzz"]      # w3 has no selected token
    y = [0, 0, 1]
    g = ["w1", "w2", "w3"]
    est = BurrowsDelta(4, "manhattan", vocabulary=["a", "b"], ablation=RELATIVE_FW_ONLY_ABLATION)
    est.fit(texts, np.array(y), np.array(g, dtype=object))
    assert est.centroids_.shape[0] == 2 and np.isfinite(est.centroids_).all()
    proba = est.predict_proba(texts)
    assert np.allclose(proba.sum(1), 1.0) and np.isfinite(proba).all()


# ═══════════════════════ predict, non-false-green, metrics ═════════════════════
def test_delta_predict_is_deterministic_and_valid():
    est = _fit("delta:4", FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    p1, p2 = est.predict_proba(TEXTS), est.predict_proba(TEXTS)
    assert np.array_equal(p1, p2)
    assert p1.shape == (len(TEXTS), 2) and np.allclose(p1.sum(1), 1.0)


def test_delta_corners_have_distinct_probabilities():
    a0 = _fit_corner_weighting("delta:4", "chunk_weighted_legacy", TEXTS, Y, GROUPS)
    a2 = _fit("delta:4", FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a3 = _fit("delta:4", RELATIVE_FW_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a4 = _fit_corner_weighting("delta:4", "work_balanced", TEXTS, Y, GROUPS)
    assert not np.allclose(a0.predict_proba(TEXTS), a3.predict_proba(TEXTS))   # R axis moves it
    assert not np.allclose(a2.predict_proba(TEXTS), a4.predict_proba(TEXTS))


def test_delta_manhattan_and_cosine_share_freq_and_meanstd():
    m = _fit("delta:4", RELATIVE_FW_ONLY_ABLATION, TEXTS, Y, GROUPS)
    c = _fit("delta_cos:4", RELATIVE_FW_ONLY_ABLATION, TEXTS, Y, GROUPS)
    np.testing.assert_allclose(m.mean_, c.mean_, atol=1e-12)          # frequency z-input identical
    np.testing.assert_allclose(m.std_, c.std_, atol=1e-12)
    assert not np.allclose(m.centroids_, c.centroids_)                # only the centroid L2 differs


def test_deltacos_centroid_is_mean_of_l2_normalized_work_z():
    est = _fit("delta_cos:4", RELATIVE_FW_ONLY_ABLATION, TEXTS, Y, GROUPS)
    mean, std, cent = _ref_z_state(TEXTS, Y, GROUPS, est.feature_names(), relative=True, metric="cosine")
    np.testing.assert_allclose(est.centroids_, cent, atol=1e-12)
    # the centroid itself is NOT re-normalized to unit norm (mean of unit vectors, generally < 1)
    norms = np.linalg.norm(est.centroids_, axis=1)
    assert (norms < 1.0 - 1e-9).any()


# ═══════════════════════ serialization / clone / refit / schema ════════════════
def test_delta_grid_pickle_refit_and_provenance():
    est = _fit("delta:4", FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    assert est._ablation == FEATURE_STATE_ONLY_ABLATION and est._schema_version == 3
    assert est.ablation_ == FEATURE_STATE_ONLY_ABLATION            # exact authoritative provenance
    # the production corners derive ablation_ from the weighting (A2/A3 carry the explicit cell)
    assert _fit_corner_weighting("delta:4", "chunk_weighted_legacy", TEXTS, Y, GROUPS).ablation_.is_legacy_corner
    assert _fit_corner_weighting("delta:4", "work_balanced", TEXTS, Y, GROUPS).ablation_.is_full_wb_corner
    rt = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(rt.predict_proba(TEXTS), est.predict_proba(TEXTS))
    assert rt._ablation == FEATURE_STATE_ONLY_ABLATION
    # refit onto a different corner-config instance leaves no stale _vec/_wv
    est.fit(list(TEXTS), np.array(Y), np.array(GROUPS, dtype=object))     # refit A2
    assert est._wv is not None and est._vec is None


def _revive(state):
    d = BurrowsDelta.__new__(BurrowsDelta)
    d.__setstate__(state)
    return d


def test_delta_schema_migration_is_version_aware_and_fail_closed():
    a0 = _fit_corner_weighting("delta:4", "chunk_weighted_legacy", TEXTS, Y, GROUPS)
    a2 = _fit("delta:4", FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    # v1/v2 (no _ablation) -> production corner, still predicts
    for v in (1, 2):
        s = a0.__dict__.copy(); s["_schema_version"] = v; s.pop("_ablation", None)
        r = _revive(s)
        assert r.__dict__["_ablation"] is None
        np.testing.assert_array_equal(r.predict_proba(TEXTS), a0.predict_proba(TEXTS))
    # v3 A2 round-trips exactly
    np.testing.assert_array_equal(_revive(a2.__dict__.copy()).predict_proba(TEXTS), a2.predict_proba(TEXTS))
    # v3 with _ablation STRIPPED is corrupt (Codex fail-open repro: an A2 loading as a corner)
    stripped = a2.__dict__.copy(); stripped.pop("_ablation")
    with pytest.raises(ValueError):
        _revive(stripped)
    # a3 for the A3-specific state branch (pooled _vec)
    a3 = _fit("delta:4", RELATIVE_FW_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a4 = _fit_corner_weighting("delta:4", "work_balanced", TEXTS, Y, GROUPS)
    np.testing.assert_array_equal(_revive(a3.__dict__.copy()).predict_proba(TEXTS), a3.predict_proba(TEXTS))
    # v3 COHERENCE state machine: every mismatched (ablation, label, vectorizer) tuple is rejected
    def mut(base, **kw):
        s = base.__dict__.copy(); s.update(kw); return s
    corrupt_cfg = AblationConfig(False, True, False)
    object.__setattr__(corrupt_cfg, "feature_fit", 1)                 # non-bool field
    bad_states = [
        mut(a2, _ablation=RELATIVE_FW_ONLY_ABLATION),                 # A2 fitted state under A3 provenance
        mut(a2, training_weighting="bogus"),                         # non-canonical label
        mut(a2, training_weighting=None),                           # None label
        mut(a2, _vec=object(), _wv=None),                          # A2 provenance over a pooled _vec
        mut(a4, _ablation=None, training_weighting="chunk_weighted_legacy"),  # A4 _wv under legacy label
        mut(a2, _ablation=corrupt_cfg),                            # corrupt non-bool ablation field
        mut(a2, _vec=object()),                                   # both _vec and _wv present (incoherent)
    ]
    for s in bad_states:
        with pytest.raises(ValueError):
            _revive(s)
    # a3 -> non-A2/A3 or wrong-typed _ablation rejected
    for bad_ab in ("A2", AblationConfig(True, True, True)):
        with pytest.raises(ValueError):
            _revive(mut(a2, _ablation=bad_ab))
    # v3 split-brain (A2 + work_balanced label) rejected at load
    with pytest.raises(ValueError):
        _revive(mut(a2, training_weighting="work_balanced"))
    # a v2 artifact with an INJECTED _ablation is forced back to a production corner (not kept)
    inj = mut(a0, _schema_version=2, _ablation=FEATURE_STATE_ONLY_ABLATION)
    assert _revive(inj).__dict__["_ablation"] is None
    # v1/v2 must ALSO validate the historically-allowed A0/A4 tuple — a bogus label or an opposite
    # label/vectorizer tuple is rejected, not silently accepted
    for v in (1, 2):
        with pytest.raises(ValueError):                              # bogus label
            _revive(mut(a0, _schema_version=v, training_weighting="bogus"))
        with pytest.raises(ValueError):                              # A4 _wv under a legacy label
            _revive(mut(a4, _schema_version=v, training_weighting="chunk_weighted_legacy"))
        with pytest.raises(ValueError):                              # A0 _vec under a work_balanced label
            _revive(mut(a0, _schema_version=v, training_weighting="work_balanced"))
    # future schema fail-closed
    with pytest.raises(ValueError):
        _revive(mut(a0, _schema_version=99))


class _FlipTW(str):
    def __new__(cls):
        from stylo.eval.work_weighting import WORK_BALANCED
        x = super().__new__(cls, WORK_BALANCED); x.c = 0; return x
    def __hash__(self):
        from stylo.eval.work_weighting import WORK_BALANCED
        return hash(WORK_BALANCED)
    def __eq__(self, o):
        self.c += 1; return self.c == 1


def test_delta_rejects_split_brain_stateful_str_and_post_construct_label():
    # constructor split-brain: work_balanced label + audit ablation
    for ab in (FEATURE_STATE_ONLY_ABLATION, RELATIVE_FW_ONLY_ABLATION):
        with pytest.raises(ValueError):
            BurrowsDelta(4, "manhattan", training_weighting="work_balanced", ablation=ab)
    # stateful str-subclass label rejected at construction
    with pytest.raises(TypeError):
        BurrowsDelta(4, "manhattan", training_weighting=_FlipTW(), ablation=FEATURE_STATE_ONLY_ABLATION)
    # a POST-construction label mutation is re-checked at fit and on provenance read
    est = BurrowsDelta(4, "manhattan", ablation=FEATURE_STATE_ONLY_ABLATION)
    est.training_weighting = "work_balanced"
    with pytest.raises(ValueError):
        est.fit(list(TEXTS), np.array(Y), np.array(GROUPS, dtype=object))
    with pytest.raises(ValueError):
        _ = est.ablation_
    # a legit A0, then a post-fit label flip -> rejected on provenance AND predict (not silently A4)
    a0 = _fit_corner_weighting("delta:4", "chunk_weighted_legacy", TEXTS, Y, GROUPS)
    a0.training_weighting = "work_balanced"
    with pytest.raises(ValueError):
        _ = a0.ablation_
    with pytest.raises(ValueError):
        a0.predict_proba(TEXTS)
    # non-A2/A3 or non-AblationConfig overrides rejected
    for bad in (WEIGHTS_ONLY_ABLATION, AblationConfig(True, True, True), AblationConfig(False, False, False)):
        with pytest.raises(ValueError):
            BurrowsDelta(4, "manhattan", ablation=bad)
    with pytest.raises(TypeError):
        BurrowsDelta(4, "manhattan", ablation="A2")


def test_delta_A2_A4_share_hand_computed_work_df_vocabulary():
    # HAND-COMPUTED expected F1 vocabulary (independent of the production WorkLevelVectorizer): 'x' is in
    # a single work -> pruned by min_df_works=2; only {a,b,c} occur in >=2 works, so the work vocab is
    # exactly {a,b,c} (alphabetical), while the pooled top-3 keeps the high-TF single-work 'x'.
    a2 = _fit("delta:3", FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a4 = _fit_corner_weighting("delta:3", "work_balanced", TEXTS, Y, GROUPS)
    assert list(a2.feature_names()) == ["a", "b", "c"] == list(a4.feature_names())
    assert "x" not in set(a2.feature_names()) and "x" in set(
        _fit_corner_weighting("delta:3", "chunk_weighted_legacy", TEXTS, Y, GROUPS).feature_names())


def test_delta_per_chunk_predict_frequency_is_exact():
    # predict frequency per chunk: A0/A2 selected/Σselected; A3/A4 selected/all-events (independent)
    a2 = _fit("delta:4", FEATURE_STATE_ONLY_ABLATION, TEXTS, Y, GROUPS)
    a3 = _fit("delta:4", RELATIVE_FW_ONLY_ABLATION, TEXTS, Y, GROUPS)
    c2 = _sel_counts(TEXTS, a2.feature_names())
    tot = c2.sum(1, keepdims=True); tot[tot == 0] = 1.0
    np.testing.assert_allclose(a2._rel_freq(list(TEXTS)), c2 / tot, atol=1e-12)      # R0
    c3 = _sel_counts(TEXTS, a3.feature_names())
    ev = _all_events(TEXTS)
    np.testing.assert_allclose(a3._rel_freq(list(TEXTS)), c3 / np.where(ev == 0, 1.0, ev)[:, None], atol=1e-12)
