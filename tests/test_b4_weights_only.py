"""B4-B increment 2: weights-only A1 (work-balanced LOSS on a strictly legacy A0 feature side).

A1 == ``AblationConfig(True, False, False)`` is runnable ONLY for the LR-family
(``stylo``/``bow_lr``/``stylo_stack``). This suite proves, on deterministic synthetic panels:

* **routing / negatives** — A1 builds for the three LR specs and fails closed (exact reason) for
  every non-LR model; the five remaining intermediates stay ``AblationNotImplementedError``; the
  increment-1 malformed/shadow/duck/subclass/None/non-bool defenses survive; reserved caller
  weight/group params are rejected; the production ``make_factory`` contract is unchanged;
* **feature side == A0** — vocabulary / IDF / feature names / block digests / the FunctionWord
  transform are the exact chunk-pooled A0 projection, no ``groups`` ever reach a block/vectorizer,
  and the projection differs from A4 where F/R actually change state;
* **weight side** — the REAL learner ``.fit`` calls (outer LR, stack SVC, stack meta-CV/final LR)
  get ``class_weight=None`` and exact fold-local work weights (sum ``W_train``, equal author mass,
  equal within-author work mass, fold-local ≠ global); no supervised branch is left unweighted
  (Platt/isotonic calibrator fits keep the signed B3 contract without ``sample_weight``);
* **non-false-green** — a strong-chunk-imbalance panel where A1 == A0 on features but A1 ≠ A0 on
  probability, so a broken W axis cannot pass;
* **stack corners** — an eligible panel confirms the group-disjoint calibration passport and the
  exact ordered fold-local weights of every SVC/meta-LR; a small panel confirms the
  disabled / equal / no-meta path; the stack feature side is proven legacy (2-arg channels).
"""
from __future__ import annotations

import contextlib
import hashlib

import numpy as np
import pytest

from stylo.config import load_config, with_overrides
from stylo.eval.dispatch import fit_estimator
from stylo.eval.lobo import make_factory, make_factory_for_ablation
from stylo.eval.work_weighting import (AblationConfig, AblationNotApplicableError,
                                       AblationNotImplementedError, CHUNK_WEIGHTED_LEGACY,
                                       FULL_WB_ABLATION, LEGACY_ABLATION, WEIGHTS_ONLY_ABLATION,
                                       WORK_BALANCED, work_sample_weights)

CFG = load_config()
SEED = CFG.get_path("seed", 42)


def _has_spacy() -> bool:
    try:
        import spacy
        spacy.load(CFG.get_path("language.spacy_model", "ru_core_news_lg"))
        return True
    except Exception:
        return False


_SPACY = _has_spacy()
requires_spacy = pytest.mark.skipif(not _SPACY, reason="spaCy ru model not installed")


# ── deterministic synthetic panels ─────────────────────────────────────────────
_VERBS = ["сидел", "смотрел", "думал", "стоял", "молчал", "ждал"]


def _panel(layout, tag):
    """(label, work#, n_chunks) rows -> distinguishable author-specific texts (no spaCy needed to
    build; blocks/BoW pick up the per-author vocabulary)."""
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


# strong chunk imbalance (8/1/1/1 vs 7/1/1/1): work weights are far from uniform. Used for the
# single-fit models (bow_lr, outer stylo LR) where extreme imbalance is the point.
RAGGED = [(0, 1, 8), (0, 2, 1), (0, 3, 1), (0, 4, 1), (1, 5, 7), (1, 6, 1), (1, 7, 1), (1, 8, 1)]
# well-conditioned for a 3-fold inner CV (6 works/class, chunk counts 6/3/2/2/2/2 so weights are still
# non-uniform) — mirrors the fixture's P1 eligible stack panel.
STACK_ELIGIBLE = ([(0, i + 1, n) for i, n in enumerate([6, 3, 2, 2, 2, 2])]
                  + [(1, i + 7, n) for i, n in enumerate([6, 3, 2, 2, 2, 2])])
# small (3 works/class) so a class falls entirely on one side of a calibration fold -> B3 disables
# calibration fail-closed (mirrors the fixture's P2 fail-disabled panel); channel inner-CV stays valid.
STACK_DISABLED = [(0, 1, 3), (0, 2, 2), (0, 3, 2), (1, 4, 3), (1, 5, 2), (1, 6, 2)]


def _cfg(tmp_path):
    cache = tmp_path / "cache"
    return with_overrides(load_config(), {"paths.data": str(cache),
                                          "paths.doc_cache": str(cache / "doc_cache")})


def _warm(cfg, texts):
    from stylo.features.reps import make_rep_cache
    make_rep_cache(cfg).warm(list(texts), n_process=1)


def _sha(items) -> str:
    return hashlib.sha256("|".join(map(str, items)).encode("utf-8")).hexdigest()


def _nrows(X):
    if X is None:
        return None
    if hasattr(X, "shape"):
        return int(X.shape[0])
    try:
        return len(X)
    except Exception:
        return None


# ── real-fit instrumentation (separate from the canonical capture tool) ─────────
def _recorder(base, kind, sink):
    class _Rec(base):
        def fit(self, X, y=None, sample_weight=None, **kw):
            sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy()
            sink.append({"kind": kind, "class_weight": self.class_weight,
                         "sample_weight": sw, "n_rows": _nrows(X)})
            return super().fit(X, y, sample_weight=sample_weight, **kw)
    return _Rec


@contextlib.contextmanager
def capture_stack_supervised():
    """Intercept ONLY the stack's own supervised learners (channel SVC + meta LR) via the module-level
    names bound inside ``stacked_clf``. Platt calibrator LRs live behind ``calibration``'s own import,
    so they are correctly NOT captured (their signed B3 contract has no sample_weight)."""
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
def capture_feature_routing():
    """Record the ``groups`` every ``StyloVectorizer.fit`` receives (block channels), count every
    ``WorkLevelVectorizer.fit``, and count every ``channels._work_sum_matrix`` (the hashing channels'
    work-level IDF path) — together these distinguish a legacy (A0/A1) feature side from a work-level
    (A4) one for BOTH channel kinds the stack uses."""
    import stylo.models.channels as ch
    from stylo.features.work_vectorizer import WorkLevelVectorizer
    from stylo.vectorizer import StyloVectorizer
    rec = {"stylo_groups": [], "work_vec_calls": 0, "work_sum_calls": 0}
    o_sv, o_wv, o_ws = StyloVectorizer.fit, WorkLevelVectorizer.fit, ch._work_sum_matrix

    def sv_fit(self, X, y=None, groups=None):
        rec["stylo_groups"].append(groups)
        return o_sv(self, X, y, groups)

    def wv_fit(self, docs, groups):
        rec["work_vec_calls"] += 1
        return o_wv(self, docs, groups)

    def ws(groups):
        rec["work_sum_calls"] += 1
        return o_ws(groups)

    StyloVectorizer.fit, WorkLevelVectorizer.fit, ch._work_sum_matrix = sv_fit, wv_fit, ws
    try:
        yield rec
    finally:
        StyloVectorizer.fit, WorkLevelVectorizer.fit, ch._work_sum_matrix = o_sv, o_wv, o_ws


def _wrap_learner(learner, sink):
    """Wrap a single pipeline learner instance's ``.fit`` (class_weight is read at call time, so the
    pipeline's forced ``class_weight=None`` is observed)."""
    orig = learner.fit

    def fit(X, y=None, sample_weight=None, **kw):
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy()
        sink.append({"class_weight": learner.class_weight, "sample_weight": sw, "n_rows": _nrows(X)})
        return orig(X, y, sample_weight=sample_weight, **kw)

    learner.fit = fit
    return learner


def _author_mass(weights, groups):
    mass = {}
    for g, w in zip(groups, weights):
        mass.setdefault(g.split("/", 1)[0], 0.0)
        mass[g.split("/", 1)[0]] += w
    return mass


def _work_mass(weights, groups):
    mass = {}
    for g, w in zip(groups, weights):
        mass.setdefault(g, 0.0)
        mass[g] += w
    return mass


def _assert_signed_weights(weights, y, groups):
    """The §3.1 invariants for the full-train weight vector."""
    W = len(set(groups.tolist()))
    A = len(set(y.tolist()))
    assert weights is not None
    assert np.isclose(weights.sum(), W)                         # sum == W_train
    am = _author_mass(weights, groups)
    assert all(np.isclose(v, W / A) for v in am.values())      # equal author mass W/A
    # equal within-author work mass
    by_author = {}
    for work, w in _work_mass(weights, groups).items():
        by_author.setdefault(work.split("/", 1)[0], []).append(w)
    for masses in by_author.values():
        assert all(np.isclose(m, masses[0]) for m in masses)


def _inner_folds(cfg) -> int:
    st = cfg.get_path("evaluation.stacking", {}) or {}
    return (st.get("inner_folds", 3) if hasattr(st, "get") else 3) or 3


def _fold_train_weights(y, groups, k):
    from sklearn.model_selection import StratifiedGroupKFold
    splits = StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups)
    return [work_sample_weights(y[tr], groups[tr]) for tr, _ in splits]


def _expected_stack_traces(cfg, y, groups, mode_):
    """The EXACT ORDERED (kind -> weight vector) sequences an A1 stack must emit — bound to specific
    folds, with pinned multiplicities. This defeats a fold permutation, a missing/extra SVC, and a
    disappeared meta-CV (any of which would slip past an unordered set-membership check)."""
    from stylo.models.channels import make_channels
    n_ch = len(make_channels(cfg))
    glob = work_sample_weights(y, groups)
    oof_w = _fold_train_weights(y, groups, _inner_folds(cfg))
    # SVC: per channel the inner-OOF train folds in split order, THEN one full-train refit per channel
    # (predict_proba). meta-LR: the 3-fold meta-CV in split order, PLUS a full-train final iff stacked.
    svc = [w for _ in range(n_ch) for w in oof_w] + [glob for _ in range(n_ch)]
    meta = _fold_train_weights(y, groups, 3) + ([glob] if mode_ == "stacked" else [])
    return svc, meta


# ═══════════════════════ 1. routing & negative tests (§6.1) ═════════════════════
def test_A1_builds_for_the_lr_family():
    for spec, cls in (("stylo", "WeightsOnlyStyloPipeline"), ("bow_lr", "WeightsOnlyBowPipeline"),
                      ("stylo_stack", "StackedChannelClassifier")):
        est = make_factory_for_ablation(spec, CFG, ablation=WEIGHTS_ONLY_ABLATION)()
        assert type(est).__name__ == cls
        assert getattr(est, "needs_groups", False) is True


def test_A1_non_lr_fails_closed_with_exact_reason_before_fit():
    cases = {"delta:500": "already_in_legacy", "delta_cos:500": "already_in_legacy",
             "char_cos": "not_applicable", "majority": "not_applicable",
             "bow_lr_ref_legacy": "not_applicable"}
    for spec, reason in cases.items():
        with pytest.raises(AblationNotApplicableError) as ei:
            make_factory_for_ablation(spec, CFG, ablation=WEIGHTS_ONLY_ABLATION)
        assert ei.value.reason == reason and ei.value.spec == spec


def test_A1_unknown_spec_is_a_value_error_not_applicability():
    with pytest.raises(ValueError) as ei:
        make_factory_for_ablation("no_such_model", CFG, ablation=WEIGHTS_ONLY_ABLATION)
    assert not isinstance(ei.value, AblationNotApplicableError)


def test_applicability_reason_is_validated_and_pickles():
    import pickle
    with pytest.raises(ValueError):
        AblationNotApplicableError("stylo", "bogus_reason")
    for r in ("already_in_legacy", "not_applicable"):
        e = AblationNotApplicableError("x", r)
        assert e.reason == r
        rt = pickle.loads(pickle.dumps(e))                    # survives a pickle round-trip (runner IPC)
        assert type(rt) is AblationNotApplicableError and rt.spec == "x" and rt.reason == r


def test_A1_invalid_delta_spec_is_rejected_not_masked_as_applicability():
    # a malformed delta spec must NOT be laundered into a clean already_in_legacy status
    for bad in ("delta:bogus", "delta:", "delta_cos:", "delta_cos:0", "delta:-3"):
        with pytest.raises(ValueError) as ei:
            make_factory_for_ablation(bad, CFG, ablation=WEIGHTS_ONLY_ABLATION)
        assert not isinstance(ei.value, AblationNotApplicableError)
    # a well-formed delta spec still fails closed as already-in-legacy
    with pytest.raises(AblationNotApplicableError) as ok:
        make_factory_for_ablation("delta:500", CFG, ablation=WEIGHTS_ONLY_ABLATION)
    assert ok.value.reason == "already_in_legacy"


def test_remaining_intermediates_still_not_implemented():
    # 010/001 became A2/A3 in increment 3; only 110/101/011 remain unwired
    for bad in [(True, True, False), (True, False, True), (False, True, True)]:
        with pytest.raises(AblationNotImplementedError):
            make_factory_for_ablation("stylo", CFG, ablation=AblationConfig(*bad))


def test_increment1_defenses_survive_for_A1_path():
    # exact-type / plain-bool / duck / subclass / None still rejected at the boundary
    class _Sub(AblationConfig):
        pass
    for bad in (_Sub(True, False, False), CHUNK_WEIGHTED_LEGACY, None, WORK_BALANCED):
        with pytest.raises(TypeError):
            make_factory_for_ablation("stylo", CFG, ablation=bad)
    corrupt = AblationConfig(True, False, False)
    object.__setattr__(corrupt, "weights", 1)                  # non-bool axis
    with pytest.raises(TypeError):
        make_factory_for_ablation("stylo", CFG, ablation=corrupt)
    # an A1 instance whose to_weighting is shadowed to WORK_BALANCED must NOT route to full-WB: the
    # factory re-reads the axes from a fresh instance via class descriptors, never the instance's.
    shadowed = AblationConfig(True, False, False)
    object.__setattr__(shadowed, "to_weighting", lambda: WORK_BALANCED)
    est = make_factory_for_ablation("stylo_stack", CFG, ablation=shadowed)()
    assert est.ablation_ == WEIGHTS_ONLY_ABLATION and est.training_weighting != WORK_BALANCED


def test_A1_stack_ablation_override_rejects_non_weights_only():
    from stylo.models.stacked_clf import StackedChannelClassifier
    for bad in (FULL_WB_ABLATION, LEGACY_ABLATION, AblationConfig(True, True, False)):
        with pytest.raises(ValueError):
            StackedChannelClassifier(CFG, ablation=bad)
    with pytest.raises(TypeError):
        StackedChannelClassifier(CFG, ablation="A1")


def test_A1_stack_rejects_split_brain_provenance():
    # ablation=A1 (legacy feature side) paired with training_weighting=work_balanced (full-WB label)
    # is a split-brain: W-on/F-off math under a full work-balanced claim. It must be rejected.
    from stylo.models.stacked_clf import StackedChannelClassifier
    with pytest.raises(ValueError) as ei:
        StackedChannelClassifier(CFG, training_weighting=WORK_BALANCED, ablation=WEIGHTS_ONLY_ABLATION)
    assert "work_balanced" in str(ei.value)
    # the legitimate A1 construction (legacy label + A1 axes) is accepted
    ok = StackedChannelClassifier(CFG, training_weighting=CHUNK_WEIGHTED_LEGACY, ablation=WEIGHTS_ONLY_ABLATION)
    assert ok.ablation_ == WEIGHTS_ONLY_ABLATION and ok.training_weighting != WORK_BALANCED


class _FlipStr(str):
    """A stateful str-subclass that passes a membership check as work_balanced but flips a later
    comparison — the exact constructor-time bypass that must be rejected fail-closed."""

    def __new__(cls):
        x = super().__new__(cls, WORK_BALANCED)
        x.calls = 0
        return x

    def __hash__(self):
        return hash(WORK_BALANCED)

    def __eq__(self, other):
        self.calls += 1
        return self.calls == 1


def test_A1_stack_rejects_stateful_str_subclass_label():
    from stylo.models.stacked_clf import StackedChannelClassifier
    # both the audit A1 path and the plain A0/A4 path must reject a polymorphic label object
    with pytest.raises(TypeError):
        StackedChannelClassifier(CFG, training_weighting=_FlipStr(), ablation=WEIGHTS_ONLY_ABLATION)
    with pytest.raises(TypeError):
        StackedChannelClassifier(CFG, training_weighting=_FlipStr())


def test_A1_stack_plain_str_corners_unchanged():
    from stylo.models.stacked_clf import StackedChannelClassifier
    cases = [(CHUNK_WEIGHTED_LEGACY, None, CHUNK_WEIGHTED_LEGACY, LEGACY_ABLATION),
             (WORK_BALANCED, None, WORK_BALANCED, FULL_WB_ABLATION),
             (CHUNK_WEIGHTED_LEGACY, WEIGHTS_ONLY_ABLATION, CHUNK_WEIGHTED_LEGACY, WEIGHTS_ONLY_ABLATION)]
    for tw, ab, exp_label, exp_axes in cases:
        est = StackedChannelClassifier(CFG, training_weighting=tw, ablation=ab)
        assert est.training_weighting == exp_label and est.ablation_ == exp_axes


def test_A1_stack_provenance_is_read_only_and_single_source():
    from stylo.models.stacked_clf import StackedChannelClassifier
    est = StackedChannelClassifier(CFG, ablation=WEIGHTS_ONLY_ABLATION)
    # (a) public provenance exactly matches the axes the math reads
    assert est.ablation_.weights == est._weights_on
    assert est.ablation_.feature_fit == est._feature_on
    assert est.training_weighting == (WORK_BALANCED if est.ablation_.is_full_wb_corner
                                      else CHUNK_WEIGHTED_LEGACY)
    # (b) ordinary assignment to either public provenance field fails closed
    for field in ("training_weighting", "ablation_"):
        with pytest.raises(AttributeError):
            setattr(est, field, WORK_BALANCED)


def test_A1_stack_pickle_preserves_readonly_single_source():
    import pickle
    from stylo.models.stacked_clf import StackedChannelClassifier
    for tw, ab in [(CHUNK_WEIGHTED_LEGACY, None), (WORK_BALANCED, None),
                   (CHUNK_WEIGHTED_LEGACY, WEIGHTS_ONLY_ABLATION)]:
        est = StackedChannelClassifier(CFG, training_weighting=tw, ablation=ab)
        rt = pickle.loads(pickle.dumps(est))
        assert rt.training_weighting == est.training_weighting and rt.ablation_ == est.ablation_
        # exactly ONE axis source survived; the public fields are NOT independent stored copies
        assert "_axes" in rt.__dict__
        assert "training_weighting" not in rt.__dict__ and "ablation_" not in rt.__dict__
        with pytest.raises(AttributeError):
            rt.ablation_ = FULL_WB_ABLATION


@pytest.mark.parametrize("spec,weight_param,group_param", [
    ("bow_lr", "lr__sample_weight", "bow__groups"),
    ("stylo", "classifier__sample_weight", "vectorizer__groups"),
])
def test_A1_wrapper_rejects_reserved_caller_params(spec, weight_param, group_param, tmp_path):
    if spec == "stylo" and not _SPACY:
        pytest.skip("spaCy ru model not installed")
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(STACK_DISABLED, "рес")
    if spec == "stylo":
        _warm(cfg, tx)
    for bad in ({"sample_weight": np.ones(len(y))}, {weight_param: np.ones(len(y))},
                {group_param: g}, {"scaler__groups": g}):
        est = make_factory_for_ablation(spec, cfg, ablation=WEIGHTS_ONLY_ABLATION)()
        with pytest.raises(ValueError):
            est.fit(tx, y, groups=g, **bad)


def test_production_make_factory_contract_unchanged():
    import inspect
    p = inspect.signature(make_factory).parameters
    assert p["weighting"].default is inspect._empty and p["weighting"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "ablation" not in p
    # A1 never routes into the production weighting enum
    with pytest.raises(AblationNotImplementedError):
        WEIGHTS_ONLY_ABLATION.to_weighting()


def test_A1_bow_is_sklearn_compatible_and_routing_invariant(tmp_path):
    import sklearn
    from sklearn.base import clone
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(RAGGED, "скл")
    est = make_factory_for_ablation("bow_lr", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    cloned = clone(est)                                         # get_params/clone survive the subclass
    assert type(cloned) is type(est) and "lr" in dict(est.get_params()["steps"])

    def _proba(flag):
        with sklearn.config_context(enable_metadata_routing=flag):
            e = make_factory_for_ablation("bow_lr", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
            fit_estimator(e, tx, y, g)
            return e, e.predict_proba(tx)

    e_off, p_off = _proba(False)
    e_on, p_on = _proba(True)
    assert list(e_off.classes_) == list(e_on.classes_)         # classes_ populated
    assert np.array_equal(p_off, p_on)                         # ambient routing mode cannot change the estimand
    # joblib round-trip preserves the fitted estimand
    import io
    import joblib
    buf = io.BytesIO()
    joblib.dump(e_off, buf)
    buf.seek(0)
    assert np.array_equal(joblib.load(buf).predict_proba(tx), p_off)


# ═══════════════════════ 2. feature-side proof: A1 == A0 (§6.2) ═════════════════
@requires_spacy
def test_A1_stylo_feature_projection_equals_A0(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(RAGGED, "приз")
    _warm(cfg, tx)

    a0 = make_factory(cfg=cfg, spec="stylo", weighting=CHUNK_WEIGHTED_LEGACY)()
    a1 = make_factory_for_ablation("stylo", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    a4 = make_factory(cfg=cfg, spec="stylo", weighting=WORK_BALANCED)()
    with capture_feature_routing() as rec0:
        fit_estimator(a0, tx, y, g)
    with capture_feature_routing() as rec1:
        fit_estimator(a1, tx, y, g)
    with capture_feature_routing() as rec4:
        fit_estimator(a4, tx, y, g)

    v0, v1, v4 = (e.named_steps["vectorizer"] for e in (a0, a1, a4))
    # (a) feature names / block digests / IDF-bearing transform identical to A0
    assert v1.feature_names() == v0.feature_names()
    d0 = {type(b).__name__: _sha(b.feature_names()) for b in v0.blocks}
    d1 = {type(b).__name__: _sha(b.feature_names()) for b in v1.blocks}
    assert d1 == d0
    X0, X1 = v0.transform(tx).toarray(), v1.transform(tx).toarray()
    assert np.array_equal(X1, X0)                              # full projection incl. FW raw counts + IDF
    # (b) the FunctionWord block specifically is raw-count A0: its sub-matrix is integer-valued
    fw_slice = next(((lo, hi) for nm, lo, hi in v1.block_slices() if nm == "function_words"), None)
    if fw_slice is not None:
        lo, hi = fw_slice
        assert np.array_equal(X1[:, lo:hi], np.round(X1[:, lo:hi]))   # raw counts, not relative fractions
    # (c) no groups reached any block/vectorizer on the A1 (or A0) feature side; no work_vec fitted
    assert all(gr is None for gr in rec1["stylo_groups"]) and rec1["work_vec_calls"] == 0
    assert all(gr is None for gr in rec0["stylo_groups"]) and rec0["work_vec_calls"] == 0
    # (d) A4 really moves feature state (work-level vocab/relative), so the equality above is non-trivial
    assert any(gr is not None for gr in rec4["stylo_groups"]) and rec4["work_vec_calls"] > 0
    assert v4.feature_names() != v1.feature_names()


def test_A1_bow_vocab_and_counts_equal_A0(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(RAGGED, "слов")
    a0 = make_factory(cfg=cfg, spec="bow_lr", weighting=CHUNK_WEIGHTED_LEGACY)()
    a1 = make_factory_for_ablation("bow_lr", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    a4 = make_factory(cfg=cfg, spec="bow_lr", weighting=WORK_BALANCED)()
    fit_estimator(a0, tx, y, g)
    fit_estimator(a1, tx, y, g)
    fit_estimator(a4, tx, y, g)
    bow0, bow1 = a0.named_steps["bow"], a1.named_steps["bow"]
    assert tuple(bow1.get_feature_names_out()) == tuple(bow0.get_feature_names_out())
    assert np.array_equal(bow1.transform(tx).toarray(), bow0.transform(tx).toarray())  # raw counts == A0
    # A4 uses a work-level count vocab -> feature set differs (equality above is a real constraint)
    assert tuple(a4.named_steps["bow"].get_feature_names_out()) != tuple(bow0.get_feature_names_out())


# ═══════════════════════ 3. weight-side proof (§6.3) ════════════════════════════
def test_A1_bow_outer_lr_exact_weights(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(RAGGED, "вес")
    est = make_factory_for_ablation("bow_lr", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    sink = []
    _wrap_learner(est.named_steps["lr"], sink)
    fit_estimator(est, tx, y, g)
    assert len(sink) == 1
    rec = sink[0]
    assert rec["class_weight"] is None                        # no double weighting
    np.testing.assert_array_equal(rec["sample_weight"], work_sample_weights(y, g))
    _assert_signed_weights(rec["sample_weight"], y, g)


@requires_spacy
def test_A1_stylo_outer_lr_exact_weights(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(RAGGED, "стил")
    _warm(cfg, tx)
    est = make_factory_for_ablation("stylo", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    sink = []
    _wrap_learner(est.named_steps["classifier"], sink)
    fit_estimator(est, tx, y, g)
    assert len(sink) == 1 and sink[0]["class_weight"] is None
    np.testing.assert_array_equal(sink[0]["sample_weight"], work_sample_weights(y, g))
    _assert_signed_weights(sink[0]["sample_weight"], y, g)


@requires_spacy
def test_A1_stack_supervised_fits_are_exact_ordered_fold_local(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(STACK_ELIGIBLE, "стек")
    _warm(cfg, tx)
    est = make_factory_for_ablation("stylo_stack", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    with capture_stack_supervised() as sink:
        fit_estimator(est, tx, y, g)
        est.predict_proba(tx)                                 # refits full-train channel SVCs
    svc = [r for r in sink if r["kind"] == "svc"]
    meta = [r for r in sink if r["kind"] == "meta_lr"]
    exp_svc, exp_meta = _expected_stack_traces(cfg, y, g, est.mode_)
    # EXACT multiplicities: a disappeared meta-CV, a missing/extra SVC, or a permuted fold fails here.
    assert len(svc) == len(exp_svc) and len(meta) == len(exp_meta)
    assert len(meta) >= 3, "meta-CV did not run on the calibration-eligible panel"
    glob = work_sample_weights(y, g)
    for got, exp in ((svc, exp_svc), (meta, exp_meta)):
        for rec, w in zip(got, exp):                          # ordered, bound to a specific fold
            assert rec["class_weight"] is None                # no double weighting
            assert rec["sample_weight"] is not None           # no unweighted supervised branch
            np.testing.assert_array_equal(rec["sample_weight"], w)
    # fold-local recomputation really differs from the global vector (not a global slice)
    assert any(not np.array_equal(r["sample_weight"], glob) for r in svc)
    _assert_signed_weights(glob, y, g)                        # the full-train vector is itself signed-correct


@requires_spacy
def test_A1_stack_feature_side_is_legacy_two_arg(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(STACK_ELIGIBLE, "sf")
    _warm(cfg, tx)
    a1 = make_factory_for_ablation("stylo_stack", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    with capture_feature_routing() as rec1:
        fit_estimator(a1, tx, y, g)
        a1.predict_proba(tx)                                  # predict refits channels too -> stay legacy
    # block channels: never a groups arg; hashing channels: never the work-sum IDF path; no work_vec
    assert rec1["stylo_groups"] and all(gr is None for gr in rec1["stylo_groups"])
    assert rec1["work_vec_calls"] == 0
    assert rec1["work_sum_calls"] == 0
    # A4 really routes work groups through BOTH channel kinds (so the A1 equalities above are real)
    a4 = make_factory(cfg=cfg, spec="stylo_stack", weighting=WORK_BALANCED)()
    with capture_feature_routing() as rec4:
        fit_estimator(a4, tx, y, g)
    assert any(gr is not None for gr in rec4["stylo_groups"])
    assert rec4["work_vec_calls"] > 0 and rec4["work_sum_calls"] > 0


def test_A1_stack_calls_a_genuine_strict_two_arg_channel(tmp_path, monkeypatch):
    # A REAL strict-signature channel (exactly two positional params, no *args, no optional third): the
    # instrumentation above shows work-level fitting is off, but a regression that passed None as a 3rd
    # arg would still satisfy it — this proves the A1 stack calls channels with EXACTLY two arguments,
    # keeping old two-arg ChannelFns working.
    import inspect

    import stylo.models.stacked_clf as sc
    from sklearn.feature_extraction.text import HashingVectorizer
    _hv = HashingVectorizer(n_features=48, alternate_sign=False, norm=None)
    calls = []

    def strict_channel(train_texts, test_texts):             # EXACTLY two positional params
        calls.append((len(list(train_texts)), len(list(test_texts))))
        return _hv.transform(list(train_texts)), _hv.transform(list(test_texts))

    params = list(inspect.signature(strict_channel).parameters.values())
    assert len(params) == 2 and all(p.kind == p.POSITIONAL_OR_KEYWORD for p in params)
    with pytest.raises(TypeError):                            # a third positional (even None) is rejected
        strict_channel(["a"], ["b"], None)

    monkeypatch.setattr(sc, "make_channels", lambda cfg: {"strict": strict_channel})
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(STACK_ELIGIBLE, "s2a")
    est = make_factory_for_ablation("stylo_stack", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    fit_estimator(est, tx, y, g)                             # inner-fold OOF channel calls (strict 2-arg)
    proba = est.predict_proba(tx)                            # full-train channel calls (strict 2-arg)
    n = len(y)
    assert any(tr < n for tr, _ in calls)                   # inner-fold calls happened (train < full)
    assert any(tr == n for tr, _ in calls)                  # full-train calls happened
    assert proba.shape == (n, len(est.classes_)) and np.allclose(proba.sum(axis=1), 1.0)
    # sanity: the SAME strict channel under A4 (feature axis ON) passes a 3rd arg -> TypeError, proving
    # the test would catch production ever routing groups into an A1 channel.
    est4 = make_factory(cfg=cfg, spec="stylo_stack", weighting=WORK_BALANCED)()
    with pytest.raises(TypeError):
        fit_estimator(est4, tx, y, g)


# ═══════════════════ 4. non-false-green scientific regression (§6.4) ════════════
def test_A1_bow_is_A0_features_but_not_A0_probability(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(RAGGED, "нфг")
    a0 = make_factory(cfg=cfg, spec="bow_lr", weighting=CHUNK_WEIGHTED_LEGACY)()
    a1 = make_factory_for_ablation("bow_lr", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    fit_estimator(a0, tx, y, g)
    fit_estimator(a1, tx, y, g)
    # features identical...
    assert np.array_equal(a1.named_steps["bow"].transform(tx).toarray(),
                          a0.named_steps["bow"].transform(tx).toarray())
    p0, p1 = a0.predict_proba(tx), a1.predict_proba(tx)
    # ...probability deterministic and NOT equal to A0 (a dead W axis would make this test fail)
    assert np.array_equal(a1.predict_proba(tx), p1)           # deterministic
    assert not np.allclose(p0, p1)                            # the W axis actually moved the model


# ═══════════════════════ 5. stack corners (§5) ═════════════════════════════════
@requires_spacy
def test_A1_stack_eligible_panel_has_group_disjoint_calibration(tmp_path):
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(STACK_ELIGIBLE, "elig")
    _warm(cfg, tx)
    est = make_factory_for_ablation("stylo_stack", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    fit_estimator(est, tx, y, g)
    assert est.passport_["calibration_disabled"] is False
    assert est.ablation_ == WEIGHTS_ONLY_ABLATION and est.training_weighting != WORK_BALANCED
    for passport in est.passport_["calibration"].values():
        assert passport.get("group_aware") is True           # W on -> group-aware selection


@requires_spacy
def test_A1_stack_disabled_calibration_panel_falls_back_to_equal_no_meta(tmp_path):
    # NB: not a literal singleton — a true 1-work class would break the channel inner-CV (a train fold
    # would see one class). This small 3-works/class panel disables calibration via the "a class is
    # absent from a calibration fold" branch (mirrors the fixture's P2 fail-disabled corner).
    import warnings
    cfg = _cfg(tmp_path)
    tx, y, g = _panel(STACK_DISABLED, "sing")
    _warm(cfg, tx)
    est = make_factory_for_ablation("stylo_stack", cfg, ablation=WEIGHTS_ONLY_ABLATION)()
    with warnings.catch_warnings():                           # tiny deliberate panel: ignore small-sample noise
        warnings.simplefilter("ignore")
        fit_estimator(est, tx, y, g)
        proba = est.predict_proba(tx)
    assert est.passport_["calibration_disabled"] is True
    assert est.mode_ == "equal" and est.meta_ is None
    assert np.allclose(proba.sum(axis=1), 1.0)
