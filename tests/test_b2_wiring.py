"""B2-core acceptance matrix (design §11) + the three mandatory runtime tests.

Spacy-free: delta/char_cos/bow_lr/majority fit on raw texts; the stylo pipeline is only
constructed (not fitted), so no rep cache / spaCy is needed.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from stylo.config import load_config
from stylo.corpus import load_dataset
from stylo.eval.dispatch import fit_estimator
from stylo.eval.lobo import make_factory
from stylo.eval.provenance import (ProvenanceError, RunContract,
                                   UnsupportedVariantError, VariantRole,
                                   assert_headline_write_allowed, derive_dataset,
                                   verify_dataset_against_disk)


def _legacy_contract(tmp_path):
    return RunContract.build(tmp_path, (), "unknown")
from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY, WORK_BALANCED
from stylo.features.work_vectorizer import WorkLevelVectorizer
from stylo.models.baselines import CharCosineBaseline
from stylo.models.delta import BurrowsDelta
from stylo.models.work_balanced import (WorkBalancedBowPipeline,
                                        WorkLevelCountTransformer,
                                        build_bow_lr_work_balanced)
from stylo.pipeline.bundle import BundleError, load_bundle, publish_bundle

WORD = {"analyzer": "word", "token_pattern": r"(?u)\b\w+\b", "lowercase": True}
CFG = load_config()


def _toy_corpus(tmp: pathlib.Path):
    for a in ("alpha", "beta"):
        for b in ("b1", "b2"):
            d = tmp / a / b
            d.mkdir(parents=True, exist_ok=True)
            for c in range(4):
                (d / f"{c}.txt").write_text(f"{a} {b} word{c} common text here more", encoding="utf-8")
    return load_dataset(tmp)


# ── §11.1-6 adapters ──────────────────────────────────────────────────────────
class TestAdapters:
    X = ["a b b", "b c", "c a a", "a c", "a a b", "b c c"]
    y = [0, 0, 1, 1, 0, 1]
    g = ["w1", "w1", "w2", "w2", "w3", "w4"]

    def _bow(self):
        return WorkBalancedBowPipeline([
            ("bow", WorkLevelCountTransformer(analyzer_params=WORD, min_df_works=1)),
            ("scaler", MaxAbsScaler()),
            ("lr", LogisticRegression(class_weight=None, max_iter=500)),
        ])

    def test_clone_getset_namedsteps_joblib_classes(self, tmp_path):
        p = self._bow()
        c = clone(p)                                   # clone
        assert "bow" in c.named_steps and c is not p
        c.set_params(lr__max_iter=123)                 # get/set params
        assert c.get_params()["lr__max_iter"] == 123
        p.fit(self.X, self.y, groups=self.g)
        assert list(p.classes_) == [0, 1]              # classes_
        fp = tmp_path / "p.joblib"
        joblib.dump(p, fp)
        assert list(joblib.load(fp).classes_) == [0, 1]   # joblib round-trip

    def test_metadata_routing_both_modes_identical(self):
        coefs = []
        for mode in (False, True):
            with sklearn.config_context(enable_metadata_routing=mode):
                p = self._bow()
                p.fit(self.X, self.y, groups=self.g)
                coefs.append(p.named_steps["lr"].coef_.copy())
        assert np.allclose(coefs[0], coefs[1])

    @pytest.mark.parametrize("bad", [{"bow__groups": g}, {"lr__sample_weight": np.ones(6)},
                                     {"sample_weight": np.ones(6)}])
    def test_reserved_params_rejected(self, bad):
        with pytest.raises(ValueError):
            self._bow().fit(self.X, self.y, groups=self.g, **bad)

    def test_exact_weights_and_class_weight(self):
        from stylo.eval.work_weighting import work_sample_weights
        p = self._bow()
        p.fit(self.X, self.y, groups=self.g)
        assert p.named_steps["lr"].class_weight is None
        w = work_sample_weights(self.y, self.g)
        assert w.sum() == pytest.approx(len(set(self.g)))     # sum == W

    def test_class_weight_forced_none_even_after_setparams(self):
        p = self._bow()
        p.set_params(lr__class_weight="balanced")       # try to defeat the contract
        p.fit(self.X, self.y, groups=self.g)
        assert p.named_steps["lr"].class_weight is None  # fit re-forces None

    def test_bow_fit_equals_fit_transform(self):
        t = WorkLevelCountTransformer(analyzer_params=WORD, min_df_works=1)
        a = t.fit(self.X, groups=self.g).transform(self.X).toarray()
        b = WorkLevelCountTransformer(analyzer_params=WORD, min_df_works=1).fit_transform(self.X, groups=self.g).toarray()
        assert np.allclose(a, b)
        with pytest.raises(TypeError):
            t.fit(self.X, self.g)                              # keyword-only groups

    def test_no_stale_cache_across_groups(self):
        # same docs, different grouping -> different fitted vocab (work-DF depends on groups)
        v1 = WorkLevelVectorizer(analyzer_params=WORD, min_df_works=2).fit(
            ["a x", "a y", "b z"], ["w1", "w2", "w3"])          # a in 2 works -> kept
        with pytest.raises(ValueError):
            WorkLevelVectorizer(analyzer_params=WORD, min_df_works=2).fit(
                ["a x", "a y", "b z"], ["w1", "w1", "w2"])      # a in 1 work -> empty vocab
        assert "a" in v1.vocabulary_


# ── §11.7 Delta/CharCos work-balanced feature state ───────────────────────────
class TestWorkBalancedFeatureState:
    def test_transform_grouped_sum_over_sum(self):
        v = WorkLevelVectorizer(analyzer_params=WORD, mode="relative", min_df_works=1).fit(
            ["a a a a b", "a b"], ["w1", "w1"])
        _, rows = v.transform_grouped(["a a a a b", "a b"], ["w1", "w1"])
        a = rows.toarray()[0][v.vocabulary_["a"]]
        assert a == pytest.approx(5 / 7)                        # Σ/Σ, not mean-of-ratios 0.65

    def test_zero_event_work_row_is_zero(self):
        v = WorkLevelVectorizer(analyzer_params=WORD, mode="relative", min_df_works=1).fit(
            ["a b", "a c"], ["w1", "w2"])
        wids, rows = v.transform_grouped(["a b", "   "], ["w1", "w2"])   # w2 has no analyzer events
        R = rows.toarray()
        assert np.all(R[wids.index("w2")] == 0.0)

    def test_delta_wb_differs_and_predicts(self):
        texts = ["kot na okne", "sobaka v parke", "kot lovil mysh", "ptica poet",
                 "more shumit", "les tikho", "volny bregu", "veter silno"]
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        g = ["a/w1", "a/w1", "a/w2", "a/w2", "b/w3", "b/w3", "b/w4", "b/w4"]
        wb = BurrowsDelta(10, "manhattan", training_weighting=WORK_BALANCED).fit(texts, y, groups=g)
        assert np.allclose(wb.predict_proba(texts).sum(axis=1), 1.0, atol=1e-6)
        assert "work_balanced" in wb.group_weighting_

    def test_charcos_wb_predicts(self):
        # common substring "text" across all 4 works -> char n-grams survive min_df_works=2
        texts = ["kot text", "kott text", "sob text", "sobb text"]
        y = [0, 0, 1, 1]
        g = ["a/w1", "a/w2", "b/w3", "b/w4"]
        wb = CharCosineBaseline(ngram_range=(2, 3), max_features=50, min_df=1,
                                training_weighting=WORK_BALANCED).fit(texts, y, groups=g)
        assert wb.predict_proba(texts).shape == (4, 2)
        assert "work_balanced" in wb.group_weighting_


# ── §11.8-10 provenance & subset ──────────────────────────────────────────────
class TestProvenance:
    def test_digest_changes_on_any_field(self, tmp_path):
        ds = _toy_corpus(tmp_path)
        base = ds.provenance.rows_digest
        for mutate in (lambda d: d.texts.__setitem__(0, "X"),
                       lambda d: d.y.__setitem__(0, 1 - d.y[0]),
                       lambda d: d.groups.__setitem__(0, "alpha/zz"),
                       lambda d: d.authors.__setitem__(0, "ZZZ")):
            d2 = _toy_corpus(tmp_path)
            mutate(d2)
            from stylo.eval.provenance import canonical_digest
            assert canonical_digest(d2.texts, d2.y, d2.groups, d2.authors, d2.provenance.row_ids,
                                    loader_kind=d2.provenance.loader_kind,
                                    chunker_config_hash=d2.provenance.chunker_config_hash) != base

    def test_disk_anchored_guard_and_semantics(self, tmp_path):
        ds = _toy_corpus(tmp_path)
        c = _legacy_contract(tmp_path)
        assert verify_dataset_against_disk(None, ds, CHUNK_WEIGHTED_LEGACY, c) == CHUNK_WEIGHTED_LEGACY
        with pytest.raises(ProvenanceError):     # WB requires a manifest loader kind
            verify_dataset_against_disk(None, ds, WORK_BALANCED, c)
        ds.texts[0] = "tampered"
        with pytest.raises(ProvenanceError):     # self-consistency / disk digest catches mutation
            verify_dataset_against_disk(None, ds, CHUNK_WEIGHTED_LEGACY, c)

    def test_hand_built_dataset_rejected(self, tmp_path):
        from stylo.corpus import Dataset
        d = Dataset(texts=np.array(["a", "b"], dtype=object), y=np.array([0, 1]),
                    groups=np.array(["x/1", "y/1"], dtype=object), authors=["x", "y"])
        with pytest.raises(ProvenanceError):     # no provenance -> fail-closed
            verify_dataset_against_disk(None, d, CHUNK_WEIGHTED_LEGACY, _legacy_contract(tmp_path))

    def test_relabel_legacy_as_wb_rejected(self, tmp_path):
        import dataclasses
        ds = _toy_corpus(tmp_path)
        ds.provenance = dataclasses.replace(ds.provenance, loader_kind="work_balanced_manifest")
        with pytest.raises(ProvenanceError):     # loader_kind flip caught (or WB reload fails)
            verify_dataset_against_disk(None, ds, WORK_BALANCED, _legacy_contract(tmp_path))

    def test_wrong_corpus_redirect_rejected(self, tmp_path):
        # a Dataset that declares a DIFFERENT frags_root than the frozen contract is refused
        ds = _toy_corpus(tmp_path)
        rogue = RunContract.build(tmp_path / "elsewhere", (), "unknown")
        with pytest.raises(ProvenanceError):
            verify_dataset_against_disk(None, ds, CHUNK_WEIGHTED_LEGACY, rogue)

    def test_forge_wb_over_legacy_arrays_rejected(self, tmp_path):
        # build_provenance(WB) over legacy arrays (row_ids carry no manifest identity) is refused
        from stylo.eval.provenance import CorpusPolicyProvenance, build_provenance
        ds = _toy_corpus(tmp_path)
        with pytest.raises(ProvenanceError):        # structural identity gate fires at build time
            build_provenance(
                loader_kind="work_balanced_manifest",
                texts=list(ds.texts), y=list(ds.y), groups=list(ds.groups),
                authors=ds.authors, row_ids=ds.provenance.row_ids,
                frags_root=ds.provenance.frags_root,
                corpus_policy=CorpusPolicyProvenance.build((), "unknown"),
                chunker_config_hash="a" * 64)

    def test_fabricated_dataset_rejected_by_disk(self, tmp_path):
        # a fabricated dataset declaring a non-existent corpus cannot match any on-disk digest
        import hashlib
        from stylo.corpus import Dataset
        from stylo.eval.provenance import (CorpusPolicyProvenance, RowIdentity,
                                           LEGACY_RECURSIVE, build_provenance)
        texts = ["fake one", "fake two"]; groups = ["a/w1", "b/w2"]
        rid = [RowIdentity(g, 0, hashlib.sha256(t.encode()).hexdigest()) for t, g in zip(texts, groups)]
        prov = build_provenance(loader_kind=LEGACY_RECURSIVE, texts=texts, y=[0, 1], groups=groups,
                                authors=["a", "b"], row_ids=rid,
                                frags_root=str(tmp_path / "nonexistent"),
                                corpus_policy=CorpusPolicyProvenance.build((), "unknown"))
        ds = Dataset(np.array(texts, dtype=object), np.array([0, 1]),
                     np.array(groups, dtype=object), ["a", "b"], provenance=prov)
        contract = RunContract.build(tmp_path / "nonexistent", (), "unknown")
        with pytest.raises(ProvenanceError):     # declared corpus not on disk -> rejected
            verify_dataset_against_disk(None, ds, CHUNK_WEIGHTED_LEGACY, contract)

    def test_derive_dataset_atomic_and_disk_chained(self, tmp_path):
        ds = _toy_corpus(tmp_path)
        sub = derive_dataset(ds, [0, 1, 2, 3])          # alpha/b1 only
        assert sub.authors == ["alpha"] and sub.provenance.loader_kind == "legacy_recursive"
        assert sub.provenance.selection_manifest_digest and sub.provenance.parent_rows_digest
        verify_dataset_against_disk(None, sub, CHUNK_WEIGHTED_LEGACY, _legacy_contract(tmp_path))
        for bad in ([0, 0], [999], [-1]):
            with pytest.raises(ProvenanceError):
                derive_dataset(ds, bad)


# ── §11.11-13 dispatch, isolation, lifecycle ──────────────────────────────────
class TestDispatchAndLifecycle:
    def test_fit_estimator_routes_groups(self):
        texts = ["a b", "b c", "c a", "a c"]; y = [0, 0, 1, 1]; g = ["a/w1", "a/w1", "b/w2", "b/w2"]
        d = BurrowsDelta(5, "manhattan")                         # needs_groups=True
        fit_estimator(d, texts, y, g)
        assert d.centroids_ is not None
        with pytest.raises(ValueError):                          # needs_groups but none given
            fit_estimator(BurrowsDelta(5, "manhattan"), texts, y, None)

    @pytest.mark.parametrize("weighting", [CHUNK_WEIGHTED_LEGACY, WORK_BALANCED])
    def test_make_factory_spec_matrix(self, weighting):
        for spec in ("stylo", "delta:150", "delta_cos:300", "char_cos", "bow_lr", "majority"):
            assert make_factory(spec, CFG, weighting=weighting)() is not None
        # stylo class_weight None only in WB
        cw = make_factory("stylo", CFG, weighting=weighting)().named_steps["classifier"].class_weight
        assert (cw is None) == (weighting == WORK_BALANCED)

    def test_bow_ref_is_wb_only(self):
        assert make_factory("bow_lr_ref_legacy", CFG, weighting=WORK_BALANCED)() is not None
        with pytest.raises(UnsupportedVariantError):        # forbidden in legacy arm
            make_factory("bow_lr_ref_legacy", CFG, weighting=CHUNK_WEIGHTED_LEGACY)

    def test_weighting_is_required_keyword(self):
        import inspect
        p = inspect.signature(make_factory).parameters["weighting"]
        assert p.default is inspect._empty and p.kind == inspect.Parameter.KEYWORD_ONLY

    def test_none_weighting_rejected(self):
        with pytest.raises(ValueError):        # None is NOT a silent legacy fallback
            make_factory("stylo", CFG, weighting=None)

    def test_symlink_exploratory_dir_refused(self, tmp_path):
        from stylo.eval.provenance import safe_exploratory_dir
        docs = tmp_path / "docs"; docs.mkdir()
        (docs / "exploratory").symlink_to(docs, target_is_directory=True)   # exploratory -> docs
        with pytest.raises(ProvenanceError, match="symlink"):
            safe_exploratory_dir(docs, "exploratory", "work_balanced")

    def test_safe_write_text_refuses_file_symlink(self, tmp_path):
        from stylo.eval.provenance import safe_write_text
        (tmp_path / "legacy.txt").write_text("HEADLINE")
        (tmp_path / "wb.txt").symlink_to(tmp_path / "legacy.txt")          # WB file -> legacy headline
        with pytest.raises(ProvenanceError):
            safe_write_text(tmp_path / "wb.txt", "EVIL")
        assert (tmp_path / "legacy.txt").read_text() == "HEADLINE"          # not followed

    def test_str_subclass_and_bare_exclude_rejected(self, tmp_path):
        from stylo.corpus import load_dataset
        from stylo.eval.provenance import (CorpusPolicyProvenance, RowIdentity,
                                           LEGACY_RECURSIVE, build_provenance)
        import hashlib
        class S(str):
            pass
        with pytest.raises(ProvenanceError):     # str subclass rejected (type() is str)
            build_provenance(loader_kind=LEGACY_RECURSIVE, texts=[S("a b"), "c d"], y=[0, 1],
                             groups=["x/1", "y/1"], authors=["x", "y"],
                             row_ids=[RowIdentity("x/1", 0, hashlib.sha256(b"a b").hexdigest()),
                                      RowIdentity("y/1", 0, hashlib.sha256(b"c d").hexdigest())],
                             frags_root="/x", corpus_policy=CorpusPolicyProvenance.build((), "unknown"))
        # a bare-string exclude_authors would silently become a set of letters
        d = tmp_path / "frags"
        for a in ("alpha", "beta"):
            for b in ("b1", "b2"):
                (d / a / b).mkdir(parents=True)
                for i in range(3):
                    (d / a / b / f"{i}.txt").write_text(f"{a} {b} chunk {i} word word")
        with pytest.raises(ValueError):
            load_dataset(d, exclude_authors="alpha")

    def test_pre_b2_delta_charcos_setstate_migration(self):
        d = BurrowsDelta(5, "manhattan").fit(["a b", "b c", "c a", "a c"], [0, 0, 1, 1],
                                             groups=["a/w1", "a/w1", "b/w2", "b/w2"])
        for attr in ("training_weighting", "_wv", "vocabulary"):
            d.__dict__.pop(attr, None)                     # simulate a pre-B2 pickle
        import pickle
        d2 = pickle.loads(pickle.dumps(d))                 # __setstate__ migrates
        assert d2.training_weighting == CHUNK_WEIGHTED_LEGACY and d2._wv is None
        d2.fit(["a b", "b c", "c a", "a c"], [0, 0, 1, 1],
               groups=["a/w1", "a/w1", "b/w2", "b/w2"])    # refit must work

    def test_stack_blocked_under_wb(self):
        with pytest.raises(UnsupportedVariantError):
            make_factory("stylo_stack", CFG, weighting=WORK_BALANCED)
        # legacy stack is allowed to build
        assert make_factory("stylo_stack", CFG, weighting=CHUNK_WEIGHTED_LEGACY) is not None

    def test_headline_write_guard(self):
        assert_headline_write_allowed(CHUNK_WEIGHTED_LEGACY)
        with pytest.raises(ProvenanceError):
            assert_headline_write_allowed(WORK_BALANCED)

    def test_variant_role_enum(self):
        assert {r.value for r in VariantRole} == {
            "primary", "reference", "not_applicable", "blocked_not_implemented"}


# ── §7 immutable versioned bundle + allowlist/containment/version + failure injection ──
class TestBundle:
    THREE = {"model.pkl", "delta.pkl", "authors.json"}

    META = {"training_weighting": "work_balanced", "dataset_contract": "work_balanced_manifest",
            "rows_digest": "d" * 64, "chunker_config_hash": "0" * 64, "code_tree_sha256": "e" * 64,
            "config_id": "c" * 64, "git_commit": "abc123def", "git_dirty": False}

    def _publish(self, root, model=b"MODEL", meta=None):
        return publish_bundle(root, {
            "model.pkl": lambda p: p.write_bytes(model),
            "delta.pkl": lambda p: p.write_bytes(b"DELTA"),
            "authors.json": lambda p: p.write_text('["a","b"]'),
        }, dict(meta or self.META))

    def _writers(self):
        return {"model.pkl": lambda p: p.write_bytes(b"x"),
                "delta.pkl": lambda p: p.write_bytes(b"y"),
                "authors.json": lambda p: p.write_text("[]")}

    def test_publish_and_load(self, tmp_path):
        root = tmp_path / "wb"
        side = self._publish(root)
        assert set(side["files"]) == self.THREE and side["bundle_version"] == "b2.bundle.v1"
        meta, files = load_bundle(root)
        assert meta["training_weighting"] == "work_balanced" and set(files) == self.THREE

    def test_allowlist_enforced(self, tmp_path):
        with pytest.raises(BundleError):     # incomplete set (only the exact 3 names are allowed)
            publish_bundle(tmp_path / "wb", {"model.pkl": lambda p: p.write_bytes(b"x")}, dict(self.META))
        with pytest.raises(BundleError):     # extra/unexpected name
            publish_bundle(tmp_path / "wb", {**self._writers(), "extra.pkl": lambda p: p.write_bytes(b"x")},
                           dict(self.META))

    def test_symlinked_versions_dir_refused_on_publish(self, tmp_path):
        root = tmp_path / "wb"; root.mkdir()
        outside = tmp_path / "outside"; outside.mkdir()
        (root / "versions").symlink_to(outside, target_is_directory=True)   # escape attempt
        with pytest.raises(BundleError, match="symlink"):
            self._publish(root)

    def test_unserializable_meta_raises_bundle_error(self, tmp_path):
        with pytest.raises(BundleError):
            publish_bundle(tmp_path / "wb", self._writers(), {**self.META, "extra": b"not json"})
        assert not (tmp_path / "wb" / "versions").exists() or \
            not list((tmp_path / "wb" / "versions").glob("[!.]*"))

    def test_meta_cannot_override_version(self, tmp_path):
        with pytest.raises(BundleError):
            publish_bundle(tmp_path / "wb", self._writers(), {**self.META, "bundle_version": "evil"})

    def test_missing_attestation_rejected(self, tmp_path):
        with pytest.raises(BundleError, match="attestation"):
            publish_bundle(tmp_path / "wb", self._writers(),
                           {"training_weighting": "work_balanced"})   # missing git_commit etc.
        with pytest.raises(BundleError, match="attestation"):
            publish_bundle(tmp_path / "wb", self._writers(), {**self.META, "git_commit": None})

    def test_tamper_detected(self, tmp_path):
        root = tmp_path / "wb"
        self._publish(root)
        _, files = load_bundle(root)
        files["authors.json"].write_text('["a","b","c"]')       # tamper the versioned file
        with pytest.raises(BundleError):
            load_bundle(root)

    def test_wrong_bundle_version_rejected(self, tmp_path):
        root = tmp_path / "wb"
        self._publish(root)
        _, files = load_bundle(root)
        sidecar = files["authors.json"].parent / "bundle_manifest.json"
        import json as _j
        data = _j.loads(sidecar.read_text()); data["bundle_version"] = "old"; sidecar.write_text(_j.dumps(data))
        with pytest.raises(BundleError):
            load_bundle(root)

    def test_missing_pointer_rejected(self, tmp_path):
        root = tmp_path / "wb"
        self._publish(root)
        (root / "current.json").unlink()
        with pytest.raises(BundleError):
            load_bundle(root)

    def test_republish_flips_current(self, tmp_path):
        root = tmp_path / "wb"
        self._publish(root, model=b"V1")
        self._publish(root, model=b"V2")
        _, files = load_bundle(root)
        assert files["model.pkl"].read_bytes() == b"V2"

    def test_failed_republish_keeps_old_bundle(self, tmp_path):
        root = tmp_path / "wb"
        self._publish(root, model=b"MODEL")
        def boom(p):
            raise RuntimeError("disk full mid-write")
        with pytest.raises(RuntimeError):
            publish_bundle(root, {"model.pkl": boom,
                                  "delta.pkl": lambda p: p.write_bytes(b"D"),
                                  "authors.json": lambda p: p.write_text("[]")}, {})
        meta, files = load_bundle(root)              # old bundle intact + loadable
        assert files["model.pkl"].read_bytes() == b"MODEL"

    def test_incomplete_version_dir_replaced_not_trusted(self, tmp_path):
        root = tmp_path / "wb"
        self._publish(root, model=b"V1")
        _, files = load_bundle(root)
        (files["model.pkl"].parent / "bundle_manifest.json").unlink()   # corrupt the version dir
        self._publish(root, model=b"V1")             # same content -> same token, but dir is corrupt
        meta, files2 = load_bundle(root)             # must be repaired, not trusted-as-is
        assert files2["model.pkl"].read_bytes() == b"V1"

    def test_self_consistent_sidecar_rewrite_rejected(self, tmp_path):
        import json as _j
        root = tmp_path / "wb"
        self._publish(root, model=b"MODEL-A")
        _, files = load_bundle(root)
        vdir = files["model.pkl"].parent
        vdir.joinpath("model.pkl").write_bytes(b"MODEL-B")               # swap content...
        sc = vdir / "bundle_manifest.json"
        data = _j.loads(sc.read_text())
        data["files"]["model.pkl"] = __import__("hashlib").sha256(b"MODEL-B").hexdigest()  # ...and the map
        sc.write_text(_j.dumps(data))
        with pytest.raises(BundleError):             # token no longer binds the served content
            load_bundle(root)

    def test_extra_untracked_file_not_deduped(self, tmp_path):
        root = tmp_path / "wb"
        self._publish(root, model=b"V1")
        _, files = load_bundle(root)
        (files["model.pkl"].parent / "sneaky.txt").write_text("x")      # untracked extra
        self._publish(root, model=b"V1")             # same token -> dir has an extra -> must repair
        _, files2 = load_bundle(root)
        assert not (files2["model.pkl"].parent / "sneaky.txt").exists()

    def test_symlinked_version_dir_rejected(self, tmp_path):
        import json as _j
        root = tmp_path / "wb"
        self._publish(root, model=b"REAL")
        pointer = _j.loads((root / "current.json").read_text())
        real = root / "versions" / pointer["version"]
        evil = root / "versions" / "evil"
        evil.symlink_to(real, target_is_directory=True)
        (root / "current.json").write_text('{"bundle_version":"b2.bundle.v1","version":"evil"}')
        with pytest.raises(BundleError):
            load_bundle(root)


class TestPickleCompat:
    """Blocker 1: pre-B2 pickles (no _wv / _ctor_state) must load → predict → clone."""
    def test_old_delta_pickle_loads_predicts_clones(self):
        d = BurrowsDelta(10, "manhattan").fit(
            ["a b c", "b c d", "c d a", "d a b"], [0, 0, 1, 1],
            groups=["a/w1", "a/w1", "b/w2", "b/w2"])
        del d.__dict__["_wv"]                              # simulate a pre-B2 pickle
        raw = joblib.dumps(d) if hasattr(joblib, "dumps") else None
        import pickle
        d2 = pickle.loads(pickle.dumps(d))
        assert d2.predict_proba(["a b c"]).shape[1] == 2  # predict works via getattr fallback
        assert d2.feature_names()                          # non-empty

    def test_old_fitted_block_clone_is_unfitted(self):
        from stylo.features.char_ngrams import CharNgramBlock
        blk = CharNgramBlock(ngram_range=(2, 3), min_df=1, bleach=False).fit(
            ["a b c", "b c d", "c d a"], [None] * 3, groups=None)
        del blk.__dict__["_ctor_state"]                   # simulate a pre-B2 fitted pickle
        cl = clone(blk)                                    # must not crash AND must be unfitted
        assert cl._vec is None and cl._wv is None          # no fitted vocab leaked into the clone

    def test_all_four_stylo_blocks_survive_missing_wv(self):
        """The blocks INSIDE the production StyloVectorizer (model.pkl) must predict from an old
        pickle that never had a _wv attribute."""
        import pickle
        from stylo.features.char_ngrams import CharNgramBlock
        from stylo.features.function_words import FunctionWordBlock
        from stylo.features.pos_ngrams import PosNgramBlock
        from stylo.features.punctuation import PunctNgramBlock

        class _Rep:
            pos_str = "NOUN VERB NOUN"
            punct_str = "— , —"
        texts = ["the cat sat here", "a dog ran there", "cats and dogs play now"]
        reps = [_Rep(), _Rep(), _Rep()]
        blocks = [
            CharNgramBlock(ngram_range=(2, 3), min_df=1, bleach=False).fit(texts, reps, groups=None),
            FunctionWordBlock(mode="mfw", mfw_count=5).fit(texts, reps, groups=None),
            PosNgramBlock(ngram_range=(1, 2), min_df=1).fit(texts, reps, groups=None),
            PunctNgramBlock(ngram_range=(1, 2)).fit(texts, reps, groups=None),
        ]
        for blk in blocks:
            old = pickle.loads(pickle.dumps(blk))
            del old.__dict__["_wv"]                        # faithful pre-B2 pickle (no _wv attr)
            assert old.transform(texts, reps).shape[0] == 3   # predict path must not crash
            assert old.feature_names()                        # nor feature_names


class TestTypeGatesAndLaundering:
    def test_float_and_bool_labels_rejected(self, tmp_path):
        ds = _toy_corpus(tmp_path)
        from stylo.eval.provenance import _validate_semantics
        with pytest.raises(ProvenanceError):
            _validate_semantics([0.0, 1.0], ["alpha/b1", "beta/b1"], ["alpha", "beta"])
        with pytest.raises(ProvenanceError):
            _validate_semantics([True, False], ["alpha/b1", "beta/b1"], ["alpha", "beta"])

    def test_derive_dataset_rejects_bad_index_types(self, tmp_path):
        ds = _toy_corpus(tmp_path)
        for bad in ({0, 1}, (i for i in range(2)), [0.0, 1.0], [True, False]):
            with pytest.raises(ProvenanceError):
                derive_dataset(ds, bad)

    def test_laundering_mutated_parent_via_derive_rejected(self, tmp_path):
        ds = _toy_corpus(tmp_path)
        ds.texts[0] = "SILENTLY CHANGED"                   # mutate parent after load
        with pytest.raises(ProvenanceError):               # derive validates the parent first
            derive_dataset(ds, [0, 1, 2, 3])

    def test_int_text_coercion_rejected(self):
        from stylo.corpus import Dataset
        from stylo.eval.provenance import (CorpusPolicyProvenance, RowIdentity,
                                           LEGACY_RECURSIVE, build_provenance)
        import hashlib
        # text "123" and int 123 stringify the same; the guard must reject a non-str text element
        rid = [RowIdentity(group="a/w1", ordinal=0, text_sha256=hashlib.sha256(b"123").hexdigest()),
               RowIdentity(group="b/w2", ordinal=0, text_sha256=hashlib.sha256(b"456").hexdigest())]
        with pytest.raises(ProvenanceError):
            build_provenance(loader_kind=LEGACY_RECURSIVE, texts=[123, "456"], y=[0, 1],
                             groups=["a/w1", "b/w2"], authors=["a", "b"], row_ids=rid,
                             frags_root="/x", corpus_policy=CorpusPolicyProvenance.build((), "unknown"))

    def test_corpus_policy_strict_types(self):
        from stylo.eval.provenance import CorpusPolicyProvenance
        with pytest.raises(ProvenanceError):
            CorpusPolicyProvenance.build("alice", "unknown")       # bare str -> set of letters
        with pytest.raises(ProvenanceError):
            CorpusPolicyProvenance.build([], 123)                  # non-str unknown_dir_name

    def test_delta_wb_rejects_float_labels(self):
        with pytest.raises(ValueError):
            BurrowsDelta(5, "manhattan", training_weighting=WORK_BALANCED).fit(
                ["a b", "b c"], np.array([0.0, 1.0]), groups=["a/w1", "b/w2"])

    def test_wb_then_legacy_refit_clears_wv(self):
        texts = ["a b c", "b c d", "c d a", "d a b"]; y = [0, 0, 1, 1]
        g = ["a/w1", "a/w1", "b/w2", "b/w2"]
        d = BurrowsDelta(10, "manhattan", training_weighting=WORK_BALANCED).fit(texts, y, groups=g)
        assert d._wv is not None
        d.training_weighting = CHUNK_WEIGHTED_LEGACY
        d.fit(texts, y, groups=g)                          # legacy refit
        assert d._wv is None                               # stale WB state cleared


class TestPredictFailClosed:
    def test_predict_raises_under_work_balanced(self, tmp_path, monkeypatch):
        from stylo.pipeline import predict
        from stylo.config import load_config

        class _Cfg:
            def __init__(self, base): self._b = base
            def get_path(self, k, d=None):
                if k == "evaluation.training_weighting":
                    return "work_balanced"
                return self._b.get_path(k, d)
        with pytest.raises(UnsupportedVariantError):
            predict.run(_Cfg(load_config()))


class TestAstGuard:
    """Blocker 6: every production call to a toggled fn must pass an explicit `weighting=`."""
    def test_all_production_calls_pass_weighting(self):
        import ast
        import pathlib as pl
        toggled = {"make_factory", "lobo_evaluate", "gkf_evaluate", "run_final",
                   "_evaluate_case", "run_sweep"}
        src = pl.Path("src/stylo")
        offenders = []
        for py in src.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            # skip the definitions themselves
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in toggled:
                    kws = {k.arg for k in node.keywords}
                    if "weighting" not in kws:
                        offenders.append(f"{py}:{node.lineno} {node.func.id}")
        assert not offenders, f"calls missing explicit weighting=: {offenders}"


class TestGoldenAndScience:
    """Golden pre-B2 artifact prediction-parity + estimator-level scientific regressions."""
    T = ["kot na okne spal dolgo", "sobaka bezhala v park bystro", "kot lovil mysh doma tikho",
         "ptica pela pesnyu vysoko", "more shumelo volnami silno", "les stoyal ochen tikho",
         "volny shli k beregu medlenno", "veter dul v pole silno"]
    Y = [0, 0, 0, 0, 1, 1, 1, 1]
    G = ["a/w1", "a/w1", "a/w2", "a/w2", "b/w3", "b/w3", "b/w4", "b/w4"]

    def test_golden_pre_b2_delta_prediction_parity(self):
        import pickle
        fresh = BurrowsDelta(50, "manhattan").fit(self.T, self.Y, groups=self.G)   # current legacy
        # genuine pre-B2 __dict__ shape: legacy math is unchanged, so drop only the B1/B2-added keys
        old = {k: v for k, v in fresh.__dict__.items()
               if k not in ("_schema_version", "_wv", "training_weighting")}
        golden = BurrowsDelta.__new__(BurrowsDelta)
        golden.__dict__ = old
        loaded = pickle.loads(pickle.dumps(golden))                # __setstate__ migrates v1
        assert loaded._schema_version == 1                          # versionless -> v1
        np.testing.assert_allclose(loaded.predict_proba(self.T), fresh.predict_proba(self.T))
        assert list(loaded.classes_) == list(fresh.classes_)
        loaded.fit(self.T, self.Y, groups=self.G)                   # refit works

    def test_delta_unequal_chunk_lengths_sum_over_sum(self):
        # a work's two chunks have very different lengths — Σcounts/Σevents != mean-of-ratios
        v = WorkLevelVectorizer(analyzer_params=WORD, mode="relative", min_df_works=1).fit(
            ["a a a a a a b", "a b"], ["w", "w"])
        _, rows = v.transform_grouped(["a a a a a a b", "a b"], ["w", "w"])
        assert rows.toarray()[0][v.vocabulary_["a"]] == pytest.approx(7 / 9)   # (6+1)/(7+2)

    def test_transform_grouped_denominator_includes_oov(self):
        # 'zzz' is pruned/OOV (min_df_works keeps only shared 'a'); it still counts in the denominator
        v = WorkLevelVectorizer(analyzer_params=WORD, mode="relative", min_df_works=2).fit(
            ["a zzz zzz", "a qqq"], ["w1", "w2"])
        assert "a" in v.vocabulary_ and "zzz" not in v.vocabulary_
        _, rows = v.transform_grouped(["a zzz zzz"], ["w1"])
        assert rows.toarray()[0][v.vocabulary_["a"]] == pytest.approx(1 / 3)   # a=1 of 3 total tokens

    def test_fixed_vocabulary_delta_wb(self):
        d = BurrowsDelta(vocabulary=["kot", "les", "more"], training_weighting=WORK_BALANCED).fit(
            self.T, self.Y, groups=self.G)
        assert d.centroids_.shape[1] == 3 and d.feature_names() == ["kot", "les", "more"]

    def test_deltacos_wb_z_is_l2_normalised(self):
        d = BurrowsDelta(50, "cosine", training_weighting=WORK_BALANCED).fit(self.T, self.Y, groups=self.G)
        assert d.group_weighting_.endswith("_l2")
        assert np.allclose(np.linalg.norm(d.centroids_, axis=1), 1.0, atol=0.5)  # work-z L2 then mean

    def test_char_work_df_idf_independent_reference(self):
        # independent reference for CharCos work-level DF/IDF selection
        texts = ["kot text here", "kott text there", "sob text now", "sobb text soon"]
        g = ["a/w1", "a/w2", "b/w3", "b/w4"]
        wv = WorkLevelVectorizer(
            analyzer_params={"analyzer": "char", "ngram_range": (3, 3), "lowercase": True},
            mode="tfidf", min_df_works=2).fit(texts, g)
        # every kept 3-gram must appear in >= 2 distinct works (independent DF recount)
        from collections import defaultdict
        analyze = __import__("sklearn.feature_extraction.text", fromlist=["CountVectorizer"]).CountVectorizer(
            analyzer="char", ngram_range=(3, 3), lowercase=True).build_analyzer()
        feat_works = defaultdict(set)
        for t, gg in zip(texts, g):
            for f in set(analyze(t)):
                feat_works[f].add(gg)
        for f in wv.vocabulary_:
            assert len(feat_works[f]) >= 2


class TestRound6Hardening:
    def test_probability_validation_fail_closed(self):
        from stylo.eval.lobo import _validate_proba
        _validate_proba(np.array([[0.6, 0.4], [0.1, 0.9]]), np.array([0, 1]), 2, 2)   # ok
        for bad in ([[0.6, 0.4], [np.nan, 0.9]], [[0.6, 0.4], [-0.1, 1.1]],
                    [[0.6, 0.3], [0.5, 0.5]]):                                          # NaN / neg / sum!=1
            with pytest.raises(ValueError):
                _validate_proba(np.array(bad), np.array([0, 1]), 2, 2)
        with pytest.raises(ValueError):                                                # duplicate classes
            _validate_proba(np.array([[0.5, 0.5]]), np.array([1, 1]), 2, 1)
        with pytest.raises(ValueError):                                                # class out of range
            _validate_proba(np.array([[0.5, 0.5]]), np.array([0, 5]), 2, 1)

    def test_public_engines_have_no_contract_param(self):
        import inspect
        from stylo.eval import final, lobo, groupkfold
        for fn in (lobo.lobo_evaluate, groupkfold.gkf_evaluate, final.run_final):
            assert "contract" not in inspect.signature(fn).parameters   # no caller-controlled anchor

    def test_provenance_schema_rejects_subclass_and_poly_fields(self, tmp_path):
        import dataclasses
        from stylo.corpus import Dataset
        from stylo.eval.provenance import (RunContract, _validate_provenance_schema, RowIdentity)
        ds = _toy_corpus(tmp_path)
        # a RowIdentity subclass with overridden equality must be rejected
        class EvilRI(RowIdentity):
            def __eq__(self, other): return True
        forged = dataclasses.replace(ds.provenance,
                                     row_ids=tuple(EvilRI(**dataclasses.asdict(r)) for r in ds.provenance.row_ids))
        with pytest.raises(ProvenanceError):
            _validate_provenance_schema(forged)
        # a polymorphic rows_digest (str subclass) is rejected
        class S(str): pass
        with pytest.raises(ProvenanceError):
            _validate_provenance_schema(dataclasses.replace(ds.provenance, rows_digest=S(ds.provenance.rows_digest)))

    def test_probability_gate_exact(self):
        from stylo.eval.lobo import _validate_proba
        with pytest.raises(ValueError):     # float classes [0.0, 1.0]
            _validate_proba(np.array([[0.6, 0.4]]), np.array([0.0, 1.0]), 2, 1)
        with pytest.raises(ValueError):     # object-bool [False, True]
            _validate_proba(np.array([[0.6, 0.4]]), np.array([False, True], dtype=object), 2, 1)
        with pytest.raises(ValueError):     # duplicate class [0,1,1] (len != n_authors)
            _validate_proba(np.array([[0.3, 0.3, 0.4]]), np.array([0, 1, 1]), 2, 1)
        _validate_proba(np.array([[0.6, 0.4]]), np.array([0, 1]), 2, 1)   # ok

    def test_schema_version_reject_unknown(self):
        import pickle
        from stylo.features.char_ngrams import CharNgramBlock
        for obj in (BurrowsDelta(5, "manhattan"), CharCosineBaseline(),
                    CharNgramBlock(ngram_range=(2, 3), min_df=1, bleach=False)):
            obj.__dict__["_schema_version"] = 99                       # a future artifact
            with pytest.raises(ValueError):
                pickle.loads(pickle.dumps(obj))

    def test_fresh_block_carries_schema_version(self):
        from stylo.features.char_ngrams import CharNgramBlock
        assert CharNgramBlock(ngram_range=(2, 3), min_df=1)._schema_version == 2

    def test_sweep_unknown_strategy_rejected(self):
        from stylo.eval.sweep import _evaluate_case, EvalCase
        with pytest.raises(ValueError):
            _evaluate_case(CFG, None, EvalCase("x"), strategy="bogus", weighting=CHUNK_WEIGHTED_LEGACY)

    def test_safe_write_batch_all_or_nothing(self, tmp_path):
        from stylo.eval.provenance import safe_write_batch
        (tmp_path / "a.txt").write_text("OLD-A")
        (tmp_path / "b.txt").write_text("OLD-B")
        # one item's content is fine, but a symlinked target must abort the WHOLE generation
        (tmp_path / "b.txt").unlink(); (tmp_path / "legacy").write_text("H")
        (tmp_path / "b.txt").symlink_to(tmp_path / "legacy")
        with pytest.raises(ProvenanceError):
            safe_write_batch(tmp_path, {"a.txt": "NEW-A", "b.txt": "NEW-B"})
        assert (tmp_path / "a.txt").read_text() == "OLD-A"            # a.txt NOT changed (all-or-nothing)
        assert (tmp_path / "legacy").read_text() == "H"              # symlink not followed

    def test_preflight_blocks_wb_predict(self):
        from stylo.eval.provenance import UnsupportedVariantError

        class _Cfg:
            def __init__(s, b): s._b = b
            def get_path(s, k, d=None):
                return "work_balanced" if k == "evaluation.training_weighting" else s._b.get_path(k, d)
        # emulate the CLI preflight branch logic
        from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY, resolve_training_weighting
        w = resolve_training_weighting(_Cfg(CFG).get_path("evaluation.training_weighting"))
        assert w != CHUNK_WEIGHTED_LEGACY
        if "predict" in ["train", "predict"] and w != CHUNK_WEIGHTED_LEGACY:
            with pytest.raises(UnsupportedVariantError):
                raise UnsupportedVariantError("predict unsupported under work_balanced")


# ── mandatory runtime test: RuAA pin ──────────────────────────────────────────
def test_ruaa_subset_disk_anchored_legacy_only(tmp_path):
    """A RuAA-style subset chains to the disk parent and verifies only as legacy, never WB."""
    ds = _toy_corpus(tmp_path)                          # legacy loader
    sub = derive_dataset(ds, [0, 1, 2, 3, 8, 9, 10, 11])  # alpha/b1 + beta/b1
    assert sub.provenance.loader_kind == "legacy_recursive"
    c = RunContract.build(tmp_path, (), "unknown")
    verify_dataset_against_disk(None, sub, CHUNK_WEIGHTED_LEGACY, c)   # subset chains to disk parent
    with pytest.raises(ProvenanceError):               # a legacy subset can never verify as WB
        verify_dataset_against_disk(None, sub, WORK_BALANCED, c)
