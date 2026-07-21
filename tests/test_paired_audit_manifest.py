"""Synthetic tests for the LOBO/RuAA fold-manifest builder, verifier, and frozen universes (§1.6/§12)."""
from __future__ import annotations

import pytest

from stylo.eval.paired_audit import manifest as mf


class _DS:
    """Minimal dataset stub — the builder derives works/authors/folds from .groups only."""
    def __init__(self, groups):
        self.groups = groups


def _groups(author_workcounts, prefix, chunks=2):
    g = []
    for i, cnt in enumerate(author_workcounts):
        for w in range(cnt):
            g += [f"{prefix}{i:02d}/w{w:03d}"] * chunks
    return g


def _lobo_groups(multi_counts, n_singletons):
    g = _groups(multi_counts, "m")
    for s in range(n_singletons):
        g += [f"s{s:02d}/only"] * 2
    return g


def _toy_lobo():
    # m00,m01,m02 multi-work (2 works each); d0,e0 single-work (train-only)
    g = _groups([2, 2, 2], "m")
    g += ["d0/only"] * 3 + ["e0/only"] * 3
    return _DS(g)


class TestBuild:
    def test_lobo_build_leaves_singletons_train_only(self):
        m = mf.build_fold_manifest("lobo", _toy_lobo(), parent_dataset_digest="p" * 64,
                                   algorithm="leave_one_work_out", seed=42, config_hash="c" * 64)
        assert m["n_train_authors"] == 5 and m["n_train_works"] == 8
        assert m["n_tested_authors"] == 3 and m["n_tested_works"] == 6
        assert len(m["probability_class_order"]) == 5 and len(m["metric_label_order"]) == 3
        tested = [w for w in m["works"] if w["tested"]]
        assert sorted(w["fold_index"] for w in tested) == list(range(6))
        train_only = [w for w in m["works"] if not w["tested"]]
        assert {w["author_id"] for w in train_only} == {"d0", "e0"}
        assert m["self_hash"] == mf.fold_manifest_self_hash(m)

    def test_ruaa_build_requires_selection_digest_and_tests_all(self):
        ds = _DS(_groups([2, 2, 2], "r"))
        with pytest.raises(mf.FoldManifestError):
            mf.build_fold_manifest("ruaa", ds, parent_dataset_digest="p" * 64,
                                   algorithm="whole_work", seed=42, config_hash="c" * 64)
        m = mf.build_fold_manifest("ruaa", ds, parent_dataset_digest="p" * 64, algorithm="whole_work",
                                   seed=42, config_hash="c" * 64, selection_digest="s" * 64)
        assert m["n_tested_works"] == m["n_train_works"] == 6   # 3 authors x 2 works, all tested
        assert all(w["tested"] for w in m["works"])


class _DSProv(_DS):
    def __init__(self, groups, sel):
        super().__init__(groups)
        self.provenance = type("P", (), {"selection_manifest_digest": sel})()


class TestRuaaSelectionBinding:
    def test_selection_digest_mismatch_rejected_at_build(self):
        ds = _DSProv(_groups([6] * 21 + [11], "r"), "s" * 64)
        with pytest.raises(mf.FoldManifestError):
            mf.build_fold_manifest("ruaa", ds, parent_dataset_digest="p" * 64, algorithm="whole_work",
                                   seed=42, config_hash="c" * 64, selection_digest="d" * 64)
        mf.build_fold_manifest("ruaa", ds, parent_dataset_digest="p" * 64, algorithm="whole_work",
                               seed=42, config_hash="c" * 64, selection_digest="s" * 64)  # matches -> ok


class TestVerify:
    def test_rebuild_equality_and_tamper(self):
        ds = _toy_lobo()
        a = mf.build_fold_manifest("lobo", ds, parent_dataset_digest="p" * 64,
                                   algorithm="leave_one_work_out", seed=42, config_hash="c" * 64)
        b = mf.build_fold_manifest("lobo", ds, parent_dataset_digest="p" * 64,
                                   algorithm="leave_one_work_out", seed=42, config_hash="c" * 64)
        mf.verify_manifest_matches_rebuilt(a, b)                 # identical -> ok
        b2 = dict(b)
        b2["seed"] = 7
        b2["self_hash"] = mf.fold_manifest_self_hash(b2)
        with pytest.raises(mf.FoldManifestError):                # rebuilt differs
            mf.verify_manifest_matches_rebuilt(a, b2)
        tampered = dict(a)
        tampered["n_tested_works"] = 999                         # tamper without re-signing
        with pytest.raises(mf.FoldManifestError):
            mf.verify_manifest_self_hash(tampered)


class TestFrozenUniverse:
    def _lobo_universe_ds(self, multi_counts, n_singletons):
        return _DS(_lobo_groups(multi_counts, n_singletons))

    def test_lobo_universe_passes_for_47_255_43_251(self):
        ds = self._lobo_universe_ds([5] * 42 + [41], 4)          # 43 authors, 251 works; +4 singletons
        m = mf.build_fold_manifest("lobo", ds, parent_dataset_digest="p" * 64,
                                   algorithm="leave_one_work_out", seed=42, config_hash="c" * 64)
        assert (m["n_train_authors"], m["n_train_works"]) == (47, 255)
        assert (m["n_tested_authors"], m["n_tested_works"]) == (43, 251)
        mf.assert_lobo_universe(m)                               # must not raise

    def test_lobo_universe_rejects_off_counts(self):
        wrong_works = mf.build_fold_manifest("lobo", self._lobo_universe_ds([5] * 42 + [40], 4),
                                             parent_dataset_digest="p" * 64, algorithm="x", seed=42,
                                             config_hash="c" * 64)  # 250 tested works
        with pytest.raises(mf.FoldManifestError):
            mf.assert_lobo_universe(wrong_works)
        wrong_authors = mf.build_fold_manifest("lobo", self._lobo_universe_ds([5] * 42 + [41], 3),
                                               parent_dataset_digest="p" * 64, algorithm="x", seed=42,
                                               config_hash="c" * 64)  # 46 authors, 3 singletons
        with pytest.raises(mf.FoldManifestError):
            mf.assert_lobo_universe(wrong_authors)

    def test_ruaa_universe_passes_for_137_22(self):
        ds = _DS(_groups([6] * 21 + [11], "r"))                  # 22 authors, 137 works
        m = mf.build_fold_manifest("ruaa", ds, parent_dataset_digest="p" * 64, algorithm="whole_work",
                                   seed=42, config_hash="c" * 64, selection_digest="s" * 64)
        assert (m["n_train_authors"], m["n_train_works"]) == (22, 137)
        mf.assert_ruaa_universe(m)                               # must not raise

    def test_ruaa_universe_rejects_off_counts(self):
        ds = _DS(_groups([6] * 21 + [10], "r"))                  # 136 works
        m = mf.build_fold_manifest("ruaa", ds, parent_dataset_digest="p" * 64, algorithm="whole_work",
                                   seed=42, config_hash="c" * 64, selection_digest="s" * 64)
        with pytest.raises(mf.FoldManifestError):
            mf.assert_ruaa_universe(m)

    def test_bogus_class_order_contents_rejected(self):
        ds = self._lobo_universe_ds([5] * 42 + [41], 4)
        m = mf.build_fold_manifest("lobo", ds, parent_dataset_digest="p" * 64, algorithm="x",
                                   seed=42, config_hash="c" * 64)
        m["probability_class_order"] = ["bogus%02d" % i for i in range(47)]   # right length, wrong names
        m["self_hash"] = mf.fold_manifest_self_hash(m)                        # re-sign the forgery
        with pytest.raises(mf.FoldManifestError):
            mf.assert_lobo_universe(m)

    def test_malformed_input_raises_typed_error(self):
        with pytest.raises(mf.FoldManifestError):
            mf.assert_lobo_universe("not a dict")
        ds = _toy_lobo()
        m = mf.build_fold_manifest("lobo", ds, parent_dataset_digest="p" * 64, algorithm="x",
                                   seed=42, config_hash="c" * 64)
        m["works"][0]["tested"] = True
        m["works"][0]["fold_index"] = None                       # tested but no fold_index
        m["self_hash"] = mf.fold_manifest_self_hash(m)
        with pytest.raises(mf.FoldManifestError):                 # typed, not a raw TypeError
            mf.assert_lobo_universe(m)
