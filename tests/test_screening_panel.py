"""Gate: the frozen screening_panel_v1 GKF fold assignment is complete, valid and reproducible.

Cheap integrity checks run on the committed manifest alone. The reproduce + end-to-end smoke need
the on-disk corpus and are skipped when it is absent; the smoke is majority-only (not a full sweep).
"""
from __future__ import annotations

import copy
import pathlib

import pytest

from stylo.config import load_config
from stylo.eval import screening_panel as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAGS = ROOT / "data" / "frags_train"
CFG = load_config()


def _manifest():
    return sp.load_manifest_file(ROOT / "docs" / "screening_panel_v1.json")


# ── committed-manifest integrity (no corpus needed) ────────────────────────────
def test_manifest_shape_and_integrity():
    m = _manifest()                                   # load_manifest_file verifies on load
    assert m["panel"] == "screening_panel_v1"
    assert m["k_folds"] == 5 and m["n_authors"] == 43 and m["n_works"] == 251
    assert len(m["works"]) == 251
    assert len({w["work_id"] for w in m["works"]}) == 251     # unique works
    assert sum(m["fold_sizes"]) == 251 and all(s > 0 for s in m["fold_sizes"])


def test_each_work_one_fold_and_train_covers_all_classes():
    m = _manifest()
    authors = m["authors"]
    for w in m["works"]:                              # exact non-bool int label + authors[label]==author
        assert type(w["label"]) is int and not isinstance(w["label"], bool)
        assert authors[w["label"]] == w["author"] == w["work_id"].split("/", 1)[0]
        assert 0 <= w["fold"] < 5
    for f in range(5):                                # every train fold contains all 43 classes
        train = {w["author"] for w in m["works"] if w["fold"] != f}
        assert train == set(authors), f"train fold {f} misses classes"


def test_self_and_config_hash_detect_tamper():
    m = _manifest()
    assert m["config_hash"] == sp._config_hash(sp.ALGORITHM, m["k_folds"], m["seed"])
    assert m["self_hash"] == sp._self_hash(m)
    bad = copy.deepcopy(m)
    bad["works"][0]["fold"] = (bad["works"][0]["fold"] + 1) % 5    # move a work to another fold
    with pytest.raises(sp.ScreeningPanelError):
        sp.verify_manifest(bad)                       # self_hash mismatch
    bad2 = copy.deepcopy(m); bad2["seed"] = 7
    with pytest.raises(sp.ScreeningPanelError):
        sp.verify_manifest(bad2)                      # config_hash mismatch


def test_verify_result_against_panel_fail_closed():
    import pandas as pd
    m = _manifest()
    good = [{"test_author": w["work_id"].split("/", 1)[0],
             "test_book": w["work_id"].split("/", 1)[1],
             "true_label": w["label"], "fold": w["fold"]} for w in m["works"]]
    sp.verify_result_against_panel(pd.DataFrame(good), m)          # ok
    # a work outside the panel
    rogue = copy.deepcopy(good); rogue[0]["test_author"] = "no_such_author"
    with pytest.raises(sp.ScreeningPanelError):
        sp.verify_result_against_panel(pd.DataFrame(rogue), m)
    # a wrong label
    wl = copy.deepcopy(good); wl[0]["true_label"] = (wl[0]["true_label"] + 1) % 43
    with pytest.raises(sp.ScreeningPanelError):
        sp.verify_result_against_panel(pd.DataFrame(wl), m)
    # a missing work (incomplete coverage)
    with pytest.raises(sp.ScreeningPanelError):
        sp.verify_result_against_panel(pd.DataFrame(good[:-1]), m)
    # the fold column is MANDATORY
    nofold = [{k: v for k, v in row.items() if k != "fold"} for row in good]
    with pytest.raises(sp.ScreeningPanelError):
        sp.verify_result_against_panel(pd.DataFrame(nofold), m)
    # a float label (2.0) is not a genuine Integral
    flt = copy.deepcopy(good); flt[0]["true_label"] = float(flt[0]["true_label"])
    with pytest.raises(sp.ScreeningPanelError):
        sp.verify_result_against_panel(pd.DataFrame(flt), m)


def test_verify_manifest_rejects_lied_fold_sizes():
    m = _manifest()
    bad = copy.deepcopy(m)
    bad["fold_sizes"] = [1, 1, 1, 1, sum(m["fold_sizes"]) - 4]     # wrong per-fold counts
    bad["self_hash"] = sp._self_hash(bad)                          # keep internal self_hash valid
    with pytest.raises(sp.ScreeningPanelError):                    # actual per-fold counts disagree
        sp.verify_manifest(bad)


# ── reproduce + end-to-end smoke (need the on-disk corpus) ─────────────────────
@pytest.mark.skipif(not FRAGS.exists(), reason="corpus data/frags_train not present")
class TestWithCorpus:
    def _dataset(self):
        from stylo.eval.dispatch import frozen_run_contract
        from stylo.eval.provenance import verify_dataset_against_disk
        from stylo.domain.work_weighting import CHUNK_WEIGHTED_LEGACY
        from stylo.dataset import resolve_dataset
        ds = resolve_dataset(
            CFG, CHUNK_WEIGHTED_LEGACY, FRAGS,
            exclude_authors=set(CFG.get_path("corpus_policy.exclude_from_benchmark", []) or []),
            unknown_name=CFG.get_path("corpus_policy.unknown_dir_name", "unknown"))
        verify_dataset_against_disk(CFG, ds, CHUNK_WEIGHTED_LEGACY, frozen_run_contract(CFG))
        return ds

    def test_manifest_reproduces_from_corpus(self):
        ds = self._dataset()
        rebuilt = sp.build_manifest(ds)
        committed = _manifest()
        assert rebuilt["self_hash"] == committed["self_hash"]      # deterministic + reproducible
        assert rebuilt["parent_dataset_digest"] == ds.provenance.rows_digest
        assert committed["parent_dataset_digest"] == ds.provenance.rows_digest

    def test_raw_panel_worker_rejects_bare_dataset_before_fit(self):
        from stylo.eval.groupkfold import bind_screening_panel, _gkf_run
        from stylo.eval.provenance import ProvenanceError
        from stylo.domain.work_weighting import CHUNK_WEIGHTED_LEGACY
        ds = self._dataset()
        sub, panel = bind_screening_panel(CFG, ds, CHUNK_WEIGHTED_LEGACY)
        assert panel is not None and sub.n_authors == 43
        with pytest.raises(ProvenanceError, match="sealed"):
            _gkf_run(CFG, sub, "majority", None, None, panel)

    def test_missing_manifest_hard_fails_no_sgkf_fallback(self, monkeypatch, tmp_path):
        from stylo.eval.groupkfold import bind_screening_panel
        from stylo.domain.work_weighting import CHUNK_WEIGHTED_LEGACY
        ds = self._dataset()
        monkeypatch.setattr(sp, "manifest_docs_path", lambda cfg: tmp_path / "screening_panel_v1.json")
        with pytest.raises(sp.ScreeningPanelError):               # missing → hard fail, never dynamic SGKF
            bind_screening_panel(CFG, ds, CHUNK_WEIGHTED_LEGACY)

    def test_truncated_self_signed_manifest_hard_fails(self, monkeypatch, tmp_path):
        from stylo.eval.groupkfold import bind_screening_panel
        from stylo.eval.provenance import derive_dataset
        from stylo.domain.work_weighting import CHUNK_WEIGHTED_LEGACY
        from stylo.jsonio import dumps_strict
        ds = self._dataset()
        # a VALID but truncated panel: rebuild the manifest on a corpus missing one author's works
        drop = _manifest()["authors"][0]
        keep = [i for i, g in enumerate(ds.groups.tolist()) if str(g).split("/", 1)[0] != drop]
        truncated = sp.build_manifest(derive_dataset(ds, keep))
        sp.verify_manifest(truncated)                            # passes its OWN integrity (self-signed)
        assert truncated != _manifest() and truncated["n_authors"] < 43
        p = tmp_path / "screening_panel_v1.json"
        p.write_text(dumps_strict(truncated, indent=2) + "\n", encoding="utf-8")
        monkeypatch.setattr(sp, "manifest_docs_path", lambda cfg: p)
        with pytest.raises(sp.ScreeningPanelError):              # rebuild-from-corpus != committed → hard fail
            bind_screening_panel(CFG, ds, CHUNK_WEIGHTED_LEGACY)
