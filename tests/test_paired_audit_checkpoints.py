"""Synthetic tests for the per-fold immutable checkpoint store and resume semantics (§4.3/§7)."""
from __future__ import annotations

import pathlib

import pytest

from stylo.jsonio import dump_strict, load_strict
from stylo.eval.paired_audit.checkpoints import CheckpointError, CheckpointStore

RUN_ID = "a" * 64          # a sha256-shaped run_id (hex)


def _bindings():
    return {
        "lobo": {"dataset_digest": "a" * 64, "parent_dataset_digest": "a" * 64,
                 "fold_manifest_digest": "b" * 64,
                 "probability_class_order_digest": "c" * 64, "metric_label_order_digest": "d" * 64},
        "ruaa": {"dataset_digest": "1" * 64, "parent_dataset_digest": "a" * 64,
                 "fold_manifest_digest": "2" * 64,
                 "probability_class_order_digest": "3" * 64, "metric_label_order_digest": "4" * 64},
    }


def _store(tmp_path, run_id=RUN_ID):
    return CheckpointStore(tmp_path / "ck", run_id, _bindings())


def _result(correct=True, **over):
    # a coherent width-2 result: correct -> pred==true==1 (rank 1); incorrect -> pred=0 true=1 (rank 2)
    if correct:
        r = {"pred_label": 1, "true_label": 1, "correct": True, "rank": 1, "probabilities": [0.4, 0.6]}
    else:
        r = {"pred_label": 0, "true_label": 1, "correct": False, "rank": 2, "probabilities": [0.6, 0.4]}
    r.update(over)
    return r


def _evi(result):
    # the canonical, authoritative proba_digest for this result's probability vector
    from stylo.eval.paired_audit.checkpoints import proba_digest
    return {"proba_digest": proba_digest(result["probabilities"])}


def _save(store, fold_index=0, work_id="stylo_book", model="stylo", cell="A0", dataset="lobo",
          result=None):
    r = result or _result()
    return store.save(dataset, model, cell, fold_index, work_id, result=r, fold_local_evidence=_evi(r))


def test_dataset_bindings_derive_class_order_digests(tmp_path):
    from stylo.eval.paired_audit.checkpoints import dataset_bindings
    from stylo.eval.paired_audit.run_plan import class_order_digest
    b = dataset_bindings("a" * 64, "p" * 64, "b" * 64, ["p", "q"], ["p"])
    assert b["probability_class_order_digest"] == class_order_digest(["p", "q"])
    assert b["metric_label_order_digest"] == class_order_digest(["p"])
    assert b["parent_dataset_digest"] == "p" * 64
    store = CheckpointStore(tmp_path / "ck", RUN_ID, {"lobo": b, "ruaa": b})   # valid 5-key bindings
    assert store.dataset_bindings["lobo"]["probability_class_order_digest"]


class TestSaveResume:
    def test_save_scan_and_idempotent(self, tmp_path):
        store = _store(tmp_path)
        path = _save(store)
        assert path.is_file()
        present = store.scan_cell("lobo", "stylo", "A0")
        assert set(present) == {0} and present[0]["work_id"] == "stylo_book"
        assert _save(store) == path                       # idempotent re-save, same path, no error

    def test_valid_resume_skips_present_and_pends_missing(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0")
        state = store.resume_cell("lobo", "stylo", "A0",
                                  [(0, "w0"), (1, "w1"), (2, "w2")])
        assert set(state["present"]) == {0}
        assert state["pending"] == [(1, "w1"), (2, "w2")]

    def test_delta_model_slug_path_is_safe(self, tmp_path):
        store = _store(tmp_path)
        store.save("lobo", "delta_cos:500", "A2", 3, "wq",
                   result=_result(correct=False), fold_local_evidence=_evi(_result(correct=False)))
        assert store.scan_cell("lobo", "delta_cos:500", "A2")[3]["model"] == "delta_cos:500"


class TestComplete:
    def test_assert_complete_success_and_incomplete_fatal(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0")
        _save(store, fold_index=1, work_id="w1")
        present = store.assert_cell_complete("lobo", "stylo", "A0", [(0, "w0"), (1, "w1")])
        assert set(present) == {0, 1}
        with pytest.raises(CheckpointError):              # missing fold -> fatal at COMPLETE
            store.assert_cell_complete("lobo", "stylo", "A0", [(0, "w0"), (1, "w1"), (2, "w2")])


class TestFailClosed:
    def test_conflicting_checkpoint_is_fatal(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0", result=_result(correct=True))
        with pytest.raises(CheckpointError):              # same identity, different result -> no overwrite
            _save(store, fold_index=0, work_id="w0", result=_result(correct=False))

    def test_corrupt_self_hash_is_fatal(self, tmp_path):
        store = _store(tmp_path)
        path = _save(store, fold_index=0, work_id="w0")
        rec = load_strict(path)
        rec["result"] = {"correct": False}               # tamper without recomputing self_hash
        dump_strict(rec, path, trailing_newline=True)
        with pytest.raises(CheckpointError):
            store.scan_cell("lobo", "stylo", "A0")

    def test_extra_checkpoint_is_fatal_on_resume(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=5, work_id="w5")
        with pytest.raises(CheckpointError):              # (5,w5) not in the expected fold set
            store.resume_cell("lobo", "stylo", "A0", [(0, "w0"), (1, "w1")])

    def test_stray_file_in_cell_dir_is_fatal(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0")
        cell_dir = (tmp_path / "ck" / "lobo" / "stylo" / "A0")
        (cell_dir / "notes.txt").write_text("stray", encoding="utf-8")
        with pytest.raises(CheckpointError):
            store.scan_cell("lobo", "stylo", "A0")

    def test_filename_not_matching_identity_is_fatal(self, tmp_path):
        store = _store(tmp_path)
        path = _save(store, fold_index=0, work_id="w0")
        renamed = path.parent / "0007-0123456789abcdef.json"   # valid shape, wrong identity
        path.rename(renamed)
        with pytest.raises(CheckpointError):
            store.scan_cell("lobo", "stylo", "A0")

    def test_wrong_run_id_rejects_on_load(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0")
        other = CheckpointStore(tmp_path / "ck", "b" * 64, _bindings())  # different valid run_id
        with pytest.raises(CheckpointError):              # run_id mismatch -> conflicting identity
            other.scan_cell("lobo", "stylo", "A0")

    def test_wrong_bindings_reject_on_load(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0")
        drifted = _bindings()
        drifted["lobo"]["dataset_digest"] = "0" * 64
        other = CheckpointStore(tmp_path / "ck", RUN_ID, drifted)
        with pytest.raises(CheckpointError):
            other.scan_cell("lobo", "stylo", "A0")

    def test_fold_index_over_9999_rejected(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(CheckpointError):
            store.checkpoint_filename(10000, "w")

    def test_symlinked_ancestor_raises_checkpoint_error(self, tmp_path):
        store = _store(tmp_path)
        root = tmp_path / "ck"
        root.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "lobo").symlink_to(outside, target_is_directory=True)
        with pytest.raises(CheckpointError):             # not a bare BundleError
            _save(store, fold_index=0, work_id="w0")

    def test_leading_dot_temp_is_ignored_by_scan(self, tmp_path):
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0")
        cell_dir = tmp_path / "ck" / "lobo" / "stylo" / "A0"
        (cell_dir / ".ckpt_orphan.tmp").write_text("partial crash orphan", encoding="utf-8")
        present = store.scan_cell("lobo", "stylo", "A0")   # transient temp ignored, not fatal
        assert set(present) == {0}

    def test_store_requires_both_datasets_and_nonempty_bindings(self, tmp_path):
        with pytest.raises(CheckpointError):
            CheckpointStore(tmp_path / "ck", RUN_ID, {"lobo": _bindings()["lobo"]})
        bad = _bindings()
        bad["ruaa"]["dataset_digest"] = ""
        with pytest.raises(CheckpointError):
            CheckpointStore(tmp_path / "ck", RUN_ID, bad)

    def test_run_id_must_be_sha256(self, tmp_path):
        with pytest.raises(CheckpointError):
            CheckpointStore(tmp_path / "ck", "not-a-sha", _bindings())

    def test_unknown_model_or_cell_rejected(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(CheckpointError):
            store.save("lobo", "bogus_model", "A0", 0, "w", result=_result(), fold_local_evidence={})
        with pytest.raises(CheckpointError):
            store.save("lobo", "stylo", "A9", 0, "w", result=_result(), fold_local_evidence={})

    def test_result_must_carry_core_keys(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(CheckpointError):
            store.save("lobo", "stylo", "A0", 0, "w", result={"correct": True},
                       fold_local_evidence={"proba_digest": "e" * 64})
        with pytest.raises(CheckpointError):
            store.save("lobo", "stylo", "A0", 0, "w",
                       result=_result(probabilities=[]), fold_local_evidence={"proba_digest": "e" * 64})

    def test_resume_rejects_reself_hashed_invalid_checkpoint(self, tmp_path):
        # owner repro: a checkpoint with pred=-9,true=99,rank=-5,proba=[-7,8] + empty evidence, RE-
        # self-hashed, was accepted by scan_cell. _validate_result now runs on LOAD, not only at save.
        from stylo.eval.paired_audit.checkpoints import _CHECKPOINT_SCHEMA, _self_hash
        from stylo.eval.prediction_contract import PREDICTION_CONTRACT_VERSION
        store = _store(tmp_path)
        record = {"schema": _CHECKPOINT_SCHEMA,
                  "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
                  "run_id": RUN_ID, "dataset": "lobo", "model": "stylo",
                  "cell": "A0", "fold_index": 0, "work_id": "x/w", "bindings": _bindings()["lobo"],
                  "result": {"pred_label": -9, "true_label": 99, "correct": True, "rank": -5,
                             "probabilities": [-7, 8]},
                  "fold_local_evidence": {}}
        record["self_hash"] = _self_hash(record)                 # a VALID self-hash over invalid content
        path = store.checkpoint_path("lobo", "stylo", "A0", 0, "x/w")
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_strict(record, path, trailing_newline=True)
        with pytest.raises(CheckpointError):                     # rejected on load/resume, not accepted
            store.scan_cell("lobo", "stylo", "A0")

    def test_strict_result_schema_rejects_incoherent_folds(self, tmp_path):
        store = _store(tmp_path)
        ev = {"proba_digest": "e" * 64}

        def bad(**over):
            with pytest.raises(CheckpointError):
                store.save("lobo", "stylo", "A0", 0, "w", result=_result(**over), fold_local_evidence=ev)

        bad(rank=-1)                                            # negative rank
        bad(rank=9)                                             # rank out of [1,width]
        bad(pred_label=5)                                       # pred out of range
        bad(true_label=5)                                       # true label out of range
        bad(probabilities=[0.6, 0.6])                           # does not sum to 1
        bad(probabilities=[1.4, -0.4])                          # outside [0,1]
        bad(pred_label=0)                                       # pred != argmax (proba[0]=0.4 < 0.6)
        bad(correct=True, true_label=0)                         # correct but pred(1) != true(0)
        bad(correct=True, rank=2)                               # rank disagrees with this non-tied vector
        # empty evidence is rejected even with a coherent result
        with pytest.raises(CheckpointError):
            store.save("lobo", "stylo", "A0", 0, "w", result=_result(), fold_local_evidence={})

    def test_tie_contract_is_stable_top1_and_conservative_rank(self, tmp_path):
        store = _store(tmp_path)
        tied = {
            "pred_label": 0,
            "true_label": 0,
            "correct": True,
            "rank": 2,
            "probabilities": [0.5, 0.5],
        }
        _save(store, result=tied)
        with pytest.raises(CheckpointError, match="stable top-1"):
            _save(
                _store(tmp_path / "other"),
                result={
                    **tied,
                    "pred_label": 1,
                    "true_label": 1,
                },
            )
        with pytest.raises(CheckpointError, match="worst-tie"):
            _save(_store(tmp_path / "third"), result={**tied, "rank": 1})

    def test_v1_checkpoint_is_not_resumable_under_v2_contract(self, tmp_path):
        store = _store(tmp_path)
        path = _save(store)
        record = load_strict(path)
        record["schema"] = "paired_audit.checkpoint.v1"
        from stylo.eval.paired_audit.checkpoints import _self_hash
        record["self_hash"] = _self_hash(record)
        dump_strict(record, path, trailing_newline=True)
        with pytest.raises(CheckpointError, match="bad schema"):
            store.scan_cell("lobo", "stylo", "A0")


class TestRunComplete:
    def test_run_complete_requires_exact_cell_set(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(CheckpointError):
            store.assert_run_complete({})            # empty != the 21 applied cells x 2 datasets

    def test_extra_cell_dir_is_fatal(self, tmp_path):
        from stylo.eval.paired_audit.applicability import registered_cells
        from stylo.eval.paired_audit.checkpoints import _DATASETS
        store = _store(tmp_path)
        _save(store, fold_index=0, work_id="w0")     # creates lobo/stylo/A0
        required = {(ds, m, c) for ds in _DATASETS for (m, c) in registered_cells()}
        store._assert_no_extra_dirs(required)        # only registered dirs so far -> ok
        (tmp_path / "ck" / "lobo" / "stylo" / "ZZZ").mkdir(parents=True)   # not a registered cell
        with pytest.raises(CheckpointError):
            store._assert_no_extra_dirs(required)
