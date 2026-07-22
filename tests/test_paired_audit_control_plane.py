"""Synthetic tests for the applicability matrix (§2.4/§3.4) and the canonical RunPlan (§4.2).

No real corpus, dataset, or confirmatory cell is touched; all bindings are synthetic.
"""
from __future__ import annotations

import platform
import re

import pytest

from stylo.config import load_config
from stylo.eval.paired_audit import applicability as ap
from stylo.eval.paired_audit import run_plan as rp

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _evidence(model, cell):
    return {key: "e" * 64 for key in ap.required_evidence_digests(model, cell)}


def _applied_record(model="stylo", cell="A4", with_vs_a0=True):
    reg = ap.cell_status(model, cell)
    r = {"status": "applied",
         "requested_axes": reg["requested_axes"], "effective_axes": reg["effective_axes"],
         "point": {"accuracy": 0.9, "macro_f1": 0.8, "top2": 0.95, "per_author_recall": {}},
         "per_work": [{"work_id": "a/w", "true_label": 0, "pred_label": 0, "correct": True,
                       "rank": 1, "proba": [0.5, 0.5]}],
         "abs_accuracy_authorclustered_ci": [0.8, 0.95],
         "evidence": _evidence(model, cell),
         "claim_status": "exploratory_internal"}
    if with_vs_a0:
        r["vs_A0"] = {"dacc": 0.02, "cluster_p": 0.01, "holm_p": 0.05, "significant": False,
                      "dacc_authorclustered_ci": [-0.01, 0.05], "mcnemar_p_diagnostic": 0.2}
    return r


def _nonapplied_record(model, cell):
    reg = ap.cell_status(model, cell)
    r = {"status": reg["status"], "requested_axes": reg["requested_axes"],
         "effective_axes": reg["effective_axes"], "claim_status": "exploratory_internal"}
    if reg["equivalent_to"] is not None:
        r["equivalent_to"] = reg["equivalent_to"]
    return r


# ── applicability matrix ─────────────────────────────────────────────────────
class TestApplicability:
    def test_invariants_21_applied_15_comparisons(self):
        ap.assert_matrix_invariants()
        assert len(ap.registered_cells()) == 21
        assert len(ap.holm_family()) == 15
        assert len(ap.applicability_matrix()) == 30

    def test_registered_statuses_match_protocol(self):
        # only the three literal §4.1 status values; Delta A1 -> not_applicable with W already-in-legacy
        assert ap.cell_status("bow_lr", "A3")["status"] == "not_applicable"
        da1 = ap.cell_status("delta_cos:500", "A1")
        assert da1["status"] == "not_applicable" and da1["effective_axes"]["W"] == "already_in_legacy"
        assert ap.cell_status("char_cos", "A1")["status"] == "not_applicable"
        assert ap.cell_status("char_cos", "A3")["status"] == "not_applicable"
        eq = ap.cell_status("char_cos", "A2")
        assert eq["status"] == "equivalent_to" and eq["equivalent_to"] == "A4"
        assert ap.cell_status("char_cos", "A4")["effective_axes"] == {"W": "already_in_legacy",
                                                                      "F": "applied", "R": "not_applicable"}
        for c in ("A1", "A2", "A3", "A4"):
            assert ap.cell_status("majority", c)["status"] == "not_applicable"
        for c in ap.CELLS:
            assert ap.cell_status("stylo_stack", c)["status"] == "applied"

    def test_holm_family_exact_membership(self):
        fam = set(ap.holm_family())
        assert ("stylo", "A4") in fam and ("stylo_stack", "A1") in fam
        assert ("char_cos", "A4") in fam
        assert ("char_cos", "A2") not in fam          # equivalent, not a comparison
        assert ("delta_cos:500", "A1") not in fam     # already_in_legacy
        assert ("bow_lr", "A3") not in fam            # not_applicable
        assert not any(m == "majority" for m, _ in fam)

    def test_matrix_digest_is_deterministic_hex(self):
        d1, d2 = ap.applicability_matrix_digest(), ap.applicability_matrix_digest()
        assert d1 == d2 and _HEX64.match(d1)

    def test_cell_record_validation(self):
        # an evidence-free applied cell is rejected (§4.1 schema)
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", {"status": "applied"})
        ap.assert_cell_record("stylo", "A4", _applied_record("stylo", "A4", with_vs_a0=True))
        ap.assert_cell_record("stylo", "A0", _applied_record("stylo", "A0", with_vs_a0=False))
        with pytest.raises(ap.ApplicabilityError):                                # non-A0 missing vs_A0
            ap.assert_cell_record("stylo", "A4", _applied_record("stylo", "A4", with_vs_a0=False))
        # a not_applicable cell carries the registry axes and no metric
        na = _nonapplied_record("bow_lr", "A3")
        ap.assert_cell_record("bow_lr", "A3", na)
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("bow_lr", "A3", {**na, "point": {"accuracy": 1}})
        # equivalent_to must name the right target
        eq = _nonapplied_record("char_cos", "A2")
        ap.assert_cell_record("char_cos", "A2", eq)
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("char_cos", "A2", {**eq, "equivalent_to": "A0"})
        # requested/effective axes must equal the registry
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", {**_applied_record("stylo", "A4"),
                                                  "effective_axes": {"W": "applied", "F": "applied", "R": "not_applicable"}})
        # vs_A0 must carry the full §4.1 key set (incl. the difference CI + diagnostic McNemar)
        bad = _applied_record("stylo", "A4", with_vs_a0=True)
        del bad["vs_A0"]["dacc_authorclustered_ci"]
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", bad)
        # status must match the registry (delta A1 is not_applicable, not applied)
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("delta_cos:500", "A1", _applied_record("delta_cos:500", "A1"))

    def test_calibration_passport_is_a_full_structure_not_a_digest(self):
        # §4.1: the stack calibration_passport is a literal structure {disabled,mode,meta}, not a digest;
        # and Delta carries delta_mean_std_centroid_digest (the literal name)
        assert "delta_mean_std_centroid_digest" in ap.required_evidence_digests("delta_cos:500", "A4")
        assert ap.required_evidence_passports("stylo_stack", "A4") == ("calibration_passport",)
        rec = _applied_record("stylo_stack", "A4")
        with pytest.raises(ap.ApplicabilityError):                 # missing passport
            ap.assert_cell_record("stylo_stack", "A4", rec)
        rec["evidence"]["calibration_passport"] = "e" * 64         # a digest is NOT a full passport
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo_stack", "A4", rec)
        rec["evidence"]["calibration_passport"] = {"calibration_disabled": False, "mode": "sigmoid",
                                                   "meta": {}}
        ap.assert_cell_record("stylo_stack", "A4", rec)            # a full structure -> ok

    def test_applied_evidence_field_validation(self):
        # proba width vs class order, non-finite metric, and non-hex evidence digest all fail closed
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", _applied_record("stylo", "A4"),
                                  probability_class_order=["a", "b", "c"])   # proba is width 2
        bad_metric = _applied_record("stylo", "A4")
        bad_metric["point"]["accuracy"] = float("nan")
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", bad_metric)
        bad_digest = _applied_record("stylo", "A4")
        bad_digest["evidence"]["proba_digest"] = "nothex"
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", bad_digest)
        # an applied axis must carry its proving digest (stylo A4 has W applied -> ordered_weight_digest)
        missing_wd = _applied_record("stylo", "A4")
        del missing_wd["evidence"]["ordered_weight_digest"]
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", missing_wd)

    def test_cell_record_whitelist_rejects_any_metric_key(self):
        na = _nonapplied_record("bow_lr", "A3")
        for leak in ({"accuracy": 0.99}, {"cluster_p": 0.001}, {"significant": True}, {"proba": [1]}):
            with pytest.raises(ap.ApplicabilityError):
                ap.assert_cell_record("bow_lr", "A3", {**na, **leak})
        ap.assert_cell_record("bow_lr", "A3", na)      # clean metadata-only record passes

    def test_invariants_catch_count_preserving_swap(self, monkeypatch):
        # flip bow_lr A3 -> applied and char_cos A4 -> not_applicable: counts stay 21/15 but the
        # frozen §2.4 per-model decomposition is violated and must fail closed
        monkeypatch.setitem(ap._STATUS["bow_lr"], "A3", ap._APPLIED)
        monkeypatch.setitem(ap._STATUS["char_cos"], "A4", ("not_applicable", "swapped", None))
        assert len(ap.registered_cells()) == 21 and len(ap.holm_family()) == 15
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_matrix_invariants()

    def test_holm_family_completeness_guard(self):
        full = list(ap.holm_family())
        ap.assert_holm_family_complete(full)
        with pytest.raises(ap.ApplicabilityError):       # missing one -> family invalid, m not reduced
            ap.assert_holm_family_complete(full[:-1])
        with pytest.raises(ap.ApplicabilityError):       # extra
            ap.assert_holm_family_complete(full + [("majority", "A0")])
        with pytest.raises(ap.ApplicabilityError):       # duplicate
            ap.assert_holm_family_complete(full + [full[0]])


# ── RunPlan + fingerprints ───────────────────────────────────────────────────
def _bindings(**over):
    b = dict(
        run_kind="confirmatory",
        git_commit="a" * 40, git_dirty=False,
        execution_source_sha256="1" * 64, env_lock_sha256="2" * 64, config_id="3" * 64,
        runtime_fingerprint={"python": "3.11.0", "libc": "glibc/2.39", "numpy": "2.0",
                             "scipy": "1.13", "sklearn": "1.5", "spacy": "3.8"},
        blas_thread_fingerprint={"threadpools": [], "thread_env": {}},
        applicability_matrix_digest=ap.applicability_matrix_digest(),
        a0_reference_shas={"lobo_books_txt": "4" * 64, "ruaa_reference_submission": "5" * 64},
        tolerances={q: {"atol": 1e-9, "rtol": 0, "dtype": "float64"}
                    for q in rp.REGISTERED_TOLERANCE_QUANTITIES},
        corpus_chain={"legacy_anchor": rp_anchor(), "semantic_parity_digest": "6" * 64},
        golden_fixture_inventory_sha="7" * 64,
        evaluator_identity={"name": "work_balanced_ablation_factory",
                            "import_module": "stylo.work_balanced_ablation_screen",
                            "import_qualname": "make_factory_for_ablation",
                            "source_digest": "a" * 64, "estimator_config_digest": "b" * 64,
                            "mechanism_passport_digest": "c" * 64},
        lobo=dict(dataset_digest="a" * 64, fold_manifest_digest="b" * 64,
                  probability_class_order=["x", "z"], metric_label_order=["x"],
                  run_contract_digest="c" * 64),
        ruaa=dict(dataset_digest="d" * 64, fold_manifest_digest="e" * 64,
                  probability_class_order=["x"], metric_label_order=["x"],
                  run_contract_digest="f" * 64, selection_digest="9" * 64),
    )
    b.update(over)
    return b


def rp_anchor():
    from stylo.eval.paired_audit.semantic_parity import LEGACY_ANCHOR
    return LEGACY_ANCHOR


class TestRunPlan:
    def test_runtime_fingerprint_binds_stack_omits_kernel(self):
        fp = rp.runtime_fingerprint()
        for k in ("python", "libc", "numpy", "scipy", "sklearn"):
            assert fp.get(k)
        release = platform.release()
        if release:
            assert release not in fp.values()             # no kernel release string
            assert release not in str(fp)

    def test_env_lock_and_source_hashes_deterministic(self):
        assert _HEX64.match(rp.env_lock_sha256())
        a, b = rp.execution_source_sha256(), rp.execution_source_sha256()
        assert a == b and _HEX64.match(a)

    def test_config_id_and_git_info(self):
        assert _HEX64.match(rp.config_id(load_config()))
        info = rp.git_commit_info()
        assert set(info) == {"git_commit", "git_dirty"} and isinstance(info["git_dirty"], bool)

    def test_build_run_plan_and_stable_run_id(self):
        plan = rp.build_run_plan(**_bindings())
        rid = rp.run_id(plan)
        assert _HEX64.match(rid)
        assert rp.run_id(rp.build_run_plan(**_bindings())) == rid       # deterministic
        assert plan["stats"]["seed"] == 42 and plan["stats"]["bootstrap_B"] == 10000

    def test_run_id_changes_on_any_binding_change(self):
        base = rp.run_id(rp.build_run_plan(**_bindings()))
        perturbations = [
            {"applicability_matrix_digest": "0" * 64},
            {"env_lock_sha256": "f" * 64},
            {"execution_source_sha256": "e" * 64},
            {"git_commit": "b" * 40},
            {"corpus_chain": {"legacy_anchor": "0" * 64, "semantic_parity_digest": "6" * 64}},
            {"lobo": _bindings()["lobo"] | {"dataset_digest": "0" * 64}},
        ]
        for over in perturbations:
            assert rp.run_id(rp.build_run_plan(**_bindings(**over))) != base

    def test_stat_setting_change_changes_run_id(self):
        # non-confirmatory kind allows a stat override; the change must re-key the run_id
        base = rp.run_id(rp.build_run_plan(**_bindings(run_kind="smoke")))
        moved = rp.build_run_plan(**_bindings(run_kind="smoke", stats={**rp.FROZEN_STATS, "seed": 43}))
        assert rp.run_id(moved) != base

    def test_missing_bindings_fail_closed(self):
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(lobo={"dataset_digest": "a" * 64}))
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(a0_reference_shas={"lobo_books_txt": "4" * 64}))
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(corpus_chain={"legacy_anchor": "6" * 64}))
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(git_dirty="no"))

    def test_empty_top_level_scalar_bindings_fail_closed(self):
        for over in ({"execution_source_sha256": ""}, {"env_lock_sha256": None}, {"config_id": ""},
                     {"applicability_matrix_digest": ""}, {"golden_fixture_inventory_sha": ""},
                     {"git_commit": ""}, {"run_kind": ""}, {"execution_source_sha256": "nothex"}):
            with pytest.raises(rp.RunPlanError):
                rp.build_run_plan(**_bindings(**over))

    def test_nonfrozen_or_empty_stats_rejected(self):
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(stats={}))                               # missing keys
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(stats={**rp.FROZEN_STATS, "seed": 43}))  # confirmatory != frozen

    def test_malformed_fingerprints_rejected(self):
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(runtime_fingerprint={"python": "3.11"}))  # missing libc/numpy/...
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(blas_thread_fingerprint={"threadpools": []}))  # no thread_env

    def test_confirmatory_requires_clean_tree_and_tolerances(self):
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(git_dirty=True))                 # confirmatory + dirty tree
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(tolerances={}))                  # empty tolerances
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(tolerances={"proba": {"atol": 1e-9}}))  # missing rtol/dtype
        assert rp.run_id(rp.build_run_plan(**_bindings(run_kind="smoke", git_dirty=True)))

    def test_confirmatory_requires_exactly_the_frozen_tolerances(self):
        assert rp.run_id(rp.build_run_plan(**_bindings(tolerances=dict(rp.FROZEN_TOLERANCES))))  # ok
        full = dict(rp.FROZEN_TOLERANCES)
        partial = {q: full[q] for q in list(full)[:-1]}                       # drop one quantity
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(tolerances=partial))
        extra = {**full, "bogus": {"atol": 1e-9, "rtol": 0, "dtype": "float64"}}
        with pytest.raises(rp.RunPlanError):                                  # an unregistered quantity
            rp.build_run_plan(**_bindings(tolerances=extra))
        oversized = {**full, "accuracy": {"atol": 1e9, "rtol": 0.0, "dtype": "float64"}}
        with pytest.raises(rp.RunPlanError):                                  # would neuter the auditor
            rp.build_run_plan(**_bindings(tolerances=oversized))

    def test_class_order_digest_deterministic_and_order_sensitive(self):
        a = rp.class_order_digest(["x", "y", "z"])
        assert a == rp.class_order_digest(["x", "y", "z"]) and _HEX64.match(a)
        assert a != rp.class_order_digest(["y", "x", "z"])

    def test_runtime_fingerprint_allowlist_rejects_unknown_field(self):
        # structural allowlist: a kernel/OS field is not even representable (no denylist scan)
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(runtime_fingerprint={
                "python": "3.11", "libc": "glibc/2.39", "numpy": "2.0", "scipy": "1", "sklearn": "1",
                "spacy": "3.8", "kernel_release": "7.1.3-arch"}))

    def test_confirmatory_requires_spacy_and_valid_run_kind(self):
        with pytest.raises(rp.RunPlanError):                        # confirmatory without spaCy
            rp.build_run_plan(**_bindings(runtime_fingerprint={
                "python": "3.11", "libc": "glibc/2.39", "numpy": "2.0", "scipy": "1", "sklearn": "1"}))
        with pytest.raises(rp.RunPlanError):                        # unknown run_kind
            rp.build_run_plan(**_bindings(run_kind="whatever"))

    def test_nonfinite_or_bad_dtype_tolerances_rejected(self):
        for bad in ({"proba": {"atol": float("nan"), "rtol": 0, "dtype": "float64"}},
                    {"proba": {"atol": float("inf"), "rtol": 0, "dtype": "float64"}},
                    {"proba": {"atol": -1e-9, "rtol": 0, "dtype": "float64"}},
                    {"proba": {"atol": 1e-9, "rtol": 0, "dtype": "float16"}}):   # unregistered dtype
            with pytest.raises(rp.RunPlanError):
                rp.build_run_plan(**_bindings(tolerances=bad))


# ── §4.2 production evaluator identity ────────────────────────────────────────
def _fake_factory(dataset, ds_obj, model, cell, fold_index, work_id, ablation):   # a stand-in estimator
    return {"pred_label": 0, "correct": True, "rank": 1, "probabilities": [1.0]}


def _spec(name, **over):
    kw = dict(estimator_config={"C": 1.0, "solver": "liblinear"},
              mechanism_passport={"W": "ordered_weight", "F": "feature_state", "R": "relative_fw"})
    kw.update(over)
    return rp.EvaluatorSpec(name=name, fn=_fake_factory, **kw)


class TestEvaluatorIdentity:
    def test_identity_recomputes_source_and_binds_config(self):
        ident = rp.evaluator_identity(_spec("work_balanced_ablation_factory"), confirmatory=True)
        assert set(ident) == set(rp._EVALUATOR_IDENTITY_KEYS)
        assert ident["import_qualname"] == "_fake_factory"
        assert _HEX64.match(ident["source_digest"])
        # a different estimator config re-keys the identity (and thus the run_id)
        other = rp.evaluator_identity(_spec("work_balanced_ablation_factory",
                                            estimator_config={"C": 2.0}), confirmatory=True)
        assert other["estimator_config_digest"] != ident["estimator_config_digest"]

    def test_bare_callable_rejected(self):
        with pytest.raises(rp.RunPlanError):
            rp.evaluator_identity(_fake_factory, confirmatory=False)     # not an EvaluatorSpec

    def test_unregistered_name_rejected_only_in_confirmatory(self):
        with pytest.raises(rp.RunPlanError):
            rp.evaluator_identity(_spec("smoke_dummy"), confirmatory=True)
        # allowed under a non-confirmatory kind
        assert rp.evaluator_identity(_spec("smoke_dummy"), confirmatory=False)["name"] == "smoke_dummy"

    def test_empty_config_or_passport_rejected(self):
        with pytest.raises(rp.RunPlanError):
            rp.evaluator_identity(_spec("work_balanced_ablation_factory", estimator_config={}),
                                  confirmatory=True)
        with pytest.raises(rp.RunPlanError):
            rp.evaluator_identity(_spec("work_balanced_ablation_factory", mechanism_passport={}),
                                  confirmatory=True)

    def test_identity_folds_into_run_id(self):
        base = rp.run_id(rp.build_run_plan(**_bindings()))
        moved = rp.build_run_plan(**_bindings(evaluator_identity={
            **_bindings()["evaluator_identity"], "source_digest": "0" * 64}))
        assert rp.run_id(moved) != base

    def test_confirmatory_plan_rejects_unregistered_evaluator(self):
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(evaluator_identity={
                **_bindings()["evaluator_identity"], "name": "smoke_dummy"}))

    def test_plan_rejects_non_hex_identity_digest(self):
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(evaluator_identity={
                **_bindings()["evaluator_identity"], "source_digest": "nothex"}))


class TestWellformedRunPlan:
    def test_a_built_plan_rebuilds_to_itself(self):
        rp.assert_wellformed_run_plan(rp.build_run_plan(**_bindings()))          # confirmatory
        rp.assert_wellformed_run_plan(rp.build_run_plan(**_bindings(run_kind="smoke", git_dirty=True)))

    def test_forged_confirmatory_plan_does_not_rebuild(self):
        plan = rp.build_run_plan(**_bindings())
        for over in ({"tolerances": {q: {"atol": 1e9, "rtol": 0.0, "dtype": "float64"}
                                     for q in rp.REGISTERED_TOLERANCE_QUANTITIES}},
                     {"stats": {**rp.FROZEN_STATS, "seed": 7}},
                     {"git_dirty": True}):
            forged = {**plan, **over}
            with pytest.raises(rp.RunPlanError):        # build invariants re-raise on the forged plan
                rp.assert_wellformed_run_plan(forged)

    def test_missing_build_field_rejected(self):
        plan = rp.build_run_plan(**_bindings())
        with pytest.raises(rp.RunPlanError):
            rp.assert_wellformed_run_plan({k: v for k, v in plan.items() if k != "config_id"})
