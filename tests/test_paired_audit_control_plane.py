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


def _applied_record(with_vs_a0=True):
    r = {"status": "applied",
         "point": {"accuracy": 0.9, "macro_f1": 0.8, "top2": 0.95, "per_author_recall": {}},
         "per_work": [{"work_id": "a/w", "pred_label": 0, "rank": 1, "proba": [0.5, 0.5]}],
         "abs_accuracy_authorclustered_ci": [0.8, 0.95],
         "evidence": {"proba_digest": "e" * 64},
         "claim_status": "exploratory_internal"}
    if with_vs_a0:
        r["vs_A0"] = {"dacc": 0.02, "cluster_p": 0.01, "holm_p": 0.05, "significant": False,
                      "dacc_authorclustered_ci": [-0.01, 0.05], "mcnemar_p_diagnostic": 0.2}
    return r


# ── applicability matrix ─────────────────────────────────────────────────────
class TestApplicability:
    def test_invariants_21_applied_15_comparisons(self):
        ap.assert_matrix_invariants()
        assert len(ap.registered_cells()) == 21
        assert len(ap.holm_family()) == 15
        assert len(ap.applicability_matrix()) == 30

    def test_registered_statuses_match_protocol(self):
        assert ap.cell_status("bow_lr", "A3")["status"] == "not_applicable"
        assert ap.cell_status("delta_cos:500", "A1")["status"] == "already_in_legacy"
        assert ap.cell_status("char_cos", "A1")["status"] == "not_applicable"
        assert ap.cell_status("char_cos", "A3")["status"] == "not_applicable"
        eq = ap.cell_status("char_cos", "A2")
        assert eq["status"] == "equivalent" and eq["equivalent_to"] == "A4"
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
        ap.assert_cell_record("stylo", "A4", _applied_record(with_vs_a0=True))
        ap.assert_cell_record("stylo", "A0", _applied_record(with_vs_a0=False))   # A0 needs no vs_A0
        with pytest.raises(ap.ApplicabilityError):                                # non-A0 missing vs_A0
            ap.assert_cell_record("stylo", "A4", _applied_record(with_vs_a0=False))
        # non-applied must carry no metric
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("bow_lr", "A3", {"status": "not_applicable", "point": {"accuracy": 1}})
        ap.assert_cell_record("bow_lr", "A3", {"status": "not_applicable"})
        # equivalent must name the right target
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("char_cos", "A2", {"status": "equivalent", "equivalent_to": "A0"})
        ap.assert_cell_record("char_cos", "A2", {"status": "equivalent", "equivalent_to": "A4"})
        # vs_A0 must carry the full §4.1 key set (incl. the difference CI + diagnostic McNemar)
        bad = _applied_record(with_vs_a0=True)
        del bad["vs_A0"]["dacc_authorclustered_ci"]
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("stylo", "A4", bad)
        # status must match the registry
        with pytest.raises(ap.ApplicabilityError):
            ap.assert_cell_record("delta_cos:500", "A1", _applied_record())

    def test_cell_record_whitelist_rejects_any_metric_key(self):
        # a non-applied cell must carry NO metric under ANY key (whitelist, not a fixed denylist)
        for leak in ({"accuracy": 0.99}, {"cluster_p": 0.001}, {"significant": True}, {"proba": [1]}):
            with pytest.raises(ap.ApplicabilityError):
                ap.assert_cell_record("bow_lr", "A3", {"status": "not_applicable", **leak})
        ap.assert_cell_record("bow_lr", "A3",
                              {"status": "not_applicable", "reason": "x", "requested_axes": {}})

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
                             "scipy": "1.13", "sklearn": "1.5"},
        blas_thread_fingerprint={"threadpools": [], "thread_env": {}},
        applicability_matrix_digest=ap.applicability_matrix_digest(),
        a0_reference_shas={"lobo_books_txt": "4" * 64, "ruaa_reference_submission": "5" * 64},
        tolerances={"proba": {"atol": 1e-9, "rtol": 0, "dtype": "float64"}},
        corpus_chain={"legacy_anchor": rp_anchor(), "semantic_parity_digest": "6" * 64},
        golden_fixture_inventory_sha="7" * 64,
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

    def test_class_order_digest_deterministic_and_order_sensitive(self):
        a = rp.class_order_digest(["x", "y", "z"])
        assert a == rp.class_order_digest(["x", "y", "z"]) and _HEX64.match(a)
        assert a != rp.class_order_digest(["y", "x", "z"])

    def test_kernel_string_injection_rejected(self):
        # run_kind='smoke' so the confirmatory tolerances validation does not pre-empt the kernel walk
        release = platform.release()
        if not release:
            pytest.skip("no kernel release string on this platform")
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(run_kind="smoke", tolerances={"proba": {"dtype": release}}))

    def test_machine_string_injection_rejected(self):
        machine = platform.machine()
        if not machine:
            pytest.skip("no machine string on this platform")
        with pytest.raises(rp.RunPlanError):
            rp.build_run_plan(**_bindings(run_kind="smoke", tolerances={"proba": {"dtype": machine}}))

    def test_bare_os_name_not_falsely_rejected(self):
        system = platform.system()
        if not system or system == platform.machine():
            pytest.skip("no distinct OS-name string")
        plan = rp.build_run_plan(**_bindings(run_kind="smoke", tolerances={"proba": {"dtype": system}}))
        assert rp.run_id(plan)
