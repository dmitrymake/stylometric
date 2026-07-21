"""Synthetic tests for the path-guarded transient store and verified publisher (§4.4/§8)."""
from __future__ import annotations

import pathlib

import pytest

import copy

from stylo.jsonio import dump_strict, load_strict
from stylo.eval.paired_audit import applicability as ap
from stylo.eval.paired_audit import headline as hl
from stylo.eval.paired_audit import inference as inf
from stylo.eval.paired_audit import publisher as pub
from stylo.eval.paired_audit import result_audit as ra
from stylo.eval.paired_audit import run_plan as rp
from stylo.eval.paired_audit.headline import HEADLINE_ENDPOINT
from stylo.eval.paired_audit.semantic_parity import LEGACY_ANCHOR

# A confirmatory-shaped fixture built with the REAL metric functions so the publisher's publish-time
# independent auditor passes it (the publisher no longer trusts a bare result_audit.passed flag).
_STATS = rp.FROZEN_STATS
_ITERS, _SEED, _Q = _STATS["bootstrap_iters"], _STATS["seed"], _STATS["quantiles"]
_MARGIN, _B = _STATS["noninferiority_margin"], _STATS["bootstrap_B"]
_PROB = ["aa", "bb"]
_MIDX = [0, 1]
# 4 works / 2 authors, one wrong; every cell shares these vectors (so A4 == A0 -> dacc 0, cluster_p 1)
_VEC = [
    {"work_id": "aa/w1", "true_label": 0, "pred_label": 0, "correct": True, "rank": 1, "proba": [0.6, 0.4]},
    {"work_id": "aa/w2", "true_label": 0, "pred_label": 0, "correct": True, "rank": 1, "proba": [0.7, 0.3]},
    {"work_id": "bb/w1", "true_label": 1, "pred_label": 1, "correct": True, "rank": 1, "proba": [0.4, 0.6]},
    {"work_id": "bb/w2", "true_label": 1, "pred_label": 0, "correct": False, "rank": 2, "proba": [0.55, 0.45]},
]


def _run_plan():
    return rp.build_run_plan(
        run_kind="confirmatory", git_commit="a" * 40, git_dirty=False,
        execution_source_sha256="1" * 64, env_lock_sha256="2" * 64, config_id="3" * 64,
        runtime_fingerprint={"python": "3.11", "libc": "glibc/2.39", "numpy": "2", "scipy": "1",
                             "sklearn": "1", "spacy": "3.8"},
        blas_thread_fingerprint={"threadpools": [], "thread_env": {}},
        applicability_matrix_digest=ap.applicability_matrix_digest(),
        a0_reference_shas={"lobo_books_txt": "4" * 64, "ruaa_reference_submission": "5" * 64},
        evaluator_identity={"name": "work_balanced_ablation_factory", "import_module": "m",
                            "import_qualname": "q", "source_digest": "a" * 64,
                            "estimator_config_digest": "b" * 64, "mechanism_passport_digest": "c" * 64},
        tolerances=dict(rp.FROZEN_TOLERANCES),
        corpus_chain={"legacy_anchor": LEGACY_ANCHOR, "semantic_parity_digest": "6" * 64},
        golden_fixture_inventory_sha="7" * 64,
        lobo=dict(dataset_digest="a" * 64, fold_manifest_digest="b" * 64,
                  probability_class_order=_PROB, metric_label_order=_PROB, run_contract_digest="c" * 64),
        ruaa=dict(dataset_digest="d" * 64, fold_manifest_digest="e" * 64,
                  probability_class_order=_PROB, metric_label_order=_PROB, run_contract_digest="f" * 64,
                  selection_digest="9" * 64))


_PLAN = _run_plan()
RUN_ID = rp.run_id(_PLAN)


def _universes():
    # match the per-dataset digests bound in _PLAN (lobo a/b, ruaa d/e)
    return {"lobo": {"dataset_digest": "a" * 64, "fold_manifest_digest": "b" * 64,
                     "probability_class_order": _PROB, "metric_label_order": _PROB},
            "ruaa": {"dataset_digest": "d" * 64, "fold_manifest_digest": "e" * 64,
                     "probability_class_order": _PROB, "metric_label_order": _PROB}}


def _attestation():
    return {"git_commit": "a" * 40, "run_kind": "confirmatory", "audit_version": rp.AUDIT_VERSION,
            "execution_source_sha256": "1" * 64, "env_lock_sha256": "2" * 64, "config_id": "3" * 64,
            "golden_fixture_inventory_sha": "7" * 64}


def _evidence(m, c):
    return {key: "e" * 64 for key in ap.required_evidence_digests(m, c)}


def _grid():
    """A consistent 30-cell grid + the Holm family, computed from _VEC with the real metric functions."""
    a = ra._arrays(_VEC)
    point = ra._point(a, _MIDX)
    abs_ci = hl.author_clustered_accuracy_ci(a["correct"], a["authors"], iters=_ITERS, seed=_SEED, quantiles=_Q)
    dacc_ci = hl.paired_accuracy_diff_ci(a["correct"], a["correct"], a["authors"], iters=_ITERS, seed=_SEED, quantiles=_Q)
    cp = inf.paired_cluster_pvalue(a["correct"], a["correct"], a["authors"], B=_B, seed=_SEED)
    mc = inf.mcnemar_diagnostic(a["correct"], a["correct"])
    grid = {}
    for m in ap.MODELS:
        for c in ap.CELLS:
            reg = ap.cell_status(m, c)
            if reg["status"] != "applied":
                r = {"status": reg["status"], "requested_axes": reg["requested_axes"],
                     "effective_axes": reg["effective_axes"], "claim_status": "exploratory_internal"}
                if reg["equivalent_to"] is not None:
                    r["equivalent_to"] = reg["equivalent_to"]
                grid[f"{m}/{c}"] = r
                continue
            r = {"status": "applied", "requested_axes": reg["requested_axes"],
                 "effective_axes": reg["effective_axes"], "point": copy.deepcopy(point),
                 "per_work": copy.deepcopy(_VEC),
                 "abs_accuracy_authorclustered_ci": [abs_ci["lo"], abs_ci["hi"]],
                 "evidence": _evidence(m, c), "claim_status": "exploratory_internal"}
            if c != "A0":
                r["vs_A0"] = {"dacc": 0.0, "dacc_authorclustered_ci": [dacc_ci["lo"], dacc_ci["hi"]],
                              "cluster_p": cp, "holm_p": 1.0, "significant": False,
                              "mcnemar_p_diagnostic": mc["mcnemar_p_diagnostic"]}
            grid[f"{m}/{c}"] = r
    holm = inf.holm_over_registered_family({(m, c): grid[f"{m}/{c}"]["vs_A0"]["cluster_p"]
                                            for (m, c) in ap.holm_family()})
    for (m, c), hp in holm.items():
        grid[f"{m}/{c}"]["vs_A0"]["holm_p"] = hp["holm_p"]
        grid[f"{m}/{c}"]["vs_A0"]["significant"] = hp["significant"]
    return grid, {f"{m}/{c}": hp for (m, c), hp in holm.items()}


def _base_summary():
    grid, holm = _grid()
    a = ra._arrays(_VEC)
    head = hl.evaluate_headline(a["correct"], a["correct"], a["authors"], margin=_MARGIN, iters=_ITERS,
                                seed=_SEED, quantiles=_Q)
    return {"run_id": RUN_ID, "claim_status": "exploratory_internal",
            "cells": {"lobo": grid, "ruaa": copy.deepcopy(grid)},
            "holm": {"lobo": holm, "ruaa": copy.deepcopy(holm)},
            "headline": {"endpoint": head["endpoint"], "decision": head["decision"],
                         "diff_ci": head["diff_ci"], "margin": _MARGIN},
            "result_audit": {"passed": True, "auditor": "independent_recompute_v1"},
            "run_plan": _PLAN, "universes": _universes(),
            "continuous_tolerances": dict(rp.FROZEN_TOLERANCES), "attestation": _attestation(),
            "run_id_source": "canonical_run_plan_sha256"}


_SUMMARY = _base_summary()


def _summary(**over):
    s = copy.deepcopy(_SUMMARY)
    s.update(over)
    return s


def _vectors():
    return {f"{ds}/{m}/{c}": copy.deepcopy(_VEC)
            for ds in ("lobo", "ruaa") for (m, c) in ap.registered_cells()}


class TestPathGuard:
    def test_rejects_headline_basename(self, tmp_path):
        with pytest.raises(pub.PublisherError):
            pub.assert_writable_audit_path(tmp_path / "lobo_books.txt",
                                           docs_root=tmp_path, allow_published=True)
        with pytest.raises(pub.PublisherError):
            pub.assert_writable_audit_path(tmp_path / "README.md",
                                           docs_root=tmp_path, allow_published=True)

    def test_rejects_more_frozen_basenames(self, tmp_path):
        for name in ("final_comparison.csv", "final_comparison.v2.csv", "p0_baseline_snapshot.json"):
            with pytest.raises(pub.PublisherError):
                pub.assert_writable_audit_path(tmp_path / name, docs_root=tmp_path, allow_published=True)

    def test_rejects_escape_and_unallowed(self, tmp_path):
        with pytest.raises(pub.PublisherError):                       # escapes docs root
            pub.assert_writable_audit_path(tmp_path.parent / "escape.json",
                                           docs_root=tmp_path, run_id=RUN_ID)
        with pytest.raises(pub.PublisherError):                       # not in any allowed namespace
            pub.assert_writable_audit_path(tmp_path / "random.json", docs_root=tmp_path)

    def test_rejects_dotdot_tail_that_escapes_after_normalization(self, tmp_path):
        # a non-existent tail climbing out with '..' must be rejected (normalized before containment)
        escaping = (tmp_path / pub.ARCHIVE_DIRNAME / pub.VERSIONS_DIR / "tok"
                    / ".." / ".." / ".." / ".." / "outside" / "pwned.json")
        with pytest.raises(pub.PublisherError):
            pub.assert_writable_audit_path(escaping, docs_root=tmp_path, allow_published=True)

    def test_allows_transient_and_published(self, tmp_path):
        t = tmp_path / pub.RUNS_SUBPATH / RUN_ID / "checkpoint.json"
        t.parent.mkdir(parents=True)
        pub.assert_writable_audit_path(t, docs_root=tmp_path, run_id=RUN_ID)
        pub.assert_writable_audit_path(tmp_path / pub.SUMMARY_NAME,
                                       docs_root=tmp_path, allow_published=True)


class TestTransient:
    def test_write_transient_lands_in_run_namespace(self, tmp_path):
        p = pub.write_transient(RUN_ID, "fold_0000.json", {"x": 1}, docs_root=tmp_path)
        assert pub.RUNS_SUBPATH.as_posix() in p.as_posix() and RUN_ID in p.parts
        assert load_strict(p) == {"x": 1}

    def test_transient_headline_name_rejected(self, tmp_path):
        with pytest.raises(pub.PublisherError):
            pub.write_transient(RUN_ID, "lobo_books.txt", {"x": 1}, docs_root=tmp_path)


class TestPublish:
    def test_publish_and_load_round_trip(self, tmp_path):
        published = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        # committed summary + archive structure
        assert (tmp_path / pub.SUMMARY_NAME).is_file()
        vdir = published["versioned_dir"]
        assert (vdir / pub.SHA256SUMS_NAME).is_file()
        assert (tmp_path / pub.ARCHIVE_DIRNAME / pub.CURRENT_NAME).is_file()
        assert (tmp_path / pub.ARCHIVE_DIRNAME / pub.COMPLETE_NAME).is_file()
        assert published["summary"]["self_hash"]
        # every per-work vector is content-addressed and referenced by hash
        inv = published["summary"]["per_work_archive"]
        assert set(inv) == set(_vectors())
        for ref in inv.values():
            assert (vdir / ref["filename"]).is_file()
        loaded = pub.load_published_audit(tmp_path)
        assert loaded["version"] == published["version"]
        assert loaded["summary"]["run_id"] == RUN_ID

    def test_idempotent_republish(self, tmp_path):
        a = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        b = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        assert a["version"] == b["version"]

    def test_summary_claim_and_run_id_required(self, tmp_path):
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(_summary(claim_status="internal_evidence"), _vectors(), docs_root=tmp_path)
        s = _summary()
        del s["run_id"]
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(s, _vectors(), docs_root=tmp_path)

    def test_recovers_from_staging_orphan(self, tmp_path):
        # a leftover .staging_ dir (crash before os.replace) is inert; publish + load still succeed
        versions = tmp_path / pub.ARCHIVE_DIRNAME / pub.VERSIONS_DIR
        versions.mkdir(parents=True)
        (versions / ".staging_orphan").mkdir()
        (versions / ".staging_orphan" / "half.json").write_text("{", encoding="utf-8")
        published = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        assert pub.load_published_audit(tmp_path)["version"] == published["version"]


class TestFinalAssembly:
    def test_arbitrary_or_partial_summary_rejected(self, tmp_path):
        with pytest.raises(pub.PublisherError):                       # arbitrary mapping
            pub.publish_audit({"run_id": "a" * 64, "claim_status": "exploratory_internal"},
                              _vectors(), docs_root=tmp_path)
        with pytest.raises(pub.PublisherError):                       # non-sha run_id
            pub.publish_audit(_summary(run_id="short"), _vectors(), docs_root=tmp_path)
        with pytest.raises(pub.PublisherError):                       # missing a dataset in cells
            bad = _summary(); bad["cells"].pop("ruaa")
            pub.publish_audit(bad, _vectors(), docs_root=tmp_path)
        with pytest.raises(pub.PublisherError):                       # incomplete Holm family
            bad = _summary(); bad["holm"]["lobo"].pop(next(iter(bad["holm"]["lobo"])))
            pub.publish_audit(bad, _vectors(), docs_root=tmp_path)
        with pytest.raises(pub.PublisherError):                       # per-work vectors missing cells
            pub.publish_audit(_summary(), {"lobo/stylo/A4": [{"work_id": "a/w"}]}, docs_root=tmp_path)
        with pytest.raises(pub.PublisherError):                       # wrong headline endpoint
            pub.publish_audit(_summary(headline={"endpoint": "bogus", "decision": "relabel"}),
                              _vectors(), docs_root=tmp_path)


class TestArchiveCommittable:
    def test_guard_flags_gitignored_archive_and_passes_whitelisted(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        docs = tmp_path / "docs"
        docs.mkdir()
        (tmp_path / ".gitignore").write_text("docs/*\n", encoding="utf-8")
        with pytest.raises(pub.PublisherError):
            pub.assert_archive_committable(docs, repo_root=tmp_path)
        (tmp_path / ".gitignore").write_text(
            "docs/*\n!docs/work_balanced_paired_audit_v1/\n!docs/work_balanced_paired_audit_v1/**\n",
            encoding="utf-8")
        pub.assert_archive_committable(docs, repo_root=tmp_path)      # whitelisted -> ok

    def test_guard_is_noop_outside_git(self, tmp_path):
        pub.assert_archive_committable(tmp_path / "docs")             # not a git repo -> no-op


class TestArtifactCompleteness:
    def test_run_id_must_recompute_from_embedded_run_plan(self):
        with pytest.raises(pub.PublisherError):                # run_id != rp.run_id(run_plan)
            pub.verify_final_assembly(_summary(run_id="f" * 64), _vectors())
        with pytest.raises(pub.PublisherError):                # a bogus/partial run_plan
            pub.verify_final_assembly(_summary(run_plan={"schema": "x"}), _vectors())
        with pytest.raises(pub.PublisherError):               # no embedded run_plan at all
            s = _summary(); s.pop("run_plan"); pub.verify_final_assembly(s, _vectors())

    def test_missing_completeness_sections_rejected(self):
        for section in ("universes", "continuous_tolerances", "attestation"):
            s = _summary(); s.pop(section)
            with pytest.raises(pub.PublisherError):
                pub.verify_final_assembly(s, _vectors())

    def test_headline_margin_must_equal_frozen_plan_margin(self):
        # a publish-boundary craft cannot decide the headline under a softer margin than the frozen δ
        s = _summary()
        s["headline"] = {**s["headline"], "margin": 0.005, "decision": "inconclusive"}
        with pytest.raises(pub.PublisherError):
            pub.verify_final_assembly(s, _vectors())

    def test_malformed_headline_ci_surfaces_as_publisher_error(self):
        s = _summary()
        s["headline"] = {**s["headline"], "diff_ci": {"point": 0.0, "lo": 0.5, "hi": -0.5}}
        with pytest.raises(pub.PublisherError):                      # lo>hi -> HeadlineError -> PublisherError
            pub.verify_final_assembly(s, _vectors())

    def test_published_summary_round_trips_and_recomputes_run_id(self, tmp_path):
        pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        loaded = pub.load_published_audit(tmp_path)
        assert rp.run_id(loaded["summary"]["run_plan"]) == loaded["summary"]["run_id"] == RUN_ID

    def test_fixture_is_independently_auditor_consistent(self):
        # the fixture summary/vectors are self-consistent under the independent auditor
        ra.audit_results(_summary(), _vectors(), _PLAN)


class TestPublishGate:
    def test_publish_run_kind_must_match_the_embedded_run_plan(self, tmp_path):
        # a confirmatory summary cannot be published under a smoke target (and vice versa) — the guard
        # is bound to the artifact's OWN run_kind, not just the caller's kwarg
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path, run_kind="smoke")

    def test_publisher_re_derives_and_rejects_a_tampered_metric(self, tmp_path):
        # the publisher does not trust result_audit.passed — it re-runs the auditor over the vectors
        s = _summary()
        s["cells"]["lobo"]["stylo/A0"]["point"]["accuracy"] = 0.999   # contradicts its own vectors
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(s, _vectors(), docs_root=tmp_path)
        s2 = _summary()
        s2["result_audit"] = {"passed": True, "auditor": "x"}          # flag says pass, but vectors lie
        s2["cells"]["ruaa"]["stylo/A4"]["vs_A0"]["cluster_p"] = 0.000001
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(s2, _vectors(), docs_root=tmp_path)

    def test_forged_confirmatory_run_plan_rejected(self, tmp_path):
        # an embedded confirmatory plan is re-validated against EVERY build invariant, not just its id:
        # a forged oversized tolerance / weakened bootstrap / inflated margin cannot publish or load
        for over in ({"tolerances": {q: {"atol": 1e9, "rtol": 0.0, "dtype": "float64"}
                                     for q in rp.REGISTERED_TOLERANCE_QUANTITIES}},
                     {"stats": {**rp.FROZEN_STATS, "bootstrap_B": 1, "bootstrap_iters": 1}},
                     {"stats": {**rp.FROZEN_STATS, "noninferiority_margin": 0.5}}):
            plan = copy.deepcopy(_PLAN)
            plan.update(over)
            s = _summary(run_plan=plan, run_id=rp.run_id(plan))
            with pytest.raises(pub.PublisherError):                    # id recomputes, but plan is forged
                pub.verify_final_assembly(s, _vectors())
            with pytest.raises(pub.PublisherError):
                pub.publish_audit(s, _vectors(), docs_root=tmp_path)

    def test_universes_class_order_must_match_run_plan(self):
        s = _summary()
        s["universes"]["lobo"] = {**s["universes"]["lobo"], "probability_class_order": ["bb", "aa"]}
        with pytest.raises(pub.PublisherError):
            pub.verify_final_assembly(s, _vectors())

    def test_malformed_vectors_surface_as_publisher_error(self, tmp_path):
        bad = {k: [{"work_id": "aa/w1", "proba": [0.5, 0.5]}] for k in _vectors()}  # no labels/rank
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(_summary(), bad, docs_root=tmp_path)

    def test_forged_diagnostic_mcnemar_is_recomputed(self, tmp_path):
        # the diagnostic-only mcnemar p is published, so the auditor recomputes it too
        s = _summary()
        s["cells"]["lobo"]["stylo/A4"]["vs_A0"]["mcnemar_p_diagnostic"] = 0.123456
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(s, _vectors(), docs_root=tmp_path)

    def test_ghost_author_in_recall_is_rejected(self, tmp_path):
        # an EXTRA (never-recomputed) author in per_author_recall must be caught by set-equality
        s = _summary()
        s["cells"]["lobo"]["stylo/A0"]["point"]["per_author_recall"]["GHOST"] = 0.9999
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(s, _vectors(), docs_root=tmp_path)

    def test_non_dict_plan_field_surfaces_as_publisher_error(self):
        # a malformed embedded plan field (non-dict) fails closed as PublisherError, never an uncaught crash
        plan = copy.deepcopy(_PLAN)
        plan["a0_reference_shas"] = None
        s = _summary(run_plan=plan, run_id=rp.run_id(plan))
        with pytest.raises(pub.PublisherError):
            pub.verify_final_assembly(s, _vectors())

    def test_decorative_echo_fields_are_bound_to_the_run_plan(self):
        # every echoed field (attestation / universes digests / tolerances) must equal the bound plan
        forgeries = [
            lambda s: s["attestation"].__setitem__("git_commit", "z" * 40),
            lambda s: s["attestation"].__setitem__("run_kind", "smoke"),
            lambda s: s["universes"]["lobo"].__setitem__("dataset_digest", "0" * 64),
            lambda s: s["universes"]["ruaa"].__setitem__("fold_manifest_digest", "0" * 64),
            lambda s: s.__setitem__("continuous_tolerances",
                                    {"probability": {"atol": 1e9, "rtol": 0.0, "dtype": "float64"}}),
        ]
        for forge in forgeries:
            s = _summary(); forge(s)
            with pytest.raises(pub.PublisherError):
                pub.verify_final_assembly(s, _vectors())

    def test_forged_in_cell_per_work_diverging_from_archive_rejected(self, tmp_path):
        s = _summary()
        s["cells"]["lobo"]["stylo/A4"]["per_work"] = [
            {"work_id": "zz/fake", "true_label": 0, "pred_label": 0, "correct": True, "rank": 1,
             "proba": [0.6, 0.4]}]
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(s, _vectors(), docs_root=tmp_path)

    def test_forged_nonapplied_reason_rejected(self):
        s = _summary()
        # majority/A1 is not_applicable; a free-text reason must not diverge from the registry
        s["cells"]["lobo"]["majority/A1"]["reason"] = "FORGED REASON"
        with pytest.raises(pub.PublisherError):
            pub.verify_final_assembly(s, _vectors())

    def test_top_level_shape_and_labels_are_pinned(self):
        with pytest.raises(pub.PublisherError):                      # an injected extra top-level key
            pub.verify_final_assembly(_summary(injected="x"), _vectors())
        with pytest.raises(pub.PublisherError):                      # forged run_id_source label
            pub.verify_final_assembly(_summary(run_id_source="forged"), _vectors())
        with pytest.raises(pub.PublisherError):                      # forged result-audit stamp
            pub.verify_final_assembly(_summary(result_audit={"passed": True, "auditor": "x"}), _vectors())

    def test_forged_holm_raw_p_is_recomputed(self, tmp_path):
        s = _summary()
        s["holm"]["lobo"]["stylo/A4"]["raw_p"] = 0.000001          # diverges from the recomputed cluster_p
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(s, _vectors(), docs_root=tmp_path)


class TestPublicationSecurity:
    def test_swapped_committed_root_summary_is_detected(self, tmp_path):
        pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)     # confirmatory
        root = tmp_path / pub.SUMMARY_NAME
        body = load_strict(root)
        body["claim_status"] = "swapped"                                 # swap the committed root file
        dump_strict(body, root, trailing_newline=True)
        with pytest.raises(pub.PublisherError):                          # root != version summary
            pub.load_published_audit(tmp_path)

    def test_write_transient_guards_symlinked_namespace_before_mkdir(self, tmp_path):
        rid = "a" * 64
        runs = tmp_path / "exploratory" / "work_balanced" / "audit" / "runs"
        runs.mkdir(parents=True)
        (tmp_path / "evil").mkdir()
        (runs / rid).symlink_to(tmp_path / "evil")                       # symlinked run dir
        with pytest.raises(pub.PublisherError):
            pub.write_transient(rid, "x.json", {"k": 1}, docs_root=tmp_path)

    def test_write_transient_writes_inside_the_run_namespace(self, tmp_path):
        p = pub.write_transient("b" * 64, "note.json", {"k": 1}, docs_root=tmp_path)
        assert p.exists() and "runs" in p.parts and p.name == "note.json"


class TestPublishFailClosed:
    def test_tampered_per_work_file_rejected_on_load(self, tmp_path):
        published = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        ref = next(iter(published["summary"]["per_work_archive"].values()))
        (published["versioned_dir"] / ref["filename"]).write_text("tampered", encoding="utf-8")
        with pytest.raises(pub.PublisherError):
            pub.load_published_audit(tmp_path)

    def test_tampered_summary_self_hash_rejected(self, tmp_path):
        published = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        sfile = published["versioned_dir"] / pub.SUMMARY_IN_VERSION
        body = load_strict(sfile)
        body["decision"] = "relabel"                     # tamper without recomputing self_hash
        dump_strict(body, sfile, trailing_newline=True)
        with pytest.raises(pub.PublisherError):
            pub.load_published_audit(tmp_path)

    def test_conflicting_version_dir_is_fatal(self, tmp_path):
        published = pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
        # tamper a file inside the immutable version dir, then republish the SAME content
        ref = next(iter(published["summary"]["per_work_archive"].values()))
        (published["versioned_dir"] / ref["filename"]).write_text("mutated in place", encoding="utf-8")
        with pytest.raises(pub.PublisherError):
            pub.publish_audit(_summary(), _vectors(), docs_root=tmp_path)
