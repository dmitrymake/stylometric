"""Synthetic tests for the path-guarded transient store and verified publisher (§4.4/§8)."""
from __future__ import annotations

import pathlib

import pytest

from stylo.jsonio import dump_strict, load_strict
from stylo.eval.paired_audit import applicability as ap
from stylo.eval.paired_audit import publisher as pub
from stylo.eval.paired_audit import run_plan as rp
from stylo.eval.paired_audit.headline import HEADLINE_ENDPOINT
from stylo.eval.paired_audit.semantic_parity import LEGACY_ANCHOR


def _run_plan():
    return rp.build_run_plan(
        run_kind="smoke", git_commit="a" * 40, git_dirty=True,
        execution_source_sha256="1" * 64, env_lock_sha256="2" * 64, config_id="3" * 64,
        runtime_fingerprint={"python": "3.11", "libc": "glibc/2.39", "numpy": "2", "scipy": "1",
                             "sklearn": "1"},
        blas_thread_fingerprint={"threadpools": [], "thread_env": {}},
        applicability_matrix_digest=ap.applicability_matrix_digest(),
        a0_reference_shas={"lobo_books_txt": "4" * 64, "ruaa_reference_submission": "5" * 64},
        evaluator_identity={"name": "smoke_dummy", "import_module": "m", "import_qualname": "q",
                            "source_digest": "a" * 64, "estimator_config_digest": "b" * 64,
                            "mechanism_passport_digest": "c" * 64},
        tolerances={}, corpus_chain={"legacy_anchor": LEGACY_ANCHOR, "semantic_parity_digest": "6" * 64},
        golden_fixture_inventory_sha="7" * 64,
        lobo=dict(dataset_digest="a" * 64, fold_manifest_digest="b" * 64,
                  probability_class_order=["x", "z"], metric_label_order=["x"], run_contract_digest="c" * 64),
        ruaa=dict(dataset_digest="d" * 64, fold_manifest_digest="e" * 64,
                  probability_class_order=["x"], metric_label_order=["x"], run_contract_digest="f" * 64,
                  selection_digest="9" * 64))


_PLAN = _run_plan()
RUN_ID = rp.run_id(_PLAN)


def _universes():
    return {ds: {"dataset_digest": "a" * 64, "fold_manifest_digest": "b" * 64,
                 "probability_class_order": ["x", "z"], "metric_label_order": ["x"],
                 "n_train_works": 2, "n_tested_works": 2, "n_train_authors": 2, "n_tested_authors": 1}
            for ds in ("lobo", "ruaa")}


def _attestation():
    return {"git_commit": "a" * 40, "run_kind": "smoke", "audit_version": rp.AUDIT_VERSION,
            "execution_source_sha256": "1" * 64, "env_lock_sha256": "2" * 64, "config_id": "3" * 64,
            "golden_fixture_inventory_sha": "7" * 64}


def _evidence(m, c):
    # exactly the digests the single-source contract requires for this applied cell (axes + passports)
    return {key: "e" * 64 for key in ap.required_evidence_digests(m, c)}


def _cell_record(m, c):
    reg = ap.cell_status(m, c)
    if reg["status"] == "applied":
        r = {"status": "applied", "requested_axes": reg["requested_axes"],
             "effective_axes": reg["effective_axes"],
             "point": {"accuracy": 0.9, "macro_f1": 0.8, "top2": 0.95, "per_author_recall": {}},
             "per_work": [{"work_id": "a/w", "true_label": 0, "pred_label": 0, "correct": True,
                           "rank": 1, "proba": [0.5, 0.5]}],
             "abs_accuracy_authorclustered_ci": [0.8, 0.95],
             "evidence": _evidence(m, c), "claim_status": "exploratory_internal"}
        if c != "A0":
            r["vs_A0"] = {"dacc": 0.02, "dacc_authorclustered_ci": [-0.01, 0.05], "cluster_p": 0.01,
                          "holm_p": 0.05, "mcnemar_p_diagnostic": 0.2, "significant": False}
        return r
    r = {"status": reg["status"], "requested_axes": reg["requested_axes"],
         "effective_axes": reg["effective_axes"], "claim_status": "exploratory_internal"}
    if reg["equivalent_to"] is not None:
        r["equivalent_to"] = reg["equivalent_to"]
    return r


def _grid():
    return {f"{m}/{c}": _cell_record(m, c) for m in ap.MODELS for c in ap.CELLS}


def _holm():
    # consistent with each cell's vs_A0 verdict (Holm<->cell consistency is checked by the publisher)
    return {f"{m}/{c}": {"raw_p": 0.001, "holm_p": 0.05, "significant": False}
            for (m, c) in ap.holm_family()}


def _summary(**over):
    s = {"run_id": RUN_ID, "claim_status": "exploratory_internal",
         "cells": {"lobo": _grid(), "ruaa": _grid()},
         "holm": {"lobo": _holm(), "ruaa": _holm()},
         "headline": {"endpoint": HEADLINE_ENDPOINT, "decision": "relabel",
                      "diff_ci": {"point": 0.02, "lo": -0.01, "hi": 0.05}, "margin": 0.02},
         "result_audit": {"passed": True, "auditor": "independent_recompute_v1"},
         "run_plan": _PLAN, "universes": _universes(),
         "continuous_tolerances": {"probability": {"atol": 1e-9, "rtol": 0, "dtype": "float64"}},
         "attestation": _attestation()}
    s.update(over)
    return s


def _vectors():
    return {f"{ds}/{m}/{c}": [{"work_id": "a/w", "proba": [0.5, 0.5]}]
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
