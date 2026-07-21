"""Synthetic tests for the path-guarded transient store and verified publisher (§4.4/§8)."""
from __future__ import annotations

import pathlib

import pytest

from stylo.jsonio import dump_strict, load_strict
from stylo.eval.paired_audit import publisher as pub

RUN_ID = "a" * 32


def _summary(**over):
    s = {"claim_status": "exploratory_internal", "run_id": RUN_ID,
         "endpoint": "stylo_lobo_a4_minus_a0_accuracy", "decision": "inconclusive"}
    s.update(over)
    return s


def _vectors():
    return {
        "lobo/stylo/A4": [{"work_id": "auth/w1", "proba": [0.1, 0.9]},
                          {"work_id": "auth/w2", "proba": [0.7, 0.3]}],
        "lobo/stylo/A0": [{"work_id": "auth/w1", "proba": [0.2, 0.8]}],
        "ruaa/char_cos/A4": [{"work_id": "b/w9", "proba": [0.5, 0.5]}],
    }


class TestPathGuard:
    def test_rejects_headline_basename(self, tmp_path):
        with pytest.raises(pub.PublisherError):
            pub.assert_writable_audit_path(tmp_path / "lobo_books.txt",
                                           docs_root=tmp_path, allow_published=True)
        with pytest.raises(pub.PublisherError):
            pub.assert_writable_audit_path(tmp_path / "README.md",
                                           docs_root=tmp_path, allow_published=True)

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
