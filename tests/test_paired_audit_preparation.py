"""Fail-closed tests for the real-input preparation command (never runs real data/cells)."""
from __future__ import annotations

import pathlib
import runpy

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = runpy.run_path(str(ROOT / "scripts" / "evaluation" / "prepare_paired_audit_inputs.py"))


def test_candidate_reuse_revalidates_every_payload(tmp_path):
    write = SCRIPT["_write_freeze_candidate"]
    body = {"schema": "test", "status": "unapproved"}
    lobo = {"self_hash": "a" * 64, "works": ["lobo"]}
    ruaa = {"self_hash": "b" * 64, "works": ["ruaa"]}
    destination = write(tmp_path / "candidates", body, lobo, ruaa)
    assert write(tmp_path / "candidates", body, lobo, ruaa) == destination

    (destination / "lobo_fold_manifest_v1.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        write(tmp_path / "candidates", body, lobo, ruaa)


def test_ruaa_drift_exception_is_exact_not_blanket():
    check = SCRIPT["_assert_only_known_ruaa_protocol_drift"]
    known = SCRIPT["_KNOWN_RUAA_PROTOCOL_DRIFT"]
    check(141, [known])
    with pytest.raises(RuntimeError):
        check(141, [{**known, "name": "manifest.json"}])
    with pytest.raises(RuntimeError):
        check(140, [known])
    with pytest.raises(RuntimeError):
        check(141, [known, {"name": "texts/x.txt", "expected": "a" * 64, "actual": "b" * 64}])
