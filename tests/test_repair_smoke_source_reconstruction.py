"""Reconstruct and verify the source behind the exploratory repair smoke."""
from __future__ import annotations

import hashlib
import json
import pathlib

from stylo import jsonio

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/evidence/stack_class_coverage_repair_smoke_v1"
MANIFEST = EVIDENCE / "source_manifest.json"
EXPECTED_SCHEMA = "stylo.stack_class_coverage_repair.source_evidence.v1"
DRIVER = "scripts/evaluation/run_stack_class_coverage_repair_smoke.py"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _smoke_self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_preserved_repair_smoke_sources_reconstruct_and_compile(tmp_path):
    manifest = jsonio.load_strict(MANIFEST)
    assert manifest["schema"] == EXPECTED_SCHEMA
    assert manifest["repair_commit"] == "07a8df82c48b62d88dd65bb262dbb3c4931b0500"
    assert set(manifest["source_files"]) == {
        DRIVER,
        "src/stylo/eval/calibration.py",
        "src/stylo/eval/lobo.py",
        "src/stylo/models/equal_channel_ensemble.py",
        "src/stylo/models/stacked_clf.py",
    }

    for original, binding in manifest["source_files"].items():
        preserved = ROOT / binding["preserved_path"]
        assert not preserved.is_symlink() and preserved.is_file()
        assert _sha256(preserved) == binding["sha256"]
        reconstructed = tmp_path / original
        reconstructed.parent.mkdir(parents=True, exist_ok=True)
        reconstructed.write_bytes(preserved.read_bytes())
        assert _sha256(reconstructed) == binding["sha256"]
        compile(reconstructed.read_bytes(), original, "exec")

    assert not (ROOT / DRIVER).exists()


def test_optional_original_inputs_match_the_preserved_bindings():
    manifest = jsonio.load_strict(MANIFEST)
    bundle = ROOT / manifest["repair_bundle"]["optional_repository_path"]
    if bundle.is_file():
        assert _sha256(bundle) == manifest["repair_bundle"]["sha256"]

    artifact_binding = manifest["artifact"]
    artifact_path = ROOT / artifact_binding["optional_repository_path"]
    if not artifact_path.is_file():
        return
    assert _sha256(artifact_path) == artifact_binding["sha256"]
    artifact = jsonio.load_strict(artifact_path)
    assert artifact["status"] == artifact_binding["status"]
    assert artifact["self_hash"] == artifact_binding["self_hash"]
    assert _smoke_self_hash(artifact) == artifact_binding["self_hash"]

    code = artifact["input_binding"]["code"]
    assert code["git_commit"] == manifest["repair_commit"]
    assert code["driver_sha256"] == manifest["source_files"][DRIVER]["sha256"]
    assert code["source_hashes"] == {
        original: binding["sha256"]
        for original, binding in manifest["source_files"].items()
        if original != DRIVER
    }
    inputs = manifest["input_bindings"]
    assert artifact["input_binding"]["corpus"]["content_digest"] == inputs["corpus_content_digest"]
    assert artifact["input_binding"]["corpus"]["manifest_self_hash"] == inputs["corpus_manifest_self_hash"]
    assert artifact["input_binding"]["corpus"]["semantic_digest"] == inputs["corpus_semantic_digest"]
    assert artifact["input_binding"]["dataset"]["rows_digest"] == inputs["dataset_rows_digest"]
    assert artifact["input_binding"]["fold_manifest"]["file_sha256"] == inputs["fold_manifest_file_sha256"]
    assert artifact["input_binding"]["fold_manifest"]["self_hash"] == inputs["fold_manifest_self_hash"]
    assert (
        artifact["input_binding"]["fold_manifest"]["probability_class_order_digest"]
        == inputs["probability_class_order_digest"]
    )
    assert (
        artifact["input_binding"]["representation_cache"]["file_sha256"]
        == inputs["representation_cache_file_sha256"]
    )
    assert (
        artifact["input_binding"]["representation_cache"]["receipt_self_hash"]
        == inputs["representation_cache_receipt_self_hash"]
    )
    assert artifact["contract"]["selection"]["folds"] == manifest["selected_folds"]
    assert (
        artifact["contract"]["excluded_inferences"]
        == manifest["claim_scope"]["excluded_inferences"]
    )
