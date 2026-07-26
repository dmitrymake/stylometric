import dataclasses
import hashlib
import json

import pytest

from stylo.benchmarks import (
    ScoringFormatError,
    load_submission,
    load_truth,
    score_files,
    score_submission,
    validate_manifest,
)


DOC1 = "doc_0000000000000001"
DOC2 = "doc_0000000000000002"
DIGESTS = {
    DOC1: "a" * 64,
    DOC2: "b" * 64,
}


def _manifest():
    documents = []
    for doc_id, task in [
        (DOC1, "spoof"),
        (DOC2, "mixed_authorship"),
    ]:
        documents.append(
            {
                "doc_id": doc_id,
                "source": {
                    "source_id": "sealed",
                    "provenance": f"sealed:{doc_id}",
                    "revision": "v1",
                    "sha256": DIGESTS[doc_id],
                },
                "split": "blind",
                "task_types": [task],
                "text_path": f"texts/{doc_id}.txt",
                "spans": [],
            }
        )
    return validate_manifest(
        {
            "schema_version": "1.0",
            "dataset": {
                "name": "blind-synthetic",
                "version": "0.1.0",
                "license": "CC0-1.0",
                "offset_unit": "token",
                "tokenizer": "stylo_unicode_word_punct_v1",
            },
            "task_types": ["spoof", "mixed_authorship"],
            "documents": documents,
        }
    )


def _write_files(tmp_path, manifest_digest: str):
    truth = {
        "schema_version": "1.0",
        "dataset_name": "blind-synthetic",
        "dataset_version": "0.1.0",
        "manifest_sha256": manifest_digest,
        "records": [
            {
                "doc_id": DOC1,
                "author_label": "author_a",
                "author_evidence": "archive:author-a",
                "document_label": "forged",
                "document_evidence": "archive:forgery-record",
                "spans": [],
            },
            {
                "doc_id": DOC2,
                "spans": [
                    {"start": 0, "end": 5, "label": "author_a", "evidence": "archive:a"},
                    {"start": 5, "end": 10, "label": "author_b", "evidence": "archive:b"},
                ],
            },
        ],
    }
    submission = {
        "schema_version": "1.0",
        "dataset_name": "blind-synthetic",
        "dataset_version": "0.1.0",
        "predictions": [
            {
                "doc_id": DOC1,
                "author_label": "author_a",
                "document_label": "forged",
                "spans": [],
            },
            {
                "doc_id": DOC2,
                "spans": [
                    {"start": 0, "end": 5, "label": "author_a"},
                    {"start": 5, "end": 10, "label": "author_b"},
                ],
            },
        ],
    }
    truth_path = tmp_path / "truth.json"
    submission_path = tmp_path / "submission.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    escrow_sha256 = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    return (
        load_truth(truth_path, expected_sha256=escrow_sha256),
        load_submission(submission_path),
    )


def test_blind_truth_is_hash_bound_and_scores_classification_and_spans(tmp_path):
    manifest_digest = hashlib.sha256(b"public manifest bytes").hexdigest()
    truth, submission = _write_files(tmp_path, manifest_digest)

    score = score_submission(
        _manifest(),
        truth,
        submission,
        manifest_sha256=manifest_digest,
        bootstrap_iters=20,
        seed=3,
        segmentation_bootstrap_unit="document",
        synthetic_integration_only=True,
    )

    assert score.authorship.accuracy == 1.0
    assert score.authorship.coverage == 1.0
    assert score.document_classification.accuracy == 1.0
    assert score.segmentation.aggregate.token_accuracy == 1.0
    assert score.segmentation.aggregate.boundaries.f1 == 1.0


def test_truth_offsets_can_be_bound_to_verified_document_lengths(tmp_path):
    manifest_digest = hashlib.sha256(b"public manifest bytes").hexdigest()
    truth, submission = _write_files(tmp_path, manifest_digest)
    bad_truth = dataclasses.replace(
        truth,
        records=(
            truth.records[0],
            dataclasses.replace(
                truth.records[1],
                spans=(dataclasses.replace(truth.records[1].spans[0], end=11),),
            ),
        ),
    )
    with pytest.raises(ScoringFormatError, match="beyond 10"):
        score_submission(
            _manifest(),
            bad_truth,
            submission,
            manifest_sha256=manifest_digest,
            document_lengths={DOC1: 10, DOC2: 10},
            bootstrap_iters=5,
            synthetic_integration_only=True,
        )


def test_scoring_rejects_wrong_manifest_hash_and_missing_blind_id(tmp_path):
    truth, submission = _write_files(tmp_path, "b" * 64)
    with pytest.raises(ScoringFormatError, match="manifest_sha256"):
        score_submission(
            _manifest(), truth, submission, manifest_sha256="c" * 64,
            bootstrap_iters=5, synthetic_integration_only=True,
        )

    incomplete = dataclasses.replace(
        submission, predictions=submission.predictions[:1]
    )
    with pytest.raises(ScoringFormatError, match="prediction ids"):
        score_submission(
            _manifest(), truth, incomplete, manifest_sha256="b" * 64,
            bootstrap_iters=5, synthetic_integration_only=True,
        )


def test_truth_requires_evidence_and_submission_forbids_it(tmp_path):
    digest = "d" * 64
    truth, _submission = _write_files(tmp_path, digest)
    raw = json.loads((tmp_path / "truth.json").read_text())
    del raw["records"][0]["author_evidence"]
    (tmp_path / "truth.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ScoringFormatError, match="author_evidence"):
        load_truth(
            tmp_path / "truth.json",
            expected_sha256=hashlib.sha256(
                (tmp_path / "truth.json").read_bytes()
            ).hexdigest(),
        )

    raw = json.loads((tmp_path / "truth.json").read_text())
    raw["records"][0]["author_evidence"] = "archive:author-a"
    del raw["records"][0]["document_evidence"]
    (tmp_path / "truth.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ScoringFormatError, match="document_evidence"):
        load_truth(
            tmp_path / "truth.json",
            expected_sha256=hashlib.sha256(
                (tmp_path / "truth.json").read_bytes()
            ).hexdigest(),
        )

    prediction_raw = json.loads((tmp_path / "submission.json").read_text())
    prediction_raw["predictions"][1]["spans"][0]["evidence"] = "leaked"
    (tmp_path / "submission.json").write_text(json.dumps(prediction_raw), encoding="utf-8")
    with pytest.raises(ScoringFormatError, match="unknown field"):
        load_submission(tmp_path / "submission.json")


def test_segmentation_requires_explicit_unit_and_rejects_character_v1(tmp_path):
    manifest_digest = hashlib.sha256(b"public manifest bytes").hexdigest()
    truth, submission = _write_files(tmp_path, manifest_digest)

    with pytest.raises(ScoringFormatError, match="segmentation_bootstrap_unit"):
        score_submission(
            _manifest(),
            truth,
            submission,
            manifest_sha256=manifest_digest,
            bootstrap_iters=5,
            synthetic_integration_only=True,
        )
    with pytest.raises(ScoringFormatError, match="work identity"):
        score_submission(
            _manifest(),
            truth,
            submission,
            manifest_sha256=manifest_digest,
            bootstrap_iters=5,
            segmentation_bootstrap_unit="work",
            synthetic_integration_only=True,
        )

    character_manifest = dataclasses.replace(
        _manifest(),
        dataset=dataclasses.replace(
            _manifest().dataset, offset_unit="character", tokenizer=None
        ),
    )
    with pytest.raises(ScoringFormatError, match="character-offset"):
        score_submission(
            character_manifest,
            truth,
            submission,
            manifest_sha256=manifest_digest,
            bootstrap_iters=5,
            segmentation_bootstrap_unit="document",
            synthetic_integration_only=True,
        )


def test_scoring_uses_task_registered_endpoints_and_exact_manifest_bytes(tmp_path):
    manifest = _manifest()
    manifest_digest = hashlib.sha256(b"public manifest bytes").hexdigest()
    truth, submission = _write_files(tmp_path, manifest_digest)
    spoof_with_spans = dataclasses.replace(
        truth,
        records=(
            dataclasses.replace(
                truth.records[0],
                spans=(truth.records[1].spans[0],),
            ),
            truth.records[1],
        ),
    )
    with pytest.raises(ScoringFormatError, match="not registered"):
        score_submission(
            manifest,
            spoof_with_spans,
            submission,
            manifest_sha256=manifest_digest,
            bootstrap_iters=5,
            segmentation_bootstrap_unit="document",
            synthetic_integration_only=True,
        )

    def without_none(value):
        if isinstance(value, dict):
            return {
                key: without_none(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, (list, tuple)):
            return [without_none(item) for item in value]
        return value

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(without_none(dataclasses.asdict(manifest))),
        encoding="utf-8",
    )
    different_object = dataclasses.replace(
        manifest,
        dataset=dataclasses.replace(manifest.dataset, description="different semantics"),
    )
    with pytest.raises(ScoringFormatError, match="does not equal the exact bytes"):
        score_files(
            different_object,
            manifest_path,
            tmp_path / "truth.json",
            tmp_path / "submission.json",
            synthetic_integration_only=True,
        )
