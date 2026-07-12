import hashlib

import pytest

from stylo.benchmarks import (
    ArtifactValidationError,
    tokenize_with_offsets,
    validate_manifest,
    verify_manifest_artifacts,
)


def _manifest_for(text: str, *, text_path: str = "texts/doc.txt", spans=None):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return validate_manifest(
        {
            "schema_version": "1.0",
            "dataset": {
                "name": "synthetic",
                "version": "0.1.0",
                "license": "CC0-1.0",
                "language": "ru",
                "offset_unit": "token",
                "tokenizer": "stylo_unicode_word_punct_v1",
            },
            "task_types": ["mixed_authorship"],
            "documents": [
                {
                    "doc_id": "doc_0000000000000001",
                    "source": {
                        "source_id": "synthetic_source",
                        "provenance": "generated:test",
                        "revision": "v1",
                        "sha256": digest,
                    },
                    "split": "train",
                    "task_types": ["mixed_authorship"],
                    "text_path": text_path,
                    "spans": spans
                    or [
                        {
                            "start": 0,
                            "end": 2,
                            "label": "a",
                            "ground_truth_known": True,
                            "evidence": "synthetic:test:a",
                        },
                        {
                            "start": 2,
                            "end": 4,
                            "label": "b",
                            "ground_truth_known": True,
                            "evidence": "synthetic:test:b",
                        },
                    ],
                }
            ],
        }
    )


def test_frozen_tokenizer_and_artifact_report(tmp_path):
    text = "Альфа, бета!"
    path = tmp_path / "texts" / "doc.txt"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")

    tokens = tokenize_with_offsets(text)
    report = verify_manifest_artifacts(_manifest_for(text), tmp_path)

    assert [token.text for token in tokens] == ["Альфа", ",", "бета", "!"]
    assert report.documents[0].n_tokens == 4
    assert report.documents[0].n_offset_units == 4


def test_artifact_verifier_rejects_hash_path_offset_and_partition_errors(tmp_path):
    text = "Альфа, бета!"
    path = tmp_path / "texts" / "doc.txt"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")

    bad_hash = _manifest_for(text)
    object.__setattr__(bad_hash.documents[0].source, "sha256", "0" * 64)
    with pytest.raises(ArtifactValidationError, match="sha256 mismatch"):
        verify_manifest_artifacts(bad_hash, tmp_path)

    unsafe = _manifest_for(text, text_path="../escape.txt")
    with pytest.raises(ArtifactValidationError, match="safe relative"):
        verify_manifest_artifacts(unsafe, tmp_path)

    out_of_range = _manifest_for(
        text,
        spans=[
            {
                "start": 0,
                "end": 5,
                "label": "a",
                "ground_truth_known": True,
                "evidence": "synthetic:test",
            }
        ],
    )
    with pytest.raises(ArtifactValidationError, match="beyond 4 token units"):
        verify_manifest_artifacts(out_of_range, tmp_path)

    gap = _manifest_for(
        text,
        spans=[
            {
                "start": 0,
                "end": 1,
                "label": "a",
                "ground_truth_known": True,
                "evidence": "synthetic:test:a",
            },
            {
                "start": 2,
                "end": 4,
                "label": "b",
                "ground_truth_known": True,
                "evidence": "synthetic:test:b",
            },
        ],
    )
    with pytest.raises(ArtifactValidationError, match="must be contiguous"):
        verify_manifest_artifacts(gap, tmp_path)
