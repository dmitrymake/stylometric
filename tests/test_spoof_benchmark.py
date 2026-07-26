import copy
import json

import pytest

from stylo.benchmarks import (
    MANIFEST_SCHEMA,
    ManifestValidationError,
    load_manifest,
    loads_manifest,
    validate_manifest,
)


SHA256 = "a" * 64


def _source(revision="2026-07-10", sha256=SHA256):
    return {
        "source_id": "archive_primary",
        "provenance": "https://example.test/archive/item",
        "revision": revision,
        "sha256": sha256,
    }


def _manifest():
    return {
        "schema_version": "1.0",
        "dataset": {
            "name": "SPOOF-RU / IDIOSHIFT-RU",
            "version": "0.1.0",
            "license": "CC-BY-4.0",
            "offset_unit": "token",
            "tokenizer": "stylo_unicode_word_punct_v1",
            "language": "ru",
        },
        "task_types": ["spoof", "idio_shift", "mixed_authorship"],
        "documents": [
            {
                "doc_id": "doc_0000000000000001",
                "source": _source(),
                "split": "train",
                "task_types": ["spoof", "mixed_authorship"],
                "text_path": "texts/doc_0000000000000001.txt",
                "author_label": "author_a",
                "document_label": "spoof",
                "work": "work_a",
                "edition": "critical-2024",
                "period": "1910s",
                "genre": "prose",
                "topic": "travel",
                "register": "literary",
                "spans": [
                    {
                        "start": 0,
                        "end": 100,
                        "label": "genuine",
                        "ground_truth_known": True,
                        "evidence": "archive:item:1",
                    },
                    {
                        "start": 100,
                        "end": 180,
                        "label": "spoof",
                        "ground_truth_known": True,
                        "evidence": "archive:item:2",
                    },
                ],
            },
            {
                "doc_id": "doc_0000000000000002",
                "source": _source("git:abc123", "b" * 64),
                "split": "validation",
                "task_types": ["idio_shift"],
                "spans": [
                    {
                        "start": 0,
                        "end": 50,
                        "label": "early",
                        "ground_truth_known": True,
                        "evidence": "archive:item:3",
                    }
                ],
            },
            {
                "doc_id": "doc_0000000000000003",
                "source": _source("archive:v2", "c" * 64),
                "split": "test",
                "task_types": ["mixed_authorship"],
                "spans": [
                    {
                        "start": 0,
                        "end": 70,
                        "label": "author_a",
                        "ground_truth_known": True,
                        "evidence": "archive:item:4",
                    },
                    {
                        "start": 90,
                        "end": 140,
                        "label": "author_b",
                        "ground_truth_known": True,
                        "evidence": "archive:item:5",
                    },
                ],
            },
            {
                "doc_id": "doc_0000000000000004",
                "source": _source("sealed:v1", "d" * 64),
                "split": "blind",
                "task_types": ["spoof"],
                "spans": [],
            },
        ],
    }


def test_valid_manifest_returns_immutable_models(tmp_path):
    raw = _manifest()
    parsed = validate_manifest(raw)

    assert parsed.dataset.name == "SPOOF-RU / IDIOSHIFT-RU"
    assert parsed.task_types == ("spoof", "idio_shift", "mixed_authorship")
    assert parsed.documents[0].source.sha256 == SHA256
    assert parsed.documents[0].spans[1].start == 100
    assert parsed.documents[-1].author_label is None
    assert parsed.documents[-1].document_label is None

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    assert load_manifest(path) == parsed


def test_public_json_schema_is_strict_and_declares_all_tasks():
    assert MANIFEST_SCHEMA["additionalProperties"] is False
    task_items = MANIFEST_SCHEMA["properties"]["task_types"]["items"]
    assert set(task_items["enum"]) == {"spoof", "idio_shift", "mixed_authorship"}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("documents", 0, "extra"), "leak"),
        (("documents", 0, "source", "mirror"), "https://mirror.test"),
        (("dataset", "citation"), "unreviewed"),
    ],
)
def test_unknown_fields_are_rejected(path, value):
    raw = _manifest()
    cursor = raw
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value

    with pytest.raises(ManifestValidationError, match="unknown field"):
        validate_manifest(raw)


def test_document_ids_must_be_opaque_and_unique():
    semantic = _manifest()
    semantic["documents"][0]["doc_id"] = "tolstoy_war_and_peace"
    with pytest.raises(ManifestValidationError, match="must be opaque"):
        validate_manifest(semantic)

    duplicate = _manifest()
    duplicate["documents"][1]["doc_id"] = duplicate["documents"][0]["doc_id"]
    with pytest.raises(ManifestValidationError, match="duplicate"):
        validate_manifest(duplicate)


@pytest.mark.parametrize("bad_hash", ["a" * 63, "A" * 64, "g" * 64])
def test_sha256_must_be_canonical_lowercase_hex(bad_hash):
    raw = _manifest()
    raw["documents"][0]["source"]["sha256"] = bad_hash
    with pytest.raises(ManifestValidationError, match="64 lowercase hexadecimal"):
        validate_manifest(raw)


@pytest.mark.parametrize("missing", ["source_id", "provenance", "revision", "sha256"])
def test_every_document_requires_complete_source_provenance(missing):
    raw = _manifest()
    del raw["documents"][0]["source"][missing]
    with pytest.raises(ManifestValidationError, match=f"source.{missing}: required"):
        validate_manifest(raw)


def test_token_offsets_require_frozen_supported_tokenizer():
    missing = _manifest()
    del missing["dataset"]["tokenizer"]
    with pytest.raises(ManifestValidationError, match="required when offset_unit"):
        validate_manifest(missing)

    unsupported = _manifest()
    unsupported["dataset"]["tokenizer"] = "mutable_tokenizer_latest"
    with pytest.raises(ManifestValidationError, match="unsupported tokenizer"):
        validate_manifest(unsupported)

    character = _manifest()
    character["dataset"]["offset_unit"] = "character"
    with pytest.raises(ManifestValidationError, match="only valid"):
        validate_manifest(character)


def test_known_ground_truth_span_requires_external_evidence_locator():
    raw = _manifest()
    del raw["documents"][0]["spans"][0]["evidence"]
    with pytest.raises(ManifestValidationError, match="evidence: required"):
        validate_manifest(raw)

    unknown = _manifest()
    unknown["documents"][0]["spans"][0]["ground_truth_known"] = False
    with pytest.raises(ManifestValidationError, match="evidence: must be omitted"):
        validate_manifest(unknown)


@pytest.mark.parametrize(
    ("spans", "message"),
    [
        (
            [{"start": -1, "end": 5, "label": "x", "ground_truth_known": True}],
            "must be non-negative",
        ),
        (
            [{"start": 5, "end": 5, "label": "x", "ground_truth_known": True}],
            "greater than start",
        ),
        (
            [
                {"start": 0, "end": 10, "label": "x", "ground_truth_known": True},
                {"start": 9, "end": 12, "label": "y", "ground_truth_known": True},
            ],
            "must not overlap",
        ),
        (
            [
                {"start": 20, "end": 30, "label": "x", "ground_truth_known": True},
                {"start": 0, "end": 10, "label": "y", "ground_truth_known": True},
            ],
            "ordered by start",
        ),
    ],
)
def test_span_ranges_are_valid_ordered_and_non_overlapping(spans, message):
    raw = _manifest()
    raw["documents"][0]["spans"] = spans
    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest(raw)


def test_span_offsets_reject_booleans_as_integers():
    raw = _manifest()
    raw["documents"][0]["spans"][0]["start"] = True
    with pytest.raises(ManifestValidationError, match="expected an integer"):
        validate_manifest(raw)


def test_blind_split_forbids_author_and_span_labels():
    labelled_author = _manifest()
    labelled_author["documents"][-1]["author_label"] = "author_a"
    with pytest.raises(ManifestValidationError, match="author_label: forbidden"):
        validate_manifest(labelled_author)

    labelled_document = _manifest()
    labelled_document["documents"][-1]["document_label"] = "spoof"
    with pytest.raises(ManifestValidationError, match="document_label: forbidden"):
        validate_manifest(labelled_document)

    labelled_span = _manifest()
    labelled_span["documents"][-1]["spans"] = copy.deepcopy(
        labelled_span["documents"][0]["spans"]
    )
    with pytest.raises(ManifestValidationError, match="must not expose span labels"):
        validate_manifest(labelled_span)


def test_document_tasks_must_be_supported_and_declared_by_manifest():
    unsupported = _manifest()
    unsupported["task_types"].append("authorship")
    with pytest.raises(ManifestValidationError, match="unsupported task type"):
        validate_manifest(unsupported)

    undeclared = _manifest()
    undeclared["task_types"] = ["spoof", "mixed_authorship"]
    with pytest.raises(ManifestValidationError, match="not declared by manifest"):
        validate_manifest(undeclared)


def test_strict_loader_rejects_duplicate_json_keys_and_non_finite_numbers():
    encoded = json.dumps(_manifest())
    duplicate = encoded.replace(
        '{"schema_version": "1.0"',
        '{"schema_version": "1.0", "schema_version": "1.0"',
        1,
    )
    with pytest.raises(ManifestValidationError, match="duplicate object key"):
        loads_manifest(duplicate)

    non_finite = _manifest()
    non_finite["documents"][0]["spans"][0]["start"] = float("nan")
    with pytest.raises(ManifestValidationError, match="non-finite number"):
        loads_manifest(json.dumps(non_finite))


def test_unknown_span_without_evidence_matches_public_schema_contract():
    raw = _manifest()
    span = raw["documents"][0]["spans"][0]
    span["ground_truth_known"] = False
    span.pop("evidence")
    parsed = validate_manifest(raw)
    assert parsed.documents[0].spans[0].ground_truth_known is False
    assert parsed.documents[0].spans[0].evidence is None

    span["evidence"] = "must-not-be-present"
    with pytest.raises(ManifestValidationError, match="must be omitted"):
        validate_manifest(raw)


def test_task_endpoint_matrix_rejects_unregistered_or_empty_endpoints():
    spoof_only_spans = _manifest()
    spoof_only_spans["documents"][0]["task_types"] = ["spoof"]
    with pytest.raises(ManifestValidationError, match="not registered"):
        validate_manifest(spoof_only_spans)

    empty_mixed = _manifest()
    empty_mixed["documents"][2]["spans"] = []
    with pytest.raises(ManifestValidationError, match="requires truth fields"):
        validate_manifest(empty_mixed)

    unused = _manifest()
    unused["documents"][1]["task_types"] = ["mixed_authorship"]
    with pytest.raises(ManifestValidationError, match="unused=.*idio_shift"):
        validate_manifest(unused)
