"""Blind truth/submission formats and benchmark scoring.

The public manifest never contains labels for ``split=blind``.  A separate
truth file is bound to the exact public manifest bytes by SHA-256 and is held by
an independent custodian until scoring.  Submissions contain only opaque
document ids and predictions.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..eval.segmentation import (
    CorpusSegmentationReport,
    LabeledSpan,
    SegmentationDocument,
    evaluate_corpus,
)
from .artifacts import file_sha256, verify_manifest_artifacts
from .schema import BenchmarkManifest, DOC_ID_PATTERN, MANIFEST_SCHEMA_VERSION, SHA256_PATTERN


@dataclasses.dataclass(frozen=True)
class ScoringSpan:
    start: int
    end: int
    label: str
    evidence: str | None = None


@dataclasses.dataclass(frozen=True)
class TruthRecord:
    doc_id: str
    author_label: str | None
    author_evidence: str | None
    document_label: str | None
    document_evidence: str | None
    spans: tuple[ScoringSpan, ...]


@dataclasses.dataclass(frozen=True)
class PredictionRecord:
    doc_id: str
    author_label: str | None
    document_label: str | None
    spans: tuple[ScoringSpan, ...]


@dataclasses.dataclass(frozen=True)
class BenchmarkTruth:
    schema_version: str
    dataset_name: str
    dataset_version: str
    manifest_sha256: str
    records: tuple[TruthRecord, ...]

    def by_id(self) -> dict[str, TruthRecord]:
        return {record.doc_id: record for record in self.records}


@dataclasses.dataclass(frozen=True)
class BenchmarkSubmission:
    schema_version: str
    dataset_name: str
    dataset_version: str
    predictions: tuple[PredictionRecord, ...]

    def by_id(self) -> dict[str, PredictionRecord]:
        return {record.doc_id: record for record in self.predictions}


@dataclasses.dataclass(frozen=True)
class ClassificationScore:
    n_documents: int
    n_predicted: int
    coverage: float
    accuracy: float
    macro_f1: float


@dataclasses.dataclass(frozen=True)
class BenchmarkScore:
    dataset_name: str
    dataset_version: str
    authorship: ClassificationScore | None
    document_classification: ClassificationScore | None
    segmentation: CorpusSegmentationReport | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class ScoringFormatError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("invalid scoring file:\n" + "\n".join(f"- {e}" for e in errors))


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate object key {key!r}")
        out[key] = value
    return out


def _reject_constant(value: str):
    raise ValueError(f"non-finite number {value!r}")


def _load_json(path: str | pathlib.Path) -> object:
    p = pathlib.Path(path)
    try:
        return json.loads(
            p.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ScoringFormatError([f"$: cannot load strict JSON from {p}: {exc}"]) from exc


def _mapping(value: object, path: str, errors: list[str]) -> Mapping[str, object] | None:
    if type(value) is not dict:
        errors.append(f"{path}: expected object")
        return None
    return value


def _array(value: object, path: str, errors: list[str]) -> list[object] | None:
    if type(value) is not list:
        errors.append(f"{path}: expected array")
        return None
    return value


def _check_keys(
    raw: Mapping[str, object], required: set[str], allowed: set[str], path: str, errors: list[str]
):
    for key in sorted(required - set(raw)):
        errors.append(f"{path}.{key}: required field is missing")
    for key in sorted(set(raw) - allowed):
        errors.append(f"{path}.{key}: unknown field")


def _string(raw: Mapping[str, object], key: str, path: str, errors: list[str], *, required=True):
    if key not in raw:
        return None
    value = raw[key]
    if type(value) is not str or not value:
        errors.append(f"{path}.{key}: expected non-empty string")
        return None
    return value


def _spans(value: object, path: str, errors: list[str], *, truth: bool) -> tuple[ScoringSpan, ...]:
    raw_spans = _array(value, path, errors)
    if raw_spans is None:
        return ()
    result: list[ScoringSpan] = []
    previous_end = 0
    for i, value in enumerate(raw_spans):
        item_path = f"{path}[{i}]"
        raw = _mapping(value, item_path, errors)
        if raw is None:
            continue
        required = {"start", "end", "label", "evidence"} if truth else {"start", "end", "label"}
        allowed = required
        _check_keys(raw, required, allowed, item_path, errors)
        start, end = raw.get("start"), raw.get("end")
        label = _string(raw, "label", item_path, errors)
        evidence = _string(raw, "evidence", item_path, errors, required=truth)
        if type(start) is not int:
            errors.append(f"{item_path}.start: expected integer")
        if type(end) is not int:
            errors.append(f"{item_path}.end: expected integer")
        if type(start) is int and type(end) is int:
            if start < 0 or end <= start:
                errors.append(f"{item_path}: invalid half-open range [{start}, {end})")
            if i and start < previous_end:
                errors.append(f"{item_path}: spans must be sorted and non-overlapping")
            previous_end = max(previous_end, end)
        if type(start) is int and type(end) is int and label is not None:
            result.append(ScoringSpan(start, end, label, evidence))
    return tuple(result)


def _header(raw: Mapping[str, object], errors: list[str]) -> tuple[str | None, str | None, str | None]:
    schema = _string(raw, "schema_version", "$", errors)
    name = _string(raw, "dataset_name", "$", errors)
    version = _string(raw, "dataset_version", "$", errors)
    if schema is not None and schema != MANIFEST_SCHEMA_VERSION:
        errors.append(f"$.schema_version: expected {MANIFEST_SCHEMA_VERSION!r}")
    return schema, name, version


def _parse_records(
    values: object, path: str, errors: list[str], *, truth: bool
) -> tuple[TruthRecord | PredictionRecord, ...]:
    rows = _array(values, path, errors)
    if rows is None:
        return ()
    result = []
    seen = set()
    for i, value in enumerate(rows):
        item_path = f"{path}[{i}]"
        raw = _mapping(value, item_path, errors)
        if raw is None:
            continue
        if truth:
            allowed = {
                "doc_id",
                "author_label",
                "author_evidence",
                "document_label",
                "document_evidence",
                "spans",
            }
        else:
            allowed = {"doc_id", "author_label", "document_label", "spans"}
        _check_keys(raw, {"doc_id", "spans"}, allowed, item_path, errors)
        doc_id = _string(raw, "doc_id", item_path, errors)
        author = _string(raw, "author_label", item_path, errors, required=False)
        author_evidence = (
            _string(raw, "author_evidence", item_path, errors, required=False) if truth else None
        )
        document_label = _string(
            raw, "document_label", item_path, errors, required=False
        )
        document_evidence = (
            _string(raw, "document_evidence", item_path, errors, required=False)
            if truth
            else None
        )
        if author is not None and truth and author_evidence is None:
            errors.append(f"{item_path}.author_evidence: required with author_label")
        if author is None and truth and "author_evidence" in raw:
            errors.append(f"{item_path}.author_evidence: requires author_label")
        if document_label is not None and truth and document_evidence is None:
            errors.append(f"{item_path}.document_evidence: required with document_label")
        if document_label is None and truth and "document_evidence" in raw:
            errors.append(f"{item_path}.document_evidence: requires document_label")
        spans = _spans(raw.get("spans"), f"{item_path}.spans", errors, truth=truth)
        if doc_id is not None:
            if re.fullmatch(DOC_ID_PATTERN, doc_id) is None:
                errors.append(f"{item_path}.doc_id: must be opaque")
            if doc_id in seen:
                errors.append(f"{item_path}.doc_id: duplicate {doc_id!r}")
            seen.add(doc_id)
            if truth:
                result.append(
                    TruthRecord(
                        doc_id,
                        author,
                        author_evidence,
                        document_label,
                        document_evidence,
                        spans,
                    )
                )
            else:
                result.append(PredictionRecord(doc_id, author, document_label, spans))
    return tuple(result)


def load_truth(path: str | pathlib.Path) -> BenchmarkTruth:
    errors: list[str] = []
    raw = _mapping(_load_json(path), "$", errors)
    if raw is None:
        raise ScoringFormatError(errors)
    required = {"schema_version", "dataset_name", "dataset_version", "manifest_sha256", "records"}
    _check_keys(raw, required, required, "$", errors)
    schema, name, version = _header(raw, errors)
    digest = _string(raw, "manifest_sha256", "$", errors)
    if digest is not None and re.fullmatch(SHA256_PATTERN, digest) is None:
        errors.append("$.manifest_sha256: expected 64 lowercase hexadecimal characters")
    records = _parse_records(raw.get("records"), "$.records", errors, truth=True)
    if errors:
        raise ScoringFormatError(errors)
    assert schema and name and version and digest
    return BenchmarkTruth(schema, name, version, digest, tuple(records))  # type: ignore[arg-type]


def load_submission(path: str | pathlib.Path) -> BenchmarkSubmission:
    errors: list[str] = []
    raw = _mapping(_load_json(path), "$", errors)
    if raw is None:
        raise ScoringFormatError(errors)
    required = {"schema_version", "dataset_name", "dataset_version", "predictions"}
    _check_keys(raw, required, required, "$", errors)
    schema, name, version = _header(raw, errors)
    records = _parse_records(raw.get("predictions"), "$.predictions", errors, truth=False)
    if errors:
        raise ScoringFormatError(errors)
    assert schema and name and version
    return BenchmarkSubmission(schema, name, version, tuple(records))  # type: ignore[arg-type]


def validate_scoring_bundle(
    manifest: BenchmarkManifest,
    truth: BenchmarkTruth,
    submission: BenchmarkSubmission,
    *,
    manifest_sha256: str,
) -> None:
    errors = []
    identity = (manifest.dataset.name, manifest.dataset.version)
    if (truth.dataset_name, truth.dataset_version) != identity:
        errors.append("truth dataset identity does not match manifest")
    if (submission.dataset_name, submission.dataset_version) != identity:
        errors.append("submission dataset identity does not match manifest")
    if truth.manifest_sha256 != manifest_sha256:
        errors.append("truth manifest_sha256 does not match public manifest bytes")
    blind = {document.doc_id: document for document in manifest.documents if document.split == "blind"}
    truth_ids = set(truth.by_id())
    prediction_ids = set(submission.by_id())
    if truth_ids != set(blind):
        errors.append(
            f"truth ids do not equal blind ids: missing={sorted(set(blind)-truth_ids)}, "
            f"extra={sorted(truth_ids-set(blind))}"
        )
    if prediction_ids != set(blind):
        errors.append(
            f"prediction ids do not equal blind ids: missing={sorted(set(blind)-prediction_ids)}, "
            f"extra={sorted(prediction_ids-set(blind))}"
        )
    for doc_id, record in truth.by_id().items():
        document = blind.get(doc_id)
        if document is None:
            continue
        if "mixed_authorship" in document.task_types and not record.spans:
            errors.append(f"{doc_id}: mixed_authorship truth requires spans")
        if (
            not record.spans
            and record.author_label is None
            and record.document_label is None
        ):
            errors.append(
                f"{doc_id}: truth has neither author_label, document_label, nor spans"
            )
    if errors:
        raise ScoringFormatError(errors)


def validate_truth_offsets(
    manifest: BenchmarkManifest,
    truth: BenchmarkTruth,
    document_lengths: Mapping[str, int],
) -> None:
    """Bind private truth offsets to the exact verified public text artifacts."""
    errors = []
    for record in truth.records:
        if record.doc_id not in document_lengths:
            errors.append(f"{record.doc_id}: verified document length is missing")
            continue
        length = int(document_lengths[record.doc_id])
        for index, span in enumerate(record.spans):
            if span.end > length:
                errors.append(
                    f"{record.doc_id}: truth span {index} ends at {span.end}, beyond {length}"
                )
        if record.spans:
            if record.spans[0].start != 0:
                errors.append(f"{record.doc_id}: labelled truth must start at 0")
            if any(left.end != right.start for left, right in zip(record.spans[:-1], record.spans[1:])):
                errors.append(f"{record.doc_id}: labelled truth spans must be contiguous")
            if record.spans[-1].end != length:
                errors.append(
                    f"{record.doc_id}: labelled truth ends at {record.spans[-1].end}, expected {length}"
                )
    if errors:
        raise ScoringFormatError(errors)


def _classification_score(
    truth: Sequence[TruthRecord],
    predictions: Mapping[str, PredictionRecord],
    *,
    field: str,
) -> ClassificationScore | None:
    rows = [record for record in truth if getattr(record, field) is not None]
    if not rows:
        return None
    y_true = np.asarray([getattr(record, field) for record in rows], dtype=object)
    raw_pred = [getattr(predictions[record.doc_id], field) for record in rows]
    n_predicted = sum(label is not None for label in raw_pred)
    y_pred = np.asarray(
        [label if label is not None else "__abstain__" for label in raw_pred], dtype=object
    )
    # The estimand is macro-F1 over registered truth classes.  Abstention is a
    # false negative for its true class, not an extra pseudo-author class.
    labels = sorted(set(y_true.tolist()))
    f1 = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return ClassificationScore(
        n_documents=len(rows),
        n_predicted=n_predicted,
        coverage=n_predicted / len(rows),
        accuracy=float(np.mean(y_true == y_pred)),
        macro_f1=float(np.mean(f1)),
    )


def score_submission(
    manifest: BenchmarkManifest,
    truth: BenchmarkTruth,
    submission: BenchmarkSubmission,
    *,
    manifest_sha256: str,
    boundary_tolerance: int = 0,
    segment_iou_threshold: float = 0.5,
    bootstrap_iters: int = 1000,
    seed: int = 42,
    document_lengths: Mapping[str, int] | None = None,
) -> BenchmarkScore:
    validate_scoring_bundle(
        manifest, truth, submission, manifest_sha256=manifest_sha256
    )
    if document_lengths is not None:
        validate_truth_offsets(manifest, truth, document_lengths)
    predictions = submission.by_id()
    authorship = _classification_score(
        truth.records, predictions, field="author_label"
    )
    document_classification = _classification_score(
        truth.records, predictions, field="document_label"
    )
    blind_by_id = {
        document.doc_id: document for document in manifest.documents if document.split == "blind"
    }
    segment_documents = []
    for record in truth.records:
        if not record.spans:
            continue
        predicted = predictions[record.doc_id]
        if not predicted.spans:
            raise ScoringFormatError([f"{record.doc_id}: segmentation prediction is missing"])
        segment_documents.append(
            SegmentationDocument(
                document_id=record.doc_id,
                work_id=blind_by_id[record.doc_id].work,
                truth=tuple(LabeledSpan(s.start, s.end, s.label) for s in record.spans),
                predicted=tuple(LabeledSpan(s.start, s.end, s.label) for s in predicted.spans),
            )
        )
    segmentation = (
        evaluate_corpus(
            segment_documents,
            boundary_tolerance=boundary_tolerance,
            segment_iou_threshold=segment_iou_threshold,
            permutation_safe=False,
            bootstrap_unit="auto",
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        )
        if segment_documents
        else None
    )
    return BenchmarkScore(
        dataset_name=manifest.dataset.name,
        dataset_version=manifest.dataset.version,
        authorship=authorship,
        document_classification=document_classification,
        segmentation=segmentation,
    )


def score_files(
    manifest: BenchmarkManifest,
    manifest_path: str | pathlib.Path,
    truth_path: str | pathlib.Path,
    submission_path: str | pathlib.Path,
    artifact_root: str | pathlib.Path | None = None,
    **kwargs,
) -> BenchmarkScore:
    document_lengths = None
    if artifact_root is not None:
        artifact_report = verify_manifest_artifacts(manifest, artifact_root)
        document_lengths = {
            document.doc_id: document.n_offset_units for document in artifact_report.documents
        }
    return score_submission(
        manifest,
        load_truth(truth_path),
        load_submission(submission_path),
        manifest_sha256=file_sha256(manifest_path),
        document_lengths=document_lengths,
        **kwargs,
    )


__all__ = [
    "BenchmarkScore",
    "BenchmarkSubmission",
    "BenchmarkTruth",
    "ClassificationScore",
    "PredictionRecord",
    "ScoringFormatError",
    "ScoringSpan",
    "TruthRecord",
    "load_submission",
    "load_truth",
    "score_files",
    "score_submission",
    "validate_scoring_bundle",
    "validate_truth_offsets",
]
