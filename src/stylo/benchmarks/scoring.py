"""Blind truth/submission formats and benchmark scoring.

The public manifest never contains labels for ``split=blind``.  A separate
truth file is bound to the exact public manifest bytes by SHA-256 and is held by
an independent custodian until scoring.  Submissions contain only opaque
document ids and predictions.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import stat
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..jsonio import artifact_self_hash
from ..eval.segmentation import (
    CorpusSegmentationReport,
    LabeledSpan,
    SegmentationDocument,
    evaluate_corpus,
)
from .artifacts import verify_manifest_artifacts
from .loader import loads_manifest
from .schema import (
    BenchmarkManifest,
    DOC_ID_PATTERN,
    MANIFEST_SCHEMA_VERSION,
    SHA256_PATTERN,
    TASK_ENDPOINT_MATRIX,
)

_VERIFIED_FILE_FLOW = object()
_ABSTAIN = object()
_RESERVED_ABSTENTION_LABELS = frozenset({"__abstain__"})
SCORE_SCHEMA_VERSION = "stylo.benchmark.score.v2"


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
    author_label_present: bool = True
    document_label_present: bool = True


@dataclasses.dataclass(frozen=True)
class BenchmarkTruth:
    schema_version: str
    dataset_name: str
    dataset_version: str
    manifest_sha256: str
    records: tuple[TruthRecord, ...]
    truth_sha256: str
    escrow_committed: bool

    def by_id(self) -> dict[str, TruthRecord]:
        return {record.doc_id: record for record in self.records}


@dataclasses.dataclass(frozen=True)
class BenchmarkSubmission:
    schema_version: str
    dataset_name: str
    dataset_version: str
    predictions: tuple[PredictionRecord, ...]
    submission_sha256: str

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
    schema_version: str
    dataset_name: str
    dataset_version: str
    input_bindings: dict[str, str]
    artifact_verification: dict[str, object]
    scoring_parameters: dict[str, object]
    protocol_binding: dict[str, object]
    code_binding: dict[str, str]
    runtime_binding: dict[str, str]
    authorship: ClassificationScore | None
    document_classification: ClassificationScore | None
    segmentation: CorpusSegmentationReport | None
    self_hash: str

    def to_dict(self) -> dict:
        value = dataclasses.asdict(self)
        if value["self_hash"] != artifact_self_hash(value):
            raise RuntimeError("benchmark score envelope self_hash mismatch")
        return value


class ScoringFormatError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("invalid scoring file:\n" + "\n".join(f"- {e}" for e in errors))


class BlindBenchmarkMigrationRequired(ScoringFormatError):
    """The legacy single public manifest cannot support a scientific blind score."""


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate object key {key!r}")
        out[key] = value
    return out


def _reject_constant(value: str):
    raise ValueError(f"non-finite number {value!r}")


def _read_regular_bytes(path: str | pathlib.Path, *, label: str) -> bytes:
    p = pathlib.Path(path)
    if p.is_symlink():
        raise ScoringFormatError([f"$: {label} path must not be a symlink: {p}"])
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(p, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError("not a regular file")
            blocks: list[bytes] = []
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                blocks.append(block)
            return b"".join(blocks)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ScoringFormatError([f"$: cannot read {label} bytes from {p}: {exc}"]) from exc


def _load_json(path: str | pathlib.Path) -> tuple[object, str]:
    p = pathlib.Path(path)
    payload = _read_regular_bytes(p, label="JSON")
    try:
        loaded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ScoringFormatError([f"$: cannot load strict JSON from {p}: {exc}"]) from exc
    return loaded, hashlib.sha256(payload).hexdigest()


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


def _nullable_prediction_string(
    raw: Mapping[str, object],
    key: str,
    path: str,
    errors: list[str],
) -> str | None:
    """Parse a prediction label whose only abstention encoding is JSON null."""

    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if type(value) is not str or not value:
        errors.append(f"{path}.{key}: expected non-empty string or null")
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
        if label in _RESERVED_ABSTENTION_LABELS:
            errors.append(
                f"{item_path}.label: reserved abstention label is forbidden"
            )
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
        author = (
            _string(raw, "author_label", item_path, errors, required=False)
            if truth
            else _nullable_prediction_string(
                raw, "author_label", item_path, errors
            )
        )
        author_evidence = (
            _string(raw, "author_evidence", item_path, errors, required=False) if truth else None
        )
        document_label = (
            _string(raw, "document_label", item_path, errors, required=False)
            if truth
            else _nullable_prediction_string(
                raw, "document_label", item_path, errors
            )
        )
        if author in _RESERVED_ABSTENTION_LABELS:
            errors.append(
                f"{item_path}.author_label: reserved abstention label is forbidden"
            )
        if document_label in _RESERVED_ABSTENTION_LABELS:
            errors.append(
                f"{item_path}.document_label: reserved abstention label is forbidden"
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
                result.append(
                    PredictionRecord(
                        doc_id,
                        author,
                        document_label,
                        spans,
                        author_label_present="author_label" in raw,
                        document_label_present="document_label" in raw,
                    )
                )
    return tuple(result)


def load_truth(
    path: str | pathlib.Path,
    *,
    expected_sha256: str | None = None,
    synthetic_integration_only: bool = False,
) -> BenchmarkTruth:
    """Load truth only after checking a pre-scoring byte commitment.

    The narrowly named synthetic bypass exists for generated integration
    controls that cannot make a blind claim.  Scientific/CLI scoring must pass
    an independently published SHA-256.
    """

    if type(synthetic_integration_only) is not bool:
        raise TypeError("synthetic_integration_only must be an exact bool")
    if expected_sha256 is None and not synthetic_integration_only:
        raise ScoringFormatError(
            ["$: expected truth SHA-256 escrow commitment is required before parsing"]
        )
    if expected_sha256 is not None and (
        type(expected_sha256) is not str
        or re.fullmatch(SHA256_PATTERN, expected_sha256) is None
    ):
        raise ScoringFormatError(["$: expected truth SHA-256 is malformed"])
    truth_path = pathlib.Path(path)
    payload = _read_regular_bytes(truth_path, label="truth")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(
        actual_sha256, expected_sha256
    ):
        raise ScoringFormatError(
            [
                "$: truth bytes do not match the pre-scoring escrow commitment "
                f"(expected {expected_sha256}, actual {actual_sha256})"
            ]
        )
    try:
        decoded = payload.decode("utf-8")
        loaded = json.loads(
            decoded,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ScoringFormatError(
            [f"$: cannot load strict JSON from committed truth bytes: {exc}"]
        ) from exc
    errors: list[str] = []
    raw = _mapping(loaded, "$", errors)
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
    return BenchmarkTruth(
        schema,
        name,
        version,
        digest,
        tuple(records),  # type: ignore[arg-type]
        actual_sha256,
        expected_sha256 is not None,
    )


def load_submission(path: str | pathlib.Path) -> BenchmarkSubmission:
    errors: list[str] = []
    loaded, submission_sha256 = _load_json(path)
    raw = _mapping(loaded, "$", errors)
    if raw is None:
        raise ScoringFormatError(errors)
    required = {"schema_version", "dataset_name", "dataset_version", "predictions"}
    _check_keys(raw, required, required, "$", errors)
    schema, name, version = _header(raw, errors)
    records = _parse_records(raw.get("predictions"), "$.predictions", errors, truth=False)
    if errors:
        raise ScoringFormatError(errors)
    assert schema and name and version
    return BenchmarkSubmission(
        schema,
        name,
        version,
        tuple(records),  # type: ignore[arg-type]
        submission_sha256,
    )


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
    predictions = submission.by_id()
    blind_tasks = {
        task for document in blind.values() for task in document.task_types
    }
    missing_blind_tasks = sorted(set(manifest.task_types) - blind_tasks)
    if missing_blind_tasks:
        errors.append(
            "declared benchmark tasks have no blind scoring observations: "
            f"{missing_blind_tasks!r}"
        )
    for field in ("author_label", "document_label"):
        allowed_labels = {
            getattr(record, field)
            for record in truth.records
            if getattr(record, field) is not None
        }
        for doc_id, prediction in predictions.items():
            label = getattr(prediction, field)
            if label is not None and label not in allowed_labels:
                errors.append(
                    f"{doc_id}: {field} {label!r} is outside the registered truth universe"
                )
    allowed_span_labels = {
        span.label for record in truth.records for span in record.spans
    }
    for doc_id, prediction in predictions.items():
        for index, span in enumerate(prediction.spans):
            if span.label not in allowed_span_labels:
                errors.append(
                    f"{doc_id}: span {index} label {span.label!r} is outside "
                    "the registered truth universe"
                )
    for doc_id, record in truth.by_id().items():
        document = blind.get(doc_id)
        if document is None:
            continue
        truth_fields = {
            field
            for field, present in (
                ("author_label", record.author_label is not None),
                ("document_label", record.document_label is not None),
                ("spans", bool(record.spans)),
            )
            if present
        }
        prediction = predictions.get(doc_id)
        prediction_fields = (
            set()
            if prediction is None
            else {
                field
                for field, present in (
                    ("author_label", prediction.author_label is not None),
                    ("document_label", prediction.document_label is not None),
                    ("spans", bool(prediction.spans)),
                )
                if present
            }
        )
        contracts = [
            TASK_ENDPOINT_MATRIX[task]
            for task in document.task_types
            if task in TASK_ENDPOINT_MATRIX
        ]
        allowed = set().union(
            *(contract["allowed_truth_fields"] for contract in contracts)
        )
        disallowed_truth = sorted(truth_fields - allowed)
        if disallowed_truth:
            errors.append(
                f"{doc_id}: truth fields {disallowed_truth!r} are not registered "
                f"for tasks {list(document.task_types)!r}"
            )
        disallowed_prediction = sorted(prediction_fields - allowed)
        if disallowed_prediction:
            errors.append(
                f"{doc_id}: prediction fields {disallowed_prediction!r} are not "
                f"registered for tasks {list(document.task_types)!r}"
            )
        if prediction is not None:
            for field in ("author_label", "document_label"):
                if (
                    field in truth_fields
                    and not getattr(prediction, f"{field}_present")
                ):
                    errors.append(
                        f"{doc_id}: classification prediction field {field!r} "
                        "must be present as a registered label or explicit JSON null"
                    )
        for task in document.task_types:
            contract = TASK_ENDPOINT_MATRIX.get(task)
            if contract is None:
                continue
            missing = sorted(
                set(contract.get("required_all", ())) - truth_fields
            )
            if missing:
                errors.append(
                    f"{doc_id}: task {task!r} requires truth fields {missing!r}"
                )
            for alternatives in contract.get("required_any", ()):
                if not (set(alternatives) & truth_fields):
                    errors.append(
                        f"{doc_id}: task {task!r} requires at least one of "
                        f"{sorted(alternatives)!r} in truth"
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
    y_true = [getattr(record, field) for record in rows]
    raw_pred = [getattr(predictions[record.doc_id], field) for record in rows]
    n_predicted = sum(label is not None for label in raw_pred)
    y_pred: list[object] = [
        label if label is not None else _ABSTAIN for label in raw_pred
    ]
    # The estimand is macro-F1 over registered truth classes.  Abstention is a
    # false negative for its true class, not an extra pseudo-author class.
    labels = sorted(set(y_true))
    f1 = []
    for label in labels:
        tp = sum(truth_label == label and pred_label == label
                 for truth_label, pred_label in zip(y_true, y_pred, strict=True))
        fp = sum(truth_label != label and pred_label == label
                 for truth_label, pred_label in zip(y_true, y_pred, strict=True))
        fn = sum(truth_label == label and pred_label != label
                 for truth_label, pred_label in zip(y_true, y_pred, strict=True))
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return ClassificationScore(
        n_documents=len(rows),
        n_predicted=n_predicted,
        coverage=n_predicted / len(rows),
        accuracy=(
            sum(
                pred_label is not _ABSTAIN and truth_label == pred_label
                for truth_label, pred_label in zip(y_true, y_pred, strict=True)
            )
            / len(rows)
        ),
        macro_f1=float(np.mean(f1)),
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _score_code_binding() -> dict[str, str]:
    stylo_root = pathlib.Path(__file__).resolve().parents[1]
    paths = (
        pathlib.Path(__file__).resolve(),
        pathlib.Path(__file__).resolve().with_name("artifacts.py"),
        pathlib.Path(__file__).resolve().with_name("loader.py"),
        pathlib.Path(__file__).resolve().with_name("schema.py"),
        pathlib.Path(__file__).resolve().with_name("validator.py"),
        stylo_root / "domain" / "segmentation.py",
        stylo_root / "jsonio.py",
        stylo_root / "eval" / "segmentation.py",
    )
    binding: dict[str, str] = {}
    for path in paths:
        payload = _read_regular_bytes(path, label="scoring code")
        binding[path.relative_to(stylo_root).as_posix()] = hashlib.sha256(payload).hexdigest()
    return binding


def _build_score_envelope(
    *,
    manifest: BenchmarkManifest,
    manifest_sha256: str,
    truth: BenchmarkTruth,
    submission: BenchmarkSubmission,
    artifact_report_sha256: str | None,
    artifact_verified: bool,
    boundary_tolerance: int,
    segment_iou_threshold: float,
    bootstrap_iters: int,
    seed: int,
    authorship: ClassificationScore | None,
    document_classification: ClassificationScore | None,
    segmentation: CorpusSegmentationReport | None,
    segmentation_bootstrap_unit: str | None,
    synthetic_integration_only: bool,
) -> BenchmarkScore:
    fields: dict[str, object] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "dataset_name": manifest.dataset.name,
        "dataset_version": manifest.dataset.version,
        "input_bindings": {
            "manifest_sha256": manifest_sha256,
            "truth_sha256": truth.truth_sha256,
            "submission_sha256": submission.submission_sha256,
        },
        "artifact_verification": {
            "verified": artifact_verified,
            "report_sha256": artifact_report_sha256,
            "mode": (
                "synthetic_integration_only"
                if synthetic_integration_only
                else "scientific_verified"
            ),
        },
        "scoring_parameters": {
            "boundary_tolerance": boundary_tolerance,
            "segment_iou_threshold": float(segment_iou_threshold),
            "bootstrap_iters": bootstrap_iters,
            "seed": seed,
        },
        "protocol_binding": {
            "manifest_schema_version": manifest.schema_version,
            "score_schema_version": SCORE_SCHEMA_VERSION,
            "offset_unit": manifest.dataset.offset_unit,
            "tokenizer": manifest.dataset.tokenizer,
            "segmentation_permutation_safe": False,
            "bootstrap_unit": segmentation_bootstrap_unit,
            "endpoint_counts": {
                task: sum(
                    task in document.task_types and document.split == "blind"
                    for document in manifest.documents
                )
                for task in manifest.task_types
            },
        },
        "code_binding": _score_code_binding(),
        "runtime_binding": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "scikit_learn": _package_version("scikit-learn"),
        },
        "authorship": (
            None if authorship is None else dataclasses.asdict(authorship)
        ),
        "document_classification": (
            None
            if document_classification is None
            else dataclasses.asdict(document_classification)
        ),
        "segmentation": (
            None if segmentation is None else dataclasses.asdict(segmentation)
        ),
    }
    fields["self_hash"] = artifact_self_hash(fields)
    return BenchmarkScore(
        schema_version=SCORE_SCHEMA_VERSION,
        dataset_name=manifest.dataset.name,
        dataset_version=manifest.dataset.version,
        input_bindings=fields["input_bindings"],  # type: ignore[arg-type]
        artifact_verification=fields["artifact_verification"],  # type: ignore[arg-type]
        scoring_parameters=fields["scoring_parameters"],  # type: ignore[arg-type]
        protocol_binding=fields["protocol_binding"],  # type: ignore[arg-type]
        code_binding=fields["code_binding"],  # type: ignore[arg-type]
        runtime_binding=fields["runtime_binding"],  # type: ignore[arg-type]
        authorship=authorship,
        document_classification=document_classification,
        segmentation=segmentation,
        self_hash=fields["self_hash"],  # type: ignore[arg-type]
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
    artifact_report_sha256: str | None = None,
    artifact_verified: bool = False,
    segmentation_bootstrap_unit: str | None = None,
    synthetic_integration_only: bool = False,
    _verified_file_flow: object | None = None,
) -> BenchmarkScore:
    if type(boundary_tolerance) is not int or boundary_tolerance < 0:
        raise ScoringFormatError(["boundary_tolerance must be an exact nonnegative integer"])
    if (
        isinstance(segment_iou_threshold, bool)
        or not isinstance(segment_iou_threshold, (int, float))
        or not np.isfinite(segment_iou_threshold)
        or not 0.0 <= float(segment_iou_threshold) <= 1.0
    ):
        raise ScoringFormatError(["segment_iou_threshold must be finite within [0, 1]"])
    if type(bootstrap_iters) is not int or bootstrap_iters <= 0:
        raise ScoringFormatError(["bootstrap_iters must be a positive exact integer"])
    if type(seed) is not int:
        raise ScoringFormatError(["seed must be an exact integer"])
    if type(artifact_verified) is not bool:
        raise ScoringFormatError(["artifact_verified must be an exact bool"])
    if not synthetic_integration_only and _verified_file_flow is not _VERIFIED_FILE_FLOW:
        raise ScoringFormatError(
            [
                "in-memory score_submission is integration-only; scientific scoring "
                "must use the sealed file-based scorer"
            ]
        )
    if not truth.escrow_committed and not synthetic_integration_only:
        raise ScoringFormatError(
            ["truth object is not bound to a pre-scoring escrow commitment"]
        )
    if not synthetic_integration_only and (
        not artifact_verified
        or document_lengths is None
        or type(artifact_report_sha256) is not str
        or re.fullmatch(SHA256_PATTERN, artifact_report_sha256) is None
    ):
        raise ScoringFormatError(
            ["scientific scoring requires a digest-bound artifact verification receipt"]
        )
    if document_lengths is not None:
        validate_truth_offsets(manifest, truth, document_lengths)
    validate_scoring_bundle(
        manifest, truth, submission, manifest_sha256=manifest_sha256
    )
    if manifest.dataset.offset_unit == "character" and any(
        record.spans for record in truth.records
    ):
        raise ScoringFormatError(
            [
                "schema v1 character-offset segmentation is not scoreable: "
                "token-named metrics cannot represent that estimand"
            ]
        )
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
    if segment_documents:
        if segmentation_bootstrap_unit not in {"work", "document"}:
            raise ScoringFormatError(
                [
                    "segmentation_bootstrap_unit must explicitly register "
                    "'work' or 'document' when segmentation is scored"
                ]
            )
        if segmentation_bootstrap_unit == "work" and any(
            document.work is None
            for document in blind_by_id.values()
            if set(document.task_types) & {"idio_shift", "mixed_authorship"}
        ):
            raise ScoringFormatError(
                ["work bootstrap requires a registered work identity for every segmentation document"]
            )
        if segmentation_bootstrap_unit == "document" and not synthetic_integration_only:
            raise ScoringFormatError(
                ["scientific segmentation requires the registered work bootstrap unit"]
            )
    elif segmentation_bootstrap_unit is not None:
        raise ScoringFormatError(
            ["segmentation_bootstrap_unit was supplied but no segmentation endpoint is registered"]
        )
    segmentation = (
        evaluate_corpus(
            segment_documents,
            boundary_tolerance=boundary_tolerance,
            segment_iou_threshold=segment_iou_threshold,
            permutation_safe=False,
            bootstrap_unit=segmentation_bootstrap_unit,
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        )
        if segment_documents
        else None
    )
    return _build_score_envelope(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        truth=truth,
        submission=submission,
        artifact_report_sha256=artifact_report_sha256,
        artifact_verified=artifact_verified,
        boundary_tolerance=boundary_tolerance,
        segment_iou_threshold=segment_iou_threshold,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
        authorship=authorship,
        document_classification=document_classification,
        segmentation=segmentation,
        segmentation_bootstrap_unit=segmentation_bootstrap_unit,
        synthetic_integration_only=synthetic_integration_only,
    )


def score_files(
    manifest: BenchmarkManifest,
    manifest_path: str | pathlib.Path,
    truth_path: str | pathlib.Path,
    submission_path: str | pathlib.Path,
    artifact_root: str | pathlib.Path | None = None,
    *,
    expected_truth_sha256: str | None = None,
    synthetic_integration_only: bool = False,
    **kwargs,
) -> BenchmarkScore:
    if type(synthetic_integration_only) is not bool:
        raise TypeError("synthetic_integration_only must be an exact bool")
    manifest_payload = _read_regular_bytes(manifest_path, label="manifest")
    try:
        parsed_manifest = loads_manifest(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ScoringFormatError(
            [f"manifest bytes cannot be parsed under the strict schema: {exc}"]
        ) from exc
    if parsed_manifest != manifest:
        raise ScoringFormatError(
            ["manifest object does not equal the exact bytes supplied to score_files"]
        )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()

    document_lengths = None
    artifact_report_sha256 = None
    artifact_verified = False
    if artifact_root is not None:
        artifact_report = verify_manifest_artifacts(manifest, artifact_root)
        document_lengths = {
            document.doc_id: document.n_offset_units for document in artifact_report.documents
        }
        artifact_report_sha256 = artifact_report.receipt_sha256()
        artifact_verified = True
    elif not synthetic_integration_only:
        raise ScoringFormatError(
            ["scientific score_files requires artifact_root; unverified offsets are forbidden"]
        )

    if not synthetic_integration_only and any(
        document.split == "blind" for document in manifest.documents
    ):
        raise BlindBenchmarkMigrationRequired(
            [
                "benchmark manifest schema 1.0 has no custodian full-provenance "
                "binding for redacted blind rows; scientific blind scoring is "
                "blocked until the versioned dual-manifest migration"
            ]
        )
    return score_submission(
        manifest,
        load_truth(
            truth_path,
            expected_sha256=expected_truth_sha256,
            synthetic_integration_only=synthetic_integration_only,
        ),
        load_submission(submission_path),
        manifest_sha256=manifest_sha256,
        document_lengths=document_lengths,
        artifact_report_sha256=artifact_report_sha256,
        artifact_verified=artifact_verified,
        synthetic_integration_only=synthetic_integration_only,
        _verified_file_flow=_VERIFIED_FILE_FLOW,
        **kwargs,
    )


__all__ = [
    "BenchmarkScore",
    "BenchmarkSubmission",
    "BenchmarkTruth",
    "BlindBenchmarkMigrationRequired",
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
