"""Dependency-free validation for benchmark JSON manifests."""
from __future__ import annotations

import re
import pathlib
from typing import Any

from .schema import (
    BenchmarkDocument,
    BenchmarkManifest,
    BenchmarkSpan,
    DatasetMetadata,
    DOC_ID_PATTERN,
    MANIFEST_SCHEMA_VERSION,
    SHA256_PATTERN,
    SUPPORTED_OFFSET_UNITS,
    SUPPORTED_SPLIT_ROLES,
    SUPPORTED_TASK_TYPES,
    SUPPORTED_TOKENIZERS,
    TASK_ENDPOINT_MATRIX,
    SourceProvenance,
)


_TOP_LEVEL_KEYS = frozenset({"schema_version", "dataset", "task_types", "documents"})
_DATASET_KEYS = frozenset(
    {"name", "version", "license", "offset_unit", "tokenizer", "language", "description"}
)
_SOURCE_KEYS = frozenset({"source_id", "provenance", "revision", "sha256"})
_DOCUMENT_KEYS = frozenset(
    {
        "doc_id",
        "source",
        "split",
        "task_types",
        "spans",
        "text_path",
        "author_label",
        "document_label",
        "work",
        "edition",
        "period",
        "genre",
        "topic",
        "register",
    }
)
_SPAN_KEYS = frozenset({"start", "end", "label", "ground_truth_known", "evidence"})
_OPTIONAL_DOCUMENT_STRINGS = (
    "text_path",
    "author_label",
    "document_label",
    "work",
    "edition",
    "period",
    "genre",
    "topic",
    "register",
)
_BLIND_IDENTITY_FIELDS = frozenset(
    {"work", "edition", "period", "genre", "topic", "register"}
)


class ManifestValidationError(ValueError):
    """Raised when a benchmark manifest violates one or more invariants."""

    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(errors)
        details = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"invalid benchmark manifest:\n{details}")


def validate_manifest(value: object) -> BenchmarkManifest:
    """Validate a JSON-compatible object and return immutable typed data.

    Validation is deliberately stricter than ordinary ``json.loads`` output:
    unknown keys, semantic document ids, duplicate ids, malformed hashes,
    overlapping spans and any label exposure in the blind split are rejected.
    All detected violations are reported together where possible.
    """

    errors: list[str] = []
    raw = _object(value, "$", errors)
    if raw is None:
        raise ManifestValidationError(errors)

    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_KEYS, "$", errors)
    schema_version = _required_string(raw, "schema_version", "$", errors)
    if schema_version and schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            "$.schema_version: expected "
            f"{MANIFEST_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    dataset = _parse_dataset(raw.get("dataset"), "$.dataset", errors)
    task_types = _parse_task_types(raw.get("task_types"), "$.task_types", errors)

    documents_value = raw.get("documents")
    documents_raw = _array(documents_value, "$.documents", errors)
    documents: list[BenchmarkDocument] = []
    seen_doc_ids: dict[str, int] = {}
    if documents_raw is not None:
        if not documents_raw:
            errors.append("$.documents: must contain at least one document")
        for index, document_value in enumerate(documents_raw):
            path = f"$.documents[{index}]"
            document = _parse_document(document_value, path, task_types, errors)
            if document is None:
                continue
            first_index = seen_doc_ids.get(document.doc_id)
            if first_index is not None:
                errors.append(
                    f"{path}.doc_id: duplicate {document.doc_id!r}; "
                    f"first declared at $.documents[{first_index}].doc_id"
                )
            else:
                seen_doc_ids[document.doc_id] = index
            documents.append(document)

    _validate_split_isolation(documents, errors)
    represented_tasks = {task for document in documents for task in document.task_types}
    unused_tasks = sorted(set(task_types) - represented_tasks)
    if unused_tasks:
        errors.append(
            "$.task_types: every declared task must have at least one document; "
            f"unused={unused_tasks!r}"
        )

    if errors:
        raise ManifestValidationError(errors)

    # Required members are guaranteed by the checks above when there are no
    # errors; these guards make that invariant explicit to type checkers.
    assert schema_version is not None
    assert dataset is not None
    assert documents_raw is not None
    return BenchmarkManifest(
        schema_version=schema_version,
        dataset=dataset,
        task_types=task_types,
        documents=tuple(documents),
    )


def _parse_dataset(
    value: object, path: str, errors: list[str]
) -> DatasetMetadata | None:
    raw = _object(value, path, errors)
    if raw is None:
        return None
    required = frozenset({"name", "version", "license", "offset_unit"})
    _check_keys(raw, required, _DATASET_KEYS, path, errors)
    name = _required_string(raw, "name", path, errors)
    version = _required_string(raw, "version", path, errors)
    license_name = _required_string(raw, "license", path, errors)
    offset_unit = _required_string(raw, "offset_unit", path, errors)
    tokenizer = _optional_string(raw, "tokenizer", path, errors)
    if offset_unit and offset_unit not in SUPPORTED_OFFSET_UNITS:
        errors.append(
            f"{path}.offset_unit: unsupported unit {offset_unit!r}; expected one of "
            f"{sorted(SUPPORTED_OFFSET_UNITS)!r}"
        )
    if offset_unit == "token":
        if tokenizer is None:
            errors.append(f"{path}.tokenizer: required when offset_unit is 'token'")
        elif tokenizer not in SUPPORTED_TOKENIZERS:
            errors.append(
                f"{path}.tokenizer: unsupported tokenizer {tokenizer!r}; expected one of "
                f"{sorted(SUPPORTED_TOKENIZERS)!r}"
            )
    elif tokenizer is not None:
        errors.append(f"{path}.tokenizer: only valid when offset_unit is 'token'")
    language = _optional_string(raw, "language", path, errors)
    description = _optional_string(raw, "description", path, errors)
    if name is None or version is None or license_name is None or offset_unit is None:
        return None
    return DatasetMetadata(
        name=name,
        version=version,
        license=license_name,
        offset_unit=offset_unit,
        tokenizer=tokenizer,
        language=language,
        description=description,
    )


def _parse_source(
    value: object, path: str, errors: list[str]
) -> SourceProvenance | None:
    raw = _object(value, path, errors)
    if raw is None:
        return None
    _check_keys(raw, _SOURCE_KEYS, _SOURCE_KEYS, path, errors)
    source_id = _required_string(raw, "source_id", path, errors)
    provenance = _required_string(raw, "provenance", path, errors)
    revision = _required_string(raw, "revision", path, errors)
    sha256 = _required_string(raw, "sha256", path, errors)
    if sha256 and re.fullmatch(SHA256_PATTERN, sha256) is None:
        errors.append(f"{path}.sha256: expected 64 lowercase hexadecimal characters")
    if source_id is None or provenance is None or revision is None or sha256 is None:
        return None
    return SourceProvenance(
        source_id=source_id,
        provenance=provenance,
        revision=revision,
        sha256=sha256,
    )


def _parse_document(
    value: object,
    path: str,
    manifest_task_types: tuple[str, ...],
    errors: list[str],
) -> BenchmarkDocument | None:
    raw = _object(value, path, errors)
    if raw is None:
        return None
    required = frozenset({"doc_id", "source", "split", "task_types", "spans"})
    _check_keys(raw, required, _DOCUMENT_KEYS, path, errors)

    doc_id = _required_string(raw, "doc_id", path, errors)
    if doc_id and re.fullmatch(DOC_ID_PATTERN, doc_id) is None:
        errors.append(
            f"{path}.doc_id: must be opaque: 'doc_' followed by 16-64 lowercase hex digits"
        )

    source = _parse_source(raw.get("source"), f"{path}.source", errors)
    split = _required_string(raw, "split", path, errors)
    if split and split not in SUPPORTED_SPLIT_ROLES:
        errors.append(
            f"{path}.split: unsupported role {split!r}; expected one of "
            f"{sorted(SUPPORTED_SPLIT_ROLES)!r}"
        )

    task_types = _parse_task_types(raw.get("task_types"), f"{path}.task_types", errors)
    undeclared_tasks = sorted(set(task_types) - set(manifest_task_types))
    if undeclared_tasks:
        errors.append(
            f"{path}.task_types: not declared by manifest task_types: {undeclared_tasks!r}"
        )

    spans_value = raw.get("spans")
    spans_raw = _array(spans_value, f"{path}.spans", errors)
    spans = _parse_spans(spans_raw, f"{path}.spans", errors)

    optional: dict[str, str | None] = {}
    for key in _OPTIONAL_DOCUMENT_STRINGS:
        optional[key] = _optional_string(raw, key, path, errors)

    if split != "blind":
        allowed_fields = set().union(
            *(
                TASK_ENDPOINT_MATRIX[task]["allowed_truth_fields"]
                for task in task_types
                if task in TASK_ENDPOINT_MATRIX
            )
        )
        present_fields = {
            field
            for field, present in (
                ("author_label", optional["author_label"] is not None),
                ("document_label", optional["document_label"] is not None),
                ("spans", bool(spans)),
            )
            if present
        }
        disallowed = sorted(present_fields - allowed_fields)
        if disallowed:
            errors.append(
                f"{path}: truth fields {disallowed!r} are not registered for "
                f"tasks {list(task_types)!r}"
            )
        for task in task_types:
            contract = TASK_ENDPOINT_MATRIX.get(task)
            if contract is None:
                continue
            missing = sorted(set(contract.get("required_all", ())) - present_fields)
            if missing:
                errors.append(
                    f"{path}: task {task!r} requires truth fields {missing!r}"
                )
            for alternatives in contract.get("required_any", ()):
                if not (set(alternatives) & present_fields):
                    errors.append(
                        f"{path}: task {task!r} requires at least one of "
                        f"{sorted(alternatives)!r}"
                    )

    if split == "blind":
        if "author_label" in raw:
            errors.append(f"{path}.author_label: forbidden in blind split")
        if "document_label" in raw:
            errors.append(f"{path}.document_label: forbidden in blind split")
        if spans_raw:
            errors.append(f"{path}.spans: blind split must not expose span labels")
        for field in sorted(_BLIND_IDENTITY_FIELDS & raw.keys()):
            errors.append(f"{path}.{field}: identity-bearing metadata is forbidden in blind split")

    if doc_id is None or source is None or split is None or spans_raw is None:
        return None
    return BenchmarkDocument(
        doc_id=doc_id,
        source=source,
        split=split,
        task_types=task_types,
        spans=spans,
        text_path=optional["text_path"],
        author_label=optional["author_label"],
        document_label=optional["document_label"],
        work=optional["work"],
        edition=optional["edition"],
        period=optional["period"],
        genre=optional["genre"],
        topic=optional["topic"],
        register=optional["register"],
    )


def _roles_are_isolated(left: str, right: str) -> bool:
    """Whether two split roles must not share a work/source identity."""

    if left == right:
        return False
    if "blind" in {left, right}:
        return True
    development = {"train", "validation"}
    return not ({left, right} <= development)


def _validate_split_isolation(
    documents: list[BenchmarkDocument], errors: list[str]
) -> None:
    """Reject content/path/revision leakage before a benchmark is accepted."""

    for index, left in enumerate(documents):
        for right in documents[index + 1 :]:
            if not _roles_are_isolated(left.split, right.split):
                continue
            if left.source.sha256 == right.source.sha256:
                errors.append(
                    "$.documents: exact source bytes cross isolated split roles: "
                    f"{left.doc_id!r} ({left.split}) vs {right.doc_id!r} ({right.split})"
                )
            if (
                left.text_path is not None
                and right.text_path is not None
                and pathlib.PurePosixPath(left.text_path)
                == pathlib.PurePosixPath(right.text_path)
            ):
                errors.append(
                    "$.documents: text_path crosses isolated split roles: "
                    f"{left.doc_id!r} ({left.split}) vs {right.doc_id!r} ({right.split})"
                )
            if (
                left.source.source_id == right.source.source_id
                and left.source.revision == right.source.revision
            ):
                errors.append(
                    "$.documents: source revision crosses isolated split roles: "
                    f"{left.doc_id!r} ({left.split}) vs {right.doc_id!r} ({right.split})"
                )
            if left.work is not None and left.work == right.work:
                errors.append(
                    "$.documents: work identity crosses isolated split roles: "
                    f"{left.doc_id!r} ({left.split}) vs {right.doc_id!r} ({right.split})"
                )


def _parse_spans(
    values: list[object] | None, path: str, errors: list[str]
) -> tuple[BenchmarkSpan, ...]:
    if values is None:
        return ()
    spans: list[BenchmarkSpan] = []
    previous_end: int | None = None
    for index, value in enumerate(values):
        span_path = f"{path}[{index}]"
        raw = _object(value, span_path, errors)
        if raw is None:
            continue
        required = frozenset({"start", "end", "label", "ground_truth_known"})
        _check_keys(raw, required, _SPAN_KEYS, span_path, errors)
        start = _required_integer(raw, "start", span_path, errors)
        end = _required_integer(raw, "end", span_path, errors)
        label = _required_string(raw, "label", span_path, errors)
        known = _required_boolean(raw, "ground_truth_known", span_path, errors)
        evidence = _optional_string(raw, "evidence", span_path, errors)
        if known is True and evidence is None:
            errors.append(
                f"{span_path}.evidence: required when ground_truth_known is true"
            )
        if known is False and evidence is not None:
            errors.append(
                f"{span_path}.evidence: must be omitted when ground_truth_known is false"
            )

        if start is not None and start < 0:
            errors.append(f"{span_path}.start: must be non-negative")
        if start is not None and end is not None and end <= start:
            errors.append(f"{span_path}.end: must be greater than start")
        if start is not None and previous_end is not None and start < previous_end:
            errors.append(
                f"{span_path}: spans must be ordered by start and must not overlap"
            )
        if end is not None:
            previous_end = end if previous_end is None else max(previous_end, end)

        if start is None or end is None or label is None or known is None:
            continue
        spans.append(
            BenchmarkSpan(
                start=start,
                end=end,
                label=label,
                ground_truth_known=known,
                evidence=evidence,
            )
        )
    return tuple(spans)


def _parse_task_types(
    value: object, path: str, errors: list[str]
) -> tuple[str, ...]:
    raw = _array(value, path, errors)
    if raw is None:
        return ()
    if not raw:
        errors.append(f"{path}: must contain at least one task type")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        if type(item) is not str or not item:
            errors.append(f"{item_path}: expected a non-empty string")
            continue
        if item not in SUPPORTED_TASK_TYPES:
            errors.append(
                f"{item_path}: unsupported task type {item!r}; expected one of "
                f"{sorted(SUPPORTED_TASK_TYPES)!r}"
            )
            continue
        if item in seen:
            errors.append(f"{item_path}: duplicate task type {item!r}")
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


def _object(value: object, path: str, errors: list[str]) -> dict[str, Any] | None:
    if type(value) is not dict:
        errors.append(f"{path}: expected an object")
        return None
    if any(type(key) is not str for key in value):
        errors.append(f"{path}: object keys must be strings")
        return None
    return value


def _array(value: object, path: str, errors: list[str]) -> list[object] | None:
    if type(value) is not list:
        errors.append(f"{path}: expected an array")
        return None
    return value


def _check_keys(
    raw: dict[str, Any],
    required: frozenset[str],
    allowed: frozenset[str],
    path: str,
    errors: list[str],
) -> None:
    for key in sorted(required - raw.keys()):
        errors.append(f"{path}.{key}: required field is missing")
    for key in sorted(raw.keys() - allowed):
        errors.append(f"{path}.{key}: unknown field")


def _required_string(
    raw: dict[str, Any], key: str, path: str, errors: list[str]
) -> str | None:
    if key not in raw:
        return None
    value = raw[key]
    if type(value) is not str or not value:
        errors.append(f"{path}.{key}: expected a non-empty string")
        return None
    return value


def _optional_string(
    raw: dict[str, Any], key: str, path: str, errors: list[str]
) -> str | None:
    if key not in raw:
        return None
    return _required_string(raw, key, path, errors)


def _required_integer(
    raw: dict[str, Any], key: str, path: str, errors: list[str]
) -> int | None:
    if key not in raw:
        return None
    value = raw[key]
    if type(value) is not int:
        errors.append(f"{path}.{key}: expected an integer")
        return None
    return value


def _required_boolean(
    raw: dict[str, Any], key: str, path: str, errors: list[str]
) -> bool | None:
    if key not in raw:
        return None
    value = raw[key]
    if type(value) is not bool:
        errors.append(f"{path}.{key}: expected a boolean")
        return None
    return value


__all__ = ["ManifestValidationError", "validate_manifest"]
