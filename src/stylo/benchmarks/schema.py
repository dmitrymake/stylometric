"""Schema declarations and immutable models for stylometry benchmarks.

``MANIFEST_SCHEMA`` is a JSON Schema document for publishers and tooling.  The
runtime validator lives in :mod:`stylo.benchmarks.validator` and deliberately
does not require the third-party ``jsonschema`` package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


MANIFEST_SCHEMA_VERSION: Final = "1.0"
SUPPORTED_TASK_TYPES: Final = frozenset(
    {"spoof", "idio_shift", "mixed_authorship"}
)
SUPPORTED_SPLIT_ROLES: Final = frozenset(
    {"train", "validation", "test", "blind"}
)
SUPPORTED_OFFSET_UNITS: Final = frozenset({"token", "character"})
SUPPORTED_TOKENIZERS: Final = frozenset({"stylo_unicode_word_punct_v1"})
DOC_ID_PATTERN: Final = r"^doc_[0-9a-f]{16,64}$"
SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"

# The v1 endpoint registry is intentionally small and explicit.  In
# particular, spans are not a generic truth field: they belong only to the two
# registered segmentation tasks.  ``spoof`` may be evaluated either as an
# author or document classification endpoint, but must expose at least one of
# those fields in released/private truth.
TASK_ENDPOINT_MATRIX: Final = {
    "spoof": {
        "allowed_truth_fields": frozenset({"author_label", "document_label"}),
        "required_any": (frozenset({"author_label", "document_label"}),),
    },
    "idio_shift": {
        "allowed_truth_fields": frozenset({"author_label", "spans"}),
        "required_all": frozenset({"spans"}),
    },
    "mixed_authorship": {
        "allowed_truth_fields": frozenset({"spans"}),
        "required_all": frozenset({"spans"}),
    },
}


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset identity and redistribution metadata."""

    name: str
    version: str
    license: str
    offset_unit: str
    tokenizer: str | None = None
    language: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class SourceProvenance:
    """Exact upstream source revision represented by a document."""

    source_id: str
    provenance: str
    revision: str
    sha256: str


@dataclass(frozen=True)
class BenchmarkSpan:
    """A half-open, character-offset ground-truth interval ``[start, end)``."""

    start: int
    end: int
    label: str
    ground_truth_known: bool
    evidence: str | None = None


@dataclass(frozen=True)
class BenchmarkDocument:
    """One document entry in a benchmark manifest."""

    doc_id: str
    source: SourceProvenance
    split: str
    task_types: tuple[str, ...]
    spans: tuple[BenchmarkSpan, ...]
    text_path: str | None = None
    author_label: str | None = None
    document_label: str | None = None
    work: str | None = None
    edition: str | None = None
    period: str | None = None
    genre: str | None = None
    topic: str | None = None
    register: str | None = None


@dataclass(frozen=True)
class BenchmarkManifest:
    """Validated, immutable representation of a benchmark manifest."""

    schema_version: str
    dataset: DatasetMetadata
    task_types: tuple[str, ...]
    documents: tuple[BenchmarkDocument, ...]


# This public declaration makes the format usable by non-Python producers.
# Cross-field invariants (unique ids, task subsets, non-overlapping spans and
# blind-label isolation) are additionally enforced by the manual validator.
MANIFEST_SCHEMA: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://stylo.invalid/schemas/benchmark-manifest-1.0.json",
    "title": "SPOOF-RU / IDIOSHIFT-RU benchmark manifest",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "dataset", "task_types", "documents"],
    "properties": {
        "schema_version": {"const": MANIFEST_SCHEMA_VERSION},
        "dataset": {"$ref": "#/$defs/dataset"},
        "task_types": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"enum": sorted(SUPPORTED_TASK_TYPES)},
        },
        "documents": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/document"},
        },
    },
    "$defs": {
        "non_empty_string": {"type": "string", "minLength": 1},
        "dataset": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "version", "license", "offset_unit"],
            "properties": {
                "name": {"$ref": "#/$defs/non_empty_string"},
                "version": {"$ref": "#/$defs/non_empty_string"},
                "license": {"$ref": "#/$defs/non_empty_string"},
                "offset_unit": {"enum": sorted(SUPPORTED_OFFSET_UNITS)},
                "tokenizer": {"enum": sorted(SUPPORTED_TOKENIZERS)},
                "language": {"$ref": "#/$defs/non_empty_string"},
                "description": {"$ref": "#/$defs/non_empty_string"},
            },
            "allOf": [
                {
                    "if": {
                        "properties": {"offset_unit": {"const": "token"}},
                        "required": ["offset_unit"],
                    },
                    "then": {"required": ["tokenizer"]},
                    "else": {"not": {"required": ["tokenizer"]}},
                }
            ],
        },
        "source": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source_id", "provenance", "revision", "sha256"],
            "properties": {
                "source_id": {"$ref": "#/$defs/non_empty_string"},
                "provenance": {"$ref": "#/$defs/non_empty_string"},
                "revision": {"$ref": "#/$defs/non_empty_string"},
                "sha256": {"type": "string", "pattern": SHA256_PATTERN},
            },
        },
        "span": {
            "type": "object",
            "additionalProperties": False,
            "required": ["start", "end", "label", "ground_truth_known"],
            "properties": {
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 1},
                "label": {"$ref": "#/$defs/non_empty_string"},
                "ground_truth_known": {"type": "boolean"},
                "evidence": {"$ref": "#/$defs/non_empty_string"},
            },
            "allOf": [
                {
                    "if": {
                        "properties": {"ground_truth_known": {"const": True}},
                        "required": ["ground_truth_known"],
                    },
                    "then": {"required": ["evidence"]},
                    "else": {"not": {"required": ["evidence"]}},
                }
            ],
        },
        "document": {
            "type": "object",
            "additionalProperties": False,
            "required": ["doc_id", "source", "split", "task_types", "spans"],
            "properties": {
                "doc_id": {"type": "string", "pattern": DOC_ID_PATTERN},
                "source": {"$ref": "#/$defs/source"},
                "split": {"enum": sorted(SUPPORTED_SPLIT_ROLES)},
                "task_types": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"enum": sorted(SUPPORTED_TASK_TYPES)},
                },
                "spans": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/span"},
                },
                "text_path": {"$ref": "#/$defs/non_empty_string"},
                "author_label": {"$ref": "#/$defs/non_empty_string"},
                "document_label": {"$ref": "#/$defs/non_empty_string"},
                "work": {"$ref": "#/$defs/non_empty_string"},
                "edition": {"$ref": "#/$defs/non_empty_string"},
                "period": {"$ref": "#/$defs/non_empty_string"},
                "genre": {"$ref": "#/$defs/non_empty_string"},
                "topic": {"$ref": "#/$defs/non_empty_string"},
                "register": {"$ref": "#/$defs/non_empty_string"},
            },
            "allOf": [
                {
                    "if": {
                        "properties": {"split": {"const": "blind"}},
                        "required": ["split"],
                    },
                    "then": {
                        "not": {
                            "anyOf": [
                                {"required": ["author_label"]},
                                {"required": ["document_label"]},
                                {"required": ["work"]},
                                {"required": ["edition"]},
                                {"required": ["period"]},
                                {"required": ["genre"]},
                                {"required": ["topic"]},
                                {"required": ["register"]},
                            ]
                        },
                        "properties": {"spans": {"maxItems": 0}},
                    },
                }
            ],
        },
    },
}


__all__ = [
    "BenchmarkDocument",
    "BenchmarkManifest",
    "BenchmarkSpan",
    "DatasetMetadata",
    "DOC_ID_PATTERN",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "SHA256_PATTERN",
    "SUPPORTED_SPLIT_ROLES",
    "SUPPORTED_OFFSET_UNITS",
    "SUPPORTED_TOKENIZERS",
    "SUPPORTED_TASK_TYPES",
    "SourceProvenance",
]
