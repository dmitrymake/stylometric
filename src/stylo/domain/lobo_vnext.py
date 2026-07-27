"""Strict, path-independent domain contracts for exploratory LOBO-vNext.

This module deliberately contains no estimator code.  It freezes and validates
the complete data/content/split/spec boundary before a representation cache or a
model factory may be created.  The historical LOBO artifacts and their readers
remain separate compatibility surfaces.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ..jsonio import StrictJSONError, dumps_strict, load_strict, loads_strict

CORPUS_VNEXT_SCHEMA_VERSION = "stylo.lobo-vnext.corpus-manifest.v1"
CONTENT_COMPONENT_SCHEMA_VERSION = "stylo.lobo-vnext.content-components.v1"
FOLD_SPEC_SCHEMA_VERSION = "stylo.lobo-vnext.fold-spec.v1"
INNER_SPLIT_SCHEMA_VERSION = "stylo.lobo-vnext.inner-split.v1"
INNER_FOLD_PLAN_SCHEMA_VERSION = "stylo.lobo-vnext.inner-fold-plan.v1"
INNER_CV_PLAN_SCHEMA_VERSION = "stylo.lobo-vnext.inner-cv-plan.v1"
FOLD_MANIFEST_SCHEMA_VERSION = "stylo.lobo-vnext.fold-manifest.v1"
MODEL_SPEC_SCHEMA_VERSION = "stylo.lobo-vnext.model-spec.v1"
INFERENCE_SPEC_SCHEMA_VERSION = "stylo.lobo-vnext.inference-spec.v1"
LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION = (
    "stylo.lobo-vnext.content-candidates.literal-bytes.v1"
)

CORPUS_KINDS = frozenset({"synthetic_fixture", "real_corpus"})
WORK_KINDS = frozenset({"work", "edition", "collection", "excerpt"})
CONTENT_EDGE_TYPES = frozenset(
    {
        "exact_duplicate",
        "edition_of",
        "contains",
        "excerpt_of",
        "collection_member",
        "manual",
    }
)
CONTENT_CANDIDATE_ORIGINS = frozenset({"automatic", "manual"})
CONTENT_DISPOSITIONS = frozenset(
    {"same_component", "separate_components", "unresolved"}
)
FOLD_MODES = frozenset({"isolated", "purged"})
_HEX64 = frozenset("0123456789abcdef")


class VNextContractError(ValueError):
    """A LOBO-vNext input is malformed, noncanonical, or scientifically unsafe."""


def canonical_json_bytes(value: object) -> bytes:
    """Return exact canonical strict-JSON bytes without type coercion."""

    _validate_json_value(value, "$")
    try:
        return dumps_strict(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VNextContractError(f"value is not canonical strict JSON: {exc}") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_json_value(value: object, path: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise VNextContractError(f"{path} must be finite")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise VNextContractError(f"{path} has a non-string object key")
            _validate_json_value(child, f"{path}.{key}")
        return
    raise VNextContractError(
        f"{path} has unsupported exact JSON type {type(value).__name__}"
    )


def _exact_object(
    value: object,
    keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise VNextContractError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != set(keys):
        missing = sorted(set(keys) - actual)
        extra = sorted(actual - set(keys))
        raise VNextContractError(
            f"{label} keys must be exact; missing={missing}, extra={extra}"
        )
    return value


def _exact_list(value: object, label: str, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = " non-empty" if nonempty else ""
        raise VNextContractError(f"{label} must be an exact{qualifier} array")
    return value


def _exact_str(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise VNextContractError(f"{label} must be an exact non-empty string")
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise VNextContractError(f"{label} must be an exact boolean")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise VNextContractError(f"{label} must be an exact integer{suffix}")
    return value


def _sha256(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if len(text) != 64 or any(char not in _HEX64 for char in text):
        raise VNextContractError(f"{label} must be 64 lowercase hex characters")
    return text


def _relative_path(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise VNextContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or text in (".", "..")
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise VNextContractError(f"{label} is not a canonical relative path")
    return text


def _ordered_unique_strings(
    value: object,
    label: str,
    *,
    nonempty: bool = True,
) -> tuple[str, ...]:
    rows = _exact_list(value, label, nonempty=nonempty)
    normalized = tuple(_exact_str(item, f"{label}[]") for item in rows)
    if len(set(normalized)) != len(normalized):
        raise VNextContractError(f"{label} contains duplicates")
    return normalized


def _sorted_unique_strings(
    value: object,
    label: str,
    *,
    nonempty: bool = True,
) -> tuple[str, ...]:
    normalized = _ordered_unique_strings(value, label, nonempty=nonempty)
    if normalized != tuple(sorted(normalized)):
        raise VNextContractError(f"{label} must be sorted")
    return normalized


def _payload_self_hash(payload: Mapping[str, object]) -> str:
    return canonical_sha256(dict(payload))


def _check_self_hash(raw: dict[str, Any], label: str) -> dict[str, Any]:
    recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    if recorded != _payload_self_hash(payload):
        raise VNextContractError(f"{label} self_hash mismatch")
    return payload


def _strict_raw(text: str, label: str) -> object:
    try:
        return loads_strict(text)
    except (StrictJSONError, TypeError) as exc:
        raise VNextContractError(f"{label}: {exc}") from exc


def _strict_file(path: str | os.PathLike[str], label: str) -> object:
    try:
        return load_strict(path)
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise VNextContractError(f"{label}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class RawInventoryEntry:
    relative_path: str
    byte_size: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "RawInventoryEntry":
        raw = _exact_object(
            value,
            {"relative_path", "byte_size", "sha256"},
            "raw inventory entry",
        )
        return cls(
            _relative_path(raw["relative_path"], "raw_inventory[].relative_path"),
            _exact_int(raw["byte_size"], "raw_inventory[].byte_size", minimum=0),
            _sha256(raw["sha256"], "raw_inventory[].sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclasses.dataclass(frozen=True)
class WorkIdentity:
    work_id: str
    author_id: str
    edition_id: str
    source_id: str
    work_kind: str
    raw_paths: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> "WorkIdentity":
        raw = _exact_object(
            value,
            {
                "work_id",
                "author_id",
                "edition_id",
                "source_id",
                "work_kind",
                "raw_paths",
            },
            "work identity",
        )
        kind = _exact_str(raw["work_kind"], "works[].work_kind")
        if kind not in WORK_KINDS:
            raise VNextContractError(f"unsupported works[].work_kind {kind!r}")
        paths = tuple(
            _relative_path(item, "works[].raw_paths[]")
            for item in _exact_list(
                raw["raw_paths"], "works[].raw_paths", nonempty=True
            )
        )
        if paths != tuple(sorted(set(paths))):
            raise VNextContractError(
                "works[].raw_paths must be sorted and duplicate-free"
            )
        return cls(
            _exact_str(raw["work_id"], "works[].work_id"),
            _exact_str(raw["author_id"], "works[].author_id"),
            _exact_str(raw["edition_id"], "works[].edition_id"),
            _exact_str(raw["source_id"], "works[].source_id"),
            kind,
            paths,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "work_id": self.work_id,
            "author_id": self.author_id,
            "edition_id": self.edition_id,
            "source_id": self.source_id,
            "work_kind": self.work_kind,
            "raw_paths": list(self.raw_paths),
        }


@dataclasses.dataclass(frozen=True)
class ContentCandidate:
    candidate_id: str
    left_work_id: str
    right_work_id: str
    edge_type: str
    origin: str
    disposition: str
    evidence_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "ContentCandidate":
        raw = _exact_object(
            value,
            {
                "candidate_id",
                "left_work_id",
                "right_work_id",
                "edge_type",
                "origin",
                "disposition",
                "evidence_sha256",
            },
            "content candidate",
        )
        edge_type = _exact_str(raw["edge_type"], "candidates[].edge_type")
        origin = _exact_str(raw["origin"], "candidates[].origin")
        disposition = _exact_str(
            raw["disposition"], "candidates[].disposition"
        )
        if edge_type not in CONTENT_EDGE_TYPES:
            raise VNextContractError(f"unsupported content edge type {edge_type!r}")
        if origin not in CONTENT_CANDIDATE_ORIGINS:
            raise VNextContractError(f"unsupported candidate origin {origin!r}")
        if disposition not in CONTENT_DISPOSITIONS:
            raise VNextContractError(
                f"unsupported candidate disposition {disposition!r}"
            )
        left = _exact_str(raw["left_work_id"], "candidates[].left_work_id")
        right = _exact_str(raw["right_work_id"], "candidates[].right_work_id")
        if left == right:
            raise VNextContractError("content candidate cannot be a self-edge")
        if edge_type == "manual" and origin != "manual":
            raise VNextContractError("manual content edge must have manual origin")
        return cls(
            _exact_str(raw["candidate_id"], "candidates[].candidate_id"),
            left,
            right,
            edge_type,
            origin,
            disposition,
            _sha256(raw["evidence_sha256"], "candidates[].evidence_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "left_work_id": self.left_work_id,
            "right_work_id": self.right_work_id,
            "edge_type": self.edge_type,
            "origin": self.origin,
            "disposition": self.disposition,
            "evidence_sha256": self.evidence_sha256,
        }

    def automatic_identity_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "left_work_id": self.left_work_id,
            "right_work_id": self.right_work_id,
            "edge_type": self.edge_type,
            "evidence_sha256": self.evidence_sha256,
        }


# A content candidate is the strict edge record carried by the component graph.
ContentEdge = ContentCandidate


@dataclasses.dataclass(frozen=True)
class ContentComponent:
    component_id: str
    work_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> "ContentComponent":
        raw = _exact_object(value, {"component_id", "work_ids"}, "content component")
        return cls(
            _exact_str(raw["component_id"], "components[].component_id"),
            _sorted_unique_strings(raw["work_ids"], "components[].work_ids"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"component_id": self.component_id, "work_ids": list(self.work_ids)}


def automatic_candidates_digest(
    candidates: Sequence[ContentCandidate],
) -> str:
    automatic = [
        candidate.automatic_identity_dict()
        for candidate in candidates
        if candidate.origin == "automatic"
    ]
    automatic.sort(key=lambda item: item["candidate_id"])
    return canonical_sha256(automatic)


@dataclasses.dataclass(frozen=True)
class ContentComponentManifest:
    schema_version: str
    automatic_candidate_policy_version: str
    work_ids: tuple[str, ...]
    components: tuple[ContentComponent, ...]
    candidates: tuple[ContentCandidate, ...]
    automatic_candidates_digest: str
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        automatic_candidate_policy_version: str,
        works: Sequence[WorkIdentity],
        components: Sequence[ContentComponent],
        candidates: Sequence[ContentCandidate],
    ) -> "ContentComponentManifest":
        payload = {
            "schema_version": CONTENT_COMPONENT_SCHEMA_VERSION,
            "automatic_candidate_policy_version": automatic_candidate_policy_version,
            "work_ids": sorted(work.work_id for work in works),
            "components": [
                component.to_dict()
                for component in sorted(components, key=lambda item: item.component_id)
            ],
            "candidates": [
                candidate.to_dict()
                for candidate in sorted(candidates, key=lambda item: item.candidate_id)
            ],
            "automatic_candidates_digest": automatic_candidates_digest(candidates),
        }
        return cls._from_payload(
            payload,
            _payload_self_hash(payload),
            works=works,
        )

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        works: Sequence[WorkIdentity] | None = None,
    ) -> "ContentComponentManifest":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "automatic_candidate_policy_version",
                "work_ids",
                "components",
                "candidates",
                "automatic_candidates_digest",
                "self_hash",
            },
            "content component manifest",
        )
        payload = _check_self_hash(raw, "content component manifest")
        return cls._from_payload(payload, raw["self_hash"], works=works)

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
        *,
        works: Sequence[WorkIdentity] | None,
    ) -> "ContentComponentManifest":
        if payload["schema_version"] != CONTENT_COMPONENT_SCHEMA_VERSION:
            raise VNextContractError(
                "content component manifest is legacy or unversioned"
            )
        policy = _exact_str(
            payload["automatic_candidate_policy_version"],
            "automatic_candidate_policy_version",
        )
        work_ids = _sorted_unique_strings(payload["work_ids"], "work_ids")
        components = tuple(
            ContentComponent.from_dict(item)
            for item in _exact_list(payload["components"], "components", nonempty=True)
        )
        candidates = tuple(
            ContentCandidate.from_dict(item)
            for item in _exact_list(payload["candidates"], "candidates")
        )
        if tuple(component.component_id for component in components) != tuple(
            sorted({component.component_id for component in components})
        ):
            raise VNextContractError(
                "components must have sorted unique component ids"
            )
        if tuple(candidate.candidate_id for candidate in candidates) != tuple(
            sorted({candidate.candidate_id for candidate in candidates})
        ):
            raise VNextContractError(
                "candidates must have sorted unique candidate ids"
            )
        recorded_automatic = _sha256(
            payload["automatic_candidates_digest"],
            "automatic_candidates_digest",
        )
        if recorded_automatic != automatic_candidates_digest(candidates):
            raise VNextContractError("automatic candidate digest mismatch")
        manifest = cls(
            CONTENT_COMPONENT_SCHEMA_VERSION,
            policy,
            work_ids,
            components,
            candidates,
            recorded_automatic,
            _sha256(self_hash, "content component manifest self_hash"),
        )
        manifest.validate(works=works)
        return manifest

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "automatic_candidate_policy_version": self.automatic_candidate_policy_version,
            "work_ids": list(self.work_ids),
            "components": [component.to_dict() for component in self.components],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "automatic_candidates_digest": self.automatic_candidates_digest,
        }

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    def validate(
        self,
        *,
        works: Sequence[WorkIdentity] | None = None,
        recomputed_automatic_candidates: Sequence[ContentCandidate] | None = None,
    ) -> "ContentComponentManifest":
        if type(self) is not ContentComponentManifest:
            raise VNextContractError(
                "content manifest must be exactly ContentComponentManifest"
            )
        if self.schema_version != CONTENT_COMPONENT_SCHEMA_VERSION:
            raise VNextContractError("content component schema mismatch")
        if self.self_hash != _payload_self_hash(self._payload()):
            raise VNextContractError("content component manifest self_hash mismatch")
        assigned: list[str] = []
        component_by_work: dict[str, str] = {}
        for component in self.components:
            for work_id in component.work_ids:
                if work_id in component_by_work:
                    raise VNextContractError(
                        f"work {work_id!r} belongs to more than one component"
                    )
                component_by_work[work_id] = component.component_id
                assigned.append(work_id)
        if tuple(sorted(assigned)) != self.work_ids:
            raise VNextContractError(
                "every included work must belong to exactly one component"
            )
        if any(
            candidate.left_work_id not in component_by_work
            or candidate.right_work_id not in component_by_work
            for candidate in self.candidates
        ):
            raise VNextContractError("content candidate references an unknown work")
        for candidate in self.candidates:
            if candidate.disposition == "unresolved":
                raise VNextContractError(
                    f"unresolved content candidate blocks run: {candidate.candidate_id}"
                )
            same = (
                component_by_work[candidate.left_work_id]
                == component_by_work[candidate.right_work_id]
            )
            if same != (candidate.disposition == "same_component"):
                raise VNextContractError(
                    f"candidate disposition conflicts with components: "
                    f"{candidate.candidate_id}"
                )
            if (
                candidate.edge_type != "manual"
                and candidate.disposition != "same_component"
            ):
                raise VNextContractError(
                    f"resolved {candidate.edge_type} relation must share a component"
                )
        # A multi-work component cannot be asserted without a connected chain of
        # resolved relations: otherwise supposedly isolated works could be hidden.
        adjacency: dict[str, set[str]] = defaultdict(set)
        for candidate in self.candidates:
            if candidate.disposition == "same_component":
                adjacency[candidate.left_work_id].add(candidate.right_work_id)
                adjacency[candidate.right_work_id].add(candidate.left_work_id)
        for component in self.components:
            if len(component.work_ids) == 1:
                continue
            reached = {component.work_ids[0]}
            frontier = [component.work_ids[0]]
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in reached:
                        reached.add(neighbor)
                        frontier.append(neighbor)
            if reached != set(component.work_ids):
                raise VNextContractError(
                    f"component {component.component_id!r} is not relation-connected"
                )
        if works is not None:
            self._validate_work_relations(works)
        if recomputed_automatic_candidates is not None:
            self.assert_automatic_candidates_match(recomputed_automatic_candidates)
        return self

    def _validate_work_relations(
        self,
        works: Sequence[WorkIdentity],
    ) -> None:
        by_id = {work.work_id: work for work in works}
        if len(by_id) != len(works) or tuple(sorted(by_id)) != self.work_ids:
            raise VNextContractError(
                "content manifest work set differs from corpus work identities"
            )
        for candidate in self.candidates:
            left = by_id[candidate.left_work_id]
            right = by_id[candidate.right_work_id]
            if candidate.edge_type == "edition_of":
                if (
                    left.author_id != right.author_id
                    or left.edition_id == right.edition_id
                    or left.work_kind == "collection"
                    or right.work_kind == "collection"
                ):
                    raise VNextContractError(
                        f"invalid edition_of relation {candidate.candidate_id!r}"
                    )
            elif candidate.edge_type == "collection_member":
                if (
                    right.work_kind != "collection"
                    or left.work_kind == "collection"
                ):
                    raise VNextContractError(
                        f"invalid collection_member relation "
                        f"{candidate.candidate_id!r}"
                    )
            elif candidate.edge_type == "excerpt_of":
                if left.work_kind != "excerpt" or right.work_kind == "excerpt":
                    raise VNextContractError(
                        f"invalid excerpt_of relation {candidate.candidate_id!r}"
                    )

    def assert_automatic_candidates_match(
        self,
        recomputed: Sequence[ContentCandidate],
    ) -> "ContentComponentManifest":
        if any(type(candidate) is not ContentCandidate for candidate in recomputed):
            raise VNextContractError(
                "recomputed candidates must be exact ContentCandidate records"
            )
        if any(candidate.origin != "automatic" for candidate in recomputed):
            raise VNextContractError(
                "recomputed candidate inventory may contain only automatic records"
            )
        if automatic_candidates_digest(recomputed) != self.automatic_candidates_digest:
            raise VNextContractError("automatic content candidates changed")
        expected = tuple(
            candidate.automatic_identity_dict()
            for candidate in self.candidates
            if candidate.origin == "automatic"
        )
        actual = tuple(
            candidate.automatic_identity_dict()
            for candidate in sorted(recomputed, key=lambda item: item.candidate_id)
        )
        if actual != expected:
            raise VNextContractError("automatic content candidate inventory mismatch")
        return self

    @property
    def component_by_work(self) -> dict[str, str]:
        return {
            work_id: component.component_id
            for component in self.components
            for work_id in component.work_ids
        }


@dataclasses.dataclass(frozen=True)
class CorpusVNextManifest:
    schema_version: str
    corpus_kind: str
    generation_id: str
    approved_for_exploratory: bool
    owner_selected: bool
    raw_inventory: tuple[RawInventoryEntry, ...]
    author_ids: tuple[str, ...]
    works: tuple[WorkIdentity, ...]
    canonical_model_row_digest: str
    chunker_policy_version: str
    canonicalizer_policy_version: str
    content_policy_version: str
    content_component_manifest_digest: str
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        corpus_kind: str,
        generation_id: str,
        approved_for_exploratory: bool,
        owner_selected: bool,
        raw_inventory: Sequence[RawInventoryEntry],
        author_ids: Sequence[str],
        works: Sequence[WorkIdentity],
        canonical_model_row_digest: str,
        chunker_policy_version: str,
        canonicalizer_policy_version: str,
        content_policy_version: str,
        content_component_manifest_digest: str,
    ) -> "CorpusVNextManifest":
        payload = {
            "schema_version": CORPUS_VNEXT_SCHEMA_VERSION,
            "corpus_kind": corpus_kind,
            "generation_id": generation_id,
            "approved_for_exploratory": approved_for_exploratory,
            "owner_selected": owner_selected,
            "raw_inventory": [
                entry.to_dict()
                for entry in sorted(
                    raw_inventory, key=lambda item: item.relative_path
                )
            ],
            "author_ids": list(author_ids),
            "works": [
                work.to_dict() for work in sorted(works, key=lambda item: item.work_id)
            ],
            "canonical_model_row_digest": canonical_model_row_digest,
            "chunker_policy_version": chunker_policy_version,
            "canonicalizer_policy_version": canonicalizer_policy_version,
            "content_policy_version": content_policy_version,
            "content_component_manifest_digest": content_component_manifest_digest,
        }
        return cls._from_payload(payload, _payload_self_hash(payload))

    @classmethod
    def from_dict(cls, value: object) -> "CorpusVNextManifest":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "corpus_kind",
                "generation_id",
                "approved_for_exploratory",
                "owner_selected",
                "raw_inventory",
                "author_ids",
                "works",
                "canonical_model_row_digest",
                "chunker_policy_version",
                "canonicalizer_policy_version",
                "content_policy_version",
                "content_component_manifest_digest",
                "self_hash",
            },
            "corpus vNext manifest",
        )
        payload = _check_self_hash(raw, "corpus vNext manifest")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "CorpusVNextManifest":
        if payload["schema_version"] != CORPUS_VNEXT_SCHEMA_VERSION:
            raise VNextContractError("corpus must be versioned vNext; legacy rejected")
        corpus_kind = _exact_str(payload["corpus_kind"], "corpus_kind")
        if corpus_kind not in CORPUS_KINDS:
            raise VNextContractError(f"unsupported corpus_kind {corpus_kind!r}")
        raw_inventory = tuple(
            RawInventoryEntry.from_dict(item)
            for item in _exact_list(
                payload["raw_inventory"], "raw_inventory", nonempty=True
            )
        )
        works = tuple(
            WorkIdentity.from_dict(item)
            for item in _exact_list(payload["works"], "works", nonempty=True)
        )
        manifest = cls(
            CORPUS_VNEXT_SCHEMA_VERSION,
            corpus_kind,
            _exact_str(payload["generation_id"], "generation_id"),
            _exact_bool(
                payload["approved_for_exploratory"], "approved_for_exploratory"
            ),
            _exact_bool(payload["owner_selected"], "owner_selected"),
            raw_inventory,
            _sorted_unique_strings(payload["author_ids"], "author_ids"),
            works,
            _sha256(
                payload["canonical_model_row_digest"],
                "canonical_model_row_digest",
            ),
            _exact_str(
                payload["chunker_policy_version"], "chunker_policy_version"
            ),
            _exact_str(
                payload["canonicalizer_policy_version"],
                "canonicalizer_policy_version",
            ),
            _exact_str(
                payload["content_policy_version"], "content_policy_version"
            ),
            _sha256(
                payload["content_component_manifest_digest"],
                "content_component_manifest_digest",
            ),
            _sha256(self_hash, "corpus vNext manifest self_hash"),
        )
        manifest.validate()
        return manifest

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_kind": self.corpus_kind,
            "generation_id": self.generation_id,
            "approved_for_exploratory": self.approved_for_exploratory,
            "owner_selected": self.owner_selected,
            "raw_inventory": [entry.to_dict() for entry in self.raw_inventory],
            "author_ids": list(self.author_ids),
            "works": [work.to_dict() for work in self.works],
            "canonical_model_row_digest": self.canonical_model_row_digest,
            "chunker_policy_version": self.chunker_policy_version,
            "canonicalizer_policy_version": self.canonicalizer_policy_version,
            "content_policy_version": self.content_policy_version,
            "content_component_manifest_digest": self.content_component_manifest_digest,
        }

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    def validate(
        self,
        *,
        content_manifest: ContentComponentManifest | None = None,
    ) -> "CorpusVNextManifest":
        if type(self) is not CorpusVNextManifest:
            raise VNextContractError(
                "corpus manifest must be exactly CorpusVNextManifest"
            )
        if self.schema_version != CORPUS_VNEXT_SCHEMA_VERSION:
            raise VNextContractError("corpus vNext schema mismatch")
        if self.self_hash != _payload_self_hash(self._payload()):
            raise VNextContractError("corpus vNext self_hash mismatch")
        paths = tuple(entry.relative_path for entry in self.raw_inventory)
        if paths != tuple(sorted(set(paths))):
            raise VNextContractError(
                "raw inventory must be path-sorted and duplicate-free"
            )
        work_ids = tuple(work.work_id for work in self.works)
        if work_ids != tuple(sorted(set(work_ids))):
            raise VNextContractError("works must be sorted by unique work_id")
        authors = tuple(sorted({work.author_id for work in self.works}))
        if authors != self.author_ids:
            raise VNextContractError(
                "author_ids must exactly equal authors represented by works"
            )
        assigned_paths = [
            path for work in self.works for path in work.raw_paths
        ]
        if len(assigned_paths) != len(set(assigned_paths)):
            raise VNextContractError("a raw path is assigned to multiple works")
        if tuple(sorted(assigned_paths)) != paths:
            raise VNextContractError(
                "work raw_paths must exactly partition the raw inventory"
            )
        if content_manifest is not None:
            content_manifest.validate(works=self.works)
            if (
                self.content_component_manifest_digest
                != content_manifest.self_hash
            ):
                raise VNextContractError(
                    "corpus/content component manifest digest mismatch"
                )
        return self

    def assert_exploratory_authorized(
        self,
        *,
        synthetic_fixture: bool,
    ) -> "CorpusVNextManifest":
        expected_kind = "synthetic_fixture" if synthetic_fixture else "real_corpus"
        if self.corpus_kind != expected_kind:
            raise VNextContractError(
                f"corpus_kind must be {expected_kind!r} for this execution"
            )
        if not self.approved_for_exploratory:
            raise VNextContractError("corpus is not approved for exploratory use")
        if self.owner_selected is not (not synthetic_fixture):
            expected = "false" if synthetic_fixture else "true"
            raise VNextContractError(
                f"owner_selected must be {expected} for this execution"
            )
        return self


def inventory_raw_files(root: str | os.PathLike[str]) -> tuple[RawInventoryEntry, ...]:
    """Read a full literal-byte inventory, rejecting every symlink/special file."""

    base = Path(root)
    try:
        root_stat = base.lstat()
    except OSError as exc:
        raise VNextContractError(f"raw corpus root is unavailable: {base}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise VNextContractError("raw corpus root must be a real directory")
    entries: list[RawInventoryEntry] = []
    stack = [base]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise VNextContractError(
                f"cannot enumerate raw corpus directory {directory}"
            ) from exc
        for child in children:
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise VNextContractError(
                    f"cannot stat raw corpus entry {child.path}"
                ) from exc
            if stat.S_ISLNK(child_stat.st_mode):
                raise VNextContractError(f"symlink rejected in raw corpus: {child.path}")
            if stat.S_ISDIR(child_stat.st_mode):
                stack.append(Path(child.path))
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise VNextContractError(
                    f"non-regular raw corpus entry rejected: {child.path}"
                )
            path = Path(child.path)
            relative = path.relative_to(base).as_posix()
            digest = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError as exc:
                raise VNextContractError(f"cannot read raw file {relative!r}") from exc
            entries.append(
                RawInventoryEntry(relative, child_stat.st_size, digest.hexdigest())
            )
    entries.sort(key=lambda entry: entry.relative_path)
    if not entries:
        raise VNextContractError("raw corpus inventory must not be empty")
    return tuple(entries)


def verify_raw_inventory(
    root: str | os.PathLike[str],
    manifest: CorpusVNextManifest,
) -> CorpusVNextManifest:
    manifest.validate()
    current = inventory_raw_files(root)
    expected_by_path = {
        entry.relative_path: entry for entry in manifest.raw_inventory
    }
    current_by_path = {entry.relative_path: entry for entry in current}
    missing = sorted(set(expected_by_path) - set(current_by_path))
    extra = sorted(set(current_by_path) - set(expected_by_path))
    if missing or extra:
        raise VNextContractError(
            f"raw inventory path mismatch; missing={missing}, extra={extra}"
        )
    for relative_path in sorted(expected_by_path):
        expected = expected_by_path[relative_path]
        observed = current_by_path[relative_path]
        if expected.byte_size != observed.byte_size:
            raise VNextContractError(
                f"raw byte size mismatch for {relative_path!r}"
            )
        if expected.sha256 != observed.sha256:
            raise VNextContractError(f"raw SHA-256 mismatch for {relative_path!r}")
    return manifest


def discover_literal_byte_content_candidates(
    root: str | os.PathLike[str],
    manifest: CorpusVNextManifest,
) -> tuple[ContentCandidate, ...]:
    """Recompute exact duplicate/containment candidates from bound literal bytes.

    A work's literal stream is the concatenation, in its already-frozen
    ``raw_paths`` order, of each file's exact bytes.  This deliberately performs
    no Unicode, orthographic, OCR, markup, or fuzzy normalization.
    """

    verify_raw_inventory(root, manifest)
    base = Path(root)
    work_bytes: dict[str, bytes] = {}
    work_receipts: dict[str, list[dict[str, str]]] = {}
    for work in manifest.works:
        blocks: list[bytes] = []
        receipts: list[dict[str, str]] = []
        for relative_path in work.raw_paths:
            path = base.joinpath(*PurePosixPath(relative_path).parts)
            try:
                literal = path.read_bytes()
            except OSError as exc:
                raise VNextContractError(
                    f"cannot recompute content candidate for {relative_path!r}"
                ) from exc
            blocks.append(literal)
            receipts.append(
                {
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(literal).hexdigest(),
                }
            )
        combined = b"".join(blocks)
        if not combined:
            raise VNextContractError(
                f"literal content for work {work.work_id!r} is empty"
            )
        work_bytes[work.work_id] = combined
        work_receipts[work.work_id] = receipts

    discovered: list[ContentCandidate] = []
    ordered_work_ids = sorted(work_bytes)
    for left_index, first_id in enumerate(ordered_work_ids):
        first = work_bytes[first_id]
        for second_id in ordered_work_ids[left_index + 1 :]:
            second = work_bytes[second_id]
            if first == second:
                edge_type = "exact_duplicate"
                left_id, right_id = first_id, second_id
            elif len(first) < len(second) and first in second:
                edge_type = "contains"
                left_id, right_id = first_id, second_id
            elif len(second) < len(first) and second in first:
                edge_type = "contains"
                left_id, right_id = second_id, first_id
            else:
                continue
            evidence = {
                "policy_version": LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION,
                "edge_type": edge_type,
                "left_work_id": left_id,
                "right_work_id": right_id,
                "left_raw_receipts": work_receipts[left_id],
                "right_raw_receipts": work_receipts[right_id],
            }
            evidence_sha256 = canonical_sha256(evidence)
            discovered.append(
                ContentCandidate(
                    candidate_id=f"automatic-{evidence_sha256}",
                    left_work_id=left_id,
                    right_work_id=right_id,
                    edge_type=edge_type,
                    origin="automatic",
                    disposition="same_component",
                    evidence_sha256=evidence_sha256,
                )
            )
    return tuple(sorted(discovered, key=lambda item: item.candidate_id))


def recompute_automatic_content_candidates(
    root: str | os.PathLike[str],
    corpus_manifest: CorpusVNextManifest,
    content_manifest: ContentComponentManifest,
) -> tuple[ContentCandidate, ...]:
    """Recompute and bind automatic candidates before any learned operation."""

    corpus_manifest.validate(content_manifest=content_manifest)
    if (
        content_manifest.automatic_candidate_policy_version
        != LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
    ):
        raise VNextContractError(
            "unsupported automatic candidate policy; no implicit fallback"
        )
    candidates = discover_literal_byte_content_candidates(root, corpus_manifest)
    content_manifest.assert_automatic_candidates_match(candidates)
    return candidates


def build_corpus_vnext_manifest(
    root: str | os.PathLike[str],
    *,
    corpus_kind: str,
    generation_id: str,
    approved_for_exploratory: bool,
    owner_selected: bool,
    author_ids: Sequence[str],
    works: Sequence[WorkIdentity],
    canonical_model_row_digest: str,
    chunker_policy_version: str,
    canonicalizer_policy_version: str,
    content_policy_version: str,
    content_component_manifest_digest: str,
) -> CorpusVNextManifest:
    return CorpusVNextManifest.build(
        corpus_kind=corpus_kind,
        generation_id=generation_id,
        approved_for_exploratory=approved_for_exploratory,
        owner_selected=owner_selected,
        raw_inventory=inventory_raw_files(root),
        author_ids=author_ids,
        works=works,
        canonical_model_row_digest=canonical_model_row_digest,
        chunker_policy_version=chunker_policy_version,
        canonicalizer_policy_version=canonicalizer_policy_version,
        content_policy_version=content_policy_version,
        content_component_manifest_digest=content_component_manifest_digest,
    )


def _freeze_json_mapping(
    value: Mapping[str, object],
    label: str,
) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping):
        raise VNextContractError(f"{label} must be an object")
    plain = dict(value)
    if any(type(key) is not str or not key for key in plain):
        raise VNextContractError(f"{label} keys must be non-empty exact strings")
    _validate_json_value(plain, label)
    # Round-trip creates an immutable-by-convention detached tree.  The outer
    # tuple prevents field replacement; to_dict always creates another tree.
    detached = json.loads(canonical_json_bytes(plain).decode("utf-8"))
    return tuple((key, detached[key]) for key in sorted(detached))


def _thaw_json_mapping(value: tuple[tuple[str, object], ...]) -> dict[str, object]:
    return json.loads(
        canonical_json_bytes({key: child for key, child in value}).decode("utf-8")
    )


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    schema_version: str
    model_id: str
    family: str
    features: tuple[str, ...]
    weighting: str
    hyperparameters: tuple[tuple[str, object], ...]
    seeds: tuple[tuple[str, int], ...]
    requires_inner_cv: bool
    inner_cv_splits: int | None
    supports_component_aware_inner_cv: bool
    approved_for_exploratory: bool
    owner_selected: bool
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        model_id: str,
        family: str,
        features: Sequence[str],
        weighting: str,
        hyperparameters: Mapping[str, object],
        seeds: Mapping[str, int],
        requires_inner_cv: bool,
        inner_cv_splits: int | None,
        supports_component_aware_inner_cv: bool,
        approved_for_exploratory: bool,
        owner_selected: bool,
    ) -> "ModelSpec":
        payload = {
            "schema_version": MODEL_SPEC_SCHEMA_VERSION,
            "model_id": model_id,
            "family": family,
            "features": list(features),
            "weighting": weighting,
            "hyperparameters": dict(hyperparameters),
            "seeds": dict(seeds),
            "requires_inner_cv": requires_inner_cv,
            "inner_cv_splits": inner_cv_splits,
            "supports_component_aware_inner_cv": supports_component_aware_inner_cv,
            "approved_for_exploratory": approved_for_exploratory,
            "owner_selected": owner_selected,
        }
        return cls._from_payload(payload, _payload_self_hash(payload))

    @classmethod
    def from_dict(cls, value: object) -> "ModelSpec":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "model_id",
                "family",
                "features",
                "weighting",
                "hyperparameters",
                "seeds",
                "requires_inner_cv",
                "inner_cv_splits",
                "supports_component_aware_inner_cv",
                "approved_for_exploratory",
                "owner_selected",
                "self_hash",
            },
            "model spec",
        )
        payload = _check_self_hash(raw, "model spec")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "ModelSpec":
        if payload["schema_version"] != MODEL_SPEC_SCHEMA_VERSION:
            raise VNextContractError("model spec is legacy or unversioned")
        features = _ordered_unique_strings(payload["features"], "features")
        hyperparameters_raw = payload["hyperparameters"]
        if type(hyperparameters_raw) is not dict:
            raise VNextContractError("hyperparameters must be an exact object")
        seeds_raw = payload["seeds"]
        if type(seeds_raw) is not dict or not seeds_raw:
            raise VNextContractError("seeds must be an exact non-empty object")
        seeds: dict[str, int] = {}
        for key, value in seeds_raw.items():
            name = _exact_str(key, "seeds key")
            seeds[name] = _exact_int(value, f"seeds.{name}", minimum=0)
        requires_inner = _exact_bool(
            payload["requires_inner_cv"], "requires_inner_cv"
        )
        raw_splits = payload["inner_cv_splits"]
        if requires_inner:
            splits: int | None = _exact_int(
                raw_splits, "inner_cv_splits", minimum=2
            )
        else:
            if raw_splits is not None:
                raise VNextContractError(
                    "inner_cv_splits must be null when inner CV is disabled"
                )
            splits = None
        spec = cls(
            MODEL_SPEC_SCHEMA_VERSION,
            _exact_str(payload["model_id"], "model_id"),
            _exact_str(payload["family"], "family"),
            features,
            _exact_str(payload["weighting"], "weighting"),
            _freeze_json_mapping(hyperparameters_raw, "hyperparameters"),
            tuple(sorted(seeds.items())),
            requires_inner,
            splits,
            _exact_bool(
                payload["supports_component_aware_inner_cv"],
                "supports_component_aware_inner_cv",
            ),
            _exact_bool(
                payload["approved_for_exploratory"],
                "approved_for_exploratory",
            ),
            _exact_bool(payload["owner_selected"], "owner_selected"),
            _sha256(self_hash, "model spec self_hash"),
        )
        spec.validate()
        return spec

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "family": self.family,
            "features": list(self.features),
            "weighting": self.weighting,
            "hyperparameters": _thaw_json_mapping(self.hyperparameters),
            "seeds": dict(self.seeds),
            "requires_inner_cv": self.requires_inner_cv,
            "inner_cv_splits": self.inner_cv_splits,
            "supports_component_aware_inner_cv": self.supports_component_aware_inner_cv,
            "approved_for_exploratory": self.approved_for_exploratory,
            "owner_selected": self.owner_selected,
        }

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    def validate(self) -> "ModelSpec":
        if type(self) is not ModelSpec:
            raise VNextContractError("model spec must be exactly ModelSpec")
        if self.schema_version != MODEL_SPEC_SCHEMA_VERSION:
            raise VNextContractError("model spec schema mismatch")
        if self.self_hash != _payload_self_hash(self._payload()):
            raise VNextContractError("model spec self_hash mismatch")
        return self

    def assert_exploratory_authorized(
        self,
        *,
        synthetic_fixture: bool,
    ) -> "ModelSpec":
        if not self.approved_for_exploratory:
            raise VNextContractError("model spec is not approved for exploratory use")
        if self.owner_selected is not (not synthetic_fixture):
            expected = "false" if synthetic_fixture else "true"
            raise VNextContractError(
                f"model owner_selected must be {expected} for this execution"
            )
        return self


@dataclasses.dataclass(frozen=True)
class InferenceSpec:
    schema_version: str
    primary_metric: str
    primary_uncertainty: str
    secondary_metrics: tuple[str, ...]
    macro_f1_uncertainty: str
    bootstrap_seed: int
    bootstrap_iterations: int
    confidence_level: float
    approved_for_exploratory: bool
    owner_selected: bool
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        primary_metric: str,
        primary_uncertainty: str,
        secondary_metrics: Sequence[str],
        macro_f1_uncertainty: str,
        bootstrap_seed: int,
        bootstrap_iterations: int,
        confidence_level: float,
        approved_for_exploratory: bool,
        owner_selected: bool,
    ) -> "InferenceSpec":
        payload = {
            "schema_version": INFERENCE_SPEC_SCHEMA_VERSION,
            "primary_metric": primary_metric,
            "primary_uncertainty": primary_uncertainty,
            "secondary_metrics": list(secondary_metrics),
            "macro_f1_uncertainty": macro_f1_uncertainty,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_iterations": bootstrap_iterations,
            "confidence_level": confidence_level,
            "approved_for_exploratory": approved_for_exploratory,
            "owner_selected": owner_selected,
        }
        return cls._from_payload(payload, _payload_self_hash(payload))

    @classmethod
    def from_dict(cls, value: object) -> "InferenceSpec":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "primary_metric",
                "primary_uncertainty",
                "secondary_metrics",
                "macro_f1_uncertainty",
                "bootstrap_seed",
                "bootstrap_iterations",
                "confidence_level",
                "approved_for_exploratory",
                "owner_selected",
                "self_hash",
            },
            "inference spec",
        )
        payload = _check_self_hash(raw, "inference spec")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "InferenceSpec":
        if payload["schema_version"] != INFERENCE_SPEC_SCHEMA_VERSION:
            raise VNextContractError("inference spec is legacy or unversioned")
        confidence = payload["confidence_level"]
        if (
            type(confidence) is not float
            or not math.isfinite(confidence)
            or not 0.0 < confidence < 1.0
        ):
            raise VNextContractError(
                "confidence_level must be an exact finite float in (0, 1)"
            )
        spec = cls(
            INFERENCE_SPEC_SCHEMA_VERSION,
            _exact_str(payload["primary_metric"], "primary_metric"),
            _exact_str(payload["primary_uncertainty"], "primary_uncertainty"),
            _ordered_unique_strings(
                payload["secondary_metrics"], "secondary_metrics"
            ),
            _exact_str(
                payload["macro_f1_uncertainty"], "macro_f1_uncertainty"
            ),
            _exact_int(payload["bootstrap_seed"], "bootstrap_seed", minimum=0),
            _exact_int(
                payload["bootstrap_iterations"],
                "bootstrap_iterations",
                minimum=1,
            ),
            confidence,
            _exact_bool(
                payload["approved_for_exploratory"],
                "approved_for_exploratory",
            ),
            _exact_bool(payload["owner_selected"], "owner_selected"),
            _sha256(self_hash, "inference spec self_hash"),
        )
        spec.validate()
        return spec

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "primary_metric": self.primary_metric,
            "primary_uncertainty": self.primary_uncertainty,
            "secondary_metrics": list(self.secondary_metrics),
            "macro_f1_uncertainty": self.macro_f1_uncertainty,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_iterations": self.bootstrap_iterations,
            "confidence_level": self.confidence_level,
            "approved_for_exploratory": self.approved_for_exploratory,
            "owner_selected": self.owner_selected,
        }

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    def validate(self) -> "InferenceSpec":
        if type(self) is not InferenceSpec:
            raise VNextContractError("inference spec must be exactly InferenceSpec")
        if self.schema_version != INFERENCE_SPEC_SCHEMA_VERSION:
            raise VNextContractError("inference spec schema mismatch")
        if self.primary_metric != "book_accuracy":
            raise VNextContractError("vNext primary metric must be book_accuracy")
        if self.primary_uncertainty != "author_clustered_percentile_bootstrap":
            raise VNextContractError(
                "vNext primary uncertainty must be author clustered"
            )
        if self.macro_f1_uncertainty != "point_only":
            raise VNextContractError("macro-F1 uncertainty must remain point_only")
        required_secondary = ("macro_f1", "top2", "per_author")
        if self.secondary_metrics != required_secondary:
            raise VNextContractError(
                f"secondary_metrics must be exactly {required_secondary!r}"
            )
        if self.self_hash != _payload_self_hash(self._payload()):
            raise VNextContractError("inference spec self_hash mismatch")
        return self

    def assert_exploratory_authorized(
        self,
        *,
        synthetic_fixture: bool,
    ) -> "InferenceSpec":
        if not self.approved_for_exploratory:
            raise VNextContractError(
                "inference spec is not approved for exploratory use"
            )
        if self.owner_selected is not (not synthetic_fixture):
            expected = "false" if synthetic_fixture else "true"
            raise VNextContractError(
                f"inference owner_selected must be {expected} for this execution"
            )
        return self


@dataclasses.dataclass(frozen=True)
class InnerSplitSpec:
    schema_version: str
    split_index: int
    train_work_ids: tuple[str, ...]
    validation_work_ids: tuple[str, ...]
    validation_component_ids: tuple[str, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        split_index: int,
        train_work_ids: Sequence[str],
        validation_work_ids: Sequence[str],
        validation_component_ids: Sequence[str],
    ) -> "InnerSplitSpec":
        payload = {
            "schema_version": INNER_SPLIT_SCHEMA_VERSION,
            "split_index": split_index,
            "train_work_ids": sorted(train_work_ids),
            "validation_work_ids": sorted(validation_work_ids),
            "validation_component_ids": sorted(validation_component_ids),
        }
        return cls._from_payload(payload, _payload_self_hash(payload))

    @classmethod
    def from_dict(cls, value: object) -> "InnerSplitSpec":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "split_index",
                "train_work_ids",
                "validation_work_ids",
                "validation_component_ids",
                "self_hash",
            },
            "inner split",
        )
        payload = _check_self_hash(raw, "inner split")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "InnerSplitSpec":
        if payload["schema_version"] != INNER_SPLIT_SCHEMA_VERSION:
            raise VNextContractError("inner split schema mismatch")
        split = cls(
            INNER_SPLIT_SCHEMA_VERSION,
            _exact_int(payload["split_index"], "split_index", minimum=0),
            _sorted_unique_strings(payload["train_work_ids"], "train_work_ids"),
            _sorted_unique_strings(
                payload["validation_work_ids"], "validation_work_ids"
            ),
            _sorted_unique_strings(
                payload["validation_component_ids"],
                "validation_component_ids",
            ),
            _sha256(self_hash, "inner split self_hash"),
        )
        if set(split.train_work_ids) & set(split.validation_work_ids):
            raise VNextContractError("inner train and validation works overlap")
        if split.self_hash != _payload_self_hash(split._payload()):
            raise VNextContractError("inner split self_hash mismatch")
        return split

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_index": self.split_index,
            "train_work_ids": list(self.train_work_ids),
            "validation_work_ids": list(self.validation_work_ids),
            "validation_component_ids": list(self.validation_component_ids),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "self_hash": self.self_hash}


def plan_component_aware_inner_splits(
    train_work_ids: Sequence[str],
    works: Sequence[WorkIdentity],
    content_manifest: ContentComponentManifest,
    *,
    n_splits: int,
    probability_class_order: Sequence[str],
) -> tuple[InnerSplitSpec, ...]:
    """Plan deterministic component-grouped, label-preserving inner splits."""

    if type(n_splits) is not int or n_splits < 2:
        raise VNextContractError("inner n_splits must be an exact integer >= 2")
    ordered_train = tuple(sorted(train_work_ids))
    if not ordered_train or len(set(ordered_train)) != len(ordered_train):
        raise VNextContractError("inner train work ids must be unique and nonempty")
    by_work = {work.work_id: work for work in works}
    if any(work_id not in by_work for work_id in ordered_train):
        raise VNextContractError("inner train references an unknown work")
    classes = tuple(probability_class_order)
    if (
        not classes
        or len(set(classes)) != len(classes)
        or any(type(author) is not str or not author for author in classes)
    ):
        raise VNextContractError("probability_class_order is malformed")
    component_by_work = content_manifest.component_by_work
    component_works: dict[str, list[str]] = defaultdict(list)
    for work_id in ordered_train:
        component_works[component_by_work[work_id]].append(work_id)
    author_component_support: dict[str, set[str]] = defaultdict(set)
    for component_id, member_ids in component_works.items():
        for work_id in member_ids:
            author_component_support[by_work[work_id].author_id].add(component_id)
    for author in classes:
        count = len(author_component_support.get(author, ()))
        if count < n_splits:
            raise VNextContractError(
                f"inner CV infeasible: author {author!r} has {count} "
                f"train components for {n_splits} splits"
            )

    authors_by_component = {
        component_id: frozenset(by_work[work_id].author_id for work_id in member_ids)
        for component_id, member_ids in component_works.items()
    }
    rarity = {
        author: len(author_component_support[author])
        for author in classes
    }
    ordered_components = sorted(
        component_works,
        key=lambda component_id: (
            min(rarity[author] for author in authors_by_component[component_id]),
            -len(authors_by_component[component_id]),
            -len(component_works[component_id]),
            component_id,
        ),
    )
    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    author_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(n_splits)]
    work_counts = [0] * n_splits
    for component_id in ordered_components:
        component_authors = authors_by_component[component_id]
        chosen = min(
            range(n_splits),
            key=lambda index: (
                sum(author_counts[index][author] for author in component_authors),
                work_counts[index],
                len(buckets[index]),
                index,
            ),
        )
        buckets[chosen].append(component_id)
        work_counts[chosen] += len(component_works[component_id])
        for author in component_authors:
            author_counts[chosen][author] += 1

    train_set = set(ordered_train)
    splits: list[InnerSplitSpec] = []
    for split_index, component_ids in enumerate(buckets):
        validation_work_ids = sorted(
            work_id
            for component_id in component_ids
            for work_id in component_works[component_id]
        )
        inner_train = sorted(train_set - set(validation_work_ids))
        if not validation_work_ids:
            raise VNextContractError("inner CV produced an empty validation split")
        validation_authors = {
            by_work[work_id].author_id for work_id in validation_work_ids
        }
        train_authors = {by_work[work_id].author_id for work_id in inner_train}
        if validation_authors != set(classes) or train_authors != set(classes):
            raise VNextContractError(
                "inner CV cannot preserve every probability class in both sides"
            )
        splits.append(
            InnerSplitSpec.build(
                split_index=split_index,
                train_work_ids=inner_train,
                validation_work_ids=validation_work_ids,
                validation_component_ids=sorted(component_ids),
            )
        )
    return tuple(splits)


@dataclasses.dataclass(frozen=True)
class FoldSpec:
    schema_version: str
    fold_id: str
    test_work_id: str
    content_component_id: str
    train_work_ids: tuple[str, ...]
    purged_work_ids: tuple[str, ...]
    probability_class_order: tuple[str, ...]
    metric_label_order: tuple[str, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        fold_id: str,
        test_work_id: str,
        content_component_id: str,
        train_work_ids: Sequence[str],
        purged_work_ids: Sequence[str],
        probability_class_order: Sequence[str],
        metric_label_order: Sequence[str],
    ) -> "FoldSpec":
        payload = {
            "schema_version": FOLD_SPEC_SCHEMA_VERSION,
            "fold_id": fold_id,
            "test_work_id": test_work_id,
            "content_component_id": content_component_id,
            "train_work_ids": sorted(train_work_ids),
            "purged_work_ids": sorted(purged_work_ids),
            "probability_class_order": list(probability_class_order),
            "metric_label_order": list(metric_label_order),
        }
        return cls._from_payload(payload, _payload_self_hash(payload))

    @classmethod
    def from_dict(cls, value: object) -> "FoldSpec":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "fold_id",
                "test_work_id",
                "content_component_id",
                "train_work_ids",
                "purged_work_ids",
                "probability_class_order",
                "metric_label_order",
                "self_hash",
            },
            "fold spec",
        )
        payload = _check_self_hash(raw, "fold spec")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "FoldSpec":
        if payload["schema_version"] != FOLD_SPEC_SCHEMA_VERSION:
            raise VNextContractError("fold spec schema mismatch")
        fold = cls(
            FOLD_SPEC_SCHEMA_VERSION,
            _exact_str(payload["fold_id"], "fold_id"),
            _exact_str(payload["test_work_id"], "test_work_id"),
            _exact_str(
                payload["content_component_id"], "content_component_id"
            ),
            _sorted_unique_strings(payload["train_work_ids"], "train_work_ids"),
            _sorted_unique_strings(
                payload["purged_work_ids"],
                "purged_work_ids",
                nonempty=False,
            ),
            _ordered_unique_strings(
                payload["probability_class_order"],
                "probability_class_order",
            ),
            _ordered_unique_strings(
                payload["metric_label_order"], "metric_label_order"
            ),
            _sha256(self_hash, "fold spec self_hash"),
        )
        if set(fold.metric_label_order) - set(fold.probability_class_order):
            raise VNextContractError(
                "metric_label_order must be a subset of probability_class_order"
            )
        expected_m = tuple(
            author
            for author in fold.probability_class_order
            if author in set(fold.metric_label_order)
        )
        if fold.metric_label_order != expected_m:
            raise VNextContractError("metric labels must preserve P order")
        if fold.test_work_id in fold.train_work_ids:
            raise VNextContractError("test work appears in outer train")
        if fold.test_work_id in fold.purged_work_ids:
            raise VNextContractError(
                "purged_work_ids contains test; purge lists only additional works"
            )
        if set(fold.train_work_ids) & set(fold.purged_work_ids):
            raise VNextContractError("outer train and purge sets overlap")
        if fold.self_hash != _payload_self_hash(fold._payload()):
            raise VNextContractError("fold spec self_hash mismatch")
        return fold

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "test_work_id": self.test_work_id,
            "content_component_id": self.content_component_id,
            "train_work_ids": list(self.train_work_ids),
            "purged_work_ids": list(self.purged_work_ids),
            "probability_class_order": list(self.probability_class_order),
            "metric_label_order": list(self.metric_label_order),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "self_hash": self.self_hash}


@dataclasses.dataclass(frozen=True)
class FoldManifest:
    schema_version: str
    mode: str
    corpus_manifest_digest: str
    content_component_manifest_digest: str
    probability_class_order: tuple[str, ...]
    metric_label_order: tuple[str, ...]
    folds: tuple[FoldSpec, ...]
    self_hash: str

    @classmethod
    def from_dict(cls, value: object) -> "FoldManifest":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "mode",
                "corpus_manifest_digest",
                "content_component_manifest_digest",
                "probability_class_order",
                "metric_label_order",
                "folds",
                "self_hash",
            },
            "fold manifest",
        )
        payload = _check_self_hash(raw, "fold manifest")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "FoldManifest":
        if payload["schema_version"] != FOLD_MANIFEST_SCHEMA_VERSION:
            raise VNextContractError("fold manifest is legacy or unversioned")
        mode = _exact_str(payload["mode"], "mode")
        if mode not in FOLD_MODES:
            raise VNextContractError(f"unsupported fold mode {mode!r}")
        probability = _ordered_unique_strings(
            payload["probability_class_order"], "probability_class_order"
        )
        metrics = _ordered_unique_strings(
            payload["metric_label_order"], "metric_label_order"
        )
        folds = tuple(
            FoldSpec.from_dict(item)
            for item in _exact_list(payload["folds"], "folds", nonempty=True)
        )
        manifest = cls(
            FOLD_MANIFEST_SCHEMA_VERSION,
            mode,
            _sha256(
                payload["corpus_manifest_digest"], "corpus_manifest_digest"
            ),
            _sha256(
                payload["content_component_manifest_digest"],
                "content_component_manifest_digest",
            ),
            probability,
            metrics,
            folds,
            _sha256(self_hash, "fold manifest self_hash"),
        )
        manifest.validate()
        return manifest

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "content_component_manifest_digest": self.content_component_manifest_digest,
            "probability_class_order": list(self.probability_class_order),
            "metric_label_order": list(self.metric_label_order),
            "folds": [fold.to_dict() for fold in self.folds],
        }

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "self_hash": self.self_hash}

    def validate(self) -> "FoldManifest":
        if type(self) is not FoldManifest:
            raise VNextContractError("fold manifest must be exactly FoldManifest")
        if self.schema_version != FOLD_MANIFEST_SCHEMA_VERSION:
            raise VNextContractError("fold manifest schema mismatch")
        if tuple(fold.fold_id for fold in self.folds) != tuple(
            sorted({fold.fold_id for fold in self.folds})
        ):
            raise VNextContractError("fold ids must be sorted and unique")
        if tuple(fold.test_work_id for fold in self.folds) != tuple(
            sorted({fold.test_work_id for fold in self.folds})
        ):
            raise VNextContractError("test works must be sorted and unique")
        if any(
            fold.probability_class_order != self.probability_class_order
            or fold.metric_label_order != self.metric_label_order
            for fold in self.folds
        ):
            raise VNextContractError("fold P/M orders differ from manifest")
        if self.self_hash != _payload_self_hash(self._payload()):
            raise VNextContractError("fold manifest self_hash mismatch")
        return self

    def validate_against(
        self,
        corpus_manifest: CorpusVNextManifest,
        content_manifest: ContentComponentManifest,
    ) -> "FoldManifest":
        rebuilt = build_fold_manifest(
            corpus_manifest,
            content_manifest,
            mode=self.mode,
        )
        if rebuilt != self:
            raise VNextContractError(
                "fold manifest differs from canonical corpus/content rebuild"
            )
        return self


@dataclasses.dataclass(frozen=True)
class InnerFoldPlan:
    schema_version: str
    fold_id: str
    fold_spec_digest: str
    splits: tuple[InnerSplitSpec, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        fold_id: str,
        fold_spec_digest: str,
        splits: Sequence[InnerSplitSpec],
    ) -> "InnerFoldPlan":
        payload = {
            "schema_version": INNER_FOLD_PLAN_SCHEMA_VERSION,
            "fold_id": fold_id,
            "fold_spec_digest": fold_spec_digest,
            "splits": [split.to_dict() for split in splits],
        }
        return cls._from_payload(payload, _payload_self_hash(payload))

    @classmethod
    def from_dict(cls, value: object) -> "InnerFoldPlan":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "fold_id",
                "fold_spec_digest",
                "splits",
                "self_hash",
            },
            "inner fold plan",
        )
        payload = _check_self_hash(raw, "inner fold plan")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "InnerFoldPlan":
        if payload["schema_version"] != INNER_FOLD_PLAN_SCHEMA_VERSION:
            raise VNextContractError("inner fold plan schema mismatch")
        plan = cls(
            INNER_FOLD_PLAN_SCHEMA_VERSION,
            _exact_str(payload["fold_id"], "inner fold plan fold_id"),
            _sha256(
                payload["fold_spec_digest"],
                "inner fold plan fold_spec_digest",
            ),
            tuple(
                InnerSplitSpec.from_dict(item)
                for item in _exact_list(payload["splits"], "inner fold plan splits")
            ),
            _sha256(self_hash, "inner fold plan self_hash"),
        )
        if tuple(split.split_index for split in plan.splits) != tuple(
            range(len(plan.splits))
        ):
            raise VNextContractError(
                "inner fold split indices must be contiguous"
            )
        if plan.self_hash != _payload_self_hash(plan._payload()):
            raise VNextContractError("inner fold plan self_hash mismatch")
        return plan

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "fold_spec_digest": self.fold_spec_digest,
            "splits": [split.to_dict() for split in self.splits],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "self_hash": self.self_hash}


@dataclasses.dataclass(frozen=True)
class InnerCVPlan:
    schema_version: str
    fold_manifest_digest: str
    content_component_manifest_digest: str
    model_spec_digest: str
    plans: tuple[InnerFoldPlan, ...]
    self_hash: str

    @classmethod
    def from_dict(cls, value: object) -> "InnerCVPlan":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "fold_manifest_digest",
                "content_component_manifest_digest",
                "model_spec_digest",
                "plans",
                "self_hash",
            },
            "inner CV plan",
        )
        payload = _check_self_hash(raw, "inner CV plan")
        return cls._from_payload(payload, raw["self_hash"])

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        self_hash: str,
    ) -> "InnerCVPlan":
        if payload["schema_version"] != INNER_CV_PLAN_SCHEMA_VERSION:
            raise VNextContractError("inner CV plan schema mismatch")
        plans = tuple(
            InnerFoldPlan.from_dict(item)
            for item in _exact_list(
                payload["plans"], "inner CV plans", nonempty=True
            )
        )
        plan = cls(
            INNER_CV_PLAN_SCHEMA_VERSION,
            _sha256(
                payload["fold_manifest_digest"], "fold_manifest_digest"
            ),
            _sha256(
                payload["content_component_manifest_digest"],
                "content_component_manifest_digest",
            ),
            _sha256(payload["model_spec_digest"], "model_spec_digest"),
            plans,
            _sha256(self_hash, "inner CV plan self_hash"),
        )
        if tuple(row.fold_id for row in plans) != tuple(
            sorted({row.fold_id for row in plans})
        ):
            raise VNextContractError(
                "inner CV plans must have sorted unique fold ids"
            )
        if plan.self_hash != _payload_self_hash(plan._payload()):
            raise VNextContractError("inner CV plan self_hash mismatch")
        return plan

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_manifest_digest": self.fold_manifest_digest,
            "content_component_manifest_digest": self.content_component_manifest_digest,
            "model_spec_digest": self.model_spec_digest,
            "plans": [plan.to_dict() for plan in self.plans],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "self_hash": self.self_hash}

    @property
    def by_fold(self) -> dict[str, InnerFoldPlan]:
        return {plan.fold_id: plan for plan in self.plans}

    def validate_against(
        self,
        fold_manifest: FoldManifest,
        corpus_manifest: CorpusVNextManifest,
        content_manifest: ContentComponentManifest,
        model_spec: ModelSpec,
    ) -> "InnerCVPlan":
        rebuilt = build_inner_cv_plan(
            fold_manifest,
            corpus_manifest,
            content_manifest,
            model_spec,
        )
        if rebuilt != self:
            raise VNextContractError(
                "inner CV plan differs from canonical pre-fit rebuild"
            )
        return self


def build_fold_manifest(
    corpus_manifest: CorpusVNextManifest,
    content_manifest: ContentComponentManifest,
    *,
    mode: str,
) -> FoldManifest:
    """Freeze model-independent outer splits before any cache/factory/fit."""

    corpus_manifest.validate(content_manifest=content_manifest)
    if type(mode) is not str or mode not in FOLD_MODES:
        raise VNextContractError("mode must be exactly 'isolated' or 'purged'")
    works = corpus_manifest.works
    work_by_id = {work.work_id: work for work in works}
    all_work_ids = tuple(sorted(work_by_id))
    component_by_work = content_manifest.component_by_work
    works_by_component = {
        component.component_id: set(component.work_ids)
        for component in content_manifest.components
    }
    if mode == "isolated" and any(
        len(component.work_ids) != 1 for component in content_manifest.components
    ):
        raise VNextContractError(
            "isolated mode requires proof that every component has one work"
        )
    probability_order = corpus_manifest.author_ids
    counts_by_author: dict[str, int] = defaultdict(int)
    for work in works:
        counts_by_author[work.author_id] += 1
    metric_order = tuple(
        author for author in probability_order if counts_by_author[author] >= 2
    )
    if not metric_order:
        raise VNextContractError("no author has enough works for a LOBO fold")
    test_work_ids = tuple(
        work.work_id for work in works if work.author_id in set(metric_order)
    )
    folds: list[FoldSpec] = []
    for test_work_id in test_work_ids:
        component_id = component_by_work[test_work_id]
        if mode == "isolated":
            purged: tuple[str, ...] = ()
        else:
            purged = tuple(
                sorted(works_by_component[component_id] - {test_work_id})
            )
        train_work_ids = tuple(
            sorted(set(all_work_ids) - {test_work_id} - set(purged))
        )
        train_authors = {work_by_id[work_id].author_id for work_id in train_work_ids}
        missing_classes = sorted(set(probability_order) - train_authors)
        if missing_classes:
            raise VNextContractError(
                f"outer fold {test_work_id!r} loses train support for "
                f"{missing_classes}"
            )
        folds.append(
            FoldSpec.build(
                fold_id=test_work_id,
                test_work_id=test_work_id,
                content_component_id=component_id,
                train_work_ids=train_work_ids,
                purged_work_ids=purged,
                probability_class_order=probability_order,
                metric_label_order=metric_order,
            )
        )
    payload = {
        "schema_version": FOLD_MANIFEST_SCHEMA_VERSION,
        "mode": mode,
        "corpus_manifest_digest": corpus_manifest.self_hash,
        "content_component_manifest_digest": content_manifest.self_hash,
        "probability_class_order": list(probability_order),
        "metric_label_order": list(metric_order),
        "folds": [fold.to_dict() for fold in folds],
    }
    return FoldManifest._from_payload(payload, _payload_self_hash(payload))


def build_inner_cv_plan(
    fold_manifest: FoldManifest,
    corpus_manifest: CorpusVNextManifest,
    content_manifest: ContentComponentManifest,
    model_spec: ModelSpec,
) -> InnerCVPlan:
    """Freeze the model-specific inner plan without contaminating outer folds."""

    fold_manifest.validate_against(corpus_manifest, content_manifest)
    model_spec.validate()
    work_by_id = {work.work_id: work for work in corpus_manifest.works}
    component_by_work = content_manifest.component_by_work
    works_by_component = {
        component.component_id: set(component.work_ids)
        for component in content_manifest.components
    }
    plans: list[InnerFoldPlan] = []
    if (
        fold_manifest.mode == "purged"
        and model_spec.requires_inner_cv
        and not model_spec.supports_component_aware_inner_cv
    ):
        raise VNextContractError(
            f"model {model_spec.model_id!r} lacks component-aware inner CV "
            "required by purged mode"
        )
    for fold in fold_manifest.folds:
        splits: tuple[InnerSplitSpec, ...] = ()
        if model_spec.requires_inner_cv:
            assert model_spec.inner_cv_splits is not None
            train_set = set(fold.train_work_ids)
            train_components = {
                component_by_work[work_id] for work_id in fold.train_work_ids
            }
            has_multiwork_component = any(
                len(works_by_component[component_id] & train_set) > 1
                for component_id in train_components
            )
            if (
                has_multiwork_component
                and not model_spec.supports_component_aware_inner_cv
            ):
                raise VNextContractError(
                    f"model {model_spec.model_id!r} lacks component-aware inner CV"
                )
            splits = plan_component_aware_inner_splits(
                fold.train_work_ids,
                corpus_manifest.works,
                content_manifest,
                n_splits=model_spec.inner_cv_splits,
                probability_class_order=fold.probability_class_order,
            )
            for split in splits:
                if set(split.train_work_ids) | set(split.validation_work_ids) != train_set:
                    raise VNextContractError(
                        f"inner split does not partition outer train for {fold.fold_id!r}"
                    )
                if any(
                    work_id not in work_by_id
                    for work_id in (
                        *split.train_work_ids,
                        *split.validation_work_ids,
                    )
                ):
                    raise VNextContractError(
                        f"inner split references an unknown work for {fold.fold_id!r}"
                    )
        plans.append(
            InnerFoldPlan.build(
                fold_id=fold.fold_id,
                fold_spec_digest=fold.self_hash,
                splits=splits,
            )
        )
    payload = {
        "schema_version": INNER_CV_PLAN_SCHEMA_VERSION,
        "fold_manifest_digest": fold_manifest.self_hash,
        "content_component_manifest_digest": content_manifest.self_hash,
        "model_spec_digest": model_spec.self_hash,
        "plans": [plan.to_dict() for plan in plans],
    }
    return InnerCVPlan._from_payload(payload, _payload_self_hash(payload))


def loads_corpus_vnext_manifest(text: str) -> CorpusVNextManifest:
    return CorpusVNextManifest.from_dict(_strict_raw(text, "corpus vNext manifest"))


def load_corpus_vnext_manifest(
    path: str | os.PathLike[str],
) -> CorpusVNextManifest:
    return CorpusVNextManifest.from_dict(_strict_file(path, "corpus vNext manifest"))


def loads_content_component_manifest(
    text: str,
    *,
    works: Sequence[WorkIdentity] | None = None,
) -> ContentComponentManifest:
    return ContentComponentManifest.from_dict(
        _strict_raw(text, "content component manifest"),
        works=works,
    )


def load_content_component_manifest(
    path: str | os.PathLike[str],
    *,
    works: Sequence[WorkIdentity] | None = None,
) -> ContentComponentManifest:
    return ContentComponentManifest.from_dict(
        _strict_file(path, "content component manifest"),
        works=works,
    )


def loads_fold_manifest(text: str) -> FoldManifest:
    return FoldManifest.from_dict(_strict_raw(text, "fold manifest"))


def load_fold_manifest(path: str | os.PathLike[str]) -> FoldManifest:
    return FoldManifest.from_dict(_strict_file(path, "fold manifest"))


def loads_inner_cv_plan(text: str) -> InnerCVPlan:
    return InnerCVPlan.from_dict(_strict_raw(text, "inner CV plan"))


def load_inner_cv_plan(path: str | os.PathLike[str]) -> InnerCVPlan:
    return InnerCVPlan.from_dict(_strict_file(path, "inner CV plan"))


def loads_model_spec(text: str) -> ModelSpec:
    return ModelSpec.from_dict(_strict_raw(text, "model spec"))


def load_model_spec(path: str | os.PathLike[str]) -> ModelSpec:
    return ModelSpec.from_dict(_strict_file(path, "model spec"))


def loads_inference_spec(text: str) -> InferenceSpec:
    return InferenceSpec.from_dict(_strict_raw(text, "inference spec"))


def load_inference_spec(path: str | os.PathLike[str]) -> InferenceSpec:
    return InferenceSpec.from_dict(_strict_file(path, "inference spec"))


__all__ = [
    "CONTENT_CANDIDATE_ORIGINS",
    "CONTENT_COMPONENT_SCHEMA_VERSION",
    "CONTENT_DISPOSITIONS",
    "CONTENT_EDGE_TYPES",
    "CORPUS_KINDS",
    "CORPUS_VNEXT_SCHEMA_VERSION",
    "FOLD_MANIFEST_SCHEMA_VERSION",
    "FOLD_MODES",
    "FOLD_SPEC_SCHEMA_VERSION",
    "INFERENCE_SPEC_SCHEMA_VERSION",
    "INNER_CV_PLAN_SCHEMA_VERSION",
    "INNER_FOLD_PLAN_SCHEMA_VERSION",
    "INNER_SPLIT_SCHEMA_VERSION",
    "LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION",
    "MODEL_SPEC_SCHEMA_VERSION",
    "WORK_KINDS",
    "ContentCandidate",
    "ContentComponent",
    "ContentComponentManifest",
    "ContentEdge",
    "CorpusVNextManifest",
    "FoldManifest",
    "FoldSpec",
    "InferenceSpec",
    "InnerCVPlan",
    "InnerFoldPlan",
    "InnerSplitSpec",
    "ModelSpec",
    "RawInventoryEntry",
    "VNextContractError",
    "WorkIdentity",
    "automatic_candidates_digest",
    "build_corpus_vnext_manifest",
    "build_fold_manifest",
    "build_inner_cv_plan",
    "canonical_json_bytes",
    "canonical_sha256",
    "discover_literal_byte_content_candidates",
    "inventory_raw_files",
    "load_content_component_manifest",
    "load_corpus_vnext_manifest",
    "load_fold_manifest",
    "load_inner_cv_plan",
    "load_inference_spec",
    "load_model_spec",
    "loads_content_component_manifest",
    "loads_corpus_vnext_manifest",
    "loads_fold_manifest",
    "loads_inner_cv_plan",
    "loads_inference_spec",
    "loads_model_spec",
    "plan_component_aware_inner_splits",
    "recompute_automatic_content_candidates",
    "verify_raw_inventory",
]
