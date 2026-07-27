"""Immutable hybrid source acquisition for the bounded RuAA R1 corpus.

This module joins two already-pinned provider contracts; it performs no title
discovery and makes no model, inference, approval, or publication decision.
An exact R1 acquisition manifest contains:

* one explicitly reviewed :class:`WikisourceCampaignSpec`;
* the one pinned FEB edition of Pushkin's ``История Пугачёва``;
* an explicit sorted inventory equal to the provider union;
* the two explicit, evidence-bound exclusions; and
* the exact content-quality policy run before publication.

Provider outputs are materialized through their own fail-closed contracts and
then copied into a new create-if-absent hybrid namespace.  A complete existing
namespace is fully revalidated without invoking either network transport.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import os
import pathlib
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any

from ..domain.corpus_identity import CONTENT_OVERLAP_POLICY_VERSION
from ..jsonio import (
    StrictJSONError,
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
    loads_strict,
)
from .feb_vnext import (
    FEBBytesTransport,
    FEBWorkReceipt,
    FEBAcquisitionError,
    PinnedFEBWorkSpec,
    materialize_pinned_feb_work,
)
from .text_quality_vnext import (
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_MINIMUM_SHINGLES,
    DEFAULT_MINIMUM_WORDS,
    DEFAULT_SAMPLE_SIZE,
    TEXT_QUALITY_AUDIT_SCHEMA_VERSION,
    TEXT_QUALITY_POLICY_VERSION,
    CorpusTextAuditReport,
    CorpusTextQualityError,
    audit_corpus_texts,
    require_text_quality,
)
from .wikisource_campaign import (
    JSONTransport,
    WikisourceCampaignError,
    WikisourceCampaignReceipt,
    WikisourceCampaignSpec,
    load_campaign_receipt,
    materialize_campaign,
)
from .wikisource_vnext import (
    PINNED_WORK_SPEC_SCHEMA_VERSION_V2,
    WholeWorkReceipt,
    WikisourceAcquisitionError,
    load_whole_work_receipt,
)


R1_ACQUISITION_MANIFEST_SCHEMA_VERSION = (
    "stylo.ruaa-r1.hybrid-acquisition-manifest.v1"
)
R1_ACQUISITION_RECEIPT_SCHEMA_VERSION = (
    "stylo.ruaa-r1.hybrid-acquisition-receipt.v1"
)
R1_ACQUISITION_KIND = "bounded_exploratory_source_acquisition_only"
R1_FEB_WORK_ID = "pushkin/история_пугачёва"
R1_COLLECTION_UMBRELLA_WORK_ID = "turgenev/записки_охотника"
R1_AUTHORSHIP_MISMATCH_WORK_ID = "serafimovich/у_нас_и_у_них"
R1_EXCLUDED_WORK_IDS = tuple(
    sorted(
        (
            R1_COLLECTION_UMBRELLA_WORK_ID,
            R1_AUTHORSHIP_MISMATCH_WORK_ID,
        )
    )
)

MANIFEST_NAME = "acquisition-manifest.json"
AUDIT_REPORT_NAME = "text-quality-audit.json"
ACQUISITION_RECEIPT_NAME = "acquisition-receipt.json"
WIKISOURCE_SPEC_PATH = "providers/wikisource/campaign-spec.json"
WIKISOURCE_RECEIPT_PATH = "providers/wikisource/campaign-receipt.json"
WIKISOURCE_WORK_RECEIPT_PREFIX = (
    "providers/wikisource/work-receipts"
)
FEB_SPEC_PATH = "providers/feb/work-spec.json"
FEB_RECEIPT_PATH = "providers/feb/work-receipt.json"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FRACTION_RE = re.compile(r"^([1-9][0-9]*)/([1-9][0-9]*)$")


class R1AcquisitionError(ValueError):
    """The hybrid R1 manifest or immutable materialization is unsafe."""


class R1AcquisitionAuditError(R1AcquisitionError):
    """All provider bytes were acquired, but the text-quality gate blocked."""

    def __init__(
        self,
        message: str,
        *,
        report: CorpusTextAuditReport,
        report_path: pathlib.Path,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.report_path = report_path


def _exact_object(
    value: object,
    keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise R1AcquisitionError(f"{label} must be an exact JSON object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise R1AcquisitionError(
            f"{label} keys must be exact; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _exact_list(
    value: object,
    label: str,
    *,
    nonempty: bool = False,
) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = " non-empty" if nonempty else ""
        raise R1AcquisitionError(
            f"{label} must be an exact{qualifier} array"
        )
    return value


def _exact_str(value: object, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise R1AcquisitionError(
            f"{label} must be an exact non-empty NUL-free string"
        )
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise R1AcquisitionError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise R1AcquisitionError(f"{label} must be an exact boolean")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _exact_str(value, label)
    if _HEX64_RE.fullmatch(digest) is None:
        raise R1AcquisitionError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return digest


def _work_id(value: object, label: str = "work_id") -> str:
    text = _exact_str(value, label)
    if "\\" in text or "\r" in text or "\n" in text:
        raise R1AcquisitionError(
            f"{label} must use canonical POSIX separators"
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise R1AcquisitionError(
            f"{label} must be a canonical author/work identifier"
        )
    return text


def _canonical_json_text(value: object) -> str:
    return dumps_strict(value, indent=2, sort_keys=True) + "\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hashed_payload(
    raw: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    if canonical_hash(payload) != recorded:
        raise R1AcquisitionError(f"{label} self_hash mismatch")
    return payload


@dataclasses.dataclass(frozen=True)
class R1TextQualitySpec:
    minimum_words: int
    containment_threshold: Fraction
    minimum_shingles: int
    sample_size: int

    @classmethod
    def build(cls) -> "R1TextQualitySpec":
        return cls(
            DEFAULT_MINIMUM_WORDS,
            DEFAULT_CONTAINMENT_THRESHOLD,
            DEFAULT_MINIMUM_SHINGLES,
            DEFAULT_SAMPLE_SIZE,
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1TextQualitySpec":
        raw = _exact_object(
            value,
            {
                "audit_schema_version",
                "policy_version",
                "content_overlap_policy_version",
                "minimum_words",
                "containment_threshold",
                "minimum_shingles",
                "sample_size",
            },
            "R1 text-quality spec",
        )
        expected_versions = {
            "audit_schema_version": TEXT_QUALITY_AUDIT_SCHEMA_VERSION,
            "policy_version": TEXT_QUALITY_POLICY_VERSION,
            "content_overlap_policy_version": (
                CONTENT_OVERLAP_POLICY_VERSION
            ),
        }
        for key, expected in expected_versions.items():
            if raw[key] != expected:
                raise R1AcquisitionError(
                    f"R1 text-quality spec {key} must be {expected!r}"
                )
        fraction_text = _exact_str(
            raw["containment_threshold"],
            "R1 text-quality spec.containment_threshold",
        )
        match = _FRACTION_RE.fullmatch(fraction_text)
        if match is None:
            raise R1AcquisitionError(
                "R1 text-quality containment_threshold must be an exact "
                "positive rational"
            )
        threshold = Fraction(int(match.group(1)), int(match.group(2)))
        if (
            f"{threshold.numerator}/{threshold.denominator}" != fraction_text
            or not Fraction(0, 1) < threshold <= Fraction(1, 1)
        ):
            raise R1AcquisitionError(
                "R1 text-quality containment_threshold is noncanonical"
            )
        result = cls(
            _exact_int(
                raw["minimum_words"],
                "R1 text-quality spec.minimum_words",
                minimum=1,
            ),
            threshold,
            _exact_int(
                raw["minimum_shingles"],
                "R1 text-quality spec.minimum_shingles",
                minimum=1,
            ),
            _exact_int(
                raw["sample_size"],
                "R1 text-quality spec.sample_size",
                minimum=1,
            ),
        )
        expected = cls.build()
        if result != expected:
            raise R1AcquisitionError(
                "R1 text-quality spec must equal the frozen v1 policy"
            )
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_schema_version": TEXT_QUALITY_AUDIT_SCHEMA_VERSION,
            "policy_version": TEXT_QUALITY_POLICY_VERSION,
            "content_overlap_policy_version": (
                CONTENT_OVERLAP_POLICY_VERSION
            ),
            "minimum_words": self.minimum_words,
            "containment_threshold": (
                f"{self.containment_threshold.numerator}/"
                f"{self.containment_threshold.denominator}"
            ),
            "minimum_shingles": self.minimum_shingles,
            "sample_size": self.sample_size,
        }

    def validate(self) -> "R1TextQualitySpec":
        if R1TextQualitySpec.from_dict(self.to_dict()) != self:
            raise R1AcquisitionError("R1 text-quality spec is noncanonical")
        return self


@dataclasses.dataclass(frozen=True)
class R1Exclusion:
    work_id: str
    reason_code: str
    evidence_sha256: str
    receipt_sha256: str | None

    @classmethod
    def collection_umbrella(cls, evidence_sha256: str) -> "R1Exclusion":
        return cls.from_dict(
            {
                "work_id": R1_COLLECTION_UMBRELLA_WORK_ID,
                "disposition": "exclude_from_corpus",
                "reason_code": "collection_umbrella",
                "evidence_sha256": evidence_sha256,
                "receipt_sha256": None,
            }
        )

    @classmethod
    def authorship_mismatch(
        cls,
        *,
        evidence_sha256: str,
        receipt_sha256: str,
    ) -> "R1Exclusion":
        return cls.from_dict(
            {
                "work_id": R1_AUTHORSHIP_MISMATCH_WORK_ID,
                "disposition": "exclude_from_corpus",
                "reason_code": "authorship_mismatch",
                "evidence_sha256": evidence_sha256,
                "receipt_sha256": receipt_sha256,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1Exclusion":
        raw = _exact_object(
            value,
            {
                "work_id",
                "disposition",
                "reason_code",
                "evidence_sha256",
                "receipt_sha256",
            },
            "R1 exclusion",
        )
        if raw["disposition"] != "exclude_from_corpus":
            raise R1AcquisitionError(
                "R1 exclusion disposition must be 'exclude_from_corpus'"
            )
        work = _work_id(raw["work_id"], "R1 exclusion.work_id")
        reason = _exact_str(raw["reason_code"], "R1 exclusion.reason_code")
        expected_reason = {
            R1_COLLECTION_UMBRELLA_WORK_ID: "collection_umbrella",
            R1_AUTHORSHIP_MISMATCH_WORK_ID: "authorship_mismatch",
        }.get(work)
        if expected_reason is None or reason != expected_reason:
            raise R1AcquisitionError(
                "R1 exclusion work/reason is not one of the two frozen "
                "exclusions"
            )
        evidence = _sha256(
            raw["evidence_sha256"],
            "R1 exclusion.evidence_sha256",
        )
        receipt_value = raw["receipt_sha256"]
        if work == R1_COLLECTION_UMBRELLA_WORK_ID:
            if receipt_value is not None:
                raise R1AcquisitionError(
                    "collection umbrella exclusion has no authorship receipt"
                )
            receipt = None
        else:
            receipt = _sha256(
                receipt_value,
                "R1 exclusion.receipt_sha256",
            )
        return cls(work, reason, evidence, receipt)

    def to_dict(self) -> dict[str, object]:
        return {
            "work_id": self.work_id,
            "disposition": "exclude_from_corpus",
            "reason_code": self.reason_code,
            "evidence_sha256": self.evidence_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


def _manifest_core(
    *,
    wikisource_campaign: WikisourceCampaignSpec,
    feb_work_spec: PinnedFEBWorkSpec,
    included_work_ids: Sequence[str],
    exclusions: Sequence[R1Exclusion],
    text_quality_spec: R1TextQualitySpec,
) -> dict[str, object]:
    return {
        "schema_version": R1_ACQUISITION_MANIFEST_SCHEMA_VERSION,
        "acquisition_kind": R1_ACQUISITION_KIND,
        "wikisource_campaign": wikisource_campaign.to_dict(),
        "feb_work_spec": feb_work_spec.to_dict(),
        "included_work_ids": list(included_work_ids),
        "exclusions": [row.to_dict() for row in exclusions],
        "text_quality_spec": text_quality_spec.to_dict(),
        "fit_performed": False,
        "confirmatory_authorized": False,
        "public_output_authorized": False,
    }


@dataclasses.dataclass(frozen=True)
class R1AcquisitionManifest:
    wikisource_campaign: WikisourceCampaignSpec
    feb_work_spec: PinnedFEBWorkSpec
    included_work_ids: tuple[str, ...]
    exclusions: tuple[R1Exclusion, ...]
    text_quality_spec: R1TextQualitySpec
    generation_id: str
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        wikisource_campaign: WikisourceCampaignSpec,
        feb_work_spec: PinnedFEBWorkSpec,
        included_work_ids: Sequence[str],
        collection_umbrella_evidence_sha256: str,
        authorship_mismatch_evidence_sha256: str,
        authorship_mismatch_receipt_sha256: str,
    ) -> "R1AcquisitionManifest":
        if type(wikisource_campaign) is not WikisourceCampaignSpec:
            raise R1AcquisitionError(
                "R1 manifest requires exactly WikisourceCampaignSpec"
            )
        if type(feb_work_spec) is not PinnedFEBWorkSpec:
            raise R1AcquisitionError(
                "R1 manifest requires exactly PinnedFEBWorkSpec"
            )
        included = tuple(
            _work_id(item, f"included_work_ids[{index}]")
            for index, item in enumerate(included_work_ids)
        )
        exclusions = tuple(
            sorted(
                (
                    R1Exclusion.collection_umbrella(
                        collection_umbrella_evidence_sha256
                    ),
                    R1Exclusion.authorship_mismatch(
                        evidence_sha256=(
                            authorship_mismatch_evidence_sha256
                        ),
                        receipt_sha256=(
                            authorship_mismatch_receipt_sha256
                        ),
                    ),
                ),
                key=lambda row: row.work_id,
            )
        )
        quality = R1TextQualitySpec.build()
        core = _manifest_core(
            wikisource_campaign=wikisource_campaign,
            feb_work_spec=feb_work_spec,
            included_work_ids=included,
            exclusions=exclusions,
            text_quality_spec=quality,
        )
        generation_id = canonical_hash(core)
        payload = {**core, "generation_id": generation_id}
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1AcquisitionManifest":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "acquisition_kind",
                "wikisource_campaign",
                "feb_work_spec",
                "included_work_ids",
                "exclusions",
                "text_quality_spec",
                "fit_performed",
                "confirmatory_authorized",
                "public_output_authorized",
                "generation_id",
                "self_hash",
            },
            "R1 acquisition manifest",
        )
        payload = _self_hashed_payload(raw, label="R1 acquisition manifest")
        if raw["schema_version"] != R1_ACQUISITION_MANIFEST_SCHEMA_VERSION:
            raise R1AcquisitionError(
                "R1 acquisition manifest is legacy or unsupported"
            )
        if raw["acquisition_kind"] != R1_ACQUISITION_KIND:
            raise R1AcquisitionError(
                f"R1 acquisition_kind must be {R1_ACQUISITION_KIND!r}"
            )
        for key in (
            "fit_performed",
            "confirmatory_authorized",
            "public_output_authorized",
        ):
            if _exact_bool(raw[key], f"R1 acquisition manifest.{key}"):
                raise R1AcquisitionError(
                    f"R1 acquisition manifest {key} must be false"
                )
        try:
            wikisource = WikisourceCampaignSpec.from_dict(
                raw["wikisource_campaign"]
            )
            feb = PinnedFEBWorkSpec.from_dict(raw["feb_work_spec"])
        except (
            WikisourceCampaignError,
            WikisourceAcquisitionError,
            FEBAcquisitionError,
        ) as exc:
            raise R1AcquisitionError(
                f"R1 embedded provider spec is invalid: {exc}"
            ) from exc
        if any(
            work.schema_version != PINNED_WORK_SPEC_SCHEMA_VERSION_V2
            for work in wikisource.works
        ):
            raise R1AcquisitionError(
                "R1 Wikisource campaign may contain only pinned-work v2 specs"
            )
        if feb.work_id != R1_FEB_WORK_ID:
            raise R1AcquisitionError(
                f"R1 FEB work must be exactly {R1_FEB_WORK_ID!r}"
            )
        if feb.work_id in set(wikisource.work_ids):
            raise R1AcquisitionError(
                "R1 provider inventories overlap on the FEB work"
            )
        raw_ids = _exact_list(
            raw["included_work_ids"],
            "R1 acquisition manifest.included_work_ids",
            nonempty=True,
        )
        included = tuple(
            _work_id(
                item,
                f"R1 acquisition manifest.included_work_ids[{index}]",
            )
            for index, item in enumerate(raw_ids)
        )
        if (
            included != tuple(sorted(included))
            or len(included) != len(set(included))
            or not included
        ):
            raise R1AcquisitionError(
                "R1 included_work_ids must be a non-empty sorted unique "
                "explicit inventory"
            )
        expected_included = tuple(
            sorted((*wikisource.work_ids, feb.work_id))
        )
        if included != expected_included:
            raise R1AcquisitionError(
                "R1 included_work_ids differ from embedded provider specs"
            )
        exclusions = tuple(
            R1Exclusion.from_dict(item)
            for item in _exact_list(
                raw["exclusions"],
                "R1 acquisition manifest.exclusions",
                nonempty=True,
            )
        )
        if (
            tuple(row.work_id for row in exclusions)
            != R1_EXCLUDED_WORK_IDS
            or len(exclusions) != 2
        ):
            raise R1AcquisitionError(
                "R1 exclusions must be the exact sorted two-work inventory"
            )
        if set(included).intersection(R1_EXCLUDED_WORK_IDS):
            raise R1AcquisitionError(
                "R1 included and excluded inventories overlap"
            )
        quality = R1TextQualitySpec.from_dict(raw["text_quality_spec"])
        generation = _sha256(
            raw["generation_id"],
            "R1 acquisition manifest.generation_id",
        )
        core = _manifest_core(
            wikisource_campaign=wikisource,
            feb_work_spec=feb,
            included_work_ids=included,
            exclusions=exclusions,
            text_quality_spec=quality,
        )
        if canonical_hash(core) != generation:
            raise R1AcquisitionError(
                "R1 acquisition manifest generation_id mismatch"
            )
        return cls(
            wikisource,
            feb,
            included,
            exclusions,
            quality,
            generation,
            _sha256(
                raw["self_hash"],
                "R1 acquisition manifest.self_hash",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        core = _manifest_core(
            wikisource_campaign=self.wikisource_campaign,
            feb_work_spec=self.feb_work_spec,
            included_work_ids=self.included_work_ids,
            exclusions=self.exclusions,
            text_quality_spec=self.text_quality_spec,
        )
        return {
            **core,
            "generation_id": self.generation_id,
            "self_hash": self.self_hash,
        }

    def validate(self) -> "R1AcquisitionManifest":
        if R1AcquisitionManifest.from_dict(self.to_dict()) != self:
            raise R1AcquisitionError(
                "R1 acquisition manifest is noncanonical"
            )
        return self


def loads_r1_acquisition_manifest(text: str) -> R1AcquisitionManifest:
    try:
        return R1AcquisitionManifest.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise R1AcquisitionError(f"R1 acquisition manifest: {exc}") from exc


def load_r1_acquisition_manifest(
    path: str | os.PathLike[str],
) -> R1AcquisitionManifest:
    try:
        return R1AcquisitionManifest.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise R1AcquisitionError(f"R1 acquisition manifest: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class R1RawInventoryRow:
    work_id: str
    relative_path: str
    byte_size: int
    sha256: str

    @classmethod
    def build(cls, work_id: str, payload: bytes) -> "R1RawInventoryRow":
        work = _work_id(work_id)
        if type(payload) is not bytes or not payload:
            raise R1AcquisitionError(
                f"R1 raw payload is empty or non-bytes: {work}"
            )
        return cls(
            work,
            f"raw/{work}.txt",
            len(payload),
            _sha256_bytes(payload),
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1RawInventoryRow":
        raw = _exact_object(
            value,
            {"work_id", "relative_path", "byte_size", "sha256"},
            "R1 raw inventory row",
        )
        work = _work_id(raw["work_id"], "R1 raw inventory row.work_id")
        relative = _exact_str(
            raw["relative_path"],
            "R1 raw inventory row.relative_path",
        )
        if relative != f"raw/{work}.txt":
            raise R1AcquisitionError(
                "R1 raw inventory relative path is noncanonical"
            )
        return cls(
            work,
            relative,
            _exact_int(
                raw["byte_size"],
                "R1 raw inventory row.byte_size",
                minimum=1,
            ),
            _sha256(raw["sha256"], "R1 raw inventory row.sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class R1AcquisitionReceipt:
    manifest_sha256: str
    generation_id: str
    wikisource_campaign_spec_sha256: str
    wikisource_campaign_receipt_sha256: str
    feb_work_spec_sha256: str
    feb_work_receipt_sha256: str
    text_quality_audit_sha256: str
    included_work_ids: tuple[str, ...]
    raw_inventory: tuple[R1RawInventoryRow, ...]
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        manifest: R1AcquisitionManifest,
        wikisource_receipt: WikisourceCampaignReceipt,
        feb_receipt: FEBWorkReceipt,
        audit_report: CorpusTextAuditReport,
        raw_payloads: Mapping[str, bytes],
    ) -> "R1AcquisitionReceipt":
        manifest.validate()
        if type(wikisource_receipt) is not WikisourceCampaignReceipt:
            raise R1AcquisitionError(
                "R1 receipt requires exactly WikisourceCampaignReceipt"
            )
        if type(feb_receipt) is not FEBWorkReceipt:
            raise R1AcquisitionError(
                "R1 receipt requires exactly FEBWorkReceipt"
            )
        if type(audit_report) is not CorpusTextAuditReport:
            raise R1AcquisitionError(
                "R1 receipt requires exactly CorpusTextAuditReport"
            )
        audit_report.validate()
        if audit_report.status != "passed":
            raise R1AcquisitionError(
                "R1 receipt cannot bind a blocked text-quality audit"
            )
        if type(raw_payloads) is not dict or set(raw_payloads) != set(
            manifest.included_work_ids
        ):
            raise R1AcquisitionError(
                "R1 receipt raw inventory differs from manifest"
            )
        rows = tuple(
            R1RawInventoryRow.build(work_id, raw_payloads[work_id])
            for work_id in manifest.included_work_ids
        )
        payload: dict[str, object] = {
            "schema_version": R1_ACQUISITION_RECEIPT_SCHEMA_VERSION,
            "acquisition_kind": R1_ACQUISITION_KIND,
            "manifest_sha256": manifest.self_hash,
            "generation_id": manifest.generation_id,
            "wikisource_campaign_spec_sha256": (
                manifest.wikisource_campaign.self_hash
            ),
            "wikisource_campaign_receipt_sha256": (
                wikisource_receipt.self_hash
            ),
            "feb_work_spec_sha256": manifest.feb_work_spec.self_hash,
            "feb_work_receipt_sha256": feb_receipt.self_hash,
            "text_quality_audit_sha256": audit_report.self_hash,
            "included_work_ids": list(manifest.included_work_ids),
            "raw_inventory": [row.to_dict() for row in rows],
            "work_count": len(rows),
            "fit_performed": False,
            "confirmatory_authorized": False,
            "public_output_authorized": False,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "R1AcquisitionReceipt":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "acquisition_kind",
                "manifest_sha256",
                "generation_id",
                "wikisource_campaign_spec_sha256",
                "wikisource_campaign_receipt_sha256",
                "feb_work_spec_sha256",
                "feb_work_receipt_sha256",
                "text_quality_audit_sha256",
                "included_work_ids",
                "raw_inventory",
                "work_count",
                "fit_performed",
                "confirmatory_authorized",
                "public_output_authorized",
                "self_hash",
            },
            "R1 acquisition receipt",
        )
        _self_hashed_payload(raw, label="R1 acquisition receipt")
        if raw["schema_version"] != R1_ACQUISITION_RECEIPT_SCHEMA_VERSION:
            raise R1AcquisitionError(
                "R1 acquisition receipt is legacy or unsupported"
            )
        if raw["acquisition_kind"] != R1_ACQUISITION_KIND:
            raise R1AcquisitionError(
                "R1 acquisition receipt kind is unsupported"
            )
        for key in (
            "fit_performed",
            "confirmatory_authorized",
            "public_output_authorized",
        ):
            if _exact_bool(raw[key], f"R1 acquisition receipt.{key}"):
                raise R1AcquisitionError(
                    f"R1 acquisition receipt {key} must be false"
                )
        included = tuple(
            _work_id(item, f"R1 acquisition receipt.included[{index}]")
            for index, item in enumerate(
                _exact_list(
                    raw["included_work_ids"],
                    "R1 acquisition receipt.included_work_ids",
                    nonempty=True,
                )
            )
        )
        if (
            included != tuple(sorted(included))
            or len(included) != len(set(included))
            or not included
        ):
            raise R1AcquisitionError(
                "R1 acquisition receipt included ids are noncanonical"
            )
        rows = tuple(
            R1RawInventoryRow.from_dict(item)
            for item in _exact_list(
                raw["raw_inventory"],
                "R1 acquisition receipt.raw_inventory",
                nonempty=True,
            )
        )
        if tuple(row.work_id for row in rows) != included:
            raise R1AcquisitionError(
                "R1 acquisition receipt raw inventory differs from ids"
            )
        if (
            _exact_int(
                raw["work_count"],
                "R1 acquisition receipt.work_count",
                minimum=1,
            )
            != len(included)
        ):
            raise R1AcquisitionError(
                "R1 acquisition receipt work_count mismatch"
            )
        return cls(
            _sha256(
                raw["manifest_sha256"],
                "R1 acquisition receipt.manifest_sha256",
            ),
            _sha256(
                raw["generation_id"],
                "R1 acquisition receipt.generation_id",
            ),
            _sha256(
                raw["wikisource_campaign_spec_sha256"],
                "R1 acquisition receipt.wikisource_campaign_spec_sha256",
            ),
            _sha256(
                raw["wikisource_campaign_receipt_sha256"],
                "R1 acquisition receipt.wikisource_campaign_receipt_sha256",
            ),
            _sha256(
                raw["feb_work_spec_sha256"],
                "R1 acquisition receipt.feb_work_spec_sha256",
            ),
            _sha256(
                raw["feb_work_receipt_sha256"],
                "R1 acquisition receipt.feb_work_receipt_sha256",
            ),
            _sha256(
                raw["text_quality_audit_sha256"],
                "R1 acquisition receipt.text_quality_audit_sha256",
            ),
            included,
            rows,
            _sha256(
                raw["self_hash"],
                "R1 acquisition receipt.self_hash",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": R1_ACQUISITION_RECEIPT_SCHEMA_VERSION,
            "acquisition_kind": R1_ACQUISITION_KIND,
            "manifest_sha256": self.manifest_sha256,
            "generation_id": self.generation_id,
            "wikisource_campaign_spec_sha256": (
                self.wikisource_campaign_spec_sha256
            ),
            "wikisource_campaign_receipt_sha256": (
                self.wikisource_campaign_receipt_sha256
            ),
            "feb_work_spec_sha256": self.feb_work_spec_sha256,
            "feb_work_receipt_sha256": self.feb_work_receipt_sha256,
            "text_quality_audit_sha256": self.text_quality_audit_sha256,
            "included_work_ids": list(self.included_work_ids),
            "raw_inventory": [row.to_dict() for row in self.raw_inventory],
            "work_count": len(self.raw_inventory),
            "fit_performed": False,
            "confirmatory_authorized": False,
            "public_output_authorized": False,
            "self_hash": self.self_hash,
        }

    def validate_for(
        self,
        *,
        manifest: R1AcquisitionManifest,
        wikisource_receipt: WikisourceCampaignReceipt,
        feb_receipt: FEBWorkReceipt,
        audit_report: CorpusTextAuditReport,
        raw_payloads: Mapping[str, bytes],
    ) -> "R1AcquisitionReceipt":
        expected = R1AcquisitionReceipt.build(
            manifest=manifest,
            wikisource_receipt=wikisource_receipt,
            feb_receipt=feb_receipt,
            audit_report=audit_report,
            raw_payloads=raw_payloads,
        )
        if self != expected:
            raise R1AcquisitionError(
                "R1 acquisition receipt differs from exact inputs"
            )
        return self


def load_r1_acquisition_receipt(
    path: str | os.PathLike[str],
) -> R1AcquisitionReceipt:
    try:
        return R1AcquisitionReceipt.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise R1AcquisitionError(f"R1 acquisition receipt: {exc}") from exc


def loads_r1_acquisition_receipt(text: str) -> R1AcquisitionReceipt:
    try:
        return R1AcquisitionReceipt.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise R1AcquisitionError(f"R1 acquisition receipt: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class MaterializedR1Acquisition:
    root: pathlib.Path
    receipt: R1AcquisitionReceipt
    audit_report: CorpusTextAuditReport
    resumed: bool


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise R1AcquisitionError(
                f"{label} must not contain symlink components: {component}"
            )


@contextlib.contextmanager
def _publication_lock(parent: pathlib.Path):
    lock = parent / ".ruaa-r1-acquisition.lock"
    if lock.is_symlink():
        raise R1AcquisitionError(
            "R1 acquisition publication lock must not be a symlink"
        )
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _relative_path(path: str) -> pathlib.PurePosixPath:
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or relative.as_posix() != path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise R1AcquisitionError(f"noncanonical relative path: {path!r}")
    return relative


def _join(root: pathlib.Path, relative: str) -> pathlib.Path:
    return root.joinpath(*_relative_path(relative).parts)


def _work_receipt_path(work_id: str) -> str:
    return f"{WIKISOURCE_WORK_RECEIPT_PREFIX}/{work_id}.json"


def _expected_inventory(
    manifest: R1AcquisitionManifest,
) -> tuple[set[str], set[str]]:
    files = {
        MANIFEST_NAME,
        AUDIT_REPORT_NAME,
        ACQUISITION_RECEIPT_NAME,
        WIKISOURCE_SPEC_PATH,
        WIKISOURCE_RECEIPT_PATH,
        FEB_SPEC_PATH,
        FEB_RECEIPT_PATH,
    }
    files.update(f"raw/{work_id}.txt" for work_id in manifest.included_work_ids)
    files.update(
        _work_receipt_path(work_id)
        for work_id in manifest.wikisource_campaign.work_ids
    )
    directories: set[str] = set()
    for relative in files:
        directories.update(
            parent.as_posix()
            for parent in PurePosixPath(relative).parents
            if parent.as_posix() != "."
        )
    return files, directories


def _tree_inventory(root: pathlib.Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        for entry in os.scandir(directory):
            metadata = entry.stat(follow_symlinks=False)
            path = pathlib.Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise R1AcquisitionError(
                    f"symlink rejected in R1 acquisition: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise R1AcquisitionError(
                    f"special file rejected in R1 acquisition: {relative}"
                )
    return files, directories


def _require_canonical_json(
    path: pathlib.Path,
    value: object,
    *,
    label: str,
) -> None:
    try:
        observed = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise R1AcquisitionError(f"{label} cannot be read: {exc}") from exc
    if observed != _canonical_json_text(value):
        raise R1AcquisitionError(f"{label} JSON bytes are noncanonical")


def _audit(
    manifest: R1AcquisitionManifest,
    raw_payloads: dict[str, bytes],
) -> CorpusTextAuditReport:
    quality = manifest.text_quality_spec
    try:
        return audit_corpus_texts(
            raw_payloads,
            expected_work_ids=manifest.included_work_ids,
            minimum_words=quality.minimum_words,
            containment_threshold=quality.containment_threshold,
            minimum_shingles=quality.minimum_shingles,
            sample_size=quality.sample_size,
        )
    except CorpusTextQualityError as exc:
        raise R1AcquisitionError(
            f"R1 text-quality audit could not be completed: {exc}"
        ) from exc


def _load_audit_report(path: pathlib.Path) -> CorpusTextAuditReport:
    try:
        raw = load_strict(path)
        report = CorpusTextAuditReport(raw).validate()
    except (
        OSError,
        UnicodeError,
        StrictJSONError,
        CorpusTextQualityError,
    ) as exc:
        raise R1AcquisitionError(
            f"R1 text-quality audit report is invalid: {exc}"
        ) from exc
    _require_canonical_json(
        path,
        report.to_dict(),
        label="R1 text-quality audit report",
    )
    return report


def _provider_receipts(
    root: pathlib.Path,
    manifest: R1AcquisitionManifest,
    raw_payloads: Mapping[str, bytes],
) -> tuple[WikisourceCampaignReceipt, FEBWorkReceipt]:
    work_receipts: list[WholeWorkReceipt] = []
    for spec in manifest.wikisource_campaign.works:
        path = _join(root, _work_receipt_path(spec.work_id))
        try:
            receipt = load_whole_work_receipt(path)
            receipt.validate_for(spec, raw_payloads[spec.work_id])
        except (
            OSError,
            UnicodeError,
            WikisourceAcquisitionError,
        ) as exc:
            raise R1AcquisitionError(
                f"R1 Wikisource work receipt is invalid for "
                f"{spec.work_id}: {exc}"
            ) from exc
        _require_canonical_json(
            path,
            receipt.to_dict(),
            label=f"R1 Wikisource receipt {spec.work_id}",
        )
        work_receipts.append(receipt)
    campaign_path = _join(root, WIKISOURCE_RECEIPT_PATH)
    try:
        campaign_receipt = load_campaign_receipt(campaign_path)
        campaign_receipt.validate_for(
            manifest.wikisource_campaign,
            work_receipts,
        )
    except (
        OSError,
        UnicodeError,
        WikisourceCampaignError,
    ) as exc:
        raise R1AcquisitionError(
            f"R1 Wikisource campaign receipt is invalid: {exc}"
        ) from exc
    _require_canonical_json(
        campaign_path,
        campaign_receipt.to_dict(),
        label="R1 Wikisource campaign receipt",
    )
    feb_path = _join(root, FEB_RECEIPT_PATH)
    try:
        feb_receipt = FEBWorkReceipt.from_dict(load_strict(feb_path))
    except (
        OSError,
        UnicodeError,
        StrictJSONError,
        FEBAcquisitionError,
    ) as exc:
        raise R1AcquisitionError(
            f"R1 FEB work receipt is invalid: {exc}"
        ) from exc
    if feb_receipt != FEBWorkReceipt.build(manifest.feb_work_spec):
        raise R1AcquisitionError(
            "R1 FEB work receipt differs from embedded pinned spec"
        )
    feb_payload = raw_payloads[manifest.feb_work_spec.work_id]
    if (
        len(feb_payload) != manifest.feb_work_spec.output_byte_size
        or _sha256_bytes(feb_payload)
        != manifest.feb_work_spec.output_sha256
    ):
        raise R1AcquisitionError(
            "R1 FEB raw output differs from embedded pinned spec"
        )
    _require_canonical_json(
        feb_path,
        feb_receipt.to_dict(),
        label="R1 FEB work receipt",
    )
    return campaign_receipt, feb_receipt


def _load_existing(
    root: pathlib.Path,
    manifest: R1AcquisitionManifest,
) -> MaterializedR1Acquisition:
    if root.is_symlink() or not root.is_dir():
        raise R1AcquisitionError(
            "R1 acquisition namespace must be a real directory"
        )
    observed = _tree_inventory(root)
    expected = _expected_inventory(manifest)
    if observed != expected:
        raise R1AcquisitionError(
            "R1 acquisition namespace has missing or extra files/directories"
        )
    manifest_path = root / MANIFEST_NAME
    loaded_manifest = load_r1_acquisition_manifest(manifest_path)
    _require_canonical_json(
        manifest_path,
        loaded_manifest.to_dict(),
        label="R1 acquisition manifest",
    )
    if loaded_manifest != manifest:
        raise R1AcquisitionError(
            "R1 acquisition namespace manifest conflicts with requested manifest"
        )
    try:
        ws_ref = WikisourceCampaignSpec.from_dict(
            load_strict(_join(root, WIKISOURCE_SPEC_PATH))
        )
        feb_ref = PinnedFEBWorkSpec.from_dict(
            load_strict(_join(root, FEB_SPEC_PATH))
        )
    except (
        OSError,
        UnicodeError,
        StrictJSONError,
        WikisourceCampaignError,
        WikisourceAcquisitionError,
        FEBAcquisitionError,
    ) as exc:
        raise R1AcquisitionError(
            f"R1 provider spec reference is invalid: {exc}"
        ) from exc
    if (
        ws_ref != manifest.wikisource_campaign
        or feb_ref != manifest.feb_work_spec
    ):
        raise R1AcquisitionError(
            "R1 provider spec references conflict with manifest"
        )
    _require_canonical_json(
        _join(root, WIKISOURCE_SPEC_PATH),
        ws_ref.to_dict(),
        label="R1 Wikisource spec reference",
    )
    _require_canonical_json(
        _join(root, FEB_SPEC_PATH),
        feb_ref.to_dict(),
        label="R1 FEB spec reference",
    )
    raw_payloads = {
        work_id: _join(root, f"raw/{work_id}.txt").read_bytes()
        for work_id in manifest.included_work_ids
    }
    ws_receipt, feb_receipt = _provider_receipts(
        root,
        manifest,
        raw_payloads,
    )
    stored_audit = _load_audit_report(root / AUDIT_REPORT_NAME)
    recomputed_audit = _audit(manifest, raw_payloads)
    if stored_audit.to_dict() != recomputed_audit.to_dict():
        raise R1AcquisitionError(
            "R1 stored text-quality audit differs from literal raw bytes"
        )
    try:
        require_text_quality(stored_audit)
    except CorpusTextQualityError as exc:
        raise R1AcquisitionError(
            f"R1 stored text-quality audit is blocked: {exc}"
        ) from exc
    receipt_path = root / ACQUISITION_RECEIPT_NAME
    receipt = load_r1_acquisition_receipt(receipt_path)
    _require_canonical_json(
        receipt_path,
        receipt.to_dict(),
        label="R1 acquisition receipt",
    )
    receipt.validate_for(
        manifest=manifest,
        wikisource_receipt=ws_receipt,
        feb_receipt=feb_receipt,
        audit_report=stored_audit,
        raw_payloads=raw_payloads,
    )
    return MaterializedR1Acquisition(
        root,
        receipt,
        stored_audit,
        True,
    )


def _persist_blocked_audit(
    parent: pathlib.Path,
    manifest: R1AcquisitionManifest,
    report: CorpusTextAuditReport,
) -> pathlib.Path:
    blocked_root = parent / ".blocked-audits"
    _reject_symlink_components(
        blocked_root,
        label="R1 blocked-audit namespace",
    )
    blocked_root.mkdir(parents=True, exist_ok=True)
    path = blocked_root / (
        f"{manifest.generation_id}-{report.self_hash}.json"
    )
    payload = _canonical_json_text(report.to_dict()).encode("utf-8")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise R1AcquisitionError(
                "R1 blocked-audit path is not a regular file"
            )
        if path.read_bytes() != payload:
            raise R1AcquisitionError(
                "R1 blocked-audit path contains conflicting bytes"
            )
        return path
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def materialize_r1_acquisition(
    manifest: R1AcquisitionManifest,
    *,
    output_parent: str | os.PathLike[str],
    wikisource_transport: JSONTransport,
    feb_transport: FEBBytesTransport,
) -> MaterializedR1Acquisition:
    """Materialize and audit one exact hybrid R1 source-corpus candidate."""

    if type(manifest) is not R1AcquisitionManifest:
        raise R1AcquisitionError(
            "R1 materialization requires exactly R1AcquisitionManifest"
        )
    manifest.validate()
    if not callable(wikisource_transport) or not callable(feb_transport):
        raise R1AcquisitionError(
            "R1 provider transports must both be callable"
        )
    parent = pathlib.Path(output_parent)
    _reject_symlink_components(parent, label="R1 acquisition output parent")
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise R1AcquisitionError(
            "R1 acquisition output parent must be a real directory"
        )
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    target = parent / manifest.generation_id
    with _publication_lock(parent):
        if target.exists() or target.is_symlink():
            return _load_existing(target, manifest)

    # Provider generations are immutable source receipts, not a model or
    # representation cache.  Keeping them allows a failed quality audit to be
    # rerun without repeating network acquisition.
    provider_materializations = parent / ".provider-materializations"
    try:
        wikisource = materialize_campaign(
            manifest.wikisource_campaign,
            output_parent=provider_materializations / "wikisource",
            transport=wikisource_transport,
        )
        feb = materialize_pinned_feb_work(
            manifest.feb_work_spec,
            output_parent=provider_materializations / "feb",
            transport=feb_transport,
        )
    except (
        OSError,
        UnicodeError,
        WikisourceCampaignError,
        WikisourceAcquisitionError,
        FEBAcquisitionError,
    ) as exc:
        raise R1AcquisitionError(
            f"R1 provider materialization was rejected: {exc}"
        ) from exc

    stage = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".ruaa-r1.{manifest.generation_id[:12]}.",
            dir=parent,
        )
    )
    try:
        dump_strict(
            manifest.to_dict(),
            stage / MANIFEST_NAME,
            sort_keys=True,
            trailing_newline=True,
        )
        dump_strict(
            manifest.wikisource_campaign.to_dict(),
            _join(stage, WIKISOURCE_SPEC_PATH),
            sort_keys=True,
            trailing_newline=True,
        )
        dump_strict(
            manifest.feb_work_spec.to_dict(),
            _join(stage, FEB_SPEC_PATH),
            sort_keys=True,
            trailing_newline=True,
        )
        dump_strict(
            wikisource.receipt.to_dict(),
            _join(stage, WIKISOURCE_RECEIPT_PATH),
            sort_keys=True,
            trailing_newline=True,
        )
        dump_strict(
            feb.receipt.to_dict(),
            _join(stage, FEB_RECEIPT_PATH),
            sort_keys=True,
            trailing_newline=True,
        )
        raw_payloads: dict[str, bytes] = {}
        for spec in manifest.wikisource_campaign.works:
            source = _join(wikisource.root, spec.output_relative_path)
            payload = source.read_bytes()
            target_path = _join(stage, f"raw/{spec.work_id}.txt")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(payload)
            receipt_source = _join(
                wikisource.root,
                f"receipts/{spec.work_id}.json",
            )
            receipt_target = _join(
                stage,
                _work_receipt_path(spec.work_id),
            )
            receipt_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(receipt_source, receipt_target)
            raw_payloads[spec.work_id] = payload
        feb_payload = feb.output_path.read_bytes()
        feb_target = _join(
            stage,
            f"raw/{manifest.feb_work_spec.work_id}.txt",
        )
        feb_target.parent.mkdir(parents=True, exist_ok=True)
        feb_target.write_bytes(feb_payload)
        raw_payloads[manifest.feb_work_spec.work_id] = feb_payload

        audit = _audit(manifest, raw_payloads)
        if audit.status != "passed":
            report_path = _persist_blocked_audit(parent, manifest, audit)
            try:
                require_text_quality(audit)
            except CorpusTextQualityError as exc:
                raise R1AcquisitionAuditError(
                    f"R1 text-quality audit blocked publication: {exc}",
                    report=audit,
                    report_path=report_path,
                ) from exc
            raise AssertionError("blocked audit unexpectedly passed")
        dump_strict(
            audit.to_dict(),
            stage / AUDIT_REPORT_NAME,
            sort_keys=True,
            trailing_newline=True,
        )
        receipt = R1AcquisitionReceipt.build(
            manifest=manifest,
            wikisource_receipt=wikisource.receipt,
            feb_receipt=feb.receipt,
            audit_report=audit,
            raw_payloads=raw_payloads,
        )
        dump_strict(
            receipt.to_dict(),
            stage / ACQUISITION_RECEIPT_NAME,
            sort_keys=True,
            trailing_newline=True,
        )
        observed = _tree_inventory(stage)
        expected = _expected_inventory(manifest)
        if observed != expected:
            raise R1AcquisitionError(
                "staged R1 acquisition inventory is noncanonical"
            )
        staged = _load_existing(stage, manifest)
        with _publication_lock(parent):
            if target.exists() or target.is_symlink():
                return _load_existing(target, manifest)
            os.rename(stage, target)
        return dataclasses.replace(staged, root=target, resumed=False)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


__all__ = [
    "ACQUISITION_RECEIPT_NAME",
    "AUDIT_REPORT_NAME",
    "FEB_RECEIPT_PATH",
    "FEB_SPEC_PATH",
    "MANIFEST_NAME",
    "MaterializedR1Acquisition",
    "R1_ACQUISITION_KIND",
    "R1_ACQUISITION_MANIFEST_SCHEMA_VERSION",
    "R1_ACQUISITION_RECEIPT_SCHEMA_VERSION",
    "R1_AUTHORSHIP_MISMATCH_WORK_ID",
    "R1_COLLECTION_UMBRELLA_WORK_ID",
    "R1_EXCLUDED_WORK_IDS",
    "R1_FEB_WORK_ID",
    "R1AcquisitionAuditError",
    "R1AcquisitionError",
    "R1AcquisitionManifest",
    "R1AcquisitionReceipt",
    "R1Exclusion",
    "R1RawInventoryRow",
    "R1TextQualitySpec",
    "WIKISOURCE_RECEIPT_PATH",
    "WIKISOURCE_SPEC_PATH",
    "WIKISOURCE_WORK_RECEIPT_PREFIX",
    "load_r1_acquisition_manifest",
    "load_r1_acquisition_receipt",
    "loads_r1_acquisition_manifest",
    "loads_r1_acquisition_receipt",
    "materialize_r1_acquisition",
]
