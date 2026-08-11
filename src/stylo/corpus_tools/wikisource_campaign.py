"""Strict, immutable multi-work Wikisource corpus acquisition.

The single-work primitives in :mod:`stylo.corpus_tools.wikisource_vnext`
deliberately know nothing about corpus membership.  This module adds the
smallest possible campaign boundary around them:

* a campaign spec embeds an exact, sorted inventory of self-hashed
  ``PinnedWorkSpec`` objects;
* its generation identifier is derived only from that path-independent
  scientific payload;
* publication is create-if-absent and a complete existing generation is
  validated before any HTTP request is made;
* the published namespace contains only pinned raw text, per-work receipts,
  and one deterministic campaign receipt.

There is intentionally no title discovery, subpage traversal, ordering
inference, or best-effort repair here.  Those decisions must already be
explicit in the pinned campaign spec.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import email.utils
import fcntl
import math
import os
import pathlib
import re
import shutil
import socket
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .._strict_fields import ExactFieldReader
from ..jsonio import (
    StrictJSONError,
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
    loads_strict,
)
from .wikisource_vnext import (
    API,
    JSONTransport,
    PinnedWorkSpec,
    WholeWorkReceipt,
    WikisourceAcquisitionError,
    load_whole_work_receipt,
    materialize_pinned_work,
)


CAMPAIGN_SPEC_SCHEMA_VERSION = "stylo.wikisource.campaign-spec.v1"
CAMPAIGN_RECEIPT_SCHEMA_VERSION = "stylo.wikisource.campaign-receipt.v1"
CAMPAIGN_KIND = "bounded_exploratory_source_acquisition_only"
CAMPAIGN_RECEIPT_NAME = "campaign-receipt.json"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class WikisourceCampaignError(WikisourceAcquisitionError):
    """A campaign spec, HTTP response, or materialized generation is unsafe."""


_STRICT = ExactFieldReader(WikisourceCampaignError)
_exact_object = _STRICT.object
_exact_list = _STRICT.array
_exact_str = _STRICT.string
_exact_int = _STRICT.integer


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise WikisourceCampaignError(f"{label} must be an exact boolean")
    return value


_sha256 = _STRICT.sha256


def _relative_path(value: object, label: str) -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise WikisourceCampaignError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WikisourceCampaignError(
            f"{label} must be a canonical relative path"
        )
    return text


def _work_id(value: object, label: str) -> str:
    text = _relative_path(value, label)
    if len(PurePosixPath(text).parts) < 2:
        raise WikisourceCampaignError(
            f"{label} must be a canonical author/work identifier"
        )
    return text


def _self_hashed_payload(raw: dict[str, Any], label: str) -> dict[str, Any]:
    recorded = _sha256(raw["self_hash"], f"{label}.self_hash")
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    if canonical_hash(payload) != recorded:
        raise WikisourceCampaignError(f"{label} self_hash mismatch")
    return payload


def _canonical_json_text(value: object) -> str:
    return dumps_strict(value, indent=2, sort_keys=True) + "\n"


def _campaign_core(
    *,
    work_ids: Sequence[str],
    works: Sequence[PinnedWorkSpec],
) -> dict[str, object]:
    return {
        "schema_version": CAMPAIGN_SPEC_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "work_ids": list(work_ids),
        "works": [work.to_dict() for work in works],
    }


@dataclasses.dataclass(frozen=True)
class WikisourceCampaignSpec:
    """Exact, self-hashed corpus acquisition instructions."""

    work_ids: tuple[str, ...]
    works: tuple[PinnedWorkSpec, ...]
    generation_id: str
    self_hash: str

    @classmethod
    def build(
        cls,
        works: Sequence[PinnedWorkSpec],
    ) -> "WikisourceCampaignSpec":
        if type(works) not in {list, tuple} or not works:
            raise WikisourceCampaignError(
                "campaign works must be an exact non-empty list or tuple"
            )
        checked: list[PinnedWorkSpec] = []
        for index, work in enumerate(works):
            if type(work) is not PinnedWorkSpec:
                raise WikisourceCampaignError(
                    f"campaign works[{index}] must be exactly PinnedWorkSpec"
                )
            try:
                checked.append(work.validate())
            except WikisourceAcquisitionError as exc:
                raise WikisourceCampaignError(
                    f"campaign works[{index}] is invalid: {exc}"
                ) from exc
        checked.sort(key=lambda row: row.work_id)
        work_ids = [row.work_id for row in checked]
        if len(work_ids) != len(set(work_ids)):
            raise WikisourceCampaignError(
                "campaign work inventory contains duplicate work ids"
            )
        core = _campaign_core(work_ids=work_ids, works=checked)
        generation_id = canonical_hash(core)
        payload = {**core, "generation_id": generation_id}
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "WikisourceCampaignSpec":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "campaign_kind",
                "work_ids",
                "works",
                "generation_id",
                "self_hash",
            },
            "campaign spec",
        )
        _self_hashed_payload(raw, "campaign spec")
        if raw["schema_version"] != CAMPAIGN_SPEC_SCHEMA_VERSION:
            raise WikisourceCampaignError(
                "campaign spec is legacy or unsupported"
            )
        if raw["campaign_kind"] != CAMPAIGN_KIND:
            raise WikisourceCampaignError(
                f"campaign spec campaign_kind must be {CAMPAIGN_KIND!r}"
            )
        raw_ids = _exact_list(
            raw["work_ids"],
            "campaign spec.work_ids",
            nonempty=True,
        )
        work_ids = tuple(
            _work_id(value, f"campaign spec.work_ids[{index}]")
            for index, value in enumerate(raw_ids)
        )
        if work_ids != tuple(sorted(work_ids)):
            raise WikisourceCampaignError(
                "campaign work_ids must be sorted exactly"
            )
        if len(work_ids) != len(set(work_ids)):
            raise WikisourceCampaignError(
                "campaign work_ids must be unique"
            )
        raw_works = _exact_list(
            raw["works"],
            "campaign spec.works",
            nonempty=True,
        )
        try:
            works = tuple(
                PinnedWorkSpec.from_dict(item) for item in raw_works
            )
        except WikisourceAcquisitionError as exc:
            raise WikisourceCampaignError(
                f"campaign contains an invalid pinned work spec: {exc}"
            ) from exc
        embedded_ids = tuple(row.work_id for row in works)
        if embedded_ids != work_ids:
            raise WikisourceCampaignError(
                "campaign works must exactly match sorted work_ids order"
            )
        if len({row.self_hash for row in works}) != len(works):
            raise WikisourceCampaignError(
                "campaign works contain duplicate pinned specs"
            )
        generation_id = _sha256(
            raw["generation_id"],
            "campaign spec.generation_id",
        )
        core = _campaign_core(work_ids=work_ids, works=works)
        if canonical_hash(core) != generation_id:
            raise WikisourceCampaignError(
                "campaign generation_id does not match campaign payload"
            )
        return cls(
            work_ids,
            works,
            generation_id,
            _sha256(raw["self_hash"], "campaign spec.self_hash"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **_campaign_core(work_ids=self.work_ids, works=self.works),
            "generation_id": self.generation_id,
            "self_hash": self.self_hash,
        }

    def validate(self) -> "WikisourceCampaignSpec":
        if WikisourceCampaignSpec.from_dict(self.to_dict()) != self:
            raise WikisourceCampaignError("campaign spec is noncanonical")
        return self


def loads_campaign_spec(text: str) -> WikisourceCampaignSpec:
    try:
        return WikisourceCampaignSpec.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise WikisourceCampaignError(f"campaign spec: {exc}") from exc


def load_campaign_spec(
    path: str | os.PathLike[str],
) -> WikisourceCampaignSpec:
    try:
        return WikisourceCampaignSpec.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise WikisourceCampaignError(f"campaign spec: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class CampaignWorkReceipt:
    work_id: str
    pinned_work_spec_sha256: str
    output_relative_path: str
    output_byte_size: int
    output_sha256: str
    word_count: int
    whole_work_receipt_relative_path: str
    whole_work_receipt_self_hash: str

    @classmethod
    def build(
        cls,
        *,
        spec: PinnedWorkSpec,
        receipt: WholeWorkReceipt,
    ) -> "CampaignWorkReceipt":
        return cls.from_dict(
            {
                "work_id": spec.work_id,
                "pinned_work_spec_sha256": spec.self_hash,
                "output_relative_path": spec.output_relative_path,
                "output_byte_size": receipt.output_byte_size,
                "output_sha256": receipt.output_sha256,
                "word_count": receipt.word_count,
                "whole_work_receipt_relative_path": (
                    f"receipts/{spec.work_id}.json"
                ),
                "whole_work_receipt_self_hash": receipt.self_hash,
            }
        )

    @classmethod
    def from_dict(cls, value: object) -> "CampaignWorkReceipt":
        raw = _exact_object(
            value,
            {
                "work_id",
                "pinned_work_spec_sha256",
                "output_relative_path",
                "output_byte_size",
                "output_sha256",
                "word_count",
                "whole_work_receipt_relative_path",
                "whole_work_receipt_self_hash",
            },
            "campaign work receipt",
        )
        work_id = _work_id(
            raw["work_id"],
            "campaign work receipt.work_id",
        )
        output = _relative_path(
            raw["output_relative_path"],
            "campaign work receipt.output_relative_path",
        )
        receipt_path = _relative_path(
            raw["whole_work_receipt_relative_path"],
            "campaign work receipt.whole_work_receipt_relative_path",
        )
        if output != f"raw/{work_id}.txt":
            raise WikisourceCampaignError(
                "campaign work output path is noncanonical"
            )
        if receipt_path != f"receipts/{work_id}.json":
            raise WikisourceCampaignError(
                "campaign work receipt path is noncanonical"
            )
        return cls(
            work_id,
            _sha256(
                raw["pinned_work_spec_sha256"],
                "campaign work receipt.pinned_work_spec_sha256",
            ),
            output,
            _exact_int(
                raw["output_byte_size"],
                "campaign work receipt.output_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["output_sha256"],
                "campaign work receipt.output_sha256",
            ),
            _exact_int(
                raw["word_count"],
                "campaign work receipt.word_count",
                minimum=1,
            ),
            receipt_path,
            _sha256(
                raw["whole_work_receipt_self_hash"],
                "campaign work receipt.whole_work_receipt_self_hash",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class WikisourceCampaignReceipt:
    campaign_spec_sha256: str
    generation_id: str
    work_ids: tuple[str, ...]
    works: tuple[CampaignWorkReceipt, ...]
    fit_performed: bool
    confirmatory_authorized: bool
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        spec: WikisourceCampaignSpec,
        work_receipts: Sequence[WholeWorkReceipt],
    ) -> "WikisourceCampaignReceipt":
        if len(work_receipts) != len(spec.works):
            raise WikisourceCampaignError(
                "campaign receipt work count differs from campaign spec"
            )
        for work, receipt in zip(
            spec.works,
            work_receipts,
            strict=True,
        ):
            if (
                receipt.work_id != work.work_id
                or receipt.pinned_work_spec_sha256 != work.self_hash
                or receipt.output_relative_path != work.output_relative_path
                or receipt.output_byte_size != work.output_byte_size
                or receipt.output_sha256 != work.output_sha256
                or receipt.word_count != work.word_count
            ):
                raise WikisourceCampaignError(
                    "campaign work receipt differs from pinned work spec: "
                    f"{work.work_id}"
                )
        rows = [
            CampaignWorkReceipt.build(spec=work, receipt=receipt)
            for work, receipt in zip(
                spec.works,
                work_receipts,
                strict=True,
            )
        ]
        payload: dict[str, object] = {
            "schema_version": CAMPAIGN_RECEIPT_SCHEMA_VERSION,
            "campaign_kind": CAMPAIGN_KIND,
            "campaign_spec_sha256": spec.self_hash,
            "generation_id": spec.generation_id,
            "work_ids": list(spec.work_ids),
            "works": [row.to_dict() for row in rows],
            "fit_performed": False,
            "confirmatory_authorized": False,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "WikisourceCampaignReceipt":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "campaign_kind",
                "campaign_spec_sha256",
                "generation_id",
                "work_ids",
                "works",
                "fit_performed",
                "confirmatory_authorized",
                "self_hash",
            },
            "campaign receipt",
        )
        _self_hashed_payload(raw, "campaign receipt")
        if raw["schema_version"] != CAMPAIGN_RECEIPT_SCHEMA_VERSION:
            raise WikisourceCampaignError(
                "campaign receipt is legacy or unsupported"
            )
        if raw["campaign_kind"] != CAMPAIGN_KIND:
            raise WikisourceCampaignError(
                f"campaign receipt campaign_kind must be {CAMPAIGN_KIND!r}"
            )
        work_ids = tuple(
            _work_id(value, f"campaign receipt.work_ids[{index}]")
            for index, value in enumerate(
                _exact_list(
                    raw["work_ids"],
                    "campaign receipt.work_ids",
                    nonempty=True,
                )
            )
        )
        if work_ids != tuple(sorted(work_ids)) or len(work_ids) != len(
            set(work_ids)
        ):
            raise WikisourceCampaignError(
                "campaign receipt work_ids must be sorted and unique"
            )
        works = tuple(
            CampaignWorkReceipt.from_dict(item)
            for item in _exact_list(
                raw["works"],
                "campaign receipt.works",
                nonempty=True,
            )
        )
        if tuple(row.work_id for row in works) != work_ids:
            raise WikisourceCampaignError(
                "campaign receipt works do not match work_ids"
            )
        fit_performed = _exact_bool(
            raw["fit_performed"],
            "campaign receipt.fit_performed",
        )
        confirmatory = _exact_bool(
            raw["confirmatory_authorized"],
            "campaign receipt.confirmatory_authorized",
        )
        if fit_performed or confirmatory:
            raise WikisourceCampaignError(
                "source acquisition receipt cannot authorize or record a fit"
            )
        return cls(
            _sha256(
                raw["campaign_spec_sha256"],
                "campaign receipt.campaign_spec_sha256",
            ),
            _sha256(
                raw["generation_id"],
                "campaign receipt.generation_id",
            ),
            work_ids,
            works,
            fit_performed,
            confirmatory,
            _sha256(raw["self_hash"], "campaign receipt.self_hash"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CAMPAIGN_RECEIPT_SCHEMA_VERSION,
            "campaign_kind": CAMPAIGN_KIND,
            "campaign_spec_sha256": self.campaign_spec_sha256,
            "generation_id": self.generation_id,
            "work_ids": list(self.work_ids),
            "works": [row.to_dict() for row in self.works],
            "fit_performed": self.fit_performed,
            "confirmatory_authorized": self.confirmatory_authorized,
            "self_hash": self.self_hash,
        }

    def validate_for(
        self,
        spec: WikisourceCampaignSpec,
        work_receipts: Sequence[WholeWorkReceipt],
    ) -> "WikisourceCampaignReceipt":
        spec.validate()
        expected = WikisourceCampaignReceipt.build(
            spec=spec,
            work_receipts=work_receipts,
        )
        if self != expected:
            raise WikisourceCampaignError(
                "campaign receipt differs from spec and work receipts"
            )
        return self


def loads_campaign_receipt(text: str) -> WikisourceCampaignReceipt:
    try:
        return WikisourceCampaignReceipt.from_dict(loads_strict(text))
    except (StrictJSONError, TypeError) as exc:
        raise WikisourceCampaignError(f"campaign receipt: {exc}") from exc


def load_campaign_receipt(
    path: str | os.PathLike[str],
) -> WikisourceCampaignReceipt:
    try:
        return WikisourceCampaignReceipt.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise WikisourceCampaignError(f"campaign receipt: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class MaterializedCampaign:
    root: pathlib.Path
    receipt: WikisourceCampaignReceipt
    resumed: bool


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise WikisourceCampaignError(
                f"{label} must not contain symlink components: {component}"
            )


@contextlib.contextmanager
def _publication_lock(parent: pathlib.Path):
    lock = parent / ".wikisource-vnext-campaign.lock"
    if lock.is_symlink():
        raise WikisourceCampaignError(
            "campaign publication lock must not be a symlink"
        )
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


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
                raise WikisourceCampaignError(
                    f"symlink rejected in campaign namespace: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise WikisourceCampaignError(
                    f"special file rejected in campaign namespace: {relative}"
                )
    return files, directories


def _expected_inventory(
    spec: WikisourceCampaignSpec,
) -> tuple[set[str], set[str]]:
    files = {CAMPAIGN_RECEIPT_NAME}
    files.update(work.output_relative_path for work in spec.works)
    files.update(f"receipts/{work.work_id}.json" for work in spec.works)
    directories: set[str] = set()
    for relative in files:
        path = PurePosixPath(relative)
        parents = list(path.parents)
        directories.update(
            parent.as_posix()
            for parent in parents
            if parent.as_posix() != "."
        )
    return files, directories


def _load_existing(
    root: pathlib.Path,
    spec: WikisourceCampaignSpec,
) -> MaterializedCampaign:
    if root.is_symlink() or not root.is_dir():
        raise WikisourceCampaignError(
            f"campaign namespace is not a real directory: {root}"
        )
    observed_files, observed_directories = _tree_inventory(root)
    expected_files, expected_directories = _expected_inventory(spec)
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise WikisourceCampaignError(
            "campaign namespace has missing or extra files/directories"
        )
    work_receipts: list[WholeWorkReceipt] = []
    for work in spec.works:
        output_path = root.joinpath(
            *PurePosixPath(work.output_relative_path).parts
        )
        receipt_path = root.joinpath(
            *PurePosixPath(f"receipts/{work.work_id}.json").parts
        )
        try:
            output_payload = output_path.read_bytes()
            text = output_payload.decode("utf-8")
            if not text.endswith("\n") or text.endswith("\n\n"):
                raise WikisourceCampaignError(
                    "campaign output has noncanonical final newline: "
                    f"{work.work_id}"
                )
            receipt = load_whole_work_receipt(receipt_path)
            if receipt_path.read_text(encoding="utf-8") != _canonical_json_text(
                receipt.to_dict()
            ):
                raise WikisourceCampaignError(
                    "campaign whole-work receipt has noncanonical JSON bytes: "
                    f"{work.work_id}"
                )
            receipt.validate_for(work, output_payload)
        except WikisourceCampaignError:
            raise
        except (OSError, UnicodeError, WikisourceAcquisitionError) as exc:
            raise WikisourceCampaignError(
                f"campaign work {work.work_id!r} is corrupt: {exc}"
            ) from exc
        work_receipts.append(receipt)
    try:
        campaign_receipt = load_campaign_receipt(
            root / CAMPAIGN_RECEIPT_NAME
        )
        if (
            (root / CAMPAIGN_RECEIPT_NAME).read_text(encoding="utf-8")
            != _canonical_json_text(campaign_receipt.to_dict())
        ):
            raise WikisourceCampaignError(
                "campaign receipt has noncanonical JSON bytes"
            )
        campaign_receipt.validate_for(spec, work_receipts)
    except WikisourceCampaignError:
        raise
    except (OSError, UnicodeError, WikisourceAcquisitionError) as exc:
        raise WikisourceCampaignError(
            f"campaign receipt is corrupt: {exc}"
        ) from exc
    return MaterializedCampaign(root, campaign_receipt, True)


def materialize_campaign(
    spec: WikisourceCampaignSpec,
    *,
    output_parent: str | os.PathLike[str],
    transport: JSONTransport,
) -> MaterializedCampaign:
    """Fetch and atomically publish one exact, pinned campaign generation."""

    if type(spec) is not WikisourceCampaignSpec:
        raise WikisourceCampaignError(
            "campaign materialization requires exactly WikisourceCampaignSpec"
        )
    spec.validate()
    parent = pathlib.Path(output_parent)
    _reject_symlink_components(parent, label="campaign output parent")
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise WikisourceCampaignError(
            "campaign output parent must be a real directory"
        )
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    target = parent / spec.generation_id

    with _publication_lock(parent):
        if target.exists() or target.is_symlink():
            return _load_existing(target, spec)

    stage = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".wikisource-campaign.{spec.generation_id[:12]}.",
            dir=parent,
        )
    )
    work_cache = stage / ".single-work-cache"
    try:
        work_receipts: list[WholeWorkReceipt] = []
        for work in spec.works:
            try:
                materialized = materialize_pinned_work(
                    work,
                    output_parent=work_cache,
                    transport=transport,
                )
            except WikisourceAcquisitionError as exc:
                raise WikisourceCampaignError(
                    f"campaign work {work.work_id!r} was rejected: {exc}"
                ) from exc
            output_target = stage.joinpath(
                *PurePosixPath(work.output_relative_path).parts
            )
            output_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(materialized.output_path, output_target)
            receipt_target = stage.joinpath(
                *PurePosixPath(f"receipts/{work.work_id}.json").parts
            )
            dump_strict(
                materialized.receipt.to_dict(),
                receipt_target,
                sort_keys=True,
                trailing_newline=True,
            )
            materialized.receipt.validate_for(
                work,
                output_target.read_bytes(),
            )
            work_receipts.append(materialized.receipt)

        shutil.rmtree(work_cache)
        campaign_receipt = WikisourceCampaignReceipt.build(
            spec=spec,
            work_receipts=work_receipts,
        )
        dump_strict(
            campaign_receipt.to_dict(),
            stage / CAMPAIGN_RECEIPT_NAME,
            sort_keys=True,
            trailing_newline=True,
        )
        expected_files, expected_directories = _expected_inventory(spec)
        observed_files, observed_directories = _tree_inventory(stage)
        if (
            observed_files != expected_files
            or observed_directories != expected_directories
        ):
            raise WikisourceCampaignError(
                "staged campaign inventory is noncanonical"
            )
        staged = _load_existing(stage, spec)
        with _publication_lock(parent):
            if target.exists() or target.is_symlink():
                return _load_existing(target, spec)
            os.rename(stage, target)
        return MaterializedCampaign(target, staged.receipt, False)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


class HTTPJSONTransport:
    """Bounded stdlib HTTP JSON transport for the MediaWiki Action API."""

    def __init__(
        self,
        *,
        user_agent: str,
        api_url: str = API,
        timeout_seconds: float = 30.0,
        max_attempts: int = 6,
        backoff_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        max_response_bytes: int = 128 * 1024 * 1024,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        if (
            type(user_agent) is not str
            or not user_agent.strip()
            or "\r" in user_agent
            or "\n" in user_agent
        ):
            raise WikisourceCampaignError(
                "HTTP user_agent must be an exact non-empty single-line string"
            )
        if type(api_url) is not str or not api_url.startswith("https://"):
            raise WikisourceCampaignError(
                "HTTP api_url must be an explicit HTTPS URL"
            )
        if (
            type(timeout_seconds) not in {int, float}
            or type(timeout_seconds) is bool
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 300
        ):
            raise WikisourceCampaignError(
                "HTTP timeout_seconds must be finite and in (0, 300]"
            )
        if (
            type(max_attempts) is not int
            or not 1 <= max_attempts <= 20
        ):
            raise WikisourceCampaignError(
                "HTTP max_attempts must be an exact integer in [1, 20]"
            )
        for value, label, allow_zero in (
            (backoff_seconds, "backoff_seconds", True),
            (max_delay_seconds, "max_delay_seconds", False),
        ):
            if (
                type(value) not in {int, float}
                or type(value) is bool
                or not math.isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
                or value > 3600
            ):
                raise WikisourceCampaignError(
                    f"HTTP {label} must be a bounded finite number"
                )
        if (
            type(max_response_bytes) is not int
            or not 1 <= max_response_bytes <= 512 * 1024 * 1024
        ):
            raise WikisourceCampaignError(
                "HTTP max_response_bytes must be in [1, 512 MiB]"
            )
        self.user_agent = user_agent.strip()
        self.api_url = api_url
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max_attempts
        self.backoff_seconds = float(backoff_seconds)
        self.max_delay_seconds = float(max_delay_seconds)
        self.max_response_bytes = max_response_bytes
        self._opener = opener
        self._sleeper = sleeper
        self._now = now

    def _retry_after_seconds(
        self,
        value: str | None,
    ) -> float:
        if not value:
            return 0.0
        stripped = value.strip()
        try:
            return max(0.0, float(int(stripped, 10)))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(stripped)
            except (TypeError, ValueError, OverflowError):
                return 0.0
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.UTC)
            return max(
                0.0,
                (parsed.astimezone(dt.UTC) - self._now()).total_seconds(),
            )

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        exponential = self.backoff_seconds * (2**attempt)
        requested = self._retry_after_seconds(retry_after)
        return min(self.max_delay_seconds, max(exponential, requested))

    def __call__(self, params: Mapping[str, str]) -> object:
        if type(params) is not dict:
            params = dict(params)
        if not params or any(
            type(key) is not str or type(value) is not str
            for key, value in params.items()
        ):
            raise WikisourceCampaignError(
                "HTTP params must be a non-empty string-to-string mapping"
            )
        query = urllib.parse.urlencode(sorted(params.items()))
        request = urllib.request.Request(
            f"{self.api_url}?{query}",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            retry_after: str | None = None
            try:
                with self._opener(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    payload = response.read(self.max_response_bytes + 1)
                if len(payload) > self.max_response_bytes:
                    raise WikisourceCampaignError(
                        "HTTP JSON response exceeds configured byte limit"
                    )
                try:
                    return loads_strict(payload.decode("utf-8"))
                except (UnicodeDecodeError, StrictJSONError) as exc:
                    raise WikisourceCampaignError(
                        f"HTTP response is not strict UTF-8 JSON: {exc}"
                    ) from exc
            except urllib.error.HTTPError as exc:
                last_error = exc
                retry_after = (
                    exc.headers.get("Retry-After")
                    if exc.headers is not None
                    else None
                )
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if not retryable:
                    raise WikisourceCampaignError(
                        f"HTTP request rejected with status {exc.code}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
            if attempt + 1 >= self.max_attempts:
                break
            self._sleeper(self._delay(attempt, retry_after))
        raise WikisourceCampaignError(
            f"HTTP request failed after {self.max_attempts} bounded attempts"
        ) from last_error


__all__ = [
    "CAMPAIGN_KIND",
    "CAMPAIGN_RECEIPT_NAME",
    "CAMPAIGN_RECEIPT_SCHEMA_VERSION",
    "CAMPAIGN_SPEC_SCHEMA_VERSION",
    "CampaignWorkReceipt",
    "HTTPJSONTransport",
    "MaterializedCampaign",
    "WikisourceCampaignError",
    "WikisourceCampaignReceipt",
    "WikisourceCampaignSpec",
    "load_campaign_receipt",
    "load_campaign_spec",
    "loads_campaign_receipt",
    "loads_campaign_spec",
    "materialize_campaign",
]
