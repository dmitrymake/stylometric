"""Pinned acquisition of the complete FEB ``История Пугачева`` body.

The RuAA R1 source page on Wikisource contains only a table of contents and
one chapter.  FEB exposes the complete 1978 academic text in one stable HTML
response.  This module pins that literal response and applies one narrow,
versioned extraction policy:

* decode strict Windows-1251;
* enter the FEB prose container;
* retain the preface, epigraph, and all eight narrative chapters;
* suppress page-number spans and footnote callouts; and
* stop before the separately headed authorial/editorial notes apparatus.

The policy is source-specific on purpose.  It does not guess headings or
silently generalise to arbitrary HTML.
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
import socket
import stat
import tempfile
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any, Protocol

from .._strict_fields import ExactFieldReader
from ..jsonio import (
    StrictJSONError,
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
)
from .wikisource_vnext import count_words


FEB_PINNED_WORK_SPEC_SCHEMA_VERSION = "stylo.feb.pinned-work-spec.v1"
FEB_WORK_RECEIPT_SCHEMA_VERSION = "stylo.feb.work-receipt.v1"
FEB_HTTP_POLICY_VERSION = "stylo.feb.https-identity-response.v1"
FEB_EXTRACTION_POLICY_VERSION = (
    "stylo.feb.pushkin-pugachev-main-narrative.v1"
)
FEB_SOURCE_HOST = "feb-web.ru"
FEB_SOURCE_URL = (
    "https://feb-web.ru/feb/pushkin/texts/push10/v08/"
    "d08-107.htm?cmd=2"
)
FEB_CONTENT_TYPE = "text/html; charset=windows-1251"
FEB_ENCODING = "windows-1251"
FEB_START_MARKER_ID = "ПРЕДИСЛОВИЕ"
FEB_END_MARKER_ID = "ПРИМЕЧАНИЯ"
FEB_PROSE_CONTAINER_ID = "prose"
FEB_RECEIPT_NAME = "receipt.json"
FEB_RESPONSE_RELATIVE_PATH = "source/feb-response.html"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_TAGS = frozenset(
    {
        "address",
        "blockquote",
        "br",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }
)


class FEBAcquisitionError(ValueError):
    """A FEB response, pinned spec, or immutable output is unsafe."""


@dataclasses.dataclass(frozen=True)
class FEBHTTPResponse:
    final_url: str
    status: int
    content_type: str
    body: bytes


class FEBBytesTransport(Protocol):
    def __call__(self, url: str) -> FEBHTTPResponse: ...


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_STRICT = ExactFieldReader(FEBAcquisitionError)
_exact_object = _STRICT.object
_exact_str = _STRICT.string
_exact_int = _STRICT.integer


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise FEBAcquisitionError(f"{label} must be an exact boolean")
    return value


_sha256 = _STRICT.sha256


def _work_id(value: object, label: str = "work_id") -> str:
    text = _exact_str(value, label)
    if "\\" in text:
        raise FEBAcquisitionError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise FEBAcquisitionError(
            f"{label} must be a canonical author/work identifier"
        )
    return text


def _canonical_json_text(value: object) -> str:
    return dumps_strict(value, indent=2, sort_keys=True) + "\n"


class _FEBMainNarrativeExtractor(HTMLParser):
    """Exact marker-driven extractor for the one pinned FEB document family."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._inside_prose = False
        self._capturing = False
        self._stopped = False
        self._suppressed_stack: list[str] = []
        self._pieces: list[str] = []
        self.prose_container_count = 0
        self.start_marker_count = 0
        self.end_marker_count = 0
        self.chapter_marker_ids: list[str] = []

    def _boundary(self) -> None:
        if self._capturing and self._pieces and self._pieces[-1] != "\n":
            self._pieces.append("\n")

    @staticmethod
    def _attrs(
        attrs: list[tuple[str, str | None]],
    ) -> dict[str, str]:
        return {key.lower(): value or "" for key, value in attrs}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        values = self._attrs(attrs)
        if (
            tag == "div"
            and values.get("id") == FEB_PROSE_CONTAINER_ID
        ):
            self.prose_container_count += 1
            self._inside_prose = True
        if not self._inside_prose or self._stopped:
            return
        if tag == "h4" and values.get("id") == FEB_START_MARKER_ID:
            self.start_marker_count += 1
            self._capturing = True
        elif tag == "h4" and values.get("id") == FEB_END_MARKER_ID:
            self.end_marker_count += 1
            self._capturing = False
            self._stopped = True
            return
        marker = values.get("id", "")
        if tag == "h4" and marker.startswith("ГЛАВА_"):
            self.chapter_marker_ids.append(marker)
        if not self._capturing:
            return
        if self._suppressed_stack:
            if tag not in _VOID_TAGS:
                self._suppressed_stack.append(tag)
            return
        classes = set(values.get("class", "").split())
        if (
            tag in {"script", "style", "noscript", "sup"}
            or (tag == "span" and "page" in classes)
            or "page-note" in classes
            or "footnote" in classes
        ):
            if tag not in _VOID_TAGS:
                self._suppressed_stack.append(tag)
            return
        if tag in _BLOCK_TAGS:
            self._boundary()

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if (
            self._inside_prose
            and self._capturing
            and not self._stopped
            and not self._suppressed_stack
            and tag.lower() in _BLOCK_TAGS
        ):
            self._boundary()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._suppressed_stack:
            for index in range(len(self._suppressed_stack) - 1, -1, -1):
                if self._suppressed_stack[index] == tag:
                    del self._suppressed_stack[index:]
                    return
            return
        if self._inside_prose and self._capturing and tag in _BLOCK_TAGS:
            self._boundary()

    def handle_data(self, data: str) -> None:
        if (
            self._inside_prose
            and self._capturing
            and not self._stopped
            and not self._suppressed_stack
            and data
        ):
            self._pieces.append(data)

    def selected_text(self) -> str:
        if self.prose_container_count != 1:
            raise FEBAcquisitionError(
                "FEB response must contain exactly one #prose container"
            )
        if self.start_marker_count != 1 or self.end_marker_count != 1:
            raise FEBAcquisitionError(
                "FEB response must contain exact narrative start/end markers"
            )
        expected_chapters = [
            "ГЛАВА_ПЕРВАЯ",
            "ГЛАВА_ВТОРАЯ",
            "ГЛАВА_ТРЕТИЯ",
            "ГЛАВА_ЧЕТВЕРТАЯ",
            "ГЛАВА_ПЯТАЯ",
            "ГЛАВА_ШЕСТАЯ",
            "ГЛАВА_СЕДЬМАЯ",
            "ГЛАВА_ОСЬМАЯ",
        ]
        if self.chapter_marker_ids != expected_chapters:
            raise FEBAcquisitionError(
                "FEB response does not contain the exact ordered eight chapters"
            )
        text = "".join(self._pieces)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = (
            text.replace("\xa0", " ")
            .replace("\u202f", " ")
            .replace("\ufeff", "")
        )
        lines: list[str] = []
        previous_blank = True
        for raw_line in text.split("\n"):
            line = re.sub(r"[\t\f\v ]+", " ", raw_line).strip()
            if line:
                lines.append(line)
                previous_blank = False
            elif not previous_blank:
                lines.append("")
                previous_blank = True
        while lines and not lines[-1]:
            lines.pop()
        selected = "\n".join(lines).strip()
        if not selected or count_words(selected) < 200:
            raise FEBAcquisitionError(
                "FEB narrative extraction produced no substantial prose"
            )
        return selected


def extract_feb_main_narrative(response_body: bytes) -> str:
    """Return exact normalized main narrative from pinned FEB response bytes."""

    if type(response_body) is not bytes or not response_body:
        raise FEBAcquisitionError(
            "FEB response body must be exact non-empty bytes"
        )
    try:
        html = response_body.decode(FEB_ENCODING, errors="strict")
    except UnicodeDecodeError as exc:
        raise FEBAcquisitionError(
            f"FEB response is not strict {FEB_ENCODING}: {exc}"
        ) from exc
    parser = _FEBMainNarrativeExtractor()
    try:
        parser.feed(html)
        parser.close()
    except FEBAcquisitionError:
        raise
    except Exception as exc:
        raise FEBAcquisitionError(f"cannot parse FEB HTML: {exc}") from exc
    return parser.selected_text()


def _spec_core(
    *,
    work_id: str,
    response_byte_size: int,
    response_sha256: str,
    output_byte_size: int,
    output_sha256: str,
    word_count: int,
    first_nonblank_line_sha256: str,
    last_nonblank_line_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": FEB_PINNED_WORK_SPEC_SCHEMA_VERSION,
        "provider": FEB_SOURCE_HOST,
        "work_id": work_id,
        "source_url": FEB_SOURCE_URL,
        "http_policy_version": FEB_HTTP_POLICY_VERSION,
        "response_content_type": FEB_CONTENT_TYPE,
        "response_encoding": FEB_ENCODING,
        "response_relative_path": FEB_RESPONSE_RELATIVE_PATH,
        "response_byte_size": response_byte_size,
        "response_sha256": response_sha256,
        "extraction_policy_version": FEB_EXTRACTION_POLICY_VERSION,
        "prose_container_id": FEB_PROSE_CONTAINER_ID,
        "start_marker_id": FEB_START_MARKER_ID,
        "end_marker_id": FEB_END_MARKER_ID,
        "output_relative_path": f"raw/{work_id}.txt",
        "output_byte_size": output_byte_size,
        "output_sha256": output_sha256,
        "word_count": word_count,
        "first_nonblank_line_sha256": first_nonblank_line_sha256,
        "last_nonblank_line_sha256": last_nonblank_line_sha256,
    }


@dataclasses.dataclass(frozen=True)
class PinnedFEBWorkSpec:
    work_id: str
    response_byte_size: int
    response_sha256: str
    output_byte_size: int
    output_sha256: str
    word_count: int
    first_nonblank_line_sha256: str
    last_nonblank_line_sha256: str
    generation_id: str
    self_hash: str

    @classmethod
    def build(
        cls,
        *,
        work_id: str,
        response_body: bytes,
    ) -> "PinnedFEBWorkSpec":
        work = _work_id(work_id)
        selected = extract_feb_main_narrative(response_body)
        output = (selected + "\n").encode("utf-8")
        nonblank = [line for line in selected.split("\n") if line]
        core = _spec_core(
            work_id=work,
            response_byte_size=len(response_body),
            response_sha256=_sha256_bytes(response_body),
            output_byte_size=len(output),
            output_sha256=_sha256_bytes(output),
            word_count=count_words(selected),
            first_nonblank_line_sha256=_sha256_bytes(
                nonblank[0].encode("utf-8")
            ),
            last_nonblank_line_sha256=_sha256_bytes(
                nonblank[-1].encode("utf-8")
            ),
        )
        generation_id = canonical_hash(core)
        payload = {**core, "generation_id": generation_id}
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "PinnedFEBWorkSpec":
        keys = {
            "schema_version",
            "provider",
            "work_id",
            "source_url",
            "http_policy_version",
            "response_content_type",
            "response_encoding",
            "response_relative_path",
            "response_byte_size",
            "response_sha256",
            "extraction_policy_version",
            "prose_container_id",
            "start_marker_id",
            "end_marker_id",
            "output_relative_path",
            "output_byte_size",
            "output_sha256",
            "word_count",
            "first_nonblank_line_sha256",
            "last_nonblank_line_sha256",
            "generation_id",
            "self_hash",
        }
        raw = _exact_object(value, keys, "pinned FEB work spec")
        expected_scalars = {
            "schema_version": FEB_PINNED_WORK_SPEC_SCHEMA_VERSION,
            "provider": FEB_SOURCE_HOST,
            "source_url": FEB_SOURCE_URL,
            "http_policy_version": FEB_HTTP_POLICY_VERSION,
            "response_content_type": FEB_CONTENT_TYPE,
            "response_encoding": FEB_ENCODING,
            "response_relative_path": FEB_RESPONSE_RELATIVE_PATH,
            "extraction_policy_version": FEB_EXTRACTION_POLICY_VERSION,
            "prose_container_id": FEB_PROSE_CONTAINER_ID,
            "start_marker_id": FEB_START_MARKER_ID,
            "end_marker_id": FEB_END_MARKER_ID,
        }
        for key, expected in expected_scalars.items():
            if raw[key] != expected:
                raise FEBAcquisitionError(
                    f"pinned FEB work spec {key} must be {expected!r}"
                )
        recorded = _sha256(raw["self_hash"], "pinned FEB work spec.self_hash")
        payload = {key: item for key, item in raw.items() if key != "self_hash"}
        if canonical_hash(payload) != recorded:
            raise FEBAcquisitionError("pinned FEB work spec self_hash mismatch")
        work = _work_id(raw["work_id"])
        if raw["output_relative_path"] != f"raw/{work}.txt":
            raise FEBAcquisitionError(
                "pinned FEB output path must be exactly raw/<work_id>.txt"
            )
        response_size = _exact_int(
            raw["response_byte_size"],
            "pinned FEB work spec.response_byte_size",
            minimum=1,
        )
        response_sha = _sha256(
            raw["response_sha256"],
            "pinned FEB work spec.response_sha256",
        )
        output_size = _exact_int(
            raw["output_byte_size"],
            "pinned FEB work spec.output_byte_size",
            minimum=1,
        )
        output_sha = _sha256(
            raw["output_sha256"],
            "pinned FEB work spec.output_sha256",
        )
        words = _exact_int(
            raw["word_count"],
            "pinned FEB work spec.word_count",
            minimum=1,
        )
        first = _sha256(
            raw["first_nonblank_line_sha256"],
            "pinned FEB work spec.first_nonblank_line_sha256",
        )
        last = _sha256(
            raw["last_nonblank_line_sha256"],
            "pinned FEB work spec.last_nonblank_line_sha256",
        )
        generation = _sha256(
            raw["generation_id"],
            "pinned FEB work spec.generation_id",
        )
        core = _spec_core(
            work_id=work,
            response_byte_size=response_size,
            response_sha256=response_sha,
            output_byte_size=output_size,
            output_sha256=output_sha,
            word_count=words,
            first_nonblank_line_sha256=first,
            last_nonblank_line_sha256=last,
        )
        if canonical_hash(core) != generation:
            raise FEBAcquisitionError(
                "pinned FEB work spec generation_id mismatch"
            )
        return cls(
            work,
            response_size,
            response_sha,
            output_size,
            output_sha,
            words,
            first,
            last,
            generation,
            recorded,
        )

    def to_dict(self) -> dict[str, object]:
        core = _spec_core(
            work_id=self.work_id,
            response_byte_size=self.response_byte_size,
            response_sha256=self.response_sha256,
            output_byte_size=self.output_byte_size,
            output_sha256=self.output_sha256,
            word_count=self.word_count,
            first_nonblank_line_sha256=self.first_nonblank_line_sha256,
            last_nonblank_line_sha256=self.last_nonblank_line_sha256,
        )
        return {
            **core,
            "generation_id": self.generation_id,
            "self_hash": self.self_hash,
        }

    def validate(self) -> "PinnedFEBWorkSpec":
        if PinnedFEBWorkSpec.from_dict(self.to_dict()) != self:
            raise FEBAcquisitionError("pinned FEB work spec is noncanonical")
        return self


def load_pinned_feb_work_spec(
    path: str | os.PathLike[str],
) -> PinnedFEBWorkSpec:
    try:
        return PinnedFEBWorkSpec.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise FEBAcquisitionError(f"pinned FEB work spec: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class FEBWorkReceipt:
    pinned_work_spec_sha256: str
    generation_id: str
    work_id: str
    response_byte_size: int
    response_sha256: str
    output_byte_size: int
    output_sha256: str
    word_count: int
    fit_performed: bool
    confirmatory_authorized: bool
    self_hash: str

    @classmethod
    def build(cls, spec: PinnedFEBWorkSpec) -> "FEBWorkReceipt":
        payload: dict[str, object] = {
            "schema_version": FEB_WORK_RECEIPT_SCHEMA_VERSION,
            "pinned_work_spec_sha256": spec.self_hash,
            "generation_id": spec.generation_id,
            "work_id": spec.work_id,
            "response_byte_size": spec.response_byte_size,
            "response_sha256": spec.response_sha256,
            "output_byte_size": spec.output_byte_size,
            "output_sha256": spec.output_sha256,
            "word_count": spec.word_count,
            "fit_performed": False,
            "confirmatory_authorized": False,
        }
        return cls.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )

    @classmethod
    def from_dict(cls, value: object) -> "FEBWorkReceipt":
        raw = _exact_object(
            value,
            {
                "schema_version",
                "pinned_work_spec_sha256",
                "generation_id",
                "work_id",
                "response_byte_size",
                "response_sha256",
                "output_byte_size",
                "output_sha256",
                "word_count",
                "fit_performed",
                "confirmatory_authorized",
                "self_hash",
            },
            "FEB work receipt",
        )
        if raw["schema_version"] != FEB_WORK_RECEIPT_SCHEMA_VERSION:
            raise FEBAcquisitionError("FEB work receipt schema is unsupported")
        recorded = _sha256(raw["self_hash"], "FEB work receipt.self_hash")
        payload = {key: item for key, item in raw.items() if key != "self_hash"}
        if canonical_hash(payload) != recorded:
            raise FEBAcquisitionError("FEB work receipt self_hash mismatch")
        fit = _exact_bool(raw["fit_performed"], "FEB work receipt.fit_performed")
        confirmatory = _exact_bool(
            raw["confirmatory_authorized"],
            "FEB work receipt.confirmatory_authorized",
        )
        if fit or confirmatory:
            raise FEBAcquisitionError(
                "FEB source receipt cannot authorize or record model fit"
            )
        return cls(
            _sha256(
                raw["pinned_work_spec_sha256"],
                "FEB work receipt.pinned_work_spec_sha256",
            ),
            _sha256(raw["generation_id"], "FEB work receipt.generation_id"),
            _work_id(raw["work_id"], "FEB work receipt.work_id"),
            _exact_int(
                raw["response_byte_size"],
                "FEB work receipt.response_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["response_sha256"],
                "FEB work receipt.response_sha256",
            ),
            _exact_int(
                raw["output_byte_size"],
                "FEB work receipt.output_byte_size",
                minimum=1,
            ),
            _sha256(
                raw["output_sha256"],
                "FEB work receipt.output_sha256",
            ),
            _exact_int(
                raw["word_count"],
                "FEB work receipt.word_count",
                minimum=1,
            ),
            fit,
            confirmatory,
            recorded,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": FEB_WORK_RECEIPT_SCHEMA_VERSION,
            "pinned_work_spec_sha256": self.pinned_work_spec_sha256,
            "generation_id": self.generation_id,
            "work_id": self.work_id,
            "response_byte_size": self.response_byte_size,
            "response_sha256": self.response_sha256,
            "output_byte_size": self.output_byte_size,
            "output_sha256": self.output_sha256,
            "word_count": self.word_count,
            "fit_performed": self.fit_performed,
            "confirmatory_authorized": self.confirmatory_authorized,
            "self_hash": self.self_hash,
        }

    def validate_for(
        self,
        spec: PinnedFEBWorkSpec,
        *,
        response_body: bytes,
        output_body: bytes,
    ) -> "FEBWorkReceipt":
        spec.validate()
        if self != FEBWorkReceipt.build(spec):
            raise FEBAcquisitionError(
                "FEB work receipt differs from pinned spec"
            )
        if (
            len(response_body) != spec.response_byte_size
            or _sha256_bytes(response_body) != spec.response_sha256
        ):
            raise FEBAcquisitionError(
                "FEB response bytes differ from pinned spec"
            )
        if (
            len(output_body) != spec.output_byte_size
            or _sha256_bytes(output_body) != spec.output_sha256
        ):
            raise FEBAcquisitionError(
                "FEB output bytes differ from pinned spec"
            )
        selected = extract_feb_main_narrative(response_body)
        expected_output = (selected + "\n").encode("utf-8")
        if expected_output != output_body:
            raise FEBAcquisitionError(
                "FEB output is not deterministic extraction of response"
            )
        return self


def load_feb_work_receipt(
    path: str | os.PathLike[str],
) -> FEBWorkReceipt:
    try:
        return FEBWorkReceipt.from_dict(load_strict(path))
    except (StrictJSONError, TypeError, OSError, UnicodeError) as exc:
        raise FEBAcquisitionError(f"FEB work receipt: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class MaterializedFEBWork:
    root: pathlib.Path
    output_path: pathlib.Path
    receipt: FEBWorkReceipt
    resumed: bool


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise FEBAcquisitionError(
                f"{label} must not contain symlink components: {component}"
            )


@contextlib.contextmanager
def _publication_lock(parent: pathlib.Path):
    lock = parent / ".feb-vnext.lock"
    if lock.is_symlink():
        raise FEBAcquisitionError("FEB publication lock must not be a symlink")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _expected_files(spec: PinnedFEBWorkSpec) -> set[str]:
    return {
        FEB_RECEIPT_NAME,
        FEB_RESPONSE_RELATIVE_PATH,
        f"raw/{spec.work_id}.txt",
    }


def _tree_files(root: pathlib.Path) -> set[str]:
    files: set[str] = set()
    for directory, directory_names, file_names in os.walk(root):
        base = pathlib.Path(directory)
        for name in tuple(directory_names) + tuple(file_names):
            path = base / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise FEBAcquisitionError(
                    f"symlink rejected in FEB namespace: {relative}"
                )
            if name in file_names:
                if not stat.S_ISREG(metadata.st_mode):
                    raise FEBAcquisitionError(
                        f"special file rejected in FEB namespace: {relative}"
                    )
                files.add(relative)
    return files


def _load_existing(
    root: pathlib.Path,
    spec: PinnedFEBWorkSpec,
) -> MaterializedFEBWork:
    if root.is_symlink() or not root.is_dir():
        raise FEBAcquisitionError("FEB namespace must be a real directory")
    if _tree_files(root) != _expected_files(spec):
        raise FEBAcquisitionError(
            "FEB namespace has missing or extra files"
        )
    response = root / FEB_RESPONSE_RELATIVE_PATH
    output = root.joinpath(
        *PurePosixPath(f"raw/{spec.work_id}.txt").parts
    )
    receipt_path = root / FEB_RECEIPT_NAME
    receipt = load_feb_work_receipt(receipt_path)
    if receipt_path.read_text(encoding="utf-8") != _canonical_json_text(
        receipt.to_dict()
    ):
        raise FEBAcquisitionError("FEB receipt JSON bytes are noncanonical")
    receipt.validate_for(
        spec,
        response_body=response.read_bytes(),
        output_body=output.read_bytes(),
    )
    return MaterializedFEBWork(root, output, receipt, True)


def materialize_pinned_feb_work(
    spec: PinnedFEBWorkSpec,
    *,
    output_parent: str | os.PathLike[str],
    transport: FEBBytesTransport,
) -> MaterializedFEBWork:
    """Fetch and immutably publish one exact pinned FEB work."""

    if type(spec) is not PinnedFEBWorkSpec:
        raise FEBAcquisitionError(
            "FEB materialization requires exactly PinnedFEBWorkSpec"
        )
    spec.validate()
    parent = pathlib.Path(output_parent)
    _reject_symlink_components(parent, label="FEB output parent")
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise FEBAcquisitionError("FEB output parent must be a real directory")
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    target = parent / spec.generation_id
    with _publication_lock(parent):
        if target.exists() or target.is_symlink():
            return _load_existing(target, spec)

    response = transport(FEB_SOURCE_URL)
    if type(response) is not FEBHTTPResponse:
        raise FEBAcquisitionError(
            "FEB transport must return exactly FEBHTTPResponse"
        )
    if (
        response.final_url != FEB_SOURCE_URL
        or response.status != 200
        or response.content_type.lower() != FEB_CONTENT_TYPE
    ):
        raise FEBAcquisitionError(
            "FEB response URL/status/content-type differs from pinned contract"
        )
    if (
        len(response.body) != spec.response_byte_size
        or _sha256_bytes(response.body) != spec.response_sha256
    ):
        raise FEBAcquisitionError("live FEB response bytes differ from pin")
    selected = extract_feb_main_narrative(response.body)
    output_body = (selected + "\n").encode("utf-8")
    if (
        len(output_body) != spec.output_byte_size
        or _sha256_bytes(output_body) != spec.output_sha256
    ):
        raise FEBAcquisitionError("live FEB extracted text differs from pin")

    stage = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".feb.{spec.generation_id[:12]}.",
            dir=parent,
        )
    )
    try:
        response_path = stage / FEB_RESPONSE_RELATIVE_PATH
        response_path.parent.mkdir(parents=True)
        response_path.write_bytes(response.body)
        output_path = stage.joinpath(
            *PurePosixPath(f"raw/{spec.work_id}.txt").parts
        )
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(output_body)
        receipt = FEBWorkReceipt.build(spec)
        dump_strict(
            receipt.to_dict(),
            stage / FEB_RECEIPT_NAME,
            sort_keys=True,
            trailing_newline=True,
        )
        staged = _load_existing(stage, spec)
        with _publication_lock(parent):
            if target.exists() or target.is_symlink():
                existing = _load_existing(target, spec)
                shutil.rmtree(stage)
                return existing
            os.rename(stage, target)
        return dataclasses.replace(
            staged,
            root=target,
            output_path=target.joinpath(
                *PurePosixPath(f"raw/{spec.work_id}.txt").parts
            ),
            resumed=False,
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


class FEBHTTPTransport:
    """Bounded HTTPS transport with redirects rejected and no compression."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 30.0,
        max_attempts: int = 6,
    ) -> None:
        if (
            type(user_agent) is not str
            or not user_agent
            or "\r" in user_agent
            or "\n" in user_agent
        ):
            raise FEBAcquisitionError(
                "FEB User-Agent must be an exact non-empty single line"
            )
        if (
            type(timeout_seconds) not in {int, float}
            or type(timeout_seconds) is bool
            or not 0 < timeout_seconds <= 300
        ):
            raise FEBAcquisitionError(
                "FEB timeout_seconds must be in (0, 300]"
            )
        if type(max_attempts) is not int or not 1 <= max_attempts <= 20:
            raise FEBAcquisitionError(
                "FEB max_attempts must be an exact integer in [1, 20]"
            )
        self.user_agent = user_agent
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max_attempts

    def __call__(self, url: str) -> FEBHTTPResponse:
        if url != FEB_SOURCE_URL:
            raise FEBAcquisitionError("FEB transport URL is not approved")
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    final_url = response.geturl()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "")
                    body = response.read()
                return FEBHTTPResponse(
                    final_url,
                    status,
                    content_type.lower(),
                    body,
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 and not 500 <= exc.code <= 599:
                    raise FEBAcquisitionError(
                        f"FEB HTTP request failed with status {exc.code}"
                    ) from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
            ) as exc:
                last_error = exc
            if attempt + 1 < self.max_attempts:
                time.sleep(min(2.0**attempt, 30.0))
        raise FEBAcquisitionError(
            f"FEB HTTP request failed after {self.max_attempts} attempts: "
            f"{last_error}"
        )


__all__ = [
    "FEBAcquisitionError",
    "FEBBytesTransport",
    "FEBHTTPResponse",
    "FEBHTTPTransport",
    "FEB_CONTENT_TYPE",
    "FEB_ENCODING",
    "FEB_END_MARKER_ID",
    "FEB_EXTRACTION_POLICY_VERSION",
    "FEB_HTTP_POLICY_VERSION",
    "FEB_PINNED_WORK_SPEC_SCHEMA_VERSION",
    "FEB_PROSE_CONTAINER_ID",
    "FEB_RECEIPT_NAME",
    "FEB_RESPONSE_RELATIVE_PATH",
    "FEB_SOURCE_URL",
    "FEB_START_MARKER_ID",
    "FEB_WORK_RECEIPT_SCHEMA_VERSION",
    "FEBWorkReceipt",
    "MaterializedFEBWork",
    "PinnedFEBWorkSpec",
    "extract_feb_main_narrative",
    "load_feb_work_receipt",
    "load_pinned_feb_work_spec",
    "materialize_pinned_feb_work",
]
