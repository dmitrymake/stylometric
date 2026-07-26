"""Verification of benchmark text artifacts against a validated manifest."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import re
import stat
from typing import Sequence

from .schema import BenchmarkDocument, BenchmarkManifest
from ..jsonio import canonical_hash


TOKENIZER_ID = "stylo_unicode_word_punct_v1"
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclasses.dataclass(frozen=True)
class TokenOffset:
    text: str
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class ArtifactCheck:
    doc_id: str
    path: str
    sha256: str
    n_bytes: int
    n_characters: int
    n_tokens: int
    n_offset_units: int


@dataclasses.dataclass(frozen=True)
class ArtifactReport:
    dataset: str
    version: str
    root: str
    documents: tuple[ArtifactCheck, ...]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def receipt_sha256(self) -> str:
        return canonical_hash(
            {
                "schema_version": "stylo.benchmark.artifact-report.v1",
                **self.to_dict(),
            }
        )


class ArtifactValidationError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__(
            "invalid benchmark artifacts:\n" + "\n".join(f"- {e}" for e in self.errors)
        )


def tokenize_with_offsets(text: str) -> tuple[TokenOffset, ...]:
    """Frozen Unicode word-or-punctuation tokenizer used by token manifests."""
    if type(text) is not str:
        raise TypeError("text must be str")
    return tuple(TokenOffset(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text))


def _resolve_text_path(root: pathlib.Path, document: BenchmarkDocument) -> pathlib.Path:
    if document.text_path is None:
        raise ValueError("text_path is required for a runnable benchmark package")
    relative = pathlib.PurePosixPath(document.text_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("text_path must be a safe relative POSIX path")
    candidate = root / pathlib.Path(*relative.parts)
    cursor = root
    if cursor.is_symlink():
        raise ValueError("benchmark root must not be a symlink")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"text_path contains a symlink: {document.text_path}")
    candidate = candidate.resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError("text_path escapes benchmark root")
    return candidate


def _read_regular_nofollow(path: pathlib.Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _check_span_partition(document: BenchmarkDocument, n_units: int) -> list[str]:
    if document.split == "blind":
        return []
    errors = []
    if not document.spans:
        return (
            ["mixed_authorship document must have a complete span partition"]
            if "mixed_authorship" in document.task_types
            else []
        )
    if document.spans[0].start != 0:
        errors.append("labelled spans must start at offset 0")
    for left, right in zip(document.spans[:-1], document.spans[1:]):
        if left.end != right.start:
            errors.append("labelled spans must be contiguous; use an unknown span for gaps")
            break
    if document.spans[-1].end != n_units:
        errors.append(
            f"labelled spans end at {document.spans[-1].end}, expected {n_units}"
        )
    return errors


def verify_manifest_artifacts(
    manifest: BenchmarkManifest,
    root: str | pathlib.Path,
) -> ArtifactReport:
    """Verify paths, UTF-8 bytes, hashes, offsets, and mixed-author coverage.

    The SHA-256 is computed over the exact packaged bytes, before decoding.
    Thus newline conversion or silent text cleaning invalidates the package.
    """
    base = pathlib.Path(root)
    errors: list[str] = []
    checks: list[ArtifactCheck] = []
    if base.is_symlink() or not base.exists() or not base.is_dir():
        raise ArtifactValidationError([f"benchmark root is not a directory: {base}"])

    for document in manifest.documents:
        prefix = document.doc_id
        try:
            path = _resolve_text_path(base, document)
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")
            continue
        if not path.exists() or not path.is_file():
            errors.append(f"{prefix}: text file does not exist: {document.text_path}")
            continue
        try:
            payload = _read_regular_nofollow(path)
        except OSError as exc:
            errors.append(f"{prefix}: cannot read {document.text_path}: {exc}")
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if digest != document.source.sha256:
            errors.append(
                f"{prefix}: sha256 mismatch: manifest={document.source.sha256}, actual={digest}"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"{prefix}: text is not valid UTF-8: {exc}")
            continue
        tokens = tokenize_with_offsets(text)
        if manifest.dataset.offset_unit == "token":
            if manifest.dataset.tokenizer != TOKENIZER_ID:
                errors.append(
                    f"{prefix}: unsupported frozen tokenizer {manifest.dataset.tokenizer!r}"
                )
                continue
            n_units = len(tokens)
        else:
            n_units = len(text)

        for index, span in enumerate(document.spans):
            if span.end > n_units:
                errors.append(
                    f"{prefix}: span {index} ends at {span.end}, beyond {n_units} "
                    f"{manifest.dataset.offset_unit} units"
                )
        errors.extend(f"{prefix}: {message}" for message in _check_span_partition(document, n_units))
        checks.append(
            ArtifactCheck(
                doc_id=document.doc_id,
                path=document.text_path or "",
                sha256=digest,
                n_bytes=len(payload),
                n_characters=len(text),
                n_tokens=len(tokens),
                n_offset_units=n_units,
            )
        )

    if errors:
        raise ArtifactValidationError(errors)
    return ArtifactReport(
        dataset=manifest.dataset.name,
        version=manifest.dataset.version,
        root=str(base.resolve()),
        documents=tuple(checks),
    )


def file_sha256(path: str | pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


__all__ = [
    "ArtifactCheck",
    "ArtifactReport",
    "ArtifactValidationError",
    "TOKENIZER_ID",
    "TokenOffset",
    "file_sha256",
    "tokenize_with_offsets",
    "verify_manifest_artifacts",
]
