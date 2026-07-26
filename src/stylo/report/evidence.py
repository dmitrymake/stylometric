"""Strict provenance envelopes for every section consumed by the HTML report."""
from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import tempfile
from collections.abc import Mapping

from ..jsonio import (
    artifact_self_hash,
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
)

SECTION_SCHEMA = "stylo.report-section-evidence.v1"
CORPUS_SECTION = "corpus_validation"
PREDICTION_SECTION = "prediction"


class SectionEvidenceError(RuntimeError):
    """A report section is missing, stale, or not byte/identity bound."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: pathlib.Path, *, label: str) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SectionEvidenceError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SectionEvidenceError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _code_tree_sha256() -> str:
    from ..pipeline.train import _code_tree_sha256 as compute

    value = compute()
    if not isinstance(value, str) or len(value) != 64:
        raise SectionEvidenceError("cannot attest the current Stylo code tree")
    return value


def _config_id(cfg) -> str:
    return _sha256(dumps_strict(cfg.to_dict(), sort_keys=True).encode("utf-8"))


def _current_fragment_identity(cfg) -> dict[str, str]:
    from ..dataset import resolve_fragment_roots

    snapshot = resolve_fragment_roots(cfg)
    return {
        "fragment_generation_id": snapshot.generation_id,
        "fragment_root": str(snapshot.root.resolve()),
        "unknown_root": str(snapshot.unknown_root.resolve()),
    }


def directory_digest(root: str | pathlib.Path) -> str:
    """Hash an exact regular-file tree; reject symlinks and special entries."""

    base = pathlib.Path(root)
    if base.is_symlink() or not base.is_dir():
        raise SectionEvidenceError(f"evidence input root must be a real directory: {base}")
    inventory: list[dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        directory = pathlib.Path(dirpath)
        for name in dirnames:
            child = directory / name
            if child.is_symlink():
                raise SectionEvidenceError(f"symlink in evidence input tree: {child}")
        for name in filenames:
            child = directory / name
            if child.is_symlink() or not child.is_file():
                raise SectionEvidenceError(
                    f"non-regular evidence input member: {child}"
                )
            inventory.append(
                {
                    "path": child.relative_to(base).as_posix(),
                    "sha256": _sha256(_read_regular(child, label="evidence input")),
                }
            )
    if not inventory:
        raise SectionEvidenceError(f"evidence input tree is empty: {base}")
    return canonical_hash(sorted(inventory, key=lambda item: item["path"]))


def _safe_docs(docs: pathlib.Path) -> None:
    if docs.is_symlink() or not docs.is_dir():
        raise SectionEvidenceError(f"docs root must be a real directory: {docs}")


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    if path.is_symlink():
        raise SectionEvidenceError(f"report section target must not be a symlink: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_section(
    docs: pathlib.Path,
    *,
    section: str,
    files: Mapping[str, str],
    identity: dict[str, object],
) -> None:
    _safe_docs(docs)
    if not files or any(
        pathlib.PurePosixPath(name).name != name or not name for name in files
    ):
        raise SectionEvidenceError("report section filenames are unsafe")
    for name, text in sorted(files.items()):
        if type(text) is not str:
            raise SectionEvidenceError("report section payloads must be exact text")
        _atomic_write_text(docs / name, text)
    envelope: dict[str, object] = {
        "schema_version": SECTION_SCHEMA,
        "section": section,
        "identity": identity,
        "files": {
            name: _sha256(text.encode("utf-8"))
            for name, text in sorted(files.items())
        },
    }
    envelope["self_hash"] = artifact_self_hash(envelope)
    dump_strict(
        envelope,
        docs / f"{section}.evidence.json",
        sort_keys=True,
    )


def _verify_section(
    docs: pathlib.Path,
    *,
    section: str,
    expected_files: set[str],
) -> tuple[dict[str, object], dict[str, str]]:
    _safe_docs(docs)
    manifest_path = docs / f"{section}.evidence.json"
    try:
        envelope = load_strict(manifest_path)
    except Exception as exc:
        raise SectionEvidenceError(
            f"invalid or missing {section} evidence envelope: {exc}"
        ) from exc
    if type(envelope) is not dict or set(envelope) != {
        "schema_version",
        "section",
        "identity",
        "files",
        "self_hash",
    }:
        raise SectionEvidenceError(f"{section} evidence field mismatch")
    if (
        envelope["schema_version"] != SECTION_SCHEMA
        or envelope["section"] != section
        or type(envelope["identity"]) is not dict
        or type(envelope["files"]) is not dict
        or envelope["self_hash"] != artifact_self_hash(envelope)
    ):
        raise SectionEvidenceError(f"{section} evidence identity/self-hash mismatch")
    if set(envelope["files"]) != expected_files:
        raise SectionEvidenceError(f"{section} evidence file inventory mismatch")
    bodies: dict[str, str] = {}
    for name, expected in envelope["files"].items():
        if type(expected) is not str or len(expected) != 64:
            raise SectionEvidenceError(f"{section} evidence digest is malformed")
        payload = _read_regular(docs / name, label=f"{section} report section")
        if _sha256(payload) != expected:
            raise SectionEvidenceError(f"{section} report section hash mismatch: {name}")
        try:
            bodies[name] = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SectionEvidenceError(f"{section} report section is not UTF-8") from exc
    return envelope["identity"], bodies


def publish_corpus_validation(
    cfg,
    *,
    corpus_root: pathlib.Path,
    text: str,
    structured: dict,
) -> None:
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    identity = {
        "corpus_sha256": directory_digest(corpus_root),
        "config_id": _config_id(cfg),
        "code_tree_sha256": _code_tree_sha256(),
    }
    _publish_section(
        docs,
        section=CORPUS_SECTION,
        files={
            "corpus_validation.json": dumps_strict(structured),
            "corpus_validation.txt": text,
        },
        identity=identity,
    )


def verify_corpus_validation(cfg) -> str:
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    identity, bodies = _verify_section(
        docs,
        section=CORPUS_SECTION,
        expected_files={"corpus_validation.json", "corpus_validation.txt"},
    )
    expected = {
        "corpus_sha256": directory_digest(
            pathlib.Path(cfg.get_path("paths.input_clean", "input_clean"))
        ),
        "config_id": _config_id(cfg),
        "code_tree_sha256": _code_tree_sha256(),
    }
    if identity != expected:
        raise SectionEvidenceError("corpus validation evidence is stale")
    return bodies["corpus_validation.txt"]


def publish_prediction(
    cfg,
    *,
    unknown_root: pathlib.Path,
    report: str,
    bundle_token: str,
    bundle_meta: dict,
) -> None:
    current_config = _config_id(cfg)
    current_code = _code_tree_sha256()
    if bundle_meta.get("config_id") != current_config:
        raise SectionEvidenceError(
            "prediction bundle config does not match the current resolved config"
        )
    if bundle_meta.get("code_tree_sha256") != current_code:
        raise SectionEvidenceError(
            "prediction bundle code tree does not match the executing code"
        )
    fragment_identity = _current_fragment_identity(cfg)
    if str(unknown_root.resolve()) != fragment_identity["unknown_root"]:
        raise SectionEvidenceError(
            "canonical prediction evidence requires the current fragment "
            "snapshot unknown root"
        )
    identity = {
        "bundle_token": bundle_token,
        "bundle_manifest_sha256": canonical_hash(bundle_meta),
        "bundle_rows_digest": bundle_meta.get("rows_digest"),
        "config_id": current_config,
        "code_tree_sha256": current_code,
        **fragment_identity,
        "unknown_sha256": directory_digest(unknown_root),
    }
    _publish_section(
        pathlib.Path(cfg.get_path("paths.docs", "docs")),
        section=PREDICTION_SECTION,
        files={"prediction.txt": report},
        identity=identity,
    )


def verify_prediction(cfg) -> str:
    from ..domain.work_weighting import CHUNK_WEIGHTED_LEGACY
    from ..pipeline.bundle import load_bundle

    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    identity, bodies = _verify_section(
        docs,
        section=PREDICTION_SECTION,
        expected_files={"prediction.txt"},
    )
    expected_fields = {
        "bundle_token",
        "bundle_manifest_sha256",
        "bundle_rows_digest",
        "config_id",
        "code_tree_sha256",
        "fragment_generation_id",
        "fragment_root",
        "unknown_root",
        "unknown_sha256",
    }
    if set(identity) != expected_fields:
        raise SectionEvidenceError("prediction evidence identity field mismatch")
    if (
        identity["config_id"] != _config_id(cfg)
        or identity["code_tree_sha256"] != _code_tree_sha256()
        or any(
            type(identity[field]) is not str
            for field in (
                "fragment_generation_id",
                "fragment_root",
                "unknown_root",
            )
        )
    ):
        raise SectionEvidenceError("prediction evidence code/config identity is stale")
    current_fragment = _current_fragment_identity(cfg)
    if {
        field: identity[field]
        for field in (
            "fragment_generation_id",
            "fragment_root",
            "unknown_root",
        )
    } != current_fragment:
        raise SectionEvidenceError(
            "prediction evidence is stale for the current fragment snapshot"
        )
    unknown_root = pathlib.Path(current_fragment["unknown_root"])
    if directory_digest(unknown_root) != identity["unknown_sha256"]:
        raise SectionEvidenceError("prediction unknown input has drifted")
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    meta, _paths = load_bundle(
        data / "deployment" / CHUNK_WEIGHTED_LEGACY,
        expected_token=identity["bundle_token"],
    )
    if (
        canonical_hash(meta) != identity["bundle_manifest_sha256"]
        or meta.get("rows_digest") != identity["bundle_rows_digest"]
    ):
        raise SectionEvidenceError("prediction bundle evidence is stale")
    return bodies["prediction.txt"]


__all__ = [
    "SectionEvidenceError",
    "directory_digest",
    "publish_corpus_validation",
    "publish_prediction",
    "verify_corpus_validation",
    "verify_prediction",
]
