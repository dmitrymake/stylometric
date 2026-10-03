"""Strict provenance envelopes for every section consumed by the HTML report."""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import tempfile
from collections.abc import Mapping

from ..config import artifact_config_id
from .._io import is_link, read_regular
from ..pipeline import _snapshot

from ..jsonio import (
    artifact_self_hash,
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
    loads_strict,
)

SECTION_SCHEMA = "stylo.report-section-evidence.v1"
CORPUS_SECTION = "corpus_validation"
PREDICTION_SECTION = "prediction"
TARGET_PREDICTION_SCHEMA = "stylo.target-prediction.v3"
_RESULT_ID = re.compile(r"^[0-9a-f]{64}$")


class SectionEvidenceError(RuntimeError):
    """A report section is missing, stale, or not byte/identity bound."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: pathlib.Path, *, label: str) -> bytes:
    try:
        return read_regular(path, label=label)
    except (OSError, ValueError) as exc:
        raise SectionEvidenceError(f"cannot read regular {label}: {path}") from exc


def _code_tree_sha256() -> str:
    from ..pipeline.train import _code_tree_sha256 as compute

    value = compute()
    if not isinstance(value, str) or len(value) != 64:
        raise SectionEvidenceError("cannot attest the current Stylo code tree")
    return value


def _config_id(cfg) -> str:
    return artifact_config_id(cfg)


def _verify_saved_bundle(meta, identity):
    from ..pipeline.bundle import (
        BundleError, REQUIRED_FILES, SUPPORTED_BUNDLE_VERSIONS, _RESERVED_META,
        _content_token, _validate_meta_schema,
    )
    if type(meta) is not dict or meta.get("bundle_version") not in SUPPORTED_BUNDLE_VERSIONS:
        raise SectionEvidenceError("saved prediction model metadata is malformed")
    files = meta.get("files")
    if (type(files) is not dict or set(files) != set(REQUIRED_FILES)
            or any(type(value) is not str or _RESULT_ID.fullmatch(value) is None for value in files.values())):
        raise SectionEvidenceError("saved prediction model file digests are malformed")
    user_meta = {key: value for key, value in meta.items() if key not in _RESERVED_META}
    try:
        _validate_meta_schema(user_meta, version=meta["bundle_version"])
    except BundleError as exc:
        raise SectionEvidenceError(f"saved prediction model metadata is malformed: {exc}") from exc
    if _content_token(files, user_meta, version=meta["bundle_version"]) != identity["bundle_token"]:
        raise SectionEvidenceError("prediction model token differs from saved metadata")


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
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
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
) -> pathlib.Path:
    _safe_docs(docs)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{section}-", dir=docs))
    try:
        _publish_locked_section(staging, section=section, files=files, identity=identity)
        _verify_locked_section(staging, section=section, expected_files=set(files))
        return _snapshot.publish_directory_snapshot(staging, docs / section)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _publish_locked_section(docs, *, section, files, identity):
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
    _atomic_write_text(docs / f"{section}.evidence.json", dumps_strict(envelope, sort_keys=True))


def _section_directory(docs, *, section, result_id=None):
    from ..pipeline.bundle import _verify_real_dir_chain
    _verify_real_dir_chain(docs)
    if result_id == "legacy":
        return docs
    target = docs / section
    store = _snapshot.snapshot_store(target)
    try:
        if result_id is not None:
            if type(result_id) is not str or _RESULT_ID.fullmatch(result_id) is None:
                raise SectionEvidenceError("result ID must be a full generation hash or legacy")
            versions = store / _snapshot.VERSIONS_DIRECTORY
            _verify_real_dir_chain(versions)
            return _snapshot._validate_generation(versions / result_id, result_id)
        if _snapshot._present(store):
            _snapshot._real_directory(store)
            if _snapshot._present(store / _snapshot.CURRENT_POINTER):
                return _snapshot.resolve_directory_snapshot(target)
    except _snapshot.SnapshotPublishError as exc:
        raise SectionEvidenceError(f"invalid {section} result generation: {exc}") from exc
    return docs


def _verify_section(
    docs: pathlib.Path,
    *,
    section: str,
    expected_files: set[str],
    result_id: str | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    resolved = _section_directory(docs, section=section, result_id=result_id)
    _safe_docs(resolved)
    return _verify_locked_section(resolved, section=section, expected_files=expected_files)


def _verify_locked_section(docs, *, section, expected_files):
    manifest_path = docs / f"{section}.evidence.json"
    try:
        envelope = loads_strict(_read_regular(manifest_path, label="section envelope").decode("utf-8"))
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
    from ..pipeline._snapshot import resolve_directory_snapshot
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    identity, bodies = _verify_section(
        docs,
        section=CORPUS_SECTION,
        expected_files={"corpus_validation.json", "corpus_validation.txt"},
    )
    expected = {
        "corpus_sha256": directory_digest(
            resolve_directory_snapshot(pathlib.Path(cfg.get_path("paths.input_clean", "input_clean")))
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
    target=None,
    selected_identity: dict | None = None,
    structured: dict | None = None,
) -> pathlib.Path | None:
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
    if target is not None:
        from ..pipeline.predict import resolve_prediction_target, target_identity
        fresh = resolve_prediction_target(cfg, target.work_id, unknown_root=target.catalog_root)
        observed = target_identity(fresh)
        if observed != selected_identity:
            raise SectionEvidenceError("selected target input changed during prediction")
        if type(structured) is not dict or structured.get("target_work_id") != target.work_id:
            raise SectionEvidenceError("structured prediction target identity mismatch")
        identity = {
            "prediction_schema_version": TARGET_PREDICTION_SCHEMA,
            "bundle_token": bundle_token,
            "bundle_manifest_sha256": canonical_hash(bundle_meta),
            "bundle_rows_digest": bundle_meta.get("rows_digest"),
            "training_weighting": bundle_meta.get("training_weighting"),
            "config_id": current_config, "code_tree_sha256": current_code,
            "presentation_settings": {
                "top_k_candidates": cfg.get_path("evaluation.top_k_candidates", 5),
                "output_directory": str(cfg.get_path("paths.docs", "docs")),
            },
            **observed,
        }
        _verify_saved_bundle(bundle_meta, identity)
        from .build import render_prediction_html
        docs = _prediction_collection(cfg, target.work_id, create=True)
        return _publish_section(
            docs, section=PREDICTION_SECTION,
            files={
                "prediction.txt": report, "prediction.json": dumps_strict(structured),
                "config.json": dumps_strict(cfg.to_dict(), sort_keys=True),
                "bundle.json": dumps_strict(bundle_meta, sort_keys=True),
                "index.html": render_prediction_html(target.work_id, report, identity=identity),
            },
            identity=identity,
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


def _prediction_collection(cfg, target_work: str, *, create=False) -> pathlib.Path:
    from ..pipeline.predict import _work_id
    from ..pipeline.bundle import _verify_real_dir_chain
    target_work = _work_id(target_work)
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs")) / "predictions" / _sha256(target_work.encode("utf-8"))
    _verify_real_dir_chain(docs)
    if create:
        docs.mkdir(parents=True, exist_ok=True)
        _verify_real_dir_chain(docs)
    return docs


def _find_result_collection(cfg, result_id):
    """Find an explicitly selected generation without consulting live inputs."""
    from ..pipeline.bundle import _verify_real_dir_chain
    if type(result_id) is not str or _RESULT_ID.fullmatch(result_id) is None:
        raise SectionEvidenceError("select a target for legacy results; otherwise use a full generation hash")
    root = pathlib.Path(cfg.get_path("paths.docs", "docs")) / "predictions"
    _verify_real_dir_chain(root)
    found = []
    if root.is_dir():
        for collection in root.iterdir():
            if is_link(collection):
                raise SectionEvidenceError("prediction collection must not be a symlink")
            candidate = _snapshot.snapshot_store(collection / PREDICTION_SECTION) / _snapshot.VERSIONS_DIRECTORY / result_id
            if candidate.exists() or is_link(candidate):
                found.append(collection)
    if len(found) != 1:
        raise SectionEvidenceError("selected result generation is missing or ambiguous")
    return found[0]


def prediction_directory(cfg, target_work: str | None = None, *, create=False,
                         result_id: str | None = None) -> pathlib.Path:
    if target_work is None:
        if result_id is None:
            from ..pipeline.predict import resolve_prediction_target
            target_work = resolve_prediction_target(cfg).work_id
        else:
            collection = _find_result_collection(cfg, result_id)
            return _section_directory(collection, section=PREDICTION_SECTION, result_id=result_id)
    collection = _prediction_collection(cfg, target_work, create=create)
    if create:
        return collection
    return _section_directory(collection, section=PREDICTION_SECTION, result_id=result_id)


def verify_prediction(cfg, target_work: str | None = None, *, result_id: str | None = None) -> str:
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    if target_work is None and result_id in {None, "legacy"}:
        legacy = _section_directory(docs, section=PREDICTION_SECTION, result_id=result_id)
        if (legacy / "prediction.evidence.json").exists():
            return _verify_legacy_prediction(cfg, result_id=result_id)
    if target_work is None:
        if result_id is None:
            from ..pipeline.predict import resolve_prediction_target
            target_work = resolve_prediction_target(cfg).work_id
    return read_prediction_result(cfg, target_work, result_id=result_id)[2]["prediction.txt"]


def verify_structured_prediction(cfg, target_work: str | None = None, *, result_id: str | None = None) -> dict:
    return loads_strict(read_prediction_result(cfg, target_work, result_id=result_id)[2]["prediction.json"])


def read_prediction_result(cfg, target_work: str | None = None, *, result_id: str | None = None):
    """Resolve and verify one complete result before returning any view path.

    Explicit selection opens historical data using its saved config/model;
    default selection additionally checks the currently requested calculation.
    """
    from ..pipeline.predict import resolve_prediction_target, target_identity
    from ..pipeline.bundle import load_bundle
    from ..domain.work_weighting import resolve_training_weighting
    if target_work is None and result_id == "legacy":
        docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
        _verify_legacy_prediction(cfg, result_id="legacy")
        identity, bodies = _verify_locked_section(docs, section=PREDICTION_SECTION,
                                                 expected_files={"prediction.txt"})
        return docs, identity, bodies
    if target_work is None and result_id is None:
        target_work = resolve_prediction_target(cfg).work_id
    docs = prediction_directory(cfg, target_work, result_id=result_id)
    versioned = (docs / _snapshot.GENERATION_MANIFEST).is_file()
    expected_files = {"prediction.txt", "prediction.json"}
    if versioned:
        expected_files |= {"index.html", "config.json", "bundle.json"}
    identity, bodies = _verify_locked_section(docs, section=PREDICTION_SECTION,
                                              expected_files=expected_files)
    expected_fields = {"prediction_schema_version", "bundle_token", "bundle_manifest_sha256",
                       "bundle_rows_digest", "training_weighting", "config_id", "code_tree_sha256",
                       "target_work_id", "target_root", "target_catalog_root", "target_files", "target_sha256"}
    if versioned:
        expected_fields.add("presentation_settings")
    if set(identity) != expected_fields or identity.get("prediction_schema_version") != (
            TARGET_PREDICTION_SCHEMA if versioned else "stylo.target-prediction.v2"):
        raise SectionEvidenceError("per-target prediction identity field mismatch")
    if target_work is None:
        target_work = identity["target_work_id"]
        if _prediction_collection(cfg, target_work) != _find_result_collection(cfg, result_id):
            raise SectionEvidenceError("selected result target differs from its collection")
    if identity["target_work_id"] != target_work:
        raise SectionEvidenceError("selected result target differs from the requested work")
    if versioned:
        from ..config import ConfigNode
        saved_config = loads_strict(bodies["config.json"])
        if type(saved_config) is not dict or _config_id(ConfigNode(saved_config)) != identity["config_id"]:
            raise SectionEvidenceError("saved prediction config identity mismatch")
        saved = ConfigNode(saved_config)
        if identity["presentation_settings"] != {
                "top_k_candidates": saved.get_path("evaluation.top_k_candidates", 5),
                "output_directory": str(saved.get_path("paths.docs", "docs"))}:
            raise SectionEvidenceError("saved presentation settings mismatch")
        meta = loads_strict(bodies["bundle.json"])
        _verify_saved_bundle(meta, identity)
    else:
        meta, _paths = load_bundle(_deployment_bundle_root(cfg, identity["training_weighting"]),
                                  expected_token=identity["bundle_token"])
    if (type(meta) is not dict or meta.get("training_weighting") != identity["training_weighting"]
            or canonical_hash(meta) != identity["bundle_manifest_sha256"]
            or meta.get("rows_digest") != identity["bundle_rows_digest"]
            or meta.get("config_id") != identity["config_id"]
            or meta.get("code_tree_sha256") != identity["code_tree_sha256"]):
        raise SectionEvidenceError("prediction bundle evidence is stale")
    if result_id is None and (identity["config_id"] != _config_id(cfg)
                              or identity["code_tree_sha256"] != _code_tree_sha256()):
        raise SectionEvidenceError("per-target prediction code/config/work identity is stale")
    expected_token = cfg.get_path("deployment.expected_bundle_token", None)
    if result_id is None and expected_token is not None and expected_token != identity["bundle_token"]:
        raise SectionEvidenceError("prediction evidence differs from the trusted bundle token")
    if result_id is None:
        target = resolve_prediction_target(cfg, target_work, unknown_root=identity["target_catalog_root"])
        if target_identity(target) != {key: identity[key] for key in target_identity(target)}:
            raise SectionEvidenceError("selected prediction target has drifted")
        weighting = resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))
        if identity["training_weighting"] != weighting:
            raise SectionEvidenceError("prediction training weighting mismatch")
        live_meta, _paths = load_bundle(_deployment_bundle_root(cfg, weighting), expected_token=identity["bundle_token"])
        if canonical_hash(live_meta) != identity["bundle_manifest_sha256"]:
            raise SectionEvidenceError("prediction bundle evidence is stale")
    structured = loads_strict(bodies["prediction.json"])
    if type(structured) is not dict or structured.get("target_work_id") != target_work:
        raise SectionEvidenceError("structured target differs from prediction evidence")
    if versioned:
        from ..config import deployment_candidates
        if (tuple(structured.get("candidate_authors", ())) != deployment_candidates(saved)
                or structured.get("training_weighting") != identity["training_weighting"]):
            raise SectionEvidenceError("structured model panel/weighting differs from saved evidence")
    return docs, identity, bodies


def _deployment_bundle_root(cfg, weighting):
    return pathlib.Path(cfg.get_path("paths.data", "data")) / "deployment" / weighting


def _verify_legacy_prediction(cfg, *, result_id=None) -> str:
    from ..domain.work_weighting import CHUNK_WEIGHTED_LEGACY
    from ..pipeline.bundle import load_bundle

    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    identity, bodies = _verify_section(
        docs,
        section=PREDICTION_SECTION,
        expected_files={"prediction.txt"},
        result_id=result_id,
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
    if result_id == "legacy":
        return bodies["prediction.txt"]
    trusted_token = cfg.get_path("deployment.expected_bundle_token", None)
    if trusted_token is not None and trusted_token != identity["bundle_token"]:
        raise SectionEvidenceError("prediction evidence differs from the trusted bundle token")
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
    "verify_structured_prediction",
    "prediction_directory",
    "read_prediction_result",
]
