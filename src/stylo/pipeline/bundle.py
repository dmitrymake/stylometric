"""B2 atomic model-bundle publish/load with immutable versions and a hashed sidecar.

A work-balanced train run must never replace the legacy production model in-place, must never
leave a loadable half-written or partial bundle, must never lose the published path to a crash,
and must never escape its bundle root (a symlinked ``versions/`` could otherwise delete an
external dir). This module publishes into an **immutable, content+meta-addressed**
``versions/<token>/`` directory and flips a single ``current.json`` pointer with one atomic
``os.replace``. The token binds BOTH file hashes and the full attestation meta. Every path in the
chain (root, versions, version dir, pointer, files) is required to be a real, non-symlink object
contained within the bundle root before any rmtree/replace. The bundle is a strict THREE-file
contract (model.pkl, delta.pkl, authors.json) with a mandatory attestation schema.
See research/P1_B2_MODEL_WIRING_DESIGN.md §7.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile
from typing import Callable, Dict

from ..jsonio import dump_strict, dumps_strict, load_strict

BUNDLE_VERSION = "b2.bundle.v1"
SIDECAR_NAME = "bundle_manifest.json"
CURRENT_NAME = "current.json"
VERSIONS_DIR = "versions"
import re as _re

REQUIRED_FILES = ("authors.json", "delta.pkl", "model.pkl")           # exact, not configurable
# mandatory attestation keys (non-null); binds the artifact to code/config/data it was trained on
REQUIRED_META = ("training_weighting", "dataset_contract", "rows_digest", "chunker_config_hash",
                 "code_tree_sha256", "config_id", "git_commit", "git_dirty")
_HEX64 = _re.compile(r"^[0-9a-f]{64}$")
_HEX64_KEYS = ("rows_digest", "chunker_config_hash", "code_tree_sha256", "config_id")
_RESERVED_META = {"bundle_version", "files"}


class BundleError(RuntimeError):
    """A bundle is missing/partial/wrong-version/wrong-schema, escapes its dir, symlinked, tampered."""


def _safe_name(name: str) -> bool:
    return (name not in ("", ".", "..") and "/" not in name and "\\" not in name
            and "\x00" not in name and name == pathlib.PurePosixPath(name).name
            and not pathlib.PurePath(name).is_absolute())


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verify_real_dir_chain(path) -> None:
    """Fail-closed if ANY existing component from the filesystem root down to ``path`` is a symlink
    (a symlinked ancestor would let publish rmtree/write outside the intended bundle root)."""
    path = pathlib.Path(path).absolute()
    cur = pathlib.Path(path.anchor or "/")
    for part in path.relative_to(cur).parts:
        cur = cur / part
        if cur.is_symlink():
            raise BundleError(f"symlink in bundle path chain: {cur}")


def _real_within(path: pathlib.Path, root: pathlib.Path, *, must_dir=False, must_file=False) -> bool:
    """True iff path exists, is NOT a symlink, is contained in root, and matches the kind."""
    if path.is_symlink():
        return False
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return False
    except (OSError, ValueError):
        return False
    if must_dir and not path.is_dir():
        return False
    if must_file and not path.is_file():
        return False
    return True


def _content_token(file_hashes: Dict[str, str], meta: Dict) -> str:
    body = "".join(f"{n}:{h}\n" for n, h in sorted(file_hashes.items()))
    body += "\x00META\x00" + dumps_strict(meta, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


def _validate_meta_schema(meta: Dict) -> None:
    missing = [k for k in REQUIRED_META if meta.get(k) in (None, "")]
    if missing:
        raise BundleError(f"attestation meta missing required non-null keys: {missing}")
    if meta["training_weighting"] != "work_balanced":
        raise BundleError("bundle attestation must be for the work_balanced arm")
    if meta["dataset_contract"] != "work_balanced_manifest":
        raise BundleError("bundle dataset_contract must be work_balanced_manifest")
    if type(meta["git_dirty"]) is not bool:
        raise BundleError("git_dirty must be a bool")
    if not (isinstance(meta["git_commit"], str) and meta["git_commit"].strip()):
        raise BundleError("git_commit must be a non-empty string")
    for k in _HEX64_KEYS:
        if not (isinstance(meta[k], str) and _HEX64.match(meta[k])):
            raise BundleError(f"attestation {k} must be a 64-hex sha256 digest")


def _versioned_dir_complete(versioned: pathlib.Path, root: pathlib.Path,
                            file_hashes: Dict[str, str], full_sidecar: Dict) -> bool:
    """A pre-existing token dir is trustworthy only if it is a real contained dir with EXACTLY the
    sidecar + tracked files (no extras/symlinks), every hash matches, and the sidecar equals what
    we are about to write (full meta, not just files)."""
    if not _real_within(versioned, root, must_dir=True):
        return False
    sidecar = versioned / SIDECAR_NAME
    if not _real_within(sidecar, versioned, must_file=True):
        return False
    try:
        meta = load_strict(sidecar)
    except Exception:
        return False
    if meta != full_sidecar:
        return False
    if {e.name for e in os.scandir(versioned)} != set(file_hashes) | {SIDECAR_NAME}:
        return False
    for name, want in file_hashes.items():
        p = versioned / name
        if not _real_within(p, versioned, must_file=True) or _sha256_file(p) != want:
            return False
    return True


def publish_bundle(bundle_root, writers: Dict[str, Callable[[pathlib.Path], None]], meta: Dict) -> Dict:
    """Publish a strict three-file bundle atomically into ``bundle_root`` (fail-closed on any
    symlink/containment/schema violation)."""
    bundle_root = pathlib.Path(bundle_root)
    if sorted(writers) != list(REQUIRED_FILES):
        raise BundleError(f"bundle must contain exactly {list(REQUIRED_FILES)}, got {sorted(writers)}")
    if any(not _safe_name(n) for n in writers):
        raise BundleError(f"unsafe bundle filename(s): {[n for n in writers if not _safe_name(n)]}")
    if _RESERVED_META & set(meta):
        raise BundleError(f"meta may not set reserved keys {_RESERVED_META & set(meta)}")
    _validate_meta_schema(meta)
    try:
        dumps_strict(meta, sort_keys=True)          # meta must be strict-JSON serializable (fail fast)
    except BundleError:
        raise
    except Exception as exc:
        raise BundleError(f"bundle meta is not JSON-serializable: {exc}") from exc

    _verify_real_dir_chain(bundle_root)             # no symlink anywhere from / down to the root
    bundle_root.mkdir(parents=True, exist_ok=True)
    _verify_real_dir_chain(bundle_root)             # re-check after mkdir (a component may be new)
    versions = bundle_root / VERSIONS_DIR
    if versions.is_symlink():                       # never follow a symlinked versions/ dir
        raise BundleError("versions/ is a symlink — refusing to publish (would escape bundle root)")
    versions.mkdir(exist_ok=True)
    if not _real_within(versions, bundle_root, must_dir=True):
        raise BundleError("versions/ escapes the bundle root")

    staging = pathlib.Path(tempfile.mkdtemp(dir=versions, prefix=".staging_"))
    try:
        file_hashes: Dict[str, str] = {}
        for name, writer in sorted(writers.items()):
            p = staging / name
            writer(p)
            if not p.is_file() or p.is_symlink():
                raise BundleError(f"writer for {name!r} did not produce a real file")
            file_hashes[name] = _sha256_file(p)
        # exact staging inventory: a writer must not create extra/nested files
        if {e.name for e in os.scandir(staging)} != set(REQUIRED_FILES):
            raise BundleError("writers created unexpected extra files in the staging bundle")
        full_sidecar = {**{k: v for k, v in meta.items()},
                        "bundle_version": BUNDLE_VERSION, "files": file_hashes}
        dump_strict(full_sidecar, staging / SIDECAR_NAME, trailing_newline=True)
        token = _content_token(file_hashes, meta)
        if not _safe_name(token):
            raise BundleError("computed token is not a safe directory name")
        versioned = versions / token
        if _versioned_dir_complete(versioned, versions, file_hashes, full_sidecar):
            shutil.rmtree(staging)                  # identical, COMPLETE version already published
        else:
            if versioned.is_symlink() or versioned.is_file():
                versioned.unlink()                  # stray symlink/file at the token path
            elif versioned.is_dir():
                if not _real_within(versioned, versions, must_dir=True):
                    raise BundleError("token path is not a real contained dir — refusing rmtree")
                shutil.rmtree(versioned)            # corrupt/incomplete prior version — replace
            os.replace(staging, versioned)          # immutable version dir (single atomic rename)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    tmp_ptr = pathlib.Path(tempfile.mktemp(dir=bundle_root, prefix=".current_"))
    dump_strict({"bundle_version": BUNDLE_VERSION, "version": token}, tmp_ptr, trailing_newline=True)
    os.replace(tmp_ptr, bundle_root / CURRENT_NAME)
    return full_sidecar


def load_bundle(bundle_root):
    """Load the current bundle only if pointer/version/token/allow-list/containment/schema hold,
    no symlinks anywhere, and every sha256 matches."""
    bundle_root = pathlib.Path(bundle_root)
    _verify_real_dir_chain(bundle_root)             # no symlink from / down to the bundle root
    ptr = bundle_root / CURRENT_NAME
    if not _real_within(ptr, bundle_root, must_file=True):
        raise BundleError("current pointer missing, a symlink, or escapes root")
    pointer = load_strict(ptr)
    if pointer.get("bundle_version") != BUNDLE_VERSION:
        raise BundleError(f"pointer bundle_version {pointer.get('bundle_version')!r} unexpected")
    token = pointer.get("version")
    if not isinstance(token, str) or not _safe_name(token):
        raise BundleError("pointer version is not a safe token")
    versions = bundle_root / VERSIONS_DIR
    versioned = versions / token
    if not _real_within(versions, bundle_root, must_dir=True) or not _real_within(versioned, versions, must_dir=True):
        raise BundleError("bundle version dir missing, a symlink, or escapes the bundle root")

    sidecar_path = versioned / SIDECAR_NAME
    if not _real_within(sidecar_path, versioned, must_file=True):
        raise BundleError("bundle sidecar missing or a symlink")
    meta = load_strict(sidecar_path)
    if meta.get("bundle_version") != BUNDLE_VERSION:
        raise BundleError(f"unexpected bundle_version {meta.get('bundle_version')!r}")
    files = meta.get("files") or {}
    if sorted(files) != list(REQUIRED_FILES):
        raise BundleError(f"bundle must list exactly {list(REQUIRED_FILES)}, got {sorted(files)}")
    user_meta = {k: v for k, v in meta.items() if k not in _RESERVED_META}
    _validate_meta_schema(user_meta)                # attestation schema also enforced on load
    if _content_token(files, user_meta) != token:   # token binds served content+meta
        raise BundleError("bundle token does not bind the served content/meta (out-of-band edit)")
    # exact inventory: no untracked extras
    if {e.name for e in os.scandir(versioned)} != set(files) | {SIDECAR_NAME}:
        raise BundleError("bundle version dir has untracked extra files")
    resolved: Dict[str, pathlib.Path] = {}
    for name, want in files.items():
        if not _safe_name(name):
            raise BundleError(f"unsafe bundle filename: {name}")
        p = versioned / name
        if not _real_within(p, versioned, must_file=True):
            raise BundleError(f"bundle file missing, a symlink, or escapes dir: {name}")
        if _sha256_file(p) != want:
            raise BundleError(f"bundle file tampered (hash mismatch): {name}")
        resolved[name] = p
    return meta, resolved
