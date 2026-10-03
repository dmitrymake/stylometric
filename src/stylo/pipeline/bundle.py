"""Atomic work-balanced model-bundle publish/load with immutable versions and a hashed sidecar.

A work-balanced train run must never replace the legacy production model in-place, must never
leave a loadable half-written or partial bundle, must never lose the published path to a crash,
and must never escape its bundle root (a symlinked ``versions/`` could otherwise delete an
external dir). This module publishes into an **immutable, content+meta-addressed**
``versions/<token>/`` directory and flips a single ``current.json`` pointer with one atomic
``os.replace``. The token binds BOTH file hashes and the full attestation meta. Every path in the
chain (root, versions, version dir, pointer, files) is required to be a real, non-symlink object
contained within the bundle root before any rmtree/replace. The bundle is a strict THREE-file
contract (model.pkl, delta.pkl, authors.json) with a mandatory attestation schema.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile
from typing import Callable, Dict

from ..jsonio import dump_strict, dumps_strict, load_strict
from .._io import exclusive_file_lock, fsync_directory, read_regular

BUNDLE_VERSION = "stylo.deployment.bundle.v3"
LEGACY_BUNDLE_VERSION = "stylo.deployment.bundle.v2"
SUPPORTED_BUNDLE_VERSIONS = {LEGACY_BUNDLE_VERSION, BUNDLE_VERSION}
SIDECAR_NAME = "bundle_manifest.json"
CURRENT_NAME = "current.json"
VERSIONS_DIR = "versions"
PUBLISH_LOCK_NAME = ".publish.lock"
import re as _re

REQUIRED_FILES = ("authors.json", "delta.pkl", "model.pkl")           # exact, not configurable
# All keys are present; only Git fields may be null in v3 packaged training.
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


def _read_regular_nofollow(path: pathlib.Path) -> bytes:
    """Read one immutable candidate through a no-follow descriptor."""
    try:
        return read_regular(path, label="bundle payload")
    except (OSError, ValueError) as exc:
        raise BundleError(f"cannot read a regular bundle payload: {path}") from exc


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


def _content_token(file_hashes: Dict[str, str], meta: Dict, *, version: str = BUNDLE_VERSION) -> str:
    body = "".join(f"{n}:{h}\n" for n, h in sorted(file_hashes.items()))
    body += "\x00META\x00" + dumps_strict(meta, sort_keys=True)
    if version != LEGACY_BUNDLE_VERSION:
        body = version + "\x00" + body
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


def _validate_meta_schema(meta: Dict, *, version: str = BUNDLE_VERSION) -> None:
    nullable = {"git_commit", "git_dirty"} if version == BUNDLE_VERSION else set()
    missing = [k for k in REQUIRED_META if k not in meta or (k not in nullable and meta[k] in (None, ""))]
    if missing:
        raise BundleError(f"attestation meta missing required non-null keys: {missing}")
    contracts = {
        "chunk_weighted_legacy": "legacy_recursive",
        "work_balanced": "work_balanced_manifest",
    }
    weighting = meta["training_weighting"]
    if weighting not in contracts:
        raise BundleError(
            f"unsupported bundle training_weighting {weighting!r}; "
            f"expected one of {sorted(contracts)}"
        )
    if meta["dataset_contract"] != contracts[weighting]:
        raise BundleError(
            f"bundle dataset_contract must be {contracts[weighting]!r} "
            f"for training_weighting={weighting!r}"
        )
    if not (version == BUNDLE_VERSION and meta["git_commit"] is None and meta["git_dirty"] is None):
        if type(meta["git_dirty"]) is not bool or not (
            type(meta["git_commit"]) is str and meta["git_commit"].strip()
        ):
            raise BundleError("Git attestation metadata must be both null, or non-null git_commit with bool git_dirty")
    for k in _HEX64_KEYS:
        if not (isinstance(meta[k], str) and _HEX64.fullmatch(meta[k])):
            raise BundleError(f"attestation {k} must be a 64-hex sha256 digest")
    if "training_work_ids" in meta:
        work_ids = meta["training_work_ids"]
        if type(work_ids) is not list or not work_ids or any(type(value) is not str or not value for value in work_ids) or len(set(work_ids)) != len(work_ids):
            raise BundleError("training_work_ids must be a nonempty unique string list")


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

    lock_path = bundle_root / PUBLISH_LOCK_NAME
    try:
        with exclusive_file_lock(lock_path):
            return _publish_locked_bundle(bundle_root, versions, writers, meta)
    except (OSError, ValueError) as exc:
        raise BundleError(f"bundle publication I/O failed: {exc}") from exc


def _publish_locked_bundle(bundle_root, versions, writers, meta):
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

        fd, tmp_name = tempfile.mkstemp(dir=bundle_root, prefix=".current_")
        os.close(fd)
        tmp_ptr = pathlib.Path(tmp_name)
        try:
            dump_strict(
                {"bundle_version": BUNDLE_VERSION, "version": token},
                tmp_ptr,
                trailing_newline=True,
            )
            with open(tmp_ptr, "r+b" if os.name == "nt" else "rb") as pointer_handle:
                os.fsync(pointer_handle.fileno())
            os.replace(tmp_ptr, bundle_root / CURRENT_NAME)
            fsync_directory(bundle_root)
        finally:
            if tmp_ptr.exists():
                tmp_ptr.unlink()

        # The token is intentionally returned out-of-band from the sidecar.  A
        # deployment must pin it through trusted configuration/CLI before
        # invoking an executable model deserializer.
        return {**full_sidecar, "bundle_token": token}
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def load_bundle(bundle_root, *, expected_token: str | None = None):
    """Load exactly the trusted immutable version, or inspect CURRENT without a token.

    A supplied token never falls back to CURRENT. The same containment, schema,
    complete inventory and byte/hash checks apply to historical versions.
    """
    bundle_root = pathlib.Path(bundle_root)
    _verify_real_dir_chain(bundle_root)             # no symlink from / down to the bundle root
    version = None
    if expected_token is not None:
        if not isinstance(expected_token, str) or not _safe_name(expected_token):
            raise BundleError("expected bundle token is malformed")
        token = expected_token
    else:
        ptr = bundle_root / CURRENT_NAME
        if not _real_within(ptr, bundle_root, must_file=True):
            raise BundleError("current pointer missing, a symlink, or escapes root")
        pointer = load_strict(ptr)
        version = pointer.get("bundle_version")
        if version not in SUPPORTED_BUNDLE_VERSIONS:
            raise BundleError(f"pointer bundle_version {version!r} unexpected")
        token = pointer.get("version")
        if not isinstance(token, str) or not _safe_name(token):
            raise BundleError("pointer version is not a safe token")
    versions = bundle_root / VERSIONS_DIR
    versioned = versions / token
    if not _real_within(versions, bundle_root, must_dir=True) or not _real_within(versioned, versions, must_dir=True):
        raise BundleError("trusted expected token version missing, a symlink, or escapes the bundle root")

    sidecar_path = versioned / SIDECAR_NAME
    if not _real_within(sidecar_path, versioned, must_file=True):
        raise BundleError("bundle sidecar missing or a symlink")
    meta = load_strict(sidecar_path)
    observed_version = meta.get("bundle_version")
    if observed_version not in SUPPORTED_BUNDLE_VERSIONS or (version is not None and observed_version != version):
        raise BundleError(f"unexpected bundle_version {meta.get('bundle_version')!r}")
    version = observed_version
    files = meta.get("files") or {}
    if sorted(files) != list(REQUIRED_FILES):
        raise BundleError(f"bundle must list exactly {list(REQUIRED_FILES)}, got {sorted(files)}")
    user_meta = {k: v for k, v in meta.items() if k not in _RESERVED_META}
    _validate_meta_schema(user_meta, version=version) # preserve the strict v2 load contract
    if _content_token(files, user_meta, version=version) != token:
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


def load_bundle_payloads(bundle_root, *, expected_token: str) -> tuple[Dict, Dict[str, bytes]]:
    """Return digest-verified bytes suitable for executable deserialisation.

    The path-based ``load_bundle`` contract is useful to inspect a bundle, but a
    consumer that verifies a pathname and later reopens it has a substitution
    race.  This helper re-reads each payload through ``O_NOFOLLOW``, verifies the
    bytes against the already token-bound sidecar, and returns those exact bytes
    to the deserializer.
    """

    meta, paths = load_bundle(bundle_root, expected_token=expected_token)
    payloads: Dict[str, bytes] = {}
    for name, path in paths.items():
        payload = _read_regular_nofollow(path)
        if hashlib.sha256(payload).hexdigest() != meta["files"][name]:
            raise BundleError(f"bundle payload changed during verified read: {name}")
        payloads[name] = payload
    return meta, payloads
