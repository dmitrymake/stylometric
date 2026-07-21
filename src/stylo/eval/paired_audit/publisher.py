"""Path-guarded transient store and the verified content-addressed publisher (§4.4/§8).

The run's transient output (per-fold checkpoints, full per-work vectors) lives under the **gitignored**
``docs/exploratory/work_balanced/audit/runs/<run_id>/`` (a path-aware guard, no headline path). The
final **verified** artifact is promoted to the unignored, committed ``docs/work_balanced_paired_audit_v1.json``
(matched by ``!docs/*.json``), and the full per-work probability vectors are durably committed as a
**content-addressed archive** ``docs/work_balanced_paired_audit_v1/`` (each per-cell per-work file named
by its content hash) with a committed ``SHA256SUMS`` the summary references by hash, published via the
immutable version-directory + single atomic ``current.json``/``COMPLETE`` pointer pattern.

The publisher can write ONLY inside the transient run namespace and the two published audit paths; it
**refuses** to write any headline/frozen artifact (an allowlist plus a headline-basename denylist),
and rejects path traversal or a symlinked path chain. It never touches ``0.8805``, the frozen baseline
snapshot, or the README/site/PAPER headline artifacts.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile
from typing import Mapping

from ...jsonio import dump_strict, dumps_strict, load_strict
from ...pipeline.bundle import (_real_within, _safe_name, _sha256_file,
                                _verify_real_dir_chain)

RUNS_SUBPATH = pathlib.PurePosixPath("exploratory/work_balanced/audit/runs")
SUMMARY_NAME = "work_balanced_paired_audit_v1.json"
ARCHIVE_DIRNAME = "work_balanced_paired_audit_v1"
VERSIONS_DIR = "versions"
CURRENT_NAME = "current.json"
COMPLETE_NAME = "COMPLETE"
SHA256SUMS_NAME = "SHA256SUMS"
SUMMARY_IN_VERSION = "summary.json"
_PUBLISH_SCHEMA = "paired_audit.publish.v1"

# frozen/headline artifacts the publisher must NEVER write (denylist, belt-and-suspenders on top of
# the allowlist)
_HEADLINE_BASENAMES = frozenset({
    "final_comparison.txt", "final_comparison.v2.txt", "lobo_books.txt", "screening_panel_v1.json",
    "corpus_manifest.json", "corpus_validation.json", "README.md", "PAPER.md", "index.html",
})


class PublisherError(RuntimeError):
    """Fail-closed: a write escapes the allowed audit namespace, targets a headline path, or the
    published archive is inconsistent."""


def _docs_root(docs_root: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(docs_root).resolve()


def assert_writable_audit_path(path: pathlib.Path | str, *, docs_root: pathlib.Path | str,
                               run_id: str | None = None, allow_published: bool = False) -> pathlib.Path:
    """Fail-closed unless ``path`` is inside an allowed audit namespace and is not a headline artifact.

    Allowed: the transient run dir ``<docs>/exploratory/work_balanced/audit/runs/<run_id>/…`` (when a
    ``run_id`` is given) and — only when ``allow_published`` — the published summary
    ``<docs>/work_balanced_paired_audit_v1.json`` and the archive dir ``<docs>/work_balanced_paired_audit_v1/…``.
    """
    droot = _docs_root(docs_root)
    p = pathlib.Path(path)
    if p.name in _HEADLINE_BASENAMES:
        raise PublisherError(f"refusing to write a headline/frozen artifact: {p.name}")
    # the existing part of the parent chain must be symlink-free
    anchor = p
    while not anchor.exists():
        anchor = anchor.parent
    _verify_real_dir_chain(anchor)
    resolved = (anchor.resolve() / p.relative_to(anchor)) if anchor != p else anchor.resolve()

    def _within(base: pathlib.Path) -> bool:
        base = base.resolve()
        return resolved == base or str(resolved).startswith(str(base) + os.sep)

    if not _within(droot):
        raise PublisherError(f"audit write escapes docs root: {p}")
    allowed = False
    if run_id is not None:
        if not _safe_name(run_id):
            raise PublisherError("run_id is not a safe path token")
        if _within(droot / RUNS_SUBPATH / run_id):
            allowed = True
    if allow_published:
        if resolved == (droot / SUMMARY_NAME) or _within(droot / ARCHIVE_DIRNAME):
            allowed = True
    if not allowed:
        raise PublisherError(f"path is not in an allowed audit namespace: {p}")
    return p


def write_transient(run_id: str, relname: str, obj: Mapping, *,
                    docs_root: pathlib.Path | str) -> pathlib.Path:
    """Atomically write a transient run artifact under the gitignored run namespace."""
    if not _safe_name(relname):
        raise PublisherError(f"unsafe transient artifact name: {relname!r}")
    droot = _docs_root(docs_root)
    path = droot / RUNS_SUBPATH / run_id / relname
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_writable_audit_path(path, docs_root=docs_root, run_id=run_id)
    dump_strict(dict(obj), path, trailing_newline=True)
    return path


# ── content-addressed per-work archive ───────────────────────────────────────
def _stage_archive(staging: pathlib.Path, per_work_vectors: Mapping[str, object]) -> dict:
    """Write each per-cell per-work payload to ``<content-sha256>.json`` and a SHA256SUMS inventory.

    Returns ``{key: {"filename", "sha256"}}`` for the summary to reference by hash.
    """
    inventory: dict[str, dict] = {}
    digest_to_name: dict[str, str] = {}
    for key in sorted(per_work_vectors):
        payload = per_work_vectors[key]
        content = dumps_strict(payload, sort_keys=True).encode("utf-8") + b"\n"
        digest = hashlib.sha256(content).hexdigest()
        fname = f"{digest}.json"
        fpath = staging / fname
        if digest not in digest_to_name:                 # identical content dedups to one file
            fpath.write_bytes(content)
            digest_to_name[digest] = fname
        inventory[key] = {"filename": fname, "sha256": digest}
    lines = sorted(f"{digest}  {name}" for digest, name in digest_to_name.items())
    (staging / SHA256SUMS_NAME).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return inventory


def _self_hash(body: Mapping) -> str:
    return hashlib.sha256(dumps_strict(body, sort_keys=True).encode("utf-8")).hexdigest()


def _archive_token(staging: pathlib.Path) -> str:
    """Deterministic token over every file in the staged version dir (name + content sha)."""
    h = hashlib.sha256()
    h.update(_PUBLISH_SCHEMA.encode("utf-8"))
    for entry in sorted(os.scandir(staging), key=lambda e: e.name):
        if entry.is_symlink():
            raise PublisherError(f"symlink in staged archive (rejected): {entry.path}")
        h.update(len(entry.name).to_bytes(8, "big") + entry.name.encode("utf-8"))
        h.update(bytes.fromhex(_sha256_file(pathlib.Path(entry.path))))
    return h.hexdigest()[:32]


def _version_complete(versioned: pathlib.Path, token: str) -> bool:
    if not _real_within(versioned, versioned.parent, must_dir=True) or versioned.name != token:
        return False
    try:
        return _archive_token(versioned) == token
    except PublisherError:
        return False


def publish_audit(summary: Mapping, per_work_vectors: Mapping[str, object], *,
                  docs_root: pathlib.Path | str) -> dict:
    """Verified publication: stage the content-addressed per-work archive, bind it into the summary
    self-hash, publish the immutable version dir + atomic ``current.json``/``COMPLETE`` pointer, and
    write the committed summary JSON — all path-guarded, never a headline path."""
    if summary.get("claim_status") != "exploratory_internal":
        raise PublisherError("audit summary must carry claim_status=exploratory_internal")
    if not summary.get("run_id"):
        raise PublisherError("audit summary must carry a run_id")

    droot = _docs_root(docs_root)
    archive_root = droot / ARCHIVE_DIRNAME
    versions = archive_root / VERSIONS_DIR
    versions.mkdir(parents=True, exist_ok=True)
    assert_writable_audit_path(versions, docs_root=docs_root, allow_published=True)
    _verify_real_dir_chain(versions)

    staging = pathlib.Path(tempfile.mkdtemp(dir=versions, prefix=".staging_"))
    published: dict = {}
    try:
        inventory = _stage_archive(staging, per_work_vectors)
        sha256sums_digest = _sha256_file(staging / SHA256SUMS_NAME)
        full_summary = dict(summary)
        full_summary["per_work_archive"] = inventory
        full_summary["archive_sha256sums_digest"] = sha256sums_digest
        full_summary["schema"] = _PUBLISH_SCHEMA
        full_summary["self_hash"] = _self_hash({k: v for k, v in full_summary.items()
                                                if k != "self_hash"})
        dump_strict(full_summary, staging / SUMMARY_IN_VERSION, trailing_newline=True)

        token = _archive_token(staging)
        versioned = versions / token
        if versioned.exists() or versioned.is_symlink():
            if not _version_complete(versioned, token):
                raise PublisherError(f"archive version {token} exists with different content (conflict)")
            shutil.rmtree(staging)
        else:
            os.replace(staging, versioned)
        staging = None
        published = {"version": token, "versioned_dir": versioned, "summary": full_summary}
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    # atomic pointer + COMPLETE marker
    for name, body in ((CURRENT_NAME, {"schema": _PUBLISH_SCHEMA, "version": published["version"]}),
                       (COMPLETE_NAME, {"schema": _PUBLISH_SCHEMA, "version": published["version"],
                                        "run_id": summary["run_id"]})):
        target = archive_root / name
        assert_writable_audit_path(target, docs_root=docs_root, allow_published=True)
        tmp = pathlib.Path(tempfile.mktemp(dir=archive_root, prefix="." + name + "_"))
        dump_strict(body, tmp, trailing_newline=True)
        os.replace(tmp, target)

    # committed summary JSON (matched by !docs/*.json)
    summary_path = droot / SUMMARY_NAME
    assert_writable_audit_path(summary_path, docs_root=docs_root, allow_published=True)
    dump_strict(published["summary"], summary_path, trailing_newline=True)
    published["summary_path"] = summary_path
    return published


def load_published_audit(docs_root: pathlib.Path | str) -> dict:
    """Load and verify the published audit: pointer → immutable version dir → token integrity →
    summary self-hash → archive files match the summary inventory by content hash."""
    droot = _docs_root(docs_root)
    archive_root = droot / ARCHIVE_DIRNAME
    _verify_real_dir_chain(archive_root)
    ptr = archive_root / CURRENT_NAME
    if not _real_within(ptr, archive_root, must_file=True):
        raise PublisherError("current pointer missing, a symlink, or escapes the archive root")
    pointer = load_strict(ptr)
    if pointer.get("schema") != _PUBLISH_SCHEMA:
        raise PublisherError("current pointer schema mismatch")
    token = pointer.get("version")
    if not isinstance(token, str) or not _safe_name(token):
        raise PublisherError("current pointer version is not a safe token")
    versioned = archive_root / VERSIONS_DIR / token
    if not _version_complete(versioned, token):
        raise PublisherError("published archive version is partial, tampered, or conflicting")

    summary = load_strict(versioned / SUMMARY_IN_VERSION)
    recorded = {k: v for k, v in summary.items() if k != "self_hash"}
    if summary.get("self_hash") != _self_hash(recorded):
        raise PublisherError("published summary self-hash mismatch")
    for key, ref in summary.get("per_work_archive", {}).items():
        fpath = versioned / ref["filename"]
        if not _real_within(fpath, versioned, must_file=True) or _sha256_file(fpath) != ref["sha256"]:
            raise PublisherError(f"archive file for {key} missing or content-hash mismatch")
    return {"version": token, "versioned_dir": versioned, "summary": summary}
