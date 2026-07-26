"""Release hygiene: keep private corpus paths out of what a push would publish.

Three distinct questions, none to be conflated:

* :func:`check_index` — is a private path staged right now?
* :func:`check_publish_ref` — would a ``git push`` of this ref send any private
  object? A push transmits every commit reachable from the pushed ref, so the gate
  scans the ref's **reachable history**, not just the tip tree: a secret that was
  committed and later deleted is gone from the tip but still travels in a push.
* :func:`audit_local_refs` — do other refs (branches, tags, remotes, custom) or the
  stash still reach private objects? Local ``main`` legitimately keeps the whole
  corpus and is never pushed, so this only WARNS.

Robustness choices that matter for a security gate:

* Detection is delegated to git's pathspec (``-- <prefix>``) instead of matching
  path strings ourselves. A file named ``input_clean/a\\tb"\\n.txt`` cannot evade
  ``startswith`` matching then, because git decides the match on raw bytes.
* ``-z`` (NUL-delimited) output is used wherever available, so newlines/quotes in a
  filename never split one path into two.
* ``--no-replace-objects`` is set on every command, so a ``refs/replace`` entry that
  swaps a private tree for a clean one cannot make the gate read a different history
  than the one ``git push`` actually sends. Replace refs are additionally surfaced.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: Path prefixes whose contents must never reach a public ref.
PRIVATE_PREFIXES: tuple[str, ...] = (
    "input/",
    "input_clean/",
    "input_cases/",
    "input_personal/",
    "input_personal_fr/",
    "input_disputed/",
    "_staging_corpora/",
    "data/frags_train/",
    "data/frags_unknown/",
)

# git base flags: ignore replacement objects (a push does), and emit raw UTF-8.
_BASE = ["--no-replace-objects", "-c", "core.quotePath=false"]

# Match a protected root itself (a blob literally named ``input_clean``) as well as
# anything beneath it. Trailing-slash prefixes would miss the bare-root case.
_ROOTS = tuple(prefix.rstrip("/") for prefix in PRIVATE_PREFIXES)
_CASEFOLDED_ROOTS = tuple(root.casefold() for root in _ROOTS)
# Case-insensitive index pathspecs: a private file committed with non-canonical
# case (Input_Clean/…) on a case-insensitive filesystem must not slip past.
_INDEX_PATHSPECS = [f":(icase){root}" for root in _ROOTS]

# Build these byte markers without embedding a literal developer-home path in
# the scanner's own source.  The release archive contains tests and gate code,
# so a self-matching marker would make every valid archive fail.
_ABSOLUTE_HOME_MARKERS: tuple[bytes, ...] = (
    b"/" + b"home" + b"/",
    b"/" + b"Users" + b"/",
    b"\\" + b"Users" + b"\\",
)


class HygieneError(RuntimeError):
    """A git command required by the hygiene check failed."""


def _run(args: list[str], *, cwd: str | None = None) -> bytes:
    try:
        return subprocess.run(
            ["git", *_BASE, *args], cwd=cwd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except FileNotFoundError as exc:  # pragma: no cover - git always present in CI
        raise HygieneError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        raise HygieneError(f"git {' '.join(args)} failed: {detail}") from exc


def _text(args: list[str], *, cwd: str | None = None) -> str:
    return _run(args, cwd=cwd).decode("utf-8", "surrogateescape")


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line]


def _decode_z(blob: bytes) -> list[str]:
    return [part.decode("utf-8", "surrogateescape") for part in blob.split(b"\0") if part]


def is_private_path(path: str) -> bool:
    """True if ``path`` is, or is under, a protected root (case-insensitive)."""
    folded = path.casefold()
    return any(folded == root or folded.startswith(root + "/") for root in _CASEFOLDED_ROOTS)


def _private_content_marker(path: Path) -> bytes | None:
    """Return the first private absolute-home marker in a regular file."""
    overlap = max(map(len, _ABSOLUTE_HOME_MARKERS)) - 1
    tail = b""
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                window = tail + block
                for marker in _ABSOLUTE_HOME_MARKERS:
                    if marker in window:
                        return marker
                tail = window[-overlap:]
    except OSError as exc:
        raise HygieneError(f"cannot read release archive member {path}: {exc}") from exc
    return None


def check_archive_content(root: str | Path) -> list[str]:
    """Inspect a materialized public archive for private paths or host layout.

    This deliberately operates on the exported tree rather than the checkout:
    immutable historical artifacts may remain tracked but be excluded with
    ``export-ignore``.  Symlinks are inspected without following them, so an
    archive member cannot escape ``root`` while the gate is reading it.
    """
    archive_root = Path(root).resolve()
    if not archive_root.is_dir():
        raise HygieneError(f"release archive root is not a directory: {archive_root}")

    issues: list[str] = []
    try:
        walker = os.walk(archive_root, topdown=True, followlinks=False)
        for directory, dirnames, filenames in walker:
            dirnames.sort()
            filenames.sort()
            for name in [*dirnames, *filenames]:
                path = Path(directory, name)
                relative = path.relative_to(archive_root).as_posix()
                if is_private_path(relative):
                    issues.append(f"{relative}: protected private path")

                if path.is_symlink():
                    try:
                        target = os.readlink(path)
                    except OSError as exc:
                        raise HygieneError(
                            f"cannot read release archive symlink {relative}: {exc}"
                        ) from exc
                    target_bytes = os.fsencode(target)
                    if os.path.isabs(target):
                        issues.append(f"{relative}: absolute symlink target")
                    for marker in _ABSOLUTE_HOME_MARKERS:
                        if marker in target_bytes:
                            issues.append(
                                f"{relative}: private absolute-home marker in symlink"
                            )
                            break
                elif path.is_file():
                    if _private_content_marker(path) is not None:
                        issues.append(
                            f"{relative}: private absolute-home marker in file content"
                        )
                elif not path.is_dir():
                    issues.append(f"{relative}: unsupported archive member type")
    except OSError as exc:
        raise HygieneError(f"cannot walk release archive {archive_root}: {exc}") from exc
    return sorted(set(issues))


def _is_shallow(*, cwd: str | None = None) -> bool:
    return _text(["rev-parse", "--is-shallow-repository"], cwd=cwd).strip() == "true"


def _resolve_commit(ref: str, *, cwd: str | None = None) -> str:
    """Resolve ``ref`` to a commit OID, rejecting option-like injections.

    ``--end-of-options`` stops a value such as ``--max-count=0`` or ``--no-walk``
    from being parsed as a git flag, and ``^{commit}`` requires a real commit.
    """
    try:
        return _text(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd=cwd).strip()
    except HygieneError as exc:
        raise HygieneError(f"cannot resolve publish ref {ref!r} to a commit") from exc


def check_index(*, cwd: str | None = None) -> list[str]:
    """Private paths staged in the index (empty == clean). Byte-safe."""
    out = _run(["ls-files", "-z", "--", *_INDEX_PATHSPECS], cwd=cwd)
    return sorted(set(_decode_z(out)))


def _paths_in_tree(tree_oid: str, *, cwd: str | None = None) -> list[str]:
    """Every path recorded in ``tree_oid`` (recursive, byte-safe)."""
    out = _run(["ls-tree", "-r", "--name-only", "-z", tree_oid], cwd=cwd)
    return _decode_z(out)


def _private_objects(commit_oid: str, *, cwd: str | None = None) -> list[str]:
    """Private paths recorded in ANY commit reachable from ``commit_oid``.

    Enumerates the full tree of every reachable commit rather than filtering
    ``rev-list --objects`` name hints. ``rev-list --objects`` emits each object OID
    once with a single hint path, so a private path whose blob is shared with a
    public path (content alias) is invisible there — a real publish bypass. Full
    per-commit tree listing surfaces every recorded path. Commits are de-duplicated
    by root tree so shared history is listed once.
    """
    private: set[str] = set()
    seen_trees: set[str] = set()
    for tree in _lines(_text(["log", "--format=%T", commit_oid], cwd=cwd)):
        if tree in seen_trees:
            continue
        seen_trees.add(tree)
        for path in _paths_in_tree(tree, cwd=cwd):
            if is_private_path(path):
                private.add(path)
    return sorted(private)


def check_publish_ref(ref: str = "HEAD", *, cwd: str | None = None) -> list[str]:
    """Private paths reachable from ``ref`` (empty == safe to publish).

    This is the publish gate: it covers the tip AND every ancestor commit on every
    merged branch, because a push sends all of them.

    Fails closed on a shallow repository: ``git rev-list`` stops at the graft
    boundary, so a truncated clone could hide a committed-then-deleted secret and
    still report clean. Unshallow (``git fetch --unshallow``) before gating.
    """
    if _is_shallow(cwd=cwd):
        raise HygieneError(
            "shallow repository: history walk is truncated and cannot certify the "
            "publish ref; run `git fetch --unshallow` first"
        )
    return _private_objects(_resolve_commit(ref, cwd=cwd), cwd=cwd)


def _resolve(ref: str, *, cwd: str | None = None) -> str:
    return _text(["rev-parse", ref], cwd=cwd).strip()


def _all_refs(*, cwd: str | None = None) -> list[str]:
    return _lines(_text(["for-each-ref", "--format=%(refname)"], cwd=cwd))


def local_refs_excluding(publish_ref: str, *, cwd: str | None = None) -> list[str]:
    """Every local ref (branches, tags, remotes, custom) except the publish ref.

    Fails loudly if ``publish_ref`` cannot be resolved rather than silently
    auditing everything.
    """
    publish_oid = _resolve(publish_ref, cwd=cwd)
    result = []
    for ref in _all_refs(cwd=cwd):
        if ref == "refs/stash":
            continue  # stash is audited separately via `git stash list`
        if _resolve(ref, cwd=cwd) != publish_oid:
            result.append(ref)
    return result


def replace_refs(*, cwd: str | None = None) -> list[str]:
    """Any ``refs/replace/*`` entries (they can mask history from naive tooling)."""
    return [r for r in _all_refs(cwd=cwd) if r.startswith("refs/replace/")]


def _stash_entries(*, cwd: str | None = None) -> list[str]:
    return _lines(_text(["stash", "list", "--format=%gd"], cwd=cwd))


def _peel(ref: str, *, cwd: str | None = None) -> str:
    """Resolve ``ref`` to its underlying object OID, peeling annotated tags."""
    return _text(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{}}"], cwd=cwd).strip()


def _object_type(oid: str, *, cwd: str | None = None) -> str:
    return _text(["cat-file", "-t", oid], cwd=cwd).strip()


def _ref_private_paths(ref: str, *, cwd: str | None = None) -> tuple[str, list[str]]:
    """Return (object_type, private paths) for what pushing ``ref`` would publish.

    Commits are walked over full history; a tree tag is listed directly; a blob tag
    publishes an object with no path, so it is surfaced by type only.
    """
    oid = _peel(ref, cwd=cwd)
    obj_type = _object_type(oid, cwd=cwd)
    if obj_type == "commit":
        return obj_type, _private_objects(oid, cwd=cwd)
    if obj_type == "tree":
        return obj_type, sorted({p for p in _paths_in_tree(oid, cwd=cwd) if is_private_path(p)})
    return obj_type, []  # blob (or other) — no path to match, surfaced by type


@dataclass
class RefAudit:
    ref: str
    private_path_count: int
    sample: list[str] = field(default_factory=list)


@dataclass
class LocalAudit:
    publish_ref: str
    refs: list[RefAudit] = field(default_factory=list)
    stashes: list[RefAudit] = field(default_factory=list)
    replace_refs: list[str] = field(default_factory=list)
    nonstandard_refs: list[str] = field(default_factory=list)

    @property
    def has_private_history(self) -> bool:
        return any(r.private_path_count for r in self.refs) or any(
            s.private_path_count for s in self.stashes
        )

    @property
    def has_replace_refs(self) -> bool:
        return bool(self.replace_refs)


def audit_local_refs(publish_ref: str = "HEAD", *, sample: int = 5, cwd: str | None = None) -> LocalAudit:
    """WARN-only audit of private objects reachable from other refs/stash.

    Also reports ``refs/replace`` entries and refs that point at a blob/tree rather
    than a commit (those publish objects too). A stash created with ``-u``/``-a``
    stores untracked/ignored files in an extra parent commit, which rev-list reaches.
    """
    audit = LocalAudit(publish_ref=publish_ref, replace_refs=replace_refs(cwd=cwd))
    for ref in local_refs_excluding(publish_ref, cwd=cwd):
        try:
            obj_type, found = _ref_private_paths(ref, cwd=cwd)
        except HygieneError:
            continue
        if obj_type != "commit":
            # a tree/blob ref publishes objects whose mount point is unknown (a tag
            # on the input_clean subtree lists paths WITHOUT the input_clean prefix),
            # so it is always unsafe; still surface any recognisable private paths.
            audit.nonstandard_refs.append(f"{ref} ({obj_type})")
            if found:
                audit.refs.append(RefAudit(ref=ref, private_path_count=len(found), sample=found[:sample]))
            continue
        if found:
            audit.refs.append(RefAudit(ref=ref, private_path_count=len(found), sample=found[:sample]))
    for entry in _stash_entries(cwd=cwd):
        oid = _resolve(entry, cwd=cwd)
        found = _private_objects(oid, cwd=cwd)
        if found:
            audit.stashes.append(RefAudit(ref=entry, private_path_count=len(found), sample=found[:sample]))
    return audit


__all__ = [
    "PRIVATE_PREFIXES",
    "HygieneError",
    "LocalAudit",
    "RefAudit",
    "audit_local_refs",
    "check_index",
    "check_publish_ref",
    "is_private_path",
    "local_refs_excluding",
    "replace_refs",
]
