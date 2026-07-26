#!/usr/bin/env python3
"""Release gate for public repository hygiene.

Two independent checks:

* Publish gate (always, blocking): the tree of the publish ref (default ``HEAD``)
  and the current index must contain no private corpus paths.
* Local-repo audit (opt-in, warning only): other local refs and the stash may
  still reach private objects in history. That is expected — local ``main`` keeps
  the full corpus and is never pushed — so it warns instead of failing.

Usage:
    python scripts/check_release_hygiene.py                     # publish gate on HEAD + index
    python scripts/check_release_hygiene.py --publish-ref origin/main
    python scripts/check_release_hygiene.py --archive           # scan a Git-free exported tree
    python scripts/check_release_hygiene.py --audit-local-refs  # + warn on other refs/stash
    python scripts/check_release_hygiene.py --history           # deprecated alias of --audit-local-refs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stylo.release.hygiene import (  # noqa: E402
    HygieneError,
    audit_local_refs,
    check_archive_content,
    check_index,
    check_publish_ref,
)


def _fail(title: str, rows: list[str]) -> None:
    print(f"✗ {title}", file=sys.stderr)
    for row in rows[:80]:
        print(f"  {row}", file=sys.stderr)
    if len(rows) > 80:
        print(f"  ... {len(rows) - 80} more", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publish-ref", default="HEAD",
                        help="ref whose tree is the publish gate (default: HEAD)")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="scan this Git-free exported tree for private paths/content",
    )
    parser.add_argument("--audit-local-refs", action="store_true",
                        help="also warn about private objects in other local refs/stash")
    parser.add_argument("--history", action="store_true", dest="audit_local_refs",
                        help="deprecated alias of --audit-local-refs (no longer blocks)")
    parser.add_argument("--ref-only", action="store_true",
                        help="check only the publish ref history, skip the index (used by the pre-push hook)")
    args = parser.parse_args()

    try:
        if args.archive:
            if args.audit_local_refs or args.ref_only:
                parser.error("--archive cannot be combined with Git ref/audit modes")
            git_metadata = ROOT / ".git"
            if git_metadata.exists() or git_metadata.is_symlink():
                raise HygieneError(
                    "--archive requires a Git-free root and cannot bypass publish-ref checks"
                )
            archive_private = check_archive_content(ROOT)
            if archive_private:
                _fail(
                    "public source archive exposes private paths or host layout",
                    archive_private,
                )
                return 1
            print("✓ public source archive contains no private paths or host layout")
            return 0

        failed = False

        if not args.ref_only:
            index_private = check_index()
            if index_private:
                _fail("private corpus paths are tracked in the index", index_private)
                failed = True
            else:
                print("✓ index contains no private corpus paths")

        ref_private = check_publish_ref(args.publish_ref)
        if ref_private:
            _fail(f"publish ref {args.publish_ref!r} history would push private corpus paths", ref_private)
            failed = True
        else:
            print(f"✓ publish ref {args.publish_ref!r} history contains no private corpus paths")

        if failed:
            return 1

        if args.audit_local_refs:
            audit = audit_local_refs(args.publish_ref)
            if audit.replace_refs:
                print("⚠ local-repo audit: refs/replace entries present (a push ignores them):")
                for ref in audit.replace_refs:
                    print(f"  {ref}")
            if audit.nonstandard_refs:
                print("⚠ local-repo audit: refs pointing at a blob/tree (would publish objects):")
                for ref in audit.nonstandard_refs:
                    print(f"  {ref}")
            if audit.has_private_history:
                print("⚠ local-repo audit: private objects are reachable from other refs/stash")
                print("  (expected for local-only branches; these must never be pushed)")
                for entry in [*audit.refs, *audit.stashes]:
                    print(f"  {entry.ref}: {entry.private_path_count} private path(s), e.g. {', '.join(entry.sample)}")
            elif not audit.replace_refs:
                print("✓ local-repo audit: no private objects in other refs/stash")
        else:
            print("i skipped local-repo audit; add --audit-local-refs to scan other refs/stash")
        return 0
    except HygieneError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
