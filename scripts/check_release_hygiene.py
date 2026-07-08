#!/usr/bin/env python3
"""Release gate for public repository hygiene.

Default mode checks the current index. Add --history before a public release to
verify that old corpus blobs are no longer reachable from any ref.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


PRIVATE_PREFIXES = (
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


def git(args: list[str]) -> list[str]:
    out = subprocess.check_output(["git", *args], text=True)
    return [line for line in out.splitlines() if line]


def is_private_path(path: str) -> bool:
    return path.startswith(PRIVATE_PREFIXES)


def fail(title: str, rows: list[str]) -> int:
    print(f"✗ {title}", file=sys.stderr)
    for row in rows[:80]:
        print(f"  {row}", file=sys.stderr)
    if len(rows) > 80:
        print(f"  ... {len(rows) - 80} more", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true",
                        help="also scan all reachable git history")
    args = parser.parse_args()

    tracked = [p for p in git(["ls-files", "--", *PRIVATE_PREFIXES]) if is_private_path(p)]
    if tracked:
        return fail("private corpus paths are tracked in the index", tracked)
    print("✓ index contains no private corpus paths")

    if args.history:
        rows = git(["rev-list", "--all", "--objects", "--", *PRIVATE_PREFIXES])
        historical = []
        for row in rows:
            _sha, _sep, path = row.partition(" ")
            if path and is_private_path(path):
                historical.append(path)
        if historical:
            return fail("private corpus paths are still reachable in git history", historical)
        print("✓ git history contains no private corpus paths")
    else:
        print("i skipped history scan; run `python scripts/check_release_hygiene.py --history` before public release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
