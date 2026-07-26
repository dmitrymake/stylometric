#!/usr/bin/env python3
"""Validate the executable-source inventory in a checkout or release archive."""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylo.release.source_inventory import (  # noqa: E402
    SourceInventoryError,
    check_source_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        action="store_true",
        help="validate a Git-free archive (local-only Python must be absent)",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="print the canonical release Python paths after validation",
    )
    args = parser.parse_args()
    try:
        report = check_source_inventory(ROOT, require_git=not args.archive)
    except SourceInventoryError as exc:
        print(f"source inventory cannot be evaluated: {exc}", file=sys.stderr)
        return 2
    if args.show_paths:
        print("\n".join(report.snapshot.paths))
    print(
        f"release Python snapshot: count={report.snapshot.file_count} "
        f"sha256={report.snapshot.paths_sha256}"
    )
    if report.issues:
        for issue in report.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print("executable-source inventory: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
