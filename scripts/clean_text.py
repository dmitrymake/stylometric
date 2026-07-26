"""Retired non-atomic corpus cleaner.

The canonical cleaner is fail-closed and publishes a complete snapshot.  This
legacy filename is retained only to give callers an actionable migration
error; it never imports NLP or writes corpus files.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/clean_text.py is retired because it overlaid partial outputs "
        "and ignored input failures. Use `stylo clean` or `./run.sh clean`."
    )


if __name__ == "__main__":
    main()
