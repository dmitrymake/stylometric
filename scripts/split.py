"""Retired direct fragment writer.

The canonical splitter publishes train, unknown, and chunk-map data inside one
immutable generation selected by a single atomic pointer.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/split.py is retired because it wrote partial fragment roots. "
        "Use `stylo split` or `./run.sh split`."
    )


if __name__ == "__main__":
    main()
