"""Retired UMAP entrypoint for unauthenticated loose model artifacts."""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/umap_vis.py is retired: its loose executable artifacts have "
        "no authenticated bundle contract. Rebuild this diagnostic against a "
        "non-executable, content-bound representation format before use."
    )


if __name__ == "__main__":
    main()
