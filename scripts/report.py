"""Compatibility entrypoint for the fail-closed canonical report builder."""
from __future__ import annotations

from stylo.report.build import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
