"""Retired predictor for unauthenticated loose pickle artifacts."""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/predict.py is retired. Use `stylo predict "
        "--model-bundle-token TOKEN` so executable model bytes are verified "
        "against an external content commitment."
    )


if __name__ == "__main__":
    main()
