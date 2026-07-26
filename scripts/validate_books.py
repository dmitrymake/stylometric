"""Retired book validator for unauthenticated loose vector artifacts."""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/validate_books.py is retired with data/train_vectors.pkl and "
        "data/scaler_delta.pkl. Use `stylo evaluate` on a validated corpus."
    )


if __name__ == "__main__":
    main()
