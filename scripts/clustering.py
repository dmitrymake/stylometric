"""Retired clustering entrypoint for unauthenticated loose model artifacts.

The historical implementation loaded ``vectorizer_fitted.pkl``,
``scaler_delta.pkl`` and an object-typed ``authors.npy`` directly from a
caller-selected directory.  Those files are not part of the authenticated
deployment bundle and therefore cannot be deserialized by an executable
release script.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/clustering.py is retired: its loose executable artifacts have "
        "no authenticated bundle contract. Rebuild this diagnostic against a "
        "non-executable, content-bound representation format before use."
    )


if __name__ == "__main__":
    main()
