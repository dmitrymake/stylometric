"""Retired loose-artifact trainer.

This path used to overwrite ``data/model.pkl`` and several unrelated siblings
without a generation manifest.  It is intentionally non-runnable: the
canonical trainer publishes one immutable, content-addressed bundle.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/train.py is retired because loose model artifacts can mix "
        "generations. Use `./run.sh train` (or `stylo train`) and retain the "
        "printed bundle_token."
    )


if __name__ == "__main__":
    main()
