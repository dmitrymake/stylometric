"""Retired unbounded legacy LOBO runner.

The historical script defaulted to all CPU workers and wrote mutable report
siblings without a run identity.  It is retained only as a migration shim.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/lobo_cv.py is retired: it used unbounded parallelism and "
        "unregistered mutable outputs. Use the bounded registered runner "
        "`scripts/evaluation/run_stylo_lobo_validation.py`."
    )


if __name__ == "__main__":
    main()
