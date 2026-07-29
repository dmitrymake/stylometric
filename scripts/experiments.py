"""Hard-disabled shim for retired automatic LOBO experiments."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/experiments.py is retired historical source: legacy experiment "
        "artifacts are no longer created. Use `stylo evaluate`."
    )


if __name__ == "__main__":
    main()
