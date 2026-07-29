"""Hard-disabled shim for the completed legacy LOBO run."""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/lobo_cv.py is retired: the legacy run is complete and "
        "preserved only as historical evidence under "
        "`research/evidence/stylo_lobo_validation_v1/`. There is no general "
        "drop-in replacement. New registered runs use the separate "
        "`lobo_vnext` control plane."
    )


if __name__ == "__main__":
    main()
