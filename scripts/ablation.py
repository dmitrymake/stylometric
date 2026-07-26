"""Retired legacy bleaching-ablation runner.

The historical script independently rebuilt the corpus and overwrote an
unregistered report.  It is retained only as a migration shim.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/ablation.py is retired: its independently rebuilt dataset "
        "and mutable output are not claim-bearing. Use the frozen-panel runner "
        "`scripts/evaluation/run_work_balanced_ablation_screen.py`."
    )


if __name__ == "__main__":
    main()
