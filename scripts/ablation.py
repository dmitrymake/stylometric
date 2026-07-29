"""Hard-disabled migration shim for completed ablation screens.

The registered work-balanced screen is complete.  Its frozen historical result
remains documented, and there is no general live replacement for this command.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/ablation.py is retired: the registered work-balanced ablation "
        "screen is complete. Its frozen historical result is preserved at "
        "`research/work_balanced/exploratory_ablation_screen.md`. There is no "
        "general live replacement for this command."
    )


if __name__ == "__main__":
    main()
