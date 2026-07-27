#!/usr/bin/env python3
"""Materialize one already-pinned FEB exploratory corpus source."""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.corpus_tools.feb_vnext import (  # noqa: E402
    FEBAcquisitionError,
    FEBHTTPTransport,
    load_pinned_feb_work_spec,
    materialize_pinned_feb_work,
)
from stylo.jsonio import dumps_strict  # noqa: E402


OUTPUT_PARENT = (
    ROOT / "docs" / "exploratory" / "lobo_vnext" / "corpora" / "feb"
)


class FEBCLIError(ValueError):
    """The pinned FEB CLI rejected an unsafe request."""


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise FEBCLIError(
                f"{label} must not contain symlink components: {component}"
            )


def _require_spec(path: pathlib.Path) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    _reject_symlink_components(candidate, label="pinned FEB spec")
    if candidate.is_symlink() or not candidate.is_file():
        raise FEBCLIError(
            "pinned FEB spec must be a regular non-symlink file: "
            f"{candidate}"
        )
    return candidate.resolve(strict=True)


def _output_parent() -> pathlib.Path:
    candidate = OUTPUT_PARENT
    _reject_symlink_components(candidate, label="FEB output parent")
    allowed = (
        ROOT / "docs" / "exploratory" / "lobo_vnext" / "corpora" / "feb"
    ).resolve(strict=False)
    if candidate.resolve(strict=False) != allowed:
        raise FEBCLIError(
            "FEB output parent must be the fixed ignored exploratory "
            f"namespace {allowed}"
        )
    if candidate.exists() and (
        candidate.is_symlink() or not candidate.is_dir()
    ):
        raise FEBCLIError(
            f"FEB output parent must be a real directory: {candidate}"
        )
    return candidate


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not 0 < number <= 300:
        raise argparse.ArgumentTypeError("must be in the interval (0, 300]")
    return number


def _bounded_attempts(value: str) -> int:
    try:
        number = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= number <= 20:
        raise argparse.ArgumentTypeError("must be in the interval [1, 20]")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the exact pinned FEB response and its bounded main "
            "narrative below the ignored exploratory LOBO-vNext corpus "
            "namespace. No cache, estimator, fit, or public writer is "
            "reachable."
        )
    )
    parser.add_argument(
        "--pinned-spec",
        type=pathlib.Path,
        required=True,
        help="strict self-hashed PinnedFEBWorkSpec JSON",
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help="explicit single-line HTTP User-Agent",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument(
        "--max-attempts",
        type=_bounded_attempts,
        default=6,
    )
    return parser


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    try:
        spec = load_pinned_feb_work_spec(_require_spec(args.pinned_spec))
        materialized = materialize_pinned_feb_work(
            spec,
            output_parent=_output_parent(),
            transport=FEBHTTPTransport(
                user_agent=args.user_agent,
                timeout_seconds=args.timeout_seconds,
                max_attempts=args.max_attempts,
            ),
        )
    except (OSError, UnicodeError, ValueError, FEBAcquisitionError) as exc:
        raise FEBCLIError(f"pinned FEB acquisition rejected: {exc}") from exc
    relative = materialized.root.relative_to(ROOT).as_posix()
    return {
        "status": "exploratory_feb_source_materialized_no_fit",
        "generation_id": spec.generation_id,
        "pinned_spec_sha256": spec.self_hash,
        "receipt_sha256": materialized.receipt.self_hash,
        "work_id": spec.work_id,
        "resumed": materialized.resumed,
        "namespace_relative_path": relative,
        "fit_performed": False,
        "confirmatory_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(argv)
    except FEBCLIError as exc:
        print(f"FEB acquisition rejected: {exc}", file=sys.stderr)
        return 2
    print(dumps_strict(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
