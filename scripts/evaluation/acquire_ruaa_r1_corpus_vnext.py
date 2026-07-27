#!/usr/bin/env python3
"""Materialize the exact pinned hybrid RuAA R1 exploratory corpus."""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.corpus_tools.feb_vnext import (  # noqa: E402
    FEBHTTPTransport,
)
from stylo.corpus_tools.ruaa_r1_acquisition import (  # noqa: E402
    R1AcquisitionAuditError,
    R1AcquisitionError,
    load_r1_acquisition_manifest,
    materialize_r1_acquisition,
)
from stylo.corpus_tools.wikisource_campaign import (  # noqa: E402
    HTTPJSONTransport,
)
from stylo.jsonio import dumps_strict  # noqa: E402


OUTPUT_PARENT = (
    ROOT
    / "docs"
    / "exploratory"
    / "lobo_vnext"
    / "corpora"
    / "ruaa_r1_hybrid"
)


class R1AcquisitionCLIError(ValueError):
    """The hybrid acquisition CLI rejected an unsafe request."""


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise R1AcquisitionCLIError(
                f"{label} must not contain symlink components: {component}"
            )


def _require_manifest(path: pathlib.Path) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    _reject_symlink_components(candidate, label="R1 acquisition manifest")
    if candidate.is_symlink() or not candidate.is_file():
        raise R1AcquisitionCLIError(
            "R1 acquisition manifest must be a regular non-symlink file: "
            f"{candidate}"
        )
    return candidate.resolve(strict=True)


def _output_parent() -> pathlib.Path:
    candidate = OUTPUT_PARENT
    _reject_symlink_components(
        candidate,
        label="R1 hybrid exploratory output parent",
    )
    allowed = (
        ROOT
        / "docs"
        / "exploratory"
        / "lobo_vnext"
        / "corpora"
        / "ruaa_r1_hybrid"
    ).resolve(strict=False)
    if candidate.resolve(strict=False) != allowed:
        raise R1AcquisitionCLIError(
            "R1 output parent must be the fixed ignored exploratory "
            f"namespace {allowed}"
        )
    if candidate.exists() and (
        candidate.is_symlink() or not candidate.is_dir()
    ):
        raise R1AcquisitionCLIError(
            f"R1 output parent must be a real directory: {candidate}"
        )
    return candidate


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not 0 < number <= 300:
        raise argparse.ArgumentTypeError(
            "must be in the interval (0, 300]"
        )
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
            "Materialize one exact reviewed Wikisource campaign plus the "
            "one pinned FEB work, audit every explicit literal text, and "
            "publish one immutable ignored exploratory corpus generation. "
            "No model, "
            "representation cache, fit, metric, or public writer is reachable."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=pathlib.Path,
        help="strict self-hashed R1AcquisitionManifest JSON",
    )
    parser.add_argument(
        "--wikisource-user-agent",
        required=True,
        help="explicit single-line Wikisource HTTP User-Agent",
    )
    parser.add_argument(
        "--feb-user-agent",
        required=True,
        help="explicit single-line FEB HTTP User-Agent",
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
        manifest = load_r1_acquisition_manifest(
            _require_manifest(args.manifest)
        )
        materialized = materialize_r1_acquisition(
            manifest,
            output_parent=_output_parent(),
            wikisource_transport=HTTPJSONTransport(
                user_agent=args.wikisource_user_agent,
                timeout_seconds=args.timeout_seconds,
                max_attempts=args.max_attempts,
            ),
            feb_transport=FEBHTTPTransport(
                user_agent=args.feb_user_agent,
                timeout_seconds=args.timeout_seconds,
                max_attempts=args.max_attempts,
            ),
        )
    except R1AcquisitionAuditError as exc:
        relative_report = (
            exc.report_path.relative_to(ROOT).as_posix()
            if exc.report_path.is_relative_to(ROOT)
            else exc.report_path.as_posix()
        )
        raise R1AcquisitionCLIError(
            "R1 hybrid text-quality audit blocked publication; "
            f"report={relative_report}; report_sha256={exc.report.self_hash}"
        ) from exc
    except (
        OSError,
        UnicodeError,
        ValueError,
        R1AcquisitionError,
    ) as exc:
        raise R1AcquisitionCLIError(
            f"R1 hybrid acquisition rejected: {exc}"
        ) from exc
    relative = materialized.root.relative_to(ROOT).as_posix()
    return {
        "status": "exploratory_ruaa_r1_corpus_materialized_no_fit",
        "generation_id": manifest.generation_id,
        "manifest_sha256": manifest.self_hash,
        "acquisition_receipt_sha256": materialized.receipt.self_hash,
        "text_quality_audit_sha256": materialized.audit_report.self_hash,
        "work_count": len(manifest.included_work_ids),
        "resumed": materialized.resumed,
        "namespace_relative_path": relative,
        "fit_performed": False,
        "confirmatory_authorized": False,
        "public_output_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(argv)
    except R1AcquisitionCLIError as exc:
        print(f"R1 acquisition rejected: {exc}", file=sys.stderr)
        return 2
    print(dumps_strict(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
