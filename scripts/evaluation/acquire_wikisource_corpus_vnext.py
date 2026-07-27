#!/usr/bin/env python3
"""Materialize one already-pinned exploratory Wikisource corpus generation.

This command is intentionally not a source-discovery interface.  It accepts
only a strict, self-hashed campaign spec containing exact page revisions,
render hashes, extraction hashes, and part order.  Output is confined to the
ignored exploratory corpus namespace and no model fit is reachable here.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.corpus_tools.wikisource_campaign import (  # noqa: E402
    HTTPJSONTransport,
    WikisourceCampaignError,
    load_campaign_spec,
    materialize_campaign,
)
from stylo.jsonio import dumps_strict  # noqa: E402


OUTPUT_PARENT = ROOT / "docs" / "exploratory" / "lobo_vnext" / "corpora"


class WikisourceCampaignCLIError(ValueError):
    """The pinned campaign CLI rejected an unsafe request."""


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise WikisourceCampaignCLIError(
                f"{label} must not contain symlink components: {component}"
            )


def _require_spec(path: pathlib.Path) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    _reject_symlink_components(candidate, label="campaign spec")
    if candidate.is_symlink() or not candidate.is_file():
        raise WikisourceCampaignCLIError(
            f"campaign spec must be a regular non-symlink file: {candidate}"
        )
    return candidate.resolve(strict=True)


def _output_parent() -> pathlib.Path:
    candidate = OUTPUT_PARENT
    _reject_symlink_components(candidate, label="exploratory output parent")
    allowed = (
        ROOT / "docs" / "exploratory" / "lobo_vnext" / "corpora"
    ).resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    if resolved != allowed:
        raise WikisourceCampaignCLIError(
            "campaign output parent must be the fixed ignored exploratory "
            f"namespace {allowed}"
        )
    if candidate.exists() and (candidate.is_symlink() or not candidate.is_dir()):
        raise WikisourceCampaignCLIError(
            f"campaign output parent must be a real directory: {candidate}"
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
            "Materialize an exact pinned Wikisource campaign below the "
            "ignored exploratory LOBO-vNext corpus namespace. This command "
            "does no discovery, title resolution, ordering inference, cache "
            "construction, or model fit."
        )
    )
    parser.add_argument(
        "--campaign-spec",
        type=pathlib.Path,
        required=True,
        help="strict self-hashed campaign spec whose revisions are already pinned",
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help="explicit single-line HTTP User-Agent identifying this acquisition",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="per-request network timeout; operational only (default: 30)",
    )
    parser.add_argument(
        "--max-attempts",
        type=_bounded_attempts,
        default=6,
        help="bounded attempts for HTTP 429/5xx/transient failures (default: 6)",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    spec_path = _require_spec(args.campaign_spec)
    output_parent = _output_parent()
    try:
        spec = load_campaign_spec(spec_path)
        transport = HTTPJSONTransport(
            user_agent=args.user_agent,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
        )
        materialized = materialize_campaign(
            spec,
            output_parent=output_parent,
            transport=transport,
        )
    except (OSError, UnicodeError, ValueError, WikisourceCampaignError) as exc:
        raise WikisourceCampaignCLIError(
            f"pinned Wikisource campaign rejected: {exc}"
        ) from exc
    relative = materialized.root.relative_to(ROOT).as_posix()
    return {
        "status": "exploratory_corpus_materialized_no_fit",
        "generation_id": spec.generation_id,
        "campaign_spec_sha256": spec.self_hash,
        "campaign_receipt_sha256": materialized.receipt.self_hash,
        "work_count": len(spec.works),
        "resumed": materialized.resumed,
        "namespace_relative_path": relative,
        "fit_performed": False,
        "confirmatory_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(argv)
    except WikisourceCampaignCLIError as exc:
        print(f"Wikisource campaign rejected: {exc}", file=sys.stderr)
        return 2
    print(dumps_strict(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
