#!/usr/bin/env python3
"""Pin a reviewed Wikisource discovery candidate without publishing a corpus."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.corpus_tools.wikisource_campaign import (  # noqa: E402
    HTTPJSONTransport,
)
from stylo.corpus_tools.wikisource_discovery import (  # noqa: E402
    WikisourceDiscoveryError,
    load_discovery_candidate,
    pin_discovery_candidate,
    write_campaign_spec_create_if_absent,
)
from stylo.jsonio import dumps_strict  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a fully resolved discovery candidate against exact "
            "Wikisource revisions and write one immutable pinned acquisition "
            "campaign. This command does not publish corpus text or run a fit."
        )
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=pathlib.Path,
        help="strict, self-hashed discovery candidate JSON",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=pathlib.Path,
        help=(
            "ignored/external directory for immutable exact API response "
            "cache files"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="create-if-absent pinned campaign-spec JSON",
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help="descriptive MediaWiki HTTP User-Agent including contact details",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-delay-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=128 * 1024 * 1024,
    )
    return parser


def _input_file(path: pathlib.Path) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    absolute = candidate.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise WikisourceDiscoveryError(
                f"candidate path must not contain symlinks: {component}"
            )
    if not absolute.is_file():
        raise WikisourceDiscoveryError(
            f"candidate is not a regular file: {absolute}"
        )
    return absolute.resolve(strict=True)


def _ignored_or_external_cache(path: pathlib.Path) -> pathlib.Path:
    candidate = path if path.is_absolute() else ROOT / path
    absolute = candidate.absolute()
    try:
        relative = absolute.relative_to(ROOT)
    except ValueError:
        return absolute
    try:
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "check-ignore",
                "--quiet",
                "--",
                relative.as_posix(),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise WikisourceDiscoveryError(
            "an in-repository cache requires git check-ignore validation"
        ) from exc
    if ignored.returncode != 0:
        detail = ignored.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise WikisourceDiscoveryError(
            "cache inside the repository must be git-ignored"
            f"{suffix}"
        )
    return absolute


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        candidate = load_discovery_candidate(_input_file(args.candidate))
        # This check deliberately precedes transport construction and all cache
        # I/O so blocked discovery is a true pre-network rejection.
        candidate.assert_ready()
        transport = HTTPJSONTransport(
            user_agent=args.user_agent,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
            backoff_seconds=args.backoff_seconds,
            max_delay_seconds=args.max_delay_seconds,
            max_response_bytes=args.max_response_bytes,
        )
        result = pin_discovery_candidate(
            candidate,
            cache_dir=_ignored_or_external_cache(args.cache_dir),
            transport=transport,
        )
        output = (
            args.output
            if args.output.is_absolute()
            else ROOT / args.output
        )
        written = write_campaign_spec_create_if_absent(
            result.campaign_spec,
            output,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"pin_wikisource_campaign: rejected: {exc}", file=sys.stderr)
        return 2
    print(
        dumps_strict(
            {
                "status": "pinned_acquisition_spec_only",
                "candidate_hash": result.candidate_hash,
                "campaign_generation_id": (
                    result.campaign_spec.generation_id
                ),
                "campaign_self_hash": result.campaign_spec.self_hash,
                "work_count": len(result.campaign_spec.works),
                "output": written.as_posix(),
                "corpus_published": False,
                "fit_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
