#!/usr/bin/env python3
"""Prepare the exact owner-selected RuAA R1 packet without model execution."""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.config import load_config  # noqa: E402
from stylo.eval.lobo_vnext_prepare import (  # noqa: E402
    R1PacketPreparationError,
    prepare_r1_packet,
)
from stylo.jsonio import dumps_strict  # noqa: E402


class R1PreparationCLIError(ValueError):
    """The explicit R1 preparation request is unsafe or incomplete."""


def _path(value: pathlib.Path) -> pathlib.Path:
    return value if value.is_absolute() else ROOT / value


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise R1PreparationCLIError(
                f"{label} must not contain symlink components: {component}"
            )


def _require_regular_file(value: pathlib.Path, *, label: str) -> pathlib.Path:
    candidate = _path(value)
    _reject_symlink_components(candidate, label=label)
    if candidate.is_symlink() or not candidate.is_file():
        raise R1PreparationCLIError(
            f"{label} must be a regular non-symlink file: {candidate}"
        )
    return candidate.resolve(strict=True)


def _require_source_root(value: pathlib.Path) -> pathlib.Path:
    candidate = _path(value)
    _reject_symlink_components(candidate, label="source root")
    if candidate.is_symlink() or not candidate.is_dir():
        raise R1PreparationCLIError(
            f"source root must be a real directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def _require_exploratory_output_parent(value: pathlib.Path) -> pathlib.Path:
    candidate = _path(value)
    _reject_symlink_components(candidate, label="output parent")
    if candidate.is_symlink():
        raise R1PreparationCLIError(
            f"output parent must not be a symlink: {candidate}"
        )
    resolved = candidate.resolve(strict=False)
    allowed = (ROOT / "docs" / "exploratory" / "lobo_vnext").resolve(
        strict=False
    )
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as exc:
        raise R1PreparationCLIError(
            f"output parent must stay below {allowed}: {resolved}"
        ) from exc
    if not relative.parts:
        raise R1PreparationCLIError(
            "output parent must name a packet namespace below "
            f"{allowed}"
        )
    if resolved.exists() and not resolved.is_dir():
        raise R1PreparationCLIError(
            f"output parent exists but is not a directory: {resolved}"
        )
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the exact owner-selected RuAA R1 LOBO-vNext packet. "
            "This command cannot construct a representation cache, model "
            "factory, estimator, checkpoint, prediction, or final result."
        )
    )
    parser.add_argument(
        "--approved-r1",
        action="store_true",
        required=True,
        help="acknowledge the exact R1 scientific selection",
    )
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--legacy-source-manifest", type=pathlib.Path, required=True
    )
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--output-parent", type=pathlib.Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    source_root = _require_source_root(args.source_root)
    source_manifest = _require_regular_file(
        args.legacy_source_manifest, label="legacy source manifest"
    )
    config_path = _require_regular_file(args.config, label="config")
    output_parent = _require_exploratory_output_parent(args.output_parent)
    try:
        cfg = load_config(config_path)
        packet = prepare_r1_packet(
            source_root=source_root,
            legacy_source_manifest=source_manifest,
            output_parent=output_parent,
            cfg=cfg,
        )
    except (OSError, UnicodeError, ValueError, R1PacketPreparationError) as exc:
        raise R1PreparationCLIError(f"R1 packet rejected: {exc}") from exc
    return {
        "status": "owner_selected_exploratory_packet_prepared_no_fit",
        "generation_id": packet.corpus_manifest.generation_id,
        "packet_self_hash": packet.packet_manifest.self_hash,
        "corpus_manifest_sha256": packet.corpus_manifest.self_hash,
        "source_selection_receipt_sha256": (
            packet.source_selection_receipt.self_hash
        ),
        "source_candidate_inventory_sha256": (
            packet.source_candidate_inventory.self_hash
        ),
        "candidate_inventory_sha256": packet.candidate_inventory.self_hash,
        "content_component_manifest_sha256": (
            packet.content_manifest.self_hash
        ),
        "fold_manifest_sha256": packet.fold_manifest.self_hash,
        "campaign_manifest_sha256": packet.campaign_manifest.self_hash,
        "representation_receipt_sha256": (
            packet.representation_receipt.self_hash
        ),
        "selected_work_count": len(packet.corpus_manifest.works),
        "fit_performed": False,
        "confirmatory_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt = run(argv)
    except R1PreparationCLIError as exc:
        print(f"LOBO-vNext R1 preparation rejected: {exc}", file=sys.stderr)
        return 2
    print(
        dumps_strict(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
