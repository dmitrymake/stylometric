#!/usr/bin/env python3
"""Create a reproducible, non-scoring transcription staging area.

This script does not OCR, transcribe, normalise, or infer mixed-authorship
boundaries.  It verifies the pinned scans, renders one stable full-page PNG per
physical page, records exact page mappings and hashes, and proves whether the
derived PDFs contain an embedded text layer.  The resulting material is an
input to the independent two-key procedure, not benchmark text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
DEFAULT_SPEC = ROOT / "research" / "yastrzhembsky_spoof_v1.yaml"
DEFAULT_OUT_DIR = (
    ROOT / "data" / "yastrzhembsky_spoof_v1" / "transcription_staging"
)
DEFAULT_MANIFEST = (
    ROOT / "research" / "yastrzhembsky_transcription_staging_manifest_v1.json"
)
GENERATOR_VERSION = "1.2"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _file_record(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_declared_file(
    path: pathlib.Path, expected_sha256: str, expected_bytes: int | None = None
) -> dict[str, Any]:
    record = _file_record(path)
    if record["sha256"] != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: {record['sha256']}, "
            f"expected {expected_sha256}"
        )
    if expected_bytes is not None and record["bytes"] != expected_bytes:
        raise RuntimeError(
            f"size mismatch for {path}: {record['bytes']}, expected {expected_bytes}"
        )
    return record


def _tool_version(command: str) -> str:
    completed = subprocess.run(
        [command, "-v"], check=True, text=True, capture_output=True
    )
    first = (completed.stdout or completed.stderr).splitlines()
    return first[0].strip() if first else "unknown"


def _text_layer_stats(path: pathlib.Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
    )
    text = completed.stdout.decode("utf-8", "replace")
    visible = "".join(character for character in text if not character.isspace())
    return {
        "utf8_bytes_including_page_breaks": len(completed.stdout),
        "non_whitespace_characters": len(visible),
        "usable": bool(visible),
    }


def _png_size(path: pathlib.Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def _page_map(
    source_start: int, source_end: int, printed_start: int, printed_end: int
) -> list[dict[str, int]]:
    source_count = source_end - source_start + 1
    printed_count = printed_end - printed_start + 1
    if source_count < 1 or source_count != printed_count:
        raise ValueError(
            "source and printed page ranges must be positive and have equal length"
        )
    return [
        {
            "derived_page_index": offset + 1,
            "source_pdf_page": source_start + offset,
            "printed_page": printed_start + offset,
        }
        for offset in range(source_count)
    ]


def _render_page(
    source: pathlib.Path,
    destination: pathlib.Path,
    derived_page_index: int,
    dpi: int,
    *,
    force: bool,
    allow_create: bool,
) -> None:
    if destination.exists() and not force:
        return
    if not allow_create:
        raise FileNotFoundError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_stem = destination.with_name(destination.stem + ".part")
    temporary_png = pathlib.Path(str(temporary_stem) + ".png")
    temporary_png.unlink(missing_ok=True)
    subprocess.run(
        [
            "pdftoppm",
            "-f", str(derived_page_index),
            "-l", str(derived_page_index),
            "-r", str(dpi),
            "-gray",
            "-png",
            "-singlefile",
            str(source),
            str(temporary_stem),
        ],
        check=True,
    )
    if not temporary_png.is_file():
        raise RuntimeError(f"pdftoppm did not create {temporary_png}")
    temporary_png.replace(destination)


def _anchors(role: str, extraction: dict[str, Any]) -> dict[str, str]:
    names = (
        ("start_anchor", "first_text_anchor", "stop_before_anchor")
        if role == "spoof"
        else ("title_anchor",)
    )
    return {name: str(extraction[name]) for name in names if name in extraction}


def _role_specs(spec: dict[str, Any]):
    yield (
        "spoof",
        spec["publication"]["scan"],
        spec["publication"]["extraction"],
    )
    primary = spec["ground_truth_evidence"]["primary"]
    yield "evidence", primary, primary


def build_staging(
    spec_path: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    dpi: int = 300,
    force: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    if dpi < 150:
        raise ValueError("dpi must be at least 150 for transcription staging")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec_record = _file_record(spec_path)
    roles = []
    for role, source, extraction in _role_specs(spec):
        raw_path = ROOT / source["downloaded_file"]
        raw = _verify_declared_file(
            raw_path, str(source["sha256"]), int(source["bytes"])
        )
        derived_path = ROOT / extraction["derived_scan_file"]
        derived = _verify_declared_file(
            derived_path, str(extraction["derived_scan_sha256"])
        )
        page_map = _page_map(
            int(extraction["source_pdf_start_page"]),
            int(extraction["source_pdf_end_page"]),
            int(extraction["printed_start_page"]),
            int(extraction["printed_end_page"]),
        )
        render_records = []
        for page in page_map:
            destination = (
                out_dir / "renders" / role
                / f"page-{page['source_pdf_page']:03d}.png"
            )
            _render_page(
                derived_path,
                destination,
                page["derived_page_index"],
                dpi,
                force=force,
                allow_create=not verify_only,
            )
            width, height = _png_size(destination)
            render_records.append(
                {
                    **page,
                    **_file_record(destination),
                    "width_px": width,
                    "height_px": height,
                }
            )
        roles.append(
            {
                "role": role,
                "raw_scan": raw,
                "derived_scan": derived,
                "source_pdf_page_range": [
                    page_map[0]["source_pdf_page"],
                    page_map[-1]["source_pdf_page"],
                ],
                "printed_page_range": [
                    page_map[0]["printed_page"],
                    page_map[-1]["printed_page"],
                ],
                "anchors": _anchors(role, extraction),
                "embedded_text_layer": _text_layer_stats(derived_path),
                "renders": render_records,
            }
        )

    return {
        "schema_version": "1.0",
        "generator": {
            "path": _relative(pathlib.Path(__file__)),
            "version": GENERATOR_VERSION,
            "pdftoppm": _tool_version("pdftoppm"),
            "pdftotext": _tool_version("pdftotext"),
            "render_dpi": dpi,
            "render_mode": "grayscale_full_page_no_crop",
        },
        "case_id": str(spec["case_id"]),
        "benchmark_role": "public_development_gold_not_blind",
        "status": "transcription_staging_only_not_scoring_text",
        "spec": spec_record,
        "roles": roles,
        "transcription_layout": {
            "page_unit": "one_utf8_file_per_source_pdf_page",
            "key_a_pattern": _relative(out_dir / "keys" / "key_a" / "page-NNN.txt"),
            "key_b_pattern": _relative(out_dir / "keys" / "key_b" / "page-NNN.txt"),
            "reconciled_pattern": _relative(
                out_dir / "keys" / "reconciled" / "page-NNN.txt"
            ),
            "normalised_output": _relative(out_dir / "normalised" / "target.txt"),
            "page_markers": "metadata_only_outside_scored_text",
            "editorial_regions": (
                "transcribe_then_tag; exclude only after human source adjudication"
            ),
        },
        "transcription_gate": {
            "key_a": "pending",
            "key_b": "pending",
            "reconciliation": "pending",
            "third_review": "pending",
            "normalised_copy": "pending",
            "scorer_status": "blocked_pending_reconciled_double_key_text",
            "mixed_boundaries": "not_claimed_not_inferred",
            "ocr_policy": "navigation_and_discrepancy_triage_only",
        },
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        dumps_strict(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only and args.force:
        parser.error("--verify-only and --force are mutually exclusive")

    if not args.verify_only:
        for directory in (
            args.out_dir / "keys" / "key_a",
            args.out_dir / "keys" / "key_b",
            args.out_dir / "keys" / "reconciled",
            args.out_dir / "normalised",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    manifest = build_staging(
        args.spec.resolve(),
        args.out_dir.resolve(),
        dpi=args.dpi,
        force=args.force,
        verify_only=args.verify_only,
    )
    encoded = _canonical_bytes(manifest)
    if args.verify_only:
        if not args.manifest.is_file():
            raise FileNotFoundError(args.manifest)
        existing = args.manifest.read_bytes()
        if existing != encoded:
            raise RuntimeError(
                f"staging manifest mismatch: {_sha256(args.manifest)} != "
                f"{hashlib.sha256(encoded).hexdigest()}"
            )
    else:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_bytes(encoded)
    print(
        dumps_strict(
            {
                "manifest": _relative(args.manifest),
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                "status": manifest["status"],
                "scorer_status": manifest["transcription_gate"]["scorer_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
