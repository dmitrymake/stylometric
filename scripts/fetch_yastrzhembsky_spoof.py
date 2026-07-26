#!/usr/bin/env python3
"""Fetch, verify, and page-extract the Yastrzhembsky SPOOF-RU intake."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile

import requests
import yaml
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402


USER_AGENT = "authorship-research/1.0 (public-domain benchmark acquisition)"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: pathlib.Path, *, sha256: str, n_bytes: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if n_bytes is not None and actual_bytes != n_bytes:
        raise RuntimeError(
            f"size mismatch for {path}: {actual_bytes}, expected {n_bytes}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: {actual_sha256}, expected {sha256}"
        )
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}


def _download(url: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=(20, 120),
        stream=True,
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def _extract_pages(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    start_page: int,
    end_page: int,
) -> None:
    if start_page < 1 or end_page < start_page:
        raise ValueError("invalid PDF page range")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yastrzhembsky-pages-") as temporary:
        directory = pathlib.Path(temporary)
        pattern = directory / "page-%04d.pdf"
        subprocess.run(
            [
                "pdfseparate",
                "-f",
                str(start_page),
                "-l",
                str(end_page),
                str(source),
                str(pattern),
            ],
            check=True,
        )
        pages = sorted(directory.glob("page-*.pdf"))
        expected = end_page - start_page + 1
        if len(pages) != expected:
            raise RuntimeError(f"pdfseparate produced {len(pages)} pages, expected {expected}")
        temporary_output = destination.with_name(destination.name + ".part")
        subprocess.run(
            ["pdfunite", *(str(page) for page in pages), str(temporary_output)],
            check=True,
        )
        temporary_output.replace(destination)


def acquire(spec_path: pathlib.Path, *, verify_only: bool = False) -> dict:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    items = [
        (
            "spoof",
            spec["publication"]["scan"],
            spec["publication"]["extraction"],
        ),
        (
            "evidence",
            spec["ground_truth_evidence"]["primary"],
            spec["ground_truth_evidence"]["primary"],
        ),
    ]
    report = {"case_id": spec["case_id"], "verify_only": verify_only, "items": []}
    for role, source, extraction in items:
        raw = pathlib.Path(source["downloaded_file"])
        if not raw.exists():
            if verify_only:
                raise FileNotFoundError(raw)
            _download(source["download_url"], raw)
        raw_report = _verify_file(
            raw, sha256=source["sha256"], n_bytes=int(source["bytes"])
        )

        derived = pathlib.Path(extraction["derived_scan_file"])
        if not derived.exists():
            if verify_only:
                raise FileNotFoundError(derived)
            _extract_pages(
                raw,
                derived,
                start_page=int(extraction["source_pdf_start_page"]),
                end_page=int(extraction["source_pdf_end_page"]),
            )
        derived_report = _verify_file(
            derived, sha256=extraction["derived_scan_sha256"]
        )
        report["items"].append(
            {"role": role, "raw": raw_report, "derived": derived_report}
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", default="research/yastrzhembsky_spoof_v1.yaml"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report = acquire(pathlib.Path(args.spec), verify_only=args.verify_only)
    print(dumps_strict(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
