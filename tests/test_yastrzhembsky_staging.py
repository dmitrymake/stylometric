import hashlib
import importlib.util
import struct
from pathlib import Path

import pytest


def _load_script():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "stage_yastrzhembsky_transcription.py"
    )
    spec = importlib.util.spec_from_file_location(
        "stage_yastrzhembsky_transcription", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


staging = _load_script()


def test_page_map_binds_physical_and_printed_pages():
    pages = staging._page_map(95, 123, 89, 117)

    assert len(pages) == 29
    assert pages[0] == {
        "derived_page_index": 1,
        "source_pdf_page": 95,
        "printed_page": 89,
    }
    assert pages[-1] == {
        "derived_page_index": 29,
        "source_pdf_page": 123,
        "printed_page": 117,
    }


def test_page_map_rejects_mismatched_ranges():
    with pytest.raises(ValueError, match="equal length"):
        staging._page_map(95, 123, 89, 116)


def test_png_dimensions_are_read_without_image_reencoding(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + struct.pack(">II", 1772, 2797)
    )

    assert staging._png_size(path) == (1772, 2797)


def test_declared_hash_must_match(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"pinned scan")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert staging._verify_declared_file(path, digest, 11)["sha256"] == digest
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        staging._verify_declared_file(path, "0" * 64)


def test_verify_mode_never_creates_a_missing_render(tmp_path):
    with pytest.raises(FileNotFoundError):
        staging._render_page(
            tmp_path / "source.pdf",
            tmp_path / "page.png",
            1,
            300,
            force=False,
            allow_create=False,
        )


def test_verify_and_force_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        staging.main(["--verify-only", "--force"])
