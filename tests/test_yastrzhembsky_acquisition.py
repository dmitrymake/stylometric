import hashlib
import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "fetch_yastrzhembsky_spoof.py"
    spec = importlib.util.spec_from_file_location("fetch_yastrzhembsky_spoof", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


acquisition = _load_script()


def test_verify_file_accepts_exact_artifact(tmp_path):
    path = tmp_path / "scan.pdf"
    payload = b"immutable scan bytes"
    path.write_bytes(payload)

    report = acquisition._verify_file(
        path,
        sha256=hashlib.sha256(payload).hexdigest(),
        n_bytes=len(payload),
    )

    assert report["bytes"] == len(payload)


def test_verify_file_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"wrong")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        acquisition._verify_file(path, sha256="0" * 64)
