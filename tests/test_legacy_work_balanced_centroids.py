"""The old LOBO entrypoint must remain a non-writing migration shim."""
from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_legacy_lobo_runner_is_hard_disabled(tmp_path):
    before = set(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lobo_cv.py")],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "retired" in (result.stdout + result.stderr)
    assert set(tmp_path.iterdir()) == before
