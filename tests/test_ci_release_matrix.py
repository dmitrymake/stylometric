"""CI must exercise the same release surfaces users receive."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def test_ci_runs_full_and_focused_suites_under_lock_constraints():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m pytest tests -q -p no:cacheprovider" in text
    assert "Focused scientific and release contracts" in text
    assert text.count("--constraint requirements.lock") >= 2
    assert text.count("--no-build-isolation --constraint requirements.lock") >= 2
    assert "pip install --quiet numpy scipy" not in text
    assert "scripts/check_executable_source_inventory.py" in text
    assert "scripts/check_release_hygiene.py --publish-ref HEAD" in text
    assert "Verify canonical portable environment binding" in text
    assert text.count("verify_installed_environment") >= 3
    assert "node scripts/gen-site-data.mjs" in text
    assert "node scripts/gen-readme.mjs --check" in text
    assert "node scripts/check-provenance.mjs" in text
    assert "git diff --exit-code -- README.md site/src/generated" in text


def test_ci_builds_wheel_and_tests_git_free_archive():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m build --no-isolation" in text
    assert "dist/stylo-*.whl" in text
    assert "working-directory: /tmp" in text
    assert "git archive HEAD | tar -x -C /tmp/stylo-archive" in text
    assert "check_executable_source_inventory.py --archive" in text
    assert "check-provenance.mjs --archive" in text
    assert "working-directory: /tmp/stylo-archive" in text
