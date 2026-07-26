"""Runtime resources and workspace-only training behave honestly after packaging."""
from __future__ import annotations

import importlib.resources
import pathlib
import tomllib

import pytest

from stylo.config import DEFAULT_CONFIG_PATH, load_config
from stylo.corpus_tools.fetch_classics import load_classics_manifest
from stylo.lang import load_author_meta
from stylo.pipeline import train

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["default.yaml", "authors.json", "classics.yaml"])
def test_runtime_resource_is_packaged_and_matches_reviewed_workspace_copy(name):
    packaged = importlib.resources.files("stylo.resources").joinpath(name)
    assert packaged.is_file()
    assert packaged.read_bytes() == (ROOT / "configs" / name).read_bytes()


def test_default_config_and_author_metadata_load_from_package_resources():
    assert DEFAULT_CONFIG_PATH.is_file()
    assert load_config().chunking.chunk_size == 500
    authors = load_author_meta()
    assert "unknown" in authors
    assert len(authors) > 1


def test_packaged_classics_manifest_loads_and_explicit_missing_override_fails(tmp_path):
    entries = load_classics_manifest()
    assert entries and {"author", "title"}.issubset(entries[0])
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_classics_manifest(tmp_path / "absent.yaml")


def test_requests_is_declared_as_a_core_runtime_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(dep.startswith("requests") for dep in project["dependencies"])


def test_training_reports_honest_workspace_only_contract(tmp_path):
    with pytest.raises(train.WorkspaceRequiredError, match="Git source workspace"):
        train._require_source_workspace(tmp_path)
    if (ROOT / ".git").exists():
        assert train._require_source_workspace(ROOT) == ROOT
