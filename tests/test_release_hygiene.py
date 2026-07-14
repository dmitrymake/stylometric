import pathlib
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_visualization_dependencies_are_not_core_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = {dep.split(";", 1)[0].strip().lower() for dep in data["project"]["dependencies"]}

    assert not any(dep.startswith("umap-learn") for dep in core)


def test_spacy_runtime_import_dependencies_are_explicit():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = {dep.split(";", 1)[0].strip().lower() for dep in data["project"]["dependencies"]}

    assert any(dep.startswith("click") for dep in core)


def test_uv_uses_canonical_python_version():
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11"


def test_uv_lock_is_not_the_canonical_lockfile():
    ignored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "uv.lock" in ignored
    assert (ROOT / "requirements.lock").exists()


def test_stylo_stack_is_explicit_experiment_not_default_headline():
    from stylo.eval.lobo import make_factory
    from stylo.eval.final import DEFAULT_SPECS, ECE_SPECS
    from stylo.config import load_config

    from stylo.eval.work_weighting import CHUNK_WEIGHTED_LEGACY
    assert "stylo_stack" not in DEFAULT_SPECS
    assert "stylo_stack" in ECE_SPECS
    assert callable(make_factory("stylo_stack", load_config(), weighting=CHUNK_WEIGHTED_LEGACY))
