import importlib.util
from pathlib import Path


def _load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_package = _load_script("build_breakthrough_synthetic").build_package
run = _load_script("run_breakthrough_pilot").run


def test_synthetic_breakthrough_stack_runs_end_to_end(tmp_path):
    package = tmp_path / "synthetic"
    built = build_package(package, seed=17)

    report = run(package, bootstrap_iters=5)

    assert built["n_documents"] == 38
    assert report["status"] == "integration_control_only"
    assert report["scientific_claim_allowed"] is False
    assert report["reference_blind_score"]["protocol_binding"]["endpoint_counts"] == {
        "idio_shift": 1,
        "mixed_authorship": 2,
        "spoof": 1,
    }
    assert report["reference_blind_score"]["authorship"]["accuracy"] == 1.0
    assert (
        report["reference_blind_score"]["document_classification"]["accuracy"]
        == 1.0
    )
    assert (
        report["reference_blind_score"]["segmentation"]["aggregate"]
        ["single_author_false_positive_rate"]
        == 0.0
    )
    for factor in ("source", "edition"):
        experiment = report["purged_factor_experiments"][factor]
        assert experiment["plan"]["test_coverage"] == 1.0
        assert experiment["plan"]["possible_split_coverage"] == 1.0
        assert experiment["baseline_char_lr"]["overall"]["accuracy"]["point"] == 1.0
        assert (
            experiment["paired_edition_residualizer"]["overall"]["accuracy"]["point"]
            == 1.0
        )
