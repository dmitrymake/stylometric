import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    "run_kolokol_gate.py",
    "run_sovremennik_gate.py",
    "run_nekrasov_panaeva_gate.py",
)


def _load(filename: str):
    name = f"output_safety_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", RUNNERS)
def test_corrected_gate_runner_default_cannot_overwrite_historical(filename):
    runner = _load(filename)

    default = runner.parse_args([])

    assert default.out == runner.DEFAULT_OUT
    assert default.out.resolve() != runner.HISTORICAL_OUT.resolve()
    with pytest.raises(SystemExit):
        runner.parse_args(["--out", str(runner.HISTORICAL_OUT)])


@pytest.mark.parametrize("filename", RUNNERS)
def test_corrected_gate_runner_requires_explicit_historical_override(filename):
    runner = _load(filename)

    args = runner.parse_args(
        ["--out", str(runner.HISTORICAL_OUT), "--overwrite-historical"]
    )

    assert args.out == runner.HISTORICAL_OUT

