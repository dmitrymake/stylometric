"""Byte identity and provenance of the historical prediction oracle.

Model behavior is replayed in test_work_balanced_ablation_config and the opt-in
live replay suite; the archived internal fit trace is not a runtime contract.
"""
import hashlib
import pathlib

from stylo import jsonio

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX_DIR = ROOT / "tests" / "fixtures"
FIXTURE = FIX_DIR / "work_balanced_ablation_goldens_v1.json"
INVENTORY = FIX_DIR / "work_balanced_ablation_goldens_v1.SHA256SUMS"

FIXTURE_SHA256 = "c66e63e7e8af36b03b6aaa28f3d8cb23b71ef8e7a433ddd0c85322463b993a25"
SOURCE_COMMIT = "f1b8e165fad4a1ae7ce30c0a613ffcf0deaa7b3d"


def test_fixture_matches_pinned_full_sha():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_inventory_is_exactly_one_valid_line():
    lines = INVENTORY.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split()
    assert name == FIXTURE.name
    assert digest == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def test_provenance_contract_and_authoritative_hashes():
    d = jsonio.load_strict(FIXTURE)
    assert d["source_commit"] == SOURCE_COMMIT and d["clean_tree"] is True
    assert d["numeric_contract"] == {"kind": "quantized_exact", "round_decimals": 12} and "tolerances" not in d
    e = d["environment"]
    assert e["spacy_model"]["name"] and e["spacy_model"]["version"] and e["threadpool"]
    assert e["pythonhashseed"] and e["omp_num_threads"]
    # recorded hashes must equal the LIVE authoritative bytes
    assert e["config_sha256"] == hashlib.sha256((ROOT / "configs" / "default.yaml").read_bytes()).hexdigest()
    assert e["lock_sha256"] == hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest()
    capture_tool = ROOT / "scripts" / "artifacts" / "capture_work_balanced_ablation_goldens.py"
    assert e["capture_tool_sha256"] == hashlib.sha256(capture_tool.read_bytes()).hexdigest()
