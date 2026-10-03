from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research" / "evidence" / "withdrawn_certificates_v1"


def test_exact_historical_bytes_and_counterexample_record_are_preserved():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "WITHDRAWN_INVALID_UNIT"
    for artifact in manifest["artifacts"]:
        if artifact["role"] == "exact_historical_falsification_record":
            # The prose review is local; its archived identity remains in the
            # historical manifest but is not a required executable input.
            assert artifact["sha256"] == (
                "60f58853c34a2dacf9f1e12d1e2bca28e5b2eff7dce923e151ddb5a9c042321f"
            )
            continue
        preserved = ROOT / artifact["preserved_path"]
        assert preserved.is_file()
        assert hashlib.sha256(preserved.read_bytes()).hexdigest() == artifact["sha256"]

    historical_source = (
        EVIDENCE / "certificates_historical.py.txt"
    ).read_text(encoding="utf-8")
    historical_output = json.loads(
        (EVIDENCE / "certificates_historical_output.json").read_text(encoding="utf-8")
    )
    assert "CERTIFY_INDISTINGUISHABLE" in historical_source
    assert historical_output["n_pairs"] == 903
    assert sum(historical_output["verdict_counts"].values()) == historical_output["n_pairs"]


def test_historical_information_bound_is_withdrawn():
    artifact = json.loads((ROOT / "docs" / "fano_frontier.json").read_text(encoding="utf-8"))
    assert artifact["artifact_status"] == "WITHDRAWN_INVALID_SCIENTIFIC_SEMANTICS"
    assert artifact["valid_lower_bound_fields"] == []
    assert "I_AF_bits" in artifact["invalid_inferential_fields"]
    assert artifact["withdrawal"]["historical_values_retained"] is True
