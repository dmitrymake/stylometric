from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from stylo.eval import certificates


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research" / "evidence" / "withdrawn_certificates_v1"


@pytest.mark.parametrize(
    "operation",
    [
        lambda: certificates.floor_curve(0.01, 100),
        lambda: certificates.separability_horizon(0.01),
        lambda: certificates.certify_pair(0, 1, [], {}, {}, {}, {}),
        lambda: certificates.certify_all_pairs([]),
    ],
)
def test_invalid_certificate_entrypoints_are_hard_disabled(operation):
    with pytest.raises(
        certificates.WithdrawnCertificateError,
        match=certificates.WITHDRAWN_INVALID_UNIT,
    ):
        operation()


def test_only_descriptive_divergence_remains_available():
    same = np.array([0.2, 0.3, 0.5])
    disjoint_a = np.array([1.0, 0.0])
    disjoint_b = np.array([0.0, 1.0])
    assert certificates.hellinger2(same, same) == pytest.approx(0.0)
    assert certificates.hellinger2(disjoint_a, disjoint_b) == pytest.approx(1.0)


def test_exact_historical_bytes_and_counterexample_record_are_preserved():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == certificates.WITHDRAWN_INVALID_UNIT
    for artifact in manifest["artifacts"]:
        preserved = ROOT / artifact["preserved_path"]
        assert preserved.is_file()
        assert hashlib.sha256(preserved.read_bytes()).hexdigest() == artifact["sha256"]

    historical_source = (
        EVIDENCE / "certificates_historical.py.txt"
    ).read_text(encoding="utf-8")
    falsification = (
        EVIDENCE / "breakthrough_leads_historical.md"
    ).read_text(encoding="utf-8")
    historical_output = json.loads(
        (EVIDENCE / "certificates_historical_output.json").read_text(encoding="utf-8")
    )
    assert "CERTIFY_INDISTINGUISHABLE" in historical_source
    assert "546/903" in falsification
    assert historical_output["n_pairs"] == 903
