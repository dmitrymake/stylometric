from __future__ import annotations

import pytest

from stylo.eval import heterogeneity as heterogeneity


def test_heterogeneity_requires_one_shared_outsider_basis():
    with pytest.raises(ValueError, match="shared outsider-fitted"):
        heterogeneity.heterogeneity_score(["a", "b", "c"], cfg=object())
    with pytest.raises(ValueError, match="independent reference_texts"):
        heterogeneity.compare_to_controls(
            ["target"],
            "target",
            {"control": ["control"]},
            cfg=object(),
        )


def test_control_comparison_is_descriptive_and_reuses_one_basis(monkeypatch):
    shared_basis = object()
    seen_bases = []

    monkeypatch.setattr(
        heterogeneity,
        "fit_style_basis",
        lambda reference_texts, cfg: shared_basis,
    )

    def fake_score(texts, cfg, k=2, basis=None):
        seen_bases.append(basis)
        score = 0.6 if texts == ["target"] else 0.2
        return {
            "silhouette_k2": score,
            "cluster_sizes": [1, 1],
            "labels": [0, 1],
        }

    monkeypatch.setattr(heterogeneity, "heterogeneity_score", fake_score)
    result = heterogeneity.compare_to_controls(
        ["target"],
        "target",
        {"c1": ["control-1"], "c2": ["control-2"]},
        cfg=object(),
        reference_texts=["independent-1", "independent-2"],
    )

    assert seen_bases == [shared_basis, shared_basis, shared_basis]
    assert result["claim_status"] == "descriptive_only_no_significance_or_authorship_verdict"
    assert result["inference"] is None
    assert "verdict" not in result
    assert "z_vs_controls" not in result
