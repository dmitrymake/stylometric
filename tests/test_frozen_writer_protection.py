"""Active result writers preserve frozen inputs and publish separate candidates."""
import json
import pathlib
import runpy

import pandas as pd
import pytest

from stylo.config import load_config, with_overrides
from stylo.jsonio import load_strict

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN_VALIDATION = {"validation.json", "validation_pd.json"}
FROZEN = {"final_comparison.csv", "final_comparison.txt", "ruaa_bench_v1.json",
          "ruaa_bench_leaderboard.md", "stylo_lobo_authorci.json", *FROZEN_VALIDATION}


def _sentinels(docs):
    docs.mkdir()
    expected = {name: ("historical fixture: " + name).encode() for name in FROZEN}
    for name, value in expected.items():
        (docs / name).write_bytes(value)
    return expected


def test_final_cli_preserves_frozen_inputs(tmp_path, monkeypatch):
    import yaml
    from stylo.cli import main
    from stylo import dataset
    from stylo.eval import final
    docs = tmp_path / "docs"
    prior = _sentinels(docs)
    cfg = with_overrides(load_config(), {"paths.docs": str(docs), "paths.data": str(tmp_path / "data")})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()))
    table = pd.DataFrame([dict(model="stylo", accuracy=0.5, acc_ci="[0,1]", macro_f1=0.5,
                              top2=1.0, vs_stylo_mcnemar_p=1.0, vs_stylo_dacc=0.0, ece=None)])
    books = pd.DataFrame([dict(test_author="alpha", test_book="one", correct=True, rank=1,
                              top_candidates=[("alpha", 0.8), ("beta", 0.2)])])
    result = {"table": table, "results": {"stylo": {"df": books, "summary": {"per_author_recall": {"alpha": 1.0}}}}}
    # Only scientific computation is supplied; CLI routing, formatting and writes are real.
    monkeypatch.setattr(dataset, "resolve_dataset", lambda *a, **kw: object())
    monkeypatch.setattr(final, "run_final", lambda *a, **kw: result)
    assert main(["evaluate", "--config", str(config_path)]) == 0
    assert {name: (docs / name).read_bytes() for name in prior} == prior
    generated = pd.read_csv(docs / "exploratory/legacy_recompute/final_comparison.csv")
    assert generated["model"].tolist() == ["stylo"]
    assert generated["accuracy"].tolist() == [0.5]


@pytest.mark.parametrize("pd_only,fast", [(False, False), (False, True), (True, False), (True, True)])
def test_benchmark_writer_preserves_historical_inputs(tmp_path, pd_only, fast):
    runner = runpy.run_path(str(ROOT / "scripts/run_benchmark.py"))
    docs = tmp_path / "docs"
    prior = _sentinels(docs)
    result = {"claim_status": "exploratory_internal", "dataset_identity": {"synthetic": True},
              "attestation": {"synthetic": True}, "macro_f1_authorclustered_CI": None,
              "macro_f1_authorclustered_interval_status": "withdrawn_invalid_resampling_labels_not_remapped"}
    outputs = runner["publish_benchmark_result"](result, docs=docs, pd_only=pd_only, fast=fast)
    provenance = load_strict(outputs["run_provenance.json"])
    candidate = load_strict(outputs[provenance["candidate_file"]])
    assert candidate == result
    assert provenance["public_headline_authorized"] is False
    assert provenance["supersedes"] is None
    assert {name: (docs / name).read_bytes() for name in prior} == prior


def test_historical_validation_inputs_match_the_p0_snapshot():
    import hashlib
    import json

    snapshot = json.loads(
        (ROOT / "docs" / "p0_baseline_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    for name in sorted(FROZEN_VALIDATION):
        relative = f"docs/{name}"
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == snapshot["artifacts"]["sha256"][relative]


def test_canonical_ruaa_runner_refuses_to_replace_frozen_results(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from stylo.eval.ci_erratum import CiErratumError
    runner = runpy.run_path(str(ROOT / "scripts/run_ruaa_baselines.py"))
    namespace = runner["main"].__globals__
    docs = tmp_path / "docs"
    prior = _sentinels(docs)
    bench = tmp_path / "bench.json"
    bench.write_text("{}")
    fake_dataset = SimpleNamespace(n_authors=2, groups=pd.Series(["a/one", "b/one"]))
    class Rows:
        n_authors = fake_dataset.n_authors
        groups = fake_dataset.groups
        def __len__(self):
            return 2
    monkeypatch.setattr("sys.argv", ["run_ruaa_baselines.py"])
    for key, value in {
        "BENCH_DOC": bench, "ROOT": tmp_path, "OUT_JSON": docs / "ruaa_bench_v1.json",
        "OUT_MD": docs / "ruaa_bench_leaderboard.md",
        "resolve_fragment_snapshot": lambda *_: SimpleNamespace(train_root=tmp_path),
        "load_dataset": lambda *_: Rows(), "bench_subset": lambda *a: (Rows(), []),
    }.items():
        monkeypatch.setitem(namespace, key, value)
    with pytest.raises(CiErratumError):
        runner["main"]()
    assert {name: (docs / name).read_bytes() for name in prior} == prior
