import hashlib
import json

import pandas as pd

from stylo.cli import main


def _write_benchmark(tmp_path):
    text = "Альфа, бета!"
    text_path = tmp_path / "texts" / "doc.txt"
    text_path.parent.mkdir()
    text_path.write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "dataset": {
            "name": "cli-synthetic",
            "version": "0.1.0",
            "license": "CC0-1.0",
            "offset_unit": "token",
            "tokenizer": "stylo_unicode_word_punct_v1",
        },
        "task_types": ["mixed_authorship"],
        "documents": [
            {
                "doc_id": "doc_0000000000000001",
                "source": {
                    "source_id": "synthetic",
                    "provenance": "generated:test",
                    "revision": "v1",
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                },
                "split": "train",
                "task_types": ["mixed_authorship"],
                "text_path": "texts/doc.txt",
                "spans": [
                    {
                        "start": 0,
                        "end": 4,
                        "label": "a",
                        "ground_truth_known": True,
                        "evidence": "synthetic:test",
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_benchmark_schema_and_artifact_validation_cli(tmp_path, capsys):
    assert main(["benchmark", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"].startswith("SPOOF-RU")

    manifest = _write_benchmark(tmp_path)
    assert main(["benchmark", "validate", str(manifest), "--root", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["artifacts"]["documents"][0]["n_tokens"] == 4


def test_invariance_cli_scores_precomputed_predictions(tmp_path, capsys):
    rows = []
    for author in ["a", "b"]:
        for source, edition in [("s1", "e1"), ("s2", "e2")]:
            rows.append(
                {
                    "true_label": author,
                    "pred_label": author,
                    "author": author,
                    "work": f"{author}_{source}",
                    "source": source,
                    "edition": edition,
                }
            )
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    assert main(["invariance", str(path), "--bootstrap-iters", "10"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["accuracy"]["point"] == 1.0
    assert report["factors"]["source"]["unconfounded_split_coverage"] == 1.0
