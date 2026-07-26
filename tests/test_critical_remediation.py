from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stylo.benchmarks import ManifestValidationError, ScoringFormatError
from stylo.benchmarks.scoring import load_truth
from stylo.benchmarks.validator import validate_manifest
from stylo.cases import framework as case_framework
from stylo.domain.corpus_identity import (
    ContentIsolationError,
    assert_cross_work_content_isolation,
    find_cross_work_content_overlaps,
)
from stylo.pipeline import clean, predict
from stylo.pipeline.bundle import BundleError, publish_bundle


class _Cfg:
    def __init__(self, values):
        self.values = dict(values)

    def get_path(self, path, default=None):
        return self.values.get(path, default)


def test_aud001_exact_and_short_in_collection_content_are_rejected():
    exact = "один и тот же зарегистрированный фрагмент " * 20
    with pytest.raises(ContentIsolationError, match="exact_cross_work_chunk"):
        assert_cross_work_content_isolation(
            [exact, exact], ["author/story", "author/collection"]
        )

    short = " ".join(f"слово{i}" for i in range(100))
    collection = "вступление редактора перед текстом " + short + " конец сборника"
    overlaps = find_cross_work_content_overlaps(
        [short, collection], ["author/story", "author/collection"]
    )
    assert any(
        overlap.kind == "word5_asymmetric_containment"
        and overlap.containment == 1.0
        for overlap in overlaps
    )


def test_aud001_known_frozen_turgenev_chunk_is_rejected():
    root = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "audit_corpus"
        / "15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029"
        / "frags"
        / "turgenev"
    )
    paths = [
        root / "хорь_и_калиныч" / "хорь_и_калиныч_00002.txt",
        root / "записки_охотника" / "записки_охотника_00084.txt",
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("ignored frozen audit corpus is not present in this checkout")
    texts = [path.read_text(encoding="utf-8").strip() for path in paths]
    with pytest.raises(ContentIsolationError, match="exact_cross_work_chunk"):
        assert_cross_work_content_isolation(
            texts,
            ["turgenev/хорь_и_калиныч", "turgenev/записки_охотника"],
        )


def _bundle_meta():
    return {
        "training_weighting": "chunk_weighted_legacy",
        "dataset_contract": "legacy_recursive",
        "rows_digest": "1" * 64,
        "chunker_config_hash": "2" * 64,
        "code_tree_sha256": "3" * 64,
        "config_id": "4" * 64,
        "git_commit": "test",
        "git_dirty": False,
    }


def test_aud002_substitution_is_rejected_before_joblib(monkeypatch, tmp_path):
    data = tmp_path / "data"
    root = data / "deployment" / "chunk_weighted_legacy"
    published = publish_bundle(
        root,
        {
            "model.pkl": lambda path: path.write_bytes(b"model"),
            "delta.pkl": lambda path: path.write_bytes(b"delta"),
            "authors.json": lambda path: path.write_text(
                '["author_a","author_b"]', encoding="utf-8"
            ),
        },
        _bundle_meta(),
    )
    calls = []
    monkeypatch.setattr(predict.joblib, "load", lambda value: calls.append(value))
    from stylo.pipeline import bundle

    original = bundle._read_regular_nofollow

    def substituted(path):
        payload = original(path)
        return b"substituted" if path.name == "model.pkl" else payload

    monkeypatch.setattr(bundle, "_read_regular_nofollow", substituted)
    cfg = _Cfg(
        {
            "evaluation.training_weighting": "chunk_weighted_legacy",
            "paths.data": str(data),
            "paths.docs": str(tmp_path / "docs"),
        }
    )
    with pytest.raises(BundleError, match="changed during verified read"):
        predict.run(cfg, expected_bundle_token=published["bundle_token"])
    assert calls == []


def test_aud002_release_has_no_loose_executable_deserializers():
    root = Path(__file__).resolve().parents[1]
    executable_loads: list[tuple[str, str]] = []
    object_numpy_loads: list[tuple[str, str]] = []
    for source_root in (root / "src", root / "scripts"):
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(root).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(
                    node.func, ast.Attribute
                ):
                    continue
                owner = node.func.value
                if (
                    isinstance(owner, ast.Name)
                    and owner.id in {"joblib", "pickle"}
                    and node.func.attr in {"load", "loads"}
                ):
                    executable_loads.append((relative, ast.unparse(node.args[0])))
                if (
                    isinstance(owner, ast.Name)
                    and owner.id in {"np", "numpy"}
                    and node.func.attr == "load"
                    and any(
                        keyword.arg == "allow_pickle"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in node.keywords
                    )
                ):
                    object_numpy_loads.append((relative, ast.unparse(node)))

    assert object_numpy_loads == []
    assert executable_loads == [
        ("src/stylo/pipeline/predict.py", "io.BytesIO(payloads['model.pkl'])"),
        ("src/stylo/pipeline/predict.py", "io.BytesIO(payloads['delta.pkl'])"),
    ]


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/clustering.py",
        "scripts/umap_vis.py",
        "scripts/validate_books.py",
        "scripts/statistic/anomaly_stats.py",
        "scripts/statistic/consistency.py",
    ],
)
def test_aud002_retired_loose_artifact_scripts_exit_without_writing(
    relative, tmp_path
):
    root = Path(__file__).resolve().parents[1]
    before = set(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, str(root / relative)],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "retired" in (result.stdout + result.stderr)
    assert set(tmp_path.iterdir()) == before


def _clean_cfg(tmp_path):
    return _Cfg(
        {
            "paths.input_raw": str(tmp_path / "raw"),
            "paths.input_clean": str(tmp_path / "clean"),
            "language.spacy_model": "unused",
            "language.spacy_fallback": None,
            "evaluation.n_jobs": 1,
        }
    )


def test_aud003_clean_snapshot_removes_stale_and_preserves_old_on_failure(
    monkeypatch, tmp_path
):
    raw = tmp_path / "raw" / "author"
    raw.mkdir(parents=True)
    (raw / "one.txt").write_text("Первый исходный текст", encoding="utf-8")
    (raw / "two.txt").write_text("Второй исходный текст", encoding="utf-8")
    monkeypatch.setattr(clean, "normalize", lambda text, _model, _fallback: text.lower())
    cfg = _clean_cfg(tmp_path)

    clean.run(cfg)
    current = tmp_path / "clean"
    assert (current / "author" / "one.txt").is_file()
    (raw / "one.txt").unlink()
    clean.run(cfg)
    assert not (current / "author" / "one.txt").exists()
    before = (current / "author" / "two.txt").read_bytes()

    monkeypatch.setattr(
        clean,
        "normalize",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected normalizer failure")),
    )
    with pytest.raises(RuntimeError, match="injected normalizer failure"):
        clean.run(cfg)
    assert (current / "author" / "two.txt").read_bytes() == before


def test_aud003_invalid_utf8_never_replaces_current_snapshot(monkeypatch, tmp_path):
    raw = tmp_path / "raw" / "author"
    raw.mkdir(parents=True)
    source = raw / "one.txt"
    source.write_text("исходный текст", encoding="utf-8")
    monkeypatch.setattr(clean, "normalize", lambda text, *_args: text)
    cfg = _clean_cfg(tmp_path)
    clean.run(cfg)
    before = (tmp_path / "clean" / "author" / "one.txt").read_bytes()
    source.write_bytes(b"\xff\xfe")
    with pytest.raises(RuntimeError, match="valid UTF-8"):
        clean.run(cfg)
    assert (tmp_path / "clean" / "author" / "one.txt").read_bytes() == before


def test_aud003_nested_or_unexpected_raw_payload_fails_closed(
    monkeypatch, tmp_path
):
    raw = tmp_path / "raw"
    for author in ("beta", "gamma"):
        path = raw / author / "book.txt"
        path.parent.mkdir(parents=True)
        path.write_text(f"{author} text", encoding="utf-8")
    hidden = raw / "alpha" / "nested" / "hidden.txt"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("must not be silently omitted", encoding="utf-8")
    monkeypatch.setattr(clean, "normalize", lambda text, *_args: text)
    with pytest.raises(RuntimeError, match="nested raw corpus"):
        clean.run(_clean_cfg(tmp_path))
    assert not (tmp_path / "clean").exists()

    hidden.unlink()
    hidden.parent.rmdir()
    (raw / "alpha" / "notes.md").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected raw corpus payload"):
        clean.run(_clean_cfg(tmp_path))
    assert not (tmp_path / "clean").exists()


def test_aud003_partial_clean_cannot_mix_preprocessing_generations(
    monkeypatch, tmp_path
):
    raw = tmp_path / "raw"
    for author in ("alpha", "beta"):
        path = raw / author / "book.txt"
        path.parent.mkdir(parents=True)
        path.write_text(author, encoding="utf-8")
    cfg = _clean_cfg(tmp_path)
    monkeypatch.setattr(clean, "normalize", lambda text, *_args: f"OLD:{text}")
    clean.run(cfg)
    before = {
        path.relative_to(tmp_path / "clean").as_posix(): path.read_bytes()
        for path in (tmp_path / "clean").rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(clean, "normalize", lambda text, *_args: f"NEW:{text}")
    with pytest.raises(ValueError, match="partial clean is disabled"):
        clean.run(cfg, only=["alpha"])
    after = {
        path.relative_to(tmp_path / "clean").as_posix(): path.read_bytes()
        for path in (tmp_path / "clean").rglob("*")
        if path.is_file()
    }
    assert after == before


def test_aud004_different_path_target_copy_fails_and_manifest_binds_bytes(tmp_path):
    candidate_a = tmp_path / "a"
    candidate_b = tmp_path / "b"
    candidate_a.mkdir()
    candidate_b.mkdir()
    copied = ("общий целевой текст с устойчивыми словами " * 20).strip()
    for index in range(2):
        (candidate_a / f"a{index}.txt").write_text(
            copied if index == 0 else ("другой текст автора а " * 20),
            encoding="utf-8",
        )
        (candidate_b / f"b{index}.txt").write_text(
            ("совсем иной текст автора б " * 20) + str(index),
            encoding="utf-8",
        )
    target = tmp_path / "target-copy.txt"
    target.write_text(copied, encoding="utf-8")
    spec = case_framework.CaseSpec(
        case_id="copy",
        title="copy",
        candidates=(
            case_framework.CorpusSource("a", candidate_a),
            case_framework.CorpusSource("b", candidate_b),
        ),
        target=target,
        feature_sets=("char3",),
        min_work_words=5,
    )
    passport = case_framework.run_case(spec).to_dict()
    assert passport["status"] == "fail"
    assert any(
        failure.startswith("target_content_exact_duplicate")
        for failure in passport["failure_modes"]
    )
    assert passport["data"]["input_manifest_sha256"]
    assert all(
        len(row["sha256"]) == 64
        for row in passport["data"]["input_manifest"]["files"]
    )


def _benchmark_document(doc_id, split, sha, *, work=None, extra=None):
    document = {
        "doc_id": doc_id,
        "source": {
            "source_id": f"source:{doc_id}",
            "provenance": "archive",
            "revision": "v1",
            "sha256": sha,
        },
        "split": split,
        "task_types": ["spoof"],
        "spans": [],
    }
    if work is not None:
        document["work"] = work
    document.update(extra or {})
    return document


def _benchmark_manifest(documents):
    return {
        "schema_version": "1.0",
        "dataset": {
            "name": "isolation",
            "version": "1",
            "license": "CC0",
            "offset_unit": "token",
            "tokenizer": "stylo_unicode_word_punct_v1",
        },
        "task_types": ["spoof"],
        "documents": documents,
    }


def test_aud005_train_blind_content_work_and_identity_metadata_are_rejected():
    train = _benchmark_document(
        "doc_0000000000000001", "train", "a" * 64, work="same-work"
    )
    blind = _benchmark_document(
        "doc_0000000000000002",
        "blind",
        "a" * 64,
        work="same-work",
        extra={"edition": "leaked-edition"},
    )
    with pytest.raises(ManifestValidationError) as caught:
        validate_manifest(_benchmark_manifest([train, blind]))
    message = str(caught.value)
    assert "exact source bytes cross isolated split roles" in message
    assert "work identity crosses" in message
    assert "identity-bearing metadata" in message


def test_aud054_truth_commitment_is_checked_before_json_parse(tmp_path):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text("{malformed", encoding="utf-8")
    committed = hashlib.sha256(b"different committed bytes").hexdigest()
    with pytest.raises(ScoringFormatError, match="escrow commitment"):
        load_truth(truth_path, expected_sha256=committed)

    valid = {
        "schema_version": "1.0",
        "dataset_name": "dataset",
        "dataset_version": "1",
        "manifest_sha256": "a" * 64,
        "records": [],
    }
    truth_path.write_text(json.dumps(valid), encoding="utf-8")
    digest = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    loaded = load_truth(truth_path, expected_sha256=digest)
    assert loaded.truth_sha256 == digest
    truth_path.write_text(json.dumps({**valid, "records": [{}]}), encoding="utf-8")
    with pytest.raises(ScoringFormatError, match="escrow commitment"):
        load_truth(truth_path, expected_sha256=digest)
