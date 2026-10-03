"""Selected-work isolation uses existing corpus coverage before expensive fit."""
from pathlib import Path

import pytest

from stylo.cli import main
from stylo.config import load_config, with_overrides
from stylo.corpus_tools import validate_corpus as validation
from stylo.pipeline import clean, split, train, predict
from stylo.report import build, evidence


def _text(prefix):
    return ", ".join(f"{prefix}{i}" for i in range(100)) + "."


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True, "0.4"])
def test_invalid_threshold_cannot_disable_isolation(tmp_path, threshold):
    with pytest.raises(ValueError, match="finite number in"):
        validation.validate(tmp_path, near_dup_threshold=threshold)


def _analyze_fixture(tmp_path, monkeypatch, *, overlap):
    root = tmp_path / "clean"
    texts = {
        "periodic/target": _text("цель"),
        "periodic/reference1": _text("эталон"),
        "periodic/reference2": "Предисловие. " + _text("эталон").replace(",", ";"),
        "lyrical/reference1": _text("парус"),
        "lyrical/reference2": _text("волна"),
        "inactive/collection": "Вступление. " + _text("цель").replace(",", ";"),
        "unknown/unselected": _text("цель"),
    }
    if overlap:
        texts["periodic/collection"] = "Вступление. " + _text("цель").replace(",", ";") + " Конец."
    for work, text in texts.items():
        path = root / (work + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    cfg = with_overrides(load_config(), {"paths.input_clean": str(root),
        "paths.docs": str(tmp_path / "results"), "deployment.candidate_authors": ["periodic", "lyrical"]})
    import stylo.cli as cli
    monkeypatch.setattr(cli, "_cfg", lambda _: cfg)
    monkeypatch.setattr(clean, "run", lambda _: None)
    captured = {}
    monkeypatch.setattr(evidence, "publish_corpus_validation", lambda _cfg, **kw: captured.update(kw))
    return cfg, captured


def test_same_author_edition_is_rejected_before_split_warm_or_fit(tmp_path, monkeypatch):
    _cfg, captured = _analyze_fixture(tmp_path, monkeypatch, overlap=True)
    monkeypatch.setattr(split, "run", lambda *_a, **_kw: pytest.fail("overlap must fail before split"))
    monkeypatch.setattr(train, "make_rep_cache", lambda *_a: pytest.fail("warm forbidden"))
    monkeypatch.setattr(train, "fit_estimator", lambda *_a: pytest.fail("fit forbidden"))
    with pytest.raises(validation.CorpusValidationError, match="target_reference_overlap"):
        main(["analyze", "--target-work", "periodic/target"])
    scope = captured["structured"]["scope"]
    assert scope["target_work_id"] == "periodic/target"
    assert scope["reference_authors"] == ["lyrical", "periodic"]
    assert "inactive/collection" not in scope["selected_work_ids"]
    assert "unknown/unselected" not in scope["selected_work_ids"]


def test_independent_target_allows_train_only_warnings_and_inactive_duplicates(tmp_path, monkeypatch):
    _cfg, captured = _analyze_fixture(tmp_path, monkeypatch, overlap=False)
    stages = []
    monkeypatch.setattr(split, "run", lambda *_a, **_kw: stages.append("split"))
    monkeypatch.setattr(train, "run", lambda *_a, **_kw: stages.append("fit") or {"bundle_token": "f" * 32})
    monkeypatch.setattr(predict, "run", lambda *_a, **_kw: stages.append("predict"))
    monkeypatch.setattr(build, "run", lambda *_a, **_kw: stages.append("report"))
    assert main(["analyze", "--target-work", "periodic/target"]) == 0
    assert stages == ["split", "fit", "predict", "report"]
    findings = captured["structured"]["findings"]
    assert any(row["severity"] == "warn" and row["code"] == "near_dup" for row in findings)
    assert not any(row["severity"] == "error" for row in findings)


def test_target_book_count_is_not_treated_as_reference_count(tmp_path, monkeypatch):
    cfg, captured = _analyze_fixture(tmp_path, monkeypatch, overlap=False)
    (Path(cfg.paths.input_clean) / "unknown/unselected.txt").write_text(_text("новаяцель"), encoding="utf-8")
    report = validation.run(cfg, target_work_id="unknown/unselected", reference_authors=("periodic", "lyrical"))
    assert not any(row.code == "few_books" and row.message.startswith("unknown:") for row in report.findings)
    assert captured["structured"]["scope"]["kind"] == "target_and_training_references"


def test_normalized_short_content_is_still_detected():
    assert validation.text_overlap_coverage("Раз два!", "РАЗ, ДВА.") == 1.0
    assert validation.text_overlap_coverage("Раз два!", "Совсем другие.") == 0.0
