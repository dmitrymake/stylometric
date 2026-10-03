import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from stylo.config import load_config
from stylo.domain.work_weighting import work_sample_weights


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "publication_comparison", _ROOT / "scripts/evaluation/run_publication_comparison.py",
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)
SETTINGS = {"delta_mfw": 300, "char_ngram_range": [3, 5],
            "char_max_features": 5000, "char_min_df": 1}


def test_mass_control_has_same_solution_as_scaling_weights():
    # Compare two mathematically equivalent weighted objectives on unequal work
    # lengths; this tests the C direction against sklearn, not only arithmetic.
    X = np.asarray([[-2.], [-1.7], [-1.1], [.4], [1.2], [2.]])
    y = np.asarray([0, 0, 0, 1, 1, 1])
    groups = ["A/a", "A/a", "A/b", "B/c", "B/d", "B/d"]
    weights = work_sample_weights(y, groups)
    multiplier = len(y) / len(set(groups))
    controlled = LogisticRegression(C=multiplier, tol=1e-10).fit(X, y, sample_weight=weights)
    rescaled = LogisticRegression(C=1, tol=1e-10).fit(X, y, sample_weight=weights * multiplier)
    assert np.allclose(controlled.coef_, rescaled.coef_, atol=1e-8)
    assert np.allclose(controlled.intercept_, rescaled.intercept_, atol=1e-8)
    cfg = load_config()
    estimator = runner.make_estimator("stylo_A4_current_mass_control", cfg, 60, 4, SETTINGS)
    assert estimator.named_steps["classifier"].C == 15
    assert estimator.named_steps["classifier"].class_weight is None


def test_all_token_delta_uses_unselected_tokens_in_denominator():
    delta = runner.make_estimator("delta_300", load_config(), 4, 4, SETTINGS)
    delta.mfw_count = 1
    delta.fit(["и и слово", "и другое", "и и и", "и текст"],
              [0, 0, 1, 1], groups=["A/a", "A/b", "B/c", "B/d"])
    assert delta._grid_rel_freq(["и незнакомое незнакомое"])[0, 0] == pytest.approx(1 / 3)


@pytest.mark.parametrize("arm", ["delta_300", "cosine_delta_300"])
def test_delta_work_score_does_not_depend_on_chunk_boundaries(arm):
    delta = runner.make_estimator(arm, load_config(), 4, 4, SETTINGS)
    delta.fit(["и и слово", "и другое слово", "не не текст", "не иной текст"],
              [0, 0, 1, 1], groups=["A/a", "A/b", "B/c", "B/d"])
    whole = delta.predict_proba(["и и слово не не текст"])[0]
    unequal = delta.predict_proba(["и", "и слово не не текст"])
    assert np.allclose(unequal, whole)
    assert np.allclose(delta.predict_proba(["и и слово", "не не текст"]), whole)


def test_every_development_and_locked_fit_excludes_all_locked_works(monkeypatch):
    fit_inputs = []

    class Spy:
        def fit(self, texts, y):
            fit_inputs.append(set(texts))
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, texts):
            return np.tile([.8, .2], (len(texts), 1))

    monkeypatch.setattr(runner, "make_estimator", lambda *args: Spy())
    works = ["A/a", "A/b", "B/c", "B/d", "A/locked", "B/locked"]
    texts = np.asarray(works, dtype=object)
    y = np.asarray([0, 0, 1, 1, 0, 1])
    roles = np.asarray(["development"] * 4 + ["locked_test"] * 2)
    results, _ = runner.evaluate_panel(texts, y, np.asarray(works), roles,
                                      ["A", "B"], load_config(), SETTINGS,
                                      arms=("stylo_A0_current",))
    assert len(fit_inputs) == 5  # four LOBO fits and exactly one final training fit
    assert all(not {"A/locked", "B/locked"} & fitted for fitted in fit_inputs)
    for held, fitted in zip(works[:4], fit_inputs[:4]):
        assert held not in fitted
        assert len(fitted) == 3
    assert fit_inputs[-1] == set(works[:4])
    assert results["locked_test"]["metrics"]["stylo_A0_current"]["total"] == 2


def test_paired_summary_retains_losses_and_equal_author_estimand():
    base = [dict(work_id=w, author=a, correct=c) for w, a, c in
            [("A/a", "A", True), ("A/b", "A", False), ("B/c", "B", False)]]
    other = [dict(row, correct=correct) for row, correct in zip(base, [False, True, True])]
    result = runner.summarize({"stylo_A0_current": base, "char_tfidf_lr": other}, ["A", "B"])
    assert result["metrics"]["char_tfidf_lr"]["macro_author_recall"] == .75
    assert result["paired_against_stylo_A0_current"]["char_tfidf_lr"] == {
        "both_correct": 0, "gained": 2, "lost": 1, "both_wrong": 0,
    }
    with pytest.raises(ValueError, match="identical ordered works"):
        runner.summarize({"stylo_A0_current": base, "char_tfidf_lr": other[::-1]}, ["A", "B"])
    mass = runner.summarize({"stylo_A0_current": base, "stylo_A4_current": base,
                             "stylo_A4_current_mass_control": other}, ["A", "B"])
    assert mass["mass_control_against_A4_current"]["lost"] == 1
    assert mass["mass_control_against_A4_current"]["gained"] == 2
    strict = [dict(row, correct=correct) for row, correct in zip(base, [False, True, False])]
    topic = runner.summarize({"stylo_A0_current": base, "stylo_A4_current": base,
                             "stylo_A4_topic_strict": strict}, ["A", "B"])
    assert topic["metrics"]["stylo_A4_current"]["correct"] == topic["metrics"]["stylo_A4_topic_strict"]["correct"]
    assert topic["topic_strict_against_A4_current"] == {
        "both_correct": 0, "gained": 1, "lost": 1, "both_wrong": 1,
    }


def test_duplicate_work_bytes_are_not_independent_observations():
    rows = [dict(work_id=f"{a}/{w}", author_id=a, role=role, text_sha256="same")
            for a in ["A", "B"] for w, role in
            [("one", "development"), ("two", "development"), ("three", "locked_test")]]
    with pytest.raises(ValueError, match="duplicate input bytes"):
        runner.validate_catalog({"works": rows})


def test_length_check_reads_only_development_and_keeps_two_prefix_chunks(tmp_path, monkeypatch):
    import hashlib
    from collections import Counter

    monkeypatch.syspath_prepend(str(_ROOT / "scripts/evaluation"))
    spec = importlib.util.spec_from_file_location(
        "publication_length_check", _ROOT / "scripts/evaluation/run_publication_length_check.py",
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rows = []
    for author in ("A", "B"):
        for i in range(4):
            payload = f"synthetic work {author} {i}".encode()
            role = "development" if i < 3 else "locked_test"
            if role == "development":
                path = tmp_path / author / f"work{i}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            rows.append(dict(work_id=f"{author}/work{i}", author_id=author, role=role,
                             text_sha256=hashlib.sha256(payload).hexdigest()))
    # No locked file exists: any read of one fails. The stubs isolate loader routing
    # from NLP cost, while the literal missing files test its actual I/O boundary.
    class Sentencizer:
        def make_doc(self, text):
            return text.split()
    monkeypatch.setattr(helper.primary, "load_sentencizer", lambda lang: Sentencizer())
    monkeypatch.setattr(helper.primary, "normalize", lambda text, *args: text)
    monkeypatch.setattr(helper.primary, "sentences_for_text", lambda text, nlp: [text])
    monkeypatch.setattr(helper.primary, "make_sent_chunks", lambda *args, **kwargs: ["first", "second", "third"])
    texts, _, groups, _, lengths, _ = helper.development_prefix({"works": rows}, tmp_path, load_config())
    assert texts.tolist() == ["first", "second"] * 6
    assert set(Counter(groups).values()) == {2}
    assert set(lengths.values()) == {3}
    assert len(lengths) == 6
