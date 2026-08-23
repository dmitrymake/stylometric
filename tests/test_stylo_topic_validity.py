from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from stylo.config import ConfigNode, load_config
from stylo.domain.work_weighting import work_sample_weights
from stylo.eval.dispatch import fit_estimator
from stylo.eval.lobo import make_factory_for_ablation
from stylo.eval.paired_audit.applicability_v3_2 import resolve_cell_v3_2
from stylo.features.function_words import FunctionWordBlock
from stylo.jsonio import canonical_hash
from stylo.vectorizer import StyloVectorizer


CELLS = ("A0", "A1", "A2", "A3", "A4")
SCAFFOLD = "И в, на — но; потому что этот тот был очень! Затем снова? "


def _text(noun: str, strength: int) -> str:
    return (SCAFFOLD + ((noun + ", ") * strength) + ((noun + "! ") * strength) + SCAFFOLD) * 3


def _panel(noun_pair=("космос", "компас"), strength=8):
    texts, labels, groups = [], [], []
    for label, noun in enumerate(noun_pair):
        for work in range(2):
            for _chunk in range(2):
                texts.append(_text(noun, strength))
                labels.append(label)
                groups.append(f"a{label}/w{work}")
    clean = [_text(noun, strength) for noun in noun_pair]
    return texts, np.asarray(labels), np.asarray(groups, dtype=object), clean, clean[::-1]


def _fit_block(mode: str, texts, labels, groups):
    block = FunctionWordBlock(mode=mode, mfw_count=300, lang="ru")
    block.fit(texts, None, groups=groups)
    matrix = block.transform(texts, None)
    scaler = MaxAbsScaler().fit(matrix)
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, class_weight=None, random_state=42
    )
    classifier.fit(
        scaler.transform(matrix), labels,
        sample_weight=work_sample_weights(labels, groups),
    )
    return block, scaler, classifier


def _block_predictions(fitted, texts):
    block, scaler, classifier = fitted
    return classifier.predict(scaler.transform(block.transform(texts, None)))


def test_mfw_content_token_causal_oracle_and_fixed_list_control():
    for noun_pair in (("космос", "компас"), ("ландыш", "баркас")):
        for strength in (4, 12):
            texts, labels, groups, clean, swapped = _panel(noun_pair, strength)
            mfw = _fit_block("mfw", texts, labels, groups)
            strict = _fit_block("fixed_list", texts, labels, groups)

            mfw_names = {name.removeprefix("fw::") for name in mfw[0].feature_names()}
            strict_names = {name.removeprefix("fw::") for name in strict[0].feature_names()}
            assert set(noun_pair) <= mfw_names
            assert set(noun_pair).isdisjoint(strict_names)

            current_clean = _block_predictions(mfw, clean)
            current_swapped = _block_predictions(mfw, swapped)
            strict_clean = _block_predictions(strict, clean)
            strict_swapped = _block_predictions(strict, swapped)
            assert current_clean.tolist() == [0, 1]
            assert current_swapped.tolist() == [1, 0]
            assert np.array_equal(strict_clean, strict_swapped)
            assert (strict[0].transform(clean, None) != strict[0].transform(swapped, None)).nnz == 0

            # The fixed-list block is live: changing genuine function words remains observable.
            function_word_probe = strict[0].transform(["и в на", "но в на"], None)
            assert (function_word_probe[0] != function_word_probe[1]).nnz > 0


def _test_config(tmp_path) -> ConfigNode:
    raw = load_config().to_dict()
    raw["language"]["parse_n_process"] = 1
    raw["paths"]["data"] = str(tmp_path / "data")
    raw["paths"]["doc_cache"] = str(tmp_path / "doc_cache")
    return ConfigNode(raw)


def _topic_strict_estimator(cfg, cell):
    row = resolve_cell_v3_2("stylo", cell)
    estimator = make_factory_for_ablation("stylo", cfg, ablation=row.ablation)()
    relative_fw = {"A2": False, "A3": True}.get(cell)
    estimator.set_params(vectorizer=StyloVectorizer.from_config(
        cfg, topic_strict=True, relative_fw=relative_fw
    ))
    return estimator


def _run_all_cells(cfg):
    texts, labels, groups, clean, swapped = _panel()
    summary = {}
    for cell in CELLS:
        row = resolve_cell_v3_2("stylo", cell)
        current = make_factory_for_ablation("stylo", cfg, ablation=row.ablation)()
        strict = _topic_strict_estimator(cfg, cell)
        assert type(current) is type(strict)
        for step in ("scaler", "classifier"):
            assert type(current.named_steps[step]) is type(strict.named_steps[step])
            assert current.named_steps[step].get_params() == strict.named_steps[step].get_params()

        fit_estimator(current, texts, labels, groups)
        fit_estimator(strict, texts, labels, groups)
        assert np.array_equal(current.classes_, strict.classes_)
        current_clean = current.predict(clean)
        current_swapped = current.predict(swapped)
        strict_clean = strict.predict(clean)
        strict_swapped = strict.predict(swapped)
        strict_vectorizer = strict.named_steps["vectorizer"]
        current_vectorizer = current.named_steps["vectorizer"]

        current_fw = next(block for block in current_vectorizer.blocks if block.name == "function_words")
        strict_fw = next(block for block in strict_vectorizer.blocks if block.name == "function_words")
        strict_syntax = next(block for block in strict_vectorizer.blocks if block.name == "syntax")
        assert current_fw.mode == "mfw" and strict_fw.mode == "fixed_list"
        assert {"pos_ratios", "lexical_richness"}.isdisjoint(strict_syntax._active)
        assert (current_vectorizer.transform(clean) != current_vectorizer.transform(swapped)).nnz > 0
        assert (strict_vectorizer.transform(clean) != strict_vectorizer.transform(swapped)).nnz == 0
        assert current_clean.tolist() == [0, 1]
        assert current_swapped.tolist() == [1, 0]
        assert np.array_equal(strict_clean, strict_swapped)

        current_drop = float(np.mean(current_clean == [0, 1]) - np.mean(current_swapped == [0, 1]))
        strict_drop = float(np.mean(strict_clean == [0, 1]) - np.mean(strict_swapped == [0, 1]))
        summary[cell] = {
            "current_clean_accuracy": float(np.mean(current_clean == [0, 1])),
            "current_swapped_accuracy": float(np.mean(current_swapped == [0, 1])),
            "current_flip_rate": float(np.mean(current_clean != current_swapped)),
            "strict_flip_rate": float(np.mean(strict_clean != strict_swapped)),
            "topic_stress_delta": current_drop - strict_drop,
        }
    return summary


def test_all_v32_stylo_routes_expose_mfw_and_topic_strict_is_invariant(tmp_path):
    pytest.importorskip("ru_core_news_lg")
    first = _run_all_cells(_test_config(tmp_path / "first"))
    second = _run_all_cells(_test_config(tmp_path / "second"))
    assert canonical_hash(first) == canonical_hash(second)
    for row in first.values():
        assert row == {
            "current_clean_accuracy": 1.0,
            "current_swapped_accuracy": 0.0,
            "current_flip_rate": 1.0,
            "strict_flip_rate": 0.0,
            "topic_stress_delta": 1.0,
        }
