import dataclasses
import itertools
import lzma
import math

import pytest

from stylo.models import compression
from stylo.models.compression import (
    ReferenceWork, compressor_spec, conditional_compression_score,
    rank_candidate_references,
)


TARGET = "Учебный пример о короткой прогулке. " * 10


def _references():
    return [
        ReferenceWork("alpha", "alpha/short", "Учебная история о море. " * 3),
        ReferenceWork("alpha", "alpha/long", "Наблюдение описывает геометрию узора. " * 80),
        ReferenceWork("beta", "beta/one", "Синтетическая запись о деревьях и ветре. " * 10),
    ]


def _rank(references, candidate_ids=("alpha", "beta")):
    return rank_candidate_references(
        TARGET, target_work_id="unknown/target", candidate_ids=candidate_ids,
        references=references,
    )


def test_conditional_score_matches_exact_utf8_concatenation_and_fixed_options():
    reference, target = "  Ёж\r\nидёт! ", "\ufeffНовый\tпример.  "
    spec = compressor_spec()
    filters = [{
        "id": lzma.FILTER_LZMA1, "dict_size": spec["dict_size"],
        "lc": spec["lc"], "lp": spec["lp"], "pb": spec["pb"],
        "mode": lzma.MODE_NORMAL, "nice_len": spec["nice_len"],
        "mf": lzma.MF_BT4, "depth": spec["depth"],
    }]
    reference_length = len(lzma.compress(reference.encode("utf-8"), format=lzma.FORMAT_RAW, filters=filters))
    joint_length = len(lzma.compress((reference + target).encode("utf-8"), format=lzma.FORMAT_RAW, filters=filters))

    score = conditional_compression_score(reference, target)

    assert score.compressed_reference_bytes == reference_length
    assert score.compressed_joint_bytes == joint_length
    assert score.conditional_bytes == joint_length - reference_length
    assert score.target_utf8_bytes == len(target.encode("utf-8"))
    assert score.reference_utf8_bytes == len(reference.encode("utf-8"))
    assert score.bytes_per_target_byte == score.conditional_bytes / score.target_utf8_bytes
    assert score.algorithm_version == spec["algorithm_version"]
    assert score.score_unit == "compressed_bytes_per_target_utf8_byte"


def test_scores_are_deterministic_finite_diagnostics():
    references = _references()
    first = _rank(references)

    assert _rank(references) == first
    for score in first:
        assert math.isfinite(score.mean_bytes_per_target_byte)
        assert score.diagnostic_only is True
        assert score.aggregation == "equal_work_mean"
        assert "probability" not in dataclasses.asdict(score)
        assert "winner" not in dataclasses.asdict(score)


def test_reference_and_panel_permutations_preserve_ranking_exactly():
    references = _references()
    expected = _rank(references)

    for permuted in itertools.permutations(references):
        assert _rank(permuted, candidate_ids=("beta", "alpha")) == expected


def test_unequal_reference_volumes_use_equal_work_mean():
    references = _references()
    scores = {result.candidate_id: result for result in _rank(references)}
    short = conditional_compression_score(references[0].text, TARGET)
    long = conditional_compression_score(references[1].text, TARGET)

    assert short.reference_utf8_bytes < long.reference_utf8_bytes
    assert short.bytes_per_target_byte != long.bytes_per_target_byte
    equal_work = (short.bytes_per_target_byte + long.bytes_per_target_byte) / 2
    volume_weighted = (
        short.bytes_per_target_byte * short.reference_utf8_bytes
        + long.bytes_per_target_byte * long.reference_utf8_bytes
    ) / (short.reference_utf8_bytes + long.reference_utf8_bytes)
    assert scores["alpha"].n_reference_works == 2
    assert scores["alpha"].mean_bytes_per_target_byte == pytest.approx(equal_work)
    assert scores["alpha"].mean_bytes_per_target_byte != pytest.approx(volume_weighted)


@pytest.mark.parametrize("change, error", [
    (lambda refs: refs + [ReferenceWork("beta", "unknown/target", "Иной эталон")], "target_work_id"),
    (lambda refs: refs + [ReferenceWork("beta", "alpha/short", "Иной эталон")], "work_id collision"),
    (lambda refs: refs + [ReferenceWork("alpha", "alpha/short", "Иной эталон")], "work_id collision"),
    (lambda refs: refs + [ReferenceWork("gamma", "gamma/work", "Иной эталон")], "outside the explicit panel"),
    (lambda refs: refs + [ReferenceWork("beta", "beta/copy", TARGET)], "exactly matches the target"),
    (lambda refs: refs + [ReferenceWork("beta", "beta/copy", refs[0].text)], "duplicate reference content"),
    (lambda refs: refs[:2], "every candidate requires"),
    (lambda refs: refs + [ReferenceWork("beta", "beta/blank", "  \n")], "non-whitespace"),
])
def test_metadata_and_exact_content_gates_run_before_compression(monkeypatch, change, error):
    def forbidden_compression(data):
        raise AssertionError("invalid panel must fail before compression")

    monkeypatch.setattr(compression, "_compressed_length", forbidden_compression)

    with pytest.raises(ValueError, match=error):
        _rank(change(_references()))


@pytest.mark.parametrize("text", ["", " \n\t"])
@pytest.mark.parametrize("side", ["reference", "target"])
def test_pair_rejects_empty_or_whitespace_only_inputs(text, side):
    with pytest.raises(ValueError, match="non-whitespace"):
        conditional_compression_score(text if side == "reference" else TARGET,
                                      text if side == "target" else TARGET)


@pytest.mark.parametrize("text", [None, b"bytes", 7])
def test_pair_requires_exact_text_strings(text):
    with pytest.raises(TypeError, match="exact str"):
        conditional_compression_score(text, TARGET)


def test_invalid_unicode_is_not_silently_replaced():
    with pytest.raises(UnicodeEncodeError):
        conditional_compression_score("invalid \ud800", TARGET)


@pytest.mark.parametrize("panel", [(), ("alpha", "alpha"), ("alpha", " ")])
def test_panel_requires_unique_nonempty_candidate_ids(panel):
    with pytest.raises(ValueError):
        _rank(_references(), candidate_ids=panel)


def test_negative_conditional_length_is_not_clipped_or_mislabeled(monkeypatch):
    lengths = iter([30, 28])
    monkeypatch.setattr(compression, "_compressed_length", lambda data: next(lengths))

    score = conditional_compression_score("reference", "abc")

    assert score.conditional_bytes == -2
    assert score.bytes_per_target_byte == -2 / 3


def test_ties_have_deterministic_candidate_id_order(monkeypatch):
    monkeypatch.setattr(compression, "_compressed_length", lambda data: len(data))

    scores = _rank(_references(), candidate_ids=("beta", "alpha"))

    assert [score.candidate_id for score in scores] == ["alpha", "beta"]


def test_reference_repr_hides_text_and_options_are_detached():
    reference = ReferenceWork("alpha", "alpha/work", "Синтетический скрытый текст")
    spec = compressor_spec()
    spec["dict_size"] = 1

    assert reference.text not in repr(reference)
    assert compressor_spec()["dict_size"] == 8 * 1024 * 1024
