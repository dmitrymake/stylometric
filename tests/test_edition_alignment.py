import pytest

from stylo.benchmarks import (
    align_editions,
    extract_block_texts,
    extract_multi_block_texts,
    intersect_reference_alignments,
    normalise_historical_word,
)


def test_historical_normalisation_is_conservative():
    assert normalise_historical_word("міръ") == "мир"
    assert normalise_historical_word("Ѳёдоръ") == "федор"
    assert normalise_historical_word("объём") == "объем"  # internal hard sign survives


def test_alignment_ignores_front_matter_and_retains_original_surfaces():
    common = [f"слово{i}" for i in range(120)]
    common[50] = "слове"
    old = "Предисловіе издателя. " + " ".join(common).replace("слове", "словѣъ")
    # The final hard sign normalises away, while punctuation and front matter
    # remain distinct in the extracted original surfaces.
    modern = "Справка. Комментарий. " + " ".join(common)

    report = align_editions(old, modern, min_block_words=50, autojunk=False)
    pairs = extract_block_texts(old, modern, report.blocks)

    assert report.matched_words == 120
    assert report.symmetric_coverage > 0.98
    assert len(report.blocks) == 1
    assert "словѣъ" in pairs[0][0]
    assert "словѣъ" not in pairs[0][1]
    assert "Предислов" not in pairs[0][0]
    assert "Справка" not in pairs[0][1]


def test_alignment_does_not_count_inserted_commentary_as_shared_content():
    left = " ".join([f"начало{i}" for i in range(70)] + [f"конец{i}" for i in range(70)])
    right = " ".join(
        [f"начало{i}" for i in range(70)]
        + ["редакторский", "комментарий"] * 30
        + [f"конец{i}" for i in range(70)]
    )
    report = align_editions(left, right, min_block_words=50, autojunk=False)

    assert [block.n_words for block in report.blocks] == [70, 70]
    assert report.matched_words == 140
    assert report.coverage_a == 1.0
    assert report.coverage_b == pytest.approx(0.7)


def test_multi_alignment_keeps_only_content_shared_by_all_versions():
    common_left = [f"лево{i}" for i in range(60)]
    common_right = [f"право{i}" for i in range(70)]
    reference = " ".join(common_left + common_right)
    version_b = "вступление " + " ".join(common_left + ["вставка"] * 20 + common_right)
    version_c = " ".join(common_left + common_right) + " приложение"

    reports = [
        align_editions(reference, version_b, min_block_words=40, autojunk=False),
        align_editions(reference, version_c, min_block_words=40, autojunk=False),
    ]
    blocks = intersect_reference_alignments(reports, min_block_words=40)
    extracted = extract_multi_block_texts([reference, version_b, version_c], blocks)

    assert [block.n_words for block in blocks] == [60, 70]
    assert len(extracted) == 2
    for row, block in zip(extracted, blocks):
        assert len(row) == 3
        assert all(len(text.split()) == block.n_words for text in row)
