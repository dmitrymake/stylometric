import copy

import pytest

from stylo.corpus_tools.text_quality_vnext import (
    CorpusTextAuditReport,
    CorpusTextQualityError,
    audit_corpus_texts,
    require_text_quality,
)
from stylo.jsonio import canonical_hash


def _text(prefix: str, *, count: int = 260) -> bytes:
    words = " ".join(f"{prefix}{index}" for index in range(count))
    return f"{prefix} title\n\n{words}\n{prefix} end\n".encode()


def test_clean_exact_inventory_passes_deterministically():
    payloads = {
        "alpha/one": _text("alpha"),
        "beta/two": _text("beta"),
    }

    first = audit_corpus_texts(
        payloads,
        expected_work_ids=("alpha/one", "beta/two"),
    )
    second = audit_corpus_texts(
        dict(reversed(tuple(payloads.items()))),
        expected_work_ids=("alpha/one", "beta/two"),
    )

    assert first.status == "passed"
    assert first.to_dict() == second.to_dict()
    assert first.payload["work_count"] == 2
    require_text_quality(first)


@pytest.mark.parametrize(
    ("body", "kind"),
    [
        ("{{шаблон}}", "wiki_template"),
        ("[[Категория:Романы]]", "wiki_link"),
        ("https://example.test/source", "web_url"),
        ("'''ГЛАВА'''", "wiki_bold_italic"),
        (
            "перенаправление Война и мир (Толстой)/Том 1",
            "rendered_redirect_notice",
        ),
        ("--------------------------------", "source_separator"),
    ],
)
def test_transport_residue_blocks_with_exact_line_evidence(body, kind):
    payload = _text("alpha").decode().replace("alpha end", body).encode()

    report = audit_corpus_texts(
        {"alpha/one": payload},
        expected_work_ids=("alpha/one",),
    )

    assert report.status == "blocked"
    finding = report.payload["works"][0]["transport_residue_findings"][0]
    assert finding["kind"] == kind
    assert type(finding["line_number"]) is int
    assert len(finding["line_sha256"]) == 64
    with pytest.raises(CorpusTextQualityError, match="transport_residue"):
        require_text_quality(report)


@pytest.mark.parametrize(
    "tail",
    [
        "Тираж 300 000 экз.",
        "Книжная фабрика № 5",
        "ISBN 978-0-00-000000-0",
        "См. также",
    ],
)
def test_high_confidence_tail_apparatus_blocks(tail):
    payload = _text("alpha").decode().replace("alpha end", tail).encode()

    report = audit_corpus_texts(
        {"alpha/one": payload},
        expected_work_ids=("alpha/one",),
    )

    assert report.status == "blocked"
    assert report.payload["works"][0]["tail_apparatus_findings"]


def test_short_stub_and_noncanonical_bytes_fail_closed():
    report = audit_corpus_texts(
        {"alpha/one": b"title\n\nshort prose\n"},
        expected_work_ids=("alpha/one",),
    )
    assert report.status == "blocked"
    assert report.payload["blocking_findings"][0]["kind"] == (
        "work_below_minimum_words"
    )

    with pytest.raises(CorpusTextQualityError, match="exactly one final LF"):
        audit_corpus_texts(
            {"alpha/one": _text("alpha") + b"\n"},
            expected_work_ids=("alpha/one",),
        )
    with pytest.raises(CorpusTextQualityError, match="strict UTF-8"):
        audit_corpus_texts(
            {"alpha/one": b"\xff\n"},
            expected_work_ids=("alpha/one",),
        )


def test_exact_duplicate_and_short_in_long_are_blocking():
    duplicate = _text("same")
    duplicate_report = audit_corpus_texts(
        {"alpha/one": duplicate, "beta/two": duplicate},
        expected_work_ids=("alpha/one", "beta/two"),
    )
    assert any(
        row["kind"] == "cross_work_exact_cross_work_chunk"
        for row in duplicate_report.payload["blocking_findings"]
    )

    short_body = " ".join(f"shared{index}" for index in range(280))
    long_body = f"prefix unique words\n\n{short_body}\n\nsuffix unique words"
    contained_report = audit_corpus_texts(
        {
            "alpha/one": f"short\n\n{short_body}\nend\n".encode(),
            "beta/two": f"long\n\n{long_body}\nend\n".encode(),
        },
        expected_work_ids=("alpha/one", "beta/two"),
    )
    assert any(
        row["kind"] == "cross_work_word5_asymmetric_containment"
        for row in contained_report.payload["blocking_findings"]
    )


def test_repeated_whole_prefix_at_tail_is_blocking():
    unit = " ".join(f"token{index}" for index in range(260))
    payload = f"title\n\n{unit}\n\ntitle\n\n{unit}\n".encode()

    report = audit_corpus_texts(
        {"alpha/one": payload},
        expected_work_ids=("alpha/one",),
    )

    assert any(
        row["kind"] == "internal_whole_text_duplication"
        for row in report.payload["blocking_findings"]
    )


def test_inventory_and_scalar_types_are_exact():
    with pytest.raises(CorpusTextQualityError, match="exact dictionary"):
        audit_corpus_texts(  # type: ignore[arg-type]
            [("alpha/one", _text("alpha"))],
            expected_work_ids=("alpha/one",),
        )
    with pytest.raises(CorpusTextQualityError, match="differs"):
        audit_corpus_texts(
            {"alpha/one": _text("alpha")},
            expected_work_ids=("alpha/one", "beta/two"),
        )
    with pytest.raises(CorpusTextQualityError, match="exact positive"):
        audit_corpus_texts(
            {"alpha/one": _text("alpha")},
            expected_work_ids=("alpha/one",),
            minimum_words=True,
        )


def test_report_rejects_rehashed_schema_drift():
    report = audit_corpus_texts(
        {"alpha/one": _text("alpha")},
        expected_work_ids=("alpha/one",),
    )
    raw = copy.deepcopy(report.to_dict())
    raw["extra"] = "not allowed"

    with pytest.raises(CorpusTextQualityError, match="missing or extra"):
        CorpusTextAuditReport(raw).validate()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["works"][0].update({"extra": "forbidden"}),
        lambda raw: raw["works"][0].update({"word_count": "260"}),
        lambda raw: raw["works"][0]["transport_residue_findings"].append(
            {
                "kind": "wiki_link",
                "line_number": True,
                "line_sha256": "0" * 64,
                "excerpt": "[[bad]]",
            }
        ),
    ],
)
def test_report_rejects_rehashed_nested_schema_and_type_drift(mutation):
    report = audit_corpus_texts(
        {"alpha/one": _text("alpha")},
        expected_work_ids=("alpha/one",),
    )
    raw = copy.deepcopy(report.to_dict())
    mutation(raw)
    raw["self_hash"] = canonical_hash(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )

    with pytest.raises(CorpusTextQualityError):
        CorpusTextAuditReport(raw).validate()
