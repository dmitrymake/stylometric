from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from stylo.corpus_tools import wikisource_vnext as ws
from stylo.jsonio import canonical_hash


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _body(label: str, offset: int) -> str:
    words = " ".join(
        f"{label.lower()}слово{index + offset}" for index in range(28)
    )
    return (
        '<div class="mw-content-ltr mw-parser-output">'
        '<div id="headertemplate">'
        '<table class="headertemplate ws-noexport"><tr><td>'
        f"Служебный заголовок {label}; автор Авторов"
        "</td></tr></table></div>"
        '<div class="prp-pages-output">'
        '<p><span class="noprint ws-noexport pagenumber">[1]</span>'
        f"{label}&nbsp;начало {words}.</p>"
        f"<h3>{label} продолжение</h3>"
        f"<p>Финальная фраза части {label}.</p>"
        "</div>"
        '<sup class="reference">[99]</sup>'
        '<ol class="references"><li>Редакторское примечание</li></ol>'
        "</div>"
    )


def _record(
    ordinal: int,
    label: str,
    *,
    requested_title: str | None = None,
    resolved_title: str | None = None,
) -> dict[str, object]:
    requested = requested_title or f"Произведение/{label}"
    resolved = resolved_title or requested
    revision_id = 1000 + ordinal
    page_id = 2000 + ordinal
    wikitext = f'<pages index="{label}.pdf" from="1" to="10" />'
    rendered_html = _body(label, ordinal * 100)
    plain = ws.extract_rendered_html(rendered_html)
    chain = (
        [{"from": requested, "to": resolved}]
        if requested != resolved
        else []
    )
    return {
        "ordinal": ordinal,
        "requested_title": requested,
        "resolved_title": resolved,
        "redirect_chain": chain,
        "page_id": page_id,
        "revision_id": revision_id,
        "parent_revision_id": revision_id - 1,
        "timestamp": f"2026-07-{ordinal + 1:02d}T12:00:00Z",
        "mediawiki_sha1": f"{revision_id:040x}",
        "wikitext": wikitext,
        "wikitext_sha256": _sha(wikitext.encode("utf-8")),
        "rendered_html": rendered_html,
        "rendered_html_sha256": _sha(rendered_html.encode("utf-8")),
        "plain": plain,
        "plain_byte_size": len(plain.encode("utf-8")),
        "plain_sha256": _sha(plain.encode("utf-8")),
        "word_count": ws.count_words(plain),
    }


def _part_spec(record: dict[str, object]) -> dict[str, object]:
    keys = {
        "ordinal",
        "requested_title",
        "resolved_title",
        "redirect_chain",
        "page_id",
        "revision_id",
        "mediawiki_sha1",
        "wikitext_sha256",
        "rendered_html_sha256",
        "plain_byte_size",
        "plain_sha256",
        "word_count",
    }
    return {key: record[key] for key in keys}


def _spec_raw(
    records: list[dict[str, object]],
    *,
    work_id: str = "author/work",
) -> dict[str, object]:
    output = ws.assemble_plain_parts(
        [str(record["plain"]) for record in records]
    )
    payload: dict[str, object] = {
        "schema_version": ws.PINNED_WORK_SPEC_SCHEMA_VERSION,
        "work_id": work_id,
        "assembly_policy_version": ws.ASSEMBLY_POLICY_VERSION,
        "extraction_policy_version": ws.EXTRACTION_POLICY_VERSION,
        "residue_policy_version": ws.RESIDUE_POLICY_VERSION,
        "word_count_policy_version": ws.WORD_COUNT_POLICY_VERSION,
        "parts": [_part_spec(record) for record in records],
        "output_relative_path": f"raw/{work_id}.txt",
        "output_byte_size": len(output),
        "output_sha256": _sha(output),
        "word_count": ws.count_words(output.decode("utf-8")),
    }
    return {**payload, "self_hash": canonical_hash(payload)}


class _Transport:
    def __init__(self, records: list[dict[str, object]]):
        self.records = {
            int(record["revision_id"]): copy.deepcopy(record)
            for record in records
        }
        self.calls: list[dict[str, str]] = []

    def __call__(self, params):
        params = dict(params)
        self.calls.append(params)
        action = params["action"]
        if action == "query" and "revids" in params:
            record = self.records[int(params["revids"])]
            return {
                "query": {
                    "pages": [
                        {
                            "pageid": record["page_id"],
                            "title": record["resolved_title"],
                            "revisions": [
                                {
                                    "revid": record["revision_id"],
                                    "parentid": record["parent_revision_id"],
                                    "timestamp": record["timestamp"],
                                    "sha1": record["mediawiki_sha1"],
                                    "slots": {
                                        "main": {
                                            "content": record["wikitext"]
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                }
            }
        if action == "parse":
            record = self.records[int(params["oldid"])]
            return {
                "parse": {
                    "title": record["resolved_title"],
                    "pageid": record["page_id"],
                    "revid": record["revision_id"],
                    "text": record["rendered_html"],
                }
            }
        raise AssertionError(f"unexpected transport request: {params}")


def _material(tmp_path: Path, *, redirected: bool = False):
    first = _record(
        0,
        "Первый",
        requested_title=(
            "Старое название/Первый"
            if redirected
            else "Произведение/Первый"
        ),
        resolved_title="Произведение/Первый",
    )
    second = _record(1, "Второй")
    records = [first, second]
    spec = ws.PinnedWorkSpec.from_dict(_spec_raw(records))
    transport = _Transport(records)
    result = ws.materialize_pinned_work(
        spec,
        output_parent=tmp_path,
        transport=transport,
    )
    return records, spec, transport, result


def test_title_resolution_records_exact_redirect_chain_and_revision():
    seen = {}

    def transport(params):
        seen.update(params)
        return {
            "query": {
                "redirects": [
                    {"from": "Старое имя", "to": "Промежуточное имя"},
                    {"from": "Промежуточное имя", "to": "Полное имя"},
                ],
                "pages": [
                    {
                        "pageid": 42,
                        "title": "Полное имя",
                        "revisions": [
                            {
                                "revid": 123,
                                "sha1": "a" * 40,
                                "timestamp": "2026-07-28T00:00:00Z",
                                "slots": {"main": {"content": "текст"}},
                            }
                        ],
                    }
                ],
            }
        }

    resolved = ws.resolve_page("Старое имя", transport=transport)

    assert seen["titles"] == "Старое имя"
    assert seen["redirects"] == "1"
    assert "revids" not in seen
    assert resolved.requested_title == "Старое имя"
    assert resolved.resolved_title == "Полное имя"
    assert [row.to_dict() for row in resolved.redirect_chain] == [
        {"from": "Старое имя", "to": "Промежуточное имя"},
        {"from": "Промежуточное имя", "to": "Полное имя"},
    ]
    assert (resolved.page_id, resolved.revision_id) == (42, 123)


def test_pinned_work_spec_is_strict_self_hashed_and_rejects_duplicate_keys():
    records = [_record(0, "Один"), _record(1, "Два")]
    raw = _spec_raw(records)

    spec = ws.PinnedWorkSpec.from_dict(raw)
    assert spec.to_dict() == raw
    assert spec.validate() is spec

    extra = copy.deepcopy(raw)
    extra["unexpected"] = None
    with pytest.raises(ws.WikisourceAcquisitionError, match="keys must be exact"):
        ws.PinnedWorkSpec.from_dict(extra)

    bad_hash = copy.deepcopy(raw)
    bad_hash["word_count"] = int(bad_hash["word_count"]) + 1
    with pytest.raises(ws.WikisourceAcquisitionError, match="self_hash mismatch"):
        ws.PinnedWorkSpec.from_dict(bad_hash)

    encoded = json.dumps(raw, ensure_ascii=False)
    duplicated = encoded.replace(
        '"work_id": "author/work"',
        '"work_id": "author/work", "work_id": "other/work"',
        1,
    )
    with pytest.raises(
        ws.WikisourceAcquisitionError,
        match="duplicate object key",
    ):
        ws.loads_pinned_work_spec(duplicated)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda raw: raw["parts"][0].__setitem__("ordinal", True),
            "exact integer",
        ),
        (
            lambda raw: raw["parts"][1].__setitem__("ordinal", 4),
            "contiguous manifest order",
        ),
        (
            lambda raw: raw["parts"][1].__setitem__(
                "revision_id", raw["parts"][0]["revision_id"]
            ),
            "duplicate revisions",
        ),
        (
            lambda raw: raw.__setitem__(
                "output_relative_path", "/host/private/work.txt"
            ),
            "canonical relative path",
        ),
    ],
)
def test_pinned_work_spec_rejects_noncanonical_types_order_and_paths(
    mutator, message
):
    raw = _spec_raw([_record(0, "Один"), _record(1, "Два")])
    mutator(raw)
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    raw["self_hash"] = canonical_hash(payload)
    with pytest.raises(ws.WikisourceAcquisitionError, match=message):
        ws.PinnedWorkSpec.from_dict(raw)


def test_rendered_html_extraction_keeps_transcluded_prose_and_removes_chrome():
    rendered = _body("Том", 0)
    plain = ws.extract_rendered_html(rendered)

    assert "томслово0" in plain
    assert "Том продолжение" in plain
    assert "Финальная фраза части Том." in plain
    assert "Служебный заголовок" not in plain
    assert "Авторов" not in plain
    assert "Редакторское примечание" not in plain
    assert "[1]" not in plain
    assert "[99]" not in plain
    assert "\xa0" not in plain
    assert "Том начало" in plain


def test_materialization_fetches_by_revision_renders_oldid_and_assembles_order(
    tmp_path,
):
    records, spec, transport, result = _material(tmp_path, redirected=True)

    expected = ws.assemble_plain_parts(
        [str(record["plain"]) for record in records]
    )
    assert result.resumed is False
    assert result.root == tmp_path / spec.self_hash
    assert result.output_path.read_bytes() == expected
    assert result.receipt.output_sha256 == spec.output_sha256
    assert result.receipt.pinned_work_spec_sha256 == spec.self_hash
    assert [
        row.requested_title for row in result.receipt.parts
    ] == [
        "Старое название/Первый",
        "Произведение/Второй",
    ]
    assert [
        (
            row.output_byte_start,
            row.output_byte_end,
        )
        for row in result.receipt.parts
    ] == [
        (0, len(str(records[0]["plain"]).encode("utf-8"))),
        (
            len(str(records[0]["plain"]).encode("utf-8")) + 2,
            len(expected) - 1,
        ),
    ]
    query_calls = [row for row in transport.calls if row["action"] == "query"]
    parse_calls = [row for row in transport.calls if row["action"] == "parse"]
    assert [row["revids"] for row in query_calls] == ["1000", "1001"]
    assert all("titles" not in row for row in query_calls)
    assert [row["oldid"] for row in parse_calls] == ["1000", "1001"]
    assert all(row["formatversion"] == "2" for row in transport.calls)


def test_materialization_is_path_independent_and_exact_existing_is_resumed(
    tmp_path,
):
    records = [_record(0, "Один"), _record(1, "Два")]
    spec = ws.PinnedWorkSpec.from_dict(_spec_raw(records))
    first = ws.materialize_pinned_work(
        spec,
        output_parent=tmp_path / "first",
        transport=_Transport(records),
    )
    second = ws.materialize_pinned_work(
        spec,
        output_parent=tmp_path / "second",
        transport=_Transport(records),
    )

    assert first.receipt == second.receipt
    assert first.output_path.read_bytes() == second.output_path.read_bytes()

    def should_not_fetch(_params):
        raise AssertionError("exact create-if-absent resume reached network")

    resumed = ws.materialize_pinned_work(
        spec,
        output_parent=tmp_path / "first",
        transport=should_not_fetch,
    )
    assert resumed.resumed is True
    assert resumed.receipt == first.receipt


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("page_id", 9999, "revision identity mismatch"),
        ("resolved_title", "Чужая страница", "revision identity mismatch"),
        ("revision_id", 9999, "revision identity mismatch"),
        ("mediawiki_sha1", "f" * 40, "revision identity mismatch"),
        ("wikitext", "изменившийся текст", "revision identity mismatch"),
    ],
)
def test_pinned_query_identity_or_content_drift_is_rejected_before_parse(
    tmp_path, field, replacement, message
):
    record = _record(0, "Один")
    spec = ws.PinnedWorkSpec.from_dict(_spec_raw([record]))
    transport = _Transport([record])
    transport.records[1000][field] = replacement

    with pytest.raises(ws.WikisourceAcquisitionError, match=message):
        ws.materialize_pinned_work(
            spec,
            output_parent=tmp_path,
            transport=transport,
        )
    assert [call["action"] for call in transport.calls] == ["query"]
    assert not (tmp_path / spec.self_hash).exists()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("page_id", 9999, "parsed identity mismatch"),
        ("resolved_title", "Чужая страница", "parsed identity mismatch"),
        ("revision_id", 9999, "parsed identity mismatch"),
        ("rendered_html", "<p>совсем другой текст книги</p>", "rendered HTML drifted"),
    ],
)
def test_pinned_parse_identity_or_render_drift_is_rejected(
    tmp_path, field, replacement, message
):
    record = _record(0, "Один")
    spec = ws.PinnedWorkSpec.from_dict(_spec_raw([record]))
    transport = _Transport([record])
    if field in {"resolved_title", "revision_id"}:
        # Keep the query response pinned and alter only the parse response.
        original_call = transport.__call__

        def parse_only(params):
            response = original_call(params)
            if params["action"] == "parse":
                response["parse"][
                    "title" if field == "resolved_title" else "revid"
                ] = replacement
            return response

        selected_transport = parse_only
    else:
        transport.records[1000][field] = replacement
        if field == "page_id":
            # Alter only parse pageid; query identity remains pinned.
            transport.records[1000][field] = record[field]
            original_call = transport.__call__

            def parse_only(params):
                response = original_call(params)
                if params["action"] == "parse":
                    response["parse"]["pageid"] = replacement
                return response

            selected_transport = parse_only
        else:
            selected_transport = transport

    with pytest.raises(ws.WikisourceAcquisitionError, match=message):
        ws.materialize_pinned_work(
            spec,
            output_parent=tmp_path,
            transport=selected_transport,
        )
    assert not (tmp_path / spec.self_hash).exists()


def test_redirect_revision_is_rejected_instead_of_becoming_corpus_text(tmp_path):
    record = _record(0, "Один")
    redirected_wikitext = "#перенаправление [[Произведение/Другой том]]"
    record["wikitext"] = redirected_wikitext
    record["wikitext_sha256"] = _sha(redirected_wikitext.encode("utf-8"))
    spec = ws.PinnedWorkSpec.from_dict(_spec_raw([record]))

    with pytest.raises(
        ws.WikisourceAcquisitionError,
        match="redirect revision",
    ):
        ws.materialize_pinned_work(
            spec,
            output_parent=tmp_path,
            transport=_Transport([record]),
        )


@pytest.mark.parametrize(
    "residue",
    [
        "перенаправление Произведение/Том 1",
        "#REDIRECT [[Other]]",
        "{{Шаблон}}",
        "[[Ссылка]]",
        "<pages index='book.pdf' />",
        "<ref>note</ref>",
        "literal nbsp residue",
        "&nbsp;",
        "Материал из Викитеки",
        "[править]",
        "Категория:Романы",
        "\ufffd",
        "\u200b",
        "\x01",
    ],
)
def test_residue_is_rejected_before_assembly(residue):
    with pytest.raises(
        ws.WikisourceAcquisitionError,
        match="rejected acquisition residue",
    ):
        ws.assemble_plain_parts([f"Нормальное начало. {residue} Конец."])


def test_exact_duplicate_and_exact_contained_parts_are_rejected():
    long = " ".join(f"слово{index}" for index in range(40))
    containing = f"Вступление отдельно. {long} Дополнительный конец."

    with pytest.raises(ws.WikisourceAcquisitionError, match="duplicate parts"):
        ws.assemble_plain_parts([long, long])
    with pytest.raises(
        ws.WikisourceAcquisitionError,
        match="wholly inside another",
    ):
        ws.assemble_plain_parts([long, containing])


def test_corrupt_extra_or_conflicting_existing_namespace_blocks_without_fetch(
    tmp_path,
):
    records, spec, _transport, result = _material(tmp_path)
    (result.root / "extra.bin").write_bytes(b"extra")

    def should_not_fetch(_params):
        raise AssertionError("conflicting namespace reached network")

    with pytest.raises(
        ws.WikisourceAcquisitionError,
        match="missing or extra",
    ):
        ws.materialize_pinned_work(
            spec,
            output_parent=tmp_path,
            transport=should_not_fetch,
        )
    (result.root / "extra.bin").unlink()
    original = result.output_path.read_bytes()
    result.output_path.write_bytes(original[:-1] + b"tamper\n")
    with pytest.raises(
        ws.WikisourceAcquisitionError,
        match="receipt/spec/output mismatch",
    ):
        ws.materialize_pinned_work(
            spec,
            output_parent=tmp_path,
            transport=should_not_fetch,
        )


def test_whole_work_receipt_rejects_nested_extras_and_rehashed_tamper(tmp_path):
    _records, _spec, _transport, result = _material(tmp_path)
    raw = result.receipt.to_dict()
    raw["parts"][0]["extra"] = "forbidden"
    part_payload = {
        key: value
        for key, value in raw["parts"][0].items()
        if key != "self_hash"
    }
    raw["parts"][0]["self_hash"] = canonical_hash(part_payload)
    work_payload = {key: value for key, value in raw.items() if key != "self_hash"}
    raw["self_hash"] = canonical_hash(work_payload)

    with pytest.raises(ws.WikisourceAcquisitionError, match="keys must be exact"):
        ws.WholeWorkReceipt.from_dict(raw)


def test_assembled_output_expectation_mismatch_never_publishes(tmp_path):
    records = [_record(0, "Один")]
    raw = _spec_raw(records)
    raw["output_sha256"] = "0" * 64
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    raw["self_hash"] = canonical_hash(payload)
    spec = ws.PinnedWorkSpec.from_dict(raw)

    with pytest.raises(
        ws.WikisourceAcquisitionError,
        match="assembled whole-work output",
    ):
        ws.materialize_pinned_work(
            spec,
            output_parent=tmp_path,
            transport=_Transport(records),
        )
    assert not (tmp_path / spec.self_hash).exists()
