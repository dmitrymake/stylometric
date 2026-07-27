from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from stylo.corpus_tools import wikisource_discovery as discovery
from stylo.corpus_tools import wikisource_vnext as ws
from stylo.jsonio import canonical_hash


def _part(
    ordinal: int,
    *,
    work_number: int = 1,
    disposition: str = "whole_rendered_body",
    start: int = 0,
    end: int | None = None,
) -> dict[str, object]:
    revision = work_number * 100 + ordinal + 1
    wikitext = f"Точный викитекст {revision}".encode("utf-8")
    return {
        "ordinal": ordinal,
        "requested_title": f"Произведение {work_number}/{ordinal + 1}",
        "resolved_title": f"Произведение {work_number}/{ordinal + 1}",
        "redirect_chain": [],
        "page_id": work_number * 1000 + ordinal + 1,
        "revision_id": revision,
        "revision_parent_id": revision - 1,
        "revision_sha1": hashlib.sha1(wikitext).hexdigest(),
        "revision_timestamp": f"2026-07-{ordinal + 1:02d}T00:00:00Z",
        "status": "resolved",
        "acquisition_strategy": "parse_oldid",
        "body_start_line": start,
        "body_end_line_exclusive": end,
        "body_disposition": disposition,
    }


def _work(
    work_id: str,
    *,
    number: int,
    parts: int = 1,
) -> dict[str, object]:
    return {
        "work_id": work_id,
        "include_in_corpus": True,
        "legacy_requested_root_title": f"Произведение {number}",
        "legacy_root_revision_id": number * 100,
        "selection_basis": "explicit_test_selection",
        "selection_status": "selected",
        "issues": [],
        "parts": [_part(index, work_number=number) for index in range(parts)],
    }


def _candidate(
    works: list[dict[str, object]] | None = None,
    *,
    status: str = "ready_for_pinning",
    schema_version: str = discovery.DISCOVERY_CANDIDATE_SCHEMA_VERSION_V2,
) -> dict[str, object]:
    rows = works or [_work("author/work", number=1, parts=2)]
    rows = sorted(rows, key=lambda row: str(row["work_id"]))
    part_rows = [
        part
        for work in rows
        for part in work["parts"]  # type: ignore[index]
    ]
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "status": status,
        "source": {
            "site": "ru.wikisource.org",
            "api": ws.API,
            "root_probe_schema_version": "test.root-probe.v1",
            "root_probe_sha256": "a" * 64,
        },
        "generated_at": "2026-07-28T00:00:00Z",
        "rejected_work_receipts": [],
        "selection_contract": {
            "source_work_count": len(rows),
            "selected_work_count": len(rows),
            "excluded_work_ids": [],
            "part_order_source": "explicit test order",
            "one_edition_per_work": True,
            "materialization": (
                discovery.RENDERED_OLDID_MATERIALIZATION
            ),
            "candidate_only": True,
        },
        "summary": {
            "work_count": len(rows),
            "part_count": len(part_rows),
            "resolved_part_count": len(part_rows),
            "missing_part_count": 0,
            "blocked_work_count": 0,
            "authorship_rejected_work_count": 0,
        },
        "unresolved_pages": [],
        "unresolved_choices": [],
        "works": rows,
    }
    return {**payload, "candidate_hash": canonical_hash(payload)}


def _rehash(raw: dict[str, object]) -> dict[str, object]:
    payload = {
        key: value for key, value in raw.items() if key != "candidate_hash"
    }
    raw["candidate_hash"] = canonical_hash(payload)
    return raw


def _html(revision: int) -> str:
    return (
        '<div class="mw-parser-output">'
        '<div id="headertemplate">Служебный заголовок</div>'
        f"<p>Начало части {revision}.</p>"
        f"<p>{' '.join(f'слово{revision}_{index}' for index in range(30))}</p>"
        f"<p>Конец части {revision}.</p>"
        '<ol class="references"><li>Служебная сноска</li></ol>'
        "</div>"
    )


class _Transport:
    def __init__(self, parts: list[dict[str, object]]):
        self.parts = {
            int(part["revision_id"]): copy.deepcopy(part) for part in parts
        }
        self.calls: list[dict[str, str]] = []

    def __call__(self, params):
        request = dict(params)
        self.calls.append(request)
        if request["action"] == "query":
            revision = int(request["revids"])
            part = self.parts[revision]
            return {
                "batchcomplete": True,
                "query": {
                    "pages": [
                        {
                            "pageid": part["page_id"],
                            "title": part["resolved_title"],
                            "revisions": [
                                {
                                    "revid": part["revision_id"],
                                    "parentid": part["revision_parent_id"],
                                    "sha1": part["revision_sha1"],
                                    "timestamp": part["revision_timestamp"],
                                    "slots": {
                                        "main": {
                                            "content": (
                                                f"Точный викитекст {revision}"
                                            )
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                },
            }
        if request["action"] == "parse":
            revision = int(request["oldid"])
            part = self.parts[revision]
            return {
                "parse": {
                    "title": part["resolved_title"],
                    "pageid": part["page_id"],
                    "revid": part["revision_id"],
                    "text": _html(revision),
                }
            }
        raise AssertionError(f"unexpected request: {request}")


def _parsed_material():
    raw = _candidate()
    candidate = discovery.DiscoveryCandidate.from_dict(raw)
    parts = [
        part
        for work in raw["works"]  # type: ignore[union-attr]
        for part in work["parts"]
    ]
    return raw, candidate, _Transport(parts)


def test_ready_candidate_pins_v2_campaign_and_exact_order(tmp_path):
    _, candidate, transport = _parsed_material()

    result = discovery.pin_discovery_candidate(
        candidate,
        cache_dir=tmp_path / "cache",
        transport=transport,
    )

    assert result.candidate_hash == candidate.candidate_hash
    assert result.campaign_spec.work_ids == ("author/work",)
    work = result.campaign_spec.works[0]
    assert work.schema_version == ws.PINNED_WORK_SPEC_SCHEMA_VERSION_V2
    assert (
        work.body_boundary_policy_version
        == ws.BODY_BOUNDARY_POLICY_VERSION_V2
    )
    assert [part.ordinal for part in work.parts] == [0, 1]
    assert all(
        part.body_boundary_v2 is not None
        and part.body_boundary_v2.body_disposition == "whole_rendered_body"
        for part in work.parts
    )
    output = result.assembled_outputs["author/work"]
    first = ws.extract_rendered_html(_html(101)).encode("utf-8")
    second = ws.extract_rendered_html(_html(102)).encode("utf-8")
    assert output == first + b"\n\n" + second + b"\n"
    assert [call["action"] for call in transport.calls] == [
        "query",
        "parse",
        "query",
        "parse",
    ]
    assert all(
        ("revids" in call) != ("oldid" in call)
        for call in transport.calls
    )


def test_boundary_selection_is_explicit_and_bound_into_v2_part(tmp_path):
    work = _work("author/work", number=2)
    work["parts"][0].update(  # type: ignore[index, union-attr]
        {
            "body_start_line": 1,
            "body_end_line_exclusive": 2,
            "body_disposition": "strip_both_apparatus",
        }
    )
    raw = _candidate([work])
    candidate = discovery.DiscoveryCandidate.from_dict(raw)
    transport = _Transport(work["parts"])  # type: ignore[arg-type]

    result = discovery.pin_discovery_candidate(
        candidate,
        cache_dir=tmp_path / "cache",
        transport=transport,
    )

    part = result.campaign_spec.works[0].parts[0]
    assert part.body_boundary_v2 is not None
    assert part.body_boundary_v2.start_line == 1
    assert part.body_boundary_v2.end_line_exclusive == 2
    assert (
        result.assembled_outputs["author/work"].decode("utf-8")
        == " ".join(f"слово201_{index}" for index in range(30)) + "\n"
    )


def test_response_cache_replays_without_network_and_rejects_tampering(tmp_path):
    _, candidate, transport = _parsed_material()
    cache = tmp_path / "cache"
    first = discovery.pin_discovery_candidate(
        candidate,
        cache_dir=cache,
        transport=transport,
    )
    assert len(list(cache.iterdir())) == 4

    def forbidden(_params):
        raise AssertionError("resumed pinning must not make an HTTP request")

    resumed = discovery.pin_discovery_candidate(
        candidate,
        cache_dir=cache,
        transport=forbidden,
    )
    assert resumed.campaign_spec == first.campaign_spec

    target = sorted(cache.glob("query-*.json"))[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["response"]["query"]["pages"][0]["title"] = "tampered"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="response hash mismatch",
    ):
        discovery.pin_discovery_candidate(
            candidate,
            cache_dir=cache,
            transport=forbidden,
        )


def test_conflicting_cache_entry_blocks_resume(tmp_path):
    _, candidate, transport = _parsed_material()
    cache = tmp_path / "cache"
    discovery.pin_discovery_candidate(
        candidate,
        cache_dir=cache,
        transport=transport,
    )
    original = sorted(cache.glob("query-101-*.json"))[0]
    conflicting = cache / f"query-101-{'f' * 64}.json"
    conflicting.write_bytes(original.read_bytes())

    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="conflicting query cache",
    ):
        discovery.pin_discovery_candidate(
            candidate,
            cache_dir=cache,
            transport=lambda _: {},
        )


def test_blocked_or_legacy_candidate_stops_before_cache_and_transport(tmp_path):
    for mutation in ("blocked", "legacy"):
        raw = _candidate()
        if mutation == "blocked":
            raw["status"] = "blocked"
        else:
            raw["schema_version"] = (
                discovery.DISCOVERY_CANDIDATE_SCHEMA_VERSION
            )
            for work in raw["works"]:  # type: ignore[union-attr]
                for part in work["parts"]:
                    for key in discovery._PART_BOUNDARY_V2_KEYS:
                        del part[key]
        _rehash(raw)
        candidate = discovery.DiscoveryCandidate.from_dict(raw)
        calls = []

        with pytest.raises(
            discovery.WikisourceDiscoveryError,
            match="not eligible for pinning",
        ):
            discovery.pin_discovery_candidate(
                candidate,
                cache_dir=tmp_path / mutation,
                transport=lambda params: calls.append(params),
            )
        assert calls == []
        assert not (tmp_path / mutation).exists()


@pytest.mark.parametrize("disposition", ["blocked", "manual"])
def test_included_work_issue_requires_exact_selected_disposition(
    tmp_path,
    disposition,
):
    raw = _candidate()
    raw["works"][0]["issues"] = [  # type: ignore[index, union-attr]
        {
            "chosen_disposition": disposition,
            "kind": "composition",
            "reason": "explicit regression",
        }
    ]
    candidate = discovery.DiscoveryCandidate.from_dict(_rehash(raw))
    calls = []

    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="chosen_disposition",
    ):
        discovery.pin_discovery_candidate(
            candidate,
            cache_dir=tmp_path / disposition,
            transport=lambda params: calls.append(params),
        )
    assert calls == []
    assert not (tmp_path / disposition).exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw.update({"unexpected": None}),
            "keys must be exact",
        ),
        (
            lambda raw: raw["source"].update(  # type: ignore[union-attr]
                {"api": "https://example.invalid/w/api.php"}
            ),
            "not the trusted",
        ),
        (
            lambda raw: raw["works"][0]["parts"][0].update(  # type: ignore[index, union-attr]
                {"acquisition_strategy": "latest_title"}
            ),
            "must be 'parse_oldid'",
        ),
        (
            lambda raw: raw["works"][0]["parts"][1].update(  # type: ignore[index, union-attr]
                {"ordinal": 3}
            ),
            "contiguous",
        ),
        (
            lambda raw: raw["works"][0]["parts"][1].update(  # type: ignore[index, union-attr]
                {
                    "page_id": (
                        raw["works"][0]["parts"][0]["page_id"]  # type: ignore[index, union-attr]
                    ),
                }
            ),
            "duplicate page ids",
        ),
        (
            lambda raw: raw["works"][0]["parts"][0].update(  # type: ignore[index, union-attr]
                {"revision_id": True}
            ),
            "exact integer",
        ),
    ],
)
def test_candidate_schema_fails_closed(mutate, message):
    raw = _candidate()
    mutate(raw)
    _rehash(raw)

    with pytest.raises(discovery.WikisourceDiscoveryError, match=message):
        discovery.DiscoveryCandidate.from_dict(raw)


def test_candidate_hash_and_duplicate_json_keys_are_rejected():
    raw = _candidate()
    raw["summary"]["part_count"] = 999  # type: ignore[index]
    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="candidate_hash mismatch",
    ):
        discovery.DiscoveryCandidate.from_dict(raw)

    encoded = json.dumps(_candidate(), ensure_ascii=False)
    duplicated = encoded.replace(
        '"status": "ready_for_pinning"',
        '"status": "ready_for_pinning", "status": "blocked"',
        1,
    )
    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="duplicate object key",
    ):
        discovery.loads_discovery_candidate(duplicated)


def test_revision_or_parse_identity_mismatch_is_rejected(tmp_path):
    _, candidate, transport = _parsed_material()
    transport.parts[101]["revision_sha1"] = "f" * 40
    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="identity differs from discovery",
    ):
        discovery.pin_discovery_candidate(
            candidate,
            cache_dir=tmp_path / "query-mismatch",
            transport=transport,
        )

    _, candidate, transport = _parsed_material()
    original = transport.__call__

    def parse_mismatch(params):
        response = original(params)
        if params["action"] == "parse":
            response["parse"]["revid"] += 1
        return response

    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="parsed identity differs from discovery",
    ):
        discovery.pin_discovery_candidate(
            candidate,
            cache_dir=tmp_path / "parse-mismatch",
            transport=parse_mismatch,
        )


def test_wikitext_bytes_must_match_mediawiki_sha1(tmp_path):
    _, candidate, transport = _parsed_material()
    original = transport.__call__

    def altered_content(params):
        response = original(params)
        if params["action"] == "query":
            response["query"]["pages"][0]["revisions"][0]["slots"]["main"][
                "content"
            ] += " изменено"
        return response

    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="does not match MediaWiki SHA-1",
    ):
        discovery.pin_discovery_candidate(
            candidate,
            cache_dir=tmp_path / "sha-mismatch",
            transport=altered_content,
        )


def test_campaign_output_is_create_if_absent(tmp_path):
    _, candidate, transport = _parsed_material()
    result = discovery.pin_discovery_candidate(
        candidate,
        cache_dir=tmp_path / "cache",
        transport=transport,
    )
    output = tmp_path / "campaign.json"

    assert (
        discovery.write_campaign_spec_create_if_absent(
            result.campaign_spec,
            output,
        )
        == output
    )
    assert (
        discovery.write_campaign_spec_create_if_absent(
            result.campaign_spec,
            output,
        )
        == output
    )
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        discovery.WikisourceDiscoveryError,
        match="conflicting bytes",
    ):
        discovery.write_campaign_spec_create_if_absent(
            result.campaign_spec,
            output,
        )
