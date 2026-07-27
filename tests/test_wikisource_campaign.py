from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from stylo.corpus_tools import wikisource_campaign as campaign
from stylo.corpus_tools import wikisource_vnext as ws
from stylo.jsonio import canonical_hash, dump_strict


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _work_material(
    *,
    work_id: str,
    revision_id: int,
) -> tuple[ws.PinnedWorkSpec, dict[str, object]]:
    label = work_id.replace("/", "-")
    words = " ".join(f"слово{revision_id}_{index}" for index in range(32))
    html = (
        '<div class="mw-parser-output">'
        '<div id="headertemplate" class="ws-noexport">Служебное</div>'
        f"<p>{label}&nbsp;начало {words}.</p>"
        f"<p>Финальная фраза {label}.</p>"
        "</div>"
    )
    plain = ws.extract_rendered_html(html)
    wikitext = f'<pages index="{revision_id}.pdf" from="1" to="9" />'
    plain_payload = plain.encode("utf-8")
    part = {
        "ordinal": 0,
        "requested_title": f"Произведение {revision_id}",
        "resolved_title": f"Произведение {revision_id}",
        "redirect_chain": [],
        "page_id": revision_id + 10000,
        "revision_id": revision_id,
        "mediawiki_sha1": f"{revision_id:040x}",
        "wikitext_sha256": _sha(wikitext.encode("utf-8")),
        "rendered_html_sha256": _sha(html.encode("utf-8")),
        "plain_byte_size": len(plain_payload),
        "plain_sha256": _sha(plain_payload),
        "word_count": ws.count_words(plain),
    }
    output = ws.assemble_plain_parts([plain])
    payload: dict[str, object] = {
        "schema_version": ws.PINNED_WORK_SPEC_SCHEMA_VERSION,
        "work_id": work_id,
        "assembly_policy_version": ws.ASSEMBLY_POLICY_VERSION,
        "extraction_policy_version": ws.EXTRACTION_POLICY_VERSION,
        "residue_policy_version": ws.RESIDUE_POLICY_VERSION,
        "word_count_policy_version": ws.WORD_COUNT_POLICY_VERSION,
        "parts": [part],
        "output_relative_path": f"raw/{work_id}.txt",
        "output_byte_size": len(output),
        "output_sha256": _sha(output),
        "word_count": ws.count_words(output.decode("utf-8")),
    }
    record: dict[str, object] = {
        **part,
        "parent_revision_id": revision_id - 1,
        "timestamp": "2026-07-28T00:00:00Z",
        "wikitext": wikitext,
        "rendered_html": html,
    }
    raw = {**payload, "self_hash": canonical_hash(payload)}
    return ws.PinnedWorkSpec.from_dict(raw), record


class _Transport:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = {
            int(row["revision_id"]): copy.deepcopy(row) for row in records
        }
        self.calls: list[dict[str, str]] = []

    def __call__(self, params):
        request = dict(params)
        self.calls.append(request)
        if request["action"] == "query":
            row = self.records[int(request["revids"])]
            return {
                "query": {
                    "pages": [
                        {
                            "pageid": row["page_id"],
                            "title": row["resolved_title"],
                            "revisions": [
                                {
                                    "revid": row["revision_id"],
                                    "parentid": row["parent_revision_id"],
                                    "timestamp": row["timestamp"],
                                    "sha1": row["mediawiki_sha1"],
                                    "slots": {
                                        "main": {"content": row["wikitext"]}
                                    },
                                }
                            ],
                        }
                    ]
                }
            }
        if request["action"] == "parse":
            row = self.records[int(request["oldid"])]
            return {
                "parse": {
                    "title": row["resolved_title"],
                    "pageid": row["page_id"],
                    "revid": row["revision_id"],
                    "text": row["rendered_html"],
                }
            }
        raise AssertionError(f"unexpected request: {request}")


def _fixture():
    second, second_record = _work_material(
        work_id="beta/second",
        revision_id=2202,
    )
    first, first_record = _work_material(
        work_id="alpha/first",
        revision_id=1101,
    )
    spec = campaign.WikisourceCampaignSpec.build([second, first])
    return spec, [first_record, second_record]


def test_campaign_spec_is_strict_sorted_self_hashed_and_generation_derived():
    spec, _records = _fixture()

    assert spec.work_ids == ("alpha/first", "beta/second")
    assert tuple(row.work_id for row in spec.works) == spec.work_ids
    assert spec.validate() is spec
    raw = spec.to_dict()
    core = {
        key: value
        for key, value in raw.items()
        if key not in {"generation_id", "self_hash"}
    }
    assert spec.generation_id == canonical_hash(core)
    assert spec.self_hash == canonical_hash(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    replacement, _record = _work_material(
        work_id="alpha/first",
        revision_id=1102,
    )
    changed = campaign.WikisourceCampaignSpec.build(
        [replacement, spec.works[1]]
    )
    assert changed.generation_id != spec.generation_id

    reordered = copy.deepcopy(raw)
    reordered["work_ids"].reverse()
    reordered["works"].reverse()
    reordered_core = {
        key: value
        for key, value in reordered.items()
        if key not in {"generation_id", "self_hash"}
    }
    reordered["generation_id"] = canonical_hash(reordered_core)
    reordered["self_hash"] = canonical_hash(
        {
            key: value
            for key, value in reordered.items()
            if key != "self_hash"
        }
    )
    with pytest.raises(campaign.WikisourceCampaignError, match="sorted exactly"):
        campaign.WikisourceCampaignSpec.from_dict(reordered)

    extra = copy.deepcopy(raw)
    extra["unexpected"] = None
    with pytest.raises(campaign.WikisourceCampaignError, match="keys must be exact"):
        campaign.WikisourceCampaignSpec.from_dict(extra)

    encoded = json.dumps(raw, ensure_ascii=False)
    duplicate = encoded.replace(
        '"campaign_kind":',
        '"campaign_kind": "duplicate", "campaign_kind":',
        1,
    )
    with pytest.raises(campaign.WikisourceCampaignError, match="duplicate object key"):
        campaign.loads_campaign_spec(duplicate)


def test_serial_materialization_is_deterministic_path_independent_and_resumable(
    tmp_path,
):
    spec, records = _fixture()
    first_transport = _Transport(records)
    first = campaign.materialize_campaign(
        spec,
        output_parent=tmp_path / "one",
        transport=first_transport,
    )
    second = campaign.materialize_campaign(
        spec,
        output_parent=tmp_path / "two",
        transport=_Transport(records),
    )

    assert first.resumed is False
    assert first.root == tmp_path / "one" / spec.generation_id
    assert first.receipt == second.receipt
    assert first.receipt.fit_performed is False
    assert first.receipt.confirmatory_authorized is False
    expected_files = {
        "campaign-receipt.json",
        "raw/alpha/first.txt",
        "raw/beta/second.txt",
        "receipts/alpha/first.json",
        "receipts/beta/second.json",
    }
    assert {
        path.relative_to(first.root).as_posix()
        for path in first.root.rglob("*")
        if path.is_file()
    } == expected_files
    assert [row["revids"] for row in first_transport.calls[::2]] == [
        "1101",
        "2202",
    ]
    assert [row["oldid"] for row in first_transport.calls[1::2]] == [
        "1101",
        "2202",
    ]
    assert all("titles" not in row for row in first_transport.calls)

    def no_network(_params):
        raise AssertionError("exact campaign resume reached the network")

    resumed = campaign.materialize_campaign(
        spec,
        output_parent=tmp_path / "one",
        transport=no_network,
    )
    assert resumed.resumed is True
    assert resumed.receipt == first.receipt


@pytest.mark.parametrize(
    "mutation",
    ["tamper", "missing", "extra", "symlink", "noncanonical_receipt"],
)
def test_existing_campaign_tamper_missing_extra_or_symlink_blocks_without_network(
    tmp_path,
    mutation,
):
    spec, records = _fixture()
    materialized = campaign.materialize_campaign(
        spec,
        output_parent=tmp_path,
        transport=_Transport(records),
    )
    raw = materialized.root / "raw" / "alpha" / "first.txt"
    if mutation == "tamper":
        payload = raw.read_bytes()
        raw.write_bytes(payload[:-2] + b"x\n")
    elif mutation == "missing":
        (materialized.root / "receipts" / "alpha" / "first.json").unlink()
    elif mutation == "extra":
        (materialized.root / "extra.bin").write_bytes(b"extra")
    elif mutation == "symlink":
        raw.unlink()
        raw.symlink_to(materialized.root / "raw" / "beta" / "second.txt")
    else:
        receipt_path = materialized.root / "campaign-receipt.json"
        parsed = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_path.write_text(
            json.dumps(parsed, ensure_ascii=False),
            encoding="utf-8",
        )

    def no_network(_params):
        raise AssertionError("conflicting campaign reached the network")

    with pytest.raises(
        campaign.WikisourceCampaignError,
        match=(
            "receipt/spec/output mismatch"
            if mutation == "tamper"
            else "missing or extra|symlink rejected|noncanonical JSON bytes"
        ),
    ):
        campaign.materialize_campaign(
            spec,
            output_parent=tmp_path,
            transport=no_network,
        )


def test_campaign_receipt_rejects_rehashed_nested_extra_and_bool_as_int(
    tmp_path,
):
    spec, records = _fixture()
    result = campaign.materialize_campaign(
        spec,
        output_parent=tmp_path,
        transport=_Transport(records),
    )
    raw = result.receipt.to_dict()
    raw["works"][0]["extra"] = "forbidden"
    raw["self_hash"] = canonical_hash(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    with pytest.raises(campaign.WikisourceCampaignError, match="keys must be exact"):
        campaign.WikisourceCampaignReceipt.from_dict(raw)

    raw = result.receipt.to_dict()
    raw["fit_performed"] = 0
    raw["self_hash"] = canonical_hash(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    with pytest.raises(campaign.WikisourceCampaignError, match="exact boolean"):
        campaign.WikisourceCampaignReceipt.from_dict(raw)


def test_http_transport_retries_429_and_5xx_with_bounded_exponential_delay():
    headers = Message()
    headers["Retry-After"] = "3"
    attempts = []
    sleeps = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"query":{"ok":true}}'

    def opener(request, *, timeout):
        attempts.append((request, timeout))
        if len(attempts) == 1:
            raise HTTPError(request.full_url, 429, "rate", headers, None)
        if len(attempts) == 2:
            raise HTTPError(request.full_url, 503, "busy", Message(), None)
        return _Response()

    transport = campaign.HTTPJSONTransport(
        user_agent="stylometric-test/1.0 (test@example.invalid)",
        timeout_seconds=7,
        max_attempts=3,
        backoff_seconds=1,
        max_delay_seconds=10,
        opener=opener,
        sleeper=sleeps.append,
    )
    result = transport({"format": "json", "action": "query"})

    assert result == {"query": {"ok": True}}
    assert sleeps == [3.0, 2.0]
    assert [row[1] for row in attempts] == [7.0, 7.0, 7.0]
    assert attempts[0][0].get_header("User-agent").startswith(
        "stylometric-test/"
    )
    assert "action=query&format=json" in attempts[0][0].full_url


def test_cli_materializes_only_pinned_spec_in_fixed_exploratory_namespace(
    tmp_path,
    monkeypatch,
):
    spec, records = _fixture()
    spec_path = tmp_path / "campaign.json"
    dump_strict(spec.to_dict(), spec_path, sort_keys=True)
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "acquire_wikisource_corpus_vnext.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "acquire_wikisource_corpus_vnext_test",
        script_path,
    )
    assert module_spec is not None and module_spec.loader is not None
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)

    fake_root = tmp_path / "repository"
    fake_output = (
        fake_root / "docs" / "exploratory" / "lobo_vnext" / "corpora"
    )
    transport = _Transport(records)
    monkeypatch.setattr(cli, "ROOT", fake_root)
    monkeypatch.setattr(cli, "OUTPUT_PARENT", fake_output)
    monkeypatch.setattr(
        cli,
        "HTTPJSONTransport",
        lambda **_kwargs: transport,
    )

    result = cli.run(
        [
            "--campaign-spec",
            str(spec_path),
            "--user-agent",
            "stylometric-test/1.0",
        ]
    )

    assert result["status"] == "exploratory_corpus_materialized_no_fit"
    assert result["generation_id"] == spec.generation_id
    assert result["namespace_relative_path"] == (
        f"docs/exploratory/lobo_vnext/corpora/{spec.generation_id}"
    )
    assert result["work_count"] == 2
    assert result["fit_performed"] is False
    assert result["confirmatory_authorized"] is False
    assert all("titles" not in row for row in transport.calls)
