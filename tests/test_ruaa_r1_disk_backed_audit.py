from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stylo.corpus_tools import ruaa_r1_acquisition as r1
from stylo.corpus_tools import ruaa_r1_disk_backed_audit as audit
from stylo.corpus_tools import text_quality_vnext as tq
from stylo.corpus_tools import wikisource_discovery as discovery
from stylo.corpus_tools import wikisource_vnext as ws
from stylo.corpus_tools.feb_vnext import PinnedFEBWorkSpec
from stylo.corpus_tools.reviewed_text_vnext import (
    ReviewedTextArtifactRef,
    ReviewedTextCampaignSpec,
    ReviewedTextWorkSpec,
)
from stylo.corpus_tools.wikisource_campaign import WikisourceCampaignSpec
from stylo.jsonio import canonical_hash, dump_strict, dumps_strict, loads_strict


def _prose(stem: str, words: int = 240) -> bytes:
    return (
        f"{stem} title\n\n"
        + " ".join(f"{stem}{index}" for index in range(words))
        + f"\n{stem} end\n"
    ).encode("utf-8")


def _synthetic_payloads() -> dict[str, bytes]:
    shared = " ".join(f"shared{index}" for index in range(300))
    duplicate = (
        "Duplicate title\n\n"
        + " ".join(f"duplicate{index}" for index in range(280))
        + "\nDuplicate end\n"
    ).encode("utf-8")
    short = f"Short title\n\n{shared}\nShort end\n".encode("utf-8")
    long = (
        "Long prefix "
        + " ".join(f"prefix{index}" for index in range(40))
        + "\n\n"
        + short.decode("utf-8").rstrip("\n")
        + "\n\n"
        + " ".join(f"suffix{index}" for index in range(40))
        + "\nLong end\n"
    ).encode("utf-8")
    clean = (
        "Clean title\n\n"
        + " ".join(f"clean{index}" for index in range(270))
        + "\nClean end\n"
    ).encode("utf-8")
    return {
        "alpha/exact": duplicate,
        "beta/exact": duplicate,
        "delta/short": short,
        "epsilon/long": long,
        "gamma/exact": duplicate,
        "zeta/clean": clean,
    }


def _inventory(
    root: Path,
    payloads: dict[str, bytes],
    *,
    symlink_work: str | None = None,
) -> Path:
    text_root = root / "texts"
    text_root.mkdir()
    rows = []
    for ordinal, work_id in enumerate(sorted(payloads)):
        payload = payloads[work_id]
        relative = Path("texts") / f"{ordinal:04d}.txt"
        target = root / relative
        if symlink_work == work_id:
            source = root / f"source-{ordinal:04d}.txt"
            source.write_bytes(payload)
            target.symlink_to(source)
        else:
            target.write_bytes(payload)
        rows.append(
            {
                "work_id": work_id,
                "path": relative.as_posix(),
                "byte_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    core = {
        "schema_version": audit.INPUT_INVENTORY_SCHEMA,
        "works": rows,
    }
    document = {**core, "self_hash": canonical_hash(core)}
    path = root / "inventory.json"
    dump_strict(document, path, sort_keys=True, trailing_newline=True)
    return path


def test_disk_backed_reports_have_exact_direct_parity(tmp_path):
    payloads = _synthetic_payloads()
    source = tmp_path / "source"
    source.mkdir()
    inventory = audit.load_input_inventory(_inventory(source, payloads))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    disk = audit.audit_disk_backed(
        inventory.entries,
        selected_exclusions=("zeta/clean",),
        scratch_parent=scratch,
        expected_full_count=6,
        expected_selected_count=5,
    )
    direct_full = tq.audit_corpus_texts(
        payloads,
        expected_work_ids=tuple(sorted(payloads)),
    )
    selected_ids = tuple(
        work_id for work_id in sorted(payloads) if work_id != "zeta/clean"
    )
    direct_selected = tq.audit_corpus_texts(
        {work_id: payloads[work_id] for work_id in selected_ids},
        expected_work_ids=selected_ids,
    )

    assert disk.full_report.to_dict() == direct_full.to_dict()
    assert disk.selected_report.to_dict() == direct_selected.to_dict()
    assert {
        row["kind"]
        for row in disk.full_report.to_dict()["cross_work_overlaps"]
    } == {
        "exact_cross_work_chunk",
        "word5_asymmetric_containment",
    }
    assert list(scratch.iterdir()) == []


def test_inventory_and_each_text_are_read_once(tmp_path, monkeypatch):
    payloads = {
        "alpha/clean": _prose("alpha"),
        "beta/excluded": _prose("tiny", 40),
        "gamma/clean": _prose("gamma"),
    }
    source = tmp_path / "source"
    source.mkdir()
    inventory_path = _inventory(source, payloads)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    reads: list[str] = []
    original = audit._read_stable_regular

    def observed_read(path, *, label, **kwargs):
        reads.append(label)
        return original(path, label=label, **kwargs)

    monkeypatch.setattr(audit, "_read_stable_regular", observed_read)
    inventory = audit.load_input_inventory(inventory_path)
    result = audit.audit_disk_backed(
        inventory.entries,
        selected_exclusions=("beta/excluded",),
        scratch_parent=scratch,
        expected_full_count=3,
        expected_selected_count=2,
    )

    assert result.full_report.status == "blocked"
    assert result.selected_report.status == "passed"
    assert reads.count("materialization inventory") == 1
    for work_id in payloads:
        assert reads.count(f"materialized text {work_id}") == 1
    assert list(scratch.iterdir()) == []


def test_inventory_symlink_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    inventory = _inventory(
        source,
        {
            "alpha/clean": _prose("alpha"),
            "beta/clean": _prose("beta"),
        },
    )
    real_inventory = source / "real-inventory.json"
    inventory.rename(real_inventory)
    inventory.symlink_to(real_inventory.name)

    with pytest.raises(
        audit.DiskBackedAuditError,
        match="missing, unsafe, or has the wrong file type",
    ):
        audit.load_input_inventory(inventory)


def test_text_symlink_is_rejected(tmp_path):
    payloads = {
        "alpha/clean": _prose("alpha"),
        "beta/clean": _prose("beta"),
    }
    source = tmp_path / "source"
    source.mkdir()
    inventory = audit.load_input_inventory(
        _inventory(
            source,
            payloads,
            symlink_work="beta/clean",
        )
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(
        audit.DiskBackedAuditError,
        match="missing, unsafe, or has the wrong file type",
    ):
        audit.audit_disk_backed(
            inventory.entries,
            selected_exclusions=(),
            scratch_parent=scratch,
            expected_full_count=2,
            expected_selected_count=2,
        )

    assert list(scratch.iterdir()) == []


def test_reports_are_canonical_self_hashed_and_direct_equivalent(tmp_path):
    payloads = {
        "alpha/clean": _prose("alpha"),
        "beta/clean": _prose("beta"),
    }
    source = tmp_path / "source"
    source.mkdir()
    inventory = audit.load_input_inventory(_inventory(source, payloads))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = audit.audit_disk_backed(
        inventory.entries,
        selected_exclusions=(),
        scratch_parent=scratch,
        expected_full_count=2,
        expected_selected_count=2,
    )
    direct = tq.audit_corpus_texts(
        payloads,
        expected_work_ids=tuple(sorted(payloads)),
    )

    assert result.full_report.to_dict() == direct.to_dict()
    assert result.selected_report.to_dict() == direct.to_dict()
    for report in (result.full_report, result.selected_report):
        raw = report.to_dict()
        payload = dumps_strict(raw, indent=2, sort_keys=True) + "\n"
        parsed = loads_strict(payload)
        assert parsed == raw
        assert raw["self_hash"] == canonical_hash(
            {
                key: value
                for key, value in raw.items()
                if key != "self_hash"
            }
        )
        assert tq.CorpusTextAuditReport(parsed).validate().to_dict() == raw
    assert list(scratch.iterdir()) == []


def test_inventory_hardlink_alias_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    payloads = {
        "alpha/clean": _prose("alpha"),
        "beta/clean": _prose("alpha"),
    }
    inventory = audit.load_input_inventory(_inventory(source, payloads))
    first = source / "texts" / "0000.txt"
    second = source / "texts" / "0001.txt"
    second.unlink()
    os.link(first, second)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(
        audit.DiskBackedAuditError,
        match="aliases one inode",
    ):
        audit.audit_disk_backed(
            inventory.entries,
            selected_exclusions=(),
            scratch_parent=scratch,
            expected_full_count=2,
            expected_selected_count=2,
        )

    assert list(scratch.iterdir()) == []


# ---------------------------------------------------------------------------
# Real replay orchestration over a compact synthetic corpus.
#
# The fixture mirrors the production shape at 1/20 scale: two Wikisource
# works, one reviewed-text work, one FEB work, and the same two excluded
# evidence works.  The replay entry point itself is never stubbed; only the
# frozen production identities are repointed at the fixture.
# ---------------------------------------------------------------------------

ACQUISITION_CLI_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluation"
    / "acquire_ruaa_r1_corpus_vnext.py"
)
WS_WORK_IDS = ("wsauthor/work000", "wsauthor/work001")
REVIEWED_WORK_ID = "revauthor/work000"
UMBRELLA_WORK_ID = "turgenev/записки_охотника"
AUTHORSHIP_WORK_ID = r1.R1_AUTHORSHIP_MISMATCH_WORK_ID
SOURCE_QUALITY_WORK_ID = r1.R1_SOURCE_QUALITY_REJECTED_WORK_ID
EVIDENCE_BUILDER_NAME = "synthetic-reviewed-builder.py"


def _load_acquisition_cli(name: str):
    module_spec = importlib.util.spec_from_file_location(
        name,
        ACQUISITION_CLI_PATH,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tokens(stem: str, count: int = 240) -> str:
    return " ".join(f"{stem}слово{index}" for index in range(count))


def _feb_html() -> bytes:
    names = (
        "ПЕРВАЯ",
        "ВТОРАЯ",
        "ТРЕТИЯ",
        "ЧЕТВЕРТАЯ",
        "ПЯТАЯ",
        "ШЕСТАЯ",
        "СЕДЬМАЯ",
        "ОСЬМАЯ",
    )
    parts = [
        '<html><body><div id="prose">',
        '<h4 id="ПРЕДИСЛОВИЕ"></h4>',
        "<p>ПРЕДИСЛОВИЕ</p>",
        "<p>" + _tokens("пугачевпредисловие", 45) + "</p>",
    ]
    for chapter, name in enumerate(names):
        parts.extend(
            [
                f'<h4 id="ГЛАВА_{name}"></h4>',
                f"<p>ГЛАВА {name}</p>",
                "<p>" + _tokens(f"пугачевглава{chapter}", 45) + "</p>",
            ]
        )
    parts.extend(
        [
            '<h4 id="ПРИМЕЧАНИЯ"></h4>',
            "<p>Редакторские примечания вне корпуса</p>",
            "</div></body></html>",
        ]
    )
    return "".join(parts).encode("windows-1251")


def _as_unrepaired_v4(spec: ws.PinnedWorkSpec) -> ws.PinnedWorkSpec:
    raw = copy.deepcopy(spec.to_dict())
    raw["schema_version"] = ws.PINNED_WORK_SPEC_SCHEMA_VERSION_V4
    raw["source_repair_policy_version"] = ws.SOURCE_REPAIR_POLICY_VERSION_V1
    for part in raw["parts"]:
        part["pre_repair_plain_byte_size"] = part["plain_byte_size"]
        part["pre_repair_plain_sha256"] = part["plain_sha256"]
        part["pre_repair_word_count"] = part["word_count"]
        part["source_repairs"] = []
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    raw["self_hash"] = canonical_hash(payload)
    return ws.PinnedWorkSpec.from_dict(raw)


def _wikisource_work(index: int, work_id: str):
    """Return the pinned spec, candidate row, and cache record of one work."""

    title = f"Синтетическое произведение {index}"
    revision = 100_000 + index
    page_id = 200_000 + index
    wikitext = f"Синтетический викитекст {index}"
    rendered = (
        '<div class="mw-parser-output"><p>'
        + _tokens(f"викиработа{index}")
        + "</p></div>"
    )
    full_plain = ws.extract_rendered_html(rendered)
    selection = ws.build_body_selection_v2(
        full_plain,
        start_line=0,
        end_line_exclusive=len(full_plain.split("\n")),
        body_disposition="whole_rendered_body",
    )
    revision_sha1 = hashlib.sha1(wikitext.encode("utf-8")).hexdigest()
    part = ws.PinnedPartSpec.from_dict(
        {
            "ordinal": 0,
            "requested_title": title,
            "resolved_title": title,
            "redirect_chain": [],
            "page_id": page_id,
            "revision_id": revision,
            "mediawiki_sha1": revision_sha1,
            "wikitext_sha256": _sha(wikitext.encode("utf-8")),
            "rendered_html_sha256": _sha(rendered.encode("utf-8")),
            **selection.to_part_fields(),
        },
        schema_version=ws.PINNED_WORK_SPEC_SCHEMA_VERSION_V2,
    )
    output = ws.assemble_plain_parts([selection.selected_plain])
    payload: dict[str, object] = {
        "schema_version": ws.PINNED_WORK_SPEC_SCHEMA_VERSION_V2,
        "work_id": work_id,
        "assembly_policy_version": ws.ASSEMBLY_POLICY_VERSION,
        "extraction_policy_version": ws.EXTRACTION_POLICY_VERSION,
        "residue_policy_version": ws.RESIDUE_POLICY_VERSION,
        "word_count_policy_version": ws.WORD_COUNT_POLICY_VERSION,
        "body_boundary_policy_version": ws.BODY_BOUNDARY_POLICY_VERSION_V2,
        "parts": [part.to_dict()],
        "output_relative_path": f"raw/{work_id}.txt",
        "output_byte_size": len(output),
        "output_sha256": _sha(output),
        "word_count": ws.count_words(output.decode("utf-8")),
    }
    spec = _as_unrepaired_v4(
        ws.PinnedWorkSpec.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        )
    )
    record = {
        "page_id": page_id,
        "title": title,
        "revision_id": revision,
        "parent_id": revision - 1,
        "timestamp": f"2026-07-{index + 1:02d}T00:00:00Z",
        "revision_sha1": revision_sha1,
        "wikitext": wikitext,
        "rendered": rendered,
    }
    return spec, _candidate_work(work_id, record), record


def _candidate_work(
    work_id: str,
    record: dict[str, object],
    *,
    include: bool = True,
    selection_status: str = discovery.SELECTED_WORK_STATUS,
) -> dict[str, object]:
    return {
        "work_id": work_id,
        "include_in_corpus": include,
        "legacy_requested_root_title": str(record["title"]),
        "legacy_root_revision_id": int(record["revision_id"]),
        "selection_basis": "synthetic replay fixture",
        "selection_status": selection_status,
        "issues": [],
        "parts": [
            {
                "ordinal": 0,
                "requested_title": record["title"],
                "resolved_title": record["title"],
                "redirect_chain": [],
                "page_id": record["page_id"],
                "revision_id": record["revision_id"],
                "revision_parent_id": record["parent_id"],
                "revision_sha1": record["revision_sha1"],
                "revision_timestamp": record["timestamp"],
                "status": discovery.RESOLVED_PART_STATUS,
                "acquisition_strategy": discovery.PARSE_OLDID_STRATEGY,
                "body_start_line": 0,
                "body_end_line_exclusive": None,
                "body_disposition": "whole_rendered_body",
            }
        ],
    }


def _plain_record(index: int, title: str) -> dict[str, object]:
    revision = 300_000 + index
    return {
        "page_id": 400_000 + index,
        "title": title,
        "revision_id": revision,
        "parent_id": revision - 1,
        "timestamp": f"2026-06-{index + 1:02d}T00:00:00Z",
        "revision_sha1": hashlib.sha1(
            title.encode("utf-8")
        ).hexdigest(),
    }


def _rejected_receipt(work_id: str, record: dict[str, object]) -> dict:
    payload = {
        "schema_version": "stylo.ruaa-r1-rejected-work-receipt.v1",
        "work_id": work_id,
        "reason_code": "authorship_mismatch",
        "disposition": "exclude_from_corpus",
        "evidence": {
            "requested_title": record["title"],
            "resolved_title": record["title"],
            "page_id": record["page_id"],
            "revision_id": record["revision_id"],
            "revision_sha1": record["revision_sha1"],
            "publication": "Синтетическая газета, 1933",
            "closing_signature": "Синтетическая подпись",
            "body_characterization": "third-person essay by another author",
        },
    }
    return {**payload, "self_hash": canonical_hash(payload)}


def _write_cache(cache: Path, records: list[dict[str, object]]) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    for record in records:
        revision = int(record["revision_id"])
        discovery._write_cached_response(
            cache,
            request_kind="query",
            revision_id=revision,
            request=discovery._request_parameters("query", revision),
            response={
                "query": {
                    "pages": [
                        {
                            "pageid": record["page_id"],
                            "title": record["title"],
                            "revisions": [
                                {
                                    "revid": revision,
                                    "parentid": record["parent_id"],
                                    "timestamp": record["timestamp"],
                                    "sha1": record["revision_sha1"],
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
            },
        )
        discovery._write_cached_response(
            cache,
            request_kind="parse",
            revision_id=revision,
            request=discovery._request_parameters("parse", revision),
            response={
                "parse": {
                    "title": record["title"],
                    "pageid": record["page_id"],
                    "revid": revision,
                    "text": record["rendered"],
                }
            },
        )
    return cache


def _write_json(document: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_strict(document, path, sort_keys=True, trailing_newline=True)
    return path


def _self_hashed(payload: dict) -> dict:
    return {**payload, "self_hash": canonical_hash(payload)}


def _source_candidate_document(
    works: list[dict[str, object]],
    receipts: list[dict],
) -> dict:
    rows = sorted(works, key=lambda row: str(row["work_id"]))
    parts = [part for row in rows for part in row["parts"]]
    payload: dict[str, object] = {
        "schema_version": discovery.DISCOVERY_CANDIDATE_SCHEMA_VERSION_V2,
        "status": discovery.READY_STATUS,
        "source": {
            "site": "ru.wikisource.org",
            "api": ws.API,
            "root_probe_schema_version": "synthetic.root-probe.v1",
            "root_probe_sha256": "a" * 64,
        },
        "generated_at": "2026-07-28T00:00:00Z",
        "rejected_work_receipts": receipts,
        "selection_contract": {
            "source_work_count": len(rows),
            "selected_work_count": sum(
                bool(row["include_in_corpus"]) for row in rows
            ),
            "excluded_work_ids": [],
            "part_order_source": "synthetic replay fixture order",
            "one_edition_per_work": True,
            "materialization": discovery.RENDERED_OLDID_MATERIALIZATION,
            "candidate_only": True,
        },
        "summary": {
            "work_count": len(rows),
            "part_count": len(parts),
            "resolved_part_count": len(parts),
            "missing_part_count": 0,
            "blocked_work_count": sum(
                row["selection_status"] != discovery.SELECTED_WORK_STATUS
                for row in rows
            ),
            "authorship_rejected_work_count": sum(
                row["selection_status"] == "authorship_rejected"
                for row in rows
            ),
        },
        "unresolved_pages": [],
        "unresolved_choices": [],
        "works": rows,
    }
    return {**payload, "candidate_hash": canonical_hash(payload)}


def _dispositions_document(
    fixture: SimpleNamespace,
    *,
    generated_ready_candidate: dict,
) -> dict:
    return _self_hashed(
        {
            "schema_version": "stylo.ruaa-r1.source-dispositions.v1",
            "models_executed": False,
            "expected_inventory": {
                "feb_work_count": 1,
                "included_work_count": 4,
                "reviewed_text_work_count": 1,
                "source_part_count": 6,
                "source_work_count": 6,
                "wikisource_work_count": 2,
            },
            "source_candidate": {
                "candidate_hash": fixture.source_candidate_hash,
                "file_sha256": fixture.source_candidate_file_sha256,
            },
            "pinned_wikisource_campaign": {
                "file_sha256": fixture.wikisource_campaign_file_sha256,
                "generation_id": fixture.wikisource_campaign.generation_id,
                "self_hash": fixture.wikisource_campaign.self_hash,
            },
            "reviewed_text_campaign": {
                "file_sha256": fixture.reviewed_campaign_file_sha256,
                "self_hash": fixture.reviewed_campaign.self_hash,
            },
            "generated_ready_candidate": generated_ready_candidate,
            "work_dispositions": [
                {
                    "work_id": REVIEWED_WORK_ID,
                    "disposition": "pinned_external_provider",
                    "provider": "reviewed_text",
                    "output_sha256": fixture.reviewed_text_sha256,
                    "reason": "synthetic reviewed composition",
                },
                {
                    "work_id": r1.R1_FEB_WORK_ID,
                    "disposition": "pinned_external_provider",
                    "provider": "feb",
                    "output_sha256": fixture.feb_spec.output_sha256,
                    "provider_spec_file_sha256": fixture.feb_spec_file_sha256,
                    "provider_spec_self_hash": fixture.feb_spec.self_hash,
                    "reason": "synthetic pinned FEB narrative",
                },
                {
                    "work_id": AUTHORSHIP_WORK_ID,
                    "disposition": "exclude_authorship_mismatch",
                    "provider": "excluded",
                    "reason": "synthetic authorship mismatch",
                    "rejected_work_receipt_self_hash": (
                        fixture.rejected_receipt["self_hash"]
                    ),
                },
                {
                    "work_id": SOURCE_QUALITY_WORK_ID,
                    "disposition": "exclude_source_quality",
                    "provider": "excluded",
                    "reason": "synthetic source-quality rejection",
                    "evidence_file_sha256": fixture.source_quality_sha256,
                    "evidence_self_hash": canonical_hash(
                        {"evidence": fixture.source_quality_sha256}
                    ),
                },
                {
                    "work_id": UMBRELLA_WORK_ID,
                    "disposition": "exclude_collection_umbrella",
                    "provider": "excluded",
                    "reason": "synthetic collection umbrella",
                },
            ],
        }
    )


def _prepare_replay_fixture(root: Path, cli, monkeypatch) -> SimpleNamespace:
    """Write the synthetic inputs and pin the identities the replay asserts."""

    candidate_builder = cli.candidate_builder
    manifest_builder = cli.manifest_builder
    fixture = SimpleNamespace(
        cli=cli,
        production_selected_work_count=cli.EXPECTED_SELECTED_WORK_COUNT,
        production_full_work_count=cli.EXPECTED_FULL_WORK_COUNT,
        production_excluded_evidence=dict(cli.EXCLUDED_EVIDENCE_TEXTS),
    )

    # The editorial evidence is bound by name and hash only, so the campaign
    # names a builder artifact that is deliberately absent from the tree.
    builder_ref = ReviewedTextArtifactRef.build(
        logical_name=EVIDENCE_BUILDER_NAME,
        payload=b"# synthetic reviewed builder\n",
    )

    reviewed_payload = (_tokens("вычитанныйтекст") + "\n").encode("utf-8")
    probe = ReviewedTextWorkSpec.build(
        work_id=REVIEWED_WORK_ID,
        text_payload=reviewed_payload,
        builder_artifacts=[builder_ref],
        provenance_artifacts=[builder_ref],
        source_part_count=1,
        reviewed_part_count=1,
    )
    artifact_cache = root / "artifact-cache"
    (artifact_cache / "sha256").mkdir(parents=True)
    (artifact_cache / "sha256" / f"{probe.sha256}.txt").write_bytes(
        reviewed_payload
    )
    fixture.artifact_cache = artifact_cache
    fixture.reviewed_text_sha256 = probe.sha256

    fixture.reviewed_campaign = ReviewedTextCampaignSpec.build([probe])
    reviewed_campaign_path = _write_json(
        fixture.reviewed_campaign.to_dict(),
        root / "tracked" / "reviewed-campaign.json",
    )
    fixture.reviewed_campaign_path = reviewed_campaign_path
    fixture.reviewed_campaign_file_sha256 = _sha(
        reviewed_campaign_path.read_bytes()
    )
    assert not (root / "tracked" / EVIDENCE_BUILDER_NAME).exists()
    monkeypatch.setattr(cli, "REVIEWED_CAMPAIGN_PATH", reviewed_campaign_path)

    pinned_specs = []
    candidate_works = []
    cache_records = []
    for index, work_id in enumerate(WS_WORK_IDS):
        spec, work_row, record = _wikisource_work(index, work_id)
        pinned_specs.append(spec)
        candidate_works.append(work_row)
        cache_records.append(record)
    fixture.wikisource_campaign = WikisourceCampaignSpec.build(pinned_specs)
    wikisource_campaign_path = _write_json(
        fixture.wikisource_campaign.to_dict(),
        root / "pin" / "campaign-spec.json",
    )
    fixture.wikisource_campaign_path = wikisource_campaign_path
    fixture.wikisource_campaign_file_sha256 = _sha(
        wikisource_campaign_path.read_bytes()
    )
    fixture.wikisource_cache = _write_cache(
        root / "wikisource-cache",
        cache_records,
    )

    feb_body = _feb_html()
    feb_response_path = root / "inputs" / "feb-response.body"
    feb_response_path.parent.mkdir(parents=True, exist_ok=True)
    feb_response_path.write_bytes(feb_body)
    fixture.feb_response_path = feb_response_path
    fixture.feb_response_file_sha256 = _sha(feb_body)
    monkeypatch.setattr(
        cli,
        "EXPECTED_FEB_RESPONSE_FILE_SHA256",
        fixture.feb_response_file_sha256,
    )
    fixture.feb_spec = PinnedFEBWorkSpec.build(
        work_id=r1.R1_FEB_WORK_ID,
        response_body=feb_body,
    )
    feb_spec_path = _write_json(
        fixture.feb_spec.to_dict(),
        root / "tracked" / "feb-work-spec.json",
    )
    fixture.feb_spec_file_sha256 = _sha(feb_spec_path.read_bytes())

    reviewed_record = _plain_record(0, "Синтетическая вычитанная работа")
    feb_record = _plain_record(1, "Синтетическая история")
    authorship_record = _plain_record(2, "У нас и у них")
    quality_record = _plain_record(3, "Дон на костылях")
    fixture.rejected_receipt = _rejected_receipt(
        AUTHORSHIP_WORK_ID,
        authorship_record,
    )
    candidate_works.extend(
        [
            _candidate_work(REVIEWED_WORK_ID, reviewed_record),
            _candidate_work(r1.R1_FEB_WORK_ID, feb_record),
            _candidate_work(
                AUTHORSHIP_WORK_ID,
                authorship_record,
                include=False,
                selection_status="authorship_rejected",
            ),
            _candidate_work(SOURCE_QUALITY_WORK_ID, quality_record),
        ]
    )
    source_candidate_path = _write_json(
        _source_candidate_document(
            candidate_works,
            [fixture.rejected_receipt],
        ),
        root / "inputs" / "source-candidate.json",
    )
    fixture.source_candidate_path = source_candidate_path
    fixture.source_candidate_file_sha256 = _sha(
        source_candidate_path.read_bytes()
    )
    fixture.source_candidate_hash = loads_strict(
        source_candidate_path.read_text(encoding="utf-8")
    )["candidate_hash"]

    excluded_root = root / "excluded-evidence"
    fixture.excluded_texts = {
        AUTHORSHIP_WORK_ID: (
            _tokens("исключённыйавторство") + "\n"
        ).encode("utf-8"),
        SOURCE_QUALITY_WORK_ID: (
            _tokens("исключённоекачество")
            + "\nматчиш <sic!> в граммофоне\n"
        ).encode("utf-8"),
    }
    fixture.clean_excluded_texts = {
        AUTHORSHIP_WORK_ID: fixture.excluded_texts[AUTHORSHIP_WORK_ID],
        SOURCE_QUALITY_WORK_ID: (
            _tokens("исключённоекачество") + "\nматчиш в граммофоне\n"
        ).encode("utf-8"),
    }
    fixture.excluded_evidence_root = _write_excluded_evidence(
        excluded_root,
        fixture.excluded_texts,
    )
    fixture.source_quality_sha256 = _sha(
        fixture.excluded_texts[SOURCE_QUALITY_WORK_ID]
    )

    fixture.parse_audit_path = _write_json(
        _self_hashed(
            {
                "schema_version": "synthetic.parts-audit.v2",
                "status": "complete_with_findings",
                "summary": {
                    "work_count": 6,
                    "part_count": 6,
                    "distinct_parsed_revision_count": 6,
                    "oldid_query_error_count": 0,
                    "parse_error_count": 0,
                },
            }
        ),
        root / "inputs" / "parse-audit.json",
    )

    dispositions_path = root / "tracked" / "source-dispositions.json"
    monkeypatch.setattr(
        candidate_builder,
        "SOURCE_CANDIDATE_FILE_SHA256",
        fixture.source_candidate_file_sha256,
    )
    monkeypatch.setattr(
        candidate_builder,
        "SOURCE_CANDIDATE_HASH",
        fixture.source_candidate_hash,
    )
    monkeypatch.setattr(
        candidate_builder,
        "DISPOSITIONS_PATH",
        dispositions_path,
    )
    monkeypatch.setattr(candidate_builder, "EXPECTED_WORK_COUNT", 6)
    monkeypatch.setattr(candidate_builder, "EXPECTED_PART_COUNT", 6)
    monkeypatch.setattr(
        candidate_builder,
        "EXPECTED_WIKISOURCE_WORK_COUNT",
        2,
    )
    monkeypatch.setattr(candidate_builder, "EXPECTED_REVIEWED_WORK_COUNT", 1)
    monkeypatch.setattr(candidate_builder, "EXPECTED_INCLUDED_WORK_COUNT", 4)
    monkeypatch.setattr(candidate_builder, "EXPECTED_DISPOSITION_COUNT", 5)
    monkeypatch.setattr(candidate_builder, "SELECTED_QUALITY_PART_PATCHES", {})

    _pin_dispositions(
        fixture,
        dispositions_path,
        {
            "candidate_hash": "0" * 64,
            "file_sha256": "0" * 64,
            "schema_version": discovery.DISCOVERY_CANDIDATE_SCHEMA_VERSION_V4,
        },
        monkeypatch,
    )
    ready_raw = _capture_ready_candidate(
        candidate_builder,
        fixture,
        monkeypatch,
        reviewed_campaign_path=reviewed_campaign_path,
    )
    ready_candidate_path = _write_json(
        ready_raw,
        root / "pin" / "ready-candidate.json",
    )
    fixture.ready_candidate_file_sha256 = _sha(
        ready_candidate_path.read_bytes()
    )
    fixture.ready_candidate_hash = ready_raw["candidate_hash"]
    _pin_dispositions(
        fixture,
        dispositions_path,
        {
            "candidate_hash": fixture.ready_candidate_hash,
            "file_sha256": fixture.ready_candidate_file_sha256,
            "schema_version": ready_raw["schema_version"],
        },
        monkeypatch,
    )

    monkeypatch.setattr(
        manifest_builder,
        "DISPOSITIONS_PATH",
        dispositions_path,
    )
    monkeypatch.setattr(manifest_builder, "FEB_SPEC_PATH", feb_spec_path)
    monkeypatch.setattr(
        manifest_builder,
        "DISPOSITIONS_FILE_SHA256",
        fixture.dispositions_file_sha256,
    )
    monkeypatch.setattr(
        manifest_builder,
        "DISPOSITIONS_SELF_HASH",
        fixture.dispositions_self_hash,
    )
    monkeypatch.setattr(manifest_builder, "EXPECTED_WIKISOURCE_WORK_COUNT", 2)
    monkeypatch.setattr(manifest_builder, "EXPECTED_REVIEWED_WORK_COUNT", 1)
    monkeypatch.setattr(manifest_builder, "EXPECTED_INCLUDED_WORK_COUNT", 4)
    monkeypatch.setattr(manifest_builder, "EXPECTED_SOURCE_WORK_COUNT", 6)
    monkeypatch.setattr(manifest_builder, "EXPECTED_SOURCE_PART_COUNT", 6)
    manifest = manifest_builder.build_manifest(
        ready_candidate_path=ready_candidate_path,
        wikisource_campaign_path=wikisource_campaign_path,
        reviewed_campaign_path=reviewed_campaign_path,
    )
    fixture.manifest = manifest
    fixture.manifest_file_sha256 = _sha(
        _write_json(
            manifest.to_dict(),
            root / "pin" / "acquisition-manifest.json",
        ).read_bytes()
    )

    pin_parent = root / "pin" / "acquisition"
    pin_parent.mkdir(parents=True)
    materialized = r1.materialize_r1_acquisition(
        manifest,
        output_parent=pin_parent,
        wikisource_transport=cli._PinnedCacheTransport(
            fixture.wikisource_cache
        ),
        feb_transport=cli._FEBFileTransport(feb_response_path),
        reviewed_artifact_cache=artifact_cache,
    )
    fixture.selected_audit_file_sha256 = _sha(
        (materialized.root / r1.AUDIT_REPORT_NAME).read_bytes()
    )
    fixture.selected_audit_self_hash = materialized.audit_report.self_hash
    fixture.pin_parent = pin_parent

    fixture.scratch_parent = root / "scratch"
    fixture.scratch_parent.mkdir()
    fixture.tracked_root = root / "tracked"
    # The synthetic tracked contracts live outside the repository, so the
    # fail-closed production recorder cannot describe them.  The fixture
    # declares the empty inventory instead of relaxing that contract.
    monkeypatch.setattr(cli, "_tracked_contract_refs", lambda: [])
    _patch_replay_identities(cli, fixture, monkeypatch)
    return fixture


def _write_excluded_evidence(root: Path, texts: dict[str, bytes]) -> Path:
    for work_id, payload in texts.items():
        target = root.joinpath(*f"{work_id}.txt".split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


def _pin_dispositions(
    fixture: SimpleNamespace,
    path: Path,
    generated_ready_candidate: dict,
    monkeypatch,
) -> Path:
    document = _dispositions_document(
        fixture,
        generated_ready_candidate=generated_ready_candidate,
    )
    _write_json(document, path)
    fixture.dispositions_self_hash = document["self_hash"]
    fixture.dispositions_file_sha256 = _sha(path.read_bytes())
    monkeypatch.setattr(
        fixture.cli.candidate_builder,
        "DISPOSITIONS_SELF_HASH",
        fixture.dispositions_self_hash,
    )
    monkeypatch.setattr(
        fixture.cli.candidate_builder,
        "DISPOSITIONS_FILE_SHA256",
        fixture.dispositions_file_sha256,
    )
    return path


def _capture_ready_candidate(
    candidate_builder,
    fixture: SimpleNamespace,
    monkeypatch,
    *,
    reviewed_campaign_path: Path,
) -> dict:
    """Run the real candidate builder once to learn its exact ready output."""

    captured: dict[str, dict] = {}
    original = discovery.DiscoveryCandidate.from_dict

    def recorder(value):
        captured["raw"] = copy.deepcopy(value)
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(
            discovery.DiscoveryCandidate,
            "from_dict",
            staticmethod(recorder),
        )
        with pytest.raises(candidate_builder.R1CandidateBuilderError):
            candidate_builder.build_candidate(
                fixture.source_candidate_path,
                reviewed_campaign_path=reviewed_campaign_path,
            )
    return captured["raw"]


def _patch_replay_identities(
    cli,
    fixture: SimpleNamespace,
    monkeypatch,
) -> None:
    values = {
        "EXPECTED_REVIEWED_CAMPAIGN_FILE_SHA256": (
            fixture.reviewed_campaign_file_sha256
        ),
        "EXPECTED_REVIEWED_CAMPAIGN_SELF_HASH": (
            fixture.reviewed_campaign.self_hash
        ),
        "EXPECTED_READY_CANDIDATE_FILE_SHA256": (
            fixture.ready_candidate_file_sha256
        ),
        "EXPECTED_READY_CANDIDATE_HASH": fixture.ready_candidate_hash,
        "EXPECTED_MANIFEST_FILE_SHA256": fixture.manifest_file_sha256,
        "EXPECTED_MANIFEST_SELF_HASH": fixture.manifest.self_hash,
        "EXPECTED_MANIFEST_GENERATION_ID": fixture.manifest.generation_id,
        "EXPECTED_PARSE_AUDIT_FILE_SHA256": _sha(
            fixture.parse_audit_path.read_bytes()
        ),
        "EXPECTED_PARSE_AUDIT_SELF_HASH": loads_strict(
            fixture.parse_audit_path.read_text(encoding="utf-8")
        )["self_hash"],
        "EXPECTED_SELECTED_AUDIT_FILE_SHA256": (
            fixture.selected_audit_file_sha256
        ),
        "EXPECTED_SELECTED_AUDIT_SELF_HASH": fixture.selected_audit_self_hash,
        "EXPECTED_SOURCE_WORK_COUNT": 6,
        "EXPECTED_SOURCE_PART_COUNT": 6,
        "EXPECTED_WIKISOURCE_WORK_COUNT": 2,
        "EXPECTED_WIKISOURCE_PART_COUNT": 2,
        "EXPECTED_REVIEWED_WORK_COUNT": 1,
        "EXPECTED_FEB_WORK_COUNT": 1,
        "EXPECTED_SELECTED_WORK_COUNT": 4,
        "EXPECTED_FULL_WORK_COUNT": 6,
        "EXPECTED_WIKISOURCE_CACHE_ENTRY_COUNT": 4,
        "EXPECTED_WIKISOURCE_CACHE_QUERY_COUNT": 2,
        "EXPECTED_WIKISOURCE_CACHE_PARSE_COUNT": 2,
        "EXCLUDED_EVIDENCE_TEXTS": _excluded_identities(
            fixture.excluded_texts
        ),
    }
    for name, value in values.items():
        monkeypatch.setattr(cli, name, value)


def _excluded_identities(texts: dict[str, bytes]) -> dict[str, dict]:
    return {
        work_id: {"byte_size": len(payload), "sha256": _sha(payload)}
        for work_id, payload in texts.items()
    }


def _replay_argv(
    fixture: SimpleNamespace,
    output_parent: Path,
    *,
    excluded_evidence_root: Path | None = None,
) -> list[str]:
    return [
        "--artifact-cache",
        str(fixture.artifact_cache),
        "--source-candidate",
        str(fixture.source_candidate_path),
        "--wikisource-campaign",
        str(fixture.wikisource_campaign_path),
        "--wikisource-cache",
        str(fixture.wikisource_cache),
        "--feb-response",
        str(fixture.feb_response_path),
        "--excluded-evidence-root",
        str(
            fixture.excluded_evidence_root
            if excluded_evidence_root is None
            else excluded_evidence_root
        ),
        "--parse-audit",
        str(fixture.parse_audit_path),
        "--output-parent",
        str(output_parent),
        "--scratch-parent",
        str(fixture.scratch_parent),
    ]


def _inventory_document(path: Path) -> dict:
    return loads_strict(path.read_text(encoding="utf-8"))


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_replay_builds_fresh_inventories_and_leaves_tracked_tree_alone(
    tmp_path,
    monkeypatch,
):
    cli = _load_acquisition_cli("acquire_ruaa_r1_corpus_replay_integration")
    fixture = _prepare_replay_fixture(tmp_path / "fixture", cli, monkeypatch)

    # The production contract still declares the frozen R1 inventory.
    assert fixture.production_selected_work_count == 134
    assert fixture.production_full_work_count == 136
    assert set(fixture.production_excluded_evidence) == {
        AUTHORSHIP_WORK_ID,
        SOURCE_QUALITY_WORK_ID,
    }

    # No previously materialized tree survives into the replay.
    for path in sorted(fixture.pin_parent.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    fixture.pin_parent.rmdir()
    assert not fixture.pin_parent.exists()

    parser = cli._replay_parser()
    for removed in (
        "--full-inventory",
        "--selected-inventory",
        "--update-tracked-snapshot",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([removed, str(tmp_path / "unused.json")])

    output_parent = tmp_path / "replay-1"
    tracked_before = _tree_digest(fixture.tracked_root)
    result = cli._run_replay(_replay_argv(fixture, output_parent))

    assert result["status"] == "materialized_audit_passed"
    assert result["work_count"] == 4
    assert result["resumed"] is False
    assert result["publication_authorized"] is False
    # The declared gates pass even though the full diagnostic is blocked by
    # the excluded source-quality evidence text.
    assert result["full_audit_status"] == "blocked"

    full_path = output_parent / cli.FULL_INVENTORY_NAME
    selected_path = output_parent / cli.SELECTED_INVENTORY_NAME
    full = audit.load_input_inventory(full_path)
    selected = audit.load_input_inventory(selected_path)
    assert len(selected.entries) == 4
    assert len(full.entries) == 6
    assert result["full_inventory_file_sha256"] == full.sha256
    assert result["selected_inventory_self_hash"] == selected.self_hash

    materialized_root = (
        output_parent / "acquisition" / fixture.manifest.generation_id
    )
    for work_id, entry in selected.entries.items():
        assert entry.path == materialized_root.joinpath(
            "raw", *f"{work_id}.txt".split("/")
        )
        assert entry.sha256 == _sha(entry.path.read_bytes())
    for entry in full.entries.values():
        assert entry.path.is_relative_to(output_parent)

    excluded = set(full.entries) - set(selected.entries)
    assert excluded == {AUTHORSHIP_WORK_ID, SOURCE_QUALITY_WORK_ID}
    for work_id in excluded:
        entry = full.entries[work_id]
        assert entry.path.is_relative_to(
            output_parent / cli.EXCLUDED_EVIDENCE_DIRECTORY_NAME
        )
        assert entry.path.read_bytes() == fixture.excluded_texts[work_id]

    for document in (
        _inventory_document(full_path),
        _inventory_document(selected_path),
    ):
        for row in document["works"]:
            assert not str(row["path"]).startswith("/")

    contract = loads_strict(
        (output_parent / cli.CONTRACT_NAME).read_text(encoding="utf-8")
    )
    assert contract["status"] == "materialized_audit_passed"
    assert contract["quality_gate"] == {
        "full_cross_work_isolated": True,
        "full_inventory_report_is_evidence_only": True,
        "gating_inventory": "selected_134",
        "selected_passed": True,
    }
    assert contract["text_audits"]["full_136"]["status"] == "blocked"
    assert contract["text_audits"]["selected_134"]["status"] == "passed"
    # Every recorded source path stays repository-relative.
    recorded = [
        str(row["relative_path"])
        for key in ("executable_sources", "tracked_contracts")
        for row in contract[key]
    ]
    assert recorded and all(
        not path.startswith("/") and str(tmp_path) not in path
        for path in recorded
    )
    # Replay writes nothing outside --output-parent.
    assert _tree_digest(fixture.tracked_root) == tracked_before

    # An exact resume re-reads neither transport.
    resumed = cli._run_replay(_replay_argv(fixture, output_parent))
    assert resumed["resumed"] is True
    assert resumed["wikisource_cache_call_count"] == 0
    assert resumed["feb_response_call_count"] == 0
    assert (
        resumed["execution_contract_self_hash"]
        == result["execution_contract_self_hash"]
    )
    assert _tree_digest(fixture.tracked_root) == tracked_before

    # The tracked campaign is an input under a pinned identity: one drifted
    # byte stops the replay before any provider runs.
    drifted_campaign = tmp_path / "drifted-campaign.json"
    drifted_campaign.write_bytes(
        fixture.reviewed_campaign_path.read_bytes().replace(
            b'"campaign_kind"',
            b'"campaign_kind" ',
            1,
        )
    )
    with monkeypatch.context() as patch:
        patch.setattr(cli, "REVIEWED_CAMPAIGN_PATH", drifted_campaign)
        with pytest.raises(
            cli.R1AcquisitionCLIError,
            match="tracked reviewed campaign file SHA-256 mismatch",
        ):
            cli._run_replay(_replay_argv(fixture, tmp_path / "replay-drift"))

    # Cleaning the excluded evidence text does not break the replay.
    clean_root = _write_excluded_evidence(
        tmp_path / "excluded-clean",
        fixture.clean_excluded_texts,
    )
    monkeypatch.setattr(
        cli,
        "EXCLUDED_EVIDENCE_TEXTS",
        _excluded_identities(fixture.clean_excluded_texts),
    )
    cleaned = cli._run_replay(
        _replay_argv(
            fixture,
            tmp_path / "replay-clean",
            excluded_evidence_root=clean_root,
        )
    )
    assert cleaned["status"] == "materialized_audit_passed"
    assert cleaned["full_audit_status"] == "passed"
    assert cleaned["selected_audit_self_hash"] == (
        result["selected_audit_self_hash"]
    )


def test_excluded_evidence_staging_never_clobbers_a_planted_target(
    tmp_path,
    monkeypatch,
):
    cli = _load_acquisition_cli("acquire_ruaa_r1_corpus_replay_staging")
    fixture = _prepare_replay_fixture(tmp_path / "fixture", cli, monkeypatch)
    payload = fixture.excluded_texts[AUTHORSHIP_WORK_ID]
    relative = Path(cli.EXCLUDED_EVIDENCE_DIRECTORY_NAME).joinpath(
        *f"{AUTHORSHIP_WORK_ID}.txt".split("/")
    )

    def plant(name: str, content: bytes | None) -> Path:
        output_parent = tmp_path / name
        target = output_parent / relative
        target.parent.mkdir(parents=True)
        if content is None:
            sentinel = tmp_path / f"{name}-sentinel.txt"
            sentinel.write_bytes(b"sentinel outside the replay tree\n")
            target.symlink_to(sentinel)
        else:
            target.write_bytes(content)
        return output_parent

    # A pre-planted symlink never redirects the staged write.
    outside = plant("replay-symlink", None)
    sentinel = tmp_path / "replay-symlink-sentinel.txt"
    before = sentinel.read_bytes()
    with pytest.raises(
        cli.R1AcquisitionCLIError,
        match="staged excluded evidence text .* symlink components",
    ):
        cli._run_replay(_replay_argv(fixture, outside))
    assert sentinel.read_bytes() == before
    assert (outside / relative).is_symlink()

    # A mismatched regular file is rejected instead of being overwritten.
    drifted = plant("replay-drifted", payload + b"tampered\n")
    with pytest.raises(
        cli.R1AcquisitionCLIError,
        match="is unsafe or drifted",
    ):
        cli._run_replay(_replay_argv(fixture, drifted))
    assert (drifted / relative).read_bytes() == payload + b"tampered\n"

    # An exact existing copy is accepted, which is the resume path.
    exact = plant("replay-exact", payload)
    result = cli._run_replay(_replay_argv(fixture, exact))
    assert result["status"] == "materialized_audit_passed"
    assert (exact / relative).read_bytes() == payload
