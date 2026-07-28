from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from stylo.corpus_tools import ruaa_r1_acquisition as r1
from stylo.corpus_tools import wikisource_vnext as ws
from stylo.corpus_tools.feb_vnext import (
    FEBHTTPResponse,
    FEB_CONTENT_TYPE,
    FEB_SOURCE_URL,
    PinnedFEBWorkSpec,
)
from stylo.corpus_tools.reviewed_text_vnext import (
    ReviewedTextArtifactRef,
    ReviewedTextCampaignSpec,
    ReviewedTextWorkSpec,
)
from stylo.corpus_tools.wikisource_campaign import WikisourceCampaignSpec
from stylo.jsonio import canonical_hash, dump_strict, dumps_strict


SYNTHETIC_WIKISOURCE_COUNT = 3
ACQUISITION_CLI_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluation"
    / "acquire_ruaa_r1_corpus_vnext.py"
)


def _load_acquisition_cli(name: str):
    module_spec = importlib.util.spec_from_file_location(
        name,
        ACQUISITION_CLI_PATH,
    )
    assert module_spec is not None and module_spec.loader is not None
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    return cli


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plain(index: int, *, audit_blocker: bool = False) -> str:
    text = " ".join(
        f"уникальный{index}токен{word}" for word in range(220)
    )
    if audit_blocker:
        text += "\nИздательство тест, тираж 1000"
    return text


def _rendered(index: int, *, audit_blocker: bool = False) -> str:
    lines = _plain(index, audit_blocker=audit_blocker).split("\n")
    return (
        '<div class="mw-parser-output">'
        + "".join(f"<p>{line}</p>" for line in lines)
        + "</div>"
    )


def _ws_work(
    index: int,
    *,
    work_id: str | None = None,
    audit_blocker: bool = False,
) -> tuple[ws.PinnedWorkSpec, dict[str, object]]:
    identity = work_id or f"author{index:03d}/work{index:03d}"
    title = f"Синтетическое произведение {index}"
    revision = 100_000 + index
    page_id = 200_000 + index
    wikitext = f"Синтетический викитекст {index}"
    rendered = _rendered(index, audit_blocker=audit_blocker)
    full_plain = ws.extract_rendered_html(rendered)
    selection = ws.build_body_selection_v2(
        full_plain,
        start_line=0,
        end_line_exclusive=len(full_plain.split("\n")),
        body_disposition="whole_rendered_body",
    )
    part = ws.PinnedPartSpec.from_dict(
        {
            "ordinal": 0,
            "requested_title": title,
            "resolved_title": title,
            "redirect_chain": [],
            "page_id": page_id,
            "revision_id": revision,
            "mediawiki_sha1": hashlib.sha1(
                wikitext.encode("utf-8")
            ).hexdigest(),
            "wikitext_sha256": _sha(wikitext.encode("utf-8")),
            "rendered_html_sha256": _sha(rendered.encode("utf-8")),
            **selection.to_part_fields(),
        },
        schema_version=ws.PINNED_WORK_SPEC_SCHEMA_VERSION_V2,
    )
    output = ws.assemble_plain_parts([selection.selected_plain])
    payload: dict[str, object] = {
        "schema_version": ws.PINNED_WORK_SPEC_SCHEMA_VERSION_V2,
        "work_id": identity,
        "assembly_policy_version": ws.ASSEMBLY_POLICY_VERSION,
        "extraction_policy_version": ws.EXTRACTION_POLICY_VERSION,
        "residue_policy_version": ws.RESIDUE_POLICY_VERSION,
        "word_count_policy_version": ws.WORD_COUNT_POLICY_VERSION,
        "body_boundary_policy_version": (
            ws.BODY_BOUNDARY_POLICY_VERSION_V2
        ),
        "parts": [part.to_dict()],
        "output_relative_path": f"raw/{identity}.txt",
        "output_byte_size": len(output),
        "output_sha256": _sha(output),
        "word_count": ws.count_words(output.decode("utf-8")),
    }
    spec = ws.PinnedWorkSpec.from_dict(
        {**payload, "self_hash": canonical_hash(payload)}
    )
    record = {
        "page_id": page_id,
        "title": title,
        "revision_id": revision,
        "parent_id": revision - 1,
        "timestamp": "2026-07-28T00:00:00Z",
        "mediawiki_sha1": part.mediawiki_sha1,
        "wikitext": wikitext,
        "rendered": rendered,
    }
    return spec, record


def _feb_html(*, audit_blocker: bool = False) -> bytes:
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
        "<p>"
        + " ".join(f"пугачевпредисловие{word}" for word in range(45))
        + "</p>",
    ]
    for chapter, name in enumerate(names):
        parts.extend(
            [
                f'<h4 id="ГЛАВА_{name}"></h4>',
                f"<p>ГЛАВА {name}</p>",
                "<p>"
                + " ".join(
                    f"пугачевглава{chapter}слово{word}"
                    for word in range(45)
                )
                + "</p>",
            ]
        )
    if audit_blocker:
        parts.append("<p>Издательство тест, тираж 1000</p>")
    parts.extend(
        [
            '<h4 id="ПРИМЕЧАНИЯ"></h4>',
            "<p>Редакторские примечания вне корпуса</p>",
            "</div></body></html>",
        ]
    )
    return "".join(parts).encode("windows-1251")


def _reviewed_payload(index: int) -> bytes:
    return (
        " ".join(
            f"reviewed{index}уникальныйтокен{word}"
            for word in range(220)
        )
        + "\n"
    ).encode("utf-8")


def _reviewed_work(
    index: int,
    *,
    work_id: str | None = None,
    payload: bytes | None = None,
) -> tuple[ReviewedTextWorkSpec, bytes]:
    text = _reviewed_payload(index) if payload is None else payload
    builder = ReviewedTextArtifactRef.build(
        logical_name="synthetic-reviewed-builder.py",
        payload=f"builder {index}".encode("utf-8"),
    )
    provenance = ReviewedTextArtifactRef.build(
        logical_name="synthetic-review-receipt.json",
        payload=f"review receipt {index}".encode("utf-8"),
    )
    return (
        ReviewedTextWorkSpec.build(
            work_id=work_id or f"reviewed{index:03d}/work{index:03d}",
            text_payload=text,
            builder_artifacts=[builder],
            provenance_artifacts=[provenance],
            source_part_count=1,
            reviewed_part_count=1,
        ),
        text,
    )


def _as_unrepaired_ws_v4(spec: ws.PinnedWorkSpec) -> ws.PinnedWorkSpec:
    raw = copy.deepcopy(spec.to_dict())
    raw["schema_version"] = ws.PINNED_WORK_SPEC_SCHEMA_VERSION_V4
    raw["source_repair_policy_version"] = (
        ws.SOURCE_REPAIR_POLICY_VERSION_V1
    )
    for part in raw["parts"]:
        part["pre_repair_plain_byte_size"] = part["plain_byte_size"]
        part["pre_repair_plain_sha256"] = part["plain_sha256"]
        part["pre_repair_word_count"] = part["word_count"]
        part["source_repairs"] = []
    payload = {key: value for key, value in raw.items() if key != "self_hash"}
    raw["self_hash"] = canonical_hash(payload)
    return ws.PinnedWorkSpec.from_dict(raw)


class _WikisourceTransport:
    def __init__(self, records: list[dict[str, object]]):
        self.records = {
            int(record["revision_id"]): copy.deepcopy(record)
            for record in records
        }
        self.calls: list[dict[str, str]] = []

    def __call__(self, params):
        request = dict(params)
        self.calls.append(request)
        if request["action"] == "query":
            record = self.records[int(request["revids"])]
            return {
                "query": {
                    "pages": [
                        {
                            "pageid": record["page_id"],
                            "title": record["title"],
                            "revisions": [
                                {
                                    "revid": record["revision_id"],
                                    "parentid": record["parent_id"],
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
        if request["action"] == "parse":
            record = self.records[int(request["oldid"])]
            return {
                "parse": {
                    "title": record["title"],
                    "pageid": record["page_id"],
                    "revid": record["revision_id"],
                    "text": record["rendered"],
                }
            }
        raise AssertionError(f"unexpected request: {request}")


@dataclass(frozen=True)
class _Harness:
    manifest: r1.R1AcquisitionManifest
    records: tuple[dict[str, object], ...]
    feb_body: bytes

    def ws_transport(self) -> _WikisourceTransport:
        return _WikisourceTransport(list(self.records))

    def feb_transport(self, calls: list[str]):
        def transport(url):
            calls.append(url)
            return FEBHTTPResponse(
                FEB_SOURCE_URL,
                200,
                FEB_CONTENT_TYPE,
                self.feb_body,
            )

        return transport


@dataclass(frozen=True)
class _V3Harness:
    manifest: r1.R1AcquisitionManifest
    records: tuple[dict[str, object], ...]
    feb_body: bytes
    reviewed_payloads: dict[str, bytes]

    def ws_transport(self) -> _WikisourceTransport:
        return _WikisourceTransport(list(self.records))

    def feb_transport(self, calls: list[str]):
        return _Harness(
            self.manifest,
            self.records,
            self.feb_body,
        ).feb_transport(calls)

    def populate_cache(self, cache: Path) -> Path:
        campaign = self.manifest.reviewed_text_campaign
        assert campaign is not None
        for work in campaign.works:
            path = cache.joinpath(*Path(work.artifact_key).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.reviewed_payloads[work.work_id])
        return cache


def _harness(*, audit_blocker: bool = False) -> _Harness:
    works: list[ws.PinnedWorkSpec] = []
    records: list[dict[str, object]] = []
    for index in range(SYNTHETIC_WIKISOURCE_COUNT):
        work, record = _ws_work(
            index,
            audit_blocker=audit_blocker and index == 1,
        )
        works.append(work)
        records.append(record)
    campaign = WikisourceCampaignSpec.build(works)
    feb_body = _feb_html()
    feb = PinnedFEBWorkSpec.build(
        work_id=r1.R1_FEB_WORK_ID,
        response_body=feb_body,
    )
    manifest = r1.R1AcquisitionManifest.build(
        wikisource_campaign=campaign,
        wikisource_discovery_candidate_sha256="d" * 64,
        source_curation_receipt_sha256="e" * 64,
        feb_work_spec=feb,
        included_work_ids=tuple(
            sorted((*campaign.work_ids, feb.work_id))
        ),
        collection_umbrella_evidence_sha256="a" * 64,
        authorship_mismatch_evidence_sha256="b" * 64,
        authorship_mismatch_receipt_sha256="c" * 64,
    )
    return _Harness(manifest, tuple(records), feb_body)


def _v3_harness(
    *,
    reviewed_payload_override: bytes | None = None,
) -> _V3Harness:
    base = _harness()
    reviewed_rows = [
        _reviewed_work(
            index,
            payload=(
                reviewed_payload_override
                if index == 0 and reviewed_payload_override is not None
                else None
            ),
        )
        for index in range(2)
    ]
    reviewed_campaign = ReviewedTextCampaignSpec.build(
        [row[0] for row in reviewed_rows]
    )
    reviewed_payloads = {
        spec.work_id: payload for spec, payload in reviewed_rows
    }
    included = tuple(
        sorted(
            (
                *base.manifest.wikisource_campaign.work_ids,
                base.manifest.feb_work_spec.work_id,
                *reviewed_campaign.work_ids,
            )
        )
    )
    manifest = r1.R1AcquisitionManifest.build(
        wikisource_campaign=base.manifest.wikisource_campaign,
        wikisource_discovery_candidate_sha256="d" * 64,
        source_curation_receipt_sha256="e" * 64,
        feb_work_spec=base.manifest.feb_work_spec,
        reviewed_text_campaign=reviewed_campaign,
        included_work_ids=included,
        collection_umbrella_evidence_sha256="a" * 64,
        authorship_mismatch_evidence_sha256="b" * 64,
        authorship_mismatch_receipt_sha256="c" * 64,
        source_quality_rejected_evidence_sha256="f" * 64,
        source_quality_rejected_receipt_sha256="9" * 64,
    )
    return _V3Harness(
        manifest,
        base.records,
        base.feb_body,
        reviewed_payloads,
    )


@pytest.fixture(scope="module")
def harness() -> _Harness:
    return _harness()


def test_manifest_binds_exact_hybrid_inventory_and_exclusions(harness):
    manifest = harness.manifest
    rebuilt = r1.R1AcquisitionManifest.build(
        wikisource_campaign=manifest.wikisource_campaign,
        wikisource_discovery_candidate_sha256="d" * 64,
        source_curation_receipt_sha256="e" * 64,
        feb_work_spec=manifest.feb_work_spec,
        included_work_ids=manifest.included_work_ids,
        collection_umbrella_evidence_sha256="a" * 64,
        authorship_mismatch_evidence_sha256="b" * 64,
        authorship_mismatch_receipt_sha256="c" * 64,
    )

    assert rebuilt == manifest
    assert len(manifest.wikisource_campaign.works) == 3
    assert len(manifest.included_work_ids) == 4
    assert manifest.included_work_ids == tuple(
        sorted(manifest.included_work_ids)
    )
    assert tuple(row.work_id for row in manifest.exclusions) == (
        r1.R1_AUTHORSHIP_MISMATCH_WORK_ID,
        r1.R1_COLLECTION_UMBRELLA_WORK_ID,
    )
    assert manifest.exclusions[0].receipt_sha256 == "c" * 64
    assert manifest.exclusions[1].receipt_sha256 is None
    encoded = json.dumps(manifest.to_dict(), ensure_ascii=False)
    assert "owner_id" not in encoded
    assert "owner_role" not in encoded
    assert manifest.wikisource_discovery_candidate_sha256 == "d" * 64
    assert manifest.source_curation_receipt_sha256 == "e" * 64
    assert manifest.generation_id == rebuilt.generation_id

    changed_selection = r1.R1AcquisitionManifest.build(
        wikisource_campaign=manifest.wikisource_campaign,
        wikisource_discovery_candidate_sha256="f" * 64,
        source_curation_receipt_sha256="e" * 64,
        feb_work_spec=manifest.feb_work_spec,
        included_work_ids=manifest.included_work_ids,
        collection_umbrella_evidence_sha256="a" * 64,
        authorship_mismatch_evidence_sha256="b" * 64,
        authorship_mismatch_receipt_sha256="c" * 64,
    )
    assert changed_selection.generation_id != manifest.generation_id


def test_manifest_rejects_inventory_mismatch_provider_overlap_and_extra_keys(
    harness,
):
    with pytest.raises(r1.R1AcquisitionError, match="differ from embedded"):
        r1.R1AcquisitionManifest.build(
            wikisource_campaign=harness.manifest.wikisource_campaign,
            wikisource_discovery_candidate_sha256="d" * 64,
            source_curation_receipt_sha256="e" * 64,
            feb_work_spec=harness.manifest.feb_work_spec,
            included_work_ids=harness.manifest.included_work_ids[:-1],
            collection_umbrella_evidence_sha256="a" * 64,
            authorship_mismatch_evidence_sha256="b" * 64,
            authorship_mismatch_receipt_sha256="c" * 64,
        )

    overlap, _ = _ws_work(
        999,
        work_id=r1.R1_FEB_WORK_ID,
    )
    overlapping_campaign = WikisourceCampaignSpec.build(
        [
            *harness.manifest.wikisource_campaign.works[:-1],
            overlap,
        ]
    )
    with pytest.raises(r1.R1AcquisitionError, match="overlap"):
        r1.R1AcquisitionManifest.build(
            wikisource_campaign=overlapping_campaign,
            wikisource_discovery_candidate_sha256="d" * 64,
            source_curation_receipt_sha256="e" * 64,
            feb_work_spec=harness.manifest.feb_work_spec,
            included_work_ids=tuple(
                sorted(
                    (
                        *overlapping_campaign.work_ids,
                        harness.manifest.feb_work_spec.work_id,
                    )
                )
            ),
            collection_umbrella_evidence_sha256="a" * 64,
            authorship_mismatch_evidence_sha256="b" * 64,
            authorship_mismatch_receipt_sha256="c" * 64,
        )

    raw = copy.deepcopy(harness.manifest.to_dict())
    raw["owner_id"] = "forbidden"
    with pytest.raises(r1.R1AcquisitionError, match="keys must be exact"):
        r1.R1AcquisitionManifest.from_dict(raw)


def test_v2_manifest_remains_byte_exact_and_requires_no_reviewed_cache(
    harness,
):
    payload = (
        dumps_strict(harness.manifest.to_dict(), indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")

    assert harness.manifest.schema_version == (
        r1.R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V2
    )
    assert harness.manifest.reviewed_text_campaign is None
    assert "reviewed_text_campaign" not in harness.manifest.to_dict()
    assert harness.manifest.generation_id == (
        "b3a7aff8b07795bb85ebf84d99ebd3c2"
        "e3c94a4423ae4e2dc0c68a2bdf826115"
    )
    assert harness.manifest.self_hash == (
        "1eee53ce520376efeaa75db5f2b306f63"
        "65d8acff22185791f585d58cf2738e5"
    )
    assert _sha(payload) == (
        "8064be6ffe2537fa9e21b49867683965e"
        "19860f02efd002f882ac0c500b98b72"
    )


def test_v3_manifest_is_exact_disjoint_three_provider_union():
    harness = _v3_harness()
    manifest = harness.manifest
    reviewed = manifest.reviewed_text_campaign

    assert manifest.schema_version == (
        r1.R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V3
    )
    assert reviewed is not None
    assert manifest.included_work_ids == tuple(
        sorted(
            (
                *manifest.wikisource_campaign.work_ids,
                manifest.feb_work_spec.work_id,
                *reviewed.work_ids,
            )
        )
    )
    assert tuple(row.work_id for row in manifest.exclusions) == (
        r1.R1_AUTHORSHIP_MISMATCH_WORK_ID,
        r1.R1_SOURCE_QUALITY_REJECTED_WORK_ID,
        r1.R1_COLLECTION_UMBRELLA_WORK_ID,
    )
    source_quality = manifest.exclusions[1]
    assert source_quality.reason_code == "source_quality_rejected"
    assert source_quality.evidence_sha256 == "f" * 64
    assert source_quality.receipt_sha256 == "9" * 64
    assert manifest.to_dict()["reviewed_text_campaign"] == reviewed.to_dict()


def test_v3_manifest_rejects_provider_work_id_overlap():
    base = _harness()
    duplicate, _ = _reviewed_work(
        9,
        work_id=base.manifest.wikisource_campaign.work_ids[0],
    )
    reviewed = ReviewedTextCampaignSpec.build([duplicate])
    included = tuple(
        sorted(
            set(
                (
                    *base.manifest.wikisource_campaign.work_ids,
                    base.manifest.feb_work_spec.work_id,
                    *reviewed.work_ids,
                )
            )
        )
    )

    with pytest.raises(r1.R1AcquisitionError, match="overlap"):
        r1.R1AcquisitionManifest.build(
            wikisource_campaign=base.manifest.wikisource_campaign,
            wikisource_discovery_candidate_sha256="d" * 64,
            source_curation_receipt_sha256="e" * 64,
            feb_work_spec=base.manifest.feb_work_spec,
            reviewed_text_campaign=reviewed,
            included_work_ids=included,
            collection_umbrella_evidence_sha256="a" * 64,
            authorship_mismatch_evidence_sha256="b" * 64,
            authorship_mismatch_receipt_sha256="c" * 64,
            source_quality_rejected_evidence_sha256="f" * 64,
            source_quality_rejected_receipt_sha256="9" * 64,
        )


def test_wikisource_v4_is_accepted_by_manifest_v3_but_not_v2():
    base = _harness()
    upgraded = _as_unrepaired_ws_v4(
        base.manifest.wikisource_campaign.works[0]
    )
    campaign = WikisourceCampaignSpec.build(
        [
            upgraded,
            *base.manifest.wikisource_campaign.works[1:],
        ]
    )
    reviewed_work, _ = _reviewed_work(8)
    reviewed = ReviewedTextCampaignSpec.build([reviewed_work])
    v3_included = tuple(
        sorted(
            (
                *campaign.work_ids,
                base.manifest.feb_work_spec.work_id,
                *reviewed.work_ids,
            )
        )
    )

    accepted = r1.R1AcquisitionManifest.build(
        wikisource_campaign=campaign,
        wikisource_discovery_candidate_sha256="d" * 64,
        source_curation_receipt_sha256="e" * 64,
        feb_work_spec=base.manifest.feb_work_spec,
        reviewed_text_campaign=reviewed,
        included_work_ids=v3_included,
        collection_umbrella_evidence_sha256="a" * 64,
        authorship_mismatch_evidence_sha256="b" * 64,
        authorship_mismatch_receipt_sha256="c" * 64,
        source_quality_rejected_evidence_sha256="f" * 64,
        source_quality_rejected_receipt_sha256="9" * 64,
    )
    assert accepted.schema_version == (
        r1.R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V3
    )

    with pytest.raises(r1.R1AcquisitionError, match="unsupported"):
        r1.R1AcquisitionManifest.build(
            wikisource_campaign=campaign,
            wikisource_discovery_candidate_sha256="d" * 64,
            source_curation_receipt_sha256="e" * 64,
            feb_work_spec=base.manifest.feb_work_spec,
            included_work_ids=tuple(
                sorted(
                    (*campaign.work_ids, base.manifest.feb_work_spec.work_id)
                )
            ),
            collection_umbrella_evidence_sha256="a" * 64,
            authorship_mismatch_evidence_sha256="b" * 64,
            authorship_mismatch_receipt_sha256="c" * 64,
        )


def test_r1_candidate_builder_binds_selected_quality_fixes_to_exact_parts():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "artifacts"
        / "build_ruaa_r1_wikisource_candidate.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "build_ruaa_r1_wikisource_candidate_quality_test",
        script,
    )
    assert module_spec is not None and module_spec.loader is not None
    builder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(builder)
    patches = builder.SELECTED_QUALITY_PART_PATCHES

    expected_boundaries = {
        ("bunin/деревня", 0, 5585755): (
            0,
            None,
            "whole_rendered_body",
        ),
        ("bunin/суходол", 0, 5588281): (
            0,
            None,
            "whole_rendered_body",
        ),
        ("dostoevsky/бесы", 23, 5625680): (
            0,
            None,
            "whole_rendered_body",
        ),
        ("furmanov/красный_десант", 0, 5604500): (
            0,
            531,
            "strip_trailing_apparatus",
        ),
        ("furmanov/мятеж", 0, 5307372): (
            0,
            6526,
            "strip_trailing_apparatus",
        ),
        ("gogol/мёртвые_души", 16, 3575878): (
            0,
            None,
            "whole_rendered_body",
        ),
        ("grin/алые_паруса", 0, 5636844): (
            20,
            None,
            "strip_leading_apparatus",
        ),
        ("korolenko/соколинец", 0, 5588120): (
            8,
            None,
            "strip_leading_apparatus",
        ),
        ("serafimovich/пески", 0, 5595521): (
            0,
            885,
            "strip_trailing_apparatus",
        ),
        ("serafimovich/степные_люди", 0, 5596256): (
            0,
            339,
            "strip_trailing_apparatus",
        ),
        ("uspensky/будка", 0, 5588776): (
            0,
            1691,
            "strip_trailing_apparatus",
        ),
    }
    assert {
        key: (
            patch["body_start_line"],
            patch["body_end_line_exclusive"],
            patch["body_disposition"],
        )
        for key, patch in patches.items()
    } == expected_boundaries

    repairs = {
        key: patch["source_repairs"]
        for key, patch in patches.items()
        if patch["source_repairs"]
    }
    assert set(repairs) == {
        ("bunin/деревня", 0, 5585755),
        ("bunin/суходол", 0, 5588281),
        ("dostoevsky/бесы", 23, 5625680),
        ("gogol/мёртвые_души", 16, 3575878),
    }
    bunin_literal = (
        "Оригинал здесь: Электронная библиотека Яблучанского."
    )
    expected_repairs = {
        ("bunin/деревня", 0, 5585755): (
            bunin_literal,
            "",
            "109eaa4d1662f9c091019b6788b508c26c5ffc047e2a50531b842a82c47f3d56",
        ),
        ("bunin/суходол", 0, 5588281): (
            bunin_literal,
            "",
            "109eaa4d1662f9c091019b6788b508c26c5ffc047e2a50531b842a82c47f3d56",
        ),
        ("dostoevsky/бесы", 23, 5625680): (
            "<III>",
            "III",
            "5699a51036ab8d8d2bd4eeb47b50218b18f3fad1aab284d2161b68a0a3b7fe36",
        ),
        ("gogol/мёртвые_души", 16, 3575878): (
            "<Myразов>",
            "<Муразов>",
            "d96b545baf5edd063376bd43d73d8923e3f1df1229060fc3292f39a7677539aa",
        ),
    }
    assert {
        key: (
            rows[0]["literal"],
            rows[0]["replacement"],
            rows[0]["occurrence_line_sha256"][0],
        )
        for key, rows in repairs.items()
    } == expected_repairs
    assert all(
        rows[0]["expected_count"] == 1
        and rows[0]["review_receipt_sha256"]
        == builder.SELECTED_QUALITY_REVIEW_RECEIPT_SHA256
        for rows in repairs.values()
    )


def test_v3_materializes_union_resumes_and_rejects_namespace_tamper(tmp_path):
    harness = _v3_harness()
    cache = harness.populate_cache(tmp_path / "reviewed-cache")
    ws_transport = harness.ws_transport()
    feb_calls: list[str] = []

    first = r1.materialize_r1_acquisition(
        harness.manifest,
        output_parent=tmp_path / "output",
        wikisource_transport=ws_transport,
        feb_transport=harness.feb_transport(feb_calls),
        reviewed_artifact_cache=cache,
    )

    reviewed = harness.manifest.reviewed_text_campaign
    assert reviewed is not None
    assert first.receipt.schema_version == (
        r1.R1_ACQUISITION_RECEIPT_SCHEMA_VERSION_V2
    )
    assert first.receipt.reviewed_text_campaign_spec_sha256 == (
        reviewed.self_hash
    )
    assert len(first.receipt.raw_inventory) == len(
        harness.manifest.included_work_ids
    )
    assert (first.root / r1.REVIEWED_TEXT_SPEC_PATH).is_file()
    assert (first.root / r1.REVIEWED_TEXT_RECEIPT_PATH).is_file()
    for work_id, payload in harness.reviewed_payloads.items():
        assert (
            first.root / "raw" / f"{work_id}.txt"
        ).read_bytes() == payload

    resumed = r1.materialize_r1_acquisition(
        harness.manifest,
        output_parent=tmp_path / "output",
        wikisource_transport=lambda _: pytest.fail("no Wikisource on resume"),
        feb_transport=lambda _: pytest.fail("no FEB on resume"),
        reviewed_artifact_cache=None,
    )
    assert resumed.resumed is True
    assert resumed.receipt == first.receipt

    reviewed_work_id = reviewed.work_ids[0]
    raw_path = first.root / "raw" / f"{reviewed_work_id}.txt"
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b"tamper\n")
    with pytest.raises(
        r1.R1AcquisitionError,
        match="reviewed-text campaign receipt is invalid",
    ):
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path / "output",
            wikisource_transport=lambda _: pytest.fail("no network"),
            feb_transport=lambda _: pytest.fail("no network"),
        )


def test_v3_rejects_tampered_artifact_cache_before_publication(tmp_path):
    harness = _v3_harness()
    cache = harness.populate_cache(tmp_path / "reviewed-cache")
    reviewed = harness.manifest.reviewed_text_campaign
    assert reviewed is not None
    first_work = reviewed.works[0]
    cache_entry = cache.joinpath(*Path(first_work.artifact_key).parts)
    cache_entry.write_bytes(b"x" * first_work.byte_size)

    with pytest.raises(
        r1.R1AcquisitionError,
        match="provider materialization was rejected",
    ):
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path / "output",
            wikisource_transport=harness.ws_transport(),
            feb_transport=harness.feb_transport([]),
            reviewed_artifact_cache=cache,
        )
    assert not (
        tmp_path / "output" / harness.manifest.generation_id
    ).exists()


def test_v3_union_audit_rejects_cross_provider_content_overlap(tmp_path):
    duplicate_wikisource_payload = (_plain(0) + "\n").encode("utf-8")
    harness = _v3_harness(
        reviewed_payload_override=duplicate_wikisource_payload
    )
    cache = harness.populate_cache(tmp_path / "reviewed-cache")

    with pytest.raises(r1.R1AcquisitionAuditError) as raised:
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path / "output",
            wikisource_transport=harness.ws_transport(),
            feb_transport=harness.feb_transport([]),
            reviewed_artifact_cache=cache,
        )

    assert raised.value.report.status == "blocked"
    assert raised.value.report.to_dict()["cross_work_overlaps"]
    assert not (
        tmp_path / "output" / harness.manifest.generation_id
    ).exists()

    with pytest.raises(r1.R1AcquisitionAuditError) as resumed:
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path / "output",
            wikisource_transport=lambda _: pytest.fail(
                "no Wikisource after blocked audit"
            ),
            feb_transport=lambda _: pytest.fail(
                "no FEB after blocked audit"
            ),
            reviewed_artifact_cache=None,
        )
    assert resumed.value.report == raised.value.report


def test_materialization_resume_and_all_tamper_modes_are_fail_closed(
    tmp_path,
    harness,
):
    ws_transport = harness.ws_transport()
    feb_calls: list[str] = []
    first = r1.materialize_r1_acquisition(
        harness.manifest,
        output_parent=tmp_path,
        wikisource_transport=ws_transport,
        feb_transport=harness.feb_transport(feb_calls),
    )

    assert first.resumed is False
    assert first.receipt.schema_version == (
        r1.R1_ACQUISITION_RECEIPT_SCHEMA_VERSION_V1
    )
    assert "reviewed_text_campaign_spec_sha256" not in (
        first.receipt.to_dict()
    )
    assert first.audit_report.status == "passed"
    assert len(first.receipt.raw_inventory) == 4
    assert len(ws_transport.calls) == 6
    assert feb_calls == [FEB_SOURCE_URL]

    resumed = r1.materialize_r1_acquisition(
        harness.manifest,
        output_parent=tmp_path,
        wikisource_transport=lambda _: pytest.fail(
            "hybrid resume must not use Wikisource"
        ),
        feb_transport=lambda _: pytest.fail(
            "hybrid resume must not use FEB"
        ),
    )
    assert resumed.resumed is True
    assert resumed.receipt == first.receipt

    raw_path = (
        first.root
        / "raw"
        / "author001"
        / "work001.txt"
    )
    original_raw = raw_path.read_bytes()
    raw_path.write_bytes(original_raw + b"tamper\n")
    with pytest.raises(r1.R1AcquisitionError, match="receipt is invalid"):
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path,
            wikisource_transport=lambda _: pytest.fail("no network"),
            feb_transport=lambda _: pytest.fail("no network"),
        )
    raw_path.write_bytes(original_raw)

    extra = first.root / "extra.txt"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(r1.R1AcquisitionError, match="missing or extra"):
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path,
            wikisource_transport=lambda _: pytest.fail("no network"),
            feb_transport=lambda _: pytest.fail("no network"),
        )
    extra.unlink()

    audit_path = first.root / r1.AUDIT_REPORT_NAME
    original_audit = audit_path.read_bytes()
    audit_path.write_bytes(original_audit + b" ")
    with pytest.raises(r1.R1AcquisitionError, match="noncanonical"):
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path,
            wikisource_transport=lambda _: pytest.fail("no network"),
            feb_transport=lambda _: pytest.fail("no network"),
        )
    audit_path.write_bytes(original_audit)

    receipt_path = (
        first.root
        / "providers"
        / "wikisource"
        / "work-receipts"
        / "author001"
        / "work001.json"
    )
    original_receipt = receipt_path.read_bytes()
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(r1.R1AcquisitionError, match="receipt is invalid"):
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path,
            wikisource_transport=lambda _: pytest.fail("no network"),
            feb_transport=lambda _: pytest.fail("no network"),
        )
    receipt_path.write_bytes(original_receipt)

    link = first.root / "forbidden-link"
    link.symlink_to(raw_path)
    with pytest.raises(r1.R1AcquisitionError, match="symlink rejected"):
        r1.materialize_r1_acquisition(
            harness.manifest,
            output_parent=tmp_path,
            wikisource_transport=lambda _: pytest.fail("no network"),
            feb_transport=lambda _: pytest.fail("no network"),
        )
    link.unlink()


def test_audit_blocker_prevents_publish_and_provider_resume_is_network_free(
    tmp_path,
):
    blocked = _harness(audit_blocker=True)
    ws_transport = blocked.ws_transport()
    feb_calls: list[str] = []

    with pytest.raises(r1.R1AcquisitionAuditError) as raised:
        r1.materialize_r1_acquisition(
            blocked.manifest,
            output_parent=tmp_path,
            wikisource_transport=ws_transport,
            feb_transport=blocked.feb_transport(feb_calls),
        )
    error = raised.value
    assert error.report.status == "blocked"
    assert error.report_path.is_file()
    assert not (tmp_path / blocked.manifest.generation_id).exists()
    assert len(ws_transport.calls) == 6
    assert feb_calls == [FEB_SOURCE_URL]

    with pytest.raises(r1.R1AcquisitionAuditError) as resumed:
        r1.materialize_r1_acquisition(
            blocked.manifest,
            output_parent=tmp_path,
            wikisource_transport=lambda _: pytest.fail(
                "provider cache resume must be network-free"
            ),
            feb_transport=lambda _: pytest.fail(
                "provider cache resume must be network-free"
            ),
        )
    assert resumed.value.report_path == error.report_path
    assert resumed.value.report.self_hash == error.report.self_hash


def test_manifest_duplicate_keys_are_rejected(harness):
    encoded = json.dumps(harness.manifest.to_dict(), ensure_ascii=False)
    duplicated = encoded.replace(
        '"acquisition_kind": '
        '"bounded_exploratory_source_acquisition_only"',
        '"acquisition_kind": '
        '"bounded_exploratory_source_acquisition_only", '
        '"acquisition_kind": "other"',
        1,
    )
    with pytest.raises(r1.R1AcquisitionError, match="duplicate object key"):
        r1.loads_r1_acquisition_manifest(duplicated)


def test_cli_requires_manifest_and_uses_only_fixed_ignored_namespace(
    tmp_path,
    harness,
    monkeypatch,
):
    cli = _load_acquisition_cli(
        "acquire_ruaa_r1_corpus_vnext_test",
    )
    fake_root = tmp_path / "repository"
    fake_output = (
        fake_root
        / "docs"
        / "exploratory"
        / "lobo_vnext"
        / "corpora"
        / "ruaa_r1_hybrid"
    )
    manifest_path = tmp_path / "manifest.json"
    dump_strict(
        harness.manifest.to_dict(),
        manifest_path,
        sort_keys=True,
    )
    seen = {}

    def materialize(manifest, **kwargs):
        seen["manifest"] = manifest
        seen.update(kwargs)
        return SimpleNamespace(
            root=fake_output / manifest.generation_id,
            receipt=SimpleNamespace(self_hash="d" * 64),
            audit_report=SimpleNamespace(self_hash="e" * 64),
            resumed=False,
        )

    monkeypatch.setattr(cli, "ROOT", fake_root)
    monkeypatch.setattr(cli, "OUTPUT_PARENT", fake_output)
    monkeypatch.setattr(
        cli,
        "HTTPJSONTransport",
        lambda **kwargs: ("wikisource", kwargs),
    )
    monkeypatch.setattr(
        cli,
        "FEBHTTPTransport",
        lambda **kwargs: ("feb", kwargs),
    )
    monkeypatch.setattr(cli, "materialize_r1_acquisition", materialize)

    result = cli.run(
        [
            "--manifest",
            str(manifest_path),
            "--wikisource-user-agent",
            "synthetic-wikisource/1",
            "--feb-user-agent",
            "synthetic-feb/1",
        ]
    )

    assert seen["manifest"] == harness.manifest
    assert seen["output_parent"] == fake_output
    assert seen["reviewed_artifact_cache"] is None
    assert result["status"] == (
        "exploratory_ruaa_r1_corpus_materialized_no_fit"
    )
    assert result["work_count"] == 4
    assert result["namespace_relative_path"].endswith(
        harness.manifest.generation_id
    )
    assert result["fit_performed"] is False
    assert result["confirmatory_authorized"] is False
    assert result["public_output_authorized"] is False


def test_cli_requires_safe_reviewed_cache_only_for_manifest_v3(
    tmp_path,
    monkeypatch,
):
    harness = _v3_harness()
    cli = _load_acquisition_cli(
        "acquire_ruaa_r1_corpus_vnext_v3_test",
    )
    fake_root = tmp_path / "repository"
    fake_output = (
        fake_root
        / "docs"
        / "exploratory"
        / "lobo_vnext"
        / "corpora"
        / "ruaa_r1_hybrid"
    )
    manifest_path = tmp_path / "manifest-v3.json"
    dump_strict(
        harness.manifest.to_dict(),
        manifest_path,
        sort_keys=True,
    )
    seen = {}

    def materialize(manifest, **kwargs):
        seen["manifest"] = manifest
        seen.update(kwargs)
        return SimpleNamespace(
            root=fake_output / manifest.generation_id,
            receipt=SimpleNamespace(self_hash="d" * 64),
            audit_report=SimpleNamespace(self_hash="e" * 64),
            resumed=False,
        )

    monkeypatch.setattr(cli, "ROOT", fake_root)
    monkeypatch.setattr(cli, "OUTPUT_PARENT", fake_output)
    monkeypatch.setattr(
        cli,
        "HTTPJSONTransport",
        lambda **kwargs: ("wikisource", kwargs),
    )
    monkeypatch.setattr(
        cli,
        "FEBHTTPTransport",
        lambda **kwargs: ("feb", kwargs),
    )
    monkeypatch.setattr(cli, "materialize_r1_acquisition", materialize)
    common = [
        "--manifest",
        str(manifest_path),
        "--wikisource-user-agent",
        "synthetic-wikisource/1",
        "--feb-user-agent",
        "synthetic-feb/1",
    ]

    with pytest.raises(
        cli.R1AcquisitionCLIError,
        match="requires --reviewed-artifact-cache",
    ):
        cli.run(common)

    real_cache = tmp_path / "real-reviewed-cache"
    real_cache.mkdir()
    linked_cache = tmp_path / "linked-reviewed-cache"
    linked_cache.symlink_to(real_cache, target_is_directory=True)
    with pytest.raises(
        cli.R1AcquisitionCLIError,
        match="symlink components",
    ):
        cli.run(
            [
                *common,
                "--reviewed-artifact-cache",
                str(linked_cache),
            ]
        )

    result = cli.run(
        [
            *common,
            "--reviewed-artifact-cache",
            str(real_cache),
        ]
    )
    assert seen["manifest"] == harness.manifest
    assert seen["reviewed_artifact_cache"] == real_cache.resolve()
    assert result["work_count"] == len(harness.manifest.included_work_ids)


def test_replay_dispatch_and_closed_file_transports(
    tmp_path,
    monkeypatch,
):
    cli = _load_acquisition_cli("acquire_ruaa_r1_corpus_replay_test")
    seen = []
    monkeypatch.setattr(
        cli,
        "_run_live",
        lambda argv: seen.append(("live", argv)) or {"mode": "live"},
    )
    monkeypatch.setattr(
        cli,
        "_run_replay",
        lambda argv: seen.append(("replay", argv)) or {"mode": "replay"},
    )
    assert cli.run(["--manifest", "x"]) == {"mode": "live"}
    assert cli.run(["replay", "--artifact-cache", "x"]) == {
        "mode": "replay"
    }
    assert seen == [
        ("live", ["--manifest", "x"]),
        ("replay", ["--artifact-cache", "x"]),
    ]

    cache = tmp_path / "cache"
    cache.mkdir()
    response = {"query": {"pages": []}}
    monkeypatch.setattr(
        cli,
        "_read_cached_response",
        lambda *_args, **_kwargs: response,
    )
    transport = cli._PinnedCacheTransport(cache)
    request = cli._request_parameters("query", 101)
    assert transport(request) is response
    assert transport.call_count == 1
    with pytest.raises(
        cli.R1AcquisitionCLIError,
        match="differs from the pinned contract",
    ):
        transport({**request, "prop": "ids"})

    feb_payload = b"captured FEB response"
    feb_path = tmp_path / "feb-response.html"
    feb_path.write_bytes(feb_payload)
    monkeypatch.setattr(
        cli,
        "EXPECTED_FEB_RESPONSE_FILE_SHA256",
        _sha(feb_payload),
    )
    feb_transport = cli._FEBFileTransport(feb_path)
    observed = feb_transport(FEB_SOURCE_URL)
    assert observed.body == feb_payload
    assert observed.content_type == FEB_CONTENT_TYPE
    assert feb_transport.call_count == 1
    with pytest.raises(
        cli.R1AcquisitionCLIError,
        match="rejects URL",
    ):
        feb_transport("https://example.invalid/")
