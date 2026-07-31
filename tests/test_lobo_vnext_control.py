from __future__ import annotations

import ast
import hashlib
import inspect
import os
import shutil
from pathlib import Path

import pytest

from stylo.config import load_config
from stylo.corpus_tools import ruaa_r1_acquisition as r1
from stylo.corpus_tools import wikisource_vnext as ws
from stylo.corpus_tools.feb_vnext import (
    PinnedFEBWorkSpec,
    extract_feb_main_narrative,
)
from stylo.corpus_tools.reviewed_text_vnext import (
    ReviewedTextArtifactRef,
    ReviewedTextCampaignSpec,
    ReviewedTextWorkSpec,
)
from stylo.corpus_tools.text_quality_vnext import audit_corpus_texts
from stylo.corpus_tools.wikisource_campaign import WikisourceCampaignSpec
from stylo.domain.lobo_vnext import VNextContractError, canonical_sha256
from stylo.domain.lobo_vnext_packet import (
    CanonicalRowEntry,
    PacketFileEntry,
    R1PacketManifest,
    R1_PACKET_MANIFEST_SCHEMA_VERSION,
)
from stylo.domain.lobo_vnext_real import REQUIRED_RECEIPT_KINDS
from stylo.eval import lobo_vnext_control as control
from stylo.eval import lobo_vnext_prepare as prep
from stylo.eval import lobo_vnext_receipts as receipts
from stylo.jsonio import (
    canonical_hash,
    dump_strict,
    dumps_strict,
    load_strict,
)
from stylo.nlp import ResolvedNLPIdentity


def _sha(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _literal_payload(index: int) -> bytes:
    return (
        " ".join(
            f"литературный{index}уникальныйтокен{word}"
            for word in range(220)
        )
        + "\n"
    ).encode("utf-8")


def _wikisource_work(
    *,
    work_id: str,
    index: int,
) -> tuple[ws.PinnedWorkSpec, bytes]:
    title = f"Синтетическое произведение {index}"
    revision = 100_000 + index
    page_id = 200_000 + index
    wikitext = f"Синтетический викитекст {index}"
    rendered = (
        '<div class="mw-parser-output"><p>'
        + _literal_payload(index).decode("utf-8").strip()
        + "</p></div>"
    )
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
            "wikitext_sha256": _sha(wikitext),
            "rendered_html_sha256": _sha(rendered),
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
    return (
        ws.PinnedWorkSpec.from_dict(
            {**payload, "self_hash": canonical_hash(payload)}
        ),
        output,
    )


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
        "<p>"
        + " ".join(f"пугачевпредисловие{word}" for word in range(45))
        + "</p>",
    ]
    for chapter, name in enumerate(names):
        parts.extend(
            (
                f'<h4 id="ГЛАВА_{name}"></h4>',
                f"<p>ГЛАВА {name}</p>",
                "<p>"
                + " ".join(
                    f"пугачевглава{chapter}слово{word}"
                    for word in range(45)
                )
                + "</p>",
            )
        )
    parts.extend(
        (
            '<h4 id="ПРИМЕЧАНИЯ"></h4>',
            "<p>Редакторские примечания вне корпуса</p>",
            "</div></body></html>",
        )
    )
    return "".join(parts).encode("windows-1251")


def _reviewed_work(
    *,
    work_id: str,
    index: int,
    payload: bytes,
) -> ReviewedTextWorkSpec:
    builder = ReviewedTextArtifactRef.build(
        logical_name="synthetic-reviewed-builder.py",
        payload=f"builder {index}".encode("utf-8"),
    )
    provenance = ReviewedTextArtifactRef.build(
        logical_name="synthetic-review-receipt.json",
        payload=f"review receipt {index}".encode("utf-8"),
    )
    return ReviewedTextWorkSpec.build(
        work_id=work_id,
        text_payload=payload,
        builder_artifacts=[builder],
        provenance_artifacts=[provenance],
        source_part_count=1,
        reviewed_part_count=1,
    )


def _work_ids() -> tuple[str, ...]:
    authors = ("pushkin", *(f"author_{index:02d}" for index in range(21)))
    rows: list[str] = []
    for author in authors:
        count = 7 if author == "author_00" else 6
        rows.extend(f"{author}/work_{index:02d}" for index in range(count))
    rows.append(r1.R1_FEB_WORK_ID)
    result = tuple(sorted(rows))
    assert len(result) == 134
    assert len({row.split("/", 1)[0] for row in result}) == 22
    assert not set(result) & set(r1.R1_EXCLUDED_WORK_IDS_V3)
    return result


def _acquisition_fixture(root: Path) -> r1.MaterializedR1Acquisition:
    acquisition_root = root / "acquisition"
    acquisition_root.mkdir(parents=True)
    work_ids = _work_ids()
    raw_payloads = {
        work_id: _literal_payload(index)
        for index, work_id in enumerate(work_ids)
    }

    wikisource_id = "pushkin/work_00"
    wikisource, wikisource_payload = _wikisource_work(
        work_id=wikisource_id,
        index=10_000,
    )
    raw_payloads[wikisource_id] = wikisource_payload
    wikisource_campaign = WikisourceCampaignSpec.build([wikisource])

    feb_body = _feb_html()
    feb = PinnedFEBWorkSpec.build(
        work_id=r1.R1_FEB_WORK_ID,
        response_body=feb_body,
    )
    raw_payloads[r1.R1_FEB_WORK_ID] = (
        extract_feb_main_narrative(feb_body) + "\n"
    ).encode("utf-8")

    reviewed = ReviewedTextCampaignSpec.build(
        [
            _reviewed_work(
                work_id=work_id,
                index=index,
                payload=raw_payloads[work_id],
            )
            for index, work_id in enumerate(work_ids)
            if work_id not in {wikisource_id, r1.R1_FEB_WORK_ID}
        ]
    )
    manifest = r1.R1AcquisitionManifest.build(
        wikisource_campaign=wikisource_campaign,
        wikisource_discovery_candidate_sha256="d" * 64,
        source_curation_receipt_sha256="e" * 64,
        feb_work_spec=feb,
        reviewed_text_campaign=reviewed,
        included_work_ids=work_ids,
        collection_umbrella_evidence_sha256="a" * 64,
        authorship_mismatch_evidence_sha256="b" * 64,
        authorship_mismatch_receipt_sha256="c" * 64,
        source_quality_rejected_evidence_sha256="f" * 64,
        source_quality_rejected_receipt_sha256="9" * 64,
    )
    audit = audit_corpus_texts(
        raw_payloads,
        expected_work_ids=work_ids,
    )
    assert audit.status == "passed"

    raw_rows = tuple(
        r1.R1RawInventoryRow.build(work_id, raw_payloads[work_id])
        for work_id in work_ids
    )
    receipt_payload: dict[str, object] = {
        "schema_version": r1.R1_ACQUISITION_RECEIPT_SCHEMA_VERSION_V2,
        "acquisition_kind": r1.R1_ACQUISITION_KIND,
        "manifest_sha256": manifest.self_hash,
        "generation_id": manifest.generation_id,
        "wikisource_campaign_spec_sha256": (
            manifest.wikisource_campaign.self_hash
        ),
        "wikisource_campaign_receipt_sha256": "1" * 64,
        "feb_work_spec_sha256": manifest.feb_work_spec.self_hash,
        "feb_work_receipt_sha256": "2" * 64,
        "text_quality_audit_sha256": audit.self_hash,
        "reviewed_text_campaign_spec_sha256": reviewed.self_hash,
        "reviewed_text_campaign_receipt_sha256": "3" * 64,
        "included_work_ids": list(work_ids),
        "raw_inventory": [row.to_dict() for row in raw_rows],
        "work_count": len(work_ids),
        "fit_performed": False,
        "confirmatory_authorized": False,
        "public_output_authorized": False,
    }
    receipt = r1.R1AcquisitionReceipt.from_dict(
        {
            **receipt_payload,
            "self_hash": canonical_hash(receipt_payload),
        }
    )

    for work_id, payload in raw_payloads.items():
        path = acquisition_root / f"raw/{work_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    dump_strict(
        manifest.to_dict(),
        acquisition_root / r1.MANIFEST_NAME,
        sort_keys=True,
        trailing_newline=True,
    )
    dump_strict(
        receipt.to_dict(),
        acquisition_root / r1.ACQUISITION_RECEIPT_NAME,
        sort_keys=True,
        trailing_newline=True,
    )
    dump_strict(
        audit.to_dict(),
        acquisition_root / r1.AUDIT_REPORT_NAME,
        sort_keys=True,
        trailing_newline=True,
    )
    return r1.MaterializedR1Acquisition(
        acquisition_root,
        manifest,
        receipt,
        audit,
        True,
    )


def _fake_canonical_rows(
    *,
    cfg,
    raw_root: Path,
    packet_root: Path,
    works,
) -> tuple[CanonicalRowEntry, ...]:
    del cfg
    rows = []
    for work in works:
        source_relative = work.raw_paths[0]
        source_payload = (raw_root / source_relative).read_bytes()
        text = f"canonical representation for {work.work_id}"
        payload = text.encode("utf-8")
        relative = f"canonical_rows/{work.work_id}/000000.txt"
        output = packet_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        rows.append(
            CanonicalRowEntry.from_dict(
                {
                    "row_id": f"{work.work_id}#000000",
                    "relative_path": relative,
                    "work_id": work.work_id,
                    "author_id": work.author_id,
                    "ordinal": 0,
                    "source_relative_path": source_relative,
                    "source_raw_sha256": _sha(source_payload),
                    "canonical_byte_size": len(payload),
                    "canonical_sha256": _sha(payload),
                    "word_count": len(text.split()),
                }
            )
        )
    return tuple(rows)


def _pin_verified_r1_ner(monkeypatch) -> None:
    material = {
        "requested_model": "ru_core_news_lg",
        "resolved_model": "ru_core_news_lg",
        "fallback_used": False,
        "package_version": "3.8.0",
        "package_payload_sha256": "a" * 64,
        "spacy_version": prep.spacy.__version__,
        "disabled_pipes": [
            "attribute_ruler",
            "lemmatizer",
            "morphologizer",
            "parser",
            "sentencizer",
            "tagger",
            "textcat",
        ],
        "active_pipes": ["tok2vec", "ner"],
        "max_length": 5_000_000,
    }
    identity = ResolvedNLPIdentity(
        requested_model=material["requested_model"],
        resolved_model=material["resolved_model"],
        fallback_used=material["fallback_used"],
        package_version=material["package_version"],
        package_payload_sha256=material["package_payload_sha256"],
        spacy_version=material["spacy_version"],
        disabled_pipes=tuple(material["disabled_pipes"]),
        active_pipes=tuple(material["active_pipes"]),
        max_length=material["max_length"],
        identity_sha256=canonical_sha256(material),
    )
    pipeline = object()

    def load(model: str, fallback: str):
        assert (model, fallback) == ("ru_core_news_lg", "ru_core_news_md")
        return pipeline

    def resolve(loaded) -> ResolvedNLPIdentity:
        assert loaded is pipeline
        return identity

    monkeypatch.setattr(prep, "load_ner", load)
    monkeypatch.setattr(prep, "resolved_nlp_identity", resolve)


@pytest.fixture(scope="module")
def prepared_packet(tmp_path_factory):
    root = tmp_path_factory.mktemp("control-plane")
    acquisition = _acquisition_fixture(root)
    works, raw_inventory = prep._acquisition_catalog(acquisition)
    audit_path = acquisition.root / r1.AUDIT_REPORT_NAME
    identities = {
        "R1_ACQUISITION_GENERATION_ID": acquisition.manifest.generation_id,
        "R1_ACQUISITION_MANIFEST_SELF_HASH": (
            acquisition.manifest.self_hash
        ),
        "R1_ACQUISITION_RECEIPT_SELF_HASH": (
            acquisition.receipt.self_hash
        ),
        "R1_SELECTED_AUDIT_FILE_SHA256": _sha(audit_path.read_bytes()),
        "R1_SELECTED_AUDIT_SELF_HASH": acquisition.audit_report.self_hash,
        "R1_RAW_INVENTORY_DIGEST": canonical_sha256(
            [row.to_dict() for row in raw_inventory]
        ),
        "R1_WORK_IDENTITY_CATALOG_DIGEST": canonical_sha256(
            [work.to_dict() for work in works]
        ),
    }
    patch = pytest.MonkeyPatch()
    for module in (prep, control):
        for name, value in identities.items():
            patch.setattr(module, name, value)
    patch.setattr(
        prep,
        "load_materialized_r1_acquisition",
        lambda acquisition_root: acquisition,
    )
    patch.setattr(prep, "_canonical_rows", _fake_canonical_rows)
    _pin_verified_r1_ner(patch)
    try:
        packet = prep.prepare_r1_packet(
            acquisition_root=acquisition.root,
            output_parent=(
                root / "exploratory" / "lobo_vnext" / "packets"
            ),
            cfg=load_config("configs/default.yaml"),
        )
        yield packet
    finally:
        patch.undo()


def _copy_packet(packet, destination: Path) -> Path:
    copied = destination / packet.root.name
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(packet.root, copied)
    return copied


def _reseal_packet(root: Path) -> Path:
    existing = R1PacketManifest.from_dict(load_strict(root / "packet.json"))
    files = tuple(
        PacketFileEntry(
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            _sha(path.read_bytes()),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "packet.json"
    )
    rebuilt = R1PacketManifest.build(
        acquisition_binding=existing.acquisition_binding,
        corpus_generation_material=existing.corpus_generation_material,
        content_policy_spec_sha256=existing.content_policy_spec_sha256,
        candidate_inventory_sha256=existing.candidate_inventory_sha256,
        corpus_manifest_sha256=existing.corpus_manifest_sha256,
        content_component_manifest_sha256=(
            existing.content_component_manifest_sha256
        ),
        fold_manifest_sha256=existing.fold_manifest_sha256,
        primary_model_spec_sha256=existing.primary_model_spec_sha256,
        baseline_model_spec_sha256=existing.baseline_model_spec_sha256,
        inference_spec_sha256=existing.inference_spec_sha256,
        primary_inner_cv_plan_sha256=(
            existing.primary_inner_cv_plan_sha256
        ),
        baseline_inner_cv_plan_sha256=(
            existing.baseline_inner_cv_plan_sha256
        ),
        model_role_manifest_sha256=existing.model_role_manifest_sha256,
        campaign_manifest_sha256=existing.campaign_manifest_sha256,
        representation_receipt_sha256=(
            existing.representation_receipt_sha256
        ),
        files=files,
    )
    dump_strict(rebuilt.to_dict(), root / "packet.json", trailing_newline=True)
    target = root.with_name(rebuilt.packet_generation_id)
    root.rename(target)
    return target


def _observation(kind: str) -> receipts.DerivedObservation:
    return receipts.DerivedObservation(
        kind=kind,
        derivation_version=f"test.{kind}.v1",
        digest=_sha(f"{kind}:digest"),
        evidence_digest=_sha(f"{kind}:evidence"),
        observation_count=1,
    ).validate()


def _patch_live_observations(monkeypatch) -> None:
    monkeypatch.setattr(
        control,
        "derive_executable_source_observation",
        lambda repository_root: _observation("executable_sources"),
    )
    monkeypatch.setattr(
        control,
        "derive_dependency_observation",
        lambda repository_root: _observation("dependencies"),
    )
    monkeypatch.setattr(
        control,
        "derive_runtime_observation",
        lambda: _observation("runtime"),
    )
    monkeypatch.setattr(
        control,
        "derive_thread_observation",
        lambda: _observation("thread_contract"),
    )
    monkeypatch.setattr(
        control,
        "derive_config_and_adapter_observations",
        lambda **kwargs: (
            _observation("config"),
            _observation("primary_model_adapter"),
            _observation("baseline_model_adapter"),
        ),
    )


def test_control_plane_has_no_cache_factory_fit_or_prediction_reachability():
    tree = ast.parse(inspect.getsource(control))
    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            reached.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                reached.add(function.id)
            elif isinstance(function, ast.Attribute):
                reached.add(function.attr)

    assert reached.isdisjoint(
        {
            "build_representation_cache",
            "fit",
            "fit_estimator",
            "make_factory",
            "predict",
            "predict_proba",
            "run_real_lobo_vnext",
        }
    )


def test_loader_rebuilds_strict_acquisition_bound_selected_134_packet(
    prepared_packet,
):
    loaded = control.load_prepared_r1_packet(prepared_packet.root)
    work_ids = tuple(
        work.work_id for work in loaded.corpus_manifest.works
    )
    canonical_work_ids = {
        row.work_id for row in loaded.representation_receipt.rows
    }

    assert type(loaded.packet_manifest) is R1PacketManifest
    assert (
        loaded.packet_manifest.schema_version
        == R1_PACKET_MANIFEST_SCHEMA_VERSION
    )
    assert (
        loaded.packet_manifest.self_hash
        == prepared_packet.packet_manifest.self_hash
    )
    assert loaded.acquisition_binding == (
        loaded.packet_manifest.acquisition_binding
    )
    assert loaded.candidate_inventory.candidates == ()
    assert loaded.packet_manifest.selected_work_count == 134
    assert loaded.root.name == loaded.packet_manifest.packet_generation_id
    assert loaded.corpus_manifest.generation_id == (
        loaded.packet_manifest.corpus_generation_id
    )
    assert loaded.candidate_inventory.generation_id == (
        loaded.packet_manifest.corpus_generation_id
    )
    assert loaded.representation_receipt.generation_id == (
        loaded.packet_manifest.corpus_generation_id
    )
    assert loaded.packet_manifest.corpus_generation_material.self_hash == (
        loaded.packet_manifest.corpus_generation_id
    )
    assert canonical_sha256(
        loaded.packet_manifest.packet_generation_material.to_dict()
    ) == loaded.packet_manifest.packet_generation_id
    assert len(
        {
            loaded.acquisition_binding.acquisition_generation_id,
            loaded.packet_manifest.corpus_generation_id,
            loaded.packet_manifest.packet_generation_id,
        }
    ) == 3
    assert len(work_ids) == 134
    assert len(loaded.corpus_manifest.author_ids) == 22
    assert loaded.acquisition_binding.upstream_excluded_work_ids == (
        r1.R1_EXCLUDED_WORK_IDS_V3
    )
    assert not set(r1.R1_EXCLUDED_WORK_IDS_V3) & set(work_ids)
    assert not set(r1.R1_EXCLUDED_WORK_IDS_V3) & canonical_work_ids
    assert {
        work.source_id.split(":")[-2]
        for work in loaded.corpus_manifest.works
    } == {"wikisource", "feb", "reviewed-text"}
    assert control.load_prepared_r1_packet(
        prepared_packet.root
    ).corpus_manifest.works == loaded.corpus_manifest.works


@pytest.mark.parametrize(
    "relative",
    [
        "acquisition/acquisition-manifest.json",
        "acquisition/acquisition-receipt.json",
        "acquisition/text-quality-audit.json",
    ],
)
def test_loader_rejects_tampered_acquisition_copies(
    tmp_path, prepared_packet, relative
):
    root = _copy_packet(prepared_packet, tmp_path)
    target = root / relative
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(control.RealControlPlaneError, match="bytes drifted"):
        control.load_prepared_r1_packet(root)


@pytest.mark.parametrize(
    "relative",
    [
        "acquisition/acquisition-manifest.json",
        "acquisition/acquisition-receipt.json",
    ],
)
def test_loader_rejects_reformatted_acquisition_copy_after_packet_reseal(
    tmp_path, prepared_packet, relative
):
    root = _copy_packet(prepared_packet, tmp_path)
    target = root / relative
    target.write_text(
        dumps_strict(
            load_strict(target),
            sort_keys=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    root = _reseal_packet(root)

    with pytest.raises(
        control.RealControlPlaneError,
        match="bytes are noncanonical",
    ):
        control.load_prepared_r1_packet(root)


@pytest.mark.parametrize("mutation", ["extra", "missing", "raw", "canonical"])
def test_loader_rejects_extra_or_tampered_packet_bytes(
    tmp_path, prepared_packet, mutation
):
    root = _copy_packet(prepared_packet, tmp_path)
    if mutation == "extra":
        (root / "unexpected.txt").write_text("extra", encoding="utf-8")
    elif mutation == "missing":
        next((root / "raw").rglob("*.txt")).unlink()
    elif mutation == "raw":
        target = next((root / "raw").rglob("*.txt"))
        target.write_bytes(target.read_bytes() + b" tampered")
    else:
        target = next((root / "canonical_rows").rglob("*.txt"))
        target.write_bytes(target.read_bytes() + b" tampered")

    with pytest.raises(
        control.RealControlPlaneError,
        match="inventory mismatch|bytes drifted",
    ):
        control.load_prepared_r1_packet(root)


@pytest.mark.parametrize("mutation", ["symlink", "special"])
def test_loader_rejects_symlink_and_special_files(
    tmp_path, prepared_packet, mutation
):
    root = _copy_packet(prepared_packet, tmp_path)
    unsafe = root / "unsafe"
    if mutation == "symlink":
        unsafe.symlink_to(root / "packet.json")
        message = "symlink rejected"
    else:
        os.mkfifo(unsafe)
        message = "special file rejected"

    with pytest.raises(control.RealControlPlaneError, match=message):
        control.load_prepared_r1_packet(root)


@pytest.mark.parametrize(
    "schema_version",
    [
        "stylo.lobo-vnext.ruaa-r1-packet.v2",
        "stylo.lobo-vnext.ruaa-r1-packet.v3",
        "stylo.lobo-vnext.ruaa-r1-packet.v4",
    ],
    ids=("v2", "v3", "v4"),
)
def test_loader_explicitly_rejects_legacy_packet_schema(
    tmp_path, prepared_packet, schema_version
):
    root = _copy_packet(prepared_packet, tmp_path)
    packet_path = root / "packet.json"
    payload = load_strict(packet_path)
    payload["schema_version"] = schema_version
    dump_strict(payload, packet_path, trailing_newline=True)

    with pytest.raises(
        control.RealControlPlaneError,
        match="legacy or unsupported",
    ):
        control.load_prepared_r1_packet(root)


def test_canonicalizer_policy_v1_is_rejected_without_compatibility_mode(
    prepared_packet,
):
    documents = {
        name: load_strict(prepared_packet.root / f"policies/{name}.json")
        for name in ("canonicalizer", "chunker", "ocr")
    }
    documents["canonicalizer"]["schema_version"] = (
        "stylo.lobo-vnext.canonicalizer-policy-doc.v1"
    )
    with pytest.raises(
        control.RealControlPlaneError,
        match="canonicalizer policy document.schema_version",
    ):
        control._validate_policy_documents(documents)


def test_loader_rejects_packet_root_basename_tamper(
    tmp_path, prepared_packet
):
    root = _copy_packet(prepared_packet, tmp_path)
    tampered = root.with_name("0" * 64)
    root.rename(tampered)

    with pytest.raises(
        control.RealControlPlaneError,
        match="generation id differs",
    ):
        control.load_prepared_r1_packet(tampered)


def test_execution_assembly_binds_packet_selection_and_exact_15_receipts(
    prepared_packet, monkeypatch
):
    _pin_verified_r1_ner(monkeypatch)
    _patch_live_observations(monkeypatch)

    execution, observations = control.assemble_real_execution_spec(
        packet=prepared_packet,
        cfg=load_config("configs/default.yaml"),
        repository_root=Path.cwd(),
    )

    assert tuple(row.kind for row in observations) == REQUIRED_RECEIPT_KINDS
    assert len(observations) == 15
    assert observations[0].kind == "packet_selection"
    assert observations[0].digest == prepared_packet.packet_manifest.self_hash
    assert execution.bindings.packet_manifest_digest == (
        prepared_packet.packet_manifest.self_hash
    )
    assert execution.independent_receipts[0].expected_digest == (
        prepared_packet.packet_manifest.self_hash
    )
    assert execution.confirmatory_execution_authorized is False
    assert execution.public_evidence_update_authorized is False
    assert execution.headline_update_authorized is False
    assert execution.frozen_evidence_mutation_authorized is False
    assert execution.output_namespace.namespace_id == (
        prepared_packet.packet_manifest.packet_generation_id
    )
    assert (
        execution.output_namespace.public_evidence_update_authorized is False
    )
    assert execution.output_namespace.confirmatory_output_authorized is False
    assert not any(
        marker in path.relative_to(prepared_packet.root).as_posix()
        for path in prepared_packet.root.rglob("*")
        for marker in ("authorization", "prediction", "result")
    )


@pytest.mark.parametrize(
    "pin_name",
    ["R1_RAW_INVENTORY_DIGEST", "R1_WORK_IDENTITY_CATALOG_DIGEST"],
)
def test_exact_corpus_pin_drift_stops_before_nlp_or_live_identity(
    prepared_packet, monkeypatch, pin_name
):
    monkeypatch.setattr(control, pin_name, "0" * 64)
    monkeypatch.setattr(
        control,
        "build_r1_content_policy",
        lambda _cfg: pytest.fail("NLP/content-policy resolution was reached"),
    )

    with pytest.raises(
        control.RealControlPlaneError,
        match="selected-134 acquisition",
    ):
        control.assemble_real_execution_spec(
            packet=prepared_packet,
            cfg=load_config("configs/default.yaml"),
            repository_root=Path.cwd(),
        )


def test_dirty_or_drifted_executable_source_stops_before_other_live_receipts(
    prepared_packet, monkeypatch
):
    def dirty(_repository_root):
        raise receipts.RealReceiptError("scientific worktree is dirty")

    monkeypatch.setattr(
        control, "derive_executable_source_observation", dirty
    )
    monkeypatch.setattr(
        control,
        "derive_config_and_adapter_observations",
        lambda **kwargs: pytest.fail("adapter observation ran after dirty code"),
    )

    with pytest.raises(receipts.RealReceiptError, match="dirty"):
        control.assemble_real_execution_spec(
            packet=prepared_packet,
            cfg=load_config("configs/default.yaml"),
            repository_root=Path.cwd(),
        )


def test_live_config_drift_stops_before_adapter_dependency_or_runtime_identity(
    prepared_packet, monkeypatch
):
    monkeypatch.setattr(
        control,
        "derive_executable_source_observation",
        lambda repository_root: _observation("executable_sources"),
    )
    monkeypatch.setattr(
        control,
        "derive_config_and_adapter_observations",
        lambda **kwargs: pytest.fail("adapter identity ran after config drift"),
    )
    drifted = load_config(
        "configs/default.yaml",
        overrides={"chunking.chunk_size": 501},
    )

    with pytest.raises(VNextContractError, match="chunk_size"):
        control.assemble_real_execution_spec(
            packet=prepared_packet,
            cfg=drifted,
            repository_root=Path.cwd(),
        )
