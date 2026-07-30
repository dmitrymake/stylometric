from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import inspect
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from stylo.config import load_config
from stylo.corpus_tools import ruaa_r1_acquisition as r1
from stylo.corpus_tools.text_quality_vnext import CorpusTextAuditReport
from stylo.domain.corpus_identity import ContentOverlap
from stylo.domain.lobo_vnext import WorkIdentity, canonical_sha256
from stylo.domain.lobo_vnext_packet import CanonicalRowEntry
from stylo.eval import lobo_vnext_prepare as prep
from stylo.jsonio import dump_strict
from stylo.nlp import ResolvedNLPIdentity


PREP_CLI_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluation"
    / "prepare_stylo_lobo_vnext_packet.py"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _ProviderWorkSpec:
    work_id: str
    identity_material: str

    def to_dict(self) -> dict[str, str]:
        return {
            "work_id": self.work_id,
            "identity_material": self.identity_material,
        }


@dataclass(frozen=True)
class _ProviderCampaign:
    works: tuple[_ProviderWorkSpec, ...]
    self_hash: str

    @property
    def work_ids(self) -> tuple[str, ...]:
        return tuple(work.work_id for work in self.works)

    def to_dict(self) -> dict[str, object]:
        return {
            "works": [work.to_dict() for work in self.works],
            "self_hash": self.self_hash,
        }


@dataclass(frozen=True)
class _Sentence:
    text: str

    def __len__(self) -> int:
        return len(self.text.split())


def _selected_work_ids() -> tuple[str, ...]:
    work_ids = []
    for author_index in range(prep.R1_AUTHOR_COUNT):
        work_count = 7 if author_index < 2 else 6
        work_ids.extend(
            f"author_{author_index:02d}/work_{work_index:02d}"
            for work_index in range(work_count)
        )
    result = tuple(sorted(work_ids))
    assert len(result) == prep.R1_WORK_COUNT
    return result


def _exclusions() -> tuple[r1.R1Exclusion, ...]:
    reasons = {
        r1.R1_AUTHORSHIP_MISMATCH_WORK_ID: "authorship_mismatch",
        r1.R1_SOURCE_QUALITY_REJECTED_WORK_ID: "source_quality_rejected",
        r1.R1_COLLECTION_UMBRELLA_WORK_ID: "collection_umbrella",
    }
    return tuple(
        r1.R1Exclusion(
            work_id,
            reasons[work_id],
            _sha(f"evidence:{work_id}".encode()),
            (
                None
                if work_id == r1.R1_COLLECTION_UMBRELLA_WORK_ID
                else _sha(f"receipt:{work_id}".encode())
            ),
        )
        for work_id in prep.R1_UPSTREAM_EXCLUDED_WORK_IDS
    )


def _build_materialized_acquisition(root: Path) -> r1.MaterializedR1Acquisition:
    work_ids = _selected_work_ids()
    payloads = {
        work_id: (
            " ".join(
                f"token_{work_id.replace('/', '_')}_{index:03d}"
                for index in range(32)
            )
            + "\n"
        ).encode("utf-8")
        for work_id in work_ids
    }
    raw_rows = []
    for work_id in work_ids:
        path = root / "raw" / f"{work_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[work_id])
        raw_rows.append(r1.R1RawInventoryRow.build(work_id, payloads[work_id]))

    provider_specs = tuple(
        _ProviderWorkSpec(work_id, f"provider-spec:{index:03d}")
        for index, work_id in enumerate(work_ids)
    )
    wikisource = _ProviderCampaign(
        provider_specs[:127],
        _sha(b"synthetic-wikisource-campaign"),
    )
    feb = provider_specs[127]
    reviewed = _ProviderCampaign(
        provider_specs[128:],
        _sha(b"synthetic-reviewed-text-campaign"),
    )
    generation_id = _sha(b"synthetic-selected-134-generation")
    manifest_self_hash = _sha(b"synthetic-acquisition-manifest")
    manifest = r1.R1AcquisitionManifest(
        wikisource_campaign=wikisource,
        wikisource_discovery_candidate_sha256=_sha(b"discovery-candidate"),
        source_curation_receipt_sha256=_sha(b"source-curation"),
        feb_work_spec=feb,
        included_work_ids=work_ids,
        exclusions=_exclusions(),
        text_quality_spec=r1.R1TextQualitySpec.build(),
        generation_id=generation_id,
        self_hash=manifest_self_hash,
        reviewed_text_campaign=reviewed,
        schema_version=r1.R1_ACQUISITION_MANIFEST_SCHEMA_VERSION_V3,
    )
    audit_core = {
        "status": "passed",
        "blocking_findings": [],
        "cross_work_overlaps": [],
        "work_count": prep.R1_WORK_COUNT,
    }
    audit = CorpusTextAuditReport(
        {**audit_core, "self_hash": canonical_sha256(audit_core)}
    )
    receipt = r1.R1AcquisitionReceipt(
        manifest_sha256=manifest.self_hash,
        generation_id=manifest.generation_id,
        wikisource_campaign_spec_sha256=wikisource.self_hash,
        wikisource_campaign_receipt_sha256=_sha(
            b"synthetic-wikisource-receipt"
        ),
        feb_work_spec_sha256=canonical_sha256(feb.to_dict()),
        feb_work_receipt_sha256=_sha(b"synthetic-feb-receipt"),
        text_quality_audit_sha256=audit.self_hash,
        included_work_ids=work_ids,
        raw_inventory=tuple(raw_rows),
        self_hash=_sha(b"synthetic-acquisition-receipt"),
        reviewed_text_campaign_spec_sha256=reviewed.self_hash,
        reviewed_text_campaign_receipt_sha256=_sha(
            b"synthetic-reviewed-text-receipt"
        ),
        schema_version=r1.R1_ACQUISITION_RECEIPT_SCHEMA_VERSION_V2,
    )
    dump_strict(
        manifest.to_dict(),
        root / r1.MANIFEST_NAME,
        sort_keys=True,
        trailing_newline=True,
    )
    dump_strict(
        receipt.to_dict(),
        root / r1.ACQUISITION_RECEIPT_NAME,
        sort_keys=True,
        trailing_newline=True,
    )
    dump_strict(
        audit.to_dict(),
        root / r1.AUDIT_REPORT_NAME,
        sort_keys=True,
        trailing_newline=True,
    )
    return r1.MaterializedR1Acquisition(
        root=root,
        manifest=manifest,
        receipt=receipt,
        audit_report=audit,
        resumed=True,
    )


@pytest.fixture
def acquisition(tmp_path) -> r1.MaterializedR1Acquisition:
    return _build_materialized_acquisition(tmp_path / "acquisition")


def _pin_canonical_acquisition(
    monkeypatch,
    acquisition: r1.MaterializedR1Acquisition,
) -> None:
    monkeypatch.setattr(
        prep,
        "R1_ACQUISITION_GENERATION_ID",
        acquisition.manifest.generation_id,
    )
    monkeypatch.setattr(
        prep,
        "R1_ACQUISITION_MANIFEST_SELF_HASH",
        acquisition.manifest.self_hash,
    )
    monkeypatch.setattr(
        prep,
        "R1_ACQUISITION_RECEIPT_SELF_HASH",
        acquisition.receipt.self_hash,
    )
    monkeypatch.setattr(
        prep,
        "R1_SELECTED_AUDIT_FILE_SHA256",
        _sha((acquisition.root / r1.AUDIT_REPORT_NAME).read_bytes()),
    )
    monkeypatch.setattr(
        prep,
        "R1_SELECTED_AUDIT_SELF_HASH",
        acquisition.audit_report.self_hash,
    )


def _pin_verified_r1_ner(
    monkeypatch,
    *,
    package_record_sha256: str = "a" * 64,
) -> ResolvedNLPIdentity:
    material = {
        "requested_model": "ru_core_news_lg",
        "resolved_model": "ru_core_news_lg",
        "fallback_used": False,
        "package_version": "3.8.0",
        "package_record_sha256": package_record_sha256,
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
        package_record_sha256=material["package_record_sha256"],
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
    return identity


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
        relative_path = f"canonical_rows/{work.work_id}/000000.txt"
        output = packet_root / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        rows.append(
            CanonicalRowEntry.from_dict(
                {
                    "row_id": f"{work.work_id}#000000",
                    "relative_path": relative_path,
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


def _patch_packet_dependencies(
    monkeypatch,
    acquisition: r1.MaterializedR1Acquisition,
) -> None:
    _pin_canonical_acquisition(monkeypatch, acquisition)
    _pin_verified_r1_ner(monkeypatch)

    def load(root):
        assert Path(root) == acquisition.root
        return acquisition

    monkeypatch.setattr(prep, "load_materialized_r1_acquisition", load)
    monkeypatch.setattr(
        prep,
        "find_cross_work_content_overlaps",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(prep, "_canonical_rows", _fake_canonical_rows)


def _file_map(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_prepare_cli(name: str):
    module_spec = importlib.util.spec_from_file_location(name, PREP_CLI_PATH)
    assert module_spec is not None and module_spec.loader is not None
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    return cli


def test_prepare_module_has_no_estimator_factory_fit_or_runner_reachability():
    tree = ast.parse(inspect.getsource(prep))
    imported_or_called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_or_called.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                imported_or_called.add(function.id)
            elif isinstance(function, ast.Attribute):
                imported_or_called.add(function.attr)

    assert imported_or_called.isdisjoint(
        {
            "fit",
            "fit_estimator",
            "make_factory",
            "run_lobo_vnext",
            "run_real_lobo_vnext",
            "predict",
            "predict_proba",
            "create_authorization",
            "publish_result",
        }
    )


def test_acquisition_selected_134_is_exact_and_exclusions_are_upstream_only(
    acquisition,
):
    works, raw_inventory = prep._acquisition_catalog(acquisition)
    provider_counts = Counter(
        kind
        for kind, _spec in prep._provider_work_specs(
            acquisition.manifest
        ).values()
    )
    author_counts = Counter(work.author_id for work in works)
    selected = {work.work_id for work in works}

    assert len(works) == len(raw_inventory) == prep.R1_WORK_COUNT == 134
    assert len(author_counts) == prep.R1_AUTHOR_COUNT == 22
    assert min(author_counts.values()) >= 2
    assert provider_counts == {
        "wikisource": 127,
        "feb": 1,
        "reviewed-text": 6,
    }
    assert tuple(
        row.work_id for row in acquisition.manifest.exclusions
    ) == prep.R1_UPSTREAM_EXCLUDED_WORK_IDS
    assert selected.isdisjoint(prep.R1_UPSTREAM_EXCLUDED_WORK_IDS)
    for excluded in prep.R1_UPSTREAM_EXCLUDED_WORK_IDS:
        assert not (acquisition.root / "raw" / f"{excluded}.txt").exists()
    assert all(work.work_kind == "work" for work in works)
    assert all(
        work.raw_paths == (f"raw/{work.work_id}.txt",)
        for work in works
    )


def test_provider_bound_work_identity_is_deterministic_and_drift_sensitive(
    acquisition,
):
    first, _ = prep._acquisition_catalog(acquisition)
    second, _ = prep._acquisition_catalog(acquisition)
    assert first == second

    first_by_id = {work.work_id: work for work in first}
    work_spec = acquisition.manifest.wikisource_campaign.works[0]
    work = first_by_id[work_spec.work_id]
    raw = acquisition.receipt.raw_inventory[0]
    assert work.author_id == work.work_id.split("/", 1)[0]
    assert work.edition_id.endswith(f"sha256:{raw.sha256}")
    assert work.source_id == (
        "stylo.lobo-vnext.ruaa-r1-provider-work-spec.v1:"
        f"wikisource:{canonical_sha256(work_spec.to_dict())}"
    )

    drifted_spec = dataclasses.replace(
        work_spec,
        identity_material="provider-spec:drifted",
    )
    drifted_campaign = dataclasses.replace(
        acquisition.manifest.wikisource_campaign,
        works=(
            drifted_spec,
            *acquisition.manifest.wikisource_campaign.works[1:],
        ),
    )
    provider_drift = dataclasses.replace(
        acquisition,
        manifest=dataclasses.replace(
            acquisition.manifest,
            wikisource_campaign=drifted_campaign,
        ),
    )
    provider_works, _ = prep._acquisition_catalog(provider_drift)
    provider_by_id = {row.work_id: row for row in provider_works}
    assert provider_by_id[work.work_id].source_id != work.source_id
    assert provider_by_id[work.work_id].edition_id == work.edition_id

    drifted_raw = dataclasses.replace(raw, sha256="f" * 64)
    raw_drift = dataclasses.replace(
        acquisition,
        receipt=dataclasses.replace(
            acquisition.receipt,
            raw_inventory=(
                drifted_raw,
                *acquisition.receipt.raw_inventory[1:],
            ),
        ),
    )
    raw_works, _ = prep._acquisition_catalog(raw_drift)
    raw_by_id = {row.work_id: row for row in raw_works}
    assert raw_by_id[work.work_id].edition_id != work.edition_id
    assert raw_by_id[work.work_id].source_id == work.source_id


def test_provider_ambiguity_is_rejected(acquisition):
    duplicate_id = acquisition.manifest.wikisource_campaign.work_ids[0]
    reviewed = acquisition.manifest.reviewed_text_campaign
    assert reviewed is not None
    ambiguous_reviewed = dataclasses.replace(
        reviewed,
        works=(
            *reviewed.works,
            _ProviderWorkSpec(duplicate_id, "ambiguous-provider-spec"),
        ),
    )
    ambiguous = dataclasses.replace(
        acquisition.manifest,
        reviewed_text_campaign=ambiguous_reviewed,
    )

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="ambiguous or incomplete",
    ):
        prep._provider_work_specs(ambiguous)


def test_unknown_provider_kind_is_rejected(acquisition, monkeypatch):
    providers = prep._provider_work_specs(acquisition.manifest)
    work_id = acquisition.manifest.included_work_ids[0]
    _kind, spec = providers[work_id]
    providers[work_id] = ("unknown-provider", spec)
    monkeypatch.setattr(
        prep,
        "_provider_work_specs",
        lambda _manifest: providers,
    )

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="unknown R1 provider kind",
    ):
        prep._acquisition_catalog(acquisition)


def test_malformed_acquisition_work_id_is_rejected(acquisition):
    old_spec = acquisition.manifest.wikisource_campaign.works[0]
    malformed_id = "author_00/nested/work"
    malformed_spec = dataclasses.replace(old_spec, work_id=malformed_id)
    campaign = dataclasses.replace(
        acquisition.manifest.wikisource_campaign,
        works=(
            malformed_spec,
            *acquisition.manifest.wikisource_campaign.works[1:],
        ),
    )
    included = tuple(
        sorted(
            (
                malformed_id,
                *acquisition.manifest.included_work_ids[1:],
            )
        )
    )
    malformed = dataclasses.replace(
        acquisition,
        manifest=dataclasses.replace(
            acquisition.manifest,
            wikisource_campaign=campaign,
            included_work_ids=included,
        ),
    )

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="work_id is malformed",
    ):
        prep._acquisition_catalog(malformed)


def test_bom_is_hash_bound_but_rejected_by_packet_text_policy(acquisition):
    works, _ = prep._acquisition_catalog(acquisition)
    work = works[0]
    path = acquisition.root / work.raw_paths[0]
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="BOM-reject policy",
    ):
        prep._read_source_texts(acquisition.root, (work,))


def test_selected_content_screen_requires_exact_zero_candidates(
    acquisition,
    monkeypatch,
):
    works, _ = prep._acquisition_catalog(acquisition)
    calls = []

    def zero(texts, groups, **kwargs):
        calls.append((texts, groups, kwargs))
        return ()

    monkeypatch.setattr(prep, "find_cross_work_content_overlaps", zero)
    prep._screen_selected_content(acquisition.root, works)

    assert len(calls) == 1
    texts, groups, kwargs = calls[0]
    assert len(texts) == len(groups) == prep.R1_WORK_COUNT
    assert tuple(groups) == tuple(work.work_id for work in works)
    assert kwargs == {
        "containment_threshold": prep.R1_WORD5_THRESHOLD,
        "min_shingles": prep.R1_WORD5_MIN_SHINGLES,
        "sample_size": prep.R1_WORD5_SAMPLE_SIZE,
    }


def test_candidate_screen_blocks_before_nlp(acquisition, tmp_path, monkeypatch):
    _pin_canonical_acquisition(monkeypatch, acquisition)
    monkeypatch.setattr(
        prep,
        "load_materialized_r1_acquisition",
        lambda root: acquisition,
    )
    works, _ = prep._acquisition_catalog(acquisition)
    candidate = ContentOverlap(
        works[0].work_id,
        works[1].work_id,
        "word5_asymmetric_containment",
        1.0,
        "synthetic exact candidate",
    )
    monkeypatch.setattr(
        prep,
        "find_cross_work_content_overlaps",
        lambda *_args, **_kwargs: (candidate,),
    )
    monkeypatch.setattr(
        prep,
        "load_ner",
        lambda *_args, **_kwargs: pytest.fail(
            "candidate screen must stop before NLP"
        ),
    )

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="post-selection content screen found candidates",
    ):
        prep.prepare_r1_packet(
            acquisition_root=acquisition.root,
            output_parent=(
                tmp_path / "exploratory" / "lobo_vnext" / "packets"
            ),
            cfg=load_config("configs/default.yaml"),
        )


def test_r1_content_policy_freezes_exact_owner_selected_values(monkeypatch):
    identity = _pin_verified_r1_ner(monkeypatch)
    policy, documents = prep.build_r1_content_policy(
        load_config("configs/default.yaml")
    )
    word5 = policy.automatic_candidates.word5_containment

    assert policy.strict_utf8.encoding == "utf-8"
    assert policy.strict_utf8.errors == "strict"
    assert policy.strict_utf8.bom_disposition == "reject"
    assert policy.canonical_row_policy.disposition == "transform_versioned"
    assert policy.yo_e_policy.disposition == "transform_versioned"
    assert (
        policy.historical_orthography_policy.disposition
        == "transform_versioned"
    )
    assert policy.ocr_policy.disposition == "preserve"
    assert policy.markup_policy.disposition == "transform_versioned"
    assert word5 is not None
    assert (
        word5.threshold.numerator,
        word5.threshold.denominator,
        word5.min_shingles,
        word5.sample_size,
    ) == (9, 10, 20, 64)
    assert word5.threshold_boundary == "inclusive"
    assert (
        word5.final_verification == "exact_intersection_authoritative"
    )
    assert documents["chunker"]["chunk_size"] == 500
    assert documents["chunker"]["min_words"] == 200
    assert documents["chunker"]["overlap"] == 0.0
    assert documents["chunker"]["sentence_aware"] is True
    assert documents["canonicalizer"]["yo_to_e"] is True
    assert (
        documents["canonicalizer"]["historical_orthography_to_modern"]
        is True
    )
    assert documents["canonicalizer"]["ocr_correction"] is False
    assert (
        documents["canonicalizer"]["resolved_person_model_identity"]
        == {
            **identity.to_dict(),
            "disabled_pipes": list(identity.disabled_pipes),
            "active_pipes": list(identity.active_pipes),
        }
    )

    _pin_verified_r1_ner(monkeypatch, package_record_sha256="b" * 64)
    changed_policy, _ = prep.build_r1_content_policy(
        load_config("configs/default.yaml")
    )
    assert changed_policy.self_hash != policy.self_hash


def test_r1_content_policy_fails_closed_without_verified_ner(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("no verified R1 NER package")

    monkeypatch.setattr(prep, "load_ner", unavailable)
    with pytest.raises(
        prep.R1PacketPreparationError,
        match="cannot resolve and verify the R1 NER pipeline",
    ):
        prep.build_r1_content_policy(load_config("configs/default.yaml"))


def test_canonical_rows_bind_literal_raw_and_versioned_transform(
    acquisition,
    tmp_path,
    monkeypatch,
):
    works, _ = prep._acquisition_catalog(acquisition)
    work = works[0]
    raw_payload = (acquisition.root / work.raw_paths[0]).read_bytes()
    clean_text = " ".join(f"cleaned_{index:03d}" for index in range(250))
    sentencizer = object()

    def normalize(text, model, fallback):
        assert text == raw_payload.decode("utf-8")
        assert (model, fallback) == (
            "ru_core_news_lg",
            "ru_core_news_md",
        )
        return clean_text

    def sentences(text, nlp):
        assert text == clean_text
        assert nlp is sentencizer
        return [_Sentence(text)]

    monkeypatch.setattr(prep, "normalize", normalize)
    monkeypatch.setattr(
        prep,
        "load_sentencizer",
        lambda language: sentencizer if language == "ru" else None,
    )
    monkeypatch.setattr(prep, "sentences_for_text", sentences)
    packet_root = tmp_path / "packet"
    rows = prep._canonical_rows(
        cfg=load_config("configs/default.yaml"),
        raw_root=acquisition.root,
        packet_root=packet_root,
        works=(work,),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.source_relative_path == work.raw_paths[0]
    assert row.source_raw_sha256 == _sha(raw_payload)
    assert row.word_count == 250
    assert (packet_root / row.relative_path).read_text() == clean_text


def test_packet_trees_are_byte_identical_and_target_is_no_clobber(
    acquisition,
    tmp_path,
    monkeypatch,
):
    _patch_packet_dependencies(monkeypatch, acquisition)
    cfg = load_config("configs/default.yaml")
    first_parent = (
        tmp_path / "first" / "exploratory" / "lobo_vnext" / "packets"
    )
    second_parent = (
        tmp_path / "second" / "exploratory" / "lobo_vnext" / "packets"
    )

    first = prep.prepare_r1_packet(
        acquisition_root=acquisition.root,
        output_parent=first_parent,
        cfg=cfg,
    )
    second = prep.prepare_r1_packet(
        acquisition_root=acquisition.root,
        output_parent=second_parent,
        cfg=cfg,
    )

    assert first.root.name == second.root.name
    assert _file_map(first.root) == _file_map(second.root)
    assert first.packet_manifest == second.packet_manifest
    assert first.acquisition_binding == second.acquisition_binding
    assert first.packet_manifest.selected_work_count == prep.R1_WORK_COUNT
    assert first.acquisition_binding.work_count == prep.R1_WORK_COUNT
    assert first.acquisition_binding.author_count == prep.R1_AUTHOR_COUNT
    assert (
        first.acquisition_binding.upstream_excluded_work_ids
        == prep.R1_UPSTREAM_EXCLUDED_WORK_IDS
    )
    assert first.candidate_inventory.candidates == ()
    assert len(first.corpus_manifest.works) == prep.R1_WORK_COUNT
    assert len(first.corpus_manifest.author_ids) == prep.R1_AUTHOR_COUNT
    assert len(first.content_manifest.components) == prep.R1_WORK_COUNT
    assert all(
        len(component.work_ids) == 1
        for component in first.content_manifest.components
    )
    assert first.fold_manifest.mode == "isolated"
    assert first.packet_manifest.confirmatory_authorized is False
    assert len(list((first.root / "raw").rglob("*.txt"))) == 134
    assert len(list((first.root / "canonical_rows").rglob("*.txt"))) == 134
    for excluded in prep.R1_UPSTREAM_EXCLUDED_WORK_IDS:
        assert not (first.root / "raw" / f"{excluded}.txt").exists()
        assert not (first.root / "canonical_rows" / excluded).exists()
    assert all(
        forbidden not in relative
        for relative in _file_map(first.root)
        for forbidden in ("authorization", "prediction", "result")
    )

    before = _file_map(first.root)
    with pytest.raises(
        prep.R1PacketPreparationError,
        match="already exists",
    ):
        prep.prepare_r1_packet(
            acquisition_root=acquisition.root,
            output_parent=first_parent,
            cfg=cfg,
        )
    assert _file_map(first.root) == before


def test_atomic_publication_never_replaces_an_existing_empty_target(tmp_path):
    source = tmp_path / "staged"
    target = tmp_path / "published"
    source.mkdir()
    target.mkdir()
    (source / "packet.json").write_bytes(b"staged")

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="immutable R1 packet conflict",
    ):
        prep._publish_directory_no_replace(source, target)

    assert source.joinpath("packet.json").read_bytes() == b"staged"
    assert list(target.iterdir()) == []


def test_packet_output_must_be_explicit_exploratory_namespace(
    acquisition,
    tmp_path,
    monkeypatch,
):
    _patch_packet_dependencies(monkeypatch, acquisition)

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="explicit exploratory/.+ namespace",
    ):
        prep.prepare_r1_packet(
            acquisition_root=acquisition.root,
            output_parent=tmp_path / "packet-output",
            cfg=load_config("configs/default.yaml"),
        )


@pytest.mark.parametrize(
    "audit_changes",
    (
        {
            "status": "blocked",
            "blocking_findings": [
                {
                    "kind": "synthetic_blocker",
                    "work_ids": ["author_00/work_00"],
                    "evidence": "synthetic",
                }
            ],
        },
        {
            "cross_work_overlaps": [
                {
                    "left_work": "author_00/work_00",
                    "right_work": "author_00/work_01",
                    "kind": "word5_asymmetric_containment",
                    "containment": "1",
                    "evidence": "synthetic",
                }
            ]
        },
    ),
    ids=("blocked", "overlap"),
)
def test_blocked_or_overlapping_acquisition_stops_before_nlp(
    acquisition,
    audit_changes,
    tmp_path,
    monkeypatch,
):
    payload = acquisition.audit_report.to_dict()
    payload.update(audit_changes)
    rejected = dataclasses.replace(
        acquisition,
        audit_report=CorpusTextAuditReport(payload),
    )
    _pin_canonical_acquisition(monkeypatch, rejected)
    monkeypatch.setattr(
        prep,
        "load_materialized_r1_acquisition",
        lambda root: rejected,
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("acquisition quality boundary must stop before NLP")

    monkeypatch.setattr(prep, "_acquisition_catalog", forbidden)
    monkeypatch.setattr(prep, "load_ner", forbidden)
    with pytest.raises(
        prep.R1PacketPreparationError,
        match="quality/receipt boundary is not passed and exact",
    ):
        prep.prepare_r1_packet(
            acquisition_root=acquisition.root,
            output_parent=(
                tmp_path / "exploratory" / "lobo_vnext" / "packets"
            ),
            cfg=load_config("configs/default.yaml"),
        )


def test_cli_accepts_exact_three_args_and_rejects_legacy_before_loader_nlp(
    tmp_path,
    monkeypatch,
):
    cli = _load_prepare_cli("prepare_lobo_vnext_packet_cli_contract_test")
    parser = cli._parser()
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest != "help"
    }
    assert tuple(actions) == (
        "acquisition_root",
        "config",
        "output_parent",
    )
    assert all(action.required for action in actions.values())
    parsed = parser.parse_args(
        [
            "--acquisition-root",
            str(tmp_path / "acquisition"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-parent",
            str(tmp_path / "output"),
        ]
    )
    assert vars(parsed) == {
        "acquisition_root": tmp_path / "acquisition",
        "config": tmp_path / "config.yaml",
        "output_parent": tmp_path / "output",
    }
    assert tuple(inspect.signature(prep.prepare_r1_packet).parameters) == (
        "acquisition_root",
        "output_parent",
        "cfg",
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("legacy argparse rejection must precede loader/NLP")

    monkeypatch.setattr(cli, "_require_acquisition_root", forbidden)
    monkeypatch.setattr(cli, "load_config", forbidden)
    monkeypatch.setattr(cli, "prepare_r1_packet", forbidden)
    base = [
        "--acquisition-root",
        str(tmp_path / "acquisition"),
        "--config",
        str(tmp_path / "config.yaml"),
        "--output-parent",
        str(tmp_path / "output"),
    ]
    legacy = (
        ("--approved-r1",),
        ("--source-root", str(tmp_path / "legacy-source")),
        (
            "--legacy-source-manifest",
            str(tmp_path / "legacy-manifest.json"),
        ),
    )
    for old_args in legacy:
        with pytest.raises(SystemExit) as raised:
            cli.run([*base, *old_args])
        assert raised.value.code == 2
