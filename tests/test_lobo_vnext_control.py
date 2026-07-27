from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import shutil
from pathlib import Path

import pytest

from stylo.config import load_config
from stylo.domain.corpus_identity import ContentOverlap
from stylo.domain.lobo_vnext import (
    VNextContractError,
    canonical_sha256,
)
from stylo.domain.lobo_vnext_packet import CanonicalRowEntry, R1PacketManifest
from stylo.domain.lobo_vnext_real import REQUIRED_RECEIPT_KINDS
from stylo.eval import lobo_vnext_control as control
from stylo.eval import lobo_vnext_prepare as prep
from stylo.eval import lobo_vnext_receipts as receipts
from stylo.jsonio import dump_strict, load_strict
from stylo.nlp import ResolvedNLPIdentity


def _sha(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir(parents=True)
    authors: dict[str, object] = {}
    author_ids = ["turgenev", *(f"author_{index:02d}" for index in range(21))]
    total = 0
    for author_id in sorted(author_ids):
        if author_id == "turgenev":
            book_ids = [
                "бирюк",
                "вешние_воды",
                "дворянское_гнездо",
                "записки_охотника",
                "муму",
                "накануне",
                "отцы_и_дети",
                "певцы",
                "первая_любовь",
                "рудин",
                "хорь_и_калиныч",
            ]
        else:
            book_ids = [f"work_{index:02d}" for index in range(6)]
        books = []
        for book_id in book_ids:
            work_id = f"{author_id}/{book_id}"
            payload = f"literal source words for {work_id}".encode("utf-8")
            path = source / f"{work_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            books.append(
                {
                    "book": book_id,
                    "sha256": _sha(payload),
                    "words": len(payload.decode("utf-8").split()),
                    "source": f"public-source:{work_id}",
                }
            )
            total += 1
        authors[author_id] = {
            "death_year": 1900,
            "n_books": len(books),
            "books": books,
        }
    assert total == prep.R1_SOURCE_BOOK_COUNT
    manifest = {
        "name": prep.R1_SOURCE_NAME,
        "version": prep.R1_SOURCE_VERSION,
        "claim_status": "exploratory_internal",
        "benchmark_role": "reproducible_cv_legacy_not_blind",
        "training_weighting": "chunk_weighted_training_legacy",
        "task": "synthetic control-plane source fixture",
        "n_authors": prep.R1_AUTHOR_COUNT,
        "n_books": prep.R1_SOURCE_BOOK_COUNT,
        "legal": "synthetic",
        "authors": authors,
        "dropped": {},
    }
    manifest_path = root / "legacy-source-manifest.json"
    dump_strict(manifest, manifest_path, trailing_newline=True)
    return source, manifest_path


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
        "package_record_sha256": "a" * 64,
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


@pytest.fixture(scope="module")
def prepared_packet(tmp_path_factory):
    root = tmp_path_factory.mktemp("control-plane")
    source, source_manifest = _source_fixture(root)
    overlaps = tuple(
        ContentOverlap(
            left_work=member,
            right_work=prep.R1_EXCLUDED_WORK_ID,
            kind="word5_asymmetric_containment",
            containment=numerator / denominator,
            evidence=f"{numerator}/{denominator} unique word-5-grams",
        )
        for member, (numerator, denominator) in zip(
            prep.R1_COLLECTION_MEMBERS,
            ((2019, 2090), (5093, 5295), (3542, 3637)),
            strict=True,
        )
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(
        prep,
        "R1_SOURCE_MANIFEST_SHA256",
        _sha(source_manifest.read_bytes()),
    )
    _pin_verified_r1_ner(patch)
    patch.setattr(prep, "_canonical_rows", _fake_canonical_rows)
    patch.setattr(
        prep,
        "find_cross_work_content_overlaps",
        lambda *args, **kwargs: overlaps,
    )
    try:
        packet = prep.prepare_r1_packet(
            source_root=source,
            legacy_source_manifest=source_manifest,
            output_parent=(
                root / "exploratory" / "lobo_vnext" / "packets"
            ),
            cfg=load_config("configs/default.yaml"),
        )
    finally:
        patch.undo()
    return packet


def _copy_packet(packet, destination: Path) -> Path:
    copied = destination / packet.root.name
    shutil.copytree(packet.root, copied)
    return copied


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


def test_loader_rebuilds_strict_v2_packet_and_exposes_selection_proof(
    prepared_packet,
):
    loaded = control.load_prepared_r1_packet(prepared_packet.root)

    assert type(loaded.packet_manifest) is R1PacketManifest
    assert loaded.packet_manifest.self_hash == (
        prepared_packet.packet_manifest.self_hash
    )
    assert loaded.source_candidate_inventory == (
        prepared_packet.source_candidate_inventory
    )
    assert loaded.source_selection_receipt == (
        prepared_packet.source_selection_receipt
    )
    assert len(loaded.source_selection_receipt.source_raw_inventory) == 137
    assert len(loaded.source_selection_receipt.source_works) == 137
    assert loaded.candidate_inventory.candidates == ()
    assert loaded.packet_manifest.generation_material.candidate_evidence_digest
    assert loaded.packet_manifest.selected_work_count == 136
    assert loaded.packet_manifest.confirmatory_authorized is False


@pytest.mark.parametrize("mutation", ["extra", "raw", "canonical"])
def test_loader_rejects_extra_or_tampered_packet_bytes(
    tmp_path, prepared_packet, mutation
):
    root = _copy_packet(prepared_packet, tmp_path)
    if mutation == "extra":
        (root / "unexpected.txt").write_text("extra", encoding="utf-8")
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


def test_loader_rejects_symlink_and_duplicate_json_keys(
    tmp_path, prepared_packet
):
    symlink_root = _copy_packet(prepared_packet, tmp_path / "symlink")
    (symlink_root / "unsafe").symlink_to(symlink_root / "packet.json")
    with pytest.raises(control.RealControlPlaneError, match="symlink rejected"):
        control.load_prepared_r1_packet(symlink_root)

    duplicate_root = _copy_packet(prepared_packet, tmp_path / "duplicate")
    packet_path = duplicate_root / "packet.json"
    text = packet_path.read_text(encoding="utf-8")
    packet_path.write_text(
        '{"status":"duplicate",' + text[1:],
        encoding="utf-8",
    )
    with pytest.raises(
        control.RealControlPlaneError, match="duplicate object key"
    ):
        control.load_prepared_r1_packet(duplicate_root)


def test_loader_rejects_rehashed_selection_or_evidence_mismatch(
    tmp_path, prepared_packet
):
    root = _copy_packet(prepared_packet, tmp_path)
    evidence_path = root / "candidates" / "evidence.json"
    evidence = load_strict(evidence_path)
    evidence[0]["reported_containment"] = 0.91
    evidence[0]["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in evidence[0].items()
            if key != "evidence_sha256"
        }
    )
    dump_strict(evidence, evidence_path, trailing_newline=True)

    manifest = prepared_packet.packet_manifest
    files = []
    for row in manifest.files:
        if row.relative_path == "candidates/evidence.json":
            payload = evidence_path.read_bytes()
            files.append(
                dataclasses.replace(
                    row, byte_size=len(payload), sha256=_sha(payload)
                )
            )
        else:
            files.append(row)
    rehashed = R1PacketManifest.build(
        generation_material=manifest.generation_material,
        source_candidate_inventory_sha256=(
            manifest.source_candidate_inventory_sha256
        ),
        candidate_inventory_sha256=manifest.candidate_inventory_sha256,
        corpus_manifest_sha256=manifest.corpus_manifest_sha256,
        content_component_manifest_sha256=(
            manifest.content_component_manifest_sha256
        ),
        fold_manifest_sha256=manifest.fold_manifest_sha256,
        primary_model_spec_sha256=manifest.primary_model_spec_sha256,
        baseline_model_spec_sha256=manifest.baseline_model_spec_sha256,
        inference_spec_sha256=manifest.inference_spec_sha256,
        primary_inner_cv_plan_sha256=(
            manifest.primary_inner_cv_plan_sha256
        ),
        baseline_inner_cv_plan_sha256=(
            manifest.baseline_inner_cv_plan_sha256
        ),
        model_role_manifest_sha256=manifest.model_role_manifest_sha256,
        campaign_manifest_sha256=manifest.campaign_manifest_sha256,
        representation_receipt_sha256=(
            manifest.representation_receipt_sha256
        ),
        files=files,
    )
    dump_strict(rehashed.to_dict(), root / "packet.json", trailing_newline=True)

    with pytest.raises(
        control.RealControlPlaneError,
        match="generation material|evidence",
    ):
        control.load_prepared_r1_packet(root)


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
