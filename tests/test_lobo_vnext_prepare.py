from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from stylo.config import load_config
from stylo.domain.corpus_identity import ContentOverlap
from stylo.domain.lobo_vnext import VNextContractError
from stylo.domain.lobo_vnext_packet import CanonicalRowEntry
from stylo.eval import lobo_vnext_prepare as prep
from stylo.jsonio import dump_strict


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_fixture(
    tmp_path: Path,
    *,
    bom_work_id: str | None = None,
) -> tuple[Path, Path]:
    """Build the exact 22-author/137-work legacy source shape, with tiny texts."""

    source = tmp_path / "source"
    source.mkdir(parents=True)
    author_ids = ["turgenev", *(f"author_{index:02d}" for index in range(21))]
    authors: dict[str, object] = {}
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
            payload = (
                b"\xef\xbb\xbf" if work_id == bom_work_id else b""
            ) + f"unique literal text for {work_id}".encode("utf-8")
            path = source / f"{work_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            books.append(
                {
                    "book": book_id,
                    "sha256": _sha(payload),
                    "words": 5,
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
        "task": "synthetic source-evidence fixture",
        "n_authors": prep.R1_AUTHOR_COUNT,
        "n_books": prep.R1_SOURCE_BOOK_COUNT,
        "legal": "synthetic",
        "authors": authors,
        "dropped": {},
    }
    manifest_path = tmp_path / "legacy-source-manifest.json"
    dump_strict(manifest, manifest_path, trailing_newline=True)
    return source, manifest_path


def _pin_manifest(monkeypatch, manifest: Path) -> None:
    monkeypatch.setattr(
        prep, "R1_SOURCE_MANIFEST_SHA256", _sha(manifest.read_bytes())
    )


def _catalog(tmp_path: Path, monkeypatch, **kwargs):
    source, manifest = _source_fixture(tmp_path, **kwargs)
    _pin_manifest(monkeypatch, manifest)
    works, metadata, inventory, manifest_sha = prep._source_catalog(
        source, manifest
    )
    return source, manifest, works, metadata, inventory, manifest_sha


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


def test_prepare_module_has_no_estimator_factory_fit_or_runner_reachability():
    tree = ast.parse(inspect.getsource(prep))
    imported_or_called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_or_called.update(
                alias.name for alias in node.names
            )
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
        }
    )


def test_source_catalog_is_exact_and_r1_selection_is_136_constituents_only(
    tmp_path, monkeypatch
):
    source, manifest, works, _metadata, inventory, manifest_sha = _catalog(
        tmp_path, monkeypatch
    )
    selected = prep._selected_works(works)

    assert source.is_dir()
    assert manifest.is_file()
    assert len(works) == len(inventory) == prep.R1_SOURCE_BOOK_COUNT
    assert len(selected) == prep.R1_SELECTED_BOOK_COUNT
    assert prep.R1_EXCLUDED_WORK_ID not in {
        work.work_id for work in selected
    }
    assert set(prep.R1_COLLECTION_MEMBERS) <= {
        work.work_id for work in selected
    }
    assert len(manifest_sha) == 64


def test_source_catalog_rejects_hash_tamper_extra_missing_and_symlink(
    tmp_path, monkeypatch
):
    source, manifest = _source_fixture(tmp_path)
    _pin_manifest(monkeypatch, manifest)
    target = source / "author_00/work_00.txt"
    original = target.read_bytes()

    target.write_bytes(original + b" tampered")
    with pytest.raises(prep.R1PacketPreparationError, match="bytes differ"):
        prep._source_catalog(source, manifest)
    target.write_bytes(original)

    extra = source / "author_00/extra.txt"
    extra.write_bytes(b"extra")
    with pytest.raises(
        prep.R1PacketPreparationError, match="missing/extra"
    ):
        prep._source_catalog(source, manifest)
    extra.unlink()

    target.unlink()
    with pytest.raises(
        prep.R1PacketPreparationError, match="bytes differ"
    ):
        prep._source_catalog(source, manifest)

    target.symlink_to(source / "author_00/work_01.txt")
    with pytest.raises(VNextContractError, match="symlink rejected"):
        prep._source_catalog(source, manifest)


def test_bom_is_hash_bound_source_evidence_but_rejected_by_r1_text_policy(
    tmp_path, monkeypatch
):
    bom_work = "author_00/work_00"
    source, _manifest, works, *_ = _catalog(
        tmp_path, monkeypatch, bom_work_id=bom_work
    )

    with pytest.raises(
        prep.R1PacketPreparationError, match="BOM-reject policy"
    ):
        prep._read_source_texts(source, works)


def test_r1_content_policy_freezes_exact_owner_selected_values():
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


def test_candidate_drafts_bind_three_exact_word5_evidence_and_manual_relations(
    tmp_path, monkeypatch
):
    source, _manifest, works, *_ = _catalog(tmp_path, monkeypatch)
    ratios = ((2019, 2090), (5093, 5295), (3542, 3637))
    overlaps = tuple(
        ContentOverlap(
            left_work=member,
            right_work=prep.R1_EXCLUDED_WORK_ID,
            kind="word5_asymmetric_containment",
            containment=numerator / denominator,
            evidence=f"{numerator}/{denominator} unique word-5-grams",
        )
        for member, (numerator, denominator) in zip(
            prep.R1_COLLECTION_MEMBERS, ratios, strict=True
        )
    )
    monkeypatch.setattr(
        prep,
        "find_cross_work_content_overlaps",
        lambda *args, **kwargs: overlaps,
    )

    drafts, evidence = prep._candidate_drafts(source, works)

    assert len(drafts) == 6
    automatic = [
        row for row in drafts
        if row.edge_type == "word5_asymmetric_containment"
    ]
    manual = [
        row for row in drafts if row.edge_type == "collection_member"
    ]
    assert len(automatic) == len(manual) == 3
    assert all(row.origin == "automatic" for row in automatic)
    assert all(row.origin == "manual" for row in manual)
    assert all(
        row.disposition == "same_component"
        and row.manual_disposition is not None
        for row in drafts
    )
    assert tuple(
        row["exact_containment_evidence"] for row in evidence
    ) == tuple(
        f"{numerator}/{denominator} unique word-5-grams"
        for numerator, denominator in ratios
    )
    assert all(
        row["owner_selected_relation"] == "collection_member"
        and row["owner_selected_corpus_action"]
        == "exclude_collection_retain_constituent"
        for row in evidence
    )


def test_unreviewed_or_missing_candidate_blocks_packet_preparation(
    tmp_path, monkeypatch
):
    source, _manifest, works, *_ = _catalog(tmp_path, monkeypatch)
    monkeypatch.setattr(
        prep,
        "find_cross_work_content_overlaps",
        lambda *args, **kwargs: (),
    )

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="differs from the owner-reviewed three pairs",
    ):
        prep._candidate_drafts(source, works)


def test_packet_is_path_independent_and_create_if_absent(
    tmp_path, monkeypatch
):
    source, manifest = _source_fixture(tmp_path / "inputs")
    _pin_manifest(monkeypatch, manifest)
    cfg = load_config("configs/default.yaml")
    monkeypatch.setattr(prep, "_canonical_rows", _fake_canonical_rows)
    monkeypatch.setattr(
        prep,
        "_candidate_drafts",
        lambda source_root, works: ((), ()),
    )
    first_parent = (
        tmp_path / "first" / "exploratory" / "lobo_vnext" / "packets"
    )
    second_parent = (
        tmp_path / "second" / "exploratory" / "lobo_vnext" / "packets"
    )

    first = prep.prepare_r1_packet(
        source_root=source,
        legacy_source_manifest=manifest,
        output_parent=first_parent,
        cfg=cfg,
    )
    second = prep.prepare_r1_packet(
        source_root=source,
        legacy_source_manifest=manifest,
        output_parent=second_parent,
        cfg=cfg,
    )

    assert first.root.name == second.root.name
    assert first.packet_manifest == second.packet_manifest
    assert first.corpus_manifest == second.corpus_manifest
    assert first.representation_receipt == second.representation_receipt
    assert first.packet_manifest.selected_work_count == 136
    assert first.packet_manifest.confirmatory_authorized is False
    assert all(
        len(component.work_ids) == 1
        for component in first.content_manifest.components
    )
    assert first.fold_manifest.mode == "isolated"
    assert all(
        not plan.splits for plan in first.primary_inner_cv_plan.plans
    )
    assert all(
        not plan.splits for plan in first.baseline_inner_cv_plan.plans
    )

    with pytest.raises(
        prep.R1PacketPreparationError, match="already exists"
    ):
        prep.prepare_r1_packet(
            source_root=source,
            legacy_source_manifest=manifest,
            output_parent=first_parent,
            cfg=cfg,
        )


def test_packet_output_must_be_explicit_exploratory_namespace(
    tmp_path, monkeypatch
):
    source, manifest = _source_fixture(tmp_path / "inputs")
    _pin_manifest(monkeypatch, manifest)
    monkeypatch.setattr(
        prep,
        "_candidate_drafts",
        lambda source_root, works: ((), ()),
    )

    with pytest.raises(
        prep.R1PacketPreparationError,
        match="explicit exploratory/.+ namespace",
    ):
        prep.prepare_r1_packet(
            source_root=source,
            legacy_source_manifest=manifest,
            output_parent=tmp_path / "packet-output",
            cfg=load_config("configs/default.yaml"),
        )
