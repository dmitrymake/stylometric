from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylo.domain.lobo_vnext import (
    LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION,
    ContentCandidate,
    ContentComponent,
    ContentComponentManifest,
    CorpusVNextManifest,
    InferenceSpec,
    ModelSpec,
    VNextContractError,
    WorkIdentity,
    build_corpus_vnext_manifest,
    build_fold_manifest,
    build_inner_cv_plan,
    canonical_sha256,
    discover_literal_byte_content_candidates,
    load_corpus_vnext_manifest,
    loads_content_component_manifest,
    loads_corpus_vnext_manifest,
    loads_inference_spec,
    recompute_automatic_content_candidates,
    verify_raw_inventory,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _work(
    work_id: str,
    *,
    kind: str = "work",
    author: str | None = None,
    edition: str | None = None,
) -> WorkIdentity:
    author_id = author or work_id.split("/", 1)[0]
    leaf = work_id.split("/", 1)[-1]
    return WorkIdentity.from_dict(
        {
            "work_id": work_id,
            "author_id": author_id,
            "edition_id": edition or f"edition-{leaf}",
            "source_id": f"source-{leaf}",
            "work_kind": kind,
            "raw_paths": [f"{work_id}.txt"],
        }
    )


def _write_works(root: Path, works: tuple[WorkIdentity, ...], *, same=()) -> None:
    same_ids = set(same)
    for index, work in enumerate(works):
        path = root / work.raw_paths[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"literal duplicate" if work.work_id in same_ids else (
            f"literal-{index}-{work.work_id}".encode()
        )
        path.write_bytes(payload)


def _singleton_content(
    works: tuple[WorkIdentity, ...],
) -> ContentComponentManifest:
    return ContentComponentManifest.build(
        automatic_candidate_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        works=works,
        components=tuple(
            ContentComponent(f"component-{work.work_id}", (work.work_id,))
            for work in works
        ),
        candidates=(),
    )


def _corpus(
    root: Path,
    works: tuple[WorkIdentity, ...],
    content: ContentComponentManifest,
    *,
    generation_id: str = "synthetic-generation-1",
) -> CorpusVNextManifest:
    return build_corpus_vnext_manifest(
        root,
        corpus_kind="synthetic_fixture",
        generation_id=generation_id,
        approved_for_exploratory=True,
        owner_selected=False,
        author_ids=tuple(sorted({work.author_id for work in works})),
        works=works,
        canonical_model_row_digest=_digest("canonical-model-rows"),
        chunker_policy_version="synthetic-chunker.v1",
        canonicalizer_policy_version="literal-canonicalizer.v1",
        content_policy_version="literal-content.v1",
        content_component_manifest_digest=content.self_hash,
    )


def _model(
    *,
    inner: bool = False,
    component_aware: bool = False,
) -> ModelSpec:
    return ModelSpec.build(
        model_id="synthetic-centroid",
        family="centroid",
        features=("literal-counts",),
        weighting="work-balanced",
        hyperparameters={"alpha": 1.0},
        seeds={"model": 7},
        requires_inner_cv=inner,
        inner_cv_splits=2 if inner else None,
        supports_component_aware_inner_cv=component_aware,
        approved_for_exploratory=True,
        owner_selected=False,
    )


def _inference() -> InferenceSpec:
    return InferenceSpec.build(
        primary_metric="book_accuracy",
        primary_uncertainty="author_clustered_percentile_bootstrap",
        secondary_metrics=("macro_f1", "top2", "per_author"),
        macro_f1_uncertainty="point_only",
        bootstrap_seed=42,
        bootstrap_iterations=100,
        confidence_level=0.95,
        approved_for_exploratory=True,
        owner_selected=False,
    )


def _rehash(raw: dict) -> dict:
    raw["self_hash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    return raw


def test_corpus_manifest_is_strict_self_hashed_and_path_independent(tmp_path):
    works = (_work("a/a1"), _work("b/b1"))
    _write_works(tmp_path, works)
    content = _singleton_content(works)
    manifest = _corpus(tmp_path, works, content)
    path = tmp_path.parent / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    loaded = load_corpus_vnext_manifest(path)

    assert loaded == manifest
    assert all(not Path(row.relative_path).is_absolute() for row in loaded.raw_inventory)
    assert tuple(row.relative_path for row in loaded.raw_inventory) == (
        "a/a1.txt",
        "b/b1.txt",
    )
    assert verify_raw_inventory(tmp_path, loaded) is loaded
    assert loaded.assert_exploratory_authorized(synthetic_fixture=True) is loaded


def test_corpus_loader_rejects_duplicate_extra_and_absolute_path_keys(tmp_path):
    works = (_work("a/a1"), _work("b/b1"))
    _write_works(tmp_path, works)
    manifest = _corpus(tmp_path, works, _singleton_content(works))
    text = json.dumps(manifest.to_dict())
    duplicate = '{"schema_version":"duplicate",' + text[1:]
    with pytest.raises(VNextContractError, match="duplicate object key"):
        loads_corpus_vnext_manifest(duplicate)

    extra = manifest.to_dict()
    extra["unexpected"] = "rehashed-extra"
    _rehash(extra)
    with pytest.raises(VNextContractError, match="keys must be exact"):
        loads_corpus_vnext_manifest(json.dumps(extra))

    absolute = manifest.to_dict()
    absolute["raw_inventory"][0]["relative_path"] = "/host/private/a1.txt"
    _rehash(absolute)
    with pytest.raises(VNextContractError, match="relative path"):
        loads_corpus_vnext_manifest(json.dumps(absolute))


def test_raw_byte_mutation_changes_identity_even_when_canonical_rows_do_not(
    tmp_path,
):
    works = (_work("a/a1"), _work("b/b1"))
    _write_works(tmp_path, works)
    content = _singleton_content(works)
    first = _corpus(tmp_path, works, content)
    path = tmp_path / "a/a1.txt"
    path.write_bytes(path.read_bytes() + b" ")
    second = _corpus(tmp_path, works, content)

    assert first.canonical_model_row_digest == second.canonical_model_row_digest
    assert first.generation_id == second.generation_id
    assert first.raw_inventory != second.raw_inventory
    assert first.self_hash != second.self_hash


def test_raw_preflight_rejects_missing_extra_tampered_and_symlink(tmp_path):
    works = (_work("a/a1"), _work("b/b1"))
    _write_works(tmp_path, works)
    manifest = _corpus(tmp_path, works, _singleton_content(works))
    a1 = tmp_path / "a/a1.txt"
    original = a1.read_bytes()

    a1.write_bytes(b"x" * len(original))
    with pytest.raises(VNextContractError, match="SHA-256 mismatch"):
        verify_raw_inventory(tmp_path, manifest)
    a1.write_bytes(original)

    extra = tmp_path / "extra.txt"
    extra.write_bytes(b"extra")
    with pytest.raises(VNextContractError, match="extra="):
        verify_raw_inventory(tmp_path, manifest)
    extra.unlink()

    a1.unlink()
    with pytest.raises(VNextContractError, match="missing="):
        verify_raw_inventory(tmp_path, manifest)
    a1.symlink_to(tmp_path / "b/b1.txt")
    with pytest.raises(VNextContractError, match="symlink rejected"):
        verify_raw_inventory(tmp_path, manifest)


def test_automatic_literal_candidates_are_recomputed_and_omission_blocks(
    tmp_path,
):
    works = (_work("a/a1"), _work("a/a2"), _work("b/b1"))
    _write_works(tmp_path, works, same=("a/a1", "a/a2"))
    omitted = _singleton_content(works)
    provisional_corpus = _corpus(tmp_path, works, omitted)

    discovered = discover_literal_byte_content_candidates(
        tmp_path, provisional_corpus
    )

    assert len(discovered) == 1
    assert discovered[0].edge_type == "exact_duplicate"
    with pytest.raises(VNextContractError, match="automatic content candidates"):
        recompute_automatic_content_candidates(
            tmp_path, provisional_corpus, omitted
        )

    bound = ContentComponentManifest.build(
        automatic_candidate_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        works=works,
        components=(
            ContentComponent("component-a-duplicate", ("a/a1", "a/a2")),
            ContentComponent("component-b", ("b/b1",)),
        ),
        candidates=discovered,
    )
    corpus = _corpus(tmp_path, works, bound)
    assert recompute_automatic_content_candidates(
        tmp_path, corpus, bound
    ) == discovered


def test_literal_short_in_long_is_a_contains_candidate(tmp_path):
    works = (_work("a/short"), _work("a/long"), _work("b/b1"))
    _write_works(tmp_path, works)
    (tmp_path / "a/short.txt").write_bytes(b"literal body")
    (tmp_path / "a/long.txt").write_bytes(b"prefix literal body suffix")
    content = _singleton_content(works)
    corpus = _corpus(tmp_path, works, content)

    candidates = discover_literal_byte_content_candidates(tmp_path, corpus)

    assert [(row.edge_type, row.left_work_id, row.right_work_id) for row in candidates] == [
        ("contains", "a/short", "a/long")
    ]


def test_content_manifest_rejects_unresolved_and_bad_partition(tmp_path):
    works = (_work("a/a1"), _work("a/a2"))
    unresolved = ContentCandidate(
        "candidate-1",
        "a/a1",
        "a/a2",
        "manual",
        "manual",
        "unresolved",
        _digest("evidence"),
    )
    with pytest.raises(VNextContractError, match="unresolved content candidate"):
        ContentComponentManifest.build(
            automatic_candidate_policy_version=(
                LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
            ),
            works=works,
            components=(
                ContentComponent("component-a1", ("a/a1",)),
                ContentComponent("component-a2", ("a/a2",)),
            ),
            candidates=(unresolved,),
        )

    raw = _singleton_content(works).to_dict()
    raw["components"][1]["work_ids"] = ["a/a1"]
    _rehash(raw)
    with pytest.raises(VNextContractError, match="more than one component"):
        loads_content_component_manifest(json.dumps(raw), works=works)


@pytest.mark.parametrize(
    ("works", "candidate", "message"),
    [
        (
            (
                _work("a/a1", author="a", edition="same-edition"),
                _work("b/b1", author="b", edition="same-edition"),
            ),
            ContentCandidate(
                "bad-edition",
                "a/a1",
                "b/b1",
                "edition_of",
                "manual",
                "same_component",
                _digest("bad-edition"),
            ),
            "invalid edition_of",
        ),
        (
            (_work("a/member"), _work("a/not_collection")),
            ContentCandidate(
                "bad-collection",
                "a/member",
                "a/not_collection",
                "collection_member",
                "manual",
                "same_component",
                _digest("bad-collection"),
            ),
            "invalid collection_member",
        ),
    ],
)
def test_invalid_edition_and_collection_relations_are_rejected(
    works,
    candidate,
    message,
):
    with pytest.raises(VNextContractError, match=message):
        ContentComponentManifest.build(
            automatic_candidate_policy_version=(
                LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
            ),
            works=works,
            components=(
                ContentComponent(
                    "component-joint",
                    tuple(sorted(work.work_id for work in works)),
                ),
            ),
            candidates=(candidate,),
        )


def test_fold_manifest_freezes_p_m_and_excludes_train_only_singleton(tmp_path):
    works = (
        _work("a/a1"),
        _work("a/a2"),
        _work("b/b1"),
        _work("b/b2"),
        _work("singleton/s1"),
    )
    _write_works(tmp_path, works)
    content = _singleton_content(works)
    corpus = _corpus(tmp_path, works, content)

    manifest = build_fold_manifest(corpus, content, mode="isolated")

    assert manifest.probability_class_order == ("a", "b", "singleton")
    assert manifest.metric_label_order == ("a", "b")
    assert tuple(fold.test_work_id for fold in manifest.folds) == (
        "a/a1",
        "a/a2",
        "b/b1",
        "b/b2",
    )
    assert all(not fold.purged_work_ids for fold in manifest.folds)
    assert all(fold.test_work_id not in fold.train_work_ids for fold in manifest.folds)


def _multi_component_fixture(tmp_path):
    works = tuple(
        _work(work_id)
        for work_id in (
            "a/a1",
            "a/a2",
            "a/a3",
            "a/a4",
            "b/b1",
            "b/b2",
            "b/b3",
            "b/b4",
        )
    )
    _write_works(tmp_path, works)
    links = (
        ContentCandidate(
            "manual-a-pair",
            "a/a1",
            "a/a2",
            "manual",
            "manual",
            "same_component",
            _digest("manual-a-pair"),
        ),
        ContentCandidate(
            "manual-b-pair",
            "b/b1",
            "b/b2",
            "manual",
            "manual",
            "same_component",
            _digest("manual-b-pair"),
        ),
    )
    content = ContentComponentManifest.build(
        automatic_candidate_policy_version=(
            LITERAL_BYTES_AUTOMATIC_CANDIDATE_POLICY_VERSION
        ),
        works=works,
        components=(
            ContentComponent("component-a12", ("a/a1", "a/a2")),
            ContentComponent("component-a3", ("a/a3",)),
            ContentComponent("component-a4", ("a/a4",)),
            ContentComponent("component-b12", ("b/b1", "b/b2")),
            ContentComponent("component-b3", ("b/b3",)),
            ContentComponent("component-b4", ("b/b4",)),
        ),
        candidates=links,
    )
    return works, content, _corpus(tmp_path, works, content)


def test_isolated_requires_singleton_components_and_purged_removes_whole_component(
    tmp_path,
):
    _, content, corpus = _multi_component_fixture(tmp_path)
    with pytest.raises(VNextContractError, match="every component has one work"):
        build_fold_manifest(corpus, content, mode="isolated")

    manifest = build_fold_manifest(corpus, content, mode="purged")
    a1 = next(fold for fold in manifest.folds if fold.test_work_id == "a/a1")
    assert a1.content_component_id == "component-a12"
    assert a1.purged_work_ids == ("a/a2",)
    assert "a/a1" not in a1.train_work_ids
    assert "a/a2" not in a1.train_work_ids


def test_purged_inner_cv_fails_closed_or_uses_whole_components(tmp_path):
    _, content, corpus = _multi_component_fixture(tmp_path)
    with pytest.raises(VNextContractError, match="lacks component-aware inner CV"):
        build_inner_cv_plan(
            build_fold_manifest(corpus, content, mode="purged"),
            corpus,
            content,
            _model(inner=True, component_aware=False),
        )

    outer = build_fold_manifest(corpus, content, mode="purged")
    inner = build_inner_cv_plan(
        outer,
        corpus,
        content,
        _model(inner=True, component_aware=True),
    )
    baseline_inner = build_inner_cv_plan(outer, corpus, content, _model())
    assert baseline_inner.fold_manifest_digest == outer.self_hash
    assert all(not plan.splits for plan in baseline_inner.plans)
    assert "model_spec_digest" not in outer.to_dict()
    assert all("splits" not in fold.to_dict() for fold in outer.folds)
    component_members = {
        component.component_id: set(component.work_ids)
        for component in content.components
    }
    for fold, fold_plan in zip(outer.folds, inner.plans, strict=True):
        assert fold_plan.fold_id == fold.fold_id
        assert fold_plan.fold_spec_digest == fold.self_hash
        assert len(fold_plan.splits) == 2
        for split in fold_plan.splits:
            validation = set(split.validation_work_ids)
            for component_id in split.validation_component_ids:
                expected = component_members[component_id] & set(fold.train_work_ids)
                assert expected <= validation
                assert not expected & set(split.train_work_ids)


def test_model_and_inference_specs_are_strict_and_synthetic_owner_false():
    model = _model()
    inference = _inference()
    assert model.assert_exploratory_authorized(synthetic_fixture=True) is model
    assert (
        inference.assert_exploratory_authorized(synthetic_fixture=True)
        is inference
    )
    with pytest.raises(VNextContractError, match="owner_selected must be true"):
        model.assert_exploratory_authorized(synthetic_fixture=False)

    raw = inference.to_dict()
    raw["bootstrap_iterations"] = True
    _rehash(raw)
    with pytest.raises(VNextContractError, match="exact integer"):
        loads_inference_spec(json.dumps(raw))


def test_real_corpus_cannot_be_mislabeled_as_synthetic(tmp_path):
    works = (_work("a/a1"), _work("b/b1"))
    _write_works(tmp_path, works)
    content = _singleton_content(works)
    raw = _corpus(tmp_path, works, content).to_dict()
    raw["corpus_kind"] = "real_corpus"
    raw["owner_selected"] = True
    _rehash(raw)
    real = CorpusVNextManifest.from_dict(raw)

    with pytest.raises(VNextContractError, match="corpus_kind"):
        real.assert_exploratory_authorized(synthetic_fixture=True)
