"""Synthetic fail-closed tests for the audit-corpus contract, immutable builder, and RuAA subset.

Gate 2 of research/work_balanced/paired_audit_protocol.md (§1.2/§1.3/§1.4/§1.5). Every test builds a
toy work-balanced corpus in a tmp dir; none touches the real closed corpus.
"""
from __future__ import annotations

import pathlib

import pytest

from stylo import workdoc as wd
from stylo.config import load_config, with_overrides
from stylo.corpus import load_dataset
from stylo.eval.provenance import RunContract, verify_dataset_against_disk
from stylo.eval.work_weighting import WORK_BALANCED
from stylo.jsonio import dump_strict, load_strict
from stylo.workdoc import chunker_config_hash, load_work_balanced_dataset
from stylo.eval.paired_audit import corpus as ac
from stylo.eval.paired_audit import semantic_parity as sp
from stylo.eval.paired_audit import work_subset as ws

CFG = load_config()
_CHASH = chunker_config_hash(CFG)


def _make_wb_corpus(tmp: pathlib.Path, spec: dict[str, dict[str, list[str]]], *, name="frags"):
    """Build input_clean sources + a work-balanced frags tree with valid manifests.

    ``spec`` = {author: {book: [chunk_text, ...]}}. Chunk files are named ``c_000.txt`` in order, so
    the legacy sorted-filename order matches the manifest ordinal order (the parity invariant).
    Returns (frags_root, input_clean_root).
    """
    frags, ic = tmp / name, tmp / f"{name}_clean"
    for author, books in spec.items():
        for book, texts in books.items():
            src = ic / author / f"{book}.txt"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(f"source {author} {book}", encoding="utf-8")
            wdir = frags / author / book
            wdir.mkdir(parents=True, exist_ok=True)
            names = [f"c_{i:03d}.txt" for i in range(len(texts))]
            for nm, tx in zip(names, texts):
                (wdir / nm).write_text(tx, encoding="utf-8")
            m = wd.build_work_manifest(
                f"{author}/{book}", author, texts, names,
                provenance_sha256=wd.source_provenance_sha256(src),
                chunker_config_hash=_CHASH, overlap=0.0,
            )
            dump_strict(m.to_dict(), wdir / wd.MANIFEST_NAME, trailing_newline=False)
    return frags, ic


def _default_spec(n=4):
    """Toy corpus: 3 authors × 2 works × ``n`` distinct chunks (≥10 total so the loaders' minimum
    fragment floor is met even for small whole-work subsets)."""
    authors = {"alpha": ["a1", "a2"], "beta": ["b1", "b2"], "gamma": ["g1", "g2"]}
    return {a: {b: [f"{a} {b} chunk {i} more words here" for i in range(n)] for b in books}
            for a, books in authors.items()}


def _toy_anchor(frags):
    return load_dataset(frags).provenance.rows_digest


# ── §1.2 semantic parity + legacy anchor ─────────────────────────────────────
class TestSemanticParity:
    def test_legacy_and_wb_loads_are_parity_equal(self, tmp_path):
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        legacy = load_dataset(frags)
        wb = load_work_balanced_dataset(frags, cfg=CFG, input_clean_root=ic)
        digest = sp.assert_semantic_parity(legacy, wb)
        assert digest == sp.dataset_semantic_digest(legacy) == sp.dataset_semantic_digest(wb)

    def test_legacy_anchor_ok_and_mismatch(self, tmp_path):
        frags, _ = _make_wb_corpus(tmp_path, _default_spec())
        legacy = load_dataset(frags)
        anchor = _toy_anchor(frags)
        assert sp.verify_legacy_anchor(legacy, expected=anchor) == anchor
        with pytest.raises(sp.SemanticParityError):
            sp.verify_legacy_anchor(legacy, expected="0" * 64)

    def test_parity_mismatch_between_different_corpora(self, tmp_path):
        frags_a, ic_a = _make_wb_corpus(tmp_path, _default_spec(), name="a")
        spec_b = _default_spec()
        spec_b["alpha"]["a1"][0] = "alpha one CHANGED"
        frags_b, ic_b = _make_wb_corpus(tmp_path, spec_b, name="b")
        legacy_a = load_dataset(frags_a)
        wb_b = load_work_balanced_dataset(frags_b, cfg=CFG, input_clean_root=ic_b)
        with pytest.raises(sp.SemanticParityError):
            sp.assert_semantic_parity(legacy_a, wb_b)

    def test_row_order_parity_invariant_catches_reordering(self, tmp_path):
        """The core §1.2 property: legacy orders rows by sorted filename, WB by manifest ordinal. A
        work whose manifest ordinal order differs from its sorted-filename order must break parity."""
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        author, book = "alpha", "a1"
        wdir = frags / author / book
        for f in list(wdir.glob("*.txt")):
            f.unlink()
        (wdir / wd.MANIFEST_NAME).unlink()
        # manifest ordinal order (zzz, aaa, mmm, bbb) != sorted-filename order (aaa, bbb, mmm, zzz)
        names = ["c_zzz.txt", "c_aaa.txt", "c_mmm.txt", "c_bbb.txt"]
        texts = ["zzz text one", "aaa text two", "mmm text three", "bbb text four"]
        for nm, tx in zip(names, texts):
            (wdir / nm).write_text(tx, encoding="utf-8")
        src = ic / author / f"{book}.txt"
        m = wd.build_work_manifest(f"{author}/{book}", author, texts, names,
                                   provenance_sha256=wd.source_provenance_sha256(src),
                                   chunker_config_hash=_CHASH, overlap=0.0)
        dump_strict(m.to_dict(), wdir / wd.MANIFEST_NAME, trailing_newline=False)
        legacy = load_dataset(frags)
        wb = load_work_balanced_dataset(frags, cfg=CFG, input_clean_root=ic)
        with pytest.raises(sp.SemanticParityError):
            sp.assert_semantic_parity(legacy, wb)


class TestSemanticRowDigest:
    """Focused unit tests pinning the low-level §1.2 digest guard surface."""

    BASE = dict(texts=["a", "b", "c"], y=[0, 0, 1], groups=["x/w", "x/w", "z/v"],
                authors=["x", "z"])

    def test_author_permutation_changes_digest(self):
        a = sp.semantic_row_digest(**self.BASE)
        b = sp.semantic_row_digest(texts=["a", "b", "c"], y=[1, 1, 0],
                                   groups=["x/w", "x/w", "z/v"], authors=["z", "x"])
        assert a != b

    def test_bool_and_float_labels_rejected(self):
        with pytest.raises(sp.SemanticParityError):
            sp.semantic_row_digest(texts=["a"], y=[True], groups=["x/w"], authors=["x"])
        with pytest.raises(sp.SemanticParityError):
            sp.semantic_row_digest(texts=["a"], y=[0.0], groups=["x/w"], authors=["x"])

    def test_non_str_text_or_group_rejected(self):
        with pytest.raises(sp.SemanticParityError):
            sp.semantic_row_digest(texts=[123], y=[0], groups=["x/w"], authors=["x"])
        with pytest.raises(sp.SemanticParityError):
            sp.semantic_row_digest(texts=["a"], y=[0], groups=[123], authors=["x"])

    def test_duplicate_authors_and_length_mismatch_rejected(self):
        with pytest.raises(sp.SemanticParityError):
            sp.semantic_row_digest(texts=["a"], y=[0], groups=["x/w"], authors=["x", "x"])
        with pytest.raises(sp.SemanticParityError):
            sp.semantic_row_digest(texts=["a", "b"], y=[0], groups=["x/w"], authors=["x"])


# ── §1.3 immutable builder + published-root loader ───────────────────────────
class TestBuilder:
    def _build(self, tmp_path, **kw):
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        return frags, ic, ac.build_audit_corpus(
            source_frags_root=frags, input_clean_root=ic, cfg=CFG,
            audit_parent=tmp_path / "audit", legacy_anchor=_toy_anchor(frags), **kw)

    def test_build_publishes_immutable_root_and_pointer(self, tmp_path):
        frags, ic, root = self._build(tmp_path)
        assert root.parent == tmp_path / "audit"
        assert root.name == ac._tree_content_digest(root, ac._CONTENT_SUBDIRS)
        pointer = load_strict(tmp_path / "audit" / ac.CURRENT_NAME)
        assert pointer["version"] == root.name
        assert ac.resolve_current_root(tmp_path / "audit") == root

    def test_build_mints_manifests_only_inside_staging_for_legacy_source(self, tmp_path):
        """The real frozen source is legacy-recursive and intentionally has no per-work manifests.

        Preparation must make the work-balanced view atomically in the immutable copy; it must never
        mutate ``data/frags_train`` (or the synthetic source standing in for it here).
        """
        from stylo.pipeline.split import resolve_fragment_snapshot, run as split_corpus

        ic = tmp_path / "legacy_clean"
        data = tmp_path / "legacy_data"
        replay_cfg = with_overrides(CFG, {
            "paths.input_clean": str(ic), "paths.data": str(data),
            "chunking.chunk_size": 20, "chunking.min_words": 5,
        })
        for author in ("alpha", "beta", "gamma"):
            for book in ("one", "two"):
                path = ic / author / f"{book}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(" ".join(
                    f"Предложение {i} книги {book} автора {author} содержит достаточно обычных слов."
                    for i in range(18)
                ), encoding="utf-8")
        split_corpus(replay_cfg)
        frags = resolve_fragment_snapshot(
            data, require_versioned=True
        ).train_root
        for path in frags.rglob(wd.MANIFEST_NAME):
            path.unlink()
        assert not list(frags.rglob(wd.MANIFEST_NAME))

        root = ac.build_audit_corpus(
            source_frags_root=frags, input_clean_root=ic, cfg=replay_cfg,
            audit_parent=tmp_path / "audit", legacy_anchor=_toy_anchor(frags),
            expected_n_works=6,
        )

        assert not list(frags.rglob(wd.MANIFEST_NAME))
        assert len(list((root / ac.FRAGS_SUBDIR).rglob(wd.MANIFEST_NAME))) == 6
        legacy = load_dataset(root / ac.FRAGS_SUBDIR)
        wb = load_work_balanced_dataset(root / ac.FRAGS_SUBDIR, cfg=replay_cfg,
                                        input_clean_root=root / ac.INPUT_CLEAN_SUBDIR)
        assert sp.assert_semantic_parity(legacy, wb)

    def test_rejects_output_inside_source_and_symlinked_source_root(self, tmp_path):
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        with pytest.raises(ac.AuditCorpusError):
            ac.build_audit_corpus(
                source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                audit_parent=frags / "audit-output", legacy_anchor=_toy_anchor(frags),
            )

        linked = tmp_path / "linked-frags"
        linked.symlink_to(frags, target_is_directory=True)
        with pytest.raises(ac.AuditCorpusError):
            ac.build_audit_corpus(
                source_frags_root=linked, input_clean_root=ic, cfg=CFG,
                audit_parent=tmp_path / "safe-output", legacy_anchor=_toy_anchor(frags),
            )

    def test_audit_dataset_is_work_balanced_for_every_cell(self, tmp_path):
        # §1.4: the audit dataset is the same WB-manifest dataset for A0..A4 (no legacy-loaded A0)
        _, _, root = self._build(tmp_path)
        ds = ac.load_audit_dataset(root, CFG)
        assert ds.provenance.loader_kind == "work_balanced_manifest"
        assert ac.verify_audit_dataset(ds) == sp.dataset_semantic_digest(ds)

    def test_published_root_legacy_and_wb_at_parity(self, tmp_path):
        # the §1.2 anchor/parity proof uses the raw loaders directly (not the confirmatory loader)
        from stylo.corpus import load_dataset
        from stylo.workdoc import load_work_balanced_dataset
        _, _, root = self._build(tmp_path)
        rf, rc = root / ac.FRAGS_SUBDIR, root / ac.INPUT_CLEAN_SUBDIR
        legacy = load_dataset(rf)
        wb = load_work_balanced_dataset(rf, cfg=CFG, input_clean_root=rc)
        assert sp.assert_semantic_parity(legacy, wb)

    def test_audit_only_verifier_rejects_legacy_dataset(self, tmp_path):
        # a runner must never feed the A0 cell a legacy-recursive-loaded dataset (§1.4)
        from stylo.corpus import load_dataset
        _, _, root = self._build(tmp_path)
        legacy = load_dataset(root / ac.FRAGS_SUBDIR)
        with pytest.raises(ac.AuditCorpusError):
            ac.verify_audit_dataset(legacy)

    def test_audit_only_verifier_rejects_mutated_arrays(self, tmp_path):
        _, _, root = self._build(tmp_path)
        wb = ac.load_audit_dataset(root, CFG)
        wb.texts[0] = "forged content after load"
        with pytest.raises(ac.AuditCorpusError):
            ac.verify_audit_dataset(wb)

    def test_idempotent_rebuild_reuses_same_root(self, tmp_path):
        frags, ic, root = self._build(tmp_path)
        again = ac.build_audit_corpus(
            source_frags_root=frags, input_clean_root=ic, cfg=CFG,
            audit_parent=tmp_path / "audit", legacy_anchor=_toy_anchor(frags))
        assert again == root

    def test_selection_subset_and_expected_count(self, tmp_path):
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        root = ac.build_audit_corpus(
            source_frags_root=frags, input_clean_root=ic, cfg=CFG,
            audit_parent=tmp_path / "audit", legacy_anchor=_toy_anchor(frags),
            work_ids=["alpha/a1", "beta/b1", "gamma/g1"], expected_n_works=3)
        wb = ac.load_audit_dataset(root, CFG)
        assert sorted(set(str(g) for g in wb.groups)) == ["alpha/a1", "beta/b1", "gamma/g1"]

    def test_wrong_expected_count_fails(self, tmp_path):
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        with pytest.raises(ac.AuditCorpusError):
            ac.build_audit_corpus(
                source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                audit_parent=tmp_path / "audit", legacy_anchor=_toy_anchor(frags),
                expected_n_works=99)


# ── §1.3 fail-closed corpus contracts ────────────────────────────────────────
class TestBuilderFailClosed:
    def _src(self, tmp_path):
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        return frags, ic, _toy_anchor(frags)

    def test_legacy_anchor_mismatch_aborts_build(self, tmp_path):
        frags, ic, _ = self._src(tmp_path)
        with pytest.raises(sp.SemanticParityError):
            ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                  audit_parent=tmp_path / "audit", legacy_anchor="0" * 64)

    def test_missing_work_selection_fails(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        with pytest.raises(ac.AuditCorpusError):
            ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                  audit_parent=tmp_path / "audit", legacy_anchor=anchor,
                                  work_ids=["alpha/a1", "nope/x"])

    def test_duplicate_work_selection_fails(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        with pytest.raises(ac.AuditCorpusError):
            ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                  audit_parent=tmp_path / "audit", legacy_anchor=anchor,
                                  work_ids=["alpha/a1", "alpha/a1"])

    def test_mutated_chunk_bytes_abort_build_via_anchor_drift(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        (frags / "alpha" / "a1" / "c_000.txt").write_text("tampered different text", encoding="utf-8")
        # the pristine legacy anchor drifts first, so the byte mutation aborts the build
        with pytest.raises(sp.SemanticParityError):
            ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                  audit_parent=tmp_path / "audit", legacy_anchor=anchor)

    def test_extra_stray_chunk_aborts_build_via_anchor_drift(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        (frags / "alpha" / "a1" / "stray.txt").write_text("extra content here", encoding="utf-8")
        # legacy load counts the stray .txt as an extra row -> anchor drifts
        with pytest.raises(sp.SemanticParityError):
            ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                  audit_parent=tmp_path / "audit", legacy_anchor=anchor)

    def test_wb_manifest_guard_catches_byte_mutation_directly(self, tmp_path):
        # exercise the WB per-chunk text_sha256 guard on its own (no anchor involved)
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        (frags / "alpha" / "a1" / "c_000.txt").write_text("byte mutated content", encoding="utf-8")
        with pytest.raises(wd.ManifestError):
            load_work_balanced_dataset(frags, cfg=CFG, input_clean_root=ic)

    def test_wb_manifest_guard_catches_stray_chunk_directly(self, tmp_path):
        # exercise the WB manifest/file bijection guard on its own (no anchor involved)
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        (frags / "alpha" / "a1" / "stray.txt").write_text("extra content here", encoding="utf-8")
        with pytest.raises(wd.ManifestError):
            load_work_balanced_dataset(frags, cfg=CFG, input_clean_root=ic)

    def test_stray_toplevel_file_in_published_root_rejected(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        root = ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                     audit_parent=tmp_path / "audit", legacy_anchor=anchor)
        (root / "SMUGGLED.txt").write_text("not part of the corpus", encoding="utf-8")
        with pytest.raises(ac.AuditCorpusError):
            ac.verify_published_corpus(root)

    def test_corpus_manifest_self_hash_tamper_rejected(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        root = ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                     audit_parent=tmp_path / "audit", legacy_anchor=anchor)
        body = load_strict(root / ac.CORPUS_MANIFEST_NAME)
        body["n_works"] = body["n_works"] + 1          # tamper a field, leave self_hash stale
        dump_strict(body, root / ac.CORPUS_MANIFEST_NAME, trailing_newline=True)
        with pytest.raises(ac.AuditCorpusError):
            ac.verify_published_corpus(root)

    def test_current_pointer_to_nonexistent_digest_rejected(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                              audit_parent=tmp_path / "audit", legacy_anchor=anchor)
        dump_strict({"schema": ac._CORPUS_DIGEST_VERSION, "version": "a" * 64},
                    tmp_path / "audit" / ac.CURRENT_NAME, trailing_newline=True)
        with pytest.raises(ac.AuditCorpusError):
            ac.resolve_current_root(tmp_path / "audit")

    def test_empty_pointer_token_rejected(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                              audit_parent=tmp_path / "audit", legacy_anchor=anchor)
        dump_strict({"schema": ac._CORPUS_DIGEST_VERSION, "version": ""},
                    tmp_path / "audit" / ac.CURRENT_NAME, trailing_newline=True)
        with pytest.raises(ac.AuditCorpusError):
            ac.resolve_current_root(tmp_path / "audit")

    def test_pointer_schema_mismatch_rejected(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        root = ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                     audit_parent=tmp_path / "audit", legacy_anchor=anchor)
        dump_strict({"schema": "wrong", "version": root.name},
                    tmp_path / "audit" / ac.CURRENT_NAME, trailing_newline=True)
        with pytest.raises(ac.AuditCorpusError):
            ac.resolve_current_root(tmp_path / "audit")

    def test_conflicting_immutable_root_is_fatal(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        root = ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                     audit_parent=tmp_path / "audit", legacy_anchor=anchor)
        # tamper a published chunk file (same dir name, different content)
        victim = next((root / ac.FRAGS_SUBDIR).rglob("c_000.txt"))
        victim.write_text("tampered inside the immutable root", encoding="utf-8")
        with pytest.raises(ac.AuditCorpusError):
            ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                  audit_parent=tmp_path / "audit", legacy_anchor=anchor)

    def test_partial_root_never_valid(self, tmp_path):
        frags, ic, anchor = self._src(tmp_path)
        root = ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                                     audit_parent=tmp_path / "audit", legacy_anchor=anchor)
        next((root / ac.FRAGS_SUBDIR).rglob("c_000.txt")).unlink()
        with pytest.raises(ac.AuditCorpusError):
            ac.verify_published_corpus(root)
        with pytest.raises(ac.AuditCorpusError):
            ac.resolve_current_root(tmp_path / "audit")


# ── §1.5 RuAA whole-work subset with three-digest binding ────────────────────
class TestWorkSubset:
    def _parent(self, tmp_path):
        frags, ic = _make_wb_corpus(tmp_path, _default_spec())
        parent = load_work_balanced_dataset(frags, cfg=CFG, input_clean_root=ic)
        return frags, ic, parent

    def test_whole_work_subset_binds_three_digests(self, tmp_path):
        frags, ic, parent = self._parent(tmp_path)
        child = ws.derive_work_subset(parent, ["alpha/a1", "beta/b1"], expected_n_works=2)
        p = child.provenance
        assert p.parent_rows_digest == parent.provenance.rows_digest      # (1) full parent
        assert p.selection_manifest_digest is not None                    # (2) selection
        assert p.rows_digest not in (None, parent.provenance.rows_digest)  # (3) child
        assert sorted(set(str(g) for g in child.groups)) == ["alpha/a1", "beta/b1"]

    def test_subset_passes_verify_against_disk(self, tmp_path):
        frags, ic, parent = self._parent(tmp_path)
        child = ws.derive_work_subset(parent, ["alpha/a1", "alpha/a2", "beta/b1"])
        contract = RunContract.build(frags, (), "unknown")
        # the runtime gate re-loads the WB parent from disk via cfg.paths.input_clean; point it at ic
        cfg = with_overrides(CFG, {"paths.input_clean": str(pathlib.Path(ic).resolve())})
        verify_dataset_against_disk(cfg, child, WORK_BALANCED, contract)  # must not raise

    def test_whole_work_includes_all_chunks(self, tmp_path):
        _, _, parent = self._parent(tmp_path)
        child = ws.derive_work_subset(parent, ["alpha/a1"])
        # alpha/a1 has 4 chunks in the toy corpus — the subset must include all of them
        assert sum(1 for g in child.groups if str(g) == "alpha/a1") == 4

    def test_missing_and_extra_and_duplicate_rejected(self, tmp_path):
        _, _, parent = self._parent(tmp_path)
        with pytest.raises(ws.WorkSubsetError):
            ws.derive_work_subset(parent, ["alpha/a1", "nope/x"])
        with pytest.raises(ws.WorkSubsetError):
            ws.derive_work_subset(parent, ["alpha/a1", "alpha/a1"])
        with pytest.raises(ws.WorkSubsetError):
            ws.derive_work_subset(parent, ["alpha/a1"], expected_n_works=5)

    def test_mutation_guard_on_parent(self, tmp_path):
        _, _, parent = self._parent(tmp_path)
        parent.texts[0] = "mutated after load"
        with pytest.raises(ws.WorkSubsetError):
            ws.derive_work_subset(parent, ["alpha/a1"])

    def test_legacy_parent_rejected(self, tmp_path):
        frags, _ = _make_wb_corpus(tmp_path, _default_spec())
        legacy = load_dataset(frags)
        with pytest.raises(ws.WorkSubsetError):
            ws.derive_work_subset(legacy, ["alpha/a1"])

    def test_forged_selection_digest_rejected_at_disk_verify(self, tmp_path):
        import dataclasses
        frags, ic, parent = self._parent(tmp_path)
        child = ws.derive_work_subset(parent, ["alpha/a1", "alpha/a2", "beta/b1"])
        child.provenance = dataclasses.replace(child.provenance, selection_manifest_digest="0" * 64)
        cfg = with_overrides(CFG, {"paths.input_clean": str(pathlib.Path(ic).resolve())})
        with pytest.raises(Exception):  # ProvenanceError: selection_manifest_digest mismatch
            verify_dataset_against_disk(cfg, child, WORK_BALANCED, RunContract.build(frags, (), "unknown"))

    def test_forged_parent_digest_rejected_at_disk_verify(self, tmp_path):
        import dataclasses
        frags, ic, parent = self._parent(tmp_path)
        child = ws.derive_work_subset(parent, ["alpha/a1", "alpha/a2", "beta/b1"])
        child.provenance = dataclasses.replace(child.provenance, parent_rows_digest="0" * 64)
        cfg = with_overrides(CFG, {"paths.input_clean": str(pathlib.Path(ic).resolve())})
        with pytest.raises(Exception):  # ProvenanceError: parent_rows_digest != disk parent digest
            verify_dataset_against_disk(cfg, child, WORK_BALANCED, RunContract.build(frags, (), "unknown"))
