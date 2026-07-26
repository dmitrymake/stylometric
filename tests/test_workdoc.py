"""Canonical work documents, strict manifests, and the manifest-driven corpus loader."""
from __future__ import annotations

import os
import pathlib

import pytest

from stylo import workdoc as wd
from stylo.config import load_config, parse_set_overrides
from stylo.jsonio import dump_strict, load_strict

_CH = wd.sha256_text("chash")


def _make_work(tmp, author, book, texts, *, overlap=0.0, chunker_hash=_CH):
    """Create input_clean source + frags work dir + a valid manifest. Returns (frags, ic, wdir)."""
    tmp = pathlib.Path(tmp)
    frags, ic = tmp / "frags", tmp / "input_clean"
    src = ic / author / f"{book}.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(f"source {author} {book}", encoding="utf-8")
    wdir = frags / author / book
    wdir.mkdir(parents=True, exist_ok=True)
    names = [f"c_{i:03d}.txt" for i in range(len(texts))]
    for name, text in zip(names, texts):
        (wdir / name).write_text(text, encoding="utf-8")
    m = wd.build_work_manifest(
        f"{author}/{book}", author, texts, names,
        provenance_sha256=wd.source_provenance_sha256(src), chunker_config_hash=chunker_hash, overlap=overlap,
    )
    dump_strict(m.to_dict(), wdir / wd.MANIFEST_NAME, trailing_newline=False)
    return frags, ic, wdir


def _write_raw(wdir, raw):
    dump_strict(raw, pathlib.Path(wdir) / wd.MANIFEST_NAME, trailing_newline=False)


class TestValid:
    def test_load_valid_returns_canonical_texts(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["  alpha beta  ", "gamma"])
        m, texts = wd.load_work_manifest(wdir, input_clean_root=ic, expected_chunker_config_hash=_CH)
        assert texts == ["alpha beta", "gamma"]          # stripped canonical, model representation
        assert [c.span_ordinal for c in m.chunks] == [0, 1]

    def test_round_trip(self, tmp_path):
        _, _, wdir = _make_work(tmp_path, "auth", "book", ["a a", "b b"])
        assert wd.WorkManifest.from_dict(load_strict(wdir / wd.MANIFEST_NAME)).work_id == "auth/book"

    def test_same_text_different_ordinal_kept(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["repeat me", "repeat me"])
        m, _ = wd.load_work_manifest(wdir, input_clean_root=ic)
        assert wd.chunk_identity(m, m.chunks[0]) != wd.chunk_identity(m, m.chunks[1])


class TestIntegrity:
    def test_missing_manifest(self, tmp_path):
        (tmp_path / "input_clean").mkdir()
        d = tmp_path / "frags" / "auth" / "book"; d.mkdir(parents=True)
        (d / "c.txt").write_text("x", encoding="utf-8")
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(d, input_clean_root=tmp_path / "input_clean")

    def test_extra_missing_tampered(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha", "beta"])
        (wdir / "stray.txt").write_text("x", encoding="utf-8")
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_nested_txt_rejected(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha", "beta"])
        (wdir / "nested").mkdir(); (wdir / "nested" / "r.txt").write_text("x", encoding="utf-8")
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_symlink_chunk_rejected(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha", "beta"])
        target = tmp_path / "outside.txt"; target.write_text("x", encoding="utf-8")
        (wdir / "c_000.txt").unlink(); (wdir / "c_000.txt").symlink_to(target)
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_invalid_utf8_rejected(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha", "beta"])
        (wdir / "c_000.txt").write_bytes(b"\xff\xfe")
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_whitespace_only_chunk_rejected(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha", "beta"])
        (wdir / "c_000.txt").write_text("   \n\t  ", encoding="utf-8")  # empty after strip
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_build_rejects_empty_chunk(self, tmp_path):
        with pytest.raises(wd.ManifestError):
            wd.build_work_manifest("a/b", "a", ["ok", "   "], ["c0.txt", "c1.txt"],
                                   provenance_sha256=_CH, chunker_config_hash=_CH, overlap=0.0)


class TestStrictSchema:
    @pytest.fixture
    def wdir(self, tmp_path):
        d = tmp_path / "frags" / "auth" / "book"; d.mkdir(parents=True)
        (d / "c_000.txt").write_text("x", encoding="utf-8")
        (tmp_path / "input_clean" / "auth").mkdir(parents=True)
        (tmp_path / "input_clean" / "auth" / "book.txt").write_text("s", encoding="utf-8")
        return d

    def _base(self):
        return {"work_id": "auth/book", "author_id": "auth", "provenance_sha256": wd.sha256_text("p"),
                "chunker_config_hash": _CH, "overlap": 0.0,
                "chunks": [{"span_ordinal": 0, "text_sha256": wd.sha256_text("x"), "path": "c_000.txt"}]}

    @pytest.mark.parametrize("mutate", [
        lambda r: r["chunks"][0].__setitem__("span_ordinal", 0.9),
        lambda r: r["chunks"][0].__setitem__("span_ordinal", False),
        lambda r: r["chunks"][0].__setitem__("text_sha256", "NOTHEX" + "0" * 58),
        lambda r: r.__setitem__("surprise", 1),
        lambda r: r["chunks"][0].__setitem__("path", "../escape.txt"),
    ])
    def test_schema_rejections(self, wdir, mutate):
        raw = self._base(); mutate(raw); _write_raw(wdir, raw)
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=wdir.parents[1] / "input_clean")


class TestBinding:
    def test_author_binding_mismatch(self, tmp_path):
        d = tmp_path / "frags" / "realauthor" / "book"; d.mkdir(parents=True)
        (d / "c_000.txt").write_text("x", encoding="utf-8")
        (tmp_path / "input_clean" / "realauthor").mkdir(parents=True)
        (tmp_path / "input_clean" / "realauthor" / "book.txt").write_text("s", encoding="utf-8")
        raw = {"work_id": "wrong/book", "author_id": "wrong", "provenance_sha256": wd.sha256_text("p"),
               "chunker_config_hash": _CH, "overlap": 0.0,
               "chunks": [{"span_ordinal": 0, "text_sha256": wd.sha256_text("x"), "path": "c_000.txt"}]}
        _write_raw(d, raw)
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(d, input_clean_root=tmp_path / "input_clean")

    def test_overlap_nonzero(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha"], overlap=0.2)
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_stale_hash(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha"], chunker_hash=wd.sha256_text("old"))
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic, expected_chunker_config_hash=wd.sha256_text("new"))

    def test_provenance_mismatch(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha"])
        (ic / "auth" / "book.txt").write_text("DIFFERENT", encoding="utf-8")
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_provenance_source_missing(self, tmp_path):
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha"])
        (ic / "auth" / "book.txt").unlink()
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)

    def test_symlinked_input_clean_author_dir_rejected(self, tmp_path):
        # symlink at the AUTHOR component of the source chain must be rejected, even though
        # book.txt itself is a regular file
        _, ic, wdir = _make_work(tmp_path, "auth", "book", ["alpha"])
        outside = tmp_path / "outside_author"
        outside.mkdir()
        (outside / "book.txt").write_text("source auth book", encoding="utf-8")
        import shutil
        shutil.rmtree(ic / "auth")
        (ic / "auth").symlink_to(outside, target_is_directory=True)
        with pytest.raises(wd.ManifestError):
            wd.load_work_manifest(wdir, input_clean_root=ic)


class TestFrozenConfigHash:
    def test_noninteger_min_words_rejected(self):
        cfg = load_config(overrides=parse_set_overrides(["chunking.min_words=200.5"]))
        with pytest.raises(wd.ManifestError):
            wd.chunker_config_hash(cfg)

    def test_size_change_changes_hash(self):
        a = wd.chunker_config_hash(load_config())
        b = wd.chunker_config_hash(load_config(overrides=parse_set_overrides(["chunking.chunk_size=999"])))
        assert a != b and len(a) == 64

    def test_split_and_hash_use_same_config(self):
        cfg = load_config()
        cc = wd.frozen_chunker_config(cfg)
        assert isinstance(cc.chunk_size, int) and isinstance(cc.min_words, int)

    def test_fallback_model_changes_hash(self):
        a = wd.chunker_config_hash(load_config())
        b = wd.chunker_config_hash(load_config(overrides=parse_set_overrides(["language.spacy_fallback=other_md"])))
        assert a != b

    def test_none_masking_field_rejected(self):
        cfg = load_config(overrides=parse_set_overrides(["language.spacy_model=null"]))
        with pytest.raises(wd.ManifestError):
            wd.chunker_config_hash(cfg)

    def test_negative_zero_overlap_canonicalised(self):
        base = wd.chunker_config_hash(load_config())
        neg = wd.chunker_config_hash(load_config(overrides=parse_set_overrides(["chunking.overlap=-0.0"])))
        assert base == neg


class TestCanonicalLoader:
    def _corpus(self, tmp_path):
        for a, b, texts in [("a1", "b1", ["alpha beta", "gamma", "delta"]),
                            ("a1", "b2", ["one two", "three"]),
                            ("a2", "b3", ["red", "green", "blue", "cyan", "magenta"])]:
            _make_work(tmp_path, a, b, texts)
        return tmp_path / "frags", tmp_path / "input_clean"

    def _cfg(self, ic, monkeypatch):
        monkeypatch.setattr(wd, "chunker_config_hash", lambda cfg: _CH)
        return load_config(overrides=parse_set_overrides([f"paths.input_clean={ic}"]))

    def test_loads_and_path_invariant(self, tmp_path, monkeypatch):
        frags, ic = self._corpus(tmp_path)
        ds = wd.load_work_balanced_dataset(frags, cfg=self._cfg(ic, monkeypatch))
        assert len(ds) == 10 and sorted(ds.authors) == ["a1", "a2"]
        assert set(ds.y.tolist()) == {0, 1}
        expected = [str(frags / a / b / f"c_{i:03d}.txt")
                    for a, b, n in [("a1", "b1", 3), ("a1", "b2", 2), ("a2", "b3", 5)] for i in range(n)]
        assert list(ds._manifest_paths) == expected

    def test_stray_author_txt_rejected(self, tmp_path, monkeypatch):
        frags, ic = self._corpus(tmp_path)
        (frags / "a1" / "rogue.txt").write_text("x", encoding="utf-8")
        with pytest.raises(wd.ManifestError):
            wd.load_work_balanced_dataset(frags, cfg=self._cfg(ic, monkeypatch))

    def test_symlinked_work_dir_rejected(self, tmp_path, monkeypatch):
        frags, ic = self._corpus(tmp_path)
        outside = tmp_path / "evil"; outside.mkdir()
        (outside / "c_000.txt").write_text("x", encoding="utf-8")
        (frags / "a1" / "b_link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(wd.ManifestError):
            wd.load_work_balanced_dataset(frags, cfg=self._cfg(ic, monkeypatch))

    def test_empty_author_dir_rejected(self, tmp_path, monkeypatch):
        frags, ic = self._corpus(tmp_path)
        (frags / "a3_empty").mkdir()
        with pytest.raises(wd.ManifestError):
            wd.load_work_balanced_dataset(frags, cfg=self._cfg(ic, monkeypatch))

    def test_exclusion_before_stray_check(self, tmp_path, monkeypatch):
        frags, ic = self._corpus(tmp_path)
        (frags / "unknown").mkdir()
        (frags / "unknown" / "junk.txt").write_text("x", encoding="utf-8")  # stray under excluded dir: OK
        ds = wd.load_work_balanced_dataset(frags, cfg=self._cfg(ic, monkeypatch))
        assert len(ds) == 10


def test_resolve_dataset_dispatch(tmp_path, monkeypatch):
    frags = tmp_path / "frags"
    for a, b, texts in [("a1", "b1", ["x1", "x2", "x3", "x4", "x5"]),
                        ("a2", "b2", ["y1", "y2", "y3", "y4", "y5"])]:
        _make_work(tmp_path, a, b, texts)
    monkeypatch.setattr(wd, "chunker_config_hash", lambda cfg: _CH)
    cfg = load_config(overrides=parse_set_overrides([f"paths.input_clean={tmp_path / 'input_clean'}"]))
    from stylo.dataset import resolve_dataset

    ds_wb = resolve_dataset(cfg, "work_balanced", frags)
    assert hasattr(ds_wb, "_manifest_paths")
    ds_legacy = resolve_dataset(cfg, "chunk_weighted_legacy", frags)
    assert not hasattr(ds_legacy, "_manifest_paths")
