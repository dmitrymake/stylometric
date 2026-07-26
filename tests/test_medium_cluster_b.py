"""Adversarial durability/concurrency gates for MEDIUM remediation cluster B."""
from __future__ import annotations

import multiprocessing
import os
import pathlib
import sqlite3
from collections import Counter
from types import SimpleNamespace

import pytest


def _resolved_identity():
    from stylo.nlp import ResolvedNLPIdentity

    return ResolvedNLPIdentity(
        requested_model="synthetic",
        resolved_model="synthetic",
        fallback_used=False,
        package_version="1",
        package_record_sha256="a" * 64,
        spacy_version="test",
        disabled_pipes=("ner",),
        active_pipes=(),
        max_length=1000,
        identity_sha256="b" * 64,
    )


def _doc_cache(root):
    import spacy

    from stylo.nlp import DocCache

    cache = DocCache(root, "synthetic", "v1")
    cache._identity = _resolved_identity()
    cache._nlp = spacy.blank("ru")
    return cache


def _doc_writer(root, key, barrier, queue):
    try:
        cache = _doc_cache(root)
        barrier.wait()
        cache._store_one(key, cache._nlp.make_doc("shared concurrent text"))
        queue.put(None)
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        queue.put(repr(exc))


def _simple_rep(text):
    from stylo.features.reps import Rep

    return Rep(
        text=text,
        bleach=text,
        pos_str="",
        punct_str="",
        morph=Counter(),
        dep_n=0,
        dep_counts=Counter(),
        dep_agg=[],
        syntax_all={},
        word_len_hist=[0],
        sent_lens=[],
    )


def _rep_writer(root, key, text, barrier, queue):
    try:
        from stylo.features import reps as reps_module

        doc_cache = SimpleNamespace(
            identity=_resolved_identity(),
            version="v1",
            model="synthetic",
        )
        params = reps_module.RepParams({}, "а", "я", max_word_len=1)
        cache = reps_module.RepCache(doc_cache, root, params)
        cache._ensure_loaded()
        reps_module._MEM_REPS[key] = _simple_rep(text)
        barrier.wait()
        cache._save([key])
        queue.put(None)
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        queue.put(repr(exc))


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="two-process cache race gate requires POSIX fork",
)
def test_doc_and_rep_cache_concurrent_writers_publish_readable_merged_state(tmp_path):
    ctx = multiprocessing.get_context("fork")

    doc_root = tmp_path / "docs"
    doc_barrier = ctx.Barrier(2)
    doc_queue = ctx.Queue()
    doc_key = "d" * 40
    doc_processes = [
        ctx.Process(
            target=_doc_writer,
            args=(str(doc_root), doc_key, doc_barrier, doc_queue),
        )
        for _ in range(2)
    ]
    for process in doc_processes:
        process.start()
    for process in doc_processes:
        process.join(30)
        assert process.exitcode == 0
    assert [doc_queue.get(timeout=2) for _ in doc_processes] == [None, None]
    cache = _doc_cache(doc_root)
    assert cache._load_one(doc_key, "shared concurrent text") is not None
    assert not list(doc_root.rglob("*.spacy.tmp"))

    rep_root = tmp_path / "reps"
    rep_barrier = ctx.Barrier(2)
    rep_queue = ctx.Queue()
    rep_rows = (("1" * 40, "one"), ("2" * 40, "two"))
    rep_processes = [
        ctx.Process(
            target=_rep_writer,
            args=(str(rep_root), key, text, rep_barrier, rep_queue),
        )
        for key, text in rep_rows
    ]
    for process in rep_processes:
        process.start()
    for process in rep_processes:
        process.join(30)
        assert process.exitcode == 0
    assert [rep_queue.get(timeout=2) for _ in rep_processes] == [None, None]

    database = next(rep_root.glob("*.sqlite3"))
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute(
            "SELECT COUNT(*) FROM representations"
        ).fetchone() == (2,)


def test_doc_cache_failed_replace_preserves_prior_canonical_payload(
        tmp_path, monkeypatch):
    from stylo import nlp as nlp_module

    cache = _doc_cache(tmp_path)
    key = "e" * 40
    cache._store_one(key, cache._nlp.make_doc("prior canonical text"))
    prior_path = cache._path_for(key)
    prior_bytes = prior_path.read_bytes()

    def crash_before_commit(_source, _target):
        raise OSError("synthetic crash before atomic replace")

    monkeypatch.setattr(nlp_module.os, "replace", crash_before_commit)
    with pytest.raises(OSError, match="synthetic crash"):
        cache._store_one(key, cache._nlp.make_doc("new uncommitted text"))
    assert prior_path.read_bytes() == prior_bytes
    assert cache._load_one(key, "prior canonical text") is not None
    assert not list(prior_path.parent.glob("*.spacy.tmp"))


def test_versioned_batch_pointer_never_resolves_a_mixed_generation(
        tmp_path, monkeypatch):
    from stylo.eval import provenance

    publication = "adversarial-batch"
    first = provenance.safe_write_batch(
        tmp_path,
        {"a.txt": "old-a", "b.txt": "old-b"},
        publication_id=publication,
    )
    assert {name: path.read_text() for name, path in first.items()} == {
        "a.txt": "old-a",
        "b.txt": "old-b",
    }

    real_replace = os.replace
    calls = 0

    def fail_first_pointer(source, target, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic pointer commit failure")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_first_pointer)
    with pytest.raises(OSError, match="pointer commit"):
        provenance.safe_write_batch(
            tmp_path,
            {"a.txt": "new-a", "b.txt": "new-b"},
            publication_id=publication,
        )
    still_first = provenance.resolve_published_batch(
        tmp_path,
        publication_id=publication,
        expected_names={"a.txt", "b.txt"},
    )
    assert {name: path.read_text() for name, path in still_first.items()} == {
        "a.txt": "old-a",
        "b.txt": "old-b",
    }

    second = provenance.safe_write_batch(
        tmp_path,
        {"a.txt": "new-a", "b.txt": "new-b"},
        publication_id=publication,
    )
    assert {name: path.read_text() for name, path in second.items()} == {
        "a.txt": "new-a",
        "b.txt": "new-b",
    }
    assert len({path.parent for path in second.values()}) == 1


def test_generic_lobo_worker_resolution_is_bounded(monkeypatch):
    from stylo.eval import lobo

    class Cfg:
        def __init__(self, cap=8):
            self.cap = cap

        def get_path(self, key, default=None):
            values = {
                "evaluation.n_jobs": -1,
                "evaluation.max_parallel_folds": self.cap,
            }
            return values.get(key, default)

    monkeypatch.setattr(lobo.os, "cpu_count", lambda: 64)
    assert lobo._bounded_lobo_workers(Cfg(), None) == 8
    assert lobo._bounded_lobo_workers(Cfg(), 10_000) == 8
    assert lobo._bounded_lobo_workers(Cfg(cap=3), -1) == 3
    with pytest.raises(ValueError):
        lobo._bounded_lobo_workers(Cfg(), 0)
    with pytest.raises(ValueError):
        lobo._bounded_lobo_workers(Cfg(cap=9), 1)
