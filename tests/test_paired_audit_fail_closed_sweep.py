"""Gap-fill tests for the confirmatory paired-audit fail-closed sweep.

The executable requirement-to-nodeid mapping is
``research/governance/requirements.json`` and is collect-checked by
``tests/test_medium_governance.py``. This module owns only gap tests that do not
belong naturally to one component suite; it is not a comment-only coverage map.
No test here touches the real closed corpus.
"""
from __future__ import annotations

import pathlib

import pytest

from stylo import workdoc as wd
from stylo.config import load_config
from stylo.jsonio import dump_strict, load_strict
from stylo.workdoc import chunker_config_hash
from stylo.eval.paired_audit import corpus as ac
from stylo.eval.paired_audit import semantic_parity as sp

CFG = load_config()
_CHASH = chunker_config_hash(CFG)


def _toy_wb(tmp: pathlib.Path):
    """A minimal valid work-balanced corpus (>=10 chunks, >=2 authors)."""
    frags, ic = tmp / "frags", tmp / "clean"
    spec = {"alpha": ["a1", "a2"], "beta": ["b1", "b2"], "gamma": ["g1", "g2"]}
    for author, books in spec.items():
        for book in books:
            src = ic / author / f"{book}.txt"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(f"source {author} {book}", encoding="utf-8")
            wdir = frags / author / book
            wdir.mkdir(parents=True, exist_ok=True)
            texts = [f"{author} {book} chunk {i} words" for i in range(3)]
            names = [f"c_{i:03d}.txt" for i in range(3)]
            for nm, tx in zip(names, texts):
                (wdir / nm).write_text(tx, encoding="utf-8")
            m = wd.build_work_manifest(f"{author}/{book}", author, texts, names,
                                       provenance_sha256=wd.source_provenance_sha256(src),
                                       chunker_config_hash=_CHASH, overlap=0.0)
            dump_strict(m.to_dict(), wdir / wd.MANIFEST_NAME, trailing_newline=False)
    return frags, ic


def test_duplicate_chunk_identity_aborts_build(tmp_path):
    frags, ic = _toy_wb(tmp_path)
    from stylo.corpus import load_dataset
    anchor = load_dataset(frags).provenance.rows_digest
    # inject a duplicate chunk entry into one work's manifest (non-contiguous ordinals / dup path)
    mpath = frags / "alpha" / "a1" / wd.MANIFEST_NAME
    raw = load_strict(mpath)
    raw["chunks"].append(dict(raw["chunks"][0]))            # duplicate the first chunk identity
    dump_strict(raw, mpath, trailing_newline=False)
    with pytest.raises((wd.ManifestError, sp.SemanticParityError)):
        ac.build_audit_corpus(source_frags_root=frags, input_clean_root=ic, cfg=CFG,
                              audit_parent=tmp_path / "audit", legacy_anchor=anchor)


# publisher crash-orphan recovery is proven in test_paired_audit_publisher::TestPublishFailClosed
# (it needs a complete verified assembly), so it lives there rather than here.
