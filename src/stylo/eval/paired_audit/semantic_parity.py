"""Loader-agnostic semantic parity, the frozen legacy anchor, and the exact equality proof (§1.2).

Old↔WB parity alone could hide coordinated loader drift, so the corpus chain is anchored to a
**committed** digest. This module provides the two independent pins:

1. **Legacy anchor** — ``load_dataset(frags_train).provenance.rows_digest`` must reproduce the
   committed ``parent_dataset_digest`` (:data:`LEGACY_ANCHOR`). That digest is the loader-bound
   canonical digest (``loader_kind=legacy_recursive``), *not* the semantic digest below; it proves
   the corpus has not drifted from the signed ancestor.
2. **Semantic parity** — a **loader-agnostic** digest over ``(texts, y, groups, authors)`` **only**
   (no ``loader_kind``, no per-row identity), so it is comparable across the legacy and the
   work-balanced loaders. Equality of the legacy load and the WB load pins WB to legacy; transitively
   WB is anchored to :data:`LEGACY_ANCHOR`.

Both must hold. The semantic digest is deliberately narrower than
:func:`stylo.eval.provenance.canonical_digest` (which binds ``loader_kind`` and the full per-row
identity and therefore differs between the two arms by construction).
"""
from __future__ import annotations

import hashlib
from typing import Sequence

# The committed frozen ancestor: ``load_dataset(data/frags_train)`` rows_digest recorded in
# ``docs/screening_panel_v1.json`` (parent_dataset_digest). Any drift from this fails closed.
LEGACY_ANCHOR = "b4886a7cd723c04515b43f042467bc372af0aeaf28c47f517f0b2aa9d46b8c92"

# Versioned tag so a future change to the semantic serialization cannot silently collide.
_SEMANTIC_DIGEST_VERSION = "paired_audit.semantic_row.v1"


class SemanticParityError(ValueError):
    """Fail-closed: the legacy anchor, the loader-agnostic parity digest, or the exact row/byte
    equality proof did not hold."""


def _lp(b: bytes) -> bytes:
    """Length-prefix a byte string so no field boundary is ambiguous."""
    return len(b).to_bytes(8, "big") + b


def _as_int_label(value, i: int) -> int:
    """Exact non-bool integral label — ``0.5``/``True`` must not silently become an int."""
    import numbers
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise SemanticParityError(f"y[{i}]={value!r} must be a non-bool integer label")
    return int(value)


def semantic_row_digest(texts: Sequence, y: Sequence, groups: Sequence, authors: Sequence) -> str:
    """Versioned, length-prefixed sha256 over ``(texts, y, groups, authors)`` **only**.

    No ``loader_kind`` and no per-row identity are bound, so a legacy load and a work-balanced load
    of the same corpus produce the **same** digest iff they yield the identical ordered semantic
    rows (same texts, labels, groups) and the identical author label space. ``authors`` is bound
    because permuting it reassigns every label.
    """
    n = len(texts)
    if not (len(y) == len(groups) == n):
        raise SemanticParityError("texts/y/groups length mismatch")
    h = hashlib.sha256()
    h.update(_lp(_SEMANTIC_DIGEST_VERSION.encode("utf-8")))
    h.update(n.to_bytes(8, "big"))
    for i in range(n):
        t = texts[i]
        g = groups[i]
        if type(t) is not str:
            raise SemanticParityError(f"row {i}: text must be exactly str")
        if type(g) is not str:
            raise SemanticParityError(f"row {i}: group must be exactly str")
        h.update(_lp(t.encode("utf-8")))
        h.update(_as_int_label(y[i], i).to_bytes(8, "big", signed=True))
        h.update(_lp(g.encode("utf-8")))
    if len(set(authors)) != len(authors):
        raise SemanticParityError("authors must be unique")
    for a in authors:
        if type(a) is not str:
            raise SemanticParityError("authors must all be exactly str")
        h.update(_lp(a.encode("utf-8")))
    return h.hexdigest()


def dataset_semantic_digest(dataset) -> str:
    """Loader-agnostic semantic digest of a :class:`stylo.corpus.Dataset`.

    Labels are passed through unchanged (not pre-coerced with ``int``) so the strict bool/non-integral
    rejection in :func:`semantic_row_digest` is genuinely exercised on the real path.
    """
    return semantic_row_digest(
        [str(t) for t in dataset.texts],
        list(dataset.y),
        [str(g) for g in dataset.groups],
        list(dataset.authors),
    )


def verify_legacy_anchor(legacy_dataset, *, expected: str = LEGACY_ANCHOR) -> str:
    """Fail-closed unless the legacy dataset's loader-bound ``rows_digest`` equals ``expected``.

    ``expected`` defaults to the committed :data:`LEGACY_ANCHOR`; synthetic tests pass a toy anchor.
    Returns the verified digest.
    """
    prov = getattr(legacy_dataset, "provenance", None)
    rows_digest = getattr(prov, "rows_digest", None)
    if not isinstance(rows_digest, str):
        raise SemanticParityError("legacy dataset carries no provenance rows_digest to anchor")
    if rows_digest != expected:
        raise SemanticParityError(
            f"legacy anchor mismatch: got {rows_digest!r}, expected {expected!r} "
            "(corpus drifted from the signed ancestor)"
        )
    return rows_digest


def assert_row_exact_equality(legacy_dataset, wb_dataset) -> None:
    """Exact per-row equality of ``(text bytes, label, group)`` between the two loads.

    This is the row half of the §1.3 equality proof (the byte/filename half is proven against the
    on-disk files by the corpus builder). A length or per-row mismatch fails closed.
    """
    lt, wt = legacy_dataset.texts, wb_dataset.texts
    ly, wy = legacy_dataset.y, wb_dataset.y
    lg, wg = legacy_dataset.groups, wb_dataset.groups
    if not (len(lt) == len(wt) == len(ly) == len(wy) == len(lg) == len(wg)):
        raise SemanticParityError(
            f"row count differs: legacy={len(lt)} vs work_balanced={len(wt)}"
        )
    if list(legacy_dataset.authors) != list(wb_dataset.authors):
        raise SemanticParityError("author label space differs between the two loads")
    for i in range(len(lt)):
        if str(lt[i]) != str(wt[i]):
            raise SemanticParityError(f"row {i}: text bytes differ between legacy and work_balanced")
        if int(ly[i]) != int(wy[i]):
            raise SemanticParityError(f"row {i}: label differs between legacy and work_balanced")
        if str(lg[i]) != str(wg[i]):
            raise SemanticParityError(f"row {i}: group differs between legacy and work_balanced")


def assert_semantic_parity(legacy_dataset, wb_dataset) -> str:
    """Prove §1.2 semantic parity: the loader-agnostic digest of the legacy load equals that of the
    work-balanced load, and every row is exactly equal. Returns the shared semantic digest.
    """
    assert_row_exact_equality(legacy_dataset, wb_dataset)
    legacy_digest = dataset_semantic_digest(legacy_dataset)
    wb_digest = dataset_semantic_digest(wb_dataset)
    if legacy_digest != wb_digest:
        raise SemanticParityError(
            f"semantic parity digest mismatch: legacy={legacy_digest!r} vs work_balanced={wb_digest!r}"
        )
    return legacy_digest
