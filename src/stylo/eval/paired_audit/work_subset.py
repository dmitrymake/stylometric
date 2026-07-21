"""RuAA nested-panel whole-work subset with three-digest binding (§1.5).

``derive_dataset`` refuses a work-balanced parent (a WB subset needs its own disk-derived anchor),
so the RuAA nested sensitivity panel uses this purpose-built derivation. It binds all three digests
required by §1.5: the **full WB parent digest**, the **exact selection-manifest digest**, and the
**derived child digest**. It requires **whole works** (every chunk of each selected work) and the
**exact committed work set** (a missing or extra work is a hard fail, not merely an ordered
subsequence). The child provenance is accepted by
:func:`stylo.eval.provenance.verify_dataset_against_disk`'s subset branch (parent-digest chain +
selection digest + subsequence).

RuAA is a **nested secondary sensitivity** panel only; it never selects the headline result.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..provenance import (WORK_BALANCED_MANIFEST, CorpusPolicyProvenance,
                          DatasetProvenance, ProvenanceError, _selection_digest,
                          build_provenance, canonical_digest)


class WorkSubsetError(ProvenanceError):
    """Fail-closed: the RuAA subset is not a whole-work exact-set selection of a valid WB parent."""


def _validate_exact_work_set(requested: Iterable[str], available: Sequence[str]) -> set[str]:
    if isinstance(requested, (str, bytes)):
        raise WorkSubsetError("work_ids must be an iterable of work ids, not a string")
    req = list(requested)
    if any(type(w) is not str for w in req):
        raise WorkSubsetError("every work id must be exactly str")
    if len(set(req)) != len(req):
        raise WorkSubsetError("duplicate work id in RuAA selection")
    avail = set(available)
    missing = sorted(set(req) - avail)
    if missing:
        raise WorkSubsetError(f"RuAA selection references works absent from the parent: {missing[:3]}")
    return set(req)


def derive_work_subset(parent_wb, work_ids: Iterable[str], *,
                       expected_n_works: Optional[int] = None):
    """Derive the whole-work RuAA subset of a valid work-balanced parent; returns a Dataset whose
    provenance carries the three-digest binding of §1.5.

    The in-memory mutation guard below recomputes the parent's loader-bound digest, so a byte-mutated
    or relabeled parent is rejected here. The runner never subsets a caller-supplied parent: it builds
    the parent from the byte-verified immutable audit root (``verify_published_corpus`` +
    ``load_audit_dataset``), so an off-disk forged in-memory parent cannot reach this function.
    (:func:`stylo.eval.provenance.verify_dataset_against_disk`'s subset branch remains available to
    re-anchor a child to disk if a future caller needs it.)
    """
    import numpy as np

    from ...corpus import Dataset

    prov: Optional[DatasetProvenance] = getattr(parent_wb, "provenance", None)
    if not isinstance(prov, DatasetProvenance):
        raise WorkSubsetError("derive_work_subset requires a parent with DatasetProvenance")
    if prov.loader_kind != WORK_BALANCED_MANIFEST:
        raise WorkSubsetError("RuAA subset parent must be the work_balanced_manifest dataset")

    # mutation guard: the stored digest must recompute over the current arrays
    recomputed = canonical_digest(
        [str(t) for t in parent_wb.texts], [int(v) for v in parent_wb.y],
        [str(g) for g in parent_wb.groups], list(parent_wb.authors), prov.row_ids,
        loader_kind=prov.loader_kind, chunker_config_hash=prov.chunker_config_hash,
    )
    if recomputed != prov.rows_digest:
        raise WorkSubsetError("parent rows_digest mismatch — parent was mutated, relabeled or forged")

    groups = [str(g) for g in parent_wb.groups]
    available = sorted(set(groups))
    requested = _validate_exact_work_set(work_ids, available)
    if expected_n_works is not None and len(requested) != expected_n_works:
        raise WorkSubsetError(
            f"RuAA selection has {len(requested)} works, expected {expected_n_works}"
        )

    # whole-work selection in parent order (every chunk of each selected work)
    idx = [i for i, g in enumerate(groups) if g in requested]
    if not idx:
        raise WorkSubsetError("RuAA selection is empty")
    selected_groups = {groups[i] for i in idx}
    if selected_groups != requested:
        raise WorkSubsetError("RuAA selection is not an exact whole-work set")

    sub_texts = [str(parent_wb.texts[i]) for i in idx]
    sub_groups = [groups[i] for i in idx]
    sub_row_ids = [prov.row_ids[i] for i in idx]
    present = sorted({g.split("/", 1)[0] for g in sub_groups})
    a2i = {a: i for i, a in enumerate(present)}
    sub_y = [a2i[g.split("/", 1)[0]] for g in sub_groups]

    child_prov = build_provenance(
        loader_kind=WORK_BALANCED_MANIFEST,
        texts=sub_texts, y=sub_y, groups=sub_groups, authors=present, row_ids=sub_row_ids,
        frags_root=prov.frags_root,
        corpus_policy=CorpusPolicyProvenance.build(
            list(prov.corpus_policy.exclude_from_benchmark), prov.corpus_policy.unknown_dir_name),
        chunker_config_hash=prov.chunker_config_hash,
        manifest_hash=None,
        parent_rows_digest=prov.rows_digest,                 # (1) full WB parent digest
        selection_manifest_digest=_selection_digest(sub_row_ids),  # (2) exact selection digest
    )
    # (3) derived child digest is child_prov.rows_digest, computed by build_provenance
    child = Dataset(
        texts=np.array(sub_texts, dtype=object),
        y=np.array(sub_y, dtype=int),
        groups=np.array(sub_groups, dtype=object),
        authors=present,
    )
    child.provenance = child_prov  # type: ignore[attr-defined]
    return child
