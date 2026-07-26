import numpy as np
import pytest

from stylo.models.invariant import PairedEditionResidualizer, PairedInvariantAuthorshipModel


def test_paired_residualizer_removes_source_and_preserves_author_axis():
    rows = []
    works = []
    sources = []
    authors = []
    # Same four works occur in both sources.  Source dominates dimension 1;
    # authorship is dimension 0 and must survive within-work residualisation.
    for author, author_x in [("a", -2.0), ("b", 2.0)]:
        for work_no in range(2):
            work = f"{author}{work_no}"
            for source, source_x in [("scan", -10.0), ("web", 10.0)]:
                rows.append([author_x, source_x, float(work_no), 1.0])
                works.append(work)
                sources.append(source)
                authors.append(author)
    X = np.asarray(rows)

    residualizer = PairedEditionResidualizer(
        embedding_dim=4, nuisance_rank=1, max_nuisance_rank=1
    ).fit(X, works, sources)
    embedded = residualizer.embedding_transform(X)
    Z = residualizer.transform(X)

    source_gap = np.linalg.norm(Z[np.asarray(sources) == "scan"].mean(0) - Z[np.asarray(sources) == "web"].mean(0))
    author_gap = np.linalg.norm(Z[np.asarray(authors) == "a"].mean(0) - Z[np.asarray(authors) == "b"].mean(0))
    assert source_gap < 1e-8
    assert author_gap > 3.9
    assert residualizer.diagnostics_.n_paired_works == 4
    assert residualizer.diagnostics_.nuisance_rank == 1
    assert residualizer.diagnostics_.paired_variance_after < 1e-12
    assert embedded.shape == Z.shape
    assert not np.allclose(embedded, Z)


def test_paired_residualizer_requires_identifying_pairs():
    X = np.eye(4)
    with pytest.raises(ValueError, match="no paired work"):
        PairedEditionResidualizer().fit(
            X,
            work_ids=["w1", "w2", "w3", "w4"],
            nuisance_ids=["s1", "s2", "s1", "s2"],
        )


def test_invariant_text_model_runs_and_exposes_diagnostics():
    texts = []
    labels = []
    works = []
    domains = []
    for author, phrase in [("a", "алый мягкий голос"), ("b", "бурый резкий голос")]:
        for work_no in range(2):
            work = f"{author}_work_{work_no}"
            for domain, marker in [("scan", " OCR_SCAN "), ("web", " WEB_HTML ")]:
                texts.append(((phrase + " ") * 30) + (marker * 20) + str(work_no))
                labels.append(author)
                works.append(work)
                domains.append(domain)

    model = PairedInvariantAuthorshipModel(
        min_df=1,
        max_features=2_000,
        embedding_dim=16,
        nuisance_rank=1,
        max_nuisance_rank=1,
    ).fit(texts, labels, work_ids=works, nuisance_ids=domains)

    pred = model.predict(["алый мягкий голос " * 20, "бурый резкий голос " * 20])
    probs = model.predict_proba(["алый мягкий голос " * 20])
    assert pred.tolist() == ["a", "b"]
    assert probs.shape == (1, 2)
    assert model.diagnostics_.usable is True
