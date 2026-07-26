"""Train-only source/edition residualisation for authorship representations.

The key identification assumption is deliberately narrow: when the *same work*
is available from two or more editions/sources, differences between their
representations are evidence about nuisance variation rather than authorship or
content.  We learn a low-rank subspace from those within-work differences and
project it out before fitting the author classifier.

This is a research baseline, not a claim that all editorial influence is
low-rank.  All vectoriser, dimensionality-reduction, nuisance-subspace, and
classifier parameters must be fitted on the outer training split only.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

try:  # sklearn keeps this import stable, but keeping it local helps type checkers.
    from sklearn.decomposition import TruncatedSVD
except ImportError:  # pragma: no cover - sklearn is a required dependency.
    TruncatedSVD = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class ResidualizerDiagnostics:
    n_samples: int
    n_works: int
    n_domains: int
    n_paired_works: int
    n_pair_deviations: int
    embedding_dim: int
    nuisance_rank: int
    nuisance_variance_explained: float
    paired_variance_before: float
    paired_variance_after: float

    @property
    def usable(self) -> bool:
        return self.n_paired_works > 0 and self.nuisance_rank > 0


class PairedEditionResidualizer(TransformerMixin, BaseEstimator):
    """Remove a nuisance subspace learned from same-work source/edition pairs.

    Parameters
    ----------
    embedding_dim:
        Maximum dense representation dimension.  High-dimensional sparse input
        is reduced with train-fitted TruncatedSVD before nuisance estimation.
    nuisance_rank:
        Fixed number of nuisance directions.  If ``None``, choose the smallest
        rank reaching ``variance_threshold``.
    max_nuisance_rank:
        Safety cap; at least one embedding dimension is always retained.
    require_pairs:
        Refuse to fit if no work occurs in at least two nuisance domains.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        nuisance_rank: int | None = None,
        variance_threshold: float = 0.90,
        max_nuisance_rank: int = 16,
        require_pairs: bool = True,
        random_state: int = 42,
    ):
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if nuisance_rank is not None and nuisance_rank < 0:
            raise ValueError("nuisance_rank must be non-negative or None")
        if not 0.0 < variance_threshold <= 1.0:
            raise ValueError("variance_threshold must be in (0, 1]")
        if max_nuisance_rank < 0:
            raise ValueError("max_nuisance_rank must be non-negative")
        self.embedding_dim = int(embedding_dim)
        self.nuisance_rank = nuisance_rank
        self.variance_threshold = float(variance_threshold)
        self.max_nuisance_rank = int(max_nuisance_rank)
        self.require_pairs = bool(require_pairs)
        self.random_state = int(random_state)

    def _fit_embedding(self, X) -> np.ndarray:
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("at least two training samples are required")
        if n_features == 0:
            raise ValueError("X must contain at least one feature")

        # Keeping a small dense matrix exact makes synthetic controls and
        # already-embedded representations deterministic.  Sparse text spaces
        # are reduced without centring, as required by TruncatedSVD.
        if n_features <= self.embedding_dim:
            self.reducer_ = None
            return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)

        n_components = min(self.embedding_dim, n_features - 1, n_samples - 1)
        if n_components < 1:  # defensive; n_features==1 was handled above.
            n_components = 1
        self.reducer_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        return np.asarray(self.reducer_.fit_transform(X), dtype=float)

    def _embed(self, X) -> np.ndarray:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature mismatch: fitted on {self.n_features_in_}, got {X.shape[1]}"
            )
        if self.reducer_ is None:
            return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        return np.asarray(self.reducer_.transform(X), dtype=float)

    def fit(self, X, work_ids: Sequence[str], nuisance_ids: Sequence[str]):
        if not hasattr(X, "shape") or len(X.shape) != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        works = np.asarray(work_ids, dtype=object)
        domains = np.asarray(nuisance_ids, dtype=object)
        if len(works) != X.shape[0] or len(domains) != X.shape[0]:
            raise ValueError("work_ids and nuisance_ids must match X rows")
        if any(str(x).strip() == "" for x in works):
            raise ValueError("work_ids cannot be empty")
        if any(str(x).strip() == "" for x in domains):
            raise ValueError("nuisance_ids cannot be empty")

        self.n_features_in_ = int(X.shape[1])
        E = self._fit_embedding(X)
        self.embedding_dim_ = int(E.shape[1])

        by_work: dict[object, np.ndarray] = {
            w: np.where(works == w)[0] for w in np.unique(works)
        }
        deviations: list[np.ndarray] = []
        paired_works = 0
        for idx in by_work.values():
            local_domains = np.unique(domains[idx])
            if len(local_domains) < 2:
                continue
            paired_works += 1
            work_mean = E[idx].mean(axis=0)
            # Each work has total weight one regardless of its edition count.
            scale = float(np.sqrt(len(local_domains)))
            for domain in local_domains:
                domain_idx = idx[domains[idx] == domain]
                deviations.append((E[domain_idx].mean(axis=0) - work_mean) / scale)

        if not deviations:
            if self.require_pairs:
                raise ValueError(
                    "no paired work: at least one work must occur in two nuisance domains"
                )
            D = np.empty((0, self.embedding_dim_), dtype=float)
            singular = np.empty(0, dtype=float)
            basis = np.empty((0, self.embedding_dim_), dtype=float)
        else:
            D = np.vstack(deviations)
            _u, singular, vh = np.linalg.svd(D, full_matrices=False)
            total = float(np.square(singular).sum())
            if total <= 1e-15 or self.max_nuisance_rank == 0:
                rank = 0
            elif self.nuisance_rank is not None:
                rank = int(self.nuisance_rank)
            else:
                cumulative = np.cumsum(np.square(singular)) / total
                rank = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            # Never project away the complete representation.
            rank = min(rank, self.max_nuisance_rank, len(singular), max(0, E.shape[1] - 1))
            basis = np.asarray(vh[:rank], dtype=float)

        self.nuisance_basis_ = basis
        total_s2 = float(np.square(singular).sum()) if len(singular) else 0.0
        kept_s2 = float(np.square(singular[: len(basis)]).sum()) if len(basis) else 0.0
        explained = kept_s2 / total_s2 if total_s2 > 0 else 0.0
        before = float(np.mean(np.sum(D * D, axis=1))) if len(D) else 0.0
        if len(D) and len(basis):
            D_after = D - (D @ basis.T) @ basis
        else:
            D_after = D
        after = float(np.mean(np.sum(D_after * D_after, axis=1))) if len(D_after) else 0.0
        self.diagnostics_ = ResidualizerDiagnostics(
            n_samples=int(X.shape[0]),
            n_works=int(len(by_work)),
            n_domains=int(len(np.unique(domains))),
            n_paired_works=int(paired_works),
            n_pair_deviations=int(len(D)),
            embedding_dim=self.embedding_dim_,
            nuisance_rank=int(len(basis)),
            nuisance_variance_explained=float(explained),
            paired_variance_before=before,
            paired_variance_after=after,
        )
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["n_features_in_", "nuisance_basis_", "diagnostics_"])
        E = self._embed(X)
        if len(self.nuisance_basis_) == 0:
            return E
        return E - (E @ self.nuisance_basis_.T) @ self.nuisance_basis_

    def embedding_transform(self, X) -> np.ndarray:
        """Return the train-fitted embedding before nuisance projection.

        This control path lets experiments compare ``embedding only`` with
        ``embedding + residualisation`` using exactly the same vectoriser and
        SVD.  Otherwise an apparent effect could be caused by dimensionality
        reduction rather than the learned nuisance subspace.
        """
        check_is_fitted(self, ["n_features_in_", "nuisance_basis_", "diagnostics_"])
        return self._embed(X)

    def fit_transform(self, X, work_ids: Sequence[str], nuisance_ids: Sequence[str]) -> np.ndarray:
        return self.fit(X, work_ids, nuisance_ids).transform(X)


class PairedInvariantAuthorshipModel(ClassifierMixin, BaseEstimator):
    """Character n-gram authorship baseline with paired-edition residualisation.

    ``nuisance_ids`` should identify the registered nuisance domain, normally a
    stable combination such as ``source_id|edition_id``.  The caller is
    responsible for invoking ``fit`` only on an outer training fold.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (3, 5),
        max_features: int = 50_000,
        min_df: int = 2,
        embedding_dim: int = 128,
        nuisance_rank: int | None = None,
        variance_threshold: float = 0.90,
        max_nuisance_rank: int = 16,
        c: float = 1.0,
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.embedding_dim = int(embedding_dim)
        self.nuisance_rank = nuisance_rank
        self.variance_threshold = float(variance_threshold)
        self.max_nuisance_rank = int(max_nuisance_rank)
        self.c = float(c)
        self.random_state = int(random_state)

    def fit(
        self,
        texts: Sequence[str],
        y: Sequence[str],
        *,
        work_ids: Sequence[str],
        nuisance_ids: Sequence[str],
    ):
        texts = list(texts)
        labels = np.asarray(y, dtype=object)
        if len(texts) != len(labels):
            raise ValueError("texts and y must have equal length")
        if len(np.unique(labels)) < 2:
            raise ValueError("at least two authors are required")
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
            dtype=np.float64,
        )
        X = self.vectorizer_.fit_transform(texts)
        self.residualizer_ = PairedEditionResidualizer(
            embedding_dim=self.embedding_dim,
            nuisance_rank=self.nuisance_rank,
            variance_threshold=self.variance_threshold,
            max_nuisance_rank=self.max_nuisance_rank,
            require_pairs=True,
            random_state=self.random_state,
        )
        Z = self.residualizer_.fit_transform(X, work_ids, nuisance_ids)
        self.classifier_ = LogisticRegression(
            C=self.c,
            class_weight="balanced",
            max_iter=2_000,
            random_state=self.random_state,
        )
        self.classifier_.fit(Z, labels)
        self.classes_ = self.classifier_.classes_
        return self

    @property
    def diagnostics_(self) -> ResidualizerDiagnostics:
        check_is_fitted(self, ["residualizer_"])
        return self.residualizer_.diagnostics_

    def _transform_texts(self, texts: Sequence[str]) -> np.ndarray:
        check_is_fitted(self, ["vectorizer_", "residualizer_", "classifier_"])
        X = self.vectorizer_.transform(list(texts))
        return self.residualizer_.transform(X)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier_.predict(self._transform_texts(texts))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier_.predict_proba(self._transform_texts(texts))
