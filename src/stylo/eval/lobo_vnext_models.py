"""Exact R1 real-model adapters for the exploratory LOBO-vNext harness.

This module is deliberately narrower than :mod:`stylo.eval.lobo`.  It does not
discover models, choose defaults, or expose the historical A0/A1/A4 axis.  It
binds the two owner-selected R1 roles to the already checked generic estimator
components:

* primary: ``stylo`` with ``work_balanced`` training;
* baseline: ``char_cos`` with ``work_balanced`` training.

The adapter is fail-closed at both boundaries.  A :class:`ModelSpec` must carry
the exact path-independent scientific configuration and the SHA-256 of this
module.  Then each worker verifies the restored :class:`FoldSpec` and exact
outer-train work receipt before the generic factory is even called.  The only
cache accepted by the primary model is the existing deterministic, text-keyed
``RepCache`` constructed internally by ``StyloVectorizer``; callers cannot
inject precomputed matrices or learned representations here.

This is exploratory execution plumbing, not a confirmatory model approval.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import stat
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from ..config import ConfigNode
from ..domain.lobo_vnext import FoldSpec, ModelSpec, canonical_sha256
from ..domain.prediction_contract import (
    PREDICTION_CONTRACT_VERSION,
    PredictionContractError,
    validate_class_indices,
    validate_probability_matrix,
)
from ..domain.work_weighting import WORK_BALANCED
from ..features.reps import RepCache
from ..models.baselines import CharCosineBaseline
from ..models.work_balanced import WorkBalancedStyloPipeline
from ..vectorizer import StyloVectorizer
from .dispatch import fit_estimator
from .lobo import make_factory


R1_MODEL_ADAPTER_SCHEMA_VERSION = "stylo.lobo-vnext.r1-model-adapter.v1"
R1_MODEL_ADAPTER_RECEIPT_SCHEMA_VERSION = (
    "stylo.lobo-vnext.r1-model-adapter-receipt.v1"
)
R1_PRIMARY_ROLE = "primary"
R1_BASELINE_ROLE = "baseline"
R1_PRIMARY_MODEL_ID = "stylo"
R1_BASELINE_MODEL_ID = "char_cos"
R1_WEIGHTING = WORK_BALANCED

_R1_ROLES = (R1_PRIMARY_ROLE, R1_BASELINE_ROLE)
_R1_MODEL_BY_ROLE = {
    R1_PRIMARY_ROLE: R1_PRIMARY_MODEL_ID,
    R1_BASELINE_ROLE: R1_BASELINE_MODEL_ID,
}
_R1_ROLE_BY_MODEL = {
    model_id: role for role, model_id in _R1_MODEL_BY_ROLE.items()
}
_R1_PRIMARY_FEATURES = (
    "char_ngrams",
    "function_words",
    "syntax",
    "pos_ngrams",
    "punctuation_ngrams",
    "dependency",
    "morphology",
    "length_dist",
)
_R1_BASELINE_FEATURES = ("char_ngrams",)
_R1_SEEDS = {"model": 42}

# Only scientific inputs consumed by the selected estimators are included.
# Host paths, worker counts, cache locations, and corpus/chunker policy are
# intentionally absent; those have separate vNext identity receipts.
_R1_SCIENTIFIC_CONFIG = {
    "seed": 42,
    "language": {
        "code": "ru",
        "spacy_model": "ru_core_news_lg",
        "spacy_model_version": "3.8.0",
        "spacy_fallback": "ru_core_news_md",
        "vowels_hard": "аоуэы",
        "vowels_soft": "иеяёю",
        "pos_bleach": {
            "PROPN": "^",
            "NOUN": "¤",
            "NUM": "%",
            "ADJ": "&",
        },
    },
    "features": {
        "char_ngrams": {
            "enabled": True,
            "ngram_range": [3, 5],
            "max_features": 5000,
            "min_df": 3,
            "sublinear_tf": True,
            "bleach": True,
        },
        "function_words": {
            "enabled": True,
            "mode": "mfw",
            "mfw_count": 300,
        },
        "syntax": {
            "enabled": True,
            "subblocks": {
                "sentence": True,
                "word_len": True,
                "pos_ratios": True,
                "punctuation": True,
                "lexical_richness": True,
                "speech": True,
                "vre": True,
                "ssa": False,
            },
        },
        "pos_ngrams": {
            "enabled": True,
            "ngram_range": [2, 4],
            "max_features": 2000,
            "min_df": 3,
        },
        "punctuation_ngrams": {
            "enabled": True,
            "ngram_range": [1, 3],
            "max_features": 500,
        },
        "dependency": {"enabled": True},
        "morphology": {"enabled": True},
        "length_dist": {"enabled": True, "max_word_len": 16},
        "embeddings": {
            "enabled": False,
            "model_name": "ai-forever/ruBert-base",
            "batch_size": 16,
            "max_length": 256,
        },
    },
    "model": {
        "classifier": {
            "type": "logistic_regression",
            "max_iter": 2000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "C": 1.0,
        },
        "calibration": {"enabled": False, "method": "isotonic"},
        "scaler": "maxabs",
    },
}

_PRIMARY_EFFECTIVE_ESTIMATOR = {
    "generic_spec": R1_PRIMARY_MODEL_ID,
    "training_weighting": R1_WEIGHTING,
    "classifier": "sklearn.linear_model.LogisticRegression",
    "classifier_C": 1.0,
    "classifier_solver": "lbfgs",
    "classifier_max_iter": 2000,
    "classifier_class_weight": None,
    "scaler": "sklearn.preprocessing.MaxAbsScaler",
    "learned_calibration": False,
    "requires_inner_cv": False,
    "fit_scope": "outer_train_only",
    "representation_input": "canonical_chunk_text",
    "representation_cache": "deterministic_text_keyed_rep_cache_only",
}

_BASELINE_EFFECTIVE_ESTIMATOR = {
    "generic_spec": R1_BASELINE_MODEL_ID,
    "training_weighting": R1_WEIGHTING,
    "analyzer": "char",
    "ngram_range": [3, 3],
    "lowercase": True,
    "max_features": 5000,
    "min_df_works": 2,
    "sublinear_tf": True,
    "work_representation": "l2_chunk_tfidf_then_normalized_work_mean",
    "author_centroid": "equal_work_mean",
    "distance": "cosine",
    "probability_link": "softmax_negative_distance",
    "learned_calibration": False,
    "requires_inner_cv": False,
    "fit_scope": "outer_train_only",
    "representation_input": "canonical_chunk_text",
    "representation_cache": "none",
}


class R1ModelAdapterError(ValueError):
    """The R1 model, fold, config, or prediction contract is not exact."""


def _adapter_path() -> pathlib.Path:
    path = pathlib.Path(__file__)
    try:
        metadata = path.lstat()
    except OSError as exc:  # pragma: no cover - broken installation
        raise R1ModelAdapterError("R1 adapter source is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise R1ModelAdapterError(
            "R1 adapter source must be a non-symlink regular file"
        )
    return path


def r1_adapter_source_sha256() -> str:
    """Return the literal-byte SHA-256 of this adapter, without a host path."""

    digest = hashlib.sha256()
    try:
        with _adapter_path().open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:  # pragma: no cover - broken installation
        raise R1ModelAdapterError("cannot hash R1 adapter source") from exc
    return digest.hexdigest()


def _exact_config_node(value: object) -> ConfigNode:
    if type(value) is not ConfigNode:
        raise R1ModelAdapterError("cfg must be exactly ConfigNode")
    return value


def _exact_tree_equal(left: object, right: object) -> bool:
    """JSON-tree equality without Python's bool/int or int/float aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if set(left) != set(right):
            return False
        return all(
            _exact_tree_equal(left[key], right[key])
            for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_tree_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _scientific_config_snapshot(cfg: ConfigNode) -> dict[str, object]:
    cfg = _exact_config_node(cfg)
    language = {
        key: cfg.get_path(f"language.{key}")
        for key in (
            "code",
            "spacy_model",
            "spacy_model_version",
            "spacy_fallback",
            "vowels_hard",
            "vowels_soft",
        )
    }
    pos_bleach = cfg.get_path("language.pos_bleach")
    features = cfg.get_path("features")
    model = cfg.get_path("model")
    if (
        type(pos_bleach) is not ConfigNode
        or type(features) is not ConfigNode
        or type(model) is not ConfigNode
    ):
        raise R1ModelAdapterError(
            "R1 scientific config sections must be exact objects"
        )
    language["pos_bleach"] = pos_bleach.to_dict()
    snapshot: dict[str, object] = {
        "seed": cfg.get_path("seed"),
        "language": language,
        "features": features.to_dict(),
        "model": model.to_dict(),
    }
    # This equality is intentionally type-sensitive for bool/int and JSON
    # containers.  It prevents a config that merely hashes itself consistently
    # from changing the owner-selected R1 estimator.
    if not _exact_tree_equal(snapshot, _R1_SCIENTIFIC_CONFIG):
        raise R1ModelAdapterError(
            "live scientific config does not equal the owner-selected R1 profile"
        )
    return snapshot


def r1_scientific_config_sha256(cfg: ConfigNode) -> str:
    """Hash the path-independent exact R1 scientific config projection."""

    return canonical_sha256(_scientific_config_snapshot(cfg))


def _role(value: object) -> str:
    if type(value) is not str or value not in _R1_ROLES:
        raise R1ModelAdapterError(
            f"role must be exactly one of {list(_R1_ROLES)!r}"
        )
    return value


def _hyperparameters(role: str, cfg: ConfigNode) -> dict[str, object]:
    snapshot = _scientific_config_snapshot(cfg)
    return {
        "adapter_schema_version": R1_MODEL_ADAPTER_SCHEMA_VERSION,
        "adapter_source_sha256": r1_adapter_source_sha256(),
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "scientific_config": snapshot,
        "scientific_config_sha256": canonical_sha256(snapshot),
        "effective_estimator": (
            dict(_PRIMARY_EFFECTIVE_ESTIMATOR)
            if role == R1_PRIMARY_ROLE
            else dict(_BASELINE_EFFECTIVE_ESTIMATOR)
        ),
    }


def build_r1_model_spec(*, role: str, cfg: ConfigNode) -> ModelSpec:
    """Build one exact owner-selectable R1 ModelSpec.

    Building does not constitute owner approval.  The returned digest must still
    be bound by the external exploratory owner-decision contract.
    """

    role = _role(role)
    model_id = _R1_MODEL_BY_ROLE[role]
    features = (
        _R1_PRIMARY_FEATURES
        if role == R1_PRIMARY_ROLE
        else _R1_BASELINE_FEATURES
    )
    return ModelSpec.build(
        model_id=model_id,
        family=model_id,
        features=features,
        weighting=R1_WEIGHTING,
        hyperparameters=_hyperparameters(role, cfg),
        seeds=_R1_SEEDS,
        requires_inner_cv=False,
        inner_cv_splits=None,
        supports_component_aware_inner_cv=False,
        approved_for_exploratory=True,
        owner_selected=True,
    )


def validate_r1_model_spec(
    model_spec: ModelSpec,
    *,
    cfg: ConfigNode,
) -> str:
    """Validate a ModelSpec byte-for-byte against the live R1 adapter/config."""

    if type(model_spec) is not ModelSpec:
        raise R1ModelAdapterError("model_spec must be exactly ModelSpec")
    try:
        model_spec.validate()
    except Exception as exc:
        raise R1ModelAdapterError(f"invalid ModelSpec: {exc}") from exc
    role = _R1_ROLE_BY_MODEL.get(model_spec.model_id)
    if role is None:
        raise R1ModelAdapterError(
            f"unsupported R1 model_id {model_spec.model_id!r}"
        )
    expected = build_r1_model_spec(role=role, cfg=cfg)
    if not _exact_tree_equal(model_spec.to_dict(), expected.to_dict()):
        raise R1ModelAdapterError(
            f"{role} ModelSpec does not equal the exact live R1 contract"
        )
    return role


def build_r1_model_adapter_receipt(
    model_spec: ModelSpec,
    *,
    cfg: ConfigNode,
) -> dict[str, object]:
    """Build a canonical source/config receipt suitable for run identity."""

    role = validate_r1_model_spec(model_spec, cfg=cfg)
    payload: dict[str, object] = {
        "schema_version": R1_MODEL_ADAPTER_RECEIPT_SCHEMA_VERSION,
        "role": role,
        "model_id": model_spec.model_id,
        "model_spec_sha256": model_spec.self_hash,
        "adapter_source_sha256": r1_adapter_source_sha256(),
        "scientific_config_sha256": r1_scientific_config_sha256(cfg),
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "training_weighting": R1_WEIGHTING,
        "outer_train_only": True,
        "learned_calibration": False,
        "inner_cv": False,
    }
    return {**payload, "self_hash": canonical_sha256(payload)}


def validate_r1_model_adapter_receipt(
    receipt: object,
    *,
    model_spec: ModelSpec,
    cfg: ConfigNode,
) -> dict[str, object]:
    """Reject a rehashed, extended, stale, or wrong-model adapter receipt."""

    expected = build_r1_model_adapter_receipt(model_spec, cfg=cfg)
    if type(receipt) is not dict or set(receipt) != set(expected):
        raise R1ModelAdapterError("R1 model adapter receipt has wrong keys")
    if not _exact_tree_equal(receipt, expected):
        raise R1ModelAdapterError(
            "R1 model adapter receipt does not match source/config/spec"
        )
    return dict(receipt)


def _string_vector(value: object, *, name: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.ndim != 1 or len(value) == 0:
        raise R1ModelAdapterError(
            f"{name} must be a nonempty exact one-dimensional ndarray"
        )
    items = value.tolist()
    if any(type(item) is not str or not item for item in items):
        raise R1ModelAdapterError(
            f"{name} must contain exact nonempty strings"
        )
    return np.asarray(items, dtype=object)


def _label_vector(value: object, *, n_classes: int) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.ndim != 1
        or len(value) == 0
        or value.dtype.kind not in "iu"
        or value.dtype.kind == "b"
    ):
        raise R1ModelAdapterError(
            "labels must be a nonempty exact 1-D integer ndarray"
        )
    labels = value.astype(np.int64, copy=False)
    if set(int(item) for item in labels.tolist()) != set(range(n_classes)):
        raise R1ModelAdapterError(
            "outer train labels must retain every class in exact P"
        )
    return labels


def _validated_fold(value: object) -> FoldSpec:
    if type(value) is not FoldSpec:
        raise R1ModelAdapterError("fold must be exactly FoldSpec")
    try:
        restored = FoldSpec.from_dict(value.to_dict())
    except Exception as exc:
        raise R1ModelAdapterError(f"invalid restored FoldSpec: {exc}") from exc
    if restored != value:
        raise R1ModelAdapterError("FoldSpec is noncanonical after restore")
    return restored


def _assert_outer_train_receipt(
    *,
    fold: FoldSpec,
    texts: object,
    labels: object,
    groups: object,
    inner_splits: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_rows = _string_vector(texts, name="texts")
    group_rows = _string_vector(groups, name="groups")
    if len(text_rows) != len(group_rows):
        raise R1ModelAdapterError("texts/groups row counts differ")
    label_rows = _label_vector(
        labels, n_classes=len(fold.probability_class_order)
    )
    if len(text_rows) != len(label_rows):
        raise R1ModelAdapterError("texts/labels row counts differ")
    observed_works = set(group_rows.tolist())
    expected_works = set(fold.train_work_ids)
    if observed_works != expected_works:
        raise R1ModelAdapterError(
            "outer train rows do not equal the frozen train_work_ids"
        )
    forbidden = set(fold.purged_work_ids) | {fold.test_work_id}
    if observed_works & forbidden:
        raise R1ModelAdapterError(
            "outer train contains the test or a purged work"
        )
    label_by_work: dict[str, int] = {}
    for work_id, raw_label in zip(
        group_rows.tolist(), label_rows.tolist(), strict=True
    ):
        label = int(raw_label)
        previous = label_by_work.setdefault(work_id, label)
        if previous != label:
            raise R1ModelAdapterError(
                f"work {work_id!r} spans multiple train labels"
            )
    if type(inner_splits) is not tuple or inner_splits:
        raise R1ModelAdapterError(
            "R1 has no learned calibration/inner CV; inner_splits must be ()"
        )
    return text_rows, label_rows, group_rows


def _verify_constructed_estimator(
    estimator: object,
    *,
    role: str,
) -> None:
    if role == R1_PRIMARY_ROLE:
        if type(estimator) is not WorkBalancedStyloPipeline:
            raise R1ModelAdapterError(
                "generic factory did not construct exact WorkBalancedStyloPipeline"
            )
        if list(estimator.named_steps) != [
            "vectorizer",
            "scaler",
            "classifier",
        ]:
            raise R1ModelAdapterError("primary pipeline step order drifted")
        vectorizer = estimator.named_steps["vectorizer"]
        scaler = estimator.named_steps["scaler"]
        classifier = estimator.named_steps["classifier"]
        if (
            type(vectorizer) is not StyloVectorizer
            or type(vectorizer.rep_cache) is not RepCache
            or tuple(block.name for block in vectorizer.blocks)
            != _R1_PRIMARY_FEATURES
        ):
            raise R1ModelAdapterError(
                "primary feature/representation adapter drifted"
            )
        if type(scaler) is not MaxAbsScaler:
            raise R1ModelAdapterError("primary scaler drifted")
        if type(classifier) is not LogisticRegression:
            raise R1ModelAdapterError("primary classifier type drifted")
        params = classifier.get_params(deep=False)
        expected = _PRIMARY_EFFECTIVE_ESTIMATOR
        if (
            params.get("C") != expected["classifier_C"]
            or params.get("solver") != expected["classifier_solver"]
            or params.get("max_iter") != expected["classifier_max_iter"]
            or params.get("class_weight")
            is not expected["classifier_class_weight"]
        ):
            raise R1ModelAdapterError(
                "primary effective classifier hyperparameters drifted"
            )
        if getattr(estimator, "needs_groups", None) is not True:
            raise R1ModelAdapterError("primary estimator lost group routing")
        return

    if type(estimator) is not CharCosineBaseline:
        raise R1ModelAdapterError(
            "generic factory did not construct exact CharCosineBaseline"
        )
    if (
        estimator.training_weighting != R1_WEIGHTING
        or estimator.ngram_range != (3, 3)
        or estimator.max_features != 5000
        or estimator.min_df != 3
        or getattr(estimator, "needs_groups", None) is not True
    ):
        raise R1ModelAdapterError(
            "baseline effective hyperparameters/group routing drifted"
        )


class R1OuterFoldEstimator:
    """One-use estimator sealed to one restored outer FoldSpec."""

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        fold: FoldSpec,
        cfg: ConfigNode,
    ) -> None:
        self._cfg = _exact_config_node(cfg)
        self._role = validate_r1_model_spec(model_spec, cfg=self._cfg)
        self._model_spec_sha256 = model_spec.self_hash
        self._fold = _validated_fold(fold)
        self._estimator: object | None = None
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        *,
        groups: np.ndarray,
        inner_splits: tuple[object, ...],
    ) -> "R1OuterFoldEstimator":
        if self._estimator is not None or self.classes_ is not None:
            raise R1ModelAdapterError(
                "R1 outer-fold estimator is immutable after its first fit"
            )
        text_rows, label_rows, group_rows = _assert_outer_train_receipt(
            fold=self._fold,
            texts=texts,
            labels=labels,
            groups=groups,
            inner_splits=inner_splits,
        )

        # Deliberately after the complete locality receipt above.
        generic_builder = make_factory(
            _R1_MODEL_BY_ROLE[self._role],
            self._cfg,
            weighting=R1_WEIGHTING,
        )
        if not callable(generic_builder):
            raise R1ModelAdapterError("generic model factory is not callable")
        estimator = generic_builder()
        _verify_constructed_estimator(estimator, role=self._role)
        try:
            fit_estimator(estimator, text_rows, label_rows, group_rows)
            classes = validate_class_indices(
                getattr(estimator, "classes_", None),
                len(self._fold.probability_class_order),
            )
        except (PredictionContractError, ValueError, TypeError) as exc:
            raise R1ModelAdapterError(
                f"R1 outer-train fit/class contract failed: {exc}"
            ) from exc
        self._estimator = estimator
        self.classes_ = classes.copy()
        return self

    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        if self._estimator is None or self.classes_ is None:
            raise R1ModelAdapterError("fit must complete before prediction")
        text_rows = _string_vector(texts, name="prediction texts")
        raw = self._estimator.predict_proba(text_rows)
        object_view = np.asarray(raw, dtype=object)
        if any(
            isinstance(value, (bool, np.bool_))
            or isinstance(value, (str, bytes))
            for value in object_view.flat
        ):
            raise R1ModelAdapterError(
                "probabilities must contain non-bool numeric scalars"
            )
        try:
            # Re-read the wrapped class order at prediction time: mutating it
            # after fit must not silently relabel columns.
            classes = validate_class_indices(
                getattr(self._estimator, "classes_", None),
                len(self._fold.probability_class_order),
            )
            if not np.array_equal(classes, self.classes_):
                raise PredictionContractError(
                    "estimator classes_ changed after fit"
                )
            probabilities = validate_probability_matrix(
                raw,
                classes,
                n_classes=len(self._fold.probability_class_order),
                n_rows=len(text_rows),
            )
        except PredictionContractError as exc:
            raise R1ModelAdapterError(
                f"R1 probability/P-order contract failed: {exc}"
            ) from exc
        return probabilities


@dataclasses.dataclass(frozen=True)
class R1ModelFactory:
    """Pickle/thread-safe factory bound to one exact ModelSpec digest."""

    cfg: ConfigNode
    model_spec_sha256: str

    def __call__(
        self,
        model_spec: ModelSpec,
        fold: FoldSpec,
    ) -> R1OuterFoldEstimator:
        if (
            type(model_spec) is not ModelSpec
            or model_spec.self_hash != self.model_spec_sha256
        ):
            raise R1ModelAdapterError(
                "runtime ModelSpec does not match the factory-bound digest"
            )
        return R1OuterFoldEstimator(
            model_spec=model_spec,
            fold=fold,
            cfg=self.cfg,
        )


def make_r1_model_factory(
    *,
    cfg: ConfigNode,
    model_spec: ModelSpec,
) -> Callable[[ModelSpec, FoldSpec], R1OuterFoldEstimator]:
    """Return the exact two-argument factory expected by LOBO-vNext."""

    cfg = _exact_config_node(cfg)
    validate_r1_model_spec(model_spec, cfg=cfg)
    return R1ModelFactory(cfg=cfg, model_spec_sha256=model_spec.self_hash)


__all__ = [
    "R1_BASELINE_MODEL_ID",
    "R1_BASELINE_ROLE",
    "R1_MODEL_ADAPTER_RECEIPT_SCHEMA_VERSION",
    "R1_MODEL_ADAPTER_SCHEMA_VERSION",
    "R1_PRIMARY_MODEL_ID",
    "R1_PRIMARY_ROLE",
    "R1_WEIGHTING",
    "R1ModelAdapterError",
    "R1ModelFactory",
    "R1OuterFoldEstimator",
    "build_r1_model_adapter_receipt",
    "build_r1_model_spec",
    "make_r1_model_factory",
    "r1_adapter_source_sha256",
    "r1_scientific_config_sha256",
    "validate_r1_model_adapter_receipt",
    "validate_r1_model_spec",
]
