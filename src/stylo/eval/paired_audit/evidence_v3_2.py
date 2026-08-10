"""Truthful, canonical in-memory evidence for one v3.2 whole-work vote."""
from __future__ import annotations

import contextlib
import hashlib
from collections import Counter
from typing import Iterator, Mapping, Sequence

import numpy as np

from ...domain.prediction_contract import validate_prediction_record
from ...domain.work_weighting import work_sample_weights
from ...features.work_vectorizer import WorkLevelVectorizer, _group_indicator
from ...jsonio import artifact_self_hash, canonical_hash

RECEIPT_SCHEMA = "paired_audit.fold_receipt.v3_2.candidate"
NUMERIC_CONTRACT = {
    "float_dtype": "float64",
    "integer_dtype": "int64",
    "order": "C",
    "round_decimals": 12,
    "nonfinite": "rejected",
}


class V32EvidenceError(ValueError):
    """Receipt or fitted-state evidence is incomplete, incoherent, or mutated."""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def numeric_evidence(value, *, integer: bool = False) -> dict:
    """Canonical numeric digest: fixed dtype, shape, C order, and 12-decimal float rounding."""
    dtype = np.int64 if integer else np.float64
    array = np.asarray(value, dtype=dtype)
    if not integer:
        if not np.isfinite(array).all():
            raise V32EvidenceError("numeric evidence contains non-finite values")
        array = np.round(array, NUMERIC_CONTRACT["round_decimals"])
    array = np.ascontiguousarray(array)
    return {
        "dtype": NUMERIC_CONTRACT["integer_dtype" if integer else "float_dtype"],
        "shape": list(array.shape),
        "order": "C",
        "round_decimals": None if integer else NUMERIC_CONTRACT["round_decimals"],
        "sha256": _sha(array.tobytes(order="C")),
    }


def ordered_strings_evidence(values: Sequence[str]) -> dict:
    ordered = list(values)
    if not all(type(item) is str for item in ordered):
        raise V32EvidenceError("ordered string evidence requires exact strings")
    return {"count": len(ordered), "sha256": canonical_hash(ordered)}


def _vocabulary_evidence(vectorizer) -> dict | None:
    vocab = getattr(vectorizer, "vocabulary_", None)
    if not isinstance(vocab, dict) or not vocab:
        return None
    ordered = [term for term, _ in sorted(vocab.items(), key=lambda row: row[1])]
    out = {"ordered_terms": ordered_strings_evidence(ordered)}
    idf = getattr(vectorizer, "idf_", None)
    if idf is not None:
        out["idf"] = numeric_evidence(idf)
    return out


def _df_evidence(matrix, groups: Sequence[str] | None) -> dict:
    binary = matrix.copy().tocsr()
    binary.data = np.ones_like(binary.data)
    if groups is None:
        df = binary.getnnz(axis=0)
        return {"grain": "training_chunk", "df": numeric_evidence(df, integer=True)}
    _, indicator = _group_indicator(list(groups))
    df = (indicator @ binary).getnnz(axis=0)
    return {"grain": "training_work", "df": numeric_evidence(df, integer=True)}


@contextlib.contextmanager
def observe_fit_v3_2() -> Iterator[list[dict]]:
    """Observe actual fit calls without altering estimator inputs or fitted mathematics.

    The trace records actual classifier sample weights and fitted vocabulary/DF/IDF.  DF is derived
    immediately from the exact fit input and the just-fitted vocabulary and is labelled as such.
    All patched methods are restored before prediction or receipt construction.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    trace: list[dict] = []
    patched: list[tuple[type, str, object]] = []

    def patch(cls, name, wrapper_builder):
        original = getattr(cls, name)
        setattr(cls, name, wrapper_builder(original))
        patched.append((cls, name, original))

    def lr_wrapper(original):
        def wrapped(self, X, y, sample_weight=None, **kwargs):
            result = original(self, X, y, sample_weight=sample_weight, **kwargs)
            row = {
                "kind": "classifier_fit",
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "n_rows": int(len(y)),
                "sample_weight": None,
            }
            if sample_weight is not None:
                row["sample_weight"] = numeric_evidence(sample_weight)
            trace.append(row)
            return result
        return wrapped

    def cv_fit_wrapper(original):
        def wrapped(self, raw_documents, y=None):
            docs = list(raw_documents)
            result = original(self, docs, y)
            matrix = self.transform(docs)
            row = {
                "kind": "fitted_vocabulary",
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "source": "actual_fit_input_plus_actual_fitted_state",
                "vocabulary": _vocabulary_evidence(self),
                "document_frequency": _df_evidence(matrix, None),
            }
            trace.append(row)
            return result
        return wrapped

    def cv_ft_wrapper(original):
        def wrapped(self, raw_documents, y=None):
            docs = list(raw_documents)
            matrix = original(self, docs, y)
            trace.append({
                "kind": "fitted_vocabulary",
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "source": "actual_fit_transform_input_plus_actual_fitted_state",
                "vocabulary": _vocabulary_evidence(self),
                "document_frequency": _df_evidence(matrix, None),
            })
            return matrix
        return wrapped

    def wv_wrapper(original):
        def wrapped(self, docs, groups):
            texts = list(docs)
            work_ids = list(groups)
            result = original(self, texts, work_ids)
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(**self.analyzer_params, vocabulary=self.vocabulary_)
            matrix = cv.transform(texts)
            trace.append({
                "kind": "fitted_work_vocabulary",
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "source": "actual_fit_input_plus_actual_fitted_state",
                "vocabulary": _vocabulary_evidence(self),
                "document_frequency": _df_evidence(matrix, work_ids),
            })
            return result
        return wrapped

    patch(LogisticRegression, "fit", lr_wrapper)
    # TfidfVectorizer overrides fit/fit_transform; CountVectorizer covers plain BoW/Delta paths.
    patch(CountVectorizer, "fit", cv_fit_wrapper)
    patch(CountVectorizer, "fit_transform", cv_ft_wrapper)
    patch(TfidfVectorizer, "fit", cv_fit_wrapper)
    patch(TfidfVectorizer, "fit_transform", cv_ft_wrapper)
    patch(WorkLevelVectorizer, "fit", wv_wrapper)
    try:
        yield trace
    finally:
        for cls, name, original in reversed(patched):
            setattr(cls, name, original)


def _step_state(step) -> dict:
    out = {"class": f"{type(step).__module__}.{type(step).__qualname__}"}
    classes = getattr(step, "classes_", None)
    if classes is not None:
        out["classes"] = numeric_evidence(classes, integer=True)
    for name in ("coef_", "intercept_", "scale_", "max_abs_", "mean_", "var_"):
        value = getattr(step, name, None)
        if value is not None:
            out[name] = numeric_evidence(value)
    vocabulary = _vocabulary_evidence(step)
    if vocabulary is not None:
        out["vocabulary"] = vocabulary
    inner_wv = getattr(step, "_wv", None)
    if inner_wv is not None:
        out["work_vectorizer"] = _step_state(inner_wv)
    return out


def fitted_state_v3_2(estimator, *, model: str) -> dict:
    """Project only actual post-fit fields; never pickle or repr an estimator."""
    state = {
        "schema": "paired_audit.actual_fitted_state.v3_2",
        "model": model,
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "classes": numeric_evidence(getattr(estimator, "classes_"), integer=True),
    }
    if model == "majority":
        state["majority_class"] = int(estimator._maj)
    elif model == "char_cos":
        state["vectorizer"] = _step_state(estimator._wv or estimator._vec)
        state["centroids"] = numeric_evidence(estimator._centroids)
        state["group_weighting"] = estimator.group_weighting_
    elif model.startswith("delta_cos:"):
        state["vectorizer"] = _step_state(estimator._wv or estimator._vec)
        state["selected_features"] = ordered_strings_evidence(list(map(str, estimator.feature_names())))
        state["mean"] = numeric_evidence(estimator.mean_)
        state["std"] = numeric_evidence(estimator.std_)
        state["centroids"] = numeric_evidence(estimator.centroids_)
        state["group_weighting"] = estimator.group_weighting_
    else:
        steps = getattr(estimator, "named_steps", None)
        if not isinstance(steps, Mapping):
            raise V32EvidenceError(f"{model} fitted estimator lacks named_steps")
        state["steps"] = {name: _step_state(step) for name, step in steps.items()}
        vectorizer = steps.get("vectorizer")
        if vectorizer is not None and hasattr(vectorizer, "blocks"):
            state["stylo_blocks"] = {
                f"{index}:{type(block).__name__}": {
                    "feature_names": ordered_strings_evidence(list(map(str, block.feature_names()))),
                    "pooled_vectorizer": _step_state(block._vec) if getattr(block, "_vec", None) is not None else None,
                    "work_vectorizer": _step_state(block._wv) if getattr(block, "_wv", None) is not None else None,
                    "feature_fit": getattr(block, "feature_fit_", None),
                    "relative_fw": getattr(block, "relative_fw_", None),
                }
                for index, block in enumerate(vectorizer.blocks)
            }
    state["digest"] = canonical_hash(state)
    return state


def training_axis_evidence(*, model: str, requested_axes: Mapping[str, bool],
                           texts: Sequence[str], y, groups, fit_trace: Sequence[dict]) -> dict:
    groups = [str(value) for value in groups]
    labels = np.asarray(y, dtype=np.int64)
    work_order = list(dict.fromkeys(groups))
    work_totals = Counter(groups)
    author_work_totals = Counter(work.split("/", 1)[0] for work in work_order)
    row = {
        "origin_contract": {
            "actual_observation": "fit_trace records actual calls and fitted state",
            "train_input_derivation": "weights/work totals and R event totals are canonical derivations, not labelled observations",
        },
        "W": {
            "requested": bool(requested_axes["W"]),
            "ordered_works": ordered_strings_evidence(work_order),
            "work_chunk_totals": [[work, work_totals[work]] for work in work_order],
            "author_work_totals": sorted(author_work_totals.items()),
            "derived_weights": numeric_evidence(work_sample_weights(labels, groups)),
            "source": "train_inputs_canonical_derivation",
        },
        "F": {
            "requested": bool(requested_axes["F"]),
            "actual_fit_trace": list(fit_trace),
        },
        "R": {"requested": bool(requested_axes["R"]), "source": "not_applicable"},
    }
    if model in ("stylo", "delta_cos:500"):
        from ...features.work_vectorizer import analyzer_event_counts
        events = analyzer_event_counts(
            {"analyzer": "word", "token_pattern": r"(?u)\b\w+\b", "lowercase": True},
            list(texts),
        )
        row["R"] = {
            "requested": bool(requested_axes["R"]),
            "policy": "all_analyzer_events_before_pruning",
            "per_train_row_event_totals": numeric_evidence(events),
            "source": "train_inputs_canonical_derivation_against_fitted_policy",
        }
    if model == "majority":
        counts = np.bincount(labels, minlength=int(labels.max()) + 1)
        row["majority"] = {
            "train_counts": numeric_evidence(counts, integer=True),
            "prior": numeric_evidence(counts / counts.sum()),
            "source": "train_inputs_canonical_derivation",
        }
    return row


def build_receipt_v3_2(body: dict) -> dict:
    if type(body) is not dict or "self_hash" in body:
        raise V32EvidenceError("receipt body must be a plain dict without self_hash")
    receipt = {"schema": RECEIPT_SCHEMA, "numeric_contract": NUMERIC_CONTRACT, **body}
    receipt["self_hash"] = artifact_self_hash(receipt)
    validate_receipt_v3_2(receipt)
    return receipt


def validate_receipt_v3_2(receipt: Mapping) -> None:
    if not isinstance(receipt, Mapping) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise V32EvidenceError("wrong v3.2 receipt schema")
    if receipt.get("numeric_contract") != NUMERIC_CONTRACT:
        raise V32EvidenceError("numeric contract drift")
    if receipt.get("self_hash") != artifact_self_hash(dict(receipt)):
        raise V32EvidenceError("receipt self-hash mismatch")
    vote = receipt.get("vote")
    probabilities = receipt.get("whole_work_probabilities")
    if not isinstance(vote, Mapping) or not isinstance(probabilities, list):
        raise V32EvidenceError("receipt vote/probabilities missing")
    validate_prediction_record(
        probabilities=probabilities,
        pred_label=vote["pred_label"],
        true_label=vote["true_label"],
        correct=vote["correct"],
        rank=vote["rank"],
        expected_width=len(probabilities),
    )
    state = receipt.get("actual_fitted_state")
    if not isinstance(state, Mapping) or state.get("digest") != canonical_hash(
        {key: value for key, value in state.items() if key != "digest"}
    ):
        raise V32EvidenceError("actual fitted-state digest mismatch")


__all__ = [
    "NUMERIC_CONTRACT", "RECEIPT_SCHEMA", "V32EvidenceError", "build_receipt_v3_2",
    "fitted_state_v3_2", "numeric_evidence", "observe_fit_v3_2", "ordered_strings_evidence",
    "training_axis_evidence", "validate_receipt_v3_2",
]
