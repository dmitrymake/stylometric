"""Truthful, canonical in-memory evidence for one v3.2 whole-work vote."""
from __future__ import annotations

import contextlib
import hashlib
from collections import Counter
from typing import Iterator, Mapping, Sequence

import numpy as np

from ...domain.prediction_contract import (
    PredictionContractError,
    validate_author_universe,
    validate_prediction_record,
)
from ...domain.work_weighting import work_sample_weights
from ...features.work_vectorizer import WorkLevelVectorizer, _group_indicator
from ...jsonio import artifact_self_hash, canonical_hash
from .applicability_v3_2 import V32ApplicabilityError, resolve_cell_v3_2

RECEIPT_SCHEMA = "paired_audit.fold_receipt.v3_2.candidate"
NUMERIC_CONTRACT = {
    "float_dtype": "float64",
    "integer_dtype": "int64",
    "order": "C",
    "round_decimals": 12,
    "nonfinite": "rejected",
}
_TOP_LEVEL_KEYS = frozenset({
    "schema", "numeric_contract", "context_identity", "bindings", "dataset", "model", "cell", "fold_index",
    "work_id", "work_content_identity", "content_component_identity", "requested_axes", "effective_axes",
    "train", "test", "class_orders", "factory_route", "estimator_class", "estimator_classes",
    "class_alignment", "whole_work_probabilities", "whole_work_probability_digest", "vote",
    "axis_evidence", "actual_fitted_state", "self_hash",
})
_EXPECTATION_KEYS = frozenset({
    "context_identity", "bindings", "dataset", "model", "cell", "fold_index", "work_id", "work_content_identity",
    "content_component_identity", "probability_class_order", "metric_label_order", "train", "test",
    "estimator_class", "estimator_classes", "whole_work_probabilities", "axis_evidence_digest",
    "actual_fitted_state_digest",
})
_BINDING_KEYS = frozenset({
    "candidate", "corrected_corpus", "corpus_manifest", "fold_manifest", "applicability", "config", "protocol",
    "content_isolation", "work_identity_catalog", "dataset_rows", "ruaa_work_selection", "dataset_row_selection",
})


class V32EvidenceError(ValueError):
    """Receipt or fitted-state evidence is incomplete, incoherent, or mutated."""


def _exact_dict(value, keys, where: str) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        raise V32EvidenceError(f"{where} must carry exactly {sorted(keys)}; got {type(value).__name__}")
    return value

def _same(value, expected) -> bool:
    if type(value) is not type(expected): return False
    if type(value) is dict: return value.keys() == expected.keys() and all(_same(value[k], expected[k]) for k in value)
    if type(value) is list: return len(value) == len(expected) and all(map(_same, value, expected))
    return value == expected


def _validated_expectation(expected: Mapping) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    expected = _exact_dict(expected, _EXPECTATION_KEYS, "receipt expectation")
    try:
        probability_order = validate_author_universe(expected["probability_class_order"])
        metric_order = validate_author_universe(expected["metric_label_order"])
    except PredictionContractError as exc:
        raise V32EvidenceError(f"invalid expected class order: {exc}") from exc
    if not set(metric_order).issubset(probability_order):
        raise V32EvidenceError("expected metric order is outside the probability order")
    if (expected["dataset"] not in ("lobo", "ruaa") or type(expected["fold_index"]) is not int
            or expected["fold_index"] < 0 or type(expected["work_id"]) is not str
            or "/" not in expected["work_id"]):
        raise V32EvidenceError("expected dataset/fold/full work identity is invalid")
    if expected["work_id"].split("/", 1)[0] not in set(metric_order):
        raise V32EvidenceError("expected held-out author is outside the metric order")
    bindings = _exact_dict(expected["bindings"], _BINDING_KEYS, "expected bindings")
    if ((expected["dataset"] == "ruaa" and None in (
            bindings["ruaa_work_selection"], bindings["dataset_row_selection"]))
            or (expected["dataset"] == "lobo" and bindings["ruaa_work_selection"] is not None)):
        raise V32EvidenceError("expected dataset selection identities are invalid")
    for split in ("train", "test"):
        row = _exact_dict(expected[split], {"n_rows", "work_ids", "row_identity_digest"},
                          f"expected {split}")
        if (type(row["n_rows"]) is not int or row["n_rows"] < 1
                or type(row["work_ids"]) is not list or not row["work_ids"]
                or any(type(work) is not str or "/" not in work for work in row["work_ids"])
                or len(set(row["work_ids"])) != len(row["work_ids"])):
            raise V32EvidenceError(f"expected {split} identity is invalid")
    if (expected["test"]["work_ids"] != [expected["work_id"]]
            or expected["work_id"] in expected["train"]["work_ids"]):
        raise V32EvidenceError("expected held-out work partition is invalid")
    classes = expected["estimator_classes"]
    width = len(probability_order)
    if (type(classes) is not list or any(type(label) is not int for label in classes)
            or sorted(classes) != list(range(width))):
        raise V32EvidenceError("expected estimator classes must permute the complete class universe")
    if type(expected["whole_work_probabilities"]) is not list:
        raise V32EvidenceError("expected whole-work probabilities must be a list")
    return expected, probability_order, metric_order


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


def build_receipt_v3_2(body: dict, *, expected: Mapping) -> dict:
    if type(body) is not dict or "self_hash" in body:
        raise V32EvidenceError("receipt body must be a plain dict without self_hash")
    receipt = {"schema": RECEIPT_SCHEMA, "numeric_contract": NUMERIC_CONTRACT, **body}
    receipt["self_hash"] = artifact_self_hash(receipt)
    validate_receipt_v3_2(receipt, expected=expected)
    return receipt


def validate_receipt_v3_2(receipt: Mapping, *, expected: Mapping) -> None:
    """Re-derive receipt semantics from context/live-state expectations, never from the receipt."""
    expected, probability_order, metric_order = _validated_expectation(expected)
    receipt = _exact_dict(receipt, _TOP_LEVEL_KEYS, "receipt")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise V32EvidenceError("wrong v3.2 receipt schema")
    if not _same(receipt["numeric_contract"], NUMERIC_CONTRACT):
        raise V32EvidenceError("numeric contract drift")
    if receipt["self_hash"] != artifact_self_hash(receipt):
        raise V32EvidenceError("receipt self-hash mismatch")
    for key in ("context_identity", "dataset", "model", "cell", "fold_index", "work_id",
                "work_content_identity", "content_component_identity", "estimator_class",
                "estimator_classes", "whole_work_probabilities"):
        if not _same(receipt[key], expected[key]):
            raise V32EvidenceError(f"receipt {key} differs from the verified expectation")
    if not _same(receipt["bindings"], expected["bindings"]):
        raise V32EvidenceError("receipt bindings differ from the verified expectation")

    try:
        cell = resolve_cell_v3_2(receipt["model"], receipt["cell"], require_applied=True)
    except V32ApplicabilityError as exc:
        raise V32EvidenceError(f"receipt route is not an applied v3.2 cell: {exc}") from exc
    if not _same(receipt["requested_axes"], cell.requested_axes):
        raise V32EvidenceError("receipt requested axes differ from the applicability registry")
    if not _same(receipt["effective_axes"], cell.effective_axes):
        raise V32EvidenceError("receipt effective axes differ from the applicability registry")
    if receipt["factory_route"] != "stylo.eval.lobo.make_factory_for_ablation":
        raise V32EvidenceError("receipt factory route is not the exact v3.2 evaluator route")

    class_orders = _exact_dict(receipt["class_orders"], {"probability", "metric"}, "class orders")
    if not _same(class_orders["probability"], ordered_strings_evidence(probability_order)):
        raise V32EvidenceError("receipt probability-order evidence differs from the verified order")
    if not _same(class_orders["metric"], ordered_strings_evidence(metric_order)):
        raise V32EvidenceError("receipt metric-order evidence differs from the verified order")

    classes = expected["estimator_classes"]
    alignment = receipt["class_alignment"]
    if type(alignment) is not list or len(alignment) != len(probability_order):
        raise V32EvidenceError("receipt class alignment is not class-complete")
    derived_alignment = []
    for column, label in enumerate(classes):
        author = probability_order[label]
        derived_alignment.append({
            "estimator_column": column,
            "dataset_label": label,
            "author": author,
            "probability_column": probability_order.index(author),
        })
    if not _same(alignment, derived_alignment):
        raise V32EvidenceError("receipt class alignment is not the verified class bijection")

    probabilities = receipt["whole_work_probabilities"]
    if not _same(receipt["whole_work_probability_digest"], numeric_evidence(probabilities)):
        raise V32EvidenceError("whole-work probability digest differs from literal probabilities")
    vote = _exact_dict(receipt["vote"], {"true_label", "pred_label", "pred_author", "correct", "rank"},
                       "receipt vote")
    try:
        decision = validate_prediction_record(
            probabilities=probabilities,
            pred_label=vote["pred_label"],
            true_label=vote["true_label"],
            correct=vote["correct"],
            rank=vote["rank"],
            expected_width=len(probability_order),
        )
    except (PredictionContractError, KeyError, TypeError, ValueError) as exc:
        raise V32EvidenceError(f"invalid receipt vote: {exc}") from exc
    true_author = receipt["work_id"].split("/", 1)[0]
    if vote["true_label"] != probability_order.index(true_author):
        raise V32EvidenceError("receipt true label differs from the held-out work author")
    if vote["pred_author"] != probability_order[decision.top1]:
        raise V32EvidenceError("receipt predicted author differs from the probability decision")

    for split in ("train", "test"):
        actual = _exact_dict(receipt[split], {"n_rows", "n_works", "work_ids", "row_identity_digest"},
                             f"receipt {split}")
        wanted = expected[split]
        if not _same(actual, {
            "n_rows": wanted["n_rows"],
            "n_works": len(wanted["work_ids"]),
            "work_ids": ordered_strings_evidence(wanted["work_ids"]),
            "row_identity_digest": wanted["row_identity_digest"],
        }):
            raise V32EvidenceError(f"receipt {split} identities differ from the verified split")

    axis = _exact_dict(
        receipt["axis_evidence"],
        {"origin_contract", "W", "F", "R", "majority"} if receipt["model"] == "majority"
        else {"origin_contract", "W", "F", "R"},
        "receipt axis evidence",
    )
    for name in "WFR":
        item = axis[name]
        if not isinstance(item, Mapping) or item.get("requested") is not cell.requested_axes[name]:
            raise V32EvidenceError(f"receipt axis evidence {name} has the wrong requested state")
    if canonical_hash(axis) != expected["axis_evidence_digest"]:
        raise V32EvidenceError("receipt axis evidence differs from the live evaluation expectation")

    state = receipt["actual_fitted_state"]
    if not isinstance(state, Mapping) or state.get("digest") != canonical_hash(
        {key: value for key, value in state.items() if key != "digest"}
    ):
        raise V32EvidenceError("actual fitted-state digest mismatch")
    if state.get("digest") != expected["actual_fitted_state_digest"]:
        raise V32EvidenceError("actual fitted state differs from the live evaluation expectation")
    if (state.get("schema") != "paired_audit.actual_fitted_state.v3_2"
            or state.get("model") != receipt["model"]
            or state.get("estimator_class") != receipt["estimator_class"]
            or not _same(state.get("classes"), numeric_evidence(classes, integer=True))):
        raise V32EvidenceError("actual fitted state disagrees with the receipt route/classes")


__all__ = [
    "NUMERIC_CONTRACT", "RECEIPT_SCHEMA", "V32EvidenceError", "build_receipt_v3_2",
    "fitted_state_v3_2", "numeric_evidence", "observe_fit_v3_2", "ordered_strings_evidence",
    "training_axis_evidence", "validate_receipt_v3_2",
]
