"""Evaluation primitives for mixed-authorship segmentation.

The public coordinate convention is a half-open token interval ``[start, end)``.
Document metrics require both the reference and the prediction to be complete,
gap-free partitions of the same token range.  In particular, a token chunk is
never treated as an independent statistical unit here: corpus uncertainty is
estimated by resampling whole documents or whole works.

Anonymous clustering outputs must opt into
``evaluation_mode="anonymous_partition"``.  That mode finds a one-to-one label
alignment maximising token overlap; named attribution explicitly rejects such
truth-fitted permutation.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..domain.segmentation import LabeledSpan, Span

EvaluationMode = Literal["named_attribution", "anonymous_partition"]
TOPOLOGY_ROLE = "mixed_authorship_evaluation"


@dataclass(frozen=True, slots=True)
class PRF1:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class TokenMetrics:
    n_tokens: int
    accuracy: float
    macro_f1: float
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundaryMatch:
    true_offset: int
    predicted_offset: int
    absolute_error: int
    true_transition: tuple[str, str] | None = None
    predicted_transition: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class BoundaryReport:
    tolerance: int
    scores: PRF1
    matches: tuple[BoundaryMatch, ...]


@dataclass(frozen=True, slots=True)
class SegmentMatch:
    true_index: int
    predicted_index: int
    label: str
    predicted_label: str
    iou: float


@dataclass(frozen=True, slots=True)
class SegmentIoUReport:
    threshold: float
    scores: PRF1
    mean_matched_iou: float
    penalized_iou: float
    matches: tuple[SegmentMatch, ...]


@dataclass(frozen=True, slots=True)
class DocumentSegmentationReport:
    document_id: str
    work_id: str | None
    n_tokens: int
    n_true_segments: int
    n_predicted_segments: int
    token: TokenMetrics
    boundaries: BoundaryReport
    labeled_boundaries: BoundaryReport
    segments: SegmentIoUReport
    evaluation_mode: EvaluationMode
    label_mapping: Mapping[str, str | None]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class SegmentationDocument:
    """One corpus item and its reference/predicted span partitions."""

    document_id: str
    truth: Sequence[LabeledSpan]
    predicted: Sequence[LabeledSpan]
    work_id: str | None = None


@dataclass(frozen=True, slots=True)
class AggregateSegmentMetrics:
    scores: PRF1
    mean_matched_iou: float
    penalized_iou: float


@dataclass(frozen=True, slots=True)
class AggregateSegmentationMetrics:
    n_documents: int
    n_tokens: int
    n_true_segments: int
    n_predicted_segments: int
    n_single_author_control_documents: int
    single_author_false_positive_documents: int
    single_author_document_false_positive_rate: float | None
    n_single_author_control_works: int
    single_author_false_positive_works: int
    single_author_work_false_positive_rate: float | None
    # Backwards-compatible aliases; these retain the old document-level meaning.
    n_single_author_controls: int
    single_author_false_positives: int
    single_author_false_positive_rate: float | None
    token_accuracy: float
    token_macro_f1: float
    boundaries: PRF1
    labeled_boundaries: PRF1
    segments: AggregateSegmentMetrics


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    point: float
    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class CorpusSegmentationReport:
    documents: tuple[DocumentSegmentationReport, ...]
    aggregate: AggregateSegmentationMetrics
    confidence_intervals: Mapping[str, BootstrapCI]
    bootstrap_unit: Literal["document", "work"]
    n_bootstrap_units: int
    bootstrap_iters: int
    evaluation_mode: EvaluationMode
    label_mapping: Mapping[str, str | None]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    document_id: str
    work_id: str | None
    truth: tuple[LabeledSpan, ...]
    predicted: tuple[LabeledSpan, ...]


def _resolve_evaluation_mode(
    evaluation_mode: EvaluationMode,
    permutation_safe: bool | None,
) -> tuple[EvaluationMode, bool]:
    if evaluation_mode not in {"named_attribution", "anonymous_partition"}:
        raise ValueError(
            "evaluation_mode must be 'named_attribution' or 'anonymous_partition'"
        )
    if permutation_safe is not None and not isinstance(permutation_safe, bool):
        raise TypeError("permutation_safe must be bool or None")
    anonymous = evaluation_mode == "anonymous_partition"
    if permutation_safe is True and not anonymous:
        raise ValueError(
            "permutation_safe is forbidden for named attribution; use "
            "evaluation_mode='anonymous_partition' for post-hoc partition scoring"
        )
    if permutation_safe is False and anonymous:
        raise ValueError(
            "anonymous_partition requires label-permutation alignment; omit "
            "permutation_safe or set it to True"
        )
    return evaluation_mode, anonymous


def validate_spans(
    spans: Sequence[LabeledSpan],
    *,
    document_length: int | None = None,
    require_contiguous: bool = False,
    require_full_coverage: bool = False,
) -> tuple[LabeledSpan, ...]:
    """Validate ordering, positive lengths and absence of overlap.

    ``require_contiguous`` rejects gaps between consecutive spans.
    ``require_full_coverage`` additionally requires coverage of
    ``[0, document_length)`` and therefore also implies contiguity.
    The immutable copy returned by the function is safe to retain in reports.
    """

    if document_length is not None:
        if isinstance(document_length, bool) or not isinstance(document_length, int):
            raise TypeError("document_length must be an integer")
        if document_length < 0:
            raise ValueError("document_length must be non-negative")
    if require_full_coverage and document_length is None:
        raise ValueError("require_full_coverage needs document_length")

    checked = tuple(spans)
    previous: LabeledSpan | None = None
    for index, span in enumerate(checked):
        if not isinstance(span, LabeledSpan):
            raise TypeError(f"span {index} must be LabeledSpan, got {type(span).__name__}")
        if isinstance(span.start, bool) or not isinstance(span.start, int):
            raise TypeError(f"span {index} start must be an integer")
        if isinstance(span.end, bool) or not isinstance(span.end, int):
            raise TypeError(f"span {index} end must be an integer")
        if span.start < 0:
            raise ValueError(f"span {index} starts before token 0")
        if span.end <= span.start:
            raise ValueError(f"span {index} must have positive length")
        if not isinstance(span.label, str) or not span.label:
            raise ValueError(f"span {index} must have a non-empty string label")
        if document_length is not None and span.end > document_length:
            raise ValueError(f"span {index} ends beyond document_length={document_length}")
        if previous is not None:
            if span.start < previous.start:
                raise ValueError(f"spans are not sorted at index {index}")
            if span.start < previous.end:
                raise ValueError(f"spans overlap at index {index}")
            if (require_contiguous or require_full_coverage) and span.start != previous.end:
                raise ValueError(f"gap before span {index}")
        previous = span

    if require_full_coverage:
        if document_length == 0:
            if checked:
                raise ValueError("a zero-length document cannot contain positive spans")
        elif not checked:
            raise ValueError("non-empty document has no spans")
        elif checked[0].start != 0 or checked[-1].end != document_length:
            raise ValueError("spans do not cover the full document range")
    return checked


def canonicalize_spans(spans: Sequence[LabeledSpan]) -> tuple[LabeledSpan, ...]:
    """Merge adjacent spans with the same label, removing artificial boundaries."""

    checked = validate_spans(spans)
    merged: list[LabeledSpan] = []
    for span in checked:
        if merged and merged[-1].end == span.start and merged[-1].label == span.label:
            previous = merged[-1]
            merged[-1] = LabeledSpan(previous.start, span.end, previous.label)
        else:
            merged.append(span)
    return tuple(merged)


def _prepare_pair(
    truth: Sequence[LabeledSpan], predicted: Sequence[LabeledSpan]
) -> tuple[tuple[LabeledSpan, ...], tuple[LabeledSpan, ...]]:
    raw_truth = validate_spans(truth, require_contiguous=True)
    if not raw_truth:
        raise ValueError("segmentation evaluation needs at least one reference span")
    if raw_truth[0].start != 0:
        raise ValueError("reference spans must start at token 0")
    document_length = raw_truth[-1].end
    raw_truth = validate_spans(
        raw_truth,
        document_length=document_length,
        require_full_coverage=True,
    )
    raw_predicted = validate_spans(
        predicted,
        document_length=document_length,
        require_full_coverage=True,
    )
    return canonicalize_spans(raw_truth), canonicalize_spans(raw_predicted)


def _token_overlap_counts(
    truth: Sequence[LabeledSpan], predicted: Sequence[LabeledSpan]
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    true_index = 0
    predicted_index = 0
    while true_index < len(truth) and predicted_index < len(predicted):
        true_span = truth[true_index]
        predicted_span = predicted[predicted_index]
        left = max(true_span.start, predicted_span.start)
        right = min(true_span.end, predicted_span.end)
        if right > left:
            counts[(true_span.label, predicted_span.label)] += right - left
        if true_span.end <= predicted_span.end:
            true_index += 1
        if predicted_span.end <= true_span.end:
            predicted_index += 1
    return dict(counts)


def _best_mapping_from_counts(
    counts: Mapping[tuple[str, str], int]
) -> dict[str, str | None]:
    true_labels = sorted({key[0] for key in counts})
    predicted_labels = sorted({key[1] for key in counts})
    mapping: dict[str, str | None] = {label: None for label in predicted_labels}
    if not true_labels or not predicted_labels:
        return mapping

    overlap = np.zeros((len(predicted_labels), len(true_labels)), dtype=np.float64)
    true_position = {label: index for index, label in enumerate(true_labels)}
    predicted_position = {label: index for index, label in enumerate(predicted_labels)}
    for (true_label, predicted_label), value in counts.items():
        overlap[predicted_position[predicted_label], true_position[true_label]] += value

    predicted_indices, true_indices = linear_sum_assignment(-overlap)
    for predicted_index, true_index in zip(predicted_indices, true_indices):
        mapping[predicted_labels[predicted_index]] = true_labels[true_index]
    return mapping


def best_label_mapping(
    truth: Sequence[LabeledSpan], predicted: Sequence[LabeledSpan]
) -> dict[str, str | None]:
    """Post-hoc mapping for anonymous clusters, never named-author attribution."""

    prepared_truth, prepared_predicted = _prepare_pair(truth, predicted)
    return _best_mapping_from_counts(_token_overlap_counts(prepared_truth, prepared_predicted))


def _apply_label_mapping(
    predicted: Sequence[LabeledSpan],
    mapping: Mapping[str, str | None],
    true_labels: set[str],
) -> tuple[LabeledSpan, ...]:
    occupied = true_labels | {span.label for span in predicted}
    unmatched: dict[str, str] = {}

    def aligned(label: str) -> str:
        mapped = mapping.get(label)
        if mapped is not None:
            return mapped
        if label not in unmatched:
            index = len(unmatched)
            candidate = f"__unmatched_prediction_{index}__:{label}"
            while candidate in occupied:
                index += 1
                candidate = f"__unmatched_prediction_{index}__:{label}"
            unmatched[label] = candidate
            occupied.add(candidate)
        return unmatched[label]

    return tuple(LabeledSpan(span.start, span.end, aligned(span.label)) for span in predicted)


def _token_metrics_from_counts(
    counts: Mapping[tuple[str, str], int], n_tokens: int
) -> TokenMetrics:
    labels = tuple(sorted({label for pair in counts for label in pair}))
    if n_tokens == 0:
        return TokenMetrics(0, 0.0, 0.0, labels)

    correct = sum(value for (true_label, predicted_label), value in counts.items()
                  if true_label == predicted_label)
    f1_values: list[float] = []
    for label in labels:
        true_positive = counts.get((label, label), 0)
        true_support = sum(value for (true_label, _), value in counts.items()
                           if true_label == label)
        predicted_support = sum(value for (_, predicted_label), value in counts.items()
                                if predicted_label == label)
        denominator = true_support + predicted_support
        f1_values.append(2.0 * true_positive / denominator if denominator else 0.0)
    return TokenMetrics(
        n_tokens=n_tokens,
        accuracy=float(correct / n_tokens),
        macro_f1=float(np.mean(f1_values)) if f1_values else 0.0,
        labels=labels,
    )


def token_metrics(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    evaluation_mode: EvaluationMode = "named_attribution",
    permutation_safe: bool | None = None,
) -> TokenMetrics:
    """Token accuracy and macro-F1 for named authors or anonymous partitions."""

    _, anonymous = _resolve_evaluation_mode(evaluation_mode, permutation_safe)
    prepared_truth, prepared_predicted = _prepare_pair(truth, predicted)
    if anonymous:
        counts = _token_overlap_counts(prepared_truth, prepared_predicted)
        mapping = _best_mapping_from_counts(counts)
        prepared_predicted = _apply_label_mapping(
            prepared_predicted, mapping, {span.label for span in prepared_truth}
        )
    counts = _token_overlap_counts(prepared_truth, prepared_predicted)
    return _token_metrics_from_counts(counts, prepared_truth[-1].end)


def token_accuracy(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    evaluation_mode: EvaluationMode = "named_attribution",
    permutation_safe: bool | None = None,
) -> float:
    return token_metrics(
        truth,
        predicted,
        evaluation_mode=evaluation_mode,
        permutation_safe=permutation_safe,
    ).accuracy


def token_macro_f1(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    evaluation_mode: EvaluationMode = "named_attribution",
    permutation_safe: bool | None = None,
) -> float:
    return token_metrics(
        truth,
        predicted,
        evaluation_mode=evaluation_mode,
        permutation_safe=permutation_safe,
    ).macro_f1


def _prf1(true_positives: int, n_predicted: int, n_true: int) -> PRF1:
    false_positives = n_predicted - true_positives
    false_negatives = n_true - true_positives
    precision = true_positives / n_predicted if n_predicted else 1.0
    recall = true_positives / n_true if n_true else 1.0
    if n_predicted == 0 and n_true == 0:
        f1 = 1.0
    elif precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return PRF1(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def _maximum_threshold_matching(
    scores: np.ndarray, eligible: np.ndarray
) -> list[tuple[int, int]]:
    """Maximum-cardinality matching, breaking ties by total score."""

    if scores.size == 0 or not np.any(eligible):
        return []
    max_matches = min(scores.shape)
    # One extra match must outweigh every possible score tie-break combined.
    bonus = float(max_matches + 1)
    weights = np.where(eligible, bonus + scores, 0.0)
    row_indices, column_indices = linear_sum_assignment(-weights)
    matches = [
        (int(row), int(column))
        for row, column in zip(row_indices, column_indices)
        if eligible[row, column]
    ]
    return sorted(matches)


def _boundary_report_prepared(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    tolerance: int,
    *,
    label_aware: bool = False,
) -> BoundaryReport:
    true_boundaries = np.asarray([span.end for span in truth[:-1]], dtype=int)
    predicted_boundaries = np.asarray([span.end for span in predicted[:-1]], dtype=int)
    true_transitions = [
        (truth[index].label, truth[index + 1].label)
        for index in range(len(truth) - 1)
    ]
    predicted_transitions = [
        (predicted[index].label, predicted[index + 1].label)
        for index in range(len(predicted) - 1)
    ]
    if len(true_boundaries) and len(predicted_boundaries):
        distances = np.abs(true_boundaries[:, None] - predicted_boundaries[None, :])
        eligible = distances <= tolerance
        if label_aware:
            same_transition = np.asarray(
                [
                    [
                        true_transition == predicted_transition
                        for predicted_transition in predicted_transitions
                    ]
                    for true_transition in true_transitions
                ],
                dtype=bool,
            )
            eligible &= same_transition
        scores = 1.0 - distances.astype(float) / (tolerance + 1.0)
        matched_indices = _maximum_threshold_matching(scores, eligible)
    else:
        distances = np.empty((len(true_boundaries), len(predicted_boundaries)), dtype=int)
        matched_indices = []
    matches = tuple(
        BoundaryMatch(
            true_offset=int(true_boundaries[true_index]),
            predicted_offset=int(predicted_boundaries[predicted_index]),
            absolute_error=int(distances[true_index, predicted_index]),
            true_transition=true_transitions[true_index],
            predicted_transition=predicted_transitions[predicted_index],
        )
        for true_index, predicted_index in matched_indices
    )
    return BoundaryReport(
        tolerance=tolerance,
        scores=_prf1(len(matches), len(predicted_boundaries), len(true_boundaries)),
        matches=matches,
    )


def boundary_metrics(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    tolerance: int = 0,
) -> BoundaryReport:
    """One-to-one, location-only boundary scores within ``tolerance`` tokens."""

    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise TypeError("boundary tolerance must be an integer")
    if tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative")
    prepared_truth, prepared_predicted = _prepare_pair(truth, predicted)
    return _boundary_report_prepared(prepared_truth, prepared_predicted, tolerance)


def labeled_boundary_metrics(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    tolerance: int = 0,
) -> BoundaryReport:
    """One-to-one boundary score requiring the same directed label transition."""

    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise TypeError("boundary tolerance must be an integer")
    if tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative")
    prepared_truth, prepared_predicted = _prepare_pair(truth, predicted)
    return _boundary_report_prepared(
        prepared_truth,
        prepared_predicted,
        tolerance,
        label_aware=True,
    )


def transition_boundary_metrics(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    tolerance: int = 0,
) -> BoundaryReport:
    """Alias spelling for :func:`labeled_boundary_metrics`."""

    return labeled_boundary_metrics(truth, predicted, tolerance=tolerance)


def _span_iou(first: LabeledSpan, second: LabeledSpan) -> float:
    overlap = max(0, min(first.end, second.end) - max(first.start, second.start))
    if overlap == 0:
        return 0.0
    union = max(first.end, second.end) - min(first.start, second.start)
    return float(overlap / union)


def _segment_iou_report_prepared(
    truth: Sequence[LabeledSpan],
    original_predicted: Sequence[LabeledSpan],
    aligned_predicted: Sequence[LabeledSpan],
    threshold: float,
) -> SegmentIoUReport:
    scores = np.zeros((len(truth), len(aligned_predicted)), dtype=np.float64)
    eligible = np.zeros_like(scores, dtype=bool)
    for true_index, true_span in enumerate(truth):
        for predicted_index, predicted_span in enumerate(aligned_predicted):
            if true_span.label != predicted_span.label:
                continue
            iou = _span_iou(true_span, predicted_span)
            scores[true_index, predicted_index] = iou
            eligible[true_index, predicted_index] = iou > 0.0 and iou >= threshold
    matched_indices = _maximum_threshold_matching(scores, eligible)
    matches = tuple(
        SegmentMatch(
            true_index=true_index,
            predicted_index=predicted_index,
            label=truth[true_index].label,
            predicted_label=original_predicted[predicted_index].label,
            iou=float(scores[true_index, predicted_index]),
        )
        for true_index, predicted_index in matched_indices
    )
    padded_denominator = max(len(truth), len(aligned_predicted))
    matched_iou_sum = float(sum(match.iou for match in matches))
    return SegmentIoUReport(
        threshold=threshold,
        scores=_prf1(len(matches), len(aligned_predicted), len(truth)),
        mean_matched_iou=matched_iou_sum / len(matches) if matches else 0.0,
        # Unmatched true/predicted segments are dummy matches with IoU=0.
        penalized_iou=(
            matched_iou_sum / padded_denominator if padded_denominator else 0.0
        ),
        matches=matches,
    )


def segment_iou_metrics(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    threshold: float = 0.5,
    evaluation_mode: EvaluationMode = "named_attribution",
    permutation_safe: bool | None = None,
) -> SegmentIoUReport:
    """One-to-one segment matching for named authors or anonymous partitions."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("segment IoU threshold must be in [0, 1]")
    _, anonymous = _resolve_evaluation_mode(evaluation_mode, permutation_safe)
    prepared_truth, prepared_predicted = _prepare_pair(truth, predicted)
    aligned_predicted = prepared_predicted
    if anonymous:
        mapping = _best_mapping_from_counts(
            _token_overlap_counts(prepared_truth, prepared_predicted)
        )
        aligned_predicted = _apply_label_mapping(
            prepared_predicted, mapping, {span.label for span in prepared_truth}
        )
    return _segment_iou_report_prepared(
        prepared_truth, prepared_predicted, aligned_predicted, threshold
    )


def _document_report_prepared(
    document: _PreparedDocument,
    *,
    mapping: Mapping[str, str | None],
    boundary_tolerance: int,
    segment_iou_threshold: float,
    evaluation_mode: EvaluationMode,
) -> DocumentSegmentationReport:
    aligned_predicted = (
        _apply_label_mapping(
            document.predicted, mapping, {span.label for span in document.truth}
        )
        if mapping
        else document.predicted
    )
    counts = _token_overlap_counts(document.truth, aligned_predicted)
    token = _token_metrics_from_counts(counts, document.truth[-1].end)
    boundaries = _boundary_report_prepared(
        document.truth, document.predicted, boundary_tolerance
    )
    labeled_boundaries = _boundary_report_prepared(
        document.truth,
        aligned_predicted,
        boundary_tolerance,
        label_aware=True,
    )
    segments = _segment_iou_report_prepared(
        document.truth, document.predicted, aligned_predicted, segment_iou_threshold
    )
    return DocumentSegmentationReport(
        document_id=document.document_id,
        work_id=document.work_id,
        n_tokens=document.truth[-1].end,
        n_true_segments=len(document.truth),
        n_predicted_segments=len(document.predicted),
        token=token,
        boundaries=boundaries,
        labeled_boundaries=labeled_boundaries,
        segments=segments,
        evaluation_mode=evaluation_mode,
        label_mapping=dict(mapping),
    )


def evaluate_document(
    truth: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    *,
    document_id: str = "document",
    work_id: str | None = None,
    boundary_tolerance: int = 0,
    segment_iou_threshold: float = 0.5,
    evaluation_mode: EvaluationMode = "named_attribution",
    permutation_safe: bool | None = None,
) -> DocumentSegmentationReport:
    """Evaluate named attribution or an explicitly anonymous partition.

    Named attribution never permits truth-fitted label permutation.  Anonymous
    partition scoring uses a post-hoc one-to-one mapping and must not be reported
    as named-author accuracy.
    """

    if not document_id:
        raise ValueError("document_id must be non-empty")
    if work_id is not None and not work_id:
        raise ValueError("work_id must be non-empty when provided")
    if isinstance(boundary_tolerance, bool) or not isinstance(boundary_tolerance, int):
        raise TypeError("boundary tolerance must be an integer")
    if boundary_tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative")
    if not 0.0 <= segment_iou_threshold <= 1.0:
        raise ValueError("segment IoU threshold must be in [0, 1]")
    evaluation_mode, anonymous = _resolve_evaluation_mode(
        evaluation_mode, permutation_safe
    )

    prepared_truth, prepared_predicted = _prepare_pair(truth, predicted)
    document = _PreparedDocument(
        document_id, work_id, prepared_truth, prepared_predicted
    )
    mapping: dict[str, str | None] = {}
    if anonymous:
        mapping = _best_mapping_from_counts(
            _token_overlap_counts(prepared_truth, prepared_predicted)
        )
    return _document_report_prepared(
        document,
        mapping=mapping,
        boundary_tolerance=boundary_tolerance,
        segment_iou_threshold=segment_iou_threshold,
        evaluation_mode=evaluation_mode,
    )


def _prepare_documents(
    documents: Sequence[SegmentationDocument],
) -> tuple[_PreparedDocument, ...]:
    if not documents:
        raise ValueError("corpus evaluation needs at least one document")
    prepared: list[_PreparedDocument] = []
    seen_ids: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, SegmentationDocument):
            raise TypeError(
                f"document {index} must be SegmentationDocument, "
                f"got {type(document).__name__}"
            )
        if not document.document_id:
            raise ValueError(f"document {index} has an empty document_id")
        if document.document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document.document_id}")
        if document.work_id is not None and not document.work_id:
            raise ValueError(f"document {document.document_id} has an empty work_id")
        seen_ids.add(document.document_id)
        truth, predicted = _prepare_pair(document.truth, document.predicted)
        prepared.append(
            _PreparedDocument(document.document_id, document.work_id, truth, predicted)
        )
    return tuple(prepared)


def _best_corpus_mapping(
    documents: Sequence[_PreparedDocument],
) -> dict[str, str | None]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for document in documents:
        for pair, value in _token_overlap_counts(
            document.truth, document.predicted
        ).items():
            counts[pair] += value
    return _best_mapping_from_counts(counts)


def best_corpus_label_mapping(
    documents: Sequence[SegmentationDocument],
) -> dict[str, str | None]:
    """Find a post-hoc anonymous-cluster permutation shared by the corpus."""

    return _best_corpus_mapping(_prepare_documents(documents))


def _aggregate_prepared(
    documents: Sequence[_PreparedDocument],
    *,
    mapping: Mapping[str, str | None],
    boundary_tolerance: int,
    segment_iou_threshold: float,
    evaluation_mode: EvaluationMode,
) -> tuple[tuple[DocumentSegmentationReport, ...], AggregateSegmentationMetrics]:
    reports: list[DocumentSegmentationReport] = []
    token_counts: dict[tuple[str, str], int] = defaultdict(int)
    total_tokens = 0
    total_true_segments = 0
    total_predicted_segments = 0
    boundary_true_positives = 0
    boundary_predicted = 0
    boundary_true = 0
    labeled_boundary_true_positives = 0
    labeled_boundary_predicted = 0
    labeled_boundary_true = 0
    segment_true_positives = 0
    segment_ious: list[float] = []
    segment_penalized_denominator = 0
    n_single_author_control_documents = 0
    single_author_false_positive_documents = 0
    control_work_failures: dict[str, list[bool]] = defaultdict(list)

    for document in documents:
        report = _document_report_prepared(
            document,
            mapping=mapping,
            boundary_tolerance=boundary_tolerance,
            segment_iou_threshold=segment_iou_threshold,
            evaluation_mode=evaluation_mode,
        )
        reports.append(report)
        aligned_predicted = (
            _apply_label_mapping(
                document.predicted, mapping, {span.label for span in document.truth}
            )
            if mapping
            else document.predicted
        )
        if len(document.truth) == 1:
            n_single_author_control_documents += 1
            expected = document.truth[0].label
            if evaluation_mode == "anonymous_partition":
                false_positive = len(document.predicted) != 1
            else:
                false_positive = (
                    len(aligned_predicted) != 1
                    or aligned_predicted[0].label != expected
                )
            single_author_false_positive_documents += int(false_positive)
            if document.work_id is not None:
                control_work_failures[document.work_id].append(false_positive)
        for pair, value in _token_overlap_counts(document.truth, aligned_predicted).items():
            token_counts[pair] += value
        total_tokens += report.n_tokens
        total_true_segments += report.n_true_segments
        total_predicted_segments += report.n_predicted_segments
        boundary_true_positives += report.boundaries.scores.true_positives
        boundary_predicted += (
            report.boundaries.scores.true_positives
            + report.boundaries.scores.false_positives
        )
        boundary_true += (
            report.boundaries.scores.true_positives
            + report.boundaries.scores.false_negatives
        )
        labeled_boundary_true_positives += (
            report.labeled_boundaries.scores.true_positives
        )
        labeled_boundary_predicted += (
            report.labeled_boundaries.scores.true_positives
            + report.labeled_boundaries.scores.false_positives
        )
        labeled_boundary_true += (
            report.labeled_boundaries.scores.true_positives
            + report.labeled_boundaries.scores.false_negatives
        )
        segment_true_positives += report.segments.scores.true_positives
        segment_ious.extend(match.iou for match in report.segments.matches)
        segment_penalized_denominator += max(
            report.n_true_segments, report.n_predicted_segments
        )

    token = _token_metrics_from_counts(token_counts, total_tokens)
    n_single_author_control_works = len(control_work_failures)
    single_author_false_positive_works = sum(
        int(any(failures)) for failures in control_work_failures.values()
    )
    document_fpr = (
        single_author_false_positive_documents / n_single_author_control_documents
        if n_single_author_control_documents
        else None
    )
    work_fpr = (
        single_author_false_positive_works / n_single_author_control_works
        if n_single_author_control_works
        else None
    )
    aggregate = AggregateSegmentationMetrics(
        n_documents=len(documents),
        n_tokens=total_tokens,
        n_true_segments=total_true_segments,
        n_predicted_segments=total_predicted_segments,
        n_single_author_control_documents=n_single_author_control_documents,
        single_author_false_positive_documents=single_author_false_positive_documents,
        single_author_document_false_positive_rate=document_fpr,
        n_single_author_control_works=n_single_author_control_works,
        single_author_false_positive_works=single_author_false_positive_works,
        single_author_work_false_positive_rate=work_fpr,
        n_single_author_controls=n_single_author_control_documents,
        single_author_false_positives=single_author_false_positive_documents,
        single_author_false_positive_rate=document_fpr,
        token_accuracy=token.accuracy,
        token_macro_f1=token.macro_f1,
        boundaries=_prf1(
            boundary_true_positives, boundary_predicted, boundary_true
        ),
        labeled_boundaries=_prf1(
            labeled_boundary_true_positives,
            labeled_boundary_predicted,
            labeled_boundary_true,
        ),
        segments=AggregateSegmentMetrics(
            scores=_prf1(
                segment_true_positives,
                total_predicted_segments,
                total_true_segments,
            ),
            mean_matched_iou=(
                float(np.mean(segment_ious)) if segment_ious else 0.0
            ),
            penalized_iou=(
                float(np.sum(segment_ious)) / segment_penalized_denominator
                if segment_penalized_denominator
                else 0.0
            ),
        ),
    )
    return tuple(reports), aggregate


def _metric_values(metrics: AggregateSegmentationMetrics) -> dict[str, float]:
    values = {
        "token_accuracy": metrics.token_accuracy,
        "token_macro_f1": metrics.token_macro_f1,
        "boundary_precision": metrics.boundaries.precision,
        "boundary_recall": metrics.boundaries.recall,
        "boundary_f1": metrics.boundaries.f1,
        "labeled_boundary_precision": metrics.labeled_boundaries.precision,
        "labeled_boundary_recall": metrics.labeled_boundaries.recall,
        "labeled_boundary_f1": metrics.labeled_boundaries.f1,
        "segment_precision": metrics.segments.scores.precision,
        "segment_recall": metrics.segments.scores.recall,
        "segment_f1": metrics.segments.scores.f1,
        "mean_matched_iou": metrics.segments.mean_matched_iou,
        "penalized_segment_iou": metrics.segments.penalized_iou,
    }
    if metrics.single_author_document_false_positive_rate is not None:
        values["single_author_document_false_positive_rate"] = (
            metrics.single_author_document_false_positive_rate
        )
        # Legacy metric name retains its historical document-level meaning.
        values["single_author_false_positive_rate"] = (
            metrics.single_author_document_false_positive_rate
        )
    if metrics.single_author_work_false_positive_rate is not None:
        values["single_author_work_false_positive_rate"] = (
            metrics.single_author_work_false_positive_rate
        )
    return values


def _resolve_bootstrap_groups(
    documents: Sequence[_PreparedDocument],
    bootstrap_unit: Literal["auto", "document", "work"],
) -> tuple[Literal["document", "work"], tuple[tuple[int, ...], ...]]:
    if bootstrap_unit not in {"auto", "document", "work"}:
        raise ValueError("bootstrap_unit must be 'auto', 'document', or 'work'")
    if bootstrap_unit == "auto":
        has_work_ids = [document.work_id is not None for document in documents]
        if any(has_work_ids) and not all(has_work_ids):
            raise ValueError(
                "auto bootstrap refuses partially missing work_id values; choose an "
                "explicit unit or complete the work identifiers"
            )
        resolved: Literal["document", "work"] = (
            "work" if all(has_work_ids) else "document"
        )
    else:
        resolved = bootstrap_unit
    if resolved == "work" and any(document.work_id is None for document in documents):
        raise ValueError("work bootstrap requires work_id for every document")

    grouped: dict[str, list[int]] = {}
    for index, document in enumerate(documents):
        key = document.document_id if resolved == "document" else str(document.work_id)
        grouped.setdefault(key, []).append(index)
    return resolved, tuple(tuple(indices) for indices in grouped.values())


def block_bootstrap_confidence_intervals(
    documents: Sequence[SegmentationDocument],
    *,
    boundary_tolerance: int = 0,
    segment_iou_threshold: float = 0.5,
    evaluation_mode: EvaluationMode = "named_attribution",
    permutation_safe: bool | None = None,
    bootstrap_unit: Literal["auto", "document", "work"] = "auto",
    iters: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[
    Mapping[str, BootstrapCI], Literal["document", "work"], int
]:
    """Cluster bootstrap CIs; the resampled units are documents or whole works."""

    if isinstance(iters, bool) or not isinstance(iters, int) or iters <= 0:
        raise ValueError("iters must be a positive integer")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(boundary_tolerance, bool) or not isinstance(boundary_tolerance, int):
        raise TypeError("boundary tolerance must be an integer")
    if boundary_tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative")
    if not 0.0 <= segment_iou_threshold <= 1.0:
        raise ValueError("segment IoU threshold must be in [0, 1]")
    evaluation_mode, anonymous = _resolve_evaluation_mode(
        evaluation_mode, permutation_safe
    )

    prepared = _prepare_documents(documents)
    resolved_unit, groups = _resolve_bootstrap_groups(prepared, bootstrap_unit)
    point_mapping = _best_corpus_mapping(prepared) if anonymous else {}
    _, point_metrics = _aggregate_prepared(
        prepared,
        mapping=point_mapping,
        boundary_tolerance=boundary_tolerance,
        segment_iou_threshold=segment_iou_threshold,
        evaluation_mode=evaluation_mode,
    )
    point_values = _metric_values(point_metrics)
    if resolved_unit != "work":
        point_values.pop("single_author_work_false_positive_rate", None)
    samples = {
        metric: np.full(iters, np.nan, dtype=np.float64) for metric in point_values
    }

    rng = np.random.default_rng(seed)
    for iteration in range(iters):
        sampled_group_indices = rng.integers(0, len(groups), size=len(groups))
        sampled = []
        for draw_index, group_index in enumerate(sampled_group_indices):
            for document_index in groups[int(group_index)]:
                document = prepared[document_index]
                if resolved_unit == "work":
                    # Repeated draws of one work are distinct bootstrap units;
                    # suffixing prevents work-level aggregation from collapsing them.
                    document = dataclasses.replace(
                        document,
                        document_id=f"{document.document_id}#bootstrap_{draw_index}",
                        work_id=f"{document.work_id}#bootstrap_{draw_index}",
                    )
                sampled.append(document)
        sampled_documents = tuple(sampled)
        sampled_mapping = (
            _best_corpus_mapping(sampled_documents) if anonymous else {}
        )
        _, sampled_metrics = _aggregate_prepared(
            sampled_documents,
            mapping=sampled_mapping,
            boundary_tolerance=boundary_tolerance,
            segment_iou_threshold=segment_iou_threshold,
            evaluation_mode=evaluation_mode,
        )
        sampled_values = _metric_values(sampled_metrics)
        for metric in samples:
            if metric in sampled_values:
                samples[metric][iteration] = sampled_values[metric]

    alpha = (1.0 - level) / 2.0
    intervals = {}
    for metric, values in samples.items():
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        intervals[metric] = BootstrapCI(
            point=float(point_values[metric]),
            lo=float(np.percentile(finite, 100.0 * alpha)),
            hi=float(np.percentile(finite, 100.0 * (1.0 - alpha))),
        )
    return intervals, resolved_unit, len(groups)


def evaluate_corpus(
    documents: Sequence[SegmentationDocument],
    *,
    boundary_tolerance: int = 0,
    segment_iou_threshold: float = 0.5,
    evaluation_mode: EvaluationMode = "named_attribution",
    permutation_safe: bool | None = None,
    bootstrap_unit: Literal["auto", "document", "work"] = "auto",
    bootstrap_iters: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> CorpusSegmentationReport:
    """Aggregate document metrics and block-bootstrap corpus uncertainty.

    Anonymous label permutation is global across the corpus rather than
    independently optimised for every document.  It is explicitly separated
    from named-author attribution in both the API and returned report.
    """

    evaluation_mode, anonymous = _resolve_evaluation_mode(
        evaluation_mode, permutation_safe
    )
    prepared = _prepare_documents(documents)
    mapping = _best_corpus_mapping(prepared) if anonymous else {}
    reports, aggregate = _aggregate_prepared(
        prepared,
        mapping=mapping,
        boundary_tolerance=boundary_tolerance,
        segment_iou_threshold=segment_iou_threshold,
        evaluation_mode=evaluation_mode,
    )
    intervals, resolved_unit, n_units = block_bootstrap_confidence_intervals(
        documents,
        boundary_tolerance=boundary_tolerance,
        segment_iou_threshold=segment_iou_threshold,
        evaluation_mode=evaluation_mode,
        permutation_safe=None,
        bootstrap_unit=bootstrap_unit,
        iters=bootstrap_iters,
        level=ci_level,
        seed=seed,
    )
    return CorpusSegmentationReport(
        documents=reports,
        aggregate=aggregate,
        confidence_intervals=intervals,
        bootstrap_unit=resolved_unit,
        n_bootstrap_units=n_units,
        bootstrap_iters=bootstrap_iters,
        evaluation_mode=evaluation_mode,
        label_mapping=dict(mapping),
    )


__all__ = [
    "AggregateSegmentMetrics",
    "AggregateSegmentationMetrics",
    "BootstrapCI",
    "BoundaryMatch",
    "BoundaryReport",
    "CorpusSegmentationReport",
    "DocumentSegmentationReport",
    "EvaluationMode",
    "LabeledSpan",
    "PRF1",
    "SegmentIoUReport",
    "SegmentMatch",
    "SegmentationDocument",
    "Span",
    "TokenMetrics",
    "best_corpus_label_mapping",
    "best_label_mapping",
    "block_bootstrap_confidence_intervals",
    "boundary_metrics",
    "canonicalize_spans",
    "evaluate_corpus",
    "evaluate_document",
    "labeled_boundary_metrics",
    "segment_iou_metrics",
    "token_accuracy",
    "token_macro_f1",
    "token_metrics",
    "transition_boundary_metrics",
    "validate_spans",
]
