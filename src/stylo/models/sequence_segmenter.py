"""Sequence decoding primitives for mixed-authorship segmentation.

This module intentionally starts *after* a leakage-free authorship model has
produced per-window class probabilities.  It adds only temporal structure and
abstention.  Decoder parameters are selected on labelled development controls,
never on the disputed target.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from ..domain.segmentation import LabeledSpan


@dataclasses.dataclass(frozen=True)
class TokenWindow:
    """A scored half-open token window ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self):
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.end, bool)
            or not isinstance(self.end, int)
        ):
            raise TypeError("token window offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid token window [{self.start}, {self.end})")

    @property
    def centre(self) -> float:
        return (self.start + self.end) / 2.0


# Backwards-compatible public name, now deliberately shared with the scorer.
# Decoder output can therefore be passed directly to evaluate_document().
DecodedSpan = LabeledSpan


@dataclasses.dataclass(frozen=True)
class MixedControl:
    """A labelled positive development control with at least one real switch.

    ``small_contribution_labels`` identifies the donor/minority labels whose
    window recall must pass a separate sensitivity gate.  If omitted, labels
    with fewer windows than the most frequent label are used automatically.
    """

    probs: np.ndarray
    true_labels: Sequence[int]
    small_contribution_labels: Sequence[int] | None = None
    boundary_tolerance: int = 0


@dataclasses.dataclass(frozen=True)
class DecoderGridRow:
    transition_penalty: float
    abstain_threshold: float | None
    window_accuracy: float
    work_false_positive_rate: float
    abstention_rate: float
    mixed_window_accuracy: float
    mixed_boundary_recall: float
    small_contribution_recall: float
    negative_gate_pass: bool
    positive_gate_pass: bool
    feasible: bool


@dataclasses.dataclass(frozen=True)
class DecoderCalibration:
    transition_penalty: float
    abstain_threshold: float | None
    rows: tuple[DecoderGridRow, ...]
    max_work_false_positive_rate: float
    min_mixed_boundary_recall: float
    min_small_contribution_recall: float
    feasible: bool
    failure_reason: str | None
    validation_scope: str


def _normalise_probs(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("probs must have shape (n_windows, n_labels>=2)")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probs must be finite and non-negative")
    totals = p.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each probability row must have positive mass")
    return p / totals


def viterbi_decode(
    probs: np.ndarray,
    *,
    transition_penalty: float = 1.0,
    abstain_threshold: float | None = None,
) -> np.ndarray:
    """Decode the maximum-score label path with a penalty for label changes.

    If ``abstain_threshold`` is set, a final state represents ``unknown``.  Its
    emission probability is the threshold, so it wins locally when every known
    class is less plausible; temporal evidence can still bridge a weak window.
    The returned unknown-state index equals the original number of labels.
    """
    if transition_penalty < 0:
        raise ValueError("transition_penalty must be non-negative")
    if abstain_threshold is not None and not 0.0 < abstain_threshold < 1.0:
        raise ValueError("abstain_threshold must be in (0, 1)")
    p = _normalise_probs(probs)
    if abstain_threshold is not None:
        unknown = np.full((len(p), 1), float(abstain_threshold), dtype=float)
        emissions = np.hstack([p, unknown])
    else:
        emissions = p
    log_e = np.log(np.clip(emissions, 1e-15, None))
    n_steps, n_states = log_e.shape

    dp = np.empty((n_steps, n_states), dtype=float)
    back = np.zeros((n_steps, n_states), dtype=int)
    dp[0] = log_e[0]
    states = np.arange(n_states)
    for t in range(1, n_steps):
        for state in range(n_states):
            prev = dp[t - 1] - transition_penalty * (states != state)
            best = int(np.argmax(prev))
            back[t, state] = best
            dp[t, state] = prev[best] + log_e[t, state]

    path = np.empty(n_steps, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def windows_to_spans(
    windows: Sequence[TokenWindow],
    decoded: Sequence[int],
    labels: Sequence[str],
    *,
    document_length: int,
    unknown_label: str = "__unknown__",
) -> list[DecodedSpan]:
    """Convert ordered window labels to full-coverage token spans.

    Boundaries between adjacent windows are placed in the middle of their
    overlap.  Consequently every emitted token region is covered by the window
    whose decoded label it receives.  The deterministic boundary convention is
    evaluated explicitly by the registered tolerance.
    """
    windows = list(windows)
    raw_path = np.asarray(decoded)
    if raw_path.ndim != 1 or not np.issubdtype(raw_path.dtype, np.integer):
        raise TypeError("decoded path must be a one-dimensional integer sequence")
    path = raw_path.astype(int, copy=False)
    if not windows or len(windows) != len(path):
        raise ValueError("windows and decoded must have equal non-zero length")
    if (
        isinstance(document_length, bool)
        or not isinstance(document_length, int)
        or document_length <= 0
    ):
        raise ValueError("document_length must be positive")
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    if not isinstance(unknown_label, str) or not unknown_label:
        raise ValueError("unknown_label must be a non-empty string")
    if unknown_label in labels:
        raise ValueError("unknown_label must not collide with a known label")

    for index, window in enumerate(windows):
        if not isinstance(window, TokenWindow):
            raise TypeError(f"window {index} must be TokenWindow")
        if window.end > document_length:
            raise ValueError(f"window {index} lies outside the document")
        if index:
            previous = windows[index - 1]
            if window.start <= previous.start or window.end <= previous.end:
                raise ValueError("windows must have strictly increasing starts and ends")
            if window.start > previous.end:
                raise ValueError("windows leave an uncovered token gap")
    if windows[0].start != 0 or windows[-1].end != document_length:
        raise ValueError("windows must cover the full document from token 0 to document_length")

    unknown_idx = len(labels)
    if np.any(path < 0) or np.any(path > unknown_idx):
        raise ValueError("decoded path contains an invalid label index")

    edges = [0]
    for left, right in zip(windows[:-1], windows[1:]):
        edges.append(int(round((right.start + left.end) / 2.0)))
    edges.append(int(document_length))
    if any(a >= b for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("window geometry produces empty token regions")

    names = [unknown_label if int(i) == unknown_idx else str(labels[int(i)]) for i in path]
    spans: list[DecodedSpan] = []
    for i, name in enumerate(names):
        start, end = edges[i], edges[i + 1]
        if spans and spans[-1].label == name:
            previous = spans[-1]
            spans[-1] = DecodedSpan(previous.start, end, name)
        else:
            spans.append(DecodedSpan(start, end, name))
    return spans


def decode_windows(
    windows: Sequence[TokenWindow],
    probs: np.ndarray,
    labels: Sequence[str],
    *,
    document_length: int,
    transition_penalty: float = 1.0,
    abstain_threshold: float | None = None,
    unknown_label: str = "__unknown__",
) -> list[DecodedSpan]:
    p = np.asarray(probs)
    if p.ndim != 2 or p.shape[1] != len(labels):
        raise ValueError("probability columns must match labels")
    path = viterbi_decode(
        p,
        transition_penalty=transition_penalty,
        abstain_threshold=abstain_threshold,
    )
    return windows_to_spans(
        windows,
        path,
        labels,
        document_length=document_length,
        unknown_label=unknown_label,
    )


def _change_boundaries(path: np.ndarray) -> np.ndarray:
    return np.flatnonzero(path[1:] != path[:-1]) + 1


def _matched_boundaries(
    true_path: np.ndarray, predicted_path: np.ndarray, tolerance: int
) -> tuple[int, int]:
    true_boundaries = _change_boundaries(true_path)
    predicted_boundaries = _change_boundaries(predicted_path)
    true_index = predicted_index = matches = 0
    while true_index < len(true_boundaries) and predicted_index < len(predicted_boundaries):
        true_boundary = int(true_boundaries[true_index])
        predicted_boundary = int(predicted_boundaries[predicted_index])
        if abs(true_boundary - predicted_boundary) <= tolerance:
            matches += 1
            true_index += 1
            predicted_index += 1
        elif predicted_boundary < true_boundary - tolerance:
            predicted_index += 1
        else:
            true_index += 1
    return matches, len(true_boundaries)


def _prepare_mixed_controls(
    mixed_controls: Sequence[MixedControl] | None,
    n_labels: int,
) -> tuple[tuple[np.ndarray, np.ndarray, frozenset[int], int], ...]:
    if not mixed_controls:
        raise ValueError(
            "mixed_controls are required to validate boundary and small-contribution sensitivity"
        )
    prepared = []
    sensitive_windows = 0
    for index, control in enumerate(mixed_controls):
        if not isinstance(control, MixedControl):
            raise TypeError(f"mixed control {index} must be MixedControl")
        probs = _normalise_probs(control.probs)
        if probs.shape[1] != n_labels:
            raise ValueError("all controls must use the same probability columns")
        raw_path = np.asarray(control.true_labels)
        if raw_path.ndim != 1 or not np.issubdtype(raw_path.dtype, np.integer):
            raise TypeError("mixed-control true_labels must be one-dimensional integers")
        true_path = raw_path.astype(int, copy=False)
        if len(true_path) != len(probs):
            raise ValueError("mixed-control probabilities and true_labels must align")
        if np.any(true_path < 0) or np.any(true_path >= n_labels):
            raise ValueError("mixed-control label index is outside probability columns")
        labels, counts = np.unique(true_path, return_counts=True)
        if len(labels) < 2:
            raise ValueError("each mixed control must contain at least two labels")
        if isinstance(control.boundary_tolerance, bool) or not isinstance(
            control.boundary_tolerance, int
        ):
            raise TypeError("mixed-control boundary_tolerance must be an integer")
        if control.boundary_tolerance < 0:
            raise ValueError("mixed-control boundary_tolerance must be non-negative")

        if control.small_contribution_labels is None:
            maximum = int(np.max(counts))
            small_labels = frozenset(
                int(label) for label, count in zip(labels, counts) if int(count) < maximum
            )
        else:
            raw_small = np.asarray(control.small_contribution_labels)
            if raw_small.ndim != 1 or not np.issubdtype(raw_small.dtype, np.integer):
                raise TypeError("small_contribution_labels must be integer indices")
            small_labels = frozenset(int(label) for label in raw_small)
            if not small_labels:
                raise ValueError("small_contribution_labels cannot be empty")
            if not small_labels.issubset({int(label) for label in labels}):
                raise ValueError("small_contribution_labels must occur in true_labels")
        sensitive_windows += int(np.sum(np.isin(true_path, list(small_labels))))
        prepared.append(
            (probs, true_path, small_labels, int(control.boundary_tolerance))
        )
    if sensitive_windows == 0:
        raise ValueError(
            "mixed controls must contain or explicitly identify a small-contribution label"
        )
    return tuple(prepared)


def tune_decoder_on_controls(
    control_probs: Sequence[np.ndarray],
    control_labels: Sequence[int],
    *,
    mixed_controls: Sequence[MixedControl] | None = None,
    transition_penalties: Sequence[float],
    abstain_thresholds: Sequence[float | None] = (None,),
    max_work_false_positive_rate: float = 0.05,
    min_mixed_boundary_recall: float = 0.8,
    min_small_contribution_recall: float = 0.8,
) -> DecoderCalibration:
    """Tune decoder parameters on labelled *development* controls.

    Feasibility requires both a single-author work-FPR gate and positive gates
    for mixed-control boundary recall and small-contribution recall.  This is
    internal parameter tuning, not an estimate of external validation quality;
    claims still require untouched work-level evaluation.
    """
    if len(control_probs) == 0 or len(control_probs) != len(control_labels):
        raise ValueError("control_probs and control_labels must have equal non-zero length")
    if not 0.0 <= max_work_false_positive_rate <= 1.0:
        raise ValueError("max_work_false_positive_rate must be in [0, 1]")
    if not transition_penalties:
        raise ValueError("transition_penalties cannot be empty")
    if not abstain_thresholds:
        raise ValueError("abstain_thresholds cannot be empty")
    if not 0.0 <= min_mixed_boundary_recall <= 1.0:
        raise ValueError("min_mixed_boundary_recall must be in [0, 1]")
    if not 0.0 <= min_small_contribution_recall <= 1.0:
        raise ValueError("min_small_contribution_recall must be in [0, 1]")

    prepared_single = []
    n_labels = None
    for probs, true_idx in zip(control_probs, control_labels):
        p = _normalise_probs(probs)
        if n_labels is None:
            n_labels = p.shape[1]
        elif p.shape[1] != n_labels:
            raise ValueError("all controls must use the same probability columns")
        if isinstance(true_idx, bool) or not isinstance(true_idx, (int, np.integer)):
            raise TypeError("control label indices must be integers")
        if true_idx < 0 or true_idx >= p.shape[1]:
            raise ValueError("control label index is outside probability columns")
        prepared_single.append((p, int(true_idx)))
    assert n_labels is not None
    prepared_mixed = _prepare_mixed_controls(mixed_controls, n_labels)

    rows: list[DecoderGridRow] = []
    for penalty in transition_penalties:
        for threshold in abstain_thresholds:
            single_total = single_correct = false_works = 0
            total_abstained = total_windows = 0
            for p, true_idx in prepared_single:
                path = viterbi_decode(
                    p,
                    transition_penalty=float(penalty),
                    abstain_threshold=threshold,
                )
                single_total += len(path)
                single_correct += int(np.sum(path == true_idx))
                total_windows += len(path)
                total_abstained += int(np.sum(path == p.shape[1]))
                false_works += int(np.any(path != true_idx))
            mixed_total = mixed_correct = matched_boundaries = true_boundaries = 0
            small_total = small_correct = 0
            for p, true_path, small_labels, tolerance in prepared_mixed:
                path = viterbi_decode(
                    p,
                    transition_penalty=float(penalty),
                    abstain_threshold=threshold,
                )
                mixed_total += len(path)
                mixed_correct += int(np.sum(path == true_path))
                total_windows += len(path)
                total_abstained += int(np.sum(path == p.shape[1]))
                matches, boundaries = _matched_boundaries(true_path, path, tolerance)
                matched_boundaries += matches
                true_boundaries += boundaries
                sensitive = np.isin(true_path, list(small_labels))
                small_total += int(np.sum(sensitive))
                small_correct += int(np.sum((path == true_path) & sensitive))

            work_fpr = false_works / len(prepared_single)
            boundary_recall = matched_boundaries / true_boundaries
            small_recall = small_correct / small_total
            negative_gate = work_fpr <= max_work_false_positive_rate
            positive_gate = (
                boundary_recall >= min_mixed_boundary_recall
                and small_recall >= min_small_contribution_recall
            )
            row = DecoderGridRow(
                transition_penalty=float(penalty),
                abstain_threshold=threshold,
                window_accuracy=single_correct / single_total,
                work_false_positive_rate=work_fpr,
                abstention_rate=total_abstained / total_windows,
                mixed_window_accuracy=mixed_correct / mixed_total,
                mixed_boundary_recall=boundary_recall,
                small_contribution_recall=small_recall,
                negative_gate_pass=negative_gate,
                positive_gate_pass=positive_gate,
                feasible=negative_gate and positive_gate,
            )
            rows.append(row)

    feasible = [row for row in rows if row.feasible]
    if feasible:
        best = min(
            feasible,
            key=lambda row: (
                -min(row.mixed_boundary_recall, row.small_contribution_recall),
                -row.mixed_window_accuracy,
                -row.window_accuracy,
                row.abstention_rate,
                row.transition_penalty,
                -1.0 if row.abstain_threshold is None else row.abstain_threshold,
            ),
        )
        failure_reason = None
    else:
        def gate_deficit(row: DecoderGridRow) -> float:
            return (
                max(0.0, row.work_false_positive_rate - max_work_false_positive_rate)
                + max(0.0, min_mixed_boundary_recall - row.mixed_boundary_recall)
                + max(0.0, min_small_contribution_recall - row.small_contribution_recall)
            )

        best = min(
            rows,
            key=lambda row: (
                gate_deficit(row),
                -row.mixed_window_accuracy,
                -row.window_accuracy,
                row.abstention_rate,
                row.transition_penalty,
            ),
        )
        failed = []
        if not best.negative_gate_pass:
            failed.append("single_author_work_fpr")
        if best.mixed_boundary_recall < min_mixed_boundary_recall:
            failed.append("mixed_boundary_recall")
        if best.small_contribution_recall < min_small_contribution_recall:
            failed.append("small_contribution_recall")
        failure_reason = "no_joint_setting_passed:" + ",".join(failed)
    return DecoderCalibration(
        transition_penalty=best.transition_penalty,
        abstain_threshold=best.abstain_threshold,
        rows=tuple(rows),
        max_work_false_positive_rate=float(max_work_false_positive_rate),
        min_mixed_boundary_recall=float(min_mixed_boundary_recall),
        min_small_contribution_recall=float(min_small_contribution_recall),
        feasible=bool(feasible),
        failure_reason=failure_reason,
        validation_scope="development_controls_only",
    )


def calibrate_decoder_on_controls(*args, **kwargs) -> DecoderCalibration:
    """Backward-compatible name for :func:`tune_decoder_on_controls`.

    The result is development-set tuning only and explicitly carries its joint
    gate status; it must not be reported as external calibration performance.
    """

    return tune_decoder_on_controls(*args, **kwargs)
