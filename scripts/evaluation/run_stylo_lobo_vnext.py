#!/usr/bin/env python3
"""Run the fail-closed LOBO-vNext synthetic exploratory harness.

This entry point is deliberately only a control-plane adapter.  It has no
corpus, model, inference, scheduling, or output defaults and it never routes to
the historical A0/A1/A4 runner.  The only executable mode in this implementation
pass is an explicitly authorised synthetic dry run.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.jsonio import StrictJSONError, dumps_strict, load_strict  # noqa: E402


OUTPUT_ROOT = ROOT / "docs" / "exploratory" / "lobo_vnext"
EXPECTED_SCHEMAS = {
    "corpus_manifest": "stylo.lobo-vnext.corpus-manifest.v1",
    "content_manifest": "stylo.lobo-vnext.content-components.v1",
    "fold_manifest": "stylo.lobo-vnext.fold-manifest.v1",
    "inner_cv_plan": "stylo.lobo-vnext.inner-cv-plan.v1",
    "model_spec": "stylo.lobo-vnext.model-spec.v1",
    "inference_spec": "stylo.lobo-vnext.inference-spec.v1",
    "execution_spec": "stylo.lobo-vnext.execution-spec.v1",
}
_GKF_NAMES = frozenset(
    {
        "gkf",
        "group k fold",
        "group-k-fold",
        "group_k_fold",
        "groupkfold",
    }
)


class VNextCLIError(ValueError):
    """The CLI or evaluator rejected a noncanonical/unauthorised request."""


def _candidate_path(value: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else ROOT / path


def _reject_symlink_components(path: pathlib.Path, *, label: str) -> None:
    """Reject a symlink in any existing component without dereferencing it."""

    candidate = path.absolute()
    chain = (candidate, *candidate.parents)
    for component in chain:
        if component.is_symlink():
            raise VNextCLIError(f"{label} must not contain symlinks: {component}")


def _require_input_file(value: pathlib.Path | str, *, label: str) -> pathlib.Path:
    candidate = _candidate_path(value)
    _reject_symlink_components(candidate, label=label)
    if not candidate.is_file():
        raise VNextCLIError(f"{label} is not a regular file: {candidate}")
    return candidate.resolve(strict=True)


def _require_corpus_root(value: pathlib.Path | str) -> pathlib.Path:
    candidate = _candidate_path(value)
    _reject_symlink_components(candidate, label="corpus root")
    if not candidate.is_dir():
        raise VNextCLIError(f"corpus root is not a directory: {candidate}")
    return candidate.resolve(strict=True)


def _require_output_namespace(value: pathlib.Path | str) -> pathlib.Path:
    candidate = _candidate_path(value)
    _reject_symlink_components(candidate, label="output namespace")
    resolved_root = OUTPUT_ROOT.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise VNextCLIError(
            f"output namespace must stay under {OUTPUT_ROOT}: {resolved}"
        ) from exc
    if not relative.parts:
        raise VNextCLIError(
            "output namespace must name a run below "
            f"{OUTPUT_ROOT}, not the namespace root itself"
        )
    if resolved.exists() and not resolved.is_dir():
        raise VNextCLIError(
            f"output namespace exists but is not a directory: {resolved}"
        )
    return resolved


def _load_mapping(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        payload = load_strict(path)
    except (OSError, UnicodeError, StrictJSONError, ValueError) as exc:
        raise VNextCLIError(f"{label} is not strict JSON: {path}: {exc}") from exc
    if type(payload) is not dict:
        raise VNextCLIError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_schema(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected: str,
) -> None:
    observed = payload.get("schema_version")
    if type(observed) is not str:
        raise VNextCLIError(f"{label}.schema_version must be an exact string")
    if observed != expected:
        legacy_note = (
            " (legacy schemas are read-only)"
            if "vnext" not in observed.lower()
            else ""
        )
        raise VNextCLIError(
            f"{label}.schema_version must be {expected!r}, got {observed!r}"
            f"{legacy_note}"
        )


def _walk_strings(value: Any):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for key, nested in value.items():
            yield key
            yield from _walk_strings(nested)
    elif type(value) is list:
        for nested in value:
            yield from _walk_strings(nested)


def _reject_gkf(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    for label, payload in payloads.items():
        for value in _walk_strings(payload):
            normalised = value.strip().lower()
            if normalised in _GKF_NAMES:
                raise VNextCLIError(
                    f"{label} requests forbidden GKF strategy {value!r}; "
                    "GKF is not LOBO"
                )


def _require_literal(
    payload: Mapping[str, Any],
    *,
    label: str,
    key: str,
    expected: Any,
) -> None:
    value = payload.get(key)
    if type(value) is not type(expected) or value != expected:
        raise VNextCLIError(
            f"{label}.{key} must be the explicit literal {expected!r}, got {value!r}"
        )


def _validate_control_plane(
    *,
    corpus_manifest_path: pathlib.Path,
    content_manifest_path: pathlib.Path,
    fold_manifest_path: pathlib.Path,
    inner_cv_plan_path: pathlib.Path,
    model_spec_path: pathlib.Path,
    inference_spec_path: pathlib.Path,
    execution_spec_path: pathlib.Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "corpus_manifest": corpus_manifest_path,
        "content_manifest": content_manifest_path,
        "fold_manifest": fold_manifest_path,
        "inner_cv_plan": inner_cv_plan_path,
        "model_spec": model_spec_path,
        "inference_spec": inference_spec_path,
        "execution_spec": execution_spec_path,
    }
    payloads = {
        label: _load_mapping(path, label=label)
        for label, path in paths.items()
    }
    for label, payload in payloads.items():
        _validate_schema(
            payload,
            label=label,
            expected=EXPECTED_SCHEMAS[label],
        )

    _reject_gkf(payloads)
    execution = payloads["execution_spec"]
    _require_literal(
        execution,
        label="execution_spec",
        key="execution_mode",
        expected="synthetic_fixture",
    )
    _require_literal(
        execution,
        label="execution_spec",
        key="authorization",
        expected="approved_for_exploratory",
    )
    _require_literal(
        execution,
        label="execution_spec",
        key="evaluation_strategy",
        expected="lobo",
    )
    _require_literal(
        payloads["corpus_manifest"],
        label="corpus_manifest",
        key="corpus_kind",
        expected="synthetic_fixture",
    )
    for label in ("corpus_manifest", "model_spec", "inference_spec"):
        _require_literal(
            payloads[label],
            label=label,
            key="approved_for_exploratory",
            expected=True,
        )
        _require_literal(
            payloads[label],
            label=label,
            key="owner_selected",
            expected=False,
        )
    return payloads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run/resume the LOBO-vNext synthetic exploratory dry-run harness. "
            "Real-corpus and confirmatory execution are not authorised."
        )
    )
    parser.add_argument("--synthetic-dry-run", action="store_true", required=True)
    parser.add_argument("--corpus-root", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--content-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--fold-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--inner-cv-plan", type=pathlib.Path, required=True)
    parser.add_argument("--model-spec", type=pathlib.Path, required=True)
    parser.add_argument("--inference-spec", type=pathlib.Path, required=True)
    parser.add_argument("--execution-spec", type=pathlib.Path, required=True)
    parser.add_argument("--output-namespace", type=pathlib.Path, required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    return parser


def _run_public_entrypoint(**kwargs):
    # This is intentionally the sole hand-off from the CLI into evaluation.
    # Importing it only after local control-plane checks also keeps malformed or
    # unauthorised requests away from cache/factory construction.
    from stylo.eval.lobo_vnext import LoboVNextError, run_lobo_vnext_from_specs

    try:
        return run_lobo_vnext_from_specs(**kwargs)
    except LoboVNextError as exc:
        raise VNextCLIError(f"vNext evaluator rejected the request: {exc}") from exc


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    if type(args.n_jobs) is not int or args.n_jobs <= 0:
        raise VNextCLIError("--n-jobs must be an explicit positive integer")

    corpus_root = _require_corpus_root(args.corpus_root)
    corpus_manifest_path = _require_input_file(
        args.corpus_manifest, label="corpus manifest"
    )
    content_manifest_path = _require_input_file(
        args.content_manifest, label="content manifest"
    )
    fold_manifest_path = _require_input_file(
        args.fold_manifest, label="fold manifest"
    )
    inner_cv_plan_path = _require_input_file(
        args.inner_cv_plan, label="inner CV plan"
    )
    model_spec_path = _require_input_file(args.model_spec, label="model spec")
    inference_spec_path = _require_input_file(
        args.inference_spec, label="inference spec"
    )
    execution_spec_path = _require_input_file(
        args.execution_spec, label="execution spec"
    )
    output_namespace = _require_output_namespace(args.output_namespace)

    _validate_control_plane(
        corpus_manifest_path=corpus_manifest_path,
        content_manifest_path=content_manifest_path,
        fold_manifest_path=fold_manifest_path,
        inner_cv_plan_path=inner_cv_plan_path,
        model_spec_path=model_spec_path,
        inference_spec_path=inference_spec_path,
        execution_spec_path=execution_spec_path,
    )

    result = _run_public_entrypoint(
        corpus_root=corpus_root,
        corpus_manifest_path=corpus_manifest_path,
        content_manifest_path=content_manifest_path,
        fold_manifest_path=fold_manifest_path,
        inner_cv_plan_path=inner_cv_plan_path,
        model_spec_path=model_spec_path,
        inference_spec_path=inference_spec_path,
        execution_spec_path=execution_spec_path,
        output_namespace=output_namespace,
        n_jobs=args.n_jobs,
    )
    if not isinstance(result, Mapping):
        raise VNextCLIError("vNext evaluator returned a non-object result")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(argv)
    except VNextCLIError as exc:
        print(f"LOBO-vNext rejected: {exc}", file=sys.stderr)
        return 2

    receipt = {
        "status": "exploratory_synthetic_dry_run_complete",
        "run_id": result.get("run_id"),
        "artifact_self_hash": result.get("self_hash"),
    }
    print(
        dumps_strict(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
