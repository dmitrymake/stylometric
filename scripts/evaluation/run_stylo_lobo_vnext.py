#!/usr/bin/env python3
"""Run the fail-closed LOBO-vNext exploratory harness.

This entry point is deliberately only a control-plane adapter.  It has no
corpus, model, inference, scheduling, or output defaults and it never routes to
the historical A0/A1/A4 runner.  Synthetic fixtures retain their original CLI
surface.  Real-corpus operation has two separate explicit modes: an
authorization-free, fit-free execution-spec preflight and an exact
interactive-authorization-bound bounded exploratory dry run.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stylo.jsonio import StrictJSONError, dumps_strict, load_strict  # noqa: E402


OUTPUT_ROOT = ROOT / "docs" / "exploratory" / "lobo_vnext"
REAL_OUTPUT_RELATIVE = pathlib.PurePosixPath(
    "docs/exploratory/lobo_vnext/real_corpus"
)
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


_MODE_FLAGS = {
    "--synthetic-dry-run": "synthetic",
    "--real-preflight": "real_preflight",
    "--real-exploratory-dry-run": "real_exploratory",
}
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


def _requested_mode(argv: Sequence[str]) -> str:
    selected = [
        mode for flag, mode in _MODE_FLAGS.items() if flag in argv
    ]
    if len(selected) > 1:
        raise VNextCLIError(
            "exactly one execution mode must be selected; "
            "synthetic and real modes cannot be combined"
        )
    if selected:
        return selected[0]
    # Preserve the original parser's required --synthetic-dry-run diagnostic
    # when no mode was supplied.
    return "synthetic"


def _parser() -> argparse.ArgumentParser:
    """Return the original synthetic-only parser unchanged."""

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


def _real_preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reload an immutable R1 packet, derive live clean-repository "
            "identity, and create an authorization-unbound ExecutionSpec v2. "
            "This mode cannot construct a model factory or fit."
        )
    )
    parser.add_argument("--real-preflight", action="store_true", required=True)
    parser.add_argument("--packet-root", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument(
        "--execution-spec-output", type=pathlib.Path, required=True
    )
    return parser


def _real_exploratory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run/resume the exact authorization-bound bounded real-corpus R1 "
            "exploratory campaign.  Confirmatory and public output are "
            "not authorised."
        )
    )
    parser.add_argument(
        "--real-exploratory-dry-run", action="store_true", required=True
    )
    parser.add_argument("--packet-root", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--execution-spec", type=pathlib.Path, required=True)
    parser.add_argument("--approval-record", type=pathlib.Path, required=True)
    parser.add_argument("--output-namespace", type=pathlib.Path, required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    return parser


def _require_packet_root(value: pathlib.Path | str) -> pathlib.Path:
    candidate = _candidate_path(value)
    _reject_symlink_components(candidate, label="prepared packet root")
    if candidate.is_symlink() or not candidate.is_dir():
        raise VNextCLIError(
            "prepared packet root must be a real non-symlink directory: "
            f"{candidate}"
        )
    return candidate.resolve(strict=True)


def _exact_real_output_root() -> pathlib.Path:
    return ROOT.joinpath(*REAL_OUTPUT_RELATIVE.parts).resolve(strict=False)


def _require_real_output_namespace(
    value: pathlib.Path | str,
) -> pathlib.Path:
    candidate = _candidate_path(value)
    _reject_symlink_components(candidate, label="real output namespace")
    resolved = candidate.resolve(strict=False)
    expected = _exact_real_output_root()
    if resolved != expected:
        raise VNextCLIError(
            "real output namespace must equal the exact ignored exploratory "
            f"root {expected}, got {resolved}"
        )
    if resolved.exists() and not resolved.is_dir():
        raise VNextCLIError(
            f"real output namespace is not a directory: {resolved}"
        )
    return resolved


def _require_real_execution_output(
    value: pathlib.Path | str,
) -> pathlib.Path:
    candidate = _candidate_path(value)
    _reject_symlink_components(candidate, label="execution spec output")
    resolved = candidate.resolve(strict=False)
    expected_root = _exact_real_output_root()
    try:
        relative = resolved.relative_to(expected_root)
    except ValueError as exc:
        raise VNextCLIError(
            "execution spec output must stay below the exact ignored "
            f"exploratory root {expected_root}: {resolved}"
        ) from exc
    if not relative.parts:
        raise VNextCLIError(
            "execution spec output must name a JSON file below the real "
            "exploratory root"
        )
    if resolved.suffix != ".json":
        raise VNextCLIError("execution spec output must use a .json filename")
    if resolved.exists() or resolved.is_symlink():
        raise VNextCLIError(
            "execution spec output is immutable create-if-absent and already "
            f"exists: {resolved}"
        )
    return resolved


def _load_real_packet(packet_root: pathlib.Path):
    from stylo.eval.lobo_vnext_control import load_prepared_r1_packet

    return load_prepared_r1_packet(packet_root)


def _load_real_config(config_path: pathlib.Path):
    from stylo.config import load_config

    return load_config(config_path)


def _load_real_execution(execution_spec_path: pathlib.Path):
    from stylo.domain.lobo_vnext_real import load_real_execution_spec

    return load_real_execution_spec(execution_spec_path)


def _load_real_approval_record(approval_record_path: pathlib.Path):
    from stylo.domain.lobo_vnext_approval import load_owner_decision_record

    return load_owner_decision_record(approval_record_path)


def _assemble_real_execution(*, packet, cfg):
    from stylo.eval.lobo_vnext_control import assemble_real_execution_spec

    return assemble_real_execution_spec(
        packet=packet,
        cfg=cfg,
        repository_root=ROOT,
    )


def _validate_real_execution_literals(execution: Any) -> None:
    required = {
        "schema_version": "stylo.lobo-vnext.execution-spec.v2",
        "execution_mode": "real_corpus",
        "authorization_scope": (
            "owner_bound_real_corpus_exploratory_dry_run_only"
        ),
        "evaluation_strategy": "lobo",
        "confirmatory_execution_authorized": False,
        "public_evidence_update_authorized": False,
        "headline_update_authorized": False,
        "frozen_evidence_mutation_authorized": False,
    }
    for field, expected in required.items():
        observed = getattr(execution, field, None)
        if type(observed) is not type(expected) or observed != expected:
            if field == "evaluation_strategy" and type(observed) is str:
                if observed.strip().lower() in _GKF_NAMES:
                    raise VNextCLIError(
                        "real execution requests forbidden GKF strategy; "
                        "GKF is not LOBO"
                    )
            raise VNextCLIError(
                f"real execution {field} must be the exact literal "
                f"{expected!r}, got {observed!r}"
            )


def _validate_real_approval_record(approval_record: Any) -> None:
    for personal_field in ("owner", "owner_id", "owner_role"):
        if hasattr(approval_record, personal_field):
            raise VNextCLIError(
                "approval record must use the non-personal v2 schema; "
                f"legacy personal field {personal_field!r} is forbidden"
            )
    required = {
        "authorization_basis": "explicit_interactive_user_authorization",
        "approved_for_exploratory": True,
        "confirmatory_execution_authorized": False,
        "public_evidence_update_authorized": False,
        "headline_update_authorized": False,
        "frozen_evidence_mutation_authorized": False,
    }
    for field, expected in required.items():
        observed = getattr(approval_record, field, None)
        if type(observed) is not type(expected) or observed != expected:
            raise VNextCLIError(
                f"approval record {field} must be the exact literal "
                f"{expected!r}, got {observed!r}"
            )


def _write_execution_spec_create_if_absent(
    path: pathlib.Path,
    execution: Any,
) -> None:
    to_dict = getattr(execution, "to_dict", None)
    if not callable(to_dict):
        raise VNextCLIError(
            "assembled real execution spec is not canonically serializable"
        )
    encoded = (
        dumps_strict(
            to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(path, label="execution spec output")
        temporary: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".execution-spec.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as stream:
                temporary = pathlib.Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o644)
            os.link(temporary, path, follow_symlinks=False)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except FileExistsError as exc:
        raise VNextCLIError(
            "execution spec output is immutable create-if-absent and already "
            f"exists: {path}"
        ) from exc
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise VNextCLIError(
            f"cannot publish immutable execution spec {path}: {exc}"
        ) from exc


def _run_real_entrypoint(**kwargs):
    from stylo.eval.lobo_vnext_real import (
        RealLoboVNextError,
        run_lobo_vnext_real,
    )

    try:
        return run_lobo_vnext_real(**kwargs)
    except RealLoboVNextError as exc:
        raise VNextCLIError(
            f"real vNext evaluator rejected the request: {exc}"
        ) from exc


def _run_public_entrypoint(**kwargs):
    # This is intentionally the sole hand-off from the CLI into evaluation.
    # Importing it only after local control-plane checks also keeps malformed or
    # unauthorised requests away from cache/factory construction.
    from stylo.eval.lobo_vnext import LoboVNextError, run_lobo_vnext_from_specs

    try:
        return run_lobo_vnext_from_specs(**kwargs)
    except LoboVNextError as exc:
        raise VNextCLIError(f"vNext evaluator rejected the request: {exc}") from exc


def _run_synthetic(argv: Sequence[str]) -> Mapping[str, Any]:
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


def _run_real_preflight(argv: Sequence[str]) -> Mapping[str, Any]:
    args = _real_preflight_parser().parse_args(argv)
    packet_root = _require_packet_root(args.packet_root)
    config_path = _require_input_file(args.config, label="real config")
    output_path = _require_real_execution_output(
        args.execution_spec_output
    )
    try:
        packet = _load_real_packet(packet_root)
        cfg = _load_real_config(config_path)
        execution, _observations = _assemble_real_execution(
            packet=packet,
            cfg=cfg,
        )
        _validate_real_execution_literals(execution)
        _write_execution_spec_create_if_absent(output_path, execution)
    except VNextCLIError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise VNextCLIError(f"real preflight rejected: {exc}") from exc
    return {
        "status": "real_corpus_execution_spec_preflight_complete_no_fit",
        "execution_spec_digest": execution.self_hash,
        "packet_self_hash": packet.packet_manifest.self_hash,
        "campaign_manifest_digest": packet.campaign_manifest.self_hash,
        "approval_record_present": False,
        "fit_performed": False,
        "confirmatory_authorized": False,
    }


def _run_real_exploratory(argv: Sequence[str]) -> Mapping[str, Any]:
    args = _real_exploratory_parser().parse_args(argv)
    if type(args.n_jobs) is not int or not 1 <= args.n_jobs <= 8:
        raise VNextCLIError(
            "--n-jobs must be an explicit integer in [1, 8] for real R1"
        )
    packet_root = _require_packet_root(args.packet_root)
    config_path = _require_input_file(args.config, label="real config")
    execution_path = _require_input_file(
        args.execution_spec, label="real execution spec"
    )
    approval_path = _require_input_file(
        args.approval_record, label="approval record"
    )
    output_namespace = _require_real_output_namespace(
        args.output_namespace
    )
    try:
        packet = _load_real_packet(packet_root)
        cfg = _load_real_config(config_path)
        supplied_execution = _load_real_execution(execution_path)
        _validate_real_execution_literals(supplied_execution)
        approval_record = _load_real_approval_record(approval_path)
        _validate_real_approval_record(approval_record)
        assembled_execution, observations = _assemble_real_execution(
            packet=packet,
            cfg=cfg,
        )
        _validate_real_execution_literals(assembled_execution)
        if supplied_execution != assembled_execution:
            raise VNextCLIError(
                "supplied ExecutionSpec v2 differs from the exact live "
                "clean-repository assembly"
            )
        supplied_execution.assert_owner_decision(approval_record)
        outcome = _run_real_entrypoint(
            packet_root=packet_root,
            packet_manifest=packet.packet_manifest,
            corpus_manifest=packet.corpus_manifest,
            content_policy_spec=packet.content_policy,
            candidate_inventory=packet.candidate_inventory,
            content_manifest=packet.content_manifest,
            fold_manifest=packet.fold_manifest,
            primary_inner_cv_plan=packet.primary_inner_cv_plan,
            baseline_inner_cv_plan=packet.baseline_inner_cv_plan,
            primary_model_spec=packet.primary_model_spec,
            baseline_model_spec=packet.baseline_model_spec,
            inference_spec=packet.inference_spec,
            model_role_manifest=packet.model_role_manifest,
            campaign_manifest=packet.campaign_manifest,
            execution_spec=supplied_execution,
            owner_decision=approval_record,
            representation_receipt=packet.representation_receipt,
            cfg=cfg,
            observations=observations,
            output_namespace=output_namespace,
            n_jobs=args.n_jobs,
        )
    except VNextCLIError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise VNextCLIError(
            f"real exploratory dry run rejected before completion: {exc}"
        ) from exc
    artifact = getattr(outcome, "artifact", None)
    run_id = getattr(outcome, "run_id", None)
    if (
        type(artifact) is not dict
        or type(artifact.get("self_hash")) is not str
        or type(run_id) is not str
    ):
        raise VNextCLIError(
            "real vNext evaluator returned a noncanonical outcome"
        )
    computed = getattr(outcome, "computed_checkpoints", None)
    resumed = getattr(outcome, "resumed_checkpoints", None)
    if (
        type(computed) is not int
        or computed < 0
        or type(resumed) is not int
        or resumed < 0
    ):
        raise VNextCLIError(
            "real vNext evaluator returned invalid checkpoint counts"
        )
    return {
        "status": "bounded_real_corpus_exploratory_dry_run_complete",
        "run_id": run_id,
        "artifact_self_hash": artifact["self_hash"],
        "computed_checkpoints": computed,
        "resumed_checkpoints": resumed,
        "confirmatory_authorized": False,
        "public_update_authorized": False,
    }


def run(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    actual_argv = tuple(sys.argv[1:] if argv is None else argv)
    mode = _requested_mode(actual_argv)
    if mode == "real_preflight":
        return _run_real_preflight(actual_argv)
    if mode == "real_exploratory":
        return _run_real_exploratory(actual_argv)
    return _run_synthetic(actual_argv)


def main(argv: Sequence[str] | None = None) -> int:
    actual_argv = tuple(sys.argv[1:] if argv is None else argv)
    try:
        mode = _requested_mode(actual_argv)
        result = run(actual_argv)
    except VNextCLIError as exc:
        print(f"LOBO-vNext rejected: {exc}", file=sys.stderr)
        return 2

    if mode == "synthetic":
        receipt = {
            "status": "exploratory_synthetic_dry_run_complete",
            "run_id": result.get("run_id"),
            "artifact_self_hash": result.get("self_hash"),
        }
    else:
        receipt = dict(result)
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
