from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation import run_stylo_lobo_vnext as cli


class _FakeExecution:
    schema_version = "stylo.lobo-vnext.execution-spec.v2"
    execution_mode = "real_corpus"
    authorization_scope = (
        "owner_bound_real_corpus_exploratory_dry_run_only"
    )
    evaluation_strategy = "lobo"
    confirmatory_execution_authorized = False
    public_evidence_update_authorized = False
    headline_update_authorized = False
    frozen_evidence_mutation_authorized = False

    def __init__(self, digest: str, owner_calls: list[object] | None = None):
        self.self_hash = digest
        self._owner_calls = owner_calls

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is _FakeExecution
            and self.self_hash == other.self_hash
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_mode": self.execution_mode,
            "authorization_scope": self.authorization_scope,
            "evaluation_strategy": self.evaluation_strategy,
            "safety": {
                "confirmatory_execution_authorized": False,
                "public_evidence_update_authorized": False,
                "headline_update_authorized": False,
                "frozen_evidence_mutation_authorized": False,
            },
            "self_hash": self.self_hash,
        }

    def assert_owner_decision(self, owner: object) -> "_FakeExecution":
        if self._owner_calls is not None:
            self._owner_calls.append(owner)
        return self


def _packet() -> SimpleNamespace:
    names = (
        "packet_manifest",
        "corpus_manifest",
        "content_policy",
        "candidate_inventory",
        "content_manifest",
        "fold_manifest",
        "primary_inner_cv_plan",
        "baseline_inner_cv_plan",
        "primary_model_spec",
        "baseline_model_spec",
        "inference_spec",
        "model_role_manifest",
        "campaign_manifest",
        "representation_receipt",
    )
    return SimpleNamespace(
        **{
            name: SimpleNamespace(self_hash=f"{index + 1:064x}")
            for index, name in enumerate(names)
        }
    )


@pytest.fixture
def real_cli_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "OUTPUT_ROOT",
        tmp_path / "docs" / "exploratory" / "lobo_vnext",
    )
    return tmp_path


def _files(root: Path) -> dict[str, Path]:
    packet_root = root / "packet"
    packet_root.mkdir()
    paths = {
        "packet_root": packet_root,
        "config": root / "config.yaml",
        "execution": root / "execution.json",
        "owner": root / "owner.json",
    }
    for key in ("config", "execution", "owner"):
        paths[key].write_text("{}\n", encoding="utf-8")
    return paths


def _real_argv(
    root: Path,
    paths: dict[str, Path],
    *,
    output: Path | None = None,
    n_jobs: int = 3,
) -> list[str]:
    return [
        "--real-exploratory-dry-run",
        "--packet-root",
        str(paths["packet_root"]),
        "--config",
        str(paths["config"]),
        "--execution-spec",
        str(paths["execution"]),
        "--owner-decision",
        str(paths["owner"]),
        "--output-namespace",
        str(
            output
            or root
            / "docs"
            / "exploratory"
            / "lobo_vnext"
            / "real_corpus"
        ),
        "--n-jobs",
        str(n_jobs),
    ]


def _patch_valid_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    packet: SimpleNamespace,
    execution: _FakeExecution,
    owner: object,
    cfg: object,
    observations: tuple[object, ...] = ("observations",),
) -> None:
    monkeypatch.setattr(cli, "_load_real_packet", lambda _: packet)
    monkeypatch.setattr(cli, "_load_real_config", lambda _: cfg)
    monkeypatch.setattr(cli, "_load_real_execution", lambda _: execution)
    monkeypatch.setattr(
        cli, "_load_real_owner_decision", lambda _: owner
    )
    monkeypatch.setattr(
        cli,
        "_assemble_real_execution",
        lambda **_: (execution, observations),
    )


def test_real_parsers_have_no_scientific_defaults_and_require_every_input():
    preflight = {
        action.dest: action
        for action in cli._real_preflight_parser()._actions
        if action.dest != "help"
    }
    assert set(preflight) == {
        "real_preflight",
        "packet_root",
        "config",
        "execution_spec_output",
    }
    assert all(action.required for action in preflight.values())

    exploratory = {
        action.dest: action
        for action in cli._real_exploratory_parser()._actions
        if action.dest != "help"
    }
    assert set(exploratory) == {
        "real_exploratory_dry_run",
        "packet_root",
        "config",
        "execution_spec",
        "owner_decision",
        "output_namespace",
        "n_jobs",
    }
    assert all(action.required for action in exploratory.values())

    with pytest.raises(SystemExit):
        cli.run(["--real-preflight"])
    with pytest.raises(SystemExit):
        cli.run(["--real-exploratory-dry-run"])


def test_modes_are_mutually_exclusive_before_any_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        cli,
        "_load_real_packet",
        lambda _: pytest.fail("loader reached after mixed modes"),
    )
    with pytest.raises(cli.VNextCLIError, match="cannot be combined"):
        cli.run(
            [
                "--synthetic-dry-run",
                "--real-preflight",
            ]
        )


def test_real_preflight_writes_immutable_owner_free_spec_without_evaluator(
    real_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _files(real_cli_root)
    packet = _packet()
    cfg = object()
    execution = _FakeExecution("a" * 64)
    monkeypatch.setattr(cli, "_load_real_packet", lambda _: packet)
    monkeypatch.setattr(cli, "_load_real_config", lambda _: cfg)
    monkeypatch.setattr(
        cli,
        "_assemble_real_execution",
        lambda **kwargs: (
            execution,
            (kwargs["packet"], kwargs["cfg"]),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_real_owner_decision",
        lambda _: pytest.fail("preflight attempted to load an owner record"),
    )
    monkeypatch.setattr(
        cli,
        "_run_real_entrypoint",
        lambda **_: pytest.fail("preflight reached evaluator"),
    )
    output = (
        real_cli_root
        / "docs"
        / "exploratory"
        / "lobo_vnext"
        / "real_corpus"
        / "control"
        / "execution.json"
    )
    argv = [
        "--real-preflight",
        "--packet-root",
        str(paths["packet_root"]),
        "--config",
        str(paths["config"]),
        "--execution-spec-output",
        str(output),
    ]

    receipt = cli.run(argv)

    assert receipt == {
        "status": "real_corpus_execution_spec_preflight_complete_no_fit",
        "execution_spec_digest": "a" * 64,
        "packet_self_hash": packet.packet_manifest.self_hash,
        "campaign_manifest_digest": packet.campaign_manifest.self_hash,
        "owner_decision_present": False,
        "fit_performed": False,
        "confirmatory_authorized": False,
    }
    assert json.loads(output.read_text(encoding="utf-8"))[
        "self_hash"
    ] == "a" * 64
    with pytest.raises(cli.VNextCLIError, match="create-if-absent"):
        cli.run(argv)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("execution_mode", "synthetic_fixture", "execution_mode"),
        ("evaluation_strategy", "GKF", "GKF.*not LOBO"),
        (
            "authorization_scope",
            "owner_approved_confirmatory",
            "authorization_scope",
        ),
    ],
)
def test_wrong_real_mode_or_gkf_is_rejected_before_assembly_and_eval(
    real_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    match: str,
):
    paths = _files(real_cli_root)
    execution = _FakeExecution("b" * 64)
    setattr(execution, field, value)
    monkeypatch.setattr(cli, "_load_real_packet", lambda _: _packet())
    monkeypatch.setattr(cli, "_load_real_config", lambda _: object())
    monkeypatch.setattr(cli, "_load_real_execution", lambda _: execution)
    monkeypatch.setattr(
        cli,
        "_assemble_real_execution",
        lambda **_: pytest.fail("invalid execution reached assembly"),
    )
    monkeypatch.setattr(
        cli,
        "_run_real_entrypoint",
        lambda **_: pytest.fail("invalid execution reached evaluator"),
    )

    with pytest.raises(cli.VNextCLIError, match=match):
        cli.run(_real_argv(real_cli_root, paths))


@pytest.mark.parametrize(
    "owner_id,owner_role",
    [
        ("…", "scientific owner"),
        ("owner:real", "..."),
        ("placeholder", "scientific owner"),
    ],
)
def test_placeholder_owner_is_rejected_before_live_assembly_or_eval(
    real_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_id: str,
    owner_role: str,
):
    paths = _files(real_cli_root)
    execution = _FakeExecution("c" * 64)
    monkeypatch.setattr(cli, "_load_real_packet", lambda _: _packet())
    monkeypatch.setattr(cli, "_load_real_config", lambda _: object())
    monkeypatch.setattr(cli, "_load_real_execution", lambda _: execution)
    monkeypatch.setattr(
        cli,
        "_load_real_owner_decision",
        lambda _: SimpleNamespace(
            owner_id=owner_id,
            owner_role=owner_role,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_assemble_real_execution",
        lambda **_: pytest.fail("placeholder owner reached assembly"),
    )
    monkeypatch.setattr(
        cli,
        "_run_real_entrypoint",
        lambda **_: pytest.fail("placeholder owner reached evaluator"),
    )

    with pytest.raises(cli.VNextCLIError, match="placeholder"):
        cli.run(_real_argv(real_cli_root, paths))


def test_execution_drift_is_rejected_before_owner_binding_and_eval(
    real_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _files(real_cli_root)
    owner_calls: list[object] = []
    supplied = _FakeExecution("d" * 64, owner_calls)
    assembled = _FakeExecution("e" * 64, owner_calls)
    owner = SimpleNamespace(
        owner_id="owner:real",
        owner_role="scientific owner",
    )
    _patch_valid_loaders(
        monkeypatch,
        packet=_packet(),
        execution=supplied,
        owner=owner,
        cfg=object(),
    )
    monkeypatch.setattr(
        cli,
        "_assemble_real_execution",
        lambda **_: (assembled, ("observations",)),
    )
    monkeypatch.setattr(
        cli,
        "_run_real_entrypoint",
        lambda **_: pytest.fail("drifted execution reached evaluator"),
    )

    with pytest.raises(cli.VNextCLIError, match="differs.*live"):
        cli.run(_real_argv(real_cli_root, paths))
    assert owner_calls == []


def test_real_output_namespace_must_equal_exact_ignored_root_before_load(
    real_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _files(real_cli_root)
    monkeypatch.setattr(
        cli,
        "_load_real_packet",
        lambda _: pytest.fail("unsafe output reached packet loader"),
    )

    with pytest.raises(cli.VNextCLIError, match="must equal"):
        cli.run(
            _real_argv(
                real_cli_root,
                paths,
                output=real_cli_root / "docs" / "exploratory" / "wrong",
            )
        )


def test_exact_real_handoff_reuses_reloaded_packet_execution_and_receipts(
    real_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _files(real_cli_root)
    packet = _packet()
    cfg = object()
    observations = (object(), object())
    owner_calls: list[object] = []
    execution = _FakeExecution("f" * 64, owner_calls)
    owner = SimpleNamespace(
        owner_id="owner:real",
        owner_role="scientific owner",
    )
    _patch_valid_loaders(
        monkeypatch,
        packet=packet,
        execution=execution,
        owner=owner,
        cfg=cfg,
        observations=observations,
    )
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            artifact={"self_hash": "1" * 64, "accuracy": 0.99},
            run_id="2" * 64,
            computed_checkpoints=17,
            resumed_checkpoints=255,
        )

    monkeypatch.setattr(cli, "_run_real_entrypoint", fake_runner)

    receipt = cli.run(_real_argv(real_cli_root, paths, n_jobs=4))

    assert owner_calls == [owner]
    assert len(calls) == 1
    call = calls[0]
    assert call == {
        "packet_root": paths["packet_root"].resolve(),
        "packet_manifest": packet.packet_manifest,
        "corpus_manifest": packet.corpus_manifest,
        "content_policy_spec": packet.content_policy,
        "candidate_inventory": packet.candidate_inventory,
        "content_manifest": packet.content_manifest,
        "fold_manifest": packet.fold_manifest,
        "primary_inner_cv_plan": packet.primary_inner_cv_plan,
        "baseline_inner_cv_plan": packet.baseline_inner_cv_plan,
        "primary_model_spec": packet.primary_model_spec,
        "baseline_model_spec": packet.baseline_model_spec,
        "inference_spec": packet.inference_spec,
        "model_role_manifest": packet.model_role_manifest,
        "campaign_manifest": packet.campaign_manifest,
        "execution_spec": execution,
        "owner_decision": owner,
        "representation_receipt": packet.representation_receipt,
        "cfg": cfg,
        "observations": observations,
        "output_namespace": (
            real_cli_root
            / "docs"
            / "exploratory"
            / "lobo_vnext"
            / "real_corpus"
        ).resolve(),
        "n_jobs": 4,
    }
    assert receipt == {
        "status": "bounded_real_corpus_exploratory_dry_run_complete",
        "run_id": "2" * 64,
        "artifact_self_hash": "1" * 64,
        "computed_checkpoints": 17,
        "resumed_checkpoints": 255,
        "confirmatory_authorized": False,
        "public_update_authorized": False,
    }
    assert "accuracy" not in receipt
