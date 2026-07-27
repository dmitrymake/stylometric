from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.evaluation import run_stylo_lobo_vnext as cli
from stylo.jsonio import dump_strict


def _control_plane(tmp_path: Path) -> dict[str, Path]:
    payloads = {
        name: {"schema_version": schema}
        for name, schema in cli.EXPECTED_SCHEMAS.items()
    }
    payloads["corpus_manifest"].update(
        {
            "corpus_kind": "synthetic_fixture",
            "approved_for_exploratory": True,
            "owner_selected": False,
        }
    )
    payloads["model_spec"]["approved_for_exploratory"] = True
    payloads["model_spec"]["owner_selected"] = False
    payloads["inference_spec"]["approved_for_exploratory"] = True
    payloads["inference_spec"]["owner_selected"] = False
    payloads["execution_spec"].update(
        {
            "execution_mode": "synthetic_fixture",
            "authorization": "approved_for_exploratory",
            "evaluation_strategy": "lobo",
        }
    )
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = tmp_path / "specs" / f"{name}.json"
        dump_strict(payload, path, sort_keys=True)
        paths[name] = path
    return paths


def _argv(
    tmp_path: Path,
    paths: dict[str, Path],
    *,
    n_jobs: int = 1,
    output_namespace: Path | None = None,
) -> list[str]:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(exist_ok=True)
    output = output_namespace or (
        tmp_path / "docs" / "exploratory" / "lobo_vnext" / "synthetic-test"
    )
    return [
        "--synthetic-dry-run",
        "--corpus-root",
        str(corpus_root),
        "--corpus-manifest",
        str(paths["corpus_manifest"]),
        "--content-manifest",
        str(paths["content_manifest"]),
        "--fold-manifest",
        str(paths["fold_manifest"]),
        "--inner-cv-plan",
        str(paths["inner_cv_plan"]),
        "--model-spec",
        str(paths["model_spec"]),
        "--inference-spec",
        str(paths["inference_spec"]),
        "--execution-spec",
        str(paths["execution_spec"]),
        "--output-namespace",
        str(output),
        "--n-jobs",
        str(n_jobs),
    ]


@pytest.fixture
def isolated_cli_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output_root = tmp_path / "docs" / "exploratory" / "lobo_vnext"
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_ROOT", output_root)
    return tmp_path


def _replace_payload(path: Path, payload: dict) -> None:
    dump_strict(payload, path, sort_keys=True)


def test_cli_has_no_scientifically_meaningful_argument_defaults():
    actions = {
        action.dest: action
        for action in cli._parser()._actions
        if action.dest != "help"
    }
    assert set(actions) == {
        "synthetic_dry_run",
        "corpus_root",
        "corpus_manifest",
        "content_manifest",
        "fold_manifest",
        "inner_cv_plan",
        "model_spec",
        "inference_spec",
        "execution_spec",
        "output_namespace",
        "n_jobs",
    }
    assert all(action.required for action in actions.values())


def test_valid_synthetic_request_calls_only_public_eval_adapter(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _control_plane(isolated_cli_root)
    observed = []

    def fake_entrypoint(**kwargs):
        observed.append(kwargs)
        return {"run_id": "a" * 64, "self_hash": "b" * 64}

    monkeypatch.setattr(cli, "_run_public_entrypoint", fake_entrypoint)

    result = cli.run(_argv(isolated_cli_root, paths, n_jobs=3))

    assert result["run_id"] == "a" * 64
    assert len(observed) == 1
    call = observed[0]
    assert call["n_jobs"] == 3
    assert call["corpus_root"] == (isolated_cli_root / "corpus").resolve()
    assert call["output_namespace"] == (
        isolated_cli_root
        / "docs"
        / "exploratory"
        / "lobo_vnext"
        / "synthetic-test"
    ).resolve()
    assert {
        key for key in call if key.endswith("_path")
    } == {
        "corpus_manifest_path",
        "content_manifest_path",
        "fold_manifest_path",
        "inner_cv_plan_path",
        "model_spec_path",
        "inference_spec_path",
        "execution_spec_path",
    }


def test_duplicate_json_key_is_rejected_before_eval(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _control_plane(isolated_cli_root)
    paths["model_spec"].write_text(
        '{"schema_version":"stylo.lobo-vnext.model-spec.v1",'
        '"approved_for_exploratory":true,'
        '"approved_for_exploratory":true}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("eval reached after duplicate JSON key"),
    )

    with pytest.raises(cli.VNextCLIError, match="strict JSON.*duplicate"):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_is_rejected_before_eval(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
):
    paths = _control_plane(isolated_cli_root)
    paths["inference_spec"].write_text(
        '{"schema_version":"stylo.lobo-vnext.inference-spec.v1",'
        '"approved_for_exploratory":true,'
        f'"poison":{constant}}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("eval reached after non-finite JSON"),
    )

    with pytest.raises(cli.VNextCLIError, match="strict JSON"):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize("label", sorted(cli.EXPECTED_SCHEMAS))
def test_legacy_or_wrong_schema_is_rejected_before_eval(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
):
    paths = _control_plane(isolated_cli_root)
    payload = {
        "schema_version": "stylo.historical-lobo.v2",
        "approved_for_exploratory": True,
        "execution_mode": "synthetic_fixture",
        "authorization": "approved_for_exploratory",
        "evaluation_strategy": "lobo",
    }
    _replace_payload(paths[label], payload)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("legacy schema reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="schema_version.*legacy"):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize(
    "label,payload_update",
    [
        ("execution_spec", {"evaluation_strategy": "GKF"}),
        ("model_spec", {"hyperparameters": {"outer_cv": "GroupKFold"}}),
        ("fold_manifest", {"metadata": {"strategy": "group_k_fold"}}),
        ("inner_cv_plan", {"metadata": {"strategy": "GroupKFold"}}),
    ],
)
def test_gkf_is_never_accepted_as_lobo(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    payload_update: dict,
):
    paths = _control_plane(isolated_cli_root)
    payload = copy.deepcopy(cli.load_strict(paths[label]))
    payload.update(payload_update)
    _replace_payload(paths[label], payload)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("GKF request reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="GKF.*not LOBO"):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("execution_mode", "real_corpus", "synthetic_fixture"),
        ("execution_mode", "confirmatory", "synthetic_fixture"),
        ("authorization", "owner_approved_confirmatory", "approved_for_exploratory"),
        ("authorization", False, "approved_for_exploratory"),
        ("evaluation_strategy", "leave_one_chunk_out", "lobo"),
    ],
)
def test_non_synthetic_or_unauthorised_execution_is_rejected_before_eval(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value,
    match: str,
):
    paths = _control_plane(isolated_cli_root)
    payload = cli.load_strict(paths["execution_spec"])
    payload[key] = value
    _replace_payload(paths["execution_spec"], payload)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("unauthorised request reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match=match):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize(
    "label", ["corpus_manifest", "model_spec", "inference_spec"]
)
@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_exploratory_approval_is_exact_non_bool_coercion_free(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    value,
):
    paths = _control_plane(isolated_cli_root)
    payload = cli.load_strict(paths[label])
    payload["approved_for_exploratory"] = value
    _replace_payload(paths[label], payload)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("invalid approval reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="approved_for_exploratory"):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize(
    "label", ["corpus_manifest", "model_spec", "inference_spec"]
)
@pytest.mark.parametrize("value", [True, 0, "false", None])
def test_synthetic_specs_cannot_imply_an_owner_selection(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    value,
):
    paths = _control_plane(isolated_cli_root)
    payload = cli.load_strict(paths[label])
    payload["owner_selected"] = value
    _replace_payload(paths[label], payload)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("owner-selected spec reached synthetic eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="owner_selected.*False"):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize("corpus_kind", ["real_corpus", "historical", None, 1])
def test_only_explicit_synthetic_corpus_is_allowed(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    corpus_kind,
):
    paths = _control_plane(isolated_cli_root)
    payload = cli.load_strict(paths["corpus_manifest"])
    payload["corpus_kind"] = corpus_kind
    _replace_payload(paths["corpus_manifest"], payload)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("non-synthetic corpus reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="corpus_kind.*synthetic_fixture"):
        cli.run(_argv(isolated_cli_root, paths))


def test_output_must_stay_below_ignored_exploratory_namespace(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _control_plane(isolated_cli_root)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("unsafe output reached eval"),
    )

    unsafe = isolated_cli_root / "docs" / "headline.json"
    with pytest.raises(cli.VNextCLIError, match="must stay under"):
        cli.run(
            _argv(
                isolated_cli_root,
                paths,
                output_namespace=unsafe,
            )
        )


def test_output_namespace_root_itself_is_not_a_run_namespace(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _control_plane(isolated_cli_root)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("namespace root reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="not the namespace root"):
        cli.run(
            _argv(
                isolated_cli_root,
                paths,
                output_namespace=cli.OUTPUT_ROOT,
            )
        )


def test_symlinked_input_and_output_are_rejected_before_eval(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _control_plane(isolated_cli_root)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("symlink reached eval"),
    )

    linked_model = isolated_cli_root / "linked-model.json"
    linked_model.symlink_to(paths["model_spec"])
    symlinked_paths = dict(paths)
    symlinked_paths["model_spec"] = linked_model
    with pytest.raises(cli.VNextCLIError, match="symlinks"):
        cli.run(_argv(isolated_cli_root, symlinked_paths))

    output_parent = cli.OUTPUT_ROOT.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    target = isolated_cli_root / "redirected-output"
    target.mkdir()
    cli.OUTPUT_ROOT.symlink_to(target, target_is_directory=True)
    with pytest.raises(cli.VNextCLIError, match="symlinks"):
        cli.run(_argv(isolated_cli_root, paths))


@pytest.mark.parametrize("n_jobs", [0, -1])
def test_n_jobs_must_be_explicitly_positive_before_eval(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_jobs: int,
):
    paths = _control_plane(isolated_cli_root)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("invalid n_jobs reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="positive integer"):
        cli.run(_argv(isolated_cli_root, paths, n_jobs=n_jobs))


def test_missing_manifest_is_rejected_before_eval(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _control_plane(isolated_cli_root)
    paths["content_manifest"].unlink()
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: pytest.fail("missing manifest reached eval"),
    )

    with pytest.raises(cli.VNextCLIError, match="not a regular file"):
        cli.run(_argv(isolated_cli_root, paths))


def test_evaluator_must_return_an_object(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _control_plane(isolated_cli_root)
    monkeypatch.setattr(cli, "_run_public_entrypoint", lambda **_: [])

    with pytest.raises(cli.VNextCLIError, match="non-object"):
        cli.run(_argv(isolated_cli_root, paths))


def test_main_emits_only_exploratory_receipt(
    isolated_cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    paths = _control_plane(isolated_cli_root)
    monkeypatch.setattr(
        cli,
        "_run_public_entrypoint",
        lambda **_: {
            "run_id": "1" * 64,
            "self_hash": "2" * 64,
            "accuracy": 0.999,
        },
    )

    assert cli.main(_argv(isolated_cli_root, paths)) == 0
    captured = capsys.readouterr()
    assert "exploratory_synthetic_dry_run_complete" in captured.out
    assert "accuracy" not in captured.out
    assert captured.err == ""
