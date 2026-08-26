from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
GOVERNANCE = ROOT / "research" / "governance"
HEX64 = set("0123456789abcdef")


def _strict_json(path: pathlib.Path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r} in {path}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _symbols(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            found.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
    return found


def _assert_binding(binding: dict, *, require_hash: bool) -> None:
    expected = {"path", "symbol", *(["sha256"] if require_hash else [])}
    assert set(binding) == expected
    path = ROOT / binding["path"]
    assert path.is_file()
    assert binding["symbol"] in _symbols(path)
    if require_hash:
        assert len(binding["sha256"]) == 64
        assert set(binding["sha256"]) <= HEX64
        assert _sha256(path) == binding["sha256"]


def test_status_ledger_keeps_every_authorization_latch_closed():
    """The ledger may be reworded freely; the latches that gate publication may not drift."""
    ledger = _strict_json(GOVERNANCE / "status_ledger.json")
    assert set(ledger) == {
        "schema", "as_of", "authority", "paired_audit", "bounded_exploratory_milestones"
    }
    assert ledger["schema"] == "stylo.governance.status_ledger.v2"
    latches = {
        "manifest_freeze": "unapproved",
        "production_evaluator": "unregistered",
        "preflight": "absent",
        "execution_authorization": "absent",
        "confirmatory_execution": "hard_disabled",
        "headline": "not_authorized",
    }
    for key, expected in latches.items():
        assert ledger["paired_audit"][key]["status"] == expected, key
    for state in ledger["paired_audit"].values():
        assert set(state) in ({"status", "claim"}, {"status", "claim", "bindings"})
        assert state["status"] and state["claim"]
        for binding in state.get("bindings", []):
            _assert_binding(binding, require_hash=False)


def test_bounded_exploratory_milestone_is_exact_and_non_authorizing():
    ledger = _strict_json(GOVERNANCE / "status_ledger.json")
    milestones = ledger["bounded_exploratory_milestones"]
    assert set(milestones) == {"ruaa_r1_v5"}
    milestone = milestones["ruaa_r1_v5"]
    assert set(milestone) == {
        "status",
        "source_commit",
        "packet_generation_id",
        "run_id",
        "result_self_hash",
        "result_file_sha256",
        "sealed_bundle_manifest_self_hash",
        "sealed_bundle_archive_sha256",
        "authorization_exhausted",
        "evidence_tier",
        "evidence_location",
        "release_artifact",
        "confirmatory_authorized",
        "publication_authorized",
        "headline_authorized",
        "claim",
    }
    assert {
        key: milestone[key]
        for key in (
            "status",
            "source_commit",
            "packet_generation_id",
            "run_id",
            "result_self_hash",
            "result_file_sha256",
            "sealed_bundle_manifest_self_hash",
            "sealed_bundle_archive_sha256",
            "evidence_tier",
            "evidence_location",
            "claim",
        )
    } == {
        "status": "completed_bounded_exploratory_local_not_published",
        "source_commit": "3c17766c3154fd515e7b5788e1b0278be108f2e1",
        "packet_generation_id": "d08f8cb772d70df33f5356a3b043bc26345e91d41b6c2da408da694753a13770",
        "run_id": "c3a97f662cc65788f5ac859cd6ac80f9f6353e95ddc1ce96e5beafb318c7fa3e",
        "result_self_hash": "d1ddf80a83ff844bbd04a5276b6d8985f1073220414204638d2cf6c24153011b",
        "result_file_sha256": "78e6d7b48457e050f2e5b6a60f4e6dca9edccfabe1c3b56bcc5822ac21ca8580",
        "sealed_bundle_manifest_self_hash": "224665b40117ef5ddd93c496b9683c0250c26120ba868227f19c3fb5cc9e2d88",
        "sealed_bundle_archive_sha256": "e37dfeaf7f5fc423459522b6eb480dce49bdb6fcb79623342ee2e18dfd37df0f",
        "evidence_tier": "bounded_exploratory",
        "evidence_location": "ignored-local",
        "claim": (
            "This bounded exploratory run does not complete or replace the paired "
            "audit, is not confirmatory evidence or an external replication, and "
            "authorizes no publication or headline."
        ),
    }
    assert milestone["authorization_exhausted"] is True
    for field in (
        "release_artifact",
        "confirmatory_authorized",
        "publication_authorized",
        "headline_authorized",
    ):
        assert milestone[field] is False


def test_corpus_exclusion_arithmetic_is_mechanically_accounted():
    """The three excluded works reconcile across every registry that counts them."""
    ledger = _strict_json(GOVERNANCE / "status_ledger.json")
    registry = _strict_json(
        ROOT / "research" / "evidence" / "ineligible_corpus_registrations_v1.json"
    )
    dispositions = _strict_json(
        ROOT / "research" / "corpus_sources" / "ruaa_r1_source_dispositions_v1.json"
    )
    lobo = _strict_json(ROOT / "docs" / "screening_panel_v1.json")
    ruaa = _strict_json(ROOT / "docs" / "ruaa_bench_manifest.json")

    excluded = {
        "turgenev/записки_охотника",
        "serafimovich/у_нас_и_у_них",
        "sevsky/дон_на_костылях",
    }
    lobo_ids = {row["work_id"] for row in lobo["works"]}
    ruaa_ids = {
        f"{author}/{book['book']}"
        for author, author_record in ruaa["authors"].items()
        for book in author_record["books"]
    }
    disposition_ids = {row["work_id"] for row in dispositions["work_dispositions"]}

    assert registry["status"] == "ineligible_for_new_scientific_runs"
    assert excluded <= disposition_ids
    assert len(lobo_ids) == lobo["n_works"] == 251
    assert len(ruaa_ids) == ruaa["n_books"] == 137
    assert excluded <= lobo_ids and excluded <= ruaa_ids
    assert len(lobo_ids - excluded) == 248
    assert len(ruaa_ids - excluded) == 134
    # The tracked independent audit fixes the immutable historical parent count at 255.
    historical_audit = (
        ROOT / "research" / "evidence" / "stylo_lobo_validation_v1" / "independent_audit.md"
    ).read_text(encoding="utf-8")
    assert "47 classes, 255 works, and 251 tested works" in historical_audit
    assert 255 - len(excluded) == 252

    assert ledger["paired_audit"]["protocol_v3_1"]["status"] == (
        "superseded_ineligible_corpus"
    )


def _provenance_registry(generator: bytes, source: bytes, output: bytes) -> dict:
    digest = lambda value: hashlib.sha256(value).hexdigest()
    return {
        "schema": "stylo.site_generation_provenance.v2",
        "generator": {
            "path": "scripts/gen-site-data.mjs",
            "sha256": digest(generator),
        },
        "sources": [{"path": "docs/source.json", "sha256": digest(source)}],
        "outputs": [{
            "path": "site/src/generated/site-data.json",
            "sha256": digest(output),
        }],
        "entries": [{
            "key": "rendered",
            "sources": ["docs/source.json"],
            "note": "synthetic field binding",
        }],
    }


def _run_checker(
    root: pathlib.Path,
    *,
    mode: str = "test-skip",
) -> subprocess.CompletedProcess[str]:
    mode_args = {
        "checkout": [],
        "test-skip": ["--skip-tracked"],
        "archive": ["--archive"],
    }[mode]
    return subprocess.run(
        [
            "node",
            str(ROOT / "scripts" / "check-provenance.mjs"),
            "--root",
            str(root),
            *mode_args,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_typed_site_provenance_detects_source_and_output_mutation(tmp_path):
    generator = b"// synthetic generator\n"
    source = b'{"value": 1}\n'
    output = b'{"rendered": 1}\n'
    paths = {
        "generator": tmp_path / "scripts" / "gen-site-data.mjs",
        "source": tmp_path / "docs" / "source.json",
        "output": tmp_path / "site" / "src" / "generated" / "site-data.json",
        "manifest": tmp_path / "site" / "src" / "generated" / "manifest.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["generator"].write_bytes(generator)
    paths["source"].write_bytes(source)
    paths["output"].write_bytes(output)
    paths["manifest"].write_text(
        json.dumps(_provenance_registry(generator, source, output)) + "\n",
        encoding="utf-8",
    )
    assert _run_checker(tmp_path).returncode == 0

    paths["source"].write_bytes(b'{"value": 2}\n')
    source_failure = _run_checker(tmp_path)
    assert source_failure.returncode == 1
    assert "digest mismatch" in source_failure.stderr

    paths["source"].write_bytes(source)
    paths["output"].write_bytes(b'{"rendered": 2}\n')
    output_failure = _run_checker(tmp_path)
    assert output_failure.returncode == 1
    assert "digest mismatch" in output_failure.stderr


def test_typed_site_provenance_rejects_forged_field_map(tmp_path):
    generator = b"// synthetic generator\n"
    source = b'{"value": 1}\n'
    output = b'{"rendered": 1}\n'
    paths = {
        "generator": tmp_path / "scripts" / "gen-site-data.mjs",
        "source": tmp_path / "docs" / "source.json",
        "output": tmp_path / "site" / "src" / "generated" / "site-data.json",
        "manifest": tmp_path / "site" / "src" / "generated" / "manifest.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["generator"].write_bytes(generator)
    paths["source"].write_bytes(source)
    paths["output"].write_bytes(output)

    registry = _provenance_registry(generator, source, output)
    registry["entries"][0]["sources"] = ["../source.json"]
    paths["manifest"].write_text(json.dumps(registry) + "\n", encoding="utf-8")
    unsafe = _run_checker(tmp_path)
    assert unsafe.returncode == 1
    assert "repository-relative path" in unsafe.stderr

    registry = _provenance_registry(generator, source, output)
    registry["entries"][0]["sources"] = ["docs/unregistered.json"]
    paths["manifest"].write_text(json.dumps(registry) + "\n", encoding="utf-8")
    unverified = _run_checker(tmp_path)
    assert unverified.returncode == 1
    assert "not digest-verified" in unverified.stderr

    registry = _provenance_registry(generator, source, output)
    registry["entries"].append(dict(registry["entries"][0]))
    paths["manifest"].write_text(json.dumps(registry) + "\n", encoding="utf-8")
    duplicate = _run_checker(tmp_path)
    assert duplicate.returncode == 1
    assert "duplicated" in duplicate.stderr

    registry = _provenance_registry(generator, source, output)
    registry["entries"][0]["key"] = "missing"
    paths["manifest"].write_text(json.dumps(registry) + "\n", encoding="utf-8")
    missing = _run_checker(tmp_path)
    assert missing.returncode == 1
    assert "does not resolve" in missing.stderr


def test_git_free_provenance_mode_is_explicit_and_cannot_bypass_checkout(
    tmp_path,
):
    generator = b"// archived generator\n"
    source = b'{"archived": true}\n'
    output = b'{"rendered": true}\n'
    paths = {
        "generator": tmp_path / "scripts" / "gen-site-data.mjs",
        "source": tmp_path / "docs" / "source.json",
        "output": tmp_path / "site" / "src" / "generated" / "site-data.json",
        "manifest": tmp_path / "site" / "src" / "generated" / "manifest.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["generator"].write_bytes(generator)
    paths["source"].write_bytes(source)
    paths["output"].write_bytes(output)
    paths["manifest"].write_text(
        json.dumps(_provenance_registry(generator, source, output)) + "\n",
        encoding="utf-8",
    )

    implicit_checkout = _run_checker(tmp_path, mode="checkout")
    assert implicit_checkout.returncode == 1
    assert "Git metadata is absent; a verified source archive must use --archive" in (
        implicit_checkout.stderr
    )

    archived = _run_checker(tmp_path, mode="archive")
    assert archived.returncode == 0, archived.stderr

    paths["source"].write_bytes(b'{"archived": false}\n')
    drifted = _run_checker(tmp_path, mode="archive")
    assert drifted.returncode == 1
    assert "digest mismatch" in drifted.stderr

    paths["source"].write_bytes(source)
    (tmp_path / ".git").mkdir()
    checkout_bypass = _run_checker(tmp_path, mode="archive")
    assert checkout_bypass.returncode == 1
    assert "cannot bypass checkout trackedness" in checkout_bypass.stderr


def test_real_site_registry_and_pages_workflow_are_reproducible():
    command = ["node", str(ROOT / "scripts" / "check-provenance.mjs")]
    if not (ROOT / ".git").exists():
        command.append("--archive")
    checked = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    workflow = (ROOT / ".github" / "workflows" / "deploy-pages.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_run:" in workflow
    assert 'workflows: ["CI (release integrity)"]' in workflow
    assert "types: [completed]" in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event.workflow_run.event == 'push'" in workflow
    assert "github.event.workflow_run.head_branch == 'main'" in workflow
    assert "github.event.workflow_run.head_sha == github.sha" in workflow
    assert "ref: ${{ github.event.workflow_run.head_sha }}" in workflow
    assert "\n  push:" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert "npm ci --no-audit --no-fund" in workflow
    assert "npm run gen" in workflow
    assert "node ../scripts/check-provenance.mjs" in workflow
    assert "git -C .. diff --exit-code -- site/src/generated" in workflow
    assert "npm install" not in workflow


def _all_registered_nodeids() -> tuple[set[str], list[str]]:
    contracts = _strict_json(GOVERNANCE / "contracts.json")
    requested = {
        nodeid
        for contract in contracts["contracts"]
        for nodeid in contract["tests"]
    }
    requested.update(
        nodeid
        for runner in contracts["runners"]["entries"]
        for nodeid in runner["required_nodeids"]
    )
    return requested, sorted({nodeid.split("::", 1)[0] for nodeid in requested})


def _missing_nodeids(collected: set[str], requested: set[str]) -> set[str]:
    return requested - collected


def test_contract_bindings_and_nodeids_are_executable():
    contracts = _strict_json(GOVERNANCE / "contracts.json")
    assert set(contracts) == {
        "schema", "authority", "collection_command", "contracts", "runners",
        "entry_points", "output_owners"
    }
    assert contracts["schema"] == "stylo.governance.contracts.v1"
    ids = [item["id"] for item in contracts["contracts"]]
    assert len(ids) == len(set(ids))
    for item in contracts["contracts"]:
        assert set(item) == {"id", "description", "code", "tests"}
        assert item["code"] and item["tests"]
        for binding in item["code"]:
            _assert_binding(binding, require_hash=False)

    requested, files = _all_registered_nodeids()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *files,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    collected = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }
    assert not _missing_nodeids(collected, requested)


def test_missing_nodeid_comparison_fails_closed():
    assert _missing_nodeids({"tests/x.py::test_a"}, {
        "tests/x.py::test_a", "tests/x.py::test_removed"
    }) == {"tests/x.py::test_removed"}


def test_registered_runners_and_entry_points_match_the_tree():
    contracts = _strict_json(GOVERNANCE / "contracts.json")
    runners = contracts["runners"]
    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / runners["directory"]).glob("*.py")
    }
    assert {runner["path"] for runner in runners["entries"]} == discovered
    for runner in runners["entries"]:
        assert set(runner) == {
            "path", "status", "claim_scope", "output_contract", "required_nodeids",
        }
        assert runner["required_nodeids"]

    paths = [entry["path"] for entry in contracts["entry_points"]]
    assert len(paths) == len(set(paths))
    assert all((ROOT / path).is_file() for path in paths)

    namespaces = [item["namespace"] for item in contracts["output_owners"]]
    assert len(namespaces) == len(set(namespaces))
    for item in contracts["output_owners"]:
        if item["owner"] is None:
            assert item["contract"] in {
                "retired_output_disabled",
                "frozen_historical_output_no_live_writer",
            }
        else:
            assert item["owner"] in paths, item["namespace"]


def test_active_taras_masking_consumer_imports_without_running_retired_cleaner():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.mask_taras_case_texts",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr


