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


def test_normative_status_ledger_is_symbol_and_byte_bound():
    ledger = _strict_json(GOVERNANCE / "status_ledger.json")
    assert set(ledger) == {
        "schema", "as_of", "authority", "paired_audit",
        "bounded_exploratory_milestones", "historical_records"
    }
    assert ledger["schema"] == "stylo.governance.status_ledger.v2"
    assert ledger["as_of"] == "2026-08-23"
    expected_states = {
        "protocol_v3_1": "superseded_ineligible_corpus",
        "protocol_v3_2": "owner_accepted_for_evaluator_implementation",
        "independent_security_audit": "not_claimed_review_terminated_by_owner",
        "evaluator_candidate": "implemented_single_review_blocker_corrected_verified_frozen_inputs_reconciled",
        "topic_validity_challenger": "implemented_review_blocker_corrected_unexecuted",
        "manifest_freeze": "unapproved",
        "production_evaluator": "unregistered",
        "preflight": "absent",
        "execution_authorization": "absent",
        "confirmatory_execution": "hard_disabled",
        "headline": "not_authorized",
    }
    assert {
        key: value["status"] for key, value in ledger["paired_audit"].items()
    } == expected_states
    for state in ledger["paired_audit"].values():
        assert set(state) in ({"status", "claim"}, {"status", "claim", "bindings"})
        for binding in state.get("bindings", []):
            _assert_binding(binding, require_hash=True)
    for record in ledger["historical_records"]:
        assert set(record) == {"path", "status", "sha256"}
        assert _sha256(ROOT / record["path"]) == record["sha256"]


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


@pytest.mark.parametrize(
    "relative",
    [
        "research/ROADMAP.md",
        "research/work_balanced/README.md",
        "research/work_balanced/estimand.md",
        "research/work_balanced/paired_audit_protocol.md",
        "research/work_balanced/paired_audit_review_provenance.md",
    ],
)
def test_current_research_documents_defer_to_the_ledger(relative):
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert "governance/status_ledger.json" in text


def test_stale_paired_audit_status_claims_are_not_current_prose():
    roadmap = (ROOT / "research" / "ROADMAP.md").read_text(encoding="utf-8")
    contract = (
        ROOT / "research" / "work_balanced" / "estimand.md"
    ).read_text(encoding="utf-8")
    protocol = (
        ROOT / "research" / "work_balanced" / "paired_audit_protocol.md"
    ).read_text(encoding="utf-8")
    assert "Still required beyond the narrow stylo validation" not in roadmap
    assert "paired audit has not yet been implemented" not in contract
    assert "No confirmatory audit-corpus builder, paired-audit runner" not in protocol


def test_paired_audit_v3_2_contract_is_mechanically_accounted_and_non_authorizing():
    """The v3.2 preparation contract is local/unapproved and leaves v3.1 evidence immutable."""
    ledger = _strict_json(GOVERNANCE / "status_ledger.json")
    protocol = (
        ROOT / "research" / "work_balanced" / "paired_audit_protocol.md"
    ).read_text(encoding="utf-8")
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
    assert ledger["paired_audit"]["protocol_v3_2"]["status"] == (
        "owner_accepted_for_evaluator_implementation"
    )
    claim = ledger["paired_audit"]["protocol_v3_2"]["claim"]
    assert "full NFC author_id/work_slug" in claim
    assert "immutable design-freeze protocol bytes" in claim
    assert "recorded manifest identity" in claim
    assert "not a freeze, execution grant, or independent security verdict" in claim
    assert ledger["paired_audit"]["independent_security_audit"]["status"] == (
        "not_claimed_review_terminated_by_owner"
    )
    evaluator = ledger["paired_audit"]["evaluator_candidate"]
    assert evaluator["status"] == (
        "implemented_single_review_blocker_corrected_verified_frozen_inputs_reconciled"
    )
    assert "one AC-07 incomplete-class-universe blocker" in evaluator["claim"]
    assert "No second review of the original evaluator acceptance is claimed" in evaluator["claim"]
    assert "bounded review of commit a4908d08 passed" in evaluator["claim"]
    assert "MFW can encode label-correlated content nouns" in evaluator["claim"]
    assert "not factology-ready for registration" in evaluator["claim"]
    assert [binding["symbol"] for binding in evaluator["bindings"]] == [
        "REGISTRY_V3_2", "evaluate_fold_v3_2", "validate_receipt_v3_2",
    ]
    challenger = ledger["paired_audit"]["topic_validity_challenger"]
    assert "no CLI, runner, registry entry" in challenger["claim"]
    assert "separately approved R3b task" in challenger["claim"]
    assert "without exposing paths" in challenger["claim"]
    assert "no second-review PASS is claimed" in challenger["claim"]
    for marker in (
        "(v3.2)",
        "47 authors / 252 works",
        "43 authors / 248 works",
        "22 authors / 134 works",
        "16 applied cells",
        "11-member set",
        "`stylo_stack` is withdrawn",
        "historical evidence only",
        "No corpus build, fit,",
    ):
        assert marker in protocol
    assert "pending implementation" not in protocol


def test_work_balanced_estimand_is_the_compact_implementation_contract():
    contract = (
        ROOT / "research" / "work_balanced" / "estimand.md"
    ).read_text(encoding="utf-8")
    assert 100 <= len(contract.splitlines()) <= 140
    assert len(contract.encode("utf-8")) <= 12 * 1024
    for marker in (
        "WorkLevelVectorizer",
        "fit_estimator",
        "needs_groups",
        "## Dataset identity, provenance, and atomic subsets",
        "## Single fit dispatch",
        "## Output and artifact isolation",
    ):
        assert marker in contract
    prose = " ".join(contract.casefold().replace("’", "'").split())
    for marker in (
        "selected-mass delta",
        "not canonical burrows's delta",
        "group-aware calibration",
        "artifact isolation",
    ):
        assert marker in prose


def test_normative_protocol_matches_the_single_canonical_environment_lock():
    protocol = (
        ROOT / "research" / "work_balanced" / "paired_audit_protocol.md"
    ).read_text(encoding="utf-8")
    assert "SHA-256 of the tracked `requirements.lock` only" in protocol
    assert "ignored local `uv.lock` is explicitly outside the run identity" in protocol
    assert "`requirements.lock`/`uv.lock` fingerprint" not in protocol


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
    requirements = _strict_json(GOVERNANCE / "requirements.json")
    runners = _strict_json(GOVERNANCE / "runner_catalog.json")
    requested = {
        nodeid
        for requirement in requirements["requirements"]
        for nodeid in requirement["tests"]
    }
    requested.update(
        nodeid
        for runner in runners["runners"]
        for nodeid in runner["required_nodeids"]
    )
    return requested, sorted({nodeid.split("::", 1)[0] for nodeid in requested})


def _missing_nodeids(collected: set[str], requested: set[str]) -> set[str]:
    return requested - collected


def test_requirement_bindings_and_nodeids_are_executable():
    requirements = _strict_json(GOVERNANCE / "requirements.json")
    assert set(requirements) == {
        "schema", "source", "collection_command", "requirements"
    }
    assert requirements["schema"] == "stylo.governance.requirements.v1"
    ids = [item["id"] for item in requirements["requirements"]]
    assert len(ids) == len(set(ids))
    assert "PA-V3-2-APPLICABILITY-EXACTNESS" in ids
    assert "PA-V3-1-HISTORICAL-HOLM-FAMILY-EXACTNESS" in ids
    assert "PA-HOLM-FAMILY-EXACTNESS" not in ids
    for item in requirements["requirements"]:
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


def test_runner_catalog_covers_the_evaluation_directory_exactly():
    catalog = _strict_json(GOVERNANCE / "runner_catalog.json")
    assert set(catalog) == {"schema", "directory", "runners"}
    assert catalog["schema"] == "stylo.governance.runner_catalog.v1"
    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / catalog["directory"]).glob("*.py")
    }
    registered = {runner["path"] for runner in catalog["runners"]}
    assert registered == discovered
    corrected = next(
        runner for runner in catalog["runners"]
        if runner["path"] == "scripts/evaluation/prepare_corrected_paired_audit_v3_2.py"
    )
    assert corrected["status"] == "preparation_only_non_authorizing"
    assert "owned only by the governance ledger" in corrected["claim_scope"]
    for runner in catalog["runners"]:
        assert set(runner) == {
            "path", "status", "claim_scope", "output_contract",
            "identity_bindings", "required_nodeids",
        }
        assert runner["identity_bindings"]
        assert runner["required_nodeids"]


def test_topology_has_one_owner_per_output_and_distinct_eval_roles():
    topology = _strict_json(GOVERNANCE / "topology.json")
    assert set(topology) == {
        "schema", "historical_evidence", "discovery_contract",
        "directories", "entries", "output_owners"
    }
    assert topology["schema"] == "stylo.governance.topology.v1"
    paths = [entry["path"] for entry in topology["entries"]]
    assert len(paths) == len(set(paths))
    assert all((ROOT / path).is_file() for path in paths)
    discovery = topology["discovery_contract"]
    assert set(discovery) == {
        "canonical_paths", "legacy_script_basenames", "contract"
    }
    discovered_legacy = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").rglob("*.py")
        if path.name in set(discovery["legacy_script_basenames"])
    }
    assert set(paths) == set(discovery["canonical_paths"]) | discovered_legacy
    statuses = {entry["path"]: entry["status"] for entry in topology["entries"]}
    assert statuses["scripts/report.py"] == "compatibility_entrypoint"
    assert discovered_legacy == {"scripts/report.py"}
    assert not any(status.startswith("retired_hard_disabled") for status in statuses.values())
    namespaces = [item["namespace"] for item in topology["output_owners"]]
    assert len(namespaces) == len(set(namespaces))
    for item in topology["output_owners"]:
        if item["owner"] is None:
            assert item["contract"] in {
                "retired_output_disabled",
                "frozen_historical_output_no_live_writer",
            }
        else:
            assert item["owner"] in paths
    owners = {
        item["namespace"]: (item["owner"], item["contract"])
        for item in topology["output_owners"]
    }
    assert owners["<paths.input_clean>"] == (
        "src/stylo/pipeline/clean.py",
        "atomic_complete_clean_snapshot",
    )
    assert owners[
        "<paths.data>/deployment/chunk_weighted_legacy/versions/<bundle-token>"
    ][0] == "src/stylo/pipeline/train.py"
    assert owners[
        "<paths.docs>/{prediction.txt,prediction.evidence.json}"
    ][0] == "src/stylo/report/evidence.py"
    assert owners[
        "<paths.docs>/{corpus_validation.txt,corpus_validation.json,corpus_validation.evidence.json}"
    ][0] == "src/stylo/report/evidence.py"
    assert owners["docs/final_comparison.csv"] == (
        None,
        "frozen_historical_output_no_live_writer",
    )
    assert owners["docs/{validation.json,validation_pd.json}"] == (
        None,
        "frozen_historical_output_no_live_writer",
    )
    assert owners[
        "<paths.docs>/exploratory/channel_benchmark/{full,pd_only}/{all_channels,fast}/**"
    ] == (
        "scripts/run_benchmark.py",
        "atomic_content_addressed_exploratory_candidate",
    )
    assert owners[
        "<paths.docs>/exploratory/{legacy_recompute,work_balanced}/**/final_comparison.{txt,csv}"
    ][0] == "src/stylo/cli.py"
    clean_source = (ROOT / "src/stylo/pipeline/clean.py").read_text(encoding="utf-8")
    train_source = (ROOT / "src/stylo/pipeline/train.py").read_text(encoding="utf-8")
    evidence_source = (ROOT / "src/stylo/report/evidence.py").read_text(encoding="utf-8")
    assert '"paths.input_clean"' in clean_source
    assert '"deployment" / CHUNK_WEIGHTED_LEGACY' in train_source
    for filename in (
        "prediction.txt",
        "corpus_validation.txt",
        "corpus_validation.json",
    ):
        assert filename in evidence_source
    assert 'f"{section}.evidence.json"' in evidence_source
    roles = {
        entry["path"]: entry["responsibility"] for entry in topology["entries"]
    }
    assert roles["src/stylo/eval/segment.py"] != roles["src/stylo/eval/segmentation.py"]
    assert (ROOT / "scripts" / "experimental").is_dir()
    assert not (ROOT / "scripts" / "experemental").exists()
    evidence = _strict_json(ROOT / topology["historical_evidence"]["path"])
    historical_hashes = evidence["artifacts"]["sha256"]
    for path in (
        "scripts/ablation.py",
        "scripts/clean_text.py",
        "scripts/clustering.py",
        "scripts/lobo_cv.py",
        "scripts/train.py",
        "scripts/predict.py",
        "scripts/experiments.py",
        "scripts/split.py",
        "scripts/statistic/anomaly_stats.py",
        "scripts/statistic/consistency.py",
        "scripts/umap_vis.py",
        "scripts/validate_books.py",
        "scripts/experemental/core_dependency.py",
    ):
        assert len(historical_hashes[path]) == 64
        assert set(historical_hashes[path]) <= HEX64


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


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("src/stylo/eval/final.py", "exploratory_model_comparison_compatibility_module"),
        ("src/stylo/eval/segment.py", "rolling_attribution_diagnostic"),
        ("src/stylo/eval/segmentation.py", "mixed_authorship_evaluation"),
    ],
)
def test_ambiguous_eval_modules_declare_distinct_topology_roles(relative, expected):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "TOPOLOGY_ROLE"
    }
    assert assignments["TOPOLOGY_ROLE"] == expected
