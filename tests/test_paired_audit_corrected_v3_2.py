"""Adversarial identity/content contracts for corrected paired-audit v3.2."""
from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import shutil
import stat

import pytest

from stylo.jsonio import dump_strict, load_strict
from stylo.eval.paired_audit import corrected_v3_2 as v3
from stylo.eval.paired_audit import corpus as historical_corpus


def _record(work_id: str, text: str, *, author: str | None = None) -> dict:
    author_id, slug = work_id.split("/", 1)
    author_id = author if author is not None else author_id
    byte = __import__("hashlib").sha256(text.encode()).hexdigest()
    normalized = __import__("hashlib").sha256(" ".join(text.casefold().split()).encode()).hexdigest()
    return {
        "work_id": work_id, "author_id": author_id, "work_slug": slug,
        "manifest_sha256": "1" * 64, "source_sha256": byte, "source_normalized_sha256": normalized,
        "work_content_identity": "2" * 64, "content_component_identity": "3" * 64,
        "chunks": [{"path": "c.txt", "span_ordinal": 0, "byte_sha256": byte,
                    "text_sha256": byte, "normalized_sha256": normalized, "text": text}],
    }


def test_adjudicated_basename_pairs_round_trip_by_full_id_only():
    records = [_record("radov/rasskazi", "radov unique words"), _record("zoshenko/rasskazi", "zoshenko unique words"),
               _record("bunin/деревня", "bunin unique words"), _record("grigorovich/деревня", "grigorovich unique words")]
    catalog = {row["work_id"]: row for row in records}
    assert v3.resolve_full_work(catalog, "radov/rasskazi")["author_id"] == "radov"
    assert v3.resolve_full_work(catalog, "zoshenko/rasskazi")["author_id"] == "zoshenko"
    assert v3.basename_collision_inventory(records, expected=v3.EXPECTED_LOBO_BASENAME_COLLISIONS)["components"] == [
        ["bunin/деревня", "grigorovich/деревня"], ["radov/rasskazi", "zoshenko/rasskazi"]
    ]
    for bare in ("rasskazi", "деревня"):
        with pytest.raises(v3.CorrectedCorpusError, match="never a basename"):
            v3.resolve_full_work(catalog, bare)


def test_duplicate_full_id_and_author_prefix_mismatch_are_fatal():
    record = _record("radov/rasskazi", "one distinct text")
    with pytest.raises(v3.CorrectedCorpusError, match="duplicate"):
        v3.work_identity_catalog([record, copy.deepcopy(record)])
    forged = copy.deepcopy(record)
    forged["author_id"] = "zoshenko"
    with pytest.raises(v3.CorrectedCorpusError, match="author-prefix"):
        v3.work_identity_catalog([forged])


def test_collision_inventory_rejects_missing_unexpected_or_third_member():
    records = [_record("radov/rasskazi", "one"), _record("zoshenko/rasskazi", "two"), _record("other/rasskazi", "three")]
    with pytest.raises(v3.CorrectedCorpusError, match="unexpected/missing/third"):
        v3.basename_collision_inventory(records, expected=(("radov/rasskazi", "zoshenko/rasskazi"),))


def test_content_overlap_inside_adjudicated_pair_remains_fatal():
    records = [_record("radov/rasskazi", "the exact component is shared across works"),
               _record("zoshenko/rasskazi", "the exact component is shared across works")]
    with pytest.raises(v3.CorrectedCorpusError, match="content-isolation"):
        v3.content_isolation_audit(records)


def test_old_fold_schema_is_rejected_before_any_universe_interpretation():
    with pytest.raises(v3.CorrectedCorpusError, match="identity-first"):
        v3.assert_v3_2_fold_manifest({"schema": "lobo_fold_manifest_v1", "works": []}, kind="lobo")


def test_tampered_child_digest_is_fatal_and_never_creates_current_pointer(tmp_path):
    root = tmp_path / "staging"
    (root / "frags").mkdir(parents=True)
    (root / "input_clean").mkdir()
    inventory, digest = v3._inventory(root)
    root = root.rename(tmp_path / digest)
    body = {
        "schema": v3.CORPUS_SCHEMA,
        "protocol_version": v3.PROTOCOL_VERSION,
        "corrected_content_inventory": inventory,
        "corrected_content_inventory_digest": digest,
    }
    body["self_hash"] = v3._self_hash(body)
    from stylo.jsonio import dump_strict

    manifest_path = root / "corrected_corpus_manifest_v3_2.json"
    dump_strict(body, manifest_path, trailing_newline=True)
    body["corrected_content_inventory_digest"] = "0" * 64
    dump_strict(body, manifest_path, trailing_newline=True)
    with pytest.raises(v3.CorrectedCorpusError, match="tamper"):
        v3._verify_child(root, body)
    assert not (tmp_path / "current.json").exists()


def test_full_work_id_rejects_non_nfc_and_bare_values():
    with pytest.raises(v3.CorrectedCorpusError):
        v3.full_work_id("деревня")
    with pytest.raises(v3.CorrectedCorpusError):
        v3.full_work_id("bunin/cafe\u0301")


def _chmod_tree(root: pathlib.Path) -> None:
    for current, dirs, files in os.walk(root):
        os.chmod(current, 0o755)
        for name in dirs:
            os.chmod(pathlib.Path(current) / name, 0o755)
        for name in files:
            os.chmod(pathlib.Path(current) / name, 0o644)


def _synthetic_parent(tmp_path: pathlib.Path, monkeypatch) -> tuple[pathlib.Path, list[str]]:
    works = {
        "a": ("a1", "a2", "a3", "a_ex"),
        "b": ("b1", "b2", "b_ex"),
        "c": ("c1", "c_ex"),
    }
    building = tmp_path / "historical-building"
    for author, slugs in works.items():
        for slug in slugs:
            work = building / "frags" / author / slug
            clean = building / "input_clean" / author
            work.mkdir(parents=True, exist_ok=True)
            clean.mkdir(parents=True, exist_ok=True)
            text = f"unique corpus text for {author} {slug} with isolated lexical material"
            chunk = work / "0000.txt"
            source = clean / f"{slug}.txt"
            chunk.write_text(text, encoding="utf-8")
            source.write_text(text, encoding="utf-8")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            dump_strict({
                "work_id": f"{author}/{slug}", "author_id": author,
                "provenance_sha256": digest, "chunker_config_hash": "c" * 64,
                "overlap": 0.0,
                "chunks": [{"span_ordinal": 0, "text_sha256": digest, "path": "0000.txt"}],
            }, work / "manifest.json", trailing_newline=True)
    _chmod_tree(building)
    digest = historical_corpus._tree_content_digest(building, ("frags", "input_clean"))
    parent = building.rename(tmp_path / digest)
    manifest = {
        "schema": v3.HISTORICAL_PARENT_SCHEMA, "audit_corpus_digest": digest,
        "n_works": 9, "synthetic_test_fixture": True,
    }
    manifest["self_hash"] = historical_corpus._self_hash(manifest)
    dump_strict(manifest, parent / "corpus_manifest.json", trailing_newline=True)
    os.chmod(parent / "corpus_manifest.json", 0o644)

    exclusions = (("a/a_ex", "x"), ("b/b_ex", "y"), ("c/c_ex", "z"))
    monkeypatch.setattr(v3, "HISTORICAL_PARENT_DIGEST", digest)
    monkeypatch.setattr(v3, "HISTORICAL_WORK_COUNT", 9)
    monkeypatch.setattr(v3, "EXCLUSIONS", exclusions)
    monkeypatch.setattr(v3, "EXCLUDED_WORK_IDS", tuple(row[0] for row in exclusions))
    monkeypatch.setattr(v3, "LOBO_COUNTS", (3, 6, 2, 5))
    monkeypatch.setattr(v3, "LOBO_SINGLETON_AUTHORS", ("c",))
    monkeypatch.setattr(v3, "RUAA_PARENT_COUNTS", (3, 6))
    monkeypatch.setattr(v3, "RUAA_COUNTS", (3, 3))
    monkeypatch.setattr(v3, "EXPECTED_LOBO_BASENAME_COLLISIONS", ())
    monkeypatch.setattr(v3, "PROVENANCE_LIMITATIONS", ())
    return parent, ["a/a1", "a/a_ex", "b/b1", "b/b_ex", "c/c1", "c/c_ex"]


@pytest.fixture
def atomic_fixture(tmp_path, monkeypatch):
    parent, selection = _synthetic_parent(tmp_path, monkeypatch)
    return {
        "parent": parent, "selection": selection,
        "config": "d" * 64, "protocol": "e" * 64,
        "output": tmp_path / "output",
    }


def _prepare(contract, output=None, fault=None):
    return v3.prepare_corrected_v3_2(
        historical_parent_root=contract["parent"],
        output_root=output or contract["output"],
        ruaa_parent_selection=contract["selection"],
        config_hash=contract["config"], protocol_sha256=contract["protocol"],
        fault_inject=fault,
    )


def _verify(contract, root):
    return v3.verify_v3_2_candidate(
        root, historical_parent_root=contract["parent"],
        ruaa_parent_selection=contract["selection"],
        config_hash=contract["config"], protocol_sha256=contract["protocol"],
    )


def _tree_state(root: pathlib.Path) -> list[tuple]:
    rows = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        kind = ("symlink" if stat.S_ISLNK(info.st_mode) else "dir" if stat.S_ISDIR(info.st_mode)
                else "file" if stat.S_ISREG(info.st_mode) else "special")
        payload = (os.readlink(path) if kind == "symlink"
                   else hashlib.sha256(path.read_bytes()).hexdigest() if kind == "file" else None)
        rows.append((path.relative_to(root).as_posix(), kind, stat.S_IMODE(info.st_mode), info.st_nlink, payload))
    return rows


def _reseal_forgery(root: pathlib.Path) -> pathlib.Path:
    inventory = {
        "schema": "paired_audit.preparation_bundle_inventory.v1",
        "storage_contract_version": v3.BUNDLE_STORAGE_CONTRACT_VERSION,
        "root_mode": "0755",
        "entries": v3._recursive_inventory(
            root, excluded=(v3.BUNDLE_INVENTORY_NAME, "candidate.json", v3.SHA256SUMS_NAME),
        ),
    }
    inventory["self_hash"] = v3._self_hash(inventory)
    v3._write_json(inventory, root / v3.BUNDLE_INVENTORY_NAME)
    candidate = load_strict(root / "candidate.json")
    lobo = load_strict(root / "lobo_fold_manifest_v3_2.json")
    ruaa = load_strict(root / "ruaa_fold_manifest_v3_2.json")
    candidate["folds"] = {"lobo_self_hash": lobo["self_hash"], "ruaa_self_hash": ruaa["self_hash"]}
    candidate["exact_inventory_self_hash"] = inventory["self_hash"]
    candidate["files"] = v3._file_map(root)
    candidate.pop("self_hash", None)
    candidate["self_hash"] = v3._self_hash(candidate)
    v3._write_json(candidate, root / "candidate.json")
    (root / v3.SHA256SUMS_NAME).write_text(v3._canonical_sums(root), encoding="utf-8")
    os.chmod(root / v3.SHA256SUMS_NAME, 0o644)
    destination = root.with_name(candidate["self_hash"])
    return root.rename(destination)


def test_content_equivalent_hardlink_is_accepted(atomic_fixture, tmp_path):
    source = atomic_fixture["parent"] / "frags/a/a1/0000.txt"
    external = tmp_path / "external-hardlink.txt"
    os.link(source, external)
    identity, records = v3.verify_historical_parent(atomic_fixture["parent"])
    assert identity["historical_parent_digest"] == v3.HISTORICAL_PARENT_DIGEST
    assert len(records) == v3.HISTORICAL_WORK_COUNT
    assert source.read_bytes() == external.read_bytes() and source.stat().st_nlink == 2


def test_child_manifest_external_symlink_rejected_before_parse(atomic_fixture, tmp_path):
    bundle = _prepare(atomic_fixture)["bundle_root"]
    manifest = bundle / v3.CORRECTED_CORPUS_DIR / v3.CORRECTED_MANIFEST_NAME
    external = tmp_path / "byte-identical-manifest.json"
    external.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(external)
    before = external.read_bytes()
    with pytest.raises(v3.CorrectedCorpusError, match="symlink"):
        _verify(atomic_fixture, bundle)
    assert external.read_bytes() == before and manifest.is_symlink()


def test_output_symlink_is_rejected_without_target_mutation(atomic_fixture, tmp_path):
    external = tmp_path / "external-output"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_bytes(b"untouched")
    atomic_fixture["output"].symlink_to(external, target_is_directory=True)
    with pytest.raises(v3.CorrectedCorpusError, match="symlink path component"):
        _prepare(atomic_fixture)
    assert marker.read_bytes() == b"untouched"


def test_mode_drift_is_rejected_without_verifier_mutation(atomic_fixture):
    bundle = _prepare(atomic_fixture)["bundle_root"]
    payload = bundle / "lobo_fold_manifest_v3_2.json"
    os.chmod(payload, 0o600)
    with pytest.raises(v3.CorrectedCorpusError, match="mode drift"):
        _verify(atomic_fixture, bundle)
    assert stat.S_IMODE(payload.lstat().st_mode) == 0o600


@pytest.mark.parametrize("rogue_kind", ["file", "directory", "symlink", "hardlink", "fifo"])
def test_exact_resume_rejects_rogue_member_without_repair(atomic_fixture, tmp_path, rogue_kind):
    original = _prepare(atomic_fixture)["bundle_root"]
    clone_parent = tmp_path / f"clone-{rogue_kind}"
    clone_parent.mkdir()
    bundle = shutil.copytree(original, clone_parent / original.name)
    rogue = bundle / "rogue-extra.txt"
    if rogue_kind == "file":
        rogue.write_text("rogue", encoding="utf-8")
    elif rogue_kind == "directory":
        rogue.mkdir()
    elif rogue_kind == "symlink":
        rogue.symlink_to(tmp_path / "external")
    elif rogue_kind == "hardlink":
        os.link(bundle / "candidate.json", rogue)
    else:
        os.mkfifo(rogue)
    state = _tree_state(bundle)
    with pytest.raises(v3.CorrectedCorpusError):
        _verify(atomic_fixture, bundle)
    assert _tree_state(bundle) == state


def _mutate_fold(fold: dict, mutation: str) -> None:
    if mutation in {"corrected_corpus_digest", "full_work_identity_catalog_digest",
                    "content_isolation_audit_digest", "selection_digest", "config_hash",
                    "applicability_matrix_digest"}:
        fold[mutation] = "f" * 64
    elif mutation == "algorithm":
        fold[mutation] = "forged_algorithm"
    elif mutation == "seed":
        fold[mutation] += 1
    elif mutation == "dataset_kind":
        fold[mutation] = "forged_kind"
    elif mutation == "work_content_identity":
        fold["works"][0][mutation] = "f" * 64
    elif mutation == "content_component_identity":
        fold["works"][0][mutation] = "f" * 64
    elif mutation == "author_assignment":
        fold["works"][0]["author_id"] = next(
            row["author_id"] for row in fold["works"]
            if row["author_id"] != fold["works"][0]["author_id"]
        )
    elif mutation == "tested_fold_assignment":
        tested = [row for row in fold["works"] if row["tested"]]
        tested[0]["fold_index"], tested[1]["fold_index"] = tested[1]["fold_index"], tested[0]["fold_index"]
    elif mutation == "label_order":
        fold["probability_class_order"] = list(reversed(fold["probability_class_order"]))
    fold.pop("self_hash", None)
    fold["self_hash"] = v3._self_hash(fold)


@pytest.mark.parametrize("mutation", [
    "corrected_corpus_digest", "full_work_identity_catalog_digest",
    "content_isolation_audit_digest", "selection_digest", "config_hash",
    "applicability_matrix_digest", "algorithm", "seed", "dataset_kind",
    "work_content_identity", "content_component_identity", "author_assignment",
    "tested_fold_assignment", "label_order",
])
def test_self_rehashed_coherent_fold_forgery_rejected_by_reconstruction(atomic_fixture, tmp_path, mutation):
    original = _prepare(atomic_fixture)["bundle_root"]
    clone_parent = tmp_path / f"forgery-{mutation}"
    clone_parent.mkdir()
    bundle = shutil.copytree(original, clone_parent / original.name)
    path = bundle / "lobo_fold_manifest_v3_2.json"
    fold = load_strict(path)
    _mutate_fold(fold, mutation)
    v3._write_json(fold, path)
    bundle = _reseal_forgery(bundle)
    with pytest.raises(v3.CorrectedCorpusError):
        _verify(atomic_fixture, bundle)


@pytest.mark.parametrize("point", [
    "during_child_assembly", "building_lobo_fold", "building_ruaa_fold",
    "verifying_ruaa_fold", "writing_audits", "writing_candidate",
    "writing_sha256sums", "before_final_rename", "final_rename",
])
def test_fault_injection_never_publishes_partial_bundle(atomic_fixture, tmp_path, point):
    output = tmp_path / f"fault-{point}"
    parent_state = _tree_state(atomic_fixture["parent"])

    def fail(at):
        if at == point:
            raise OSError(f"injected {point}")

    with pytest.raises(OSError, match="injected"):
        _prepare(atomic_fixture, output=output, fault=fail)
    bundle_parent = output / v3.BUNDLE_PARENT_NAME
    if bundle_parent.exists():
        assert not [entry for entry in bundle_parent.iterdir() if not entry.name.startswith(".")]
        assert not [entry for entry in bundle_parent.iterdir() if entry.name.startswith(".staging")]
    assert not list(output.rglob("candidate.json"))
    assert not list(output.rglob("current.json"))
    assert _tree_state(atomic_fixture["parent"]) == parent_state


def test_actual_final_rename_failure_and_success_resume_semantics(atomic_fixture, monkeypatch):
    output = atomic_fixture["output"]
    original_rename = v3._rename_noreplace
    monkeypatch.setattr(v3, "_rename_noreplace", lambda *_: (_ for _ in ()).throw(OSError("rename failed")))
    with pytest.raises(OSError, match="rename failed"):
        _prepare(atomic_fixture)
    parent = output / v3.BUNDLE_PARENT_NAME
    assert not [entry for entry in parent.iterdir() if not entry.name.startswith(".")]
    monkeypatch.setattr(v3, "_rename_noreplace", original_rename)
    first = _prepare(atomic_fixture)
    state = _tree_state(first["bundle_root"])
    second = _prepare(atomic_fixture)
    assert second["reused"] is True and second["bundle_root"] == first["bundle_root"]
    assert _tree_state(first["bundle_root"]) == state


def test_storage_contract_rejects_old_split_layout(atomic_fixture, tmp_path):
    old = tmp_path / ("a" * 64)
    old.mkdir()
    (old / "candidate.json").write_text('{"schema":"paired_audit.local_candidate_preparation.v3_2"}\n')
    os.chmod(old / "candidate.json", 0o644)
    with pytest.raises(v3.CorrectedCorpusError):
        _verify(atomic_fixture, old)


@pytest.mark.parametrize("binding", [
    "absolute", "traversal", "dot", "empty", "alternate", "disagreement", "wrong_type",
])
def test_invalid_self_rehashed_child_binding_rejected_with_zero_child_use(
    atomic_fixture, tmp_path, monkeypatch, binding,
):
    original = _prepare(atomic_fixture)["bundle_root"]
    clone_parent = tmp_path / f"binding-{binding}"
    clone_parent.mkdir()
    bundle = shutil.copytree(original, clone_parent / original.name)
    candidate = load_strict(bundle / "candidate.json")
    if binding == "absolute":
        left = right = "/tmp/corrected_corpus"
    elif binding == "traversal":
        left = right = "../corrected_corpus"
    elif binding == "dot":
        left = right = "."
    elif binding == "empty":
        left = right = ""
    elif binding == "alternate":
        left = right = "alternate_child"
    elif binding == "disagreement":
        left, right = v3.CORRECTED_CORPUS_DIR, "alternate_child"
    else:
        left = right = [v3.CORRECTED_CORPUS_DIR]
    candidate["bundle_layout"]["corrected_corpus_relative_root"] = left
    candidate["corrected_corpus"]["relative_root"] = right
    candidate.pop("self_hash")
    candidate["self_hash"] = v3._self_hash(candidate)
    v3._write_json(candidate, bundle / "candidate.json")
    bundle = bundle.rename(bundle.with_name(candidate["self_hash"]))

    calls = 0

    def forbidden_snapshot(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("child snapshot/descent occurred before binding rejection")

    monkeypatch.setattr(v3, "_snapshot_tree_fd", forbidden_snapshot)
    with pytest.raises(v3.CorrectedCorpusError, match="exact literal corrected_corpus"):
        _verify(atomic_fixture, bundle)
    assert calls == 0


def test_ambiguous_child_binding_encoding_rejected_before_child_use(
    atomic_fixture, tmp_path, monkeypatch,
):
    original = _prepare(atomic_fixture)["bundle_root"]
    clone_parent = tmp_path / "binding-ambiguous-encoding"
    clone_parent.mkdir()
    bundle = shutil.copytree(original, clone_parent / original.name)
    candidate_path = bundle / "candidate.json"
    canonical = candidate_path.read_bytes()
    ambiguous = canonical.replace(
        b'": "corrected_corpus"', b'": "corrected\\u005fcorpus"',
    )
    assert ambiguous != canonical
    candidate_path.write_bytes(ambiguous)
    os.chmod(candidate_path, 0o644)
    calls = 0

    def forbidden_snapshot(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("child snapshot/descent occurred before canonical envelope rejection")

    monkeypatch.setattr(v3, "_snapshot_tree_fd", forbidden_snapshot)
    with pytest.raises(v3.CorrectedCorpusError, match="non-canonical JSON"):
        _verify(atomic_fixture, bundle)
    assert calls == 0


def test_crash_after_successful_rename_is_exactly_reused(atomic_fixture, monkeypatch):
    original = v3._rename_noreplace

    def publish_then_crash(source, destination):
        original(source, destination)
        raise SystemExit("simulated process crash after successful publication")

    monkeypatch.setattr(v3, "_rename_noreplace", publish_then_crash)
    with pytest.raises(SystemExit, match="after successful publication"):
        _prepare(atomic_fixture)
    monkeypatch.setattr(v3, "_rename_noreplace", original)
    resumed = _prepare(atomic_fixture)
    assert resumed["reused"] is True
    assert resumed["bundle_root"].name == resumed["candidate"]["self_hash"]
    parent = atomic_fixture["output"] / v3.BUNDLE_PARENT_NAME
    assert len([entry for entry in parent.iterdir() if not entry.name.startswith(".")]) == 1
    assert not [entry for entry in parent.iterdir() if entry.name.startswith(".staging")]
