#!/usr/bin/env python3
"""Prepare the real immutable corpus and an explicitly UNAPPROVED manifest freeze candidate.

This command performs §11 data preparation only.  It cannot run an estimator, compute a headline,
or publish a result.  The generated LOBO/RuAA fold manifests remain ``unapproved`` until an
independent review commits an approved freeze root and the execution control plane pins its digest.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile

from stylo.config import load_config
from stylo.eval.paired_audit import corpus as audit_corpus
from stylo.eval.paired_audit import manifest as fold_manifest
from stylo.eval.paired_audit import references
from stylo.eval.paired_audit import run_plan
from stylo.eval.paired_audit.work_subset import derive_work_subset
from stylo.jsonio import dump_strict, dumps_strict, load_strict
from stylo.pipeline.bundle import _verify_real_dir_chain

SCHEMA = "paired_audit.freeze_candidate.v1"
_KNOWN_RUAA_PROTOCOL_DRIFT = {
    "name": "protocol.md",
    "expected": "4a58c00ada1bf3748ab74c80e7dd9d3089e2b100ff748f9895a8f476ef825b4a",
    "actual": "97e5f8e25ba8a9603afebe2e4669c83a8da2f445340b98ee31db2c28465b3fb4",
}


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _self_hash(body: dict) -> str:
    return hashlib.sha256(dumps_strict(body, sort_keys=True).encode("utf-8")).hexdigest()


def _verify_existing_candidate(destination: pathlib.Path, body: dict,
                               lobo: dict, ruaa: dict) -> None:
    """Require an idempotently reused candidate directory to be complete and byte-authentic."""
    expected_names = {
        "freeze_candidate.json", "lobo_fold_manifest_v1.json",
        "ruaa_fold_manifest_v1.json", "SHA256SUMS",
    }
    entries = list(destination.iterdir()) if destination.is_dir() and not destination.is_symlink() else []
    if {p.name for p in entries} != expected_names or any(p.is_symlink() or not p.is_file() for p in entries):
        raise RuntimeError(f"freeze candidate {destination} is incomplete or has unexpected entries")
    candidate_path = destination / "freeze_candidate.json"
    if load_strict(candidate_path) != body:
        raise RuntimeError(f"conflicting freeze candidate at {destination}")
    if load_strict(destination / "lobo_fold_manifest_v1.json") != lobo:
        raise RuntimeError(f"LOBO manifest drift in existing candidate {destination}")
    if load_strict(destination / "ruaa_fold_manifest_v1.json") != ruaa:
        raise RuntimeError(f"RuAA manifest drift in existing candidate {destination}")

    recorded: dict[str, str] = {}
    for line in (destination / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, sep, name = line.partition("  ")
        if sep != "  " or name in recorded or name not in expected_names - {"SHA256SUMS"}:
            raise RuntimeError(f"malformed freeze-candidate SHA256SUMS entry: {line!r}")
        recorded[name] = digest
    payload_names = expected_names - {"SHA256SUMS"}
    if set(recorded) != payload_names:
        raise RuntimeError("freeze-candidate SHA256SUMS does not cover exactly the three payloads")
    for name, digest in recorded.items():
        if _sha256_file(destination / name) != digest:
            raise RuntimeError(f"freeze-candidate SHA256 mismatch for {name}")


def _git_head(repo: pathlib.Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _assert_clean_committed_preparation(repo: pathlib.Path) -> tuple[str, dict[str, str]]:
    """Bind preparation to a clean commit containing the protocol, builder, and this command."""
    info = run_plan.git_commit_info(repo)
    if not info.get("git_commit") or info.get("git_dirty"):
        raise RuntimeError("real paired-audit preparation requires a clean committed checkout")
    paths = (
        "research/work_balanced/paired_audit_protocol.md",
        "scripts/evaluation/prepare_paired_audit_inputs.py",
        "src/stylo/eval/paired_audit/corpus.py",
    )
    digests: dict[str, str] = {}
    for rel in paths:
        disk = repo / rel
        try:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{rel}"], cwd=repo, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"preparation input {rel} is not committed at HEAD") from exc
        if not disk.is_file() or disk.is_symlink() or disk.read_bytes() != committed:
            raise RuntimeError(f"preparation input {rel} differs from committed HEAD")
        digests[rel] = hashlib.sha256(committed).hexdigest()
    return str(info["git_commit"]), digests


def _inventory_mismatches(sums_path: pathlib.Path, root: pathlib.Path) -> tuple[int, list[dict]]:
    mismatches: list[dict] = []
    n = 0
    for raw in sums_path.read_text(encoding="utf-8").splitlines():
        digest, sep, name = raw.partition("  ")
        rel = pathlib.PurePosixPath(name)
        if (sep != "  " or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or not name or rel.is_absolute() or ".." in rel.parts):
            raise RuntimeError(f"malformed RuAA SHA256SUMS entry: {raw!r}")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"RuAA inventory file missing or a symlink: {name}")
        actual = _sha256_file(path)
        if actual != digest:
            mismatches.append({"name": name, "expected": digest, "actual": actual})
        n += 1
    return n, mismatches


def _assert_only_known_ruaa_protocol_drift(n: int, mismatches: list[dict]) -> None:
    if n != 141 or mismatches != [_KNOWN_RUAA_PROTOCOL_DRIFT]:
        raise RuntimeError(
            "--allow-unapproved-ruaa-drift permits only the exact registered protocol.md metadata "
            f"drift across an otherwise-valid 141-file inventory; got {mismatches!r}"
        )


def _ruaa_work_ids(ruaa_manifest_path: pathlib.Path) -> list[str]:
    raw = load_strict(ruaa_manifest_path)
    authors = raw.get("authors") if isinstance(raw, dict) else None
    if not isinstance(authors, dict):
        raise RuntimeError("RuAA manifest must carry an authors mapping")
    work_ids: list[str] = []
    for author, record in authors.items():
        books = record.get("books") if isinstance(record, dict) else None
        if not isinstance(author, str) or not isinstance(books, list):
            raise RuntimeError("RuAA manifest author records are malformed")
        declared = record.get("n_books")
        if declared != len(books):
            raise RuntimeError(f"RuAA author {author}: n_books != len(books)")
        for book in books:
            name = book.get("book") if isinstance(book, dict) else None
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"RuAA author {author}: malformed book entry")
            work_ids.append(f"{author}/{name}")
    if len(work_ids) != 137 or len(set(work_ids)) != 137 or len(authors) != 22:
        raise RuntimeError(
            f"RuAA selection must be 137 unique works / 22 authors, got "
            f"{len(work_ids)}/{len(authors)}"
        )
    return sorted(work_ids)


def _write_freeze_candidate(parent: pathlib.Path, body: dict,
                            lobo: dict, ruaa: dict) -> pathlib.Path:
    body = dict(body)
    body.pop("self_hash", None)
    if parent.is_symlink():
        raise RuntimeError("freeze-candidate parent must not be a symlink")
    _verify_real_dir_chain(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _verify_real_dir_chain(parent)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".staging_", dir=parent))
    try:
        lobo_path = staging / "lobo_fold_manifest_v1.json"
        ruaa_path = staging / "ruaa_fold_manifest_v1.json"
        dump_strict(lobo, lobo_path, trailing_newline=True)
        dump_strict(ruaa, ruaa_path, trailing_newline=True)
        body["fold_manifest_files"] = {
            "lobo": {"name": lobo_path.name, "sha256": _sha256_file(lobo_path),
                     "self_hash": lobo["self_hash"]},
            "ruaa": {"name": ruaa_path.name, "sha256": _sha256_file(ruaa_path),
                     "self_hash": ruaa["self_hash"]},
        }
        body["self_hash"] = _self_hash(body)
        candidate_path = staging / "freeze_candidate.json"
        dump_strict(body, candidate_path, trailing_newline=True)
        files = (candidate_path, lobo_path, ruaa_path)
        sums = "".join(f"{_sha256_file(p)}  {p.name}\n" for p in sorted(files))
        (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")

        destination = parent / body["self_hash"]
        if destination.exists():
            _verify_existing_candidate(destination, body, lobo, ruaa)
            shutil.rmtree(staging)
            return destination
        os.replace(staging, destination)
        staging = None
        return destination
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def prepare(repo: pathlib.Path, *, data_root: pathlib.Path | None = None,
            input_clean_root: pathlib.Path | None = None,
            output_data_root: pathlib.Path | None = None,
            allow_unapproved_ruaa_drift: bool) -> pathlib.Path:
    cfg = load_config(repo / "configs" / "default.yaml")
    data = (data_root or (repo / "data")).resolve()
    input_clean = (input_clean_root or (repo / "input_clean")).resolve()
    output_data = (output_data_root or data).resolve()
    ruaa_root = data / "ruaa_bench_v1"
    from stylo.pipeline.split import resolve_fragment_snapshot

    source_fragments = resolve_fragment_snapshot(data).train_root
    protocol = repo / "research" / "work_balanced" / "paired_audit_protocol.md"
    git_head, preparation_digests = _assert_clean_committed_preparation(repo)

    root = audit_corpus.build_audit_corpus(
        source_frags_root=source_fragments,
        input_clean_root=input_clean,
        cfg=cfg,
        audit_parent=output_data / "audit_corpus",
        exclude_authors=cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [],
        unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"),
        expected_n_works=255,
    )
    corpus_record = audit_corpus.verify_published_corpus(root)
    lobo_dataset = audit_corpus.load_audit_dataset(root, cfg)
    work_ids = _ruaa_work_ids(ruaa_root / "manifest.json")
    ruaa_dataset = derive_work_subset(lobo_dataset, work_ids)

    config_digest = run_plan.config_id(cfg)
    parent_digest = lobo_dataset.provenance.rows_digest
    lobo = fold_manifest.build_fold_manifest(
        "lobo", lobo_dataset, parent_dataset_digest=parent_digest,
        algorithm=fold_manifest.REGISTERED_ALGORITHM["lobo"],
        seed=fold_manifest.REGISTERED_SEED, config_hash=config_digest,
    )
    ruaa = fold_manifest.build_fold_manifest(
        "ruaa", ruaa_dataset, parent_dataset_digest=parent_digest,
        algorithm=fold_manifest.REGISTERED_ALGORITHM["ruaa"],
        seed=fold_manifest.REGISTERED_SEED, config_hash=config_digest,
        selection_digest=ruaa_dataset.provenance.selection_manifest_digest,
    )
    fold_manifest.assert_lobo_universe(lobo)
    fold_manifest.assert_ruaa_universe(ruaa)
    fold_manifest.assert_manifest_consistent_with_dataset(lobo, lobo_dataset)
    fold_manifest.assert_manifest_consistent_with_dataset(ruaa, ruaa_dataset)

    inventory_error = None
    try:
        n_inventory = references.verify_ruaa_inventory(
            ruaa_root / "SHA256SUMS",
            ruaa_root,
            expected_files=references.RUAA_REGISTERED_INVENTORY_FILES,
        )
    except references.ReferenceError as exc:
        inventory_error = str(exc)
        if not allow_unapproved_ruaa_drift:
            raise
        listed, mismatches = _inventory_mismatches(ruaa_root / "SHA256SUMS", ruaa_root)
        _assert_only_known_ruaa_protocol_drift(listed, mismatches)
        n_inventory = listed - len(mismatches)

    body = {
        "schema": SCHEMA,
        "status": "unapproved",
        "hard_stop": "independent manifest review and approved freeze-root pin required before execution",
        "git_head": git_head,
        "git_dirty": False,
        "preparation_file_sha256": preparation_digests,
        "execution_source_sha256": run_plan.execution_source_sha256(repo / "src"),
        "protocol_sha256": _sha256_file(protocol),
        "config_id": config_digest,
        "audit_corpus": {
            "relative_root": str(root.relative_to(output_data)),
            "digest": root.name,
            "manifest_self_hash": corpus_record["self_hash"],
            "legacy_anchor": corpus_record["legacy_anchor"],
            "semantic_parity_digest": corpus_record["source_semantic_parity_digest"],
            "n_works": corpus_record["n_works"],
            "n_chunks": corpus_record["n_chunks"],
        },
        "datasets": {
            "lobo": {"dataset_digest": lobo_dataset.provenance.rows_digest,
                     "n_authors": len(lobo_dataset.authors), "n_rows": len(lobo_dataset)},
            "ruaa": {"dataset_digest": ruaa_dataset.provenance.rows_digest,
                     "parent_dataset_digest": ruaa_dataset.provenance.parent_rows_digest,
                     "selection_digest": ruaa_dataset.provenance.selection_manifest_digest,
                     "n_authors": len(ruaa_dataset.authors), "n_works": len(work_ids),
                     "n_rows": len(ruaa_dataset)},
        },
        "ruaa_source_inventory": {
            "sha256sums_file_sha256": _sha256_file(ruaa_root / "SHA256SUMS"),
            "verified_files": n_inventory,
            "expected_files": references.RUAA_REGISTERED_INVENTORY_FILES,
            "verification_error": inventory_error,
        },
    }
    return _write_freeze_candidate(output_data / "paired_audit_preparation", body, lobo, ruaa)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=pathlib.Path,
                        help="read-only source data root (defaults to <repo>/data)")
    parser.add_argument("--input-clean-root", type=pathlib.Path,
                        help="read-only cleaned-source root (defaults to <repo>/input_clean)")
    parser.add_argument("--output-data-root", type=pathlib.Path,
                        help="where immutable preparation artifacts are written (defaults to data root)")
    parser.add_argument(
        "--allow-unapproved-ruaa-drift", action="store_true",
        help="record the known RuAA inventory mismatch in an UNAPPROVED candidate; never authorizes execution",
    )
    args = parser.parse_args()
    out = prepare(args.repo_root.resolve(),
                  data_root=args.data_root,
                  input_clean_root=args.input_clean_root,
                  output_data_root=args.output_data_root,
                  allow_unapproved_ruaa_drift=args.allow_unapproved_ruaa_drift)
    print(out)


if __name__ == "__main__":
    main()
