#!/usr/bin/env python3
"""Repack a verified aligned panel after metadata-only spec corrections.

This never realigns or mutates text.  It is useful when immutable source
revisions and artifact hashes have already been verified, but manifest metadata
such as period or split semantics needs a new exploratory dataset version.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import shutil

import yaml

from stylo.benchmarks import load_manifest, verify_manifest_artifacts
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402


def _canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value independently of whitespace/key order."""
    payload = dumps_strict(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def repack(source: pathlib.Path, spec_path: pathlib.Path, out: pathlib.Path) -> dict:
    manifest_path = source / "manifest.json"
    source_manifest_bytes = manifest_path.read_bytes()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()

    alignment_path = source / "alignment_report.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment.get("manifest_sha256") != source_manifest_sha256:
        raise ValueError(
            "source alignment manifest_sha256 does not match source manifest bytes"
        )
    source_spec_sha256 = _canonical_json_sha256(alignment.get("spec"))
    if alignment.get("spec_sha256") != source_spec_sha256:
        raise ValueError(
            "source alignment spec_sha256 does not match its embedded spec"
        )

    manifest = load_manifest(manifest_path)
    verify_manifest_artifacts(manifest, source)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    audits = {
        (row["author"], row["work_key"]): row for row in alignment["works"]
    }
    entries = {
        (row["author"], row["work_key"]): row for row in spec["works"]
    }
    if set(audits) != set(entries):
        raise ValueError("source alignment works do not match the current spec")

    period_by_work_id = {}
    for key, entry in entries.items():
        audit = audits[key]
        versions = entry.get("versions", [])
        revisions = audit.get("wikisource_revisions", [])
        if len(versions) != 2 or len(revisions) != 2:
            raise ValueError(f"{key}: expected two pinned revisions")
        for version, revision in zip(versions, revisions):
            recorded_title = revision.get("requested_title", revision.get("title"))
            if (
                version.get("title") != recorded_title
                or version.get("revid") != revision.get("revid")
            ):
                raise ValueError(f"{key}: cached revision does not match current spec")
        period_by_work_id[audit["work_id"]] = str(entry["period"])

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    raw_manifest = json.loads(source_manifest_bytes)
    for document in raw_manifest["documents"]:
        work_id = document["work"]
        if work_id not in period_by_work_id:
            raise ValueError(f"manifest work is absent from alignment audit: {work_id}")
        document["split"] = "train"
        document["period"] = period_by_work_id[work_id]
        relative = pathlib.PurePosixPath(document["text_path"])
        target = out.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.joinpath(*relative.parts), target)

    rendered_manifest_bytes = (
        dumps_strict(raw_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_sha256 = hashlib.sha256(rendered_manifest_bytes).hexdigest()
    (out / "manifest.json").write_bytes(rendered_manifest_bytes)

    repacked_alignment = copy.deepcopy(alignment)
    repacked_alignment["spec"] = spec
    repacked_alignment["spec_sha256"] = _canonical_json_sha256(spec)
    repacked_alignment["manifest_sha256"] = manifest_sha256
    repacked_alignment["repack"] = {
        "operation": "metadata_only_text_artifacts_unchanged",
        "source_manifest_sha256": source_manifest_sha256,
        "source_spec_sha256": source_spec_sha256,
        "source_artifacts_reverified": True,
    }
    (out / "alignment_report.json").write_text(
        dumps_strict(repacked_alignment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    repacked = load_manifest(out / "manifest.json")
    verify_manifest_artifacts(repacked, out)
    return {
        "out": str(out.resolve()),
        "n_documents": len(repacked.documents),
        "source_manifest_sha256": source_manifest_sha256,
        "manifest_sha256": manifest_sha256,
        "text_artifacts_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument(
        "--spec", default="research/wikisource_edition_pilot_v1.yaml"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = repack(
        pathlib.Path(args.source), pathlib.Path(args.spec), pathlib.Path(args.out)
    )
    print(dumps_strict(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
