import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_script("build_wikisource_edition_panel")
repacker = _load_script("repack_edition_panel")


def _spec(period: str) -> dict:
    return {
        "schema_version": "1.0",
        "status": "exploratory_internal_not_preregistered",
        "description": "test acquisition spec",
        "min_block_words": 80,
        "min_total_matched_words": 80,
        "works": [
            {
                "author": "author",
                "work_key": "work",
                "period": period,
                "local_path": "input_clean/author/work.txt",
                "versions": [
                    {"title": "Main", "revid": 101},
                    {"title": "Version 2", "revid": 102},
                ],
            }
        ],
    }


def _build_source(tmp_path, monkeypatch):
    project = tmp_path / "project"
    research = project / "research"
    local = project / "input_clean" / "author" / "work.txt"
    research.mkdir(parents=True)
    local.parent.mkdir(parents=True)
    text = " ".join(f"слово{index}" for index in range(120))
    local.write_text(text, encoding="utf-8")

    source_spec = _spec("1900s")
    source_spec_path = research / "source.yaml"
    source_spec_path.write_text(
        builder.yaml.safe_dump(source_spec, allow_unicode=True), encoding="utf-8"
    )

    def fake_fetch(title, revid):
        return {
            "requested_title": title,
            "title": title,
            "revid": revid,
            "parentid": revid - 1,
            "timestamp": "2026-07-10T00:00:00Z",
            "wiki_sha1": f"sha1-{revid}",
            "url": f"https://example.test/?oldid={revid}",
            "plain": text,
        }

    monkeypatch.setattr(builder, "_fetch_exact_page", fake_fetch)
    source = project / "source"
    builder.build_panel(source_spec_path, source, delay=0)
    return project, source, source_spec


def test_repack_rebinds_embedded_spec_and_written_manifest_bytes(
    tmp_path, monkeypatch
):
    project, source, source_spec = _build_source(tmp_path, monkeypatch)
    source_manifest_sha256 = hashlib.sha256(
        (source / "manifest.json").read_bytes()
    ).hexdigest()

    updated_spec = _spec("1910s")
    updated_spec_path = project / "research" / "updated.yaml"
    updated_spec_path.write_text(
        builder.yaml.safe_dump(updated_spec, allow_unicode=True), encoding="utf-8"
    )
    out = project / "repacked"

    result = repacker.repack(source, updated_spec_path, out)

    manifest_bytes = (out / "manifest.json").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    alignment = json.loads(
        (out / "alignment_report.json").read_text(encoding="utf-8")
    )
    assert alignment["spec"] == updated_spec
    assert alignment["spec_sha256"] == repacker._canonical_json_sha256(updated_spec)
    assert alignment["manifest_sha256"] == manifest_sha256
    assert alignment["repack"]["source_manifest_sha256"] == source_manifest_sha256
    assert alignment["repack"]["source_spec_sha256"] == (
        repacker._canonical_json_sha256(source_spec)
    )
    assert result["manifest_sha256"] == manifest_sha256
    assert {row["period"] for row in json.loads(manifest_bytes)["documents"]} == {
        "1910s"
    }


@pytest.mark.parametrize("tamper", ["manifest_bytes", "embedded_spec"])
def test_repack_rejects_tampered_source_bindings(tmp_path, monkeypatch, tamper):
    project, source, _source_spec = _build_source(tmp_path, monkeypatch)
    if tamper == "manifest_bytes":
        manifest_path = source / "manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        match = "manifest_sha256 does not match source manifest bytes"
    else:
        alignment_path = source / "alignment_report.json"
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        alignment["spec"] = copy.deepcopy(alignment["spec"])
        alignment["spec"]["description"] = "tampered after acquisition"
        alignment_path.write_text(
            json.dumps(alignment, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        match = "spec_sha256 does not match its embedded spec"

    with pytest.raises(ValueError, match=match):
        repacker.repack(
            source,
            project / "research" / "source.yaml",
            project / "out",
        )
