import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "build_wikisource_edition_panel.py"
    spec = importlib.util.spec_from_file_location("build_wikisource_edition_panel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_script()


class _Response:
    def __init__(self, revid=123, status_code=200, headers=None):
        self.revid = revid
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
        return None

    def json(self):
        return {
            "query": {
                "pages": [
                    {
                        "title": "Pinned title",
                        "revisions": [
                            {
                                "revid": self.revid,
                                "parentid": 122,
                                "timestamp": "2026-07-10T00:00:00Z",
                                "sha1": "wiki-sha1",
                                "slots": {"main": {"content": "Pinned body"}},
                            }
                        ],
                    }
                ]
            }
        }


def test_fetch_uses_spec_revision_instead_of_latest_title(monkeypatch):
    seen = {}

    def fake_get(url, *, params, headers, timeout):
        seen.update(params)
        return _Response(123)

    monkeypatch.setattr(builder.requests, "get", fake_get)

    result = builder._fetch_exact_page("Pinned title", 123)

    assert seen["revids"] == "123"
    assert "titles" not in seen
    assert result["requested_title"] == "Pinned title"
    assert result["revid"] == 123
    assert "oldid=123" in result["url"]


def test_fetch_rejects_revision_mismatch(monkeypatch):
    monkeypatch.setattr(
        builder.requests,
        "get",
        lambda *args, **kwargs: _Response(999),
    )

    with pytest.raises(RuntimeError, match="expected 123"):
        builder._fetch_exact_page("Pinned title", 123)


def test_fetch_retries_rate_limit_without_changing_revision(monkeypatch):
    responses = iter(
        [
            _Response(status_code=429, headers={"Retry-After": "0"}),
            _Response(123),
        ]
    )
    sleeps = []
    monkeypatch.setattr(builder.requests, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(builder.time, "sleep", sleeps.append)

    result = builder._fetch_exact_page(
        "Pinned title", 123, max_retries=1, retry_backoff=0.25
    )

    assert result["revid"] == 123
    assert sleeps == [0.25]


def test_fetch_retries_transient_connection_failure(monkeypatch):
    calls = 0
    sleeps = []

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise builder.requests.ConnectionError("temporary DNS failure")
        return _Response(123)

    monkeypatch.setattr(builder.requests, "get", fake_get)
    monkeypatch.setattr(builder.time, "sleep", sleeps.append)

    result = builder._fetch_exact_page(
        "Pinned title", 123, max_retries=1, retry_backoff=0.5
    )

    assert result["revid"] == 123
    assert calls == 2
    assert sleeps == [0.5]


@pytest.mark.parametrize("revid", [True, 0, -1, "123"])
def test_fetch_rejects_invalid_revision_id(revid):
    with pytest.raises(ValueError, match="positive integer"):
        builder._fetch_exact_page("Requested title", revid)


def test_fetch_rejects_revision_from_another_title(monkeypatch):
    monkeypatch.setattr(
        builder.requests,
        "get",
        lambda *args, **kwargs: _Response(123),
    )

    with pytest.raises(RuntimeError, match="not requested title"):
        builder._fetch_exact_page("Different title", 123)


def test_alignment_artifact_binds_embedded_spec_and_manifest(tmp_path, monkeypatch):
    project = tmp_path / "project"
    research = project / "research"
    local = project / "input_clean" / "author" / "work.txt"
    research.mkdir(parents=True)
    local.parent.mkdir(parents=True)
    text = " ".join(f"слово{index}" for index in range(120))
    local.write_text(text, encoding="utf-8")
    spec = {
        "schema_version": "1.0",
        "status": "exploratory_internal_not_preregistered",
        "min_block_words": 80,
        "min_total_matched_words": 80,
        "works": [
            {
                "author": "author",
                "work_key": "work",
                "period": "1900s",
                "local_path": "input_clean/author/work.txt",
                "versions": [
                    {"title": "Main", "revid": 101},
                    {"title": "Version 2", "revid": 102},
                ],
            }
        ],
    }
    spec_path = research / "spec.yaml"
    spec_path.write_text(builder.yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")

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
    monkeypatch.setattr(builder.time, "sleep", lambda _: None)
    out = project / "panel"
    builder.build_panel(spec_path, out, delay=0)

    manifest_payload = (out / "manifest.json").read_bytes()
    artifact = json.loads((out / "alignment_report.json").read_text(encoding="utf-8"))
    assert artifact["artifact_schema_version"] == "1.0"
    assert artifact["acquisition_mode"] == "spec_pinned_revision"
    assert artifact["spec"] == spec
    assert artifact["spec_sha256"] == builder._canonical_json_sha256(spec)
    assert artifact["manifest_sha256"] == hashlib.sha256(manifest_payload).hexdigest()
