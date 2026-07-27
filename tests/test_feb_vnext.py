import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from stylo.corpus_tools.feb_vnext import (
    FEBAcquisitionError,
    FEBHTTPResponse,
    FEB_CONTENT_TYPE,
    FEB_SOURCE_URL,
    PinnedFEBWorkSpec,
    extract_feb_main_narrative,
    materialize_pinned_feb_work,
)
from stylo.jsonio import dump_strict


def _fixture_html(*, chapters: int = 8, notes: str = "EDITORIAL NOTES") -> bytes:
    names = (
        "ПЕРВАЯ",
        "ВТОРАЯ",
        "ТРЕТИЯ",
        "ЧЕТВЕРТАЯ",
        "ПЯТАЯ",
        "ШЕСТАЯ",
        "СЕДЬМАЯ",
        "ОСЬМАЯ",
    )
    parts = [
        '<html><body><div class="text" id="prose">',
        '<span class="page">107</span>',
        '<h4 L="0" id="Заголовок"></h4>',
        "<p>ИСТОРИЯ ПУГАЧЕВА</p>",
        '<h4 L="0" id="Текст"></h4>',
        '<h4 L="1" id="ПРЕДИСЛОВИЕ"></h4>',
        "<p>ПРЕДИСЛОВИЕ</p>",
        "<p>" + " ".join(f"предисловие{i}" for i in range(40)) + "</p>",
    ]
    for index, name in enumerate(names[:chapters]):
        parts.extend(
            [
                f'<h4 L="1" id="ГЛАВА_{name}"></h4>',
                f"<p>ГЛАВА {name}</p>",
                "<p>"
                + " ".join(
                    f"глава{index}слово{word}" for word in range(45)
                )
                + f'<sup><a href="#note">1</a></sup></p>',
            ]
        )
    parts.extend(
        [
            '<h4 L="0" id="ПРИМЕЧАНИЯ"></h4>',
            f"<p>{notes}</p>",
            "</div></body></html>",
        ]
    )
    return "".join(parts).encode("windows-1251")


def _response(body: bytes) -> FEBHTTPResponse:
    return FEBHTTPResponse(
        FEB_SOURCE_URL,
        200,
        FEB_CONTENT_TYPE,
        body,
    )


def test_exact_feb_extraction_keeps_eight_chapters_and_stops_before_notes():
    selected = extract_feb_main_narrative(_fixture_html())

    assert selected.startswith("ПРЕДИСЛОВИЕ\n")
    assert "ИСТОРИЯ ПУГАЧЕВА" not in selected
    assert "ГЛАВА ПЕРВАЯ" in selected
    assert "ГЛАВА ОСЬМАЯ" in selected
    assert "EDITORIAL NOTES" not in selected
    assert "\n107\n" not in selected
    assert "\n1\n" not in selected


@pytest.mark.parametrize(
    "payload",
    [
        b"<html></html>",
        _fixture_html(chapters=7),
        _fixture_html().replace(
            b'id="\xcf\xd0\xc8\xcc\xc5\xd7\xc0\xcd\xc8\xdf"',
            b'id="OTHER"',
        ),
    ],
)
def test_feb_extraction_rejects_missing_markers_or_chapters(payload):
    with pytest.raises(FEBAcquisitionError):
        extract_feb_main_narrative(payload)


def test_pinned_spec_is_strict_self_hashed_and_byte_sensitive():
    body = _fixture_html()
    first = PinnedFEBWorkSpec.build(
        work_id="pushkin/история_пугачёва",
        response_body=body,
    )
    second = PinnedFEBWorkSpec.build(
        work_id="pushkin/история_пугачёва",
        response_body=body.replace(b"<body>", b"<body >"),
    )

    assert first.response_sha256 == hashlib.sha256(body).hexdigest()
    assert first.generation_id != second.generation_id
    assert first.output_sha256 == second.output_sha256

    raw = copy.deepcopy(first.to_dict())
    raw["extra"] = "rehashed but forbidden"
    raw["self_hash"] = hashlib.sha256(b"irrelevant").hexdigest()
    with pytest.raises(FEBAcquisitionError, match="keys must be exact"):
        PinnedFEBWorkSpec.from_dict(raw)


def test_materialization_and_resume_are_immutable_and_network_free(tmp_path):
    body = _fixture_html()
    spec = PinnedFEBWorkSpec.build(
        work_id="pushkin/история_пугачёва",
        response_body=body,
    )
    calls = []

    def transport(url):
        calls.append(url)
        return _response(body)

    first = materialize_pinned_feb_work(
        spec,
        output_parent=tmp_path,
        transport=transport,
    )
    second = materialize_pinned_feb_work(
        spec,
        output_parent=tmp_path,
        transport=lambda _: pytest.fail("resume must not use network"),
    )

    assert calls == [FEB_SOURCE_URL]
    assert first.resumed is False
    assert second.resumed is True
    assert first.output_path.read_bytes().endswith(b"\n")
    assert first.receipt == second.receipt


def test_materialization_rejects_drift_tamper_and_extra_files(tmp_path):
    body = _fixture_html()
    spec = PinnedFEBWorkSpec.build(
        work_id="pushkin/история_пугачёва",
        response_body=body,
    )
    with pytest.raises(FEBAcquisitionError, match="bytes differ"):
        materialize_pinned_feb_work(
            spec,
            output_parent=tmp_path / "drift",
            transport=lambda _: _response(body + b" "),
        )

    materialized = materialize_pinned_feb_work(
        spec,
        output_parent=tmp_path / "tamper",
        transport=lambda _: _response(body),
    )
    materialized.output_path.write_bytes(b"tampered\n")
    with pytest.raises(FEBAcquisitionError, match="output bytes differ"):
        materialize_pinned_feb_work(
            spec,
            output_parent=tmp_path / "tamper",
            transport=lambda _: pytest.fail("resume must not use network"),
        )

    clean = materialize_pinned_feb_work(
        spec,
        output_parent=tmp_path / "extra",
        transport=lambda _: _response(body),
    )
    (clean.root / "extra.txt").write_text("extra")
    with pytest.raises(FEBAcquisitionError, match="missing or extra"):
        materialize_pinned_feb_work(
            spec,
            output_parent=tmp_path / "extra",
            transport=lambda _: pytest.fail("resume must not use network"),
        )


@pytest.mark.parametrize(
    "response",
    [
        FEBHTTPResponse("https://redirect.invalid", 200, FEB_CONTENT_TYPE, b"x"),
        FEBHTTPResponse(FEB_SOURCE_URL, 404, FEB_CONTENT_TYPE, b"x"),
        FEBHTTPResponse(FEB_SOURCE_URL, 200, "text/plain", b"x"),
    ],
)
def test_materialization_rejects_url_status_or_content_type(response, tmp_path):
    body = _fixture_html()
    spec = PinnedFEBWorkSpec.build(
        work_id="pushkin/история_пугачёва",
        response_body=body,
    )
    with pytest.raises(FEBAcquisitionError, match="URL/status/content-type"):
        materialize_pinned_feb_work(
            spec,
            output_parent=tmp_path,
            transport=lambda _: response,
        )


def test_cli_uses_only_pinned_spec_and_fixed_exploratory_namespace(
    tmp_path,
    monkeypatch,
):
    body = _fixture_html()
    spec = PinnedFEBWorkSpec.build(
        work_id="pushkin/история_пугачёва",
        response_body=body,
    )
    spec_path = tmp_path / "feb-spec.json"
    dump_strict(spec.to_dict(), spec_path, sort_keys=True)
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "acquire_feb_corpus_vnext.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "acquire_feb_corpus_vnext_test",
        script_path,
    )
    assert module_spec is not None and module_spec.loader is not None
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)

    fake_root = tmp_path / "repository"
    fake_output = (
        fake_root
        / "docs"
        / "exploratory"
        / "lobo_vnext"
        / "corpora"
        / "feb"
    )
    monkeypatch.setattr(cli, "ROOT", fake_root)
    monkeypatch.setattr(cli, "OUTPUT_PARENT", fake_output)
    monkeypatch.setattr(
        cli,
        "FEBHTTPTransport",
        lambda **_kwargs: (lambda _url: _response(body)),
    )

    result = cli.run(
        [
            "--pinned-spec",
            str(spec_path),
            "--user-agent",
            "stylometric-test/1.0",
        ]
    )

    assert result["status"] == "exploratory_feb_source_materialized_no_fit"
    assert result["generation_id"] == spec.generation_id
    assert result["namespace_relative_path"] == (
        f"docs/exploratory/lobo_vnext/corpora/feb/{spec.generation_id}"
    )
    assert result["fit_performed"] is False
    assert result["confirmatory_authorized"] is False
