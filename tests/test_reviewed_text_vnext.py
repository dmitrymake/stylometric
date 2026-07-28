import copy
import hashlib
import inspect
import json
import socket
from pathlib import Path

import pytest

from stylo.corpus_tools import reviewed_text_vnext as reviewed
from stylo.jsonio import canonical_hash, dump_strict, load_strict


ROOT = Path(__file__).parents[1]
EXTERNAL_EDITORIAL_EVIDENCE_NAMES = frozenset(
    {
        "build_stylo_prutkov_clean_source_probe.py",
        "build_stylo_prutkov_fruits_rev5658784.py",
        "build_stylo_tsushima_sheba_main.py",
        "build_stylo_tsushima_sheba_scan_corrected-final-independent.py",
        "build_stylo_warpeace_tolstoyru_reviewed_final.py",
        "stylo-prutkov-clean-source-probe.json",
        "stylo-prutkov-fruits-rev5658784.clean-manifest.json",
        "stylo-tsushima-sheba-1984-main-scan-corrected-final-independent.json",
        "stylo-tsushima-sheba-1984-main.json",
        "stylo-warpeace-tolstoyru-reviewed-final.json",
    }
)


def _payload(label: str) -> bytes:
    return (
        f"{label} НАЧАЛО\n"
        "\n"
        f"{label} первая строка литературного текста\n"
        f"{label} последняя строка\n"
    ).encode("utf-8")


def _artifact_ref(name: str, marker: str) -> reviewed.ReviewedTextArtifactRef:
    return reviewed.ReviewedTextArtifactRef.build(
        logical_name=name,
        payload=marker.encode("utf-8"),
    )


def _work(
    work_id: str,
    payload: bytes,
    *,
    source_parts: int,
    reviewed_parts: int,
) -> reviewed.ReviewedTextWorkSpec:
    return reviewed.ReviewedTextWorkSpec.build(
        work_id=work_id,
        text_payload=payload,
        builder_artifacts=[
            _artifact_ref("z_builder", f"{work_id}: builder z"),
            _artifact_ref("a_builder", f"{work_id}: builder a"),
        ],
        provenance_artifacts=[
            _artifact_ref("source_manifest", f"{work_id}: manifest"),
        ],
        source_part_count=source_parts,
        reviewed_part_count=reviewed_parts,
    )


def _fixture():
    payloads = {
        "alpha/первое": _payload("УНИКАЛЬНАЯ АЛЬФА"),
        "beta/второе": _payload("УНИКАЛЬНАЯ БЕТА"),
    }
    second = _work(
        "beta/второе",
        payloads["beta/второе"],
        source_parts=3,
        reviewed_parts=10,
    )
    first = _work(
        "alpha/первое",
        payloads["alpha/первое"],
        source_parts=1,
        reviewed_parts=2,
    )
    return reviewed.ReviewedTextCampaignSpec.build([second, first]), payloads


def _populate_cache(
    cache: Path,
    spec: reviewed.ReviewedTextCampaignSpec,
    payloads: dict[str, bytes],
) -> None:
    for work in spec.works:
        target = cache.joinpath(*work.artifact_key.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[work.work_id])


def _rehash(raw: dict) -> dict:
    raw["self_hash"] = canonical_hash(
        {key: value for key, value in raw.items() if key != "self_hash"}
    )
    return raw


def test_campaign_spec_is_strict_sorted_self_hashed_and_hash_only():
    spec, payloads = _fixture()

    assert spec.work_ids == ("alpha/первое", "beta/второе")
    assert tuple(row.work_id for row in spec.works) == spec.work_ids
    assert spec.validate() is spec
    assert spec.self_hash == canonical_hash(
        {
            key: value
            for key, value in spec.to_dict().items()
            if key != "self_hash"
        }
    )
    raw = spec.to_dict()
    assert (
        raw["upstream_inventory_file_sha256"]
        == reviewed.UPSTREAM_INVENTORY_FILE_SHA256
    )
    assert (
        raw["upstream_quality_audit_file_sha256"]
        == reviewed.UPSTREAM_QUALITY_AUDIT_FILE_SHA256
    )
    assert (
        raw["upstream_quality_audit_self_hash"]
        == reviewed.UPSTREAM_QUALITY_AUDIT_SELF_HASH
    )
    encoded = json.dumps(raw, ensure_ascii=False)
    assert all(
        payload.decode("utf-8").strip() not in encoded
        for payload in payloads.values()
    )

    alpha = spec.works[0]
    assert alpha.artifact_key == f"sha256/{alpha.sha256}.txt"
    assert alpha.line_count == 4
    assert alpha.source_part_count == 1
    assert alpha.reviewed_part_count == 2
    assert [row.logical_name for row in alpha.builder_artifacts] == [
        "a_builder",
        "z_builder",
    ]
    assert alpha.first_nonblank_line_sha256 == hashlib.sha256(
        "УНИКАЛЬНАЯ АЛЬФА НАЧАЛО".encode("utf-8")
    ).hexdigest()
    assert alpha.last_nonblank_line_sha256 == hashlib.sha256(
        "УНИКАЛЬНАЯ АЛЬФА последняя строка".encode("utf-8")
    ).hexdigest()


def test_campaign_spec_rejects_rehashed_reordering_extras_and_duplicate_json():
    spec, _payloads = _fixture()

    reordered = copy.deepcopy(spec.to_dict())
    reordered["work_ids"].reverse()
    reordered["works"].reverse()
    _rehash(reordered)
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="sorted exactly",
    ):
        reviewed.ReviewedTextCampaignSpec.from_dict(reordered)

    extra = copy.deepcopy(spec.to_dict())
    extra["works"][0]["extra"] = "rehashed but forbidden"
    _rehash(extra)
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="keys must be exact",
    ):
        reviewed.ReviewedTextCampaignSpec.from_dict(extra)

    wrong_upstream = copy.deepcopy(spec.to_dict())
    wrong_upstream["upstream_inventory_file_sha256"] = "0" * 64
    _rehash(wrong_upstream)
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="upstream_inventory_file_sha256",
    ):
        reviewed.ReviewedTextCampaignSpec.from_dict(wrong_upstream)

    encoded = json.dumps(spec.to_dict(), ensure_ascii=False)
    duplicate = encoded.replace(
        '"campaign_kind":',
        '"campaign_kind": "duplicate", "campaign_kind":',
        1,
    )
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="duplicate object key",
    ):
        reviewed.loads_reviewed_text_campaign_spec(duplicate)

    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="duplicate work ids",
    ):
        reviewed.ReviewedTextCampaignSpec.build(
            [spec.works[0], spec.works[0]]
        )


def test_nested_work_contract_rejects_noncanonical_key_refs_and_bool_count():
    spec, _payloads = _fixture()
    raw = spec.works[0].to_dict()
    raw["artifact_key"] = f"sha256/{'0' * 64}.txt"
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="artifact_key",
    ):
        reviewed.ReviewedTextWorkSpec.from_dict(raw)

    raw = spec.works[0].to_dict()
    raw["builder_artifacts"].reverse()
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="sorted by unique",
    ):
        reviewed.ReviewedTextWorkSpec.from_dict(raw)

    raw = spec.works[0].to_dict()
    raw["source_part_count"] = True
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="exact integer",
    ):
        reviewed.ReviewedTextWorkSpec.from_dict(raw)


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"\xff\n", "strict UTF-8"),
        ("Текст".encode("utf-8"), "exactly one final LF"),
        ("Текст\n\n".encode("utf-8"), "exactly one final LF"),
        ("Текст\r\n".encode("utf-8"), "rejected CR"),
        ("\ufeffТекст\n".encode("utf-8"), "UTF-8 BOM"),
    ],
)
def test_work_build_rejects_noncanonical_text(payload, message):
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match=message,
    ):
        _work(
            "author/work",
            payload,
            source_parts=1,
            reviewed_parts=1,
        )


def test_local_materialization_is_exact_compact_resumable_and_network_free(
    tmp_path,
    monkeypatch,
):
    spec, payloads = _fixture()
    cache = tmp_path / "cache"
    _populate_cache(cache, spec, payloads)

    def reject_socket(*_args, **_kwargs):
        raise AssertionError("reviewed-text provider attempted network access")

    monkeypatch.setattr(socket, "socket", reject_socket)
    first = reviewed.materialize_reviewed_text_campaign(
        spec,
        artifact_cache=cache,
        output_parent=tmp_path / "published",
    )

    assert first.resumed is False
    assert first.root == tmp_path / "published" / spec.self_hash
    assert first.receipt.campaign_spec_sha256 == spec.self_hash
    assert first.receipt.self_hash == canonical_hash(
        {
            key: value
            for key, value in first.receipt.to_dict().items()
            if key != "self_hash"
        }
    )
    assert {
        path.relative_to(first.root).as_posix()
        for path in first.root.rglob("*")
        if path.is_file()
    } == {
        "campaign-spec.json",
        "campaign-receipt.json",
        "raw/alpha/первое.txt",
        "raw/beta/второе.txt",
    }
    assert first.output_path("alpha/первое").read_bytes() == payloads[
        "alpha/первое"
    ]
    assert tuple(path.read_bytes() for path in first.output_paths) == tuple(
        payloads[work_id] for work_id in spec.work_ids
    )

    resumed = reviewed.materialize_reviewed_text_campaign(
        spec,
        artifact_cache=tmp_path / "cache-does-not-exist",
        output_parent=tmp_path / "published",
    )
    assert resumed.resumed is True
    assert resumed.receipt == first.receipt

    assert list(
        inspect.signature(
            reviewed.materialize_reviewed_text_campaign
        ).parameters
    ) == ["spec", "artifact_cache", "output_parent"]


@pytest.mark.parametrize("mutation", ["root_symlink", "entry_symlink"])
def test_artifact_cache_rejects_symlinks(tmp_path, mutation):
    spec, payloads = _fixture()
    real_cache = tmp_path / "real-cache"
    _populate_cache(real_cache, spec, payloads)
    cache = real_cache
    if mutation == "root_symlink":
        cache = tmp_path / "cache-link"
        cache.symlink_to(real_cache, target_is_directory=True)
    else:
        first = spec.works[0]
        target = real_cache.joinpath(*first.artifact_key.split("/"))
        elsewhere = tmp_path / "elsewhere.txt"
        elsewhere.write_bytes(payloads[first.work_id])
        target.unlink()
        target.symlink_to(elsewhere)

    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="symlink components|missing or unsafe",
    ):
        reviewed.materialize_reviewed_text_campaign(
            spec,
            artifact_cache=cache,
            output_parent=tmp_path / "published",
        )


def test_artifact_cache_rejects_byte_and_text_identity_drift(tmp_path):
    spec, payloads = _fixture()
    cache = tmp_path / "cache"
    _populate_cache(cache, spec, payloads)
    work = spec.works[0]
    target = cache.joinpath(*work.artifact_key.split("/"))
    payload = target.read_bytes()
    target.write_bytes(payload[:-1] + b"x")

    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="exactly one final LF",
    ):
        reviewed.materialize_reviewed_text_campaign(
            spec,
            artifact_cache=cache,
            output_parent=tmp_path / "published",
        )

    target.write_bytes(b"x")
    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="byte size mismatch",
    ):
        reviewed.materialize_reviewed_text_campaign(
            spec,
            artifact_cache=cache,
            output_parent=tmp_path / "published-2",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "missing",
        "extra",
        "symlink",
        "spec_noncanonical",
        "receipt_noncanonical",
        "receipt_rehashed",
    ],
)
def test_resume_rejects_full_generation_tamper(tmp_path, mutation):
    spec, payloads = _fixture()
    cache = tmp_path / f"cache-{mutation}"
    _populate_cache(cache, spec, payloads)
    result = reviewed.materialize_reviewed_text_campaign(
        spec,
        artifact_cache=cache,
        output_parent=tmp_path / f"published-{mutation}",
    )
    alpha = result.output_path("alpha/первое")
    if mutation == "raw":
        payload = alpha.read_bytes()
        alpha.write_bytes(payload[:-1] + b"x")
    elif mutation == "missing":
        result.receipt_path.unlink()
    elif mutation == "extra":
        (result.root / "extra.bin").write_bytes(b"extra")
    elif mutation == "symlink":
        alpha.unlink()
        alpha.symlink_to(result.output_path("beta/второе"))
    elif mutation == "spec_noncanonical":
        raw = json.loads(result.spec_path.read_text(encoding="utf-8"))
        result.spec_path.write_text(
            json.dumps(raw, ensure_ascii=False),
            encoding="utf-8",
        )
    elif mutation == "receipt_noncanonical":
        raw = json.loads(result.receipt_path.read_text(encoding="utf-8"))
        result.receipt_path.write_text(
            json.dumps(raw, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        raw = result.receipt.to_dict()
        raw["works"][0]["byte_size"] += 1
        _rehash(raw)
        dump_strict(raw, result.receipt_path, sort_keys=True)

    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match=(
            "identity mismatch|missing or extra|symlink rejected|"
            "noncanonical JSON bytes|receipt/spec/output mismatch|"
            "exactly one final LF"
        ),
    ):
        reviewed.materialize_reviewed_text_campaign(
            spec,
            artifact_cache=tmp_path / "must-not-be-read",
            output_parent=tmp_path / f"published-{mutation}",
        )


def test_receipt_rejects_rehashed_nested_extra():
    spec, _payloads = _fixture()
    receipt = reviewed.ReviewedTextCampaignReceipt.build(spec)
    raw = receipt.to_dict()
    raw["works"][0]["extra"] = "forbidden"
    _rehash(raw)

    with pytest.raises(
        reviewed.ReviewedTextMaterializationError,
        match="keys must be exact",
    ):
        reviewed.ReviewedTextCampaignReceipt.from_dict(raw)


def test_tracked_r1_reviewed_contract_is_compact_self_hashed_and_disjoint():
    sources = ROOT / "research" / "corpus_sources"
    provenance_path = (
        sources / "ruaa_r1_reviewed_text_provenance_v1.json"
    )
    dispositions_path = sources / "ruaa_r1_source_dispositions_v1.json"
    campaign_path = sources / "ruaa_r1_reviewed_text_campaign_v1.json"

    for path in (provenance_path, dispositions_path):
        raw = load_strict(path)
        assert type(raw) is dict
        recorded = raw["self_hash"]
        assert canonical_hash(
            {key: value for key, value in raw.items() if key != "self_hash"}
        ) == recorded

    campaign = reviewed.load_reviewed_text_campaign_spec(campaign_path)
    assert campaign.work_ids == (
        "korolenko/дети_подземелья",
        "korolenko/слепой_музыкант",
        "novikov_priboy/цусима",
        "prutkov/выдержки_из_записок_деда",
        "prutkov/плоды_раздумья",
        "tolstoy/война_и_мир",
    )
    assert "pushkin/история_пугачёва" not in campaign.work_ids
    assert campaign.self_hash == (
        "c87deecb01ea9db922e02305efee9cfda9c76fe4e4e03d38e28dc6438eb63f7f"
    )
    assert campaign_path.stat().st_size < 10_000
    assert provenance_path.stat().st_size < 12_000

    # The external editorial evidence is bound by name, SHA-256, and byte
    # size only.  Those files are curated outside this repository, and the
    # reviewed provider never opens them: it consumes finished hash-bound
    # texts from the content-addressed cache.
    referenced = {
        ref.logical_name: ref
        for work in campaign.works
        for ref in (*work.builder_artifacts, *work.provenance_artifacts)
    }
    assert (
        set(referenced) - {provenance_path.name}
        == EXTERNAL_EDITORIAL_EVIDENCE_NAMES
    )
    for name in EXTERNAL_EDITORIAL_EVIDENCE_NAMES:
        ref = referenced[name]
        assert len(ref.sha256) == 64 and ref.byte_size > 0
        assert not (sources / name).exists()
        assert not (
            sources / "ruaa_r1_reviewed_text_evidence_v1" / name
        ).exists()
