import hashlib
import types
import sys

import numpy as np
import pytest

from stylo.config import load_config, with_overrides
from stylo.features.embeddings import EmbeddingBlock
from stylo.features.registry import build_blocks


REVISION = "a" * 40
TEXT = "Синтетический текст для проверки кеша."


def _fake_block(monkeypatch, tmp_path, dimension=3, **kwargs):
    block = EmbeddingBlock(cache_dir=tmp_path, **{"revision": REVISION, **kwargs})
    calls = []

    def ensure_model():
        block._dimension = dimension
        block._device = "cpu"

    def encode(texts):
        calls.append(list(texts))
        return np.full((len(texts), dimension), block.max_length, dtype=np.float32)

    monkeypatch.setattr(block, "_ensure_model", ensure_model)
    monkeypatch.setattr(block, "_encode_batch", encode)
    return block, calls


def test_max_length_change_recomputes_instead_of_reusing_stale_vector(monkeypatch, tmp_path):
    short, _ = _fake_block(monkeypatch, tmp_path, max_length=256)
    short._store(TEXT, np.full(3, 256, dtype=np.float32))
    long, calls = _fake_block(monkeypatch, tmp_path, max_length=512)

    result = long.transform([TEXT], []).toarray()

    assert calls == [[TEXT]]
    np.testing.assert_array_equal(result, [[512, 512, 512]])
    assert short._cache_path(TEXT) != long._cache_path(TEXT)


@pytest.mark.parametrize("change", [
    {"revision": "b" * 40},
    {"tokenizer_revision": "b" * 40},
    {"model_name": "synthetic/other-model"},
    {"batch_size": 8},
])
def test_model_tokenizer_and_inference_identity_changes_recompute(monkeypatch, tmp_path, change):
    original, _ = _fake_block(monkeypatch, tmp_path)
    original._store(TEXT, np.ones(3, dtype=np.float32))
    changed, calls = _fake_block(monkeypatch, tmp_path, **change)

    changed.transform([TEXT], [])

    assert calls == [[TEXT]]
    assert original._key(TEXT) != changed._key(TEXT)


def test_same_pinned_identity_reuses_cache_and_runtime_change_invalidates(monkeypatch, tmp_path):
    original, _ = _fake_block(monkeypatch, tmp_path)
    monkeypatch.setattr(original, "_runtime_identity", lambda: {"numpy": "synthetic-v1"})
    original._store(TEXT, np.ones(3, dtype=np.float32))
    same, same_calls = _fake_block(monkeypatch, tmp_path)
    monkeypatch.setattr(same, "_runtime_identity", lambda: {"numpy": "synthetic-v1"})
    np.testing.assert_array_equal(same.transform([TEXT], []).toarray(), [[1, 1, 1]])
    assert same_calls == []

    changed, changed_calls = _fake_block(monkeypatch, tmp_path)
    monkeypatch.setattr(changed, "_runtime_identity", lambda: {"numpy": "synthetic-v2"})
    changed.transform([TEXT], [])
    assert changed_calls == [[TEXT]]


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "abc123"])
def test_unpinned_revision_computes_without_disk_cache(monkeypatch, tmp_path, revision, caplog):
    block, calls = _fake_block(monkeypatch, tmp_path, revision=revision)

    block.transform([TEXT], [])
    block.transform([TEXT], [])

    assert calls == [[TEXT], [TEXT]]
    assert list(tmp_path.rglob("*.npy")) == []
    assert "disk cache disabled" in caplog.text
    with pytest.raises(ValueError, match="immutable commit revisions"):
        block._key(TEXT)


def test_unpinned_tokenizer_and_local_model_disable_cache(monkeypatch, tmp_path):
    branch, calls = _fake_block(monkeypatch, tmp_path, tokenizer_revision="main")
    branch.transform([TEXT], [])
    assert calls == [[TEXT]]
    assert not branch._cache_enabled()
    local_model = tmp_path / "local_model"
    local_model.mkdir()
    local, _ = _fake_block(monkeypatch, tmp_path, model_name=str(local_model))
    assert not local._cache_enabled()


@pytest.mark.parametrize("invalid", [
    "corrupt", np.zeros(2, dtype=np.float32), np.zeros((1, 3), dtype=np.float32),
    np.array([1, np.nan, 3], dtype=np.float32),
    np.array([1, np.inf, 3], dtype=np.float32),
    np.ones(3, dtype=np.float64), np.array(["a", "b", "c"], dtype=object),
])
def test_invalid_cache_arrays_are_recomputed_and_repaired(monkeypatch, tmp_path, invalid):
    block, calls = _fake_block(monkeypatch, tmp_path)
    path = block._cache_path(TEXT)
    path.parent.mkdir(parents=True)
    if isinstance(invalid, str):
        path.write_bytes(b"invalid npy payload")
    else:
        np.save(path, invalid)

    result = block.transform([TEXT], []).toarray()

    assert calls == [[TEXT]]
    np.testing.assert_array_equal(result, [[256, 256, 256]])
    stored = np.load(path, allow_pickle=False)
    assert stored.shape == (3,)
    assert stored.dtype == np.float32
    assert np.isfinite(stored).all()


@pytest.mark.parametrize("invalid", [
    np.ones((2, 3)), np.ones((1, 2)), np.array([[1, np.nan, 3]]),
    np.array([[1, np.inf, 3]]), np.array([[1e300, 2, 3]]),
])
def test_invalid_encoder_output_fails_before_cache_write(monkeypatch, tmp_path, invalid):
    block, _ = _fake_block(monkeypatch, tmp_path)
    monkeypatch.setattr(block, "_encode_batch", lambda texts: invalid)

    with pytest.raises(ValueError, match="embedding array"):
        block.transform([TEXT], [])

    assert list(tmp_path.rglob("*.npy")) == []


def test_actual_model_dimension_controls_columns_and_empty_output(monkeypatch, tmp_path):
    block, _ = _fake_block(monkeypatch, tmp_path, dimension=5)

    assert block.feature_names() == [f"emb::{i}" for i in range(5)]
    assert block.transform([TEXT], []).shape == (1, 5)
    assert block.transform([], []).shape == (0, 5)


def test_legacy_sha1_namespace_is_ignored(monkeypatch, tmp_path):
    block, calls = _fake_block(monkeypatch, tmp_path)
    legacy_key = hashlib.sha1(block.model_name.encode() + b"\x00" + TEXT.encode()).hexdigest()
    legacy_path = tmp_path / legacy_key[:2] / f"{legacy_key}.npy"
    legacy_path.parent.mkdir()
    np.save(legacy_path, np.ones(3, dtype=np.float32))

    block.transform([TEXT], [])

    assert calls == [[TEXT]]
    assert legacy_path.exists()
    assert block._cache_path(TEXT).exists()


def test_failed_atomic_write_preserves_existing_cache_and_removes_temp(monkeypatch, tmp_path):
    block, _ = _fake_block(monkeypatch, tmp_path)
    block._store(TEXT, np.ones(3, dtype=np.float32))
    path = block._cache_path(TEXT)

    def fail_replace(source, destination):
        raise OSError("synthetic failed atomic rename")

    monkeypatch.setattr("stylo.features.embeddings.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic failed atomic rename"):
        block._store(TEXT, np.zeros(3, dtype=np.float32))

    np.testing.assert_array_equal(np.load(path, allow_pickle=False), [1, 1, 1])
    assert list(path.parent.iterdir()) == [path]


def test_loader_forwards_explicit_model_and_tokenizer_revisions(monkeypatch, tmp_path):
    loaded = []
    model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=5))
    model.eval = lambda: None
    model.to = lambda device: None

    def load_model(name, **kwargs):
        loaded.append(("model", name, kwargs))
        return model

    def load_tokenizer(name, **kwargs):
        loaded.append(("tokenizer", name, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    ))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=load_tokenizer),
    ))
    block = EmbeddingBlock(
        cache_dir=tmp_path, revision=REVISION, tokenizer_revision="b" * 40,
    )

    assert block.feature_names() == [f"emb::{i}" for i in range(5)]
    assert loaded == [
        ("tokenizer", block.model_name, {"revision": "b" * 40}),
        ("model", block.model_name, {"revision": REVISION}),
    ]


def test_registry_forwards_revisions_and_default_stays_disabled(tmp_path):
    cfg = load_config()
    assert all(block.name != "embeddings" for block in build_blocks(cfg))
    cfg = with_overrides(cfg, {
        "paths.data": str(tmp_path), "features.embeddings.enabled": True,
        "features.embeddings.revision": REVISION,
        "features.embeddings.tokenizer_revision": "b" * 40,
    })

    block = next(block for block in build_blocks(cfg) if block.name == "embeddings")

    assert block.revision == REVISION
    assert block.tokenizer_revision == "b" * 40


def test_legacy_pickled_state_is_uncached_and_gets_actual_dimension():
    state = {
        "model_name": "synthetic/model", "batch_size": 16, "max_length": 256,
        "cache_dir": "unused", "_model": types.SimpleNamespace(
            config=types.SimpleNamespace(hidden_size=5),
        ),
    }
    block = EmbeddingBlock.__new__(EmbeddingBlock)
    block.__setstate__(state)

    assert not block._cache_enabled()
    assert len(block.feature_names()) == 5
