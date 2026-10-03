from stylo.config import load_config, parse_set_overrides
import pytest


def test_load_default():
    cfg = load_config()
    assert cfg.chunking.chunk_size == 500
    assert cfg.get_path("features.char_ngrams.max_features") == 5000
    assert cfg.get_path("nope.missing", "x") == "x"


def test_overrides_and_coercion():
    cfg = load_config(overrides={"features.char_ngrams.bleach": False,
                                 "chunking.chunk_size": 300})
    assert cfg.get_path("features.char_ngrams.bleach") is False
    assert cfg.chunking.chunk_size == 300


def test_parse_set_overrides_types():
    ov = parse_set_overrides(["a.b=true", "c=10", "d=1.5", "e=hello"])
    from stylo.config import load_config
    cfg = load_config(overrides=ov)
    assert cfg.get_path("a.b") is True
    assert cfg.get_path("c") == 10
    assert cfg.get_path("d") == 1.5
    assert cfg.get_path("e") == "hello"


def test_partial_yaml_inherits_defaults_and_cli_override_wins(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("seed: 7\nfeatures:\n  char_ngrams:\n    max_features: 123\n"
                    "deployment:\n  candidate_authors: [alpha, beta]\n", encoding="utf-8")
    cfg = load_config(path, overrides={"features.char_ngrams.max_features": "321"})
    assert cfg.seed == 7
    assert cfg.features.char_ngrams.max_features == 321
    assert cfg.features.char_ngrams.bleach == load_config().features.char_ngrams.bleach
    assert cfg.chunking.chunk_size == 500
    assert cfg.deployment.candidate_authors == ["alpha", "beta"]


def test_partial_yaml_replaces_lists_and_preserves_explicit_null(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("delta:\n  mfw_sizes: [100]\nmodel:\n  classifier:\n    class_weight: null\n",
                    encoding="utf-8")
    cfg = load_config(path)
    assert cfg.delta.mfw_sizes == [100]
    assert cfg.model.classifier.class_weight is None
    assert cfg.model.classifier.max_iter == load_config().model.classifier.max_iter
    assert load_config().delta.mfw_sizes != [100]


def test_mapping_can_overlay_a_null_default_without_mutating_it(tmp_path, monkeypatch):
    from stylo import config
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text("optional: null\nvalues: [1, 2]\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", defaults)
    supplied = tmp_path / "supplied.yaml"
    supplied.write_text("optional:\n  threshold: 0.5\nvalues: []\n", encoding="utf-8")
    assert load_config(supplied).optional.threshold == 0.5
    assert load_config(supplied)["values"] == []
    assert load_config().optional is None
    assert load_config()["values"] == [1, 2]


@pytest.mark.parametrize("content", ["", "null", "[]", "42", "1: invalid-key"])
def test_invalid_yaml_config_shapes_fail(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_partial_yaml_does_not_hide_scalar_override_conflicts(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("features: null\n", encoding="utf-8")
    assert load_config(path).features is None
    with pytest.raises(ValueError, match="conflicts with scalar"):
        load_config(path, overrides={"features.char_ngrams.enabled": True})
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
