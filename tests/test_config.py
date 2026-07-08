from stylo.config import load_config, parse_set_overrides


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
