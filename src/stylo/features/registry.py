"""Реестр фич-блоков: имя -> конструктор из конфига.

build_blocks(cfg) собирает список включённых блоков по configs/default.yaml.
enabled_override позволяет sweep'у форсить вкл/выкл блоков, не трогая YAML.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import FeatureBlock
from .char_ngrams import CharNgramBlock
from .dependency import DependencyBlock
from .embeddings import EmbeddingBlock
from .function_words import FunctionWordBlock
from .length_dist import LengthDistBlock
from .morphology import MorphologyBlock
from .pos_ngrams import PosNgramBlock
from .punctuation import PunctNgramBlock
from .syntax import SyntaxBlock

# Порядок определяет порядок столбцов в итоговом векторе.
BLOCK_ORDER = [
    "char_ngrams", "function_words", "syntax", "pos_ngrams",
    "punctuation_ngrams", "dependency", "morphology", "length_dist", "embeddings",
]


def _build_one(name: str, fb, cfg, topic_strict: bool = False,
               relative_fw: bool | None = None) -> Optional[FeatureBlock]:
    """fb — ConfigNode для features.<name>; cfg — корневой конфиг.

    topic_strict: строгий topic-control — function_words=fixed_list
    (закрытый класс, без частотных СОДЕРЖАТЕЛЬНЫХ слов) + глушение жанро-несущих субблоков
    синтаксиса (pos_ratios, lexical_richness). Применяется к тематически-инвариантному набору.

    relative_fw: R-axis policy for the FunctionWord block only (B4-B increment 3). ``None`` keeps the
    legacy corner coupling (byte-exact A0/A4); an explicit bool selects the A2 (raw) / A3 (relative)
    transform independently of F. Every other block ignores R.
    """
    lang = cfg.get_path("language.code", "ru")
    if name == "char_ngrams":
        return CharNgramBlock(
            ngram_range=list(fb.get("ngram_range", [3, 5])),
            max_features=fb.get("max_features", 5000),
            min_df=fb.get("min_df", 3),
            sublinear_tf=fb.get("sublinear_tf", True),
            bleach=fb.get("bleach", True),
            pos_replacements=cfg.get_path("language.pos_bleach").to_dict(),
        )
    if name == "function_words":
        mode = "fixed_list" if topic_strict else fb.get("mode", "mfw")
        return FunctionWordBlock(
            mode=mode,
            mfw_count=fb.get("mfw_count", 300),
            lang=lang,
            relative_fw=relative_fw,
        )
    if name == "syntax":
        sub = fb.get_path("subblocks")
        sub_d = sub.to_dict() if sub is not None else None
        if topic_strict:                       # глушим жанро-несущие субблоки
            sub_d = dict(sub_d or {})
            sub_d["pos_ratios"] = False
            sub_d["lexical_richness"] = False
        return SyntaxBlock(
            subblocks=sub_d,
            vowels_hard=cfg.get_path("language.vowels_hard", "аоуэы"),
            vowels_soft=cfg.get_path("language.vowels_soft", "иеяёю"),
        )
    if name == "pos_ngrams":
        return PosNgramBlock(
            ngram_range=list(fb.get("ngram_range", [2, 4])),
            max_features=fb.get("max_features", 2000),
            min_df=fb.get("min_df", 3),
        )
    if name == "punctuation_ngrams":
        return PunctNgramBlock(
            ngram_range=list(fb.get("ngram_range", [1, 3])),
            max_features=fb.get("max_features", 500),
        )
    if name == "dependency":
        return DependencyBlock()
    if name == "morphology":
        return MorphologyBlock()
    if name == "length_dist":
        return LengthDistBlock(max_word_len=fb.get("max_word_len", 16))
    if name == "embeddings":
        return EmbeddingBlock(
            model_name=fb.get("model_name", "ai-forever/ruBert-base"),
            batch_size=fb.get("batch_size", 16),
            max_length=fb.get("max_length", 256),
            cache_dir=cfg.get_path("paths.data", "data") + "/emb_cache",
        )
    raise KeyError(f"Неизвестный блок: {name}")


def build_blocks(cfg, enabled_override: Optional[Dict[str, bool]] = None,
                 topic_strict: bool = False,
                 relative_fw: bool | None = None) -> List[FeatureBlock]:
    override = enabled_override or {}
    blocks: List[FeatureBlock] = []
    feats = cfg.get_path("features")
    for name in BLOCK_ORDER:
        fb = feats.get_path(name) if feats is not None else None
        if fb is None:
            continue
        enabled = override.get(name, fb.get("enabled", False))
        if not enabled:
            continue
        block = _build_one(name, fb, cfg, topic_strict=topic_strict, relative_fw=relative_fw)
        if block is not None:
            blocks.append(block)
    if not blocks:
        raise ValueError("Не включён ни один фич-блок (проверь config.features.*.enabled).")
    return blocks
