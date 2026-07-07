"""Лёгкие предвычисленные представления документов (Rep) — ядро масштабируемости.

Проблема: в leakage-free LOBO/sweep блоки фич многократно проходят по spaCy Doc
(десериализация DocBin из множества файлов в каждом воркере → I/O-bound, часы на
полном корпусе).

Решение: один раз посчитать по каждому чанку фолд-НЕЗАВИСИМые представления
(bleached-строка, POS/пунктуационные строки, счётчики morph/dep, синтаксис, длины)
и хранить их в ОДНОМ файле. Каждый процесс грузит его один раз в память; блоки
работают с Rep (строки/маленькие dict), а не с тяжёлыми Doc. spaCy в eval не нужен.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import pathlib
import pickle
from collections import Counter
from typing import Dict, List, Optional, Sequence

from spacy.tokens import Doc

from .char_ngrams import bleach_doc
from .pos_ngrams import _pos_string_raw
from .punctuation import _punct_string_raw
from .morphology import _morph_items_raw
from .dependency import DependencyBlock
from .syntax import SyntaxBlock

log = logging.getLogger("stylo.features.reps")


@dataclasses.dataclass
class RepParams:
    pos_replacements: Dict[str, str]
    vowels_hard: str
    vowels_soft: str
    max_word_len: int = 16

    def version(self) -> str:
        s = repr(sorted(self.pos_replacements.items())) + self.vowels_hard + \
            self.vowels_soft + str(self.max_word_len)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


@dataclasses.dataclass
class Rep:
    text: str                       # сырой текст (function_words, bow, embeddings)
    bleach: str                     # bleached-строка для char-n-грамм
    pos_str: str
    punct_str: str
    morph: Counter
    dep_n: int
    dep_counts: Counter
    dep_agg: List[float]
    syntax_all: Dict[str, List[float]]
    word_len_hist: List[int]        # гистограмма длин слов (1..max+1)
    sent_lens: List[int]            # длины предложений (в словах)


def build_rep(text: str, doc: Doc, params: RepParams,
              _synt: Optional[SyntaxBlock] = None) -> Rep:
    synt = _synt or SyntaxBlock(vowels_hard=params.vowels_hard, vowels_soft=params.vowels_soft)
    n, dep_counts, dep_agg = DependencyBlock._raw(doc)
    hist = [0] * (params.max_word_len + 1)
    for t in doc:
        if t.is_alpha:
            L = min(len(t.text), params.max_word_len + 1)
            hist[L - 1] += 1
    sent_lens = [sum(1 for t in s if t.is_alpha) for s in doc.sents]
    return Rep(
        text=text,
        bleach=bleach_doc(doc, params.pos_replacements),
        pos_str=_pos_string_raw(doc),
        punct_str=_punct_string_raw(doc),
        morph=_morph_items_raw(doc),
        dep_n=n,
        dep_counts=dep_counts,
        dep_agg=dep_agg,
        syntax_all=synt._compute_all(doc),
        word_len_hist=hist,
        sent_lens=sent_lens,
    )


# Процесс-глобальный кеш Rep (грузится один раз из единого файла).
_MEM_REPS: Dict[str, Rep] = {}
_LOADED_FILES: set = set()


def _key(text: str, model: str, version: str, rep_ver: str) -> str:
    h = hashlib.sha1()
    for part in (model, version, rep_ver, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class RepCache:
    """Единый файл представлений: data/reps_<repver>.pkl = {key: Rep}.

    warm(): строит недостающие Rep (через DocCache+spaCy) и пишет файл.
    get_reps(): грузит файл один раз в _MEM_REPS, недостающие строит на лету.
    """

    def __init__(self, doc_cache, data_dir: pathlib.Path | str, params: RepParams):
        self.doc_cache = doc_cache
        self.data_dir = pathlib.Path(data_dir)
        self.params = params
        self.rep_ver = params.version()
        self.path = self.data_dir / f"reps_{doc_cache.model}_{doc_cache.version}_{self.rep_ver}.pkl"

    def _ensure_loaded(self) -> None:
        if str(self.path) in _LOADED_FILES:
            return
        if self.path.exists():
            try:
                with open(self.path, "rb") as fh:
                    data = pickle.load(fh)
                _MEM_REPS.update(data)
                log.info("RepCache: загружено %d представлений из %s", len(data), self.path.name)
            except Exception as exc:  # pragma: no cover
                log.warning("RepCache: не удалось загрузить %s (%s)", self.path, exc)
        _LOADED_FILES.add(str(self.path))

    def _keys(self, texts: Sequence[str]) -> List[str]:
        return [_key(t, self.doc_cache.model, self.doc_cache.version, self.rep_ver) for t in texts]

    def warm(self, texts: Sequence[str], n_process: int = 4, batch_size: int = 32) -> int:
        self._ensure_loaded()
        keys = self._keys(texts)
        miss = [(k, t) for k, t in zip(keys, texts) if k not in _MEM_REPS]
        if not miss:
            log.info("RepCache.warm: всё на месте (%d)", len(texts))
            return 0
        log.info("RepCache.warm: строю %d/%d представлений…", len(miss), len(texts))
        miss_texts = [t for _, t in miss]
        miss_keys = [k for k, _ in miss]
        # разбираем недостающие через DocCache (он сам кеширует DocBin)
        synt = SyntaxBlock(vowels_hard=self.params.vowels_hard, vowels_soft=self.params.vowels_soft)
        n = 0
        # построчно, но spaCy-разбор внутри DocCache.get_docs уже многопроцессный через warm
        self.doc_cache.warm(miss_texts, n_process=n_process, batch_size=batch_size)
        docs = self.doc_cache.get_docs(miss_texts, batch_size=batch_size)
        for k, t, d in zip(miss_keys, miss_texts, docs):
            _MEM_REPS[k] = build_rep(t, d, self.params, _synt=synt)
            n += 1
        self._save()
        log.info("RepCache.warm: построено %d", n)
        return n

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # сохраняем только наши ключи (весь _MEM_REPS относится к этому rep_ver/модели)
        tmp = self.path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(_MEM_REPS, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)

    def get_reps(self, texts: Sequence[str]) -> List[Rep]:
        self._ensure_loaded()
        keys = self._keys(texts)
        missing = [(i, k, texts[i]) for i, k in enumerate(keys) if k not in _MEM_REPS]
        if missing:
            # достроить на лету (напр. unknown-тексты при predict)
            synt = SyntaxBlock(vowels_hard=self.params.vowels_hard, vowels_soft=self.params.vowels_soft)
            mtexts = [t for _, _, t in missing]
            docs = self.doc_cache.get_docs(mtexts)
            for (i, k, t), d in zip(missing, docs):
                _MEM_REPS[k] = build_rep(t, d, self.params, _synt=synt)
        return [_MEM_REPS[k] for k in keys]


def make_rep_cache(cfg, doc_cache=None) -> RepCache:
    from ..nlp import make_doc_cache
    dc = doc_cache or make_doc_cache(cfg)
    params = RepParams(
        pos_replacements=cfg.get_path("language.pos_bleach").to_dict(),
        vowels_hard=cfg.get_path("language.vowels_hard", "аоуэы"),
        vowels_soft=cfg.get_path("language.vowels_soft", "иеяёю"),
        max_word_len=cfg.get_path("features.length_dist.max_word_len", 16),
    )
    return RepCache(dc, cfg.get_path("paths.data", "data"), params)
