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
import math
import os
import pathlib
import sqlite3
import threading
from collections import Counter
from typing import Dict, List, Optional, Sequence

from spacy.tokens import Doc

from ..jsonio import dumps_strict, loads_strict
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
_INITIALISED_FILES: set[str] = set()
_REP_MEMORY_LOCK = threading.RLock()
_REP_CACHE_SCHEMA = "stylo.rep-cache.v3.resolved-nlp.sqlite-json"


def _key(text: str, nlp_identity: str, version: str, rep_ver: str) -> str:
    h = hashlib.sha1()
    for part in (nlp_identity, version, rep_ver, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class RepCache:
    """Safe SQLite/strict-JSON representation cache.

    warm(): строит недостающие Rep (через DocCache+spaCy) и пишет файл.
    get_reps(): reads only requested rows; missing representations are rebuilt.

    The former pickle format is intentionally not read: a cache is untrusted,
    regenerable input and executable deserialisation is not an acceptable
    runtime boundary.
    """

    def __init__(self, doc_cache, data_dir: pathlib.Path | str, params: RepParams):
        self.doc_cache = doc_cache
        self.data_dir = pathlib.Path(data_dir)
        self.params = params
        self.rep_ver = params.version()
        self.path = (
            self.data_dir
            / (
                f"reps_{doc_cache.identity.identity_sha256}_"
                f"{doc_cache.version}_{self.rep_ver}.sqlite3"
            )
        )

    def _metadata(self) -> dict[str, str]:
        return {
            "schema": _REP_CACHE_SCHEMA,
            "requested_model": str(self.doc_cache.model),
            "configured_version": str(self.doc_cache.version),
            "nlp_identity": dumps_strict(
                self.doc_cache.identity.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "rep_version": self.rep_ver,
        }

    def _connect(self) -> sqlite3.Connection:
        if self.path.is_symlink():
            raise RuntimeError(f"RepCache path must not be a symlink: {self.path}")
        if self.path.exists() and not self.path.is_file():
            raise RuntimeError(f"RepCache path is not a regular file: {self.path}")
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA journal_mode=DELETE")
        return connection

    def _ensure_loaded(self) -> None:
        identity = str(self.path.absolute())
        with _REP_MEMORY_LOCK:
            if identity in _INITIALISED_FILES:
                return
            existed = self.path.exists()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS representations "
                    "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                existing = dict(connection.execute("SELECT key, value FROM metadata"))
                expected = self._metadata()
                if existing and existing != expected:
                    raise RuntimeError(
                        f"RepCache metadata mismatch for {self.path}: "
                        f"expected {expected!r}, got {existing!r}"
                    )
                if not existing:
                    connection.executemany(
                        "INSERT INTO metadata(key, value) VALUES (?, ?)",
                        sorted(expected.items()),
                    )
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
                if not integrity or integrity[0] != "ok":
                    raise RuntimeError(
                        f"RepCache SQLite integrity check failed: {integrity!r}"
                    )
            if not existed:
                dfd = os.open(
                    self.data_dir,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            _INITIALISED_FILES.add(identity)

    def _keys(self, texts: Sequence[str]) -> List[str]:
        return [
            _key(
                t,
                self.doc_cache.identity.identity_sha256,
                self.doc_cache.version,
                self.rep_ver,
            )
            for t in texts
        ]

    @staticmethod
    def _rep_payload(rep: Rep) -> dict:
        return {
            "text": rep.text,
            "bleach": rep.bleach,
            "pos_str": rep.pos_str,
            "punct_str": rep.punct_str,
            "morph": dict(rep.morph),
            "dep_n": rep.dep_n,
            "dep_counts": dict(rep.dep_counts),
            "dep_agg": list(rep.dep_agg),
            "syntax_all": rep.syntax_all,
            "word_len_hist": list(rep.word_len_hist),
            "sent_lens": list(rep.sent_lens),
        }

    @staticmethod
    def _decode_rep(payload: str, *, expected_text: str) -> Rep:
        raw = loads_strict(payload)
        required = {
            "text",
            "bleach",
            "pos_str",
            "punct_str",
            "morph",
            "dep_n",
            "dep_counts",
            "dep_agg",
            "syntax_all",
            "word_len_hist",
            "sent_lens",
        }
        if type(raw) is not dict or set(raw) != required:
            raise ValueError("RepCache payload has an unexpected schema")
        if raw["text"] != expected_text:
            raise ValueError("RepCache payload text does not match its requested key")
        for key in ("text", "bleach", "pos_str", "punct_str"):
            if type(raw[key]) is not str:
                raise ValueError(f"RepCache {key} must be a string")
        for key in ("morph", "dep_counts"):
            value = raw[key]
            if type(value) is not dict or any(
                type(k) is not str or type(v) is not int or v < 0
                for k, v in value.items()
            ):
                raise ValueError(f"RepCache {key} must be string->nonnegative-int")
        if type(raw["dep_n"]) is not int or raw["dep_n"] < 0:
            raise ValueError("RepCache dep_n must be a nonnegative exact integer")
        for key in ("dep_agg", "word_len_hist", "sent_lens"):
            if type(raw[key]) is not list:
                raise ValueError(f"RepCache {key} must be a list")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw["dep_agg"]
        ):
            raise ValueError("RepCache dep_agg must contain finite numbers")
        for key in ("word_len_hist", "sent_lens"):
            if any(type(value) is not int or value < 0 for value in raw[key]):
                raise ValueError(f"RepCache {key} must contain nonnegative exact integers")
        syntax = raw["syntax_all"]
        if type(syntax) is not dict or any(
            type(name) is not str
            or type(values) is not list
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
            for name, values in syntax.items()
        ):
            raise ValueError("RepCache syntax_all has invalid values")
        return Rep(
            text=raw["text"],
            bleach=raw["bleach"],
            pos_str=raw["pos_str"],
            punct_str=raw["punct_str"],
            morph=Counter(raw["morph"]),
            dep_n=raw["dep_n"],
            dep_counts=Counter(raw["dep_counts"]),
            dep_agg=[float(v) for v in raw["dep_agg"]],
            syntax_all={
                name: [float(value) for value in values]
                for name, values in syntax.items()
            },
            word_len_hist=list(raw["word_len_hist"]),
            sent_lens=list(raw["sent_lens"]),
        )

    def _load_requested(self, keys: Sequence[str], texts: Sequence[str]) -> None:
        with _REP_MEMORY_LOCK:
            requested = [
                (key, text)
                for key, text in zip(keys, texts, strict=True)
                if key not in _MEM_REPS
            ]
        if not requested:
            return
        with self._connect() as connection:
            for key, text in requested:
                row = connection.execute(
                    "SELECT payload FROM representations WHERE cache_key = ?", (key,)
                ).fetchone()
                if row is None:
                    continue
                try:
                    decoded = self._decode_rep(row[0], expected_text=text)
                    with _REP_MEMORY_LOCK:
                        _MEM_REPS.setdefault(key, decoded)
                except Exception as exc:
                    # A corrupt cache is never silently used.  Delete only the
                    # bad regenerable row and make the caller rebuild it.
                    connection.execute(
                        "DELETE FROM representations WHERE cache_key = ?", (key,)
                    )
                    log.warning("RepCache: rejected corrupt row %s (%s)", key, exc)

    def warm(self, texts: Sequence[str], n_process: int = 4, batch_size: int = 32) -> int:
        self._ensure_loaded()
        keys = self._keys(texts)
        self._load_requested(keys, texts)
        with _REP_MEMORY_LOCK:
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
            built = build_rep(t, d, self.params, _synt=synt)
            with _REP_MEMORY_LOCK:
                _MEM_REPS.setdefault(k, built)
            n += 1
        self._save(miss_keys)
        log.info("RepCache.warm: построено %d", n)
        return n

    def _save(self, keys: Sequence[str]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with _REP_MEMORY_LOCK:
            rows = [
                (
                    key,
                    dumps_strict(
                        self._rep_payload(_MEM_REPS[key]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                for key in keys
            ]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for key, payload in rows:
                connection.execute(
                    "INSERT OR IGNORE INTO representations(cache_key, payload) "
                    "VALUES (?, ?)",
                    (key, payload),
                )
                existing = connection.execute(
                    "SELECT payload FROM representations WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if existing is None or existing[0] != payload:
                    raise RuntimeError(
                        f"RepCache immutable content conflict for key {key}"
                    )

    def get_reps(self, texts: Sequence[str]) -> List[Rep]:
        self._ensure_loaded()
        keys = self._keys(texts)
        self._load_requested(keys, texts)
        with _REP_MEMORY_LOCK:
            missing = [
                (i, k, texts[i])
                for i, k in enumerate(keys)
                if k not in _MEM_REPS
            ]
        if missing:
            # достроить на лету (напр. unknown-тексты при predict)
            synt = SyntaxBlock(vowels_hard=self.params.vowels_hard, vowels_soft=self.params.vowels_soft)
            mtexts = [t for _, _, t in missing]
            docs = self.doc_cache.get_docs(mtexts)
            for (i, k, t), d in zip(missing, docs):
                built = build_rep(t, d, self.params, _synt=synt)
                with _REP_MEMORY_LOCK:
                    _MEM_REPS.setdefault(k, built)
            self._save([key for _index, key, _text in missing])
        with _REP_MEMORY_LOCK:
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
