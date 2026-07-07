"""Загрузка spaCy и ДИСКОВЫЙ КЕШ разобранных документов (DocBin).

Почему это критично: честный leakage-free LOBO переобучает векторизатор на каждом
из ~100+ фолдов, и без кеша spaCy пришлось бы заново разбирать ОДНИ И ТЕ ЖЕ чанки
в каждом форке.

Решение: один раз разобрать все чанки корпуса, сохранить Doc (POS/lemma/dep/morph/
границы предложений) на диск через DocBin. Все фичи читают готовые Doc из кеша,
а внутри LOBO spaCy не вызывается вовсе.

Ключ кеша = sha1(text + model + version), шардинг по первым 2 символам хэша.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pathlib
from typing import Dict, List, Optional, Sequence

import spacy
from spacy.tokens import Doc, DocBin

log = logging.getLogger("stylo.nlp")

# Атрибуты, которые сохраняем в DocBin (достаточно для всех стилометрических фич).
_DOCBIN_ATTRS = ["ORTH", "SPACY", "LEMMA", "POS", "TAG", "MORPH", "DEP", "HEAD", "SENT_START"]

# Компоненты, не нужные для стилометрии (NER замаскирован на этапе clean_text).
_DISABLE = ["ner"]

_NLP_CACHE: Dict[str, "spacy.Language"] = {}

# Процесс-глобальный кеш РАЗОБРАННЫХ Doc (после десериализации из DocBin).
# Ключ — text-key. Устраняет повторную десериализацию одних и тех же чанков
# при многократных transform внутри LOBO/sweep (главный фактор скорости eval).
_MEM_DOCS: Dict[str, Doc] = {}
_MEM_CAP = 60_000  # мягкий предел числа Doc в памяти на процесс


def load_nlp(model: str, fallback: Optional[str] = None, max_length: int = 5_000_000):
    """Загрузить (и закешировать в процессе) spaCy-модель для стилометрии.

    Оставляет tagger/morphologizer/parser/lemmatizer (нужны для POS/dep/morph/sents),
    отключает NER. Падает на fallback-модель, если основная не установлена.
    """
    if model in _NLP_CACHE:
        return _NLP_CACHE[model]
    try:
        nlp = spacy.load(model, disable=_DISABLE)
    except OSError:
        if fallback:
            log.warning("Модель %s не найдена, использую fallback %s", model, fallback)
            nlp = spacy.load(fallback, disable=_DISABLE)
        else:
            raise
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names \
            and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = max_length
    _NLP_CACHE[model] = nlp
    log.info("spaCy %s загружена (pipes: %s)", nlp.meta.get("name"), nlp.pipe_names)
    return nlp


def load_sentencizer(lang: str = "ru"):
    """Лёгкий пайплайн только для сегментации предложений (нарезка корпуса).

    Использует blank-модель + rule-based sentencizer — быстро и без тяжёлой lg.
    """
    key = f"__sent__{lang}"
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    nlp.max_length = 5_000_000
    _NLP_CACHE[key] = nlp
    return nlp


def load_ner(model: str, fallback: Optional[str] = None):
    """Модель только с NER (для маскировки имён на этапе очистки)."""
    key = f"__ner__{model}"
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]
    disable = ["parser", "tagger", "attribute_ruler", "lemmatizer",
               "morphologizer", "sentencizer", "textcat"]
    try:
        nlp = spacy.load(model, disable=disable)
    except OSError:
        if fallback:
            nlp = spacy.load(fallback, disable=disable)
        else:
            raise
    nlp.max_length = 5_000_000
    _NLP_CACHE[key] = nlp
    return nlp


def _text_key(text: str, model: str, version: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(version.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class DocCache:
    """Дисковый кеш разобранных spaCy Doc.

    get_docs(texts) возвращает Doc в порядке texts; промахи разбираются батчем и
    дописываются в кеш. Запись атомарная (через tmp+rename) — безопасно при
    последовательном прогреве перед параллельным LOBO (в фолдах только чтение).
    """

    def __init__(self, cache_dir: pathlib.Path | str, model: str, version: str,
                 fallback: Optional[str] = None, lang: str = "ru"):
        self.cache_dir = pathlib.Path(cache_dir)
        self.model = model
        self.version = version
        self.fallback = fallback
        self.lang = lang
        self._nlp = None        # полная модель — только для разбора промахов
        self._recon_vocab = None  # лёгкий vocab для реконструкции из DocBin

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = load_nlp(self.model, self.fallback)
        return self._nlp

    @property
    def recon_vocab(self):
        """Лёгкий blank-vocab для чтения DocBin (без загрузки тяжёлой lg-модели).

        DocBin несёт собственный StringStore, поэтому pos_/dep_/lemma_/morph и границы
        предложений восстанавливаются корректно даже на пустом vocab. Это позволяет
        параллельным воркерам LOBO читать кеш, не загружая ru_core_news_lg.
        """
        if self._recon_vocab is None:
            self._recon_vocab = spacy.blank(self.lang).vocab
        return self._recon_vocab

    def _path_for(self, key: str) -> pathlib.Path:
        return self.cache_dir / self.model / key[:2] / f"{key}.spacy"

    def _load_one(self, key: str) -> Optional[Doc]:
        p = self._path_for(key)
        if not p.exists():
            return None
        try:
            db = DocBin().from_disk(p)
            docs = list(db.get_docs(self.recon_vocab))
            return docs[0] if docs else None
        except Exception as exc:  # pragma: no cover - повреждённый кеш
            log.warning("Битый кеш %s (%s) — переразбор", p, exc)
            return None

    def _store_one(self, key: str, doc: Doc) -> None:
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        db = DocBin(attrs=_DOCBIN_ATTRS, store_user_data=False)
        db.add(doc)
        tmp = p.with_suffix(".spacy.tmp")
        db.to_disk(tmp)
        os.replace(tmp, p)

    def get_docs(self, texts: Sequence[str], batch_size: int = 32) -> List[Doc]:
        """Вернуть Doc для каждого текста (из кеша или разобрав промахи)."""
        keys = [_text_key(t, self.model, self.version) for t in texts]
        result: List[Optional[Doc]] = [None] * len(texts)

        disk_idx: List[int] = []
        for i, key in enumerate(keys):
            doc = _MEM_DOCS.get(key)          # 1) память
            if doc is not None:
                result[i] = doc
            else:
                disk_idx.append(i)

        miss_idx: List[int] = []
        for i in disk_idx:
            doc = self._load_one(keys[i])     # 2) диск (DocBin)
            if doc is None:
                miss_idx.append(i)
            else:
                result[i] = doc
                if len(_MEM_DOCS) < _MEM_CAP:
                    _MEM_DOCS[keys[i]] = doc

        if miss_idx:                          # 3) разбор spaCy
            miss_texts = [texts[i] for i in miss_idx]
            log.info("DocCache: %d/%d промахов — разбираю spaCy", len(miss_idx), len(texts))
            for j, doc in zip(miss_idx, self.nlp.pipe(miss_texts, batch_size=batch_size)):
                self._store_one(keys[j], doc)
                result[j] = doc
                if len(_MEM_DOCS) < _MEM_CAP:
                    _MEM_DOCS[keys[j]] = doc

        return [d for d in result]  # type: ignore[return-value]

    def warm(self, texts: Sequence[str], batch_size: int = 32, n_process: int = 1) -> int:
        """Прогреть кеш для всего корпуса. Возвращает число вновь разобранных.

        n_process>1 включает многопроцессный разбор spaCy (важно для ~12k чанков:
        одно-процессный lg-разбор всего корпуса занимает ~час, на 4-8 ядрах — кратно
        быстрее). Память: каждый воркер держит копию lg-модели (~0.6 ГБ).
        """
        keys = [_text_key(t, self.model, self.version) for t in texts]
        miss = [(k, t) for k, t in zip(keys, texts) if not self._path_for(k).exists()]
        if not miss:
            log.info("DocCache.warm: всё уже в кеше (%d текстов)", len(texts))
            return 0
        log.info("DocCache.warm: разбираю %d/%d новых текстов (n_process=%d)…",
                 len(miss), len(texts), n_process)
        miss_keys = [k for k, _ in miss]
        miss_texts = [t for _, t in miss]
        n = 0
        pipe = self.nlp.pipe(miss_texts, batch_size=batch_size,
                             n_process=n_process if n_process and n_process > 1 else 1)
        for key, doc in zip(miss_keys, pipe):
            self._store_one(key, doc)
            n += 1
            if n % 500 == 0:
                log.info("  …%d/%d", n, len(miss))
        log.info("DocCache.warm: готово, разобрано %d", n)
        return n


def make_doc_cache(cfg) -> DocCache:
    """Собрать DocCache из конфига (stylo.config.ConfigNode)."""
    return DocCache(
        cache_dir=cfg.get_path("paths.doc_cache", "data/doc_cache"),
        model=cfg.get_path("language.spacy_model", "ru_core_news_lg"),
        version=str(cfg.get_path("language.spacy_model_version", "0")),
        fallback=cfg.get_path("language.spacy_fallback", None),
        lang=cfg.get_path("language.code", "ru"),
    )
