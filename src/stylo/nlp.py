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

import base64
import csv
import dataclasses
import hashlib
import importlib.metadata
import io
import logging
import os
import pathlib
import tempfile
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import spacy
from spacy.tokens import Doc, DocBin

from .jsonio import dumps_strict

try:  # Linux is the canonical bound-run platform; keep import portable.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback is process-local only
    fcntl = None

log = logging.getLogger("stylo.nlp")

# Атрибуты, которые сохраняем в DocBin (достаточно для всех стилометрических фич).
_DOCBIN_ATTRS = ["ORTH", "SPACY", "LEMMA", "POS", "TAG", "MORPH", "DEP", "HEAD", "SENT_START"]

# Компоненты, не нужные для стилометрии (NER замаскирован на этапе clean_text).
_DISABLE = ["ner"]

@dataclasses.dataclass(frozen=True)
class ResolvedNLPIdentity:
    """Identity of the pipeline that actually produced cached annotations."""

    requested_model: str
    resolved_model: str
    fallback_used: bool
    package_version: str
    package_record_sha256: str
    spacy_version: str
    disabled_pipes: tuple[str, ...]
    active_pipes: tuple[str, ...]
    max_length: int
    identity_sha256: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_NLP_CACHE: Dict[tuple, "spacy.Language"] = {}
_NLP_IDENTITIES: Dict[int, ResolvedNLPIdentity] = {}

# Процесс-глобальный кеш РАЗОБРАННЫХ Doc (после десериализации из DocBin).
# Ключ — text-key. Устраняет повторную десериализацию одних и тех же чанков
# при многократных transform внутри LOBO/sweep (главный фактор скорости eval).
_MEM_DOCS: Dict[str, Doc] = {}
_MEM_CAP = 60_000  # мягкий предел числа Doc в памяти на процесс
_MEM_DOCS_LOCK = threading.RLock()
_DOC_WRITE_LOCK = threading.RLock()


def _fsync_directory(path: pathlib.Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _exclusive_cache_key(lock_path: pathlib.Path):
    """Serialize one cache-key publication across threads and POSIX processes."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    with _DOC_WRITE_LOCK:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | nofollow, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _mem_doc_get(key: str) -> Optional[Doc]:
    with _MEM_DOCS_LOCK:
        return _MEM_DOCS.get(key)


def _mem_doc_put(key: str, doc: Doc) -> None:
    with _MEM_DOCS_LOCK:
        if len(_MEM_DOCS) < _MEM_CAP:
            _MEM_DOCS[key] = doc


def verified_installed_package_record(name: str) -> tuple[str, str]:
    """Verify every hashed wheel ``RECORD`` member and return its identity.

    Hashing the ``RECORD`` text alone only identifies what an installer claimed
    to install.  Scientific attestations need the stronger statement that the
    files currently on disk still match those recorded hashes and sizes.
    """

    distribution = importlib.metadata.distribution(name)
    record = distribution.read_text("RECORD")
    if not record:
        raise RuntimeError(
            f"installed spaCy model {name!r} has no wheel RECORD"
        )

    root = pathlib.Path(distribution.locate_file("")).resolve()
    seen: set[str] = set()
    verified = 0
    try:
        rows = csv.reader(io.StringIO(record), strict=True)
        for row_number, row in enumerate(rows, start=1):
            if len(row) != 3:
                raise RuntimeError(
                    f"installed spaCy model {name!r} has malformed RECORD "
                    f"row {row_number}"
                )
            member, digest_spec, size_text = row
            member_path = pathlib.PurePosixPath(member)
            if (
                not member
                or member in seen
                or member_path.is_absolute()
                or any(part in {"", ".", ".."} for part in member_path.parts)
            ):
                raise RuntimeError(
                    f"installed spaCy model {name!r} has unsafe/duplicate "
                    f"RECORD member {member!r}"
                )
            seen.add(member)

            # Wheel installers may add unhashed RECORD/bytecode rows.  Model
            # source, configuration and weight payloads are required to carry
            # hashes and are verified below.
            if not digest_spec:
                continue
            algorithm, separator, expected_digest = digest_spec.partition("=")
            if (
                separator != "="
                or algorithm != "sha256"
                or not expected_digest
                or not size_text.isascii()
                or not size_text.isdecimal()
            ):
                raise RuntimeError(
                    f"installed spaCy model {name!r} has unsupported RECORD "
                    f"identity for {member!r}"
                )

            path = pathlib.Path(distribution.locate_file(member))
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"installed spaCy model {name!r} RECORD member is "
                    f"missing/unsafe: {member}"
                )
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise RuntimeError(
                    f"installed spaCy model {name!r} RECORD member escapes "
                    f"the environment: {member}"
                )

            digest = hashlib.sha256()
            observed_size = 0
            with resolved.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
                    observed_size += len(block)
            observed_digest = (
                base64.urlsafe_b64encode(digest.digest())
                .rstrip(b"=")
                .decode("ascii")
            )
            if (
                observed_size != int(size_text)
                or observed_digest != expected_digest
            ):
                raise RuntimeError(
                    f"installed spaCy model {name!r} RECORD mismatch: {member}"
                )
            verified += 1
    except csv.Error as exc:
        raise RuntimeError(
            f"installed spaCy model {name!r} has invalid wheel RECORD"
        ) from exc

    if verified == 0:
        raise RuntimeError(
            f"installed spaCy model {name!r} RECORD has no verified members"
        )
    return (
        str(distribution.version),
        hashlib.sha256(record.encode("utf-8")).hexdigest(),
    )


def _installed_package_identity(name: str, nlp) -> tuple[str, str]:
    """Return installed version and a verified package RECORD/metadata digest.

    Real wheel installations are verified against every recorded file hash.
    Test-only/fake packages may not exist as distributions; only that absent
    distribution case falls back to hashing the exact live spaCy metadata.
    """

    try:
        return verified_installed_package_record(name)
    except importlib.metadata.PackageNotFoundError:
        version = str(nlp.meta.get("version", "unknown"))
        material = dumps_strict(
            nlp.meta,
            sort_keys=True,
            separators=(",", ":"),
        )
        return version, hashlib.sha256(material.encode("utf-8")).hexdigest()


def _build_nlp_identity(
    *,
    requested: str,
    resolved: str,
    nlp,
    max_length: int,
) -> ResolvedNLPIdentity:
    version, record_sha = _installed_package_identity(resolved, nlp)
    payload = {
        "requested_model": requested,
        "resolved_model": resolved,
        "fallback_used": resolved != requested,
        "package_version": version,
        "package_record_sha256": record_sha,
        "spacy_version": spacy.__version__,
        "disabled_pipes": sorted(_DISABLE),
        "active_pipes": list(nlp.pipe_names),
        "max_length": max_length,
    }
    digest = hashlib.sha256(
        dumps_strict(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResolvedNLPIdentity(
        requested_model=requested,
        resolved_model=resolved,
        fallback_used=resolved != requested,
        package_version=version,
        package_record_sha256=record_sha,
        spacy_version=spacy.__version__,
        disabled_pipes=tuple(sorted(_DISABLE)),
        active_pipes=tuple(nlp.pipe_names),
        max_length=max_length,
        identity_sha256=digest,
    )


def resolved_nlp_identity(nlp) -> ResolvedNLPIdentity:
    try:
        return _NLP_IDENTITIES[id(nlp)]
    except KeyError as exc:
        raise RuntimeError("spaCy pipeline was not loaded through stylo.load_nlp") from exc


def load_nlp(model: str, fallback: Optional[str] = None, max_length: int = 5_000_000):
    """Загрузить (и закешировать в процессе) spaCy-модель для стилометрии.

    Оставляет tagger/morphologizer/parser/lemmatizer (нужны для POS/dep/morph/sents),
    отключает NER. Падает на fallback-модель, если основная не установлена.
    """
    if type(model) is not str or not model:
        raise ValueError("spaCy model must be a nonempty string")
    if fallback is not None and (type(fallback) is not str or not fallback):
        raise ValueError("spaCy fallback must be None or a nonempty string")
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("spaCy max_length must be a positive exact integer")
    cache_key = ("full", model, fallback, max_length, tuple(_DISABLE))
    if cache_key in _NLP_CACHE:
        return _NLP_CACHE[cache_key]
    resolved = model
    try:
        nlp = spacy.load(model, disable=_DISABLE)
    except OSError:
        if fallback:
            log.warning("Модель %s не найдена, использую fallback %s", model, fallback)
            nlp = spacy.load(fallback, disable=_DISABLE)
            resolved = fallback
        else:
            raise
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names \
            and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = max_length
    identity = _build_nlp_identity(
        requested=model, resolved=resolved, nlp=nlp, max_length=max_length
    )
    _NLP_CACHE[cache_key] = nlp
    _NLP_IDENTITIES[id(nlp)] = identity
    log.info(
        "spaCy %s загружена как %s (identity=%s, pipes: %s)",
        model,
        resolved,
        identity.identity_sha256[:12],
        nlp.pipe_names,
    )
    return nlp


def load_sentencizer(lang: str = "ru"):
    """Лёгкий пайплайн только для сегментации предложений (нарезка корпуса).

    Использует blank-модель + rule-based sentencizer — быстро и без тяжёлой lg.
    """
    key = ("sentencizer", lang, spacy.__version__, 5_000_000)
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    nlp.max_length = 5_000_000
    _NLP_CACHE[key] = nlp
    return nlp


def load_ner(model: str, fallback: Optional[str] = None):
    """Модель только с NER (для маскировки имён на этапе очистки)."""
    disable = ["parser", "tagger", "attribute_ruler", "lemmatizer",
               "morphologizer", "sentencizer", "textcat"]
    key = ("ner", model, fallback, 5_000_000, tuple(disable))
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]
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


def _text_key(text: str, identity_sha256: str, version: str) -> str:
    h = hashlib.sha1()
    h.update(identity_sha256.encode("utf-8"))
    h.update(b"\x00")
    h.update(version.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class DocCache:
    """Дисковый кеш разобранных spaCy Doc.

    get_docs(texts) возвращает Doc в порядке texts; промахи разбираются батчем и
    дописываются в кеш. Каждая запись публикуется под per-key lock через
    уникальный same-directory temp + fsync + atomic replace; конкурентный warm
    и авария до replace не повреждают уже опубликованный DocBin.
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
        self._identity: ResolvedNLPIdentity | None = None

    def __getstate__(self):
        """Do not serialize process-local spaCy/vocab objects.

        A restored estimator resolves the runtime pipeline again and compares
        it with the persisted scientific identity before touching cache rows.
        """

        state = dict(self.__dict__)
        state["_nlp"] = None
        state["_recon_vocab"] = None
        return state

    @property
    def nlp(self):
        if self._nlp is None:
            loaded = load_nlp(self.model, self.fallback)
            observed = resolved_nlp_identity(loaded)
            recorded = getattr(self, "_identity", None)
            if recorded is not None and recorded != observed:
                raise RuntimeError(
                    "resolved spaCy identity changed across estimator serialization"
                )
            self._identity = observed
            self._nlp = loaded
        return self._nlp

    @property
    def identity(self) -> ResolvedNLPIdentity:
        _nlp = self.nlp
        identity = getattr(self, "_identity", None)
        if identity is None:
            identity = resolved_nlp_identity(_nlp)
            self._identity = identity
        return identity

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
        return (
            self.cache_dir
            / self.identity.identity_sha256
            / key[:2]
            / f"{key}.spacy"
        )

    def _load_one(self, key: str, expected_text: str) -> Optional[Doc]:
        p = self._path_for(key)
        if not p.exists():
            return None
        if p.is_symlink() or not p.is_file():
            log.warning("Небезопасный кеш %s — переразбор", p)
            return None
        try:
            db = DocBin().from_disk(p)
            docs = list(db.get_docs(self.recon_vocab))
            if len(docs) != 1 or docs[0].text != expected_text:
                raise ValueError("DocBin payload text/count does not match its cache key")
            return docs[0]
        except Exception as exc:  # pragma: no cover - повреждённый кеш
            log.warning("Битый кеш %s (%s) — переразбор", p, exc)
            return None

    def _store_one(self, key: str, doc: Doc) -> None:
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        lock_path = p.parent / f".{key}.lock"
        with _exclusive_cache_key(lock_path):
            # Another writer may have completed while this process parsed.
            if self._load_one(key, doc.text) is not None:
                return
            db = DocBin(attrs=_DOCBIN_ATTRS, store_user_data=False)
            db.add(doc)
            fd, tmp_name = tempfile.mkstemp(
                dir=p.parent,
                prefix=f".{key}.",
                suffix=".spacy.tmp",
            )
            os.close(fd)
            tmp = pathlib.Path(tmp_name)
            try:
                db.to_disk(tmp)
                with tmp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(tmp, p)
                _fsync_directory(p.parent)
            except BaseException:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
                raise

    def get_docs(self, texts: Sequence[str], batch_size: int = 32) -> List[Doc]:
        """Вернуть Doc для каждого текста (из кеша или разобрав промахи)."""
        keys = [
            _text_key(t, self.identity.identity_sha256, self.version)
            for t in texts
        ]
        result: List[Optional[Doc]] = [None] * len(texts)

        disk_idx: List[int] = []
        for i, key in enumerate(keys):
            doc = _mem_doc_get(key)          # 1) память
            if doc is not None:
                result[i] = doc
            else:
                disk_idx.append(i)

        miss_idx: List[int] = []
        for i in disk_idx:
            doc = self._load_one(keys[i], texts[i])     # 2) диск (DocBin)
            if doc is None:
                miss_idx.append(i)
            else:
                result[i] = doc
                _mem_doc_put(keys[i], doc)

        if miss_idx:                          # 3) разбор spaCy
            miss_texts = [texts[i] for i in miss_idx]
            log.info("DocCache: %d/%d промахов — разбираю spaCy", len(miss_idx), len(texts))
            for j, doc in zip(miss_idx, self.nlp.pipe(miss_texts, batch_size=batch_size)):
                if doc.text != texts[j]:
                    raise RuntimeError("spaCy pipeline changed input text during parsing")
                self._store_one(keys[j], doc)
                result[j] = doc
                _mem_doc_put(keys[j], doc)

        return [d for d in result]  # type: ignore[return-value]

    def warm(self, texts: Sequence[str], batch_size: int = 32, n_process: int = 1) -> int:
        """Прогреть кеш для всего корпуса. Возвращает число вновь разобранных.

        n_process>1 включает многопроцессный разбор spaCy (важно для ~12k чанков:
        одно-процессный lg-разбор всего корпуса занимает ~час, на 4-8 ядрах — кратно
        быстрее). Память: каждый воркер держит копию lg-модели (~0.6 ГБ).
        """
        keys = [
            _text_key(t, self.identity.identity_sha256, self.version)
            for t in texts
        ]
        miss = []
        for key, text in zip(keys, texts, strict=True):
            doc = _mem_doc_get(key)
            if doc is None:
                doc = self._load_one(key, text)
            if doc is None:
                miss.append((key, text))
            else:
                _mem_doc_put(key, doc)
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
