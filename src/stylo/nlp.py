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
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import logging
import marshal
import os
import pathlib
import tempfile
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import spacy
from spacy.tokens import Doc, DocBin
from spacy.util import load_model_from_init_py as _spacy_load_model_from_init_py

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


def _safe_distribution_member(
    distribution,
    root: pathlib.Path,
    member: str,
    *,
    model: str,
) -> pathlib.Path:
    path = pathlib.Path(distribution.locate_file(member))
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"installed spaCy model {model!r} RECORD member is "
            f"missing/unsafe: {member}"
        )
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"installed spaCy model {model!r} RECORD member escapes "
            f"the environment: {member}"
        )
    return resolved


def _pyc_optimization(
    source: pathlib.Path,
    bytecode: pathlib.Path,
) -> int | None:
    for level, tag in ((0, ""), (1, "1"), (2, "2")):
        candidate = pathlib.Path(
            importlib.util.cache_from_source(str(source), optimization=tag)
        )
        if candidate == bytecode:
            return level
    return None


def _verify_derived_bytecode(
    *,
    model: str,
    member: str,
    bytecode: pathlib.Path,
    source: pathlib.Path,
    source_bytes: bytes,
) -> None:
    raw = bytecode.read_bytes()
    if len(raw) < 16 or raw[:4] != importlib.util.MAGIC_NUMBER:
        raise RuntimeError(
            f"installed spaCy model {model!r} has invalid generated "
            f"bytecode in RECORD: {member!r}"
        )

    flags = int.from_bytes(raw[4:8], "little")
    if flags not in {0, 1, 3}:
        raise RuntimeError(
            f"installed spaCy model {model!r} has unsupported generated "
            f"bytecode flags in RECORD: {member!r}"
        )
    if flags == 0:
        source_stat = source.stat()
        expected_header = (
            (int(source_stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
            + (len(source_bytes) & 0xFFFFFFFF).to_bytes(4, "little")
        )
    else:
        expected_header = importlib.util.source_hash(source_bytes)
    if raw[8:16] != expected_header:
        raise RuntimeError(
            f"installed spaCy model {model!r} generated bytecode header "
            f"does not match its verified source: {member!r}"
        )

    optimization = _pyc_optimization(source, bytecode)
    if optimization is None:
        raise RuntimeError(
            f"installed spaCy model {model!r} has an unsupported generated "
            f"bytecode path in RECORD: {member!r}"
        )
    try:
        expected_code = compile(
            source_bytes,
            str(source),
            "exec",
            dont_inherit=True,
            optimize=optimization,
        )
        expected_body = marshal.dumps(expected_code)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"installed spaCy model {model!r} has an invalid verified "
            f"source for generated bytecode: {member!r}"
        ) from exc
    if raw[16:] != expected_body:
        raise RuntimeError(
            f"installed spaCy model {model!r} generated bytecode does "
            f"not derive from its verified source: {member!r}"
        )


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

    root = pathlib.Path(distribution.locate_file("")).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(
            f"installed spaCy model {name!r} has no filesystem package root"
        )

    seen: set[str] = set()
    parsed: dict[str, tuple[pathlib.PurePosixPath, str, str]] = {}
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
            parsed[member] = (member_path, digest_spec, size_text)
    except csv.Error as exc:
        raise RuntimeError(
            f"installed spaCy model {name!r} has invalid wheel RECORD"
        ) from exc

    metadata_root = getattr(distribution, "_path", None)
    if metadata_root is None:
        raise RuntimeError(
            f"installed spaCy model {name!r} has no filesystem metadata root"
        )
    own_record_path = pathlib.Path(metadata_root) / "RECORD"
    if own_record_path.is_symlink() or not own_record_path.is_file():
        raise RuntimeError(
            f"installed spaCy model {name!r} has an unsafe wheel RECORD"
        )
    own_record = own_record_path.resolve(strict=True)
    if not own_record.is_relative_to(root):
        raise RuntimeError(
            f"installed spaCy model {name!r} has an unsafe wheel RECORD"
        )
    own_record_member = own_record.relative_to(root).as_posix()
    own_row = parsed.get(own_record_member)
    if (
        own_row is None
        or own_row[0].name != "RECORD"
        or not own_row[0].parent.name.endswith(".dist-info")
        or own_row[1]
        or own_row[2]
        or own_record.read_text(encoding="utf-8") != record
    ):
        raise RuntimeError(
            f"installed spaCy model {name!r} does not bind its own wheel RECORD"
        )

    empty_members = {
        member
        for member, (_member_path, digest_spec, _size_text) in parsed.items()
        if not digest_spec
    }
    bytecode_sources: dict[str, str] = {}
    for member in sorted(empty_members - {own_record_member}):
        member_path, _digest_spec, size_text = parsed[member]
        if (
            size_text
            or member_path.suffix != ".pyc"
            or member_path.parent.name != "__pycache__"
        ):
            raise RuntimeError(
                f"installed spaCy model {name!r} has an unhashed "
                f"model payload in RECORD: {member!r}"
            )
        bytecode_path = pathlib.Path(distribution.locate_file(member))
        try:
            source_path = pathlib.Path(
                importlib.util.source_from_cache(str(bytecode_path))
            ).resolve(strict=True)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"installed spaCy model {name!r} has unbound generated "
                f"bytecode in RECORD: {member!r}"
            ) from exc
        if not source_path.is_relative_to(root):
            raise RuntimeError(
                f"installed spaCy model {name!r} generated bytecode source "
                f"escapes the environment: {member!r}"
            )
        source_member = source_path.relative_to(root).as_posix()
        source_row = parsed.get(source_member)
        if source_row is None or not source_row[1] or not source_row[2]:
            raise RuntimeError(
                f"installed spaCy model {name!r} generated bytecode lacks "
                f"a hashed source in RECORD: {member!r}"
            )
        bytecode_sources[member] = source_member

    # RECORD may omit bytecode generated after installation.  Such a cache is
    # still preferred over the verified source at import time, so discover and
    # validate every current-runtime cache path for every hashed Python source.
    for source_member, (
        source_member_path,
        digest_spec,
        _size_text,
    ) in parsed.items():
        if source_member_path.suffix != ".py" or not digest_spec:
            continue
        source_path = pathlib.Path(distribution.locate_file(source_member))
        for _optimization, tag in ((0, ""), (1, "1"), (2, "2")):
            bytecode_path = pathlib.Path(
                importlib.util.cache_from_source(
                    str(source_path),
                    optimization=tag,
                )
            )
            if not bytecode_path.exists() and not bytecode_path.is_symlink():
                continue
            try:
                bytecode_member = bytecode_path.relative_to(root).as_posix()
            except ValueError as exc:
                raise RuntimeError(
                    f"installed spaCy model {name!r} has generated bytecode "
                    f"outside the environment for source {source_member!r}"
                ) from exc
            existing_source = bytecode_sources.setdefault(
                bytecode_member,
                source_member,
            )
            if existing_source != source_member:
                raise RuntimeError(
                    f"installed spaCy model {name!r} has ambiguous generated "
                    f"bytecode source: {bytecode_member!r}"
                )

    owned_roots: set[pathlib.Path] = set()
    for member_path, _digest_spec, _size_text in parsed.values():
        if (
            len(member_path.parts) < 2
            or member_path.parts[0].endswith(".dist-info")
        ):
            continue
        candidate = root / member_path.parts[0]
        if candidate.is_symlink():
            raise RuntimeError(
                f"installed spaCy model {name!r} has a symlinked package root"
            )
        if candidate.is_dir():
            owned_roots.add(candidate)

    allowed_unrecorded = set(bytecode_sources)
    for owned_root in sorted(owned_roots):
        for directory, directory_names, file_names in os.walk(
            owned_root,
            followlinks=False,
        ):
            directory_path = pathlib.Path(directory)
            for directory_name in directory_names:
                child = directory_path / directory_name
                if child.is_symlink():
                    raise RuntimeError(
                        f"installed spaCy model {name!r} has an unrecorded "
                        f"symlinked package path: "
                        f"{child.relative_to(root).as_posix()!r}"
                    )
            for file_name in file_names:
                child = directory_path / file_name
                member = child.relative_to(root).as_posix()
                if member in parsed or member in allowed_unrecorded:
                    continue
                raise RuntimeError(
                    f"installed spaCy model {name!r} has an unrecorded "
                    f"model payload: {member!r}"
                )

    verified = 0
    verified_sources: dict[str, tuple[pathlib.Path, bytes]] = {}
    required_sources = set(bytecode_sources.values())
    for member, (_member_path, digest_spec, size_text) in parsed.items():
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

        resolved = _safe_distribution_member(
            distribution,
            root,
            member,
            model=name,
        )
        digest = hashlib.sha256()
        observed_size = 0
        source_bytes = bytearray() if member in required_sources else None
        with resolved.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                observed_size += len(block)
                if source_bytes is not None:
                    source_bytes.extend(block)
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
        if source_bytes is not None:
            verified_sources[member] = (resolved, bytes(source_bytes))
        verified += 1

    for bytecode_member, source_member in bytecode_sources.items():
        bytecode_path = pathlib.Path(
            distribution.locate_file(bytecode_member)
        )
        if not bytecode_path.exists() and not bytecode_path.is_symlink():
            continue
        resolved_bytecode = _safe_distribution_member(
            distribution,
            root,
            bytecode_member,
            model=name,
        )
        source_path, source_bytes = verified_sources[source_member]
        _verify_derived_bytecode(
            model=name,
            member=bytecode_member,
            bytecode=resolved_bytecode,
            source=source_path,
            source_bytes=source_bytes,
        )

    if verified == 0:
        raise RuntimeError(
            f"installed spaCy model {name!r} RECORD has no verified members"
        )
    return (
        str(distribution.version),
        hashlib.sha256(record.encode("utf-8")).hexdigest(),
    )


def _verified_model_package_binding(
    name: str,
    package_identity: tuple[str, str],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    """Bind a standard model package and its exact init path to its RECORD."""

    if not name.isidentifier():
        raise RuntimeError(
            f"spaCy model {name!r} is not a top-level installed package"
        )
    distribution = importlib.metadata.distribution(name)
    record = distribution.read_text("RECORD")
    if not record:
        raise RuntimeError(f"installed spaCy model {name!r} has no wheel RECORD")
    observed_identity = (
        str(distribution.version),
        hashlib.sha256(record.encode("utf-8")).hexdigest(),
    )
    if observed_identity != package_identity:
        raise RuntimeError(
            f"spaCy model {name!r} distribution changed before import binding"
        )

    root = pathlib.Path(distribution.locate_file("")).resolve(strict=True)
    # Search the current import path directly instead of consulting
    # ``sys.modules``.  A previously imported model module is mutable process
    # state and must never become authority for a scientific model load.
    spec = importlib.machinery.PathFinder.find_spec(name)
    if (
        spec is None
        or spec.name != name
        or type(spec.loader) is not importlib.machinery.SourceFileLoader
        or type(spec.origin) is not str
        or spec.submodule_search_locations is None
    ):
        raise RuntimeError(
            f"spaCy model {name!r} import target is not a standard source package"
        )

    origin_path = pathlib.Path(spec.origin)
    if (
        origin_path.is_symlink()
        or not origin_path.is_file()
        or origin_path.name != "__init__.py"
    ):
        raise RuntimeError(
            f"spaCy model {name!r} import target is missing or unsafe"
        )
    origin = origin_path.resolve(strict=True)
    if not origin.is_relative_to(root):
        raise RuntimeError(
            f"spaCy model {name!r} import target is outside its verified wheel"
        )
    origin_member = origin.relative_to(root).as_posix()
    expected_origin_member = f"{name}/__init__.py"
    if origin_member != expected_origin_member:
        raise RuntimeError(
            f"spaCy model {name!r} lacks its canonical package init"
        )

    rows: dict[str, tuple[str, str]] = {}
    try:
        for row in csv.reader(io.StringIO(record), strict=True):
            if len(row) != 3:
                raise RuntimeError(
                    f"installed spaCy model {name!r} has malformed RECORD"
                )
            rows[row[0]] = (row[1], row[2])
    except csv.Error as exc:
        raise RuntimeError(
            f"installed spaCy model {name!r} has invalid wheel RECORD"
        ) from exc
    origin_identity = rows.get(origin_member)
    if (
        origin_identity is None
        or not origin_identity[0].startswith("sha256=")
        or not origin_identity[1].isascii()
        or not origin_identity[1].isdecimal()
    ):
        raise RuntimeError(
            f"spaCy model {name!r} import target is not RECORD-hashed"
        )
    recorded_origin = _safe_distribution_member(
        distribution,
        root,
        origin_member,
        model=name,
    )
    if recorded_origin != origin:
        raise RuntimeError(
            f"spaCy model {name!r} import target differs from its RECORD member"
        )

    locations = tuple(spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"spaCy model {name!r} has ambiguous package search roots"
        )
    location_path = pathlib.Path(locations[0])
    if location_path.is_symlink() or not location_path.is_dir():
        raise RuntimeError(
            f"spaCy model {name!r} has an unsafe package search root"
        )
    location = location_path.resolve(strict=True)
    if location != origin.parent:
        raise RuntimeError(
            f"spaCy model {name!r} package search root differs from its origin"
        )
    if location.relative_to(root).as_posix() != name:
        raise RuntimeError(
            f"spaCy model {name!r} has a noncanonical package search root"
        )

    return (
        str(root),
        origin_member,
        origin_identity[0],
        origin_identity[1],
        (location.relative_to(root).as_posix(),),
    )


def _load_verified_model_from_binding(
    binding: tuple[str, str, str, str, tuple[str, ...]],
    *,
    disable: Sequence[str],
):
    """Load verified model data without executing its mutable package module."""

    root_text, origin_member, _digest, _size, _locations = binding
    root = pathlib.Path(root_text)
    origin = root.joinpath(*pathlib.PurePosixPath(origin_member).parts)
    if (
        origin.is_symlink()
        or not origin.is_file()
        or origin.resolve(strict=True) != origin
    ):
        raise RuntimeError("verified spaCy model init changed before data load")
    # Standard spaCy trained-pipeline wheels implement package ``load`` as this
    # exact helper call.  Calling it by the verified init path preserves those
    # semantics while keeping caller-preloaded ``name`` / ``name.*`` modules
    # outside the scientific trust boundary.
    return _spacy_load_model_from_init_py(origin, disable=disable)


class _VerifiedModelLoadUnavailable(OSError):
    """The direct model load failed after an unchanged-package recheck."""


def _verified_spacy_load(
    model: str,
    fallback: Optional[str],
    *,
    disable: Sequence[str],
):
    """Verify wheel payloads around a direct load of their trained-pipeline data."""

    def reverify_bound(
        name: str,
        before: tuple[str, str],
        before_binding: tuple[str, str, str, str, tuple[str, ...]],
        *,
        outcome: str,
    ) -> tuple[str, str]:
        try:
            after = verified_installed_package_record(name)
            after_binding = _verified_model_package_binding(name, after)
        except Exception as exc:
            raise RuntimeError(
                f"cannot reverify spaCy model {name!r} after {outcome}"
            ) from exc
        if after != before or after_binding != before_binding:
            raise RuntimeError(
                f"spaCy model {name!r} changed during {outcome}"
            )
        return after

    def load_bound(
        name: str,
        before: tuple[str, str],
        before_binding: tuple[str, str, str, str, tuple[str, ...]],
    ):
        try:
            loaded = _load_verified_model_from_binding(
                before_binding,
                disable=disable,
            )
        except OSError as exc:
            # A load-time OSError is eligible for the configured fallback only
            # when the package stayed byte-identical.  Otherwise a concurrent
            # integrity failure could be laundered into an ordinary absence.
            reverify_bound(
                name,
                before,
                before_binding,
                outcome="a failed direct load",
            )
            raise _VerifiedModelLoadUnavailable(str(exc)) from exc
        except Exception:
            reverify_bound(
                name,
                before,
                before_binding,
                outcome="a failed direct load",
            )
            raise
        after = reverify_bound(
            name,
            before,
            before_binding,
            outcome="a successful direct load",
        )
        return loaded, after

    def verified_fallback():
        assert fallback is not None
        try:
            before = verified_installed_package_record(fallback)
        except importlib.metadata.PackageNotFoundError as exc:
            if pathlib.Path(fallback).exists():
                raise RuntimeError(
                    f"spaCy fallback {fallback!r} is not an installed wheel"
                ) from exc
            raise OSError(
                f"spaCy fallback {fallback!r} is not installed as a wheel"
            ) from exc
        before_binding = _verified_model_package_binding(
            fallback,
            before,
        )
        loaded, after = load_bound(
            fallback,
            before,
            before_binding,
        )
        return loaded, fallback, after

    try:
        before = verified_installed_package_record(model)
    except importlib.metadata.PackageNotFoundError as exc:
        if pathlib.Path(model).exists():
            raise RuntimeError(
                f"spaCy model {model!r} is not an installed wheel"
            ) from exc
        if fallback is None:
            raise OSError(
                f"spaCy model {model!r} is not installed as a wheel"
            ) from exc
        log.warning(
            "Модель %s не установлена как wheel, использую fallback %s",
            model,
            fallback,
        )
        return verified_fallback()

    before_binding = _verified_model_package_binding(
        model,
        before,
    )
    try:
        loaded, after = load_bound(
            model,
            before,
            before_binding,
        )
    except _VerifiedModelLoadUnavailable:
        if fallback is None:
            raise
        log.warning("Модель %s не найдена, использую fallback %s", model, fallback)
        return verified_fallback()
    return loaded, model, after


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
    package_identity: tuple[str, str] | None = None,
    disabled_pipes: Sequence[str] = _DISABLE,
) -> ResolvedNLPIdentity:
    version, record_sha = (
        package_identity
        if package_identity is not None
        else _installed_package_identity(resolved, nlp)
    )
    payload = {
        "requested_model": requested,
        "resolved_model": resolved,
        "fallback_used": resolved != requested,
        "package_version": version,
        "package_record_sha256": record_sha,
        "spacy_version": spacy.__version__,
        "disabled_pipes": sorted(disabled_pipes),
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
        disabled_pipes=tuple(sorted(disabled_pipes)),
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
    nlp, resolved, package_identity = _verified_spacy_load(
        model,
        fallback,
        disable=_DISABLE,
    )
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names \
            and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = max_length
    identity = _build_nlp_identity(
        requested=model,
        resolved=resolved,
        nlp=nlp,
        max_length=max_length,
        package_identity=package_identity,
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
    if type(model) is not str or not model:
        raise ValueError("spaCy model must be a nonempty string")
    if fallback is not None and (type(fallback) is not str or not fallback):
        raise ValueError("spaCy fallback must be None or a nonempty string")
    disable = ["parser", "tagger", "attribute_ruler", "lemmatizer",
               "morphologizer", "sentencizer", "textcat"]
    key = ("ner", model, fallback, 5_000_000, tuple(disable))
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]
    nlp, resolved, package_identity = _verified_spacy_load(
        model,
        fallback,
        disable=disable,
    )
    nlp.max_length = 5_000_000
    _NLP_IDENTITIES[id(nlp)] = _build_nlp_identity(
        requested=model,
        resolved=resolved,
        nlp=nlp,
        max_length=5_000_000,
        package_identity=package_identity,
        disabled_pipes=disable,
    )
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
