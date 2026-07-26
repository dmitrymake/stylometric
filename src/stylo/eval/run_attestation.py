"""Live immutable-input checks for long-running scientific orchestration."""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from collections.abc import Mapping


class LiveRunAttestationError(RuntimeError):
    """Code, config, or cache bytes changed after the run identity was minted."""


def _sha256_regular(path: pathlib.Path, *, label: str) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise LiveRunAttestationError(f"{label} is missing or unsafe: {path}")
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


@dataclasses.dataclass(frozen=True)
class LiveRunAttestor:
    """Picklable verifier suitable for the parent and joblib workers."""

    repository_root: str
    code_hashes: tuple[tuple[str, str], ...]
    config_path: str
    config_sha256: str
    cache_path: str | None = None
    cache_sha256: str | None = None
    cache_size_bytes: int | None = None

    @classmethod
    def build(
        cls,
        *,
        repository_root: str | pathlib.Path,
        code_hashes: Mapping[str, str],
        config_path: str | pathlib.Path,
        config_sha256: str,
        cache_path: str | pathlib.Path | None = None,
        cache_sha256: str | None = None,
        cache_size_bytes: int | None = None,
    ) -> "LiveRunAttestor":
        if type(code_hashes) is not dict or not code_hashes:
            raise LiveRunAttestationError("code_hashes must be a nonempty exact dict")
        rows: list[tuple[str, str]] = []
        for relative, digest in sorted(code_hashes.items()):
            if (
                type(relative) is not str
                or not relative
                or pathlib.PurePosixPath(relative).is_absolute()
                or ".." in pathlib.PurePosixPath(relative).parts
                or type(digest) is not str
                or len(digest) != 64
            ):
                raise LiveRunAttestationError("malformed code-hash inventory")
            rows.append((relative, digest))
        if (cache_path is None) != (cache_sha256 is None):
            raise LiveRunAttestationError(
                "cache path and digest must either both be supplied or both omitted"
            )
        instance = cls(
            repository_root=str(pathlib.Path(repository_root).resolve()),
            code_hashes=tuple(rows),
            config_path=str(pathlib.Path(config_path).resolve()),
            config_sha256=str(config_sha256),
            cache_path=(
                None if cache_path is None else str(pathlib.Path(cache_path).resolve())
            ),
            cache_sha256=cache_sha256,
            cache_size_bytes=cache_size_bytes,
        )
        instance.verify("initial")
        return instance

    def verify(self, stage: str) -> None:
        if type(stage) is not str or not stage:
            raise LiveRunAttestationError("attestation stage must be a nonempty string")
        root = pathlib.Path(self.repository_root)
        for relative, expected in self.code_hashes:
            observed, _size = _sha256_regular(
                root / relative, label=f"{stage}: code input {relative}"
            )
            if observed != expected:
                raise LiveRunAttestationError(
                    f"{stage}: code input drifted: {relative}"
                )
        observed_config, _size = _sha256_regular(
            pathlib.Path(self.config_path), label=f"{stage}: config"
        )
        if observed_config != self.config_sha256:
            raise LiveRunAttestationError(f"{stage}: config bytes drifted")
        if self.cache_path is not None:
            observed_cache, size = _sha256_regular(
                pathlib.Path(self.cache_path), label=f"{stage}: representation cache"
            )
            if (
                observed_cache != self.cache_sha256
                or (
                    self.cache_size_bytes is not None
                    and size != self.cache_size_bytes
                )
            ):
                raise LiveRunAttestationError(
                    f"{stage}: representation cache bytes drifted"
                )


__all__ = ["LiveRunAttestationError", "LiveRunAttestor"]
