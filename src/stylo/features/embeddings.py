"""Контекстные эмбеддинги (ruBERT, mean-pooled) — ЭКСПЕРИМЕНТАЛЬНЫЙ блок.

ВНИМАНИЕ — РИСК УТЕЧКИ ТЕМЫ: BERT кодирует прежде всего тему/семантику, а в корпусе
по 2–8 книг на автора темы коррелируют с автором. Поэтому LOBO-точность с эмбеддингами
может отражать тему, а не стиль. Блок off по умолчанию и используется ТОЛЬКО в
topic-controlled эксперименте (см. eval/sweep.py: cross-topic фолд) и всегда
сравнивается с char-baseline.

Зависимости (torch/transformers) импортируются лениво — нужны лишь при enabled=true.
Disk cache v2 binds text, immutable model/tokenizer commits, inference settings,
pooling, dimension and numerical runtime. Pass full 40-hex commit IDs through
``revision`` and optionally ``tokenizer_revision`` (defaults to ``revision``).
Branches, tags, unspecified revisions and local model directories disable disk
caching; inference remains available. Model loading is still lazy, including
when checking cached vector dimensions. Old sha1(text+model) cache files are
ignored and can be removed separately; they are never migrated or reused.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import pathlib
import re
import tempfile
from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from ..jsonio import dumps_strict
from .base import FeatureBlock

log = logging.getLogger("stylo.features.embeddings")


class EmbeddingBlock(FeatureBlock):
    group = "embeddings"
    name = "embeddings"

    def __init__(self, model_name: str = "ai-forever/ruBert-base",
                 batch_size: int = 16, max_length: int = 256,
                 cache_dir: str | pathlib.Path = "data/emb_cache",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("embedding batch_size must be a positive integer")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("embedding max_length must be a positive integer")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = pathlib.Path(cache_dir)
        self.revision = revision
        self.tokenizer_revision = revision if tokenizer_revision is None else tokenizer_revision
        self._tok = None
        self._model = None
        self._torch = None
        self._dimension = None
        if not self._cache_enabled():
            log.warning(
                "Embedding disk cache disabled: model and tokenizer require immutable "
                "40-hex commit revisions and a remote model ID."
            )

    def __setstate__(self, state):
        # Old pickled models remain usable but their unbound disk cache is disabled.
        super().__setstate__(state)
        defaults = {"revision": None, "tokenizer_revision": None, "_dimension": None}
        for key, value in defaults.items():
            self.__dict__.setdefault(key, value)
            if isinstance(self.__dict__.get("_ctor_state"), dict):
                self._ctor_state.setdefault(key, value)

    def _cache_enabled(self) -> bool:
        revisions = (self.revision, self.tokenizer_revision)
        return (
            not pathlib.Path(self.model_name).is_dir()
            and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
                    for value in revisions)
        )

    def _ensure_model(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Для embeddings нужен extra: pip install '.[embeddings]' "
                    "(torch, transformers)."
                ) from exc
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.tokenizer_revision
            )
            self._model = AutoModel.from_pretrained(self.model_name, revision=self.revision)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            log.info("ruBERT %s загружена на %s", self.model_name, self._device)
        dimension = getattr(self._model.config, "hidden_size", None)
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("embedding model config requires a positive hidden_size")
        self._dimension = dimension

    def _runtime_identity(self) -> dict:
        versions = {}
        for package in ("numpy", "torch", "transformers", "tokenizers"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        versions["device"] = self._device
        if self._device == "cuda":
            versions["cuda"] = self._torch.version.cuda
            versions["gpu"] = self._torch.cuda.get_device_name()
        return versions

    def _key(self, text: str) -> str:
        if not self._cache_enabled():
            raise ValueError("embedding cache identity requires immutable commit revisions")
        self._ensure_model()
        identity = {
            "schema": "stylo.embedding-cache.v2",
            "model_name": self.model_name,
            "model_revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "pooling": "attention_mask_mean_including_special_tokens.v1",
            "dtype": "float32",
            "dimension": self._dimension,
            "runtime": self._runtime_identity(),
            "text": text,
        }
        return hashlib.sha256(
            dumps_strict(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _cache_path(self, text: str) -> pathlib.Path:
        key = self._key(text)
        return self.cache_dir / "v2" / key[:2] / f"{key}.npy"

    def _validate_array(self, values, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != shape or array.dtype.kind not in "fiu":
            raise ValueError(f"embedding array must be numeric with shape {shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError("embedding array must contain only finite float32 values")
        return array

    def _cached(self, text: str) -> Optional[np.ndarray]:
        if not self._cache_enabled():
            return None
        p = self._cache_path(text)
        try:
            vector = np.load(p, allow_pickle=False)
            if not isinstance(vector, np.ndarray):
                vector.close()
                return None
            if vector.dtype != np.float32:
                return None
            return self._validate_array(vector, (self._dimension,))
        except (OSError, ValueError, EOFError):
            return None

    def _store(self, text: str, vec: np.ndarray) -> None:
        if not self._cache_enabled():
            return
        p = self._cache_path(text)
        vector = self._validate_array(vec, (self._dimension,))
        p.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=p.parent, suffix=".npy", delete=False) as stream:
                temporary_path = pathlib.Path(stream.name)
                np.save(stream, vector, allow_pickle=False)
            os.replace(temporary_path, p)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        torch = self._torch
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=self.max_length, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state          # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()       # (B, T, 1)
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        mean = (summed / counts).cpu().numpy()
        return mean

    def fit(self, texts, reps, groups=None):
        return self

    def transform(self, texts, reps) -> csr_matrix:
        texts = list(texts)
        self._ensure_model()
        out: List[Optional[np.ndarray]] = [self._cached(t) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start:start + self.batch_size]
            vecs = self._validate_array(
                self._encode_batch([texts[i] for i in idx]), (len(idx), self._dimension)
            )
            for i, v in zip(idx, vecs):
                out[i] = v
                self._store(texts[i], v)
        arr = np.vstack(out).astype(np.float32) if out else np.zeros((0, self._dimension), np.float32)
        return csr_matrix(arr)

    def feature_names(self) -> List[str]:
        self._ensure_model()
        return [f"emb::{i}" for i in range(self._dimension)]
