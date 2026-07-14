"""Контекстные эмбеддинги (ruBERT, mean-pooled) — ЭКСПЕРИМЕНТАЛЬНЫЙ блок.

ВНИМАНИЕ — РИСК УТЕЧКИ ТЕМЫ: BERT кодирует прежде всего тему/семантику, а в корпусе
по 2–8 книг на автора темы коррелируют с автором. Поэтому LOBO-точность с эмбеддингами
может отражать тему, а не стиль. Блок off по умолчанию и используется ТОЛЬКО в
topic-controlled эксперименте (см. eval/sweep.py: cross-topic фолд) и всегда
сравнивается с char-baseline.

Зависимости (torch/transformers) импортируются лениво — нужны лишь при enabled=true.
Эмбеддинги кешируются на диск по sha1(text+model), т.к. инференс дорогой.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
from typing import List, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from spacy.tokens import Doc

from .base import FeatureBlock

log = logging.getLogger("stylo.features.embeddings")


class EmbeddingBlock(FeatureBlock):
    group = "embeddings"
    name = "embeddings"

    def __init__(self, model_name: str = "ai-forever/ruBert-base",
                 batch_size: int = 16, max_length: int = 256,
                 cache_dir: str | pathlib.Path = "data/emb_cache"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = pathlib.Path(cache_dir)
        self._tok = None
        self._model = None
        self._torch = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Для embeddings нужен extra: pip install '.[embeddings]' "
                "(torch, transformers)."
            ) from exc
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        log.info("ruBERT %s загружена на %s", self.model_name, self._device)

    def _key(self, text: str) -> str:
        h = hashlib.sha1(self.model_name.encode() + b"\x00" + text.encode("utf-8"))
        return h.hexdigest()

    def _cached(self, text: str) -> Optional[np.ndarray]:
        p = self.cache_dir / self._key(text)[:2] / f"{self._key(text)}.npy"
        if p.exists():
            try:
                return np.load(p)
            except Exception:
                return None
        return None

    def _store(self, text: str, vec: np.ndarray) -> None:
        p = self.cache_dir / self._key(text)[:2] / f"{self._key(text)}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, vec.astype(np.float32))

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
        out: List[Optional[np.ndarray]] = [self._cached(t) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start:start + self.batch_size]
            vecs = self._encode_batch([texts[i] for i in idx])
            for i, v in zip(idx, vecs):
                out[i] = v
                self._store(texts[i], v)
        arr = np.vstack(out).astype(np.float32) if out else np.zeros((0, 768), np.float32)
        return csr_matrix(arr)

    def feature_names(self) -> List[str]:
        # размерность определяется моделью (768 для ruBert-base); имена — после первого encode
        dim = 768
        return [f"emb::{i}" for i in range(dim)]
