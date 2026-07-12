"""Обучение продакшен-модели (config-driven, без тяжёлых артефактов).

Сохраняет:
  data/model.pkl    — Pipeline(StyloVectorizer -> Scaler -> LR)
  data/delta.pkl    — BurrowsDelta (настоящая, на MFW) для ансамбля
  data/authors.json — список авторов (индекс == метка)

НЕ сохраняет train_vectors.pkl: диагностики берут векторы из модели.
"""
from __future__ import annotations

import logging
import pathlib

import joblib

from ..config import load_config
from ..corpus import load_dataset
from ..features.reps import make_rep_cache
from ..jsonio import dump_strict
from ..models.delta import BurrowsDelta
from ..models.lr import make_full_pipeline
from ..vectorizer import StyloVectorizer

log = logging.getLogger("stylo.pipeline.train")


def run(cfg=None, warm: bool = True) -> None:
    cfg = cfg or load_config()
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    data.mkdir(parents=True, exist_ok=True)

    exclude = set(cfg.get_path("corpus_policy.exclude_from_benchmark", []) or [])
    ds = load_dataset(data / "frags_train", exclude_authors=exclude,
                      unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"))
    log.info("Train: %d чанков, %d авторов", len(ds), ds.n_authors)

    if warm:
        make_rep_cache(cfg).warm(list(ds.texts),
                                 n_process=cfg.get_path("language.parse_n_process", 4))

    vec = StyloVectorizer.from_config(cfg)
    pipe = make_full_pipeline(cfg, vec)
    log.info("Обучаю основной пайплайн…")
    pipe.fit(list(ds.texts), ds.y, groups=ds.groups)
    joblib.dump(pipe, data / "model.pkl")
    log.info("Сохранено: %s", data / "model.pkl")

    mfw = cfg.get_path("delta.mfw_sizes", [300])
    delta = BurrowsDelta(mfw_count=mfw[len(mfw) // 2] if mfw else 300,
                         metric=cfg.get_path("delta.metric", "manhattan"))
    delta.fit(list(ds.texts), ds.y, groups=ds.groups)
    joblib.dump(delta, data / "delta.pkl")
    log.info("Сохранено: %s", data / "delta.pkl")

    dump_strict(ds.authors, data / "authors.json", trailing_newline=False)
    log.info("Обучение завершено.")
