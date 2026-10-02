"""Rank an unknown text within an explicit panel; abstain from authorship decisions.

The LR/legacy-Delta ensemble is a relative score, not a calibrated probability of
historical authorship. A calibrated open-set applicability gate is not available.
"""
from __future__ import annotations

import datetime
import io
import logging
import pathlib
from typing import List

import joblib
import numpy as np
from scipy.stats import trim_mean

from ..config import deployment_candidates, load_config
from ..corpus import load_unknown
from ..dataset import resolve_fragment_roots
from ..domain.prediction_contract import (
    PredictionContractError,
    validate_author_universe,
    validate_class_indices,
    validate_distances,
    validate_probabilities,
)
from ..jsonio import loads_strict
from ..lang import display_name
from .bundle import BundleError, load_bundle_payloads

log = logging.getLogger("stylo.pipeline.predict")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


def run(
    cfg=None,
    unknown_dir: str | None = None,
    *,
    expected_bundle_token: str | None = None,
) -> dict:
    cfg = cfg or load_config()
    from ..eval.provenance import UnsupportedVariantError
    from ..domain.work_weighting import (CHUNK_WEIGHTED_LEGACY,
                                       resolve_training_weighting)
    weighting = resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))
    if weighting != CHUNK_WEIGHTED_LEGACY:
        # Work-balanced training has no supported deployment inference path. Fail closed rather
        # than silently score the unknown with the legacy production model (data/model.pkl).
        raise UnsupportedVariantError(
            f"predict under training_weighting={weighting!r} is unsupported for deployment; "
            "the legacy model must not be used to score a work_balanced run"
        )
    data = pathlib.Path(cfg.get_path("paths.data", "data"))
    docs_dir = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    if docs_dir.is_symlink():
        raise BundleError(f"docs root must not be a symlink: {docs_dir}")
    docs_dir.mkdir(parents=True, exist_ok=True)

    pinned_token = cfg.get_path("deployment.expected_bundle_token", None)
    if pinned_token is not None and (
        not isinstance(pinned_token, str) or not pinned_token
    ):
        raise BundleError("deployment.expected_bundle_token must be a nonempty string")
    if expected_bundle_token is not None and pinned_token is not None and expected_bundle_token != pinned_token:
        raise BundleError("CLI and configured trusted bundle tokens conflict")
    trusted_token = expected_bundle_token if expected_bundle_token is not None else pinned_token
    if not isinstance(trusted_token, str) or not trusted_token:
        raise BundleError(
            "predict requires a trusted deployment bundle token "
            "(--model-bundle-token or deployment.expected_bundle_token); "
            "refusing executable deserialisation without an external commitment"
        )
    bundle_root = data / "deployment" / CHUNK_WEIGHTED_LEGACY
    bundle_meta, payloads = load_bundle_payloads(
        bundle_root, expected_token=trusted_token
    )
    # Deserialize the exact bytes whose hashes were checked, never a pathname
    # that could be substituted between verification and joblib.load.
    pipe = joblib.load(io.BytesIO(payloads["model.pkl"]))
    delta = joblib.load(io.BytesIO(payloads["delta.pkl"]))
    try:
        authors_raw = loads_strict(payloads["authors.json"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError(f"bundle authors.json is invalid strict UTF-8 JSON: {exc}") from exc
    try:
        authors = list(validate_author_universe(authors_raw))
        validate_class_indices(pipe.classes_, len(authors), name="model.classes_")
        validate_class_indices(delta.classes_, len(authors), name="delta.classes_")
    except (AttributeError, PredictionContractError) as exc:
        raise BundleError(f"bundle class-universe contract failed: {exc}") from exc
    candidates = deployment_candidates(cfg)
    if tuple(authors) != candidates:
        raise BundleError("bundle authors do not match deployment.candidate_authors")

    unk_root = (
        pathlib.Path(unknown_dir)
        if unknown_dir
        else resolve_fragment_roots(cfg).unknown_root
    )
    # ``load_unknown`` is strict: unreadable, non-UTF-8, empty and symlinked
    # fragments abort attribution instead of silently changing the evidence.
    texts = load_unknown(unk_root.parent, unknown_name=unk_root.name)
    if not texts:
        raise RuntimeError(f"Нет фрагментов unknown в {unk_root}")
    log.info("Unknown фрагментов: %d", len(texts))

    try:
        lr_probs = validate_probabilities(
            pipe.predict_proba(texts),
            rows=len(texts),
            n_classes=len(authors),
            name="model.predict_proba",
        )
    except PredictionContractError as exc:
        raise BundleError(f"bundle probability contract failed: {exc}") from exc
    lr_mean = lr_probs.mean(axis=0)
    lr_full = lr_mean

    try:
        d_dist = validate_distances(
            delta.distances(texts),
            rows=len(texts),
            n_classes=len(authors),
            name="delta.distances",
        )
    except PredictionContractError as exc:
        raise BundleError(f"bundle distance contract failed: {exc}") from exc
    d_mean = trim_mean(d_dist, 0.1, axis=0)
    delta_full = _softmax(-d_mean)
    if not np.isfinite(delta_full).all() or not np.isclose(
        delta_full.sum(), 1.0, rtol=0.0, atol=1e-8
    ):
        raise BundleError("delta softmax produced an invalid probability vector")

    ens = 0.6 * lr_full + 0.4 * delta_full
    order = np.argsort(-ens, kind="stable")
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("evaluation.top_k_candidates must be a positive integer")
    margin = float(ens[order[0]] - ens[order[1]]) if len(order) > 1 else 0.0

    lines: List[str] = []
    lines.append("=== Диагностический рейтинг кандидатов (LR + legacy Delta) ===")
    lines.append(f"Дата: {datetime.datetime.now():%d.%m.%Y %H:%M}")
    lines.append(f"Фрагментов: {len(texts)}")
    lines.append("Панель: " + ", ".join(display_name(author) for author in authors))
    lines.append("Оценки сравнивают кандидатов внутри панели; это не вероятности авторства.")
    lines.append("")
    lines.append(f"Топ-{top_k} кандидатов (оценка ансамбля):")
    for i in order[:top_k]:
        lines.append(f"  {display_name(authors[int(i)]):24} ens={ens[int(i)]:.4f} "
                     f"(LR={lr_full[int(i)]:.4f}, Delta={delta_full[int(i)]:.4f})")
    lines.append("")
    lines.append(f"Ближайший кандидат внутри панели: {display_name(authors[int(order[0])])}")
    lines.append(f"Margin над 2-м местом: {margin:.4f}")
    lines.append("Авторство не установлено: проверка применимости к неизвестному автору не откалибрована.")
    lines.append("")
    lines.append("Отдельные методы:")
    lines.append(f"  LR    → {display_name(authors[int(np.argmax(lr_full))])}")
    lines.append(f"  Delta → {display_name(authors[int(np.argmax(delta_full))])}")

    report = "\n".join(lines)
    from ..report.evidence import publish_prediction

    publish_prediction(
        cfg,
        unknown_root=unk_root,
        report=report,
        bundle_token=trusted_token,
        bundle_meta=bundle_meta,
    )
    print(report)
    return {"winner": None, "diagnostic_closed_set_top": authors[int(order[0])],
            "candidate_authors": authors, "abstained": True,
            "abstention_reason": "open_set_applicability_unavailable",
            "score_scope": "closed_set_ensemble", "margin": margin,
            "ensemble": {authors[i]: float(ens[i]) for i in range(len(authors))}}
