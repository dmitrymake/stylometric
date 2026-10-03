"""Rank an unknown text within an explicit panel; abstain from authorship decisions.

The LR/legacy-Delta ensemble is a relative score, not a calibrated probability of
historical authorship. A calibrated open-set applicability gate is not available.
"""
from __future__ import annotations

import io
import hashlib
from dataclasses import dataclass
import logging
import pathlib

import joblib
import numpy as np

from ..config import deployment_candidates, load_config
from ..corpus import list_authors
from .._io import read_regular
from ..dataset import resolve_fragment_roots
from ..domain.prediction_contract import (
    PredictionContractError,
    validate_author_universe,
    validate_class_indices,
    validate_distances,
    validate_probabilities,
)
from ..jsonio import canonical_hash, load_strict, loads_strict
from ..workdoc import WorkManifest
from .bundle import BundleError, load_bundle_payloads, _verify_real_dir_chain

log = logging.getLogger("stylo.pipeline.predict")


@dataclass(frozen=True)
class PredictionTarget:
    work_id: str
    root: pathlib.Path
    files: tuple[pathlib.Path, ...]
    manifest: WorkManifest | None = None
    catalog_root: pathlib.Path | None = None


def _work_id(value: str) -> str:
    if type(value) is not str or len(value.split("/")) != 2 or any(
        not part or part in {".", ".."} or part != part.strip() or "\\" in part or "\x00" in part
        for part in value.split("/")
    ):
        raise BundleError("target-work must be an exact author/work ID")
    return value


def resolve_prediction_target(cfg, target_work=None, *, unknown_root=None) -> PredictionTarget:
    """Select one declared work; never pool several unknown works implicitly."""
    base = pathlib.Path(unknown_root) if unknown_root is not None else resolve_fragment_roots(cfg).unknown_root
    _verify_real_dir_chain(base)
    if not base.is_dir():
        raise BundleError("unknown root must be a real directory")
    targets = {}
    flat = []
    requested = _work_id(target_work) if target_work is not None else None
    for author in sorted(base.iterdir()):
        if requested is not None and author.name != requested.split("/")[0] and author.suffix != ".txt":
            continue
        if author.is_symlink():
            raise BundleError("symlinked unknown input rejected")
        if author.is_file():
            if author.suffix == ".txt":
                flat.append(author)
            elif author.name != "split_manifest.json":
                raise BundleError("unexpected file in unknown root")
            continue
        if not author.is_dir():
            raise BundleError("unknown input must contain regular files/directories")
        for work in sorted(author.iterdir()):
            if requested is not None and work.name != requested.split("/")[1]:
                continue
            if work.is_symlink() or not work.is_dir():
                raise BundleError("unknown inputs must be grouped as author/work directories")
            work_id = _work_id(author.name + "/" + work.name)
            entries = list(work.iterdir())
            if any(p.is_symlink() or not p.is_file() for p in entries):
                raise BundleError("target work contains symlinked, nested or special input")
            files = tuple(sorted(p for p in entries if p.suffix == ".txt"))
            if not files or any(p.suffix != ".txt" and p.name != "manifest.json" for p in entries):
                raise BundleError("target work has an invalid fragment inventory")
            manifest = None
            if (work / "manifest.json").exists():
                manifest = WorkManifest.from_dict(load_strict(work / "manifest.json"))
                if manifest.work_id != work_id or manifest.author_id != author.name:
                    raise BundleError("target manifest work/author identity mismatch")
                if {p.name for p in files} != {row.path for row in manifest.chunks}:
                    raise BundleError("target manifest/file inventory mismatch")
                if [row.span_ordinal for row in manifest.chunks] != list(range(len(files))):
                    raise BundleError("target fragment ordinals must be contiguous")
                files = tuple(work / row.path for row in manifest.chunks)
            targets[work_id] = PredictionTarget(work_id, work, files, manifest, base)
    if requested is not None and requested in targets:
        return targets[requested]
    if flat:
        if targets or len(flat) != 1:
            raise BundleError("ambiguous flat unknown inputs; group fragments into author/work directories")
        work_id = _work_id(cfg.get_path("corpus_policy.unknown_dir_name", "unknown") + "/" + flat[0].stem)
        targets[work_id] = PredictionTarget(work_id, base, tuple(flat), None, base)
    if target_work is not None:
        target_work = requested
        if target_work not in targets:
            raise BundleError("selected target-work is absent from unknown inputs")
        return targets[target_work]
    if len(targets) != 1:
        raise BundleError("unknown inputs contain multiple/no works; select --target-work author/work")
    return next(iter(targets.values()))


def target_identity(target: PredictionTarget) -> dict:
    paths = list(target.files)
    if target.manifest is not None:
        paths.append(target.root / "manifest.json")
    return {
        "target_work_id": target.work_id,
        "target_root": str(target.root.resolve()),
        "target_catalog_root": str(target.catalog_root.resolve()),
        "target_files": [path.name for path in target.files],
        "target_sha256": canonical_hash([
            [path.name, hashlib.sha256(read_regular(path, label="target input")).hexdigest()]
            for path in paths
        ]),
    }


def _target_texts(target: PredictionTarget) -> list[str]:
    texts = []
    for index, path in enumerate(target.files):
        text = read_regular(path, label="target fragment").decode("utf-8").strip()
        if not text:
            raise BundleError("target fragment is empty")
        if target.manifest is not None and hashlib.sha256(text.encode()).hexdigest() != target.manifest.chunks[index].text_sha256:
            raise BundleError("target fragment hash differs from its manifest")
        texts.append(text)
    return texts


def _overlap_status(cfg, target, texts, authors, meta) -> dict:
    from ..corpus_tools.validate_corpus import text_overlap_coverage, validate_threshold
    threshold = validate_threshold(cfg.get_path("corpus_policy.near_dup_threshold", 0.4))
    trained = meta.get("training_work_ids")
    if trained is not None and target.work_id in trained:
        raise BundleError("target work_id overlaps the trained references")
    result = {"work_id": "checked_bundle_metadata" if trained is not None else "unavailable_legacy_bundle",
              "exact_content": "unavailable_no_training_texts",
              "near_duplicates": "unavailable_no_training_texts", "near_dup_threshold": threshold}
    try:
        snapshot = resolve_fragment_roots(cfg)
    except (RuntimeError, FileNotFoundError):
        # Standalone inference can use an explicitly supplied unknown directory.
        # Existing damaged reference publications are not relabelled as absent.
        data = pathlib.Path(cfg.get_path("paths.data", "data"))
        if (data / "fragment_snapshots").exists() or (data / "frags_train").exists():
            raise
        return result
    if not snapshot.train_root.is_dir():
        return result
    from ..dataset import resolve_dataset
    available = set(list_authors(snapshot.train_root, exclude_unknown=cfg.get_path("corpus_policy.unknown_dir_name", "unknown")))
    dataset = resolve_dataset(cfg, meta["training_weighting"], snapshot.train_root,
                              exclude_authors=available - set(authors),
                              unknown_name=cfg.get_path("corpus_policy.unknown_dir_name", "unknown"))
    if dataset.provenance.rows_digest != meta["rows_digest"]:
        raise BundleError("available training references differ from the authenticated bundle")
    if target.work_id in set(dataset.groups):
        raise BundleError("target work_id overlaps the training references")
    fragments = set(map(str, dataset.texts))
    whole_works = {}
    for group, text in zip(dataset.groups, dataset.texts, strict=True):
        whole_works.setdefault(str(group), []).append(str(text))
    if any(text in fragments for text in texts) or " ".join(texts) in {" ".join(parts) for parts in whole_works.values()}:
        raise BundleError("target exactly copies available training content")
    target_text = " ".join(texts)
    for parts in whole_works.values():
        coverage = text_overlap_coverage(target_text, " ".join(parts))
        if coverage > 0 and coverage >= threshold:
            raise BundleError("target normalized/shingle content overlaps available training references")
    result.update(work_id="checked_training_references", exact_content="checked_matching_training_references",
                  near_duplicates="checked_matching_training_references")
    return result


def run(
    cfg=None,
    unknown_dir: str | None = None,
    *,
    expected_bundle_token: str | None = None,
    target_work: str | None = None,
) -> dict:
    cfg = cfg or load_config()
    from ..domain.work_weighting import resolve_training_weighting
    weighting = resolve_training_weighting(cfg.get_path("evaluation.training_weighting"))
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
    bundle_root = data / "deployment" / weighting
    bundle_meta, payloads = load_bundle_payloads(
        bundle_root, expected_token=trusted_token
    )
    if bundle_meta["training_weighting"] != weighting:
        raise BundleError("bundle training weighting does not match the selected mode")
    target = resolve_prediction_target(cfg, target_work, unknown_root=unknown_dir)
    selected_identity = target_identity(target)
    # Deserialize the exact bytes whose hashes were checked, never a pathname
    # that could be substituted between verification and joblib.load.
    try:
        authors_raw = loads_strict(payloads["authors.json"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError(f"bundle authors.json is invalid strict UTF-8 JSON: {exc}") from exc
    try:
        authors = list(validate_author_universe(authors_raw))
    except (AttributeError, PredictionContractError) as exc:
        raise BundleError(f"bundle class-universe contract failed: {exc}") from exc
    candidates = deployment_candidates(cfg)
    if tuple(authors) != candidates:
        raise BundleError("bundle authors do not match deployment.candidate_authors")
    from ..report.evidence import _config_id, _code_tree_sha256, SectionEvidenceError
    if bundle_meta["config_id"] != _config_id(cfg) or bundle_meta["code_tree_sha256"] != _code_tree_sha256():
        raise SectionEvidenceError("prediction bundle code/config does not match the current resolved config")
    pipe = joblib.load(io.BytesIO(payloads["model.pkl"]))
    delta = joblib.load(io.BytesIO(payloads["delta.pkl"]))
    try:
        validate_class_indices(pipe.classes_, len(authors), name="model.classes_")
        validate_class_indices(delta.classes_, len(authors), name="delta.classes_")
    except (AttributeError, PredictionContractError) as exc:
        raise BundleError(f"bundle class-universe contract failed: {exc}") from exc

    texts = _target_texts(target)
    overlap = _overlap_status(cfg, target, texts, authors, bundle_meta)
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
    d_mean = d_dist.mean(axis=0)
    order = np.argsort(-lr_full, kind="stable")
    delta_order = np.argsort(d_mean, kind="stable")
    top_k = cfg.get_path("evaluation.top_k_candidates", 5)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("evaluation.top_k_candidates must be a positive integer")
    margin = float(lr_full[order[0]] - lr_full[order[1]]) if len(order) > 1 else 0.0
    ablation = getattr(delta, "ablation_", None)
    denominator = (
        "all_analyzer_events" if ablation.relative_fw else "sum_selected_mfw_counts"
    ) if ablation is not None else "not_recorded"
    result = {
        "schema_version": "stylo.target-prediction.v2", "target_work_id": target.work_id,
        "n_fragments": len(texts), "candidate_authors": authors,
        "winner": None, "diagnostic_closed_set_top": authors[int(order[0])],
        "abstained": True, "abstention_reason": "open_set_applicability_unavailable",
        "score_scope": "closed_set_method_diagnostics", "margin": margin,
        "aggregation": "mean_over_selected_work_fragments", "overlap_check": overlap,
        "training_weighting": weighting,
        "view_settings": {"top_k_candidates": top_k, "output_directory": str(docs_dir)},
        "methods": {
            "stylo_lr": {"scores": {author: float(lr_full[i]) for i, author in enumerate(authors)},
                         "top_candidate": authors[int(order[0])]},
            "delta": {"distances": {author: float(d_mean[i]) for i, author in enumerate(authors)},
                      "top_candidate": authors[int(delta_order[0])],
                      "frequency_denominator": denominator,
                      "training_weighting": weighting,
                      "reference_group_weighting": getattr(delta, "group_weighting_", "not_recorded")},
        },
        "fragments": [{"ordinal": i, "lr_scores": lr_probs[i].tolist(),
                       "delta_distances": d_dist[i].tolist()} for i in range(len(texts))],
    }
    from ..report.build import format_prediction_report
    report = format_prediction_report(result, top_k)
    from ..report.evidence import publish_prediction

    published = publish_prediction(
        cfg,
        unknown_root=target.root,
        target=target,
        selected_identity=selected_identity,
        structured=result,
        report=report,
        bundle_token=trusted_token,
        bundle_meta=bundle_meta,
    )
    print(report)
    if published is not None:
        print(f"Result ID: {published.name}\nОтчёт: {published / 'index.html'}")
    return result
