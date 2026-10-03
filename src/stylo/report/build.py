"""Build an HTML report only from verified, current v2 evidence."""
from __future__ import annotations

import datetime
import hashlib
import html
import logging
import os
import pathlib
import tempfile
from typing import Any

from ..jsonio import load_strict

log = logging.getLogger("stylo.report")

SWEEP_PROVENANCE_SCHEMA = "stylo.sweep.v2.provenance"


def format_prediction_report(result: dict, top_k: int) -> str:
    """Format already-computed method outputs; this function never scores text."""
    from ..lang import display_name
    authors = result["candidate_authors"]
    top_k = min(top_k, len(authors))
    lr = result["methods"]["stylo_lr"]
    delta = result["methods"]["delta"]
    denominator = {
        "sum_selected_mfw_counts": "частоты по выбранному словарю",
        "all_analyzer_events": "частоты по всем словам анализатора",
        "not_recorded": "знаменатель не записан",
    }[delta["frequency_denominator"]]
    content_check = {
        "checked_matching_training_references": "точные совпадения с эталонами проверены",
        "unavailable_no_training_texts": "проверка точных совпадений недоступна: исходных эталонов нет",
    }[result["overlap_check"]["exact_content"]]
    lines = ["=== Диагностический рейтинг кандидатов (LR и Delta) ===",
             f"Дата: {datetime.datetime.now():%d.%m.%Y %H:%M}",
             f"Произведение: {result['target_work_id']}",
             f"Фрагментов: {result['n_fragments']}",
             "Панель: " + ", ".join(display_name(author) for author in authors),
             "Сравнение внутри панели: LR score выше — ближе, Delta distance ниже — ближе; авторство не установлено.",
             "", f"Топ-{top_k} по Stylo LR:"]
    for author in sorted(authors, key=lambda value: -lr["scores"][value])[:top_k]:
        lines.append(f"  {display_name(author):24} LR={lr['scores'][author]:.4f}")
    lines.extend(["", f"Margin LR над 2-м местом: {result['margin']:.4f}",
                  f"Топ-{top_k} по Delta ({denominator}):"])
    for author in sorted(authors, key=lambda value: delta["distances"][value])[:top_k]:
        lines.append(f"  {display_name(author):24} distance={delta['distances'][author]:.4f}")
    lines.extend(["", "Проверка пересечения с эталонами: " + content_check])
    return "\n".join(lines)


class ReportEvidenceError(RuntimeError):
    """Required report evidence is missing, stale, or digest-invalid."""


def _read_required(path: pathlib.Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReportEvidenceError(f"required report evidence missing/unsafe: {path}")
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReportEvidenceError(f"report evidence is not UTF-8: {path}") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verified_sweep(cfg, docs: pathlib.Path) -> tuple[str, str]:
    from ..eval.provenance import ProvenanceError, resolve_published_batch

    expected_files = {
        "sweep_table.v2.csv",
        "sweep_table.v2.provenance.json",
        "sweep_table.v2.txt",
    }
    try:
        resolved = resolve_published_batch(
            docs,
            publication_id="sweep-table-v2",
            expected_names=expected_files,
        )
        provenance_path = resolved["sweep_table.v2.provenance.json"]
        provenance = load_strict(provenance_path)
    except Exception as exc:
        if isinstance(exc, ReportEvidenceError):
            raise
        if isinstance(exc, ProvenanceError):
            raise ReportEvidenceError(f"invalid sweep publication: {exc}") from exc
        raise ReportEvidenceError(f"invalid sweep provenance: {exc}") from exc
    required = {
        "schema_version",
        "training_weighting",
        "strategy",
        "dataset_contract",
        "rows_digest",
        "attestation",
        "files",
        "note",
    }
    if type(provenance) is not dict or set(provenance) != required:
        raise ReportEvidenceError("sweep provenance field inventory mismatch")
    if provenance["schema_version"] != SWEEP_PROVENANCE_SCHEMA:
        raise ReportEvidenceError("unsupported sweep provenance schema")
    strategy = provenance["strategy"]
    if strategy not in {"gkf", "lobo"}:
        raise ReportEvidenceError(f"unsupported sweep strategy: {strategy!r}")

    from ..domain.work_weighting import resolve_training_weighting

    weighting = resolve_training_weighting(
        cfg.get_path("evaluation.training_weighting")
    )
    if provenance["training_weighting"] != weighting:
        raise ReportEvidenceError(
            "sweep weighting does not match the current resolved configuration"
        )
    expected_contract = {
        "chunk_weighted_legacy": "legacy_recursive",
        "work_balanced": "work_balanced_manifest",
    }[weighting]
    if provenance["dataset_contract"] != expected_contract:
        raise ReportEvidenceError("sweep dataset contract/weighting mismatch")
    if (
        type(provenance["rows_digest"]) is not str
        or len(provenance["rows_digest"]) != 64
    ):
        raise ReportEvidenceError("sweep rows_digest is malformed")

    files = provenance["files"]
    expected_payload_files = {"sweep_table.v2.csv", "sweep_table.v2.txt"}
    if (
        type(files) is not dict
        or set(files) != expected_payload_files
        or any(type(value) is not str or len(value) != 64 for value in files.values())
    ):
        raise ReportEvidenceError("sweep file-digest inventory mismatch")
    bodies = {name: _read_required(resolved[name]) for name in expected_payload_files}
    for name, body in bodies.items():
        if _sha256_text(body) != files[name]:
            raise ReportEvidenceError(f"sweep evidence hash mismatch: {name}")

    # A report must not relabel evidence from an older corpus/code/config as a
    # newly generated current run.
    from ..pipeline.train import _attestation

    current_attestation = _attestation(cfg)
    recorded = provenance["attestation"]
    if type(recorded) is not dict or any(
        recorded.get(field) != current_attestation.get(field)
        for field in ("code_tree_sha256", "config_id", "git_commit")
    ):
        raise ReportEvidenceError("sweep code/config attestation is stale")

    from ..dataset import resolve_dataset

    dataset = resolve_dataset(
        cfg,
        weighting,
        exclude_authors=set(
            cfg.get_path("corpus_policy.exclude_from_benchmark", []) or []
        ),
        unknown_name=cfg.get_path(
            "corpus_policy.unknown_dir_name", "unknown"
        ),
    )
    if dataset.provenance.rows_digest != provenance["rows_digest"]:
        raise ReportEvidenceError("sweep corpus rows_digest is stale")

    title = (
        "Что работает: ablation-sweep (полный LOBO)"
        if strategy == "lobo"
        else "Что работает: ablation-sweep (GKF screening proxy)"
    )
    return title, bodies["sweep_table.v2.txt"]


def run_prediction(cfg, target_work: str) -> pathlib.Path:
    """Render one verified prediction independently of corpus/sweep reports."""
    from .evidence import prediction_directory, verify_prediction
    body = verify_prediction(cfg, target_work)
    docs = prediction_directory(cfg, target_work)
    output = docs / "index.html"
    if output.is_symlink():
        raise ReportEvidenceError("prediction report output must not be a symlink")
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>Stylo — сравнение произведения</title>'
        '<style>body{font:16px system-ui;max-width:960px;margin:3rem auto;padding:0 1rem}'
        'pre{white-space:pre-wrap;line-height:1.6}</style>'
        f'<h1>{html.escape(target_work)}</h1><pre>{html.escape(body)}</pre></html>'
    )
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=docs, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(page)
        handle.flush(); os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Отчёт произведения: {output}")
    return output


def run(cfg=None, *, target_work: str | None = None, prediction_only: bool = False) -> None:
    from ..config import load_config

    cfg = cfg or load_config()
    if target_work is not None or prediction_only:
        if target_work is None:
            from ..pipeline.predict import resolve_prediction_target
            target_work = resolve_prediction_target(cfg).work_id
        run_prediction(cfg, target_work)
        return
    docs = pathlib.Path(cfg.get_path("paths.docs", "docs"))
    if docs.is_symlink() or not docs.is_dir():
        raise ReportEvidenceError(f"docs root must be a real existing directory: {docs}")

    from .evidence import (
        SectionEvidenceError,
        verify_corpus_validation,
        verify_prediction,
    )

    try:
        corpus = verify_corpus_validation(cfg)
        prediction = verify_prediction(cfg)
    except SectionEvidenceError as exc:
        raise ReportEvidenceError(str(exc)) from exc
    sweep_title, sweep = _verified_sweep(cfg, docs)
    strategy_claim = (
        "full per-work LOBO"
        if "полный LOBO" in sweep_title
        else "group-aware GKF screening proxy (not LOBO)"
    )

    model = cfg.get_path("language.spacy_model", "?")
    mver = cfg.get_path("language.spacy_model_version", "?")
    parts: list[str] = [
        """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Стилометрия авторства — отчёт</title>
<style>
 body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
 h1 { border-bottom: 2px solid #333; padding-bottom: .3rem; }
 h2 { margin-top: 2rem; color: #234; }
 pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
        padding: 1rem; overflow-x: auto; font-size: 13px; line-height: 1.4; }
 .meta { color: #666; font-size: 13px; }
</style></head><body>""",
        "<h1>Стилометрия авторства русской прозы</h1>",
        (
            f'<p class="meta">Сгенерировано: '
            f"{datetime.datetime.now():%d.%m.%Y %H:%M} · "
            f"модель spaCy: {html.escape(str(model))} "
            f"v{html.escape(str(mver))} · "
            f"evidence strategy: {html.escape(strategy_claim)}</p>"
        ),
    ]
    for title, body in (
        ("Состав и качество корпуса", corpus),
        (sweep_title, sweep),
        ("Атрибуция спорного текста", prediction),
    ):
        parts.append(f"<h2>{html.escape(title)}</h2><pre>{html.escape(body)}</pre>")
    parts.append("</body></html>")

    fd, temporary_name = tempfile.mkstemp(prefix=".index.", dir=docs)
    os.close(fd)
    temporary = pathlib.Path(temporary_name)
    try:
        temporary.write_text("\n".join(parts), encoding="utf-8")
        os.replace(temporary, docs / "index.html")
    finally:
        if temporary.exists():
            temporary.unlink()
    log.info("Отчёт: %s", docs / "index.html")
    print(f"Отчёт собран: {docs / 'index.html'}")
