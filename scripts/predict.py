from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
from scipy.stats import mannwhitneyu, trim_mean
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("=== Predict (Unified Pipeline: LR + Delta + Ensemble) ===")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict author for unknown fragments.")
    p.add_argument("--lang", default="ru", help="Language code: ru|en|fr")
    p.add_argument(
        "--datadir", default="data", help="Artifacts directory (model, centroids, etc.)"
    )
    p.add_argument(
        "--unknown-dir", default="", help="Override unknown fragments directory"
    )
    return p.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


def softmax_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    ex = np.exp(arr)
    return ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)


def format_p_value(p_val: float, is_greater: bool = True) -> str:
    # is_greater: метка для читабельности на вызове; внутри функции не учитывается
    _ = is_greater
    if p_val < 0.001:
        return "p < 0.001 (очень значимо)"
    if p_val < 0.01:
        return f"p = {p_val:.3f} (значимо)"
    if p_val < 0.05:
        return f"p = {p_val:.3f} (умеренно значимо)"
    return f"p = {p_val:.3f} (не значимо)"


def resolve_unknown_dir(project_root: Path, override: str) -> Path:
    if override:
        return Path(override).resolve()

    unk_dir = project_root / "data" / "frags_unknown"
    if unk_dir.exists():
        return unk_dir

    # fallback: запасной каталог под альтернативным именем
    unk_dir2 = project_root / "data" / "frags_unknown_plain"
    return unk_dir2


def load_unknown_texts(unk_dir: Path) -> tuple[List[Path], List[str]]:
    if not unk_dir.exists():
        raise FileNotFoundError(f"Unknown folder not found: {unk_dir}")

    unk_files = sorted([p for p in unk_dir.rglob("*.txt") if p.is_file()])
    texts: List[str] = []
    files_kept: List[Path] = []

    for fp in unk_files:
        try:
            txt = fp.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if txt:
            files_kept.append(fp)
            texts.append(txt)

    return files_kept, texts


def main() -> None:
    args = parse_args()

    # IMPORTANT: set language before importing meta/meta-dependent modules
    os.environ["STYLO_LANG"] = args.lang

    from meta.meta import display_name  # noqa: WPS433 (import after env set)

    project_root = Path(__file__).parent.parent.resolve()
    datadir = Path(args.datadir).resolve()

    model_path = datadir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} (run train.py)")

    pipe = joblib.load(model_path)

    delta_scaler_path = datadir / "scaler_delta.pkl"
    centroids_path = datadir / "centroids.npy"
    authors_path = datadir / "authors.npy"

    if (
        not delta_scaler_path.exists()
        or not centroids_path.exists()
        or not authors_path.exists()
    ):
        raise FileNotFoundError(
            "Delta assets not found (scaler_delta.pkl / centroids.npy / authors.npy). Run train.py."
        )

    delta_scaler = joblib.load(delta_scaler_path)
    centroids = np.load(centroids_path, allow_pickle=True)
    authors = np.load(authors_path, allow_pickle=True)

    centroids = np.squeeze(centroids)
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)

    unk_dir = resolve_unknown_dir(project_root, args.unknown_dir)
    logging.info(f"Unknown dir: {unk_dir}")

    unk_files, unk_texts = load_unknown_texts(unk_dir)
    if not unk_texts:
        raise RuntimeError(f"No valid texts found in: {unk_dir}")

    logging.info(f"Unknown fragments loaded: {len(unk_texts)}")

    # Logistic Regression (through the unified pipeline)
    probs = pipe.predict_proba(unk_texts)
    logreg_mean = probs.mean(axis=0)

    log_sorted = np.argsort(logreg_mean)[::-1]
    log_best_idx = int(log_sorted[0])
    log_runner_up_idx = (
        int(log_sorted[1]) if len(log_sorted) > 1 else int(log_sorted[0])
    )

    logreg_p_value = 1.0
    if len(unk_texts) > 1 and probs.shape[1] > 1:
        try:
            _, p_val = mannwhitneyu(
                probs[:, log_best_idx],
                probs[:, log_runner_up_idx],
                alternative="greater",
            )
            logreg_p_value = float(p_val)
        except Exception:
            pass

    # Burrows Delta (Manhattan + Cosine) in Z-space
    # We must use the SAME fitted vectorizer as in the pipeline.
    vec = pipe.named_steps["vectorizer"]
    X_vec = vec.transform(unk_texts)
    X_z = delta_scaler.transform(X_vec)

    delta_manh = manhattan_distances(X_z, centroids)
    delta_cos = cosine_distances(X_z, centroids)

    # robust aggregation across fragments
    manh_mean = trim_mean(delta_manh, 0.1, axis=0)
    cos_mean = trim_mean(delta_cos, 0.1, axis=0)

    manh_sorted = np.argsort(manh_mean)
    cos_sorted = np.argsort(cos_mean)

    manh_best_idx = int(manh_sorted[0])
    manh_runner_up_idx = (
        int(manh_sorted[1]) if len(manh_sorted) > 1 else int(manh_sorted[0])
    )

    cos_best_idx = int(cos_sorted[0])
    cos_runner_up_idx = (
        int(cos_sorted[1]) if len(cos_sorted) > 1 else int(cos_sorted[0])
    )

    manh_p_value = 1.0
    cos_p_value = 1.0
    if len(unk_texts) > 1 and centroids.shape[0] > 1:
        try:
            _, p_val = mannwhitneyu(
                delta_manh[:, manh_best_idx],
                delta_manh[:, manh_runner_up_idx],
                alternative="less",
            )
            manh_p_value = float(p_val)
        except Exception:
            pass

        try:
            _, p_val = mannwhitneyu(
                delta_cos[:, cos_best_idx],
                delta_cos[:, cos_runner_up_idx],
                alternative="less",
            )
            cos_p_value = float(p_val)
        except Exception:
            pass

    # Ensemble
    manh_norm = softmax(-manh_mean)  # smaller dist => bigger score
    cos_norm = softmax(-cos_mean)

    ensemble_scores = 0.50 * logreg_mean + 0.25 * manh_norm + 0.25 * cos_norm

    ens_sorted = np.argsort(ensemble_scores)[::-1]
    ens_best_idx = int(ens_sorted[0])
    ens_runner_up_idx = (
        int(ens_sorted[1]) if len(ens_sorted) > 1 else int(ens_sorted[0])
    )

    delta_manh_norm_per_frag = softmax_rows(-delta_manh)
    delta_cos_norm_per_frag = softmax_rows(-delta_cos)
    ensemble_scores_per_frag = (
        0.50 * probs + 0.25 * delta_manh_norm_per_frag + 0.25 * delta_cos_norm_per_frag
    )

    ensemble_p_value = 1.0
    if len(unk_texts) > 1 and probs.shape[1] > 1:
        try:
            _, p_val = mannwhitneyu(
                ensemble_scores_per_frag[:, ens_best_idx],
                ensemble_scores_per_frag[:, ens_runner_up_idx],
                alternative="greater",
            )
            ensemble_p_value = float(p_val)
        except Exception:
            pass

    top_sorted = np.sort(ensemble_scores)[::-1]
    confidence = float(ensemble_scores[ens_best_idx])
    margin = float(top_sorted[0] - top_sorted[1]) if len(top_sorted) > 1 else 0.0

    # REPORT (make compatible with report.py parser)
    out: List[str] = []
    out.append("=== Авторская атрибуция ===")
    out.append(f"Дата создания отчёта: {datetime.datetime.now():%d.%m.%Y %H:%M:%S}")
    out.append(f"Фрагментов: {len(unk_texts)}")
    out.append("Авторы: " + ", ".join(display_name(a) for a in authors))
    out.append("")

    out.append("Logistic Regression:")
    for i, a in enumerate(authors):
        out.append(f" {display_name(a)}: {logreg_mean[i]:.4f}")
    out.append(f" Предсказание: {display_name(authors[log_best_idx])}")
    out.append(
        f" P-value (vs {display_name(authors[log_runner_up_idx])}): "
        f"{format_p_value(logreg_p_value, is_greater=True)}"
    )
    out.append("")

    out.append("Burrows Delta (Manhattan):")
    for i, a in enumerate(authors):
        out.append(f" {display_name(a)}: {manh_mean[i]:.4f}")
    out.append(f" Предсказание: {display_name(authors[manh_best_idx])}")
    out.append(
        f" P-value (vs {display_name(authors[manh_runner_up_idx])}): "
        f"{format_p_value(manh_p_value, is_greater=False)}"
    )
    out.append("")

    out.append("Burrows Delta (Cosine):")
    for i, a in enumerate(authors):
        out.append(f" {display_name(a)}: {cos_mean[i]:.4f}")
    out.append(f" Предсказание: {display_name(authors[cos_best_idx])}")
    out.append(
        f" P-value (vs {display_name(authors[cos_runner_up_idx])}): "
        f"{format_p_value(cos_p_value, is_greater=False)}"
    )
    out.append("")

    out.append("Ensemble:")
    for i, a in enumerate(authors):
        out.append(f" {display_name(a)}: {ensemble_scores[i]:.4f}")
    out.append(f" Победитель ансамбля: {display_name(authors[ens_best_idx])}")
    out.append(f" Margin: {margin:.4f}")
    out.append(
        f" P-value (vs {display_name(authors[ens_runner_up_idx])}): "
        f"{format_p_value(ensemble_p_value, is_greater=True)}"
    )
    out.append("")

    # These keys are parsed by scripts/report.py (it expects this exact phrase)
    out.append(f"=== ИТОГОВОЕ ЗАКЛЮЧЕНИЕ: {display_name(authors[ens_best_idx])} ===")
    out.append(f"Уверенность: {confidence * 100:.2f}%")
    out.append(f"p-value: {ensemble_p_value:.6f}")
    out.append("")

    out.append("Методы:")
    out.append(f" Logistic Regression → {display_name(authors[log_best_idx])}")
    out.append(f" Manhattan Delta → {display_name(authors[manh_best_idx])}")
    out.append(f" Cosine Delta → {display_name(authors[cos_best_idx])}")
    out.append(f" Ensemble → {display_name(authors[ens_best_idx])}")
    out.append("")

    out.append("Справка по уверенности:")
    out.append(" 0.00–0.25 : слабая связь")
    out.append(" 0.25–0.45 : умеренная")
    out.append(" 0.45–0.70 : сильная")
    out.append(" 0.70–1.00 : очень сильная")
    out.append("")

    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Both output names: prediction.txt and dual_prediction_report.txt (read by report.py)
    (docs_dir / "prediction.txt").write_text("\n".join(out), encoding="utf-8")
    (docs_dir / "dual_prediction_report.txt").write_text(
        "\n".join(out), encoding="utf-8"
    )

    print("\n".join(out))
    logging.info("Done. Saved: docs/prediction.txt and docs/dual_prediction_report.txt")


if __name__ == "__main__":
    main()
