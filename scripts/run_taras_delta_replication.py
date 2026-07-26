"""Method-independent replication of the Taras Bulba additions attribution.

Burrows Delta (Manhattan over z-scored frequencies — a different classifier
family than the fw_fixed cosine-centroid channel) in two modes:
  - delta_fw   — vocabulary fixed to function words (theme-neutral);
  - delta_mfw  — classic top-300 MFW of the train corpus.

Panel mirrors the suspects spec: Gogol anchor (without Taras Bulba),
Annenkov 1840s prose, Pushkin, Turgenev, Dostoevsky. Case texts are taken
from input_cases/taras_bulba/masked/ (symmetric NER masking) when present.

Protocol per mode: leave-one-work-out gate (held-out work classified by the
majority of its chunks, one work = one vote), random work-label permutation
with a full LOO model refit under every assignment (implemented by an exactly
equivalent cached transform), then target attribution by chunk majority with
the margin between the two closest authors.

Default output:
docs/cases/work_balanced_audit/custom/taras_delta_full_refit_work_balanced.json
The historical report is preserved unless --overwrite-historical is explicit.
Run: .venv/bin/python scripts/run_taras_delta_replication.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
from stylo.lang import function_words  # noqa: E402
from stylo.models.delta import BurrowsDelta  # noqa: E402

IC = ROOT / "input_clean"
CASE = ROOT / "input_cases" / "taras_bulba"
MASKED = CASE / "masked"
HISTORICAL_OUT = (
    ROOT / "docs" / "cases" / "taras_hardened" / "reports" / "delta_replication.json"
)
DEFAULT_OUT = (
    ROOT / "docs" / "cases" / "work_balanced_audit" / "custom"
    / "taras_delta_full_refit_work_balanced.json"
)

WIN = 1500
MFW = 300
GATE_THRESHOLD = 0.80
N_PERM = 2000
SEED = 42
FW = sorted(function_words("ru"))


def masked_path(rel: str) -> pathlib.Path:
    p = MASKED / rel
    return p if p.exists() else CASE / rel

_GOGOL = [p for p in sorted((IC / "gogol").glob("*.txt"))
          if p.name != "тарас_бульба.txt"]

PANELS: Dict[str, Dict[str, List[pathlib.Path]]] = {
    "suspects": {
        "gogol": _GOGOL,
        "annenkov_1840s": sorted(masked_path("cand_annenkov_1840s").glob("*.txt")),
        "pushkin": sorted((IC / "pushkin").glob("*.txt")),
        "turgenev": sorted((IC / "turgenev").glob("*.txt")),
        "dostoevsky": sorted((IC / "dostoevsky").glob("*.txt")),
    },
    "somov_binary": {
        "gogol": _GOGOL,
        "somov": sorted(masked_path("cand_somov").glob("*.txt")),
    },
    "topic": {
        "gogol": _GOGOL,
        "somov": sorted(masked_path("cand_somov").glob("*.txt")),
        "narezhny": sorted(masked_path("cand_narezhny").glob("*.txt")),
        "grebenka": sorted(masked_path("cand_grebenka").glob("*.txt")),
    },
}

TARGETS = {
    "strict_additions": masked_path("dobavleniya1842_strict.txt"),
    "loose_additions": masked_path("dobavleniya1842_loose.txt"),
    "gogol1835_base_control": masked_path("gogol1835_mirgorod.txt"),
}


def chunk_words(text: str, win: int = WIN) -> List[str]:
    w = text.split()
    return [" ".join(w[i:i + win]) for i in range(0, len(w), win)
            if len(w[i:i + win]) >= win // 2]


def load_panel(panel: Dict[str, List[pathlib.Path]]) -> List[Tuple[str, str, str]]:
    """[(author, work, chunk_text)]"""
    rows = []
    for author, files in panel.items():
        for f in files:
            for ch in chunk_words(f.read_text("utf-8", "ignore")):
                rows.append((author, f.stem, ch))
    return rows


def make_model(mode: str) -> BurrowsDelta:
    if mode == "delta_fw":
        return BurrowsDelta(metric="manhattan", vocabulary=FW)
    return BurrowsDelta(mfw_count=MFW, metric="manhattan")


def work_loo(rows, mode: str) -> Tuple[float, Dict[str, str], List[Tuple[str, str]]]:
    works = sorted({(a, w) for a, w, _ in rows})
    labels = [a for a, _w in works]
    return _work_loo_for_labels(rows, mode, labels)


def _work_loo_for_labels(
    rows, mode: str, labels: List[str]
) -> Tuple[float, Dict[str, str], List[Tuple[str, str]]]:
    """Reference full-refit LOO for one work-label assignment.

    This intentionally remains a straightforward implementation: tests use it
    as the oracle for the cached permutation path below.
    """
    works = sorted({(a, w) for a, w, _ in rows})
    if len(labels) != len(works):
        raise ValueError("labels must contain one value per work")
    label_of = dict(zip(works, labels))
    votes: List[Tuple[str, str]] = []  # (true_author, predicted)
    for held_key in works:
        train = [
            (a, w, c) for a, w, c in rows if (a, w) != held_key
        ]
        test = [c for a, w, c in rows if (a, w) == held_key]
        model = make_model(mode)
        model.fit(
            [c for _, _, c in train],
            [label_of[(a, w)] for a, w, _ in train],
            groups=[(a, w) for a, w, _ in train],
        )
        preds = model.predict(test)
        top = Counter(preds).most_common(1)[0][0]
        votes.append((label_of[held_key], top))
    per_author: Dict[str, List[int]] = {}
    for true, pred in votes:
        per_author.setdefault(true, []).append(int(true == pred))
    recalls = {a: sum(v) / len(v) for a, v in per_author.items()}
    macro = sum(recalls.values()) / len(recalls)
    detail = {a: f"{sum(v)}/{len(v)}" for a, v in per_author.items()}
    return macro, detail, votes


@dataclass(frozen=True)
class DeltaFoldCache:
    """Label-independent Delta transform for one held-out work."""

    heldout_index: int
    train_indices: np.ndarray
    train_work_z: np.ndarray
    test_z: np.ndarray
    feature_names: Tuple[str, ...]


@dataclass(frozen=True)
class DeltaPermutationCache:
    """Transforms that can be reused under arbitrary work-label assignments.

    In Delta, the fold vocabulary, relative frequencies, and train z-scaling
    depend on the training *texts/works*, but not on their author labels.  Only
    the final mean of work z-profiles into author centroids depends on labels.
    Caching the former and rebuilding the latter for every permutation is
    therefore mathematically identical to a full model refit, while avoiding
    thousands of repeated vectorizer fits.
    """

    works: Tuple[Tuple[str, str], ...]
    authors: Tuple[str, ...]
    observed_labels: Tuple[str, ...]
    folds: Tuple[DeltaFoldCache, ...]


def build_permutation_cache(rows, mode: str) -> DeltaPermutationCache:
    works = tuple(sorted({(a, w) for a, w, _ in rows}))
    by_work: Dict[Tuple[str, str], List[str]] = {key: [] for key in works}
    for author, work, chunk in rows:
        by_work[(author, work)].append(chunk)

    all_texts: List[str] = []
    chunk_work_indices: List[int] = []
    for work_index, key in enumerate(works):
        all_texts.extend(by_work[key])
        chunk_work_indices.extend([work_index] * len(by_work[key]))
    chunk_work = np.asarray(chunk_work_indices, dtype=int)

    # Tokenize/count once.  For MFW, each fold still chooses its own top-MFW
    # columns strictly from that fold's training term totals.  The global
    # alphabetic column universe is only a lossless sparse cache: held-out-only
    # words have zero training frequency and cannot enter a fold vocabulary.
    if mode == "delta_fw":
        vectorizer = CountVectorizer(
            vocabulary=FW, lowercase=True, token_pattern=r"(?u)\b\w+\b"
        )
    elif mode == "delta_mfw":
        vectorizer = CountVectorizer(
            lowercase=True, token_pattern=r"(?u)\b\w+\b"
        )
    else:
        raise ValueError(f"unknown Delta mode: {mode}")
    all_counts = vectorizer.fit_transform(all_texts)
    all_feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)

    folds = []
    for heldout_index, heldout_key in enumerate(works):
        train_indices = np.asarray(
            [i for i in range(len(works)) if i != heldout_index], dtype=int
        )
        train_chunk_mask = chunk_work != heldout_index
        if mode == "delta_mfw":
            train_tf = np.asarray(
                all_counts[train_chunk_mask].sum(axis=0)
            ).ravel()
            present = np.flatnonzero(train_tf > 0)
            # CountVectorizer sorts candidate terms alphabetically, applies
            # this same argsort to training term frequency, then returns kept
            # columns in alphabetic order.
            if len(present) > MFW:
                ranked = (-train_tf[present]).argsort()[:MFW]
                selected = np.sort(present[ranked])
            else:
                selected = present
        else:
            selected = np.arange(all_counts.shape[1])
        selected_counts = all_counts[:, selected].toarray().astype(np.float64)
        totals = selected_counts.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        relative_frequencies = selected_counts / totals

        train_work_freq = np.vstack(
            [relative_frequencies[chunk_work == int(i)].mean(axis=0)
             for i in train_indices]
        )
        mean = train_work_freq.mean(axis=0)
        std = train_work_freq.std(axis=0)
        std[std == 0] = 1e-9
        train_work_z = (train_work_freq - mean) / std
        test_z = (relative_frequencies[chunk_work == heldout_index] - mean) / std
        folds.append(
            DeltaFoldCache(
                heldout_index=heldout_index,
                train_indices=train_indices,
                train_work_z=train_work_z,
                test_z=test_z,
                feature_names=tuple(str(name) for name in all_feature_names[selected]),
            )
        )

    observed_labels = tuple(author for author, _work in works)
    return DeltaPermutationCache(
        works=works,
        authors=tuple(sorted(set(observed_labels))),
        observed_labels=observed_labels,
        folds=tuple(folds),
    )


def _majority_codes(predictions: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-wise majority with Counter.most_common's first-seen tie break."""
    counts = np.stack(
        [(predictions == code).sum(axis=1) for code in range(n_classes)], axis=1
    )
    max_count = counts.max(axis=1)
    winners = np.full(predictions.shape[0], -1, dtype=int)
    rows = np.arange(predictions.shape[0])
    for position in range(predictions.shape[1]):
        candidate = predictions[:, position]
        take = (winners < 0) & (counts[rows, candidate] == max_count)
        winners[take] = candidate[take]
    return winners


def work_macro_scores(
    cache: DeltaPermutationCache,
    assignments: List[List[str]],
    batch_size: int = 64,
) -> np.ndarray:
    """Score work-label assignments through the cached full-refit equivalent."""
    if not assignments:
        return np.empty(0, dtype=float)
    author_code = {author: i for i, author in enumerate(cache.authors)}
    try:
        codes = np.asarray(
            [[author_code[label] for label in labels] for labels in assignments],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(f"unknown author label: {exc.args[0]}") from exc
    if codes.ndim != 2 or codes.shape[1] != len(cache.works):
        raise ValueError("each assignment must contain one label per work")

    n_classes = len(cache.authors)
    scores = np.empty(len(codes), dtype=float)
    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size]
        correct = np.zeros((len(batch), n_classes), dtype=float)
        denominators = np.stack(
            [(batch == code).sum(axis=1) for code in range(n_classes)], axis=1
        )
        for fold in cache.folds:
            train_labels = batch[:, fold.train_indices]
            distances = np.full(
                (len(batch), len(fold.test_z), n_classes), np.inf, dtype=float
            )
            for code in range(n_classes):
                mask = train_labels == code
                count = mask.sum(axis=1)
                valid = count > 0
                if not np.any(valid):
                    continue
                centroids = (
                    mask[valid].astype(float) @ fold.train_work_z
                ) / count[valid, None]
                distances[valid, :, code] = np.abs(
                    fold.test_z[None, :, :] - centroids[:, None, :]
                ).mean(axis=2)
            chunk_predictions = distances.argmin(axis=2)
            work_predictions = _majority_codes(chunk_predictions, n_classes)
            truth = batch[:, fold.heldout_index]
            for code in range(n_classes):
                correct[:, code] += (truth == code) & (work_predictions == code)

        with np.errstate(divide="ignore", invalid="ignore"):
            recalls = np.divide(
                correct,
                denominators,
                out=np.full_like(correct, np.nan),
                where=denominators > 0,
            )
        scores[start:start + len(batch)] = np.nanmean(recalls, axis=1)
    return scores


def cached_work_loo(
    cache: DeltaPermutationCache, labels: List[str] | None = None
) -> Tuple[float, Dict[str, str], List[Tuple[str, str]]]:
    """Observed/reporting metrics without repeating the expensive fold fits."""
    labels = list(cache.observed_labels if labels is None else labels)
    if len(labels) != len(cache.works):
        raise ValueError("labels must contain one value per work")
    author_code = {author: i for i, author in enumerate(cache.authors)}
    try:
        codes = np.asarray([author_code[label] for label in labels], dtype=int)
    except KeyError as exc:
        raise ValueError(f"unknown author label: {exc.args[0]}") from exc

    votes: List[Tuple[str, str]] = []
    for fold in cache.folds:
        train_labels = codes[fold.train_indices]
        distances = np.full(
            (1, len(fold.test_z), len(cache.authors)), np.inf, dtype=float
        )
        for code in range(len(cache.authors)):
            mask = train_labels == code
            if not np.any(mask):
                continue
            centroid = fold.train_work_z[mask].mean(axis=0)
            distances[0, :, code] = np.abs(
                fold.test_z - centroid[None, :]
            ).mean(axis=1)
        chunk_predictions = distances.argmin(axis=2)
        predicted = int(_majority_codes(chunk_predictions, len(cache.authors))[0])
        truth = int(codes[fold.heldout_index])
        votes.append((cache.authors[truth], cache.authors[predicted]))

    per_author: Dict[str, List[int]] = {}
    for truth, predicted in votes:
        per_author.setdefault(truth, []).append(int(truth == predicted))
    recalls = {author: sum(values) / len(values)
               for author, values in per_author.items()}
    macro = sum(recalls.values()) / len(recalls)
    detail = {
        author: f"{sum(values)}/{len(values)}"
        for author, values in per_author.items()
    }
    return macro, detail, votes


def permutation_p(
    rows,
    mode: str,
    n: int = N_PERM,
    seed: int = SEED,
    cache: DeltaPermutationCache | None = None,
) -> float:
    """Random work-label permutation with model refit under every assignment.

    The refit is evaluated through :func:`work_macro_scores`, whose cached
    transforms are label-independent and are parity-tested against literal
    ``BurrowsDelta.fit`` leave-one-work-out fits.
    """
    if n < 1:
        raise ValueError("n must be positive")
    cache = cache or build_permutation_cache(rows, mode)
    observed_labels = list(cache.observed_labels)
    rng = random.Random(seed)
    assignments = []
    for _ in range(n):
        shuffled = observed_labels[:]
        rng.shuffle(shuffled)
        assignments.append(shuffled)
    all_scores = work_macro_scores(cache, [observed_labels, *assignments])
    observed = all_scores[0]
    hits = int(np.sum(all_scores[1:] >= observed - 1e-12))
    return (hits + 1) / (n + 1)


def gate_passes(macro_recall: float, permutation_p_value: float) -> bool:
    """Confirmatory gate requires both separation and a significant null."""
    return bool(
        macro_recall >= GATE_THRESHOLD - 1e-9
        and permutation_p_value <= 0.05 + 1e-12
    )


def attribute(rows, mode: str, target_text: str) -> Dict[str, object]:
    model = make_model(mode)
    model.fit(
        [c for _, _, c in rows],
        [a for a, _, _ in rows],
        groups=[(a, w) for a, w, _ in rows],
    )
    chunks = chunk_words(target_text)
    if not chunks:
        chunks = [target_text]
    counts = Counter(model.predict(chunks))
    top = counts.most_common(1)[0][0]
    # Margin between the two closest authors by mean Delta distance.
    mean_d = model.distances(chunks).mean(axis=0)
    order = np.argsort(mean_d)
    return {
        "top": top,
        "per_chunk_winners": dict(counts),
        "n_chunks": len(chunks),
        "winner_share": round(counts[top] / len(chunks), 3),
        "mean_distances": {str(model.classes_[i]): round(float(mean_d[i]), 6)
                           for i in order},
        "margin": round(float(mean_d[order[1]] - mean_d[order[0]]), 6),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help="report path (default: work-balanced audit location)",
    )
    parser.add_argument(
        "--n-perm", type=int, default=N_PERM,
        help=f"number of random work-label assignments (default: {N_PERM})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"deterministic permutation seed (default: {SEED})",
    )
    parser.add_argument(
        "--overwrite-historical",
        action="store_true",
        help="allow --out to replace the preserved legacy report",
    )
    args = parser.parse_args(argv)
    if args.n_perm < 1:
        parser.error("--n-perm must be positive")
    if (
        args.out.expanduser().resolve() == HISTORICAL_OUT.resolve()
        and not args.overwrite_historical
    ):
        parser.error(
            "refusing to overwrite the historical report without "
            "--overwrite-historical"
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out.expanduser()
    try:
        report_path = str(out.resolve().relative_to(ROOT))
    except ValueError:
        report_path = str(out.resolve())
    report = {
        "case_id": "taras_bulba_1842_additions",
        "status": "exploratory_adversarial_rerun",
        "lineage": {
            "historical_report": str(HISTORICAL_OUT.relative_to(ROOT)),
            "historical_report_status": "superseded_for_scientific_interpretation",
            "historical_report_superseded_by": report_path,
            "supersession_reason": (
                "equal-work train centroids and full-refit work-label permutation null"
            ),
        },
        "protocol": {
            "unit": "work", "chunk_words": WIN, "gate_threshold": GATE_THRESHOLD,
            "permutation": f"random_{args.n_perm}", "seed": args.seed,
            "permutation_unit": "work",
            "permutation_alpha": 0.05,
            "permutation_refit": "full_loo_cached_label_independent_transform",
            "train_centroid_weighting": "equal_work_after_within_work_chunk_mean",
            "masked_inputs": MASKED.exists(),
        },
        "panels": {},
    }
    for panel_name, panel in PANELS.items():
        rows = load_panel(panel)
        panel_entry = {"composition": {a: len(fs) for a, fs in panel.items()},
                       "modes": {}}
        for mode in ("delta_fw", "delta_mfw"):
            cache = build_permutation_cache(rows, mode)
            macro, detail, _votes = cached_work_loo(cache)
            p = permutation_p(
                rows, mode, n=args.n_perm, seed=args.seed, cache=cache
            )
            gate_pass = gate_passes(macro, p)
            entry = {
                "gate": {
                    "work_macro_recall": round(macro, 4),
                    "gate_pass": gate_pass,
                    "work_recall": detail,
                    "permutation_p": round(p, 4),
                },
                "targets": {},
            }
            for name, path in TARGETS.items():
                attr = attribute(rows, mode, path.read_text("utf-8", "ignore"))
                attr["interpreted"] = gate_pass
                entry["targets"][name] = attr
            panel_entry["modes"][mode] = entry
            print(f"[{panel_name}/{mode}] gate={macro:.4f} p={p:.4f} " +
                  " ".join(f"{n}:{t['top']}({t['winner_share']})"
                           for n, t in entry["targets"].items()), flush=True)
        report["panels"][panel_name] = panel_entry
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_strict(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
